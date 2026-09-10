"""Freeze HB4 inputs and reconstruct visual observations for recovery branches.

This module deliberately keeps source experiments read-only.  It writes only
small manifests to the code export tree and reconstructed observations to the
separate HB4 run root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_array, sha256_file, write_table


CODE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = CODE_ROOT.parent.parent / "cr_scizor"
HIST_ROOT = SOURCE_ROOT / "experiments/handback/hb3p_stop_continue_formal_v1"
BASE_CHECKPOINT = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
TEACHER_CHECKPOINT = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_teacher_epoch_100.pth")
SOURCE_HDF5 = SOURCE_ROOT / "data/robomimic/square/ph/image.hdf5"
BASE_MANIFEST = SOURCE_ROOT / "experiments/handback/hb1_v1/base/square/data/bc_source_manifest.json"
RUN_ROOT = Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1")
EXPORT_ROOT = CODE_ROOT / "experiments/handback/hb4_square_absorb_v1"
BASE_SHA = "e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6"
TEACHER_SHA = "7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62"
HIST_PROTOCOL_SHA = "040b679569eaab02f577c056fdc85f2049a19af3ad8351fc50d35c7afe46c3d3"
EXPECTED_OBS = {
    "agentview_image": [3, 84, 84],
    "robot0_eye_in_hand_image": [3, 84, 84],
    "robot0_eef_pos": [3],
    "robot0_eef_quat": [4],
    "robot0_gripper_qpos": [2],
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a real, atomic CSV without changing the shared JSONL helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    fd, tmp_name = __import__("tempfile").mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _file_record(path: Path, role: str, expected_sha: str | None = None) -> dict:
    path = path.expanduser().resolve()
    actual = _sha(path)
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_sha,
        "hash_match": expected_sha is None or actual == expected_sha,
        "placeholder": path.name.endswith(".placeholder.md"),
        "readable": os.access(path, os.R_OK),
    }


def _dir_digest(path: Path, pattern: str) -> dict:
    files = sorted(path.glob(pattern))
    digest = hashlib.sha256()
    rows = []
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha(item)))
        rows.append({"name": item.name, "bytes": item.stat().st_size, "sha256": _sha(item)})
    return {"path": str(path.resolve()), "pattern": pattern, "count": len(files), "digest": digest.hexdigest(), "files": rows}


def _checkpoint_metadata(path: Path) -> dict:
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = json.loads(payload["config"])
    shape = payload.get("shape_metadata", {})
    return {
        "algo_name": payload.get("algo_name"),
        "shape_metadata": shape,
        "config": config,
        "env_metadata": payload.get("env_metadata", {}),
        "model_key_count": len(payload.get("model", {})),
    }


def _environment_lock() -> dict:
    packages = {}
    for name in ("torch", "robomimic", "robosuite", "mujoco", "numpy", "h5py", "pandas"):
        try:
            module = __import__(name)
            packages[name] = {
                "version": getattr(module, "__version__", "unknown"),
                "file": str(getattr(module, "__file__", "unknown")),
            }
        except Exception as exc:
            packages[name] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch

        cuda = {
            "torch_version": torch.version.cuda,
            "is_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    except Exception as exc:
        cuda = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "schema_version": "hb4_environment_lock_v1",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {key: os.environ.get(key) for key in ("MUJOCO_GL", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "packages": packages,
        "cuda": cuda,
    }


def _git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(CODE_ROOT), *args], text=True).strip()

    return {
        "code_root": str(CODE_ROOT.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
        "remote": run("remote", "-v"),
    }


def freeze() -> dict:
    for path in (BASE_CHECKPOINT, TEACHER_CHECKPOINT, SOURCE_HDF5, BASE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha(BASE_CHECKPOINT) != BASE_SHA:
        raise RuntimeError("base checkpoint hash mismatch")
    if _sha(TEACHER_CHECKPOINT) != TEACHER_SHA:
        raise RuntimeError("teacher checkpoint hash mismatch")
    draft = _json(CODE_ROOT / "../.." / "HB4_square_absorb_v1_bundle.zip") if False else None
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("config", "assets", "data", "metrics", "report", "package"):
        (EXPORT_ROOT / name).mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

    base_meta = _checkpoint_metadata(BASE_CHECKPOINT)
    teacher_meta = _checkpoint_metadata(TEACHER_CHECKPOINT)
    base_shape = base_meta["shape_metadata"]
    expected_keys = sorted(EXPECTED_OBS)
    actual_keys = list(base_shape.get("all_obs_keys", []))
    compatibility = {
        "action_dim": int(base_shape.get("ac_dim", -1)),
        "rnn_enabled": bool(base_meta["config"].get("algo", {}).get("rnn", {}).get("enabled")),
        "rnn_horizon": int(base_meta["config"].get("algo", {}).get("rnn", {}).get("horizon", -1)),
        "expected_obs_keys": expected_keys,
        "actual_obs_keys": actual_keys,
        "obs_keys_match": actual_keys == expected_keys,
        "shape_match": base_shape.get("all_shapes", {}) == EXPECTED_OBS,
        "action_match": int(base_shape.get("ac_dim", -1)) == 7,
        "rnn_match": bool(base_meta["config"].get("algo", {}).get("rnn", {}).get("enabled")) and int(base_meta["config"].get("algo", {}).get("rnn", {}).get("horizon", -1)) == 10,
        "teacher_privileged_low_dim": teacher_meta["shape_metadata"].get("all_obs_keys", []),
    }
    assets = {
        "schema_version": "hb4_resolved_inputs_v1",
        "code_root": str(CODE_ROOT.resolve()),
        "run_root": str(RUN_ROOT.resolve()),
        "export_root": str(EXPORT_ROOT.resolve()),
        "source_experiment": str(HIST_ROOT.resolve()),
        "historical_protocol_sha256": _sha(HIST_ROOT / "config/frozen_protocol.json"),
        "files": [
            _file_record(BASE_CHECKPOINT, "square_base_checkpoint", BASE_SHA),
            _file_record(TEACHER_CHECKPOINT, "square_frozen_repair_teacher", TEACHER_SHA),
            _file_record(SOURCE_HDF5, "square_visual_source_hdf5"),
            _file_record(BASE_MANIFEST, "square_D0_source_manifest"),
            _file_record(HIST_ROOT / "metrics/episodes.jsonl", "historical_episode_table"),
            _file_record(HIST_ROOT / "config/frozen_protocol.json", "historical_protocol", HIST_PROTOCOL_SHA),
            _file_record(HIST_ROOT / "metrics/branch_parity.csv", "historical_branch_parity"),
        ],
        "directories": [
            _dir_digest(HIST_ROOT / "roots/payloads", "*"),
            _dir_digest(HIST_ROOT / "roots", "rollout_*.npz"),
            _dir_digest(HIST_ROOT / "episodes", "shard*/FIXED_L60/*.npz"),
            _dir_digest(HIST_ROOT / "episodes", "shard*/FIXED_L80/*.npz"),
        ],
        "checkpoint_metadata": {"base": base_meta, "teacher": teacher_meta},
        "compatibility": compatibility,
        "historical_assets_are_read_only": True,
        "placeholder_assets_allowed": False,
    }
    roles = {
        "schema_version": "hb4_data_roles_v1",
        "historical_cohort": {
            "seed_start": 800000, "seed_stop_exclusive": 800400,
            "old_role": "HB3_FORMAL_TEST", "new_role": "HB4_source_development",
            "test_reuse_forbidden": True,
        },
        "standard_source": {"seed_start": 920000, "count": 200, "role": "HB4_standard_source"},
        "pilot": {"seed_start": 929000, "count": 4, "role": "HB4_pilot"},
        "development": {"seed_start": 930000, "count": 80, "role": "HB4_development"},
        "test": {"seed_start": 940000, "count": 400, "role": "HB4_test"},
        "collision_audit": "pending_before_new_root_collection",
    }
    seed_registry = {
        "schema_version": "hb4_seed_registry_v1",
        "ranges": roles,
        "historical_roots": "registered_from_hb3_formal_episode_table",
        "new_ranges_collision_checked": False,
    }
    policy_schema = {
        "schema_version": "hb4_policy_input_schema_v1",
        "student_input_keys": expected_keys,
        "student_input_shapes": EXPECTED_OBS,
        "student_input_layout": {"images": "HWC uint8 source; checkpoint preprocessing exactly once", "proprio": "float arrays"},
        "student_action_dim": 7,
        "student_action_bounds": [-1.0, 1.0],
        "student_extra_features_forbidden": ["object", "base_suggestion", "success", "reward", "seed", "future_result", "helper_mask", "stop_continue"],
        "teacher_input_keys": teacher_meta["shape_metadata"].get("all_obs_keys", []),
        "teacher_is_offline_label_source_only": True,
    }
    frozen = {
        "schema_version": "hb4_resolved_config_v1",
        "experiment_id": "hb4_square_absorb_v1",
        "status": "A_FROZEN_ASSETS_B_PENDING",
        "source_commit": _git_state()["commit"],
        "research_branch": "exp/hb4-square-absorb-v1",
        "paths": {"code_root": str(CODE_ROOT.resolve()), "run_root": str(RUN_ROOT.resolve()), "export_root": str(EXPORT_ROOT.resolve()), "base_checkpoint": str(BASE_CHECKPOINT.resolve()), "teacher_checkpoint": str(TEACHER_CHECKPOINT.resolve()), "source_hdf5": str(SOURCE_HDF5.resolve()), "base_source_manifest": str(BASE_MANIFEST.resolve()), "historical_root": str(HIST_ROOT.resolve())},
        "protocol": {"horizon_steps": 400, "entry_t": 20, "short_length": 60, "fixed_length": 80, "state_atol": 1e-10, "block_length": 10, "blocks_per_root": 4, "labels_per_root": 40},
        "scope": {"task": "square", "no_help_main_evaluation": True, "new_rl": False, "online_updates": False, "secondary_assisted_eval_enabled": False},
        "compatibility": compatibility,
    }
    atomic_json_dump(assets, EXPORT_ROOT / "assets/resolved_inputs.json")
    atomic_json_dump({"schema_version": "hb4_resolved_inputs_public_v1", "file_roles": [{"role": x["role"], "bytes": x["bytes"], "sha256": x["sha256"], "path_basename": Path(x["path"]).name} for x in assets["files"]], "compatibility": compatibility}, EXPORT_ROOT / "assets/resolved_inputs_public.json")
    atomic_json_dump(roles, EXPORT_ROOT / "assets/data_roles.json")
    atomic_json_dump(seed_registry, EXPORT_ROOT / "assets/seed_registry.json")
    atomic_json_dump(_environment_lock(), EXPORT_ROOT / "assets/environment_lock.json")
    atomic_json_dump(policy_schema, EXPORT_ROOT / "assets/policy_input_schema.json")
    atomic_json_dump(frozen, EXPORT_ROOT / "config/hb4_resolved.json")
    atomic_json_dump(_git_state(), EXPORT_ROOT / "config/git_state.json")
    atomic_json_dump({"base": base_meta, "teacher": teacher_meta, "compatibility": compatibility}, EXPORT_ROOT / "metrics/checkpoint_load_audit.json")
    return {"status": frozen["status"], "compatibility": compatibility, "export_root": str(EXPORT_ROOT)}


def _episode_rows() -> list[dict]:
    rows = [_json_line for _json_line in (json.loads(line) for line in (HIST_ROOT / "metrics/episodes.jsonl").read_text(encoding="utf-8").splitlines())]
    return rows


def _branch_path(seed: int, length: int, suffix: str = ".npz") -> Path:
    return next((HIST_ROOT / "episodes").glob(f"shard*/FIXED_L{length}/square__hb3p_stop_continue_formal__{seed}{suffix}"))


def build_source_table() -> dict:
    rows = _episode_rows()
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["root_seed"]), {})[row["method_id"]] = row
    output = []
    eligible = []
    for seed in sorted(grouped):
        group = grouped[seed]
        none, l60, l80 = group["NONE"], group["FIXED_L60"], group["FIXED_L80"]
        reasons = []
        if none.get("system_success"):
            reasons.append("NONE_SUCCESS")
        if not l60.get("genuine_handoff_success"):
            reasons.append("L60_NOT_GENUINE_HANDOFF")
        if not l60.get("engineering_ok") or not l80.get("engineering_ok"):
            reasons.append("ENGINEERING_FAILURE")
        if l60.get("protocol_hash") != HIST_PROTOCOL_SHA or l80.get("protocol_hash") != HIST_PROTOCOL_SHA:
            reasons.append("PROTOCOL_HASH_MISMATCH")
        p60, p80 = _branch_path(seed, 60), _branch_path(seed, 80)
        with np.load(p60, allow_pickle=False) as a, np.load(p80, allow_pickle=False) as b:
            s60, s80 = np.asarray(a["states"], np.float64), np.asarray(b["states"], np.float64)
            a60, a80 = np.asarray(a["actions"], np.float32), np.asarray(b["actions"], np.float32)
            if not np.array_equal(s60[:21], s80[:21]):
                reasons.append("SHARED_PREFIX_STATE_MISMATCH")
            if not np.array_equal(a60[:20], a80[:20]):
                reasons.append("SHARED_PREFIX_ACTION_MISMATCH")
            if int(a["helper_mask"].sum()) != 60 or int(b["helper_mask"].sum()) != 80:
                reasons.append("HELPER_MASK_COUNT_MISMATCH")
            if not np.array_equal(a["helper_mask"][20:80], np.ones(60, dtype=bool)):
                reasons.append("L60_HELPER_INTERVAL_MISMATCH")
            if not np.array_equal(b["helper_mask"][20:100], np.ones(80, dtype=bool)):
                reasons.append("L80_HELPER_INTERVAL_MISMATCH")
            if len(a60) < 80 or len(a80) < 100:
                reasons.append("SUPERVISION_INTERVAL_UNAVAILABLE")
        root_rollout = HIST_ROOT / "roots" / f"rollout_{seed - 800000:03d}.npz"
        payload = HIST_ROOT / "roots/payloads" / f"{seed - 800000:03d}.npz"
        if not root_rollout.is_file() or not payload.is_file():
            reasons.append("MISSING_CANONICAL_OR_RGB_ROOT")
        row = {
            "canonical_group_id": f"square:hb4_source:{seed}", "task": "square", "root_seed": seed,
            "root_id": f"square:hb3p_stop_continue_formal:{seed}", "none_system_success": bool(none.get("system_success")),
            "l60_system_success": bool(l60.get("system_success")), "l60_genuine_handoff_success": bool(l60.get("genuine_handoff_success")),
            "l80_system_success": bool(l80.get("system_success")), "l80_genuine_handoff_success": bool(l80.get("genuine_handoff_success")),
            "l60_autonomous_completion": bool(l60.get("autonomous_completion", False)), "l80_autonomous_completion": bool(l80.get("autonomous_completion", False)),
            "l60_first_raw_success_state": l60.get("first_raw_success_state"), "l80_first_raw_success_state": l80.get("first_raw_success_state"),
            "l60_handoff_state_index": l60.get("handoff_state_index"), "l80_handoff_state_index": l80.get("handoff_state_index"),
            "l60_episode_end_state_index": l60.get("episode_end_state_index"), "l80_episode_end_state_index": l80.get("episode_end_state_index"),
            "initial_state_hash": none.get("initial_state_hash"), "protocol_hash": HIST_PROTOCOL_SHA,
            "canonical_payload_path": str(payload.resolve()), "root_rollout_path": str(root_rollout.resolve()),
            "l60_episode_json": str(_branch_path(seed, 60, ".json").resolve()), "l80_episode_json": str(_branch_path(seed, 80, ".json").resolve()),
            "l60_trajectory_path": str(p60.resolve()), "l80_trajectory_path": str(p80.resolve()),
            "eligible_short_handoff": not reasons, "exclusion_reasons": ";".join(reasons),
        }
        output.append(row)
        if not reasons:
            eligible.append(seed)
    _write_csv(output, EXPORT_ROOT / "data/source_root_table.csv")
    write_table(output, EXPORT_ROOT / "data/source_root_table.jsonl")
    eligible_rows = [{"canonical_group_id": f"square:hb4_source:{s}", "root_seed": s, "root_id": f"square:hb3p_stop_continue_formal:{s}"} for s in sorted(eligible)]
    atomic_json_dump({"schema_version": "hb4_eligible_roots_v1", "definition": "NONE.system_success==0 AND FIXED_L60.genuine_handoff_success==1 AND complete_pair_and_shared_prefix", "count": len(eligible_rows), "root_seeds": sorted(eligible), "rows": eligible_rows}, EXPORT_ROOT / "data/eligible_roots.json")
    return {"roots": len(output), "eligible": len(eligible), "eligible_root_seeds": sorted(eligible)}


def _load_payload(seed: int) -> dict:
    p = HIST_ROOT / "roots/payloads" / f"{seed - 800000:03d}.npz"
    with np.load(p, allow_pickle=False) as handle:
        states = np.asarray(handle["states"], dtype=np.float64)
    return {"states": states, "model": p.with_suffix(".xml").read_text(encoding="utf-8"), "seed": seed}


def reconstruct(seed: int, length: int, state_atol: float = 1e-10) -> dict:
    source = SOURCE_HDF5.resolve()
    branch = _branch_path(seed, length)
    with np.load(branch, allow_pickle=False) as handle:
        actions = np.asarray(handle["actions"], dtype=np.float32)
        saved_states = np.asarray(handle["states"], dtype=np.float64)
        saved_success = np.asarray(handle["success"], dtype=np.bool_)
        saved_rewards = np.asarray(handle["rewards"], dtype=np.float64)
    env = EnvAdapter(source, **load_observation_spec(source))
    obs_rows: list[dict[str, np.ndarray]] = []
    max_pre = max_post = max_reward = 0.0
    success_mismatch = 0
    try:
        obs = env.reset_canonical(_load_payload(seed), seed)
        for t, action in enumerate(actions):
            pre = env.physical_state()
            max_pre = max(max_pre, float(np.max(np.abs(pre - saved_states[t]))))
            if 20 <= t < 100:
                obs_rows.append({key: np.asarray(obs[key]).copy() for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")})
            obs, reward, success, _ = env.step(action)
            post = env.physical_state()
            max_post = max(max_post, float(np.max(np.abs(post - saved_states[t + 1]))))
            max_reward = max(max_reward, abs(float(reward) - float(saved_rewards[t])))
            success_mismatch += int(bool(success) != bool(saved_success[t]))
    except Exception as exc:
        return {"seed": seed, "length": length, "verified": False, "failure": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()
    verified = max(max_pre, max_post) <= state_atol and success_mismatch == 0
    output_dir = RUN_ROOT / "observations" / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"FIXED_L{length}.npz"
    if verified:
        arrays: dict[str, np.ndarray] = {
            "actions": np.asarray(actions[20:min(100, len(actions))], dtype=np.float32),
            "absolute_t": np.arange(20, min(100, len(actions)), dtype=np.int64),
        }
        for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            arrays[key] = np.asarray([row[key] for row in obs_rows])
        np.savez_compressed(cache_path, **arrays)
    result = {
        "schema_version": "hb4_reconstruction_audit_row_v1", "seed": seed, "length": length,
        "branch_path": str(branch.resolve()), "cache_path": str(cache_path.resolve()) if verified else None,
        "image_origin": "RECONSTRUCTED_FOR_HB4", "reconstruction_verified": bool(verified),
        "state_atol": state_atol, "max_pre_state_abs": max_pre, "max_post_state_abs": max_post,
        "max_reward_abs": max_reward, "success_mismatch_count": success_mismatch,
        "cached_frames": len(obs_rows), "failure": None if verified else "STATE_OR_SUCCESS_MISMATCH",
    }
    atomic_json_dump(result, output_dir / f"FIXED_L{length}.json")
    return result


def reconstruct_all() -> dict:
    eligible = _json(EXPORT_ROOT / "data/eligible_roots.json")["root_seeds"]
    rows = []
    for seed in eligible:
        for length in (60, 80):
            audit = RUN_ROOT / "observations" / str(seed) / f"FIXED_L{length}.json"
            if audit.is_file():
                rows.append(_json(audit))
            else:
                rows.append(reconstruct(int(seed), length))
    verified_seeds = sorted({r["seed"] for r in rows if r["reconstruction_verified"]} & set(eligible))
    failed = [r for r in rows if not r["reconstruction_verified"]]
    atomic_json_dump({"schema_version": "hb4_reconstruction_audit_v1", "candidate_roots": len(eligible), "branch_records": len(rows), "verified_roots_with_both_branches": len(verified_seeds), "verified_root_seeds": verified_seeds, "failed_records": failed, "status": "PASS_B_READY" if len(verified_seeds) >= 30 else "HOLD_HB4_OBSERVATION_RECONSTRUCTION"}, EXPORT_ROOT / "data/reconstruction_audit.json")
    _write_csv(rows, EXPORT_ROOT / "data/reconstruction_audit.csv")
    return {"candidate_roots": len(eligible), "verified_roots": len(verified_seeds), "failed_records": len(failed), "status": "PASS_B_READY" if len(verified_seeds) >= 30 else "HOLD_HB4_OBSERVATION_RECONSTRUCTION"}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("source-table")
    sub.add_parser("reconstruct")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "source-table":
        result = build_source_table()
    else:
        result = reconstruct_all()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
