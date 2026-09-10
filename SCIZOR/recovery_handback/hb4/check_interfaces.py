"""Run HB4-D checkpoint, input-whitelist, and step-0 parity checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.common import atomic_json_dump, sha256_file


BASE = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
EXPORT = Path("/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1/experiments/handback/hb4_square_absorb_v1")
RUN = Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1")
STUDENT_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _strict_checkpoint_audit(device: torch.device) -> dict:
    from robomimic.algo import algo_factory
    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import obs_utils as ObsUtils

    config, ckpt = FileUtils.config_from_checkpoint(ckpt_path=str(BASE), verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    shape = ckpt["shape_metadata"]
    model = algo_factory(config.algo_name, config, shape["all_shapes"], shape["ac_dim"], device)
    checkpoint_keys = set(ckpt["model"])
    model_keys = set(model.serialize())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    model.deserialize(ckpt["model"])
    return {
        "checkpoint": str(BASE.resolve()), "checkpoint_sha256": sha256_file(BASE),
        "algo_name": ckpt.get("algo_name"), "model_key_count": len(checkpoint_keys),
        "missing_keys": missing, "unexpected_keys": unexpected,
        "shape_metadata": shape, "strict_deserialize": True,
        "device": str(device),
    }


def _first_cache() -> Path:
    paths = sorted((RUN / "observations").glob("*/FIXED_L60.npz"))
    if not paths:
        raise FileNotFoundError("no verified HB4 observation cache")
    return paths[0]


def _input_and_step0_audit(device: str) -> tuple[dict, dict]:
    cache_path = _first_cache()
    with np.load(cache_path, allow_pickle=False) as data:
        keys = sorted(data.files)
        obs = {key: np.asarray(data[key][0]).copy() for key in STUDENT_KEYS}
    expected = {
        "agentview_image": [84, 84, 3], "robot0_eye_in_hand_image": [84, 84, 3],
        "robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2],
    }
    observed = {key: list(value.shape) for key, value in obs.items()}
    forbidden = sorted(set(keys) - set(STUDENT_KEYS) - {"actions", "absolute_t"})
    if observed != expected or forbidden:
        raise RuntimeError(f"input whitelist mismatch: observed={observed}, forbidden={forbidden}")
    first = BasePolicyAdapter(BASE, device=device)
    second = BasePolicyAdapter(BASE, device=device)
    first.start_episode()
    second.start_episode()
    action_a = first.suggest_once(obs, 20, "square:hb4:pilot:929000")
    action_b = second.suggest_once(obs, 20, "square:hb4:pilot:929000")
    delta = float(np.max(np.abs(action_a - action_b)))
    if delta > 1e-6:
        raise RuntimeError(f"step-0 base parity failed: max_abs={delta}")
    input_audit = {
        "cache": str(cache_path.resolve()), "cache_sha256": sha256_file(cache_path),
        "student_keys": list(STUDENT_KEYS), "observed_shapes": observed,
        "forbidden_cache_keys": forbidden, "raw_image_layout": "HWC_uint8",
        "student_extra_features": False, "input_whitelist_passed": True,
    }
    parity = {
        "same_checkpoint": True, "same_observation": True, "same_random_tape": True,
        "action_max_abs": delta, "action_dim": int(action_a.shape[0]),
        "base_calls_each": [first._calls, second._calls], "passed": True,
    }
    return input_audit, parity


def run(output_root: Path, device: str) -> dict:
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_audit = _strict_checkpoint_audit(torch_device)
    input_audit, parity = _input_and_step0_audit(str(torch_device))
    metrics = output_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(checkpoint_audit, metrics / "checkpoint_load_audit.json")
    atomic_json_dump(input_audit, metrics / "input_audit.json")
    atomic_json_dump({"schema_version": "hb4_d_reference_interface_tests_v1", "checkpoint": parity, "strict_load": True, "input_whitelist": True, "passed": True}, metrics / "reference_tests.json")
    atomic_json_dump({"schema_version": "hb4_pilot_interface_summary_v1", "status": "PASS_HB4_D_INTERFACE", "pilot_roots": 0, "note": "interface parity only; environment pilot remains pending", "step0_parity": parity}, metrics / "pilot_summary.json")
    return {"status": "PASS_HB4_D_INTERFACE", "checkpoint_keys": checkpoint_audit["model_key_count"], "step0_action_max_abs": parity["action_max_abs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=EXPORT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
