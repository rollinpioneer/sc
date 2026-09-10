"""Run one frozen no-help policy checkpoint over canonical evaluation roots."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_file, write_table


SOURCE = Path("/home/__compress_data/xushijie/work/cr_scizor/data/robomimic/square/ph/image.hdf5")
DEFAULT_PROTOCOL = Path("/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1/experiments/handback/hb4_square_absorb_v1/config/test_frozen_protocol.json")
HORIZON = 400
STABLE = 5


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _payload(row: dict) -> dict:
    with np.load(row["payload_path"], allow_pickle=False) as handle:
        states = np.asarray(handle["states"], dtype=np.float64)
    return {"states": states, "model": Path(row["model_path"]).read_text(encoding="utf-8"), "seed": int(row["seed"])}


def _protocol_sha256(protocol: Path) -> str:
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    return sha256_file(protocol)


def evaluate(roots_path: Path, checkpoint: Path, method: str, training_seed: int | None, output_dir: Path, device: str, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    roots = _rows(roots_path)
    protocol_sha = _protocol_sha256(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "episodes.jsonl"
    existing = {(row["canonical_group_id"], row.get("training_seed")): row for row in _rows(result_path)} if result_path.is_file() else {}
    policy = BasePolicyAdapter(checkpoint, device=device)
    env = EnvAdapter(SOURCE, **load_observation_spec(SOURCE))
    started = time.time()
    output_rows = dict(existing)
    try:
        for root in roots:
            key = (root["canonical_group_id"], training_seed)
            if key in output_rows and output_rows[key].get("complete_record") and output_rows[key].get("engineering_ok"):
                continue
            job_id = f"hb4-{method}-{training_seed if training_seed is not None else 'base'}-{root['seed']}"
            row = {
                "method_id": method, "training_seed": training_seed, "canonical_group_id": root["canonical_group_id"],
                "environment_seed": int(root["seed"]), "policy_checkpoint_sha256": sha256_file(checkpoint),
                "initial_state_hash": root["initial_state_hash"], "protocol_sha256": protocol_sha,
                "helper_policy_loaded": False, "selector_loaded": False, "repair_policy_calls": 0,
                "takeover_count": 0, "helper_steps_actual": 0, "complete_record": False,
            }
            exception_reason = None
            actions = []
            successes = []
            first_raw = None
            stable_state = None
            try:
                obs = env.reset_canonical(_payload(root), int(root["seed"]))
                policy.start_episode()
                history = []
                for t in range(HORIZON):
                    action = policy.suggest_once(obs, t, root["canonical_group_id"])
                    if action.shape != (7,):
                        raise RuntimeError(f"action shape {action.shape} != (7,)")
                    actions.append(action.copy())
                    obs, _reward, success, _info = env.step(action)
                    successes.append(bool(success))
                    if first_raw is None and success:
                        first_raw = t + 1
                    history.append(bool(success))
                    if len(history) > STABLE:
                        history.pop(0)
                    if len(history) == STABLE and all(history):
                        stable_state = t + 1
                        break
            except Exception as exc:
                exception_reason = f"{type(exc).__name__}: {exc}"
            row.update({
                "engineering_ok": exception_reason is None, "no_help_task_success": stable_state is not None if exception_reason is None else None,
                "first_raw_success_state": first_raw, "stable_success_state": stable_state,
                "executed_env_steps": len(actions), "base_policy_calls": policy._calls,
                "wall_seconds": time.time() - started, "exception_reason": exception_reason,
                "job_id": job_id, "complete_record": True,
            })
            output_rows[key] = row
            write_table(sorted(output_rows.values(), key=lambda item: (item["canonical_group_id"], str(item.get("training_seed")))), result_path)
    finally:
        env.close()
    rows_out = sorted(output_rows.values(), key=lambda item: (item["canonical_group_id"], str(item.get("training_seed"))))
    summary = {
        "schema_version": "hb4_no_help_evaluation_summary_v1", "method_id": method, "training_seed": training_seed,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "roots": len(rows_out), "engineering_failures": sum(not bool(row.get("engineering_ok")) for row in rows_out),
        "successes": sum(bool(row.get("no_help_task_success")) for row in rows_out if row.get("engineering_ok")),
        "rows_path": str(result_path.resolve()), "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.roots, args.checkpoint, args.method, args.training_seed, args.output_dir, args.device, args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
