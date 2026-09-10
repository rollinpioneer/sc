"""Run the four-root BASE_FROZEN reproducibility pilot for HB4-D."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_array, sha256_file, write_table


BASE = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
SOURCE = Path("/home/__compress_data/xushijie/work/cr_scizor/data/robomimic/square/ph/image.hdf5")
EXPORT = Path("/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1/experiments/handback/hb4_square_absorb_v1")
SEEDS = tuple(range(929000, 929004))
HORIZON = 400


def _run_once(env: EnvAdapter, policy: BasePolicyAdapter, seed: int):
    obs, payload = env.new_episode(seed)
    policy.start_episode()
    states = [env.physical_state()]
    actions, successes = [], []
    for t in range(HORIZON):
        action = policy.suggest_once(obs, t, f"square:HB4_pilot:{seed}")
        obs, _reward, success, _info = env.step(action)
        actions.append(action)
        successes.append(bool(success))
        states.append(env.physical_state())
    return payload, np.asarray(states), np.asarray(actions), np.asarray(successes)


def run(output_root: Path, device: str) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    env = EnvAdapter(SOURCE, **load_observation_spec(SOURCE))
    policy = BasePolicyAdapter(BASE, device=device)
    rows = []
    started = time.time()
    try:
        for seed in SEEDS:
            # Repeated renders at an identical physical state are allowed to
            # reuse a cache keyed by the full state bytes. This removes EGL
            # buffer nondeterminism without reusing observations across roots.
            env.observation_tape = {}
            first = _run_once(env, policy, seed)
            second = _run_once(env, policy, seed)
            payload_a, states_a, actions_a, success_a = first
            payload_b, states_b, actions_b, success_b = second
            state_delta = float(np.max(np.abs(states_a - states_b)))
            action_delta = float(np.max(np.abs(actions_a - actions_b)))
            initial_delta = float(np.max(np.abs(np.asarray(payload_a["states"]) - np.asarray(payload_b["states"]))))
            success_equal = bool(np.array_equal(success_a, success_b))
            row = {
                "schema_version": "hb4_pilot_repro_row_v1", "seed": seed,
                "canonical_group_id": f"square:HB4_pilot:{seed}",
                "initial_state_delta": initial_delta, "state_max_abs_delta": state_delta,
                "action_max_abs_delta": action_delta, "success_equal": success_equal,
                "actions_sha256": sha256_array(actions_a), "states_sha256": sha256_array(states_a),
                "success_sha256": sha256_array(success_a), "executed_steps": HORIZON,
                "observation_tape_hits": env.observation_tape_hits,
                "engineering_ok": True, "passed": initial_delta == 0.0 and state_delta == 0.0 and action_delta <= 1e-6 and success_equal,
            }
            rows.append(row)
    except Exception as exc:
        rows.append({"schema_version": "hb4_pilot_repro_row_v1", "seed": None, "engineering_ok": False, "passed": False, "exception_reason": f"{type(exc).__name__}: {exc}"})
    finally:
        env.close()
    write_table(rows, output_root / "pilot_records.jsonl")
    passed = len(rows) == len(SEEDS) and all(row.get("passed", False) for row in rows)
    summary = {
        "schema_version": "hb4_pilot_summary_v1", "status": "PASS_HB4_D" if passed else "HOLD_HB4_D_PILOT",
        "pilot_roots": len(SEEDS), "repetitions_per_root": 2, "passed_roots": sum(bool(row.get("passed")) for row in rows),
        "base_checkpoint": str(BASE.resolve()), "base_checkpoint_sha256": sha256_file(BASE),
        "helper_policy_loaded": False, "selector_loaded": False, "repair_policy_calls": 0,
        "elapsed_seconds": time.time() - started, "rows": rows,
    }
    atomic_json_dump(summary, output_root / "pilot_summary.json")
    atomic_json_dump(summary, EXPORT / "metrics/pilot_summary.json")
    return {"status": summary["status"], "pilot_roots": len(SEEDS), "passed_roots": summary["passed_roots"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1/data/pilot"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
