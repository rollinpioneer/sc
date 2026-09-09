from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from recovery_handback.hb3p.metrics import episode_value
from recovery_handback.hb3p_stop.aggregate import aggregate
from recovery_handback.hb3p_stop.evaluate import evaluate
from recovery_handback.hb3p_stop.features import FEATURE_DIM, feature_vector
from recovery_handback.hb3p_stop.io import write_jsonl
from recovery_handback.hb3p_stop.online import _method_spec


class StopContinueReferences(unittest.TestCase):
    def test_feature_contract_is_non_privileged_and_fixed(self):
        proprio = np.zeros((4, 9), dtype=np.float32)
        actions = np.zeros((4, 7), dtype=np.float32)
        times = np.asarray([77, 78, 79, 80], dtype=np.int64)
        features = feature_vector(proprio, actions, times)
        self.assertEqual(features.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[-1]), 0.2)

    def test_frozen_method_intervals(self):
        self.assertEqual(_method_spec("FIXED_L60"), ("STOP", 60))
        self.assertEqual(_method_spec("FIXED_L80"), ("CONTINUE", 80))
        self.assertEqual(_method_spec("LEARNED_STOP_CONTINUE"), ("LEARNED", None))

    def test_label_uses_realized_utility_not_success_only(self):
        stop = {"engineering_ok": True, "takeover_count": 1, "helper_steps_actual": 60,
                "genuine_handoff_success": True, "system_success": True}
        continue_ = {"engineering_ok": True, "takeover_count": 1, "helper_steps_actual": 80,
                     "genuine_handoff_success": True, "system_success": True}
        self.assertGreater(episode_value(stop), episode_value(continue_))

    def test_downstream_pipeline_and_selected_branch_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes"
            roots = []
            protocol = {
                "protocol_sha256": "fixture-protocol",
                "semantic_pair_id": "fixture-pair",
                "test_seeds": [1, 2],
                "new_test_roots": 2,
                "lambda": 0.25,
                "horizon_steps": 400,
                "noninferiority_margin_absolute": 0.05,
                "noninferiority_comparator": "FIXED_L80",
                "statistics": {"bootstrap_repeats": 50, "seed": 7},
            }
            for seed in protocol["test_seeds"]:
                root_id = f"square:fixture:{seed}"
                base_states = np.zeros((101, 3), dtype=np.float64)
                base_suggestions = np.zeros((100, 7), dtype=np.float32)
                base_path = root / f"baseline_{seed}.npz"
                np.savez_compressed(
                    base_path,
                    states=base_states,
                    suggestions=base_suggestions,
                )
                roots.append({
                    "root_id": root_id,
                    "stat_group_id": root_id,
                    "seed": seed,
                    "role": "fixture",
                    "baseline_success": False,
                    "actual_steps": 100,
                    "first_raw_success_state": None,
                    "stable_success_state": None,
                    "initial_state_hash": f"state-{seed}",
                    "rollout_path": str(base_path),
                    "exception_reason": None,
                })
                for method, length in (("FIXED_L60", 60), ("FIXED_L80", 80), ("LEARNED_STOP_CONTINUE", 60)):
                    method_dir = episodes / "shard0" / method
                    method_dir.mkdir(parents=True, exist_ok=True)
                    states = base_states.copy()
                    actions = np.zeros((100, 7), dtype=np.float32)
                    base_actions = base_suggestions.copy()
                    helper = np.zeros(100, dtype=np.bool_)
                    helper[20:20 + length] = True
                    actions[helper, 0] = 0.5
                    trajectory = method_dir / f"{seed}.npz"
                    np.savez_compressed(
                        trajectory,
                        states=states,
                        actions=actions,
                        base_actions=base_actions,
                        helper_mask=helper,
                        success=np.zeros(100, dtype=np.bool_),
                    )
                    result = {
                        "root_id": root_id,
                        "stat_group_id": root_id,
                        "root_seed": seed,
                        "initial_state_hash": f"state-{seed}",
                        "method_id": method,
                        "protocol_hash": protocol["protocol_sha256"],
                        "semantic_pair_id": protocol["semantic_pair_id"],
                        "engineering_ok": True,
                        "exception_reason": None,
                        "system_success": False,
                        "genuine_handoff_success": False,
                        "takeover_count": 1,
                        "takeover_t": 20,
                        "selected_length": length,
                        "helper_steps_actual": length,
                        "repair_policy_calls": length,
                        "repair_calls_after_handoff": 0,
                        "changed_action_steps": length,
                        "first_raw_success_state": None,
                        "stable_success_state": None,
                        "handoff_executed": True,
                        "handoff_t": 20 + length,
                        "helper_completed_task": False,
                        "query_count": int(method == "LEARNED_STOP_CONTINUE"),
                        "queried_times": [80] if method == "LEARNED_STOP_CONTINUE" else [],
                        "inference_wall_seconds": 0.001 if method == "LEARNED_STOP_CONTINUE" else 0.0,
                        "stop_continue_decision": "STOP" if method == "LEARNED_STOP_CONTINUE" else None,
                        "trajectory_path": str(trajectory),
                        "decision_trace_path": None,
                    }
                    (method_dir / f"{seed}.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
            root_manifest = root / "roots.jsonl"
            write_jsonl(roots, root_manifest)
            output = root / "metrics"
            aggregate(root_manifest, episodes, output, protocol)
            coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["prefix_verified_roots"], 2)
            self.assertEqual(coverage["branch_parity_verified_roots"], 2)
            result = evaluate(output / "episodes.jsonl", output / "coverage.json", protocol, output)
            self.assertEqual(result["decision"]["status"], "PILOT_ONLY")
            self.assertEqual(
                result["decision"]["noninferiority_comparator"], "FIXED_L80"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
