from __future__ import annotations

import unittest

from recovery_handback.hb3p.controller import (
    Rule,
    SingleIntervention,
    select_episode_from_cached_predictions,
)
from recovery_handback.hb3p.evaluate import _mean, _optional_int
from recovery_handback.hb3p.exit_diagnosis import _oracle_component, _pair_case
from recovery_handback.hb3p.metrics import binary_ranking, episode_value, paired_bootstrap
from recovery_handback.hb3p.report import _fmt

NO = {"p0": 0.9, "p_genuine_5": 0.01, "p_genuine_20": 0.02, "p_genuine_80": 0.03}
YES = {"p0": 0.01, "p_genuine_5": 0.02, "p_genuine_20": 0.9, "p_genuine_80": 0.1}


class References(unittest.TestCase):
    def test_tie_prefers_no_help(self):
        probabilities = {
            "p0": 0.2,
            "p_genuine_5": 0.2 + 0.25 * 5 / 400,
            "p_genuine_20": 0.2 + 0.25 * 20 / 400,
            "p_genuine_80": 0.2 + 0.25 * 80 / 400,
        }
        controller = SingleIntervention(Rule("model"))
        for t in range(20):
            controller.owner_before_action(t)
        self.assertEqual(controller.owner_before_action(20, probabilities)[0], "base")

    def test_wait_then_one_help_then_permanent_handoff(self):
        controller = SingleIntervention(Rule("model", "M1_proprio"))
        owners = []
        for t in range(400):
            probabilities = (NO if t == 20 else YES) if controller.needs_prediction(t) else None
            owners.append(controller.owner_before_action(t, probabilities)[0])
        self.assertEqual([i for i, owner in enumerate(owners) if owner == "repair"], list(range(80, 100)))
        self.assertEqual((controller.takeovers, controller.query_count, controller.handoff_t), (1, 2, 100))

    def test_early_success_blocks_all_help(self):
        controller = SingleIntervention(Rule("scheduled", scheduled_t=20, scheduled_length=80))
        for t in range(120):
            self.assertEqual(controller.owner_before_action(t, raw_success_seen=t >= 19)[0], "base")
        self.assertEqual(controller.takeovers, 0)

    def test_scheduled_interval_and_missing_prediction(self):
        controller = SingleIntervention(Rule("scheduled", scheduled_t=20, scheduled_length=5))
        owners = [controller.owner_before_action(t)[0] for t in range(40)]
        self.assertEqual(owners.count("repair"), 5)
        for length in (40, 60):
            controller = SingleIntervention(Rule("scheduled", scheduled_t=20, scheduled_length=length))
            owners = [controller.owner_before_action(t)[0] for t in range(120)]
            self.assertEqual(owners.count("repair"), length)
        missing = SingleIntervention(Rule("model"))
        for t in range(20):
            missing.owner_before_action(t)
        with self.assertRaises(ValueError):
            missing.owner_before_action(20)

    def test_reconstruction_does_not_choose_future_best(self):
        result = select_episode_from_cached_predictions(
            Rule("model"), {20: YES, 80: NO, 160: NO}
        )
        self.assertEqual((result["takeover_t"], result["length"]), (20, 20))

    def test_tied_ranking_order_invariant(self):
        left = binary_ranking([1, 0], [0.5, 0.5])
        right = binary_ranking([0, 1], [0.5, 0.5])
        self.assertEqual(left, right)
        self.assertEqual((left["auroc"], left["average_precision"]), (0.5, 0.5))

    def test_perfect_and_reversed_ranking(self):
        self.assertEqual(binary_ranking([0, 1], [0.0, 1.0])["auroc"], 1.0)
        self.assertEqual(binary_ranking([0, 1], [1.0, 0.0])["auroc"], 0.0)
        self.assertEqual(binary_ranking([0, 1], [1.0, 0.0])["average_precision"], 0.5)
        self.assertIsNone(binary_ranking([1, 1], [0.1, 0.9])["auroc"])

    def test_no_help_and_genuine_distinct(self):
        common = {
            "engineering_ok": True,
            "system_success": True,
            "genuine_handoff_success": False,
            "helper_steps_actual": 0,
            "takeover_count": 0,
        }
        self.assertEqual(episode_value(common), 1.0)
        self.assertAlmostEqual(episode_value({**common, "takeover_count": 1, "helper_steps_actual": 80}), -0.05)
        self.assertAlmostEqual(
            episode_value({**common, "takeover_count": 1, "helper_steps_actual": 80, "genuine_handoff_success": True}),
            0.95,
        )

    def test_paired_interval_identity_and_missing_root(self):
        values = {"a": 0.2, "b": 0.3}
        self.assertEqual(paired_bootstrap(values, values)["ci95_percentile"], [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired_bootstrap(values, {"a": 0.2})

    def test_missing_numeric_values_are_normalized(self):
        self.assertIsNone(_optional_int(None))
        self.assertIsNone(_optional_int(float("nan")))
        self.assertEqual(_optional_int(20.0), 20)
        self.assertIsNone(_mean([{"value": None}, {"value": float("nan")}], "value"))
        self.assertEqual(_mean([{"value": float("nan")}, {"value": 0.25}], "value"), 0.25)

    def test_stop_continue_pair_cases(self):
        self.assertEqual(_pair_case(False, True), "stop_failure_continue_success")
        self.assertEqual(_pair_case(True, False), "stop_success_continue_failure")
        self.assertEqual(_pair_case(True, True), "stop_success_continue_success")
        self.assertEqual(_pair_case(False, False), "both_failure")

    def test_oracle_component_decomposes_success_and_cost(self):
        fixed = {"utility": 0.8, "autonomous_completion": True}
        self.assertEqual(
            _oracle_component({"utility": 0.9, "autonomous_completion": True}, fixed),
            "success_with_lower_cost",
        )
        self.assertEqual(
            _oracle_component({"utility": 0.1, "autonomous_completion": False}, {"utility": 0.0, "autonomous_completion": False}),
            "both_failed_cost_only",
        )

    def test_report_number_formatting(self):
        self.assertEqual(_fmt(None), "NA")
        self.assertEqual(_fmt(0.25), "0.2500")
        self.assertEqual(_fmt(-0.068457, 6), "-0.068457")


if __name__ == "__main__":
    unittest.main(verbosity=2)
