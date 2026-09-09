from __future__ import annotations

import unittest

from recovery_handback.hb3p.controller import (
    Rule,
    SingleIntervention,
    select_episode_from_cached_predictions,
)
from recovery_handback.hb3p.metrics import binary_ranking, episode_value, paired_bootstrap

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
