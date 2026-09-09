from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recovery_handback.hb3p.metrics import episode_value
from recovery_handback.hb3p_stop.features import FEATURE_DIM, feature_vector
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

