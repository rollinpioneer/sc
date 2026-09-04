"""Three directional smoke checks for Stage 1D metrics."""

import pandas as pd

from .evaluate_localization import _measure


def _case(scores, recovery=False):
    return pd.DataFrame({
        "pair_id": ["case"] * 8, "task": ["can"] * 8, "base_demo_id": ["demo"] * 8, "t": list(range(8)),
        "score": scores, "is_responsible_point": [False, False, False, True, False, False, False, False],
        "is_responsibility_region": [False, False, True, True, True, False, False, False],
        "is_recovery": [False, False, False, False, False, recovery, recovery, False],
        "is_rare": [False] * 8, "is_innocent_downstream": [False] * 8,
        "is_no_effect_intervention": [False] * 8,
    })


def main():
    perfect = _measure(_case([0, 0, 0.9, 1.0, 0.8, 0, 0, 0], recovery=True), 0.75)
    delayed = _measure(_case([0, 0, 0, 0.2, 0, 0, 0, 1.0]), 0.75)
    recovery_deleted = _measure(_case([0, 0, 0, 0.1, 0, 1.0, 1.0, 0], recovery=True), 0.75)
    delayed_f1 = delayed["transition_f1"] or 0.0
    if not (perfect["transition_f1"] > delayed_f1 and perfect["responsibility_region_iou"] > delayed["responsibility_region_iou"]):
        raise RuntimeError("perfect localization did not outrank delayed localization")
    if not recovery_deleted["recovery_retention"] < perfect["recovery_retention"]:
        raise RuntimeError("recovery deletion did not reduce retention")
    if delayed["mean_localization_delay"] <= 0:
        raise RuntimeError("delayed prediction has non-positive localization delay")
    print({"passed": True, "perfect": perfect, "delayed": delayed, "recovery_deleted": recovery_deleted})


if __name__ == "__main__":
    main()
