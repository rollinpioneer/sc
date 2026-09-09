"""Fill the HB2 AUROC/AP reporting gap without changing any HB2 decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb3p.io import mark
from recovery_handback.hb3p.metrics import (
    binary_brier,
    binary_nll,
    binary_ranking,
    equal_root_weights,
)

OUTPUTS = {
    "p0": "y0",
    "p_sys_5": "y_sys_l5",
    "p_sys_20": "y_sys_l20",
    "p_sys_80": "y_sys_l80",
    "p_genuine_5": "y_genuine_l5",
    "p_genuine_20": "y_genuine_l20",
    "p_genuine_80": "y_genuine_l80",
    "p_full": "y_full",
}


def clarify(hb2_root: Path, output_dir: Path) -> dict:
    truth_rows = [
        row for row in read_table(hb2_root / "data/test_attached/anchor_examples.parquet")
        if bool(row.get("complete_pair"))
    ]
    truth = {str(row["example_id"]): row for row in truth_rows}
    if len(truth) != len(truth_rows):
        raise ValueError("duplicate truth example_id")
    original = json.loads((hb2_root / "metrics/test/probability_metrics.json").read_text())
    results = {
        "schema_version": "hb3p_hb2_ranking_clarification_v1",
        "hb2_status_preserved": "HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN",
        "selection_changed": False,
        "models": {},
    }
    for prediction_path in sorted((hb2_root / "predictions/test").glob("*.json")):
        if prediction_path.name == "summary.json" or prediction_path.name.startswith("handoff_"):
            continue
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        model_key = f"{payload['model']}/{payload['seed']}"
        records = payload["records"]
        seen = {str(row["example_id"]) for row in records}
        missing = sorted(set(truth) - seen)
        extra = sorted(seen - set(truth))
        if len(seen) != len(records):
            raise ValueError(f"duplicate prediction example_id: {model_key}")
        model_result = {"records": len(records), "missing_predictions": missing, "extra_predictions": extra, "outputs": {}}
        for probability_key, truth_key in OUTPUTS.items():
            joined = [(truth[str(row["example_id"])], row) for row in records if str(row["example_id"]) in truth]
            y = [int(left[truth_key]) for left, _ in joined]
            p = [float(right[probability_key]) for _, right in joined]
            groups = [str(left["stat_group_id"]) for left, _ in joined]
            weights = equal_root_weights(groups)
            pooled = binary_ranking(y, p)
            root_equal = binary_ranking(y, p, weights)
            original_metric = original.get(model_key, {}).get(probability_key, {})
            positive_groups = {group for group, value in zip(groups, y) if value}
            negative_groups = {group for group, value in zip(groups, y) if not value}
            model_result["outputs"][probability_key] = {
                "valid_samples": len(y),
                "positive_samples": int(sum(y)),
                "negative_samples": int(len(y) - sum(y)),
                "positive_roots": len(positive_groups),
                "negative_roots": len(negative_groups),
                "anchor_pooled": pooled,
                "root_equal_weighted": root_equal,
                "original_root_equal_brier": original_metric.get("brier"),
                "original_root_equal_nll": original_metric.get("nll"),
                "recomputed_anchor_pooled_brier": binary_brier(y, p),
                "recomputed_anchor_pooled_nll": binary_nll(y, p),
                "small_sample": min(len(positive_groups), len(negative_groups)) < 5,
            }
        results["models"][model_key] = model_result
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(results, output_dir / "ranking_metrics.json")
    notes = """# HB2 Report Notes\n\n- HB2 status remains `HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN`; ranking completion does not rerun selection.\n- `local_feature_gain`, `history_gain`, and `paired_supervision_gain` are validation single-seed point-estimate comparisons, not independent test confirmation.\n- Historical interference denominators differed by the anchors each method actually helped. HB3-P reports both all baseline-success roots and helped baseline-success roots.\n- Historical files named `*.csv` may contain JSONL because the old generic writer keyed only on `.parquet`; HB3-P emits standards-compliant CSV.\n- AUROC/AP below use tied-score threshold groups. `anchor_pooled` and `root_equal_weighted` are distinct estimands. Single-class outputs remain NA with an explicit reason.\n"""
    (output_dir / "HB2_REPORT_NOTES.md").write_text(notes, encoding="utf-8")
    mark(output_dir.parents[1], "hb3p-A.done", "HB2 ranking metrics clarified without reselection")
    return {"models": len(results["models"]), "truth_rows": len(truth), "output": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hb2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(clarify(args.hb2_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
