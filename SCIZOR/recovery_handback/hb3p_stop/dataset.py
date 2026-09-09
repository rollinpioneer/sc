"""Build paired L60/L80 labels from the frozen HB3-P probe."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import sha256_array, sha256_file
from recovery_handback.hb3p.metrics import episode_value
from recovery_handback.hb3p_stop.features import audit_input_schema, feature_from_history
from recovery_handback.hb3p_stop.io import atomic_json_dump, write_jsonl


def _records(probe_root: Path, method: str) -> dict[str, dict]:
    result = {}
    for path in sorted((probe_root / "episodes").glob(f"**/{method}/*.json")):
        if path.name.endswith("_decision_trace.json") or path.name.endswith("_handoff.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        root_id = str(row["root_id"])
        if root_id in result:
            raise ValueError(f"duplicate {method} record: {root_id}")
        if not row.get("engineering_ok"):
            raise RuntimeError(f"probe record is not engineering-valid: {path}")
        result[root_id] = row
    return result


def build(probe_root: Path, output_dir: Path, *, horizon: int = 400, lambda_: float = 0.25) -> dict:
    l60 = _records(Path(probe_root), "FIXED_L60")
    l80 = _records(Path(probe_root), "FIXED_L80")
    if set(l60) != set(l80) or len(l60) < 2:
        raise RuntimeError("L60 and L80 must cover the same nonempty root set")
    rows = []
    features = []
    labels = []
    for root_id in sorted(l60):
        left, right = l60[root_id], l80[root_id]
        u60 = episode_value(left, lambda_, horizon)
        u80 = episode_value(right, lambda_, horizon)
        feature, audit = feature_from_history(Path(left["handoff_history_path"]), horizon)
        # CONTINUE is a paired, realized L80 utility improvement over STOP/L60.
        continue_label = int(u80 > u60 + 1e-8)
        rows.append({
            "root_id": root_id,
            "stat_group_id": str(left["stat_group_id"]),
            "label": continue_label,
            "label_name": "CONTINUE" if continue_label else "STOP",
            "utility_l60": float(u60),
            "utility_l80": float(u80),
            "utility_delta_l80_minus_l60": float(u80 - u60),
            "l60_autonomous_completion": bool(left["genuine_handoff_success"]),
            "l80_autonomous_completion": bool(right["genuine_handoff_success"]),
            "l60_system_success": bool(left["system_success"]),
            "l80_system_success": bool(right["system_success"]),
            "handoff_history_path": audit["history_path"],
            "absolute_times": audit["absolute_times"],
            "feature_sha256": audit["feature_sha256"],
        })
        features.append(feature)
        labels.append(continue_label)
    features_array = np.asarray(features, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if features_array.ndim != 2 or features_array.shape[1] != 65 or not np.isfinite(features_array).all():
        raise ValueError(f"invalid paired feature matrix: {features_array.shape}")
    # Deterministic root-level folds. No frames from a root cross a fold boundary.
    for index, row in enumerate(rows):
        row["fold"] = int(index % 5)
        row["split"] = "oof_holdout" if row["fold"] == 0 else "oof_train_candidate"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "paired_dataset.npz", features=features_array, labels=labels_array)
    write_jsonl(rows, output_dir / "paired_labels.jsonl")
    atomic_json_dump({
        "schema_version": "hb3p_stop_continue_paired_dataset_v1",
        "probe_root": str(Path(probe_root).resolve()),
        "probe_frozen_protocol_sha256": sha256_file(Path(probe_root) / "config/frozen_protocol.json"),
        "roots": len(rows),
        "continue_labels": int(labels_array.sum()),
        "stop_labels": int((labels_array == 0).sum()),
        "label_rule": "CONTINUE iff realized utility(FIXED_L80) > realized utility(FIXED_L60) by >1e-8",
        "utility": {"lambda": lambda_, "denominator": horizon},
        "input_schema": audit_input_schema(),
        "feature_shape": list(features_array.shape),
        "feature_array_sha256": sha256_array(features_array),
        "label_array_sha256": sha256_array(labels_array),
        "dataset_file_sha256": sha256_file(output_dir / "paired_dataset.npz"),
    }, output_dir / "dataset_manifest.json")
    return {
        "output_dir": str(output_dir.resolve()),
        "roots": len(rows),
        "continue_labels": int(labels_array.sum()),
        "stop_labels": int((labels_array == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(build(args.probe_root, args.output_dir, horizon=args.horizon, lambda_=args.lambda_), indent=2))


if __name__ == "__main__":
    main()
