import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _safe_int(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()

    rows = _load_rows(args.metadata)
    task_failure = Counter()
    perturb_mag = Counter()
    onset_delays = []
    failure_types = Counter()
    final_success = Counter()
    delta_norms = []
    label_status = Counter()

    for row in rows:
        task_failure[(row["task"], row["failure_type"])] += 1
        perturb_mag[(row["perturbation_type"], float(row["magnitude"]))] += 1
        failure_types[row["failure_type"]] += 1
        label_status[row.get("label_status", "") or "unknown"] += 1
        final_success[(bool(row.get("final_success_clean", False)), bool(row.get("final_success_perturbed", False)))] += 1
        delta_norms.append(float(row.get("action_delta_norm", 0.0)))
        onset = _safe_int(row.get("failure_onset"))
        perturb_t = _safe_int(row.get("perturb_t"))
        if onset is not None and perturb_t is not None:
            onset_delays.append(onset - perturb_t)

    onset_arr = np.asarray(onset_delays, dtype=np.float32) if onset_delays else np.asarray([], dtype=np.float32)
    delta_arr = np.asarray(delta_norms, dtype=np.float32) if delta_norms else np.asarray([], dtype=np.float32)

    summary = {
        "pair_count": len(rows),
        "task_failure_counts": {f"{task}::{failure}": count for (task, failure), count in sorted(task_failure.items())},
        "perturbation_magnitude_counts": {f"{perturbation}::{magnitude}": count for (perturbation, magnitude), count in sorted(perturb_mag.items())},
        "failure_type_counts": dict(sorted(failure_types.items())),
        "label_status_counts": dict(sorted(label_status.items())),
        "final_success_counts": {f"clean={clean},perturbed={perturbed}": count for (clean, perturbed), count in sorted(final_success.items())},
        "ambiguous_rate": float(failure_types.get("ambiguous", 0) / max(1, len(rows))),
        "recovery_success_rate": float(failure_types.get("recovery_success", 0) / max(1, len(rows))),
        "recovery_failure_rate": float(failure_types.get("recovery_failure", 0) / max(1, len(rows))),
        "onset_delay_mean": float(onset_arr.mean()) if len(onset_arr) else None,
        "onset_delay_std": float(onset_arr.std()) if len(onset_arr) else None,
        "onset_delay_min": float(onset_arr.min()) if len(onset_arr) else None,
        "onset_delay_max": float(onset_arr.max()) if len(onset_arr) else None,
        "action_delta_norm_mean": float(delta_arr.mean()) if len(delta_arr) else None,
        "action_delta_norm_std": float(delta_arr.std()) if len(delta_arr) else None,
        "action_delta_norm_min": float(delta_arr.min()) if len(delta_arr) else None,
        "action_delta_norm_max": float(delta_arr.max()) if len(delta_arr) else None,
    }

    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with Path(args.output_csv).open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in summary.items():
            if isinstance(value, dict):
                f.write(f"{key},{json.dumps(value, sort_keys=True)}\n")
            else:
                f.write(f"{key},{json.dumps(value)}\n")


if __name__ == "__main__":
    main()
