import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _onset_rate(rows):
    total = len(rows)
    if total == 0:
        return 0.0
    hits = sum(1 for row in rows if row.get("failure_onset") is not None)
    return hits / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--target-min-rate", type=float, default=0.30)
    p.add_argument("--target-max-rate", type=float, default=0.70)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = []
    for path in args.metadata:
        rows.extend(_load_rows(path))
    grouped = defaultdict(list)
    tasks = set()
    perturbation_types = set()
    for row in rows:
        task = row["task"]
        perturbation = row["perturbation_type"]
        magnitude = float(row["magnitude"])
        grouped[(task, perturbation, magnitude)].append(row)
        tasks.add(task)
        perturbation_types.add(perturbation)

    magnitudes = sorted({float(row["magnitude"]) for row in rows})
    summary = {
        "target_min_rate": args.target_min_rate,
        "target_max_rate": args.target_max_rate,
        "selected": {},
        "rates": {},
    }

    for task in sorted(tasks):
        summary["selected"][task] = {}
        summary["rates"][task] = {}
        for perturbation in sorted(perturbation_types):
            rates = {}
            for magnitude in magnitudes:
                key = (task, perturbation, magnitude)
                rate = _onset_rate(grouped.get(key, []))
                rates[str(magnitude)] = rate
            summary["rates"][task][perturbation] = rates

            chosen = None
            above_floor = []
            for magnitude in magnitudes:
                rate = rates[str(magnitude)]
                if rate >= args.target_min_rate:
                    above_floor.append((magnitude, rate))
                    if chosen is None:
                        chosen = magnitude
            if chosen is None:
                chosen = 1.0 if all(rates[str(m)] < args.target_min_rate for m in magnitudes) else magnitudes[0]
            summary["selected"][task][perturbation] = {
                "magnitude": float(chosen),
                "rates": rates,
                "within_target_window": bool(args.target_min_rate <= rates.get(str(chosen), 0.0) <= args.target_max_rate),
                "any_candidate_over_floor": bool(above_floor),
                "fallback_added_1.0": bool(chosen == 1.0),
            }

    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
