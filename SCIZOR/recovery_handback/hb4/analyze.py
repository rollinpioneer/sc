"""Analyze root-paired HB4 development or formal no-help records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, write_table


ARMS = ("REPLAY_ONLY", "MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY", "HANDOFF_RECOVERY")
# The handoff arm is the method under test, not a comparator against itself.
COMPARATORS = ("BASE_FROZEN",) + tuple(arm for arm in ARMS if arm != "HANDOFF_RECOVERY")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _success(row: dict) -> float:
    return float(bool(row.get("no_help_task_success")))


def _mean_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = values[rng.integers(0, values.size, size=(repeats, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analyze(records_root: Path, output_dir: Path, expected_roots: int, mode: str) -> dict:
    files = sorted(records_root.glob("*/episodes.jsonl"))
    rows = []
    for path in files:
        rows.extend(_rows(path))
    by_method: dict[str, dict[int | None, dict[str, float]]] = {}
    engineering_failures = 0
    for row in rows:
        if not row.get("engineering_ok"):
            engineering_failures += 1
            continue
        by_method.setdefault(row["method_id"], {}).setdefault(row.get("training_seed"), {})[row["canonical_group_id"]] = _success(row)
    expected_keys = {("BASE_FROZEN", None)} | {(arm, seed) for arm in ARMS for seed in (0, 1, 2)}
    observed_keys = [(row.get("method_id"), row.get("training_seed"), row.get("canonical_group_id")) for row in rows]
    duplicate_records = len(observed_keys) - len(set(observed_keys))
    expected_method_roots = {
        (method, seed): expected_roots
        for method, seed in expected_keys
    }
    observed_method_roots = {
        (method, seed): len(values)
        for method, seeds in by_method.items()
        for seed, values in seeds.items()
    }
    missing_method_keys = sorted(
        f"{method}:{seed}"
        for method, seed in expected_keys
        if observed_method_roots.get((method, seed), 0) != expected_roots
    )
    unexpected_method_keys = sorted(
        f"{method}:{seed}"
        for method, seed in observed_method_roots
        if (method, seed) not in expected_method_roots
    )
    root_ids = {row.get("canonical_group_id") for row in rows}
    expected_records = expected_roots * len(expected_keys)
    coverage = {
        "mode": mode,
        "expected_roots": expected_roots,
        "expected_records": expected_records,
        "records": len(rows),
        "unique_record_keys": len(set(observed_keys)),
        "duplicate_records": duplicate_records,
        "unique_roots": len(root_ids),
        "engineering_failures": engineering_failures,
        "methods": {method: {str(seed): len(values) for seed, values in seeds.items()} for method, seeds in by_method.items()},
        "missing_method_keys": missing_method_keys,
        "unexpected_method_keys": unexpected_method_keys,
        "complete": (
            len(rows) == expected_records
            and len(set(observed_keys)) == expected_records
            and duplicate_records == 0
            and len(root_ids) == expected_roots
            and engineering_failures == 0
            and not missing_method_keys
            and not unexpected_method_keys
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(coverage, output_dir / "coverage.json")
    if not by_method:
        return coverage
    base_seed = next(iter(by_method.get("BASE_FROZEN", {})), None)
    base = by_method["BASE_FROZEN"][base_seed]
    root_ids = sorted(base)
    rng = np.random.default_rng(20260910)
    methods = {"BASE_FROZEN": {"training_seed": None, "root_mean": float(np.mean([base[r] for r in root_ids]))}}
    for arm in ARMS:
        seed_stats = {}
        for seed in (0, 1, 2):
            values = by_method.get(arm, {}).get(seed, {})
            paired = np.asarray([values[root] for root in root_ids if root in values and root in base], dtype=float)
            seed_stats[str(seed)] = {"roots": int(paired.size), "success_rate": float(paired.mean()) if paired.size else None, "gain_vs_base": float((paired - np.asarray([base[root] for root in root_ids if root in values and root in base])).mean()) if paired.size else None}
        per_root = np.asarray([np.mean([by_method[arm][seed][root] for seed in (0, 1, 2)]) - base[root] for root in root_ids if all(root in by_method.get(arm, {}).get(seed, {}) for seed in (0, 1, 2))], dtype=float)
        mean, lo, hi = _mean_ci(per_root, rng, 5000)
        methods[arm] = {"seed_stats": seed_stats, "root_mean_success_rate": float(np.mean([base[root] + diff for root, diff in zip(root_ids, per_root)])) if per_root.size else None, "gain_vs_base": mean, "gain_ci95": [lo, hi], "paired_roots": int(per_root.size)}
    comparisons = []
    for comparator in COMPARATORS:
        if comparator == "BASE_FROZEN":
            diff = np.asarray([np.mean([by_method["HANDOFF_RECOVERY"][seed][root] for seed in (0, 1, 2)]) - base[root] for root in root_ids if all(root in by_method.get("HANDOFF_RECOVERY", {}).get(seed, {}) for seed in (0, 1, 2))], dtype=float)
        else:
            diff = np.asarray([np.mean([by_method["HANDOFF_RECOVERY"][seed][root] for seed in (0, 1, 2)]) - np.mean([by_method[comparator][seed][root] for seed in (0, 1, 2)]) for root in root_ids if all(root in by_method.get("HANDOFF_RECOVERY", {}).get(seed, {}) and root in by_method.get(comparator, {}).get(seed, {}) for seed in (0, 1, 2))], dtype=float)
        mean, lo, hi = _mean_ci(diff, rng, 5000)
        comparisons.append({"comparison": f"HANDOFF_RECOVERY-{comparator}", "paired_roots": int(diff.size), "estimate": mean, "ci95": [lo, hi], "lower_bound_gt_zero": bool(lo > 0) if diff.size else False})
    handoff = methods.get("HANDOFF_RECOVERY", {})
    handoff_seed_stats = handoff.get("seed_stats", {})
    positive_vs_base = sum(float(handoff_seed_stats.get(str(seed), {}).get("gain_vs_base") or 0.0) > 0 for seed in (0, 1, 2))
    # Seed-level signs are computed against the same root set, not pooled rows.
    seed_control_gains = {}
    for control in ("MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY"):
        seed_control_gains[control] = {}
        for seed in (0, 1, 2):
            common = sorted(set(by_method.get("HANDOFF_RECOVERY", {}).get(seed, {})) & set(by_method.get(control, {}).get(seed, {})))
            seed_control_gains[control][str(seed)] = float(np.mean([by_method["HANDOFF_RECOVERY"][seed][root] - by_method[control][seed][root] for root in common])) if common else None
    handoff_mean = handoff.get("root_mean_success_rate")
    controls_not_better = all(handoff_mean is not None and handoff_mean >= methods.get(control, {}).get("root_mean_success_rate", float("inf")) for control in ARMS if control != "HANDOFF_RECOVERY")
    gate = {
        "engineering_valid": bool(coverage["complete"]),
        "handoff_point_gain_vs_base_at_least_0_05": bool((handoff.get("gain_vs_base") or -1.0) >= 0.05),
        "handoff_mean_not_below_trained_controls": controls_not_better,
        "handoff_positive_seeds_vs_base_at_least_2": positive_vs_base >= 2,
        "handoff_positive_seeds_vs_base": positive_vs_base,
        "seed_mean_gains_vs_controls": seed_control_gains,
        "development_go": bool(coverage["complete"] and (handoff.get("gain_vs_base") or -1.0) >= 0.05 and controls_not_better and positive_vs_base >= 2) if mode == "development" else None,
    }
    atomic_json_dump({"schema_version": "hb4_analysis_summary_v1", "mode": mode, "coverage": coverage, "methods": methods, "comparisons": comparisons, "gate": gate}, output_dir / "summary.json")
    if mode == "development":
        atomic_json_dump({"schema_version": "hb4_development_decision_v1", "decision": "HB4_DEVELOPMENT_GO" if gate["development_go"] else "HB4_DEVELOPMENT_NO_GO", "gate": gate}, output_dir / "decision.json")
    else:
        passed = coverage["complete"] and all(item["lower_bound_gt_zero"] for item in comparisons) and positive_vs_base == 3 and all(sum((value or 0.0) > 0 for value in seed_control_gains[c].values()) >= 2 for c in seed_control_gains)
        atomic_json_dump({"schema_version": "hb4_formal_decision_v1", "decision": "HB4_HANDOFF_DATA_ADVANTAGE_SUPPORTED" if passed else "HB4_FORMAL_COMPARISON_NOT_FULLY_SUPPORTED", "gate": gate, "comparisons": comparisons}, output_dir / "decision.json")
    write_table(comparisons, output_dir / "paired_comparisons.csv")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-roots", type=int, required=True)
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.records_root, args.output_dir, args.expected_roots, args.mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
