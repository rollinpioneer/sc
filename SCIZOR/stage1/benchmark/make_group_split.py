"""Create a deterministic, label-stratified group split for Stage 1C."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test")
EFFECTIVE_TYPES = {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["base_demo_id"])].append(row)
    result = []
    for (task, base_demo_id), records in grouped.items():
        non_ambiguous = [row for row in records if row.get("label_status") != "ambiguous"]
        result.append({
            "task": task, "base_demo_id": base_demo_id, "pair_ids": sorted(row["pair_id"] for row in records),
            "final_failure": sum(not row.get("final_success_perturbed", True) for row in non_ambiguous),
            "recovery_success": sum(row["failure_type"] == "recovery_success" for row in non_ambiguous),
            "effective": sum(row["failure_type"] in EFFECTIVE_TYPES for row in non_ambiguous),
        })
    return result


def _assign_task(groups, ratios, rng):
    n = len(groups)
    targets = {"train": round(n * ratios["train"]), "validation": round(n * ratios["validation"])}
    targets["test"] = n - targets["train"] - targets["validation"]
    features = ("final_failure", "recovery_success", "effective")
    totals = {feature: sum(group[feature] for group in groups) for feature in features}
    ideal = {feature: {split: totals[feature] * targets[split] / n for split in SPLITS} for feature in features}
    tie = {id(group): float(rng.random()) for group in groups}
    ordered = sorted(groups, key=lambda group: (-group["final_failure"] - group["recovery_success"] - 0.1 * group["effective"], tie[id(group)], group["base_demo_id"]))
    assignments, split_counts = {}, Counter()
    observed = {split: Counter() for split in SPLITS}
    for group in ordered:
        candidates = [split for split in SPLITS if split_counts[split] < targets[split]]
        def cost(split):
            # Measure the post-assignment error across all splits. Considering
            # only the selected split overfills validation/test with signals.
            balance = sum(
                ((observed[target_split][feature] + (group[feature] if target_split == split else 0) - ideal[feature][target_split]) / max(1, ideal[feature][target_split])) ** 2
                for target_split in SPLITS for feature in features
            )
            coverage = sum(2.0 for feature in features if split in {"validation", "test"} and group[feature] > 0 and observed[split][feature] == 0)
            return balance - coverage
        split = min(candidates, key=lambda name: (cost(name), SPLITS.index(name)))
        assignments[(group["task"], group["base_demo_id"])] = split
        split_counts[split] += 1
        for feature in features:
            observed[split][feature] += group[feature]
    return assignments, targets, observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ratios = {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio}
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to 1")
    rows, groups, rng = _rows(args.metadata), None, np.random.default_rng(args.seed)
    groups = _groups(rows)
    assignments, target_groups, observed = {}, {}, {}
    for task in sorted({group["task"] for group in groups}):
        task_assignments, targets, task_observed = _assign_task([group for group in groups if group["task"] == task], ratios, rng)
        assignments.update(task_assignments)
        target_groups[task] = targets
        observed[task] = {split: dict(counts) for split, counts in task_observed.items()}
    split_pairs, split_groups = {split: [] for split in SPLITS}, {split: [] for split in SPLITS}
    for group in sorted(groups, key=lambda item: (item["task"], item["base_demo_id"])):
        split = assignments[(group["task"], group["base_demo_id"])]
        split_pairs[split].extend(group["pair_ids"])
        split_groups[split].append(f"{group['task']}:{group['base_demo_id']}")
    coverage = {}
    for split in SPLITS:
        assigned_groups = [group for group in groups if assignments[(group["task"], group["base_demo_id"])] == split]
        coverage[split] = {
            "tasks": sorted({group["task"] for group in assigned_groups}), "group_count": len(assigned_groups), "pair_count": len(split_pairs[split]),
            "final_failure_pair_count": sum(group["final_failure"] for group in assigned_groups),
            "recovery_success_pair_count": sum(group["recovery_success"] for group in assigned_groups),
            "effective_pair_count": sum(group["effective"] for group in assigned_groups),
        }
    missing = [f"{split}:{field}" for split in ("validation", "test") for field in ("final_failure_pair_count", "recovery_success_pair_count", "effective_pair_count") if coverage[split][field] == 0]
    if missing or any(set(coverage[split]["tasks"]) != {"can", "square"} for split in SPLITS):
        raise RuntimeError(f"split coverage requirement not met: {missing}")
    manifest = {
        "seed": args.seed, "group_key": ["task", "base_demo_id"], "ratios": ratios,
        "source_benchmark_sha256": _sha256(args.metadata),
        "train": sorted(split_pairs["train"]), "validation": sorted(split_pairs["validation"]), "test": sorted(split_pairs["test"]),
        "group_assignments": {f"{task}:{base_demo_id}": split for (task, base_demo_id), split in sorted(assignments.items())},
        "counts": {"target_group_counts_by_task": target_groups, "coverage": coverage, "feature_counts_by_task": observed},
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
