"""Create the frozen Stage 1D leaderboard and concise stage report."""

import argparse
import json
from pathlib import Path

import pandas as pd


LEADERBOARD_COLUMNS = ["method", "transition_f1", "responsibility_region_iou", "mean_abs_localization_delay", "top1_within_1", "top5_hit", "recovery_retention", "innocent_downstream_retention", "expert_retention", "rare_retention", "no_effect_false_attribution_rate", "test_delete_rate"]
SUBGROUP_COLUMNS = ["method", "group", "pair_count", "transition_count", "transition_f1", "responsibility_region_iou", "recovery_retention", "no_effect_false_attribution_rate"]


def _markdown_table(frame):
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in frame.itertuples(index=False, name=None):
        values = []
        for value in record:
            if pd.isna(value):
                text = "NA"
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _subgroup_table(metrics, dimension):
    rows = []
    for method, detail in metrics["methods"].items():
        for group, values in detail.get(dimension, {}).items():
            rows.append({"method": method, "group": group, **{column: values.get(column) for column in SUBGROUP_COLUMNS[2:]}})
    return _markdown_table(pd.DataFrame(rows, columns=SUBGROUP_COLUMNS)) if rows else "No rows."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-stats", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    stats = json.loads(Path(args.benchmark_stats).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    operating = json.loads(Path(args.operating_points).read_text(encoding="utf-8"))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    rows = []
    for method, detail in metrics["methods"].items():
        overall = detail["overall"]
        rows.append({
            "method": method, "transition_f1": overall.get("transition_f1"), "responsibility_region_iou": overall.get("responsibility_region_iou"),
            "mean_abs_localization_delay": overall.get("mean_abs_localization_delay"), "top1_within_1": overall.get("top1_within_1"), "top5_hit": overall.get("top5_hit"),
            "recovery_retention": overall.get("recovery_retention"), "innocent_downstream_retention": overall.get("innocent_downstream_retention"),
            "expert_retention": overall.get("expert_retention"), "rare_retention": overall.get("rare_retention"),
            "no_effect_false_attribution_rate": overall.get("no_effect_false_attribution_rate"), "test_delete_rate": overall.get("delete_rate"),
        })
    leaderboard = pd.DataFrame(rows)[LEADERBOARD_COLUMNS]
    Path(args.leaderboard).parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.leaderboard, index=False)
    table = _markdown_table(leaderboard)
    report = f"""# Stage 1 Report

## Frozen inputs

- Benchmark pairs: {stats['pair_count']}; failure-type counts: `{stats['failure_type_counts']}`.
- Split seed: {manifest['seed']}; group key: `{manifest['group_key']}`; metadata SHA-256: `{manifest['source_benchmark_sha256']}`.
- Checkpoint: `{Path(args.checkpoint_hash).read_text(encoding='utf-8').strip()}`.
- Repository record: `{Path(args.repo_commit).read_text(encoding='utf-8').strip()}`.
- Operating points were selected only on `{operating['selection_split']}` at the original-score percentile `{operating['reference_percentile']}` and applied unchanged to test.

## Test leaderboard

{table}

## Task results

{_subgroup_table(metrics, 'task')}

## Outcome results

{_subgroup_table(metrics, 'outcome_group')}

## Interpretation boundary

Original SCIZOR, Uniform split, and Future discount are evaluated on the same frozen transition labels. The task and outcome tables above are the primary subgroup summaries. Sparse perturbation subgroups are descriptive only; the JSON metrics retain their `pair_count` and `transition_count` values. The benchmark provides fixed labels, group split, chunk evidence, and baseline error measurements for the Stage 2 responsibility model. No test-set threshold tuning or split regeneration was performed.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote {args.leaderboard} and {args.report}")


if __name__ == "__main__":
    main()
