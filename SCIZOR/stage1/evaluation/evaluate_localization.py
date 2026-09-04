"""Evaluate frozen Stage 1D operating points without test-set tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _measure(part, threshold):
    part = part.copy()
    if part.empty:
        return {"transition_count": 0}
    part["deleted"] = part["score"] >= threshold
    truth, pred = part["is_responsible_point"].to_numpy(bool), part["deleted"].to_numpy(bool)
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    precision, recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    result = {
        "transition_count": int(len(part)), "pair_count": int(part["pair_id"].nunique()), "delete_rate": float(pred.mean()),
        "transition_precision": precision, "transition_recall": recall, "transition_f1": f1,
        "recovery_retention": _safe_ratio(int((part["is_recovery"] & ~part["deleted"]).sum()), int(part["is_recovery"].sum())),
        "recovery_false_deletion_rate": _safe_ratio(int((part["is_recovery"] & part["deleted"]).sum()), int(part["is_recovery"].sum())),
        "innocent_downstream_retention": _safe_ratio(int((part["is_innocent_downstream"] & ~part["deleted"]).sum()), int(part["is_innocent_downstream"].sum())),
        "rare_retention": _safe_ratio(int((part["is_rare"] & ~part["deleted"]).sum()), int(part["is_rare"].sum())),
        "no_effect_false_attribution_rate": _safe_ratio(int((part["is_no_effect_intervention"] & part["deleted"]).sum()), int(part["is_no_effect_intervention"].sum())),
    }
    effective = part[part["is_responsible_point"].groupby(part["pair_id"]).transform("any")]
    ious, delays, abs_delays, top1, top5 = [], [], [], [], []
    for _, episode in effective.groupby("pair_id", dropna=False):
        region, deleted = episode["is_responsibility_region"].to_numpy(bool), episode["deleted"].to_numpy(bool)
        union = (region | deleted).sum()
        ious.append(float((region & deleted).sum() / union) if union else 0.0)
        ordered = episode.sort_values(["score", "t"], ascending=[False, True], kind="stable")
        responsible_t = int(episode.loc[episode["is_responsible_point"], "t"].iloc[0])
        peak_t = int(ordered.iloc[0]["t"])
        delays.append(peak_t - responsible_t)
        abs_delays.append(abs(peak_t - responsible_t))
        top1.append(bool(ordered.iloc[0]["is_responsibility_region"]))
        top5.append(bool((ordered.head(5)["t"] == responsible_t).any()))
    result.update({
        "effective_pair_count": len(ious), "responsibility_region_iou": float(np.mean(ious)) if ious else None,
        "mean_localization_delay": float(np.mean(delays)) if delays else None,
        "mean_abs_localization_delay": float(np.mean(abs_delays)) if abs_delays else None,
        "top1_within_1": float(np.mean(top1)) if top1 else None, "top5_hit": float(np.mean(top5)) if top5 else None,
    })
    return result


def _bootstrap(part, threshold, samples, seed):
    groups = list(part.groupby(["task", "base_demo_id"], sort=False))
    if not groups:
        return {}
    rng, values = np.random.default_rng(seed), {key: [] for key in ("transition_f1", "responsibility_region_iou", "mean_abs_localization_delay", "recovery_retention")}
    for _ in range(samples):
        sampled = [groups[index][1] for index in rng.integers(0, len(groups), len(groups))]
        metrics = _measure(pd.concat(sampled, ignore_index=True), threshold)
        for key in values:
            if metrics.get(key) is not None:
                values[key].append(metrics[key])
    return {key: {"mean": float(np.mean(items)), "ci95": [float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))]} for key, items in values.items() if items}


def _flat(method, dimension, group, metrics):
    return [{"method": method, "dimension": dimension, "group": str(group), **metrics}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    operating = json.loads(Path(args.operating_points).read_text(encoding="utf-8"))["methods"]
    active = predictions[(predictions["split"] == args.split) & (predictions["variant"] == "perturbed") & (predictions["label_status"] != "ambiguous")]
    clean = predictions[(predictions["split"] == args.split) & (predictions["variant"] == "clean")]
    report, flat = {"split": args.split, "methods": {}}, []
    for method in sorted(operating):
        threshold = float(operating[method]["threshold"])
        part, clean_part = active[active["method"] == method], clean[clean["method"] == method]
        overall = _measure(part, threshold)
        overall["expert_retention"] = _safe_ratio(int((clean_part["score"] < threshold).sum()), len(clean_part))
        method_report = {"threshold": threshold, "overall": overall, "bootstrap": _bootstrap(part, threshold, args.bootstrap_samples, args.seed)}
        flat.extend(_flat(method, "overall", "all", overall))
        for dimension, column in (("task", "task"), ("failure_type", "failure_type"), ("perturbation_type", "perturbation_type")):
            method_report[dimension] = {}
            for group, group_part in part.groupby(column, dropna=False):
                metrics = _measure(group_part, threshold)
                method_report[dimension][str(group)] = metrics
                flat.extend(_flat(method, dimension, group, metrics))
        outcome = part.assign(outcome_group=np.where(part["failure_type"] == "recovery_success", "recovery_success", "final_failure"))
        method_report["outcome_group"] = {}
        for group, group_part in outcome[outcome["failure_type"] != "no_effect"].groupby("outcome_group"):
            metrics = _measure(group_part, threshold)
            method_report["outcome_group"][group] = metrics
            flat.extend(_flat(method, "outcome_group", group, metrics))
        for flag in ("is_slow_precise", "is_rare"):
            metrics = _measure(part[part[flag]], threshold)
            method_report[flag] = metrics
            flat.extend(_flat(method, flag, "true", metrics))
        report["methods"][method] = method_report
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(flat).to_csv(args.output_csv, index=False)
    print(json.dumps({"split": args.split, "methods": list(report["methods"]), "rows": len(flat)}, indent=2))


if __name__ == "__main__":
    main()
