from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .common import env_for_dataset, replay, text


def horizon_mean(values, t, horizon):
    end = min(len(values), int(t) + int(horizon))
    if end <= int(t):
        return np.nan
    return float(np.mean(np.asarray(values)[int(t):end]))


def binary_metrics(labels, scores):
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any(): return {}
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64); ranks[order] = np.arange(1, len(s) + 1)
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
    desc = np.argsort(-s, kind="mergesort"); yy = y[desc]; tp = np.cumsum(yy); precision = tp / np.arange(1, len(y) + 1); ap = float((precision * yy).sum() / pos.sum())
    return {"auroc": auc, "auprc": ap}


def run_pair(env_ref, env_clean, source, pert_group, clean_group, horizons, model_xml=None):
    initial = np.asarray(source[f"data/{text(pert_group.attrs['base_demo_id'])}"]["states"][0]).copy()
    pert_actions = np.asarray(pert_group["actions"]); clean_actions = np.asarray(clean_group["actions"])
    pert = replay(env_ref, initial, pert_actions, render_images=False, model_xml=model_xml)
    clean = replay(env_clean, initial, clean_actions, render_images=False, model_xml=model_xml)
    t = int(pert_group.attrs["perturb_t"])
    pre_ref = pert["states_pre"][t]
    pre_clean = clean["states_pre"][t]
    ref_max = float(np.max(np.abs(pert["states_post"] - np.asarray(pert_group["states_post"]))))
    clean_max = float(np.max(np.abs(clean["states_post"] - np.asarray(clean_group["states_post"]))))
    row = {"pair_id": text(pert_group.attrs["pair_id"]), "task": text(pert_group.attrs["task"]), "base_demo_id": text(pert_group.attrs["base_demo_id"]), "perturb_t": t, "failure_type": text(pert_group.attrs.get("failure_type", "")), "is_effective_intervention": bool(pert_group.attrs.get("is_effective_intervention", False)), "branch_pre_state_equal": bool(np.array_equal(pre_ref, pre_clean)), "branch_pre_state_max_abs": float(np.max(np.abs(pre_ref - pre_clean))), "reference_max_abs": ref_max, "paired_clean_max_abs": clean_max, "reference_exact_all_horizons": bool(ref_max == 0.0), "paired_clean_exact_all_horizons": bool(clean_max == 0.0), "finite_target": bool(np.isfinite(pert["rewards"]).all() and np.isfinite(clean["rewards"]).all()), "actual_horizon": int(min(len(pert_actions), len(clean_actions)) - t)}
    for h in horizons:
        row[f"dense_delta_h{h}"] = horizon_mean(clean["rewards"], t, h) - horizon_mean(pert["rewards"], t, h)
        row[f"stage_delta_h{h}"] = horizon_mean(np.max(clean["staged_rewards"], axis=1), t, h) - horizon_mean(np.max(pert["staged_rewards"], axis=1), t, h)
        row[f"success_delta_h{h}"] = horizon_mean(clean["success"].astype(float), t, h) - horizon_mean(pert["success"].astype(float), t, h)
    horizon_weights = {10: 0.2, 20: 0.3, 40: 0.5}
    if set(horizons) != set(horizon_weights):
        raise ValueError("v0.2 freezes oracle horizons to 10,20,40")
    row["counterfactual_improvement"] = float(sum(
        horizon_weights[h] * (0.4 * row[f"dense_delta_h{h}"] + 0.5 * row[f"stage_delta_h{h}"] + 0.1 * row[f"success_delta_h{h}"])
        for h in horizons if np.isfinite(row[f"dense_delta_h{h}"])
    ))
    return row


def main():
    p = argparse.ArgumentParser(); p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True); p.add_argument("--horizons", default="10,20,40"); p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True); p.add_argument("--part-index", type=int, default=0); p.add_argument("--num-parts", type=int, default=1); a = p.parse_args(); horizons = [int(x) for x in a.horizons.split(",")]
    metadata = {str(r["pair_id"]): r for r in (json.loads(x) for x in a.metadata.read_text(encoding="utf-8").splitlines() if x.strip())}
    rows = []; envs = {}; sources = {}; models = {}; seen = 0
    with h5py.File(a.benchmark, "r") as h5:
        for name, g in h5["data"].items():
            if text(g.attrs.get("variant", "")) != "perturbed": continue
            if seen % a.num_parts != a.part_index:
                seen += 1
                continue
            seen += 1
            task = text(g.attrs["task"])
            if task not in envs:
                source = text(g.attrs["source_dataset"]); envs[task] = (env_for_dataset(source)[0], env_for_dataset(source)[0])
                sources[task] = h5py.File(source, "r")
                models[task] = envs[task][0].env.model.get_xml()
            clean_name = text(g.attrs["clean_demo_id"])
            # XML replacement is needed once to synchronize the two freshly
            # created environments. Subsequent resets retain that model.
            model_xml = models[task]
            row = run_pair(envs[task][0], envs[task][1], sources[task], g, h5[f"data/{clean_name}"], horizons, model_xml)
            if row["pair_id"] not in metadata:
                raise ValueError(f"metadata missing {row['pair_id']}")
            row["split"] = str(metadata[row["pair_id"]].get("split", "pilot"))
            if row["split"] not in {"train", "validation", "pilot"}:
                raise ValueError(f"forbidden split: {row['split']}")
            rows.append(row)
    for pair in envs.values(): pair[0].close(); pair[1].close()
    for source in sources.values(): source.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.suffix == ".jsonl":
        a.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    else:
        raise RuntimeError("Parquet output requires an installed parquet engine; use .jsonl output in the replay runtime")
    engineering = {"pair_count": len(rows), "branch_pre_state_equal_rate": float(sum(x["branch_pre_state_equal"] for x in rows) / len(rows)) if rows else 0.0, "reference_exact_all_horizons_rate": float(sum(x["reference_exact_all_horizons"] for x in rows) / len(rows)) if rows else 0.0, "paired_clean_exact_all_horizons_rate": float(sum(x["paired_clean_exact_all_horizons"] for x in rows) / len(rows)) if rows else 0.0, "finite_target_rate": float(sum(x["finite_target"] for x in rows) / len(rows)) if rows else 0.0}
    metrics = {}
    if rows and len({x["is_effective_intervention"] for x in rows}) == 2:
        y = [int(x["is_effective_intervention"]) for x in rows]; s = [float(x["counterfactual_improvement"]) for x in rows]; overall = binary_metrics(y, s); metrics = {"overall_auroc": overall["auroc"], "overall_auprc": overall["auprc"]}
        for task in ("can", "square"):
            sub = [x for x in rows if x["task"] == task]
            if sub and len({x["is_effective_intervention"] for x in sub}) == 2: metrics[f"{task}_auroc"] = binary_metrics([int(x["is_effective_intervention"]) for x in sub], [x["counterfactual_improvement"] for x in sub])["auroc"]; metrics[f"{task}_n"] = len(sub)
    failure_counts = {}
    for x in rows: failure_counts[x["failure_type"]] = failure_counts.get(x["failure_type"], 0) + 1
    summary = {"engineering": engineering, "metrics": metrics, "failure_type_counts": failure_counts, "effective_count": sum(bool(x["is_effective_intervention"]) for x in rows), "no_effect_count": sum(not bool(x["is_effective_intervention"]) for x in rows)}
    a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
