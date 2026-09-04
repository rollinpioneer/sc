"""Evaluate frozen verifier checkpoints against blind teacher-forced oracle rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage3a.method_v02r.data.counterfactual_verifier_dataset import CounterfactualVerifierDataset
from stage3a.method_v02r.models.long_horizon_verifier import LongHorizonCounterfactualVerifier
from .metrics import binary


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer(frame: pd.DataFrame, checkpoints: list[Path], feature_index: Path, normalizer: Path, config: Path) -> pd.DataFrame:
    predictions = []
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("mode") != "full":
            raise ValueError(f"blind teacher-forced evaluator requires full checkpoint: {checkpoint}")
        dataset = CounterfactualVerifierDataset(frame, feature_index, normalizer, int(saved["config"]["feature"]["continuation_horizon"]), int(saved["config"]["feature"]["history_length"]))
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LongHorizonCounterfactualVerifier(int(saved["state_dim"]), float(saved["config"]["model"]["dropout"])).to(device)
        model.load_state_dict(saved["model_state"]); model.eval()
        score, positive = [], []
        with torch.inference_mode():
            for batch in loader:
                moved = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = model(moved, "full")
                score.extend(output["pred_score"].cpu().numpy().tolist())
                positive.extend(torch.sigmoid(output["pred_positive_logit"]).cpu().numpy().tolist())
        predictions.append((np.asarray(score, float), np.asarray(positive, float)))
    score = np.stack([item[0] for item in predictions], axis=1)
    positive = np.stack([item[1] for item in predictions], axis=1)
    out = frame.copy()
    out["pred_score_mean"] = score.mean(1); out["pred_score_std"] = score.std(1)
    out["pred_positive_mean"] = positive.mean(1); out["pred_positive_std"] = positive.std(1)
    out["pred_score_lcb"] = np.clip(out.pred_score_mean - out.pred_score_std, 0, 0.9)
    out["pred_positive_lcb"] = np.clip(out.pred_positive_mean - out.pred_positive_std, 0, 1)
    out["replacement_cf_score"] = out.pred_score_lcb / 0.9 * out.pred_positive_lcb
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs=3, type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oracle = rows(args.oracle)
    if not oracle:
        raise ValueError("blind teacher-forced oracle is empty")
    with h5py.File(args.benchmark, "r") as handle:
        intervention = {str(name): int(group.attrs["perturb_t"]) for name, group in handle["data"].items() if str(group.attrs.get("variant", "")) == "perturbed"}
    frame = pd.DataFrame(oracle)
    frame["intervention_t"] = frame.perturbed_demo_id.map(intervention)
    if frame.intervention_t.isna().any():
        raise ValueError("blind oracle contains a demo absent from benchmark")
    frame["query_group_id"] = frame["query_id"].astype(str)
    frame["target_positive"] = frame.counterfactual_improvement_long.astype(float) >= 0.5
    frame["target_valid"] = frame[["branch_pre_state_equal", "reference_exact", "finite_target", "state_in_domain", "action_in_domain"]].all(axis=1)
    frame["is_teacher_forced"] = True
    if not frame.target_valid.all():
        raise ValueError("blind teacher-forced oracle contains invalid rows")
    prediction = infer(frame, args.checkpoints, args.feature_index, args.normalizer, args.config)
    primary = prediction[(prediction.replacement_rank.eq(0)) & prediction.query_t.eq(prediction.intervention_t)].drop_duplicates("pair_id")
    metrics = {
        "rows": int(len(prediction)),
        "query_groups": int(prediction.query_group_id.nunique()),
        "engineering": {
            "branch_pre_state_equal_rate": float(prediction.branch_pre_state_equal.astype(bool).mean()),
            "reference_exact_rate": float(prediction.reference_exact.astype(bool).mean()),
            "finite_target_rate": float(prediction.finite_target.astype(bool).mean()),
        },
        "replacement_oracle_positive": binary(prediction.target_positive.astype(int), prediction.pred_positive_mean),
        "score_mae": float(np.abs(prediction.pred_score_mean - prediction.counterfactual_improvement_long).mean()),
        "score_spearman": float(prediction.pred_score_mean.corr(prediction.counterfactual_improvement_long, method="spearman")),
        "primary_effective_vs_no_effect": binary(primary.is_effective_intervention.astype(int), primary.pred_score_mean),
        "primary_by_task": {task: binary(part.is_effective_intervention.astype(int), part.pred_score_mean) for task, part in primary.groupby("task")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
