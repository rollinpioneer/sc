"""Train the fixed learned counterfactual verifier without simulator access."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage3a.method_v02r.data.counterfactual_verifier_dataset import CounterfactualVerifierDataset
from stage3a.method_v02r.data.query_group_batch_sampler import QueryGroupBatchSampler
from stage3a.method_v02r.models.long_horizon_verifier import LongHorizonCounterfactualVerifier
from stage3a.method_v02r.train.losses import verifier_loss


def auc(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    valid = np.isfinite(scores); labels, scores = labels[valid], scores[valid]
    pos, neg = int(labels.sum()), int(len(labels) - labels.sum())
    if not pos or not neg: return None
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), float); i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]: j += 1
        ranks[order[i:j]] = (i + j + 1) / 2; i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort"); y = labels[order]; pos = int(y.sum())
    return None if not pos else float(((np.cumsum(y) / np.arange(1, len(y) + 1)) * y).sum() / pos)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate(model, loader, mode, device, sample_frame):
    model.eval(); ids = []; scores = []; probabilities = []
    with torch.inference_mode():
        for raw in loader:
            batch = move(raw, device); out = model(batch, mode)
            ids.extend(raw["replacement_id"]); scores.extend(out["pred_score"].cpu().numpy()); probabilities.extend(torch.sigmoid(out["pred_positive_logit"]).cpu().numpy())
    table = pd.DataFrame({"replacement_id": ids, "pred_score": scores, "pred_positive_probability": probabilities}).merge(sample_frame, on="replacement_id", validate="one_to_one")
    truth = table.target_positive.astype(int).to_numpy(); score = table.pred_positive_probability.to_numpy()
    mae = float(np.abs(table.pred_score - table.counterfactual_improvement_long).mean())
    spearman = float(pd.Series(table.pred_score).corr(pd.Series(table.counterfactual_improvement_long), method="spearman"))
    primary = table[
        table.is_teacher_forced
        & table.query_t.eq(table.intervention_t)
        & table.replacement_rank.eq(0)
    ].copy()
    pair = primary.drop_duplicates("pair_id")
    return {"replacement_auroc": auc(truth, score), "replacement_auprc": auprc(truth, score), "score_mae": mae, "score_spearman": spearman,
            "teacher_forced_primary_auroc": auc(pair.is_effective_intervention.astype(int), pair.pred_score), "prediction_rows": table}


def better(metrics, best):
    if best is None: return True
    current_ap, best_ap = metrics["replacement_auprc"], best["replacement_auprc"]
    if current_ap is None: return False
    if best_ap is None or current_ap > best_ap + 0.005: return True
    if abs(current_ap - best_ap) <= 0.005 and metrics["score_mae"] < best["score_mae"] - 1e-9: return True
    return abs(current_ap - best_ap) <= 0.005 and abs(metrics["score_mae"] - best["score_mae"]) <= 1e-9 and (metrics["teacher_forced_primary_auroc"] or -1) > (best["teacher_forced_primary_auroc"] or -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True); p.add_argument("--train-samples", type=Path, required=True); p.add_argument("--validation-samples", type=Path, required=True)
    p.add_argument("--feature-index", type=Path, required=True); p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "action_only"), required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--max-steps", type=int)
    a = p.parse_args(); config = json.loads(a.config.read_text())
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    train_frame, validation_frame = pd.read_parquet(a.train_samples), pd.read_parquet(a.validation_samples)
    horizon = int(config["feature"]["continuation_horizon"]); history = int(config["feature"]["history_length"])
    train = CounterfactualVerifierDataset(train_frame, a.feature_index, a.normalizer, horizon, history)
    validation = CounterfactualVerifierDataset(validation_frame, a.feature_index, a.normalizer, horizon, history)
    sampler = QueryGroupBatchSampler(train_frame, int(config["training"]["group_batch_size"]), a.seed)
    train_loader = DataLoader(train, batch_sampler=sampler, num_workers=int(config["training"]["num_workers"]), pin_memory=True)
    val_loader = DataLoader(validation, batch_size=128, shuffle=False, num_workers=int(config["training"]["num_workers"]), pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongHorizonCounterfactualVerifier(int(train[0]["state_t"].numel()), float(config["model"]["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"]["mixed_precision"]))
    a.output_dir.mkdir(parents=True, exist_ok=True); history_rows, best, stale, steps = [], None, 0, 0
    max_epochs = int(config["training"]["max_epochs"])
    for epoch in range(max_epochs):
        model.train(); sampler.set_epoch(epoch); aggregate = np.zeros(4); count = 0
        for raw in train_loader:
            batch = move(raw, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and bool(config["training"]["mixed_precision"])):
                out = model(batch, a.mode)
                losses = verifier_loss(
                    out,
                    batch,
                    raw["query_group_id"],
                    margin=float(config["training"]["rank_margin"]),
                    rank_target_gap=float(config["training"]["rank_target_gap"]),
                )
                loss = losses["loss"]
                regression = losses["regression_loss"]
                classification = losses["classification_loss"]
                ranking = losses["rank_loss"]
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"])); scaler.step(optimizer); scaler.update()
            aggregate += [float(loss.detach()), float(regression.detach()), float(classification.detach()), float(ranking.detach())]; count += 1; steps += 1
            if a.max_steps is not None and steps >= a.max_steps: break
        metrics = evaluate(model, val_loader, a.mode, device, validation_frame)
        row = {"epoch": epoch, "steps": steps, "train_loss": aggregate[0] / max(count, 1), "train_regression_loss": aggregate[1] / max(count, 1), "train_classification_loss": aggregate[2] / max(count, 1), "train_rank_loss": aggregate[3] / max(count, 1), **{key: value for key, value in metrics.items() if key != "prediction_rows"}}
        history_rows.append(row)
        if better(metrics, best):
            best = {key: value for key, value in metrics.items() if key != "prediction_rows"}; stale = 0
            torch.save({"model_state": model.state_dict(), "state_dim": int(train[0]["state_t"].numel()), "mode": a.mode, "config": config, "seed": a.seed}, a.output_dir / "best.pt")
        else: stale += 1
        if a.max_steps is not None or stale >= int(config["training"]["early_stop_patience"]): break
    torch.save({"model_state": model.state_dict(), "state_dim": int(train[0]["state_t"].numel()), "mode": a.mode, "config": config, "seed": a.seed}, a.output_dir / "last.pt")
    with (a.output_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys()); writer.writeheader(); writer.writerows(history_rows)
    (a.output_dir / "run_config.json").write_text(json.dumps({"mode": a.mode, "seed": a.seed, "max_steps": a.max_steps, "best": best}, indent=2), encoding="utf-8")
    print(json.dumps({"epochs": len(history_rows), "best": best}, indent=2))


if __name__ == "__main__": main()
