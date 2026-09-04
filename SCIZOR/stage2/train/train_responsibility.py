"""Train the fixed Stage 2 responsibility network on frozen train/validation splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from stage2.data.responsibility_dataset import ResponsibilityDataset, make_dataloader
from stage2.models.losses import responsibility_losses
from stage2.models.responsibility_model import ResponsibilityNet


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True); p.add_argument("--chunk-index", type=Path, required=True)
    p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--model-variant", choices=["full", "action_only"], required=True)
    p.add_argument("--epochs", type=int); p.add_argument("--max-train-batches", type=int); p.add_argument("--max-validation-batches", type=int)
    return p.parse_args()


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def metrics(model, loader, device, loss_config, amp, max_batches=None) -> dict[str, float]:
    model.eval(); losses = []; positive_top1 = []; positive_top5 = []; delays = []; predicted = []; truth = []; gates = []
    gate_sums, gate_counts = {group: 0.0 for group in range(4)}, {group: 0 for group in range(4)}
    loss_parts = {key: [] for key in ("loss_effect_positive", "loss_effect_negative", "loss_effect_no_effect", "loss_effect_hard_negative", "loss_effect_clean")}
    with torch.inference_mode():
        for number, batch in enumerate(loader):
            if max_batches is not None and number >= max_batches: break
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch); out_losses = responsibility_losses(out, batch, **loss_config)
            losses.append(float(out_losses["loss"]))
            for key in loss_parts: loss_parts[key].append(float(out_losses[key]))
            target, effect = batch["target_distribution"], batch["target_effect"] > .5
            index = out.rho.argmax(dim=-1)
            if effect.any():
                rows = torch.where(effect)[0]
                positive_top1.extend((target[rows, index[rows]] > 0).float().cpu().tolist())
                centers = batch["target_center_index"][rows]
                top5 = torch.topk(out.rho[rows], k=min(5, out.rho.shape[1]), dim=-1).indices
                positive_top5.extend((top5 == centers[:, None]).any(dim=-1).float().cpu().tolist())
                delays.extend((index[rows] - centers).abs().float().cpu().tolist())
            predicted.extend((out.p_effect >= .5).cpu().tolist()); truth.extend(effect.cpu().tolist()); gates.extend(out.p_effect.cpu().tolist())
            groups = batch["effect_group_id"].long()
            for group in range(4):
                mask = groups == group
                if mask.any():
                    gate_sums[group] += float(out.p_effect[mask].sum())
                    gate_counts[group] += int(mask.sum())
    truth_array, pred_array, gate_array = np.asarray(truth, bool), np.asarray(predicted, bool), np.asarray(gates, float)
    positive_recall = pred_array[truth_array].mean() if truth_array.any() else 0.
    negative_recall = (~pred_array[~truth_array]).mean() if (~truth_array).any() else 0.
    group_means = {group: gate_sums[group] / gate_counts[group] if gate_counts[group] else 0.0 for group in range(4)}
    result = {"loss": float(np.mean(losses)), "chunk_top1_within_region": float(np.mean(positive_top1)) if positive_top1 else 0., "chunk_top5_hit_center": float(np.mean(positive_top5)) if positive_top5 else 0., "chunk_mean_abs_delay": float(np.mean(delays)) if delays else float("inf"), "effect_balanced_accuracy": float((positive_recall + negative_recall) / 2), "positive_gate_mean": group_means[0], "negative_gate_mean": float(gate_array[~truth_array].mean()) if (~truth_array).any() else 0., "no_effect_gate_mean": group_means[1], "hard_negative_gate_mean": group_means[2], "clean_gate_mean": group_means[3], "gate_gap": group_means[0] - group_means[1]}
    result.update({key: float(np.mean(values)) if values else 0.0 for key, values in loss_parts.items()})
    return result


def candidate_key(candidate_name: str, metrics: dict[str, float]) -> tuple:
    if candidate_name == "best_localization":
        return (float(metrics["chunk_top1_within_region"]), -float(metrics["chunk_mean_abs_delay"]), float(metrics["effect_balanced_accuracy"]), float(metrics["gate_gap"]))
    if candidate_name == "best_effect_bacc":
        return (float(metrics["effect_balanced_accuracy"]), float(metrics["gate_gap"]), float(metrics["chunk_top1_within_region"]), -float(metrics["chunk_mean_abs_delay"]))
    if candidate_name == "best_gate_gap":
        return (float(metrics["gate_gap"]), float(metrics["effect_balanced_accuracy"]), float(metrics["chunk_top1_within_region"]), -float(metrics["chunk_mean_abs_delay"]))
    raise ValueError(candidate_name)


def main() -> None:
    args = parse(); config = json.loads(args.config.read_text()); args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = bool(config["train"]["mixed_precision"] and device.type == "cuda")
    epochs, batch_size = int(args.epochs or config["train"]["epochs"]), int(config["train"]["batch_size"])
    train_set = ResponsibilityDataset(args.chunk_index, args.normalizer, int(config["data"]["max_horizon"]), "train")
    valid_set = ResponsibilityDataset(args.chunk_index, args.normalizer, int(config["data"]["max_horizon"]), "validation")
    train_loader = make_dataloader(train_set, batch_size, train=True, num_workers=int(config["train"]["num_workers"]), seed=args.seed); valid_loader = make_dataloader(valid_set, batch_size, train=False, num_workers=int(config["train"]["num_workers"]))
    state_dim = int(train_set[0]["state"].shape[-1])
    model_kwargs = dict(config["model"])
    model_kwargs.update(state_dim=state_dim, action_feature_dim=23, max_horizon=int(config["data"]["max_horizon"]), model_variant=args.model_variant)
    model = ResponsibilityNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    steps_per_epoch = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader); total_steps = max(1, epochs * steps_per_epoch); warmup = min(int(config["train"]["warmup_steps"]), max(0, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / max(1, warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
    scaler = torch.cuda.amp.GradScaler(enabled=amp); loss_config = config["loss"]; history = []; patience = 0
    candidate_names = ("best_localization", "best_effect_bacc", "best_gate_gap")
    candidate_best = {name: None for name in candidate_names}
    run_config = {"seed": args.seed, "model_variant": args.model_variant, "device": str(device), "batch_size": batch_size, "epochs": epochs, "state_dim": state_dim, "action_feature_dim": 23, "config": config, "sampler": getattr(train_loader, "sampler_summary", None)}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"sampler": run_config["sampler"]}, sort_keys=True), flush=True)
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []; train_parts = {key: [] for key in ("loss_effect_positive", "loss_effect_negative", "loss_effect_no_effect", "loss_effect_hard_negative", "loss_effect_clean")}; optimizer_steps = 0
        for number, batch in enumerate(train_loader):
            if args.max_train_batches is not None and number >= args.max_train_batches: break
            batch = to_device(batch, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp): out = model(batch); losses = responsibility_losses(out, batch, **loss_config)
            if not torch.isfinite(losses["loss"]): raise FloatingPointError("non-finite loss")
            scaler.scale(losses["loss"]).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip_norm"])); scaler.step(optimizer); scaler.update(); scheduler.step()
            train_losses.append(float(losses["loss"].detach()))
            for key in train_parts: train_parts[key].append(float(losses[key].detach()))
            optimizer_steps += 1
        valid = metrics(model, valid_loader, device, loss_config, amp, args.max_validation_batches)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "optimizer_steps": optimizer_steps, "learning_rate": float(optimizer.param_groups[0]["lr"]), **{f"train_{key}": float(np.mean(values)) if values else 0.0 for key, values in train_parts.items()}, **{f"validation_{key}": value for key, value in valid.items()}}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint = {"model_state": model.state_dict(), "config": config, "state_dim": state_dim, "action_feature_dim": 23, "seed": args.seed, "model_variant": args.model_variant, "epoch": epoch, "validation_metrics": valid}
        torch.save(checkpoint, args.output_dir / "last.pt")
        improved_any = False
        for name in candidate_names:
            key = candidate_key(name, valid)
            previous = candidate_best[name]
            if previous is None or key > tuple(previous["key"]):
                improved_any = True
                candidate = {"path": str(args.output_dir / f"{name}.pt"), "epoch": epoch, "metrics": valid, "key": list(key)}
                candidate_best[name] = candidate
                checkpoint["candidate_name"], checkpoint["best_epoch"] = name, epoch
                torch.save(checkpoint, args.output_dir / f"{name}.pt")
                if name == "best_localization":
                    torch.save(checkpoint, args.output_dir / "best.pt")
                    (args.output_dir / "best_metrics.json").write_text(json.dumps({"best_epoch": epoch, **valid}, indent=2, sort_keys=True), encoding="utf-8")
        (args.output_dir / "candidate_checkpoints.json").write_text(json.dumps(candidate_best, indent=2, sort_keys=True), encoding="utf-8")
        patience = 0 if improved_any else patience + 1
        if patience >= int(config["train"]["early_stop_patience"]): break
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    # Smoke/full invariant: rho has mass only on valid transition slots.
    check = next(iter(valid_loader)); check = to_device(check, device); model.eval(); out = model(check)
    assert torch.allclose((out.rho * check["valid_mask"]).sum(dim=-1), torch.ones(len(check["rho"]) if "rho" in check else out.rho.shape[0], device=device), atol=1e-4)
    assert float(out.rho[~check["valid_mask"]].max()) < 1e-6
    print(json.dumps({"best": json.loads((args.output_dir / "best_metrics.json").read_text()), "output_dir": str(args.output_dir)}))


if __name__ == "__main__": main()
