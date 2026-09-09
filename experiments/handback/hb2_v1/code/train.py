"""Train the fixed HB2 model matrix with root-balanced batches."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from recovery_handback.common import atomic_json_dump
from recovery_handback.hb2.dataset import AnchorDataset, HandoffDataset, RootBalancedBatchSampler
from recovery_handback.hb2.models import build_model


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    names = ("global", "local", "proprio", "base_actions", "time", "helper_elapsed")
    return {key: value.to(device) for key, value in batch.items() if key in names}


def _outcome_loss(logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    device = logits.device
    y0 = batch["y0"].to(device=device, dtype=logits.dtype)
    yfull = batch["yfull"].to(device=device, dtype=logits.dtype)
    categories = batch["categories"].to(device=device)
    delta = batch["delta"].to(device=device, dtype=logits.dtype)
    losses = [F.binary_cross_entropy_with_logits(logits[:, 0], y0, reduction="none"),
              F.cross_entropy(logits[:, 1:5], categories[:, 0], reduction="none"),
              F.cross_entropy(logits[:, 5:9], categories[:, 1], reduction="none"),
              F.cross_entropy(logits[:, 9:13], categories[:, 2], reduction="none"),
              F.binary_cross_entropy_with_logits(logits[:, 13], yfull, reduction="none")]
    outcome_per_example = torch.stack(losses, dim=1).mean(dim=1)
    probs = [torch.sigmoid(logits[:, 0])]
    for start in (1, 5, 9):
        probs.append(logits[:, start:start + 4].softmax(-1)[:, 3])
    p0 = probs[0]
    pair_per_example = torch.stack([
        F.smooth_l1_loss(probs[index + 1] - p0, delta[:, index], reduction="none")
        for index in range(3)
    ], dim=1).mean(dim=1)
    outcome = outcome_per_example.mean()
    pair = pair_per_example.mean()
    return outcome, {"outcome": outcome, "pair": pair, "outcome_per_example": outcome_per_example,
                     "pair_per_example": pair_per_example}


def _m0_fit(rows: list[dict], output_dir: Path) -> None:
    groups = {}
    for row in rows:
        key = int(row["anchor_t"])
        groups.setdefault(key, []).append(row)

    def frequencies(values: list[dict]) -> dict:
        root_counts = defaultdict(int)
        for row in values:
            root_counts[str(row["stat_group_id"])] += 1
        weights = [1.0 / root_counts[str(row["stat_group_id"])] for row in values]
        total = float(sum(weights))
        return {
            "effective_root_weight": total,
            "p0": (sum(weight * int(row["y0"]) for weight, row in zip(weights, values)) + 1) / (total + 2),
            "pfull": (sum(weight * int(row["y_full"]) for weight, row in zip(weights, values)) + 1) / (total + 2),
            "categories": [[
                (sum(weight * (int(row[f"category_l{length}"]) == category)
                     for weight, row in zip(weights, values)) + 1) / (total + 4)
                for category in range(4)
            ] for length in (5, 20, 80)],
        }

    payload = {
        "kind": "M0_time",
        "alpha": 1.0,
        "weighting": "root_equal_then_anchor",
        "overall": frequencies(rows),
        "by_time": {str(key): frequencies(values) for key, values in groups.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "best.pt")
    torch.save(payload, output_dir / "last.pt")
    atomic_json_dump({"kind": "M0_time", "training_rows": len(rows), "stopped_epoch": 0}, output_dir / "training_summary.json")


def _root_equal_mean(values: list[float], roots: list[str]) -> tuple[float, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for root, value in zip(roots, values):
        grouped[str(root)].append(float(value))
    result = float(np.mean([np.mean(group) for group in grouped.values()])) if grouped else float("inf")
    return result, len(grouped)


def _evaluate(model, loader, device) -> dict:
    model.eval(); values: list[float] = []; roots: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(_inputs(batch, device))
            _, pieces = _outcome_loss(logits, batch)
            values.extend(pieces["outcome_per_example"].detach().cpu().numpy().tolist())
            roots.extend(str(root) for root in batch["root_id"])
    nll, root_count = _root_equal_mean(values, roots)
    return {"root_equal_outcome_nll": nll, "roots": root_count, "rows": len(values)}


def _train_anchor(args, config: dict, rows: list[dict], output_dir: Path) -> None:
    seed = int(args.seed); _seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Path(args.features) / "normalizer.json"
    train_ds = AnchorDataset(rows, Path(args.features), split="train", normalizer_path=normalizer)
    val_ds = AnchorDataset(rows, Path(args.features), split="validation", normalizer_path=normalizer)
    if args.model == "M0_time":
        _m0_fit(train_ds.rows, output_dir); return
    train_sampler = RootBalancedBatchSampler(train_ds, int(config["hb2"]["training"]["batch_size"]), seed)
    loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(config["hb2"]["training"]["batch_size"]), shuffle=False, num_workers=0)
    training = config["hb2"]["training"]
    device = _device()
    model = build_model(args.model, hidden_size=int(training["hidden_size"]),
                        dropout=float(training["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    max_epochs = 1 if args.smoke_batches else int(config["hb2"]["training"]["max_epochs"])
    best = float("inf"); no_improve = 0; history = []
    for epoch in range(max_epochs):
        model.train(); train_seen = 0; train_losses = []
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(_inputs(batch, device))
            outcome, pieces = _outcome_loss(logits, batch)
            total = outcome + (float(training["pair_loss_weight"]) * pieces["pair"] if args.model in ("M4_paired", "M5_single") else 0.0)
            total.backward(); clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step()
            train_seen += len(batch["y0"])
            train_losses.append(float(total.detach().cpu()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        torch.save({"model_id": args.model, "seed": seed, "state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "last.pt")
        should_validate = bool(args.smoke_batches) or (epoch + 1) % int(training["validate_every"]) == 0 or epoch + 1 == max_epochs
        if not should_validate:
            continue
        val = _evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_rows": train_seen,
                  "train_total_loss": float(np.mean(train_losses)) if train_losses else None, **val}
        history.append(record)
        if val["root_equal_outcome_nll"] < best:
            best = val["root_equal_outcome_nll"]; no_improve = 0
            torch.save({"model_id": args.model, "seed": seed, "state_dict": model.state_dict(),
                        "epoch": epoch + 1, "validation": val}, output_dir / "best.pt")
        else:
            no_improve += 1
        if args.smoke_batches or no_improve >= int(training["early_stop_validation_checks"]):
            break
    atomic_json_dump({"model": args.model, "seed": seed, "history": history,
                      "best_validation_nll": best,
                      "stopped_epoch": history[-1]["epoch"] if history else 0,
                      "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                      "history_length": model.history_length,
                      "checkpoint_criterion": training["checkpoint_criterion"]}, output_dir / "training_summary.json")


def _train_handoff(args, config: dict, rows: list[dict], output_dir: Path) -> None:
    _seed(int(args.seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Path(args.features) / "normalizer.json"
    train_ds = HandoffDataset(rows, Path(args.features), split="train", normalizer_path=normalizer)
    val_ds = HandoffDataset(rows, Path(args.features), split="validation", normalizer_path=normalizer)
    sampler = RootBalancedBatchSampler(train_ds, 32, int(args.seed))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    training = config["hb2"]["training"]
    resolved_model = args.model
    selected_f_model = None
    if args.model == "selected_visual":
        if not args.protocol or not args.protocol.is_file():
            raise ValueError("selected_visual handoff model requires --protocol")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        selected_f_model = str(protocol["selected_visual_model"])
        resolved_model = "selected_visual_local" if selected_f_model in ("M3_local", "M4_paired", "M5_single") else "selected_visual_global"
    device = _device()
    model = build_model(resolved_model, head="handoff", hidden_size=int(training["hidden_size"]),
                        dropout=float(training["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    best = float("inf"); no_improve = 0; history = []
    max_epochs = int(training["max_epochs"])
    for epoch in range(max_epochs):
        model.train(); sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(_inputs(batch, device)), batch["target"].to(device))
            loss.backward(); clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step()
        torch.save({"head": "handoff", "model_id": args.model, "architecture": resolved_model,
                    "selected_f_model": selected_f_model, "seed": int(args.seed),
                    "state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "last.pt")
        should_validate = (epoch + 1) % int(training["validate_every"]) == 0 or epoch + 1 == max_epochs
        if not should_validate:
            continue
        model.eval(); values = []; roots = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(_inputs(batch, device))
                values.extend(F.cross_entropy(logits, batch["target"].to(device), reduction="none").cpu().numpy().tolist())
                roots.extend(str(root) for root in batch["root_id"])
        nll, root_count = _root_equal_mean(values, roots)
        history.append({"epoch": epoch + 1, "root_equal_nll": nll, "roots": root_count, "rows": len(values)})
        if nll < best:
            best = nll; no_improve = 0
            torch.save({"head": "handoff", "model_id": args.model, "architecture": resolved_model,
                        "selected_f_model": selected_f_model, "seed": int(args.seed),
                        "state_dict": model.state_dict(), "epoch": epoch + 1,
                        "validation": {"root_equal_nll": nll, "roots": root_count}}, output_dir / "best.pt")
        else: no_improve += 1
        if no_improve >= int(training["early_stop_validation_checks"]): break
    atomic_json_dump({"head": "handoff", "model": args.model, "seed": int(args.seed), "history": history,
                      "architecture": resolved_model, "selected_f_model": selected_f_model,
                      "best_validation_nll": best,
                      "stopped_epoch": history[-1]["epoch"] if history else 0,
                      "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                      "checkpoint_criterion": "root_equal_handoff_three_class_nll"}, output_dir / "training_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head", choices=("anchor", "handoff"), default="anchor")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    table = "anchor_examples.parquet" if args.head == "anchor" else "handoff_examples.parquet"
    rows = __import__("recovery_handback.common", fromlist=["read_table"]).read_table(args.dataset / table)
    if args.head == "handoff": _train_handoff(args, config, rows, args.output_dir)
    else: _train_anchor(args, config, rows, args.output_dir)


if __name__ == "__main__":
    main()
