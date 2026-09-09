"""Train the fixed HB2 model matrix with root-balanced batches."""
from __future__ import annotations

import argparse
import json
import random
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
    return {key: value.to(device) for key, value in batch.items() if key in ("global", "local", "proprio", "base_actions", "time")}


def _outcome_loss(logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    device = logits.device
    y0 = batch["y0"].to(device=device, dtype=logits.dtype)
    yfull = batch["yfull"].to(device=device, dtype=logits.dtype)
    categories = batch["categories"].to(device=device)
    delta = batch["delta"].to(device=device, dtype=logits.dtype)
    losses = [F.binary_cross_entropy_with_logits(logits[:, 0], y0),
              F.cross_entropy(logits[:, 1:5], categories[:, 0]),
              F.cross_entropy(logits[:, 5:9], categories[:, 1]),
              F.cross_entropy(logits[:, 9:13], categories[:, 2]),
              F.binary_cross_entropy_with_logits(logits[:, 13], yfull)]
    outcome = torch.stack(losses).mean()
    probs = [torch.sigmoid(logits[:, 0])]
    for start in (1, 5, 9):
        probs.append(1 - logits[:, start:start + 4].softmax(-1)[:, 0])
    p0 = probs[0]
    pair = torch.stack([F.smooth_l1_loss(probs[index + 1] - p0, delta[:, index]) for index in range(3)]).mean()
    return outcome, {"outcome": outcome, "pair": pair, "total": outcome}


def _m0_fit(rows: list[dict], output_dir: Path) -> None:
    groups = {}
    for row in rows:
        key = int(row["anchor_t"])
        groups.setdefault(key, []).append(row)
    total = len(rows)
    payload = {"kind": "M0_time", "alpha": 1.0, "overall": {
        "p0": (sum(int(row["y0"]) for row in rows) + 1) / (total + 2),
        "pfull": (sum(int(row["y_full"]) for row in rows) + 1) / (total + 2),
    }, "by_time": {str(key): {
        "p0": (sum(int(row["y0"]) for row in values) + 1) / (len(values) + 2),
        "pfull": (sum(int(row["y_full"]) for row in values) + 1) / (len(values) + 2),
        "categories": [[(sum(int(row[f"category_l{length}"]) == category for row in values) + 1) /
                        (len(values) + 4) for category in range(4)] for length in (5, 20, 80)],
    } for key, values in groups.items()}}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "best.pt")
    torch.save(payload, output_dir / "last.pt")
    atomic_json_dump({"kind": "M0_time", "training_rows": len(rows), "stopped_epoch": 0}, output_dir / "training_summary.json")


def _evaluate(model, loader, device, *, pair_weight: float, include_pair: bool) -> dict:
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(_inputs(batch, device))
            outcome, pieces = _outcome_loss(logits, batch)
            roots = batch["root_id"]
            rows.extend([(str(root), float(outcome.detach().cpu())) for root in roots])
    by_root = {}
    for root, value in rows:
        by_root.setdefault(root, []).append(value)
    nll = float(np.mean([np.mean(values) for values in by_root.values()])) if by_root else float("inf")
    return {"root_equal_outcome_nll": nll, "roots": len(by_root), "rows": len(rows)}


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
    model = build_model(args.model, hidden_size=128, dropout=0.1).to(_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_epochs = 1 if args.smoke_batches else int(config["hb2"]["training"]["max_epochs"])
    best = float("inf"); no_improve = 0; history = []
    for epoch in range(max_epochs):
        model.train(); train_seen = 0
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(_inputs(batch, _device()))
            outcome, pieces = _outcome_loss(logits, batch)
            total = outcome + (0.25 * pieces["pair"] if args.model in ("M4_paired", "M5_single") else 0.0)
            total.backward(); clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_seen += len(batch["y0"])
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        val = _evaluate(model, val_loader, _device(), pair_weight=0.25, include_pair=False)
        record = {"epoch": epoch + 1, "train_rows": train_seen, **val}
        history.append(record)
        if val["root_equal_outcome_nll"] < best:
            best = val["root_equal_outcome_nll"]; no_improve = 0
            torch.save({"model_id": args.model, "seed": seed, "state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "best.pt")
        else:
            no_improve += 1
        torch.save({"model_id": args.model, "seed": seed, "state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "last.pt")
        if args.smoke_batches or (epoch + 1) % int(config["hb2"]["training"]["validate_every"]) == 0 and no_improve >= int(config["hb2"]["training"]["early_stop_validation_checks"]):
            if args.smoke_batches or no_improve >= int(config["hb2"]["training"]["early_stop_validation_checks"]):
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({"model": args.model, "seed": seed, "history": history,
                      "best_validation_nll": best, "stopped_epoch": len(history),
                      "parameters": sum(parameter.numel() for parameter in model.parameters())}, output_dir / "training_summary.json")


def _train_handoff(args, config: dict, rows: list[dict], output_dir: Path) -> None:
    _seed(int(args.seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Path(args.features) / "normalizer.json"
    train_ds = HandoffDataset(rows, Path(args.features), split="train", normalizer_path=normalizer)
    val_ds = HandoffDataset(rows, Path(args.features), split="validation", normalizer_path=normalizer)
    sampler = RootBalancedBatchSampler(train_ds, 32, int(args.seed))
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    model = build_model(args.model, head="handoff").to(_device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf"); no_improve = 0; history = []
    for epoch in range(int(config["hb2"]["training"]["max_epochs"])):
        model.train(); sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(_inputs(batch, _device())), batch["target"].to(_device()))
            loss.backward(); clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval(); values = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(_inputs(batch, _device()))
                values.extend(F.cross_entropy(logits, batch["target"].to(_device()), reduction="none").cpu().numpy().tolist())
        nll = float(np.mean(values)) if values else float("inf")
        history.append({"epoch": epoch + 1, "nll": nll})
        if nll < best:
            best = nll; no_improve = 0
            torch.save({"head": "handoff", "model_id": args.model, "seed": int(args.seed), "state_dict": model.state_dict(), "epoch": epoch + 1}, output_dir / "best.pt")
        else: no_improve += 1
        if no_improve >= 4: break
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({"head": "handoff", "model": args.model, "seed": int(args.seed), "history": history,
                      "best_validation_nll": best, "stopped_epoch": len(history)}, output_dir / "training_summary.json")


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
