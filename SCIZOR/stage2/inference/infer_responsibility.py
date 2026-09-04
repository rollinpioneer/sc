"""Run a fixed responsibility checkpoint over every Stage 1 evidence chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage2.data.responsibility_dataset import ResponsibilityDataset, make_dataloader
from stage2.models.responsibility_model import ResponsibilityNet

KEYS = ["task", "demo_id", "t"]
META = ["task", "demo_id", "pair_id", "variant", "base_demo_id", "split"]


def parse():
    p = argparse.ArgumentParser()
    # Stage 3 v0.2-R reuses this frozen proposer on its isolated blind split;
    # the model and inference logic are unchanged.
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--model-name", required=True); p.add_argument("--split", choices=["train", "validation", "test", "blind_test"], required=True)
    p.add_argument("--chunk-evidence", type=Path, required=True); p.add_argument("--transition-labels", type=Path, required=True); p.add_argument("--feature-index", type=Path, required=True); p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-chunks", type=Path, required=True); p.add_argument("--output-transitions", type=Path, required=True); p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def move(batch, device): return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main():
    args = parse(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text()); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(checkpoint["config"]["model"]); kwargs.update(state_dim=int(checkpoint["state_dim"]), action_feature_dim=int(checkpoint["action_feature_dim"]), max_horizon=int(config["data"]["max_horizon"]), model_variant=str(checkpoint["model_variant"]))
    model = ResponsibilityNet(**kwargs).to(device).eval(); model.load_state_dict(checkpoint["model_state"])
    labels = pd.read_parquet(args.transition_labels); labels_split = labels[labels["split"] == args.split].copy()
    meta = labels_split[META].drop_duplicates(["task", "demo_id"])
    chunks = pd.read_parquet(args.chunk_evidence)
    # Stage 1 stores a terminal evidence row with horizon_steps=1 but the
    # half-open interval is [T-1, T-1). It has no transition to score.
    chunks = chunks[(chunks.variant.isin(["clean", "perturbed"])) & ((chunks.end_t - chunks.start_t) > 0)]
    chunks = chunks.merge(meta.drop(columns="pair_id"), on=["task", "demo_id", "variant", "base_demo_id"], how="inner", validate="many_to_one")
    chunks.loc[chunks.variant == "clean", "pair_id"] = pd.NA
    features = pd.read_parquet(args.feature_index)[["task", "demo_id", "feature_path"]]
    chunks = chunks.merge(features, on=["task", "demo_id"], how="inner", validate="many_to_one")
    chunks["sample_id"] = [f"{task}|{demo}|{chunk}" for task, demo, chunk in chunks[["task", "demo_id", "chunk_id"]].itertuples(index=False, name=None)]
    chunks["sample_type"] = "inference"; chunks["target_effect"] = 0; chunks["target_center_t"] = np.nan; chunks["target_region_start_t"] = np.nan; chunks["target_region_end_t"] = np.nan; chunks["pair_sample_weight"] = 1.0
    dataset = ResponsibilityDataset(chunks, args.normalizer, max_horizon=int(config["data"]["max_horizon"]))
    loader = make_dataloader(dataset, args.batch_size, train=False, num_workers=4)
    by_id = chunks.set_index("sample_id")
    chunk_rows = []; aggregate: dict[tuple[str, str, int], list[float]] = {}
    with torch.inference_mode():
        for batch in loader:
            ids = list(batch["sample_id"]); raw = by_id.loc[ids]
            dev = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(dev)
            rho, gate = out.rho.float().cpu().numpy(), out.p_effect.float().cpu().numpy()
            valid = batch["valid_mask"].numpy()
            for i, (_, row) in enumerate(raw.iterrows()):
                length, start = int(row.end_t - row.start_t), int(row.start_t)
                values = rho[i, :length]; p = float(gate[i]); raw_mass = float(row.V_c) * p * length * values
                peak = int(np.argmax(values)) + start
                entropy = float(-(values * np.log(np.maximum(values, 1e-12))).sum())
                chunk_rows.append({"method": args.model_name, "task": row.task, "demo_id": row.demo_id, "pair_id": row.pair_id, "chunk_id": row.chunk_id, "start_t": start, "end_t": int(row.end_t), "V_c": float(row.V_c), "p_effect": p, "rho_entropy": entropy, "peak_rho": float(values.max()), "predicted_t": peak})
                for offset, mass in enumerate(raw_mass):
                    key = (row.task, row.demo_id, start + offset); item = aggregate.setdefault(key, [0., 0., 0., 0.])
                    item[0] += float(mass); item[1] += 1.; item[2] += p; item[3] = max(item[3], p)
    out_chunks = pd.DataFrame(chunk_rows); args.output_chunks.parent.mkdir(parents=True, exist_ok=True); out_chunks.to_parquet(args.output_chunks, index=False)
    transitions = labels_split[META + ["t"]].copy(); raw_score = []; coverage = []; mean_gate = []; max_gate = []
    for task, demo, t in transitions[KEYS].itertuples(index=False, name=None):
        item = aggregate.get((task, demo, int(t)), [0., 0., 0., 0.]); mean = item[0] / max(1., item[1]); raw_score.append(mean); coverage.append(int(item[1])); mean_gate.append(item[2] / max(1., item[1])); max_gate.append(item[3])
    transitions["method"] = args.model_name; transitions["score_raw"] = raw_score; transitions["score"] = 1 - np.exp(-np.asarray(raw_score)); transitions["num_covering_chunks"] = coverage; transitions["mean_effect_gate"] = mean_gate; transitions["max_effect_gate"] = max_gate
    transitions = transitions[["method", *META, "t", "score", "score_raw", "num_covering_chunks", "mean_effect_gate", "max_effect_gate"]]
    if transitions.duplicated(KEYS).any() or len(transitions) != len(labels_split): raise RuntimeError("inference output failed frozen-label coverage")
    args.output_transitions.parent.mkdir(parents=True, exist_ok=True); transitions.to_parquet(args.output_transitions, index=False)
    print(json.dumps({"method": args.model_name, "split": args.split, "chunks": len(out_chunks), "transitions": len(transitions)}, indent=2))


if __name__ == "__main__": main()
