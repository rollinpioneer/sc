"""Run one frozen verifier checkpoint over a sample table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage3a.method_v02r.data.counterfactual_verifier_dataset import CounterfactualVerifierDataset
from stage3a.method_v02r.models.long_horizon_verifier import LongHorizonCounterfactualVerifier


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--samples", type=Path, required=True); p.add_argument("--feature-index", type=Path, required=True); p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--config", type=Path, required=True); p.add_argument("--mode", choices=("full", "action_only"), required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if saved["mode"] != a.mode: raise ValueError("checkpoint mode mismatch")
    data = CounterfactualVerifierDataset(a.samples, a.feature_index, a.normalizer, int(saved["config"]["feature"]["continuation_horizon"]), int(saved["config"]["feature"]["history_length"]))
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongHorizonCounterfactualVerifier(int(saved["state_dim"]), float(saved["config"]["model"]["dropout"])).to(device).eval(); model.load_state_dict(saved["model_state"])
    rows = []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            out = model(batch, a.mode)
            for identifier, score, logit in zip(raw["replacement_id"], out["pred_score"].cpu().numpy(), out["pred_positive_logit"].cpu().numpy()): rows.append({"replacement_id": identifier, "pred_score": float(score), "pred_positive_probability": float(torch.sigmoid(torch.tensor(logit))), "mode": a.mode})
    result = pd.DataFrame(rows); a.output.parent.mkdir(parents=True, exist_ok=True); result.to_parquet(a.output, index=False); print({"rows": len(result), "output": str(a.output)})


if __name__ == "__main__": main()
