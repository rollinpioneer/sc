from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def text(x): return x.decode() if isinstance(x, bytes) else x


def stage_delta(clean, pert):
    clean, pert = np.asarray(clean, float), np.asarray(pert, float)
    if clean.shape != pert.shape: raise ValueError(f"stage shape mismatch: {clean.shape}/{pert.shape}")
    return clean - pert if clean.ndim == 1 else (np.max(clean - pert, axis=1) if clean.shape[1] else np.zeros(len(clean)))


def robust(values, q, eps):
    x = np.abs(np.asarray(values, float)); x = x[np.isfinite(x) & (x > eps)]
    return float(np.quantile(x, q)) if len(x) else 1.0


def main():
    p = argparse.ArgumentParser(); p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True); p.add_argument("--spec", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    spec = json.loads(a.spec.read_text()); meta = {r["pair_id"]: r for r in (json.loads(x) for x in a.metadata.read_text().splitlines() if x.strip())}; h, q, eps = int(spec["max_horizon"]), float(spec["normalizer_quantile"]), float(spec["normalizer_zero_epsilon"])
    dense, stage, count = {x: [] for x in ("can", "square")}, {x: [] for x in ("can", "square")}, {x: 0 for x in ("can", "square")}
    with h5py.File(a.benchmark, "r") as f:
        for g in f["data"].values():
            if text(g.attrs.get("variant")) != "perturbed": continue
            r = meta[text(g.attrs["pair_id"])]
            if r["split"] != "train": continue
            task, t = r["task"], int(r["perturb_t"]); clean = f[f"data/{text(g.attrs['clean_demo_id'])}"]; end = min(len(g["rewards"]), t + h)
            dense[task].extend((np.asarray(clean["rewards"])[t:end] - np.asarray(g["rewards"])[t:end]).tolist())
            stage[task].extend(stage_delta(np.asarray(clean["staged_rewards"])[t:end], np.asarray(g["staged_rewards"])[t:end]).tolist()); count[task] += 1
    result = {"source": "v0.2 train paired-clean framewise deltas", "max_horizon": h, "quantile": q, "zero_epsilon": eps, "tasks": {}, "normalizer": {}}
    for task in ("can", "square"):
        ds, ss = robust(dense[task], q, eps), robust(stage[task], q, eps)
        result["tasks"][task] = {"dense_scale": ds, "stage_scale": ss, "train_pair_count": count[task], "dense_nonzero_count": sum(abs(x) > eps for x in dense[task]), "stage_nonzero_count": sum(abs(x) > eps for x in stage[task])}; result["normalizer"][task] = {"dense_scale": ds, "stage_scale": ss}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
