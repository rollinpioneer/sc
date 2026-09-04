"""Build the frozen v0.2 train-clean real-action library.

This is intentionally independent of the simulator runtime: it is run in the
curation environment, where FAISS, sklearn and a parquet engine are pinned.
Only clean groups tagged ``split=train`` are eligible as library members.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--state-key", default="states_pre")
    p.add_argument("--split", default="train")
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260902)
    a = p.parse_args()
    if a.split != "train":
        raise ValueError("v0.2 action library must be built from train only")
    if a.state_key != "states_pre":
        raise ValueError("v0.2 protocol freezes state key to states_pre")
    manifest = json.loads(a.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("test"):
        raise ValueError("v0.1 test / test split is forbidden in v0.2")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = {"can": [], "square": []}
    raw_states = {"can": [], "square": []}
    actions = {"can": [], "square": []}
    with h5py.File(a.benchmark, "r") as h5:
        for name, g in h5["data"].items():
            if text(g.attrs.get("variant", "")) != "clean" or text(g.attrs.get("split", "")) != "train":
                continue
            task = text(g.attrs["task"])
            if task not in records:
                raise ValueError(f"unexpected task: {task}")
            states = np.asarray(g[a.state_key], dtype=np.float32)
            acts = np.asarray(g["actions"], dtype=np.float32)
            if len(states) != len(acts):
                raise ValueError(f"{name}: state/action length mismatch")
            base = text(g.attrs["base_demo_id"])
            for t in range(len(acts)):
                records[task].append({
                    "library_row": len(records[task]), "library_id": f"{task}_lib_{len(records[task]):06d}",
                    "task": task, "base_demo_id": base, "t": int(t), "clean_demo_id": name,
                })
            raw_states[task].append(states)
            actions[task].append(acts)
    support, summary = {}, {"rows": {}, "train_clean_demos": {}}
    all_index = []
    for task in ("can", "square"):
        if not records[task]:
            raise ValueError(f"no train clean rollouts for {task}")
        states = np.concatenate(raw_states[task], axis=0).astype(np.float32)
        acts = np.concatenate(actions[task], axis=0).astype(np.float32)
        tab = pd.DataFrame(records[task])
        mean, std = states.mean(0), states.std(0)
        normalized = np.clip((states - mean) / np.maximum(std, 1e-6), -10.0, 10.0).astype(np.float32)
        index = faiss.IndexFlatL2(normalized.shape[1]); index.add(np.ascontiguousarray(normalized))
        # The support calibration excludes the originating base demo exactly as
        # online retrieval does.  Query progressively deeper only if needed.
        base_ids = tab.base_demo_id.to_numpy()
        k = min(len(tab), 128)
        distances, neighbors = index.search(np.ascontiguousarray(normalized), k)
        best_d, best_delta = [], []
        for i in range(len(tab)):
            chosen = next((int(j) for d, j in zip(distances[i], neighbors[i]) if j >= 0 and base_ids[int(j)] != base_ids[i]), None)
            if chosen is not None:
                best_d.append(float(np.sum((normalized[i] - normalized[chosen]) ** 2)))
                best_delta.append(float(np.linalg.norm(acts[i] - acts[chosen])))
        if not best_d:
            raise ValueError(f"{task}: no cross-demo neighbors for support")
        clusters = min(int(a.codebook_size), len(acts))
        km = MiniBatchKMeans(n_clusters=clusters, random_state=a.seed, n_init=3, batch_size=min(2048, len(acts))).fit(acts)
        tab["action_cluster_id"] = km.labels_.astype(np.int64)
        medoids = []
        for cid, center in enumerate(km.cluster_centers_):
            members = np.flatnonzero(km.labels_ == cid)
            j = int(members[np.argmin(np.sum((acts[members] - center) ** 2, axis=1))])
            medoids.append({"cluster_id": int(cid), "library_row": j, "library_id": str(tab.iloc[j].library_id), "action": acts[j].astype(float).tolist(), "cluster_size": int(len(members))})
        np.savez(a.output_dir / f"library_{task}.npz", states=normalized, actions=acts, state_mean=mean.astype(np.float32), state_std=std.astype(np.float32))
        faiss.write_index(index, str(a.output_dir / f"state_{task}.faiss"))
        (a.output_dir / f"codebook_{task}.json").write_text(json.dumps({"task": task, "medoids": medoids}, indent=2), encoding="utf-8")
        support[task] = {"state_distance_q95": float(np.quantile(best_d, 0.95)), "action_delta_q99": float(np.quantile(best_delta, 0.99)), "num_queries": len(best_d)}
        summary["rows"][task] = len(tab); summary["train_clean_demos"][task] = int(tab.base_demo_id.nunique())
        all_index.append(tab)
    pd.concat(all_index, ignore_index=True).to_parquet(a.output_dir / "action_library_index.parquet", index=False)
    (a.output_dir / "support_thresholds.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
    summary["support_thresholds"] = support
    (a.output_dir / "action_library_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
