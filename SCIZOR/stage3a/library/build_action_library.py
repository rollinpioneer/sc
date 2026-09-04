"""Build a train-only, real-action nearest-neighbour library."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

try:
    import faiss
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise RuntimeError(
        "3A-C requires faiss-cpu so state_{task}.faiss is a real IndexFlatL2"
    ) from exc


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feature-index", type=Path, required=True)
    p.add_argument("--normalizer", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260902)
    return p.parse_args()


def normalized(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / np.maximum(std, 1e-6), -10.0, 10.0).astype(np.float32)


def main() -> None:
    a = args(); cfg = json.loads(a.config.read_text()); a.output_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(a.feature_index)
    idx = idx[(idx["split"] == "train") & (idx["variant"] == "clean")].copy().reset_index(drop=True)
    if idx.empty: raise ValueError("train clean feature index is empty")
    with np.load(a.normalizer) as z:
        state_mean, state_std = z["state_mean"], z["state_std"]
    records = {"can": [], "square": []}
    arrays = {"can": {k: [] for k in ("vectors", "states", "actions", "previous_actions", "relative_positions")}, "square": {k: [] for k in ("vectors", "states", "actions", "previous_actions", "relative_positions")}}
    for row in idx.itertuples(index=False):
        f = np.load(row.feature_path)
        states, actions = f["states"].astype(np.float32), f["actions"].astype(np.float32)
        valid = f["state_valid_mask"].astype(bool)
        ns = normalized(states, state_mean, state_std); ns[:, ~valid] = 0.0
        task = str(row.task); n = len(actions)
        for t in range(n):
            prev = float(actions[max(0, t - 1), -1]); rel = float(t / max(n - 1, 1))
            q = np.concatenate([ns[t], np.asarray([rel, prev], dtype=np.float32)])
            q /= max(float(np.linalg.norm(q)), 1e-8)
            lid = len(records[task])
            records[task].append({"library_id": f"{task}_lib_{lid:06d}", "task": task, "demo_id": str(row.demo_id), "base_demo_id": str(row.base_demo_id), "t": t, "episode_length": n, "feature_path": str(row.feature_path), "state_dim": int(row.state_dim), "relative_position": rel, "previous_gripper_action": prev})
            for key, val in (("vectors", q), ("states", ns[t]), ("actions", actions[t]), ("previous_actions", np.asarray([prev], dtype=np.float32)), ("relative_positions", np.asarray([rel], dtype=np.float32))): arrays[task][key].append(val)
        f.close()
    thresholds = {}
    for task in ("can", "square"):
        tab = pd.DataFrame(records[task]); ar = {k: np.asarray(v, dtype=np.float32) for k,v in arrays[task].items()}
        tab["library_row"] = np.arange(len(tab), dtype=np.int64)
        tab["action_cluster_id"] = -1
        # Support is computed from clean queries while excluding their own demo.
        # Pairwise matrices are small for the train-clean split and avoid a
        # Python loop over every transition while preserving the same nearest
        # valid neighbour definition.
        n = len(tab)
        vv = ar["vectors"].astype(np.float32)
        dist_matrix = np.maximum(0.0, (vv * vv).sum(1)[:, None] + (vv * vv).sum(1)[None, :] - 2.0 * vv.dot(vv.T))
        same_demo = tab.base_demo_id.to_numpy()[:, None] == tab.base_demo_id.to_numpy()[None, :]
        dist_matrix[same_demo] = np.inf
        nearest = np.argmin(dist_matrix, axis=1)
        finite = np.isfinite(dist_matrix[np.arange(n), nearest])
        dists = dist_matrix[np.arange(n)[finite], nearest[finite]].tolist()
        deltas = np.linalg.norm(ar["actions"][nearest[finite]] - ar["actions"][np.arange(n)[finite]], axis=1).tolist()
        thresholds[task] = {"state_distance_q95": float(np.quantile(dists, 0.95)) if dists else 0.0, "action_delta_q99": float(np.quantile(deltas, 0.99)) if deltas else 0.0, "num_queries": len(dists)}
        k = min(int(cfg["library"]["action_codebook_size_per_task"]), len(tab))
        km = MiniBatchKMeans(n_clusters=max(1, k), random_state=a.seed, n_init=3, batch_size=min(2048, len(tab))).fit(ar["actions"])
        tab["action_cluster_id"] = km.labels_.astype(np.int64)
        medoids = []
        for cid, center in enumerate(km.cluster_centers_):
            members = np.flatnonzero(km.labels_ == cid)
            if len(members): medoids.append({"cluster_id": int(cid), "library_id": str(tab.iloc[members[np.argmin(((ar["actions"][members] - center) ** 2).sum(axis=1))]].library_id), "action": ar["actions"][members[np.argmin(((ar["actions"][members] - center) ** 2).sum(axis=1))]].astype(float).tolist(), "cluster_size": int(len(members))})
        ar["state_mean"] = np.asarray(state_mean, dtype=np.float32)
        ar["state_std"] = np.asarray(state_std, dtype=np.float32)
        np.savez(a.output_dir / f"library_{task}.npz", **ar)
        state_index = faiss.IndexFlatL2(int(ar["vectors"].shape[1]))
        state_index.add(np.ascontiguousarray(ar["vectors"].astype(np.float32)))
        faiss.write_index(state_index, str(a.output_dir / f"state_{task}.faiss"))
        (a.output_dir / f"codebook_{task}.json").write_text(json.dumps({"task": task, "medoids": medoids}, indent=2), encoding="utf-8")
        records[task] = tab.to_dict("records")
    pd.DataFrame(records["can"] + records["square"]).to_parquet(a.output_dir / "action_library_index.parquet", index=False)
    (a.output_dir / "support_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(records["can"]) + len(records["square"]), "by_task": {k: len(v) for k,v in records.items()}, "thresholds": thresholds}, indent=2))


if __name__ == "__main__": main()
