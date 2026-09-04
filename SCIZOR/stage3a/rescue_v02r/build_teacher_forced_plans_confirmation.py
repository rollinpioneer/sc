"""Create deterministic, train-library replacement plans at each true perturb_t."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def json_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieve(task, state, action, base_id, lib, thresholds, codebook, count):
    table, arrays, index = lib[task]
    ar = arrays
    q = np.clip((np.asarray(state, np.float32) - ar["state_mean"]) / np.maximum(ar["state_std"], 1e-6), -10.0, 10.0).astype(np.float32)
    k = min(64, len(table)); distances, indices = index.search(np.ascontiguousarray(q[None]), k)
    max_delta = float(thresholds[task]["action_delta_q99"]); state_limit = float(thresholds[task]["state_distance_q95"])
    chosen, used_clusters = [], set()
    def valid(j, state_distance, require_state_support, require_diversity=True):
        row = table.iloc[int(j)]
        delta = float(np.linalg.norm(ar["actions"][j] - action))
        if str(row.base_demo_id) == str(base_id) or (require_state_support and state_distance > state_limit) or not (0.02 <= delta <= max_delta):
            return None
        if require_diversity and any(float(np.linalg.norm(ar["actions"][j] - ar["actions"][old])) < 0.10 for old in chosen):
            return None
        return row, delta
    for d, j in zip(distances[0], indices[0]):
        if j < 0: continue
        got = valid(int(j), float(d), True)
        if got is None: continue
        row, delta = got; chosen.append(int(j)); used_clusters.add(int(row.action_cluster_id))
        if len(chosen) == count: break
    if len(chosen) < count:
        for medoid in sorted(codebook[task]["medoids"], key=lambda x: (-x["cluster_size"], x["cluster_id"])):
            if len(chosen) == count: break
            if int(medoid["cluster_id"]) in used_clusters: continue
            # Frozen Stage 3A-C policy: a real-action medoid may fill an
            # otherwise insufficient neighborhood. Its state distance remains
            # recorded and is marked out-of-domain below when above q95.
            j = int(medoid["library_row"]); d = float(np.sum((ar["states"][j] - q) ** 2)); got = valid(j, d, False)
            if got is None: continue
            row, delta = got; chosen.append(j); used_clusters.add(int(row.action_cluster_id))
    # The frozen primary retrieval remains above.  A tiny number of extreme
    # reverse-motion queries cannot meet four-way diversity within the fixed
    # 64-medoid set; fill only their non-primary tail slot with a real action.
    diversity_fallback = set(); global_fallback = set()
    if len(chosen) < count:
        for medoid in sorted(codebook[task]["medoids"], key=lambda x: (-x["cluster_size"], x["cluster_id"])):
            if len(chosen) == count: break
            j = int(medoid["library_row"])
            if j in chosen: continue
            d = float(np.sum((ar["states"][j] - q) ** 2)); got = valid(j, d, False, require_diversity=False)
            if got is None: continue
            chosen.append(j); diversity_fallback.add(j)
    if len(chosen) < count:
        action_order = np.argsort(np.linalg.norm(ar["actions"] - action, axis=1), kind="stable")
        for j in action_order:
            if len(chosen) == count: break
            j = int(j)
            if j in chosen: continue
            d = float(np.sum((ar["states"][j] - q) ** 2)); got = valid(j, d, False, require_diversity=False)
            if got is None: continue
            chosen.append(j); global_fallback.add(j)
    out = []
    for rank, j in enumerate(chosen):
        row = table.iloc[j]; d = float(np.sum((ar["states"][j] - q) ** 2)); delta = float(np.linalg.norm(ar["actions"][j] - action))
        source = "nn_real" if rank < 3 else "codebook_medoid"
        if j in diversity_fallback: source = "codebook_medoid_diversity_fallback"
        if j in global_fallback: source = "global_real_action_fallback"
        out.append({"replacement_rank": rank, "replacement_source": source, "replacement_action": ar["actions"][j].astype(float).tolist(), "library_id": str(row.library_id), "library_base_demo_id": str(row.base_demo_id), "library_clean_demo_id": str(row.clean_demo_id), "library_t": int(row.t), "state_distance": d, "action_delta_l2": delta, "state_in_domain": bool(d <= state_limit), "action_in_domain": bool(0.02 <= delta <= max_delta)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True); p.add_argument("--action-library", type=Path, required=True)
    p.add_argument("--query-source", default="perturb_t"); p.add_argument("--num-replacements", type=int, default=4)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--jsonl-output", type=Path); a = p.parse_args()
    if a.query_source != "perturb_t" or a.num_replacements != 4:
        raise ValueError("v0.2 protocol freezes true perturb_t and four replacements")
    manifest = json.loads(a.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("test"): raise ValueError("test split is forbidden")
    metadata = {str(r["pair_id"]): r for r in json_rows(a.metadata)}
    thresholds = json.loads((a.action_library / "support_thresholds.json").read_text())
    lib, codebook = {}, {}
    for task in ("can", "square"):
        table = pd.read_parquet(a.action_library / "action_library_index.parquet"); table = table[table.task.eq(task)].reset_index(drop=True)
        with np.load(a.action_library / f"library_{task}.npz") as z: arrays = {k: z[k].copy() for k in z.files}
        lib[task] = (table, arrays, faiss.read_index(str(a.action_library / f"state_{task}.faiss")))
        codebook[task] = json.loads((a.action_library / f"codebook_{task}.json").read_text())
    rows, short = [], []
    with h5py.File(a.benchmark, "r") as h5:
        for name, g in h5["data"].items():
            if text(g.attrs.get("variant", "")) != "perturbed": continue
            pair_id, task = text(g.attrs["pair_id"]), text(g.attrs["task"])
            meta = metadata.get(pair_id)
            if meta is None: raise ValueError(f"metadata missing {pair_id}")
            split = str(meta["split"])
            if split not in {"train", "validation", "confirmation"}: raise ValueError(f"forbidden split {split}")
            t = int(g.attrs["perturb_t"]); actions = np.asarray(g["actions"], np.float32); states = np.asarray(g["states_pre"], np.float32)
            found = retrieve(task, states[t], actions[t], text(g.attrs["base_demo_id"]), lib, thresholds, codebook, a.num_replacements)
            if len(found) != a.num_replacements: short.append({"pair_id": pair_id, "found": len(found)}); continue
            for repl in found:
                rid = f"{pair_id}|t{t}|r{repl['replacement_rank']}|{repl['replacement_source']}"
                rows.append({"replacement_id": rid, "query_id": f"{pair_id}|t{t}", "pair_id": pair_id, "task": task, "base_demo_id": text(g.attrs["base_demo_id"]), "split": split, "perturbed_demo_id": name, "clean_demo_id": text(g.attrs["clean_demo_id"]), "query_t": t, "episode_length": len(actions), "query_source": "teacher_forced_perturb_t", "failure_type": text(g.attrs.get("failure_type", "")), "is_effective_intervention": bool(g.attrs.get("is_effective_intervention", False)), **repl})
    if short: raise RuntimeError(f"could not produce four replacements for {len(short)} pairs: {short[:5]}")
    a.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(a.output, index=False)
    if a.jsonl_output:
        a.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        a.jsonl_output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(rows) // 4, "replacement_rows": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")}}, indent=2))


if __name__ == "__main__": main()
