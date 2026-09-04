"""Build frozen-library action replacements at proposer-selected transitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd

from stage3a.rescue_v02.build_teacher_forced_plans import retrieve, text


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--proposal-candidates", type=Path, required=True)
    p.add_argument("--action-library", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation", "blind_test"), required=True)
    p.add_argument("--proposal-source", default="union_top5", choices=("full_top5", "action_top5", "union_top5"))
    p.add_argument("--num-replacements", type=int, default=4)
    p.add_argument("--min-future-steps", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jsonl-output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    if a.num_replacements != 4:
        raise ValueError("protocol fixes four real-action replacements")
    metadata = {str(row["pair_id"]): row for row in load_rows(a.metadata)}
    candidates = pd.read_parquet(a.proposal_candidates)
    if set(candidates.split.unique()) != {a.split}:
        raise ValueError("proposal split does not match requested split")
    membership = f"in_{a.proposal_source}"
    if membership not in candidates:
        raise ValueError(f"missing {membership}")
    candidates = candidates[candidates[membership].fillna(False)].copy()
    candidates = candidates.sort_values(["pair_id", "union_rank", "t"]).drop_duplicates(["pair_id", "t"])
    thresholds = json.loads((a.action_library / "support_thresholds.json").read_text(encoding="utf-8"))
    library, codebook = {}, {}
    table_all = pd.read_parquet(a.action_library / "action_library_index.parquet")
    for task in ("can", "square"):
        table = table_all[table_all.task.eq(task)].reset_index(drop=True)
        with np.load(a.action_library / f"library_{task}.npz") as item:
            arrays = {key: item[key].copy() for key in item.files}
        library[task] = (table, arrays, faiss.read_index(str(a.action_library / f"state_{task}.faiss")))
        codebook[task] = json.loads((a.action_library / f"codebook_{task}.json").read_text(encoding="utf-8"))
    output = []
    skipped = []
    with h5py.File(a.benchmark, "r") as h5:
        for candidate in candidates.itertuples(index=False):
            pair_id, t, task = str(candidate.pair_id), int(candidate.t), str(candidate.task)
            meta = metadata.get(pair_id)
            if meta is None or str(meta["split"]) != a.split:
                raise ValueError(f"metadata mismatch for {pair_id}")
            group = h5[f"data/{candidate.demo_id}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            states = np.asarray(group["states_pre"], dtype=np.float32)
            if len(actions) - t < a.min_future_steps:
                skipped.append({"pair_id": pair_id, "query_t": t, "reason": "insufficient_future"})
                continue
            selected = retrieve(task, states[t], actions[t], str(candidate.base_demo_id), library, thresholds, codebook, a.num_replacements)
            if len(selected) != a.num_replacements:
                raise RuntimeError(f"failed fixed retrieval for {pair_id}:t{t}")
            query_id = f"{pair_id}:t{t}"
            for replacement in selected:
                rank = int(replacement["replacement_rank"])
                output.append({
                    "replacement_id": f"{query_id}:r{rank}", "query_id": query_id, "pair_id": pair_id, "task": task,
                    "base_demo_id": str(candidate.base_demo_id), "split": a.split, "perturbed_demo_id": str(candidate.demo_id),
                    "clean_demo_id": str(meta["clean_demo_id"]), "query_t": t, "episode_length": int(len(actions)),
                    "query_source": f"proposal_{a.proposal_source}", "proposal_full_rank": None if pd.isna(candidate.full_rank) else int(candidate.full_rank),
                    "proposal_action_rank": None if pd.isna(candidate.action_rank) else int(candidate.action_rank),
                    "union_rank": int(candidate.union_rank), "raw_full_score": float(candidate.raw_full_score),
                    "raw_action_score": float(candidate.raw_action_score), "raw_union_score": float(candidate.raw_union_score),
                    "proposal_rank_weight": float(candidate.proposal_rank_weight), "failure_type": str(meta["failure_type"]),
                    "is_effective_intervention": bool(meta["is_effective_intervention"]), "intervention_t": int(meta["perturb_t"]),
                    "responsible_start": max(0, int(meta["perturb_t"]) - 1) if bool(meta["is_effective_intervention"]) else None,
                    "responsible_end": min(len(actions) - 1, int(meta["perturb_t"]) + 1) if bool(meta["is_effective_intervention"]) else None,
                    **replacement,
                })
    table = pd.DataFrame(output).sort_values(["pair_id", "query_t", "replacement_rank"]).reset_index(drop=True)
    if table.duplicated("replacement_id").any() or table.groupby("query_id").size().ne(4).any():
        raise RuntimeError("candidate plans do not contain exactly four unique replacements per query")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.output, index=False)
    a.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    a.jsonl_output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in table.to_dict("records")) + "\n", encoding="utf-8")
    summary = {"split": a.split, "proposal_source": a.proposal_source, "candidate_rows_before_future_filter": int(len(candidates)),
               "query_count": int(table.query_id.nunique()), "replacement_rows": int(len(table)), "skipped_queries": skipped,
               "max_replacements_per_query": int(table.groupby("query_id").size().max()) if len(table) else 0}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
