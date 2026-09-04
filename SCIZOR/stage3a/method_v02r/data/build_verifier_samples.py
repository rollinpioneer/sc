"""Combine candidate and teacher-forced oracle rows into verifier sample tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EFFECTIVE = {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rows(path: Path) -> list[dict]:
    """Read either parquet or JSONL plan rows without changing their schema."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return read_jsonl(path)


def canonical(row: dict) -> tuple[str, int, int]:
    return str(row["pair_id"]), int(row["query_t"]), int(row["replacement_rank"])


def target_is_valid(row: dict) -> bool:
    return all(bool(row.get(key, False)) for key in ("branch_pre_state_equal", "reference_exact", "finite_target", "state_in_domain", "action_in_domain"))


def normalize_row(row: dict, metadata: dict, feature_lookup: dict, teacher_forced: bool) -> dict | None:
    pair_id = str(row["pair_id"]); meta = metadata[pair_id]
    task = str(row["task"]); demo = str(row["perturbed_demo_id"])
    if (task, demo) not in feature_lookup:
        raise ValueError(f"missing feature for {task}/{demo}")
    effective = bool(meta["is_effective_intervention"])
    intervention = int(meta["perturb_t"])
    if not target_is_valid(row):
        return None
    score = float(row["counterfactual_improvement_long"])
    return {
        "replacement_id": str(row["replacement_id"]), "query_group_id": str(row.get("query_id", f"{pair_id}:t{int(row['query_t'])}")),
        "pair_id": pair_id, "task": task, "split": str(meta["split"]), "perturbed_demo_id": demo, "query_t": int(row["query_t"]),
        "replacement_rank": int(row["replacement_rank"]), "replacement_action": row["replacement_action"],
        "state_distance": float(row["state_distance"]), "action_delta_l2": float(row["action_delta_l2"]),
        "state_in_domain": bool(row["state_in_domain"]), "action_in_domain": bool(row["action_in_domain"]), "target_valid": True,
        "counterfactual_improvement_long": score, "target_positive": bool(score >= 0.5), "is_teacher_forced": teacher_forced,
        "failure_type": str(meta["failure_type"]), "is_effective_intervention": effective, "intervention_t": intervention,
        "responsible_start": max(0, intervention - 1) if effective else None, "responsible_end": intervention + 1 if effective else None,
        "proposal_full_rank": row.get("proposal_full_rank"), "proposal_action_rank": row.get("proposal_action_rank"),
        "union_rank": row.get("union_rank"), "raw_full_score": row.get("raw_full_score"), "raw_action_score": row.get("raw_action_score"), "raw_union_score": row.get("raw_union_score"), "proposal_rank_weight": row.get("proposal_rank_weight"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-train", type=Path, required=True); p.add_argument("--candidate-validation", type=Path, required=True)
    p.add_argument("--teacher-forced", type=Path, required=True); p.add_argument("--teacher-plans", type=Path, required=True)
    p.add_argument("--plans-train", type=Path); p.add_argument("--plans-validation", type=Path)
    p.add_argument("--metadata", type=Path, required=True); p.add_argument("--feature-index", type=Path, required=True)
    p.add_argument("--output-train", type=Path, required=True); p.add_argument("--output-validation", type=Path, required=True); p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    metadata = {str(row["pair_id"]): row for row in read_jsonl(a.metadata)}
    features = pd.read_parquet(a.feature_index)
    feature_lookup = {(str(row.task), str(row.demo_id)): str(row.feature_path) for row in features.itertuples(index=False)}
    teacher_rows = read_jsonl(a.teacher_forced)
    teacher_plans = {str(row["replacement_id"]): row for row in read_rows(a.teacher_plans)}
    if not teacher_plans:
        raise RuntimeError("teacher-forced plan table is empty")
    result = {}
    for split, candidate_path, output in (("train", a.candidate_train, a.output_train), ("validation", a.candidate_validation, a.output_validation)):
        candidate_all = [row for row in read_jsonl(candidate_path) if str(row["split"]) == split]
        candidate = [row for row in candidate_all if target_is_valid(row)]
        # A valid candidate occupies the semantic slot. An invalid candidate
        # cannot suppress a valid teacher-forced target for the same query.
        candidate_keys = {canonical(row) for row in candidate}
        teacher_all = [row for row in teacher_rows if str(row["split"]) == split and canonical(row) not in candidate_keys]
        enriched_teacher = []
        for row in teacher_all:
            # Older feasible-long artifacts did not copy replacement_action
            # from their plan. Recover it from the frozen v0.2 teacher plan;
            # oracle labels and engineering flags remain authoritative.
            if not row.get("replacement_action"):
                plan = teacher_plans.get(str(row["replacement_id"]))
                if plan is None:
                    raise RuntimeError(f"missing frozen teacher plan {row['replacement_id']}")
                merged = dict(plan)
                merged.update(row)
                row = merged
            enriched_teacher.append(row)
        teacher = [row for row in enriched_teacher if target_is_valid(row)]
        records = [normalize_row(row, metadata, feature_lookup, False) for row in candidate]
        records.extend(normalize_row(row, metadata, feature_lookup, True) for row in teacher)
        frame = pd.DataFrame(records).sort_values(["pair_id", "query_t", "replacement_rank", "is_teacher_forced"]).reset_index(drop=True)
        if frame.empty or frame.replacement_id.duplicated().any() or not frame.target_valid.all() or set(frame.split.unique()) != {split}:
            raise RuntimeError(f"invalid {split} verifier sample table")
        sizes = frame.groupby("query_group_id").size()
        if sizes.min() < 1 or sizes.max() > 4:
            raise RuntimeError(f"invalid query group sizes for {split}")
        output.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(output, index=False)
        result[split] = {"sample_rows": int(len(frame)), "query_groups": int(frame.query_group_id.nunique()), "teacher_forced_rows_added": int(frame.is_teacher_forced.sum()), "candidate_rows_excluded_invalid": int(len(candidate_all) - len(candidate)), "teacher_rows_excluded_invalid": int(len(teacher_all) - len(teacher)), "positive_rate": float(frame.target_positive.mean()), "valid_rate": float(frame.target_valid.mean()), "max_group_size": int(sizes.max())}
    a.summary.parent.mkdir(parents=True, exist_ok=True); a.summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
