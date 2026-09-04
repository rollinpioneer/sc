"""Build the non-deduplicated validation teacher-forced primary table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage3a.method_v02r.data.build_verifier_samples import normalize_row, target_is_valid


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return read_jsonl(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-forced", type=Path, required=True)
    p.add_argument("--teacher-plans", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--feature-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()

    metadata = {str(row["pair_id"]): row for row in read_jsonl(args.metadata)}
    plans = {str(row["replacement_id"]): row for row in read_table(args.teacher_plans)}
    features = pd.read_parquet(args.feature_index)
    feature_lookup = {(str(row.task), str(row.demo_id)): str(row.feature_path)
                      for row in features.itertuples(index=False)}

    source = [row for row in read_jsonl(args.teacher_forced)
              if str(row.get("split")) == "validation"
              and int(row.get("query_t", -1)) == int(metadata[str(row["pair_id"])]["perturb_t"])
              and int(row.get("replacement_rank", -1)) == 0]
    rows: list[dict] = []
    invalid = 0
    for row in source:
        # Feasible-long rows intentionally omit action vectors in older runs.
        action = row.get("replacement_action")
        if action is None or (hasattr(action, "__len__") and len(action) == 0):
            plan = plans.get(str(row["replacement_id"]))
            if plan is None:
                raise RuntimeError(f"missing frozen teacher plan {row['replacement_id']}")
            merged = dict(plan)
            merged.update(row)
            row = merged
        valid = target_is_valid(row)
        if not valid:
            invalid += 1
        normalized = normalize_row(row, metadata, feature_lookup, True) if valid else None
        if normalized is None:
            # Keep the complete primary universe even when an oracle branch
            # is outside the frozen state domain.  The model dataset does not
            # consume target_valid; preserving the row makes this diagnostic
            # auditable while its engineering flag remains false.
            pair_id = str(row["pair_id"]); meta = metadata[pair_id]
            normalized = {
                "replacement_id": str(row["replacement_id"]), "query_group_id": str(row.get("query_id", f"{pair_id}:t{int(row['query_t'])}")),
                "pair_id": pair_id, "task": str(row["task"]), "split": "validation", "perturbed_demo_id": str(row["perturbed_demo_id"]),
                "query_t": int(row["query_t"]), "replacement_rank": int(row["replacement_rank"]), "replacement_action": row["replacement_action"],
                "state_distance": float(row["state_distance"]), "action_delta_l2": float(row["action_delta_l2"]),
                "state_in_domain": bool(row["state_in_domain"]), "action_in_domain": bool(row["action_in_domain"]), "target_valid": False,
                "counterfactual_improvement_long": float(row["counterfactual_improvement_long"]), "target_positive": bool(float(row["counterfactual_improvement_long"]) >= .5),
                "is_teacher_forced": True, "failure_type": str(meta["failure_type"]), "is_effective_intervention": bool(meta["is_effective_intervention"]), "intervention_t": int(meta["perturb_t"]),
                "responsible_start": max(0, int(meta["perturb_t"]) - 1) if bool(meta["is_effective_intervention"]) else None, "responsible_end": int(meta["perturb_t"]) + 1 if bool(meta["is_effective_intervention"]) else None,
                "proposal_full_rank": None, "proposal_action_rank": None, "union_rank": None, "raw_full_score": None, "raw_action_score": None, "raw_union_score": None, "proposal_rank_weight": None,
            }
        rows.append(normalized)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("teacher-forced primary table is empty")
    frame = frame.sort_values(["pair_id", "query_t", "replacement_rank"]).reset_index(drop=True)
    if frame.pair_id.duplicated().any():
        raise RuntimeError("teacher-forced primary table has duplicate pair IDs")
    expected_pairs = 256
    expected_effective = 29
    if len(frame) != expected_pairs or int(frame.is_effective_intervention.sum()) != expected_effective:
        raise RuntimeError({"pairs": len(frame), "effective_pairs": int(frame.is_effective_intervention.sum()),
                            "source_rows": len(source), "invalid": invalid})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    summary = {"schema": "stage3f_r_complete_teacher_forced_v1", "split": "validation",
               "pairs": int(len(frame)), "effective_pairs": int(frame.is_effective_intervention.sum()),
               "source_rows": len(source), "invalid_rows": invalid, "teacher_forced": True}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
