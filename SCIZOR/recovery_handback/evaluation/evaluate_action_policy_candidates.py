"""Evaluate fixed Robomimic action-policy candidates on one HB1 role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump
from recovery_handback.evaluation.evaluate_base import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--role", default="base_val")
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    payload = json.loads(args.candidate_metadata.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"no candidates in {args.candidate_metadata}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, candidate in enumerate(candidates):
        checkpoint = Path(candidate["checkpoint"]).expanduser().resolve()
        summary = evaluate(
            config,
            assets,
            args.task,
            args.role,
            checkpoint,
            args.output_dir / f"candidate_{index:02d}",
            device=args.device,
        )
        engineering_failures = sum(
            bool(row.get("exception_reason")) for row in summary.get("rows", [])
        )
        rows.append({
            "candidate_index": index,
            "candidate": candidate,
            "checkpoint": str(checkpoint),
            "roots": int(summary["roots"]),
            "successes": int(summary["successes"]),
            "success_rate": float(summary["success_rate"]),
            "engineering_failures": engineering_failures,
            "action_dim": int(summary.get("action_dim", -1)),
            "eligible": (
                engineering_failures == 0
                and int(summary.get("action_dim", -1)) == 7
            ),
        })

    rows.sort(key=lambda row: (
        -int(row["eligible"]),
        -row["successes"],
        int(row["candidate"].get("training_steps", 0)),
    ))
    selected = rows[0]
    minimum = int(
        config["capability_repair"][
            "can_diagnostic" if args.task == "can" else "square_teacher"
        ].get("minimum_successes_on_base_val", 0)
    )
    status = (
        "QUALIFIED"
        if selected["eligible"] and selected["successes"] >= minimum
        else ("HOLD_ENGINEERING_FIX" if not selected["eligible"] else "INSUFFICIENT_CAPABILITY")
    )
    result = {
        "schema_version": "hb1_action_policy_candidate_evaluation_v1",
        "task": args.task,
        "role": args.role,
        "status": status,
        "minimum_successes": minimum,
        "selected_candidate": selected["candidate"],
        "selection": selected,
        "candidates": rows,
    }
    atomic_json_dump(result, args.output_dir / "summary.json")
    atomic_json_dump(selected["candidate"], args.output_dir / "selected_candidate.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
