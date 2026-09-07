"""Qualify and freeze one HB1 repair checkpoint on repair_val roots."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import re
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json, write_table
from recovery_handback.execution.paired_branch import run_branch


def _run_branch_job(job: tuple[dict, dict, dict, int | str, str, str, str | None, str | None]) -> dict:
    """Execute one uniquely-owned branch in a fresh worker process."""
    config, anchor, pair, repair_length, output_path, device, base_device, repair_device = job
    return run_branch(
        config,
        anchor,
        pair,
        repair_length,
        output_path,
        device=device,
        base_device=base_device,
        repair_device=repair_device,
    )


def _run_jobs(jobs: list[tuple[dict, dict, dict, int | str, str, str, str | None, str | None]], workers: int) -> list[dict]:
    if workers <= 1:
        return [_run_branch_job(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    rows: list[dict | None] = [None] * len(jobs)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        future_to_index = {
            executor.submit(_run_branch_job, job): index
            for index, job in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            rows[index] = future.result()
    return [row for row in rows if row is not None]


def _reusable_record(path: Path, config_hash: str, pair_hash: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not bool(row.get("engineering_ok"))
        or row.get("config_hash") != config_hash
        or row.get("policy_pair_hash") != pair_hash
    ):
        return None
    return row


def _run_or_resume(
    config: dict,
    anchors: list[dict],
    pair: dict,
    repair_length: int | str,
    output_dir: Path,
    subdir: str,
    config_hash: str,
    workers: int,
    device: str,
    base_device: str | None,
    repair_device: str | None,
    resume: bool,
) -> list[dict]:
    rows: list[dict | None] = [None] * len(anchors)
    jobs = []
    positions = []
    pair_hash = sha256_json(pair)
    for index, anchor in enumerate(anchors):
        path = output_dir / "records" / subdir / f"{index:04d}.json"
        reused = _reusable_record(path, config_hash, pair_hash) if resume else None
        if reused is not None:
            rows[index] = reused
            continue
        jobs.append((config, anchor, pair, repair_length, str(path), device, base_device, repair_device))
        positions.append(index)
    for position, row in zip(positions, _run_jobs(jobs, workers)):
        rows[position] = row
    if any(row is None for row in rows):
        raise RuntimeError(f"missing qualification result in {subdir}")
    return [row for row in rows if row is not None]


def _evaluation_cost(rows: list[dict]) -> dict:
    return {
        "branches": len(rows),
        "prefix_env_steps": sum(int(row.get("prefix_env_steps") or 0) for row in rows),
        "continuation_env_steps": sum(int(row.get("continuation_env_steps") or 0) for row in rows),
        "base_policy_calls": sum(int(row.get("base_policy_calls") or 0) for row in rows),
        "repair_policy_calls": sum(int(row.get("repair_policy_calls") or 0) for row in rows),
        "wall_seconds": sum(float(row.get("wall_seconds") or 0.0) for row in rows),
    }


def _anchors(config: dict, roots_path: Path, task: str) -> list[dict]:
    output = []
    for root in read_table(roots_path):
        if root.get("exception_reason"):
            continue
        with np.load(root["rollout_path"], allow_pickle=False) as handle:
            success = np.asarray(handle["success"], dtype=bool)
        for anchor_t in sorted(int(value) for value in config["anchor_times"]):
            if anchor_t >= len(success) or bool(success[:anchor_t].any()):
                continue
            if int(config["horizon_steps"]) - anchor_t < int(config["minimum_remaining_steps"]):
                continue
            output.append({
                "anchor_id": f"{root['root_id']}:{anchor_t}", "task": task,
                "role": "repair_val", "root_id": root["root_id"],
                "stat_group_id": root.get("stat_group_id", root["root_id"]),
                "anchor_t": anchor_t, "remaining_steps": int(config["horizon_steps"]) - anchor_t,
                "rollout_path": root["rollout_path"],
                "canonical_payload_path": root.get("canonical_payload_path", root.get("payload_path")),
                "anchor_history_path": root.get("anchor_history_path"),
                "policy_memory_check_path": root.get("policy_memory_path", root.get("policy_memory_check_path")),
                "baseline_success": bool(root.get("baseline_success")),
                "initial_state_hash": root.get("initial_state_hash"),
            })
    return sorted(output, key=lambda row: (row["root_id"], row["anchor_t"]))


def _metadata(checkpoint: Path, normalizer: Path) -> dict:
    candidates = [checkpoint.with_suffix(".json"), checkpoint.parent / "training_summary.json"]
    meta = {}
    for candidate in candidates:
        if candidate.is_file():
            meta.update(json.loads(candidate.read_text(encoding="utf-8")))
            break
    shape = meta.get("observation_shape")
    if not shape:
        raise RuntimeError(f"repair observation_shape missing beside {checkpoint}")
    match = re.search(r"(?:sac_)(\d+)", checkpoint.stem)
    return {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "normalizer": str(normalizer.resolve()), "normalizer_sha256": sha256_file(normalizer),
        "observation_shape": list(shape), "algorithm": meta.get("algorithm", "sac_bounded_residual_adapter"),
        "action_mode": meta.get("action_mode", "residual"), "uses_privileged_input": bool(meta.get("privileged_repairer", True)),
        "training_steps": int(match.group(1)) if match else int(meta.get("sac_transitions", 0)),
    }


def _candidates(args, config: dict) -> list[dict]:
    if args.repair_metadata:
        payload = json.loads(args.repair_metadata.read_text(encoding="utf-8"))
        items = payload.get("candidates")
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"repair metadata has no candidates: {args.repair_metadata}")
        candidates = []
        for item in items:
            candidate = dict(item)
            checkpoint = Path(candidate["checkpoint"]).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            candidate["checkpoint"] = str(checkpoint)
            candidate["checkpoint_sha256"] = sha256_file(checkpoint)
            normalizer = candidate.get("normalizer")
            if normalizer:
                normalizer_path = Path(normalizer).expanduser().resolve()
                if not normalizer_path.is_file():
                    raise FileNotFoundError(normalizer_path)
                candidate["normalizer"] = str(normalizer_path)
                candidate["normalizer_sha256"] = sha256_file(normalizer_path)
            candidates.append(candidate)
        return candidates
    if args.repair_checkpoint:
        if not args.repair_normalizer:
            raise SystemExit("--repair-normalizer is required with --repair-checkpoint")
        return [_metadata(args.repair_checkpoint, args.repair_normalizer)]
    if not args.repair_runs:
        raise SystemExit("provide --repair-runs or an explicit repair checkpoint and normalizer")
    items = []
    for step in config["repair_training"]["checkpoint_steps"]:
        checkpoint = args.repair_runs / f"sac_{int(step)}.zip"
        normalizer = args.repair_runs / f"vecnormalize_{int(step)}.pkl"
        if checkpoint.is_file() and normalizer.is_file():
            items.append(_metadata(checkpoint, normalizer))
    final_checkpoint = args.repair_runs / "sac_final.zip"
    final_normalizer = args.repair_runs / "vecnormalize_final.pkl"
    if not items and final_checkpoint.is_file() and final_normalizer.is_file():
        items.append(_metadata(final_checkpoint, final_normalizer))
    if not items:
        raise RuntimeError(f"no complete repair checkpoint/normalizer pair under {args.repair_runs}")
    return items[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--task", choices=("can", "square"), required=True)
    parser.add_argument("--base-policy-json", type=Path, required=True)
    parser.add_argument("--repair-runs", type=Path)
    parser.add_argument("--repair-checkpoint", type=Path)
    parser.add_argument("--repair-normalizer", type=Path)
    parser.add_argument("--repair-metadata", type=Path)
    parser.add_argument("--validation-roots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-device")
    parser.add_argument("--repair-device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent worker processes for branch evaluation; each output path is unique.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads(args.base_policy_json.read_text(encoding="utf-8"))
    anchors = _anchors(config, args.validation_roots, args.task)
    if not anchors:
        raise RuntimeError("repair validation roots produced no fixed-time anchors")
    candidates = _candidates(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_json(config)
    baseline = {}
    all_rows = []
    baseline_rows = _run_or_resume(
        config, anchors, {"base": base}, 0, args.output_dir, "baseline",
        config_hash, args.workers, args.device, args.base_device,
        args.repair_device, args.resume,
    )
    for anchor, row in zip(anchors, baseline_rows):
        baseline[anchor["anchor_id"]] = row
        all_rows.append(row)
    baseline_engineering_complete = sum(bool(row["engineering_ok"]) for row in baseline.values())
    baseline_cost = _evaluation_cost(list(baseline.values()))
    summaries = []
    for candidate_index, candidate in enumerate(candidates):
        pair = {
            "schema_version": "hb1_qualification_pair_v2_no_extra_observation",
            "task": args.task,
            "base": base,
            "repair": candidate,
        }
        rows = _run_or_resume(
            config, anchors, pair, "full", args.output_dir,
            f"candidate_{candidate_index:02d}", config_hash, args.workers,
            args.device, args.base_device, args.repair_device, args.resume,
        )
        for row in rows:
            row["qualification_candidate"] = candidate_index
            all_rows.append(row)
        base_success = {key: bool(value["system_success"]) for key, value in baseline.items()}
        repair_success = {row["anchor_id"]: bool(row["system_success"]) for row in rows}
        rescued_roots = {
            row["root_id"] for row in rows
            if not base_success[row["anchor_id"]] and repair_success[row["anchor_id"]]
        }
        harmed_roots = {
            row["root_id"] for row in rows
            if base_success[row["anchor_id"]] and not repair_success[row["anchor_id"]]
        }
        successes = sum(repair_success.values())
        baseline_successes = sum(base_success.values())
        summaries.append({
            "candidate_index": candidate_index, "repair": candidate,
            "anchors": len(anchors), "engineering_complete": sum(bool(row["engineering_ok"]) for row in rows),
            "baseline_engineering_complete": baseline_engineering_complete,
            "baseline_evaluation_cost": baseline_cost,
            "repair_evaluation_cost": _evaluation_cost(rows),
            "baseline_successes": baseline_successes, "repair_successes": successes,
            "baseline_success_rate": baseline_successes / len(anchors),
            "repair_success_rate": successes / len(anchors),
            "paired_success_rate_difference": (successes - baseline_successes) / len(anchors),
            "base_failed_repair_succeeded_roots": len(rescued_roots),
            "base_succeeded_repair_failed_roots": len(harmed_roots),
            "rescued_root_ids": sorted(rescued_roots), "harmed_root_ids": sorted(harmed_roots),
        })
    summaries.sort(key=lambda row: (
        -int(
            row["engineering_complete"] == len(anchors)
            and row["baseline_engineering_complete"] == len(anchors)
        ),
        -row["base_failed_repair_succeeded_roots"],
        -row["repair_success_rate"],
        row["base_succeeded_repair_failed_roots"],
        row["repair"]["training_steps"],
    ))
    selected = summaries[0]
    if (
        selected["engineering_complete"] != len(anchors)
        or selected["baseline_engineering_complete"] != len(anchors)
    ):
        status = "HOLD_ENGINEERING_FIX"
    else:
        required = int(
            config["capability_repair"]["square_teacher"]["minimum_rescued_roots"]
        )
        status = (
            "QUALIFIED"
            if selected["base_failed_repair_succeeded_roots"] >= required
            else "NEED_STRONGER_REPAIRER"
        )
    payload = {
        "schema_version": "hb1_repair_qualification_v1", "task": args.task,
        "status": status, "anchors": len(anchors), "independent_roots": len({row["root_id"] for row in anchors}),
        "selected_repair": selected["repair"], "selection": selected,
        "candidates": summaries,
    }
    write_table(all_rows, args.output_dir / "qualification_results.parquet")
    atomic_json_dump(payload, args.output_dir / "summary.json")
    print(json.dumps({"task": args.task, "status": status, "selection": selected}, indent=2))


if __name__ == "__main__":
    main()
