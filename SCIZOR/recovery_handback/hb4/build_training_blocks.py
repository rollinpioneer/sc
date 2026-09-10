"""Freeze HB4-C root matching and recurrent supervision blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file, write_table


SALT = "hb4_square_absorb_v1_data_selection"
BLOCK_LENGTH = 10
BLOCKS_PER_ROOT = 4
MAX_K = 64


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def _choose_starts(root_id: str, branch: str, candidates: list[int]) -> list[int]:
    ranked = sorted(candidates, key=lambda start: _rank(SALT, root_id, start))
    chosen = sorted(ranked[:BLOCKS_PER_ROOT])
    if len(chosen) != BLOCKS_PER_ROOT:
        raise ValueError(f"not enough aligned blocks for {root_id} {branch}")
    return chosen


def _cache_hash(path: Path) -> str:
    return sha256_file(path)


def _recovery_rows(source_table: Path, audit_path: Path, export_root: Path) -> tuple[list[dict], dict]:
    source_rows = {int(row["root_seed"]): row for row in _load_csv(source_table)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verified = set(map(int, audit["verified_root_seeds"]))
    eligible = [seed for seed in map(int, json.loads((export_root / "data/eligible_roots.json").read_text())["root_seeds"]) if seed in verified]
    ranked = sorted(eligible, key=lambda seed: _rank(SALT, "recovery_root", seed))
    return [source_rows[seed] for seed in ranked], {"verified_eligible": len(eligible), "verified_seeds": ranked}


def _load_csv(path: Path) -> list[dict]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _standard_candidates(records_path: Path) -> list[dict]:
    return [row for row in _load_jsonl(records_path) if bool(row.get("valid_for_matching"))]


def _int_list(value) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return [int(item) for item in value]


def build(export_root: Path, run_root: Path) -> dict:
    source_table = export_root / "data/source_root_table.csv"
    audit_path = export_root / "data/reconstruction_audit.json"
    standard_records = run_root / "data/standard_teacher/records.jsonl"
    if not source_table.is_file() or not audit_path.is_file() or not standard_records.is_file():
        raise FileNotFoundError("HB4-B outputs are not complete")
    recovery, recovery_info = _recovery_rows(source_table, audit_path, export_root)
    standard = sorted(_standard_candidates(standard_records), key=lambda row: _rank(SALT, "standard_root", row["seed"]))
    k = min(MAX_K, len(recovery), len(standard))
    if k < 30:
        raise RuntimeError(f"HOLD_HB4_STANDARD_CONTROL_DATA: K={k}, recovery={len(recovery)}, standard={len(standard)}")
    recovery = recovery[:k]
    standard = standard[:k]
    recovery_ids = {int(row["root_seed"]) for row in recovery}
    standard_ids = {int(row["seed"]) for row in standard}
    if recovery_ids & standard_ids:
        raise RuntimeError("source and standard seed ranges collided")

    rows: list[dict] = []
    arm_counts: dict[str, int] = {}
    per_root: dict[str, dict[str, int]] = {}
    reconstruction_lookup = {
        (int(row["seed"]), int(row["length"])): row
        for row in _load_jsonl_file(run_root / "observations")
    }

    def add_recovery(arm: str, branch: str, source: dict, length: int) -> None:
        seed = int(source["root_seed"])
        audit = reconstruction_lookup[(seed, length)]
        cache = Path(audit["cache_path"])
        if not audit["reconstruction_verified"] or not cache.is_file():
            raise RuntimeError(f"missing verified observation cache: {seed} L{length}")
        starts = _choose_starts(source["canonical_group_id"], branch, list(range(20, 20 + length - BLOCK_LENGTH + 1, BLOCK_LENGTH)))
        root_key = source["canonical_group_id"]
        for start in starts:
            rows.append({
                "schema_version": "hb4_training_block_v1", "arm": arm,
                "canonical_group_id": source["canonical_group_id"], "root_seed": seed,
                "trajectory_id": f"square:hb3p_stop_continue_formal:{seed}:{branch}",
                "source_branch": branch, "trajectory_path": source["l60_trajectory_path"] if length == 60 else source["l80_trajectory_path"],
                "block_start": start, "block_length": BLOCK_LENGTH,
                "event_ids": [f"{source['canonical_group_id']}:{branch}:{t}" for t in range(start, start + BLOCK_LENGTH)],
                "action_owners": ["TEACHER"] * BLOCK_LENGTH,
                "label_mask": [True] * BLOCK_LENGTH,
                "obs_artifact": str(cache.resolve()), "obs_index": start - 20,
                "obs_cache_hash": _cache_hash(cache), "teacher_sha256": "7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62",
                "base_sha256": "e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6",
                "image_origin": "RECONSTRUCTED_FOR_HB4", "reconstruction_verified": True,
                "source_protocol_hash": source["protocol_hash"], "selection_salt": SALT,
            })
        per_root.setdefault(root_key, {})[arm] = BLOCKS_PER_ROOT * BLOCK_LENGTH

    for source in recovery:
        add_recovery("HANDOFF_RECOVERY", "FIXED_L60", source, 60)
        add_recovery("FIXED_L80_RECOVERY", "FIXED_L80", source, 80)

    for record in standard:
        seed = int(record["seed"])
        root_id = record["root_id"]
        starts = _choose_starts(root_id, "STANDARD", _int_list(record["candidate_block_starts"]))
        cache = Path(record["hdf5"] if "hdf5" in record else run_root / "data/standard_teacher/ordinary_teacher.hdf5")
        if not cache.is_file():
            cache = run_root / "data/standard_teacher/ordinary_teacher.hdf5"
        for start in starts:
            rows.append({
                "schema_version": "hb4_training_block_v1", "arm": "MATCHED_STANDARD_DATA",
                "canonical_group_id": root_id, "root_seed": seed,
                "trajectory_id": f"{root_id}:{record['demo_id']}", "source_branch": "STANDARD",
                "source_demo_id": record["demo_id"],
                "block_start": start, "block_length": BLOCK_LENGTH,
                "event_ids": [f"{root_id}:STANDARD:{t}" for t in range(start, start + BLOCK_LENGTH)],
                "action_owners": ["TEACHER"] * BLOCK_LENGTH, "label_mask": [True] * BLOCK_LENGTH,
                "obs_artifact": str(cache.resolve()), "obs_index": start,
                "obs_cache_hash": _cache_hash(cache), "teacher_sha256": record["teacher_checkpoint_sha256"],
                "base_sha256": "e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6",
                "image_origin": "RECORDED", "reconstruction_verified": False,
                "source_protocol_hash": "HB4_STANDARD_TEACHER", "selection_salt": SALT,
            })
        per_root.setdefault(root_id, {})["MATCHED_STANDARD_DATA"] = BLOCKS_PER_ROOT * BLOCK_LENGTH

    for row in rows:
        arm_counts[row["arm"]] = arm_counts.get(row["arm"], 0) + int(row["block_length"])
    expected = k * BLOCKS_PER_ROOT * BLOCK_LENGTH
    if set(arm_counts) != {"HANDOFF_RECOVERY", "FIXED_L80_RECOVERY", "MATCHED_STANDARD_DATA"} or set(arm_counts.values()) != {expected}:
        raise RuntimeError(f"unbalanced block counts: {arm_counts}")
    per_root_expected = BLOCKS_PER_ROOT * BLOCK_LENGTH
    if any(set(values) != {"HANDOFF_RECOVERY", "FIXED_L80_RECOVERY"} or any(v != per_root_expected for v in values.values()) for key, values in per_root.items() if key.startswith("square:hb4_source:")):
        raise RuntimeError("recovery root block mismatch")
    output = export_root / "data/training_blocks_manifest.jsonl"
    write_table(rows, output)
    audit = {
        "schema_version": "hb4_matching_audit_v1", "status": "PASS_HB4_C_READY",
        "K": k, "N": expected, "recovery_candidate_count": len(recovery),
        "standard_candidate_count": len(standard), "recovery_root_seeds": sorted(recovery_ids),
        "standard_root_seeds": sorted(standard_ids), "arms": arm_counts,
        "blocks_per_root": BLOCKS_PER_ROOT, "block_length": BLOCK_LENGTH,
        "teacher_executed_actions_only": True, "post_handoff_base_actions_supervised": False,
        "selection_salt": SALT, "reconstruction": recovery_info,
        "source_protocol_hashes": sorted({row["source_protocol_hash"] for row in rows}),
        "data_manifest_sha256": sha256_file(output),
    }
    atomic_json_dump(audit, export_root / "data/matching_audit.json")
    atomic_json_dump({"schema_version": "hb4_data_manifest_v1", "K": k, "N": expected, "manifest": str(output.resolve()), "sha256": sha256_file(output), "arms": arm_counts}, export_root / "data/data_manifest.json")
    return {"K": k, "N": expected, "arms": arm_counts, "status": audit["status"]}


def _load_jsonl_file(observation_root: Path) -> list[dict]:
    if not observation_root.is_dir():
        return []
    rows = []
    for path in sorted(observation_root.glob("*/FIXED_L*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.export_root, args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
