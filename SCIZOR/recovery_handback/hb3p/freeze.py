"""Freeze the HB3-P protocol after development and pilot parity checks."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _code_manifest(package: Path) -> dict:
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".sh"}:
            files[str(path.relative_to(package))] = sha256_file(path)
    return {"files": files, "tree_sha256": sha256_json(files)}


def _prior_seed_rows(handback_root: Path) -> list[dict]:
    rows = []
    patterns = (
        "hb1_v1/roots/**/roots.jsonl",
        "hb1_repair_v1/roots/**/roots.jsonl",
        "hb2_v1/roots/**/roots.jsonl",
    )
    seen = set()
    for pattern in patterns:
        for path in sorted(handback_root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            rows.extend(read_table(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    if not pilot.get("all_checks_passed") or not pilot.get("all_engineering_ok"):
        raise RuntimeError("pilot parity did not pass; refusing to freeze")
    if int(pilot.get("pilot_roots", -1)) != 4:
        raise RuntimeError("pilot did not cover the four frozen roots")
    if int(pilot.get("repair_calls_after_handoff", -1)) != 0:
        raise RuntimeError("pilot called the repair policy after handback")
    repo = args.code_root.parent
    if _git(repo, "status", "--porcelain", "--", "SCIZOR/recovery_handback/hb3p"):
        raise RuntimeError("HB3-P code has uncommitted changes; commit before freezing")
    head = _git(repo, "rev-parse", "HEAD")
    package = args.code_root / "recovery_handback/hb3p"
    code = _code_manifest(package)
    seeds = list(range(int(draft["test_seed_start"]), int(draft["test_seed_start"]) + int(draft["new_test_roots"])))
    if seeds != list(range(500000, 500080)):
        raise RuntimeError(f"unexpected HB3-P test seed range: {seeds[:1]}..{seeds[-1:]}")
    handback_root = Path(config["output_root"]).parent
    prior = _prior_seed_rows(handback_root)
    prior_seeds = {int(row["seed"]) for row in prior if row.get("seed") is not None}
    overlap = sorted(prior_seeds.intersection(seeds))
    if overlap:
        raise RuntimeError(f"HB3-P preregistered seeds overlap prior HB1/HB2 manifests: {overlap}")
    seed_rows = [{
        "index": index,
        "task": "square",
        "role": "hb3p_test",
        "seed": seed,
        "root_id": f"square:hb3p_test:{seed}",
        "initial_state_hash": None,
        "status": "preregistered_before_collection",
    } for index, seed in enumerate(seeds)]
    seed_manifest = {
        "schema_version": "hb3p_test_seed_manifest_v1",
        "source_ref": draft["source_ref"],
        "code_commit": head,
        "count": len(seed_rows),
        "adaptive_expansion_allowed": False,
        "prior_manifest_seed_overlap": overlap,
        "rows": seed_rows,
    }
    seed_path = args.output.with_name("test_seed_manifest.json")
    atomic_json_dump(seed_manifest, seed_path)
    frozen = dict(draft)
    frozen.update({
        "schema_version": "hb3p_frozen_protocol_v1",
        "frozen": True,
        "code_commit": head,
        "code_manifest": code,
        "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        "draft": {"path": str(args.draft.resolve()), "sha256": sha256_file(args.draft)},
        "pilot": {"path": str(args.pilot.resolve()), "sha256": sha256_file(args.pilot)},
        "test_seed_manifest": {"path": str(seed_path.resolve()), "sha256": sha256_file(seed_path)},
        "test_seeds": seeds,
        "test_locked": True,
        "adaptive_test_expansion_allowed": False,
        "online_constraints": {
            "candidate_times_only": True,
            "max_takeovers": 1,
            "fixed_exit": True,
            "repair_calls_after_handoff_required": 0,
            "learned_exit_enabled": False,
            "repeated_takeover_tested": False,
            "arbitrary_query_times_tested": False,
        },
    })
    atomic_json_dump(frozen, args.output)
    digest = sha256_file(args.output)
    args.output.with_name("frozen.sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    experiment_root = Path(config["output_root"])
    atomic_json_dump({
        "completed": True,
        "frozen_protocol": str(args.output.resolve()),
        "sha256": digest,
        "code_commit": head,
    }, experiment_root / "status/hb3p-D.done")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": digest,
        "code_commit": head,
        "test_seeds": [seeds[0], seeds[-1]],
    }, indent=2))


if __name__ == "__main__":
    main()
