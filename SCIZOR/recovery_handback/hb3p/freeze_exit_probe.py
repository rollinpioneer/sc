"""Freeze the bounded HB3-P intermediate-exit probe before collection."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json
from recovery_handback.hb3p.io import mark


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _code_manifest(package: Path) -> dict:
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".sh"}:
            files[str(path.relative_to(package))] = sha256_file(path)
    return {"files": files, "tree_sha256": sha256_json(files)}


def _prior_seed_rows(handback_root: Path, current_root: Path) -> list[dict]:
    rows = []
    for path in sorted(handback_root.glob("*/roots/**/roots.jsonl")):
        if current_root in path.parents:
            continue
        rows.extend(read_table(path))
    return rows


def freeze(config_path: Path, draft_path: Path, code_root: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    repo = code_root.parent
    if _git(repo, "status", "--porcelain", "--", "SCIZOR/recovery_handback/hb3p"):
        raise RuntimeError("HB3-P probe code has uncommitted changes; commit before freezing")
    head = _git(repo, "rev-parse", "HEAD")
    seeds = [int(value) for value in draft["test_seeds"]]
    expected = list(range(600000, 600040))
    if seeds != expected:
        raise RuntimeError(f"unexpected probe seed range: {seeds[:1]}..{seeds[-1:]}")
    prior = _prior_seed_rows(output_root.parent, output_root)
    prior_seeds = {int(row["seed"]) for row in prior if row.get("seed") is not None}
    overlap = sorted(prior_seeds.intersection(seeds))
    if overlap:
        raise RuntimeError(f"probe seeds overlap an earlier root manifest: {overlap}")

    rows = [{
        "index": index,
        "task": "square",
        "role": "hb3p_exit_probe",
        "seed": seed,
        "root_id": f"square:hb3p_exit_probe:{seed}",
        "initial_state_hash": None,
        "status": "preregistered_before_collection",
    } for index, seed in enumerate(seeds)]
    seed_manifest = {
        "schema_version": "hb3p_exit_probe_seed_manifest_v1",
        "source_ref": draft["source_ref"],
        "code_commit": head,
        "count": len(rows),
        "adaptive_expansion_allowed": False,
        "prior_manifest_seed_overlap": overlap,
        "rows": rows,
    }
    seed_path = output_root / "config/test_seed_manifest.json"
    atomic_json_dump(seed_manifest, seed_path)
    frozen = dict(draft)
    frozen.update({
        "schema_version": "hb3p_exit_probe_protocol_v1",
        "frozen": True,
        "test_locked": True,
        "code_commit": head,
        "code_manifest": _code_manifest(code_root / "recovery_handback/hb3p"),
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "draft": {"path": str(draft_path.resolve()), "sha256": sha256_file(draft_path)},
        "test_seed_manifest": {"path": str(seed_path.resolve()), "sha256": sha256_file(seed_path)},
        "test_seeds": seeds,
        "test_locked": True,
        "adaptive_test_expansion_allowed": False,
        "probe_limits": {
            "roots": 40,
            "methods": ["NONE", "FIXED_L40", "FIXED_L60", "FIXED_L80"],
            "maximum_complete_method_records": 160,
            "maximum_environment_steps": 64000,
        },
    })
    protocol_path = output_root / "config/frozen_protocol.json"
    atomic_json_dump(frozen, protocol_path)
    digest = sha256_file(protocol_path)
    (output_root / "config/frozen.sha256").write_text(
        f"{digest}  {protocol_path.name}\n", encoding="utf-8"
    )
    config["probe_protocol_sha256"] = digest
    atomic_json_dump(config, config_path)
    mark(output_root, "exit-probe-frozen.done", f"code={head} seeds=600000..600039")
    return {"protocol": str(protocol_path.resolve()), "sha256": digest, "code_commit": head}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        config_path=args.config,
        draft_path=args.draft,
        code_root=args.code_root,
        output_root=args.output_root,
    ), indent=2))


if __name__ == "__main__":
    main()
