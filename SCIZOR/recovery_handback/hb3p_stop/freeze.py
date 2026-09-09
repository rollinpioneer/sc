"""Freeze the learned stop/continue protocol before test-root collection."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _code_manifest(code_root: Path) -> dict:
    files = {}
    for path in sorted(Path(code_root).iterdir()):
        if path.is_file() and path.suffix in {".py", ".sh", ".json"}:
            files[path.name] = sha256_file(path)
    return {"files": files, "tree_sha256": sha256_json(files)}


def _prior_seeds(handback_root: Path, current_root: Path) -> set[int]:
    seeds = set()
    for path in sorted(Path(handback_root).glob("*/roots/**/roots.jsonl")):
        if current_root in path.parents:
            continue
        for row in read_table(path):
            if row.get("seed") is not None:
                seeds.add(int(row["seed"]))
    return seeds


def freeze(config_path: Path, draft_path: Path, training_summary_path: Path, code_root: Path, output: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    draft = json.loads(Path(draft_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    model = Path(training["checkpoint"])
    normalizer = Path(training["normalizer"])
    if not model.is_file() or not normalizer.is_file():
        raise FileNotFoundError("trained stop/continue model or normalizer is missing")
    repo = Path(code_root).parents[2]
    if _git(repo, "status", "--porcelain", "--", "SCIZOR/recovery_handback/hb3p_stop"):
        raise RuntimeError("stop/continue code must be committed before protocol freeze")
    head = _git(repo, "rev-parse", "HEAD")
    seeds = list(range(int(draft["test_seed_start"]), int(draft["test_seed_start"]) + int(draft["new_test_roots"])))
    experiment_root = Path(config["output_root"])
    overlap = sorted(_prior_seeds(experiment_root.parent, experiment_root).intersection(seeds))
    if overlap:
        raise RuntimeError(f"stop/continue test seeds overlap prior handback roots: {overlap}")
    frozen = dict(draft)
    frozen.update({
        "schema_version": "hb3p_stop_continue_protocol_v1",
        "frozen": True,
        "test_locked": True,
        "code_commit": head,
        "code_manifest": _code_manifest(code_root),
        "config": {"path": str(Path(config_path).resolve()), "sha256": sha256_file(Path(config_path))},
        "draft": {"path": str(Path(draft_path).resolve()), "sha256": sha256_file(Path(draft_path))},
        "training_summary": {"path": str(Path(training_summary_path).resolve()), "sha256": sha256_file(Path(training_summary_path))},
        "model": {
            "checkpoint": str(model.resolve()),
            "checkpoint_sha256": sha256_file(model),
            "normalizer": str(normalizer.resolve()),
            "normalizer_sha256": sha256_file(normalizer),
            "threshold": float(training["threshold_selection"]["threshold"]),
            "input_dim": 65,
            "architecture": [64, 32],
            "label_source": "paired realized utility of FIXED_L60 vs FIXED_L80",
        },
        "test_seeds": seeds,
        "prior_manifest_seed_overlap": overlap,
        "online_constraints": {
            "fixed_entry_t": 20,
            "model_query_only_at_t": 80,
            "max_takeovers": 1,
            "stop_handoff_t": 80,
            "continue_handoff_t": 100,
            "repair_interval_stop": [20, 80],
            "repair_interval_continue": [20, 100],
            "repair_calls_after_handoff_required": 0,
            "base_calls_once_per_executed_step": True,
        },
        "pilot": True,
        "formal_claim_allowed": False,
        "formal_claim_blocker": "40 test roots is under the estimated ~354 roots for 0.05 precision and therefore this is a pilot",
    })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(frozen, output)
    digest = sha256_file(output)
    output.with_name("frozen.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    config["frozen_protocol_sha256"] = digest
    atomic_json_dump(config, Path(config_path))
    return {"protocol": str(output.resolve()), "sha256": digest, "code_commit": head, "threshold": frozen["model"]["threshold"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(
        config_path=args.config,
        draft_path=args.draft,
        training_summary_path=args.training_summary,
        code_root=args.code_root,
        output=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
