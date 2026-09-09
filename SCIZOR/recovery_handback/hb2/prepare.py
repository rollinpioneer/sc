"""Resolve frozen HB1-R inputs and construct the HB2 experiment configuration."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from recovery_handback.common import atomic_json_dump, read_table, sha256_file, sha256_json

BASE_SHA = "e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6"
REPAIR_SHA = "7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62"


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _require(path: Path, label: str) -> Path:
    if not path.is_file() or path.name.endswith(".placeholder.md"):
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.resolve()


def _resolve_checkpoint(record: dict, expected_sha: str, roots: list[Path], label: str) -> Path:
    candidates = []
    raw = record.get("checkpoint")
    if raw:
        candidates.append(Path(raw))
    for root in roots:
        checkpoint_dir = root / "checkpoints"
        if checkpoint_dir.is_dir():
            candidates.extend(sorted(checkpoint_dir.glob("*.pth")))
    checked = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in checked or not candidate.is_file():
            continue
        checked.add(candidate)
        if sha256_file(candidate) == expected_sha:
            return candidate.resolve()
    raise FileNotFoundError(f"missing {label} checkpoint with SHA-256 {expected_sha}")


def _schema(rows: list[dict]) -> dict:
    return {"rows": len(rows), "fields": sorted({key for row in rows for key in row})}


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def prepare(source_root: Path, runtime_root: Path, output_root: Path, task: str) -> None:
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    output_root = output_root.resolve()
    if task != "square":
        raise ValueError("HB2-v1 accepts only square")
    for name in ("config", "assets", "data", "roots", "branches", "features", "models",
                 "predictions", "metrics", "report", "package", "logs", "status", "bin"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    decision_path = _require(source_root / "metrics/hb1_repair_decision.json", "HB1-R decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    square = decision.get("tasks", {}).get("square", {})
    if square.get("formal_probe", {}).get("status") != "READY_HB2":
        raise ValueError("HB1-R decision does not authorize formal Square HB2 data reuse")
    source_config_path = _require(source_root / "config/hb1_repair.json", "HB1-R config")
    source_runtime_path = _require(source_root / "config/runtime.txt", "HB1-R runtime record")
    source_pair_path = _require(source_root / "assets/policy_pair_square.json", "Square policy pair")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_pair = json.loads(source_pair_path.read_text(encoding="utf-8"))
    assets_candidates = [runtime_root / "assets/assets.json", source_root / "assets/assets.runtime.json",
                         source_root / "assets/assets.json"]
    source_assets_path = next((path for path in assets_candidates if path.is_file()), None)
    if source_assets_path is None:
        raise FileNotFoundError("missing HB1-R assets.json")
    source_assets = json.loads(source_assets_path.read_text(encoding="utf-8"))

    base_checkpoint = _resolve_checkpoint(source_pair["base"], BASE_SHA, [runtime_root, source_root], "base")
    repair_checkpoint = _resolve_checkpoint(source_pair["repair"], REPAIR_SHA, [runtime_root, source_root], "repair")
    pair = json.loads(json.dumps(source_pair))
    pair["base"]["checkpoint"] = str(base_checkpoint)
    pair["base"]["checkpoint_sha256"] = BASE_SHA
    pair["repair"]["checkpoint"] = str(repair_checkpoint)
    pair["repair"]["checkpoint_sha256"] = REPAIR_SHA

    roots_dirs = [runtime_root / "roots/square/probe", source_root / "roots/square/probe"]
    roots_dir = next((path for path in roots_dirs if (path / "roots.parquet").is_file()
                      and (path / "anchors.parquet").is_file()), None)
    branches_dirs = [runtime_root / "branches/square", source_root / "branches/square"]
    branches_dir = next((path for path in branches_dirs if (path / "branch_results.parquet").is_file()), None)
    if roots_dir is None:
        raise FileNotFoundError("missing Square probe roots.parquet/anchors.parquet")
    if branches_dir is None:
        raise FileNotFoundError("missing Square probe branch_results.parquet")
    roots_path = _require(roots_dir / "roots.parquet", "legacy roots")
    anchors_path = _require(roots_dir / "anchors.parquet", "legacy anchors")
    branch_results_path = _require(branches_dir / "branch_results.parquet", "legacy branch results")
    if not (branches_dir / "records").is_dir():
        raise FileNotFoundError(f"missing branch records directory: {branches_dir / 'records'}")

    repo = Path(__file__).resolve().parents[3]
    adapter_files = [
        repo / "SCIZOR/recovery_handback/adapters/base_policy.py",
        repo / "SCIZOR/recovery_handback/adapters/env_adapter.py",
        repo / "SCIZOR/recovery_handback/adapters/repair_policy.py",
        repo / "SCIZOR/recovery_handback/execution/paired_branch.py",
        repo / "SCIZOR/recovery_handback/execution/label_reference.py",
    ]
    adapter_hashes = {str(path.relative_to(repo)): sha256_file(path) for path in adapter_files}
    original_pair_hash = sha256_json(source_pair)
    semantic_payload = {
        "base_checkpoint_sha256": BASE_SHA,
        "repair_checkpoint_sha256": REPAIR_SHA,
        "base_algorithm": pair["base"].get("algorithm"),
        "repair_algorithm": pair["repair"].get("algorithm"),
        "base_observation_protocol": pair["base"].get("observation_protocol"),
        "action_dim": pair["base"].get("action_dim"),
        "policy_history_protocol": source_config.get("policy_history_protocol"),
        "random_tape": source_config.get("random_tape"),
        "minimum_autonomous_steps": source_config.get("minimum_autonomous_steps"),
        "success_consecutive_steps": source_config.get("success_consecutive_steps"),
        "adapter_source_sha256": adapter_hashes,
    }
    semantic_pair_id = sha256_json(semantic_payload)
    pair["semantic_pair_id"] = semantic_pair_id
    pair["source_policy_pair_hash"] = original_pair_hash

    pair_output = output_root / "assets/policy_pair_square.json"
    assets_output = output_root / "assets/assets.json"
    atomic_json_dump(pair, pair_output)
    atomic_json_dump(source_assets, assets_output)
    shutil.copy2(source_config_path, output_root / "config/hb1_repair_source.json")
    shutil.copy2(source_runtime_path, output_root / "config/hb1_runtime_source.txt")
    shutil.copy2(source_pair_path, output_root / "assets/policy_pair_square.source.json")
    shutil.copy2(decision_path, output_root / "assets/hb1_repair_decision.source.json")

    template_path = Path(__file__).with_name("hb2_template.json")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    config = _deep_merge(source_config, template)
    config["assets_file"] = str(assets_output)
    config["output_root"] = str(output_root)
    config["hb2"]["selected_policy_pair_path"] = str(pair_output)
    config["hb2"]["semantic_pair_id"] = semantic_pair_id
    config["hb2"]["source_policy_pair_hash"] = original_pair_hash
    config["hb2"]["source_root"] = str(source_root)
    config["hb2"]["runtime_root"] = str(runtime_root)
    atomic_json_dump(config, output_root / "config/hb2.json")
    head = _git_head(repo)
    (output_root / "config/source_commit.txt").write_text(head + "\n", encoding="utf-8")

    root_rows = read_table(roots_path)
    anchor_rows = read_table(anchors_path)
    branch_rows = read_table(branch_results_path)
    inventory = {
        "schema_version": "hb2_input_inventory_v1",
        "task": task,
        "source_commit": head,
        "semantic_pair_id": semantic_pair_id,
        "tables": {
            "roots": {"path": str(roots_path), **_schema(root_rows)},
            "anchors": {"path": str(anchors_path), **_schema(anchor_rows)},
            "branches": {"path": str(branch_results_path), **_schema(branch_rows)},
        },
        "anchors": [{key: row.get(key) for key in (
            "anchor_id", "root_id", "stat_group_id", "anchor_t", "initial_state_hash",
            "rollout_path", "anchor_history_path")}
            for row in anchor_rows],
        "handoff_history_records": sum(
            bool(row.get("handoff_history_path")) and Path(str(row["handoff_history_path"])).is_file()
            for row in branch_rows
        ),
    }
    atomic_json_dump(inventory, output_root / "data/input_inventory.json")
    resolved = {
        "schema_version": "hb2_resolved_inputs_v1",
        "task": task,
        "semantic_pair_id": semantic_pair_id,
        "source_policy_pair_hash": original_pair_hash,
        "resolution_scope": [str(source_root), str(runtime_root), str(runtime_root / "checkpoints")],
        "inputs": {
            "decision": {"source_path": str(decision_path), "resolved_path": str(decision_path), "basis": "source_root fixed path"},
            "roots": {"source_path": str(roots_path), "resolved_path": str(roots_path), "basis": "complete runtime Square probe"},
            "anchors": {"source_path": str(anchors_path), "resolved_path": str(anchors_path), "basis": "complete runtime Square probe"},
            "branch_results": {"source_path": str(branch_results_path), "resolved_path": str(branch_results_path), "basis": "complete runtime Square probe"},
            "branch_records": {"source_path": str(branches_dir / "records"), "resolved_path": str(branches_dir), "basis": "branch JSON and handoff histories"},
            "base_checkpoint": {"source_path": source_pair["base"].get("checkpoint"), "resolved_path": str(base_checkpoint), "sha256": BASE_SHA, "basis": "fixed SHA-256"},
            "repair_checkpoint": {"source_path": source_pair["repair"].get("checkpoint"), "resolved_path": str(repair_checkpoint), "sha256": REPAIR_SHA, "basis": "fixed SHA-256"},
        },
    }
    atomic_json_dump(resolved, output_root / "assets/resolved_inputs.json")
    atomic_json_dump({
        "schema_version": "hb2_selected_policy_pair_manifest_v1",
        "task": task,
        "semantic_pair_id": semantic_pair_id,
        "source_policy_pair_hash": original_pair_hash,
        "resolved_policy_pair_hash": sha256_json(pair),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": BASE_SHA,
        "repair_checkpoint": str(repair_checkpoint),
        "repair_checkpoint_sha256": REPAIR_SHA,
        "semantic_components": semantic_payload,
    }, output_root / "assets/selected_policy_pair_manifest.json")
    print(json.dumps({"output_root": str(output_root), "source_commit": head,
                      "semantic_pair_id": semantic_pair_id, "legacy_anchors": len(anchor_rows),
                      "legacy_branches": len(branch_rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("square",), required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.runtime_root, args.output_root, args.task)


if __name__ == "__main__":
    main()

