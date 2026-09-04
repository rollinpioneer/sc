"""Freeze the small, auditable input bundle for any Stage 4 hand-off."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(root: Path, path: Path | None, role: str) -> dict:
    resolved = path.resolve() if path is not None else None
    return {"path": str(resolved) if resolved else None, "sha256": sha256(resolved) if resolved else None, "role": role}


def maybe(root: Path, relative: str, role: str) -> dict:
    return artifact(root, root / relative, role)


def git_commit(scizor_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(scizor_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--scizor-root", type=Path, required=True)
    parser.add_argument("--action-library", type=Path)
    parser.add_argument("--result-definition", type=Path)
    parser.add_argument("--full-proposer-checkpoint", type=Path)
    parser.add_argument("--action-proposer-checkpoint", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.method_root.resolve()
    library_arg = args.action_library or (Path(os.environ["ACTION_LIBRARY_V02"]) if os.environ.get("ACTION_LIBRARY_V02") else None)
    library = library_arg.resolve() if library_arg else None
    decision = json.loads((root / "metrics/stage3_final_decision.json").read_text(encoding="utf-8")) if (root / "metrics/stage3_final_decision.json").exists() else {}
    protocol = root / "metrics/validation_frozen_protocol.json"
    confirmation = root / "config/v02r_confirmation_decision.json"
    result_definition = args.result_definition or (Path(os.environ["SCORE_SPEC_V02R"]) if os.environ.get("SCORE_SPEC_V02R") else root / "config/oracle_score_spec_v02r.json")
    blind_freeze = root / "config/blind_test_freeze.sha256"
    blind_benchmark = root / "blind_test/benchmark_v0.2_final_test.hdf5"

    checkpoints = {}
    for seed in (0, 1, 2):
        checkpoints[f"full_seed_{seed}"] = maybe(root, f"runs/full_seed_{seed}/best.pt", "full verifier checkpoint")
    selected = protocol
    selected_source = None
    if protocol.exists():
        selected_source = json.loads(protocol.read_text(encoding="utf-8")).get("selected_proposer")
    proposer_paths = {
        "full_proposer": args.full_proposer_checkpoint,
        "action_proposer": args.action_proposer_checkpoint,
    }
    if selected_source == "full_top5":
        selected_proposer = proposer_paths["full_proposer"]
    elif selected_source == "action_top5":
        selected_proposer = proposer_paths["action_proposer"]
    else:
        selected_proposer = None

    payload = {
        "schema": "stage3_v02r_method_bundle_v1",
        "code_commit": args.code_commit or git_commit(args.scizor_root),
        "final_decision": decision.get("decision"),
        "selected_proposer": selected_source,
        "selected_threshold": (json.loads(protocol.read_text(encoding="utf-8")).get("selected_threshold") if protocol.exists() else None),
        "verifier_checkpoints": checkpoints,
        "selected_proposer_checkpoint": artifact(root, selected_proposer, "selected frozen proposer checkpoint"),
        "action_library": {
            "index": artifact(root, library / "action_library_index.parquet" if library else None, "frozen action library index"),
            "support_thresholds": artifact(root, library / "support_thresholds.json" if library else None, "frozen action support thresholds"),
        },
        "feature_normalizer": maybe(root, "features/verifier_normalizer.npz", "train-only verifier normalizer"),
        "proposal_calibration": maybe(root, "proposals/proposal_score_calibration_v02.json", "frozen proposer calibration"),
        "validation_protocol": artifact(root, protocol, "frozen validation protocol"),
        "confirmation_decision": artifact(root, confirmation, "v0.2-R confirmation decision"),
        "long_result_definition": artifact(root, result_definition, "long-horizon oracle score definition"),
        "blind_benchmark": artifact(root, blind_benchmark, "frozen final blind benchmark"),
        "blind_benchmark_freeze": artifact(root, blind_freeze, "blind benchmark freeze checksum"),
        "large_artifact_manifest": maybe(root, "report/large_artifact_manifest.json", "large artifact manifest"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_proposer": selected_source, "decision": decision.get("decision")}, indent=2))


if __name__ == "__main__":
    main()
