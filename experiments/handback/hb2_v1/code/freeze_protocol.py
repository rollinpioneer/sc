"""Write the immutable F protocol manifest after validation selection."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


MODEL_IDS = ("M0_time", "M1_proprio", "M2_global", "M3_local", "M4_paired", "M5_single")


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _validation_path(validation_dir: Path, model_id: str, seed_name: str) -> Path:
    suffix = "" if seed_name == "seed0" else f"_{seed_name}"
    return validation_dir / f"{model_id}{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    experiment_root = Path(config["output_root"])
    validation_dir = experiment_root / "metrics/validation"
    selected = str(selection["selected_visual_model"])
    required_seeds = {
        model_id: ([0, *config["hb2"]["training"]["selected_model_additional_seeds"]]
                   if model_id == selected else [0])
        for model_id in MODEL_IDS
    }
    models: dict[str, dict] = {}
    for model_id, seeds in required_seeds.items():
        for seed in seeds:
            seed_name = f"seed{int(seed)}"
            checkpoint = args.models_root / model_id / seed_name / "best.pt"
            validation_path = _validation_path(validation_dir, model_id, seed_name)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"required checkpoint missing: {checkpoint}")
            if not validation_path.is_file():
                raise FileNotFoundError(f"required validation artifact missing: {validation_path}")
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            metrics = validation["metrics"]
            models.setdefault(model_id, {})[seed_name] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "temperature": float(metrics["temperature"]),
                "validation_root_equal_nll_uncalibrated": metrics["root_equal_five_head_nll_uncalibrated"],
                "validation_root_equal_nll_calibrated": metrics["root_equal_five_head_nll_calibrated"],
                "parameters": metrics.get("parameters"),
                "history_length": metrics.get("history_length"),
                "validation_artifact": str(validation_path.resolve()),
                "validation_artifact_sha256": sha256_file(validation_path),
            }

    repo = Path(__file__).resolve().parents[3]
    input_schema = experiment_root / "features/input_schema.json"
    normalizer = experiment_root / "features/normalizer.json"
    feature_manifest = experiment_root / "features/anchor.manifest.json"
    split_manifest = experiment_root / "data/frozen/split_manifest.json"
    sufficiency = experiment_root / "data/frozen/data_sufficiency.json"
    policy_pair = experiment_root / "assets/selected_policy_pair_manifest.json"
    payload = {
        "schema_version": "hb2_frozen_protocol_v2",
        "task": "square",
        "code_commit": _git_head(repo),
        "source_ref": config.get("source_ref"),
        "semantic_pair_id": config["hb2"]["semantic_pair_id"],
        "selected_visual_model": selected,
        "canonical_seed": 0,
        "models": models,
        "selection": selection,
        "decision": config["hb2"]["decision"],
        "statistics": config["hb2"]["statistics"],
        "data_roles": config["roles"],
        "artifacts": {
            "config": _file_record(args.config),
            "selection": _file_record(args.selection),
            "input_schema": _file_record(input_schema),
            "normalizer": _file_record(normalizer),
            "feature_manifest": _file_record(feature_manifest),
            "split_manifest": _file_record(split_manifest),
            "data_sufficiency": _file_record(sufficiency),
            "policy_pair_manifest": _file_record(policy_pair),
        },
        "input_schema": str(input_schema.resolve()),
        "normalizer": str(normalizer.resolve()),
        "encoder": json.loads(feature_manifest.read_text(encoding="utf-8"))["encoder"],
        "test_role": "hb2_test",
        "test_seed_start": int(config["roles"]["hb2_test"]["seed_start"]),
        "test_locked": True,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({
        "selected_visual_model": selected,
        "canonical_seed": 0,
        "model_seeds": {key: sorted(value) for key, value in models.items()},
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
