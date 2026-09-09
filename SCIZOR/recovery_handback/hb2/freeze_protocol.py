"""Write the immutable F protocol manifest after validation selection."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True); parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); selection = json.loads(args.selection.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[3]
    try: commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    models = {}
    for model_id in ["M0_time", "M1_proprio", "M2_global", "M3_local", "M4_paired", "M5_single"]:
        checkpoint = args.models_root / model_id / "seed0" / "best.pt"
        if checkpoint.is_file(): models.setdefault(model_id, {})["seed0"] = {"checkpoint": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
    selected = selection["selected_visual_model"]
    for seed in config["hb2"]["training"]["selected_model_additional_seeds"]:
        checkpoint = args.models_root / selected / f"seed{seed}" / "best.pt"
        if checkpoint.is_file(): models.setdefault(selected, {})[f"seed{seed}"] = {"checkpoint": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
    payload = {
        "schema_version": "hb2_frozen_protocol_v1", "task": "square", "code_commit": commit,
        "semantic_pair_id": config["hb2"]["semantic_pair_id"], "source_ref": config.get("source_ref"),
        "selected_visual_model": selected, "canonical_seed": 0, "models": models,
        "selection": selection, "input_schema": str((Path(config["output_root"]) / "features/input_schema.json").resolve()),
        "normalizer": str((Path(config["output_root"]) / "features/normalizer.json").resolve()),
        "decision": config["hb2"]["decision"], "statistics": config["hb2"]["statistics"],
        "test_locked": True,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"selected_visual_model": selected, "checkpoint_models": list(models)}, indent=2))


if __name__ == "__main__": main()

