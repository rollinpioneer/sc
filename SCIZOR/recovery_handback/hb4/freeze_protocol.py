"""Create reproducible HB4 training and test protocol manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


ARMS = ("REPLAY_ONLY", "MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY", "HANDOFF_RECOVERY")
SEEDS = (0, 1, 2)
BASE = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(export_root: Path, run_root: Path, roots_path: Path | None = None, test: bool = False, output_name: str | None = None) -> dict:
    checkpoints = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = run_root / "training" / arm / f"seed{seed}" / "model_step_4000.pth"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints[f"{arm}:seed{seed}"] = {"path": str(path.resolve()), "sha256": sha256_file(path), "step": 4000}
    data_manifest = export_root / "data/training_hdf5_manifest.json"
    blocks_manifest = export_root / "data/training_blocks_manifest.jsonl"
    input_manifest = export_root / "assets/resolved_inputs_public.json"
    for path in (data_manifest, blocks_manifest, input_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema_version": "hb4_test_frozen_protocol_v1" if test else "hb4_training_protocol_v1",
        "experiment_id": "hb4_square_absorb_v1",
        "frozen_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": _json(export_root / "config/hb4_resolved.json").get("source_commit"),
        "scope": {"task": "square", "horizon_steps": 400, "stable_success_steps": 5, "no_help": True, "secondary_assisted_eval_enabled": False},
        "base_checkpoint": {"path": str(BASE.resolve()), "sha256": sha256_file(BASE)},
        "training": {"arms": list(ARMS), "seeds": list(SEEDS), "updates": 4000, "batch_sequences": 32, "sequence_length": 10, "learning_rate": 3e-5, "checkpoint_step": 4000, "checkpoints": checkpoints},
        "data_manifest": {"path": str(data_manifest.resolve()), "sha256": sha256_file(data_manifest)},
        "training_blocks_manifest": {"path": str(blocks_manifest.resolve()), "sha256": sha256_file(blocks_manifest)},
        "input_manifest": {"path": str(input_manifest.resolve()), "sha256": sha256_file(input_manifest)},
        "statistics": {"unit": "canonical_group_id", "bootstrap_repeats": 5000, "bootstrap_seed": 20260910, "interval": "two_sided_95_percentile_paired_root"},
    }
    if roots_path is not None:
        payload["evaluation_roots"] = {"path": str(roots_path.resolve()), "sha256": sha256_file(roots_path), "count": sum(1 for line in roots_path.read_text().splitlines() if line.strip())}
    name = output_name or ("test_frozen_protocol.json" if test else "training_protocol.json")
    output = export_root / "config" / name
    atomic_json_dump(payload, output)
    atomic_json_dump({"protocol": name, "sha256": sha256_file(output)}, export_root / "config" / f"{name}.sha256.json")
    if not test:
        (export_root / "config" / "training_frozen.sha256").write_text(
            f"{sha256_file(output)}  {name}\n",
            encoding="ascii",
        )
    return {"path": str(output.resolve()), "sha256": sha256_file(output), "checkpoints": len(checkpoints), "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--roots", type=Path)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    print(json.dumps(freeze(args.export_root, args.run_root, args.roots, args.test, args.output_name), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
