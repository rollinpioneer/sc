"""Build direct-action repair metadata from Robomimic teacher checkpoints."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


EPOCH_PATTERN = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)


def discover(
    runs: Path,
    epochs: list[int],
    *,
    algorithm: str = "robomimic_bc_gmm_direct",
    uses_privileged_input: bool = True,
) -> list[dict]:
    targets = set(epochs)
    matches: dict[int, list[Path]] = {}
    for checkpoint in runs.rglob("*.pth"):
        match = EPOCH_PATTERN.search(checkpoint.stem)
        if match and int(match.group(1)) in targets:
            matches.setdefault(int(match.group(1)), []).append(checkpoint)
    missing = sorted(targets - set(matches))
    if missing:
        raise RuntimeError(f"missing checkpoints for epochs {missing} under {runs}")
    candidates = []
    for epoch in sorted(targets):
        checkpoint = max(matches[epoch], key=lambda path: (path.stat().st_mtime_ns, str(path)))
        candidates.append({
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "algorithm": algorithm,
            "action_mode": "direct",
            "uses_privileged_input": uses_privileged_input,
            "training_steps": epoch,
            "training_unit": "epoch",
            "normalizer": None,
            "observation_shape": None,
        })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--algorithm", default="robomimic_bc_gmm_direct")
    parser.add_argument("--non-privileged", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "hb1_direct_repair_candidates_v1",
        "candidates": discover(
            args.runs,
            args.epochs,
            algorithm=args.algorithm,
            uses_privileged_input=not args.non_privileged,
        ),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
