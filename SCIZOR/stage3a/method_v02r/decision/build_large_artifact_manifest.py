"""Record non-lightweight Stage 3 artifacts without copying them into the ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LARGE_SUFFIXES = {".pt", ".pth", ".hdf5", ".h5", ".npz", ".parquet", ".faiss", ".mp4", ".jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(path: Path) -> str:
    parts = set(path.parts)
    if "features" in parts:
        return "transition feature cache or verifier normalizer"
    if "labels" in parts:
        return "transition label table"
    if "evidence" in parts:
        return "frozen SCIZOR evidence or scoring adapter"
    if "oracle" in parts:
        return "counterfactual oracle output"
    if "predictions" in parts:
        return "verifier prediction output"
    if "runs" in parts:
        return "training checkpoint or run artifact"
    if "blind_test" in parts:
        return "blind benchmark artifact"
    return "large Stage 3 artifact"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"missing artifact root: {root}")
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in LARGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        artifacts.append(
            {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(relative),
            }
        )
    payload = {
        "schema": "stage3_v02r_large_artifact_manifest_v1",
        "root": str(root),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_count": len(artifacts), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
