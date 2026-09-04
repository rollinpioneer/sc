"""Render a bounded, class-balanced set of benchmark label-review videos."""

import argparse
import json
from pathlib import Path

import h5py

from .freeze_stage1c import _render_pair_video


REVIEW_TYPES = ("direct_failure", "delayed_failure", "recovery_success", "recovery_failure", "no_effect")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.metadata).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for failure_type in REVIEW_TYPES:
            candidates = [row for row in records if row["failure_type"] == failure_type and row.get("label_status") == "ok"]
            for row in sorted(candidates, key=lambda item: item["pair_id"])[:args.per_class]:
                filename = f"{failure_type}__{row['pair_id'].replace(':', '_').replace('.', '_')}.mp4"
                _render_pair_video(h5, row, output_dir / filename)
                selected.append({"failure_type": failure_type, "pair_id": row["pair_id"], "video": filename})
    (output_dir / "index.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video_count": len(selected), "counts": {kind: sum(row["failure_type"] == kind for row in selected) for kind in REVIEW_TYPES}}, indent=2))


if __name__ == "__main__":
    main()
