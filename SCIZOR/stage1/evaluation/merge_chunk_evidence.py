"""Merge disjoint chunk-evidence shards and verify frozen-label coverage."""

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["task", "demo_id", "start_t"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_parquet(path) for path in args.shards], ignore_index=True)
    labels = pd.read_parquet(args.labels)
    expected = labels[["task", "demo_id", "t"]].rename(columns={"t": "start_t"})
    if frame.duplicated(KEYS).any():
        raise RuntimeError("duplicate chunk-evidence keys across shards")
    check = expected.merge(frame[KEYS], on=KEYS, how="outer", indicator=True)
    if (check["_merge"] != "both").any():
        raise RuntimeError(f"chunk-evidence coverage mismatch: {check['_merge'].value_counts().to_dict()}")
    frame.sort_values(KEYS, inplace=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"wrote {len(frame)} chunks from {len(args.shards)} shards to {args.output}")


if __name__ == "__main__":
    main()
