"""Project raw SCIZOR chunk deficits to transition scores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["task", "demo_id", "t"]
CHUNK_KEYS = ["task", "demo_id", "start_t"]


def _project(chunks, labels, gamma_per_second, method):
    output = []
    for (task, demo_id), label_part in labels.groupby(["task", "demo_id"], sort=False):
        chunk_part = chunks[(chunks["task"] == task) & (chunks["demo_id"] == demo_id)]
        length = len(label_part)
        numerator, denominator, coverage = np.zeros(length), np.zeros(length), np.zeros(length, dtype=np.int32)
        for chunk in chunk_part.itertuples(index=False):
            start, end = int(chunk.start_t), min(int(chunk.end_t), length)
            indices = np.arange(start, max(start + 1, end), dtype=np.int64)
            if method == "uniform":
                weights = np.full(len(indices), 1.0 / len(indices))
            else:
                gamma_step = gamma_per_second ** (1.0 / float(chunk.control_freq))
                weights = gamma_step ** (indices[-1] - indices)
                weights = weights / weights.sum()
            numerator[indices] += float(chunk.V_c) * weights
            denominator[indices] += weights
            coverage[indices] += 1
        part = label_part.sort_values("t")[["task", "demo_id", "pair_id", "t"]].copy()
        part["score"] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        part["num_covering_chunks"] = coverage
        part["method"] = method
        output.append(part)
    return pd.concat(output, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-evidence", required=True)
    parser.add_argument("--transition-labels", required=True)
    parser.add_argument("--gamma-per-second", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    chunks, labels = pd.read_parquet(args.chunk_evidence), pd.read_parquet(args.transition_labels)
    labels = labels.sort_values(KEYS).reset_index(drop=True)
    if chunks.duplicated(CHUNK_KEYS).any():
        raise RuntimeError("chunk evidence has duplicate trajectory start keys")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, filename in (("uniform", "uniform_scores.parquet"), ("future_discount", "future_discount_scores.parquet")):
        scores = _project(chunks, labels, args.gamma_per_second, method)
        if len(scores) != len(labels) or scores.duplicated(KEYS).any():
            raise RuntimeError(f"{method} projection does not cover transition labels exactly")
        scores.to_parquet(output_dir / filename, index=False)
        print(f"wrote {len(scores)} {method} transition scores")


if __name__ == "__main__":
    main()
