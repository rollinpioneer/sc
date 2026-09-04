import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import DataLoader

from curation.video_encoding.suboptimal_hdf5 import Evaluator, PreprocessDataset


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file(path):
    location = Path(path)
    return location if location.suffix == ".pth" else sorted(location.glob("*.pth"))[-1]


def attr(demo, key):
    value = demo.attrs.get(key, "")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def task(path, demo):
    return attr(demo, "task") or ("square" if "square" in str(path).lower() else "can" if "can" in str(path).lower() else path.stem)


def expected_rank(rank_thresholds, goal_seconds, max_rank):
    thresholds = np.asarray(rank_thresholds, dtype=float)
    matching = np.flatnonzero((goal_seconds >= thresholds[:, 0]) & (goal_seconds < thresholds[:, 1]))
    if not len(matching):
        matching = np.array([0 if goal_seconds < thresholds[0, 0] else len(thresholds) - 1])
    index = int(matching[0])
    lower, upper = thresholds[index]
    value = (goal_seconds - lower) / (upper - lower) + index - 0.5
    return float(np.clip(value, 0, max_rank))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--goal-time", type=float, default=2.0)
    parser.add_argument("--image-key", default="agentview_image")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--demo-start", type=int, default=0)
    parser.add_argument("--demo-end", type=int)
    args = parser.parse_args()
    evaluator = Evaluator(args.model_path)
    freq = int(evaluator.hdf5_config["freq"])
    horizon = int(round(args.goal_time * freq))
    image_size = int(evaluator.hdf5_config["obs_keys"][args.image_key])
    checkpoint = model_file(args.model_path)
    checkpoint_digest = sha256(checkpoint)
    rows = []
    for path in sorted(Path(args.data_dir).rglob("*.hdf5")):
        with h5py.File(path, "r") as file:
            demos = natsorted(file["data"].keys())
            if args.max_demos is not None:
                demos = demos[:args.max_demos]
            demos = demos[int(args.demo_start):int(args.demo_end) if args.demo_end is not None else None]
            for demo_id in demos:
                demo = file[f"data/{demo_id}"]
                images = np.asarray(demo[f"obs/{args.image_key}"])
                dataset = PreprocessDataset(images, np.asarray([horizon]), image_size)
                probs = np.concatenate([evaluator.inference.get_score(batch) for batch in DataLoader(dataset, batch_size=args.batch_size)])
                ranks = np.sum(probs * np.arange(probs.shape[-1]), axis=-1).reshape(-1)
                for t, pred_rank in enumerate(ranks):
                    pred_rank = float(pred_rank)
                    end_t = min(t + horizon, len(images) - 1)
                    steps = max(1, end_t - t)
                    goal_seconds = steps / freq
                    expected = expected_rank(evaluator.config["discriminator"]["rank_thres"], goal_seconds, probs.shape[-1] - 1)
                    deficit = float(np.clip((expected - pred_rank) / steps, 0, 1))
                    rows.append({"dataset_file": str(path.resolve()), "task": task(path, demo), "demo_id": demo_id,
                        "pair_id": attr(demo, "pair_id"), "variant": attr(demo, "variant"), "base_demo_id": attr(demo, "base_demo_id"),
                        "chunk_id": f"{demo_id}:{t}", "start_t": t, "end_t": end_t, "horizon_steps": steps,
                        "control_freq": freq, "goal_time": args.goal_time, "pred_rank": float(pred_rank),
                        "expected_rank": expected, "V_c": deficit, "checkpoint_sha256": checkpoint_digest})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)} chunks to {output}")


if __name__ == "__main__":
    main()
