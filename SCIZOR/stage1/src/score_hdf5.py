import argparse
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from torch.utils.data import DataLoader
from tqdm import tqdm

from curation.video_encoding.suboptimal_hdf5 import Evaluator, PreprocessDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--goal-time", nargs="+", type=float, default=[2.0])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-key", default="agentview_image")
    args = parser.parse_args()
    evaluator = Evaluator(args.model_path)
    evaluator.goal_time = list(args.goal_time)
    freq = int(evaluator.hdf5_config["freq"])
    goal_dist = np.asarray([int(round(time * freq)) for time in args.goal_time], dtype=np.int64)
    image_size = int(evaluator.hdf5_config["obs_keys"][args.image_key])
    files = sorted(Path(args.data_dir).rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no hdf5 under {args.data_dir}")
    for path in files:
        print(f"scoring {path}")
        with h5py.File(path, "a") as file:
            for demo_key in tqdm(natsorted(file["data"].keys())):
                demo = file[f"data/{demo_key}"]
                images = np.asarray(demo[f"obs/{args.image_key}"])
                dataset = PreprocessDataset(images, goal_dist, image_size)
                probabilities = [evaluator.inference.get_score(batch) for batch in DataLoader(dataset, batch_size=args.batch_size)]
                scores = np.asarray(evaluator.rank_prob_to_score(np.concatenate(probabilities), dataset.dist, freq), dtype=np.float32).reshape(-1)
                if "subop_score" in demo:
                    del demo["subop_score"]
                demo.create_dataset("subop_score", data=scores, compression="gzip")


if __name__ == "__main__":
    main()
