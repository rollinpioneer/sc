import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from natsort import natsorted


def text_attr(demo, key):
    value = demo.attrs.get(key, "")
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def infer_task(path, demo):
    task = text_attr(demo, "task")
    if task:
        return task
    return "square" if "square" in str(path).lower() else "can" if "can" in str(path).lower() else path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.data_dir).rglob("*.hdf5")):
        with h5py.File(path, "r") as file:
            for demo_id in natsorted(file["data"].keys()):
                demo = file[f"data/{demo_id}"]
                if "subop_score" not in demo:
                    raise KeyError(f"{path}:{demo_id} missing subop_score")
                scores, actions = np.asarray(demo["subop_score"]).reshape(-1), np.asarray(demo["actions"])
                length = min(len(scores), len(actions))
                for t in range(length):
                    motion = actions[t, :-1] if actions.shape[1] > 1 else actions[t]
                    rows.append({"dataset_file": str(path.resolve()), "task": infer_task(path, demo), "demo_id": demo_id,
                        "pair_id": text_attr(demo, "pair_id"), "variant": text_attr(demo, "variant"), "base_demo_id": text_attr(demo, "base_demo_id"),
                        "t": t, "episode_length": length, "subop_score": float(scores[t]), "action_norm": float(np.linalg.norm(motion)),
                        "gripper_action": float(actions[t, -1]) if actions.shape[1] > 1 else float("nan")})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
