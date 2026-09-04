import argparse
import json
from pathlib import Path

import h5py


def inspect(path: str) -> dict:
    result = {"path": str(Path(path).resolve()), "errors": []}
    with h5py.File(path, "r") as f:
        if "data" not in f:
            result["errors"].append("missing /data")
            return result
        demos = sorted(f["data"].keys(), key=lambda item: int(item.split("_")[-1]))
        result["num_demos"] = len(demos)
        result["has_env_args"] = "env_args" in f["data"].attrs
        result["num_transitions"] = sum(
            int(f[f"data/{demo}"]["actions"].shape[0])
            for demo in demos if "actions" in f[f"data/{demo}"]
        )
        result["sample_demos"] = []
        for demo_key in demos[:3]:
            demo = f[f"data/{demo_key}"]
            row = {"demo": demo_key, "has_model_file": "model_file" in demo.attrs}
            for required in ("actions", "states", "obs/agentview_image"):
                if required not in demo:
                    result["errors"].append(f"{demo_key}: missing {required}")
            if "actions" in demo:
                row.update(actions_shape=list(demo["actions"].shape), actions_dtype=str(demo["actions"].dtype))
            if "states" in demo:
                row["states_shape"] = list(demo["states"].shape)
            if "obs/agentview_image" in demo:
                row.update(image_shape=list(demo["obs/agentview_image"].shape), image_dtype=str(demo["obs/agentview_image"].dtype))
            result["sample_demos"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = [inspect(path) for path in args.inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(item["errors"] for item in report):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
