"""Expose v0.2 pre-action images through the read-only Stage 1 scoring schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py


ATTRS = ("task", "pair_id", "variant", "base_demo_id", "split", "perturb_t", "failure_type", "is_effective_intervention")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = str(a.benchmark.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.benchmark, "r") as original, h5py.File(a.output, "w") as adapter:
        data = adapter.create_group("data")
        data.attrs["source_benchmark"] = source
        data.attrs["read_only_external_link_adapter"] = True
        for name, group in original["data"].items():
            out = data.create_group(name)
            out["actions"] = h5py.ExternalLink(source, f"/data/{name}/actions")
            obs = out.create_group("obs")
            obs["agentview_image"] = h5py.ExternalLink(source, f"/data/{name}/obs/agentview_image_pre")
            for key in ATTRS:
                if key in group.attrs:
                    out.attrs[key] = group.attrs[key]
    with h5py.File(a.output, "r") as checked:
        if len(checked["data"]) == 0:
            raise RuntimeError("empty scoring adapter")
        first = checked["data"][next(iter(checked["data"]))]
        if len(first["actions"]) != len(first["obs/agentview_image"]):
            raise RuntimeError("adapter image/action alignment failed")
    print({"groups": len(h5py.File(a.benchmark, "r")["data"]), "output": str(a.output)})


if __name__ == "__main__":
    main()
