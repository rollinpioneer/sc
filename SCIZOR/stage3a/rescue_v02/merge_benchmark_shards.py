from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py


def text(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5-shards", nargs="+", type=Path, required=True)
    p.add_argument("--meta-shards", nargs="+", type=Path, required=True)
    p.add_argument("--output-hdf5", type=Path, required=True)
    p.add_argument("--output-meta", type=Path, required=True)
    p.add_argument("--output-split", type=Path, required=True)
    a = p.parse_args()
    if len(a.hdf5_shards) != len(a.meta_shards):
        raise ValueError("HDF5 and metadata shard counts differ")
    if a.output_hdf5.exists():
        raise FileExistsError(a.output_hdf5)
    a.output_hdf5.parent.mkdir(parents=True, exist_ok=True); a.output_meta.parent.mkdir(parents=True, exist_ok=True)
    groups, rows, splits = set(), [], defaultdict(list)
    with h5py.File(a.output_hdf5, "w") as out:
        data = out.create_group("data")
        copied_root = False
        for shard in a.hdf5_shards:
            with h5py.File(shard, "r") as src:
                if not copied_root:
                    for k, v in src["data"].attrs.items(): data.attrs[k] = v
                    copied_root = True
                for name in src["data"]:
                    if name in groups: raise ValueError(f"duplicate group: {name}")
                    src.copy(src["data"][name], data, name=name); groups.add(name)
        data.attrs["total_groups"] = len(groups)
        data.attrs["total_pairs"] = sum(1 for name in groups if text(data[name].attrs.get("variant", "")) == "perturbed")
    for shard in a.meta_shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line); rows.append(row); splits[str(row["split"])].append(str(row["pair_id"]))
    if len({r["pair_id"] for r in rows}) != len(rows): raise ValueError("duplicate pair metadata")
    a.output_meta.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    a.output_split.write_text(json.dumps({k: sorted(v) for k, v in splits.items()}, indent=2), encoding="utf-8")
    print(json.dumps({"groups": len(groups), "pairs": len(rows), "split_counts": {k: len(v) for k, v in splits.items()}}, indent=2))


if __name__ == "__main__": main()
