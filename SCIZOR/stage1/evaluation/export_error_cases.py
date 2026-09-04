"""Export a bounded set of Stage 1D error-case videos and index rows."""

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd

from stage1.benchmark.freeze_stage1c import _render_pair_video


def _row_from_group(group):
    attrs = group.attrs
    def value(key, default=None):
        item = attrs.get(key, default)
        item = item.item() if hasattr(item, "item") else item
        return None if isinstance(item, int) and item < 0 and key in {"failure_onset", "recovery_start", "recovery_end"} else item
    return {key: value(key) for key in ("pair_id", "clean_demo_id", "perturbed_demo_id", "perturb_t", "failure_onset", "recovery_start", "recovery_end", "control_freq")}


def _cases(part, threshold, per_case_type):
    part = part.copy()
    part["deleted"] = part["score"] >= threshold
    effective = part[part["is_responsible_point"].groupby(part["pair_id"]).transform("any")]
    quality = []
    for pair_id, episode in effective.groupby("pair_id"):
        peak = episode.sort_values(["score", "t"], ascending=[False, True], kind="stable").iloc[0]
        responsible = episode.loc[episode["is_responsible_point"], "t"].iloc[0]
        quality.append({"pair_id": pair_id, "abs_delay": abs(int(peak.t) - int(responsible)), "responsible_deleted": bool(episode.loc[episode["is_responsible_point"], "deleted"].iloc[0])})
    quality = pd.DataFrame(quality)
    recovery = part.groupby("pair_id").apply(lambda group: int((group["is_recovery"] & group["deleted"]).sum()), include_groups=False).rename("recovery_deleted").reset_index()
    innocent = part.groupby("pair_id").apply(lambda group: int((group["is_innocent_downstream"] & group["deleted"]).sum()), include_groups=False).rename("innocent_deleted").reset_index()
    result = []
    selectors = {
        "accurate": quality.sort_values(["abs_delay", "pair_id"]).head(per_case_type),
        "responsible_miss": quality[~quality["responsible_deleted"]].sort_values(["abs_delay", "pair_id"], ascending=[False, True]).head(per_case_type),
        "recovery_false_delete": recovery[recovery["recovery_deleted"] > 0].sort_values(["recovery_deleted", "pair_id"], ascending=[False, True]).head(per_case_type),
        "innocent_false_delete": innocent[innocent["innocent_deleted"] > 0].sort_values(["innocent_deleted", "pair_id"], ascending=[False, True]).head(per_case_type),
    }
    for case_type, frame in selectors.items():
        for record in frame.to_dict("records"):
            result.append({"case_type": case_type, **record})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-case-type", type=int, default=5)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    operating = json.loads(Path(args.operating_points).read_text(encoding="utf-8"))["methods"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for method, spec in sorted(operating.items()):
            part = predictions[(predictions["method"] == method) & (predictions["split"] == "test") & (predictions["variant"] == "perturbed") & (predictions["label_status"] != "ambiguous")]
            for record in _cases(part, float(spec["threshold"]), args.per_case_type):
                pair_id = record["pair_id"]
                demo_id = part.loc[part["pair_id"] == pair_id, "demo_id"].iloc[0]
                metadata = _row_from_group(h5["data"][demo_id])
                filename = f"{method}__{record['case_type']}__{str(pair_id).replace(':', '_').replace('.', '_')}.mp4"
                _render_pair_video(h5, metadata, output_dir / filename)
                selected.append({"method": method, "video": filename, **record})
    pd.DataFrame(selected).to_csv(output_dir / "case_index.csv", index=False)
    print(f"wrote {len(selected)} case videos")


if __name__ == "__main__":
    main()
