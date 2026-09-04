"""Export at most twenty annotated canonical-responsibility test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


def parse():
    p = argparse.ArgumentParser(); p.add_argument("--benchmark-hdf5", required=True); p.add_argument("--predictions", required=True); p.add_argument("--operating-points", required=True); p.add_argument("--canonical-seed", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--per-case-type", type=int, default=5); return p.parse_args()


def select_cases(part, threshold, k):
    part = part.copy(); part["deleted"] = part.score >= threshold; effective = part[part.is_responsible_point.groupby(part.pair_id).transform("any")]
    quality = []
    for pair, episode in effective.groupby("pair_id"):
        peak = episode.sort_values(["score", "t"], ascending=[False, True]).iloc[0]; resp = int(episode.loc[episode.is_responsible_point, "t"].iloc[0])
        quality.append({"pair_id": pair, "abs_delay": abs(int(peak.t) - resp), "responsible_deleted": bool(episode.loc[episode.is_responsible_point, "deleted"].iloc[0])})
    quality = pd.DataFrame(quality)
    recovery = part.groupby("pair_id").apply(lambda g: int((g.is_recovery & g.deleted).sum()), include_groups=False).rename("count").reset_index()
    no_effect = part.groupby("pair_id").apply(lambda g: int((g.is_no_effect_intervention & g.deleted).sum()), include_groups=False).rename("count").reset_index()
    selectors = {"top1_accurate": quality[quality.abs_delay <= 1].sort_values(["abs_delay", "pair_id"]).head(k), "responsible_miss": quality[~quality.responsible_deleted].sort_values(["abs_delay", "pair_id"], ascending=[False, True]).head(k), "recovery_false_deletion": recovery[recovery["count"] > 0].sort_values(["count", "pair_id"], ascending=[False, True]).head(k), "no_effect_false_attribution": no_effect[no_effect["count"] > 0].sort_values(["count", "pair_id"], ascending=[False, True]).head(k)}
    return [(kind, pair) for kind, frame in selectors.items() for pair in frame.pair_id.tolist()]


def render(group, episode, threshold, out_path):
    images = np.asarray(group["obs/agentview_image"], dtype=np.uint8); h, w = images.shape[1:3]; canvas_h = h + 90
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(group.attrs.get("control_freq", 20)), (w, canvas_h))
    scores = episode.sort_values("t").score.to_numpy(float); maximum = max(float(scores.max()), threshold, 1e-6)
    attrs = group.attrs; responsible = int(attrs.get("responsible_t", -1)); onset = int(attrs.get("failure_onset", -1)); recovery_start, recovery_end = int(attrs.get("recovery_start", -1)), int(attrs.get("recovery_end", -1))
    for t, image in enumerate(images):
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR); panel = np.zeros((90, w, 3), dtype=np.uint8)
        points = np.column_stack([np.linspace(0, w - 1, len(scores)).astype(int), (82 - 65 * scores / maximum).astype(int)])
        cv2.polylines(panel, [points], False, (0, 220, 255), 1); line = int(82 - 65 * threshold / maximum); cv2.line(panel, (0, line), (w - 1, line), (0, 0, 255), 1)
        for marker, color in ((responsible, (0, 255, 0)), (onset, (0, 165, 255)), (recovery_start, (255, 0, 255)), (recovery_end, (255, 0, 255))):
            if marker >= 0: cv2.line(panel, (int(marker / max(1, len(scores)-1) * (w-1)), 5), (int(marker / max(1, len(scores)-1) * (w-1)), 85), color, 1)
        cv2.putText(panel, f"t={t}  score={scores[min(t, len(scores)-1)]:.3f}  threshold={threshold:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
        writer.write(np.vstack([frame, panel]))
    writer.release()


def main():
    args = parse(); canonical = json.loads(Path(args.canonical_seed).read_text())["canonical_method"]; threshold = float(json.loads(Path(args.operating_points).read_text())["methods"][canonical]["threshold"])
    all_predictions = pd.read_parquet(args.predictions); part = all_predictions[(all_predictions.method == canonical) & (all_predictions["split"] == "test") & (all_predictions.variant == "perturbed") & (all_predictions.label_status != "ambiguous")]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); rows = []
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for kind, pair in select_cases(part, threshold, args.per_case_type):
            episode = part[part.pair_id == pair]; demo = str(episode.demo_id.iloc[0]); filename = f"{kind}__{str(pair).replace(':', '_').replace('.', '_')}.mp4"; render(h5["data"][demo], episode, threshold, output / filename); rows.append({"case_type": kind, "pair_id": pair, "demo_id": demo, "method": canonical, "threshold": threshold, "video": filename})
    pd.DataFrame(rows).to_csv(output / "case_index.csv", index=False); print(f"wrote {len(rows)} canonical case videos")


if __name__ == "__main__": main()
