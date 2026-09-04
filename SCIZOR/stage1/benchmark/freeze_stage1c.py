"""Apply the Stage 1C label freeze corrections without replaying rollouts."""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EFFECTIVE_TYPES = {
    "direct_failure",
    "delayed_failure",
    "recovery_failure",
    "recovery_success",
}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _value(value):
    return None if value is None or int(value) < 0 else int(value)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _render_pair_video(h5, row, output):
    clean = h5["data"][row["clean_demo_id"]]["obs"]["agentview_image"]
    perturbed = h5["data"][row["perturbed_demo_id"]]["obs"]["agentview_image"]
    length = min(len(clean), len(perturbed))
    scale, header, footer = 3, 82, 30
    frame_w, frame_h = clean.shape[2] * scale, clean.shape[1] * scale
    canvas_size = (frame_w * 2, header + frame_h + footer)
    title_font, body_font = _font(15), _font(13)
    onset = _value(row.get("failure_onset"))
    recovery_start = _value(row.get("recovery_start"))
    recovery_end = _value(row.get("recovery_end"))
    writer = imageio.get_writer(str(output), fps=max(1, int(row.get("control_freq", 20))), codec="libx264", quality=8)
    try:
        for t in range(length):
            canvas = Image.new("RGB", canvas_size, "white")
            draw = ImageDraw.Draw(canvas)
            left = Image.fromarray(np.asarray(clean[t])).resize((frame_w, frame_h), Image.Resampling.NEAREST)
            right = Image.fromarray(np.asarray(perturbed[t])).resize((frame_w, frame_h), Image.Resampling.NEAREST)
            canvas.paste(left, (0, header))
            canvas.paste(right, (frame_w, header))
            draw.text((6, 4), "clean", fill="black", font=title_font)
            draw.text((frame_w + 6, 4), "perturbed", fill="black", font=title_font)
            draw.text(
                (6, 26),
                f"{row['pair_id']} | intervention={row['perturb_t']} | onset={onset} | recovery={recovery_start}:{recovery_end}",
                fill="black",
                font=body_font,
            )
            status = []
            if t == int(row["perturb_t"]):
                status.append("INTERVENTION")
                draw.rectangle((frame_w, header, frame_w * 2 - 1, header + frame_h - 1), outline="red", width=4)
            if onset is not None and t == onset:
                status.append("FAILURE ONSET")
            if recovery_start is not None and recovery_start <= t <= (recovery_end if recovery_end is not None else length - 1):
                status.append("RECOVERY WINDOW")
            draw.text((6, 54), " | ".join(status) or "", fill=(0, 120, 0) if "RECOVERY WINDOW" in status else (180, 120, 0), font=body_font)
            draw.text((6, header + frame_h + 6), f"t={t}/{length - 1}", fill="black", font=body_font)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def _correction(row):
    """A fixed policy for the known no_effect/onset annotation conflict."""
    if bool(row.get("final_success_perturbed")) and row.get("recovery_start") is not None and row.get("recovery_end") is not None:
        return "recovery_success", "ok", "fixed_rule_complete_recovery"
    return "ambiguous", "ambiguous", "no_effect_with_onset_without_complete_recovery"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--correction-ledger", required=True)
    parser.add_argument("--stage1b-backup", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    backup_path = Path(args.stage1b_backup)
    rows = _rows(metadata_path)
    inconsistent = [row for row in rows if row.get("failure_type") == "no_effect" and row.get("failure_onset") is not None]
    if not inconsistent:
        raise RuntimeError("No inconsistent no_effect rows found; refusing to create an empty correction freeze.")
    if backup_path.exists():
        if _rows(backup_path) != rows:
            raise RuntimeError(f"Backup already exists and differs from current metadata: {backup_path}")
    else:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_path, backup_path)

    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    ledgers = []
    by_pair = {row["pair_id"]: row for row in rows}
    with h5py.File(args.benchmark_hdf5, "r+") as h5:
        for original in inconsistent:
            row = by_pair[original["pair_id"]]
            video_name = original["pair_id"].replace(":", "_").replace(".", "_") + ".mp4"
            video_path = review_dir / video_name
            _render_pair_video(h5, original, video_path)
            corrected_type, corrected_status, rule = _correction(original)
            before = {key: original.get(key) for key in ("failure_type", "failure_onset", "recovery_start", "recovery_end", "label_status")}
            row["failure_type"] = corrected_type
            row["label_status"] = corrected_status
            row["stage1c_correction_rule"] = rule
            row["is_inconsistent_no_effect"] = True
            row["intervention_t"] = int(row["perturb_t"])
            row["is_effective_intervention"] = corrected_type in EFFECTIVE_TYPES
            group = h5["data"][row["perturbed_demo_id"]]
            for key, value in row.items():
                if key in {"original_action", "perturbed_action", "action_delta_norm", "stage1c_correction_rule", "is_inconsistent_no_effect", "intervention_t", "is_effective_intervention"}:
                    continue
                if key in group.attrs:
                    group.attrs[key] = -1 if value is None and key in {"failure_onset", "recovery_start", "recovery_end"} else value
            group.attrs["intervention_t"] = int(row["intervention_t"])
            group.attrs["is_effective_intervention"] = bool(row["is_effective_intervention"])
            group.attrs["stage1c_correction_rule"] = rule
            group.attrs["is_inconsistent_no_effect"] = True
            ledgers.append({
                "pair_id": original["pair_id"],
                "review_video": str(video_path),
                "reviewer_rule": rule,
                "original": before,
                "corrected": {"failure_type": corrected_type, "label_status": corrected_status},
            })

    for row in rows:
        row.setdefault("intervention_t", int(row["perturb_t"]))
        row["is_effective_intervention"] = row.get("failure_type") in EFFECTIVE_TYPES
        row.setdefault("is_inconsistent_no_effect", False)
    # Persist the new Stage 1C semantics for every perturbed episode, not only
    # the 15 corrected records. The rollout arrays remain untouched.
    with h5py.File(args.benchmark_hdf5, "r+") as h5:
        for row in rows:
            group = h5["data"][row["perturbed_demo_id"]]
            group.attrs["intervention_t"] = int(row["intervention_t"])
            group.attrs["is_effective_intervention"] = bool(row["is_effective_intervention"])
            group.attrs["is_inconsistent_no_effect"] = bool(row["is_inconsistent_no_effect"])
    _dump_jsonl(metadata_path, rows)
    _dump_jsonl(args.correction_ledger, ledgers)
    print(json.dumps({
        "reviewed_count": len(inconsistent),
        "correction_counts": {"ambiguous": sum(x["corrected"]["failure_type"] == "ambiguous" for x in ledgers), "recovery_success": sum(x["corrected"]["failure_type"] == "recovery_success" for x in ledgers)},
        "review_dir": str(review_dir),
        "correction_ledger": args.correction_ledger,
    }, indent=2))


if __name__ == "__main__":
    main()
