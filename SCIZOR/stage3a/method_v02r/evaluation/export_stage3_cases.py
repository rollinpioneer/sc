"""Export twelve bounded, annotated blind-test explanation videos.

The videos are explanatory artifacts only.  Selection is deterministic and
uses the already-frozen proposer source and pair threshold; no metric or
operating point is changed here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _attrs(group) -> dict:
    result = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif hasattr(value, "item"):
            value = value.item()
        result[str(key)] = value
    return result


def _images(group, stem: str) -> np.ndarray:
    obs = group.get("obs")
    if obs is None:
        return np.zeros((0, 84, 84, 3), dtype=np.uint8)
    for name in (f"agentview_image_{stem}", "agentview_image"):
        if name in obs:
            return np.asarray(obs[name][:], dtype=np.uint8)
    return np.zeros((0, 84, 84, 3), dtype=np.uint8)


def _first(frame: pd.DataFrame, names: tuple[str, ...], default=None):
    for name in names:
        if name in frame:
            value = frame[name].iloc[0]
            if pd.notna(value):
                return value
    return default


def _select_cases(pairs: pd.DataFrame, threshold: float, per_type: int) -> list[dict]:
    pairs = pairs.copy()
    pairs["effective"] = pairs["is_effective_intervention"].astype(bool)
    pairs["positive"] = pairs["fused_pair_score"].astype(float) >= float(threshold)
    pairs["abs_delay"] = (pairs["fused_predicted_t"].astype(int) - pairs["intervention_t"].astype(int)).abs()

    selectors = {
        "effective_correct_localization": pairs[
            pairs.effective & pairs.positive & (pairs.abs_delay <= 1)
        ].sort_values(["abs_delay", "fused_pair_score", "pair_id"], ascending=[True, False, True]),
        "effective_missed": pairs[
            pairs.effective & ~pairs.positive
        ].sort_values(["fused_pair_score", "abs_delay", "pair_id"], ascending=[True, False, True]),
        "no_effect_correct_suppression": pairs[
            ~pairs.effective & ~pairs.positive
        ].sort_values(["fused_pair_score", "pair_id"], ascending=[True, True]),
        "no_effect_wrong_attribution": pairs[
            ~pairs.effective & pairs.positive
        ].sort_values(["fused_pair_score", "pair_id"], ascending=[False, True]),
    }
    selected: list[dict] = []
    for case_type, frame in selectors.items():
        for row in frame.head(per_type).to_dict("records"):
            selected.append({"case_type": case_type, **row})
    counts = {kind: sum(row["case_type"] == kind for row in selected) for kind in selectors}
    missing = {kind: per_type - count for kind, count in counts.items() if count < per_type}
    if missing:
        raise RuntimeError(f"not enough blind cases for required export: {missing}")
    return selected


def _format(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "na"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _render_case(
    output: Path,
    pair_row: dict,
    perturbed,
    clean,
    transition_rows: pd.DataFrame,
    replacement_rows: pd.DataFrame,
    threshold: float,
) -> None:
    import imageio.v2 as imageio

    pert = _images(perturbed, "post")
    if len(pert) == 0:
        pert = _images(perturbed, "pre")
    clean_images = _images(clean, "post") if clean is not None else np.zeros_like(pert)
    if len(clean_images) == 0:
        clean_images = _images(clean, "pre") if clean is not None else np.zeros_like(pert)
    length = min(len(pert), len(clean_images)) if len(clean_images) else len(pert)
    if length <= 0:
        raise RuntimeError(f"no renderable images for {pair_row['pair_id']}")

    attrs = _attrs(perturbed)
    control_freq = int(attrs.get("control_freq", 20))
    true_t = _number(pair_row.get("intervention_t", attrs.get("perturb_t", -1)))
    pred_t = _number(pair_row.get("fused_predicted_t", -1))
    pair_score = float(pair_row.get("fused_pair_score", float("nan")))
    failure_type = str(pair_row.get("failure_type", attrs.get("failure_type", "")))
    task = str(pair_row.get("task", attrs.get("task", "")))
    pair_id = str(pair_row["pair_id"])

    # Prefer the selected proposer source.  The table can still contain all
    # sources when produced by build_pipeline_scores.
    transitions = transition_rows[transition_rows.pair_id.astype(str).eq(pair_id)].copy()
    replacements = replacement_rows[replacement_rows.pair_id.astype(str).eq(pair_id)].copy()
    title_font, body_font, tiny_font = _font(14), _font(11), _font(9)
    scale = 3
    image_h, image_w = pert.shape[1], pert.shape[2]
    frame_w, frame_h = image_w * scale, image_h * scale
    header, footer = 70, 138
    canvas_size = (frame_w * 2, header + frame_h + footer)
    writer = imageio.get_writer(str(output), fps=max(1, control_freq), codec="libx264", quality=8)
    try:
        for t in range(length):
            canvas = Image.new("RGB", canvas_size, "white")
            draw = ImageDraw.Draw(canvas)
            left = Image.fromarray(np.asarray(clean_images[t])).resize((frame_w, frame_h), Image.Resampling.NEAREST)
            right = Image.fromarray(np.asarray(pert[t])).resize((frame_w, frame_h), Image.Resampling.NEAREST)
            canvas.paste(left, (0, header))
            canvas.paste(right, (frame_w, header))
            draw.text((6, 4), "clean", fill="black", font=title_font)
            draw.text((frame_w + 6, 4), "perturbed", fill="black", font=title_font)
            draw.text((6, 25), f"{task} | {pair_id}", fill="black", font=body_font)
            draw.text((6, 43), f"case={pair_row['case_type']}  failure={failure_type}", fill="black", font=body_font)
            if t == true_t:
                draw.rectangle((frame_w, header, frame_w * 2 - 1, header + frame_h - 1), outline="red", width=5)
            if t == pred_t:
                draw.rectangle((0, header, frame_w - 1, header + frame_h - 1), outline="blue", width=5)

            # The selected transition row at this frame supplies frozen raw
            # proposal ranks/scores.  If absent, show the closest predicted
            # transition so every case remains interpretable.
            tr = transitions[transitions.query_t.astype(int).eq(t)]
            if tr.empty and len(transitions):
                tr = transitions.iloc[[int(np.abs(transitions.query_t.astype(int) - t).argmin())]]
            tr_row = tr.iloc[0].to_dict() if len(tr) else {}
            proposal_text = (
                f"t={t} true={true_t} pred={pred_t} "
                f"ranks(f/a/u)={tr_row.get('full_rank', 'na')}/{tr_row.get('action_rank', 'na')}/{tr_row.get('union_rank', 'na')}"
            )
            draw.text((6, header + frame_h + 6), proposal_text, fill="black", font=body_font)
            draw.text(
                (6, header + frame_h + 24),
                f"pair={_format(pair_score)} threshold={_format(threshold)} positive={pair_score >= threshold}",
                fill="black",
                font=body_font,
            )

            # Four replacement slots for the selected/nearest transition.
            q_t = _number(tr_row.get("query_t", pred_t if pred_t >= 0 else true_t))
            reps = replacements[replacements.query_t.astype(int).eq(q_t)].sort_values("replacement_rank") if len(replacements) else replacements
            rep_parts = []
            for _, rep in reps.head(4).iterrows():
                rank = _number(rep.get("replacement_rank", -1))
                mean = _first(pd.DataFrame([rep]), ("pred_score_mean", "pred_score", "replacement_cf_score"), None)
                std = _first(pd.DataFrame([rep]), ("pred_score_std",), None)
                rep_parts.append(f"r{rank}:{_format(mean)}+/-{_format(std)}")
            while len(rep_parts) < 4:
                rep_parts.append(f"r{len(rep_parts)}:na+/-na")
            draw.text((6, header + frame_h + 42), f"replacements q={q_t} " + " ".join(rep_parts), fill="black", font=tiny_font)
            draw.text((6, header + frame_h + 61), "ensemble mean/std shown; red=true perturbation, blue=predicted transition", fill="black", font=tiny_font)
            draw.text((6, header + frame_h + 79), f"raw={_format(tr_row.get('raw_proposer_score'))} cf={_format(tr_row.get('counterfactual_only_score'))} fused={_format(tr_row.get('fused_transition_score'))}", fill="black", font=tiny_font)
            draw.text((6, header + frame_h + 97), f"failure_onset={attrs.get('failure_onset', 'na')} recovery={attrs.get('recovery_start', 'na')}:{attrs.get('recovery_end', 'na')}", fill="black", font=tiny_font)
            draw.text((6, header + frame_h + 116), f"frame={t}/{length - 1}", fill="black", font=tiny_font)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-hdf5", type=Path, required=True)
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--transition-scores", type=Path, required=True)
    parser.add_argument("--replacement-scores", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--proposer-source")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-case-type", type=int, default=3)
    args = parser.parse_args()
    if args.threshold is None and args.protocol is None:
        raise ValueError("provide --threshold or --protocol")
    protocol = _json(args.protocol) if args.protocol else {}
    threshold = float(args.threshold if args.threshold is not None else protocol["selected_threshold"])
    source = args.proposer_source or protocol.get("selected_proposer")

    pairs = pd.read_parquet(args.pair_scores)
    transitions = pd.read_parquet(args.transition_scores)
    replacements = pd.read_parquet(args.replacement_scores)
    if source:
        selected_pairs = pairs[pairs.proposer_source.astype(str).eq(str(source))].copy()
        selected_transitions = transitions[transitions.proposer_source.astype(str).eq(str(source))].copy()
    else:
        selected_pairs, selected_transitions = pairs.copy(), transitions.copy()
    if selected_pairs.empty:
        raise ValueError(f"no pair rows for proposer source {source!r}")

    cases = _select_cases(selected_pairs, threshold, args.per_case_type)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_groups: dict[str, tuple[object, object, dict]] = {}
    with h5py.File(args.benchmark_hdf5, "r") as h5:
        for name, group in h5["data"].items():
            attrs = _attrs(group)
            if str(attrs.get("variant", "")) != "perturbed":
                continue
            pair_id = str(attrs.get("pair_id", ""))
            if pair_id not in {str(case["pair_id"]) for case in cases}:
                continue
            clean_name = str(attrs.get("clean_demo_id", ""))
            clean = h5["data"].get(clean_name)
            pair_groups[pair_id] = (group, clean, attrs)
        rows = []
        for index, case in enumerate(cases):
            pair_id = str(case["pair_id"])
            if pair_id not in pair_groups:
                raise KeyError(f"blind benchmark lacks HDF5 group for {pair_id}")
            perturbed, clean, attrs = pair_groups[pair_id]
            filename = f"{index:02d}__{case['case_type']}__{_safe_name(pair_id)}.mp4"
            output = args.output_dir / filename
            _render_case(output, case, perturbed, clean, selected_transitions, replacements, threshold)
            rows.append({
                "case_type": case["case_type"],
                "pair_id": pair_id,
                "task": case.get("task", attrs.get("task")),
                "failure_type": case.get("failure_type", attrs.get("failure_type")),
                "is_effective_intervention": bool(case.get("is_effective_intervention", False)),
                "intervention_t": _number(case.get("intervention_t", attrs.get("perturb_t", -1))),
                "predicted_t": _number(case.get("fused_predicted_t", -1)),
                "pair_score": float(case.get("fused_pair_score", float("nan"))),
                "threshold": threshold,
                "proposer_source": source,
                "video": filename,
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "case_index.csv", index=False)
    print(json.dumps({"case_count": len(rows), "counts": {kind: sum(row["case_type"] == kind for row in rows) for kind in sorted({row["case_type"] for row in rows})}, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
