"""Freeze the canonical full seed using validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(seed, values):
    overall = values["methods"][f"responsibility_seed{seed}"]["overall"]
    return (overall.get("top1_within_1") or 0., overall.get("responsibility_region_iou") or 0., overall.get("transition_f1") or 0., overall.get("recovery_retention") or 0., -(overall.get("no_effect_false_attribution_rate") or 0.), -seed)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--validation-metrics", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    values = json.loads(args.validation_metrics.read_text()); seeds = [0, 1, 2]; chosen = max(seeds, key=lambda seed: score(seed, values))
    payload = {"selection_split": "validation", "canonical_seed": chosen, "canonical_method": f"responsibility_seed{chosen}", "selection_key": ["top1_within_1", "responsibility_region_iou", "transition_f1", "recovery_retention", "-no_effect_false_attribution_rate", "-seed"], "candidate_scores": {str(seed): list(score(seed, values)) for seed in seeds}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
