# Benchmark v0.1 Data Card

## Source

Benchmark v0.1 uses successful Robomimic Robosuite image demonstrations for Can and Square: `data/robomimic/can/ph/image.hdf5` and `data/robomimic/square/ph/image.hdf5`. The source repository commit is `75051b3d45676a033533896da16a6a7abf8ac42e`.

## Contents

- 40 base episodes for Can and 40 for Square.
- 80 clean demonstrations, stored once each.
- 1,280 perturbed demonstrations: 640 Can and 640 Square.
- Every perturbed rollout has exactly one intervention at a known action step.
- Control frequency is 20 Hz.

The frozen labels contain 7 direct failures, 68 delayed failures, 66 recovery failures, 75 recovery successes, 1,049 consistent no-effect controls, and 15 ambiguous pairs. Ambiguous pairs are excluded from primary localization metrics. The perturbation types are `axis_impulse`, `flip_gripper`, `reverse_motion`, and `zero_motion`; selected magnitudes are recorded per pair in `pair_metadata.jsonl`.

## Split and evaluation

The split key is `(task, base_demo_id)`, with deterministic seed `20260831` and 60/20/20 train/validation/test group ratios. The manifest preserves all clean and perturbed variants of a base episode in one split. Ground truth comes from simulator reward/staged rewards, not from SCIZOR.

## Known limitation

Open-loop continuation can amplify a perturbed state distribution shift. The benchmark measures the consequence of a single action replacement under that controlled continuation, not a policy's closed-loop recovery behavior.
