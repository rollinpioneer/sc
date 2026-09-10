# HB3-P Stop/Continue Pilot

This report covers one frozen fixed-entry pilot. It is not a formal non-inferiority result.

## Frozen Protocol

- Entry takeover: `t=20`.
- Selector query: exactly once at `t=80` when the episode reaches it.
- STOP: helper interval `[20,80)`, permanent handback at `t=80`.
- CONTINUE: helper interval `[20,100)`, permanent handback at `t=100`.
- Methods: `NONE, FIXED_L60, FIXED_L80, LEARNED_STOP_CONTINUE`.
- Declared absolute success-rate non-inferiority margin: `0.05`.
- Non-inferiority comparator: `FIXED_L80`; short-exit comparator: `FIXED_L60`.
- Independent test roots: `40`; this run is explicitly `PILOT`.

## Label And Model

- Paired source roots: `40`; labels STOP/CONTINUE = `35/5`.
- Labels were generated from realized paired utility of the existing FIXED_L60 and FIXED_L80 runs; no old H/M1 probability was thresholded.
- OOF threshold: `0.00028433`; OOF accuracy: `30.00%`.
- Input: four frames of 9-D proprioception, four 7-D base-action suggestions, and current normalized time.
- Excluded: reward, success, object state, privileged state, future result, images, root ID, seed, repair action.

## Coverage And Engineering Checks

- Complete logical records: `160/160`.
- Unique rollouts including baseline: `160`.
- Prefix checks: `40/40`.
- Learned-to-selected-fixed full trajectory parity: `40/40`.
- Missing records / engineering failures: `0/0`.
- Required online invariants: one base-policy call per executed step, at most one takeover, no repair call after handback.

## Results

| Method | System success | Autonomous completion | Mean U | Mean helper steps |
|---|---:|---:|---:|---:|
| NONE | 5.00% (2/40) | 5.00% (2/40) | 0.050000 | 0.000 |
| FIXED_L60 | 22.50% (9/40) | 22.50% (9/40) | 0.187500 | 60.000 |
| FIXED_L80 | 20.00% (8/40) | 20.00% (8/40) | 0.150000 | 80.000 |
| LEARNED_STOP_CONTINUE | 17.50% (7/40) | 17.50% (7/40) | 0.126562 | 77.500; STOP/CONTINUE=5/35 |

## Paired Comparisons

- `LEARNED_STOP_CONTINUE_minus_FIXED_L60` utility delta: `-0.060937`, 95% root-bootstrap CI `[-0.161562, 0.038750]`.
  System-success delta: `-0.050000`, CI `[-0.150000, 0.050000]`; autonomous delta: `-0.050000`, CI `[-0.150000, 0.050000]`.
- `LEARNED_STOP_CONTINUE_minus_FIXED_L80` utility delta: `-0.023437`, 95% root-bootstrap CI `[-0.073438, 0.002188]`.
  System-success delta: `-0.025000`, CI `[-0.075000, 0.000000]`; autonomous delta: `-0.025000`, CI `[-0.075000, 0.000000]`.
- `LEARNED_STOP_CONTINUE_minus_NONE` utility delta: `0.076563`, 95% root-bootstrap CI `[-0.000250, 0.177500]`.
  System-success delta: `0.125000`, CI `[0.049375, 0.225000]`; autonomous delta: `0.125000`, CI `[0.049375, 0.225000]`.

## Interpretation

- Diagnostic system-success non-inferiority check versus FIXED_L80: `False`.
- Diagnostic autonomous-completion non-inferiority check versus FIXED_L80: `False`.
- These checks do not establish non-inferiority: the sample is 40 roots, while the pre-run estimate was roughly 354 roots for about 0.05 precision (worst-case binary proportion approximately 384 roots).
- The result should be used to decide whether a larger preregistered run is warranted, not as a deployment guarantee.

## Provenance

- Source probe protocol SHA256: `2dc31f4aefb76dd25658a90e6c989bceffc24604be44efe6b172a3cdceb60c31`.
- Frozen execution code commit: `28565d3054fb251593e5ef73b4d6f896d6a6d2a5`.
- Selector checkpoint SHA256: `25c29777ede276b2cfd83c978d25d0989a63511aae359de21b57448ef15b6863`.
- Frozen protocol SHA256: `7e762f5a5df4c06f9aa9c77e11435b594687d28c72c61dcf9ebde7fbe9ee0a8e`.
- Large trajectory, model, image, log, and IPC assets remain local and are excluded from the lightweight Git package.
