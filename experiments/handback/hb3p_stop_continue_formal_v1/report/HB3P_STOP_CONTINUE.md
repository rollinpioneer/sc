# HB3-P Stop/Continue Formal

This report covers one frozen fixed-entry formal evaluation. The result is interpreted only under the preregistered protocol.

## Frozen Protocol

- Entry takeover: `t=20`.
- Selector query: exactly once at `t=80` when the episode reaches it.
- STOP: helper interval `[20,80)`, permanent handback at `t=80`.
- CONTINUE: helper interval `[20,100)`, permanent handback at `t=100`.
- Methods: `NONE, FIXED_L60, FIXED_L80, LEARNED_STOP_CONTINUE`.
- Declared absolute success-rate non-inferiority margin: `0.05`.
- Non-inferiority comparator: `FIXED_L80`; short-exit comparator: `FIXED_L60`.
- Independent test roots: `400`; this run is explicitly `FORMAL`.

## Label And Model

- Paired source roots: `40`; labels STOP/CONTINUE = `35/5`.
- Labels were generated from realized paired utility of the existing FIXED_L60 and FIXED_L80 runs; no old H/M1 probability was thresholded.
- OOF threshold: `0.00028433`; OOF accuracy: `30.00%`.
- Input: four frames of 9-D proprioception, four 7-D base-action suggestions, and current normalized time.
- Excluded: reward, success, object state, privileged state, future result, images, root ID, seed, repair action.

## Coverage And Engineering Checks

- Complete logical records: `1600/1600`.
- Unique rollouts including baseline: `1600`.
- Prefix checks: `400/400`.
- Learned-to-selected-fixed full trajectory parity: `400/400`.
- Missing records / engineering failures: `0/0`.
- Required online invariants: one base-policy call per executed step, at most one takeover, no repair call after handback.

## Results

| Method | System success | Autonomous completion | Mean U | Mean helper steps |
|---|---:|---:|---:|---:|
| NONE | 6.75% (27/400) | 6.75% (27/400) | 0.067500 | 0.000 |
| FIXED_L60 | 16.50% (66/400) | 16.50% (66/400) | 0.127500 | 60.000 |
| FIXED_L80 | 23.25% (93/400) | 21.00% (84/400) | 0.160000 | 80.000 |
| LEARNED_STOP_CONTINUE | 22.50% (90/400) | 20.50% (82/400) | 0.155750 | 78.800; STOP/CONTINUE=24/376 |

## Paired Comparisons

- `LEARNED_STOP_CONTINUE_minus_FIXED_L60` utility delta: `0.028250`, 95% root-bootstrap CI `[-0.006781, 0.063315]`.
  System-success delta: `0.060000`, CI `[0.027500, 0.095000]`; autonomous delta: `0.040000`, CI `[0.005000, 0.075000]`.
- `LEARNED_STOP_CONTINUE_minus_FIXED_L80` utility delta: `-0.004250`, 95% root-bootstrap CI `[-0.011812, 0.000813]`.
  System-success delta: `-0.007500`, CI `[-0.017500, 0.000000]`; autonomous delta: `-0.005000`, CI `[-0.012500, 0.000000]`.
- `LEARNED_STOP_CONTINUE_minus_NONE` utility delta: `0.088250`, 95% root-bootstrap CI `[0.043279, 0.133095]`.
  System-success delta: `0.157500`, CI `[0.110000, 0.202500]`; autonomous delta: `0.137500`, CI `[0.092500, 0.182500]`.

## Interpretation

- Diagnostic system-success non-inferiority check versus FIXED_L80: `True`.
- Diagnostic autonomous-completion non-inferiority check versus FIXED_L80: `True`.
- The confidence bounds are the prespecified formal decision quantities; this is not a deployment guarantee.

## Provenance

- Source probe protocol SHA256: `2dc31f4aefb76dd25658a90e6c989bceffc24604be44efe6b172a3cdceb60c31`.
- Frozen execution code commit: `83831b21e604a60b6908b605faa799e9e415f722`.
- Selector checkpoint SHA256: `25c29777ede276b2cfd83c978d25d0989a63511aae359de21b57448ef15b6863`.
- Frozen protocol SHA256: `040b679569eaab02f577c056fdc85f2049a19af3ad8351fc50d35c7afe46c3d3`.
- Large trajectory, model, image, log, and IPC assets remain local and are excluded from the lightweight Git package.
