# HB3-P Report

## Material Passport

- Verification Status: VERIFIED
- Experiment: HB3-P-v1 Square
- Source commit: `732d2ead92b5e4117c595049496dbbc3706f62e8`
- Frozen execution code commit: `bb2d3b044dfddcb2b0aa65f0c32798d1738a1509`
- Result analysis code commit: `d243a96b4639c0713ff3d818a1485d1d01f18d57`
- Result status: **STATE_ENTRY_SIGNAL_ONLY**
- Frozen protocol SHA256: `07ed1213f636ea1db2500798b4468add685249faa7626d6bf39628f0fb5cccc3`

## Fixed Pair And Scope

- Base checkpoint SHA256: `e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6`
- Repair checkpoint SHA256: `7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62`
- Semantic pair ID: `a8ca96a4f5975b211053a3b6a1d34bc15a77b5ef2297cf6d8a7c97ac30fb1236`
- Scope: fixed candidate times 20/80/160, at most one takeover, fixed 5/20/80-step exit, permanent handback.
- Preserved HB2 status: `HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN`.
- No action policy, teacher, DINO backbone, F model, or H model was retrained.
- After collection, an analysis-only correction normalized missing numeric values in decision-count, latency, and case-display outputs. It did not change trajectories, frozen decision rules, method metrics, confidence intervals, or the research status.

## Episode-Level Change From HB2

HB2 averaged decisions over separately supplied legal anchors. HB3-P executes one complete trajectory from t=0 per method and root; after the first nonzero decision it never queries F again. Therefore HB2 anchor-level utilities are historical context and are not subtracted from these episode-level results.

## Development And Frozen Rules

- Development roots / legal anchors / complete branches: 40 / 119 / 595.
- Selected fixed schedule from old validation: `S_20_80` with validation U=0.1750.
- Compiled M0 schedule: `{"160": 0, "20": 80, "80": 20}`.
- Method aliases: `[{"alias": "M0_GRID", "canonical_execution": "S_STAR", "reason": "same_complete_validation_execution_under_frozen_rules"}, {"alias": "RISK_GRID", "canonical_execution": "S_STAR", "reason": "same_complete_validation_execution_under_frozen_rules"}]`.
- Primary method / comparator: `M1_GRID` / `S_STAR`.

## Pilot And Online Semantics

- Pilot roots / complete records: 4 / 16.
- All input, baseline, fixed-branch control-semantic, and selection parity checks passed: True.
- Pilot repair calls after handback: 0.
- Mean pilot episode wall time: 11.0673 seconds.
- Mean live M4 inference+IPC time per pilot episode: 0.4841 seconds.

## New Test Coverage

- Preregistered independent roots: 80 (seeds 500000-500079).
- Complete logical root-method records: 480 / 480.
- Actual unique rollouts including baseline: 320.
- Total executed environment steps across unique rollouts: 117135.
- Missing records / engineering failures: 0 / 0.

## Complete-Trajectory Results

| Method | System success | Autonomous completion | U (lambda=0.25) | Mean helper steps | Genuine rescued roots |
|---|---:|---:|---:|---:|---:|
| M0_GRID | 0.2500 (20/80) | 0.2375 (19/80) | 0.1875 | 80.0000 | 17 |
| M1_GRID | 0.2500 (20/80) | 0.2375 (19/80) | 0.2020 | 56.7500 | 17 |
| M4_GRID | 0.1875 (15/80) | 0.1750 (14/80) | 0.1350 | 64.0000 | 12 |
| NONE | 0.0375 (3/80) | 0.0375 (3/80) | 0.0375 | 0.0000 | 0 |
| RISK_GRID | 0.2500 (20/80) | 0.2375 (19/80) | 0.1875 | 80.0000 | 17 |
| S_STAR | 0.2500 (20/80) | 0.2375 (19/80) | 0.1875 | 80.0000 | 17 |

## Paired Comparisons

- Primary M1_GRID - S_STAR delta U: 0.014531; 95% paired-root percentile CI [-0.068457, 0.093523], n=80, 2000 resamples, seed 20260909.
- Exploratory M1_GRID_minus_NONE: delta U=0.164531, CI=[0.075937, 0.263125].
- Exploratory M1_GRID_minus_M0_GRID: delta U=0.014531, CI=[-0.068457, 0.093523].
- Exploratory M4_GRID_minus_S_STAR: delta U=-0.052500, CI=[-0.126250, 0.012516].
- Exploratory M4_GRID_minus_M1_GRID: delta U=-0.067031, CI=[-0.144707, 0.010172].
- Exploratory S_STAR_minus_NONE: delta U=0.150000, CI=[0.062500, 0.250000].

## Interference And Cost

| Method | Interference / all baseline-success | Interference / helped baseline-success | Mean takeover | Mean queries | Repair calls after handback |
|---|---:|---:|---:|---:|---:|
| M0_GRID | 1/3 (0.3333) | 1/3 (0.3333) | 1.0000 | 0.0000 | 0 |
| M1_GRID | 1/3 (0.3333) | 1/3 (0.3333) | 0.7750 | 1.5875 | 0 |
| M4_GRID | 1/3 (0.3333) | 1/3 (0.3333) | 0.8000 | 1.4375 | 0 |
| NONE | 0/3 (0.0000) | 0/0 (NA) | 0.0000 | 0.0000 | 0 |
| RISK_GRID | 1/3 (0.3333) | 1/3 (0.3333) | 1.0000 | 0.0000 | 0 |
| S_STAR | 1/3 (0.3333) | 1/3 (0.3333) | 1.0000 | 0.0000 | 0 |

## Predictor Inputs And Latency

- F received only the two current camera streams, 9-D proprioception, the current base suggestion history, and normalized absolute time. Labels, reward, privileged object state, future actions, root seed encodings, and future images were excluded.
- The base policy was advanced exactly once on every executed environment step, including helper-controlled steps. M4 used a synchronous local file queue; its backbone, head, IPC, and total wall times are reported in `metrics/test/costs.csv`.
- Simulation waits for inference and therefore does not claim a real-robot 20 Hz latency result.

## HB2 Ranking Supplement

- Ranking supplement artifact: `metrics/clarification/ranking_metrics.json` (8 top-level entries).
- AUROC/AP were recomputed from existing predictions only. This did not change HB2 model selection, temperatures, thresholds, utility, or the preserved HB2 status.
- HB2 local/history/paired gains remain validation-set single-seed point estimates, not independent significance results.

## Layered Decision

- HB3-P status: **STATE_ENTRY_SIGNAL_ONLY**.
- State-entry gain: directional.
- Time-schedule gain: supported.
- Visual increment: unproven (pre-registered secondary comparison only).
- Baseline preservation evidence: small_sample.

## Not Tested

Learned H-controlled exit, repeated takeover, arbitrary query times, Can, full-help capability on the new roots, policy improvement, distribution shift, and real-robot execution were not tested. The result does not reclassify HB2 visual evidence or claim complete HB-3.

## Next Allowed Scope

Any continuation must use a new frozen protocol. If mechanism value is supported, the next bounded question is stop-versus-continue at a fixed entry after 5/20 helper steps; current H probabilities are not a validated stopping controller.

## Selected Cases

- `effective_help`: `square:hb3p_test:500008`, M1 t=20, L=80.
- `baseline_interference`: `square:hb3p_test:500047`, M1 t=80, L=20.
- `state_time_choice_difference`: `square:hb3p_test:500000`, M1 t=NA, L=0.
- `waited_to_later_candidate`: `square:hb3p_test:500001`, M1 t=80, L=20.
