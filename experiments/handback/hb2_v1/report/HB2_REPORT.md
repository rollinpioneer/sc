# HB2 Report

## Material Passport

- Verification Status: VERIFIED
- Experiment: HB2-v1 Square
- Source commit: `b130fac48125757fcd3c573061fd51eb740c3626`
- Frozen code commit: `564c33e6a79e519c3ae7584762fb323cb53518e1`
- Result status: **HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN**

## Fixed Pair And Scope

- Base checkpoint SHA256: `e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6`
- Repair checkpoint SHA256: `7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62`
- Semantic pair ID: `a8ca96a4f5975b211053a3b6a1d34bc15a77b5ef2297cf6d8a7c97ac30fb1236`
- Scope: Square, fixed policy pair, fixed anchors, one intervention from 0/5/20/80 steps.
- The repair teacher uses privileged state. Online adaptive switching was not tested.

## Existing HB1-R Evidence

- Legacy rows reused as training/development data: 120
- Count clarification: `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb2_v1/metrics/hb1_count_clarification.json`
- Legacy full unique success roots: 18

## Data Roles And Independent Roots

- Frozen F examples: 357
- Frozen H examples: 1055
- Test valid roots / anchors: 40 / 118
- Full branch coverage: True
- Test finite-help rescuable roots: 16
- One allowed train/validation expansion used: False

## Labels And Inputs

- F uses one anchor history with none, l5, l20, l80, and full outcomes; missing engineering records are excluded, not relabeled as failures.
- H uses actual handoff histories and includes absolute time plus elapsed helper steps.
- Model tensors are restricted to two RGB cameras, 9-D proprioception, 7-D cached base actions, and time.
- DINOv2 checkpoint SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`

## Probability Evaluation

| Output | Brier | NLL | AUROC | AP | Positive roots |
|---|---:|---:|---:|---:|---:|
| p0 | 0.0392 | 0.1513 | NA | NA | 2 |
| p_sys_5 | 0.0249 | 0.0924 | NA | NA | 3 |
| p_sys_20 | 0.0599 | 0.2365 | NA | NA | 8 |
| p_sys_80 | 0.1290 | 0.4259 | NA | NA | 15 |
| p_genuine_5 | 0.0278 | 0.1030 | NA | NA | 3 |
| p_genuine_20 | 0.0634 | 0.2500 | NA | NA | 8 |
| p_genuine_80 | 0.1127 | 0.3835 | NA | NA | 14 |
| p_full | 0.1613 | 0.4976 | NA | NA | 19 |

## Fixed Matrix Ablation

| Model | Validation root-equal utility |
|---|---:|
| M0_time | 0.0831 |
| M1_proprio | 0.0705 |
| M2_global | 0.0495 |
| M3_local | 0.0719 |
| M4_paired | 0.0744 |
| M5_single | 0.0679 |

- Selected visual model: `M4_paired` (canonical seed 0).
- Visual / local / history / paired gain: unproven / supported / supported / supported.

## One-Shot Selection Value And Cost

| Method | Root utility | Autonomous completion | System success | Mean helper steps | Rescued roots |
|---|---:|---:|---:|---:|---:|
| fixed_l20 | 0.0625 | 0.0750 | 0.0750 | 20.0000 | 6 |
| fixed_l5 | 0.0302 | 0.0333 | 0.0333 | 5.0000 | 1 |
| fixed_l80 | 0.0792 | 0.1292 | 0.1458 | 80.0000 | 13 |
| fixed_none | 0.0500 | 0.0500 | 0.0500 | 0.0000 | 0 |
| model_M0_time | 0.1245 | 0.1458 | 0.1542 | 34.1667 | 12 |
| model_M1_proprio | 0.1051 | 0.1208 | 0.1375 | 25.1667 | 9 |
| model_M4_paired | 0.0871 | 0.1042 | 0.1125 | 27.3333 | 7 |
| model_M4_paired_seed1 | 0.0571 | 0.0583 | 0.0583 | 2.0000 | 1 |
| model_M4_paired_seed2 | 0.0958 | 0.1125 | 0.1208 | 26.6667 | 8 |
| model_M4_paired_system_objective | 0.0688 | 0.0958 | 0.1125 | 43.3333 | 9 |
| observed_finite_oracle | 0.1945 | 0.2000 | 0.2000 | 8.8750 | 16 |
| risk_fixed | 0.0792 | 0.1292 | 0.1458 | 80.0000 | 13 |

- Frozen B_star: `model_M0_time`; canonical 95% root-cluster CI: `[-0.08661588541666669, 0.003126302083333329]`.
- Canonical interference: `{'count': 1, 'denominator': 2, 'rate': 0.5}`.
- Canonical regret to observed finite oracle: 0.1074.

## Actual Handoff Prediction

- Exit model ready: **False**.
- Evidence: `{"beats_proprio_and_prior": false, "counts_ok": true, "frozen_train_prior_mean_brier": 0.07541599196753354, "proprio_mean_brier": 0.048933218836639716, "rule": {"criterion": "selected_visual_mean_brier_below_proprio_and_frozen_train_prior", "minimum_negative_roots_per_output": 5, "minimum_positive_roots_per_output": 5, "minimum_test_roots": 20}, "selected_visual_mean_brier": 0.058369936979013354}`
- Support is limited to observed fixed 5/20/80-step handoff exits.

## Runtime

- Runtime probe status: `runtime_probe_complete`.
- Pilot roots checked: 4.
- Repair calls after handoff: 0.
- Prefix replay cost is reported in runtime_probe.json and is not folded into formal test metrics.

## Development Cases

- effective_short_help: `F:square:hb2_val:301010:20`, root `square:hb2_val:301010`, t=20.
- help_ineffective: `F:square:hb2_val:301000:20`, root `square:hb2_val:301000`, t=20.
- baseline_success_interfered: `F:square:hb2_val:301006:20`, root `square:hb2_val:301006`, t=20.
- helper_completed: `F:square:hb2_val:301006:80`, root `square:hb2_val:301006`, t=80.
- finite_nonmonotonic: `F:square:hb2_val:301017:20`, root `square:hb2_val:301017`, t=20.

## Limitations

- Each rollout is one binary outcome under the frozen policy and random mechanism, not a per-state safety probability.
- The test is same-task and same-distribution; cross-task, distribution-shift, and real-robot claims are unsupported.
- Full repair is a capability/cost reference and is excluded from the deployable selector.
- Multiple anchors and model seeds are not treated as additional independent roots.

## Layered Decision

- Overall: **HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN**
- Counterfactual start signal: supported
- Visual gain: unproven
- Exit model ready: False

## HB3 Handoff

- Allowed scope: `fixed_pair_fixed_anchor_single_intervention`.
- A full dynamic-query or repeated-takeover controller remains out of scope.
- Continue only within the capability implied by the layered status above.
