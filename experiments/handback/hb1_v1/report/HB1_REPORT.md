# HB1 Report

## 1. Fixed policy pair and task scope

Protocol: `hb1_protocol_v1`. Tasks: can, square. Horizon: 400 steps at the recorded runtime control frequency.

- can: base `bc_rnn_gmm` checkpoint `73df842484ac3ef858a242004a46617aad3c1f1b424718500c6ec6affdefd30f`; task status `NEED_BASE_POLICY`; no repair policy was trained or claimed.
- square: base `bc_rnn_gmm` checkpoint `e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6`; repair `sac_bounded_residual_adapter` checkpoint `340a376970559855779533bfb656814706d76960dbfb6d3e1ef310093021de29`; privileged repair input: `True`; repair training steps `250000`; qualification environment steps `28339`. resumed from `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb1_v1/repair/square/train/sac_125000.zip` after an external interruption; replay buffer resumed: `False`.

Old scorer/responsibility checkpoints listed during asset resolution were not used as action policies.

## 2. Data roles

BC source is demonstration supervision. `base_val` selects only the base checkpoint; `repair_train` and `repair_val` train and select only the repairer; `pilot` is limited to engineering checks; `probe` is the exploratory paired feasibility set.

## 3. Coverage and natural scenarios

Authoritative anchors: 118; complete paired anchors: 0 (0.0%); missing branches: 590.

Independent root groups by task: `{"square": 40}`. Anchor-time counts: `{"('square', 160)": 38, "('square', 20)": 40, "('square', 80)": 40}`.
No-help successful/complete paired anchors by task: `{"can": {"complete": 0, "successful": 0}, "square": {"complete": 0, "successful": 0}}`. Missing branches are not counted as failures.

## 4. Paired outcomes

### Can

| Branch | System success | Paired difference | Raw handoff | Genuine handoff | Helper-period success | Genuine rescue roots | Mean helper actions |
|---|---:|---:|---:|---:|---:|---:|---:|

### Square

| Branch | System success | Paired difference | Raw handoff | Genuine handoff | Helper-period success | Genuine rescue roots | Mean helper actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| l5 | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| l20 | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| l80 | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| full | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## 5. Helper completion versus handoff

`system_success`, `handoff_success_raw`, and `genuine_handoff_success` are retained separately. Genuine handoff requires no success under helper control, at least the configured autonomous delay, stable success, and zero repair calls after handoff.

- square: helper-after-handoff calls `0`; median first-success waits by finite branch `{}`; mean helper actions on no-help-success anchors `{}`.

## 6. Post-hoc shortest length and cost

Shortest diagnostic categories: `{"('square', 'incomplete')": 118}`. Conditional mean helper-action saving: `None` over `0` eligible anchors.

## 7. Task decisions

Overall status: **HOLD**.

- can: `NEED_BASE_POLICY`; valid roots 0, failed roots 0, genuinely rescued roots 0, control roots 0.
- square: `NEED_STRONGER_REPAIRER`; valid roots 0, failed roots 0, genuinely rescued roots 0, control roots 0.

## 8. Blocking reason classification

Any hold is reported by its concrete task-level status: engineering coverage, base/repair capability, natural-failure count, or lack of local handoff evidence for this fixed pair and grid.

## 9. Limitations

One base seed and one repair seed per task; privileged repair observations; fixed 5/20/80 duration grid; current Can/Square task scope; exploratory probe rather than a final blind test; no new visual handoff model was trained.

## 10. HB-2 handoff

Decision: `HOLD`. Any later visual model must be compared against no-help and the strongest fixed-duration baseline, split by root scenario, and evaluated on new scenes not relabeled as a blind reuse of HB1 probe.
