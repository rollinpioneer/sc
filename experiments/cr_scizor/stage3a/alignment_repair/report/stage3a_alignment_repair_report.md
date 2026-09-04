# Stage 3A Alignment Repair Report

## Decision

`SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`

Stage 2 decision preserved: `NO_GO_SWITCH_DIRECTION`.

## Scope and frozen inputs

The existing validation plan and benchmark/source HDF5 files were used. Stage 1/2 benchmark, labels, split, checkpoints, proposal settings, and target formulas were not modified. No simulator rollouts were regenerated.

## Alignment results

The untouched provisional alignment failed with next-state L2 checks `6.8102827072` and `0.1964062005` (median `3.5033444539`).

The repair diagnostic tested 8 fixed queries across Can/Square and effective/no-effect cells. The required thresholds were pre-state/next-state median `< 1e-4` and p95 `< 1e-3`.

| Replay + layout | Pre median | Pre p95 | Next median | Next p95 | Passed |
|---|---:|---:|---:|---:|---|
| direct_state_reset + pre_action | 1.11e-16 | 3.17e-16 | 2.2839 | 4.5247 | false |
| direct_state_reset + post_action | 2.39e-16 | 3.05e-16 | 2.0252 | 2.3489 | false |
| prefix_replay + pre_action | 1.6073 | 4.3789 | 1.1646 | 1.8069 | false |
| prefix_replay + post_action | 0.2392 | 1.5060 | 0.1646 | 1.5099 | false |

All four combinations failed. Model-file-aware replay was executed using the source XML, with runtime-only texture conversion for MuJoCo 2.3.2 and bookkeeping-only compatibility for old geom/site names. The XML loads, but its transitions do not reproduce the frozen benchmark transitions.

Because no combination passed, there is no final frozen `state_layout` or `oracle_replay_mode`; both are `not_selected`.

## Oracle and continuation status

No aligned action library, aligned oracle, targets, verifier, validation selection, test read, or blind holdout was generated. Repaired train/validation reference replay rates and repaired AUROC are `not_run`.

The pre-verifier gate is `not_run` because alignment is a mandatory prerequisite.

For comparison, the untouched provisional validation oracle reported paired-clean upper-bound AUROC `0.6097`, teacher-forced primary AUROC `0.6174`, and primary-feasible replacement AUROC `0.5650`. These values remain `MISALIGNED_PROVISIONAL` and were not used for continuation.

## Failed rules

- `simulator_transition_alignment_failed`

Artifacts:

- `diagnostics/state_layout_diagnostics.json`
- `config/transition_alignment.json`
- `metrics/go_no_go.json`
- `logs/S3A-DR1-diagnose-state-layout.log`
