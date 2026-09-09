# HB1-R Capability Repair Report

## Scope

Protocol: `hb1_capability_repair_v1`. This report separates policy capability repair from the formal 0/5/20/80/full handback experiment.

## Preserved HB1 Result

The preserved HB1 decision was `HOLD`. It is historical evidence and was not overwritten.

## Square Capability Repair

- Direct privileged teacher status: `QUALIFIED`.
- Direct teacher checkpoint: `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb1_repair_v1/square/teacher/runs/handback_privileged_teacher_square/20260907151442_seed11/models/model_epoch_100.pth`.
- Direct teacher qualification: 18/72 (25.0%); rescued independent roots: `14`.
- Teacher-centered residual curriculum executed: `no`.
- Residual qualification status: `NOT_RUN`; rescued independent roots: `0`.
- Final Square rescued independent roots: `14`.
- Frozen Square pair: `True`.

The Square teacher and any teacher-centered residual repairer use privileged object-state input and are feasibility instruments, not deployable visual repair policies.

## Can Capability Diagnosis

- Low-dimensional privileged teacher status: `QUALIFIED`.
- Low-dimensional base_val result: 20/20 (100.0%).
- Successful runtime teacher rollouts: `170` of `200` attempted.
- Visual student status: `QUALIFIED`.
- Visual student base_val result: 20/20 (100.0%).
- Frozen Can base policy: `True`.

The Can low-dimensional teacher uses privileged object state. The visual student excludes object state and uses only the configured cameras and proprioception.

## Formal HB1 Paired Experiment

Tasks with completed formal paired anchors: `['square']`.
Coverage: `120` complete anchors of `120`; fraction `1.000`.

| Task | Branch | System success | Genuine rescue roots | Helper-completed anchors |
|---|---|---:|---:|---:|
| square | none | 5.0% | 0 | 0 |
| square | l5 | 2.5% | 2 | 0 |
| square | l20 | 7.5% | 7 | 0 |
| square | l80 | 18.3% | 13 | 4 |
| square | full | 25.0% | 0 | 30 |

System success, genuine handoff rescue, and helper-completed outcomes are reported separately. A full-helper success is not counted as genuine handoff evidence.

## Decision

Final HB1-R status: **READY_HB2_SINGLE_TASK**.

- Square: `READY`.
- Can: `READY`.
- Tasks cleared for HB-2: `['square']`.

HB-2 should start only for tasks meeting the formal independent-root, natural-failure, genuine-rescue, control-root, and complete-coverage gates.
