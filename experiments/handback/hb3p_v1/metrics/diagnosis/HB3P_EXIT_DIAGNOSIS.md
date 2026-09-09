# HB3-P Exit Diagnosis

This is a read-only diagnosis. It adds no training and no rollout.

## HB3-P Per-Root Attribution

- HB3-P logical records: 480 across 80 roots.
- M1 no-takeover roots: 18.
- M1 versus S_STAR outcome-difference roots: 12.
- M1 genuine rescue roots: 17.
- M1 interference roots: 1.
- `hb3p_per_root_diagnosis.csv` preserves actual query predictions and selected decisions as compact JSON.

## Fixed Entry: t=20

- Source: `hb2_val`; roots: 40.
- Shared-prefix verification: 40/40 roots passed.

| Branch | System success | Autonomous completion | Mean utility | Mean helper steps |
|---|---:|---:|---:|---:|
| none | 1/40 | 1/40 | 0.025000 | 0.000 |
| l5 | 2/40 | 2/40 | 0.046875 | 5.000 |
| l20 | 1/40 | 1/40 | 0.012500 | 20.000 |
| l80 | 9/40 | 9/40 | 0.175000 | 80.000 |

### Stop-versus-continue cases

- `l5` stop versus `l80` continue, autonomous: `7` short-failure/continue-success roots; `2` both-success roots.
- `l20` stop versus `l80` continue, autonomous: `8` short-failure/continue-success roots; `1` both-success roots.
- Observed exit-grid oracle minus fixed `l80` mean utility: `0.038672`.
- Oracle success-with-lower-cost roots: `2`.
- Oracle gains from both-failed cost-only cases: `31`.

## Route Decision

- Decision: **RUN_BOUNDED_INTERMEDIATE_EXIT_PROBE**.
- Training allowed: `False`.
- HB3-P status preserved: `STATE_ENTRY_SIGNAL_ONLY`.
- The oracle is retrospective and is not a deployable selector or a formal success-rate upper bound.
- The next bounded probe, if executed, must use a new frozen protocol with fixed t=20 and 40/60/80-step exits plus NONE.
