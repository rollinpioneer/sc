# HB3-P Intermediate Exit Probe

This is one bounded, fixed-entry probe after the HB3-P exit diagnosis. It does not train a stopping model.

## Frozen Scope

- Entry: t=20.
- Methods: NONE, FIXED_L40, FIXED_L60, FIXED_L80.
- Independent roots: 40 (seeds 600000-600039).
- Maximum complete method records: 160; maximum environment steps: 64,000.
- New roots were sampled before labels and were not selected from HB3-P test roots.

## Results

| Method | System success | Autonomous completion | Mean utility | Mean helper steps |
|---|---:|---:|---:|---:|
| NONE | 1/40 (0.0250) | 1/40 (0.0250) | 0.025000 | 0.000 |
| FIXED_L40 | 5/40 (0.1250) | 5/40 (0.1250) | 0.100000 | 40.000 |
| FIXED_L60 | 10/40 (0.2500) | 10/40 (0.2500) | 0.212500 | 60.000 |
| FIXED_L80 | 11/40 (0.2750) | 11/40 (0.2750) | 0.225000 | 80.000 |

## Stop Versus Continue

- Prefix verification: 40/40.
- Short-exit autonomous successes: `{'FIXED_L40': 5, 'FIXED_L60': 10}`.
- Short-failure / L80-success cases: `{'FIXED_L40': 7, 'FIXED_L60': 5}`.
- Retrospective fixed-grid oracle mean utility delta versus L80: `0.145313`.
- The oracle is retrospective and is not a deployable selector or a formal success-rate upper bound.

## Decision

- **INTERMEDIATE_EXIT_OPPORTUNITY_CONFIRMED**.
- Training allowed by this gate: `True`.
- Training performed: `False`.
- Any model training would require a separate frozen protocol and is outside this probe.
