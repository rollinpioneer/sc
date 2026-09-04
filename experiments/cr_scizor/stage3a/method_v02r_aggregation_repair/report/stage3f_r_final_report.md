# SCIZOR Stage 3F-R Aggregation Repair Report

> This report treats validation strictly as the development set. No blind benchmark was read or generated after the fixed development gate failed.

## Decision

- Final decision: `SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`
- Stage3 v1 decision preserved: `True`
- Aggregation repair attempted: `True`
- Blind generated: `False`
- Failed rules: `development_aggregation_gate`

## Complete Teacher-Forced Diagnostic

- Pairs: `256`; effective pairs: `29`
- Primary rows retained with original `state_in_domain=false` engineering flag: `20` (not rewritten)
- Full AUROC: `0.752089`; AUPRC: `0.245387`
- Full score Spearman: `0.789892`

## Fixed Development Matrix

| method | source | AUROC | AUPRC | FAR | effective recall | Can AUROC | Square AUROC | eligible |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| action_cf_only | action_top5 | 0.607778 | 0.181946 | 0.14977973568281938 | 0.3103448275862069 | 0.6841517857142857 | 0.46555183946488293 | False |
| action_current_fused | action_top5 | 0.648337 | 0.209509 | 0.15859030837004406 | 0.4827586206896552 | 0.6796875 | 0.5899665551839465 | False |
| action_defect_contrast | action_top5 | 0.525900 | 0.134165 | 0.19383259911894274 | 0.3103448275862069 | 0.4810267857142857 | 0.6294314381270903 | False |
| action_defect_gated | action_top5 | 0.550661 | 0.198115 | 0.14537444933920704 | 0.3103448275862069 | 0.6512276785714286 | 0.37725752508361204 | False |
| action_raw | action_top5 | 0.615069 | 0.191038 | 0.18502202643171806 | 0.3103448275862069 | 0.7751116071428571 | 0.4214046822742475 | False |
| full_raw | full_top5 | 0.661552 | 0.191173 | 0.13215859030837004 | 0.41379310344827586 | 0.6545758928571429 | 0.6668896321070235 | False |
| union_defect_contrast | union_top5 | 0.530153 | 0.121445 | 0.1894273127753304 | 0.2413793103448276 | 0.5407366071428571 | 0.5618729096989966 | False |
| union_raw | union_top5 | 0.639982 | 0.172042 | 0.13215859030837004 | 0.3448275862068966 | 0.7025669642857143 | 0.5411371237458193 | False |

## Scope And Stop Rule

The candidate replacement baseline remained above the verifier gate, and the complete teacher-forced full AUROC met the required threshold. The fixed aggregation methods did not satisfy the development gate, so the protocol stops before blind generation. No formula, threshold, source, checkpoint, proposer, oracle, label, or benchmark was changed after this failure.

## Preserved v1 Evidence

The original Stage 3 v1 report is included under `baseline_v1/`; its final decision remains `SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`.

## Reproducibility

Development input tables and their derived score tables remain outside this lightweight package because they are parquet artifacts. The package contains the deterministic source code, protocol configuration, summaries, metrics, and baseline evidence needed to audit the decision.
