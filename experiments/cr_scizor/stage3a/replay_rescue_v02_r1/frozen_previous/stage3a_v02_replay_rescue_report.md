# Stage 3A v0.2 replay-locked rescue report

## Decision

`STOP_STAGE3A_V02_ORACLE_CEILING_FAILURE`.  Stage 3A-E continuation is `False`.

## Evidence

1. The previous stop remains scoped to v0.1 recovery: its legacy runtime is unavailable; it does not invalidate the independently replay-locked v0.2 path.
2. The current mimicgen runtime is frozen by `1a310e15962111b52503a7069b6261c5fffbc7603ea0f1493e1dc265d3b893a9`.
3. Pilot selection contains 10 clean-success base demos and produced 160 pairs.
4. Pilot paired-clean AUROC: 0.7399794450154162; pilot exact branch/reference/clean rates: {'branch_pre_state_equal_rate': 1.0, 'reference_exact_all_horizons_rate': 1.0, 'paired_clean_exact_all_horizons_rate': 1.0, 'finite_target_rate': 1.0}
5. Full v0.2 contains 1024 train/validation perturbed pairs; split counts: {'train': 768, 'validation': 256}.
6. Full paired-clean validation AUROC/AUPRC: 0.6393741455263557 / 0.24404906883071248.
7. Primary feasible validation AUROC/AUPRC: 0.6630715479264773 / 0.23165953676862808.
8. Best-of-4 feasible validation AUROC: 0.6513747531520583 (upper-bound diagnostic only).
9. Proposer transfer evaluated: False; full/action-only Top-5 recall: None / None.
10. Engineering pass: True; method pass: False; failed rules: ['oracle_method_ceiling'].

## Frozen gates

- Validation paired-clean AUROC >= 0.70
- Validation primary feasible replacement AUROC >= 0.70
- Exact twin-prefix/replay requirements recorded in `metrics/oracle_ceiling_v0.2_validation.json`
