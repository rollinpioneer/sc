# HB2 Report Notes

- HB2 status remains `HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN`; ranking completion does not rerun selection.
- `local_feature_gain`, `history_gain`, and `paired_supervision_gain` are validation single-seed point-estimate comparisons, not independent test confirmation.
- Historical interference denominators differed by the anchors each method actually helped. HB3-P reports both all baseline-success roots and helped baseline-success roots.
- Historical files named `*.csv` may contain JSONL because the old generic writer keyed only on `.parquet`; HB3-P emits standards-compliant CSV.
- AUROC/AP below use tied-score threshold groups. `anchor_pooled` and `root_equal_weighted` are distinct estimands. Single-class outputs remain NA with an explicit reason.
