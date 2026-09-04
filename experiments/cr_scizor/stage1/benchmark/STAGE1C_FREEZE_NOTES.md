# Stage 1C Freeze Corrections

This freeze modifies labels and metadata only. It does not replay the simulator or regenerate any pair.

## No-effect/onset conflict

The Stage 1B metadata contained 15 records with `failure_type=no_effect` and a non-null `failure_onset`. A review MP4 was rendered from the stored clean and perturbed images for every affected pair before metadata was changed.

The fixed correction policy is:

- Promote to `recovery_success` only when the perturbed episode is successful and has both `recovery_start` and `recovery_end`.
- Otherwise label the record `failure_type=ambiguous` and `label_status=ambiguous`.

All 15 affected rows have no `recovery_end`, so all are ambiguous. The original Stage 1B metadata is retained in `pair_metadata.stage1b_original.jsonl`; every correction and its review-video path is recorded in `stage1c_label_corrections.jsonl`.

`benchmark_audit.json` remains the historical Stage 1B rollout audit. `stage1c_freeze_audit.json` is the authoritative audit for this label freeze.

## Responsibility semantics

`intervention_t` identifies the known action replacement. A responsibility point or region is positive only for `direct_failure`, `delayed_failure`, `recovery_failure`, and `recovery_success`. Consistent `no_effect` rows are negative controls; ambiguous rows are excluded from primary localization.

`no_effect_false_attribution_rate` is evaluated after a model produces responsibility scores: the fraction of consistent no-effect intervention steps assigned high responsibility. It is not a simulator-ground-truth value.

## Transition table

`transition_labels.parquet` contains one clean transition sequence per `(task, base_demo_id)` and one complete perturbed sequence per pair. Clean rows have a null `pair_id` and `variant=clean`; clean transitions are therefore not repeated once per perturbation.
