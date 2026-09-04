# Stage 2 Iteration 1 Report

## Fixed protocol

- Sampler positive/negative mass is `0.5 / 0.5`: **True**.
- Validation deletion budget is fixed at `9616`; test thresholds were not retuned.
- Canonical method: `responsibility_iter1_seed0` (full seed 0).

## Validation effect supervision

The maximum validation gate gap / effect balanced accuracy observed in the three full runs was: {"responsibility_iter1_seed0": {"max_validation_effect_balanced_accuracy": 0.7002840909090908, "max_validation_gate_gap": 0.3283874671775977}, "responsibility_iter1_seed1": {"max_validation_effect_balanced_accuracy": 0.676948051948052, "max_validation_gate_gap": 0.2309084238705935}, "responsibility_iter1_seed2": {"max_validation_effect_balanced_accuracy": 0.6153612012987013, "max_validation_gate_gap": 0.1617815777972027}}. Selected candidates were: {"action_only_seed_0": "best_effect_bacc", "full_seed_0": "best_localization", "full_seed_1": "best_gate_gap", "full_seed_2": "best_gate_gap"}.

## Test answers

1. Sampler mass is exactly balanced in all full runs: **True**.
2. Full-seed validation gate/effect metrics are recorded above; checkpoint selection remains validation-only.
3. Selected trackers: full seed 0=`best_localization`, full seed 1=`best_gate_gap`, full seed 2=`best_gate_gap`, action-only=`best_effect_bacc`.
4. Canonical test IoU/top-1/delay: `0.07014578593908122` / `0.5384615384615384` / `21.596153846153847`; strongest baseline values are `0.011985390641535707` / `0.038461538461538464` / `54.69230769230769`. Full seeds above baseline: IoU 3/3, top-1 3/3.
5. Canonical no-effect FAR is `0.7537688442211056`, fixed limit is `0.13552763819095476`; the FAR gate **fails**.
6. Canonical recovery retention is `0.45875542691751087`, baseline maximum is `0.5412445730824892`, tolerance floor is `0.4412445730824892`.
7. Canonical versus action-only test IoU/top-1/FAR: `0.07014578593908122` / `0.5384615384615384` / `0.7537688442211056` vs `0.08100250874703853` / `0.6153846153846154` / `0.7587939698492462`.
8. Final fixed decision: **NO_GO_SWITCH_DIRECTION**. Failed rules: `['canonical_no_effect_far_within_limit', 'canonical_beats_action_only']`.

## Test leaderboard

| method | transition_f1 | responsibility_region_iou | mean_abs_localization_delay | top1_within_1 | top5_hit | recovery_retention | innocent_downstream_retention | expert_retention | rare_retention | no_effect_false_attribution_rate | delete_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_responsibility_iter1 | 0.00921936 | 0.0701458 | 21.5962 | 0.538462 | 0.673077 | 0.458755 | 0.713377 | 0.748603 | 0.580292 | 0.753769 | 0.294142 |
| responsibility_iter1_seed0 | 0.00921936 | 0.0701458 | 21.5962 | 0.538462 | 0.673077 | 0.458755 | 0.713377 | 0.748603 | 0.580292 | 0.753769 | 0.294142 |
| responsibility_iter1_seed1 | 0.00801115 | 0.0564828 | 29.0192 | 0.346154 | 0.519231 | 0.659913 | 0.783612 | 0.656425 | 0.609084 | 0.768844 | 0.338736 |
| responsibility_iter1_seed2 | 0.00891963 | 0.0603707 | 40.3654 | 0.230769 | 0.557692 | 0.680174 | 0.771906 | 0.679236 | 0.611381 | 0.738693 | 0.32401 |
| action_only_iter1_seed0 | 0.00885515 | 0.0810025 | 14.2885 | 0.615385 | 0.75 | 0.53835 | 0.735177 | 0.712291 | 0.642741 | 0.758794 | 0.32638 |
| old_canonical_responsibility | 0.00939059 | 0.0629783 | 32.3269 | 0.384615 | 0.634615 | 0.643994 | 0.720417 | 0.694134 | 0.667072 | 0.859296 | 0.301372 |
| future_discount | 0.0030388 | 0.0119854 | 56.9038 | 0.0384615 | 0.0384615 | 0.376266 | 0.85724 | 0.786778 | 0.656123 | 0.175879 | 0.251978 |
| original_scizor | 0.00330974 | 0.0107681 | 59.9231 | 0 | 0.0192308 | 0.541245 | 0.837476 | 0.703445 | 0.507299 | 0.105528 | 0.32075 |
| uniform | 0.00290795 | 0.0113172 | 54.6923 | 0.0384615 | 0.0769231 | 0.315485 | 0.834252 | 0.769553 | 0.627602 | 0.140704 | 0.263386 |

All learned test metrics use operating points frozen on validation; no second Stage 2 tuning is performed.
