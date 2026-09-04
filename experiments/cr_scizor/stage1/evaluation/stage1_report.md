# Stage 1 Report

## Frozen inputs

- Benchmark pairs: 1280; failure-type counts: `{'ambiguous': 15, 'delayed_failure': 68, 'direct_failure': 7, 'no_effect': 1049, 'recovery_failure': 66, 'recovery_success': 75}`.
- Split seed: 20260831; group key: `['task', 'base_demo_id']`; metadata SHA-256: `f058b9f9811e7f1e19684256dc6ef2de1238bbbc370ceeff37c2eb7317dcc7f4`.
- Checkpoint: `ce741ef70abb1de14c0a2df352f7ab5ce95a2cc7efe44996890c6af9ca964d2d  /home/xushijie/work/cr_scizor/experiments/cr_scizor/stage1/baseline/frozen/model_10000.pth`.
- Repository record: `75051b3d45676a033533896da16a6a7abf8ac42e`.
- Operating points were selected only on `validation` at the original-score percentile `0.7` and applied unchanged to test.

## Test leaderboard

| method | transition_f1 | responsibility_region_iou | mean_abs_localization_delay | top1_within_1 | top5_hit | recovery_retention | innocent_downstream_retention | expert_retention | rare_retention | no_effect_false_attribution_rate | test_delete_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| future_discount | 0.0030388 | 0.0119854 | 56.9038 | 0.0384615 | 0.0384615 | 0.376266 | 0.85724 | 0.786778 | 0.656123 | 0.175879 | 0.251978 |
| original_scizor | 0.00330974 | 0.0107681 | 59.9231 | 0 | 0.0192308 | 0.541245 | 0.837476 | 0.703445 | 0.507299 | 0.105528 | 0.32075 |
| uniform | 0.00290795 | 0.0113172 | 54.6923 | 0.0384615 | 0.0769231 | 0.315485 | 0.834252 | 0.769553 | 0.627602 | 0.140704 | 0.263386 |

## Task results

| method | group | pair_count | transition_count | transition_f1 | responsibility_region_iou | recovery_retention | no_effect_false_attribution_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| future_discount | can | 125 | 15103 | 0.00553506 | 0.0137943 | 0.670695 | 0.09 |
| future_discount | square | 126 | 18646 | 0.00267666 | 0.0103105 | 0.105556 | 0.262626 |
| original_scizor | can | 125 | 15103 | 0.00461467 | 0.00935054 | 0.752266 | 0.12 |
| original_scizor | square | 126 | 18646 | 0.00298507 | 0.0120806 | 0.347222 | 0.0909091 |
| uniform | can | 125 | 15103 | 0.00460476 | 0.012391 | 0.561934 | 0.12 |
| uniform | square | 126 | 18646 | 0.00261849 | 0.010323 | 0.0888889 | 0.161616 |

## Outcome results

| method | group | pair_count | transition_count | transition_f1 | responsibility_region_iou | recovery_retention | no_effect_false_attribution_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| future_discount | final_failure | 33 | 4236 | 0.0120664 | 0.0172026 | 0.32247 | NA |
| future_discount | recovery_success | 19 | 3076 | 0.00312012 | 0.00292398 | 0.666667 | NA |
| original_scizor | final_failure | 33 | 4236 | 0.0123294 | 0.013896 | 0.463122 | NA |
| original_scizor | recovery_success | 19 | 3076 | 0.00533333 | 0.0053354 | 0.962963 | NA |
| uniform | final_failure | 33 | 4236 | 0.0117474 | 0.01607 | 0.255575 | NA |
| uniform | recovery_success | 19 | 3076 | 0.00255754 | 0.00306232 | 0.638889 | NA |

## Interpretation boundary

Original SCIZOR, Uniform split, and Future discount are evaluated on the same frozen transition labels. The task and outcome tables above are the primary subgroup summaries. Sparse perturbation subgroups are descriptive only; the JSON metrics retain their `pair_count` and `transition_count` values. The benchmark provides fixed labels, group split, chunk evidence, and baseline error measurements for the Stage 2 responsibility model. No test-set threshold tuning or split regeneration was performed.
