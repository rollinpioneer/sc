# Stage 2 Report

## Frozen protocol

Feature cache: 1360 demos / 176596 transitions. Chunk samples: 6547. Canonical method `responsibility_seed1` was selected on validation only; its test threshold was not retuned.

## Test leaderboard

| method | transition_f1 | responsibility_region_iou | mean_abs_localization_delay | top1_within_1 | top5_hit | recovery_retention | innocent_downstream_retention | expert_retention | rare_retention | no_effect_false_attribution_rate | test_delete_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_responsibility | 0.00939059 | 0.0629783 | 32.3269 | 0.384615 | 0.634615 | 0.643994 | 0.720417 | 0.694134 | 0.667072 | 0.859296 | 0.301372 |
| responsibility_seed0 | 0.00976319 | 0.0719857 | 52.4615 | 0.211538 | 0.480769 | 0.671491 | 0.750615 | 0.726723 | 0.611246 | 0.778894 | 0.283742 |
| responsibility_seed1 | 0.00939059 | 0.0629783 | 32.3269 | 0.384615 | 0.634615 | 0.643994 | 0.720417 | 0.694134 | 0.667072 | 0.859296 | 0.301372 |
| responsibility_seed2 | 0.00818797 | 0.0549167 | 39.1154 | 0.25 | 0.576923 | 0.599132 | 0.69497 | 0.66527 | 0.611652 | 0.773869 | 0.331388 |
| action_only_seed0 | 0.00965503 | 0.0593928 | 20.9808 | 0.538462 | 0.653846 | 0.657019 | 0.717194 | 0.6946 | 0.66572 | 0.834171 | 0.293075 |
| future_discount | 0.0030388 | 0.0119854 | 56.9038 | 0.0384615 | 0.0384615 | 0.376266 | 0.85724 | 0.786778 | 0.656123 | 0.175879 | 0.251978 |
| original_scizor | 0.00330974 | 0.0107681 | 59.9231 | 0 | 0.0192308 | 0.541245 | 0.837476 | 0.703445 | 0.507299 | 0.105528 | 0.32075 |
| uniform | 0.00290795 | 0.0113172 | 54.6923 | 0.0384615 | 0.0769231 | 0.315485 | 0.834252 | 0.769553 | 0.627602 | 0.140704 | 0.263386 |

## Direct answers

1. **Stability versus Stage 1 baselines:** all 3/3 full seeds exceed the strongest baseline IoU and all 3/3 exceed its top-1-within-1 score.
2. **Source of improvement:** canonical IoU/top-1/delay are `0.06297828382093663`, `0.38461538461538464`, and `32.32692307692308`, compared with the best baseline values `0.011985390641535707`, `0.038461538461538464`, and `54.69230769230769`. Threshold-level F1 is `0.009390589846424728`.
3. **Recovery:** canonical recovery retention is `0.6439942112879884` versus baseline maximum `0.5412445730824892`; it did not decline.
4. **No-effect control:** canonical no-effect false-attribution rate is `0.8592964824120602` versus baseline minimum `0.10552763819095477`; it exceeds the fixed tolerance, so this gate prevents GO_STAGE3.
5. **Task difference:** canonical Can top-1/IoU are `0.6` / `0.08197946772841253`; Square are `0.18518518518518517` / `0.045384595017718216`.
6. **Action-only diagnostic:** action-only top-1/IoU/FAR are `0.5384615384615384` / `0.05939284352013884` / `0.8341708542713567`, versus canonical `0.38461538461538464` / `0.06297828382093663` / `0.8592964824120602`.
7. **Next action:** the fixed automatic conclusion is **ITERATE_STAGE2_ONCE**. Per the protocol, this permits one targeted Stage 2 adjustment (effect-loss emphasis for no-effect/hard negatives *or* learning rate 1e-4), rather than proceeding to Stage 3 now.

The comparisons above all use validation-frozen operating points; the test set was not used to select seed or threshold.
