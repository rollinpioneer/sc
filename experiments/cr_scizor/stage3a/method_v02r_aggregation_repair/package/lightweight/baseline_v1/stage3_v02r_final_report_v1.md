# SCIZOR Stage 3 v0.2-R 最终报告

> 本报告只汇总已生成的冻结证据；解释性案例不参与阈值、模型或最终决策。

## 1. v0.1 重放失败与 v0.2 工程救援结论

- v0.1 重放/数据血缘报告：/home/__compress_data/xushijie/work/cr_scizor/experiments/cr_scizor/stage3a/replay_rescue_v02/report/stage3a_v02_replay_rescue_report.md。
- v0.2 replay-locked rescue 报告：/home/__compress_data/xushijie/work/cr_scizor/experiments/cr_scizor/stage3a/replay_rescue_v02_r1/report/stage3a_v02r_report.md。
- 本阶段沿用 v0.2 的显式 pre/post state、当前 mimicgen runtime 和冻结的动作/扰动定义；不读取或重放 v0.1 test。
- v0.2-R confirmation decision：`RESUME_STAGE3A_E_ON_V02R`；engineering_pass=`true`；method_pass=`true`。

## 2. v0.2-R oracle confirmation 结果

- paired-clean confirmation AUROC：`0.905154`。
- primary feasible confirmation AUROC：`0.907112`。
- confirmation failed_rules：`[]`。
- confirmation 只作为工程和 oracle 前置证据，不进入 validation proposer、checkpoint 或 operating threshold 选择。

## 3. proposer 在 v0.2 的迁移表现

- full/action/union transfer summary：`metrics/proposer_transfer_v02.json`。
- train transfer：`{'action_top5': {'by_task': {'can': {'effective_pair_count': 37, 'mean_abs_localization_delay': 2.7027027027027026, 'mean_candidates_per_pair': 5.0, 'pair_count': 384, 'responsibility_region_recall': 0.972972972972973, 'top1_within_1': 0.972972972972973, 'top5_hit': 0.972972972972973}, 'square': {'effective_pair_count': 19, 'mean_abs_localization_delay': 2.0, 'mean_candidates_per_pair': 5.0, 'pair_count': 384, 'responsibility_region_recall': 0.8421052631578947, 'top1_within_1': 0.6842105263157895, 'top5_hit': 0.8421052631578947}}, 'overall': {'effective_pair_count': 56, 'mean_abs_localization_delay': 2.4642857142857144, 'mean_candidates_per_pair': 5.0, 'pair_count': 768, 'responsibility_region_recall': 0.9285714285714286, 'top1_within_1': 0.875, 'top5_hit': 0.9285714285714286}}, 'full_top5': {'by_task': {'can': {'effective_pair_count': 37, 'mean_abs_localization_delay': 2.135135135135135, 'mean_candidates_per_pair': 5.0, 'pair_count': 384, 'responsibility_region_recall': 0.972972972972973, 'top1_within_1': 0.9459459459459459, 'top5_hit': 0.972972972972973}, 'square': {'effective_pair_count': 19, 'mean_abs_localization_delay': 7.157894736842105, 'mean_candidates_per_pair': 5.0, 'pair_count': 384, 'responsibility_region_recall': 0.6842105263157895, 'top1_within_1': 0.3157894736842105, 'top5_hit': 0.6842105263157895}}, 'overall': {'effective_pair_count': 56, 'mean_abs_localization_delay': 3.8392857142857144, 'mean_candidates_per_pair': 5.0, 'pair_count': 768, 'responsibility_region_recall': 0.875, 'top1_within_1': 0.7321428571428571, 'top5_hit': 0.875}}, 'union_top5': {'by_task': {'can': {'effective_pair_count': 37, 'mean_abs_localization_delay': 2.135135135135135, 'mean_candidates_per_pair': 8.2109375, 'pair_count': 384, 'responsibility_region_recall': 0.972972972972973, 'top1_within_1': 0.972972972972973, 'top5_hit': 0.972972972972973}, 'square': {'effective_pair_count': 19, 'mean_abs_localization_delay': 2.0, 'mean_candidates_per_pair': 8.682291666666666, 'pair_count': 384, 'responsibility_region_recall': 0.8421052631578947, 'top1_within_1': 0.5789473684210527, 'top5_hit': 0.8421052631578947}}, 'overall': {'effective_pair_count': 56, 'mean_abs_localization_delay': 2.0892857142857144, 'mean_candidates_per_pair': 8.446614583333334, 'pair_count': 768, 'responsibility_region_recall': 0.9285714285714286, 'top1_within_1': 0.8392857142857143, 'top5_hit': 0.9285714285714286}}}`。
- validation transfer：`{'action_top5': {'by_task': {'can': {'effective_pair_count': 16, 'mean_abs_localization_delay': 6.5625, 'mean_candidates_per_pair': 5.0, 'pair_count': 128, 'responsibility_region_recall': 0.875, 'top1_within_1': 0.875, 'top5_hit': 0.875}, 'square': {'effective_pair_count': 13, 'mean_abs_localization_delay': 2.6923076923076925, 'mean_candidates_per_pair': 5.0, 'pair_count': 128, 'responsibility_region_recall': 0.9230769230769231, 'top1_within_1': 0.6923076923076923, 'top5_hit': 0.9230769230769231}}, 'overall': {'effective_pair_count': 29, 'mean_abs_localization_delay': 4.827586206896552, 'mean_candidates_per_pair': 5.0, 'pair_count': 256, 'responsibility_region_recall': 0.896551724137931, 'top1_within_1': 0.7931034482758621, 'top5_hit': 0.896551724137931}}, 'full_top5': {'by_task': {'can': {'effective_pair_count': 16, 'mean_abs_localization_delay': 13.0, 'mean_candidates_per_pair': 5.0, 'pair_count': 128, 'responsibility_region_recall': 0.6875, 'top1_within_1': 0.625, 'top5_hit': 0.6875}, 'square': {'effective_pair_count': 13, 'mean_abs_localization_delay': 31.23076923076923, 'mean_candidates_per_pair': 5.0, 'pair_count': 128, 'responsibility_region_recall': 0.23076923076923078, 'top1_within_1': 0.15384615384615385, 'top5_hit': 0.23076923076923078}}, 'overall': {'effective_pair_count': 29, 'mean_abs_localization_delay': 21.17241379310345, 'mean_candidates_per_pair': 5.0, 'pair_count': 256, 'responsibility_region_recall': 0.4827586206896552, 'top1_within_1': 0.41379310344827586, 'top5_hit': 0.4827586206896552}}, 'union_top5': {'by_task': {'can': {'effective_pair_count': 16, 'mean_abs_localization_delay': 5.0625, 'mean_candidates_per_pair': 8.515625, 'pair_count': 128, 'responsibility_region_recall': 0.9375, 'top1_within_1': 0.75, 'top5_hit': 0.9375}, 'square': {'effective_pair_count': 13, 'mean_abs_localization_delay': 0.5384615384615384, 'mean_candidates_per_pair': 8.984375, 'pair_count': 128, 'responsibility_region_recall': 0.9230769230769231, 'top1_within_1': 0.6153846153846154, 'top5_hit': 0.9230769230769231}}, 'overall': {'effective_pair_count': 29, 'mean_abs_localization_delay': 3.0344827586206895, 'mean_candidates_per_pair': 8.75, 'pair_count': 256, 'responsibility_region_recall': 0.9310344827586207, 'top1_within_1': 0.6896551724137931, 'top5_hit': 0.9310344827586207}}}`。
- proposer 仅产生冻结 Top-5 候选；长期反事实 oracle/verifier 才承担候选有效性判断。

## 4. verifier 三 seed 结果与 action-only 诊断

- verifier learning metrics：`metrics/validation_verifier_learning.json`。
- candidate replacement full AUROC/AUPRC：`0.848613` / `0.693273`。
- teacher-forced primary full AUROC/AUPRC：`0.742424` / `0.317226`。
- full vs action-only replacement AUPRC difference：`0.0325205`。
- full/action-only matched-recall no-effect FAR：`0.151515` / `0.171717`。
- teacher-forced primary 只使用 `is_teacher_forced=true`, `query_t=intervention_t`, `replacement_rank=0`。

## 5. validation protocol

- validation gate：`false`。
- selected proposer：`not available`。
- selected score：`fused_transition_score`；frozen threshold：`not available`。
- validation pipeline metrics：`metrics/validation_pipeline_metrics.json`。
- selected validation pair AUROC/AUPRC：`not available` / `not available`。
- validation failed_rules：`['no_valid_proposer_pipeline']`。
- 该 protocol 在 blind 读取前冻结；不得重新搜索 threshold、Top-k、模型结构、学习率或融合系数。

## 6. blind test 完整流程

- blind benchmark pipeline metrics：`metrics/blind_test_pipeline_metrics.json` (not generated)。
- blind teacher-forced metrics：`metrics/blind_test_teacher_forced_metrics.json` (not generated)。
- benchmark check pass：`not available`。
- blind selected proposer/threshold：`not available` / `not available`。
- blind pair count：`not available`；blind benchmark and full oracle artifacts are listed in `report/large_artifact_manifest.json`。
- blind 数据只在 protocol、checkpoint、normalizer、proposal calibration 全部冻结后读取；blind 结果不回流训练或选择。

## 7. no-effect、recovery、Can、Square 分组

- blind pair no-effect FAR / effective recall / Top-1 within ±1：`not available` / `not available` / `not available`。
- blind recovery false-attribution：`not available`。
- blind Can/Square pair AUROC：`not available` / `not available`。
- teacher-forced engineering branch/reference/finite rates：`not available` / `not available` / `not available`。
- teacher-forced oracle-positive AUROC：`not available`；primary effective-vs-no_effect AUROC：`not available`。

## 8. 最终决策

- **`SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`**。
- decision artifact：`metrics/stage3_final_decision.json`。
- failed rules：`['validation_gate']`。

## 9. 下一步（仅在 GO 时）

若最终决策为 `GO_STAGE4_POLICY_VALIDATION`，下一步只运行相同数据预算下的 policy 训练；Stage 4 必须使用冻结 method bundle，不得重新选择 threshold。

## 10. 失败时的切换方案

若最终决策不是 GO，则切换 `SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`，不进行第二轮大范围 verifier、Top-k、学习率或阈值实验。

## 交付索引

- leaderboard：`report/stage3_leaderboard.csv`。
- representative cases：`report/cases/case_index.csv` (not generated)。
- large-artifact manifest：`report/large_artifact_manifest.json`。
- method bundle：`report/stage3_method_bundle.json`。
