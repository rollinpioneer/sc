# SCIZOR Stage 3F-R：反事实责任分数聚合修复
## Agent 详细操作命令文档

> 适用起点：`Stage 3 v0.2-R` 的 verifier、validation candidate oracle 和 proposer transfer 已完成，但完整 validation pipeline 因 `no_valid_proposer_pipeline` 未通过。
>
> 当前正式结果必须保留：`stage3_v02r_method_v1 = SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`。
>
> 本文档只允许进行一次范围固定的“分数聚合修复”。不重训 verifier，不重跑 simulator oracle，不修改长期反事实标签，不调整 Stage 2 proposer，不提前生成或读取 blind benchmark。

---

# 0. 总体任务、已有证据与本轮边界

## 0.1 总体上要干什么

当前三个基础模块已经工作：

1. v0.2-R simulator 反事实分支可确定性运行；
2. learned verifier 能够近似 candidate-level simulator target；
3. `action_top5` 和 `union_top5` 通常能将真实责任区域包含在候选集合中。

当前失败集中在最后一步：

> 如何将“每个候选 transition 的 4 个替代动作分数”和“每条轨迹的多个候选 transition 分数”聚合成一个稳定的责任分数。

本轮只做以下修复：

```text
补全 train split 的冻结 verifier 预测
        ↓
用 train no-effect 数据估计正常改善背景
        ↓
完整评估 raw proposer / counterfactual-only / 当前融合
        ↓
增加 replacement 内与 pair 内的对比聚合
        ↓
增加原轨迹局部 outcome deficit 软门控
        ↓
只在当前 validation（后文称 development）上选择一个固定公式
        ↓
若 development 通过，冻结公式和阈值后才生成全新 blind benchmark
        ↓
blind 只运行一次，决定是否进入 Stage 4 policy 实验
```

## 0.2 当前结果事实

当前已有结果：

```text
candidate replacement verifier：
  full AUROC  = 0.848613
  full AUPRC  = 0.693273
  Spearman    = 0.677837

action_top5 proposer transfer：
  responsibility-region recall = 0.896552
  Top-1 within ±1              = 0.793103

当前 action_top5 完整流程：
  pair AUROC       = 0.645517
  pair AUPRC       = 0.209556
  no-effect FAR    = 0.160000
  effective recall = 0.482759
  Can AUROC        = 0.675000
  Square AUROC     = 0.589967
```

当前流程大部分 operating-point 指标已经达到要求，但总体 AUROC 和 Square AUROC 未达到冻结门槛。现有代码同时计算了 raw proposer、counterfactual-only 和 fused 分数，却只用 `fused_pair_score` 做最终 gate；另外连续两次 `max` 会放大 no-effect pair 中偶然出现的高分。

## 0.3 本轮最终只允许三个状态

Development 修复完成后：

```text
RESUME_STAGE3_BLIND_AFTER_AGGREGATION_REPAIR
SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR
```

若 development 通过并完成 blind：

```text
GO_STAGE4_POLICY_VALIDATION
SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR
```

本轮结束后不再允许：

```text
ITERATE_STAGE3F_AGAIN
SECOND_AGGREGATION_REPAIR
RETRAIN_VERIFIER_AFTER_BLIND
```

## 0.4 本轮明确不做什么

不要执行：

```text
重新训练 full verifier 或 action-only verifier
修改 verifier 网络、学习率、loss 或 checkpoint
重新运行 train / validation simulator oracle
修改 100 帧长期结果定义
修改 dense / stage / success 权重
修改 positive threshold=0.5
重新训练 Stage 2 responsibility network
重新选择 Stage 2 checkpoint
改变 Top-5 候选规模
改变每个 query 的 4 个 replacement
重新构建动作库或 support threshold
重新调扰动强度或干预位置
读取尚未生成的 blind 结果后再修改公式
运行全仓库测试、lint、依赖审计或无关性能检查
```

本轮只保留直接服务于推进的最小检查：

```text
输入文件存在
train / development 行数和 ID 唯一性
完整 teacher-forced 诊断为 256 pairs / 29 effective
每个方法 development pair 表覆盖全部 256 pairs
固定公式矩阵全部生成
冻结 gate 和 blind 一次性执行
```

---

# 1. 统一路径、环境和目录

每个新 shell 先执行本节。不要依赖旧 shell 的变量。

```bash
set -euo pipefail

export PROJECT_ROOT=/home/__compress_data/xushijie/work/cr_scizor
if [ ! -d "$PROJECT_ROOT/SCIZOR" ]; then
  export PROJECT_ROOT=/home/xushijie/work/cr_scizor
fi

export SCIZOR_ROOT="$(readlink -f "$PROJECT_ROOT/SCIZOR")"
export EXP_ROOT="$PROJECT_ROOT/experiments/cr_scizor"
export STAGE1_ROOT="$EXP_ROOT/stage1"
export STAGE2_BASE_ROOT="$EXP_ROOT/stage2"
export STAGE2_ITER_ROOT="$EXP_ROOT/stage2_iter1"
export STAGE3A_ROOT="$EXP_ROOT/stage3a"
export V02_ROOT="$STAGE3A_ROOT/replay_rescue_v02"
export V02R_ROOT="$STAGE3A_ROOT/replay_rescue_v02_r1"
export STAGE3_V1_ROOT="$STAGE3A_ROOT/method_v02r"
export STAGE3_AGG_ROOT="$STAGE3A_ROOT/method_v02r_aggregation_repair"

export BENCHMARK_V02="$V02_ROOT/benchmark/benchmark_v0.2_train_val.hdf5"
export META_V02="$V02_ROOT/metadata/pair_metadata_v0.2_train_val.jsonl"
export SPLIT_V02="$V02_ROOT/metadata/split_manifest_v0.2.json"
export BASES_V02="$V02_ROOT/metadata/base_demos_v0.2_train_val.json"
export ACTION_LIBRARY_V02="$V02_ROOT/library"

export SCORE_SPEC_V02R="$V02R_ROOT/config/oracle_score_spec_v02r.json"
export ORACLE_NORMALIZER_V02R="$V02R_ROOT/config/train_normalizer_v02r.json"
export TEACHER_FORCED_LONG="$V02R_ROOT/oracle/dev/feasible_long.jsonl"
export TEACHER_PLANS="$V02_ROOT/oracle/teacher_forced_plans.parquet"
export CONFIRMATION_DECISION="$V02R_ROOT/metrics/v02r_final_decision.json"
export CONFIRMATION_BASES="$V02R_ROOT/metadata/base_demos_v0.2_confirmation.json"

export FULL_PROPOSER_CKPT="$STAGE2_ITER_ROOT/runs/full_seed_0/selected.pt"
export ACTION_PROPOSER_CKPT="$STAGE2_ITER_ROOT/runs/action_only_seed_0/selected.pt"
export STAGE2_ITER_CONFIG="$SCIZOR_ROOT/stage2/configs/responsibility_iter1.json"
export STAGE2_TRAIN_NORMALIZER="$STAGE2_BASE_ROOT/features/normalizer.npz"
export FROZEN_SCIZOR_DIR="$STAGE1_ROOT/baseline/frozen"

export LABELS_V02="$STAGE3_V1_ROOT/labels/transition_labels_v02.parquet"
export FEATURE_INDEX_V02="$STAGE3_V1_ROOT/features/feature_index.parquet"
export VERIFIER_NORMALIZER="$STAGE3_V1_ROOT/features/verifier_normalizer.npz"
export CHUNK_EVIDENCE_V02="$STAGE3_V1_ROOT/evidence/chunk_evidence_v02.parquet"
export PROPOSALS_TRAIN="$STAGE3_V1_ROOT/proposals/proposal_candidates_train.parquet"
export PROPOSALS_DEV="$STAGE3_V1_ROOT/proposals/proposal_candidates_validation.parquet"
export SAMPLES_TRAIN="$STAGE3_V1_ROOT/oracle/datasets/verifier_samples_train.parquet"
export SAMPLES_DEV="$STAGE3_V1_ROOT/oracle/datasets/verifier_samples_validation.parquet"
export ENSEMBLE_DEV="$STAGE3_V1_ROOT/predictions/ensemble/validation_ensemble.parquet"
export ACTION_ONLY_DEV="$STAGE3_V1_ROOT/predictions/validation/action_only_seed_0.parquet"
export V1_PAIR_SCORES="$STAGE3_V1_ROOT/predictions/validation/pair_scores.parquet"
export V1_TRANSITION_SCORES="$STAGE3_V1_ROOT/predictions/validation/transition_scores.parquet"
export V1_REPLACEMENT_SCORES="$STAGE3_V1_ROOT/predictions/validation/replacement_scores.parquet"
export V1_METRICS="$STAGE3_V1_ROOT/metrics/validation_pipeline_metrics.json"
export V1_PROTOCOL="$STAGE3_V1_ROOT/metrics/validation_frozen_protocol.json"
export V1_DECISION="$STAGE3_V1_ROOT/metrics/stage3_final_decision.json"
export STAGE3_CONFIG="$SCIZOR_ROOT/stage3a/method_v02r/config_v02r.json"

export MIMICGEN_PYTHON=/home/xushijie/.conda/envs/mimicgen/bin/python
if [ ! -x "$MIMICGEN_PYTHON" ]; then
  export MIMICGEN_PYTHON="$(command -v python)"
fi

for candidate in \
  /home/xushijie/.conda/envs/scizor-curation/bin/python \
  /home/xushijie/.conda/envs/curation/bin/python; do
  if [ -x "$candidate" ]; then
    export CURATION_PYTHON="$candidate"
    break
  fi
done
: "${CURATION_PYTHON:?cannot locate curation python}"

export PYTHONPATH="$SCIZOR_ROOT/robomimic:$SCIZOR_ROOT"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export USE_TF=0
export TRANSFORMERS_NO_TF=1

mkdir -p \
  "$STAGE3_AGG_ROOT/config" \
  "$STAGE3_AGG_ROOT/baseline_v1" \
  "$STAGE3_AGG_ROOT/predictions/train" \
  "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete" \
  "$STAGE3_AGG_ROOT/background" \
  "$STAGE3_AGG_ROOT/development" \
  "$STAGE3_AGG_ROOT/metrics" \
  "$STAGE3_AGG_ROOT/blind_test" \
  "$STAGE3_AGG_ROOT/logs" \
  "$STAGE3_AGG_ROOT/report" \
  "$STAGE3_AGG_ROOT/package"
```

只检查本轮直接需要的输入：

```bash
for file in \
  "$LABELS_V02" \
  "$FEATURE_INDEX_V02" \
  "$VERIFIER_NORMALIZER" \
  "$CHUNK_EVIDENCE_V02" \
  "$PROPOSALS_TRAIN" \
  "$PROPOSALS_DEV" \
  "$SAMPLES_TRAIN" \
  "$SAMPLES_DEV" \
  "$ENSEMBLE_DEV" \
  "$ACTION_ONLY_DEV" \
  "$V1_PAIR_SCORES" \
  "$V1_TRANSITION_SCORES" \
  "$V1_REPLACEMENT_SCORES" \
  "$V1_METRICS" \
  "$V1_PROTOCOL" \
  "$V1_DECISION" \
  "$TEACHER_FORCED_LONG" \
  "$TEACHER_PLANS" \
  "$META_V02" \
  "$STAGE3_CONFIG" \
  "$STAGE3_V1_ROOT/runs/full_seed_0/best.pt" \
  "$STAGE3_V1_ROOT/runs/full_seed_1/best.pt" \
  "$STAGE3_V1_ROOT/runs/full_seed_2/best.pt" \
  "$STAGE3_V1_ROOT/runs/action_only_seed_0/best.pt"; do
  test -s "$file"
done
```

确认旧结果仍是 validation gate 失败：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
v1=json.load(open(os.environ['V1_DECISION']))
assert v1['decision']=='SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR', v1
assert v1['failed_rules']==['validation_gate'], v1
print('v1 result frozen:', v1['decision'])
PY
```

---

# 小阶段 3F-R-A：冻结方法 v1、创建修复分支和唯一配置

## 3F-R-A.1 本小阶段总体上要干什么

本小阶段只做三件事：

1. 将现有 Stage 3 方法 v1 的结果复制为只读基线；
2. 建立独立代码包和独立实验目录；
3. 在读取新的聚合结果前，写死本轮允许比较的公式和门槛。

## 3F-R-A.2 建立 Git 分支

当前 Stage 3 v1 最终代码提交为：

```text
c72ff2e087c6dbef2a5fe9e44509bc14b44c2ec8
```

执行：

```bash
set -euo pipefail
cd "$SCIZOR_ROOT"

git rev-parse --verify c72ff2e087c6dbef2a5fe9e44509bc14b44c2ec8 >/dev/null

git checkout -b exp/stage3f-r-aggregation-repair \
  c72ff2e087c6dbef2a5fe9e44509bc14b44c2ec8

mkdir -p \
  stage3a/method_v02r_aggregation_repair/evaluation \
  stage3a/method_v02r_aggregation_repair/decision \
  stage3a/method_v02r_aggregation_repair/benchmark

touch \
  stage3a/method_v02r_aggregation_repair/__init__.py \
  stage3a/method_v02r_aggregation_repair/evaluation/__init__.py \
  stage3a/method_v02r_aggregation_repair/decision/__init__.py \
  stage3a/method_v02r_aggregation_repair/benchmark/__init__.py
```

## 3F-R-A.3 冻结旧结果副本

只复制小型 JSON / CSV / Markdown，不复制 checkpoint、Parquet、HDF5 或 NPZ：

```bash
cp "$V1_METRICS" \
  "$STAGE3_AGG_ROOT/baseline_v1/validation_pipeline_metrics_v1.json"
cp "$V1_PROTOCOL" \
  "$STAGE3_AGG_ROOT/baseline_v1/validation_frozen_protocol_v1.json"
cp "$V1_DECISION" \
  "$STAGE3_AGG_ROOT/baseline_v1/stage3_final_decision_v1.json"
cp "$STAGE3_V1_ROOT/metrics/validation_pipeline_leaderboard.csv" \
  "$STAGE3_AGG_ROOT/baseline_v1/validation_pipeline_leaderboard_v1.csv"
cp "$STAGE3_V1_ROOT/report/stage3_v02r_final_report.md" \
  "$STAGE3_AGG_ROOT/baseline_v1/stage3_v02r_final_report_v1.md"
```

写入状态说明：

```bash
cat > "$STAGE3_AGG_ROOT/baseline_v1/STATUS.json" <<'JSON'
{
  "schema": "stage3f_r_v1_status_v1",
  "method_version": "stage3_v02r_method_v1",
  "decision": "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR",
  "reason": "validation gate failed under fused_pair_score-only protocol",
  "status": "FROZEN_BASELINE_NOT_OVERWRITTEN",
  "aggregation_repair_allowed": true,
  "blind_generated": false
}
JSON
```

## 3F-R-A.4 写入唯一修复配置

创建：

```text
stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json
```

执行：

```bash
cat > "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" <<'JSON'
{
  "version": "stage3f-r-aggregation-repair-v1",
  "development_split": "validation",
  "development_pair_count": 256,
  "development_effective_pair_count": 29,
  "replacement_background_quantile": 0.80,
  "outcome_deficit": {
    "source": "max_V_c_over_covering_chunks",
    "clip_min": 0.0,
    "clip_max": 1.0,
    "chunk_interval": "start_t_inclusive_end_t_exclusive"
  },
  "candidate_sources": {
    "action_top5": {
      "membership_column": "in_action_top5",
      "raw_score_column": "raw_action_score",
      "rank_column": "action_rank"
    },
    "union_top5": {
      "membership_column": "in_union_top5",
      "raw_score_column": "raw_union_score",
      "rank_column": "union_rank"
    },
    "full_top5_diagnostic": {
      "membership_column": "in_full_top5",
      "raw_score_column": "raw_full_score",
      "rank_column": "full_rank"
    }
  },
  "diagnostic_methods": [
    "action_raw",
    "union_raw",
    "full_raw"
  ],
  "eligible_methods": [
    "action_cf_only",
    "action_current_fused",
    "action_defect_gated",
    "action_defect_contrast",
    "union_defect_contrast"
  ],
  "score_formulas": {
    "replacement_cf": "replacement_cf_score",
    "transition_cf_max": "max_k replacement_cf",
    "transition_cf_contrast": "max_k replacement_cf - median_k replacement_cf",
    "train_background_max": "q80 transition_cf_max on train no_effect, grouped by task and source",
    "train_background_contrast": "q80 transition_cf_contrast on train no_effect, grouped by task and source",
    "local_deficit": "clip(max covering V_c,0,1)",
    "action_cf_only": "pair=max_j transition_cf_max",
    "action_current_fused": "pair=max_j transition_cf_max*proposal_rank_weight",
    "action_defect_gated": "pair=max_j local_deficit*relu(transition_cf_max-background_max)",
    "defect_contrast_transition": "local_deficit*relu(transition_cf_contrast-background_contrast)",
    "defect_contrast_pair": "max_j defect_contrast_transition - median_j defect_contrast_transition"
  },
  "pair_universe": {
    "include_all_perturbed_pairs": true,
    "missing_pipeline_score": 0.0,
    "missing_predicted_t": -1
  },
  "complete_teacher_forced": {
    "expected_pairs": 256,
    "expected_effective_pairs": 29,
    "required_full_auroc": 0.70
  },
  "development_gate": {
    "no_effect_far_cap": 0.20,
    "min_pair_auroc": 0.70,
    "min_effective_recall": 0.40,
    "min_proposal_region_recall": 0.50,
    "min_task_pair_auroc": 0.60
  },
  "method_tiebreak": [
    "AUPRC within 0.01 of best",
    "lowest no_effect FAR within 0.02",
    "highest effective recall",
    "highest top1_within_1",
    "fewest mean candidates",
    "lexicographic method_id"
  ],
  "blind_gate": {
    "max_no_effect_far": 0.25,
    "min_pair_auroc": 0.70,
    "min_effective_recall": 0.35,
    "min_top1_within_1": 0.25,
    "min_task_pair_auroc": 0.60,
    "min_auprc_prevalence_multiple": 2.0
  },
  "second_repair_allowed": false
}
JSON
```

## 3F-R-A.5 写入环境文件

```bash
cat > "$STAGE3_AGG_ROOT/config/stage3f_r.env" <<EOF_ENV
export PROJECT_ROOT="$PROJECT_ROOT"
export SCIZOR_ROOT="$SCIZOR_ROOT"
export EXP_ROOT="$EXP_ROOT"
export STAGE1_ROOT="$STAGE1_ROOT"
export STAGE2_BASE_ROOT="$STAGE2_BASE_ROOT"
export STAGE2_ITER_ROOT="$STAGE2_ITER_ROOT"
export STAGE3A_ROOT="$STAGE3A_ROOT"
export V02_ROOT="$V02_ROOT"
export V02R_ROOT="$V02R_ROOT"
export STAGE3_V1_ROOT="$STAGE3_V1_ROOT"
export STAGE3_AGG_ROOT="$STAGE3_AGG_ROOT"
export LABELS_V02="$LABELS_V02"
export FEATURE_INDEX_V02="$FEATURE_INDEX_V02"
export VERIFIER_NORMALIZER="$VERIFIER_NORMALIZER"
export CHUNK_EVIDENCE_V02="$CHUNK_EVIDENCE_V02"
export PROPOSALS_TRAIN="$PROPOSALS_TRAIN"
export PROPOSALS_DEV="$PROPOSALS_DEV"
export SAMPLES_TRAIN="$SAMPLES_TRAIN"
export SAMPLES_DEV="$SAMPLES_DEV"
export ENSEMBLE_DEV="$ENSEMBLE_DEV"
export ACTION_ONLY_DEV="$ACTION_ONLY_DEV"
export TEACHER_FORCED_LONG="$TEACHER_FORCED_LONG"
export TEACHER_PLANS="$TEACHER_PLANS"
export META_V02="$META_V02"
export BENCHMARK_V02="$BENCHMARK_V02"
export BASES_V02="$BASES_V02"
export CONFIRMATION_BASES="$CONFIRMATION_BASES"
export ACTION_LIBRARY_V02="$ACTION_LIBRARY_V02"
export STAGE3_CONFIG="$STAGE3_CONFIG"
export MIMICGEN_PYTHON="$MIMICGEN_PYTHON"
export CURATION_PYTHON="$CURATION_PYTHON"
export PYTHONPATH="$PYTHONPATH"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export USE_TF=0
export TRANSFORMERS_NO_TF=1
EOF_ENV
```

## 3F-R-A.6 冻结输入哈希

```bash
sha256sum \
  "$V1_METRICS" \
  "$V1_PROTOCOL" \
  "$V1_DECISION" \
  "$LABELS_V02" \
  "$FEATURE_INDEX_V02" \
  "$VERIFIER_NORMALIZER" \
  "$CHUNK_EVIDENCE_V02" \
  "$PROPOSALS_TRAIN" \
  "$PROPOSALS_DEV" \
  "$SAMPLES_TRAIN" \
  "$SAMPLES_DEV" \
  "$ENSEMBLE_DEV" \
  "$ACTION_ONLY_DEV" \
  "$TEACHER_FORCED_LONG" \
  "$TEACHER_PLANS" \
  "$STAGE3_V1_ROOT/runs/full_seed_0/best.pt" \
  "$STAGE3_V1_ROOT/runs/full_seed_1/best.pt" \
  "$STAGE3_V1_ROOT/runs/full_seed_2/best.pt" \
  "$STAGE3_V1_ROOT/runs/action_only_seed_0/best.pt" \
  "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  > "$STAGE3_AGG_ROOT/config/frozen_inputs.sha256"
```

### 3F-R-A 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/baseline_v1/STATUS.json"
test -s "$STAGE3_AGG_ROOT/config/stage3f_r.env"
test -s "$STAGE3_AGG_ROOT/config/frozen_inputs.sha256"
test -s "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json"
```

提交配置：

```bash
cd "$SCIZOR_ROOT"
git add stage3a/method_v02r_aggregation_repair

git -c user.name="Experiment Agent" -c user.email="agent@local" \
  commit -m "stage3F-R-A: freeze bounded aggregation repair protocol"
```

---

# 小阶段 3F-R-B：补全 train 预测和完整 teacher-forced 诊断

## 3F-R-B.1 本小阶段总体上要干什么

本小阶段不训练模型，只用已有冻结 checkpoint 完成两项推理：

1. 对 train sample table 运行 3 个 full seed，生成 train ensemble，用于估计 no-effect 背景；
2. 单独构造完整的 validation teacher-forced primary 表，确保每个 validation pair 都有一条真实干预位置的 primary replacement 预测。

现有 `verifier_samples_validation.parquet` 中，teacher-forced row 会在与 proposal candidate 占用相同 `(pair_id, query_t, replacement_rank)` 时被省略，因此当前 teacher-forced 指标只有 103 pairs / 4 effective。新表必须达到：

```text
256 pairs
29 effective pairs
每个 pair 恰好一条 primary teacher-forced row
```

## 3F-R-B.2 对 train split 运行冻结 verifier

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

infer_train_full() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r.inference.infer_long_horizon_verifier \
    --checkpoint "$STAGE3_V1_ROOT/runs/full_seed_${seed}/best.pt" \
    --samples "$SAMPLES_TRAIN" \
    --feature-index "$FEATURE_INDEX_V02" \
    --normalizer "$VERIFIER_NORMALIZER" \
    --config "$STAGE3_CONFIG" \
    --mode full \
    --output "$STAGE3_AGG_ROOT/predictions/train/full_seed_${seed}.parquet" \
    > "$STAGE3_AGG_ROOT/logs/S3FR-B-train-full-seed${seed}.log" 2>&1
}

infer_train_full 0 0 & P0=$!
infer_train_full 1 1 & P1=$!
infer_train_full 2 2 & P2=$!

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.inference.infer_long_horizon_verifier \
  --checkpoint "$STAGE3_V1_ROOT/runs/action_only_seed_0/best.pt" \
  --samples "$SAMPLES_TRAIN" \
  --feature-index "$FEATURE_INDEX_V02" \
  --normalizer "$VERIFIER_NORMALIZER" \
  --config "$STAGE3_CONFIG" \
  --mode action_only \
  --output "$STAGE3_AGG_ROOT/predictions/train/action_only_seed_0.parquet" \
  > "$STAGE3_AGG_ROOT/logs/S3FR-B-train-action-only.log" 2>&1 & P3=$!

wait "$P0" "$P1" "$P2" "$P3"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.inference.merge_verifier_ensemble \
  --inputs \
    "$STAGE3_AGG_ROOT/predictions/train/full_seed_0.parquet" \
    "$STAGE3_AGG_ROOT/predictions/train/full_seed_1.parquet" \
    "$STAGE3_AGG_ROOT/predictions/train/full_seed_2.parquet" \
  --samples "$SAMPLES_TRAIN" \
  --std-multiplier 1.0 \
  --score-max 0.9 \
  --output "$STAGE3_AGG_ROOT/predictions/train/train_ensemble.parquet" \
  --summary "$STAGE3_AGG_ROOT/predictions/train/train_ensemble_summary.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-B-merge-train-ensemble.log"
```

直接检查行数与 ID：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import os, pandas as pd
root=os.environ['STAGE3_AGG_ROOT']
samples=pd.read_parquet(os.environ['SAMPLES_TRAIN'])
ens=pd.read_parquet(f'{root}/predictions/train/train_ensemble.parquet')
assert len(ens)==len(samples), (len(ens),len(samples))
assert ens.replacement_id.is_unique
assert set(ens.replacement_id)==set(samples.replacement_id)
print({'train_rows':len(ens),'query_groups':ens.query_group_id.nunique()})
PY
```

## 3F-R-B.3 创建完整 teacher-forced sample 构建器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/build_complete_teacher_forced_samples.py
```

实现要求：

1. 读取 `$TEACHER_FORCED_LONG`；
2. 只保留 `split=validation`；
3. 只保留 `query_t == intervention_t` 且 `replacement_rank == 0`；
4. 不与 proposal candidate 去重；
5. 若 oracle row 缺少 `replacement_action`，从 `$TEACHER_PLANS` 按 `replacement_id` 回填；
6. 使用与原 `build_verifier_samples.py` 相同的工程有效条件：

```python
branch_pre_state_equal
reference_exact
finite_target
state_in_domain
action_in_domain
```

7. 输出字段必须兼容 `CounterfactualVerifierDataset`；
8. 每个 pair 恰好一行；
9. 断言 256 pairs 和 29 个 effective pairs。

推荐直接复用原模块中的：

```python
from stage3a.method_v02r.data.build_verifier_samples import normalize_row, target_is_valid
```

关键逻辑：

```python
rows = [r for r in teacher_rows if r['split']=='validation']
rows = [r for r in rows if int(r['query_t'])==int(r['intervention_t'])]
rows = [r for r in rows if int(r['replacement_rank'])==0]
rows = [r for r in rows if target_is_valid(r)]

records = [normalize_row(r, metadata, feature_lookup, True) for r in rows]
frame = pd.DataFrame(records).drop_duplicates('pair_id')

assert len(frame) == 256
assert int(frame.is_effective_intervention.sum()) == 29
assert frame.pair_id.is_unique
```

执行：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.build_complete_teacher_forced_samples \
  --teacher-forced "$TEACHER_FORCED_LONG" \
  --teacher-plans "$TEACHER_PLANS" \
  --metadata "$META_V02" \
  --feature-index "$FEATURE_INDEX_V02" \
  --output "$STAGE3_AGG_ROOT/development/teacher_forced_complete_samples.parquet" \
  --summary "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_samples.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-B-build-complete-teacher.log"
```

## 3F-R-B.4 对完整 teacher-forced 表运行冻结 verifier

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

infer_teacher_full() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r.inference.infer_long_horizon_verifier \
    --checkpoint "$STAGE3_V1_ROOT/runs/full_seed_${seed}/best.pt" \
    --samples "$STAGE3_AGG_ROOT/development/teacher_forced_complete_samples.parquet" \
    --feature-index "$FEATURE_INDEX_V02" \
    --normalizer "$VERIFIER_NORMALIZER" \
    --config "$STAGE3_CONFIG" \
    --mode full \
    --output "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/full_seed_${seed}.parquet" \
    > "$STAGE3_AGG_ROOT/logs/S3FR-B-teacher-full-seed${seed}.log" 2>&1
}

infer_teacher_full 0 0 & P0=$!
infer_teacher_full 1 1 & P1=$!
infer_teacher_full 2 2 & P2=$!

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.inference.infer_long_horizon_verifier \
  --checkpoint "$STAGE3_V1_ROOT/runs/action_only_seed_0/best.pt" \
  --samples "$STAGE3_AGG_ROOT/development/teacher_forced_complete_samples.parquet" \
  --feature-index "$FEATURE_INDEX_V02" \
  --normalizer "$VERIFIER_NORMALIZER" \
  --config "$STAGE3_CONFIG" \
  --mode action_only \
  --output "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/action_only_seed_0.parquet" \
  > "$STAGE3_AGG_ROOT/logs/S3FR-B-teacher-action-only.log" 2>&1 & P3=$!

wait "$P0" "$P1" "$P2" "$P3"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.inference.merge_verifier_ensemble \
  --inputs \
    "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/full_seed_0.parquet" \
    "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/full_seed_1.parquet" \
    "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/full_seed_2.parquet" \
  --samples "$STAGE3_AGG_ROOT/development/teacher_forced_complete_samples.parquet" \
  --std-multiplier 1.0 \
  --score-max 0.9 \
  --output "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/ensemble.parquet" \
  --summary "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/ensemble_summary.json"
```

## 3F-R-B.5 创建完整 teacher-forced 评估器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/evaluate_complete_teacher_forced.py
```

输出：

```text
full ensemble effective-vs-no_effect AUROC / AUPRC
full score MAE / Spearman against oracle continuous score
action-only effective-vs-no_effect AUROC / AUPRC
Can / Square 分组
pair count / positive count
```

评估标签：

```python
y = is_effective_intervention
score_full = pred_score_mean
score_action = action_pred_score
```

执行：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.evaluate_complete_teacher_forced \
  --ensemble "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/ensemble.parquet" \
  --action-only "$STAGE3_AGG_ROOT/predictions/teacher_forced_complete/action_only_seed_0.parquet" \
  --output "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_metrics.json" \
  --csv "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_metrics.csv" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-B-evaluate-complete-teacher.log"
```

直接检查：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
p=f"{os.environ['STAGE3_AGG_ROOT']}/metrics/teacher_forced_complete_metrics.json"
x=json.load(open(p))
assert x['pair_count']==256, x
assert x['effective_pair_count']==29, x
assert x['full']['auroc'] is not None
print(x)
PY
```

### 3F-R-B 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/predictions/train/train_ensemble.parquet"
test -s "$STAGE3_AGG_ROOT/development/teacher_forced_complete_samples.parquet"
test -s "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_metrics.json"
```

提交新评估代码：

```bash
cd "$SCIZOR_ROOT"
git add stage3a/method_v02r_aggregation_repair/evaluation

git -c user.name="Experiment Agent" -c user.email="agent@local" \
  commit -m "stage3F-R-B: complete train inference and teacher-forced diagnostic"
```

---

# 小阶段 3F-R-C：估计 train 背景、加入 outcome deficit 并生成固定聚合矩阵

## 3F-R-C.1 本小阶段总体上要干什么

本小阶段构建聚合修复的核心数据层：

1. 将每个 replacement 的 conservative verifier score 聚合为 transition 分数；
2. 使用 train no-effect pair 估计“正常情况下替换动作也可能带来的改善背景”；
3. 将原 SCIZOR 的局部 chunk deficit 映射到每个候选 transition；
4. 为 development 的固定五种方法生成 replacement、transition 和 pair 三层表；
5. 所有方法都覆盖完整 256 个 development pairs，不能静默丢弃没有有效候选的 pair。

## 3F-R-C.2 固定数学定义

对候选 transition `j` 的第 `k` 个替代动作：

\[
c_{j,k}=\text{replacement\_cf\_score}_{j,k}
\]

四个 replacement 的 transition 汇总：

\[
c_j^{\max}=\max_k c_{j,k}
\]

\[
c_j^{\mathrm{med}}=\operatorname{median}_k c_{j,k}
\]

\[
c_j^{\mathrm{contrast}}
=\max\left(c_j^{\max}-c_j^{\mathrm{med}},0\right)
\]

train-only no-effect 背景：

\[
b_{q}^{\max}(task,source)
=Q_{0.80}\left(c_j^{\max}\mid\text{train no-effect}\right)
\]

\[
b_{q}^{\mathrm{contrast}}(task,source)
=Q_{0.80}\left(c_j^{\mathrm{contrast}}\mid\text{train no-effect}\right)
\]

局部 outcome deficit：

\[
d_j=\operatorname{clip}\left(
\max_{c:\,start_c\le j<end_c}V_c,
0,1
\right)
\]

固定方法：

### 方法 1：`action_cf_only`

\[
T_j=c_j^{\max}
\]

\[
S_{pair}=\max_j T_j
\]

### 方法 2：`action_current_fused`

\[
T_j=c_j^{\max}\cdot w_j^{rank}
\]

\[
S_{pair}=\max_j T_j
\]

### 方法 3：`action_defect_gated`

\[
T_j=d_j\left[c_j^{\max}-b_q^{\max}\right]_+
\]

\[
S_{pair}=\max_j T_j
\]

### 方法 4：`action_defect_contrast`

\[
T_j=d_j\left[c_j^{\mathrm{contrast}}-b_q^{\mathrm{contrast}}\right]_+
\]

\[
S_{pair}=\max_j T_j-\operatorname{median}_j T_j
\]

### 方法 5：`union_defect_contrast`

公式与方法 4 相同，但候选来自 full/action Top-5 去重并集。

所有方法的 predicted transition 都定义为：

\[
\hat{t}=\arg\max_j T_j
\]

诊断方法另外输出：

```text
action_raw
union_raw
full_raw
```

raw 方法不参与 repaired method 选择，只用于判断“新聚合是否真正超越 proposer 本身”。

## 3F-R-C.3 创建聚合工具模块

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/aggregation_utils.py
```

至少实现：

```python
def read_jsonl(path): ...
def binary_metrics(labels, scores): ...
def threshold_metrics(labels, scores, threshold): ...
def build_pair_universe(labels, split): ...
def attach_local_deficit(candidate_rows, chunk_evidence, labels, split): ...
def replacement_summary(group): ...
def source_columns(source): ...
def fill_missing_pairs(pair_scores, pair_universe): ...
```

`attach_local_deficit` 必须使用：

```text
chunk interval = [start_t, end_t)
只使用 perturbed chunk
同 task + demo_id
覆盖 query_t 的所有 chunk 取最大 V_c
无覆盖时 local_deficit=0.0，并记录 deficit_coverage=false
```

推荐实现方式：

```python
for (task, demo_id), chunks in evidence.groupby(['task','demo_id']):
    max_t = int(chunks.end_t.max())
    deficit = np.zeros(max_t + 1, dtype=np.float32)
    covered = np.zeros(max_t + 1, dtype=bool)
    for row in chunks.itertuples(index=False):
        start, end = int(row.start_t), int(row.end_t)
        if end <= start:
            continue
        deficit[start:end] = np.maximum(deficit[start:end], float(row.V_c))
        covered[start:end] = True
    deficit_map[(task,demo_id)] = (deficit,covered)
```

## 3F-R-C.4 创建 train 背景估计器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/build_train_background.py
```

输入：

```text
train ensemble
train verifier samples
train proposal candidates
v0.2 labels
chunk evidence
aggregation repair config
```

步骤：

1. 从 train ensemble 过滤 `is_teacher_forced=false`；
2. 与 proposal candidates 按 `pair_id + query_t` 合并；
3. 分别取 `action_top5`、`union_top5` 和 full diagnostic；
4. 每个 `(pair_id, query_t)` 计算 `cf_max / cf_median / cf_contrast`；
5. 只使用 `is_effective_intervention=false` 的 train rows；
6. 按 `task + source` 计算 q80；
7. 同时统计 local deficit 的覆盖率；
8. 输出 JSON，不输出任何 development 统计。

输出 JSON 结构：

```json
{
  "schema": "stage3f_r_train_background_v1",
  "quantile": 0.8,
  "sources": {
    "action_top5": {
      "can": {"cf_max_q80": 0.0, "cf_contrast_q80": 0.0, "n": 0},
      "square": {"cf_max_q80": 0.0, "cf_contrast_q80": 0.0, "n": 0}
    },
    "union_top5": {},
    "full_top5": {}
  },
  "local_deficit": {
    "formula": "clip(max covering V_c,0,1)",
    "coverage_rate": 1.0
  }
}
```

执行：

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.build_train_background \
  --ensemble "$STAGE3_AGG_ROOT/predictions/train/train_ensemble.parquet" \
  --samples "$SAMPLES_TRAIN" \
  --proposals "$PROPOSALS_TRAIN" \
  --labels "$LABELS_V02" \
  --chunk-evidence "$CHUNK_EVIDENCE_V02" \
  --config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  --output "$STAGE3_AGG_ROOT/background/train_background.json" \
  --transition-output "$STAGE3_AGG_ROOT/background/train_transition_background.parquet" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-C-build-train-background.log"
```

直接检查 q80 不是缺失值：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, math, os
p=f"{os.environ['STAGE3_AGG_ROOT']}/background/train_background.json"
x=json.load(open(p))
for source in ('action_top5','union_top5'):
    for task in ('can','square'):
        item=x['sources'][source][task]
        assert item['n'] > 0, (source,task,item)
        assert math.isfinite(item['cf_max_q80'])
        assert math.isfinite(item['cf_contrast_q80'])
print(json.dumps(x,indent=2))
PY
```

## 3F-R-C.5 创建通用聚合分数构建器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/build_aggregation_scores.py
```

命令参数：

```text
--ensemble
--samples
--proposals
--labels
--chunk-evidence
--background
--config
--split
--output-replacements
--output-transitions
--output-pairs
--summary
```

必须生成全部方法，不允许根据 development 结果临时增加公式。

### Replacement 表必须包含

```text
replacement_id
pair_id
task
split
query_t
replacement_rank
replacement_cf_score
pred_score_mean
pred_score_std
pred_positive_mean
pred_positive_std
source membership
raw proposer scores
rank columns
local_deficit
is_effective_intervention
intervention_t
responsible_start
responsible_end
failure_type
```

### Transition 表必须包含

```text
method_id
source
pair_id
task
split
query_t
transition_score
cf_max
cf_median
cf_contrast
background_max
background_contrast
local_deficit
proposal_rank_weight
raw_proposer_score
replacement_count
is_effective_intervention
intervention_t
responsible_start
responsible_end
is_recovery
```

### Pair 表必须包含

```text
method_id
source
pair_id
task
split
pair_score
predicted_t
candidate_count
has_valid_candidate
is_effective_intervention
intervention_t
responsible_start
responsible_end
failure_type
```

### 完整 pair universe

必须从 labels 中构造完整 perturbed pair universe，再对每个方法 left join：

```python
pair_universe = (
    labels[(labels.variant=='perturbed') & (labels.split==split)]
    .drop_duplicates('pair_id')
)
```

若某个 source 因未来长度不足而没有有效 replacement：

```python
pair_score = 0.0
predicted_t = -1
candidate_count = 0
has_valid_candidate = False
```

每个方法在 development 必须输出恰好 256 行。

执行 development 分数构建：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.build_aggregation_scores \
  --ensemble "$ENSEMBLE_DEV" \
  --samples "$SAMPLES_DEV" \
  --proposals "$PROPOSALS_DEV" \
  --labels "$LABELS_V02" \
  --chunk-evidence "$CHUNK_EVIDENCE_V02" \
  --background "$STAGE3_AGG_ROOT/background/train_background.json" \
  --config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  --split validation \
  --output-replacements "$STAGE3_AGG_ROOT/development/replacement_scores_all_methods.parquet" \
  --output-transitions "$STAGE3_AGG_ROOT/development/transition_scores_all_methods.parquet" \
  --output-pairs "$STAGE3_AGG_ROOT/development/pair_scores_all_methods.parquet" \
  --summary "$STAGE3_AGG_ROOT/metrics/development_score_build_summary.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-C-build-development-scores.log"
```

检查每个方法完整覆盖：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os, pandas as pd
root=os.environ['STAGE3_AGG_ROOT']
config=json.load(open(os.path.join(os.environ['SCIZOR_ROOT'],'stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json')))
pairs=pd.read_parquet(f'{root}/development/pair_scores_all_methods.parquet')
expected=config['diagnostic_methods']+config['eligible_methods']
for method in expected:
    part=pairs[pairs.method_id.eq(method)]
    assert len(part)==256, (method,len(part))
    assert part.pair_id.is_unique, method
    assert part.pair_score.notna().all(), method
print(pairs.groupby('method_id').size().to_dict())
PY
```

### 3F-R-C 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/background/train_background.json"
test -s "$STAGE3_AGG_ROOT/development/replacement_scores_all_methods.parquet"
test -s "$STAGE3_AGG_ROOT/development/transition_scores_all_methods.parquet"
test -s "$STAGE3_AGG_ROOT/development/pair_scores_all_methods.parquet"
```

提交核心聚合代码：

```bash
cd "$SCIZOR_ROOT"
git add stage3a/method_v02r_aggregation_repair/evaluation

git -c user.name="Experiment Agent" -c user.email="agent@local" \
  commit -m "stage3F-R-C: add fixed counterfactual aggregation matrix"
```

---

# 小阶段 3F-R-D：仅用 development 选择唯一聚合协议

## 3F-R-D.1 本小阶段总体上要干什么

当前 validation 已经被用于发现聚合问题，因此本轮将它明确称为 `development set`。

本小阶段只在固定的五个 eligible methods 中选择一个，不搜索：

```text
新的融合系数
新的 Top-k
新的背景分位数
新的 deficit 公式
新的 replacement 数量
新的 checkpoint
```

## 3F-R-D.2 创建 development protocol 选择器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/select_development_protocol.py
```

输入：

```text
all-method pair scores
all-method transition scores
complete teacher-forced metrics
原 verifier learning metrics
proposer transfer metrics
repair config
```

### 通用 threshold 选择

对每个 eligible method：

1. 枚举该方法 development pair score 的唯一值；
2. 只保留 `no_effect_far <= 0.20` 的阈值；
3. 在可行阈值中按以下顺序选择：

```text
最高 effective recall
最高 localized effective recall（预测 t 在 intervention_t ±1）
最高 threshold
```

`higher threshold` 只作为最后一个确定性 tie-break，不代表额外调参。

### 每个方法的统计

```text
pair AUROC / AUPRC
prevalence
proposal responsibility-region recall
threshold
no-effect FAR
effective recall
Top-1 within ±1
responsibility-region hit rate
mean absolute localization delay
recovery false attribution
Can AUROC / AUPRC
Square AUROC / AUPRC
mean candidates per pair
missing-candidate pair count
```

### Eligible method gate

每个方法必须同时满足：

```text
proposal responsibility-region recall >= 0.50
pair AUROC >= 0.70
存在 no-effect FAR <= 0.20 的 threshold
effective recall >= 0.40
Can pair AUROC >= 0.60
Square pair AUROC >= 0.60
```

全局 verifier gate：

```text
candidate replacement full AUROC >= 0.70
complete teacher-forced full AUROC >= 0.70
```

### 唯一方法选择

在通过 gate 的方法中：

1. 找最高 AUPRC；保留距离最高值不超过 0.01 的方法；
2. 其中找最低 no-effect FAR；保留距离最低值不超过 0.02 的方法；
3. 选择 effective recall 最高者；
4. 再选择 Top-1 within ±1 最高者；
5. 再选择 mean candidates 最少者；
6. 最后按 method_id 字典序。

### 输出协议

若有方法通过：

```json
{
  "development_gate_pass": true,
  "decision": "RESUME_STAGE3_BLIND_AFTER_AGGREGATION_REPAIR",
  "selected_method": "...",
  "selected_source": "action_top5 or union_top5",
  "selected_threshold": 0.0,
  "selected_formula": "...",
  "background_path": "...",
  "failed_rules": []
}
```

若没有方法通过：

```json
{
  "development_gate_pass": false,
  "decision": "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR",
  "selected_method": null,
  "selected_source": null,
  "selected_threshold": null,
  "failed_rules": ["no_valid_fixed_aggregation_method"]
}
```

## 3F-R-D.3 执行固定 development 选择

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.select_development_protocol \
  --pair-scores "$STAGE3_AGG_ROOT/development/pair_scores_all_methods.parquet" \
  --transition-scores "$STAGE3_AGG_ROOT/development/transition_scores_all_methods.parquet" \
  --teacher-forced-complete "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_metrics.json" \
  --verifier-learning "$STAGE3_V1_ROOT/metrics/validation_verifier_learning.json" \
  --proposer-transfer "$STAGE3_V1_ROOT/metrics/proposer_transfer_v02.json" \
  --config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  --output-metrics "$STAGE3_AGG_ROOT/metrics/development_aggregation_metrics.json" \
  --output-protocol "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
  --output-csv "$STAGE3_AGG_ROOT/metrics/development_aggregation_leaderboard.csv" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-D-select-development-protocol.log"
```

打印选择结果：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
p=f"{os.environ['STAGE3_AGG_ROOT']}/metrics/development_aggregation_protocol.json"
x=json.load(open(p))
print(json.dumps({k:x.get(k) for k in (
    'development_gate_pass','decision','selected_method',
    'selected_source','selected_threshold','failed_rules')},indent=2))
PY
```

## 3F-R-D.4 Development 失败时立即停止

执行：

```bash
if ! PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
x=json.load(open(f"{os.environ['STAGE3_AGG_ROOT']}/metrics/development_aggregation_protocol.json"))
raise SystemExit(0 if x['development_gate_pass'] else 1)
PY
then
  echo "fixed aggregation matrix did not pass; do not generate blind benchmark"
  touch "$STAGE3_AGG_ROOT/STOP_BEFORE_BLIND"
fi
```

如果生成了 `STOP_BEFORE_BLIND`：

```text
跳过 3F-R-E 中 blind freeze 部分
跳过 3F-R-F
直接执行 3F-R-G 的 validation-only 收尾
```

### 3F-R-D 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/metrics/development_aggregation_metrics.json"
test -s "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json"
test -s "$STAGE3_AGG_ROOT/metrics/development_aggregation_leaderboard.csv"
```

提交选择代码：

```bash
cd "$SCIZOR_ROOT"
git add stage3a/method_v02r_aggregation_repair/evaluation

git -c user.name="Experiment Agent" -c user.email="agent@local" \
  commit -m "stage3F-R-D: select one fixed development aggregation protocol"
```

---

# 小阶段 3F-R-E：冻结 repaired protocol 和方法包

## 3F-R-E.1 本小阶段总体上要干什么

仅当 `development_gate_pass=true` 时执行完整本阶段。

目标是将以下内容冻结为 blind 读取前不可修改的协议：

```text
selected method
selected candidate source
selected score formula
selected threshold
train no-effect background
outcome deficit 公式
full verifier checkpoints
feature normalizer
proposal calibration
action library
长期 oracle score spec
代码 commit
```

## 3F-R-E.2 创建 repaired method bundle

创建：

```text
stage3a/method_v02r_aggregation_repair/decision/build_repaired_method_bundle.py
```

输入：

```text
development_aggregation_protocol.json
train_background.json
v1 method bundle
repair config
```

输出：

```text
report/stage3f_r_method_bundle.json
```

至少包含：

```json
{
  "schema": "stage3f_r_method_bundle_v1",
  "selected_method": "...",
  "selected_source": "...",
  "selected_threshold": 0.0,
  "selected_formula": "...",
  "background_quantile": 0.8,
  "background_sha256": "...",
  "repair_config_sha256": "...",
  "development_protocol_sha256": "...",
  "verifier_checkpoints": {},
  "feature_normalizer": {},
  "action_library": {},
  "proposal_calibration": {},
  "oracle_score_spec": {},
  "base_stage3_v1_decision": "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR",
  "blind_status": "NOT_READ"
}
```

执行：

```bash
if [ ! -f "$STAGE3_AGG_ROOT/STOP_BEFORE_BLIND" ]; then
  PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r_aggregation_repair.decision.build_repaired_method_bundle \
    --protocol "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
    --background "$STAGE3_AGG_ROOT/background/train_background.json" \
    --repair-config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
    --v1-method-bundle "$STAGE3_V1_ROOT/report/stage3_method_bundle.json" \
    --output "$STAGE3_AGG_ROOT/report/stage3f_r_method_bundle.json"
fi
```

## 3F-R-E.3 冻结协议哈希和代码提交

```bash
if [ ! -f "$STAGE3_AGG_ROOT/STOP_BEFORE_BLIND" ]; then
  sha256sum \
    "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
    "$STAGE3_AGG_ROOT/background/train_background.json" \
    "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
    "$STAGE3_AGG_ROOT/report/stage3f_r_method_bundle.json" \
    > "$STAGE3_AGG_ROOT/config/repaired_protocol.sha256"

  cd "$SCIZOR_ROOT"
  git add stage3a/method_v02r_aggregation_repair
  git -c user.name="Experiment Agent" -c user.email="agent@local" \
    commit -m "stage3F-R-E: freeze aggregation-repaired blind protocol"

  git rev-parse HEAD > "$STAGE3_AGG_ROOT/config/repaired_protocol_commit.txt"
fi
```

从此以后，blind 执行期间不得修改：

```text
stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json
train_background.json
development_aggregation_protocol.json
build_aggregation_scores.py
selected threshold
selected source
```

### 3F-R-E 完成标准

仅在 development 通过时要求：

```bash
test -s "$STAGE3_AGG_ROOT/report/stage3f_r_method_bundle.json"
test -s "$STAGE3_AGG_ROOT/config/repaired_protocol.sha256"
test -s "$STAGE3_AGG_ROOT/config/repaired_protocol_commit.txt"
```

---

# 小阶段 3F-R-F：条件生成 blind benchmark 并运行冻结聚合协议

## 3F-R-F.1 本小阶段总体上要干什么

仅当：

```text
development_gate_pass=true
STOP_BEFORE_BLIND 不存在
```

时执行。

本阶段生成此前从未使用的：

```text
Can 10 条成功 clean base demos
Square 10 条成功 clean base demos
共 20 × 4 干预点 × 4 扰动 = 320 pairs
```

然后只运行一次：

```text
标签和特征
SCIZOR chunk evidence
冻结 action/full proposer
冻结 selected source 的 replacement plans
冻结 verifier ensemble inference
冻结 repaired aggregation formula
冻结 development threshold
blind metrics
真实干预位置 teacher-forced oracle 对照
```

## 3F-R-F.2 Gate 检查

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

if [ -f "$STAGE3_AGG_ROOT/STOP_BEFORE_BLIND" ]; then
  echo "development failed; blind is forbidden"
  exit 1
fi

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
x=json.load(open(f"{os.environ['STAGE3_AGG_ROOT']}/metrics/development_aggregation_protocol.json"))
assert x['development_gate_pass'] is True, x
assert x['decision']=='RESUME_STAGE3_BLIND_AFTER_AGGREGATION_REPAIR', x
print('blind generation allowed:', x['selected_method'], x['selected_threshold'])
PY
```

## 3F-R-F.3 解析源 HDF5 并选择全新 base demos

```bash
eval "$(PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" - <<'PY'
import h5py, os, shlex
found={}
with h5py.File(os.environ['BENCHMARK_V02'],'r') as h5:
    for g in h5['data'].values():
        task=g.attrs.get('task',''); task=task.decode() if isinstance(task,bytes) else task
        src=g.attrs.get('source_dataset',''); src=src.decode() if isinstance(src,bytes) else src
        if task in {'can','square'} and src: found.setdefault(task,src)
print('export CAN_SOURCE='+shlex.quote(found['can']))
print('export SQUARE_SOURCE='+shlex.quote(found['square']))
PY
)"

printf 'export CAN_SOURCE=%q\nexport SQUARE_SOURCE=%q\n' \
  "$CAN_SOURCE" "$SQUARE_SOURCE" \
  > "$STAGE3_AGG_ROOT/blind_test/source_paths.env"

PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" \
  -m stage3a.method_v02r.benchmark.select_final_blind_demos \
  --can-source "$CAN_SOURCE" \
  --square-source "$SQUARE_SOURCE" \
  --stage1-metadata "$STAGE1_ROOT/benchmark/pair_metadata.jsonl" \
  --v02-bases "$BASES_V02" \
  --confirmation-bases "$CONFIRMATION_BASES" \
  --can-count 10 \
  --square-count 10 \
  --output "$STAGE3_AGG_ROOT/blind_test/base_demos_final.json" \
  --details "$STAGE3_AGG_ROOT/blind_test/base_demo_checks.jsonl" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-select-blind-demos.log"
```

## 3F-R-F.4 生成 320-pair blind benchmark

```bash
source "$STAGE3_AGG_ROOT/blind_test/source_paths.env"

PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" \
  -m stage3a.rescue_v02.generate_benchmark_v02 \
  --base-manifest "$STAGE3_AGG_ROOT/blind_test/base_demos_final.json" \
  --can-source "$CAN_SOURCE" \
  --square-source "$SQUARE_SOURCE" \
  --stage1-pair-meta "$STAGE1_ROOT/benchmark/pair_metadata.jsonl" \
  --runtime-fingerprint "$V02_ROOT/config/runtime_fingerprint.json" \
  --output-hdf5 "$STAGE3_AGG_ROOT/blind_test/benchmark_v0.2_final_test.hdf5" \
  --output-meta "$STAGE3_AGG_ROOT/blind_test/pair_metadata_v0.2_final_test.jsonl" \
  --num-intervention-points 4 \
  --perturbations zero_motion,reverse_motion,flip_gripper,axis_impulse \
  --seed 20260904 \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-generate-blind.log"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['STAGE3_AGG_ROOT'])/'blind_test'
rows=[json.loads(x) for x in (root/'pair_metadata_v0.2_final_test.jsonl').read_text().splitlines() if x.strip()]
ids=[r['pair_id'] for r in rows]
assert len(ids)==320, len(ids)
(root/'split_manifest_v0.2_final_test.json').write_text(json.dumps({'blind_test':ids},indent=2))
print({'pairs':len(ids)})
PY

PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" \
  -m stage3a.rescue_v02.audit_benchmark_v02 \
  --benchmark "$STAGE3_AGG_ROOT/blind_test/benchmark_v0.2_final_test.hdf5" \
  --metadata "$STAGE3_AGG_ROOT/blind_test/pair_metadata_v0.2_final_test.jsonl" \
  --split-manifest "$STAGE3_AGG_ROOT/blind_test/split_manifest_v0.2_final_test.json" \
  --expected-pairs 320 \
  --output "$STAGE3_AGG_ROOT/metrics/blind_test_benchmark_check.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-audit-blind.log"

sha256sum \
  "$STAGE3_AGG_ROOT/blind_test/benchmark_v0.2_final_test.hdf5" \
  "$STAGE3_AGG_ROOT/blind_test/pair_metadata_v0.2_final_test.jsonl" \
  "$STAGE3_AGG_ROOT/blind_test/split_manifest_v0.2_final_test.json" \
  "$STAGE3_AGG_ROOT/blind_test/base_demos_final.json" \
  > "$STAGE3_AGG_ROOT/config/blind_test_freeze.sha256"
```

## 3F-R-F.5 构建 blind labels、features 和 outcome evidence

```bash
export BLIND_H5="$STAGE3_AGG_ROOT/blind_test/benchmark_v0.2_final_test.hdf5"
export BLIND_META="$STAGE3_AGG_ROOT/blind_test/pair_metadata_v0.2_final_test.jsonl"
export BLIND_SPLIT="$STAGE3_AGG_ROOT/blind_test/split_manifest_v0.2_final_test.json"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.data.build_v02_transition_labels \
  --benchmark "$BLIND_H5" \
  --metadata "$BLIND_META" \
  --split-manifest "$BLIND_SPLIT" \
  --output "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
  --summary "$STAGE3_AGG_ROOT/blind_test/transition_labels_summary.json"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.data.extract_v02_transition_features \
  --benchmark "$BLIND_H5" \
  --labels "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
  --config "$STAGE3_CONFIG" \
  --output-dir "$STAGE3_AGG_ROOT/blind_test/features/cache" \
  --output-index "$STAGE3_AGG_ROOT/blind_test/features/feature_index.parquet" \
  --output-manifest "$STAGE3_AGG_ROOT/blind_test/features/feature_manifest.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-blind-features.log"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.data.build_scizor_scoring_adapter \
  --benchmark "$BLIND_H5" \
  --output "$STAGE3_AGG_ROOT/blind_test/scoring_adapter.hdf5"

mkdir -p "$STAGE3_AGG_ROOT/blind_test/adapter_dir"
ln -sfn "$STAGE3_AGG_ROOT/blind_test/scoring_adapter.hdf5" \
  "$STAGE3_AGG_ROOT/blind_test/adapter_dir/scoring_adapter.hdf5"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  "$SCIZOR_ROOT/stage1/src/export_chunk_evidence.py" \
  --data-dir "$STAGE3_AGG_ROOT/blind_test/adapter_dir" \
  --model-path "$FROZEN_SCIZOR_DIR" \
  --goal-time 2 \
  --image-key agentview_image \
  --batch-size 256 \
  --output "$STAGE3_AGG_ROOT/blind_test/chunk_evidence.parquet" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-blind-evidence.log"
```

不要对 blind 重新计算 verifier normalizer、proposal calibration 或 train background。

## 3F-R-F.6 运行冻结 proposer 并构建 proposal candidates

```bash
infer_blind_proposer() {
  local gpu="$1"
  local checkpoint="$2"
  local model_name="$3"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage2.inference.infer_responsibility \
    --checkpoint "$checkpoint" \
    --model-name "$model_name" \
    --split blind_test \
    --chunk-evidence "$STAGE3_AGG_ROOT/blind_test/chunk_evidence.parquet" \
    --transition-labels "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
    --feature-index "$STAGE3_AGG_ROOT/blind_test/features/feature_index.parquet" \
    --normalizer "$STAGE2_TRAIN_NORMALIZER" \
    --config "$STAGE2_ITER_CONFIG" \
    --output-chunks "$STAGE3_AGG_ROOT/blind_test/${model_name}_chunks.parquet" \
    --output-transitions "$STAGE3_AGG_ROOT/blind_test/${model_name}_transitions.parquet" \
    --batch-size 128 \
    > "$STAGE3_AGG_ROOT/logs/S3FR-F-${model_name}.log" 2>&1
}

infer_blind_proposer 0 "$FULL_PROPOSER_CKPT" full_proposer_v02 & P0=$!
infer_blind_proposer 1 "$ACTION_PROPOSER_CKPT" action_proposer_v02 & P1=$!
wait "$P0" "$P1"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.proposals.build_topk_proposals \
  --full-transition-scores "$STAGE3_AGG_ROOT/blind_test/full_proposer_v02_transitions.parquet" \
  --action-transition-scores "$STAGE3_AGG_ROOT/blind_test/action_proposer_v02_transitions.parquet" \
  --transition-labels "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
  --top-k 5 \
  --splits blind_test \
  --score-calibration "$STAGE3_V1_ROOT/proposals/proposal_score_calibration_v02.json" \
  --output "$STAGE3_AGG_ROOT/blind_test/proposal_candidates.parquet" \
  --summary "$STAGE3_AGG_ROOT/blind_test/proposal_summary.json"
```

## 3F-R-F.7 只为 selected source 构建 replacement plans

读取冻结 source：

```bash
export SELECTED_SOURCE="$(PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
x=json.load(open(f"{os.environ['STAGE3_AGG_ROOT']}/metrics/development_aggregation_protocol.json"))
print(x['selected_source'])
PY
)"

echo "selected source: $SELECTED_SOURCE"
```

执行：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.oracle.build_candidate_replacement_plans \
  --benchmark "$BLIND_H5" \
  --metadata "$BLIND_META" \
  --proposal-candidates "$STAGE3_AGG_ROOT/blind_test/proposal_candidates.parquet" \
  --action-library "$ACTION_LIBRARY_V02" \
  --split blind_test \
  --proposal-source "$SELECTED_SOURCE" \
  --num-replacements 4 \
  --min-future-steps 20 \
  --output "$STAGE3_AGG_ROOT/blind_test/replacement_plans.parquet" \
  --jsonl-output "$STAGE3_AGG_ROOT/blind_test/replacement_plans.jsonl" \
  --summary "$STAGE3_AGG_ROOT/blind_test/replacement_plans_summary.json"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.data.build_verifier_inference_samples \
  --plans "$STAGE3_AGG_ROOT/blind_test/replacement_plans.parquet" \
  --feature-index "$STAGE3_AGG_ROOT/blind_test/features/feature_index.parquet" \
  --output "$STAGE3_AGG_ROOT/blind_test/verifier_inference_samples.parquet" \
  --summary "$STAGE3_AGG_ROOT/blind_test/verifier_inference_samples_summary.json"
```

## 3F-R-F.8 三 seed blind verifier inference

```bash
mkdir -p "$STAGE3_AGG_ROOT/blind_test/predictions"

infer_blind_full() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r.inference.infer_long_horizon_verifier \
    --checkpoint "$STAGE3_V1_ROOT/runs/full_seed_${seed}/best.pt" \
    --samples "$STAGE3_AGG_ROOT/blind_test/verifier_inference_samples.parquet" \
    --feature-index "$STAGE3_AGG_ROOT/blind_test/features/feature_index.parquet" \
    --normalizer "$VERIFIER_NORMALIZER" \
    --config "$STAGE3_CONFIG" \
    --mode full \
    --output "$STAGE3_AGG_ROOT/blind_test/predictions/full_seed_${seed}.parquet" \
    > "$STAGE3_AGG_ROOT/logs/S3FR-F-blind-full-seed${seed}.log" 2>&1
}

infer_blind_full 0 0 & P0=$!
infer_blind_full 1 1 & P1=$!
infer_blind_full 2 2 & P2=$!
wait "$P0" "$P1" "$P2"

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.inference.merge_verifier_ensemble \
  --inputs \
    "$STAGE3_AGG_ROOT/blind_test/predictions/full_seed_0.parquet" \
    "$STAGE3_AGG_ROOT/blind_test/predictions/full_seed_1.parquet" \
    "$STAGE3_AGG_ROOT/blind_test/predictions/full_seed_2.parquet" \
  --samples "$STAGE3_AGG_ROOT/blind_test/verifier_inference_samples.parquet" \
  --std-multiplier 1.0 \
  --score-max 0.9 \
  --output "$STAGE3_AGG_ROOT/blind_test/predictions/ensemble.parquet" \
  --summary "$STAGE3_AGG_ROOT/blind_test/predictions/ensemble_summary.json"
```

## 3F-R-F.9 使用冻结 repaired formula 构建 blind score

调用同一个 `build_aggregation_scores.py`，背景必须仍使用 train JSON：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.build_aggregation_scores \
  --ensemble "$STAGE3_AGG_ROOT/blind_test/predictions/ensemble.parquet" \
  --samples "$STAGE3_AGG_ROOT/blind_test/verifier_inference_samples.parquet" \
  --proposals "$STAGE3_AGG_ROOT/blind_test/proposal_candidates.parquet" \
  --labels "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
  --chunk-evidence "$STAGE3_AGG_ROOT/blind_test/chunk_evidence.parquet" \
  --background "$STAGE3_AGG_ROOT/background/train_background.json" \
  --config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  --split blind_test \
  --only-method "$(PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" - <<'PY'
import json, os
x=json.load(open(f"{os.environ['STAGE3_AGG_ROOT']}/metrics/development_aggregation_protocol.json"))
print(x['selected_method'])
PY
)" \
  --output-replacements "$STAGE3_AGG_ROOT/blind_test/replacement_scores.parquet" \
  --output-transitions "$STAGE3_AGG_ROOT/blind_test/transition_scores.parquet" \
  --output-pairs "$STAGE3_AGG_ROOT/blind_test/pair_scores.parquet" \
  --summary "$STAGE3_AGG_ROOT/blind_test/score_build_summary.json"
```

## 3F-R-F.10 创建冻结 blind 评估器

创建：

```text
stage3a/method_v02r_aggregation_repair/evaluation/evaluate_frozen_blind.py
```

该脚本必须直接读取：

```text
development_aggregation_protocol.json
selected_method
selected_threshold
selected_source
```

禁止在 blind 上：

```text
重新枚举 threshold
重新选择 method
重新计算 background
重新选择 candidate source
```

输出：

```text
pair AUROC / AUPRC
positive prevalence
no-effect FAR at frozen threshold
effective recall at frozen threshold
Top-1 within ±1
responsibility-region hit rate
mean localization delay
Can / Square AUROC
recovery false attribution
candidate coverage
```

Blind gate：

```text
pair AUROC >= 0.70
pair AUPRC >= 2 × positive prevalence
no-effect FAR <= 0.25
effective recall >= 0.35
Top-1 within ±1 >= 0.25
Can AUROC >= 0.60
Square AUROC >= 0.60
```

执行：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.evaluation.evaluate_frozen_blind \
  --pair-scores "$STAGE3_AGG_ROOT/blind_test/pair_scores.parquet" \
  --transition-scores "$STAGE3_AGG_ROOT/blind_test/transition_scores.parquet" \
  --labels "$STAGE3_AGG_ROOT/blind_test/transition_labels.parquet" \
  --metadata "$BLIND_META" \
  --protocol "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
  --config "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair/config_aggregation_repair.json" \
  --output "$STAGE3_AGG_ROOT/metrics/blind_aggregation_metrics.json" \
  --csv "$STAGE3_AGG_ROOT/metrics/blind_aggregation_metrics.csv" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-evaluate-blind.log"
```

## 3F-R-F.11 在真实干预位置补充 blind teacher-forced 对照

这一步只评估 verifier，不修改 repaired protocol。

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.rescue_v02.build_teacher_forced_plans \
  --benchmark "$BLIND_H5" \
  --metadata "$BLIND_META" \
  --split-manifest "$BLIND_SPLIT" \
  --action-library "$ACTION_LIBRARY_V02" \
  --query-source perturb_t \
  --num-replacements 4 \
  --output "$STAGE3_AGG_ROOT/blind_test/teacher_forced_plans.parquet" \
  --jsonl-output "$STAGE3_AGG_ROOT/blind_test/teacher_forced_plans.jsonl"

mkdir -p "$STAGE3_AGG_ROOT/blind_test/oracle_parts"
pids=()
for part in $(seq 0 7); do
  gpu=$((part % 4))
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" \
    -m stage3a.rescue_v02r.run_feasible_long \
    --benchmark "$BLIND_H5" \
    --plans "$STAGE3_AGG_ROOT/blind_test/teacher_forced_plans.jsonl" \
    --normalizer "$ORACLE_NORMALIZER_V02R" \
    --spec "$SCORE_SPEC_V02R" \
    --part-index "$part" \
    --num-parts 8 \
    --output "$STAGE3_AGG_ROOT/blind_test/oracle_parts/part${part}.jsonl" \
    --summary "$STAGE3_AGG_ROOT/blind_test/oracle_parts/part${part}.json" \
    > "$STAGE3_AGG_ROOT/logs/S3FR-F-blind-oracle-part${part}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

PYTHONPATH="$PYTHONPATH" "$MIMICGEN_PYTHON" \
  -m stage3a.rescue_v02.merge_jsonl_shards \
  --inputs "$STAGE3_AGG_ROOT"/blind_test/oracle_parts/part*.jsonl \
  --output "$STAGE3_AGG_ROOT/blind_test/teacher_forced_oracle_long.jsonl" \
  --id-key replacement_id

PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r.evaluation.evaluate_blind_teacher_forced \
  --oracle "$STAGE3_AGG_ROOT/blind_test/teacher_forced_oracle_long.jsonl" \
  --benchmark "$BLIND_H5" \
  --feature-index "$STAGE3_AGG_ROOT/blind_test/features/feature_index.parquet" \
  --checkpoints \
    "$STAGE3_V1_ROOT/runs/full_seed_0/best.pt" \
    "$STAGE3_V1_ROOT/runs/full_seed_1/best.pt" \
    "$STAGE3_V1_ROOT/runs/full_seed_2/best.pt" \
  --normalizer "$VERIFIER_NORMALIZER" \
  --config "$STAGE3_CONFIG" \
  --output "$STAGE3_AGG_ROOT/metrics/blind_teacher_forced_metrics.json" \
  2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-F-evaluate-blind-teacher.log"
```

### 3F-R-F 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/metrics/blind_test_benchmark_check.json"
test -s "$STAGE3_AGG_ROOT/metrics/blind_aggregation_metrics.json"
test -s "$STAGE3_AGG_ROOT/metrics/blind_teacher_forced_metrics.json"
test -s "$STAGE3_AGG_ROOT/blind_test/pair_scores.parquet"
test -s "$STAGE3_AGG_ROOT/blind_test/predictions/ensemble.parquet"
```

---

# 小阶段 3F-R-G：最终决策、报告、方法包和轻量交付

## 3F-R-G.1 本小阶段总体上要干什么

根据 development repaired protocol 和可选 blind 结果做最终收尾。

不再修改任何方法参数。

## 3F-R-G.2 创建最终决策脚本

创建：

```text
stage3a/method_v02r_aggregation_repair/decision/finalize_aggregation_repair.py
```

### Development 失败

若：

```text
development_gate_pass=false
```

直接输出：

```json
{
  "decision": "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR",
  "stage3_v1_decision_preserved": true,
  "aggregation_repair_attempted": true,
  "blind_generated": false,
  "failed_rules": ["development_aggregation_gate"]
}
```

### Development 通过且 blind 完成

Blind 必须同时满足：

```text
benchmark audit pass
pair AUROC >= 0.70
AUPRC >= 2 × prevalence
no-effect FAR <= 0.25
effective recall >= 0.35
Top-1 within ±1 >= 0.25
Can / Square AUROC 均 >= 0.60
teacher-forced engineering branch/reference/finite rate 通过
```

全部通过：

```text
GO_STAGE4_POLICY_VALIDATION
```

否则：

```text
SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR
```

执行：

```bash
set -euo pipefail
source "$STAGE3_AGG_ROOT/config/stage3f_r.env"

if [ -f "$STAGE3_AGG_ROOT/STOP_BEFORE_BLIND" ]; then
  PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r_aggregation_repair.decision.finalize_aggregation_repair \
    --v1-decision "$V1_DECISION" \
    --development "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
    --validation-only \
    --output "$STAGE3_AGG_ROOT/metrics/stage3f_r_final_decision.json" \
    2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-G-finalize-validation-only.log"
else
  PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
    -m stage3a.method_v02r_aggregation_repair.decision.finalize_aggregation_repair \
    --v1-decision "$V1_DECISION" \
    --development "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
    --blind "$STAGE3_AGG_ROOT/metrics/blind_aggregation_metrics.json" \
    --blind-teacher-forced "$STAGE3_AGG_ROOT/metrics/blind_teacher_forced_metrics.json" \
    --benchmark-check "$STAGE3_AGG_ROOT/metrics/blind_test_benchmark_check.json" \
    --output "$STAGE3_AGG_ROOT/metrics/stage3f_r_final_decision.json" \
    2>&1 | tee "$STAGE3_AGG_ROOT/logs/S3FR-G-finalize-blind.log"
fi
```

## 3F-R-G.3 生成对比报告

创建：

```text
stage3a/method_v02r_aggregation_repair/decision/write_aggregation_repair_report.py
```

报告必须包含：

1. Stage 3 v1 原始失败结果；
2. verifier 和 proposer 已通过的模块证据；
3. 完整 teacher-forced 256-pair 诊断；
4. 固定五方法 development leaderboard；
5. selected method、公式、source 和 threshold；
6. raw / cf-only / current fused / defect gate / contrast 的差异；
7. 若未生成 blind，明确写明原因；
8. 若生成 blind，报告 frozen blind 指标；
9. 最终决策；
10. 禁止将 development 改称未见验证集。

执行：

```bash
PYTHONPATH="$PYTHONPATH" "$CURATION_PYTHON" \
  -m stage3a.method_v02r_aggregation_repair.decision.write_aggregation_repair_report \
  --v1-report "$STAGE3_V1_ROOT/report/stage3_v02r_final_report.md" \
  --teacher-forced "$STAGE3_AGG_ROOT/metrics/teacher_forced_complete_metrics.json" \
  --development-metrics "$STAGE3_AGG_ROOT/metrics/development_aggregation_metrics.json" \
  --development-protocol "$STAGE3_AGG_ROOT/metrics/development_aggregation_protocol.json" \
  --blind-metrics "$STAGE3_AGG_ROOT/metrics/blind_aggregation_metrics.json" \
  --blind-teacher-forced "$STAGE3_AGG_ROOT/metrics/blind_teacher_forced_metrics.json" \
  --final-decision "$STAGE3_AGG_ROOT/metrics/stage3f_r_final_decision.json" \
  --output "$STAGE3_AGG_ROOT/report/stage3f_r_final_report.md"
```

脚本应允许 blind 文件不存在，并在报告中写：

```text
blind not generated because fixed development aggregation gate failed
```

## 3F-R-G.4 生成最终 leaderboard

输出：

```text
report/stage3f_r_leaderboard.csv
```

至少包含：

```text
stage3_v1_action_fused
stage3_v1_union_fused
action_raw
action_cf_only
action_current_fused
action_defect_gated
action_defect_contrast
union_defect_contrast
selected repaired method
blind selected repaired method（若生成）
```

列：

```text
split
method_id
source
pair_auroc
pair_auprc
prevalence
no_effect_far
effective_recall
top1_within_1
region_hit
mean_abs_delay
recovery_false_attribution
can_auroc
square_auroc
mean_candidates
selected
```

## 3F-R-G.5 生成轻量结果包

轻量包禁止包含：

```text
.pt
.hdf5
.npz
.faiss
.parquet
.mp4
大 JSONL oracle
```

可以包含：

```text
新增代码
配置
小型 JSON / CSV
日志尾部
最终报告
最终决策
方法 bundle
哈希清单
本操作文档
```

执行：

```bash
mkdir -p "$STAGE3_AGG_ROOT/package/lightweight"

cp -r "$SCIZOR_ROOT/stage3a/method_v02r_aggregation_repair" \
  "$STAGE3_AGG_ROOT/package/lightweight/code"
cp -r "$STAGE3_AGG_ROOT/config" \
  "$STAGE3_AGG_ROOT/package/lightweight/config"
cp -r "$STAGE3_AGG_ROOT/metrics" \
  "$STAGE3_AGG_ROOT/package/lightweight/metrics"
cp -r "$STAGE3_AGG_ROOT/report" \
  "$STAGE3_AGG_ROOT/package/lightweight/report"
cp -r "$STAGE3_AGG_ROOT/baseline_v1" \
  "$STAGE3_AGG_ROOT/package/lightweight/baseline_v1"

find "$STAGE3_AGG_ROOT/package/lightweight" -type f \
  \( -name '*.pt' -o -name '*.hdf5' -o -name '*.npz' -o \
     -name '*.faiss' -o -name '*.parquet' -o -name '*.mp4' -o \
     -name '*.jsonl' \) -delete

cd "$STAGE3_AGG_ROOT/package/lightweight"
zip -qr "$STAGE3_AGG_ROOT/package/stage3f_r_results_lightweight.zip" .
cd "$STAGE3_AGG_ROOT/package"
sha256sum stage3f_r_results_lightweight.zip \
  > stage3f_r_results_lightweight.zip.sha256
unzip -t stage3f_r_results_lightweight.zip
```

## 3F-R-G.6 最终提交

```bash
cd "$SCIZOR_ROOT"
git add stage3a/method_v02r_aggregation_repair

git -c user.name="Experiment Agent" -c user.email="agent@local" \
  commit -m "stage3F-R: finalize bounded counterfactual aggregation repair"

git rev-parse HEAD > "$STAGE3_AGG_ROOT/config/final_commit.txt"
```

### 3F-R-G 完成标准

```bash
test -s "$STAGE3_AGG_ROOT/metrics/stage3f_r_final_decision.json"
test -s "$STAGE3_AGG_ROOT/report/stage3f_r_final_report.md"
test -s "$STAGE3_AGG_ROOT/report/stage3f_r_leaderboard.csv"
test -s "$STAGE3_AGG_ROOT/package/stage3f_r_results_lightweight.zip"
test -s "$STAGE3_AGG_ROOT/package/stage3f_r_results_lightweight.zip.sha256"
test -s "$STAGE3_AGG_ROOT/config/final_commit.txt"
```

---

# 2. Agent 推荐执行顺序

严格按以下顺序：

```text
3F-R-A  冻结 v1、建分支、写死修复矩阵
    ↓
3F-R-B  补 train ensemble + 完整 256-pair teacher-forced
    ↓
3F-R-C  train no-effect background + outcome deficit + 五方法分数表
    ↓
3F-R-D  development-only 选择唯一公式和 threshold
    ├── 未通过 → 直接 3F-R-G validation-only 收尾
    └── 通过
          ↓
3F-R-E  冻结 repaired protocol、哈希和 commit
          ↓
3F-R-F  生成全新 320-pair blind，只运行一次冻结评估
          ↓
3F-R-G  最终决策、报告、leaderboard、轻量包
```

不得越过：

```text
未完成 train background 就运行 development
未冻结 development protocol 就生成 blind
blind 生成后修改公式
blind 结果出来后重新选 threshold
```

---

# 3. Agent 最终回报格式

Agent 完成后，按以下格式回报：

```markdown
## Stage 3F-R 聚合修复结果

### A：冻结与配置
- branch：
- base commit：
- repair config SHA-256：
- v1 decision preserved：

### B：完整 teacher-forced
- complete pair count：
- effective pair count：
- full AUROC / AUPRC：
- action-only AUROC / AUPRC：
- Can / Square AUROC：

### C：train background
- action Can/Square cf-max q80：
- action Can/Square contrast q80：
- union Can/Square cf-max q80：
- deficit coverage rate：

### D：development fixed matrix
- action raw AUROC：
- action counterfactual-only AUROC：
- current fused AUROC：
- action defect-gated AUROC：
- action defect-contrast AUROC：
- union defect-contrast AUROC：
- selected method：
- selected source：
- selected threshold：
- no-effect FAR：
- effective recall：
- Can / Square AUROC：
- development decision：

### E/F：blind（仅 development 通过时）
- blind pair count：
- blind positive prevalence：
- pair AUROC / AUPRC：
- no-effect FAR：
- effective recall：
- Top-1 within ±1：
- Can / Square AUROC：
- teacher-forced engineering rates：

### 最终决策
- GO_STAGE4_POLICY_VALIDATION 或 SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR
- failed rules：

### 交付
- final report：
- leaderboard：
- method bundle：
- lightweight ZIP：
- ZIP SHA-256：
- final commit：
```

---

# 4. 本轮核心判据

本轮不是再次调模型，而是验证以下研究判断：

> learned counterfactual verifier 已经能够预测单个替代动作的后果，完整方法失败是否主要来自“连续取最大值”和“缺少原轨迹 outcome deficit 门控”。

若固定的对比聚合或缺陷门控方法在 development 和全新 blind 上同时达到原门槛，则进入 Stage 4 policy validation。

若固定矩阵仍无法通过，接受结论：

```text
candidate proposal 有效
counterfactual prediction 有效
但 transition/pair responsibility aggregation 不足以形成可靠数据清洗规则
```

此时正式切换 Coverage-Constrained Soft SCIZOR，不再继续增加第三阶段复杂度。

**核心：本轮只修复最后一层责任分数聚合，不重训任何模型；development 通过后才生成 blind，blind 只运行一次。**
