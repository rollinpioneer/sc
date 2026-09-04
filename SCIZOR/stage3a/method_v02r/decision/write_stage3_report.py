"""Write the ordered, evidence-linked Stage 3 v0.2-R final report.

The report is assembled from frozen JSON/CSV artifacts already produced by the
pipeline.  Missing blind artifacts are represented explicitly for the
validation-only path; this writer never invents metrics or changes a decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def text(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric(mapping: dict | None, *keys, default=None):
    current = mapping or {}
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def existing(root: Path, relative: str) -> str:
    path = root / relative
    return f"`{relative}`" if path.exists() else f"`{relative}` (not generated)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.method_root
    metrics_root = root / "metrics"
    config_root = root / "config"
    report_root = root / "report"

    confirmation = load(config_root / "v02r_confirmation_decision.json") or {}
    transfer = load(metrics_root / "proposer_transfer_v02.json") or {}
    learning = load(metrics_root / "validation_verifier_learning.json") or {}
    validation_pipeline = load(metrics_root / "validation_pipeline_metrics.json") or {}
    protocol = load(metrics_root / "validation_frozen_protocol.json") or {}
    blind_pipeline = load(metrics_root / "blind_test_pipeline_metrics.json") or {}
    blind_teacher = load(metrics_root / "blind_test_teacher_forced_metrics.json") or {}
    decision = load(metrics_root / "stage3_final_decision.json") or {}

    # The sibling replay-rescue reports are the authoritative references for
    # the v0.1 replay failure and v0.2 engineering rescue.  We link to their
    # paths without copying stale prose into this report.
    sibling_root = root.parent
    v02_report = sibling_root / "replay_rescue_v02" / "report" / "stage3a_v02_replay_rescue_report.md"
    v02r_report = sibling_root / "replay_rescue_v02_r1" / "report" / "stage3a_v02r_report.md"
    selected = protocol.get("selected_proposer") or blind_pipeline.get("selected_proposer")
    selected_threshold = protocol.get("selected_threshold")
    if selected_threshold is None:
        selected_threshold = blind_pipeline.get("selected_threshold")
    final_decision = decision.get("decision", "not available")

    lines = [
        "# SCIZOR Stage 3 v0.2-R 最终报告",
        "",
        "> 本报告只汇总已生成的冻结证据；解释性案例不参与阈值、模型或最终决策。",
        "",
        "## 1. v0.1 重放失败与 v0.2 工程救援结论",
        "",
        f"- v0.1 重放/数据血缘报告：{v02_report if v02_report.exists() else '未找到'}。",
        f"- v0.2 replay-locked rescue 报告：{v02r_report if v02r_report.exists() else '未找到'}。",
        "- 本阶段沿用 v0.2 的显式 pre/post state、当前 mimicgen runtime 和冻结的动作/扰动定义；不读取或重放 v0.1 test。",
        f"- v0.2-R confirmation decision：`{text(confirmation.get('decision'))}`；engineering_pass=`{text(confirmation.get('engineering_pass'))}`；method_pass=`{text(confirmation.get('method_pass'))}`。",
        "",
        "## 2. v0.2-R oracle confirmation 结果",
        "",
        f"- paired-clean confirmation AUROC：`{text(confirmation.get('paired_clean_confirmation_auroc'))}`。",
        f"- primary feasible confirmation AUROC：`{text(confirmation.get('primary_feasible_confirmation_auroc'))}`。",
        f"- confirmation failed_rules：`{text(confirmation.get('failed_rules', []))}`。",
        "- confirmation 只作为工程和 oracle 前置证据，不进入 validation proposer、checkpoint 或 operating threshold 选择。",
        "",
        "## 3. proposer 在 v0.2 的迁移表现",
        "",
        f"- full/action/union transfer summary：{existing(root, 'metrics/proposer_transfer_v02.json')}。",
        f"- train transfer：`{text(metric(transfer, 'train'))}`。",
        f"- validation transfer：`{text(metric(transfer, 'validation'))}`。",
        "- proposer 仅产生冻结 Top-5 候选；长期反事实 oracle/verifier 才承担候选有效性判断。",
        "",
        "## 4. verifier 三 seed 结果与 action-only 诊断",
        "",
        f"- verifier learning metrics：{existing(root, 'metrics/validation_verifier_learning.json')}。",
        f"- candidate replacement full AUROC/AUPRC：`{text(metric(learning, 'candidate_replacement', 'full', 'auroc'))}` / `{text(metric(learning, 'candidate_replacement', 'full', 'auprc'))}`。",
        f"- teacher-forced primary full AUROC/AUPRC：`{text(metric(learning, 'teacher_forced_primary', 'full', 'auroc'))}` / `{text(metric(learning, 'teacher_forced_primary', 'full', 'auprc'))}`。",
        f"- full vs action-only replacement AUPRC difference：`{text(metric(learning, 'full_vs_action_only', 'replacement_auprc_difference_full_minus_action'))}`。",
        f"- full/action-only matched-recall no-effect FAR：`{text(metric(learning, 'full_vs_action_only', 'full_matched_recall_no_effect_far'))}` / `{text(metric(learning, 'full_vs_action_only', 'action_only_matched_recall_no_effect_far'))}`。",
        "- teacher-forced primary 只使用 `is_teacher_forced=true`, `query_t=intervention_t`, `replacement_rank=0`。",
        "",
        "## 5. validation protocol",
        "",
        f"- validation gate：`{text(protocol.get('validation_gate_pass'))}`。",
        f"- selected proposer：`{text(selected)}`。",
        f"- selected score：`{text(protocol.get('selected_score', 'fused_transition_score'))}`；frozen threshold：`{text(selected_threshold)}`。",
        f"- validation pipeline metrics：{existing(root, 'metrics/validation_pipeline_metrics.json')}。",
        f"- selected validation pair AUROC/AUPRC：`{text(metric(validation_pipeline, 'sources', selected, 'auroc'))}` / `{text(metric(validation_pipeline, 'sources', selected, 'auprc'))}`。",
        f"- validation failed_rules：`{text(protocol.get('failed_rules', validation_pipeline.get('failed_rules', [])))}`。",
        "- 该 protocol 在 blind 读取前冻结；不得重新搜索 threshold、Top-k、模型结构、学习率或融合系数。",
        "",
        "## 6. blind test 完整流程",
        "",
        f"- blind benchmark pipeline metrics：{existing(root, 'metrics/blind_test_pipeline_metrics.json')}。",
        f"- blind teacher-forced metrics：{existing(root, 'metrics/blind_test_teacher_forced_metrics.json')}。",
        f"- benchmark check pass：`{text(blind_pipeline.get('benchmark_check_pass'))}`。",
        f"- blind selected proposer/threshold：`{text(blind_pipeline.get('selected_proposer', selected))}` / `{text(blind_pipeline.get('selected_threshold', selected_threshold))}`。",
        f"- blind pair count：`{text(blind_pipeline.get('benchmark_pair_count'))}`；blind benchmark and full oracle artifacts are listed in `report/large_artifact_manifest.json`。",
        "- blind 数据只在 protocol、checkpoint、normalizer、proposal calibration 全部冻结后读取；blind 结果不回流训练或选择。",
        "",
        "## 7. no-effect、recovery、Can、Square 分组",
        "",
        f"- blind pair no-effect FAR / effective recall / Top-1 within ±1：`{text(metric(blind_pipeline, 'pair_metrics', 'no_effect_far'))}` / `{text(metric(blind_pipeline, 'pair_metrics', 'effective_recall'))}` / `{text(metric(blind_pipeline, 'pair_metrics', 'top1_within_1'))}`。",
        f"- blind recovery false-attribution：`{text(metric(blind_pipeline, 'pair_metrics', 'recovery_false_attribution'))}`。",
        f"- blind Can/Square pair AUROC：`{text(metric(blind_pipeline, 'pair_metrics', 'by_task', 'can', 'auroc'))}` / `{text(metric(blind_pipeline, 'pair_metrics', 'by_task', 'square', 'auroc'))}`。",
        f"- teacher-forced engineering branch/reference/finite rates：`{text(metric(blind_teacher, 'engineering', 'branch_pre_state_equal_rate'))}` / `{text(metric(blind_teacher, 'engineering', 'reference_exact_rate'))}` / `{text(metric(blind_teacher, 'engineering', 'finite_target_rate'))}`。",
        f"- teacher-forced oracle-positive AUROC：`{text(metric(blind_teacher, 'replacement_oracle_positive', 'auroc'))}`；primary effective-vs-no_effect AUROC：`{text(metric(blind_teacher, 'primary_effective_vs_no_effect', 'auroc'))}`。",
        "",
        "## 8. 最终决策",
        "",
        f"- **`{final_decision}`**。",
        f"- decision artifact：{existing(root, 'metrics/stage3_final_decision.json')}。",
        f"- failed rules：`{text(decision.get('failed_rules', []))}`。",
        "",
        "## 9. 下一步（仅在 GO 时）",
        "",
        "若最终决策为 `GO_STAGE4_POLICY_VALIDATION`，下一步只运行相同数据预算下的 policy 训练；Stage 4 必须使用冻结 method bundle，不得重新选择 threshold。",
        "",
        "## 10. 失败时的切换方案",
        "",
        "若最终决策不是 GO，则切换 `SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR`，不进行第二轮大范围 verifier、Top-k、学习率或阈值实验。",
        "",
        "## 交付索引",
        "",
        f"- leaderboard：{existing(root, 'report/stage3_leaderboard.csv')}。",
        f"- representative cases：{existing(root, 'report/cases/case_index.csv')}。",
        f"- large-artifact manifest：{existing(root, 'report/large_artifact_manifest.json')}。",
        f"- method bundle：{existing(root, 'report/stage3_method_bundle.json')}。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": final_decision, "validation_gate_pass": protocol.get("validation_gate_pass")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
