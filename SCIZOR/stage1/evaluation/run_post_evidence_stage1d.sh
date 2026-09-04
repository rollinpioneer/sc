#!/usr/bin/env bash
set -euo pipefail

root=/home/__compress_data/xushijie/work/cr_scizor
stage="$root/experiments/cr_scizor/stage1"
code="$root/SCIZOR"
shards="$stage/evaluation/predictions/chunk_evidence_shards"
supervisor_pid="${EVIDENCE_SUPERVISOR_PID:?set EVIDENCE_SUPERVISOR_PID to the evidence-export supervisor PID}"

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PYTHONPATH="$code"

run_python() {
  /opt/miniconda3/bin/conda run -n scizor-curation python "$@"
}

for shard in 0 1 2 3; do
  output="$shards/shard_${shard}.parquet"
  while [[ ! -s "$output" ]]; do
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
      echo "Evidence supervisor exited before shard ${shard} was written: $output" >&2
      exit 1
    fi
    sleep 60
  done
done

run_python -m stage1.evaluation.merge_chunk_evidence \
  --shards "$shards/shard_0.parquet" "$shards/shard_1.parquet" "$shards/shard_2.parquet" "$shards/shard_3.parquet" \
  --labels "$stage/benchmark/transition_labels.parquet" \
  --output "$stage/evaluation/predictions/chunk_evidence.parquet"

run_python -m stage1.evaluation.build_attribution_baselines \
  --chunk-evidence "$stage/evaluation/predictions/chunk_evidence.parquet" \
  --transition-labels "$stage/benchmark/transition_labels.parquet" \
  --gamma-per-second 0.5 \
  --output-dir "$stage/evaluation/predictions"

run_python -m stage1.evaluation.merge_predictions \
  --original "$stage/evaluation/predictions/original_scizor_scores.parquet" \
  --uniform "$stage/evaluation/predictions/uniform_scores.parquet" \
  --future "$stage/evaluation/predictions/future_discount_scores.parquet" \
  --labels "$stage/benchmark/transition_labels.parquet" \
  --output "$stage/evaluation/predictions/all_predictions.parquet"

run_python -m stage1.evaluation.select_operating_point \
  --predictions "$stage/evaluation/predictions/all_predictions.parquet" \
  --reference-method original_scizor \
  --reference-percentile 0.70 \
  --split validation \
  --output "$stage/evaluation/metrics/operating_points.json"

run_python -m stage1.evaluation.evaluate_localization \
  --predictions "$stage/evaluation/predictions/all_predictions.parquet" \
  --operating-points "$stage/evaluation/metrics/operating_points.json" \
  --split test \
  --bootstrap-samples 500 \
  --seed 20260831 \
  --output-json "$stage/evaluation/metrics/test_metrics.json" \
  --output-csv "$stage/evaluation/metrics/test_metrics.csv"

run_python -m stage1.evaluation.export_error_cases \
  --benchmark-hdf5 "$stage/evaluation/scoring_input/benchmark_v0.1.hdf5" \
  --predictions "$stage/evaluation/predictions/all_predictions.parquet" \
  --operating-points "$stage/evaluation/metrics/operating_points.json" \
  --output-dir "$stage/evaluation/cases" \
  --per-case-type 5

run_python -m stage1.evaluation.make_stage1_report \
  --benchmark-stats "$stage/benchmark/benchmark_stats.json" \
  --split-manifest "$stage/benchmark/split_manifest.json" \
  --operating-points "$stage/evaluation/metrics/operating_points.json" \
  --metrics "$stage/evaluation/metrics/test_metrics.json" \
  --repo-commit "$stage/config/repo_commit.txt" \
  --checkpoint-hash "$stage/baseline/frozen/checkpoint.sha256" \
  --leaderboard "$stage/evaluation/leaderboard.csv" \
  --report "$stage/evaluation/stage1_report.md"

archive="$stage/artifacts/stage1D_results_lightweight.zip"
rm -f "$archive"
cd "$stage"
zip -rq "$archive" \
  benchmark/ANNOTATION_SPEC.md benchmark/DATA_CARD.md benchmark/STAGE1C_FREEZE_NOTES.md \
  benchmark/split_manifest.json benchmark/split_manifest.sha256 benchmark/stage1c_final_audit.json \
  evaluation/metrics evaluation/cases evaluation/leaderboard.csv evaluation/stage1_report.md \
  evaluation/predictions/chunk_evidence.parquet \
  evaluation/predictions/original_scizor_scores.parquet \
  evaluation/predictions/uniform_scores.parquet \
  evaluation/predictions/future_discount_scores.parquet \
  evaluation/predictions/all_predictions.parquet
unzip -t "$archive" >/dev/null
echo "STAGE1D_COMPLETE archive=$archive"
