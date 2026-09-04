#!/usr/bin/env bash
set -euo pipefail

stage=/home/__compress_data/xushijie/work/cr_scizor/experiments/cr_scizor/stage1
postprocess_pid="${POSTPROCESS_PID:?set POSTPROCESS_PID to the Stage 1D postprocess PID}"
log="$stage/logs/S1-D-archive-finalize.log"

while kill -0 "$postprocess_pid" 2>/dev/null; do
  sleep 60
done

required=(
  "$stage/evaluation/predictions/chunk_evidence.parquet"
  "$stage/evaluation/predictions/original_scizor_scores.parquet"
  "$stage/evaluation/predictions/uniform_scores.parquet"
  "$stage/evaluation/predictions/future_discount_scores.parquet"
  "$stage/evaluation/predictions/all_predictions.parquet"
  "$stage/evaluation/metrics/operating_points.json"
  "$stage/evaluation/metrics/test_metrics.json"
  "$stage/evaluation/metrics/test_metrics.csv"
  "$stage/evaluation/leaderboard.csv"
  "$stage/evaluation/stage1_report.md"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "missing Stage 1D artifact: $path" >&2; exit 1; }
done

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
echo "STAGE1D_ARCHIVE_FINALIZED archive=$archive" | tee -a "$log"
