#!/usr/bin/env bash
set -euo pipefail

stage_root=/home/__compress_data/xushijie/work/cr_scizor/experiments/cr_scizor/stage1
scizor_root=/home/__compress_data/xushijie/work/cr_scizor/SCIZOR
conda_bin=/opt/miniconda3/bin/conda
export PYTHONPATH="$scizor_root${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

mkdir -p "$stage_root/evaluation/scoring_input" "$stage_root/evaluation/predictions" "$stage_root/logs"
echo "SUPERVISOR_START $(date -Is) pid=$$"
if [ ! -e "$stage_root/evaluation/scoring_input/benchmark_v0.1.hdf5" ]; then
  cp --reflink=auto "$stage_root/benchmark/benchmark_v0.1.hdf5" "$stage_root/evaluation/scoring_input/benchmark_v0.1.hdf5"
fi

CUDA_VISIBLE_DEVICES=0 "$conda_bin" run -n scizor-curation python "$scizor_root/stage1/src/score_hdf5.py" \
  --data-dir "$stage_root/evaluation/scoring_input" --model-path "$stage_root/baseline/frozen" \
  --goal-time 2 --batch-size 256 --image-key agentview_image \
  > "$stage_root/logs/S1-D-score-benchmark.log" 2>&1 &
score_pid=$!
CUDA_VISIBLE_DEVICES=1 "$conda_bin" run -n scizor-curation python "$scizor_root/stage1/src/export_chunk_evidence.py" \
  --data-dir "$stage_root/evaluation/scoring_input" --model-path "$stage_root/baseline/frozen" \
  --goal-time 2 --batch-size 256 --image-key agentview_image \
  --output "$stage_root/evaluation/predictions/chunk_evidence.parquet" \
  > "$stage_root/logs/S1-D-chunk-evidence.log" 2>&1 &
evidence_pid=$!

wait "$score_pid"
"$conda_bin" run -n scizor-curation python "$scizor_root/stage1/src/export_transition_scores.py" \
  --data-dir "$stage_root/evaluation/scoring_input" \
  --output "$stage_root/evaluation/predictions/original_scizor_scores.parquet" \
  > "$stage_root/logs/S1-D-export-original.log" 2>&1
wait "$evidence_pid"
touch "$stage_root/evaluation/predictions/scoring_complete"
echo "SUPERVISOR_COMPLETE $(date -Is)"
