#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${OUT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_stop_continue_v1}"
PROBE="${PROBE:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1}"
WT="${WT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb3p-stop-continue-v1}"
CODE="${CODE:-$WT/SCIZOR}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
GPU_SIM="${GPU_SIM:-1}"
DEVICE="${DEVICE:-cuda}"
CONFIG="$OUT/config/stop_continue.json"
DRAFT="$OUT/config/protocol.draft.json"
PROTOCOL="$OUT/config/frozen_protocol.json"
STATUS="$OUT/status"
LOGS="$OUT/logs"

export PYTHONPATH="$CODE/robomimic:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS"
printf '%s\n' "$$" > "$STATUS/stop-continue-supervisor.pid"

stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
fail() { code=$?; rm -f "$STATUS/stop-continue.running"; printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/stop-continue.failed"; exit "$code"; }
trap 'fail $LINENO' ERR

rm -f "$STATUS/stop-continue.failed" "$STATUS/stop-continue.done"
mark stop-continue.running "pid=$$ gpu=$GPU_SIM"

"$PY_SIM" -m compileall -q "$CODE/recovery_handback/hb3p_stop"
git -C "$WT" diff --check -- SCIZOR/recovery_handback/hb3p_stop

mark prepare.done "prepare frozen pilot directories"
"$PY_SIM" -m recovery_handback.hb3p_stop.prepare \
  --probe-root "$PROBE" --output-root "$OUT" --code-root "$CODE" \
  > "$LOGS/prepare.log" 2>&1

mark dataset.done "build paired L60/L80 labels"
"$PY_SIM" -m recovery_handback.hb3p_stop.dataset \
  --probe-root "$PROBE" --output-dir "$OUT/dataset" \
  > "$LOGS/dataset.log" 2>&1

mark train.running "train OOF/final stop-continue MLP"
CUDA_VISIBLE_DEVICES="${GPU_TRAIN:-$GPU_SIM}" \
  "$PY_SIM" -m recovery_handback.hb3p_stop.train \
  --dataset-dir "$OUT/dataset" --output-dir "$OUT/model" --device "$DEVICE" \
  > "$LOGS/train.log" 2>&1
mark train.done "stop-continue model trained"

mark code-freeze.required "commit hb3p_stop code before running freeze.py"
"$PY_SIM" -m recovery_handback.hb3p_stop.freeze \
  --config "$CONFIG" --draft "$DRAFT" --training-summary "$OUT/model/training_summary.json" \
  --code-root "$CODE/recovery_handback/hb3p_stop" --output "$PROTOCOL" \
  > "$LOGS/freeze.log" 2>&1

mark collect.running "collect independent label-free test roots"
CUDA_VISIBLE_DEVICES="$GPU_SIM" MUJOCO_EGL_DEVICE_ID="$GPU_SIM" \
  "$PY_SIM" -m recovery_handback.hb3p_stop.collect \
  --config "$CONFIG" --protocol "$PROTOCOL" --output-dir "$OUT/roots" --resume \
  > "$LOGS/collect.log" 2>&1
mark collect.done "40 test roots collected"

mark online.running "run NONE, FIXED_L60, FIXED_L80, and learned methods"
CUDA_VISIBLE_DEVICES="$GPU_SIM" MUJOCO_EGL_DEVICE_ID="$GPU_SIM" \
  "$PY_SIM" -m recovery_handback.hb3p_stop.run_online \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$OUT/roots/roots/runtime_root_manifest.jsonl" \
  --num-shards 1 --shard-index 0 --output-dir "$OUT/episodes" --device "$DEVICE" --resume \
  > "$LOGS/online.log" 2>&1
mark online.done "all stop-continue episodes complete"

"$PY_SIM" -m recovery_handback.hb3p_stop.aggregate \
  --roots "$OUT/roots/roots/runtime_root_manifest.jsonl" --episodes-root "$OUT/episodes" \
  --protocol "$PROTOCOL" --output-dir "$OUT/metrics" \
  > "$LOGS/aggregate.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p_stop.evaluate \
  --episodes "$OUT/metrics/episodes.jsonl" --coverage "$OUT/metrics/coverage.json" \
  --protocol "$PROTOCOL" --output-dir "$OUT/metrics" \
  > "$LOGS/evaluate.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p_stop.report \
  --config "$CONFIG" --protocol "$PROTOCOL" --dataset-summary "$OUT/dataset/dataset_manifest.json" \
  --training-summary "$OUT/model/training_summary.json" --coverage "$OUT/metrics/coverage.json" \
  --evaluation-summary "$OUT/metrics/summary.json" --decision "$OUT/metrics/decision.json" \
  --output "$OUT/report/HB3P_STOP_CONTINUE.md" \
  > "$LOGS/report.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p_stop.package \
  --experiment-root "$OUT" --code-root "$CODE" --output "$OUT/package/HB3P_stop_continue_lightweight.zip" \
  > "$LOGS/package.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p_stop.export_lightweight \
  --experiment-root "$OUT" --destination "$WT/experiments/handback/hb3p_stop_continue_v1" \
  > "$LOGS/export.log" 2>&1
mark stop-continue.done "report package and lightweight Git export complete"
rm -f "$STATUS/stop-continue.running"
