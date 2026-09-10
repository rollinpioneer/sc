#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
GPU_ID="${GPU_ID:-4}"
LOG_ROOT="$RUN_ROOT/logs/training"

mkdir -p "$LOG_ROOT"
exec 9>"$RUN_ROOT/logs/training_matrix.lock"
flock -n 9 || { echo "training matrix already running"; exit 0; }
exec > >(tee -a "$RUN_ROOT/logs/training_matrix.log") 2>&1

while :; do
  if [[ -f "$EXPORT_ROOT/data/matching_audit.json" && -f "$EXPORT_ROOT/data/training_hdf5_manifest.json" && -f "$RUN_ROOT/data/pilot/pilot_summary.json" ]] && \
    "$PYTHON" - "$EXPORT_ROOT/data/matching_audit.json" "$RUN_ROOT/data/pilot/pilot_summary.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))
raise SystemExit(0 if a.get("status") == "PASS_HB4_C_READY" and p.get("status") == "PASS_HB4_D" else 1)
PY
  then
    break
  fi
  sleep 30
done

for seed in 0 1 2; do
  for arm in REPLAY_ONLY MATCHED_STANDARD_DATA FIXED_L80_RECOVERY HANDOFF_RECOVERY; do
    out="$RUN_ROOT/training/$arm/seed$seed"
    mkdir -p "$out"
    if [[ -f "$out/training_summary.json" ]]; then
      echo "skip complete arm=$arm seed=$seed"
      continue
    fi
    echo "start arm=$arm seed=$seed"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU_ID" \
      PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
      OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false USE_TF=0 \
      "$PYTHON" -m recovery_handback.hb4.train \
      --arm "$arm" --seed "$seed" --device cuda:0 --output-dir "$out" \
      > "$LOG_ROOT/${arm}_seed${seed}.log" 2>&1
    echo "done arm=$arm seed=$seed"
  done
done

printf 'finished=%s\n' "$(date --iso-8601=seconds)"
