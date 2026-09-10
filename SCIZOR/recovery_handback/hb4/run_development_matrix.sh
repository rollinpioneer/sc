#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
ROOTS="$RUN_ROOT/data/development_roots/roots.jsonl"
PROTOCOL="$EXPORT_ROOT/config/development_frozen_protocol.json"
LOG_ROOT="$RUN_ROOT/logs/development"
mkdir -p "$LOG_ROOT" "$EXPORT_ROOT/metrics/development"
exec 9>"$RUN_ROOT/logs/development_matrix.lock"
flock -n 9 || { echo "development matrix already running"; exit 0; }
exec > >(tee -a "$RUN_ROOT/logs/development_matrix.log") 2>&1

while [[ ! -f "$ROOTS" ]] || [[ "$(wc -l < "$ROOTS")" -lt 80 ]]; do sleep 30; done
for arm in REPLAY_ONLY MATCHED_STANDARD_DATA FIXED_L80_RECOVERY HANDOFF_RECOVERY; do
  for seed in 0 1 2; do
    while [[ ! -f "$RUN_ROOT/training/$arm/seed$seed/training_summary.json" ]]; do sleep 30; done
  done
done

env PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" "$PYTHON" -m recovery_handback.hb4.freeze_protocol \
  --export-root "$EXPORT_ROOT" --run-root "$RUN_ROOT" --roots "$ROOTS" --output-name development_frozen_protocol.json

run_eval() {
  local method="$1" seed="$2" checkpoint="$3" output="$4"
  if [[ -f "$output/summary.json" ]]; then
    local roots_done
    roots_done="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("roots", 0))' "$output/summary.json" 2>/dev/null || printf '0')"
    local engineering_failures
    engineering_failures="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("engineering_failures", 1))' "$output/summary.json" 2>/dev/null || printf '1')"
    if [[ "$roots_done" -ge 80 && "$engineering_failures" -eq 0 ]]; then
      return 0
    fi
  fi
  mkdir -p "$output"
  local args=(--roots "$ROOTS" --checkpoint "$checkpoint" --method "$method" --output-dir "$output" --device cuda:0 --protocol "$PROTOCOL")
  [[ "$seed" == "null" ]] || args+=(--training-seed "$seed")
  env CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false USE_TF=0 "$PYTHON" -m recovery_handback.hb4.evaluate "${args[@]}" \
    > "$LOG_ROOT/${method}_${seed}.log" 2>&1
}

run_eval BASE_FROZEN null /tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth "$EXPORT_ROOT/metrics/development/BASE_FROZEN"
for arm in REPLAY_ONLY MATCHED_STANDARD_DATA FIXED_L80_RECOVERY HANDOFF_RECOVERY; do
  for seed in 0 1 2; do
    run_eval "$arm" "$seed" "$RUN_ROOT/training/$arm/seed$seed/model_step_4000.pth" "$EXPORT_ROOT/metrics/development/${arm}_seed${seed}"
  done
done

env PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" "$PYTHON" -m recovery_handback.hb4.analyze \
  --records-root "$EXPORT_ROOT/metrics/development" --output-dir "$EXPORT_ROOT/metrics/development" --expected-roots 80 --mode development
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
