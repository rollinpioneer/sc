#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
ROOT_DIR="$RUN_ROOT/data/formal_test_roots"
ROOTS="$ROOT_DIR/roots.jsonl"
LOG_ROOT="$RUN_ROOT/logs/formal"
DECISION="$EXPORT_ROOT/metrics/development/decision.json"
mkdir -p "$ROOT_DIR" "$LOG_ROOT" "$EXPORT_ROOT/metrics/test"
exec 9>"$RUN_ROOT/logs/formal_matrix.lock"
flock -n 9 || { echo "formal matrix already running"; exit 0; }
exec > >(tee -a "$RUN_ROOT/logs/formal_matrix.log") 2>&1

while [[ ! -f "$DECISION" ]]; do sleep 30; done
decision="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("decision", ""))' "$DECISION")"
if [[ "$decision" != "HB4_DEVELOPMENT_GO" ]]; then
  echo "formal test not unlocked: $decision"
  exit 0
fi

if [[ ! -f "$ROOTS" ]] || [[ "$(wc -l < "$ROOTS")" -lt 400 ]]; then
  env CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false USE_TF=0 "$PYTHON" -m recovery_handback.hb4.collect_eval_roots \
    --start-seed 940000 --count 400 --role HB4_test --output-dir "$ROOT_DIR" > "$LOG_ROOT/root_collection.log" 2>&1
fi
while [[ ! -f "$ROOTS" ]] || [[ "$(wc -l < "$ROOTS")" -lt 400 ]]; do sleep 30; done

env PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" "$PYTHON" -m recovery_handback.hb4.freeze_protocol \
  --export-root "$EXPORT_ROOT" --run-root "$RUN_ROOT" --roots "$ROOTS" --test

run_eval() {
  local method="$1" seed="$2" checkpoint="$3" output="$4"
  if [[ -f "$output/summary.json" ]]; then
    local roots_done
    roots_done="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("roots", 0))' "$output/summary.json" 2>/dev/null || printf '0')"
    local engineering_failures
    engineering_failures="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("engineering_failures", 1))' "$output/summary.json" 2>/dev/null || printf '1')"
    if [[ "$roots_done" -ge 400 && "$engineering_failures" -eq 0 ]]; then
      return 0
    fi
  fi
  mkdir -p "$output"
  local args=(--roots "$ROOTS" --checkpoint "$checkpoint" --method "$method" --output-dir "$output" --device cuda:0 --protocol "$EXPORT_ROOT/config/test_frozen_protocol.json")
  [[ "$seed" == "null" ]] || args+=(--training-seed "$seed")
  env CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false USE_TF=0 "$PYTHON" -m recovery_handback.hb4.evaluate "${args[@]}" \
    > "$LOG_ROOT/${method}_${seed}.log" 2>&1
}

run_eval BASE_FROZEN null /tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth "$EXPORT_ROOT/metrics/test/BASE_FROZEN"
for arm in REPLAY_ONLY MATCHED_STANDARD_DATA FIXED_L80_RECOVERY HANDOFF_RECOVERY; do
  for seed in 0 1 2; do
    run_eval "$arm" "$seed" "$RUN_ROOT/training/$arm/seed$seed/model_step_4000.pth" "$EXPORT_ROOT/metrics/test/${arm}_seed${seed}"
  done
done

env PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" "$PYTHON" -m recovery_handback.hb4.analyze \
  --records-root "$EXPORT_ROOT/metrics/test" --output-dir "$EXPORT_ROOT/metrics/test" --expected-roots 400 --mode formal
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
