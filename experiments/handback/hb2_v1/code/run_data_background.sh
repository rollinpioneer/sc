#!/usr/bin/env bash
set -Eeuo pipefail

HB2_ROOT="${HB2_ROOT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb2_v1}"
CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb2-square-v1/SCIZOR}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
CONFIG="$HB2_ROOT/config/hb2.json"
PAIR="$HB2_ROOT/assets/policy_pair_square.json"
STATUS="$HB2_ROOT/status"
LOGS="$HB2_ROOT/logs"

export PYTHONPATH="$CODE_ROOT/robomimic:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS"
printf '%s\n' "$$" > "$LOGS/hb2-data-supervisor.pid"

stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
fail() {
  code=$?
  printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/hb2-data.failed"
  exit "$code"
}
trap 'fail $LINENO' ERR

run_gpu() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$@"
}

collect_role() {
  local role="$1" count="$2" gpu="$3"
  run_gpu "$gpu" "$PY_SIM" -m recovery_handback.hb2.collect \
    --config "$CONFIG" --role "$role" --start-index 0 --count "$count" \
    --output-dir "$HB2_ROOT/roots/$role" \
    > "$LOGS/collect_${role}.log" 2>&1
}

branches_role() {
  local role="$1" shard="$2" gpu="$3"
  run_gpu "$gpu" "$PY_SIM" -m recovery_handback.execution.run_branches \
    --config "$CONFIG" --policy-pair "$PAIR" \
    --anchors "$HB2_ROOT/roots/$role/anchors.parquet" --role "$role" \
    --output-dir "$HB2_ROOT/branches/$role/shard_$(printf '%03d' "$shard")" \
    --num-shards 2 --shard-index "$shard" --resume \
    --base-device cuda --repair-device cuda \
    > "$LOGS/branches_${role}_$(printf '%03d' "$shard").log" 2>&1
}

aggregate_role() {
  local role="$1"
  "$PY_SIM" -m recovery_handback.hb2.aggregate \
    --config "$CONFIG" --role "$role" \
    --anchors "$HB2_ROOT/roots/$role/anchors.parquet" \
    --branches-root "$HB2_ROOT/branches/$role" \
    --output-dir "$HB2_ROOT/data/$role" \
    > "$LOGS/aggregate_${role}.log" 2>&1
}

run_pair_and_wait() {
  "$@" &
  local pid_a=$!
  shift 3
  "$@" &
  local pid_b=$!
  printf '%s\n' "$pid_a" "$pid_b" > "$LOGS/hb2-current-workers.pid"
  wait "$pid_a"
  wait "$pid_b"
}

mark hb2-data.running "pid=$$"

mark pilot-collect.running "gpu=6 roots=4"
collect_role hb2_pilot 4 6
mark pilot-collect.done "roots and anchors complete"

"$PY_SIM" -c \
  'from pathlib import Path; from recovery_handback.common import read_table, write_table; p=Path("'"$HB2_ROOT"'/roots/hb2_pilot/anchors.parquet"); rows=read_table(p); assert rows, "pilot has no legal anchor"; write_table(rows[:1], p.with_name("anchors_first.parquet"))'

mark pilot-first-anchor.running "gpu=6 one anchor, five branches"
run_gpu 6 "$PY_SIM" -m recovery_handback.execution.run_branches \
  --config "$CONFIG" --policy-pair "$PAIR" \
  --anchors "$HB2_ROOT/roots/hb2_pilot/anchors_first.parquet" --role hb2_pilot \
  --output-dir "$HB2_ROOT/branches/hb2_pilot/shard_000" \
  --num-shards 1 --shard-index 0 --resume \
  --base-device cuda --repair-device cuda \
  > "$LOGS/branches_hb2_pilot_first.log" 2>&1
mark pilot-first-anchor.done "one anchor interface complete"

mark pilot-all.running "gpu=6 all pilot anchors"
run_gpu 6 "$PY_SIM" -m recovery_handback.execution.run_branches \
  --config "$CONFIG" --policy-pair "$PAIR" \
  --anchors "$HB2_ROOT/roots/hb2_pilot/anchors.parquet" --role hb2_pilot \
  --output-dir "$HB2_ROOT/branches/hb2_pilot/shard_000" \
  --num-shards 1 --shard-index 0 --resume \
  --base-device cuda --repair-device cuda \
  > "$LOGS/branches_hb2_pilot_all.log" 2>&1
aggregate_role hb2_pilot
mark pilot-all.done "pilot aggregation complete"

mark train-val-collect.running "train gpu=1 validation gpu=2"
collect_role hb2_train 40 1 & pid_train=$!
collect_role hb2_val 40 2 & pid_val=$!
printf '%s\n' "$pid_train" "$pid_val" > "$LOGS/hb2-current-workers.pid"
wait "$pid_train"
wait "$pid_val"
mark train-val-collect.done "80 roots collected"

mark train-val-branches.running "two shards per role on gpus 1-4"
branches_role hb2_train 0 1 & pid_t0=$!
branches_role hb2_train 1 2 & pid_t1=$!
branches_role hb2_val 0 3 & pid_v0=$!
branches_role hb2_val 1 4 & pid_v1=$!
printf '%s\n' "$pid_t0" "$pid_t1" "$pid_v0" "$pid_v1" > "$LOGS/hb2-current-workers.pid"
wait "$pid_t0"
wait "$pid_t1"
wait "$pid_v0"
wait "$pid_v1"
aggregate_role hb2_train
aggregate_role hb2_val
mark train-val-branches.done "paired data aggregated"

"$PY_SIM" -m recovery_handback.hb2.freeze_dataset \
  --config "$CONFIG" --legacy "$HB2_ROOT/data/legacy_train" \
  --train "$HB2_ROOT/data/hb2_train" --validation "$HB2_ROOT/data/hb2_val" \
  --output-dir "$HB2_ROOT/data/frozen" \
  > "$LOGS/freeze_dataset.log" 2>&1

status="$($PY_SIM -c 'import json; print(json.load(open("'"$HB2_ROOT"'/data/frozen/data_sufficiency.json"))["status"])')"
if [[ "$status" == "EXPANSION_REQUIRED" ]]; then
  mark expansion.running "single allowed full-block expansion"
  collect_role hb2_train_extra 40 1 & pid_train=$!
  collect_role hb2_val_extra 20 2 & pid_val=$!
  printf '%s\n' "$pid_train" "$pid_val" > "$LOGS/hb2-current-workers.pid"
  wait "$pid_train"
  wait "$pid_val"
  branches_role hb2_train_extra 0 1 & pid_t0=$!
  branches_role hb2_train_extra 1 2 & pid_t1=$!
  branches_role hb2_val_extra 0 3 & pid_v0=$!
  branches_role hb2_val_extra 1 4 & pid_v1=$!
  printf '%s\n' "$pid_t0" "$pid_t1" "$pid_v0" "$pid_v1" > "$LOGS/hb2-current-workers.pid"
  wait "$pid_t0"
  wait "$pid_t1"
  wait "$pid_v0"
  wait "$pid_v1"
  aggregate_role hb2_train_extra
  aggregate_role hb2_val_extra
  "$PY_SIM" -m recovery_handback.hb2.freeze_dataset \
    --config "$CONFIG" --legacy "$HB2_ROOT/data/legacy_train" \
    --train "$HB2_ROOT/data/hb2_train" --validation "$HB2_ROOT/data/hb2_val" \
    --train-extra "$HB2_ROOT/data/hb2_train_extra" \
    --validation-extra "$HB2_ROOT/data/hb2_val_extra" \
    --output-dir "$HB2_ROOT/data/frozen" \
    >> "$LOGS/freeze_dataset.log" 2>&1
  mark expansion.done "single expansion frozen"
fi

final_status="$($PY_SIM -c 'import json; print(json.load(open("'"$HB2_ROOT"'/data/frozen/data_sufficiency.json"))["status"])')"
mark hb2-data.done "status=$final_status"
exit 0
