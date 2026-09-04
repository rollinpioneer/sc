#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
CONFIG="${1:-experiments/stage1/configs/mvp_smoke.yaml}"
SEED="${2:-0}"
RUN_KIND="${RUN_KIND:-smoke}"
EXP_ID="S1_12_${RUN_KIND}_robomimic_can_td5_budget_unfrozen_seed${SEED}_$(date +%Y%m%d-%H%M%S)"
OUT_DIR="outputs/stage1/$EXP_ID"
mkdir -p "$OUT_DIR"/{logs,influence,selection,policy,rollout,metrics}
cp "$CONFIG" "$OUT_DIR/resolved_config.yaml"
COMMAND="$0 $*"
python experiments/stage1/tools/write_run_manifest.py --config "$OUT_DIR/resolved_config.yaml" --experiment-id "$EXP_ID" --output "$OUT_DIR/run_manifest.json" --command "$COMMAND" --status started
if ! python experiments/stage1/tools/preflight_stage1.py --config "$OUT_DIR/resolved_config.yaml" 2>&1 | tee "$OUT_DIR/logs/run.log"; then
  python experiments/stage1/tools/write_run_manifest.py --config "$OUT_DIR/resolved_config.yaml" --experiment-id "$EXP_ID" --output "$OUT_DIR/run_manifest.json" --command "$COMMAND" --status failed --failed-step preflight --error-summary "Missing original DataMIL pipeline and/or frozen dataset inputs"
  printf '%s\n' "$COMMAND" >> experiments/stage1/reports/commands.log
  echo "Smoke experiment preserved at $OUT_DIR" >&2
  exit 2
fi
echo "Preflight passed, but pipeline dispatch is intentionally disabled until the original DataMIL command is audited." >&2
exit 3
