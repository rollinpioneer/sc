#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
RUN_KIND=baseline exec bash experiments/stage1/scripts/run_stage1_smoke.sh "${1:-experiments/stage1/configs/mvp_base.yaml}" "${2:-0}"
