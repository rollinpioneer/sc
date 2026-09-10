#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
KEY="${GITHUB_DEPLOY_KEY:-/home/__compress_data/xushijie/.ssh/scizor_github_sc_deploy_ed25519}"
BRANCH="${BRANCH:-exp/hb4-square-absorb-v1}"
REMOTE="${REMOTE:-github-deploy}"
LOG_ROOT="$RUN_ROOT/logs"
ARCHIVE="$EXPORT_ROOT/package/HB4_results_lightweight.zip"
SHA_FILE="$EXPORT_ROOT/package/HB4_results_lightweight.zip.sha256"

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/publish.lock"
flock -n 9 || { echo "publish already running"; exit 0; }
exec > >(tee -a "$LOG_ROOT/publish.log") 2>&1

echo "publish_wait_started=$(date --iso-8601=seconds)"
while [[ ! -f "$ARCHIVE" || ! -f "$SHA_FILE" ]]; do
  sleep 60
done

cd "$CODE_ROOT"

actual_sha="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
declared_sha="$(awk '{print $1}' "$SHA_FILE")"
[[ "$actual_sha" == "$declared_sha" ]]

tmp1="$(mktemp -d /tmp/hb4_zipcheck_1.XXXXXX)"
tmp2="$(mktemp -d /tmp/hb4_zipcheck_2.XXXXXX)"
unzip -q "$ARCHIVE" -d "$tmp1"
unzip -q "$ARCHIVE" -d "$tmp2"
! unzip -Z1 "$ARCHIVE" | grep -E 'HB4_results_lightweight\.zip(\.sha256)?$'

"$PYTHON" - <<'PY' "$tmp1" "$tmp2" "$EXPORT_ROOT"
import csv
import json
import sys
import zipfile
from pathlib import Path

roots = [Path(sys.argv[1]), Path(sys.argv[2])]
export = Path(sys.argv[3])
required = [
    "hb4_square_absorb_v1/report/HB4_REPORT.md",
    "hb4_square_absorb_v1/report/LOCAL_ONLY_ARTIFACTS.md",
    "hb4_square_absorb_v1/metrics/power_planning.json",
]
if (export / "metrics/development/decision.json").is_file():
    required.extend([
        "hb4_square_absorb_v1/metrics/development/summary.json",
        "hb4_square_absorb_v1/metrics/development/decision.json",
    ])
if (export / "metrics/test/coverage.json").is_file():
    required.extend([
        "hb4_square_absorb_v1/metrics/test/coverage.json",
        "hb4_square_absorb_v1/metrics/test/methods.csv",
        "hb4_square_absorb_v1/metrics/test/paired_comparisons.json",
        "hb4_square_absorb_v1/metrics/test/baseline_preservation.csv",
        "hb4_square_absorb_v1/metrics/decision.json",
    ])

for root in roots:
    for item in required:
        path = root / item
        if not path.is_file():
            raise SystemExit(f"missing from zip: {item}")
    for path in root.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in root.rglob("*.jsonl"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                json.loads(line)
    for path in root.rglob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            list(csv.reader(handle))

coverage = export / "metrics/test/coverage.json"
if coverage.is_file():
    cov = json.loads(coverage.read_text(encoding="utf-8"))
    if cov.get("records") != cov.get("expected_records") or not cov.get("complete"):
        raise SystemExit(f"formal coverage incomplete: {cov}")
PY

"$PYTHON" -m py_compile SCIZOR/recovery_handback/hb4/*.py
bash -n SCIZOR/recovery_handback/hb4/*.sh
git diff --check

git add \
  SCIZOR/recovery_handback/adapters/env_adapter.py \
  SCIZOR/robomimic/robomimic/utils/file_utils.py \
  SCIZOR/recovery_handback/hb4 \
  experiments/handback/hb4_square_absorb_v1

# Raw per-episode records are retained locally for audit and are excluded from
# the lightweight GitHub handoff, matching the ZIP packaging rule.
git reset -- \
  experiments/handback/hb4_square_absorb_v1/metrics/development/*/episodes.jsonl \
  experiments/handback/hb4_square_absorb_v1/metrics/test/*/episodes.jsonl

if git diff --cached --quiet; then
  echo "nothing_to_commit=1"
else
  git commit -m "Add HB4 square absorb experiment results"
fi

GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git push "$REMOTE" "HEAD:refs/heads/$BRANCH"

local_head="$(git rev-parse HEAD)"
remote_head="$(GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" git ls-remote "$REMOTE" "refs/heads/$BRANCH" | awk '{print $1}')"
[[ "$local_head" == "$remote_head" ]]

GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git fetch "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
git show "$REMOTE/$BRANCH:experiments/handback/hb4_square_absorb_v1/report/HB4_REPORT.md" >/tmp/hb4_remote_report_check.md
git show "$REMOTE/$BRANCH:experiments/handback/hb4_square_absorb_v1/metrics/decision.json" >/tmp/hb4_remote_decision_check.json 2>/dev/null || true

echo "published_head=$local_head"
echo "published_remote_head=$remote_head"
echo "published_archive_sha256=$actual_sha"
echo "publish_finished=$(date --iso-8601=seconds)"
