#!/usr/bin/env bash
# 뉴스 판정 워커(jevk5:4b via ollaya)의 크래시 재기동 래퍼 (2026-10-05).
# 수집기: scripts/live_news_judge_20261005.py (news_rss.sqlite 읽고 judgments 씀, ollaya 필요: supervisor_ollaya.sh)
# 둘이면 같은 기사를 두 번 판정한다 -- 이미 돌면 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
RUNNER="scripts/live_news_judge_20261005.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] 뉴스 판정 워커가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_news_judge_20261005.py" \
  "$ROOT/data/live/.supervisor_news_judge.lock" \
  "$ROOT/logs/supervisor/news_judge" \
  "$PY" -u "$ROOT/$RUNNER"
