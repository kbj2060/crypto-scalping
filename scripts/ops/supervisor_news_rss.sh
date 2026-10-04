#!/usr/bin/env bash
# 뉴스 RSS(The Block·Cointelegraph·CoinDesk) 수집기의 크래시 재기동 래퍼 (2026-10-05).
# 수집기: scripts/live_news_rss_collector_20261005.py (공개 RSS 읽기, 주문·바이낸스 호출 없음)
# 같은 수집기가 둘이면 같은 사이트를 두 배로 두드린다 -- 이미 돌면 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
RUNNER="scripts/live_news_rss_collector_20261005.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] 뉴스 RSS 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_news_rss_collector_20261005.py" \
  "$ROOT/data/live/.supervisor_news_rss.lock" \
  "$ROOT/logs/supervisor/news_rss" \
  "$PY" -u "$ROOT/$RUNNER"
