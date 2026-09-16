#!/usr/bin/env bash
# Crash-restart watchdog for the server-side ETH peg-maker fill shadow worker.
# Same _supervise.sh pattern as supervisor_oi_lsratio_worker.sh (crontab @reboot, not systemd --
# see that file's header for the sudoers reasoning).
#
# What it runs: scripts/maker_fill_shadow_worker.py -- virtual (no real orders, public streams
# only) peg/static maker legs every 5 minutes, recording effective cost per leg to validate the
# maker fill simulations (docs/experiments/eth_maker_fill_simulation_l2_20260822.md) against
# live fills. Completely independent from trading_bot.py -- no order path, no shared files.
#
# Writes to its OWN duckdb file (maker_fill_shadow.duckdb) -- single-writer-per-file principle,
# same as oi_lsratio.duckdb (DuckDB refuses concurrent writers across processes).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export MAKER_SHADOW_SYMBOL="${MAKER_SHADOW_SYMBOL:-ETHUSDT}"
export MAKER_SHADOW_SPACING_S="300"
export MAKER_SHADOW_TIMEOUT_S="120"
export MAKER_SHADOW_POLICIES="peg,static"

# 2026-09-16: 심볼별로 DB/lock/로그를 분리한다. ETHUSDT 는 **기존 경로 그대로**(접미사 없음)라
# 돌고 있는 워커에 영향이 없다. duckdb 는 단일 writer 이므로 심볼마다 파일이 달라야 한다.
case "$MAKER_SHADOW_SYMBOL" in
  ETHUSDT) SFX="" ;;
  *)       SFX="_$(printf '%s' "$MAKER_SHADOW_SYMBOL" | tr 'A-Z' 'a-z')" ;;
esac
export MAKER_SHADOW_DB_PATH="$ROOT/data/live/maker_fill_shadow${SFX}.duckdb"

if pgrep -af "[m]aker_fill_shadow_worker.py" | grep -q "MAKER_SHADOW_SYMBOL=$MAKER_SHADOW_SYMBOL" 2>/dev/null; then
  echo "[$(date -Iseconds)] $MAKER_SHADOW_SYMBOL 섀도우가 이미 실행 중 -- 켜지 않는다." >&2; exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "maker_fill_shadow_worker.py($MAKER_SHADOW_SYMBOL peg-maker fill shadow)" \
  "$ROOT/data/live/.supervisor_maker_fill_shadow${SFX}.lock" \
  "$ROOT/logs/supervisor/maker_fill_shadow${SFX}" \
  "$PY" -u "$ROOT/scripts/maker_fill_shadow_worker.py"
