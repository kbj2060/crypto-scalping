#!/usr/bin/env bash
# 저변동 스트래들 섀도우의 크래시 재기동 래퍼 (2026-09-15).
#
# 워커: scripts/live_eth_lowvol_straddle_shadow_20260915.py (읽기 전용, **주문 없음**)
# 이 섀도우의 목적은 표본 축적이다 — 멈춘 시간만큼 판정이 늦어진다(연 60~130쌍이라 하루도 아깝다).
# 다만 봇·대시보드와 완전 분리라 죽어도 서비스에는 영향이 없다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 심볼 워커가 둘이면
# 같은 JSONL 에 둘이 append 해서 쌍이 중복 기록되고, 그 순간 표본 수가 거짓이 된다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_lowvol_straddle_shadow_20260915.py"
SYM="${SS_SYMBOL:-ETHUSDT}"
for pid in $(pgrep -f "[l]ive_eth_lowvol_straddle_shadow_20260915.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^SS_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ETHUSDT}" = "$SYM" ]; then
    echo "[$(date -Iseconds)] 스트래들 섀도우($SYM)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SS_SYMBOL="$SYM"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_lowvol_straddle_shadow_20260915.py($SYM)" \
  "$ROOT/data/live/.supervisor_lowvol_straddle_shadow_$SYM.lock" \
  "$ROOT/logs/supervisor/lowvol_straddle_shadow_$SYM" \
  "$PY" -u "$ROOT/$RUNNER" --loop
