#!/usr/bin/env bash
# 하이퍼리퀴드 고래 포지션 수집기의 크래시 재기동 래퍼 (2026-09-24).
# 수집기: scripts/live_hyperliquid_positions_collector_20260924.py (읽기 전용 공개 API, 주문 없음)
# 포지션 스냅샷은 소급이 불가능하다 -- 멈춘 구간은 영구 손실이라 재기동을 붙인다.
#
# 2026-09-26 다코인: 대상은 HL_POS_COINS 로 가른다(기본 ETH). 🔴코인마다 따로 띄우지 말 것 --
#   한 번의 조회가 전 코인을 주므로 같은 주소를 중복 조회해 요청 한도(1,200/분)를 넘는다.
#   ETH 는 단독(사전등록 검정의 2.5분 바퀴), 나머지는 HL_POS_COINS=BTC,SOL,XRP,HYPE 한 프로세스.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_hyperliquid_positions_collector_20260924.py"
TARGET="${HL_POS_COINS:-ETH}"
for pid in $(pgrep -f "[l]${RUNNER#scripts/l}"); do
  cur=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^HL_POS_COINS=' | cut -d= -f2)
  if [ "${cur:-ETH}" = "$TARGET" ]; then
    echo "[$(date -Iseconds)] 하이퍼리퀴드 포지션 수집기($TARGET)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done
# ETH 는 기존 락·로그 경로 그대로(이미 떠 있는 supervisor 와 같은 락이어야 중복이 막힌다).
if [ "$TARGET" = "ETH" ]; then SFX=""; else SFX="_${TARGET//,/_}"; fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HL_POS_COINS="$TARGET"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_hyperliquid_positions_collector_20260924.py($TARGET)" \
  "$ROOT/data/live/.supervisor_hyperliquid_positions$SFX.lock" \
  "$ROOT/logs/supervisor/hyperliquid_positions$SFX" \
  "$PY" -u "$ROOT/$RUNNER"
