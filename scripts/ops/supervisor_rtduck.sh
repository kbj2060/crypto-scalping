#!/usr/bin/env bash
# rtduck(DuckDB 로 쓰는 Rust 수집기) 크래시 재기동 래퍼 (2026-11-21). 종류 하나 = 프로세스 하나. 지금은 **그림자** 전용 -- rtcollect/SHADOW.md §rtduck.
# 🔴실행·등록하지 않았다(사용자 결정 전). 순서는 SHADOW.md 대로.
#
# 실행 파일: rtduck/target/release/rtduck 을 개발기에서 빌드해 옮긴 것(RTDUCK_BIN, 기본 $ROOT/bin/rtduck).
#   bash scripts/ops/supervisor_rtduck.sh <oi_lsratio|hl_trades|hl_positions> [목록]
#   oi_lsratio: BOT_SYMBOLS(기본 ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT -- 서버 supervisor_oi_lsratio_worker.sh 와 같은 목록) · 🔴RTDUCK_REST=1 없으면 아무것도 안 쓴다
#               · 쓰는 곳 기본 data/live/rtduck/oi_lsratio.duckdb(파이썬 파일 data/live/oi_lsratio.duckdb 는 실행 파일이 거부 -- 전환 때만 RTDUCK_TAKEOVER=1)
#   hl_trades:  HL_COINS(기본 ETH -- 원본처럼 코인마다 하나) · 쓰는 곳 기본 data/live/rtduck/orderflow/hyperliquid/<코인_코인>/<날짜>.duckdb
#   hl_positions: HL_POS_COINS(기본 ETH) · HL_ROOT(체결 DuckDB 읽기만) · 🔴RTDUCK_REST=1 없으면 아무것도 안 한다 · 쓰는 곳 data/hot/rtduck/hl_positions[_코인들].sqlite
# 로그: logs/supervisor/rtduck_<KEY>_YYYYMMDD.log · 락: data/live/.supervisor_rtduck_<KEY>.lock
# ⚠️같은 종류·같은 목록의 rtduck 가 이미 돌면 켜지 않는다(DuckDB 는 쓰는 프로세스가 하나다 -- 두 벌이면 하나가 «Conflicting lock» 으로 죽는다).
#   목록이 다르면 켠다(hl_trades·hl_positions 는 코인 조합마다 파일이 다르다 · oi_lsratio 는 한 파일이라 종류가 같으면 막는다).
#
# crontab(서버 재부팅 때) -- 적기만 했다:
#   @reboot sleep 95 && cd <저장소> && RTDUCK_REST=1 bash scripts/ops/supervisor_rtduck.sh oi_lsratio >> logs/supervisor/rtduck_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${RTDUCK_BIN:-$ROOT/bin/rtduck}"
KIND="${1:-${RTDUCK_KIND:-oi_lsratio}}"
case "$KIND" in
  oi_lsratio) VAR=BOT_SYMBOLS; DEF=ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT ;;
  hl_trades)  VAR=HL_COINS;    DEF=ETH ;;
  hl_positions) VAR=HL_POS_COINS; DEF=ETH ;;
  *) echo "[$(date -Iseconds)] 모르는 종류 '$KIND' -- oi_lsratio|hl_trades|hl_positions" >&2; exit 2 ;;
esac
if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] 실행 파일이 없다: $BIN" >&2
  exit 1
fi
SYM="$(echo "${2:-${!VAR:-$DEF}}" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
envof() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep "^$2=" | head -1 | cut -d= -f2-; }
for pid in $(pgrep -x rtduck); do
  [ "$(envof "$pid" RTDUCK_KIND)" = "$KIND" ] || continue
  cur="$(envof "$pid" "$VAR" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  [ "$KIND" = oi_lsratio ] || [ "${cur:-$DEF}" = "$SYM" ] || continue
  echo "[$(date -Iseconds)] rtduck($KIND $(envof "$pid" "$VAR"))가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
  exit 1
done
KEY="${KIND}_${SYM//,/_}"
export RTDUCK_KIND="$KIND" "$VAR=$SYM"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtduck($KEY)" \
  "$ROOT/data/live/.supervisor_rtduck_$KEY.lock" \
  "$ROOT/logs/supervisor/rtduck_$KEY" \
  "$BIN"
