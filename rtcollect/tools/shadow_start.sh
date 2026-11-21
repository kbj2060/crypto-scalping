#!/usr/bin/env bash
# rtcollect 그림자 일괄 기동 (2026-11-21 · 사용자 결정 «전 종목 7일 병행 · /mnt/data2 에 기록»). 서버 전용.
# 파이썬 수집기의 crontab 등록에서 종목을 그대로 읽어 같은 목록으로 띄운다. 파이썬 수집기·그 파일은 건드리지 않는다.
#   bash shadow_start.sh        띄운다(종류마다 8초 간격 -- 거래소 연결을 한꺼번에 열지 않게)
#   bash shadow_start.sh cron   띄우지 않고 재부팅용 crontab 줄만 출력한다
set -uo pipefail
ROOT=/home/llewyn/crypto-scalping
cd "$ROOT" || exit 1
SH=/mnt/data2/rtcollect_shadow
MODE="${1:-up}"
BIN="$ROOT/tmp/rtbin/rtcollect"
[ -x "$BIN" ] && [ -f scripts/ops/supervisor_rtcollect.sh ] || { echo "실행 파일·감독 스크립트 없음 -- 중단"; exit 1; }
if [ "$MODE" = up ]; then
  mkdir -p "$SH/hot" "$SH/live" logs/supervisor
  [ -e data/hot/rtcollect ] || ln -s "$SH/hot" data/hot/rtcollect
  [ -e data/live/rtcollect ] || ln -s "$SH/live" data/live/rtcollect
  [ "$(readlink -f data/hot/rtcollect)" = "$SH/hot" ] && [ "$(readlink -f data/live/rtcollect)" = "$SH/live" ] || { echo "출력 폴더가 $SH 를 가리키지 않음 -- 중단"; exit 1; }
fi
N=0
up() { # up <종류> [VAR=VAL ...]
  local kind="$1"; shift
  N=$((N + 1))
  if [ "$MODE" = cron ]; then
    echo "@reboot sleep $((100 + N * 5)) && cd $ROOT && RTCOLLECT_BIN=$BIN $* bash scripts/ops/supervisor_rtcollect.sh $kind >> logs/supervisor/rtcollect_reboot.log 2>&1 &"
  else
    env RTCOLLECT_BIN="$BIN" "$@" nohup setsid bash scripts/ops/supervisor_rtcollect.sh "$kind" >> logs/supervisor/rtcollect_boot.log 2>&1 < /dev/null &
    sleep "${GAP:-8}"
  fi
}
val() { echo "$1" | grep -oE "(^| )$2=[^ ]+" | head -1 | sed -E "s/^ ?$2=//"; }
CT="$(crontab -l | grep -E '^@reboot' | grep -v supervisor_rtcollect)"

while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(val "$l" TAPE_SYMBOL)"; m="$(val "$l" TAPE_MARKET)"
  up binance_tape "TAPE_SYMBOL=${s:-ethusdt}" "TAPE_MARKET=${m:-futures}"
done <<< "$(echo "$CT" | grep 'supervisor_trade_tape.sh')"
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(val "$l" OKX_TAPE_INST)"; up okx_tape "OKX_TAPE_INST=${s:-ETH-USDT-SWAP}"
done <<< "$(echo "$CT" | grep 'supervisor_okx_trade_tape.sh')"
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(val "$l" OKX_CTX_INST)"; up okx_ctx "OKX_CTX_INST=${s:-ETH-USDT-SWAP}"
done <<< "$(echo "$CT" | grep 'supervisor_okx_context.sh')"
echo "$CT" | grep -q 'supervisor_bybit_tape.sh' && up bybit_tape
echo "$CT" | grep -q 'supervisor_bybit_liq.sh' && up bybit_liq
echo "$CT" | grep -q 'supervisor_binance_ctx.sh' && up binance_ctx
echo "$CT" | grep -q 'supervisor_xyz100.sh' && up xyz
# 하이퍼리퀴드 호가: 줄마다 코인 하나. HL_BT_CTX_ONLY=1 줄 = 맥락만(hl_ctx) · 아니면 전체(hl_bt = 호가 .bt + 맥락).
#   🔴hl_bt 와 hl_ctx 가 같이 뜨면 hl_ctx 는 data/hot/rtcollect/hl_ctx_only.sqlite 에 쓴다(한 파일에 쓰는 프로세스는 하나 -- hl_bt 가 hl_ctx.sqlite).
#   대조 때는 두 파일을 합쳐 파이썬 hl_ctx.sqlite 와 비교한다(SHADOW.md §2).
HLB=0; HLC=""
while IFS= read -r l; do [ -n "$l" ] || continue
  c="$(val "$l" HL_BT_COIN)"; c="${c:-ETH}"
  if [ "$(val "$l" HL_BT_CTX_ONLY)" = 1 ]; then HLC="${HLC:+$HLC,}$c"; else up hl_bt "HL_BT_COIN=$c"; HLB=$((HLB + 1)); fi
done <<< "$(echo "$CT" | grep 'supervisor_hyperliquid_book_ticker.sh')"
if [ -n "$HLC" ]; then
  if [ "$HLB" -eq 0 ]; then up hl_ctx "HL_BT_COIN=$HLC"; else up hl_ctx "HL_BT_COIN=$HLC" "HL_CTX_DB_PATH=$ROOT/data/hot/rtcollect/hl_ctx_only.sqlite"; fi
fi
# 호가 변화: 스냅숏 REST 필수(없으면 첫 줄이 null 이라 ob5m 이 북을 못 세운다)
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(val "$l" DD_SYMBOL)"; up depth "DD_SYMBOL=${s:-ethusdt}" RTCOLLECT_REST=1
done <<< "$(echo "$CT" | grep 'supervisor_depth_diff.sh')"
# 호가 래스터: 종목·칸 크기·칸 수를 crontab 줄 그대로(OF_SYMBOL 없으면 ethusdt · OF_BIN_SIZE·OF_N_BINS 없으면 원본 기본 0.5·240) · 스냅숏 REST 필수
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(val "$l" OF_SYMBOL)"; b="$(val "$l" OF_BIN_SIZE)"; n="$(val "$l" OF_N_BINS)"
  up raster "OF_SYMBOL=${s:-ethusdt}" ${b:+"OF_BIN_SIZE=$b"} ${n:+"OF_N_BINS=$n"} RTCOLLECT_REST=1
done <<< "$(echo "$CT" | grep 'supervisor_orderflow_raster.sh')"
# 5분 호가 요약·호가 깊이 10열: 위 depth 파일을 읽는다
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(echo "$l" | grep -oE 'supervisor_ob5m\.sh [A-Z0-9]+' | awk '{print $2}')"; up ob5m "OB_SYMBOL=${s:-ETHUSDT}"
done <<< "$(echo "$CT" | grep 'supervisor_ob5m.sh')"
echo "$CT" | grep -q 'supervisor_bookdepth.sh' && up bookdepth OB_SYMBOL=ETHUSDT

# ── rtduck(DuckDB 로 쓰는 수집기 · 따로 된 실행 파일) ──
# 웹소켓만 쓰는 종류(hl_trades)는 띄운다. 🔴REST 종류(oi_lsratio · hl_positions)는 기본 꺼짐 -- 파이썬과 합쳐 호출이 두 배다(SHADOW.md §12).
#   켜려면 RTDUCK_SHADOW_REST=1 bash shadow_start.sh (그때 실행 파일에도 RTDUCK_REST=1 을 넘긴다).
DBIN="$ROOT/tmp/rtbin/rtduck"
upd() { # upd <종류> [VAR=VAL ...]
  local kind="$1"; shift
  N=$((N + 1))
  if [ "$MODE" = cron ]; then
    echo "@reboot sleep $((100 + N * 5)) && cd $ROOT && RTDUCK_BIN=$DBIN $* bash scripts/ops/supervisor_rtduck.sh $kind >> logs/supervisor/rtduck_reboot.log 2>&1 &"
  else
    env RTDUCK_BIN="$DBIN" "$@" nohup setsid bash scripts/ops/supervisor_rtduck.sh "$kind" >> logs/supervisor/rtduck_boot.log 2>&1 < /dev/null &
    sleep "${GAP:-8}"
  fi
}
if [ -x "$DBIN" ]; then
  if [ "$MODE" = up ]; then
    mkdir -p "$SH/duck_hot" "$SH/duck_live"
    [ -e data/hot/rtduck ] || ln -s "$SH/duck_hot" data/hot/rtduck
    [ -e data/live/rtduck ] || ln -s "$SH/duck_live" data/live/rtduck
    [ "$(readlink -f data/hot/rtduck)" = "$SH/duck_hot" ] && [ "$(readlink -f data/live/rtduck)" = "$SH/duck_live" ] || { echo "rtduck 출력 폴더가 $SH 를 가리키지 않음 -- rtduck 은 안 띄운다"; DBIN=""; }
  fi
  if [ -n "$DBIN" ]; then
    while IFS= read -r l; do [ -n "$l" ] || continue
      c="$(val "$l" HL_COINS)"; upd hl_trades "HL_COINS=${c:-ETH}"
    done <<< "$(echo "$CT" | grep 'supervisor_hyperliquid_trades.sh')"
    # market_data_worker(미시구조 ETH·BTC·SOL + 청산 요격기 ETH + dashboard_state). 미시구조 REST(OI·premiumIndex 10초 · aggTrades 대체)는
    #   그림자에서 켜면 파이썬과 합쳐 두 배라 RTDUCK_SHADOW_REST=1 일 때만 -- 꺼 두면 웹소켓 칸만 쌓이고 poll_connected=false(data_stale)
    if echo "$CT" | grep -q 'supervisor_market_data_worker.sh'; then
      if [ "${RTDUCK_SHADOW_REST:-}" = 1 ]; then upd market_data TAIL_SYMBOLS=ETHUSDT BOT_SYMBOLS=ETHUSDT,BTCUSDT,SOLUSDT RTDUCK_REST=1
      else upd market_data TAIL_SYMBOLS=ETHUSDT BOT_SYMBOLS=ETHUSDT,BTCUSDT,SOLUSDT; fi
    fi
    if [ "${RTDUCK_SHADOW_REST:-}" = 1 ]; then
      echo "$CT" | grep -q 'supervisor_oi_lsratio_worker.sh' && upd oi_lsratio RTDUCK_REST=1 BOT_SYMBOLS=ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT
      while IFS= read -r l; do [ -n "$l" ] || continue
        c="$(val "$l" HL_POS_COINS)"; upd hl_positions "HL_POS_COINS=${c:-ETH}" RTDUCK_REST=1 HL_ROOT="$ROOT/data/live/rtduck/orderflow/hyperliquid"
      done <<< "$(echo "$CT" | grep 'supervisor_hyperliquid_positions.sh')"
    else
      echo "rtduck REST 종류(oi_lsratio · hl_positions) 안 띄움 -- RTDUCK_SHADOW_REST=1 이면 띄운다(호출 두 배)"
    fi
  fi
else
  echo "rtduck 실행 파일 없음($DBIN) -- rtduck 종류는 건너뜀"
fi

[ "$MODE" = cron ] && exit 0
sleep 45
echo "띄운 수 $N · 실행 중 rtcollect $(pgrep -x rtcollect | wc -l)개"
echo "--- 기동 로그 꼬리"; tail -n 12 logs/supervisor/rtcollect_boot.log 2>/dev/null | cut -c1-220
echo "--- 그림자 폴더"; du -sh "$SH" 2>/dev/null; ls "$SH/hot" 2>/dev/null | tr '\n' ' '; echo; ls "$SH/live" 2>/dev/null | tr '\n' ' '; echo
echo "--- 종류별 최근 로그 한 줄"; for f in logs/supervisor/rtcollect_*_"$(date +%Y%m%d)".log; do [ -f "$f" ] && echo "$(basename "$f" | cut -c1-60): $(tail -n 1 "$f" | cut -c1-150)"; done
