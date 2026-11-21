#!/usr/bin/env bash
# rtcollect 그림자 일괄 기동 (2026-11-21 · 사용자 결정 «전 종목 7일 병행 · /mnt/data2 에 기록»). 서버 전용.
# 파이썬 수집기의 crontab 등록에서 종목을 그대로 읽어 같은 목록으로 띄운다. 파이썬 수집기·그 파일은 건드리지 않는다.
#   bash shadow_start.sh        띄운다(종류마다 8초 간격 -- 거래소 연결을 한꺼번에 열지 않게)
#   bash shadow_start.sh cron   띄우지 않고 crontab 줄만 출력한다 -- 상주(@reboot) 묶음 뒤에 예약 종류(raster_archive·liq_tab·seal) 묶음을 따로
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
[ "$MODE" = cron ] && echo "# ── 상주 종류(@reboot) ──"
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
# 칸별 아이스버그(depthdiff·hot 테이프 읽기만 -- 거래소 호출 없음): 종목 = 감독 스크립트의 첫 인자(서버 줄 `supervisor_eth_iceberg.sh XRPUSDT >> …` · 없으면 ETHUSDT)
while IFS= read -r l; do [ -n "$l" ] || continue
  s="$(echo "$l" | grep -oE 'supervisor_eth_iceberg\.sh [A-Z0-9]+' | awk '{print $2}')"; s="${s:-$(val "$l" ICEBERG_SYMBOL)}"; up iceberg "ICEBERG_SYMBOL=${s:-ETHUSDT}"
done <<< "$(echo "$CT" | grep 'supervisor_eth_iceberg.sh')"

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
    mkdir -p "$SH/duck_hot" "$SH/duck_live" "$SH/duck_lake" data/rtduck
    [ -e data/hot/rtduck ] || ln -s "$SH/duck_hot" data/hot/rtduck
    [ -e data/live/rtduck ] || ln -s "$SH/duck_live" data/live/rtduck
    [ -e data/rtduck/lake ] || ln -s "$SH/duck_lake" data/rtduck/lake   # seal 그림자 lake(+ 원천 사본 .snap_*) -- C: 에 쓰지 않게
    [ "$(readlink -f data/hot/rtduck)" = "$SH/duck_hot" ] && [ "$(readlink -f data/live/rtduck)" = "$SH/duck_live" ] && [ "$(readlink -f data/rtduck/lake)" = "$SH/duck_lake" ] || { echo "rtduck 출력 폴더가 $SH 를 가리키지 않음 -- rtduck 은 안 띄운다"; DBIN=""; }
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
      echo "# rtduck REST 종류(oi_lsratio · hl_positions) 안 띄움 -- RTDUCK_SHADOW_REST=1 이면 띄운다(호출 두 배)"
    fi
    # Deribit 옵션 체결: 체결 WS 만(Deribit 연결 -- 바이낸스 한도와 무관) · REST(백필·10분 체인·매시 대조)는 RTDUCK_REST 를 안 넘겨 꺼짐
    if echo "$CT" | grep -q 'deribit_block_trade'; then upd deribit
    else echo "# deribit: 파이썬 수집기 @reboot 줄(deribit_block_trade)을 못 찾음 -- 안 띄움(다른 방식으로 뜨면 손으로: RTDUCK_BIN=$DBIN bash scripts/ops/supervisor_rtduck.sh deribit)"; fi
  fi
else
  echo "# rtduck 실행 파일 없음($DBIN) -- rtduck 종류는 건너뜀"
fi

if [ "$MODE" = cron ]; then
  # ── 예약 종류(한 번 돌고 끝 · REST 없음 = 거래소 호출 ×1) -- 위 @reboot 줄과 따로 등록한다(조정자가 확인 뒤).
  #   줄마다 출력 폴더가 6TB 링크인지 먼저 본다(아니면 아무것도 안 하고 끝 -- C: 에 쓰지 않게). 시각은 파이썬 줄에서 읽는다.
  ALL="$(crontab -l 2>/dev/null | grep -vE '^[[:space:]]*(#|$)')"
  echo "# ── rtduck 예약 종류(crontab 에 따로) ──"
  if [ -n "${DBIN:-}" ] && [ -x "$DBIN" ]; then
    GL="[ \"\$(readlink -f data/live/rtduck)\" = $SH/duck_live ]"
    echo "$CT" | grep -q 'supervisor_orderflow_raster.sh' \
      && echo "40 * * * * cd $ROOT && $GL && RTDUCK_KIND=raster_archive $DBIN >> logs/supervisor/rt_raster_archive.log 2>&1" \
      || echo "# raster_archive: 래스터 줄이 없어 안 냄"
    m="$(echo "$ALL" | grep 'live_liq_tab' | head -1 | awk '{print $1}')"
    if [[ "$m" =~ ^[0-9]+$ ]]; then
      echo "# liq_tab: 파이썬 줄 $m 분 + 5 · 파이썬 1h 봉 캐시를 복사한 뒤 REST 끄고(=저장된 봉만) 계산"
      echo "$(( (m + 5) % 60 )) * * * * cd $ROOT && $GL && cp -p data/live/liq_tab_k1h_*.parquet data/live/rtduck/ && RTDUCK_KIND=liq_tab $DBIN >> logs/supervisor/rt_liq_tab.log 2>&1"
    else echo "# liq_tab: 파이썬 줄(live_liq_tab · 분이 숫자)을 못 찾음 -- 안 냄"; fi
    read -r m h _ <<< "$(echo "$ALL" | grep 'seal_to_lake' | head -1)"
    if [[ "${m:-}" =~ ^[0-9]+$ && "${h:-}" =~ ^[0-9]+$ ]]; then
      echo "# seal: 파이썬 줄 $h시 $m분 + 1시간(원본과 겹치면 원천 사본이 둘) · lake = data/rtduck/lake → $SH/duck_lake · hot 정리 없음(TAKEOVER 아님)"
      echo "$m $(( (h + 1) % 24 )) * * * cd $ROOT && [ \"\$(readlink -f data/rtduck/lake)\" = $SH/duck_lake ] && $GL && RTDUCK_KIND=seal $DBIN >> logs/supervisor/rt_seal.log 2>&1"
    else echo "# seal: 파이썬 줄(seal_to_lake · 분·시가 숫자)을 못 찾음 -- 안 냄"; fi
  else echo "# rtduck 실행 파일이 없어 안 냄"; fi
  exit 0
fi
sleep 45
echo "띄운 수 $N · 실행 중 rtcollect $(pgrep -x rtcollect | wc -l)개"
echo "--- 기동 로그 꼬리"; tail -n 12 logs/supervisor/rtcollect_boot.log 2>/dev/null | cut -c1-220
echo "--- 그림자 폴더"; du -sh "$SH" 2>/dev/null; ls "$SH/hot" 2>/dev/null | tr '\n' ' '; echo; ls "$SH/live" 2>/dev/null | tr '\n' ' '; echo
echo "--- 종류별 최근 로그 한 줄"; for f in logs/supervisor/rtcollect_*_"$(date +%Y%m%d)".log; do [ -f "$f" ] && echo "$(basename "$f" | cut -c1-60): $(tail -n 1 "$f" | cut -c1-150)"; done
