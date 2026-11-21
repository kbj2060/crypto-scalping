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
echo "$CT" | grep -q 'supervisor_hyperliquid_book_ticker.sh' && up hl_ctx
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

[ "$MODE" = cron ] && exit 0
sleep 45
echo "띄운 수 $N · 실행 중 rtcollect $(pgrep -x rtcollect | wc -l)개"
echo "--- 기동 로그 꼬리"; tail -n 12 logs/supervisor/rtcollect_boot.log 2>/dev/null | cut -c1-220
echo "--- 그림자 폴더"; du -sh "$SH" 2>/dev/null; ls "$SH/hot" 2>/dev/null | tr '\n' ' '; echo; ls "$SH/live" 2>/dev/null | tr '\n' ' '; echo
echo "--- 종류별 최근 로그 한 줄"; for f in logs/supervisor/rtcollect_*_"$(date +%Y%m%d)".log; do [ -f "$f" ] && echo "$(basename "$f" | cut -c1-60): $(tail -n 1 "$f" | cut -c1-150)"; done
