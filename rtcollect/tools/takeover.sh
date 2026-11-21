#!/usr/bin/env bash
# 그림자 → 진짜 경로 넘기기 (2026-11-21 · 사용자 «기다리지 말고 바로바로 반영 배포»). 서버 전용 · 저장소 루트 기준 · **한 번에 한 스트림**.
#   bash rtcollect/tools/takeover.sh check <종류> [<종목>]  그림자가 떠 있던 창 전체를 대조(읽기 전용 · nice) -- 설명 못 하는 차이가 0 이 아니면(또는 견준 것이 0) 종료코드 1
#   bash rtcollect/tools/takeover.sh go    <종류> [<종목>]  check 가 0 일 때만: ① crontab 백업 ② 파이썬·그림자 줄 주석 + Rust 줄 · 줄 수 검증 뒤 적용
#                                                         ③ 파이썬 감독 → 수집기 끔 ④ 그림자 감독 → Rust 끔 ⑤ Rust 를 진짜 경로(TAKEOVER=1)로 ⑥ 60초 확인 · 끊긴 초
#   bash rtcollect/tools/takeover.sh back  <종류> [<종목>]  Rust 끔 → crontab 되돌림(이 종류가 바꾼 줄만) → 매니페스트 되돌림 → 파이썬 감독 다시
# 종류(상주): binance_tape binance_spot okx_tape okx_ctx bybit_tape bybit_liq binance_ctx xyz hl_ctx depth raster ob5m bookdepth iceberg hl_trades market_data oi_lsratio deribit
#      (예약): liq_tab seal    · 안 다룸: hl_bt(파이썬 짝이 Pi) · hl_positions(대조 방법 없음) · news·treasury_futures·macro_5m(그림자 없음) · liqmap_scorer·soxl_weekend·cboe_chain(전진 장부 -- 손으로)
# 죽이는 것: 명령줄 인자가 그 종류의 러너 파일로 끝나고 environ 의 종목이 맞는 프로세스와 그 부모 _supervise.sh 뿐(PID 마다 /proc/<pid>/cmdline 다시 확인 · 이름으로 한꺼번에 안 죽인다).
#   SIGTERM 뒤 기다렸다가 살아 있으면 보고만 하고 멈춘다(그때는 `back` 으로 되돌린다). 대시보드·감시·주문은 러너가 달라 걸리지 않는다(걸리면 멈춘다).
# 바꿔 끼우기(시험용): CRONTAB_CMD(기본 crontab) · TAKEOVER_ROOT · WATCH_S(60) · TERM_WAIT_S(30) · CHECK_PY · RTCOLLECT_BIN · RTDUCK_BIN
set -uo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TAKEOVER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 1
MODE="${1:-}"; KIND="${2:-}"; ARG="${3:-}"
CT="${CRONTAB_CMD:-crontab}"
WATCH_S="${WATCH_S:-60}"; TERM_WAIT_S="${TERM_WAIT_S:-30}"
PYC="${CHECK_PY:-$HOME/miniconda3/envs/quant_ai/bin/python}"; [ -x "$PYC" ] || PYC=python3
die() { echo "🔴 $*" >&2; exit 1; }
nrm() { echo "$1" | tr '[:lower:]' '[:upper:]' | tr -d ' '; }
envof() { tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep "^$2=" | head -1 | cut -d= -f2-; }
argv() { tr '\0' '\n' 2>/dev/null < "/proc/$1/cmdline"; }
val() { echo "$1" | grep -oE "(^| )$2=[^ ]+" | head -1 | sed -E "s/^ ?$2=//"; }
gone() { [ ! -e "/proc/$1" ] || [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" = Z ]; }
ppid_of() { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '; }
is_sup() { argv "$1" | grep -qE '(^|/)_supervise\.sh$'; }

# ── 종류 표 ── ENG(rtcollect|rtduck) RK(Rust 종류) PYSUP(crontab 의 파이썬 줄 글자) PYRUN(파이썬 러너 파일) SELV·SELD(종목 변수·기본)
#   XV·XD·XW(파이썬 environ 에서 XV(기본 XD) 가 XW 여야 -- 같은 러너의 다른 스트림 가르기) XR(Rust·crontab 줄에도 X 를 적용) RVAR(Rust 종목 변수)
#   COPY(파이썬 environ 에서 그대로 넘길 변수) TENV(진짜 경로·REST) CHK(대조 목록 «모드:파이썬:Rust») W(끊긴 초를 재는 파일·폴더) CRON(예약 종류) SHARED(여러 종목이 한 파일)
ENG=rtcollect; RK="$KIND"; PYSUP=""; PYRUN=""; SELV=""; SELD=""; XV=""; XD=""; XW=""; XR=0; RVAR=""; COPY=""; TENV=""; CHK=""; W=""; CRON=0; SHARED=0; GROUP=0
case "$KIND" in
  binance_tape) PYSUP=supervisor_trade_tape.sh; PYRUN=live_trade_tape_collector_20260916.py; SELV=TAPE_SYMBOL; SELD=ethusdt; XV=TAPE_MARKET; XD=futures; XW=futures; XR=1; SHARED=1
                TENV="TAPE_MARKET=futures TAPE_DB_PATH=data/hot/binance_tape.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/binance_tape.sqlite:data/hot/rtcollect/binance_tape.sqlite"; W="data/hot/binance_tape.sqlite data/hot/binance_tape.sqlite-wal" ;;
  binance_spot) RK=binance_tape; PYSUP=supervisor_trade_tape.sh; PYRUN=live_trade_tape_collector_20260916.py; SELV=TAPE_SYMBOL; SELD=ethusdt; XV=TAPE_MARKET; XD=futures; XW=spot; XR=1; SHARED=1
                TENV="TAPE_MARKET=spot TAPE_DB_PATH=data/hot/binance_spot_tape.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/binance_spot_tape.sqlite:data/hot/rtcollect/binance_spot_tape.sqlite"; W="data/hot/binance_spot_tape.sqlite data/hot/binance_spot_tape.sqlite-wal" ;;
  okx_tape)     PYSUP=supervisor_okx_trade_tape.sh; PYRUN=live_okx_trade_tape_collector_20260923.py; SELV=OKX_TAPE_INST; SELD=ETH-USDT-SWAP; SHARED=1
                TENV="OKX_TAPE_DB_PATH=data/hot/okx_tape.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/okx_tape.sqlite:data/hot/rtcollect/okx_tape.sqlite"; W="data/hot/okx_tape.sqlite data/hot/okx_tape.sqlite-wal" ;;
  okx_ctx)      PYSUP=supervisor_okx_context.sh; PYRUN=live_okx_context_collector_20260923.py; SELV=OKX_CTX_INST; SELD=ETH-USDT-SWAP; SHARED=1
                TENV="OKX_CTX_DB_PATH=data/hot/okx_ctx.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/okx_ctx.sqlite:data/hot/rtcollect/okx_ctx.sqlite"; W="data/hot/okx_ctx.sqlite data/hot/okx_ctx.sqlite-wal" ;;
  bybit_tape)   PYSUP=supervisor_bybit_tape.sh; PYRUN=live_bybit_trade_tape_collector_20261006.py; RVAR=BYBIT_TAPE_SYMBOLS; COPY=BYBIT_TAPE_SYMBOLS
                TENV="BYBIT_TAPE_DB=data/hot/bybit_tape.sqlite BYBIT_CTX_DB=data/hot/bybit_ctx.sqlite RTCOLLECT_REST=1"
                CHK="sq:data/hot/bybit_tape.sqlite:data/hot/rtcollect/bybit_tape.sqlite sq:data/hot/bybit_ctx.sqlite:data/hot/rtcollect/bybit_ctx.sqlite"; W="data/hot/bybit_tape.sqlite data/hot/bybit_tape.sqlite-wal" ;;
  bybit_liq)    PYSUP=supervisor_bybit_liq.sh; PYRUN=live_bybit_liquidation_collector_20261003.py; COPY=BYBIT_LIQ_SYMBOLS
                TENV="BYBIT_LIQ_DB=data/hot/bybit_liq.sqlite"; CHK="sq:data/hot/bybit_liq.sqlite:data/hot/rtcollect/bybit_liq.sqlite"; W="data/hot/bybit_liq.sqlite data/hot/bybit_liq.sqlite-wal" ;;
  binance_ctx)  PYSUP=supervisor_binance_ctx.sh; PYRUN=live_binance_ctx_collector_20261001.py; COPY="BN_CTX_MARK_SYMBOLS BN_CTX_OI_SYMBOLS BN_CTX_OI_SLOW_SYMBOLS"
                TENV="BN_CTX_DB=data/hot/binance_ctx.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/binance_ctx.sqlite:data/hot/rtcollect/binance_ctx.sqlite"; W="data/hot/binance_ctx.sqlite data/hot/binance_ctx.sqlite-wal" ;;
  xyz)          PYSUP=supervisor_xyz100.sh; PYRUN=live_xyz100_collector_20261014.py; COPY=HL_XYZ_COINS
                TENV="HL_XYZ_DB=data/hot/hl_xyz.sqlite RTCOLLECT_REST=1"; CHK="sq:data/hot/hl_xyz.sqlite:data/hot/rtcollect/hl_xyz.sqlite"; W="data/hot/hl_xyz.sqlite data/hot/hl_xyz.sqlite-wal" ;;
  hl_ctx)       PYSUP=supervisor_hyperliquid_book_ticker.sh; PYRUN=live_hyperliquid_book_ticker_collector_20260923.py; SELV=HL_BT_COIN; SELD=ETH; XV=HL_BT_CTX_ONLY; XD=""; XW=1; GROUP=1; RVAR=HL_BT_COIN
                TENV="HL_CTX_DB_PATH=data/hot/hl_ctx.sqlite"; W="data/hot/hl_ctx.sqlite data/hot/hl_ctx.sqlite-wal"
                CHK="sq:data/hot/hl_ctx.sqlite:data/hot/rtcollect/$( [ -e data/hot/rtcollect/hl_ctx_only.sqlite ] && echo hl_ctx_only || echo hl_ctx).sqlite" ;;
  depth)        PYSUP=supervisor_depth_diff.sh; PYRUN=live_depth_diff_collector_20260915.py; SELV=DD_SYMBOL; SELD=ethusdt; TENV="DD_ROOT=data/live/orderflow/depthdiff RTCOLLECT_REST=1" ;;
  raster)       PYSUP=supervisor_orderflow_raster.sh; PYRUN=live_orderflow_raster_collector_20260914.py; SELV=OF_SYMBOL; SELD=ethusdt; COPY="OF_BIN_SIZE OF_N_BINS"; TENV="OF_ROOT=data/live/orderflow/raster RTCOLLECT_REST=1" ;;
  ob5m)         PYSUP=supervisor_ob5m.sh; PYRUN=live_ob5m_20261026.py; SELV=OB_SYMBOL; SELD=ETHUSDT; RVAR=OB_SYMBOL ;;
  bookdepth)    PYSUP=supervisor_bookdepth.sh; PYRUN=live_bookdepth_feat_20261028.py; RVAR=OB_SYMBOL
                TENV="OB_DB=data/live/bookdepth/ethusdt.sqlite"; CHK="sq:data/live/bookdepth/ethusdt.sqlite:data/live/rtcollect/bookdepth/ethusdt.sqlite"; W="data/live/bookdepth/ethusdt.sqlite data/live/bookdepth/ethusdt.sqlite-wal" ;;
  iceberg)      PYSUP=supervisor_eth_iceberg.sh; PYRUN=live_eth_iceberg_20261016.py; SELV=ICEBERG_SYMBOL; SELD=ETHUSDT; TENV="ICEBERG_OUT=data/live/iceberg" ;;
  hl_trades)    ENG=rtduck; PYSUP=supervisor_hyperliquid_trades.sh; PYRUN=live_hyperliquid_trade_collector_20260916.py; SELV=HL_COINS; SELD=ETH; TENV="HL_ROOT=data/live/orderflow/hyperliquid" ;;
  market_data)  ENG=rtduck; PYSUP=supervisor_market_data_worker.sh; PYRUN=duckdb_persist_worker.py; XV=COLLECT_OI_LSRATIO; XD=false; XW=false; RVAR=TAIL_SYMBOLS; COPY="TAIL_SYMBOLS BOT_SYMBOLS"
                TENV="RTDUCK_LIVE_DIR=data/live QUANT_TAIL_DB_PATH=data/live/tail_risk.duckdb QUANT_MICRO_DB_PATH=data/live/microstructure.duckdb RTDUCK_REST=1"
                CHK="duck:data/live/microstructure.duckdb:data/live/rtduck/microstructure.duckdb duck:data/live/tail_risk.duckdb:data/live/rtduck/tail_risk.duckdb"; W="data/live/dashboard_state.json" ;;
  oi_lsratio)   ENG=rtduck; PYSUP=supervisor_oi_lsratio_worker.sh; PYRUN=duckdb_persist_worker.py; XV=COLLECT_OI_LSRATIO; XD=false; XW=true; RVAR=BOT_SYMBOLS; COPY=BOT_SYMBOLS
                TENV="QUANT_OI_LSRATIO_DB_PATH=data/live/oi_lsratio.duckdb RTDUCK_REST=1"; CHK="duck:data/live/oi_lsratio.duckdb:data/live/rtduck/oi_lsratio.duckdb"; W="data/live/oi_lsratio.duckdb data/live/oi_lsratio.duckdb.wal" ;;
  deribit)      ENG=rtduck; PYSUP=run_deribit_block_trade_collector_server.sh; PYRUN=live_deribit_block_trade_collector_20260928.py
                TENV="DERIBIT_BT_DIR=data/live RTDUCK_REST=1"; CHK="duck:data/live/deribit_options.duckdb:data/live/rtduck/deribit_options.duckdb"; W="data/live/deribit_options.duckdb data/live/deribit_options.duckdb.wal data/live/deribit_block_trades_state.json" ;;
  liq_tab)      ENG=rtduck; CRON=1; PYSUP=live_liq_tab_20261107; PYRUN=live_liq_tab_20261107.py; TENV="LIQ_TAB_LIVE=data/live RTDUCK_REST=1"; CHK="json:data/live:data/live/rtduck" ;;
  seal)         ENG=rtduck; CRON=1; PYSUP=seal_to_lake; PYRUN=seal_to_lake.py; TENV="SEAL_LAKE=data/lake SEAL_STATUS=data/live/ops_watchdog/seal_status.json"; CHK="lake:data/lake:data/rtduck/lake" ;;
  hl_bt|hl_positions|news|treasury_futures|macro_5m|liqmap_scorer|soxl_weekend|cboe_chain) die "$KIND 는 이 스크립트가 다루지 않는다 -- rtcollect/REMAINING.md «넘겨도 되는가» 표" ;;
  *) [ -n "$KIND" ] || { sed -n '2,12p' "$0"; exit 2; }; die "모르는 종류 '$KIND'" ;;
esac
case "$MODE" in check|go|back) ;; *) die "check|go|back 중 하나" ;; esac
ENGU="$(echo "$ENG" | tr '[:lower:]' '[:upper:]')"
BIN="$(eval echo "\${${ENGU}_BIN:-$ROOT/tmp/rtbin/$ENG}")"
SEL="$(nrm "${ARG:-$SELD}")"
SYML="$(echo "$SEL" | tr '[:upper:]' '[:lower:]')"
case "$KIND" in   # 종목마다 경로가 갈리는 종류
  depth)   CHK="depth:data/live/orderflow/depthdiff/$SEL:data/live/rtcollect/orderflow/depthdiff/$SEL"; W="data/live/orderflow/depthdiff/$SEL" ;;
  raster)  CHK="raster:data/live/orderflow/raster/$SEL:data/live/rtcollect/orderflow/raster/$SEL"; W="data/live/orderflow/raster/$SEL" ;;
  ob5m)    TENV="OB_DB=data/live/ob5m/$SYML.sqlite"; CHK="sq:data/live/ob5m/$SYML.sqlite:data/live/rtcollect/ob5m/$SYML.sqlite"; W="data/live/ob5m/$SYML.sqlite data/live/ob5m/$SYML.sqlite-wal" ;;
  iceberg) c="${SYML%usdt}"; CHK="events:data/live/iceberg/${c}_events.jsonl:data/live/rtcollect/iceberg/${c}_events.jsonl"; W="data/live/iceberg/${c}_state.json" ;;
  hl_trades) d="${SEL//,/_}"; CHK="duckdir:data/live/orderflow/hyperliquid/$d:data/live/rtduck/orderflow/hyperliquid/$d"; W="data/live/orderflow/hyperliquid/$d" ;;
esac
TAG="$KIND"; [ -z "$SELV" ] || [ "$GROUP" = 1 ] || TAG="${KIND}_${SEL//[,:]/_}"
TS="$(date +%Y%m%d_%H%M%S)"

# ── check ── 그림자가 떠 있던 창 전체(도구가 두 쪽 자료가 함께 있는 구간을 고른다)
hours() { # 그림자 폴더의 시 파일 -- 첫 시(그림자가 중간에 떴다)·끝 시(진행 중)는 뺀다
  local h; h="$(ls "$1" 2>/dev/null | sed -E 's/\..*//' | sort -u)"
  [ "$(echo "$h" | grep -c .)" -ge 3 ] || return 1
  echo "$(echo "$h" | sed -n 2p),$(echo "$h" | tail -2 | head -1)"
}
check() {
  local n=0 m=0 rc=0 e mode py rs out h f d0 d1
  for e in $CHK; do
    IFS=: read -r mode py rs <<< "$e"
    [ -e "$rs" ] || { echo "  $e: 그림자 출력 없음 -- 그림자를 먼저 띄운다"; rc=1; continue; }
    [ -e "$py" ] || { echo "  $e: 파이썬 출력 없음"; rc=1; continue; }
    case "$mode" in
      sq)     out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "$py" "$rs" --summary)" ;;
      events) out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --events "$py" "$rs" --summary)" ;;
      json)   out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --json 'liq_tab_*.json' "$py" "$rs" --summary)" ;;
      duck)   out="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" "$py" "$rs" --summary)" ;;
      depth|raster) h="$(hours "$rs")" || { echo "  $e: 그림자 시 파일이 3개 미만"; rc=1; continue; }
              out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "--$mode" "$py" "$rs" --hours "$h" --summary)" ;;
      lake)   d0="$(ls -d "$rs"/*/*/coin=*/date=* 2>/dev/null | sed 's/.*date=//' | sort | head -1)"; d1="$(ls -d "$rs"/*/*/coin=*/date=* 2>/dev/null | sed 's/.*date=//' | sort | tail -1)"
              [ -n "$d0" ] || { echo "  $e: 그림자 lake 가 비었다"; rc=1; continue; }
              out="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" --lake "$py" "$rs" --days "$d0,$d1" --summary)" ;;
      duckdir) out=""
              for f in "$rs"/*.duckdb; do [ -e "$py/$(basename "$f")" ] || continue
                out+="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" "$py/$(basename "$f")" "$f" --summary)"$'\n' || rc=1; done; true ;;
    esac
    [ "$?" = 0 ] || rc=1
    echo "$out" | grep -v '^요약' | sed 's/^/  /' | head -40
    while read -r a b; do n=$((n + a)); m=$((m + b)); done < <(echo "$out" | sed -nE 's/^요약: 견준 ([0-9,]+) · 설명 못 함 ([0-9,]+).*/\1 \2/p' | tr -d ,)
  done
  [ "$m" = 0 ] && [ "$n" -gt 0 ] || rc=1
  echo "check $KIND ${ARG:-}: 견준 $n · 설명 못 함 $m · $([ "$rc" = 0 ] && echo 통과 || echo '🔴불통과')"
  return "$rc"
}

# ── 프로세스 찾기 ── /proc 를 훑는다(pgrep -f 는 자기 명령줄에도 걸린다)
py_sel() { # 파이썬 수집기의 종목 -- ob5m 은 명령줄 --symbol(없으면 ETHUSDT)
  if [ "$KIND" = ob5m ]; then argv "$1" | grep -A1 -x -- '--symbol' | tail -1 | grep -v -- '--symbol' || echo ETHUSDT
  else local v; v="$(envof "$1" "$SELV")"; echo "${v:-$SELD}"; fi
}
py_procs() { # [any] -- any 면 종목을 안 본다
  local p pid
  for p in /proc/[0-9]*; do pid="${p#/proc/}"
    [ "$pid" = "$$" ] || [ "$pid" = "$BASHPID" ] && continue
    argv "$pid" | grep -qE "(^|/)${PYRUN//./\\.}\$" || continue
    is_sup "$pid" && continue
    [ -z "$XV" ] || { local x; x="$(envof "$pid" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
    [ -z "$SELV" ] || [ "$GROUP" = 1 ] || [ "${1:-}" = any ] || [ "$(nrm "$(py_sel "$pid")")" = "$SEL" ] || continue
    echo "$pid"
  done
}
rs_procs() { # shadow|takeover
  local pid t
  for pid in $(pgrep -x "$ENG"); do
    [ "$(envof "$pid" "${ENGU}_KIND")" = "$RK" ] || continue
    t="$(envof "$pid" "${ENGU}_TAKEOVER")"; if [ "$1" = takeover ]; then [ "$t" = 1 ] || continue; else [ "$t" = 1 ] && continue; fi
    [ "$XR" = 0 ] || { local x; x="$(envof "$pid" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
    if [ -n "$SELV" ] && [ "$GROUP" = 0 ]; then local v; v="$(envof "$pid" "${RVAR:-$SELV}")"; [ "$(nrm "${v:-$SELD}")" = "$SEL" ] || continue; fi
    echo "$pid"
  done
}
stop() { # stop <수집기 pid> <이름> -- 감독(_supervise.sh 부모) → 수집기 순 · 죽이기 직전 명령줄을 다시 본다
  local c="$1" s; s="$(ppid_of "$c")"
  argv "$c" | grep -qE 'server\.py|trading_bot|rtorder|ops_watchdog|rtserve' && die "pid $c 는 대시보드·주문·감시로 보인다 -- 멈춤"
  if [ -n "$s" ] && [ "$s" != 1 ] && is_sup "$s"; then
    kill -TERM "$s" 2>/dev/null; for _ in $(seq 50); do gone "$s" && break; sleep 0.2; done
    gone "$s" || die "$2 감독 pid $s 가 SIGTERM 10초 뒤에도 산다 -- 멈춤(되돌리기: back)"
    echo "  끔: $2 감독 pid $s"
  else echo "  ⚠️ $2 pid $c 의 부모($s)가 _supervise.sh 가 아니다 -- 수집기만 끈다"; fi
  kill -TERM "$c" 2>/dev/null; for _ in $(seq $((TERM_WAIT_S * 5))); do gone "$c" && break; sleep 0.2; done
  gone "$c" || die "$2 수집기 pid $c 가 SIGTERM ${TERM_WAIT_S}초 뒤에도 산다 -- 멈춤(되돌리기: back)"
  echo "  끔: $2 수집기 pid $c"
}

# ── crontab ──
cron_lines() { # 이 종류가 주석 처리할 줄 번호: 파이썬 줄 + 그림자 줄
  echo "$1" | awk '{print NR "\t" $0}' | while IFS=$'\t' read -r i l; do
    [[ "$l" =~ ^[[:space:]]*# ]] && continue
    if [[ "$l" == *"$PYSUP"* ]]; then
      if [ "$CRON" = 1 ]; then [[ "$l" == @reboot* ]] && continue; else [[ "$l" == @reboot* ]] || continue; fi
      if [ "$GROUP" = 1 ]; then [ "$(val "$l" "$XV")" = "$XW" ] || continue
      elif [ -n "$SELV" ]; then
        local s; s="$(val "$l" "$SELV")"
        [ "$KIND" = ob5m ] || [ "$KIND" = iceberg ] && { local p; p="$(echo "$l" | grep -oE "$PYSUP [A-Za-z0-9]+" | awk '{print $2}')"; s="${s:-$p}"; }
        [ "$(nrm "${s:-$SELD}")" = "$SEL" ] || continue
      fi
      [ "$XR" = 0 ] || { local x; x="$(val "$l" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
      echo "$i"
    elif [[ "$l" == *"supervisor_$ENG.sh $RK"* || "$l" == *"${ENGU}_KIND=$RK "* ]] && [[ "$l" != *"_TAKEOVER=1"* ]]; then   # 그림자 줄
      if [ -n "$SELV" ] && [ "$GROUP" = 0 ]; then local v; v="$(val "$l" "${RVAR:-$SELV}")"; [ "$(nrm "${v:-$SELD}")" = "$SEL" ] || continue; fi
      [ "$XR" = 0 ] || { local x; x="$(val "$l" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
      echo "$i"
    fi
  done
}
cron_apply() { # cron_apply <새 내용> <기대 줄 수>
  [ "$(printf '%s\n' "$1" | wc -l)" = "$2" ] || die "새 crontab 줄 수 $(printf '%s\n' "$1" | wc -l) ≠ 기대 $2 -- 적용 안 함"
  printf '%s\n' "$1" | $CT - || die "crontab 적용 실패"
  [ "$($CT -l)" = "$1" ] || die "적용한 crontab 을 다시 읽으니 다르다 -- 백업에서 손으로 되돌린다"
}
manifest() { # manifest take|back -- multicoin 매니페스트에서 이 스트림 항목을 taken_over 로 옮김(resume 가 파이썬을 다시 띄우지 않게) / 되돌림
  local f=data/live/multicoin_collectors.json
  [ -f "$f" ] || return 0
  [ "$1" = take ] && cp -p "$f" "logs/supervisor/multicoin_backup_${TS}_takeover_$TAG.json"
  M="$1" TAG="$TAG" PYRUN="$PYRUN" SELV="$SELV" SEL="$SEL" SELD="$SELD" GROUP="$GROUP" XV="$XV" XD="$XD" XW="$XW" F="$f" python3 - <<'PY'
import json, os
e = os.environ; p = e["F"]; m = json.load(open(p))
src, dst = ("collectors", "taken_over") if e["M"] == "take" else ("taken_over", "collectors")
moved = []
for k, c in list(m.get(src, {}).items()):
    if e["M"] == "back":
        ok = c.get("takeover_tag") == e["TAG"]
    else:
        mt = c.get("match", {})
        ok = c.get("runner") == e["PYRUN"] and (not e["XV"] or mt.get(e["XV"], e["XD"]) == e["XW"]) and (
            not e["SELV"] or e["GROUP"] == "1" or mt.get(e["SELV"], e["SELD"]).upper().replace(" ", "") == e["SEL"])
    if ok:
        c = m[src].pop(k)
        c.pop("takeover_tag", None) if e["M"] == "back" else c.update(takeover_tag=e["TAG"])
        m.setdefault(dst, {})[k] = c
        moved.append(k)
tmp = p + ".part"
json.dump(m, open(tmp, "w"), ensure_ascii=False, indent=1)
os.replace(tmp, p)
print(f"  매니페스트 {e['M']}: {moved or '해당 항목 없음'}")
PY
}
mt() { find $W -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1; }
logs_glob() { ls logs/supervisor/${ENG}_${RK}_*.log 2>/dev/null; }

# ── 실행 ──
[ "$MODE" = check ] && { check; exit $?; }
[ -x "$BIN" ] || die "실행 파일이 없다: $BIN"
CUR="$($CT -l 2>/dev/null)"; NCUR="$(printf '%s\n' "$CUR" | wc -l)"

if [ "$MODE" = back ]; then
  echo "back $TAG"
  for pid in $(rs_procs takeover); do stop "$pid" "Rust($RK)"; done
  NEW="$(printf '%s\n' "$CUR" | grep -vE " #takeover:$TAG\$" | sed "s|^#takeover:$TAG ||")"   # 더한 Rust 줄 = 끝 표지 · 주석 줄 = 앞 표지
  BACK="$(printf '%s\n' "$CUR" | grep -cE "^#takeover:$TAG ")"; ADDED="$(printf '%s\n' "$CUR" | grep -cE " #takeover:$TAG\$")"
  cron_apply "$NEW" $((NCUR - ADDED))
  echo "  crontab: 주석 푼 줄 $BACK · 지운 Rust 줄 $ADDED"
  B="$(ls -t logs/supervisor/crontab_backup_*_takeover_"$TAG".txt 2>/dev/null | head -1)"
  [ -z "$B" ] || { [ "$(cat "$B")" = "$NEW" ] && echo "  crontab = 백업 $B 와 같다" || echo "  ⚠️ crontab 이 백업 $B 와 다르다(그 뒤 다른 변경이 있었다 -- 이 종류 줄만 되돌렸다)"; }
  manifest back
  [ "$CRON" = 1 ] && { echo "back $TAG: 끝(예약 종류 -- 다음 시각부터 파이썬)"; exit 0; }
  printf '%s\n' "$CUR" | grep -E "^#takeover:$TAG @reboot" | sed "s|^#takeover:$TAG @reboot ||; s|^sleep [0-9]* && ||" | grep -F "$PYSUP" | while IFS= read -r cmd; do
    echo "  다시 띄움: $cmd"
    setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
  done
  sleep 5
  echo "back $TAG: 파이썬 수집기 $(py_procs | tr '\n' ' ')· Rust 진짜 경로 $(rs_procs takeover | wc -l)개 · 그림자는 다시 띄우지 않았다(shadow_start.sh)"
  exit 0
fi

# go
echo "go $TAG -- check 부터"
check || die "check 불통과 -- 아무것도 바꾸지 않았다"
[ -z "$(rs_procs takeover)" ] || die "이미 진짜 경로 Rust 가 돈다(pid $(rs_procs takeover | tr '\n' ' ')) -- 멈춤"
PY="$(py_procs)"
if [ "$CRON" = 0 ]; then
  [ -n "$PY" ] || die "넘길 파이썬 수집기가 안 돈다 -- 멈춤"
  [ "$GROUP" = 1 ] || [ "$(echo "$PY" | wc -l)" = 1 ] || die "맞는 파이썬 수집기가 여럿($(echo $PY)) -- 종목을 좁힌다"
  if [ "$SHARED" = 1 ] && [ -n "$(comm -13 <(echo "$PY" | sort) <(py_procs any | sort))" ]; then
    die "같은 파일에 쓰는 다른 파이썬 수집기가 있다($(comm -13 <(echo "$PY" | sort) <(py_procs any | sort) | tr '\n' ' ')) -- 단일 writer 를 지키려 멈춤"
  fi
  [ "$KIND" != hl_ctx ] || { XW=""; o="$(py_procs any)"; XW=1; [ -z "$o" ] || die "CTX_ONLY 가 아닌 HL 호가 수집기(pid $o)도 hl_ctx.sqlite 에 쓴다 -- 멈춤"; }
fi
# Rust 에 넘길 값: 진짜 경로 + 파이썬 environ 에서 그대로(COPY) + 종목
PY1="$(echo "$PY" | head -1)"
RENV="${ENGU}_TAKEOVER=1 $TENV"
for v in $COPY; do x="$(envof "$PY1" "$v")"; [ -z "$x" ] || RENV="$(echo "$RENV" | sed -E "s/(^| )$v=[^ ]*//g") $v=$x"; done
if [ "$GROUP" = 1 ]; then RENV+=" $RVAR=$(for p in $PY; do py_sel "$p"; done | paste -sd,)"
elif [ -n "$SELV" ]; then RENV+=" ${RVAR:-$SELV}=$( [ -n "$PY1" ] && py_sel "$PY1" || echo "$SEL")"
elif [ "$KIND" = bookdepth ]; then RENV+=" OB_SYMBOL=ETHUSDT"; fi
[ "$XR" = 0 ] || RENV="$(echo "$RENV" | sed -E "s/(^| )$XV=[^ ]*//g") $XV=$XW"
RENV="$(echo $RENV)"
# ① 백업 ② 새 crontab
B="logs/supervisor/crontab_backup_${TS}_takeover_$TAG.txt"; mkdir -p logs/supervisor; printf '%s\n' "$CUR" > "$B"
LN="$(cron_lines "$CUR")"
if [ "$CRON" = 1 ]; then
  first="$(echo "$LN" | head -1)"; [ -n "$first" ] || die "파이썬 예약 줄($PYSUP)을 못 찾음 -- 시각을 모른다 · 아무것도 안 바꿈"
  SCHED="$(printf '%s\n' "$CUR" | sed -n "${first}p" | awk '{print $1, $2, $3, $4, $5}')"
  ADD="$SCHED cd $ROOT && $RENV ${ENGU}_KIND=$RK $BIN >> logs/supervisor/rt_$RK.log 2>&1 #takeover:$TAG"
else
  ADD="@reboot sleep 100 && cd $ROOT && ${ENGU}_BIN=$BIN $RENV bash scripts/ops/supervisor_$ENG.sh $RK >> logs/supervisor/${ENG}_reboot.log 2>&1 & #takeover:$TAG"
fi
NEW="$(printf '%s\n' "$CUR" | awk -v L=",$(echo $LN | tr ' ' ','),"  -v T="#takeover:$TAG " 'index(L, "," NR ",") {print T $0; next} {print}')"$'\n'"$ADD"
NL="$(echo $LN | wc -w)"
[ "$(diff <(printf '%s\n' "$CUR") <(printf '%s\n' "$NEW" | sed '$d') | grep -c '^>')" = "$NL" ] || die "치환 검증 실패 -- 적용 안 함"
cron_apply "$NEW" $((NCUR + 1))
NPY="$(for i in $LN; do printf '%s\n' "$CUR" | sed -n "${i}p"; done | grep -cF "$PYSUP")"
echo "  ① crontab 백업 $B ② 주석 $NL줄(파이썬 $NPY · 그림자 $((NL - NPY))) + Rust 줄 1 · $NCUR → $((NCUR + 1))줄"
manifest take
if [ "$CRON" = 1 ]; then
  for _ in $(seq 300); do [ -z "$(py_procs any)" ] && break; sleep 1; done
  [ -z "$(py_procs any)" ] || die "지금 도는 파이썬 $PYRUN 이 5분 뒤에도 안 끝났다 -- 줄은 이미 바뀜(되돌리기: back)"
  echo "go $TAG: 예약 줄 바꿈 -- 다음 «$SCHED» 부터 Rust · 그 시각 뒤 check 한 번 더"; exit 0
fi
# ③ 파이썬 → ④ 그림자
for p in $PY; do stop "$p" "파이썬($PYRUN)"; done
T_DOWN="$(date +%s)"
[ -z "$(py_procs)" ] || die "파이썬 $PYRUN 이 다시 떴다(pid $(py_procs | tr '\n' ' ')) -- 멈춤(되돌리기: back)"
for p in $(rs_procs shadow); do stop "$p" "그림자 Rust($RK)"; done
# ⑤ 진짜 경로
declare -A LOG0; for f in $(logs_glob); do LOG0[$f]="$(stat -c %s "$f")"; done
echo "  ⑤ 띄움: $RENV $ENG $RK"
env "${ENGU}_BIN=$BIN" $RENV nohup setsid bash "scripts/ops/supervisor_$ENG.sh" "$RK" >> "logs/supervisor/${ENG}_takeover.log" 2>&1 < /dev/null &
# ⑥ 지켜보기
T_UP=""
for _ in $(seq "$WATCH_S"); do
  sleep 1
  [ -n "$T_UP" ] || { m="$(mt)"; [ -n "$m" ] && [ "$m" -gt "$T_DOWN" ] && T_UP="$m"; }
done
R="$(rs_procs takeover | tr '\n' ' ')"
BAD=0; for f in $(logs_glob); do BAD=$((BAD + $(tail -c +$(( ${LOG0[$f]:-0} + 1 )) "$f" | grep -ciE 'panic|crash-looping|SUPERVISOR .* exited'))); done
GAP="$([ -n "$T_UP" ] && echo "$((T_UP - T_DOWN))초" || echo "${WATCH_S}초 안 첫 쓰기 없음(쓰는 주기가 길면 정상 -- check 로)")"
echo "go $TAG: Rust pid ${R:-없음}· 끊김 $GAP · 로그 이상 $BAD · 다음 스트림 넘기기 전 check 한 번 더"
[ -n "$R" ] && [ "$BAD" = 0 ] || { echo "🔴 Rust 가 안 살았거나 로그에 이상 -- back 으로 되돌린다"; exit 1; }
