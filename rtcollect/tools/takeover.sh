#!/usr/bin/env bash
# 그림자 → 진짜 경로 넘기기 (2026-11-21 · 사용자 «기다리지 말고 바로바로 반영 배포»). 서버 전용 · 저장소 루트 기준 · **한 번에 한 스트림**.
#   bash rtcollect/tools/takeover.sh check <종류> [<종목>]  그림자가 떠 있던 창 전체를 대조(읽기 전용 · nice) -- 설명 못 하는 차이가 0 이 아니면(또는 견준 것이 0) 종료코드 1
#   bash rtcollect/tools/takeover.sh go    <종류> [<종목>]  check 가 0 일 때만: ① crontab 백업 ② 파이썬·그림자 줄 주석 + Rust 줄 · 줄 수 검증 뒤 적용
#                                                         ③ 파이썬 감독 → 수집기 끔 ④ 그림자 감독 → Rust 끔 ⑤ Rust 를 진짜 경로(TAKEOVER=1)로 ⑥ 60초 확인 · 끊긴 초
#   bash rtcollect/tools/takeover.sh back  <종류> [<종목>]  Rust 끔 → crontab 되돌림(이 종류가 바꾼 줄만) → 매니페스트 되돌림 → 파이썬 감독 다시
# 종류(상주): binance_tape binance_spot okx_tape okx_ctx bybit_tape bybit_liq binance_ctx xyz hl_ctx depth raster ob5m bookdepth iceberg hl_trades market_data oi_lsratio deribit
#      + hl_bt(코인마다 · 전체 모드 = .bt + hl_ctx.sqlite 맥락)
#      (예약): liq_tab seal raster_archive    · 안 다룸: hl_positions(대조 방법 없음) · news·treasury_futures·macro_5m(그림자 없음) · liqmap_scorer·soxl_weekend·cboe_chain(전진 장부 -- 손으로)
# 죽이는 것: 명령줄 인자가 그 종류의 러너 파일로 끝나고 environ 의 종목이 맞는 프로세스와 그 부모 _supervise.sh 뿐(PID 마다 /proc/<pid>/cmdline 다시 확인 · 이름으로 한꺼번에 안 죽인다).
#   SIGTERM 뒤 기다렸다가 살아 있으면 보고만 하고 멈춘다(그때는 `back` 으로 되돌린다). 대시보드·감시·주문은 러너가 달라 걸리지 않는다(걸리면 멈춘다).
#   bash rtcollect/tools/takeover.sh weight binance_ctx  바이낸스 가중치 머리(x-mbx-used-weight-1m)만 WEIGHT_N 번 읽어 최고를 찍는다(binance_ctx go 가 넘기기 전·뒤에 쓴다)
# 바꿔 끼우기(시험용): CRONTAB_CMD(기본 crontab) · TAKEOVER_ROOT · WATCH_S(60) · TERM_WAIT_S(120 -- 체결 테이프류는 SIGTERM 뒤 마지막 묶음을 쓰느라 30초를 넘긴다) · CHECK_PY · RTCOLLECT_BIN · RTDUCK_BIN
set -uo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TAKEOVER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 1
MODE="${1:-}"; KIND="${2:-}"; ARG="${3:-}"
CT="${CRONTAB_CMD:-crontab}"
WATCH_S="${WATCH_S:-60}"; TERM_WAIT_S="${TERM_WAIT_S:-120}"; STAGE=0
PYC="${CHECK_PY:-$HOME/miniconda3/envs/quant_ai/bin/python}"; [ -x "$PYC" ] || PYC=python3
die() { # crontab 을 바꾼 뒤(STAGE 2)의 실패는 그 자리에서 되돌린다 -- 어떤 길로도 «쓰는 프로세스 0» 으로 끝나지 않게
  echo "🔴 $*" >&2
  [ "$STAGE" -ge 2 ] && rollback "$*"
  exit 1
}
nrm() { echo "$1" | tr '[:lower:]' '[:upper:]' | tr -d ' '; }
envof() { tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep "^$2=" | head -1 | cut -d= -f2-; }
argv() { tr '\0' '\n' 2>/dev/null < "/proc/$1/cmdline"; }
val() { echo "$1" | grep -oE "(^| )$2=[^ ]+" | head -1 | sed -E "s/^ ?$2=//"; }
gone() { [ ! -e "/proc/$1" ] || [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" = Z ]; }
# 기다림은 «돈 횟수»가 아니라 벽시계 초(SECONDS)로 잰다 -- 11-22 hl_trades XRP go 에서 «120초 뒤에도 산다»가 9초 만에 나왔다
#   (횟수 고리 seq×sleep 0.2 가 실제로 얼마나 기다렸는지 남지 않았다 · sleep 이 바로 돌아오는 환경이면 횟수는 시간이 아니다)
nap() { sleep "$1" 2>/dev/null || read -rt "$1" _ <> <(:) 2>/dev/null || :; }
wait_gone() { # wait_gone <pid> <초> [이름] -- 그 초까지 벽시계로 기다린다 · 10초마다 «살아 있음 · mtime» · 끝에 실제로 기다린 초를 WAITED 에
  local p="$1" lim="$2" t0=$SECONDS nxt=$((SECONDS + 10)) m0 m1
  m0="$(mt)"
  while ! gone "$p" && [ $((SECONDS - t0)) -lt "$lim" ]; do
    nap 0.2
    if [ -n "${3:-}" ] && [ $SECONDS -ge $nxt ]; then
      m1="$(mt)"; echo "  기다림 $((SECONDS - t0))초: $3 pid $p 살아 있음 · 파일 mtime $([ "${m1:-0}" -gt "${m0:-0}" ] && echo 전진 || echo 멈춤)(마지막 묶음 쓰는 중이면 전진)"
      m0="$m1"; nxt=$((nxt + 10))
    fi
  done
  WAITED=$((SECONDS - t0))
  gone "$p"
}
ppid_of() { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '; }
is_sup() { argv "$1" | grep -qE '(^|/)_supervise\.sh$'; }

# ── 종류 표 ── ENG(rtcollect|rtduck) RK(Rust 종류) PYSUP(crontab 의 파이썬 줄 글자) PYRUN(파이썬 러너 파일) SELV·SELD(종목 변수·기본)
#   XV·XD·XW(파이썬 environ 에서 XV(기본 XD) 가 XW 여야 -- 같은 러너의 다른 스트림 가르기) XR(Rust·crontab 줄에도 X 를 적용) RVAR(Rust 종목 변수)
#   COPY(파이썬 environ 에서 그대로 넘길 변수) TENV(진짜 경로·REST) CHK(대조 목록 «모드:파이썬:Rust») W(끊긴 초를 재는 파일·폴더) CRON(예약 종류) SHARED(여러 종목이 한 파일)
ENG=rtcollect; RK="$KIND"; COVK=""; PYSUP=""; PYMAIN=""; ONCE=""; REQ=""; PYRUN=""; SELV=""; SELD=""; XV=""; XD=""; XW=""; XR=0; RVAR=""; COPY=""; TENV=""; CHK=""; W=""; CRON=0; SHARED=0; GROUP=0
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
                CHK="sq:data/hot/hl_ctx.sqlite:$(ls data/hot/rtcollect/hl_ctx.sqlite data/hot/rtcollect/hl_ctx_only.sqlite 2>/dev/null | paste -sd+)"   # hl_bt 그림자(전체 모드 코인) + hl_ctx 그림자(CTX_ONLY 코인)를 이어 하나로
                COVK="hl_ctx hl_bt" ;;
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
  hl_bt)        PYSUP=supervisor_hyperliquid_book_ticker.sh; PYRUN=live_hyperliquid_book_ticker_collector_20260923.py; SELV=HL_BT_COIN; SELD=ETH; XV=HL_BT_CTX_ONLY; XD=""; XW=""
                TENV="HL_BT_ROOT=data/live/orderflow/hyperliquid_bookticker HL_CTX_DB_PATH=data/hot/hl_ctx.sqlite"; COVK="hl_bt" ;;   # 경로는 종목을 안 뒤에(아래)
  raster_archive) ENG=rtduck; CRON=1; PYSUP=""; PYRUN="__none__"; TENV="OF_ROOT=data/live/orderflow/raster OF_ARCHIVE_ROOT=data/live/orderflow/raster_archive" ;;
  # 한 회차 대조 종류(그림자 없음 · check 가 Rust 를 그 자리에서 한 번 그림자 경로로 돌려 파이썬이 쓴 것과 견준다 · REQ = 그 한 번의 바깥 요청)
  treasury_futures) CRON=1; PYSUP=live_treasury_futures_collector_20261006; PYRUN=live_treasury_futures_collector_20261006.py; ONCE="--range 5d"
                TENV="TREASURY_FUTURES_DB=data/hot/treasury_futures.sqlite RTCOLLECT_REST=1"; CHK="once_sq:data/hot/treasury_futures.sqlite:data/hot/rtcollect/treasury_futures.sqlite"
                REQ="Yahoo 차트 --range 5d 종목마다 1회(cron.rs TREASURY 종목 수)" ;;
  macro_5m)     CRON=1; PYSUP=live_macro_5m_collector_20261028; PYRUN=live_macro_5m_collector_20261028.py
                TENV="MACRO_5M_DB=data/hot/macro_5m.sqlite RTCOLLECT_REST=1"; CHK="once_sq:data/hot/macro_5m.sqlite:data/hot/rtcollect/macro_5m.sqlite"
                REQ="Yahoo 5분봉 종목마다 1회(1.5초 간격)" ;;
  liqmap_scorer) CRON=1; PYSUP=live_liqmap_scorer_20261008; PYRUN=live_liqmap_scorer_20261008.py
                TENV="LIQMAP_OUT=data/live/liqmap_score.jsonl LIQMAP_STATUS=data/live/liqmap_score_status.json RTCOLLECT_REST=1"
                CHK="once_jsonl:data/live/liqmap_score.jsonl:data/live/rtcollect/liqmap_score.jsonl:origin"; REQ="바이낸스 klines 1h×198 1회(가중치 2)" ;;
  soxl_weekend) CRON=1; PYSUP=live_soxl_weekend_forward_recorder_20261022; PYRUN=live_soxl_weekend_forward_recorder_20261022.py; ONCE="--record"
                TENV="SOXL_WEEKEND_OUT=data/live/soxl_weekend_forward.jsonl RTCOLLECT_REST=1"
                CHK="once_jsonl:data/live/soxl_weekend_forward.jsonl:data/live/rtcollect/soxl_weekend_forward.jsonl:weekend"; REQ="끝난 새 주말이 있을 때만 klines 8 + fundingRate 1" ;;
  cboe_chain)   CRON=1; PYSUP="live_cboe_option_chain_collector_20261101|cboe_option_summary_20261101"; PYMAIN=live_cboe_option_chain_collector_20261101; PYRUN=live_cboe_option_chain_collector_20261101.py
                TENV="CBOE_OUT=data/live/options_equity/cboe CBOE_STATE=data/live/cboe_option_state.json RTCOLLECT_REST=1"
                CHK="once_cboe:data/live/cboe_option_state.json:data/live/rtcollect/cboe_option_state.json"; REQ="CBOE 체인 종목마다 1 + 바이낸스 klines 1h 종목마다 하루 1(캐시)" ;;
  news)         PYSUP=supervisor_news_rss.sh; PYRUN=live_news_rss_collector_20261005.py; TENV="NEWS_RSS_DB=data/hot/news_rss.sqlite"
                CHK="once_news:data/hot/news_rss.sqlite:data/hot/rtcollect/news_rss.sqlite"; REQ="RSS 8곳을 원본 주기대로 NEWS_S(기본 180)초 동안(같은 IP 두 배는 그 동안만)" ;;
  hl_positions) die "$KIND 는 이 스크립트가 다루지 않는다 -- rtcollect/REMAINING.md «넘겨도 되는가» 표" ;;
  *) [ -n "$KIND" ] || { sed -n '2,12p' "$0"; exit 2; }; die "모르는 종류 '$KIND'" ;;
esac
case "$MODE" in check|go|back|weight|weight-judge) ;; *) die "check|go|back 중 하나" ;; esac
ENGU="$(echo "$ENG" | tr '[:lower:]' '[:upper:]')"
BIN="$(eval echo "\${${ENGU}_BIN:-$ROOT/tmp/rtbin/$ENG}")"
COVK="${COVK:-$RK}"
SEL="$(nrm "${ARG:-$SELD}")"   # 아래 resolve 가 도는 파이썬 수집기의 목록 통째로 바꾼다
TS="$(date +%Y%m%d_%H%M%S)"

# ── check ── 그림자가 떠 있던 창 전체(도구가 두 쪽 자료가 함께 있는 구간을 고른다)
hours() { # 그림자 폴더의 시 파일 -- 첫 시(그림자가 중간에 떴다)·끝 시(진행 중)는 뺀다
  local h; h="$(ls "$1" 2>/dev/null | sed -E 's/\..*//' | sort -u)"
  [ "$(echo "$h" | grep -c .)" -ge 3 ] || return 1
  echo "$(echo "$h" | sed -n 2p),$(echo "$h" | tail -2 | head -1)"
}
starts() { # 그림자 감독 로그(logs/supervisor/<ENG>_<종류>_<목록>_<날짜>.log)의 «SUPERVISOR starting» 시각(초, 쉼표) -- 도구가 그 앞뒤 Rust 빈 구간을 뺀다
  local k f pre
  for k in $COVK; do
    pre="${ENG}_${k}_"; [ -z "$SELV" ] || [ "$GROUP" = 1 ] || pre="${ENG}_${k}_${SEL//[,:]/_}_"
    for f in logs/supervisor/${ENG}_${k}_*.log; do [ -f "$f" ] || continue
      [[ "$(basename "$f" | tr '[:lower:]' '[:upper:]')" == "$(echo "$pre" | tr '[:lower:]' '[:upper:]')"* ]] || continue
      grep -h 'SUPERVISOR starting' "$f" | sed -nE 's/^\[([^]]+)\].*/\1/p'
    done
  done | while read -r t; do date -d "$t" +%s 2>/dev/null; done | sort -un | paste -sd,
}
reconn() { # 같은 그림자 로그의 «끊김» 줄 수(연결 끊김 -- 재기동 아님) · shadow_tape 가 그만큼의 «Rust 만 빈 곳»을 설명으로 친다
  local k f pre n=0
  for k in $COVK; do
    pre="${ENG}_${k}_"; [ -z "$SELV" ] || [ "$GROUP" = 1 ] || pre="${ENG}_${k}_${SEL//[,:]/_}_"
    for f in logs/supervisor/${ENG}_${k}_*.log; do [ -f "$f" ] || continue
      [[ "$(basename "$f" | tr '[:lower:]' '[:upper:]')" == "$(echo "$pre" | tr '[:lower:]' '[:upper:]')"* ]] || continue
      n=$((n + $(grep -c '끊김' "$f")))
    done
  done
  echo "$n"
}
in_list() { local x; for x in ${1//,/ }; do [[ ",$2," == *",$x,"* ]] || return 1; done; }   # 앞 목록이 모두 뒤 목록에 있나
coverage() { # 그림자 Rust 프로세스들이 파이썬 목록 전부를 덮는가(덮지 않는 종목이 있으면 그 종목은 견준 적이 없다 → 불통과)
  local cv="${SELV:-${COPY%% *}}" pl="" rl="" p k v x miss=""
  [ -n "$cv" ] || return 0
  for p in $( [ "$KIND" = hl_ctx ] && { XV=""; py_procs any; } || py_procs); do   # hl_ctx 는 CTX_ONLY·전체 모드 둘 다 hl_ctx.sqlite 에 쓴다
    v="$( [ -n "$SELV" ] && py_sel "$p" || envof "$p" "$cv")"; pl="${pl:+$pl,}$(nrm "$v")"
  done
  [ -n "$pl" ] || { echo "  목록: 파이썬 $cv 를 못 읽음(수집기가 안 돌거나 기본값) -- 덮음 확인 건너뜀"; return 0; }
  for p in $(pgrep -x "$ENG"); do
    k="$(envof "$p" "${ENGU}_KIND")"; [[ " $COVK " == *" $k "* ]] || continue
    [ "$(envof "$p" "${ENGU}_TAKEOVER")" = 1 ] && continue
    [ "$XR" = 0 ] || { x="$(envof "$p" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
    v="$(envof "$p" "${RVAR:-$cv}")"; [ -z "$v" ] || rl="${rl:+$rl,}$(nrm "$v")"
  done
  for x in ${pl//,/ }; do in_list "$x" "$rl" || miss="${miss:+$miss,}$x"; done
  if [ -n "$miss" ]; then echo "  🔴목록: 파이썬 $cv=$pl 중 그림자 Rust 가 안 뜨는 종목 $miss (그림자 목록 ${rl:-없음})"; return 1; fi
  echo "  목록: 파이썬 $cv=$pl 전부 그림자가 뜬다"
}
replay_check() { # ob5m·bookdepth: 실시간 그림자 대신 «같은 시작점 재생 대조»(replay_book.py) -- 사슬 시작·이력에 달린 칸은 그림자로 못 보인다
  local s="${SEL:-ETHUSDT}"; local f="logs/supervisor/replay_book_${KIND}_${s}.txt" end
  if [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 7200 ] && grep -q '^요약' "$f"; then
    echo "  재생 대조: 2시간 안 결과를 다시 쓴다($f · $(date -d @"$(stat -c %Y "$f")" '+%T'))"
  else
    end="$(date -u -d '1 hour ago' +%Y-%m-%dT%H)"
    echo "  재생 대조: 끝 시 $end(UTC) · nice · 무겁다(파이썬 수 분)"
    # -u + 도구 안 줄 단위 출력: 진행 줄이 바로 파일·화면에 · 상한 REPLAY_LIMIT_S(기본 1800초)를 넘으면 도구가 «시간 초과(단계)» 로 끝낸다
    nice -n 10 "$PYC" -u "$TOOLS/replay_book.py" "$KIND" --symbol "$s" --end-hour "$end" --bin "${RTCOLLECT_BIN:-$ROOT/tmp/rtbin/rtcollect}" \
      --limit "${REPLAY_LIMIT_S:-1800}" 2>&1 | tee "$f.part" | sed 's/^/    /'
    mv "$f.part" "$f"; REPLAY_SHOWN=1
  fi
  [ -n "${REPLAY_SHOWN:-}" ] || sed 's/^/    /' "$f" | tail -12
  local nm; nm="$(sed -nE 's/^요약: 견준 ([0-9,]+) · 설명 못 함 ([0-9,]+).*/\1 \2/p' "$f" | tr -d ,)"
  read -r a b <<< "${nm:-0 1}"
  echo "check $KIND $s (재생): 견준 $a · 설명 못 함 $b · $([ "$b" = 0 ] && [ "$a" -gt 0 ] && echo 통과 || echo '🔴불통과')"
  [ "$b" = 0 ] && [ "$a" -gt 0 ]
}
archive_check() { # raster_archive: 원본은 래스터 수집기 안(매시 _purge_old) -- 래스터를 넘긴 종목만 Rust 보관이 맡는다
  local syms; syms="$(printf '%s\n' "$($CT -l 2>/dev/null)" | grep -oE ' #takeover:raster_[A-Z0-9]+$' | sed 's/.*raster_//' | sort -u | paste -sd,)"
  [ -n "$syms" ] || { echo "check raster_archive: 넘긴 래스터가 없다(#takeover:raster_… 표지 0) -- 🔴불통과(넘길 것이 없다)"; return 1; }
  local log=logs/supervisor/rt_raster_archive.log bad=0
  [ -f "$log" ] && bad="$(tail -n 200 "$log" | grep -ciE 'panic|error|실패')"
  echo "  넘긴 래스터 종목 $syms · 그림자 보관 로그 끝 200줄의 오류 $bad · 값 대조 = 정답지(raster_archive_parity · 양방향 읽기)뿐(그림자 보관은 그림자 래스터 파일을 보관해 원본 보관본과 견줄 수 없다)"
  echo "check raster_archive: $([ "$bad" = 0 ] && echo 통과 || echo '🔴불통과')"
  OF_SYMS="$syms"
  [ "$bad" = 0 ]
}
py_args() { # 파이썬 예약 줄의 스크립트 뒤 인자(>> 앞까지) -- 첫 줄
  printf '%s\n' "$($CT -l 2>/dev/null)" | grep -vE '^[[:space:]]*#' | grep -E "${PYMAIN:-$PYSUP}" | head -1 | sed -E "s/.*(${PYMAIN:-$PYSUP})[^ ]*//; s/>>.*//; s/2>&1//" | xargs
}
once_check() { # Rust 를 그림자 경로로 한 번 돌리고(REST 켬) 파이썬이 쓴 것과 견준다
  local e mode py rs key out rc=0 a b
  echo "  바깥 요청(이 check 한 번): ${REQ:-없음}"
  for e in $CHK; do
    IFS=: read -r mode py rs key <<< "$e"
    [ -e "$py" ] || { echo "  $py: 파이썬 출력 없음"; return 1; }
    local args="$ONCE"; [ "$KIND" != cboe_chain ] || args="$(py_args)"
    echo "  Rust 한 회차: RTCOLLECT_KIND=$RK $args (그림자 경로)"
    if [ "$mode" = once_news ]; then
      env RTCOLLECT_KIND=news timeout "${NEWS_S:-180}" "$BIN" > logs/supervisor/takeover_once_news.log 2>&1
    else
      env RTCOLLECT_KIND="$RK" RTCOLLECT_REST=1 "$BIN" $args > "logs/supervisor/takeover_once_$RK.log" 2>&1 || { echo "  🔴Rust 한 회차 실패: $(tail -3 "logs/supervisor/takeover_once_$RK.log")"; return 1; }
    fi
    case "$mode" in
      once_sq)   out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "$py" "$rs" --to $(( $(date +%s) - 600 )) --summary)" || rc=1 ;;   # 끝 10분 = 아직 자라는 봉
      once_news) out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "$py" "$rs" --subset --summary)" || rc=1 ;;   # 파이썬이 오래 모은 것은 견주지 않는다 -- Rust 가 받은 것이 파이썬에 같은 글자로 있나
      once_jsonl) out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --jsonl "$key" "$py" "$rs" --summary)" || rc=1 ;;
      once_cboe) out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --cboe "$py" "$rs" --summary)" || rc=1 ;;
    esac
    echo "$out" | sed 's/^/  /' | head -40
    read -r a b <<< "$(echo "$out" | sed -nE 's/^요약: 견준 ([0-9,]+) · 설명 못 함 ([0-9,]+).*/\1 \2/p' | tr -d ,)"
  done
  echo "check $KIND (한 회차): 견준 ${a:-0} · 설명 못 함 ${b:-?} · $([ "$rc" = 0 ] && echo 통과 || echo '🔴불통과')"
  return "$rc"
}
check() {
  [[ "$CHK" == once_* ]] && { once_check; return $?; }
  [ "$KIND" = raster_archive ] && { archive_check; return $?; }
  [ "$KIND" = ob5m ] || [ "$KIND" = bookdepth ] && { replay_check; return $?; }
  local n=0 m=0 rc=0 e mode py rs out h f d0 d1
  ST="$(starts)"; RC="$(reconn)"
  echo "  그림자 감독 기동 시각 $(echo "$ST" | tr , '\n' | grep -c .)개 · «끊김» 줄 $RC(로그) -- 그 앞뒤·그만큼의 Rust 빈 구간은 뺀다"
  coverage || rc=1
  for e in $CHK; do
    IFS=: read -r mode py rs <<< "$e"
    [ -n "$rs" ] && [ -z "$(for f in ${rs//+/ }; do [ -e "$f" ] || echo x; done)" ] || { echo "  $e: 그림자 출력 없음 -- 그림자를 먼저 띄운다"; rc=1; continue; }
    [ -e "$py" ] || { echo "  $e: 파이썬 출력 없음"; rc=1; continue; }
    case "$mode" in
      sq)     out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "$py" "$rs" --starts "$ST" --reconnects "$RC" --summary)" ;;
      events) out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --events "$py" "$rs" --summary)" ;;
      json)   out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --json 'liq_tab_*.json' "$py" "$rs" --summary)" ;;
      duck)   out="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" "$py" "$rs" --starts "$ST" --summary)" ;;
      bt)     h="$(hours "$rs")" || { echo "  $e: 그림자 시 파일이 3개 미만"; rc=1; continue; }
              out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" --bt "$py" "$rs" --hours "$h" --starts "$ST" --reconnects "$RC" --summary)" ;;
      depth|raster) h="$(hours "$rs")" || { echo "  $e: 그림자 시 파일이 3개 미만"; rc=1; continue; }
              out="$(nice -n 10 "$PYC" "$TOOLS/shadow_tape.py" "--$mode" "$py" "$rs" --hours "$h" --starts "$ST" --summary)" ;;
      lake)   d0="$(ls -d "$rs"/*/*/coin=*/date=* 2>/dev/null | sed 's/.*date=//' | sort | head -1)"; d1="$(ls -d "$rs"/*/*/coin=*/date=* 2>/dev/null | sed 's/.*date=//' | sort | tail -1)"
              [ -n "$d0" ] || { echo "  $e: 그림자 lake 가 비었다"; rc=1; continue; }
              out="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" --lake "$py" "$rs" --days "$d0,$d1" --summary)" ;;
      duckdir) out=""
              for f in "$rs"/*.duckdb; do [ -e "$py/$(basename "$f")" ] || continue
                out+="$(nice -n 10 "$PYC" "$TOOLS/shadow_duck.py" "$py/$(basename "$f")" "$f" --starts "$ST" --summary)"$'\n' || rc=1; done; true ;;
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
rs_procs() { # shadow|takeover [any]
  local pid t
  for pid in $(pgrep -x "$ENG"); do
    [ "$(envof "$pid" "${ENGU}_KIND")" = "$RK" ] || continue
    t="$(envof "$pid" "${ENGU}_TAKEOVER")"; if [ "$1" = takeover ]; then [ "$t" = 1 ] || continue; else [ "$t" = 1 ] && continue; fi
    [ "$XR" = 0 ] || { local x; x="$(envof "$pid" "$XV")"; [ "${x:-$XD}" = "$XW" ] || continue; }
    if [ -n "$SELV" ] && [ "$GROUP" = 0 ] && [ "${2:-}" != any ]; then local v; v="$(envof "$pid" "${RVAR:-$SELV}")"; [ "$(nrm "${v:-$SELD}")" = "$SEL" ] || continue; fi
    echo "$pid"
  done
}
# 끈 프로세스를 그대로 다시 띄우려고 argv·environ·cwd 를 메모리에만 둔다(디스크에 안 적는다 -- environ 에 비밀 값이 있을 수 있다)
SV_NAME=(); SV_CWD=(); SV_ARGV=(); SV_ENV=(); SV_SUP=(); SV_COLL=()
save() { # save <다시 띄울 pid(감독 또는 수집기)> <수집기 pid> <이름>
  SV_NAME+=("$3"); SV_CWD+=("$(readlink "/proc/$1/cwd")"); SV_ARGV+=("$(argv "$1")")
  SV_ENV+=("$(tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=')"); SV_SUP+=("$1"); SV_COLL+=("$2")
}
relaunch() { # relaunch <번호> -- 끈 그대로(같은 argv·environ·cwd)
  local i="$1" a e
  mapfile -t a <<< "${SV_ARGV[$i]}"; mapfile -t e <<< "${SV_ENV[$i]}"
  ( cd "${SV_CWD[$i]}" && exec env -i "${e[@]}" setsid nohup "${a[@]}" > /dev/null 2>&1 < /dev/null ) &
  echo "  다시 띄움: ${SV_NAME[$i]} -- ${a[*]:0:4} …"
}
stop() { # stop <수집기 pid> <이름> [nosave] -- 감독(_supervise.sh 부모) → 수집기 순 · 죽이기 직전 명령줄을 다시 본다 · 실패면 1(감독)·2(수집기)
  local c="$1" s m0 m1 k; s="$(ppid_of "$c")"
  argv "$c" | grep -qE 'server\.py|trading_bot|rtorder|ops_watchdog|rtserve' && die "pid $c 는 대시보드·주문·감시로 보인다 -- 멈춤"
  if [ -n "$s" ] && [ "$s" != 1 ] && is_sup "$s"; then
    [ "${3:-}" = nosave ] || save "$s" "$c" "$2 감독"
    kill -TERM "$s" 2>/dev/null
    wait_gone "$s" 10 || { echo "  🔴$2 감독 pid $s 가 SIGTERM ${WAITED}초 뒤에도 산다"; return 1; }
    echo "  끔: $2 감독 pid $s"
  else
    echo "  ⚠️ $2 pid $c 의 부모($s)가 _supervise.sh 가 아니다 -- 수집기만 끈다"
    [ "${3:-}" = nosave ] || save "$c" "$c" "$2 수집기"
  fi
  kill -TERM "$c" 2>/dev/null
  wait_gone "$c" "$TERM_WAIT_S" "$2 수집기" || {
    echo "  🔴$2 수집기 pid $c 가 SIGTERM ${WAITED}초 뒤에도 산다(기다림 한도 ${TERM_WAIT_S}초 · 감독은 이미 꺼짐) -- $(ps -o pid=,stat=,etime=,args= -p "$c" 2>/dev/null | cut -c1-160)"
    return 2; }
  echo "  끔: $2 수집기 pid $c (SIGTERM 뒤 ${WAITED}초)"
}
writers() { # 끝날 때마다 찍는다 -- 0 이면 종료코드 9
  local np nr ns
  np="$(py_procs | grep -c .)"; nr="$(rs_procs takeover | grep -c .)"; ns="$(rs_procs shadow | grep -c .)"
  echo "지금 쓰는 프로세스($TAG): 파이썬 $np · Rust(진짜 경로) $nr · 그림자 $ns"
  if [ "$CRON" = 0 ] && [ $((np + nr)) = 0 ]; then
    echo "████████████████████████████████████████████████████████████"
    echo "██  🔴🔴 $TAG 를 쓰는 프로세스가 0 개다 -- 지금 자료가 빈다  ██"
    echo "██  손으로: bash rtcollect/tools/takeover.sh back $KIND         ██"
    echo "████████████████████████████████████████████████████████████"
    return 9
  fi
}
cron_back() { # cron_back <태그> -- 그 표지가 바꾼 줄만 되돌린다(더한 Rust 줄 = 끝 표지 지움 · 주석 줄 = 앞 표지 풂)
  local cur new nb na
  cur="$($CT -l 2>/dev/null)"
  new="$(printf '%s\n' "$cur" | grep -vE " #takeover:$1\$" | sed "s|^#takeover:$1 ||")"
  nb="$(printf '%s\n' "$cur" | grep -cE "^#takeover:$1 ")"; na="$(printf '%s\n' "$cur" | grep -cE " #takeover:$1\$")"
  cron_apply "$new" $(( $(printf '%s\n' "$cur" | wc -l) - na ))
  echo "  crontab: 주석 푼 줄 $nb · 지운 Rust 줄 $na"
  CRON_RESTORED="$cur"   # 되돌리기 전 내용(back 이 파이썬 줄을 다시 띄울 때 쓴다)
}
rollback() {
  STAGE=0   # 되돌리는 동안의 실패는 다시 되돌리지 않는다
  echo "🔴 중간 실패($1) -- 스스로 되돌린다"
  local p i c
  for p in $(rs_procs takeover); do stop "$p" "Rust($RK)" nosave; done
  cron_back "$TAG"
  manifest back
  for i in "${!SV_NAME[@]}"; do
    gone "${SV_SUP[$i]}" || { echo "  ${SV_NAME[$i]} pid ${SV_SUP[$i]} 는 아직 산다 -- 그대로 둔다"; continue; }
    c="${SV_COLL[$i]}"
    if ! gone "$c"; then   # 감독 없이 수집기만 혼자 쓰는 중 -- 같은 파일을 둘이 열지 않게 끝나는 즉시 감독을 다시 띄우는 지킴이
      ( trap '' HUP; while ! gone "$c"; do sleep 1; done; relaunch "$i" >> logs/supervisor/takeover_rollback.log 2>&1 ) &
      echo "  ${SV_NAME[$i]}: 수집기 pid $c 가 혼자 산다 -- 끝나는 즉시 감독을 다시 띄우는 지킴이를 남김(logs/supervisor/takeover_rollback.log)"
    else relaunch "$i"; fi
  done
  sleep 3
  writers; local w=$?
  echo "되돌림 끝 -- 종료코드 $([ "$w" = 9 ] && echo 9 || echo 8)"
  exit $([ "$w" = 9 ] && echo 9 || echo 8)
}

# ── crontab ──
cron_lines() { # 이 종류가 주석 처리할 줄 번호: 파이썬 줄 + 그림자 줄
  echo "$1" | awk '{print NR "\t" $0}' | while IFS=$'\t' read -r i l; do
    [[ "$l" =~ ^[[:space:]]*# ]] && continue
    if [ -n "$PYSUP" ] && [[ "$l" =~ $PYSUP ]]; then
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

# ── 종목 = 도는 파이썬 수집기의 목록 통째로 ── 인자가 없으면 그 종류(XV 조건)의 수집기 하나 · 있으면 그 종목이 든 수집기(인자가 목록이면 모두 든 것)
resolve() {
  [ -n "$SELV" ] && [ "$GROUP" = 0 ] || return 0
  local want hit=() p l
  want="$(nrm "$ARG")"
  for p in $(py_procs any); do
    l="$(nrm "$(py_sel "$p")")"
    { [ -z "$want" ] || in_list "$want" "$l"; } && hit+=("$p:$l")
  done
  if [ ${#hit[@]} = 0 ] && [ "$MODE" = back ]; then   # 되돌릴 때는 파이썬이 꺼져 있다 -- 진짜 경로 Rust 의 목록으로
    for p in $(rs_procs takeover any); do
      l="$(nrm "$(envof "$p" "${RVAR:-$SELV}")")"
      { [ -z "$want" ] || in_list "$want" "$l"; } && hit+=("$p:$l")
    done
  fi
  case ${#hit[@]} in
    0) ;;
    1) SEL="${hit[0]#*:}" ;;
    *) die "맞는 파이썬 수집기가 여럿 -- 종목을 좁힌다(pid:목록): ${hit[*]}" ;;
  esac
}
resolve
SYML="$(echo "$SEL" | tr '[:upper:]' '[:lower:]')"
case "$KIND" in   # 종목마다 경로가 갈리는 종류(이 종류들은 수집기 하나 = 종목 하나)
  depth)   CHK="depth:data/live/orderflow/depthdiff/$SEL:data/live/rtcollect/orderflow/depthdiff/$SEL"; W="data/live/orderflow/depthdiff/$SEL" ;;
  raster)  CHK="raster:data/live/orderflow/raster/$SEL:data/live/rtcollect/orderflow/raster/$SEL"; W="data/live/orderflow/raster/$SEL" ;;
  ob5m)    TENV="OB_DB=data/live/ob5m/$SYML.sqlite"; CHK="sq:data/live/ob5m/$SYML.sqlite:data/live/rtcollect/ob5m/$SYML.sqlite"; W="data/live/ob5m/$SYML.sqlite data/live/ob5m/$SYML.sqlite-wal" ;;
  iceberg) c="${SYML%usdt}"; CHK="events:data/live/iceberg/${c}_events.jsonl:data/live/rtcollect/iceberg/${c}_events.jsonl"; W="data/live/iceberg/${c}_state.json" ;;
  hl_bt)   CHK="bt:data/live/orderflow/hyperliquid_bookticker/$SEL:data/live/rtcollect/orderflow/hyperliquid_bookticker/$SEL sq:data/hot/hl_ctx.sqlite:$(ls data/hot/rtcollect/hl_ctx.sqlite data/hot/rtcollect/hl_ctx_only.sqlite 2>/dev/null | paste -sd+)"
           W="data/live/orderflow/hyperliquid_bookticker/$SEL" ;;
  hl_trades) d="${SEL//,/_}"; CHK="duckdir:data/live/orderflow/hyperliquid/$d:data/live/rtduck/orderflow/hyperliquid/$d"; W="data/live/orderflow/hyperliquid/$d" ;;
esac
TAG="$KIND"; [ -z "$SELV" ] || [ "$GROUP" = 1 ] || TAG="${KIND}_${SEL//[,:]/_}"

# ── 실행 ──
# ── 바이낸스 가중치(binance_ctx 만) ── 응답 머리 x-mbx-used-weight-1m(이 IP 의 지난 1분 사용량)을 WEIGHT_GAP 초 간격으로 WEIGHT_N 번 · 가장 큰 값
#   부르는 곳 = /fapi/v1/time(가중치 1) · 밴 파일이 살아 있으면 부르지 않는다(멈춤). 시험은 WEIGHT_URL 을 127.0.0.1 로.
WEIGHT_URL="${WEIGHT_URL:-https://fapi.binance.com/fapi/v1/time}"; WEIGHT_N="${WEIGHT_N:-5}"; WEIGHT_GAP="${WEIGHT_GAP:-60}"
weight_max() { # 값 WEIGHT_N 개를 빈칸으로(이름은 옛 것 -- 최고가 아니라 전부를 낸다)
  local i w all="" ban=data/live/binance_ban_until
  for ((i = 1; i <= WEIGHT_N; i++)); do
    [ "$i" = 1 ] || nap "$WEIGHT_GAP"
    if [ -f "$ban" ] && [ "$(cut -d. -f1 < "$ban" 2>/dev/null || echo 0)" -gt "$(date +%s)" ] 2>/dev/null; then echo "밴 중" >&2; return 1; fi
    w="$(curl -s -D - -o /dev/null --max-time 10 "$WEIGHT_URL" | tr -d '\r' | sed -nE 's/^[Xx]-[Mm][Bb][Xx]-[Uu][Ss][Ee][Dd]-[Ww][Ee][Ii][Gg][Hh][Tt]-1[Mm]: *([0-9]+).*/\1/p')"
    [ -n "$w" ] || { echo "가중치 머리를 못 읽음" >&2; return 1; }
    echo "  가중치 $i/$WEIGHT_N: $w" >&2
    all="${all:+$all }$w"
  done
  echo "$all"
}
median() { tr ' ' '\n' <<< "$1" | grep . | sort -n | awk '{v[NR]=$1} END {print (NR % 2) ? v[(NR + 1) / 2] : int((v[NR / 2] + v[NR / 2 + 1]) / 2)}'; }
weight_judge() { # weight_judge "<전 값들>" "<뒤 값들>" -- 되돌릴 까닭을 찍고 1, 아니면 0
  # 근거: 가중치 머리는 같은 IP 의 모든 프로세스 몫(대시보드·봇·다른 수집기)이라 한 번 튀는 값은 우리 몫이 아니다(11-22: 뒤 다섯 중 1382 하나만 튐 ·
  #   나머지 넷은 전과 같음 · 수집기 자체 요청률은 508.8 대 512.2 = 0.66%). 늘 얹히는 몫은 «중앙값»으로, 연달아 넘는 것은 «두 번 연속»으로 잡는다.
  local mb ma lim prev=0 x
  mb="$(median "$1")"; ma="$(median "$2")"; lim=$((mb + 100))
  echo "  가중치(1분): 넘기기 전 [$1] 중앙 $mb · 뒤 [$2] 중앙 $ma"
  [ "$ma" -le "$lim" ] || { echo "  🔴중앙값이 $((ma - mb)) 늘었다(> 100)"; return 1; }
  for x in $2; do
    if [ "$x" -gt "$lim" ]; then [ "$prev" = 1 ] && { echo "  🔴두 번 연속 넘기기 전 중앙 + 100($lim) 초과"; return 1; }; prev=1; else prev=0; fi
  done
}
[ "$MODE" = weight-judge ] && { weight_judge "${4:-}" "${5:-}"; exit $?; }   # 시험용: weight-judge binance_ctx "" "<전>" "<뒤>"
[ "$MODE" = weight ] && { v="$(weight_max)" || exit 1; echo "$v"; echo "중앙 $(median "$v")"; exit 0; }   # 시험·손 확인용
[ "$MODE" = check ] && { check; exit $?; }
[ -x "$BIN" ] || die "실행 파일이 없다: $BIN"
CUR="$($CT -l 2>/dev/null)"; NCUR="$(printf '%s\n' "$CUR" | wc -l)"

if [ "$MODE" = back ]; then
  # 표지 = crontab 에 남은 «#takeover:<종류>…» 를 종류 접두사로(인자가 있으면 그 종목이 든 것) -- go 가 쓴 그대로
  TAGS="$(printf '%s\n' "$CUR" | grep -oE "#takeover:${KIND}(_[^ ]*)?( |\$)" | sed 's/^#takeover://; s/ $//' | sort -u)"
  if [ -n "$ARG" ]; then
    TAGS="$(for t in $TAGS; do ok=1; for x in $(echo "${ARG//[,:]/ }" | tr '[:lower:]' '[:upper:]'); do [[ "_${t#"$KIND"}_" == *"_${x}_"* ]] || ok=0; done; [ "$ok" = 1 ] && echo "$t"; done)"
  fi
  case "$(echo "$TAGS" | grep -c .)" in
    1) TAG="$TAGS" ;;
    0) die "crontab 에 «#takeover:$KIND…» 표지가 없다 -- 되돌릴 것이 없다(진짜 경로 Rust: $(rs_procs takeover any | tr '\n' ' '))" ;;
    *) die "표지가 여럿 -- 종목을 좁힌다: $(echo $TAGS)" ;;
  esac
  echo "back $TAG"
  for pid in $(rs_procs takeover any); do   # 이 표지의 Rust 만(종목 목록으로 표지를 다시 만들어 맞춘다)
    l="$(envof "$pid" "${RVAR:-$SELV}")"; t="$KIND"; [ -z "$SELV" ] || [ "$GROUP" = 1 ] || t="${KIND}_$(nrm "$l" | tr ',:' '__')"
    [ "$t" = "$TAG" ] && stop "$pid" "Rust($RK)" nosave
  done
  cron_back "$TAG"; NEW="$($CT -l)"
  B="$(ls -t logs/supervisor/crontab_backup_*_takeover_"$TAG".txt 2>/dev/null | head -1)"
  [ -z "$B" ] || { [ "$(cat "$B")" = "$NEW" ] && echo "  crontab = 백업 $B 와 같다" || echo "  ⚠️ crontab 이 백업 $B 와 다르다(그 뒤 다른 변경이 있었다 -- 이 종류 줄만 되돌렸다)"; }
  manifest back
  [ "$CRON" = 1 ] && { echo "back $TAG: 끝(예약 종류 -- 다음 시각부터 파이썬)"; exit 0; }
  printf '%s\n' "$CRON_RESTORED" | grep -E "^#takeover:$TAG @reboot" | sed "s|^#takeover:$TAG @reboot ||; s|^sleep [0-9]* && ||" | grep -F "$PYSUP" | while IFS= read -r cmd; do
    echo "  다시 띄움: $cmd"
    setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
  done
  sleep 5
  echo "back $TAG: 그림자는 다시 띄우지 않았다(shadow_start.sh)"
  SEL="${TAG#"$KIND"}"; SEL="${SEL#_}"; SEL="${SEL//_/,}"; [ -n "$SEL" ] || SEL="$(nrm "$SELD")"   # writers 가 이 목록으로 센다
  writers; exit $?
fi

# go
echo "go $TAG -- check 부터"
check || die "check 불통과 -- 아무것도 바꾸지 않았다"
if [ "$KIND" = binance_ctx ]; then   # 넘기면 Rust 가 OI REST(분당 약 540 · 가중치 1)를 처음 실전에서 맡는다 -- 같은 조건 분당 요청 수 + 넘기기 전 가중치
  echo "  binance_ctx: 가짜 서버로 분당 요청 수 나란히(${RATE_S:-300}초) -- rate_binance_ctx.py"
  nice -n 10 "$PYC" "$TOOLS/rate_binance_ctx.py" --seconds "${RATE_S:-300}" --latency "${RATE_LAT:-0.03}" --bin "$BIN" | sed 's/^/    /'
  [ "${PIPESTATUS[0]}" = 0 ] || die "분당 요청 수가 원본과 다르다 -- 아무것도 바꾸지 않았다"
  W_BEFORE="$(weight_max)" || die "넘기기 전 가중치를 못 읽음 -- 아무것도 바꾸지 않았다"
  echo "  넘기기 전 가중치(1분) [$W_BEFORE] 중앙 $(median "$W_BEFORE")"
fi
[ -z "$(rs_procs takeover)" ] || die "이미 진짜 경로 Rust 가 돈다(pid $(rs_procs takeover | tr '\n' ' ')) -- 멈춤"
PY="$(py_procs)"
if [ "$CRON" = 0 ]; then
  [ -n "$PY" ] || die "넘길 파이썬 수집기가 안 돈다 -- 멈춤"
  [ "$GROUP" = 1 ] || [ "$(echo "$PY" | wc -l)" = 1 ] || die "맞는 파이썬 수집기가 여럿($(echo $PY)) -- 종목을 좁힌다"
  if [ "$SHARED" = 1 ] && [ -n "$(comm -13 <(echo "$PY" | sort) <(py_procs any | sort))" ]; then
    die "같은 파일에 쓰는 다른 파이썬 수집기가 있다($(comm -13 <(echo "$PY" | sort) <(py_procs any | sort) | tr '\n' ' ')) -- 단일 writer 를 지키려 멈춤"
  fi
  # hl_ctx·hl_bt: data/hot/hl_ctx.sqlite 에는 원본도 코인마다 프로세스 넷이 같이 쓴다(WAL) -- Rust 도 WAL + busy_timeout 30초(store.rs) → 코인 하나씩 넘겨도 된다
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
  ADD=""; SCHED=""
  if [ "$KIND" = raster_archive ]; then ADD="40 * * * * cd $ROOT && $RENV OF_SYMBOLS=$OF_SYMS ${ENGU}_KIND=$RK $BIN >> logs/supervisor/rt_$RK.log 2>&1 #takeover:$TAG"; SCHED="40 * * * *"
  else   # 파이썬 예약 줄마다(PYMAIN 에 맞는 것만 -- cboe 요약 줄은 주석만) 같은 시각 · 스크립트 뒤 인자 그대로(국채선물 1d·5d 두 줄 · cboe 종목)
    for i in $LN; do
      l="$(printf '%s\n' "$CUR" | sed -n "${i}p")"
      [[ "$l" =~ ${PYMAIN:-$PYSUP} ]] || continue
      sc="$(echo "$l" | awk '{print $1, $2, $3, $4, $5}')"; ar="$(echo "$l" | sed -E "s/.*(${PYMAIN:-$PYSUP})[^ ]*//; s/>>.*//; s/2>&1//" | xargs)"
      ADD="${ADD:+$ADD$'\n'}$sc cd $ROOT && $RENV ${ENGU}_KIND=$RK $BIN $ar >> logs/supervisor/rt_$RK.log 2>&1 #takeover:$TAG"; SCHED="${SCHED:+$SCHED · }$sc"
    done
    [ -n "$ADD" ] || die "파이썬 예약 줄(${PYMAIN:-$PYSUP})을 못 찾음 -- 시각을 모른다 · 아무것도 안 바꿈"
  fi
  while IFS= read -r l; do   # 회차 시각 앞뒤 3분이면 거부(파이썬 회차와 Rust 회차가 같은 장부·파일에 두 번 쓰지 않게) -- */N 이면 N 분 주기의 앞뒤 1분
    read -r SM SH _ <<< "$l"
    if [[ "$SM" =~ ^[0-9]+$ ]] && { [ "$SH" = '*' ] || [ "$((10#$SH))" = "$((10#$(date +%H)))" ]; }; then
      d=$(( ( $((10#$(date +%M))) - SM + 60 ) % 60 )); { [ "$d" -le 3 ] || [ "$d" -ge 57 ]; } && die "회차 시각($l) 앞뒤 3분 -- 지금은 안 바꾼다(${d}분)"
    elif [[ "$SM" =~ ^\*/([0-9]+)$ ]]; then
      N="${BASH_REMATCH[1]}"; d=$(( $((10#$(date +%M))) % N )); { [ "$d" -le 1 ] || [ "$d" -ge $((N - 1)) ]; } && die "회차 시각($l) 앞뒤 1분 -- 지금은 안 바꾼다"
    fi
  done <<< "$(echo "$ADD" | awk '{print $1, $2}')"
else
  ADD="@reboot sleep 100 && cd $ROOT && ${ENGU}_BIN=$BIN $RENV bash scripts/ops/supervisor_$ENG.sh $RK >> logs/supervisor/${ENG}_reboot.log 2>&1 & #takeover:$TAG"
fi
NADD="$(printf '%s\n' "$ADD" | grep -c .)"
NEW="$(printf '%s\n' "$CUR" | awk -v L=",$(echo $LN | tr ' ' ','),"  -v T="#takeover:$TAG " 'index(L, "," NR ",") {print T $0; next} {print}')"$'\n'"$ADD"
NL="$(echo $LN | wc -w)"
[ "$(diff <(printf '%s\n' "$CUR") <(printf '%s\n' "$NEW" | head -n "$NCUR") | grep -c '^>')" = "$NL" ] || die "치환 검증 실패 -- 적용 안 함"
cron_apply "$NEW" $((NCUR + NADD))
STAGE=2
NPY="$(for i in $LN; do printf '%s\n' "$CUR" | sed -n "${i}p"; done | grep -cE "${PYSUP:-^$}")"
echo "  ① crontab 백업 $B ② 주석 $NL줄(파이썬 $NPY · 그림자 $((NL - NPY))) + Rust 줄 $NADD · $NCUR → $((NCUR + NADD))줄"
manifest take
if [ "$CRON" = 1 ]; then
  t0=$SECONDS; while [ -n "$(py_procs any)" ] && [ $((SECONDS - t0)) -lt 300 ]; do nap 1; done
  [ -z "$(py_procs any)" ] || die "지금 도는 파이썬 $PYRUN 이 5분 뒤에도 안 끝났다 -- 줄은 이미 바뀜(되돌리기: back)"
  echo "go $TAG: 예약 줄 바꿈 -- 다음 «$SCHED» 부터 Rust · 그 시각 뒤 check 한 번 더"; exit 0
fi
# ③ 파이썬 → ④ 그림자
for p in $PY; do stop "$p" "파이썬($PYRUN)" || die "파이썬을 못 껐다"; done
T_DOWN="$(date +%s)"
[ -z "$(py_procs)" ] || die "파이썬 $PYRUN 이 다시 떴다(pid $(py_procs | tr '\n' ' ')) -- 멈춤(되돌리기: back)"
for p in $(rs_procs shadow); do stop "$p" "그림자 Rust($RK)" || die "그림자를 못 껐다"; done
# ⑤ 진짜 경로
declare -A LOG0; for f in $(logs_glob); do LOG0[$f]="$(stat -c %s "$f")"; done
echo "  ⑤ 띄움: $RENV $ENG $RK"
env "${ENGU}_BIN=$BIN" $RENV nohup setsid bash "scripts/ops/supervisor_$ENG.sh" "$RK" >> "logs/supervisor/${ENG}_takeover.log" 2>&1 < /dev/null &
# ⑥ 지켜보기
T_UP=""
t0=$SECONDS
while [ $((SECONDS - t0)) -lt "$WATCH_S" ]; do
  nap 1
  [ -n "$T_UP" ] || { m="$(mt)"; [ -n "$m" ] && [ "$m" -gt "$T_DOWN" ] && T_UP="$m"; }
done
R="$(rs_procs takeover | tr '\n' ' ')"
BAD=0; for f in $(logs_glob); do BAD=$((BAD + $(tail -c +$(( ${LOG0[$f]:-0} + 1 )) "$f" | grep -ciE 'panic|crash-looping|SUPERVISOR .* exited'))); done
GAP="$([ -n "$T_UP" ] && echo "$((T_UP - T_DOWN))초" || echo "${WATCH_S}초 안 첫 쓰기 없음(쓰는 주기가 길면 정상 -- check 로)")"
echo "go $TAG: Rust pid ${R:-없음}· 끊김 $GAP · 로그 이상 $BAD · 다음 스트림 넘기기 전 check 한 번 더"
[ -n "$R" ] || die "Rust 가 ${WATCH_S}초 안에 안 살았다"
if [ "$KIND" = binance_ctx ]; then   # 넘긴 직후 가중치 -- 넘기기 전 최고보다 100 넘게 늘면 되돌린다(IP 밴은 주문 경로까지 막는다)
  W_AFTER="$(weight_max)" || die "넘긴 뒤 가중치를 못 읽음"
  weight_judge "$W_BEFORE" "$W_AFTER" || die "넘긴 뒤 가중치가 늘었다"
fi
STAGE=0
writers || exit 9
[ "$BAD" = 0 ] || { echo "🔴 Rust 는 돌지만 로그에 이상 $BAD 줄 -- 보고 판단(되돌리기: back $KIND)"; exit 1; }
