#!/usr/bin/env bash
# rtcollect(Rust 수집기) 크래시 재기동 래퍼 (2026-11-21). 종류 하나 = 프로세스 하나. 지금은 **그림자** 전용 -- rtcollect/SHADOW.md.
#
# 실행 파일: rtcollect/target/release/rtcollect 를 개발기에서 빌드해 옮긴 것(RTCOLLECT_BIN, 기본 $ROOT/bin/rtcollect).
# 종류·심볼은 환경변수(수집기 실행 파일과 같은 이름) 또는 인자:
#   bash scripts/ops/supervisor_rtcollect.sh <binance_tape|bybit_liq|bybit_tape|okx_tape|okx_ctx|binance_ctx|xyz> [심볼 쉼표 목록]
#   RTCOLLECT_KIND=okx_tape OKX_TAPE_INST=ETH-USDT-SWAP,BTC-USDT-SWAP bash scripts/ops/supervisor_rtcollect.sh
#   (binance_tape: TAPE_SYMBOL·TAPE_MARKET · bybit_liq: BYBIT_LIQ_SYMBOLS · bybit_tape: BYBIT_TAPE_SYMBOLS · okx_tape: OKX_TAPE_INST ·
#    okx_ctx: OKX_CTX_INST · binance_ctx: BN_CTX_MARK_SYMBOLS(OI·청산 목록은 BN_CTX_OI_SYMBOLS·BN_CTX_OI_SLOW_SYMBOLS) · xyz: HL_XYZ_COINS)
# 쓰는 곳: 기본 data/hot/rtcollect/<파이썬과 같은 파일 이름> -- 파이썬 수집기의 data/hot/<이름> 은 실행 파일이 거부한다(전환 때만 RTCOLLECT_TAKEOVER=1).
# 로그: logs/supervisor/rtcollect_<KEY>_YYYYMMDD.log · 락: data/live/.supervisor_rtcollect_<KEY>.lock
# 종료코드와 무관하게 다시 띄운다(_supervise.sh: 15초 뒤 · 10분에 5번이면 300초 쉼) -- 정지 감시 70(바이낸스·OKX) · 3(Bybit 테이프) · 2(설정 오류) 모두.
# ⚠️중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 종류·겹치는 심볼의 rtcollect 가 이미 돌면 켜지 않는다
#   (같은 파일에 두 벌 기록). 파이썬 수집기와는 겹쳐도 된다(그림자 = 다른 파일).
#
# crontab(서버 재부팅 때) -- 적기만 했다. 등록은 사람이 한다(SHADOW.md 의 순서대로):
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=binance_tape TAPE_SYMBOL=<파이썬과 같은 목록> bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=binance_tape TAPE_MARKET=spot TAPE_SYMBOL=<현물 목록> bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=okx_tape OKX_TAPE_INST=<파이썬과 같은 목록> bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=bybit_tape bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=bybit_liq bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=okx_ctx OKX_CTX_INST=<파이썬과 같은 목록> bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=binance_ctx bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
#   @reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=xyz bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${RTCOLLECT_BIN:-$ROOT/bin/rtcollect}"
KIND="${1:-${RTCOLLECT_KIND:-binance_tape}}"

case "$KIND" in
  binance_tape) VAR=TAPE_SYMBOL;        DEF=ethusdt ;;
  bybit_liq)    VAR=BYBIT_LIQ_SYMBOLS;  DEF=ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT ;;
  bybit_tape)   VAR=BYBIT_TAPE_SYMBOLS; DEF=ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,HYPEUSDT ;;
  okx_tape)     VAR=OKX_TAPE_INST;      DEF=ETH-USDT-SWAP ;;
  okx_ctx)      VAR=OKX_CTX_INST;       DEF=ETH-USDT-SWAP ;;
  binance_ctx)  VAR=BN_CTX_MARK_SYMBOLS; DEF=ethusdt,btcusdt,solusdt,xrpusdt,zecusdt,sndkusdt,soxlusdt,skhynixusdt,muusdt ;;
  xyz)          VAR=HL_XYZ_COINS;       DEF=xyz:XYZ100,xyz:SP500,xyz:JPY,xyz:EUR ;;
  *) echo "[$(date -Iseconds)] 모르는 종류 '$KIND' -- binance_tape|bybit_liq|bybit_tape|okx_tape|okx_ctx|binance_ctx|xyz" >&2; exit 2 ;;
esac
if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] 실행 파일이 없다: $BIN (rtcollect/SHADOW.md «실행 파일 두기»)" >&2
  exit 1
fi

# 심볼은 대문자로 맞춰 비교한다(실행 파일이 바이낸스는 소문자, 나머지는 대문자로 읽는다)
norm() { echo "$1" | tr '[:lower:]' '[:upper:]' | tr -d ' '; }
RAW="${2:-${!VAR:-$DEF}}"
SYM="$(norm "$RAW")"
MKT="${TAPE_MARKET:-futures}"
overlap() { local a=",$1,"; for x in ${2//,/ }; do case "$a" in *",$x,"*) return 0;; esac; done; return 1; }
envof() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep "^$2=" | head -1 | cut -d= -f2-; }
for pid in $(pgrep -x rtcollect); do
  kind=$(envof "$pid" RTCOLLECT_KIND)
  [ "${kind:-binance_tape}" = "$KIND" ] || continue
  mkt=$(envof "$pid" TAPE_MARKET)
  [ "$KIND" != binance_tape ] || [ "${mkt:-futures}" = "$MKT" ] || continue
  cur=$(envof "$pid" "$VAR")
  if overlap "$(norm "${cur:-$DEF}")" "$SYM"; then
    echo "[$(date -Iseconds)] rtcollect($KIND ${cur:-$DEF})가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

# 넘기는 글자: 바이낸스는 소문자(파이썬 감독·매니페스트와 같은 글자), 나머지는 대문자 -- 매니페스트 `match` 는 environ 을 글자 그대로 비교한다
# xyz 는 글자 그대로(HL 종목 이름은 대소문자를 가린다 · 원본도 split(",") 만 한다)
OUT="$SYM"; [ "$KIND" != binance_tape ] || OUT="$(echo "$SYM" | tr '[:upper:]' '[:lower:]')"; [ "$KIND" != xyz ] || OUT="$RAW"
KEY="${KIND}_${OUT//,/_}"; KEY="${KEY//:/_}"
[ "$KIND" != binance_tape ] || [ "$MKT" = futures ] || KEY="${KEY}_$MKT"
export RTCOLLECT_KIND="$KIND" "$VAR=$OUT"
[ "$KIND" != binance_tape ] || export TAPE_MARKET="$MKT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtcollect($KEY)" \
  "$ROOT/data/live/.supervisor_rtcollect_$KEY.lock" \
  "$ROOT/logs/supervisor/rtcollect_$KEY" \
  "$BIN"
