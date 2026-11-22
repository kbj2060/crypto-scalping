#!/usr/bin/env bash
# rtorder(Rust 주문 경로)의 서버 **그림자(병행) 가동** 래퍼 (2026-11-21 · 사용자 승인 «주문 안 보냄으로 병행 가동»). 설명: rtorder/SHADOW.md.
#
# 🔴그림자 고정: 이 스크립트는 실전송 스위치(RTORDER_MODE=live · RTORDER_LIVE_CONFIRM)를 절대 넘기지 않는다 --
#   환경(점-env 포함)에 그 값이 있어도 지우고 띄운다. 실행 파일 안에서도 그림자 모드면 GET 과 listenKey 만들기·연장(POST·PUT) 밖의
#   요청은 보내기 전에 panic 한다(http.rs guard). 첫 로그 줄 = «모드: 그림자 · 쓰기 막힘».
# 키: 대시보드 감독 스크립트(dashboard/scripts/supervise_server.sh 19~24행)와 **같은 줄**로 저장소의 점-env 를 프로세스 환경에만 싣는다
#   (set -a · . · set +a -- 값을 찍거나 파일에 쓰지 않는다 · set -x 금지). rtorder 가 읽는 이름: BINANCE_API_KEY · BINANCE_SECRET_KEY
#   (파이썬 live_manual_peg_execute·계좌 모듈과 같은 이름) · DASHBOARD_MANUAL_EXEC_ENABLED · DASHBOARD_MANUAL_EXEC_SYMBOL · DASHBOARD_SIZING_CAP_EQUITY_X ·
#   TELEGRAM_BOT_TOKEN·TELEGRAM_CHAT_ID(그림자에서는 안 보냄).
# 실행 파일: tmp/rtbin/rtorder (개발기에서 `cd rtorder && cargo build --release --features live -j 2 --bin rtorder` → target/release/rtorder 를 복사)
# 상태 폴더: /mnt/data2/rtcollect_shadow/rtorder (6TB · 파이썬 data/live 가 아니다 -- 무장 파일을 서로 덮지 않게)
# 로그: logs/supervisor/rtorder_YYYYMMDD.log (안 보낸 쓰기는 «그림자(안 보냄): …» 줄)
#
# 환경변수(비우면 기본):
#   RTORDER_BIN      실행 파일(기본 $ROOT/tmp/rtbin/rtorder)
#   RTORDER_DIR      상태 폴더(기본 /mnt/data2/rtcollect_shadow/rtorder)
#   RTORDER_LISTEN   루프백 문(기본 127.0.0.1:8791 · 루프백이 아니면 rtorder 가 시작을 거절)
#   RTORDER_HUB      rtcore 허브(기본 ws://127.0.0.1:8788/rt · 감시 5분봉·1초 가격은 여기서만)
#   RTORDER_SHADOW_STREAM  사용자 데이터 스트림(기본 1 = 켬). 0 = listenKey 를 만들지 않고 계좌 REST 만.
#     같은 계좌 listenKey 는 하나라 파이썬과 같은 값을 받는다(POST = 있으면 그것을 돌려주고 60분 연장 · PUT = 연장). rtorder 는 DELETE 를 보내지 않고
#     그림자 막이도 DELETE 를 막는다 -- 파이썬 스트림을 끊을 길은 없다. 그래도 걱정되면 0.
#   RTORDER_SHADOW_ACCOUNT_TTL  그림자 감시 틱 계좌 캐시 초(기본 60 · 원본은 10) -- 같은 IP 의 파이썬 주문 경로 가중치를 지킨다(rtorder/SHADOW.md 4절).
#     그림자는 그 밖에도 체결 뒤 다시 읽기 30초 간격 · userTrades 건너뜀 · 거래소 사용 가중치가 1,800 을 넘은 분에는 REST 를 쉰다.
#     로그에 매 분 «rtorder 요청 계량 · … · 가중치 W · 거래소 사용 가중치 U · 쉰 요청 S» 한 줄.
#
# crontab(서버 · 등록은 사람이 한다):
#   @reboot sleep 90 && cd <저장소> && bash scripts/ops/supervisor_rtorder.sh >> logs/supervisor/rtorder_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# 2026-11-22: 표식이 있으면 파이썬이 주문을 이 문으로 넘긴다 -- 그림자가 그 문을 쥐면 주문이 «성공처럼» 보이고 안 나간다. 실전송 판(supervisor_rtorder_live.sh)만 떠야 한다.
if [ -f "$ROOT/data/live/rtorder_live.on" ]; then
  echo "[$(date -Iseconds)] 표식(data/live/rtorder_live.on)이 있다 -- 그림자를 켜지 않는다." >&2
  exit 1
fi

# 키 = 대시보드 감독 스크립트와 같은 줄(값은 이 프로세스 환경에만)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env" || echo "[warn] .env 를 읽지 못했습니다 -- 키 없이 계속합니다(계좌를 못 읽는 그림자)" >&2
  set +a
fi
# 🔴그림자 고정 -- 어디서 왔든 실전송 스위치는 지운다
unset RTORDER_MODE RTORDER_LIVE_CONFIRM
# 시험용 손잡이도 지운다(운영은 실거래소 읽기 · 감시 무한)
unset RTORDER_BASE RTORDER_WS RTORDER_MAX_TICKS RTORDER_TELEGRAM_BASE

BIN="${RTORDER_BIN:-$ROOT/tmp/rtbin/rtorder}"
export RTORDER_DIR="${RTORDER_DIR:-/mnt/data2/rtcollect_shadow/rtorder}"
export RTORDER_LISTEN="${RTORDER_LISTEN:-127.0.0.1:8791}"
export RTORDER_HUB="${RTORDER_HUB:-ws://127.0.0.1:8788/rt}"
export RTORDER_STREAM="${RTORDER_SHADOW_STREAM:-1}"
export RTORDER_BAN_FILE="${RTORDER_BAN_FILE:-$ROOT/data/live/binance_ban_until}"   # 파이썬과 같은 418·429 차단 기록(같은 IP)
export RUST_BACKTRACE=1

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] rtorder 실행 파일이 없다: $BIN -- 켜지 않는다." >&2
  exit 1
fi
case "$RTORDER_DIR" in
  "$ROOT/data/live"|"$ROOT/data/live/") echo "[$(date -Iseconds)] RTORDER_DIR 이 파이썬 상태 폴더다 -- 켜지 않는다." >&2; exit 1 ;;
esac
mkdir -p "$RTORDER_DIR" "$ROOT/logs/supervisor"
# flock 은 감독자끼리만 막는다 -- 손으로 띄운 rtorder 가 문(8791)을 쥐고 있으면 새 것은 묶기에서 종료코드 2 로 끝난다(같은 계좌 이중 감시 막기)
if pgrep -xf -- "$BIN" >/dev/null; then
  echo "[$(date -Iseconds)] rtorder 가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtorder (그림자 · $RTORDER_LISTEN)" \
  "$ROOT/data/live/.supervisor_rtorder.lock" \
  "$ROOT/logs/supervisor/rtorder" \
  "$BIN"
