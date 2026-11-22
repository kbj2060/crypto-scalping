#!/usr/bin/env bash
# rtorder(Rust 주문 경로) **실전송** 감독 (2026-11-22 · 사용자 승인 «지금 전환해줘»). 절차·되돌리기: rtorder/SHADOW.md §9 · 전환은 scripts/ops/rtorder_cutover.sh 로.
#
# 🔴표식 파일(data/live/rtorder_live.on)이 **있을 때만** 뜬다. 표식이 없으면 파이썬 대시보드가 손절 감시 중이다 -- 둘이 같이 감시하면 두 번 청산된다.
#   표식이 있으면 파이썬은 주문·계좌 14주소를 이 프로세스의 문(127.0.0.1:8791)으로 넘기고 자기 감시·사용자 스트림을 멈춘다(dashboard/orders.py rtorder_on).
# 키·실행 스위치: 그림자 래퍼(supervisor_rtorder.sh)와 같은 줄로 저장소의 점-env 를 프로세스 환경에만 싣는다(값을 찍거나 파일에 쓰지 않는다 · set -x 금지).
# 상태 폴더: data/live/rtorder (빠른 디스크 · 파이썬 data/live 자체가 아니다 -- 무장 파일은 전환 때 한 번 복사해 넣는다).
# 로그: logs/supervisor/rtorder_YYYYMMDD.log -- 첫 줄 «모드: 실전송».
# crontab(서버 · rtorder_cutover.sh go 가 넣고 back 이 뺀다 · 표식이 없으면 이 스크립트가 스스로 안 뜬다):
#   @reboot sleep 75 && cd <저장소> && bash scripts/ops/supervisor_rtorder_live.sh >> logs/supervisor/rtorder_reboot.log 2>&1 & #rtorder-live
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [ ! -f "$ROOT/data/live/rtorder_live.on" ]; then
  echo "[$(date -Iseconds)] 표식(data/live/rtorder_live.on)이 없다 = 파이썬이 감시 중 -- rtorder 실전송을 켜지 않는다." >&2
  exit 1
fi

# 키 = 대시보드 감독 스크립트와 같은 줄(값은 이 프로세스 환경에만)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env" || echo "[warn] .env 를 읽지 못했습니다 -- 키 없이 계속합니다(계좌를 못 읽는 그림자)" >&2
  set +a
fi
# 🔴실전송 -- 사용자 승인(2026-11-22) · 표식이 있을 때만(위 검사)
export RTORDER_MODE=live
export RTORDER_LIVE_CONFIRM="실주문을-보낸다"
# 시험용 손잡이도 지운다(운영은 실거래소 읽기 · 감시 무한)
unset RTORDER_BASE RTORDER_WS RTORDER_MAX_TICKS RTORDER_TELEGRAM_BASE

BIN="${RTORDER_BIN:-$ROOT/tmp/rtbin/rtorder}"
export RTORDER_DIR="${RTORDER_DIR:-$ROOT/data/live/rtorder}"
export RTORDER_LISTEN="${RTORDER_LISTEN:-127.0.0.1:8791}"
export RTORDER_HUB="${RTORDER_HUB:-ws://127.0.0.1:8788/rt}"
export RTORDER_STREAM=1
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
  "rtorder (실전송 · $RTORDER_LISTEN)" \
  "$ROOT/data/live/.supervisor_rtorder.lock" \
  "$ROOT/logs/supervisor/rtorder" \
  "$BIN"
