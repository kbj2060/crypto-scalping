#!/usr/bin/env bash
# rtserve(Rust 응답 서버 · 꾸러미 R6)의 서버 **병행 가동** 래퍼 (2026-11-22). 설명: docs/rtserve_routes_20261121.md §8.
# 파이썬 대시보드(127.0.0.1:8787)는 그대로 두고 rtserve 를 127.0.0.1:8790 에 띄워 같은 주소를 나란히 답하게 한다(대조 = rtserve/tools/shadow_http.py).
#
# 🔴병행 고정(어디서 왔든 덮어쓴다):
#   RTSERVE_READONLY=1  메모·오답·푸시 구독 POST = 503 {"ok": false, "error": "readonly_shadow"} · 접수 기록(order_requests.jsonl) 안 적음
#                       -- 파이썬과 같은 data/live 파일을 둘이 쓰지 않게
#   RTSERVE_REST 지움   모델 주소(dir4h·dir-short·vol-forecast·open-rev·dir-signals)가 REST 를 안 보낸다(reason «RestOff: …») -- 파이썬과 같은 IP 로 겹쳐 부르지 않게.
#                       켜려면 이 스크립트 말고 손으로(RTSERVE_REST=<원본 FAPI 주소>) -- 병행 중엔 켜지 말 것
#   주문·계좌 14 주소는 rtorder 문(RTSERVE_RTORDER 기본 127.0.0.1:8791)으로 넘긴다 -- 그림자 rtorder 가 떠 있으면 그쪽이 답하고(안 보냄 모드), 없으면 502.
# 키: VAPID_PUBLIC_KEY 가 /api/push/config 에 나가므로 대시보드 감독 스크립트(dashboard/scripts/supervise_server.sh)와 **같은 줄**로 저장소 점-env 를
#   프로세스 환경에만 싣는다(set -a · . · set +a -- 값을 찍거나 파일에 쓰지 않는다 · set -x 금지). 비밀키는 «있는지»만 쓴다.
# 실행 파일: tmp/rtbin/rtserve (개발기에서 `cd rtserve && cargo build --release -j 2 --bin rtserve` → target/release/rtserve 를 복사)
# 로그: logs/supervisor/rtserve_YYYYMMDD.log
#
# 환경변수(비우면 기본):
#   RTSERVE_BIN     실행 파일(기본 $ROOT/tmp/rtbin/rtserve)
#   RTSERVE_ADDR    묶을 주소(기본 127.0.0.1:8790 · 루프백이 아니면 rtserve 가 시작을 거절 · 8787 은 파이썬 몫)
#   RTSERVE_HUB     rtcore 허브(기본 ws://127.0.0.1:8788/rt · 메모리 상태 주소 = 수급·풋프린트·oi-5m OKX·청산 원·events 가격)
#   RTSERVE_MODELS  모델 JSON 자리(기본 data/live/rtmodel · REST 꺼짐이라 병행 중엔 안 읽힌다)
#
# crontab(서버 · 등록은 사람이 한다):
#   @reboot sleep 100 && cd <저장소> && bash scripts/ops/supervisor_rtserve.sh >> logs/supervisor/rtserve_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env" || echo "[warn] 점-env 를 읽지 못했습니다 -- 알림 공개키 없이 계속합니다" >&2
  set +a
fi
# 🔴병행 고정
export RTSERVE_READONLY=1
unset RTSERVE_REST

BIN="${RTSERVE_BIN:-$ROOT/tmp/rtbin/rtserve}"
export RTSERVE_ROOT="$ROOT"
export RTSERVE_ADDR="${RTSERVE_ADDR:-127.0.0.1:8790}"
export RTSERVE_HUB="${RTSERVE_HUB:-ws://127.0.0.1:8788/rt}"
export RUST_BACKTRACE=1

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] rtserve 실행 파일이 없다: $BIN -- 켜지 않는다." >&2
  exit 1
fi
case "$RTSERVE_ADDR" in
  *:8787) echo "[$(date -Iseconds)] RTSERVE_ADDR 이 파이썬 대시보드 문(8787)이다 -- 켜지 않는다." >&2; exit 1 ;;
esac
mkdir -p "$ROOT/logs/supervisor"
if pgrep -xf -- "$BIN" >/dev/null; then
  echo "[$(date -Iseconds)] rtserve 가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtserve (병행 · $RTSERVE_ADDR · 읽기 전용)" \
  "$ROOT/data/live/.supervisor_rtserve.lock" \
  "$ROOT/logs/supervisor/rtserve" \
  "$BIN"
