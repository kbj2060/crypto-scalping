#!/usr/bin/env bash
# rtcore(Rust 실시간 코어)의 크래시 재기동 래퍼 (2026-11-20). 설계: docs/rtcore_design_20261120.md §11 R1.
#
# 하는 일: 거래소 웹소켓(시세·체결·청산, 읽기 전용 -- 주문·계정·키 없음) → 메모리 → 127.0.0.1:8788 의 /rt 웹소켓으로 화면에 민다.
# 수신 태스크가 죽으면(2026-11-20 rustls 패닉처럼) 프로세스가 종료코드 70 으로 끝난다(rtcore/src/main.rs) -- 여기서 다시 띄운다.
# 다시 띄우기는 _supervise.sh 규칙 그대로: 끝나면 15초 뒤 · 10분 안 5번째부터는 300초 쉬고 띄운다(같은 자리 패닉이 거래소를 두드리지 않게).
# 로그: logs/supervisor/rtcore_YYYYMMDD.log -- 60초마다 «rtcore 60s 수신->송신 … · 살아있는수신=N · 깨진JSON=… 건너뜀=… 재연결=N (누적)».
# ⚠️개발기에서 실행하지 않는다(기본 주소가 실제 거래소 -- 서버와 IP 한도 공유). 서버 전용.
#
# 환경변수(비우면 rtcore 기본값 -- rtcore/src/lib.rs Cfg · hub.rs HubCfg):
#   RTCORE_BIN            실행 파일(기본 rtcore/target/release/rtcore)
#   RTCORE_SYMBOLS        심볼 쉼표 목록(기본 아래 -- 대시보드가 1초 수급·풋프린트를 내는 코인과 같게)
#   RTCORE_LISTEN         듣는 주소(rtcore 기본 127.0.0.1:8788 -- 0.0.0.0 금지: 2026-09-12 LAN 경로 사고와 같은 이유)
#   RTCORE_PATH           웹소켓 경로(기본 /rt -- Cloudflare 규칙 ^/rt$ 와 같게)
#   RTCORE_ALLOW_ORIGINS  브라우저 Origin 허용 목록(쉼표, https://<호스트>) -- 비면 Origin 없는 도구만 붙는다
#   RTCORE_MAX_CONNS · RTCORE_MAX_SUBS · RTCORE_PING_SECS · RTCORE_SILENCE_SECS
#   RTCORE_WS_BASE · RTCORE_MARKET_WS_BASE · RTCORE_FORCE_WS_URL · RTCORE_SPOT_WS_BASE · RTCORE_OKX_WS_URL · RTCORE_BYBIT_WS_URL (빈 값 = 그 원천 끔)
#
# crontab(서버 · 등록은 사람이 한다):
#   @reboot sleep 60 && cd <저장소> && bash scripts/ops/supervisor_rtcore.sh >> logs/supervisor/rtcore_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="${RTCORE_BIN:-$ROOT/rtcore/target/release/rtcore}"
if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] rtcore 실행 파일이 없다: $BIN -- 켜지 않는다." >&2
  exit 1
fi
# flock 은 supervisor 끼리만 막는다. 손으로 띄운 rtcore 가 포트를 쥐고 있으면 새 것은 bind 에서 죽어 재기동만 반복한다.
if pgrep -xf -- "$BIN" >/dev/null; then
  echo "[$(date -Iseconds)] rtcore 가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

# 기본 심볼 = 대시보드 흐름 엔진이 도는 코인: server.py:1722~1723 `FLOW_SPECS 중 DASHBOARD_ASSETS 에 든 것`
#   (DASHBOARD_ASSETS 기본 server.py:425 eth,btc,sol,xrp,sndk,soxl,skhynix,mu · FLOW_SPECS coin_flow.py:269~279 같은 8개).
#   서버가 DASHBOARD_ASSETS 를 바꿔 띄우면 여기도 같이 바꾼다.
export RTCORE_SYMBOLS="${RTCORE_SYMBOLS:-ethusdt,btcusdt,solusdt,xrpusdt,sndkusdt,soxlusdt,skhynixusdt,muusdt}"
export RUST_BACKTRACE=1   # 패닉 자리가 로그에 남게(70 으로 끝나는 사유 줄과 같이 본다)

exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtcore ($RTCORE_SYMBOLS)" \
  "$ROOT/data/live/.supervisor_rtcore.lock" \
  "$ROOT/logs/supervisor/rtcore" \
  "$BIN"
