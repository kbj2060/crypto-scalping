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
#   RTCORE_QUOTE_SYMBOLS  호가·1분봉만 받는 주문 심볼(기본 ethusdc,solusdc,xrpusdc,btcusdc -- rtorder 주문 계획이 쓰는 USDC 선물 · 2026-11-21).
#                         RTCORE_SYMBOLS 마다 @trade·@bookTicker·@markPrice@1s·@kline_1m 4개 + 이것마다 @bookTicker·@kline_1m 2개 → 기본 8코인 47개 연결(나머지 원천 포함)
#   RTCORE_LISTEN         듣는 주소(rtcore 기본 127.0.0.1:8788 -- 0.0.0.0 금지: 2026-09-12 LAN 경로 사고와 같은 이유)
#   RTCORE_PATH           웹소켓 경로(기본 /rt -- Cloudflare 규칙 ^/rt$ 와 같게)
#   RTCORE_ALLOW_ORIGINS  브라우저 Origin 허용 목록(쉼표, https://<호스트>) -- 비면 Origin 없는 도구만 붙는다
#   RTCORE_MAX_CONNS · RTCORE_MAX_SUBS · RTCORE_PING_SECS · RTCORE_SILENCE_SECS
#   RTCORE_WS_BASE · RTCORE_MARKET_WS_BASE · RTCORE_FORCE_WS_URL · RTCORE_SPOT_WS_BASE · RTCORE_OKX_WS_URL · RTCORE_BYBIT_WS_URL (빈 값 = 그 원천 끔)
#   재시작 복원(2026-11-21 · rtcore/RESTORE.md · rtcore/src/restore.rs) -- 상대 경로는 아래 `cd "$ROOT"` 기준(저장소 루트):
#   RTCORE_SNAP_DIR       코어 스냅숏 폴더(기본 data/live/rtcore · 코어가 만든다 · 30초마다·SIGTERM 때 <심볼>.json 덮어쓰기) -- 빈 값 = 복원 전부 끔
#   RTCORE_TAPE_DB        바이낸스 체결 테이프(기본 data/hot/binance_tape.sqlite · 읽기 전용) -- 풋프린트 봉·1초 칸·dhl 공백 메우기
#   RTCORE_OKX_TAPE_DB    OKX 체결 테이프(기본 data/hot/okx_tape.sqlite) -- OKX 1초 칸
#   RTCORE_OKX_CTX_DB     OKX 맥락(기본 data/hot/okx_ctx.sqlite) -- OKX 미결제약정·청산
#   RTCORE_BN_CTX_DB      바이낸스 맥락(기본 data/hot/binance_ctx.sqlite) -- 강제청산
#   RTCORE_BYBIT_LIQ_DB   Bybit 청산(기본 data/hot/bybit_liq.sqlite)
#     Rust 수집기 그림자 파일로 바꿀 때는 data/hot/rtcollect/<같은 이름>(rtcollect/SHADOW.md). 파일이 없으면 로그 한 줄 «… 못 읽음 … 빈다» 를 남기고 그 원천만 빈 채로 돈다.
#     테이프는 WAL 이라 읽기 전용 연결도 -shm 에 쓴다 -- 수집기와 같은 사용자로 띄운다.
#   rtcore 에 SIGTERM(kill <rtcore pid>): 스냅숏을 한 번 쓰고 종료코드 0 으로 끝난다(메우는 중이면 안 쓴다) → _supervise.sh 가 15초 뒤 다시 띄운다
#     (= 새 실행 파일로 바꾸는 재시작). 아주 내리려면 감시자(이 스크립트) pid 를 먼저 끝내고 rtcore 에 SIGTERM.
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
