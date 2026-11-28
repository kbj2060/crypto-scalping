#!/usr/bin/env bash
# rtserve **전환 모드** 래퍼 (2026-11-28) -- thesan.xyz 를 파이썬 대시보드(8787)에서 rtserve 로 넘길 때 쓴다. 병행용은 supervisor_rtserve.sh.
#
# 🔴파이썬 대시보드가 같은 파일을 쓰는 동안 띄우지 말 것.
#   이 모드는 메모(acct_notes)·오답(acct_wrong)·알림 구독(push 구독 파일)·접수 기록(order_requests.jsonl)을 원본과 **같은 data/live 파일**에 쓰고,
#   요청 때 바이낸스 REST 를 원본처럼 부른다. 둘이 같이 돌면 rename 덮어쓰기로 서로의 쓰기를 지우고 같은 IP 로 REST 를 두 번 부른다.
#   그래서 127.0.0.1:8787 에 누가 듣고 있으면 켜지 않는다(아래 검사).
#
# 전환 순서(사람이 한다 · 서버):
#   1. 병행 rtserve 끄기: supervisor_rtserve.sh 감독과 rtserve 프로세스(같은 8790 · 같은 잠금 파일)를 멈춘다.
#   2. 파이썬 대시보드 끄기: dashboard/scripts/supervise_server.sh 감독째 멈춘다(crontab @reboot 줄도 주석) -- `ss -ltn | grep 8787` 이 비어야 한다.
#   3. 이 스크립트로 rtserve 를 8790 에 띄운다 → `curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:8790/dashboard/live/` = 200 확인.
#   4. 사용자가 Cloudflare 원격 규칙 중 «경로 빈 thesan.xyz» 줄의 서비스만 http://127.0.0.1:8790 으로 고친다(2026-11-28 실제로 한 방법).
#      원격 규칙이 있으면 터널은 --url 을 무시한다. 🔴규칙을 지우거나 더하지 말 것(11-22: 지우면 DNS 까지 삭제) · `^/rt$` → 8788 줄은 위에 그대로.
#   표식: data/live/rtserve_live.on 을 만들면 deploy_watcher·rtorder_cutover·알림 워커가 8787 대신 8790 을 본다(지우면 예전 그대로).
#   5. 되돌리기: 이 감독 멈춤 → 표식 지움 → 파이썬 감독 다시 → 사용자가 같은 규칙 줄을 8787 로.
#
# 병행 모드와 다른 점:
#   RTSERVE_READONLY=0               쓰기 주소 4 + 접수 기록이 열린다(같은 파일 · 같은 잠금 없음 = 혼자 쓸 때만 안전)
#   RTSERVE_REST=https://fapi.binance.com   원본 FAPI(바꾸려면 RTSERVE_REST_LIVE). 바이낸스 요청은 전부 model::binance_get 한 문
#                                    (밴 파일 · 가중치 1,200 넘으면 그 분 쉼 · 간격 200ms)을 지난다.
# 나머지(점-env 를 프로세스 환경에만 · 실행 파일 · 로그 · 8790 · 허브 · rtorder 문)는 supervisor_rtserve.sh 와 같다.
# 로그: logs/supervisor/rtserve_live_YYYYMMDD.log · 잠금: data/live/.supervisor_rtserve.lock(병행 감독과 같은 잠금 -- 둘이 동시에 못 뜬다)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env" || echo "[warn] 점-env 를 읽지 못했습니다 -- 알림 공개키 없이 계속합니다" >&2
  set +a
fi
# 🔴전환 고정
export RTSERVE_READONLY=0
export RTSERVE_REST="${RTSERVE_REST_LIVE:-https://fapi.binance.com}"

BIN="${RTSERVE_BIN:-$ROOT/tmp/rtbin/rtserve}"
export RTSERVE_ROOT="$ROOT"
export RTSERVE_ADDR="${RTSERVE_ADDR:-127.0.0.1:8790}"
export RTSERVE_HUB="${RTSERVE_HUB:-ws://127.0.0.1:8788/rt}"
export RUST_BACKTRACE=1
export RTSERVE_PY="${RTSERVE_PY:-${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}}"   # /api/push/test 가 원본 웹 푸시 모듈을 부를 때

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] rtserve 실행 파일이 없다: $BIN -- 켜지 않는다." >&2
  exit 1
fi
case "$RTSERVE_ADDR" in
  *:8787) echo "[$(date -Iseconds)] RTSERVE_ADDR 이 파이썬 대시보드 문(8787)이다 -- 켜지 않는다." >&2; exit 1 ;;
esac
if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(^|:)8787$'; then
  echo "[$(date -Iseconds)] 8787 에 파이썬 대시보드가 듣고 있다 -- 같은 파일을 둘이 쓰게 되므로 켜지 않는다(전환 순서 2 먼저)." >&2
  exit 1
fi
mkdir -p "$ROOT/logs/supervisor"
if pgrep -xf -- "$BIN" >/dev/null; then
  echo "[$(date -Iseconds)] rtserve 가 이미 실행 중(병행 모드일 수 있다) -- 켜지 않는다." >&2
  exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "rtserve (전환 · $RTSERVE_ADDR · 쓰기·REST 켬)" \
  "$ROOT/data/live/.supervisor_rtserve.lock" \
  "$ROOT/logs/supervisor/rtserve_live" \
  "$BIN"
