#!/usr/bin/env bash
# Zeus 섀도우 기록기의 crash-restart 감시자. `_supervise.sh` 패턴(crontab @reboot).
#
# 사용법: supervisor_zeus_shadow.sh <아티팩트 디렉터리명>
#   예) supervisor_zeus_shadow.sh zeus_v4_shadow_20260918
#
# 무엇을 돌리나: scripts/live_zeus_shadow_runner_20260918.py -- 🔴**주문을 내지 않는다.**
# 사양은 docs/zeus/README.md §2(v3)·§3(v4), 판정은 docs/zeus/shadow_prereg_v4_20260918.md.
# v3·v4 가 **같은 러너**를 쓴다 -- 사양 차이는 전부 아티팩트(model.pt/meta.json)에 있다.
# 🔴러너를 판본별로 나누면 한쪽만 고쳐진다(2026-09-18: 서버 v3 전용 러너에 원천 병합
#   버그가 있어 최근 4일 OI 결측 43.4% 였다).
#
# trading_bot.py 와 완전 분리: 주문 경로 없음, 공유 상태 없음, 피쳐를 독립 재계산한다.
# 🔴배포·코드 변경 뒤에는 사람이 먼저 파리티를 확인한다(부팅마다 돌리면 연구 parquet
#   480MB 를 실거래 기기에서 매번 읽는다):
#     python scripts/live_zeus_shadow_runner_20260918.py --parity --art=<디렉터리>
#     python scripts/live_zeus_shadow_runner_20260918.py --verify-rest
set -uo pipefail
ART="${1:?아티팩트 디렉터리명을 인자로 넘길 것 (예: zeus_v4_shadow_20260918)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ZEUS_ROOT="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"   # 라이브 봇과 CPU 경쟁하지 않는다
export MALLOC_ARENA_MAX=2                        # 2026-09-18 OOM 전례(glibc arena 미반환)

# 🔴같은 아티팩트를 두 번 띄우면 원장이 섞인다. pkill 로 죽이지 말 것(패턴이 이 셸도 잡는다).
if pgrep -af "[l]ive_zeus_shadow_runner_20260918.py .*--art=$ART" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] Zeus 섀도우($ART)가 이미 실행 중 -- 켜지 않는다." >&2; exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_zeus_shadow_runner($ART, no orders)" \
  "$ROOT/data/live/.supervisor_${ART}.lock" \
  "$ROOT/logs/supervisor/${ART}" \
  "$PY" -u "$ROOT/scripts/live_zeus_shadow_runner_20260918.py" \
        --live --art="$ART" --bars=20000 --sleep=120
