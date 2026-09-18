#!/usr/bin/env bash
# Zeus Baseline v3 섀도우 기록기의 crash-restart 감시자.
# supervisor_maker_fill_shadow.sh 와 같은 _supervise.sh 패턴(crontab @reboot, systemd 아님).
#
# 무엇을 돌리나: scripts/live_zeus_v3_shadow_20260918.py -- 🔴**주문을 내지 않는다.**
# 동결 사양(docs/zeus/README.md §2)으로 5분봉마다 결정을 기록만 한다.
# 판정 기준은 docs/zeus/shadow_prereg_v3_20260918.md (수익은 체결 436건까지 «판정 안 함»).
#
# trading_bot.py 와 완전 분리: 주문 경로 없음, 공유 파일 없음, 피쳐를 독립 재계산한다.
# 🔴OI/LSR 은 저장 패널(live_evr_gate_worker 가 갱신)에 의존한다 -- 그게 멈추면 여기도 멈춘다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ZEUS_ROOT="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"   # 라이브 봇과 CPU 경쟁하지 않는다
export MALLOC_ARENA_MAX=2                        # 2026-09-18 OOM 전례

if pgrep -af "[l]ive_zeus_v3_shadow_20260918.py" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] Zeus v3 섀도우가 이미 실행 중 -- 켜지 않는다." >&2; exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_zeus_v3_shadow_20260918.py(Zeus v3 shadow, no orders)" \
  "$ROOT/data/live/.supervisor_zeus_v3_shadow.lock" \
  "$ROOT/logs/supervisor/zeus_v3_shadow" \
  "$PY" -u "$ROOT/scripts/live_zeus_v3_shadow_20260918.py" --loop
