#!/usr/bin/env bash
# ETH L2 이상탐지 스냅샷 수집기의 크래시 재기동 래퍼 (2026-09-04).
#
# 왜 뒤늦게 생겼나: BTC/SOL/HYPE/XRP는 각자 supervisor + crontab @reboot가 있었는데 ETH만
# 없었다. ETH가 `l2_anomaly_snapshot_collector.py:476`의 **기본 심볼**(ethusdt)이라 환경변수
# 없이 맨몸으로 띄우는 방식이었고, 그래서 등록 대상에서 빠져 있었다. 결과적으로 2026-09-03
# 22:05 재부팅 때 나머지 4코인은 돌아왔지만 ETH만 20:24에 멈춘 채 18시간 방치됐다.
#
# DB 경로는 기존 파일을 그대로 쓴다(접미사 없는 l2_anomaly_snapshots.duckdb) -- 여기에 ETH
# 이력이 이미 들어있어서, 접미사를 붙이면 과거와 끊긴다.
#
# ⚠️ 중복 실행 방지: _supervise.sh의 flock은 supervisor끼리만 막는다. 같은 DB에 두 프로세스가
# 붙으면 duckdb가 단일 writer 제약으로 거부하거나 데이터가 어긋난다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

if pgrep -f "L2_ANOMALY_SYMBOL=ethusdt" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] ETH l2_anomaly가 이미 실행 중 -- 켜지 않는다(중복 방지)." >&2
  exit 1
fi
for pid in $(pgrep -f "[l]2_anomaly_snapshot_collector.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^L2_ANOMALY_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ethusdt}" = "ethusdt" ]; then
    echo "[$(date -Iseconds)] ETH l2_anomaly가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export L2_ANOMALY_SYMBOL="ethusdt"
export L2_ANOMALY_DB_PATH="$ROOT/data/live/l2_anomaly_snapshots.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "l2_anomaly_snapshot_collector.py(ETH)" \
  "$ROOT/data/live/.supervisor_l2_anomaly_eth.lock" \
  "$ROOT/logs/supervisor/l2_anomaly_eth" \
  "$PY" -u "$ROOT/l2_anomaly_snapshot_collector.py"
