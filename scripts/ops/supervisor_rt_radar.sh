#!/usr/bin/env bash
# rt_radar(Rust 위험 레이더 한 회차) cron 래퍼 (2026-11-21). 상주 프로세스가 아니다 -- 원본 scripts/live_risk_radar_20261115.py 처럼 cron 이 매분 한 번 부른다.
# 그래서 재기동 루프(_supervise.sh) 대신: 실행 파일·계수 확인 → 겹침 막기(flock -n · 첫 회차 31일 백필 1~2분이 다음 분과 겹치지 않게) → 한 번 돌고 종료코드 그대로.
# 지금은 **그림자** 전용 -- rtmodel/SHADOW.md «rt_radar». 쓰는 곳 기본 data/live/rt_risk_radar(RT_RADAR_OUT · data/live 는 실행 파일이 종료코드 2 로 거부).
# 종료코드: 0 정상 · 1 한 코인이라도 오류(cron 이 다음 분에 다시) · 2 설정 거부 · 3 이전 회차가 아직 돈다(건너뜀).
# 로그: logs/supervisor/rt_radar_YYYYMMDD.log
#
# crontab -- 적기만 했다. 등록은 사람이 한다:
#   * * * * * cd <저장소> && bash scripts/ops/supervisor_rt_radar.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${RT_RADAR_BIN:-$ROOT/bin/rt_radar}"
export RT_RADAR_OUT="${RT_RADAR_OUT:-data/live/rt_risk_radar}"
PARAMS="${RT_RADAR_PARAMS:-dashboard/risk_radar_params_20261116.json}"
mkdir -p logs/supervisor data/live
LOG="logs/supervisor/rt_radar_$(date +%Y%m%d).log"

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] 실행 파일이 없다: $BIN (rtmodel/SHADOW.md «rt_radar»)" >> "$LOG"
  exit 2
fi
if [ ! -f "$PARAMS" ]; then
  echo "[$(date -Iseconds)] 계수 파일이 없다: $PARAMS" >> "$LOG"
  exit 2
fi
exec 9>"data/live/.rt_radar_$(echo "$RT_RADAR_OUT" | md5sum | cut -c1-8).lock"   # 출력 디렉터리마다 한 회차
if ! flock -n 9; then
  echo "[$(date -Iseconds)] 이전 회차가 아직 돈다($RT_RADAR_OUT) -- 이번 분은 건너뜀" >> "$LOG"
  exit 3
fi
nice -n 10 "$BIN" >> "$LOG" 2>&1
