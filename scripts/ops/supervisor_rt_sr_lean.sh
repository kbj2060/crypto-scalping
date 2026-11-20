#!/usr/bin/env bash
# rt_sr_lean(Rust «레벨 5분» 워커) 크래시 재기동 래퍼 (2026-11-21). 지금은 **그림자** 전용 -- rtmodel/SHADOW.md.
#
# 실행 파일: rtmodel/target/release/rt_sr_lean 을 개발기에서 빌드해 옮긴 것(RT_SR_BIN, 기본 $ROOT/bin/rt_sr_lean).
# 쓰는 곳: 기본 data/live/rt_sr_lean/(RT_SR_OUT) -- 파이썬 워커의 data/live/sr_lean 은 실행 파일이 거부한다(종료코드 2 · 전환 때만 RT_SR_TAKEOVER=1).
# 모델: <RT_SR_OUT>/sr_lean.json(RT_SR_MODEL) -- `python rtmodel/tools/export_hgb.py data/live/rt_sr_lean` 이 만든다. 없으면 켜지 않는다.
# 로그: logs/supervisor/rt_sr_lean_YYYYMMDD.log · 락: data/live/.supervisor_rt_sr_lean.lock
# 종료코드와 무관하게 다시 띄운다(_supervise.sh: 15초 뒤 · 10분에 5번이면 300초 쉼) -- 1(원본도 죽는 예외) · 2(설정 오류) 모두.
# (재)시작 뒤 약 1시간은 준비(파이썬과 같다).
# ⚠️중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 출력 디렉터리의 rt_sr_lean 이 이미 돌면 켜지 않는다(같은 파일에 두 벌 기록).
#   파이썬 워커(supervisor_sr_lean.sh)와는 겹쳐도 된다(그림자 = 다른 디렉터리). 이름이 다르므로 그쪽 pgrep 규칙(-u …/live_sr_lean_20261118.py)에 안 걸린다.
#
# crontab(서버 재부팅 때) -- 적기만 했다. 등록은 사람이 한다(SHADOW.md 순서대로):
#   @reboot sleep 90 && cd <저장소> && bash scripts/ops/supervisor_rt_sr_lean.sh >> logs/supervisor/rt_sr_lean_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${RT_SR_BIN:-$ROOT/bin/rt_sr_lean}"
OUTD="${RT_SR_OUT:-data/live/rt_sr_lean}"
MODEL="${RT_SR_MODEL:-$OUTD/sr_lean.json}"

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] 실행 파일이 없다: $BIN (rtmodel/SHADOW.md «실행 파일과 모델»)" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "[$(date -Iseconds)] 모델이 없다: $MODEL -- python rtmodel/tools/export_hgb.py $OUTD" >&2
  exit 1
fi
envof() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep "^$2=" | head -1 | cut -d= -f2-; }
for pid in $(pgrep -x rt_sr_lean); do
  cur=$(envof "$pid" RT_SR_OUT)
  if [ "$(realpath -m "${cur:-data/live/rt_sr_lean}")" = "$(realpath -m "$OUTD")" ]; then
    echo "[$(date -Iseconds)] rt_sr_lean($OUTD)이 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export RT_SR_OUT="$OUTD" RT_SR_MODEL="$MODEL"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "rt_sr_lean($OUTD)" \
  "$ROOT/data/live/.supervisor_rt_sr_lean.lock" \
  "$ROOT/logs/supervisor/rt_sr_lean" \
  nice -n 10 "$BIN"
