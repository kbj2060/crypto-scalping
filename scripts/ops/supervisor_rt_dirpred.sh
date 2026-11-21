#!/usr/bin/env bash
# rt_dirpred(Rust 4h·15분/1시간 방향 예측 + 상대 강도 기록 루프) 크래시 재기동 래퍼 (2026-11-21). 지금은 **그림자** 전용 -- rtmodel/SHADOW.md.
#
# 실행 파일: rtmodel/target/release/rt_dirpred 를 개발기에서 빌드해 옮긴 것(RT_DIR_BIN, 기본 $ROOT/tmp/rtbin/rt_dirpred).
# 쓰는 곳: 기본 data/live/rt_dirpred/(RT_DIR_OUT) -- 파이썬 서버의 data/live 는 실행 파일이 거부한다(종료코드 2 · 전환 때만 RT_DIR_TAKEOVER=1).
# 모델: <RT_DIR_OUT>/dir4h_<코인>.json · ds_<코인>.json · oos_<코인>.json(RT_DIR_MODELS) -- SHADOW.md «rt_dirpred» 의 내보내기 두 줄. 하나도 없으면 켜지 않는다.
# 로그: logs/supervisor/rt_dirpred_YYYYMMDD.log · 락: data/live/.supervisor_rt_dirpred.lock
# 종료코드와 무관하게 다시 띄운다(_supervise.sh: 15초 뒤 · 10분에 5번이면 300초 쉼).
# ⚠️같은 출력 디렉터리의 rt_dirpred 가 이미 돌면 켜지 않는다(같은 기록 파일에 두 벌). 파이썬 서버와는 겹쳐도 된다(다른 디렉터리 · REST 는 두 벌 -- SHADOW.md 표).
#
# crontab(서버 재부팅 때) -- 적기만 했다. 등록은 사람이 한다:
#   @reboot sleep 120 && cd <저장소> && bash scripts/ops/supervisor_rt_dirpred.sh >> logs/supervisor/rt_dirpred_reboot.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BIN="${RT_DIR_BIN:-$ROOT/tmp/rtbin/rt_dirpred}"
OUTD="${RT_DIR_OUT:-data/live/rt_dirpred}"
MODELS="${RT_DIR_MODELS:-$OUTD}"

if [ ! -x "$BIN" ]; then
  echo "[$(date -Iseconds)] 실행 파일이 없다: $BIN (rtmodel/SHADOW.md «rt_dirpred»)" >&2
  exit 1
fi
if ! ls "$MODELS"/dir4h_*.json "$MODELS"/ds_*.json >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 모델이 없다: $MODELS/dir4h_*.json · ds_*.json -- SHADOW.md 의 export_dir4h.py · export_dirshort.py" >&2
  exit 1
fi
envof() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep "^$2=" | head -1 | cut -d= -f2-; }
for pid in $(pgrep -x rt_dirpred); do
  cur=$(envof "$pid" RT_DIR_OUT)
  if [ "$(realpath -m "${cur:-data/live/rt_dirpred}")" = "$(realpath -m "$OUTD")" ]; then
    echo "[$(date -Iseconds)] rt_dirpred($OUTD)이 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export RT_DIR_OUT="$OUTD" RT_DIR_MODELS="$MODELS"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "rt_dirpred($OUTD)" \
  "$ROOT/data/live/.supervisor_rt_dirpred.lock" \
  "$ROOT/logs/supervisor/rt_dirpred" \
  nice -n 10 "$BIN"
