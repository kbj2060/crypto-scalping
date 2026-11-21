#!/usr/bin/env bash
# 파이썬 무장 파일 사본을 rtorder 그림자 폴더로 넣는다 (2026-11-21 · rtorder/SHADOW.md 3절). **읽기만** -- 파이썬 data/live 는 건드리지 않는다.
# 그림자 rtorder 는 접수가 거래소에 안 나가 스스로 무장이 생기지 않는다 -- 파이썬이 지금 무장한 선을 같은 판정으로 보려면 이 사본이 필요하다.
#   바뀌었을 때만(cmp) 임시 파일 → mv 로 바꿔 넣는다(rtorder 가 반쪽 파일을 읽지 않게) · 주기 RTORDER_COPY_SEC(기본 2초 = 감시 주기).
#   원본이 없으면(무장 없음) 사본도 지운다.
# 실행(서버): nohup bash scripts/ops/rtorder_shadow_copy_brackets.sh >> logs/supervisor/rtorder_copy.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/data/live/manual_bracket_state.json"
DST_DIR="${RTORDER_DIR:-/mnt/data2/rtcollect_shadow/rtorder}"
DST="$DST_DIR/manual_bracket_state.json"
SEC="${RTORDER_COPY_SEC:-2}"
case "$DST_DIR" in
  "$ROOT/data/live"|"$ROOT/data/live/") echo "사본 폴더가 파이썬 상태 폴더다 -- 하지 않는다." >&2; exit 1 ;;
esac
mkdir -p "$DST_DIR"
exec 9>"$DST_DIR/.copy_brackets.lock"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] 사본 넣기가 이미 돈다 -- 끝낸다." >&2
  exit 1
fi
echo "[$(date -Iseconds)] 무장 파일 사본 시작: $SRC → $DST (${SEC}초마다)"
while true; do
  if [[ -f "$SRC" ]]; then
    if ! cmp -s "$SRC" "$DST" 2>/dev/null; then
      tmp="$DST.copy.$$"
      if cp -- "$SRC" "$tmp" 2>/dev/null && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$tmp" 2>/dev/null; then
        mv -f -- "$tmp" "$DST" && echo "[$(date -Iseconds)] 사본 바꿈 ($(wc -c < "$DST") 바이트)"
      else
        rm -f -- "$tmp"   # 파이썬이 쓰는 중(반쪽) -- 다음 바퀴에
      fi
    fi
  elif [[ -f "$DST" ]]; then
    rm -f -- "$DST" && echo "[$(date -Iseconds)] 원본 없음(무장 없음) -- 사본 지움"
  fi
  sleep "$SEC"
done
