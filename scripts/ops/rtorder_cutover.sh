#!/usr/bin/env bash
# 주문 경로 전환: 파이썬 대시보드 ↔ rtorder 실전송 (2026-11-22 · 사용자 승인 «지금 전환해줘»). 절차 원문: rtorder/SHADOW.md §9.
#   bash scripts/ops/rtorder_cutover.sh status   읽기만: 전환 조건을 본다
#   bash scripts/ops/rtorder_cutover.sh go       파이썬 → rtorder (조건이 안 맞으면 아무것도 안 바꿈 · 중간 확인이 실패하면 스스로 되돌림)
#   bash scripts/ops/rtorder_cutover.sh back     rtorder → 파이썬
# 원칙: 손절 감시가 겹치지 않게(두 번 청산) 짧게 비운다. 표식 data/live/rtorder_live.on = switching(주문 POST 503) → live(주문이 rtorder 로).
# 실주문을 내는 명령은 여기 없다 -- 프로세스와 표식만 바꾼다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
MODE="${1:?status|go|back}"
LIVE="$ROOT/data/live"; MARK="$LIVE/rtorder_live.on"; BR=manual_bracket_state.json
RDIR="$LIVE/rtorder"; SDIR="${RTORDER_SHADOW_DIR:-/mnt/data2/rtcollect_shadow/rtorder}"
# rtserve 전환 표식(data/live/rtserve_live.on)이 있으면 thesan.xyz 응답 서버 = rtserve(8790) -- 조회는 그쪽으로 묻는다.
PYD=http://127.0.0.1:8787; [ -f "$LIVE/rtserve_live.on" ] && PYD=http://127.0.0.1:8790; DOOR=http://127.0.0.1:8791; C="${CURL:-/usr/bin/curl}"
ASSETS="${CUTOVER_ASSETS:-eth sol xrp btc soxl sndk}"
TAG="#rtorder-live"
say() { echo "[cutover $MODE $(date +%H:%M:%S)] $*"; }
die() { say "🔴$*"; exit 1; }
mark_set() { printf '%s\n' "$1" > "$MARK.tmp" && mv -f "$MARK.tmp" "$MARK"; }
argv() { tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline"; }
sup_pids() { local p; for p in $(pgrep -f '_supervise.sh rtorder' || true); do argv "$p" | grep -q "$LIVE/.supervisor_rtorder.lock" && echo "$p"; done; }
copy_pids() { pgrep -f 'rtorder_shadow_copy_brackets.sh' || true; }
gone() { [ -z "$(ps -o stat= -p "$1" 2>/dev/null | grep -v '^[ZX]')" ]; }
wait_gone() { local t0=$SECONDS; while ! gone "$1" && [ $((SECONDS - t0)) -lt "$2" ]; do sleep 0.2; done; gone "$1"; }
stop_rtorder() {   # 감독자 먼저(다시 띄우지 않게) → rtorder SIGTERM
  local p
  for p in $(sup_pids); do kill -TERM "$p" 2>/dev/null; wait_gone "$p" 5 || { say "감독자 $p 가 안 죽는다"; return 1; }; done
  for p in $(pgrep -x rtorder || true); do kill -TERM "$p" 2>/dev/null; wait_gone "$p" 10 || { say "rtorder $p 가 SIGTERM 10초 뒤에도 산다"; return 1; }; done
  return 0
}
acct() { "$C" -s -m 3 "$DOOR/internal/acct"; }
mode_line() { local f="logs/supervisor/rtorder_$(date +%Y%m%d).log"; awk '/SUPERVISOR starting/{n=NR} {l[NR]=$0} END{print l[n+1]}' "$f" 2>/dev/null; }
idle_check() {   # $1 = 물을 곳(파이썬 또는 문) -- 도는 작업·대기가 없어야 0
  local a j bad=0
  for a in $ASSETS; do
    j="$("$C" -s -m 5 "$1/api/manual-entry/status?asset=$a")" || { say "status($a) 못 읽음"; return 1; }
    printf '%s' "$j" | python3 -c '
import sys, json
d = json.load(sys.stdin); st = d.get("state") or {}; ph = st.get("phase")
busy = [k for k, v in list((d.get("waits") or {}).items()) + [("wait", d.get("wait") or {})] if isinstance(v, dict) and v.get("phase") not in (None, "idle")]   # 서버 실제 모양: waits = {코인: {phase}}
ex = [k for k, v in (d.get("side_exits") or {}).items() if isinstance(v, dict) and v.get("phase") not in (None, "idle", "done", "failed", "cancelled")]
ok = ph in (None, "idle", "filled_maker", "filled_taker", "done", "failed", "cancelled", "expired", "rejected") and not d.get("wait_list") and not busy and not ex
print("  %s: 진입 칸 %s · 대기 %d · busy %s · 청산 중 %s · 무장 %s → %s" % (sys.argv[1], ph, len(d.get("wait_list") or []), busy, ex, d.get("bracket_armed"), "됨" if ok else "🔴안 됨"))
sys.exit(0 if ok else 1)' "$a" || bad=1
  done
  ls "$LIVE"/manual_limit_wait_*.json >/dev/null 2>&1 && { say "대기 파일(manual_limit_wait_*.json)이 있다"; bad=1; }
  return $bad
}
status() {
  say "표식: $([ -f "$MARK" ] && echo "있음($(tr -d '\n' < "$MARK"))" || echo 없음) · rtorder $(pgrep -x rtorder | wc -l)개 · 첫 줄 «$(mode_line)» · 감독자 $(sup_pids | wc -l) · 사본 스크립트 $(copy_pids | wc -l)"
  say "문: $(acct | cut -c1-220)"
  say "파이썬 무장 파일: $([ -f "$LIVE/$BR" ] && echo "$(wc -c < "$LIVE/$BR")바이트" || echo 없음) · rtorder 폴더: $([ -f "$RDIR/$BR" ] && echo "$(wc -c < "$RDIR/$BR")바이트" || echo 없음)"
  say "실전송 판 실행 파일: $([ -x tmp/rtbin/rtorder ] && sha256sum tmp/rtbin/rtorder | cut -c1-16) · crontab 줄 $(crontab -l 2>/dev/null | grep -c "$TAG")"
  idle_check "$([ -f "$MARK" ] && echo "$DOOR" || echo "$PYD")"; say "조건(도는 작업 없음): $([ $? = 0 ] && echo 됨 || echo 🔴안 됨)"
}
rollback() {   # go 도중 실패 -- 파이썬 감시로 되돌린다(rtorder 는 아직 주문을 받은 적이 없다: 표식이 live 가 아니었다)
  say "되돌림: $*"
  stop_rtorder
  rm -f "$MARK" "$MARK.tmp"; say "표식 지움 -- 파이썬이 다음 틱부터 감시"
  setsid nohup bash scripts/ops/rtorder_shadow_copy_brackets.sh >> logs/supervisor/rtorder_copy.log 2>&1 < /dev/null &
  setsid nohup bash scripts/ops/supervisor_rtorder.sh > /dev/null 2>> logs/supervisor/rtorder_reboot.log < /dev/null &
  sleep 3; say "그림자 다시 띄움 · rtorder $(pgrep -x rtorder | wc -l) · 표식 $([ -f "$MARK" ] && echo 있음 || echo 없음)"
  exit 1
}
case "$MODE" in
status) status ;;
go)
  [ -f "$MARK" ] && die "표식이 이미 있다($(tr -d '\n' < "$MARK")) -- status 로 먼저 본다"
  [ -f scripts/ops/supervisor_rtorder_live.sh ] || die "실전송 감독 스크립트가 없다"
  grep -q "rtorder_on" dashboard/orders.py || die "대시보드에 전환 스위치가 없다"
  [ "$(pgrep -x rtorder | wc -l)" = 1 ] || die "그림자 rtorder 가 하나 떠 있어야 한다"
  mode_line | grep -q "그림자" || die "지금 rtorder 가 그림자가 아니다: «$(mode_line)»"
  acct | grep -q '"stale": false' || die "rtorder 허브 가격 수신이 정상이 아니다: $(acct | cut -c1-160)"
  "$C" -s -o /dev/null -m 5 "$PYD/api/manual-entry/status" || die "파이썬 대시보드가 안 열려 있다"
  idle_check "$PYD" || die "도는 작업·대기가 있다 -- 아무것도 안 바꿈"
  say "조건 통과 -- 전환 시작"
  for p in $(copy_pids); do kill -TERM "$p" 2>/dev/null; done
  m0="$(stat -c %Y "$LIVE/$BR" 2>/dev/null || echo 0)"
  T0=$(date +%s.%N)
  mark_set switching; say "표식 = switching (파이썬 감시 멈춤 · 주문 POST 503) ← 빈 시간 시작"
  sleep 3
  [ "$(stat -c %Y "$LIVE/$BR" 2>/dev/null || echo 0)" = "$m0" ] || rollback "표식 뒤 파이썬 무장 파일이 바뀌었다(그 틱이 청산을 시작했을 수 있다)"
  ls "$LIVE"/manual_limit_wait_*.json >/dev/null 2>&1 && rollback "대기 파일이 생겼다"
  mkdir -p "$RDIR"
  if [ -f "$LIVE/$BR" ]; then cp -p "$LIVE/$BR" "$RDIR/$BR" && cmp -s "$LIVE/$BR" "$RDIR/$BR" || rollback "무장 파일 복사 실패"; say "무장 파일 넘김($(wc -c < "$RDIR/$BR")바이트)"; else rm -f "$RDIR/$BR"; say "무장 없음(넘길 파일 없음)"; fi
  stop_rtorder || rollback "그림자 rtorder 를 못 껐다"
  setsid nohup bash scripts/ops/supervisor_rtorder_live.sh > /dev/null 2>> logs/supervisor/rtorder_reboot.log < /dev/null &
  ok=""
  for i in $(seq 1 60); do
    sleep 0.5
    mode_line | grep -q "실전송" || continue
    a="$(acct)"; printf '%s' "$a" | grep -q '"connected": true' || continue
    printf '%s' "$a" | grep -q '"stale": false' || continue
    ok=1; break
  done
  [ -n "$ok" ] || rollback "30초 안에 실전송 rtorder 가 준비되지 않았다(첫 줄 «$(mode_line)» · 문 $(acct | cut -c1-120))"
  T1=$(date +%s.%N)
  say "rtorder 실전송 준비됨(첫 줄 «$(mode_line)») ← 빈 시간 끝 · 빈 시간 약 $(echo "$T1 - $T0" | bc | cut -c1-5)초"
  for a in $ASSETS; do   # 파이썬이 조회를 rtorder 로 넘기는지: 같은 요청의 답이 문의 답과 같아야 한다
    x="$("$C" -s -m 5 "$PYD/api/manual-entry/status?asset=$a" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("ok"), d.get("exec_enabled"), d.get("bracket_armed"), (d.get("state") or {}).get("phase"))' 2>&1)"
    y="$("$C" -s -m 5 "$DOOR/api/manual-entry/status?asset=$a" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("ok"), d.get("exec_enabled"), d.get("bracket_armed"), (d.get("state") or {}).get("phase"))' 2>&1)"
    [ "$x" = "$y" ] || rollback "파이썬(8787)과 문(8791)의 status($a) 답이 다르다: «$x» / «$y»"
  done
  [ "$("$C" -s -o /dev/null -w '%{http_code}' -m 8 "$PYD/api/binance-account")" = 200 ] || rollback "계좌 조회가 rtorder 를 거쳐 200 이 아니다"
  grep -aE "panicked" "logs/supervisor/rtorder_$(date +%Y%m%d).log" | tail -n 1 | grep -q . && say "⚠️ 오늘 로그에 panic 줄이 있다(그림자 때 것일 수 있다 -- 확인)"
  mark_set live; say "표식 = live -- 주문이 rtorder 로 간다 · 주문 막힌 시간 약 $(echo "$(date +%s.%N) - $T0" | bc | cut -c1-5)초"
  if ! crontab -l 2>/dev/null | grep -q "$TAG"; then
    crontab -l > "logs/supervisor/crontab_backup_$(date +%Y%m%d_%H%M%S)_rtorder_live.txt"
    { crontab -l; echo "@reboot sleep 75 && cd $ROOT && bash scripts/ops/supervisor_rtorder_live.sh >> logs/supervisor/rtorder_reboot.log 2>&1 & $TAG"; } | crontab -
    say "crontab: 재부팅 때 실전송 rtorder 를 띄우는 줄 추가(표식이 있을 때만 뜬다)"
  fi
  status ;;
back)
  [ -f "$MARK" ] || die "표식이 없다 -- 이미 파이썬이 감시 중"
  ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE '(^|:)8787$' || die "파이썬 대시보드(8787)가 듣고 있지 않다 -- 표식을 지우면 손절 감시가 아무 데도 없다. 먼저 파이썬 대시보드를 살린 뒤 back"
  if pgrep -x rtorder >/dev/null; then idle_check "$DOOR" || die "rtorder 에 도는 작업·대기가 있다 -- 끝난 뒤에"; fi
  mark_set switching; say "표식 = switching (주문 POST 503)"
  stop_rtorder || say "⚠️ rtorder 를 깨끗이 못 껐다 -- 계속(무장 파일은 디스크에 있다)"
  say "rtorder 끔 ← 빈 시간 시작"
  if [ -f "$RDIR/$BR" ]; then
    [ -f "$LIVE/$BR" ] && cp -p "$LIVE/$BR" "$LIVE/$BR.before_back_$(date +%Y%m%d_%H%M%S)"
    cp -p "$RDIR/$BR" "$LIVE/$BR.tmp" && mv -f "$LIVE/$BR.tmp" "$LIVE/$BR" && cmp -s "$RDIR/$BR" "$LIVE/$BR" || say "⚠️ 무장 파일 되복사 실패 -- 파이썬 쪽 옛 파일로 감시한다"
  else rm -f "$LIVE/$BR"; fi
  rm -f "$MARK" "$MARK.tmp"; say "표식 지움 -- 파이썬이 다음 틱(≤2초)부터 감시 ← 빈 시간 끝"
  if crontab -l 2>/dev/null | grep -q "$TAG"; then crontab -l > "logs/supervisor/crontab_backup_$(date +%Y%m%d_%H%M%S)_rtorder_back.txt"; crontab -l | grep -v "$TAG" | crontab -; say "crontab: 실전송 줄 뺌"; fi
  setsid nohup bash scripts/ops/rtorder_shadow_copy_brackets.sh >> logs/supervisor/rtorder_copy.log 2>&1 < /dev/null &
  setsid nohup bash scripts/ops/supervisor_rtorder.sh > /dev/null 2>> logs/supervisor/rtorder_reboot.log < /dev/null &
  sleep 5; status ;;
*) die "status|go|back" ;;
esac
