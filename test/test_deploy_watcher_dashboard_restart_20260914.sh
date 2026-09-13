#!/usr/bin/env bash
# deploy_watcher 의 restart_dashboard 계약 검사 (2026-09-14). 프레임워크 없이 그냥 실행한다.
#
# 왜 있나: pid 파일이 없으면 예전 판은 "nothing to restart" 로 **조용히 성공 반환**했다.
# 그러면 워처가 머지하고 last_deployed_sha 를 갱신하고 `deploy OK` 까지 찍는데 프로세스는
# 옛 코드를 들고 계속 돈다(2026-09-13 실장애: 디스크는 새 코드, API 응답은 옛 필드).
# 헬스체크도 못 잡는다 -- "8787 리슨 중"은 **아무것도 안 죽였을 때 가장 확실히 통과**한다.
#
# 여기서 지키는 계약:
#   · pid 파일이 있고 살아 있으면 그 pid 를 죽인다
#   · pid 파일이 없으면 **포트 주인**으로 떨어진다(그리고 죽인다)
#   · pid 파일이 낡았어도(죽은 pid) 포트 주인으로 떨어진다
#   · 포트 주인도 없으면 아무도 안 죽이고 조용히 성공(감독이 띄운다)
#   · SIGTERM 을 무시하면 SIGKILL 로 승격한다 -- server.py 의 고질 증상
#
# 실행: bash test/test_deploy_watcher_dashboard_restart_20260914.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCHER="$ROOT/scripts/ops/deploy_watcher.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# 대상 함수만 떼어낸다 -- 워처 본문을 통째로 실행하면 실제 배포가 돈다.
sed -n '/^restart_dashboard() {/,/^}/p' "$WATCHER" > "$TMP/fn.sh"
[[ -s "$TMP/fn.sh" ]] || { echo "🔴 restart_dashboard 를 못 찾았다"; exit 1; }

run_case() {   # run_case <이름> <pid파일내용|-> <살아있는pid들> <포트주인|-> <기대 kill 대상|->
  local name="$1" pidfile="$2" alive="$3" owner="$4" want="$5"
  local dir="$TMP/$name"; mkdir -p "$dir/data/live"
  [[ "$pidfile" == "-" ]] || echo "$pidfile" > "$dir/data/live/dashboard_external.child.pid"
  : > "$dir/killed"
  {
    echo "ROOT='$dir'; DASHBOARD_PORT=8787; DASHBOARD_SIGTERM_GRACE_SECONDS=0"
    echo "ALIVE='$alive'; OWNER='$owner'; KILLED='$dir/killed'"
    cat <<'STUB'
log() { echo "    log: $*"; }
# 가짜 ss -- 포트 주인을 OWNER 로 답한다
ss() { [[ "$OWNER" == "-" ]] && return 0; echo "LISTEN 0 128 127.0.0.1:8787 0.0.0.0:* users:((\"python\",pid=$OWNER,fd=11))"; }
# 가짜 kill -- -0 는 생존조회, 그 외는 기록. SIGTERM 을 받아도 계속 살아 있다(고질 증상 재현).
kill() {
  if [[ "$1" == "-0" ]]; then [[ " $ALIVE " == *" $2 "* ]] && return 0 || return 1; fi
  if [[ "$1" == "-9" ]]; then echo "KILL9 $2" >> "$KILLED"; return 0; fi
  echo "TERM $1" >> "$KILLED"; return 0
}
sleep() { :; }
STUB
    cat "$TMP/fn.sh"
    echo "restart_dashboard"
  } > "$dir/run.sh"
  echo "■ $name"
  bash "$dir/run.sh" >/dev/null 2>&1
  local got; got="$(grep -oE '[0-9]+' "$dir/killed" 2>/dev/null | head -1)"; got="${got:--}"
  if [[ "$got" == "$want" ]]; then
    echo "    OK  죽인 대상 ${got}"
  else
    echo "    🔴 기대 ${want} · 실제 ${got}"; fail=1
  fi
  printf '%s' "$(cat "$dir/killed" 2>/dev/null)" > "$dir/killed.flat"
}

# ① 정상: pid 파일이 살아 있는 프로세스를 가리킨다
run_case normal 4242 "4242" 4242 4242
# ② 🔴 pid 파일 없음 -> 포트 주인으로 떨어져야 한다(예전 판은 여기서 아무것도 안 했다)
run_case no_pidfile - "7777" 7777 7777
# ③ pid 파일이 낡음(그 pid 는 이미 죽음) -> 역시 포트 주인
run_case stale_pidfile 1111 "8888" 8888 8888
# ④ 아무도 포트를 안 쥐고 있음 -> 죽일 대상 없음(감독이 띄운다)
run_case nobody - "" - -

# ⑤ SIGTERM 무시 -> SIGKILL 승격. ①~④ 의 가짜 kill 은 TERM 후에도 계속 살아 있으므로
#    정상 케이스의 기록에 KILL9 가 남아 있어야 한다.
if grep -q 'KILL9' "$TMP/normal/killed"; then
  echo "■ sigterm_escalation"; echo "    OK  SIGTERM 무시시 SIGKILL 승격"
else
  echo "■ sigterm_escalation"; echo "    🔴 SIGKILL 승격이 없다"; fail=1
fi

echo
if [[ "$fail" == "0" ]]; then echo "통과 5/5 — 정상·pid없음·낡은pid·주인없음·SIGKILL승격"; else echo "🔴 실패"; fi
exit "$fail"
