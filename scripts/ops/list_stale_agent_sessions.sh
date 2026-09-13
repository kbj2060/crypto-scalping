#!/usr/bin/env bash
# 오래 남은 Claude 에이전트 세션을 **보여준다**. 죽이지 않는다.
#
# 🔴자동으로 죽이지 않는 이유: «고아»를 나이로 가릴 수 없다. 진짜 워커도 며칠씩 돌고,
# 지금 쓰고 있는 세션도 하루를 넘긴다. 자동 킬러가 살아 있는 세션을 죽이면 그게 더 큰 사고다.
# 그래서 여기서는 **지금 이 셸의 조상은 표시하고 제외**한 뒤 나머지를 나이순으로 보여준다.
#
# 폭주 자체는 handoff.sh 의 rate_guard 가 막는다(분당 30회 상한). 이 스크립트는
# 2026-09-13 사고 이후 «남은 게 있나»를 사람이 한눈에 보려고 둔다. 근거: 그날 앱을 지운 뒤에도
# ccd-cli 4개가 30시간 넘게 살아 있었고 그중 하나가 sleep 없는 폴링 루프를 돌리고 있었다.
set -uo pipefail

# 이 셸의 조상 PID 들 -- 죽이면 안 되는 것들이다.
ancestors() {
  local a=$$ p
  while [[ -n "$a" && "$a" != "1" ]]; do
    echo "$a"
    p=$(awk '{print $4}' "/proc/$a/stat" 2>/dev/null) || break
    a=$p
  done
}
mapfile -t SAFE < <(ancestors)
in_safe() { local x; for x in "${SAFE[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

printf '%-8s %10s  %s\n' PID 가동 명령
found=0
while read -r pid etimes rest; do
  [[ -z "${pid:-}" ]] && continue
  if in_safe "$pid"; then
    printf '%-8s %9ss  %s  <= 지금 세션(제외)\n' "$pid" "$etimes" "${rest:0:52}"
    continue
  fi
  printf '%-8s %9ss  %s\n' "$pid" "$etimes" "${rest:0:52}"
  found=$((found + 1))
done < <(ps -eo pid,etimes,args 2>/dev/null | grep -E 'ccd-cli|\.claude/remote/srv' | grep -v grep)

echo
if (( found > 0 )); then
  cat <<MSG
남은 세션 ${found}개. 확인 후 지우려면 (한 번에 하나씩, 위 «지금 세션» 은 절대 제외):
  kill \$(pgrep -P <PID>) <PID>
🔴 '.claude/remote/srv/.../server' 는 여러 세션이 공유한다 -- 죽이면 지금 대화도 끊긴다.
MSG
else
  echo "남은 세션 없음."
fi
