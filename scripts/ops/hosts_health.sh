#!/usr/bin/env bash
# dev / server / collector 상호 상태 체크. handoff.hosts.conf 를 그대로 재사용하므로
# 세 머신 어디서 돌려도 같은 표가 나온다 (각 머신에 conf 사본이 있어야 한다).
#
# 시각은 NTP 동기 여부로 본다. 수집기는 타임스탬프가 전부라 시각이 벌어지면 조인이
# 조용히 틀리는데 -- ssh 왕복으로 오차를 "계산"하면 재는 건 시계가 아니라 ssh 지연이다
# (2026-09-21 실측: 자기 자신에게도 -0.94s 가 나왔다). NTP 동기 여부가 정확한 질문이다.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTS_FILE="${HANDOFF_HOSTS_FILE:-$SCRIPT_DIR/handoff.hosts.conf}"
[[ -f "$HOSTS_FILE" ]] || { echo "missing $HOSTS_FILE" >&2; exit 1; }
declare -A HOSTS; source "$HOSTS_FILE"

# handoff.sh 와 같은 다중화 소켓을 쓴다. 디렉터리가 없으면 ssh 가 통째로 실패하므로
# (unix_listener: cannot bind to path) 여기서도 직접 만든다 -- handoff.sh 가 그렇게 한다.
mkdir -p "$HOME/.ssh/cm" 2>/dev/null && chmod 700 "$HOME/.ssh/cm" 2>/dev/null || true
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5
          -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/%C -o ControlPersist=60"

# 자기 자신에게는 ssh 하지 않는다. 그러려면 각 호스트의 키를 자기 authorized_keys 에
# 넣어야 하는데, 로컬 실행이 더 빠르고 키 표면도 안 늘린다.
#
# 판별을 IP 로 하지 않는 이유: server 는 WSL 이라 실제 인터페이스가 172.18.x 이고
# conf 의 192.168.1.89 는 윈도우 호스트 주소다 -- 자기 IP 를 자기가 안 갖고 있다.
# repo 경로가 $HOME 아래인지로 본다 (세 호스트의 홈이 각각 다르다: llewyn/kbj20/pi).

# 원격에서 한 줄을 '|' 로 붙여 돌려준다. 공백 분리로 파싱하면 필드 하나가 비었을 때
# 컬럼이 통째로 밀린다 -- 실제로 그렇게 한 번 틀렸다.
probe='
  d=$(df -P "$1" 2>/dev/null | awk "NR==2{print \$5}")
  [ -n "$d" ] || d=$(df -P "$HOME" | awk "NR==2{print \$5}")
  [ -f "$2/etc/profile.d/conda.sh" ] && e=env_ok || e=env_MISSING
  printf "%s|%s|%s|%s\n" "$(cut -d" " -f1 /proc/loadavg)" "$d" \
    "$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo ?)" "$e"
'

printf '%-11s %-24s %-6s %-6s %-6s %-6s %s\n' HOST ENDPOINT REACH LOAD DISK NTP ENV
rc=0
for h in $(echo "${!HOSTS[@]}" | tr ' ' '\n' | sort); do
  IFS='|' read -r CONN REPO CONDA_BASE _ <<< "${HOSTS[$h]}"
  IFS=':' read -r USERHOST PORT <<< "$CONN"
  # 한 번의 접속으로 전부 받는다 -- 항목마다 접속하면 09-13 폭주 사고의 그 패턴이 된다.
  if [[ "$REPO" == "$HOME/"* ]]; then
    out=$(bash -c "$probe" bash "$REPO" "$CONDA_BASE" 2>/dev/null | tail -1)
  else
    out=$(timeout 12 ssh -p "$PORT" $SSH_OPTS "$USERHOST" \
          "bash -s -- '$REPO' '$CONDA_BASE'" <<< "$probe" 2>/dev/null | tail -1)
  fi
  if [[ -z "$out" ]]; then
    printf '%-11s %-24s %-6s %-6s %-6s %-6s %s\n' "$h" "$CONN" DOWN - - - -
    rc=1; continue
  fi
  IFS='|' read -r load disk ntp envst <<< "$out"
  printf '%-11s %-24s %-6s %-6s %-6s %-6s %s\n' "$h" "$CONN" UP "$load" "$disk" "$ntp" "$envst"
  [[ "$envst" == env_ok && "$ntp" == yes ]] || rc=1
done
exit $rc
