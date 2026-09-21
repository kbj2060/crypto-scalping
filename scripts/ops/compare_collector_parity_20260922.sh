#!/usr/bin/env bash
# Pi(collector) 와 server 가 같은 시간대에 같은 건수를 받았는지 대조.
# 수집기를 서버에서 내리기 전에 «Pi 가 대체 가능한가»를 판정하는 근거다.
# dev 에서 실행. handoff.hosts.conf 를 재사용한다.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTS_FILE="${HANDOFF_HOSTS_FILE:-$SCRIPT_DIR/handoff.hosts.conf}"
[[ -f "$HOSTS_FILE" ]] || { echo "missing $HOSTS_FILE" >&2; exit 1; }
declare -A HOSTS; source "$HOSTS_FILE"
mkdir -p "$HOME/.ssh/cm" 2>/dev/null && chmod 700 "$HOME/.ssh/cm" 2>/dev/null || true
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8
          -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/%C -o ControlPersist=60"

# bookTicker 는 32B 헤더 + 32B 행이라 파일 크기로 정확히 센다.
# depthdiff 는 jsonl 줄 수. 로테이션되면 .gz 이므로 zcat 로 센다.
# 진행 중인 시각(마지막 파일)은 양쪽 시점이 달라 비교가 무의미하므로 제외한다.
probe='
  R="$1"
  cd "$R" 2>/dev/null || exit 0
  for f in data/live/orderflow/bookticker/ETHUSDT/*.bt; do
    [ -e "$f" ] || continue
    h=$(basename "$f" .bt); s=$(stat -c %s "$f")
    echo "bt|$h|$(( (s - 32) / 32 ))"
  done
  for f in data/live/orderflow/depthdiff/ETHUSDT/*.jsonl data/live/orderflow/depthdiff/ETHUSDT/*.jsonl.gz; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); h="${b%%.jsonl*}"
    if [[ "$f" == *.gz ]]; then n=$(zcat "$f" 2>/dev/null | wc -l); else n=$(wc -l < "$f"); fi
    echo "dd|$h|$n"
  done
'
fetch() {  # $1=alias
  IFS='|' read -r CONN REPO _ _ <<< "${HOSTS[$1]}"
  IFS=':' read -r UH PORT <<< "$CONN"
  if [[ "$REPO" == "$HOME/"* ]]; then bash -c "$probe" bash "$REPO" 2>/dev/null
  else timeout 90 ssh -p "$PORT" $SSH_OPTS "$UH" "bash -s -- '$REPO'" <<< "$probe" 2>/dev/null; fi
}

PI=$(fetch collector); SV=$(fetch server)
[[ -n "$PI" ]] || { echo "collector 에서 데이터를 못 읽었습니다" >&2; exit 1; }
[[ -n "$SV" ]] || { echo "server 에서 데이터를 못 읽었습니다" >&2; exit 1; }

# 양쪽 모두 가진 시각만, 그리고 각자의 마지막(진행 중) 시각은 뺀다.
LAST_PI=$(cut -d'|' -f2 <<< "$PI" | sort -u | tail -1)
LAST_SV=$(cut -d'|' -f2 <<< "$SV" | sort -u | tail -1)

printf '%-4s %-16s %12s %12s %9s %s\n' 종류 시각UTC collector server 차이 판정
rc=0
while IFS='|' read -r k h n; do
  [[ "$h" == "$LAST_PI" || "$h" == "$LAST_SV" ]] && continue
  m=$(awk -F'|' -v k="$k" -v h="$h" '$1==k && $2==h {print $3}' <<< "$SV")
  [[ -n "$m" ]] || continue
  d=$(awk -v a="$n" -v b="$m" 'BEGIN{printf "%+.2f%%", (b==0?0:100*(a-b)/b)}')
  v=$(awk -v a="$n" -v b="$m" 'BEGIN{r=(b==0?0:100*(a-b)/b); if(r<0)r=-r; print (r<=1.0 ? "OK" : "DIFF")}')
  [[ "$v" == OK ]] || rc=1
  printf '%-4s %-16s %12s %12s %9s %s\n' "$k" "$h" "$n" "$m" "$d" "$v"
done < <(sort -t'|' -k1,1 -k2,2 <<< "$PI")
echo
echo "진행 중인 시각은 제외했습니다 (collector=$LAST_PI, server=$LAST_SV)."
echo "판정: 차이 1% 이내면 OK. 전부 OK 면 서버측 수집기를 내려도 됩니다."
exit $rc
