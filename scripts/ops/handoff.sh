#!/usr/bin/env bash
# One-command sync + remote job control between the dev and server machines (same LAN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# Real IPs/ports/usernames live only in this gitignored local file, never in
# the repo (this is a public repo). See handoff.hosts.conf.example.
HOSTS_FILE="${HANDOFF_HOSTS_FILE:-$SCRIPT_DIR/handoff.hosts.conf}"
if [[ ! -f "$HOSTS_FILE" ]]; then
  echo "missing $HOSTS_FILE - copy scripts/ops/handoff.hosts.conf.example to that path and fill in real values" >&2
  exit 1
fi
declare -A HOSTS
source "$HOSTS_FILE"
JOBS_DIR="tmp/handoff_jobs"
# 🔴2026-09-17: SSH 연결 다중화(ControlMaster). 아래 rate_guard 는 **호출 빈도**만 막고
# **연결당 비용**은 못 줄인다 -- 09-13 사고의 실제 비용이 그거였다: 세션마다 TCP 핸드셰이크 +
# sshd 포크 + PAM + systemd-logind 스코프 생성·삭제. "세션은 명령을 실행하지 않는다.
# 접속·인증·즉시 종료다"(그 사고 기록). 게다가 `launch` 한 번이 rsync+ssh 여러 개라
# 상한 30회/분이 실제로는 분당 100~150 연결을 허용한다.
# ControlMaster=auto 면 첫 연결이 마스터 소켓을 만들고 이후 ssh/rsync 가 **그 하나를 재사용**한다
# -- 새 TCP 도, 인증도, logind 스코프도 없다. 마스터가 죽으면 auto 가 알아서 새로 연다.
# %C 는 (로컬·원격호스트·포트·사용자) 해시라 ControlPath 길이 제한(~100자)에 안 걸린다.
# 끄려면 HANDOFF_SSH_MUX=0.
if [[ "${HANDOFF_SSH_MUX:-1}" == "1" ]]; then
  _CM_DIR="${HOME}/.ssh/cm"
  mkdir -p "$_CM_DIR" 2>/dev/null && chmod 700 "$_CM_DIR" 2>/dev/null || true
  _MUX="-o ControlMaster=auto -o ControlPath=${_CM_DIR}/%C -o ControlPersist=300"
else
  _MUX=""
fi
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new $_MUX"

# rsync 압축(-z). 기본은 켜짐 -- 소스·설정·parquet 같은 평문에는 이득이 크다.
# **이미 압축된 파일을 보낼 때는 꺼야 한다.** 2026-09-22 실측: 수집기가 시각 로테이션에서
# 이미 gzip 한 파일을 -z 로 다시 압축하면 bookticker 1.0%, depthdiff **-0.0%**(오히려 커짐)
# 이고 CPU 만 쓴다. 압축은 원천에서 한 번 하는 게 맞고, 전송 때 또 할 이유가 없다.
#   HANDOFF_RSYNC_Z=0 bash scripts/ops/handoff.sh push server data/live/orderflow/...
_Z=$([[ "${HANDOFF_RSYNC_Z:-1}" == "0" ]] && echo "" || echo "z")

usage() {
  cat >&2 <<EOF
Usage:
  handoff.sh push   <host> <path> [path...]
  handoff.sh pull   <host> <path> [path...]
  handoff.sh launch <host> <job_name> [--sync <path> [path...]] -- <command...>
  handoff.sh stop   <host> <job_name>
  handoff.sh status <host> [job_name]
  handoff.sh logs   <host> <job_name> [-f]

Hosts: ${!HOSTS[*]}
Paths are relative to the repo root.
EOF
  exit 1
}

resolve_host() {
  local h="$1"
  [[ -n "${HOSTS[$h]:-}" ]] || { echo "unknown host '$h' (known: ${!HOSTS[*]})" >&2; exit 1; }
  IFS='|' read -r CONN REPO CONDA_BASE CONDA_ENV <<< "${HOSTS[$h]}"
  IFS=':' read -r USERHOST PORT <<< "$CONN"
}

remote_ssh() {
  ssh -p "$PORT" $SSH_OPTS "$USERHOST" "$@"
}

do_push() {
  local host="$1"; shift
  resolve_host "$host"
  for p in "$@"; do
    remote_ssh "mkdir -p '$REPO/$(dirname "$p")'"
    rsync -av$_Z --progress -e "ssh -p $PORT $SSH_OPTS" "$LOCAL_REPO/$p" "$USERHOST:$REPO/$(dirname "$p")/"
  done
}

do_pull() {
  local host="$1"; shift
  resolve_host "$host"
  for p in "$@"; do
    mkdir -p "$LOCAL_REPO/$(dirname "$p")"
    rsync -av$_Z --progress -e "ssh -p $PORT $SSH_OPTS" "$USERHOST:$REPO/$p" "$LOCAL_REPO/$(dirname "$p")/"
  done
}

do_launch() {
  local host="$1"; shift
  local job="$1"; shift
  resolve_host "$host"

  local sync_paths=()
  if [[ "${1:-}" == "--sync" ]]; then
    shift
    while [[ "${1:-}" != "--" && -n "${1:-}" ]]; do sync_paths+=("$1"); shift; done
  fi
  [[ "${1:-}" == "--" ]] && shift
  [[ $# -gt 0 ]] || { echo "no command given after --" >&2; exit 1; }

  if [[ ${#sync_paths[@]} -gt 0 ]]; then
    do_push "$host" "${sync_paths[@]}"
    resolve_host "$host"
  fi

  local jdir="$REPO/$JOBS_DIR/$job"

  # Build the conda-activation runner locally (avoids embedding the job
  # command inside a nested ssh/bash -c quoting chain) and ship it as a file.
  # It writes its own $$ to the pidfile as its first action, before doing
  # anything else. Reason: nohup/setsid can fork an extra time internally
  # (setsid() fails if the caller is already a process group leader, so the
  # `setsid` utility then forks a child to hold the new session instead) -
  # when that happens, the PID bash's own "$!" captures right after
  # backgrounding is that short-lived intermediate process, not the one that
  # ends up actually running long-term, so a pidfile written from the
  # launcher side can silently point at the wrong (already-exited) PID and
  # `stop` then kills nothing real. Every step after this line only ever
  # execs (conda activate doesn't fork; the final `exec "$@"` replaces this
  # script in place) - exec always preserves the PID, so capturing it here,
  # first, is the only point that's guaranteed to match the final process.
  local tmpdir; tmpdir="$(mktemp -d)"
  cat > "$tmpdir/run.sh" <<EOF
#!/usr/bin/env bash
echo "\$\$" > "$jdir/pid"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"
exec "\$@"
EOF
  chmod +x "$tmpdir/run.sh"

  remote_ssh "mkdir -p '$jdir'"
  rsync -avz -e "ssh -p $PORT $SSH_OPTS" "$tmpdir/run.sh" "$USERHOST:$jdir/run.sh"
  rm -rf "$tmpdir"

  local quoted="" a
  for a in "$@"; do quoted+=" $(printf '%q' "$a")"; done

  # The session that backgrounds the job tends to hang past the point the
  # job has actually detached and is running (observed even with full
  # redirection + setsid) - cap it and confirm over a fresh connection
  # instead of trusting this call's own output. run.sh (not this command)
  # now owns writing the pidfile - see comment above.
  timeout 6 ssh -p "$PORT" $SSH_OPTS "$USERHOST" \
    "cd '$REPO' && nohup setsid '$jdir/run.sh'$quoted > '$jdir/log' 2>&1 < /dev/null &" \
    || true

  sleep 1
  echo "launched '$job' on $host:"
  do_status "$host" "$job"
}

do_stop() {
  local host="$1" job="$2"
  resolve_host "$host"
  local jdir="$REPO/$JOBS_DIR/$job"
  remote_ssh "
    if [[ -f '$jdir/pid' ]]; then
      pid=\$(cat '$jdir/pid')
      if kill -0 \"\$pid\" 2>/dev/null; then
        kill -TERM \"\$pid\" 2>/dev/null || true
        sleep 2
        kill -0 \"\$pid\" 2>/dev/null && kill -KILL \"\$pid\" 2>/dev/null || true
        echo \"stopped pid \$pid\"
      else
        echo \"pid \$pid not running\"
      fi
    else
      echo \"no pidfile for job '$job' on $host\"
    fi
  "
}

do_status() {
  local host="$1" job="${2:-}"
  resolve_host "$host"
  if [[ -z "$job" ]]; then
    remote_ssh "ls '$REPO/$JOBS_DIR' 2>/dev/null || echo '(no jobs)'"
    return
  fi
  local jdir="$REPO/$JOBS_DIR/$job"
  remote_ssh "
    if [[ -f '$jdir/pid' ]]; then
      pid=\$(cat '$jdir/pid')
      if kill -0 \"\$pid\" 2>/dev/null; then echo \"RUNNING pid=\$pid\"; else echo \"STOPPED (last pid=\$pid)\"; fi
    else
      echo 'no such job: $job'
    fi
    echo '--- last 15 log lines ---'
    tail -n 15 '$jdir/log' 2>/dev/null
  "
}

do_logs() {
  local host="$1" job="$2" follow="${3:-}"
  resolve_host "$host"
  local jdir="$REPO/$JOBS_DIR/$job"
  if [[ "$follow" == "-f" ]]; then
    ssh -p "$PORT" -t -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USERHOST" "tail -n 50 -f '$jdir/log'"
  else
    remote_ssh "tail -n 100 '$jdir/log'"
  fi
}

[[ $# -ge 1 ]] || usage
# ── 폭주 방지 (2026-09-13) ────────────────────────────────────────────────────
# 🔴2026-09-13 사고: 다른 세션이 `until handoff.sh logs ...` 를 **sleep 없이** 돌려
# 28.8시간 동안 초당 2~3회 SSH 를 열었다. 서버가 21시간에 26,353 세션을 받았고(분당 156),
# 세션마다 sshd 포크 + PAM + systemd 스코프 생성·삭제가 일어났다. 21:51:55 에 서버 WSL 의
# 시계가 튀고 DNS 가 죽어 사람이 강제 재부팅했다(리눅스 쪽 OOM·커널오류는 없었다).
#
# 고아 프로세스를 나이로 죽이는 건 위험하다 -- 진짜 워커도 며칠씩 돈다. 대신 **이 파일이
# 이 저장소의 모든 SSH 가 지나는 단일 통로**이므로 여기서 호출 빈도를 막는다. 어떤 에이전트가
# 어떤 루프를 짜든 걸린다.
#
# 잠들지 않고 **거부**한다 -- 잠들면 루프가 조용히 계속되고 버그가 안 보인다.
# 사람이 쓰는 속도(분당 수 회)보다 한참 위라 정상 사용은 안 걸린다.
HANDOFF_MAX_PER_MIN="${HANDOFF_MAX_PER_MIN:-30}"
rate_guard() {
  local stamp="${TMPDIR:-/tmp}/.handoff_calls_$(id -u)"
  local now; now=$(date +%s)
  local recent=0 line
  if [[ -f "$stamp" ]]; then
    # 최근 60초치만 남긴다. 파일이 커지지 않는다.
    awk -v c="$now" '$1 > c - 60' "$stamp" > "$stamp.tmp" 2>/dev/null && mv "$stamp.tmp" "$stamp"
    recent=$(wc -l < "$stamp" 2>/dev/null || echo 0)
  fi
  if (( recent >= HANDOFF_MAX_PER_MIN )); then
    cat >&2 <<MSG
handoff.sh: 최근 60초에 ${recent}회 호출 -- 상한 ${HANDOFF_MAX_PER_MIN}회를 넘었습니다. 거부합니다.

  폴링 루프에 sleep 이 빠졌을 가능성이 큽니다. 2026-09-13 에 같은 실수로 서버가 멈췄습니다
  (28.8시간 동안 분당 156회, 세션 26,353개).
  대기하려면 루프에 'sleep 30' 이상을 넣으세요. 한 번에 오래 기다릴 일이면
  'handoff.sh launch' 로 서버에서 돌리고 결과만 한 번 가져오세요.
  상한을 바꾸려면 HANDOFF_MAX_PER_MIN 환경변수를 쓰세요.
MSG
    exit 3
  fi
  echo "$now" >> "$stamp"
}
rate_guard

cmd="$1"; shift
case "$cmd" in
  push)   [[ $# -ge 2 ]] || usage; do_push "$@" ;;
  pull)   [[ $# -ge 2 ]] || usage; do_pull "$@" ;;
  launch) [[ $# -ge 2 ]] || usage; do_launch "$@" ;;
  stop)   [[ $# -ge 2 ]] || usage; do_stop "$@" ;;
  status) [[ $# -ge 1 ]] || usage; do_status "$@" ;;
  logs)   [[ $# -ge 2 ]] || usage; do_logs "$@" ;;
  *) usage ;;
esac
