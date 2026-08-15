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
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new"

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
    rsync -avz --progress -e "ssh -p $PORT $SSH_OPTS" "$LOCAL_REPO/$p" "$USERHOST:$REPO/$(dirname "$p")/"
  done
}

do_pull() {
  local host="$1"; shift
  resolve_host "$host"
  for p in "$@"; do
    mkdir -p "$LOCAL_REPO/$(dirname "$p")"
    rsync -avz --progress -e "ssh -p $PORT $SSH_OPTS" "$USERHOST:$REPO/$p" "$LOCAL_REPO/$(dirname "$p")/"
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
