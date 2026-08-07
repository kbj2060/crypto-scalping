#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/venv/bin/python}"
LIVE_DIR="$ROOT/data/live"
SUPERVISOR_PID="$LIVE_DIR/micro_scalp_shadow_supervisor.pid"
SUPERVISOR_LOG="$LIVE_DIR/micro_scalp_shadow_supervisor.log"
SUPERVISOR_ERR="$LIVE_DIR/micro_scalp_shadow_supervisor.err"
RESTART_DELAY="${MICRO_SCALP_SHADOW_RESTART_DELAY:-10}"

mkdir -p "$LIVE_DIR"
echo "$$" >"$SUPERVISOR_PID"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$SUPERVISOR_LOG"
}

err() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$SUPERVISOR_ERR"
}

process_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  local state
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$state" != "Z" ]]
}

find_service_pid() {
  local script_name="$1"
  ps -eo pid=,stat=,comm=,args= | awk -v needle="$script_name" '
    $3 ~ /^python([0-9.]+)?$/ {
      args = ""
      for (i = 4; i <= NF; i++) args = args " " $i
      if ($2 !~ /^Z/ && index(args, needle) && args ~ /[[:space:]]serve([[:space:]]|$)/) {
        print $1
        exit
      }
    }
  '
}

supervise_service() {
  local name="$1"
  local script_name="$2"
  local pid_file="$LIVE_DIR/micro_scalp_shadow_${name}.pid"
  local stdout_log="$LIVE_DIR/micro_scalp_shadow_${name}.log"
  local stderr_log="$LIVE_DIR/micro_scalp_shadow_${name}.err"
  local child=""

  trap 'if [[ -n "$child" ]] && process_alive "$child"; then kill "$child" 2>/dev/null || true; fi; exit 0' INT TERM

  while true; do
    child="$(find_service_pid "$script_name" || true)"
    if [[ -n "$child" ]]; then
      log "service=$name attached pid=$child"
    else
      log "service=$name starting"
      case "$name" in
        eth_v4)
          "$PY" -u "$ROOT/scripts/$script_name" serve \
            --device cpu --interval-seconds 300 --max-stream-age-minutes 15 \
            >>"$stdout_log" 2>>"$stderr_log" &
          ;;
        btc_sol|reuse)
          "$PY" -u "$ROOT/scripts/$script_name" serve \
            --device cpu --interval-seconds 300 --max-stream-age-minutes 15 \
            >>"$stdout_log" 2>>"$stderr_log" &
          ;;
        hexa)
          "$PY" -u "$ROOT/scripts/$script_name" serve \
            --interval-seconds 60 --max-stream-age-minutes 5 \
            >>"$stdout_log" 2>>"$stderr_log" &
          ;;
        *)
          err "service=$name invalid configuration"
          return 1
          ;;
      esac
      child="$!"
      log "service=$name started pid=$child"
    fi
    echo "$child" >"$pid_file"

    while process_alive "$child"; do
      sleep 10
    done

    err "service=$name pid=$child exited; restarting in ${RESTART_DELAY}s"
    rm -f "$pid_file"
    sleep "$RESTART_DELAY"
  done
}

workers=()
supervise_service eth_v4 run_eth_micro_scalp_v4_shadow_bot_20260718.py &
workers+=("$!")
supervise_service btc_sol run_btc_sol_micro_scalp_shadow_bot_20260718.py &
workers+=("$!")
supervise_service reuse run_micro_scalp_reuse_shadow_bot_20260718.py &
workers+=("$!")
supervise_service hexa run_hexa_pulse_formula_shadow_20260718.py &
workers+=("$!")

cleanup() {
  trap - INT TERM
  for worker in "${workers[@]}"; do
    kill "$worker" 2>/dev/null || true
  done
  wait "${workers[@]}" 2>/dev/null || true
  rm -f "$SUPERVISOR_PID"
  exit 0
}

trap cleanup INT TERM
log "supervisor started pid=$$ services=eth_v4,btc_sol,reuse,hexa"
wait "${workers[@]}"
cleanup
