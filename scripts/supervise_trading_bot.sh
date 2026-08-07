#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PID="$ROOT/data/live/trading_bot.pid"
LOG="$ROOT/data/live/trading_bot_stdout.log"
ERR="$ROOT/data/live/trading_bot_stderr.log"
SUPERVISOR_LOG="$ROOT/data/live/trading_bot_supervisor.log"
SUPERVISOR_ERR="$ROOT/data/live/trading_bot_supervisor.err"
RESTART_DELAY="${TRADING_BOT_RESTART_DELAY:-10}"
DECISION_SNAPSHOT="${TRADING_BOT_DECISION_SNAPSHOT:-$ROOT/data/live/trading_bot_decision_heartbeat.json}"
DECISION_WATCHDOG_ENABLE="${TRADING_BOT_DECISION_WATCHDOG_ENABLE:-1}"
DECISION_WATCHDOG_MAX_AGE_SEC="${TRADING_BOT_DECISION_WATCHDOG_MAX_AGE_SEC:-900}"
DECISION_WATCHDOG_GRACE_SEC="${TRADING_BOT_DECISION_WATCHDOG_GRACE_SEC:-1800}"
DECISION_WATCHDOG_INTERVAL_SEC="${TRADING_BOT_DECISION_WATCHDOG_INTERVAL_SEC:-60}"

mkdir -p "$ROOT/data/live"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$SUPERVISOR_LOG"
}

err() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$SUPERVISOR_ERR"
}

running_bot_pid() {
  ps -eo pid=,stat=,comm=,args= | awk '
    $3 ~ /^python([0-9.]+)?$/ {
      args = ""
      for (i = 4; i <= NF; i++) args = args " " $i
      if ($2 !~ /^Z/ && args ~ /(^|[[:space:]])([^[:space:]]*\/)?trading_bot\.py([[:space:]]|$)/) {
        print $1
        exit
      }
    }
  '
}

process_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  local state
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$state" != "Z" ]]
}

decision_snapshot_age_sec() {
  if [[ ! -f "$DECISION_SNAPSHOT" ]]; then
    printf ''
    return 0
  fi
  local mtime
  mtime="$(stat -c %Y "$DECISION_SNAPSHOT" 2>/dev/null || true)"
  if [[ -z "$mtime" ]]; then
    printf ''
    return 0
  fi
  printf '%s' "$(($(date +%s) - mtime))"
}

watchdog_check() {
  local pid="$1"
  local started_epoch="$2"
  local now_epoch
  now_epoch="$(date +%s)"
  if [[ "$DECISION_WATCHDOG_ENABLE" != "1" && "$DECISION_WATCHDOG_ENABLE" != "true" ]]; then
    return 0
  fi
  if (( now_epoch - started_epoch < DECISION_WATCHDOG_GRACE_SEC )); then
    return 0
  fi
  local age
  age="$(decision_snapshot_age_sec)"
  if [[ -z "$age" ]]; then
    err "decision_watchdog restarting pid=$pid reason=snapshot_missing path=$DECISION_SNAPSHOT"
    kill "$pid" 2>/dev/null || true
    return 0
  fi
  if (( age > DECISION_WATCHDOG_MAX_AGE_SEC )); then
    err "decision_watchdog restarting pid=$pid reason=snapshot_stale age=${age}s max=${DECISION_WATCHDOG_MAX_AGE_SEC}s path=$DECISION_SNAPSHOT"
    kill "$pid" 2>/dev/null || true
  fi
}

stop_child() {
  if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
    kill "$(cat "$PID")" 2>/dev/null || true
  fi
}

trap 'stop_child; exit 0' INT TERM

cd "$ROOT"

while true; do
  existing="$(running_bot_pid || true)"
  if [[ -n "$existing" ]]; then
    log "trading_bot.py already running pid=$existing"
    echo "$existing" >"$PID"
    existing_started_epoch="$(date +%s)"
    while process_alive "$existing"; do
      watchdog_check "$existing" "$existing_started_epoch"
      sleep "$DECISION_WATCHDOG_INTERVAL_SEC"
    done
    err "trading_bot.py pid=$existing exited; restarting in ${RESTART_DELAY}s"
    rm -f "$PID"
    sleep "$RESTART_DELAY"
    continue
  fi

  rm -f "$PID"
  log "starting trading_bot.py"
  "$PY" -u trading_bot.py >>"$LOG" 2>>"$ERR" &
  child="$!"
  echo "$child" >"$PID"

  child_started_epoch="$(date +%s)"
  while process_alive "$child"; do
    watchdog_check "$child" "$child_started_epoch"
    sleep "$DECISION_WATCHDOG_INTERVAL_SEC"
  done
  wait "$child"
  code="$?"
  rm -f "$PID"
  err "trading_bot.py exited code=$code; restarting in ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done
