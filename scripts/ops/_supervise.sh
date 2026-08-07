#!/usr/bin/env bash
# Generic crash-restart watchdog for a single long-running process. Not called directly --
# see supervisor_trading_bot.sh / supervisor_tau1_shadow.sh.
#
# Usage: _supervise.sh <name> <lockfile> <logfile-prefix> <cmd...>
set -uo pipefail

NAME="$1"; shift
LOCK="$1"; shift
LOG_PREFIX="$1"; shift

# Self-contained: don't rely on the caller (botctl.sh, a systemd unit, cron, ...)
# having pre-created these directories.
mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG_PREFIX")"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Iseconds)] supervisor for $NAME already running (lock held) -- exiting" >&2
  exit 1
fi

RESTART_COUNT=0
WINDOW_START=$(date +%s)

while true; do
  LOG_FILE="${LOG_PREFIX}_$(date +%Y%m%d).log"
  echo "[$(date -Iseconds)] SUPERVISOR starting $NAME (pid will follow)" >> "$LOG_FILE"
  # Keep the supervisor lock in this process only. A child must not retain it
  # after the supervisor itself exits, otherwise a replacement supervisor
  # cannot acquire the lock.
  "$@" 9>&- >> "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  echo "[$(date -Iseconds)] SUPERVISOR $NAME exited code=$EXIT_CODE" >> "$LOG_FILE"

  NOW=$(date +%s)
  if (( NOW - WINDOW_START > 600 )); then
    RESTART_COUNT=0
    WINDOW_START=$NOW
  fi
  RESTART_COUNT=$((RESTART_COUNT + 1))

  if (( RESTART_COUNT >= 5 )); then
    echo "[$(date -Iseconds)] SUPERVISOR $NAME crash-looping ($RESTART_COUNT restarts in <10min) -- backing off 300s" >> "$LOG_FILE"
    sleep 300
  else
    sleep 15
  fi
done
