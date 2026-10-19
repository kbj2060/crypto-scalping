#!/usr/bin/env bash
# Control the operations watchdog (the only process left under this tool).
#
# Usage:
#   botctl.sh start|stop|restart   # prints the systemctl command to run (does not act, as before)
#   botctl.sh status               # systemd state + running process
#
# 2026-10-19: trading_bot / tau1_shadow / btc_multislot_shadow retired and removed; their
# bash supervisors are gone. ops_watchdog runs under systemd (ops-watchdog.service) since
# 2026-08-03, so the old bash-supervisor fallback is gone too.
set -uo pipefail
UNIT=ops-watchdog.service

status() {
  # is-active/is-enabled exit non-zero for inactive/disabled, so fall back on emptiness, not exit status.
  state="$(systemctl is-active "$UNIT" 2>/dev/null)"; state="${state:-unknown}"
  enabled="$(systemctl is-enabled "$UNIT" 2>/dev/null)"; enabled="${enabled:-unknown}"
  echo "ops_watchdog: systemd $UNIT = $state (enabled=$enabled)"
  echo "---"
  pgrep -af "[o]ps_watchdog.py" || echo "ops_watchdog: not running"
}

case "${1:-}" in
  start|stop|restart) echo "ops_watchdog: managed by systemd ($UNIT) -- use 'sudo systemctl $1 $UNIT'" ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
