#!/usr/bin/env bash
# Control the supervised live processes (trading bot, Tau1 shadow tracker, operations watchdog).
#
# Usage:
#   botctl.sh start    # launch all supervisors (idempotent -- flock prevents duplicates)
#   botctl.sh stop      # stop both supervisors AND the processes they manage
#   botctl.sh status     # show whether each supervisor + managed process is running
#   botctl.sh restart    # stop then start
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPS="$ROOT/scripts/ops"
mkdir -p "$ROOT/logs/supervisor"

# Must match the interpreter the supervisor_*.sh scripts actually launch (they honor
# PYTHON_BIN too) -- otherwise stop()'s pkill can miss a process started under a
# custom PYTHON_BIN and leave it running while status/botctl think it's dead.
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PY_PATTERN="$(printf '%s' "$PY" | sed 's/[.[\*^$()+?{|]/\\&/g')"

supervisor_process_name() {
  if [[ "$1" == "trading_bot" ]]; then
    printf '%s' 'trading_bot.py'
  else
    printf '%s' "$1"
  fi
}

# 2026-08-03: trading_bot/tau1_shadow/ops_watchdog migrated to systemd
# (scripts/ops/systemd/*.service). These two helpers detect that so start/stop/status
# don't fight the systemd-managed process -- launching a bash supervisor on top of an
# already-systemd-managed process would run two live trading_bot.py instances at once.
systemd_unit_for() {
  case "$1" in
    trading_bot) printf '%s' 'trading-bot.service' ;;
    tau1_shadow) printf '%s' 'tau1-shadow.service' ;;
    ops_watchdog) printf '%s' 'ops-watchdog.service' ;;
  esac
}

systemd_managed() {
  command -v systemctl >/dev/null 2>&1 || return 1
  systemctl list-unit-files "$(systemd_unit_for "$1")" --no-legend 2>/dev/null | grep -q .
}

start() {
  # _supervise.sh holds its own flock and exits immediately if one is already running,
  # so launching unconditionally here is safe (no duplicate supervisors can result).
  for name in trading_bot tau1_shadow ops_watchdog btc_multislot_shadow; do
    if systemd_managed "$name"; then
      unit="$(systemd_unit_for "$name")"
      echo "$name: managed by systemd ($unit) -- use 'sudo systemctl start $unit' instead, skipping bash supervisor"
      continue
    fi
    supervisor_name="$(supervisor_process_name "$name")"
    if pgrep -f "[s]cripts/ops/_supervise.sh ${supervisor_name} " >/dev/null; then
      echo "$name: supervisor already running, skipping"
      continue
    fi
    nohup setsid "$OPS/supervisor_${name}.sh" >/dev/null 2>&1 < /dev/null &
    disown
    echo "$name: supervisor launched"
  done
}

stop() {
  for name in trading_bot tau1_shadow ops_watchdog btc_multislot_shadow; do
    if systemd_managed "$name"; then
      unit="$(systemd_unit_for "$name")"
      echo "$name: managed by systemd ($unit) -- use 'sudo systemctl stop $unit' (killing it here would just get restarted by systemd)"
      continue
    fi
    supervisor_name="$(supervisor_process_name "$name")"
    pids=$(pgrep -f "[s]cripts/ops/_supervise.sh ${supervisor_name} " || true)
    if [[ -n "$pids" ]]; then
      echo "$name: stopping supervisor pids=$pids"
      kill $pids 2>/dev/null
    fi
  done
  sleep 1
  # ^-anchored and no leading ".*" on the alternatives: _supervise.sh's own argv embeds
  # this exact "<python> -u ...<script>.py" substring (it's passed the child command as
  # its own arguments), so an unanchored pattern matches the supervisor bash process too
  # and kills it alongside the child -- confirmed live 2026-08-03 (ops_watchdog supervisor
  # died from a pkill meant only for its child process).
  if ! systemd_managed trading_bot; then
    pkill -f "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*trading_bot.py" 2>/dev/null && echo "trading_bot.py: stopped" || echo "trading_bot.py: not running"
  fi
  if ! systemd_managed tau1_shadow; then
    pkill -f "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*live_sigma6_regime_tiebreak_shadow" 2>/dev/null && echo "tau1_shadow: stopped" || echo "tau1_shadow: not running"
  fi
  if ! systemd_managed ops_watchdog; then
    pkill -f "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*ops_watchdog.py" 2>/dev/null && echo "ops_watchdog: stopped" || echo "ops_watchdog: not running"
  fi
  # btc_multislot_shadow is bash-supervised only (not migrated to systemd), so it
  # has no systemd_managed guard to skip -- always safe to pkill its child here.
  pkill -f "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*run_btc_multislot_shadow_loop_20260807.py" 2>/dev/null && echo "btc_multislot_shadow: stopped" || echo "btc_multislot_shadow: not running"
}

status() {
  for name in trading_bot tau1_shadow ops_watchdog btc_multislot_shadow; do
    if systemd_managed "$name"; then
      unit="$(systemd_unit_for "$name")"
      # is-active/is-enabled print a real value (inactive/disabled/failed) AND exit
      # non-zero for those non-"active"/"enabled" states -- a `|| echo unknown`
      # fallback fires on that exit code regardless, appending a second bogus line
      # after the real one. Fall back on emptiness instead of exit status.
      state="$(systemctl is-active "$unit" 2>/dev/null)"; state="${state:-unknown}"
      enabled="$(systemctl is-enabled "$unit" 2>/dev/null)"; enabled="${enabled:-unknown}"
      echo "$name: systemd $unit = $state (enabled=$enabled)"
      continue
    fi
    supervisor_name="$(supervisor_process_name "$name")"
    sup_pid=$(pgrep -f "[s]cripts/ops/_supervise.sh ${supervisor_name} " | head -1 || true)
    if [[ -n "$sup_pid" ]]; then
      echo "$name supervisor: RUNNING (pid $sup_pid)"
    else
      echo "$name supervisor: STOPPED"
    fi
  done
  echo "---"
  pgrep -af "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*trading_bot.py" || echo "trading_bot.py: not running"
  pgrep -af "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*live_sigma6_regime_tiebreak_shadow" || echo "tau1_shadow: not running"
  pgrep -af "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*ops_watchdog.py" || echo "ops_watchdog: not running"
  pgrep -af "^($PY_PATTERN|[^ ]*venv/bin/python|[^ ]*miniconda3/envs/quant_ai/bin/python) -u .*run_btc_multislot_shadow_loop_20260807.py" || echo "btc_multislot_shadow: not running"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) stop; sleep 2; start ;;
  *) echo "usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
