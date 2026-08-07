#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/live/monitoring"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/parity_drift_monitor_cron.log"

venv/bin/python scripts/run_live_parity_drift_monitor.py 2>&1 | tee -a "$LOG_FILE"
