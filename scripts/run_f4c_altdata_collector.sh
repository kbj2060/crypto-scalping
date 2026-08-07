#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/research"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/altdata_collector_cron.log"

venv/bin/python scripts/run_f4c_altdata_collector.py 2>&1 | tee -a "$LOG_FILE"
