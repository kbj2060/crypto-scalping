#!/usr/bin/env bash
# Server-only cron job (venv/bin/python -> /usr/bin/python3 on the server). Dev has no
# venv/ and doesn't run this collector -- don't hardcode a dev python path here, a local
# uncommitted edit doing that got swept into a handoff.sh push on 2026-08-17 and silently
# broke the server's daily 1am cron until 2026-08-19 (see run_deribit_gex_collector_server.sh
# for the same failure mode with a different fix).
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/research"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/altdata_collector_cron.log"

venv/bin/python scripts/run_f4c_altdata_collector.py 2>&1 | tee -a "$LOG_FILE"
