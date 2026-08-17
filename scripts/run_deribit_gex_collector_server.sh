#!/usr/bin/env bash
# Server-side wrapper for scripts/collect_deribit_option_gex_20260815.py.
# The dev wrapper (run_deribit_gex_collector.sh) hardcodes dev's python path, which doesn't
# exist on the server -- kept as a separate file rather than parametrizing the dev one, since
# the two machines' conda bases differ (handoff.hosts.conf) and this avoids touching a working
# dev cron job.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/research"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deribit_gex_collector_cron.log"

/home/llewyn/miniconda3/envs/quant_ai/bin/python scripts/collect_deribit_option_gex_20260815.py 2>&1 | tee -a "$LOG_FILE"
