#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/research"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deribit_gex_collector_cron.log"

/home/kbj20/anaconda3/envs/quant_ai/bin/python scripts/collect_deribit_option_gex_20260815.py 2>&1 | tee -a "$LOG_FILE"
