#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="data/ensemble/metrics"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/weekly_param_update.log"

python scripts/weekly_update_ensemble_params.py --notify-telegram "$@" | tee -a "$LOG_FILE"
