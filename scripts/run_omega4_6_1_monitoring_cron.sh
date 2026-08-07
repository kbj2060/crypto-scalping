#!/usr/bin/env bash
# Phase 2 wiring: run both Omega4.6.1 monitoring checks (roadmap items 4-5) periodically via cron.
# Read-only diagnostics -- never touches trading_bot.py or any live state, only appends logs.
set -eo pipefail

cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quant_ai

LOG_DIR="data/ensemble/metrics"
mkdir -p "$LOG_DIR"

{
  echo "=== $(date -Iseconds) omega4_6_1 monitoring run ==="
  python scripts/audit_omega4_6_1_feature_drift_scheduled_20260707.py || echo "FEATURE DRIFT SCRIPT EXIT NONZERO (see status above)"
  python scripts/monitor_omega4_6_1_live_drift_20260707.py || echo "LIVE DRIFT MONITOR SCRIPT EXIT NONZERO"
} >> "$LOG_DIR/omega4_6_1_monitoring_cron.log" 2>&1
