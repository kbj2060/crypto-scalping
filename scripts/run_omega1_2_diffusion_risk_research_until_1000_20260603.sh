#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
CONDA="/home/llewyn/miniconda3/bin/conda"
LOG_DIR="$ROOT/logs"
SCRIPT="$ROOT/scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py"
mkdir -p "$LOG_DIR"

run_one() {
  local suffix="$1"
  shift
  local out_dir="$ROOT/tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_diffusion_risk_20260603_${suffix}"
  local log="$LOG_DIR/omega1_2_diffusion_risk_${suffix}.log"
  if [[ -f "$out_dir/report.json" ]]; then
    echo "[skip] $suffix already has report.json" | tee -a "$log"
    return 0
  fi
  echo "[run] $suffix $(date -Is)" | tee -a "$log"
  "$CONDA" run -n quant_ai python "$SCRIPT" \
    --steps 1800 \
    --scorer-steps 900 \
    --samples-per-row 32 \
    --keep-top-k 4 \
    --rerank-samples 32 \
    --diffusion-steps 24 \
    --batch-size 1024 \
    --device auto \
    --out-suffix "$suffix" \
    "$@" 2>&1 | tee -a "$log"
}

cd "$ROOT"
run_one "penalty_anchor_delta20_p005" --risk-bounds-preset anchor_delta20 --rerank-exposure-penalty 0.05
run_one "penalty_anchor_delta20_p010" --risk-bounds-preset anchor_delta20 --rerank-exposure-penalty 0.10
run_one "penalty_anchor_delta35_p005" --risk-bounds-preset anchor_delta35 --rerank-exposure-penalty 0.05
run_one "penalty_anchor_delta35_p010" --risk-bounds-preset anchor_delta35 --rerank-exposure-penalty 0.10
run_one "top1_anchor_delta20" --risk-bounds-preset anchor_delta20 --keep-top-k 1
run_one "top8_anchor_delta20" --risk-bounds-preset anchor_delta20 --keep-top-k 8
run_one "top1_anchor_delta35" --risk-bounds-preset anchor_delta35 --keep-top-k 1
run_one "top8_anchor_delta35" --risk-bounds-preset anchor_delta35 --keep-top-k 8
