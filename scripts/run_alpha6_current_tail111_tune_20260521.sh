#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
SCRIPT="$ROOT/scripts/alpha6_catboost_multihead_policy_20260521.py"
OUT_BASE="$ROOT/tmp/causal_regen_20260516/alpha6_catboost_current_tail111_tune_20260521"

mkdir -p "$OUT_BASE"

run_one() {
  local name="$1"
  local iterations="$2"
  local lr="$3"
  local depth="$4"
  local l2="$5"
  local out="$OUT_BASE/$name"
  mkdir -p "$out"
  echo "[alpha6-tune] start $name iterations=$iterations lr=$lr depth=$depth l2=$l2"
  env PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" \
    --variant current_tail111 \
    --iterations "$iterations" \
    --learning-rate "$lr" \
    --depth "$depth" \
    --l2-leaf-reg "$l2" \
    --thresholds 70 \
    --out-dir "$out" \
    2>&1 | tee "$out/launcher.log"
  echo "[alpha6-tune] done $name"
}

run_one "base_d6_lr045_l2_6" 700 0.045 6 6.0
run_one "conservative_d5_lr030_l2_10" 800 0.030 5 10.0
run_one "deeper_d7_lr035_l2_8" 900 0.035 7 8.0
run_one "shallow_d4_lr055_l2_5" 650 0.055 4 5.0
run_one "low_lr_d6_lr025_l2_12" 1000 0.025 6 12.0
