#!/usr/bin/env bash
set -euo pipefail
cd /home/llewyn/crypto-scalping
VENV=./venv/bin/python3

SOL_ZIG_DIR=tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_exit30k_20260722
SOL_H48_DIR=tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_h48qual_exit30k_20260722
BTC_H48_DIR=tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_exit30k_20260722
BTC_ZIG_DIR=tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_zig075_exit30k_20260722

echo "=== [1/4] SOL zig075 exit30k risk sidecar (q070) ==="
$VENV scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py \
  --baseline-bundle $SOL_ZIG_DIR/true_3head_tabm_bundle.pt \
  --precomputed-prediction-dir $SOL_ZIG_DIR \
  --precomputed-prediction-tag q070 \
  --quality-threshold 0.70 \
  --out-suffix adaptive_squeeze_exit30k_q070_20260722 \
  --device cuda

echo "=== [2/4] SOL h48qual exit30k risk sidecar (q045) ==="
$VENV scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py \
  --baseline-bundle $SOL_H48_DIR/true_3head_tabm_bundle.pt \
  --precomputed-prediction-dir $SOL_H48_DIR \
  --precomputed-prediction-tag q045 \
  --quality-threshold 0.45 \
  --out-suffix adaptive_squeeze_h48qual_exit30k_q045_20260722 \
  --device cuda

echo "=== [3/4] BTC h48qual exit30k risk sidecar (q055) ==="
$VENV scripts/train_eval_omega4_2_risk_sidecar_btc_20260708.py \
  --baseline-bundle $BTC_H48_DIR/true_3head_tabm_bundle.pt \
  --precomputed-prediction-dir $BTC_H48_DIR \
  --precomputed-prediction-tag q055 \
  --quality-threshold 0.55 \
  --out-suffix h48qual_exit30k_q055_20260722 \
  --device cuda

echo "=== [4/4] BTC zig075 exit30k risk sidecar (q065) ==="
$VENV scripts/train_eval_omega4_2_risk_sidecar_btc_20260708.py \
  --baseline-bundle $BTC_ZIG_DIR/true_3head_tabm_bundle.pt \
  --precomputed-prediction-dir $BTC_ZIG_DIR \
  --precomputed-prediction-tag q065 \
  --quality-threshold 0.65 \
  --out-suffix zig075_exit30k_q065_20260722 \
  --device cuda

echo "=== ALL SIDECARS DONE ==="
