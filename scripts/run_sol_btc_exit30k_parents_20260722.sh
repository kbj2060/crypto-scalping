#!/usr/bin/env bash
# Sequential exit30k parent retrains for SOL/BTC, matching ETH's h48qual/zig075 recipe
# (max-exit-samples 30000, same quality-mode/quality-label-dir conventions already used).
set -euo pipefail
cd /home/llewyn/crypto-scalping
VENV=./venv/bin/python3

echo "=== [1/4] SOL zig075 exit30k (adaptive_squeeze features) ==="
$VENV scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py \
  --quality-mode same_as_direction \
  --max-exit-samples 30000 \
  --out-suffix adaptive_squeeze_exit30k_20260722 \
  --device cuda

echo "=== [2/4] SOL h48qual exit30k (adaptive_squeeze features) ==="
$VENV scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py \
  --quality-mode quality_label_action \
  --quality-label-dir tmp/causal_regen_20260516/sol_h48_conservative_padded_to_zigzag_timestamps_20260707 \
  --max-exit-samples 30000 \
  --out-suffix adaptive_squeeze_h48qual_exit30k_20260722 \
  --device cuda

echo "=== [3/4] BTC h48qual exit30k ==="
$VENV scripts/train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708.py \
  --quality-mode quality_label_action \
  --quality-label-dir tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708 \
  --max-exit-samples 30000 \
  --out-suffix h48qual_exit30k_20260722 \
  --device cuda

echo "=== [4/4] BTC zig075 exit30k (new) ==="
$VENV scripts/train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708.py \
  --quality-mode same_as_direction \
  --max-exit-samples 30000 \
  --out-suffix zig075_exit30k_20260722 \
  --device cuda

echo "=== ALL DONE ==="
