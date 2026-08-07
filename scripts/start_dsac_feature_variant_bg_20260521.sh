#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN="$ROOT/scripts/train_dsac_feature_variant_20260521.py"
OUT_ROOT="${OUT_ROOT:-$ROOT/tmp/causal_regen_20260516/dsac_feature_screen_20260521}"

if [ $# -lt 1 ]; then
  echo "usage: $0 <variant> [episodes] [device]" >&2
  exit 2
fi

VARIANT="$1"
EPISODES="${2:-15}"
DEVICE="${3:-auto}"
OUT_DIR="$OUT_ROOT/$VARIANT"
LOG_PATH="$OUT_DIR/nohup.out"
PID_PATH="$OUT_DIR/bg.pid"

mkdir -p "$OUT_DIR"

nohup /bin/bash -lc "
  echo \"\$(date -Is) wrapper-start variant=$VARIANT episodes=$EPISODES device=$DEVICE\"
  exec \"$PY\" \"$RUN\" --variant \"$VARIANT\" --episodes \"$EPISODES\" --fresh-start --device \"$DEVICE\"
" >>"$LOG_PATH" 2>&1 < /dev/null &

echo $! | tee "$PID_PATH"
