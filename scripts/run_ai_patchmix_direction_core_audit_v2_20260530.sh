#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN_DIR="$ROOT/tmp/causal_regen_20260516/ai_patchmix_direction_core_audit_v2_20260530_full"

mkdir -p "$RUN_DIR"

COMMON_ARGS=(
  --context-length 512
  --stride 12
  --batch-size 192
  --iterations 700
  --task-type GPU
)

"$PY" "$ROOT/scripts/build_ai_patchmix_direction_core_20260530.py" \
  --train-csv "$ROOT/data/splits/year_oos/training_features_2024.csv" \
  --score-csv "$ROOT/data/splits/year_oos/training_features_2025.csv" \
  --out-dir "$RUN_DIR/fit2024_score2025" \
  --out-csv "$RUN_DIR/fit2024_score2025/ai_patchmix_direction_core_audit_v2_2025.csv" \
  "${COMMON_ARGS[@]}"

"$PY" "$ROOT/scripts/build_ai_patchmix_direction_core_20260530.py" \
  --train-csv "$ROOT/data/splits/year_oos/training_features_2025.csv" \
  --score-csv "$ROOT/data/splits/year_oos/training_features_2026_rebuilt.csv" \
  --out-dir "$RUN_DIR/fit2025_score2026" \
  --out-csv "$RUN_DIR/fit2025_score2026/ai_patchmix_direction_core_audit_v2_2026.csv" \
  "${COMMON_ARGS[@]}"
