#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN="$ROOT/scripts/train_dsac_feature_variant_20260521.py"
OUT_ROOT="${OUT_ROOT:-$ROOT/tmp/causal_regen_20260516/dsac_feature_screen_regime_fixed_20260521}"
INPUT_CSV="${INPUT_CSV:-$ROOT/tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv}"
SPEC_DIR="${SPEC_DIR:-$ROOT/tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521}"
EPISODES="${EPISODES:-15}"
DEVICE="${DEVICE:-auto}"
SUMMARIZE="$ROOT/scripts/summarize_dsac_feature_screen_runs_20260521.py"

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(
    current_pca32_all111
    stable48_plus_clean4_pred_tail
    stable48_plus_family_pca_regime
    stable48_plus_ai_tail
  )
fi

mkdir -p "$OUT_ROOT"

for variant in "${VARIANTS[@]}"; do
  VARIANT_DIR="$OUT_ROOT/$variant"
  rm -rf "$VARIANT_DIR"
  mkdir -p "$VARIANT_DIR"
  echo "[$(date -Is)] START variant=$variant episodes=$EPISODES" | tee -a "$OUT_ROOT/batch.log"
  (
    cd "$ROOT"
    env PYTHONUNBUFFERED=1 \
      "$PY" "$RUN" \
      --variant "$variant" \
      --spec-dir "$SPEC_DIR" \
      --input-csv "$INPUT_CSV" \
      --out-root "$OUT_ROOT" \
      --episodes "$EPISODES" \
      --fresh-start \
      --device "$DEVICE"
  ) 2>&1 | tee "$VARIANT_DIR/launcher.log" "$OUT_ROOT/${variant}_batch.log"
  python3 "$SUMMARIZE" --runs-dir "$OUT_ROOT" --episode-cutoff "$EPISODES" | tee -a "$OUT_ROOT/batch.log"
  echo "[$(date -Is)] END variant=$variant" | tee -a "$OUT_ROOT/batch.log"
done
