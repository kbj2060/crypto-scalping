#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/llewyn/crypto-scalping"
SWEEP_ROOT="$ROOT/tmp/causal_regen_20260516/dsac_feature_sweep_20260520"
PY="$ROOT/venv/bin/python"
RUN="$ROOT/scripts/alpha5_dsac_single_router5_density_20260520.py"

mkdir -p "$SWEEP_ROOT"

run_variant() {
  local name="$1"
  shift
  local out="$SWEEP_ROOT/$name"
  mkdir -p "$out"
  echo "[$(date -Is)] START $name" | tee -a "$SWEEP_ROOT/sweep_master.log"
  (
    cd "$ROOT" || exit 1
    env "$@" "$PY" "$RUN" --fresh-start --skip-score --episodes 35 --out-dir "$out"
  ) > "$out/master.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END $name code=$code" | tee -a "$SWEEP_ROOT/sweep_master.log"
}

run_variant compact54 DSAC_ALL_FEATURES_ENABLE=0 DSAC_EXTRA_PCA_ENABLE=0
run_variant raw111 DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=0
run_variant pca16 DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=16
run_variant pca32 DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=32
run_variant pca48 DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=48
