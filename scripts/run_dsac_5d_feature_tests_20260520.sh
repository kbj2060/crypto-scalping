#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN="$ROOT/scripts/alpha5_dsac_single_router5_density_20260520.py"
BASE_ROUTER="$ROOT/tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT/tmp/causal_regen_20260516/dsac_5d_feature_tests_20260520}"
EPISODES="${EPISODES:-20}"
DSAC_LR_ACTOR="${DSAC_LR_ACTOR:-3e-4}"
DSAC_LR_CRITIC="${DSAC_LR_CRITIC:-3e-4}"
DSAC_LR_ALPHA="${DSAC_LR_ALPHA:-3e-4}"

mkdir -p "$SWEEP_ROOT"

run_variant() {
  local name="$1"
  local router_dir="$2"
  shift 2
  local out="$SWEEP_ROOT/$name"
  mkdir -p "$out"
  echo "[$(date -Is)] START $name episodes=$EPISODES router=$router_dir env=$*" | tee -a "$SWEEP_ROOT/sweep_master.log"
  (
    cd "$ROOT" || exit 1
    env PYTHONUNBUFFERED=1 \
      DSAC_LR_ACTOR="$DSAC_LR_ACTOR" \
      DSAC_LR_CRITIC="$DSAC_LR_CRITIC" \
      DSAC_LR_ALPHA="$DSAC_LR_ALPHA" \
      "$@" "$PY" "$RUN" \
      --fresh-start \
      --skip-score \
      --device auto \
      --episodes "$EPISODES" \
      --router-dir "$router_dir" \
      --out-dir "$out"
  ) > "$out/master.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END $name code=$code" | tee -a "$SWEEP_ROOT/sweep_master.log"
}

run_variant compact_no_pca "$BASE_ROUTER" DSAC_ALL_FEATURES_ENABLE=0 DSAC_EXTRA_PCA_ENABLE=0
run_variant pca16 "$BASE_ROUTER" DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=16
run_variant pca32_base "$BASE_ROUTER" DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=32
run_variant pca32_zero_funding "$SWEEP_ROOT/router_zero_funding" DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=32
run_variant pca32_zero_catboost_major "$SWEEP_ROOT/router_zero_catboost_major" DSAC_ALL_FEATURES_ENABLE=1 DSAC_EXTRA_PCA_ENABLE=1 DSAC_EXTRA_PCA_COMPONENTS=32
