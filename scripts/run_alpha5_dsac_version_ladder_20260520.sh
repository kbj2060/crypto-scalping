#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN="$ROOT/scripts/alpha5_dsac_single_router5_density_20260520.py"
ROUTER="$ROOT/tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519"
LADDER_ROOT="${LADDER_ROOT:-$ROOT/tmp/causal_regen_20260516/alpha5_dsac_version_ladder_20260520}"
EPISODES="${EPISODES:-70}"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/tmp/causal_regen_20260516/dsac_5d_soft_exit_conflict_diag_20260520/compact_no_pca}"
BASELINE_PID="${BASELINE_PID:-}"

mkdir -p "$LADDER_ROOT"

if [ -n "$BASELINE_PID" ]; then
  echo "[$(date -Is)] WAIT baseline pid=$BASELINE_PID" | tee -a "$LADDER_ROOT/ladder_master.log"
  while kill -0 "$BASELINE_PID" 2>/dev/null; do
    sleep 60
  done
fi

if [ -d "$BASELINE_DIR" ] && [ ! -e "$LADDER_ROOT/alpha5_00_soft_exit_hard_bucket" ]; then
  ln -s "$BASELINE_DIR" "$LADDER_ROOT/alpha5_00_soft_exit_hard_bucket"
fi

run_case() {
  local name="$1"
  shift
  local out="$LADDER_ROOT/$name"
  mkdir -p "$out/conflict_diag"
  echo "[$(date -Is)] START $name episodes=$EPISODES env=$*" | tee -a "$LADDER_ROOT/ladder_master.log"
  (
    cd "$ROOT" || exit 1
    env PYTHONUNBUFFERED=1 \
      DSAC_ALL_FEATURES_ENABLE=0 \
      DSAC_EXTRA_PCA_ENABLE=0 \
      DSAC_LR_ACTOR="${DSAC_LR_ACTOR:-9e-4}" \
      DSAC_LR_CRITIC="${DSAC_LR_CRITIC:-9e-4}" \
      DSAC_LR_ALPHA="${DSAC_LR_ALPHA:-3e-4}" \
      DSAC_CONFLICT_DIAG_ENABLE=1 \
      DSAC_CONFLICT_GAP_TH="${DSAC_CONFLICT_GAP_TH:-0.15}" \
      DSAC_CONFLICT_DIAG_DIR="$out/conflict_diag" \
      "$@" \
      "$PY" "$RUN" \
      --fresh-start \
      --skip-score \
      --device auto \
      --episodes "$EPISODES" \
      --router-dir "$ROUTER" \
      --out-dir "$out"
  ) > "$out/master.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END $name code=$code" | tee -a "$LADDER_ROOT/ladder_master.log"
  "$PY" "$ROOT/scripts/summarize_dsac_5d_feature_tests_20260520.py" "$LADDER_ROOT" > "$LADDER_ROOT/summary_latest.json" 2>/dev/null || true
}

run_case alpha5_01_soft_exit_market_state_hard_bucket DSAC_V2_CONTINUOUS_RISK_ENABLE=0
run_case alpha5_02_soft_exit_market_state_continuous_risk DSAC_V2_CONTINUOUS_RISK_ENABLE=1
