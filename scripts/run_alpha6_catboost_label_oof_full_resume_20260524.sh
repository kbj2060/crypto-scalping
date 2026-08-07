#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
OUT_ROOT="$ROOT/tmp/causal_regen_20260516/alpha6_representation_oof_full_20260524/label_oof_full_resume"
REP_FILE="$ROOT/tmp/causal_regen_20260516/alpha6_representation_oof_full_20260524/representation/alpha6_representation_oof_features.parquet"

mkdir -p "$OUT_ROOT"
cd "$ROOT"

exec "$PY" -u scripts/train_alpha6_catboost_label_sweep_oof_full_20260524.py \
  --out-root "$OUT_ROOT" \
  --representation-feature-file "$REP_FILE" \
  --candidates current_quality:bucket5,short_horizon_robust:horizon_reg,high_precision_robust:bucket5,perturbation_robust:bucket5,adverse_conformal:bucket5,sam_conformal:bucket5,ts2vec_ood:bucket5,cost_beta_neutral:bucket5,mamba_regime_filter:bucket5,timegrad_mc:bucket5,timellm_uncertainty:bucket5 \
  --folds 2 \
  --purge-bars 96 \
  --task-type GPU \
  --learning-rate 0.035 \
  --exit-learning-rate 0.028 \
  --l2-leaf-reg 9.0 \
  --oof-iterations 420 \
  --oof-exit-iterations 180 \
  --oof-exit-max-trades 2500 \
  --oof-exit-step 8 \
  --final-iterations 900 \
  --final-exit-iterations 650 \
  --entry-thresholds 60 \
  --exit-max-trades 3500 \
  --exit-step 6 \
  --eval-costs 1,2,3 \
  --exit-threshold-grid 0.35,0.45,0.55,0.65,0.75 \
  --keep-going
