#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
OUT_ROOT="$ROOT/tmp/causal_regen_20260516/alpha6_representation_oof_full_20260524"
REP_DIR="$OUT_ROOT/representation"
REP_FILE="$REP_DIR/alpha6_representation_oof_features.parquet"
LABEL_DIR="$OUT_ROOT/label_oof_full"

mkdir -p "$OUT_ROOT" "$REP_DIR" "$LABEL_DIR"

cd "$ROOT"

echo "[run] representation full OOF started $(date -Is)"
"$PY" -u scripts/train_alpha6_representation_label_features_20260524.py \
  --out-dir "$REP_DIR" \
  --seq-len 64 \
  --horizon 24 \
  --hidden 96 \
  --emb-dim 64 \
  --batch-size 192 \
  --epochs 4 \
  --mc-paths 16 \
  --models ts2vec,cost,mamba,timegrad,timellm \
  --skip-timellm-on-fail

echo "[run] CatBoost label OOF/full started $(date -Is)"
"$PY" -u scripts/train_alpha6_catboost_label_sweep_oof_full_20260524.py \
  --out-root "$LABEL_DIR" \
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
  --oof-exit-max-trades 3500 \
  --oof-exit-step 6 \
  --final-iterations 900 \
  --final-exit-iterations 650 \
  --entry-thresholds 60 \
  --exit-max-trades 12000 \
  --exit-step 2 \
  --eval-costs 1,2,3 \
  --exit-threshold-grid 0.35,0.45,0.55,0.65,0.75 \
  --keep-going

echo "[run] completed $(date -Is)"
