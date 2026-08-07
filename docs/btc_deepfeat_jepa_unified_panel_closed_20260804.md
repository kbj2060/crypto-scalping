# BTC Deep-Feature JEPA Encoder on Unified Raw Panel — CLOSED 2026-08-04

## What was tried

Per user request: build a new BTC architecture on the union of every currently-
available raw feature source (`causalfix_final` 99 model cols + Regime3 wide24's 13
raw-input cols not already in causalfix_final + Deribit DVOL's 7 derived cols +
CoinMetrics on-chain's 6 derived cols = 124 unique raw features), then apply a
deep-learning feature-extraction method surveyed from 2024-2026 literature (JEPA
latent-prediction + TS2Vec-style temporal contrastive) to turn that union into
learned embeddings.

Plan: `C:\Users\kbj20\.claude\plans\unified-napping-lighthouse.md` (approved).

## Stages run

1. **Stage 0** — `scripts/build_btc_unified_raw_panel_20260804.py`: unified panel,
   271,797 rows, 140 cols. Output: `data/splits/year_oos/btc_unified_raw_panel_20260804.parquet`.
2. **Stage A** (cheapest falsification, raw union only, no deep learning) —
   `scripts/train_eval_btc_dense_nogate_quality_unified_raw_20260804.py`, same
   dense-nogate LightGBM pipeline/split/cost model as the live causalfix_final
   baseline. **Result: 0/7 threshold configs VAL+OOS both positive (n>=15 each
   side).** The new raw columns (DVOL, on-chain, state7/state12) do get real usage —
   several rank in the top 3-15 of 124 features by importance — but do not flip any
   config profitable on both VAL and OOS. Same failure shape as every prior
   individual test of these sources (DVOL closed 0/9, on-chain closed 0/9).
3. **Stage B** (JEPA + temporal-contrastive self-supervised encoder, cheap gate) —
   `ensemble/deep_features/tabular_jepa_encoder.py` (small transformer, window=32
   bars, d_model=64, embed_dim=24, EMA target encoder, latent-prediction loss +
   InfoNCE temporal-contrastive auxiliary), pretrained unsupervised on all rows
   before `VAL_START=2025-09-01` only (`scripts/pretrain_btc_deep_feature_encoder_20260804.py`,
   8 epochs, jepa loss 0.28->0.19, contrastive loss 1.47->1.36, no collapse), then
   evaluated via `scripts/eval_btc_deepfeat_cheap_falsification_20260804.py`
   (embeddings concatenated onto the Stage 0 panel, same LightGBM pipeline).
   **Result: 0/9 threshold configs VAL+OOS both positive.** The 24 `deepfeat_*`
   embedding columns rank *weaker* than the raw DVOL/on-chain columns did in Stage A
   (mid-to-low importance, rank 56-121 of 148), i.e. the learned representation was
   less useful to the downstream tree than the hand-derived features it was built
   from.

## Verdict

**CLOSED.** Both the raw-union cheap gate and the deep-feature-encoder cheap gate
failed, matching the project's established pattern: every recent BTC data source and
every model-family variant tried in 2026-08 has hit the same wall. The negative
result here is informative in a specific way -- the JEPA embeddings did not even
match the raw features they were derived from, suggesting the bottleneck is not "the
tree can't combine these features nonlinearly enough" but something upstream (label
definition / achievable edge on this instrument), consistent with the standing
"病목이 데이터가 아니라 라벨링/아키텍처 자체" hypothesis from
`docs/btc_new_architecture_session_summary_20260804.md`.

Per the approved plan, Stage C (TabM integration + Fresh-Forward validation via
`core/causal_futures_backtest.py`) was **not** attempted -- the Stage B gate did not
pass.

## Artifacts (kept for reference, not for reuse as a promotion basis)

- `data/splits/year_oos/btc_unified_raw_panel_20260804.parquet`
- `data/splits/year_oos/btc_deepfeat_embeddings_20260804.parquet`
- `data/ensemble/supervised/btc_deepfeat_jepa_encoder_20260804.pt`
- `tmp/btc_dense_nogate_quality_unified_raw_20260804.csv`
- `tmp/btc_deepfeat_cheap_falsification_20260804.csv`
