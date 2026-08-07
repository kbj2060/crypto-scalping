# SOL/BTC regime model retrain + tuning (current HMM + net-new future CryptoMamba), 2026-07-21

## Goal

User asked to retrain/tune both regime-related models for SOL and BTC "to the max": (1) the
current-regime HMM nowcast (already existed for both, but with no recorded quantitative
performance for BTC, and only a single untuned config for both), and (2) a future/prediction
regime model (CryptoMamba h6, previously ETH-only -- SOL/BTC never had one).

## Part 1: Current-regime HMM tuning sweep (classifier accuracy only)

Reused `scripts/experiment_regime3_current_hmm_wide24_20260529.py` unmodified against SOL/BTC's
own `data/splits/year_oos/{sol,btc}_features_{2024,2025,2026}.csv`, sweeping its own established
axes (4 feature_sets x 2 label_modes = 8 combos/asset, VAL-selected).

| | SOL VAL best (docs42, label=current) | SOL live (wide24, label=balancedish) | BTC VAL best (docs42, label=current) | BTC live (wide24, label=balancedish) |
|---|---:|---:|---:|---:|
| balanced_accuracy | 0.7845 | 0.7149 | 0.8455 | 0.7937 |

docs42 beats the live wide24+balancedish combo on pure classifier accuracy for both assets.

## Part 2: Full fresh-forward retrain test -- docs42 does NOT improve trading performance

Before swapping docs42 into live, discovered both SOL and BTC's live parent models (147 base_cols)
were **trained on the 6 `regime3_current_sensitive_wide24_*` columns as direct input features**
(confirmed via `torch.load(...)['base_cols']`), plus a separately shared frozen module
(`train_omega1_regime3_expert_direction_head_volpca_20260602.py`, `hard.ROUTE_COLS`) and another
hardcoded string reference inside `train_omega1_regime3_routed_expert_direction_quality_20260602.py`
also expect this exact column-name contract -- so a naive current-HMM path swap would break live
inference (missing/misaligned features), not just improve one input. Ran the full retrain pipeline
instead (feature relabel -> parent retrain -> risk-sidecar retrain -> scale-map -> fresh-forward
VAL/OOS), using "maskedname" sidecar CSVs (docs42's output columns renamed to the wide24 naming
convention) so the multiple hardcoded string consumers keep working unchanged without needing to
patch each one individually.

**SOL: rejected, collapses badly.**

| | live v2 (wide24) | regime_docs42 retrain |
|---|---:|---:|
| VAL | +16.75% / MDD -26.29% | **-14.29% / MDD -26.99%** |
| OOS | +57.94% / MDD -21.35% | **+0.67% / MDD -21.46%** |

**BTC: no clear win, mildly worse.**

| | live v1 (wide24) | regime_docs42 retrain |
|---|---:|---:|
| VAL (gated) | +12.39% / MDD -6.49% | +13.88% / MDD -8.48% |
| OOS (gated) | +29.23% / MDD -10.65% | +22.75% / MDD -16.56% |

**Conclusion: current-regime HMM stays on wide24 for both assets. Not live-wired.** This is the
third instance this session of "better classifier accuracy does not translate to better trading
performance" (alongside BTC's failed adaptive_squeeze test and the SOL/BTC chop-soft-sizing
transfer-check rejection).

## Part 3: Net-new future/prediction CryptoMamba models (research artifact only, per user scope)

Forked `scripts/train_regime3_cryptomamba_pred_20260531.py` for each asset
(`train_regime3_cryptomamba_pred_sol_20260721.py`, `..._btc_20260721.py`), pointing at each
asset's own raw features and its LIVE current-HMM sidecar (wide24, not docs42, since docs42 was
just rejected for live use in Part 2). Had to first generate a missing 2024 wide24 sidecar
transform for both assets (only 2025/2026 existed previously) via a one-off run of the same
frozen-joblib causal `_transform` already used for the wide24 experiment.

| | ETH (existing) | SOL (new) | BTC (new) |
|---|---:|---:|---:|
| VAL balanced_accuracy / AUC | 0.6520 / 0.8242 | 0.5682 / 0.7483 | 0.6691 / 0.8429 |
| OOS(2026) balanced_accuracy / AUC | 0.6726 / 0.8438 | 0.6136 / 0.7849 | **0.6725 / 0.8365** |

BTC's future-regime classifier is essentially as strong as ETH's own (AUC 0.8365 vs 0.8438). SOL's
is meaningfully weaker (AUC 0.7849), consistent with SOL's generally noisier data found elsewhere
in this project.

## Part 4: ETH docs42 full retrain -- same rejection

User asked to also test ETH's own docs42 current-regime swap with a full fresh-forward retrain
(same as Parts 1-2, but for ETH). Reused the maskedname trick on ETH's own docs42 sweep output
(`tmp/causal_regen_20260516/eth_regime3_current_hmm_tuning_20260721/sensitive/`), forked ETH's live
h48qual parent script (`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`) and
sidecar script (`train_eval_omega4_2_risk_sidecar_20260622.py`) with the exact live contract
(direction/quality label dirs, `entry_label_terminal_giveback` exit mode, `log_risk`/`validation_only`
sidecar selection, tail_penalty=0.5, avg-notional constraint 0.45-0.95 relaxed to get a raw read
once it rejected every candidate).

**Result: also rejected.** Even before a final scale-map stage, the risk-sidecar's own selected
replay already shows clearly negative VAL and OOS:

| | VAL | OOS |
|---|---:|---:|
| docs42 retrain (h48qual component only) | -9.49% / MDD -20.39% | -17.32% / MDD -22.77% |

Consistent with SOL/BTC: classifier accuracy improves in isolation, but the full retrained trading
pipeline gets clearly worse. **ETH's current-regime model also stays on wide24. Not live-wired.**
This closes the docs42-current-regime question for all three assets -- do not re-attempt without
new information.

## Status

`model_status=research_diagnostic_not_live_wired` for all of Part 3 (per explicit user scope:
model-own accuracy/AUC report only, no Sigma6-style trading-gate combination or backtest). Part 1/2
current-HMM retrain is also not live-wired (rejected on fresh-forward evidence). No
`trading_bot.py`/`.env`/`runtime_config.py` changes for SOL or BTC -- confirmed via `git status`/grep
that the live `FINAL_GOVERNOR_OMEGA4_6_1_SOL_REGIME3_PATH`/`..._BTC_REGIME3_PATH` still point at the
original wide24 joblib paths.

## Reusable artifacts (all new, none touching live config)

- `data/ensemble/supervised/{sol,btc}_regime3_current_hmm_docs42_20260720/` -- tuned but rejected
  current-HMM candidates.
- `data/ensemble/supervised/regime3_cryptomamba_pred_{sol,btc}_h6_nocurrent_20260721/` -- new
  future-regime models, usable for a future Sigma6-style gate experiment if ever revisited.
- `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_{sol,btc}_regime_docs42_20260721.py`,
  matching risk-sidecar and scale-map forks -- reusable pattern (maskedname column trick) for any
  future per-asset feature-swap retrain test.
