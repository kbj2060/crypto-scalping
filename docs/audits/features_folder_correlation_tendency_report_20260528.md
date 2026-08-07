# Features Folder Correlation And Tendency Report - 2026-05-28

## Scope

This audit covers the feature contracts and generators under `features/`, then maps them onto the active Alpha7/Alpha6 feature frame used by the current experiments.

Funding-family source issues were remediated after `docs/audits/last_funding_rate_source_audit_20260528.md`; active split CSVs now use ETHUSDT-only backward-asof funding.

Artifacts:

- Code-created feature inventory: `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/code_created_features.csv`
- Code literal inventory: `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/code_referenced_feature_literals.csv`
- Feature score table: `tmp/causal_regen_20260516/all_feature_usage_20260528/feature_usage.csv`
- Family score table: `tmp/causal_regen_20260516/all_feature_usage_20260528/family_usage.csv`
- Family probe AUC table: `tmp/causal_regen_20260516/all_feature_usage_20260528/family_probe_auc.csv`
- Redundancy pairs: `tmp/causal_regen_20260516/all_feature_usage_20260528/redundancy_top_pairs.csv`

Important caveat: `features/` contains Python generators and contracts, not parquet/csv feature matrices. Correlation and tendency statistics therefore use the active generated frames, while the source inventory records which columns are created or referenced by `features/*.py`.

## Method

- Train split: 2025 before 2025-10-01.
- Validation split: 2025 from 2025-10-01.
- OOS split: 2026 evaluation frame.
- Excluded target/leak-like generated columns: names containing `label`, `future`, `fwd_`, `cash_after`, `pnl_after`, `realized_net`, `exit_reason`, `dir24`, `high_abs`, `large_down`, and direct `target_` columns except audited `m7_target_*` model outputs.
- Tendency metrics:
  - OOS Spearman IC against future return horizons 6/12/24/48.
  - OOS Spearman IC against future absolute-return horizons 6/12/24/48.
  - PSI drift from train to 2026 OOS.
  - Family-level HGB probes for direction, high-volatility, and large-down labels.
- Redundancy threshold: absolute correlation `>= 0.95`.

## Top-Level Findings

The feature set is much stronger for risk, volatility, sizing, and exit control than for raw direction prediction.

- Best direction family probe is weak: `regime_pred` OOS AUC about `0.539` for 24-bar up/down.
- High-volatility prediction is strong:
  - `m7` OOS AUC about `0.733`
  - `microstructure` about `0.725`
  - `volatility` about `0.722`
  - `ai` about `0.716`
  - `ts_model` about `0.700`
- Large-down risk is moderate:
  - `m7` about `0.626`
  - `ai` about `0.621`
  - `volatility` about `0.613`
  - `ts_model` about `0.612`
  - `microstructure` about `0.610`

Practical interpretation: use these features mainly in risk/meta, notional, TP/SL, cooldown, and exit layers. Do not expect direct parent direction accuracy to improve simply by adding many raw features.

## Source Inventory

The static source scan found:

- 93 assigned columns in `features/*.py`.
- 526 referenced feature literals and contract names.

Primary source modules:

- `features/engineering.py`: base market, flow, technical, funding, volatility, temporal, synthetic alpha, and active-prune contract.
- `features/elite.py`: strategy-style elite signals, synthetic alpha engine, volatility model engine, regime engine, and new elite signal engine.
- `features/high_order_state.py`: high-order state summaries.
- `features/schema.py`: active RL/live keep-set.
- `features/registry.py`: M7 generated/core/live/deprecated column contract.
- `features/integrated_overlay.py`, `features/news_shock_guard.py`, `features/playbook_meta_controller.py`: runtime overlay feature builders.

## Strongest Feature Tendencies

### Return / Direction-ish Candidates

These have the highest OOS future-return IC, but even these should be treated as weak directional edges rather than hard signals.

| Feature | Family | OOS return IC | OOS abs-return IC | PSI | Recommendation |
|---|---:|---:|---:|---:|---|
| `last_funding_rate` | funding | 0.149 | 0.178 | 0.470 | execution/risk sizing, not direct hard entry |
| `long_squeeze_risk` | technical | 0.141 | 0.124 | 0.276 | entry context |
| `squeeze_power` | technical | 0.127 | 0.185 | 0.857 | monitor/veto only unless normalized |
| `session_us` | calendar | 0.086 | 0.204 | 0.000 | session/risk context |
| `crowding_pressure` | high-order | 0.082 | 0.072 | 0.032 | secondary entry context |
| `funding_roc_288` | funding | 0.080 | 0.027 | 0.275 | entry context |
| `funding_pressure` | funding | 0.078 | 0.199 | 0.256 | execution/risk sizing |
| `m7_quant_up` | m7 | 0.077 | 0.075 | 0.011 | compact direction summary |
| `m7_quant_dn` | m7 | 0.076 | 0.071 | 0.013 | compact direction summary |
| `m7_q50` | m7 | 0.069 | 0.073 | 0.022 | compact expectation summary |

### Risk / Volatility / Exit Candidates

These are the most useful group for sizing, TP/SL, exit, and veto.

| Feature | Family | OOS abs-return IC | PSI | Recommendation |
|---|---:|---:|---:|---|
| `m7_entry_short_offset` | m7 | 0.461 | 0.127 | risk sizing / exit |
| `parkinson_vol` | volatility | 0.450 | 0.126 | risk sizing / exit |
| `garman_klass_vol` | volatility | 0.449 | 0.116 | risk sizing / exit |
| `m7_entry_long_offset` | m7 | 0.448 | 0.115 | risk sizing / exit |
| `rogers_satchell_vol` | volatility | 0.447 | 0.113 | risk sizing / exit |
| `trades` | base market | 0.442 | 0.172 | execution context |
| `m7_quality_pred` | m7 | 0.415 | 0.082 | risk sizing / exit |
| `bb_width` | volatility | 0.410 | 0.097 | risk sizing / exit |
| `m7_target_quality` | m7 | 0.400 | 0.095 | audited model-output feature; rename in future contract if rewritten |
| `volume` | microstructure | 0.395 | 0.085 | execution/risk sizing |
| `taker_buy_base` | microstructure | 0.388 | 0.074 | execution/risk sizing |
| `teacher_tail_warning` | teacher | 0.350 | 0.033 | exit/veto |
| `m7_qwidth` / `teacher_uncertainty` | m7/teacher | 0.343 | 0.030 | uncertainty/risk |
| `ai_adverse_risk` / `tide_vol_raw` | ai/ts_model | 0.341 | 0.040 | adverse risk / exit |

## Correlation And Duplicate Clusters

Do not include all members of these groups blindly. Pick one representative per downstream role unless an ensemble explicitly benefits from redundant views.

### Exact Or Near-Exact Duplicates

- `ai_dir_edge` == `patchtst_median`
- `ai_flow_slope` == `dlinear_smf_slope`
- `ai_adverse_risk` == `tide_vol_raw`
- `oi_change_rate` == `smart_money_flow`
- `ai_dir_entropy` == `patchtst_regime_sim`
- `m7_iso_anom` == `m7_iso_pred`
- `m7_gate_block` == `m7_vae_anom`
- `regime4_pred_chop_prob` == `regime4_pred_range_prob`
- `regime4_pred_micro_prob` == `regime4_pred_trend_prob`
- `regime4_pred_instability_prob` == `regime4_pred_whipsaw_prob`
- `clean_regime4_state24_sticky090_v2_chop_prob` == `clean_regime4_state24_sticky090_v2_range_prob`
- `clean_regime4_state24_sticky090_v2_micro_prob` == `clean_regime4_state24_sticky090_v2_trend_prob`
- `clean_regime4_state24_sticky090_v2_instability_prob` == `clean_regime4_state24_sticky090_v2_whipsaw_prob`
- `clean_regime4_state24_sticky090_v2_risk_off_prob` == `clean_regime4_state24_sticky090_v2_transition_risk`

### Price-Level Redundancy

Raw OHLC and M7 raw price outputs are almost perfectly correlated and drift heavily:

- `open`, `high`, `low`, `close`
- `close_btc`
- `m7_entry_long_price`, `m7_entry_short_price`
- `m7_tp_price`, `m7_sl_price`

These should not be direct model inputs in active/live candidates. Use offsets, returns, volatility-normalized distances, or execution-only context instead.

## Drift And Bug-Risk Features

The following are not necessarily future leaks, but they are bad active-model inputs without re-normalization or a specific veto/monitor role.

| Feature | Issue |
|---|---|
| `garch_vol_z` | Extreme PSI, invalid/NaN tendency in this audit. Treat as bug-risk until regenerated or replaced. |
| `open/high/low/close`, `close_btc` | Raw level drift; do not feed directly to model. |
| `m7_entry_long_price`, `m7_entry_short_price`, `m7_tp_price`, `m7_sl_price` | Raw price-level model outputs; convert to offsets/returns only. |
| `sum_open_interest_value` | Raw OI level drift. Prefer pct/normalized variants. |
| `sum_toptrader_long_short_ratio`, `count_long_short_ratio`, `whale_retail_ratio` | High drift; use as monitor/risk context only unless normalized. |
| `trade_intensity` | High drift; execution monitor only. |
| `squeeze_power` | Good return IC but high drift; normalize before active use. |
| `btc_corr_60` | High drift; monitor/veto only. |
| `clean_regime4_state24_sticky090_v2_factor_liquidity` | High drift; monitor/veto only. |
| `clean_regime4_state24_sticky090_v2_risk_off_prob`, `transition_risk` | Redundant and drift-risk; monitor/veto only. |

No direct future-looking feature was confirmed in this pass among the audited active/live candidates. The bug-risk bucket here means contract/drift/normalization risk, not confirmed label leakage.

## Recommended Minimal Feature Use By Layer

### Entry / Direction Context

Use a compact set only. Direction signal is weak, so these should modulate existing entry models rather than own final direction.

- `last_funding_rate`
- `funding_roc_288`
- `long_squeeze_risk`
- `crowding_pressure`
- `m7_quant_up`
- `m7_quant_dn`
- `m7_q50`
- `session_us`
- One of `ai_dir_edge` or `patchtst_median`, not both.

### Risk Sizing / TP-SL / Exit

This is the strongest use case for the feature folder outputs.

- `parkinson_vol`
- `garman_klass_vol`
- `rogers_satchell_vol`
- `bb_width`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_quality_pred`
- `m7_qwidth`
- `teacher_tail_warning`
- `teacher_uncertainty`
- One of `ai_adverse_risk` or `tide_vol_raw`, not both.

### Execution Context

- `volume`
- `quote_volume`
- `trades`
- `taker_buy_base`
- `taker_buy_quote`
- `volume_btc`
- `funding_pressure`

### Regime / Meta Layer

Use the reduced regime set recorded in the model architect memory. Do not inject all regime columns directly into parent/deep models.

- `regime4_pred_chop_prob`
- `regime4_pred_trend_prob`
- `regime4_pred_bear_prob`
- `regime4_pred_instability_prob`
- `regime4_pred_directional_bias`
- `clean_regime4_state24_sticky090_v2_chop_prob`
- `clean_regime4_state24_sticky090_v2_trend_prob`
- `clean_regime4_state24_sticky090_v2_entropy`
- `clean_regime4_state24_sticky090_v2_confidence`
- `clean_regime4_state24_sticky090_v2_factor_trend`
- `ai_vol_regime_pct`
- `patchtst_regime_sim`

## Contract Recommendations

1. Keep Alpha7.1/01965 baseline compact input as the reference until a layer-specific ablation beats it.
2. Do not add broad raw feature bundles to parent/deep entry models. Prior tests showed feature expansion and PCA variants underperformed the compact baseline.
3. Promote features by role:
   - Entry: compact funding/squeeze/M7 expectation summaries.
   - Exit/risk: volatility, uncertainty, M7 offsets, teacher tail warnings.
   - Meta/veto: reduced regime set plus drift-risk monitors.
4. If the active contract is rewritten, rename `m7_target_quality` and `m7_target_hold` to prediction-style names to avoid target-name confusion. Do not silently alias in active paths; regenerate the contract fail-fast.
5. Treat raw price-level columns and raw M7 price outputs as forbidden for active/live model input unless transformed into offsets/returns and re-audited.

## Next Tests

Suggested next ablations:

1. `entry_context_minimal`: compact directional context only.
2. `risk_sizing_exit_minimal`: volatility + M7 offset + uncertainty/tail features only.
3. `regime_meta_overlay`: reduced regime set as external threshold/notional/exit modifier, not direct model input.
4. `normalized_squeeze`: replace raw `squeeze_power` with a rolling z-score or bounded transform and retest.
5. `raw_price_free_contract`: enforce exclusion of raw price-level/M7 price outputs in active model contracts.
