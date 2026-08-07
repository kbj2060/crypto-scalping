# Omega1 Processed Feature Contract - 2026-05-31

## Purpose

This is the canonical tracking document for Omega1 processed / layered
features. Update this document whenever a processed feature is added, removed,
promoted, demoted, or reclassified.

The goal is to keep feature state explicit and avoid silent aliasing or
fallbacks.

## Change Log

- 2026-05-31: Initial standalone Omega1 processed feature contract created.
- 2026-05-31: Added M7 ZigZag direction candidates from
  `alpha_catboost_action_master_like` and `trend_xgb_like_xgb`.
- 2026-05-31: Confirmed the M7+ZigZag CSV files still contain legacy M7
  columns for historical compatibility, but those legacy columns are not
  automatically approved Omega1 inputs. Usage status is tracked explicitly
  below.
- 2026-05-31: M7 target-family inputs remain excluded from active Omega1
  teacher generation after Red Team alias audit.
- 2026-05-31: Promoted a narrow explicit M7 subset into Omega1 HGB/Mamba
  teacher inputs: M7 quantile-risk summaries plus ZigZag-retrained direction
  probability/edge features. Legacy M7 target-family and ordinal action
  columns remain blocked.
- 2026-05-31: Rebuilt HGB teacher with M7 ZigZag inputs and active
  `zigzag_action` labels. Artifact:
  `tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_20260531`.
- 2026-05-31: Ran 1-epoch Mamba teacher smoke with the same M7 ZigZag
  + `zigzag_action` contract. Artifact:
  `tmp/causal_regen_20260516/omega1_mamba_teacher_m7zigzag_smoke_20260531`.
- 2026-05-31: Retired the experimental `m7_clean_*` recomputed risk/execution
  context by user decision because the signal ownership was ambiguous. These
  columns are removed from active teacher inputs and M7 CSVs.
- 2026-05-31: Promoted `regime3_cryptomamba_pred_h6_nocurrent_20260531`
  all-sanitized-128 feature variant as the active Regime3 h6 future-context
  sidecar candidate after seed retest.
- 2026-05-31: Added explicit Regime3 CryptoMamba h6 numeric outputs to Omega1
  processed-feature registry. This is a narrow exception to the generic `future`
  token ban because these columns are model predictions generated at timestamp
  `t`, not realized future labels.
- 2026-05-31: Clarified that this contract covers the full Omega1 architecture
  feature registry. Teacher input features are only one consumer subset, not
  the scope of the whole contract.
- 2026-05-31: Added Layer 2 direction feature generator tests
  `dir3_retrieval` and `dir3_cycle`. `dir3_retrieval` is retained as a
  parent/meta candidate after a small 2026 OOS label-probe lift when added to
  Omega1 core inputs. `dir3_cycle` is diagnostics-only because it did not
  improve the combined parent/meta probe.
- 2026-05-31: Completed the remaining Layer 2 direction generator tests
  `dir3_chartcnn`, `dir3_patch`, and `dir3_duet`. `dir3_patch` is the best
  active parent/meta candidate from this batch; `dir3_duet` is a weaker
  research parent/meta candidate; `dir3_chartcnn` is diagnostics-only.
- 2026-05-31: Added financial-paper-inspired DIR3 candidates:
  `dir3_vsnlstm`, `dir3_lpatchtst`, and `dir3_xtrend`. `dir3_vsnlstm` is
  retained as a parent/meta candidate after improving the Omega1 core probe.
  `dir3_lpatchtst` is a weaker research candidate; `dir3_xtrend` is
  diagnostics-only in this implementation.
- 2026-05-31: Added `dir3_cryptomamba`, a direction sidecar that ports the
  Regime3 CryptoMamba C-Block architecture to the active `zigzag_action`
  direction label. It is retained as a parent/meta research candidate after a
  small core-probe bacc/AUC lift, but it did not beat `dir3_patch` or
  `dir3_vsnlstm`.
- 2026-05-31: Completed Top2 full sweep for `dir3_patch` and `dir3_vsnlstm`.
  After seed/epoch/hyperparameter sweep, `dir3_vsnlstm_full` is the current
  best label-probe/meta-probe direction candidate; `dir3_patch_full` remains a
  strong HGB baseline.
- 2026-05-31: Reclassified the Omega1 architecture contract into explicit
  layers. Existing `dir3_*` artifact names are historical prefixes only:
  standalone direction generators trained on 2024 and scored on 2025/2026 are
  Layer 2 processed features, not Layer 3 teacher/meta outputs.
- 2026-06-01: Added the Omega1 supervised-label authority rule. Omega1 may
  borrow architectural ideas from Alpha models, but it must not reuse Alpha
  supervised label contracts as active Omega1 training targets. All supervised
  Omega1 action / direction / entry / parent / expert heads must train from the
  active `zigzag_action` contract unless this document is explicitly amended.
- 2026-06-01: Added the ZigZag-only max-feature current-Regime3 MoE risk
  redesign result. The supervised heads remain trained only on
  `zigzag_action`; the improvement came from non-supervised validation search
  over confidence, router, notional, TP/SL, max-hold, cooldown, and expert
  scale parameters.
- 2026-06-01: Retired `teacher_*` features from Omega1 active/research
  modeling by user decision. Teacher artifacts remain historical only. Active
  Omega1 feature selection must use Layer 1/2 features directly and must fail
  fast if `teacher_*` or `teacher_oof_*` appears in an active input contract.
- 2026-06-02: Added the grouped Omega1 feature inventory snapshot covering
  source/raw, Layer-1 deterministic processed, and Layer-2 model/sidecar
  processed features observed in the current max-feature MoE and confirmed
  Direction Head contracts.
- 2026-06-02: Added a feature-character grouping section so Omega1 model
  layers can select features by economic / modeling role instead of broad
  prefixes.
- 2026-06-02: Promoted `core_plus_tsfm_chronos + volatility_pca06` as the
  fixed Omega1 Direction Head contract after grouped volatility PCA testing and
  core-group PCA ablation. The confirmed Direction Head now consumes the 55 raw
  `core_plus_tsfm_chronos` features plus 6 PCA components fit from the explicit
  24-column volatility context group. Core subgroups remain raw.

## Omega1 Layer Contract

Omega1 uses this layered feature DAG:

1. Layer 1: source/current features. These are raw or directly derived
   live-computable market, execution, funding, flow, BTC-relative, and
   current-context fields.
2. Layer 2: processed OOS feature generators. These train on 2024 and score
   2025/2026 by exact timestamp. This layer includes AI/TSFM risk context,
   Chronos context, Regime3 sidecars, M7 ZigZag/quantile context, and the
   standalone direction generators currently stored under legacy `dir3_*`
   artifact names.
3. Layer 3: meta/parent stack. These models train on 2025 Layer-2 OOS scores
   and test on 2026 Layer-2 scores. `teacher_*` generation is retired and must
   not be used in active Omega1 inputs.
4. Layer 4: final policy/backtest/live execution. This layer consumes approved
   Layer-2 and Layer-3 outputs according to the consumer permissions below.

Naming rule: do not rename existing artifacts or silently alias prefixes.
`dir3_*` remains a legacy artifact prefix for reproducibility, but its
architectural layer is Layer 2 unless a future artifact explicitly trains on
2025 Layer-2 OOS scores under a new no-leak stacking contract.

Dependency rule: Layer 3 outputs must never feed Layer 2 generation. Layer 2
features may feed Layer 3 only through explicit OOS score files and exact
timestamp joins. Any violation must fail fast.

## Omega1 Grouped Feature Inventory Snapshot - 2026-06-02

This section is an inventory snapshot, not a broad usage approval. The
existing Architecture-Approved Processed Features, M7 Usage Status, Retired
Teacher Outputs, Hard Exclusions, and consumer permissions below remain the
authority for active modeling.

Snapshot sources:

- Max-feature ZigZag MoE feature contract:
  `tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_20260601/report.json`
- Confirmed Direction Head `core_plus_tsfm_chronos` input contract:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602/report.json`

Snapshot totals:

- Total unique columns: `211`
- Source/raw columns: `17`
- Layer-1 deterministic processed columns: `76`
- Layer-2 model/sidecar processed columns: `118`

### Source / Raw Columns

These are direct source, exchange, or frame-level fields. They are not model
outputs.

- `side_hint`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_volume`
- `trades`
- `taker_buy_base`
- `taker_buy_quote`
- `sum_open_interest_value`
- `sum_toptrader_long_short_ratio`
- `count_long_short_ratio`
- `last_funding_rate`
- `close_btc`
- `volume_btc`
- `quote_volume_btc`

### Layer-1 Deterministic Processed Columns

These are live-computable fields derived from source/current/past market data.
They do not require a learned OOS sidecar model.

- `whale_retail_ratio`
- `smart_money_flow`
- `squeeze_power`
- `oi_change_rate`
- `net_taker_ratio`
- `taker_acceleration`
- `trade_intensity`
- `big_trade_ratio`
- `log_return`
- `volatility_z`
- `rsi`
- `macd_hist`
- `bb_width`
- `bb_width_z`
- `hma_slope`
- `wick_ratio`
- `garman_klass_vol`
- `realized_vol_ratio`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `rogers_satchell_vol`
- `parkinson_vol`
- `amihud_illiquidity_z`
- `btc_corr_60`
- `eth_btc_ratio_change`
- `fvg_dist`
- `chop_index`
- `hour_sin`
- `hour_cos`
- `minute_sin`
- `minute_cos`
- `session_europe`
- `session_us`
- `is_hour_open`
- `cvp_poc_dist`
- `cvp_cluster_position`
- `cvp_volume_imbalance`
- `cvp_regime`
- `turtle_signal`
- `dual_momentum`
- `mean_reversion_z`
- `breakout_strength`
- `volume_profile_signal`
- `funding_roc_288`
- `long_squeeze_risk`
- `funding_price_divergence`
- `regime_trending`
- `ofi_acceleration`
- `kalman_velocity`
- `realized_skewness`
- `ofti`
- `kel`
- `mta_funding`
- `svps`
- `sig_liquidity_trap`
- `garch_vol_z`
- `liquidity_vacuum`
- `execution_quality`
- `jump_z`
- `jump_flag`
- `evt_tail_flag`
- `evt_excess_z`
- `funding_abs`
- `funding_pressure`
- `crowding_pressure`
- `whale_conviction`
- `cross_scale_curvature`
- `cvp_vah_val_width`
- `hurst_48`
- `ou_funding_z`
- `ou_halflife`
- `sig_ai_squeeze`
- `sig_oi_divergence`
- `sig_trend_health`
- `sig_volume_confirm`
- `sig_whale`

### Layer-2 Model / Sidecar Processed Columns

These are generated by AI/TSFM, M7, Regime3, DIR3, Chronos, or other learned
sidecar processes. Some are active/research candidates and some are legacy or
retired; consult the detailed registry sections below before use.

#### M7 Legacy / Quantile / Risk / Unsupervised Context

- `m7_trend_xgb_dn`
- `m7_trend_xgb_fl`
- `m7_trend_xgb_up`
- `m7_mtl_dn`
- `m7_mtl_fl`
- `m7_mtl_up`
- `m7_quant_dn`
- `m7_quant_fl`
- `m7_quant_up`
- `m7_confidence`
- `m7_action`
- `m7_size`
- `m7_q10`
- `m7_q50`
- `m7_q90`
- `m7_qwidth`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_tp_offset`
- `m7_sl_offset`
- `m7_tp_price`
- `m7_sl_price`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_expected_ret`
- `m7_tail_risk`
- `m7_composite_score`
- `m7_hdb_label`
- `m7_hdb_prob`
- `m7_prob_dn`
- `m7_prob_fl`
- `m7_prob_up`
- `m7_vae_threshold`

#### Legacy PatchTST / AI / TSFM Context

- `pred_patchtst`
- `conf_patchtst`
- `ai_dir_edge`
- `ai_dir_p_up`
- `ai_dir_p_down`
- `ai_dir_p_flat`
- `ai_dir_entropy`
- `patchtst_median`
- `patchtst_regime_sim`
- `ai_adverse_risk`
- `ai_reward_risk`
- `ai_vol_regime_pct`
- `tide_vol_raw`
- `tide_vol_zscore`
- `ai_flow_pressure`
- `ai_flow_exhaustion`
- `ai_flow_flip_prob`
- `ai_flow_slope`
- `dlinear_smf_ema`
- `dlinear_smf_slope`
- `ai_anchor_revert_prob`
- `ai_anchor_overheat`
- `ai_anchor_trend_escape_prob`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`

#### Regime3 Current / CryptoMamba / Stability-Risk Context

- `regime3_current_sensitive_wide24_bull_prob`
- `regime3_current_sensitive_wide24_bear_prob`
- `regime3_current_sensitive_wide24_chop_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_entropy`
- `regime3_current_sensitive_wide24_margin`
- `regime3_cmamba_h6_sidecar_bull_prob`
- `regime3_cmamba_h6_sidecar_bear_prob`
- `regime3_cmamba_h6_sidecar_chop_prob`
- `regime3_cmamba_h6_sidecar_class_id`
- `regime3_cmamba_h6_sidecar_confidence`
- `regime3_cmamba_h6_sidecar_transition_prob`
- `regime3_cmamba_h6_sidecar_stability_score`
- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`

#### DIR3 Direction Context

- `dir3_vsnlstm_h6_fl_prob`
- `dir3_vsnlstm_h6_up_prob`
- `dir3_vsnlstm_h6_dn_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_fl_prob`
- `dir3_patch_h6_up_prob`
- `dir3_patch_h6_dn_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`

#### Chronos Context

- `chronos_h6_q10`
- `chronos_h6_q50`
- `chronos_h6_q90`
- `chronos_h6_width`
- `chronos_h6_mean`
- `chronos_unc_atr14_q10`
- `chronos_unc_atr14_q50`
- `chronos_unc_atr14_q90`
- `chronos_unc_atr14_width`
- `chronos_unc_atr14_mean`
- `chronos_unc_atr14_width_ewm3`
- `chronos_unc_atr14_width_ewm6`
- `chronos_unc_rv24_q10`
- `chronos_unc_rv24_q50`
- `chronos_unc_rv24_q90`
- `chronos_unc_rv24_width`
- `chronos_unc_rv24_mean`
- `chronos_unc_rv24_width_ewm3`
- `chronos_unc_rv24_width_ewm6`

## Omega1 Feature Character Groups - 2026-06-02

This section groups the current inventory by economic / modeling role. It is
for model-design routing and ablation planning. It is not a permission
override: consumer permissions, hard exclusions, and retired-feature rules
still apply.

### Price / OHLC State

Role: current price state, candle range, and direct execution reference.
Recommended consumers: direction, entry-quality, risk-template, exit/hazard.

- `side_hint`
- `open`
- `high`
- `low`
- `close`

### Volume / Trade Activity

Role: market participation, trade density, and raw activity pressure.
Recommended consumers: direction, entry-quality, risk-template.

- `volume`
- `quote_volume`
- `trades`
- `taker_buy_base`
- `taker_buy_quote`
- `trade_intensity`
- `big_trade_ratio`

### Funding / Perpetual Positioning

Role: funding pressure, crowding, squeeze risk, and derivatives positioning.
Recommended consumers: risk-template, exit/hazard, governor; direction only as
context.

- `last_funding_rate`
- `funding_roc_288`
- `funding_price_divergence`
- `funding_abs`
- `funding_pressure`
- `crowding_pressure`
- `mta_funding`
- `ou_funding_z`
- `ou_halflife`
- `long_squeeze_risk`
- `sum_open_interest_value`
- `sum_toptrader_long_short_ratio`
- `count_long_short_ratio`
- `oi_change_rate`

### BTC / Cross-Asset Context

Role: BTC-relative beta, market-wide pressure, and ETH/BTC divergence.
Recommended consumers: direction, regime router, governor.

- `close_btc`
- `volume_btc`
- `quote_volume_btc`
- `btc_corr_60`
- `eth_btc_ratio_change`

### Order Flow / Whale / Smart Money

Role: directional pressure and participant-quality context.
Recommended consumers: direction, entry-quality, regime experts.

- `whale_retail_ratio`
- `smart_money_flow`
- `net_taker_ratio`
- `taker_acceleration`
- `ofi_acceleration`
- `whale_conviction`
- `sig_whale`
- `ai_flow_pressure`
- `ai_flow_exhaustion`
- `ai_flow_flip_prob`
- `ai_flow_slope`

### Momentum / Trend / Mean-Reversion

Role: trend-following, reversal, breakout, and swing-context features.
Recommended consumers: direction head and regime experts.

- `log_return`
- `rsi`
- `macd_hist`
- `hma_slope`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `turtle_signal`
- `dual_momentum`
- `mean_reversion_z`
- `breakout_strength`
- `volume_profile_signal`
- `kalman_velocity`
- `regime_trending`
- `ai_anchor_revert_prob`
- `ai_anchor_overheat`
- `ai_anchor_trend_escape_prob`
- `dlinear_smf_ema`
- `dlinear_smf_slope`
- `patchtst_median`
- `patchtst_regime_sim`

### Volatility / Range / Tail Risk

Role: sizing, TP/SL width, trade filtering, and adverse-move detection.
Recommended consumers: risk-template, entry-quality, exit/hazard.

- `volatility_z`
- `bb_width`
- `bb_width_z`
- `wick_ratio`
- `garman_klass_vol`
- `realized_vol_ratio`
- `rogers_satchell_vol`
- `parkinson_vol`
- `chop_index`
- `realized_skewness`
- `garch_vol_z`
- `jump_z`
- `jump_flag`
- `evt_tail_flag`
- `evt_excess_z`
- `hurst_48`
- `cross_scale_curvature`
- `ai_adverse_risk`
- `ai_reward_risk`
- `ai_vol_regime_pct`
- `tide_vol_raw`
- `tide_vol_zscore`

### Liquidity / Execution Quality

Role: execution feasibility, slippage risk, and liquidity-trap avoidance.
Recommended consumers: entry-quality, risk-template, governor.

- `amihud_illiquidity_z`
- `fvg_dist`
- `liquidity_vacuum`
- `execution_quality`
- `sig_liquidity_trap`
- `squeeze_power`
- `sig_ai_squeeze`
- `sig_oi_divergence`
- `sig_trend_health`
- `sig_volume_confirm`

### Time / Session Seasonality

Role: intraday session, funding-window, and cycle context.
Recommended consumers: direction, entry-quality, governor.

- `hour_sin`
- `hour_cos`
- `minute_sin`
- `minute_cos`
- `session_europe`
- `session_us`
- `is_hour_open`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`

### CVP / Volume Profile Structure

Role: price-location and volume-profile context.
Recommended consumers: direction and entry-quality.

- `cvp_poc_dist`
- `cvp_cluster_position`
- `cvp_volume_imbalance`
- `cvp_regime`
- `cvp_vah_val_width`

### Regime3 Current Context

Role: current bull / bear / chop market-state routing.
Recommended consumers: regime router, expert selection, risk-template.

- `regime3_current_sensitive_wide24_bull_prob`
- `regime3_current_sensitive_wide24_bear_prob`
- `regime3_current_sensitive_wide24_chop_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_entropy`
- `regime3_current_sensitive_wide24_margin`

### Regime3 h6 Prediction / Stability / Transition Context

Role: near-future regime reliability, transition risk, and churn risk.
Recommended consumers: risk-template, exit/hazard, governor; direction only as
context. `future`-named columns are exact-name prediction outputs generated at
timestamp `t`, not realized future labels.

- `regime3_cmamba_h6_sidecar_bull_prob`
- `regime3_cmamba_h6_sidecar_bear_prob`
- `regime3_cmamba_h6_sidecar_chop_prob`
- `regime3_cmamba_h6_sidecar_class_id`
- `regime3_cmamba_h6_sidecar_confidence`
- `regime3_cmamba_h6_sidecar_transition_prob`
- `regime3_cmamba_h6_sidecar_stability_score`
- `regime3_cmamba_h6_future_bull_prob`
- `regime3_cmamba_h6_future_bear_prob`
- `regime3_cmamba_h6_future_chop_prob`
- `regime3_cmamba_h6_confidence`
- `regime3_cmamba_h6_transition_prob`
- `regime3_cmamba_h6_stability_score`
- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`

### M7 Quantile / Risk / Legacy Context

Role: quantile-risk and historical M7 context. Use only explicitly approved
M7 subsets in active models.
Recommended consumers: risk-template and diagnostics. Most legacy fields remain
blocked unless separately promoted.

- `m7_q10`
- `m7_q50`
- `m7_q90`
- `m7_qwidth`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_tail_risk`
- `m7_composite_score`
- `m7_hdb_label`
- `m7_hdb_prob`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_vae_threshold`

### M7 Direction / Probability Context

Role: ZigZag-retrained or legacy class-probability direction context.
Recommended consumers: direction/meta-policy only when explicitly approved.
Ordinal action columns remain blocked as direct active inputs unless a new
contract promotes them.

- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_action`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_action`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`
- `m7_trend_xgb_dn`
- `m7_trend_xgb_fl`
- `m7_trend_xgb_up`
- `m7_mtl_dn`
- `m7_mtl_fl`
- `m7_mtl_up`
- `m7_quant_dn`
- `m7_quant_fl`
- `m7_quant_up`
- `m7_prob_dn`
- `m7_prob_fl`
- `m7_prob_up`
- `m7_confidence`
- `m7_action`
- `m7_size`

### M7 Price / Offset / Target-Family Legacy Fields

Role: historical audit only. These are not approved active Omega1 inputs.
Recommended consumers: diagnostics only unless a new no-leak recomputation and
promotion audit is written.

- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_tp_offset`
- `m7_sl_offset`
- `m7_tp_price`
- `m7_sl_price`
- `m7_expected_ret`
- `m7_target_quality`
- `m7_target_hold`
- `m7_clean_entry_band_width`
- `m7_clean_entry_long_abs_dist`
- `m7_clean_entry_long_ret`
- `m7_clean_entry_mid_ret`
- `m7_clean_entry_short_abs_dist`
- `m7_clean_entry_short_ret`
- `m7_clean_sl_abs_dist`
- `m7_clean_sl_ret`
- `m7_clean_tp_abs_dist`
- `m7_clean_tp_ret`
- `m7_clean_tp_sl_rr`
- `m7_clean_tp_sl_width`

### DIR3 Direction Generators

Role: standalone Layer-2 direction contexts trained under the active ZigZag
direction contract.
Recommended consumers: Direction Head, meta-policy, and regime-expert ablation.

- `dir3_vsnlstm_h6_fl_prob`
- `dir3_vsnlstm_h6_up_prob`
- `dir3_vsnlstm_h6_dn_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_fl_prob`
- `dir3_patch_h6_up_prob`
- `dir3_patch_h6_dn_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`
- `dir3_duet_h6_fl_prob`
- `dir3_duet_h6_up_prob`
- `dir3_duet_h6_dn_prob`
- `dir3_duet_h6_confidence`
- `dir3_duet_h6_side_edge`
- `dir3_duet_h6_trade_prob`
- `dir3_cryptomamba_h6_fl_prob`
- `dir3_cryptomamba_h6_up_prob`
- `dir3_cryptomamba_h6_dn_prob`
- `dir3_cryptomamba_h6_confidence`
- `dir3_cryptomamba_h6_side_edge`
- `dir3_cryptomamba_h6_trade_prob`
- `dir3_retrieval_h6_fl_prob`
- `dir3_retrieval_h6_up_prob`
- `dir3_retrieval_h6_dn_prob`
- `dir3_retrieval_h6_confidence`
- `dir3_retrieval_h6_side_edge`
- `dir3_retrieval_h6_trade_prob`
- `dir3_retrieval_h6_neighbor_edge_mean`
- `dir3_retrieval_h6_neighbor_edge_q25`
- `dir3_retrieval_h6_neighbor_edge_q75`
- `dir3_retrieval_h6_regime_consensus`
- `dir3_retrieval_h6_similarity_score`
- `dir3_lpatchtst_h6_fl_prob`
- `dir3_lpatchtst_h6_up_prob`
- `dir3_lpatchtst_h6_dn_prob`
- `dir3_lpatchtst_h6_confidence`
- `dir3_lpatchtst_h6_side_edge`
- `dir3_lpatchtst_h6_trade_prob`
- `dir3_xtrend_h6_fl_prob`
- `dir3_xtrend_h6_up_prob`
- `dir3_xtrend_h6_dn_prob`
- `dir3_xtrend_h6_confidence`
- `dir3_xtrend_h6_side_edge`
- `dir3_xtrend_h6_trade_prob`
- `dir3_cycle_h6_fl_prob`
- `dir3_cycle_h6_up_prob`
- `dir3_cycle_h6_dn_prob`
- `dir3_cycle_h6_confidence`
- `dir3_cycle_h6_side_edge`
- `dir3_cycle_h6_trade_prob`
- `dir3_cycle_h6_group_support`
- `dir3_chartcnn_h6_fl_prob`
- `dir3_chartcnn_h6_up_prob`
- `dir3_chartcnn_h6_dn_prob`
- `dir3_chartcnn_h6_confidence`
- `dir3_chartcnn_h6_side_edge`
- `dir3_chartcnn_h6_trade_prob`

### Chronos Distribution / Uncertainty Context

Role: distributional forecast width, uncertainty, and large-move risk.
Recommended consumers: risk-template, entry-quality, exit/hazard. Direction
ownership is not approved unless embedded inside the confirmed Direction Head
contract.

- `chronos_h6_q10`
- `chronos_h6_q50`
- `chronos_h6_q90`
- `chronos_h6_width`
- `chronos_h6_mean`
- `chronos_unc_atr14_q10`
- `chronos_unc_atr14_q50`
- `chronos_unc_atr14_q90`
- `chronos_unc_atr14_width`
- `chronos_unc_atr14_mean`
- `chronos_unc_atr14_width_ewm3`
- `chronos_unc_atr14_width_ewm6`
- `chronos_unc_rv24_q10`
- `chronos_unc_rv24_q50`
- `chronos_unc_rv24_q90`
- `chronos_unc_rv24_width`
- `chronos_unc_rv24_mean`
- `chronos_unc_rv24_width_ewm3`
- `chronos_unc_rv24_width_ewm6`
- `chronos_atr14_upside_band_ewm3`
- `chronos_atr14_width_ewm6`
- `chronos_atr14_width`
- `chronos_atr14_large_move_score`
- `chronos_realized_vol24_width`
- `chronos_realized_vol24_large_move_score`

### Confirmed Omega1 Direction-Head Outputs

Role: Layer-2-to-downstream direction output from the confirmed
`core_plus_tsfm_chronos` Direction Head. These may feed downstream Layer-3/4
models, but must not feed back into upstream AI/TSFM, M7, Regime3, or DIR3
generators.

- `omega1_tsfm_chronos_p_cash`
- `omega1_tsfm_chronos_p_long`
- `omega1_tsfm_chronos_p_short`
- `omega1_tsfm_chronos_confidence`
- `omega1_tsfm_chronos_side_edge`
- `omega1_tsfm_chronos_trade_prob`
- `omega1_tsfm_chronos_action`

### Retired Teacher Outputs

Role: historical research only. Active Omega1 paths must fail fast if these
appear in active input contracts.

- `teacher_hgb_p_cash`
- `teacher_hgb_p_long`
- `teacher_hgb_p_short`
- `teacher_hgb_confidence`
- `teacher_hgb_side_edge`
- `teacher_hgb_uncertainty`
- `teacher_hgb_risk_veto_score`
- `teacher_mamba_p_cash`
- `teacher_mamba_p_long`
- `teacher_mamba_p_short`
- `teacher_mamba_confidence`
- `teacher_mamba_side_edge`
- `teacher_mamba_uncertainty`
- `teacher_mamba_risk_veto_score`

## Omega1 Role-Based Feature Groups - 2026-06-02

This is the primary grouping to use when designing Omega1 model heads. It
answers "what is this feature useful for?" rather than "where did it come
from?". Provenance groups above still matter for audits; role groups below
drive model input routing.

If a feature ablation, model test, or red-team audit changes a feature's role,
update this section in the same change. Do not leave successful feature-routing
changes only in experiment reports.

### Direction Group

Purpose: long / short / cash swing direction and side confidence.

Primary consumers:

- Direction Head
- Regime expert action heads
- Layer-3 parent/meta-policy

Core features:

- `dir3_vsnlstm_h6_fl_prob`
- `dir3_vsnlstm_h6_up_prob`
- `dir3_vsnlstm_h6_dn_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_fl_prob`
- `dir3_patch_h6_up_prob`
- `dir3_patch_h6_dn_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`
- `omega1_tsfm_chronos_p_cash`
- `omega1_tsfm_chronos_p_long`
- `omega1_tsfm_chronos_p_short`
- `omega1_tsfm_chronos_confidence`
- `omega1_tsfm_chronos_side_edge`
- `omega1_tsfm_chronos_trade_prob`
- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`
- `ai_dir_edge`
- `ai_dir_p_up`
- `ai_dir_p_down`
- `ai_dir_p_flat`
- `ai_dir_entropy`
- `log_return`
- `rsi`
- `macd_hist`
- `hma_slope`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `breakout_strength`
- `dual_momentum`
- `mean_reversion_z`
- `smart_money_flow`
- `net_taker_ratio`
- `ofi_acceleration`

Secondary / research features:

- `dir3_duet_h6_fl_prob`
- `dir3_duet_h6_up_prob`
- `dir3_duet_h6_dn_prob`
- `dir3_duet_h6_confidence`
- `dir3_duet_h6_side_edge`
- `dir3_duet_h6_trade_prob`
- `dir3_cryptomamba_h6_fl_prob`
- `dir3_cryptomamba_h6_up_prob`
- `dir3_cryptomamba_h6_dn_prob`
- `dir3_cryptomamba_h6_confidence`
- `dir3_cryptomamba_h6_side_edge`
- `dir3_cryptomamba_h6_trade_prob`
- `dir3_retrieval_h6_fl_prob`
- `dir3_retrieval_h6_up_prob`
- `dir3_retrieval_h6_dn_prob`
- `dir3_retrieval_h6_confidence`
- `dir3_retrieval_h6_side_edge`
- `dir3_retrieval_h6_trade_prob`
- `dir3_retrieval_h6_neighbor_edge_mean`
- `dir3_retrieval_h6_neighbor_edge_q25`
- `dir3_retrieval_h6_neighbor_edge_q75`
- `dir3_retrieval_h6_regime_consensus`
- `dir3_retrieval_h6_similarity_score`

Do not use ordinal action columns as default direction inputs. Any promotion of
`*_action` columns requires a separate contract update.

### Entry Quality Group

Purpose: decide whether a direction candidate is executable now without
immediate whipsaw, liquidity-trap, or SL-first failure.

Primary consumers:

- Entry Quality Head
- Final veto / governor
- Regime expert entry filters

Core features:

- Direction Group outputs for the selected side
- `amihud_illiquidity_z`
- `liquidity_vacuum`
- `execution_quality`
- `sig_liquidity_trap`
- `fvg_dist`
- `cvp_poc_dist`
- `cvp_cluster_position`
- `cvp_volume_imbalance`
- `cvp_vah_val_width`
- `squeeze_power`
- `sig_ai_squeeze`
- `sig_oi_divergence`
- `sig_trend_health`
- `sig_volume_confirm`
- `trade_intensity`
- `big_trade_ratio`
- `taker_acceleration`
- `whale_retail_ratio`
- `whale_conviction`
- `chronos_unc_atr14_width`
- `chronos_unc_atr14_width_ewm3`
- `chronos_unc_rv24_width`
- `chronos_unc_rv24_width_ewm3`

Execution-quality labels may be ZigZag-derived or template-compatibility
derived, but they must not replace the Direction Group's `zigzag_action`
authority.

### Risk / Volatility Group

Purpose: sizing, leverage/notional, TP/SL width, max-hold, cooldown, and
portfolio risk throttling.

Primary consumers:

- Risk Template Selector
- Sizing / leverage layer
- Exit / hazard layer
- Governor

Core features:

- `realized_vol_ratio`
- `volatility_z`
- `garch_vol_z`
- `bb_width`
- `bb_width_z`
- `garman_klass_vol`
- `rogers_satchell_vol`
- `parkinson_vol`
- `realized_skewness`
- `jump_z`
- `jump_flag`
- `evt_tail_flag`
- `evt_excess_z`
- `ai_adverse_risk`
- `ai_reward_risk`
- `ai_vol_regime_pct`
- `tide_vol_raw`
- `tide_vol_zscore`
- `chronos_h6_q10`
- `chronos_h6_q50`
- `chronos_h6_q90`
- `chronos_h6_width`
- `chronos_h6_mean`
- `chronos_unc_atr14_q10`
- `chronos_unc_atr14_q50`
- `chronos_unc_atr14_q90`
- `chronos_unc_atr14_width`
- `chronos_unc_atr14_mean`
- `chronos_unc_atr14_width_ewm3`
- `chronos_unc_atr14_width_ewm6`
- `chronos_unc_rv24_q10`
- `chronos_unc_rv24_q50`
- `chronos_unc_rv24_q90`
- `chronos_unc_rv24_width`
- `chronos_unc_rv24_mean`
- `chronos_unc_rv24_width_ewm3`
- `chronos_unc_rv24_width_ewm6`
- `m7_q10`
- `m7_q90`
- `m7_qwidth`

### Regime / Context Group

Purpose: current market-state routing, expert selection, and regime-aware
parameter conditioning.

Primary consumers:

- Regime Router
- MoE expert selector
- Risk Template Selector
- Governor

Core features:

- `regime3_current_sensitive_wide24_bull_prob`
- `regime3_current_sensitive_wide24_bear_prob`
- `regime3_current_sensitive_wide24_chop_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_entropy`
- `regime3_current_sensitive_wide24_margin`
- `regime3_cmamba_h6_sidecar_bull_prob`
- `regime3_cmamba_h6_sidecar_bear_prob`
- `regime3_cmamba_h6_sidecar_chop_prob`
- `regime3_cmamba_h6_sidecar_confidence`
- `regime3_cmamba_h6_sidecar_transition_prob`
- `regime3_cmamba_h6_sidecar_stability_score`
- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`
- `cvp_regime`
- `regime_trending`
- `chop_index`
- `hurst_48`

Prediction-time `future`-named Regime3 sidecar outputs may be used only by
exact column name as documented above. Broad `future` selection remains
forbidden.

### Execution / Liquidity Group

Purpose: avoid bad fills, spread/liquidity traps, and noisy microstructure
entries.

Primary consumers:

- Entry Quality Head
- Limit/market execution selector
- Governor

Core features:

- `amihud_illiquidity_z`
- `liquidity_vacuum`
- `execution_quality`
- `trade_intensity`
- `big_trade_ratio`
- `taker_acceleration`
- `cvp_volume_imbalance`
- `cvp_poc_dist`
- `sig_liquidity_trap`
- `squeeze_power`
- `volume`
- `quote_volume`
- `trades`
- `taker_buy_base`
- `taker_buy_quote`

### Position / Exit Group

Purpose: monitor open-position risk and decide hold / reduce / close. The
current inventory mostly contains pre-entry features; live/runtime position
state must be appended by the execution environment, not backfilled into
historical feature generators.

Primary consumers:

- Exit / Hazard Head
- Governor

Current feature inputs:

- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`
- `regime3_cmamba_h6_sidecar_transition_prob`
- `regime3_cmamba_h6_sidecar_stability_score`
- `chronos_unc_atr14_width`
- `chronos_unc_rv24_width`
- `ai_adverse_risk`
- `liquidity_vacuum`
- `execution_quality`

Runtime-only state to append outside feature generation:

- current position side
- entry price
- unrealized PnL
- hold bars
- current notional / leverage
- active TP/SL distance
- recent trade count / cooldown state

### Time / Session Group

Purpose: session effects, funding windows, and repeated intraday behavior.

Primary consumers:

- Direction Head
- Entry Quality Head
- Governor

Core features:

- `hour_sin`
- `hour_cos`
- `minute_sin`
- `minute_cos`
- `session_europe`
- `session_us`
- `is_hour_open`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`

### Meta / Reliability Group

Purpose: model confidence, uncertainty, disagreement, and veto reliability.

Primary consumers:

- Meta Governor
- Fallback selector
- Risk Template Selector

Core features:

- `omega1_tsfm_chronos_confidence`
- `omega1_tsfm_chronos_side_edge`
- `omega1_tsfm_chronos_trade_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_entropy`
- `regime3_current_sensitive_wide24_margin`
- `regime3_cmamba_h6_sidecar_confidence`
- `chronos_h6_width`
- `chronos_unc_atr14_width`
- `chronos_unc_rv24_width`
- `ai_dir_entropy`

### Head-To-Role Input Map

Use this as the default starting point for future Omega1 tests:

| Head / Layer | Primary Role Groups | Secondary Role Groups |
| --- | --- | --- |
| Direction Head | Direction, Regime / Context | Time / Session, Order Flow / Whale / Smart Money |
| Entry Quality Head | Entry Quality, Execution / Liquidity, Risk / Volatility | Direction outputs, Regime / Context |
| Regime Router | Regime / Context | Risk / Volatility, Time / Session |
| Risk Template Selector | Risk / Volatility, Regime / Context, Funding / Perpetual Positioning | Meta / Reliability |
| Exit / Hazard | Position / Exit, Risk / Volatility, Regime / Context | Execution / Liquidity |
| Meta Governor | Meta / Reliability, Risk / Volatility, Execution / Liquidity | Direction outputs, runtime trade-count state |


## Omega1 Supervised Label Authority

Omega1's active supervised label source is:

- Label artifact root:
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531`
- Label column: `zigzag_action`
- Classes: `0=CASH`, `1=LONG`, `2=SHORT`
- Soft labels: `zigzag_soft_cash`, `zigzag_soft_long`,
  `zigzag_soft_short`

This label contract applies to every Omega1 supervised action, direction,
entry, parent, expert, teacher, and meta-policy head. Older Alpha labels and
derived targets can be used only for historical comparison or non-active
research baselines, not as active Omega1 supervised training targets.

Explicitly forbidden active Omega1 supervised targets:

- `tp_sl_action_score`
- `wave3_action`
- Alpha lifecycle / `FullyLearnedGovernor` path labels
- fixed-barrier Alpha6 action labels
- Alpha5/Alpha7 parent labels generated outside the active ZigZag contract
- realized future PnL, target, or future-path columns used directly as model
  inputs

Architecture reuse rule: Alpha models can be treated as design references.
For example, Omega1 may reuse a Regime MoE frame, parent/fallback routing
pattern, risk-template idea, or backtest utility. It must retrain supervised
heads from `zigzag_action` and must not load an Alpha parent/governor model as
an Omega1 active supervised component.

Risk/TP/SL/notional layers are allowed only when they are either:

1. rule/template/search layers selected by validation without supervised
   labels, or
2. supervised models retrained from the same active `zigzag_action` contract
   or an explicitly documented ZigZag-derived soft target.

Any code path that silently falls back to a TP/SL path label, wave label, or
legacy Alpha label must fail fast in active Omega1 experiments.

## Canonical Data Artifacts

Base M7 + ZigZag direction candidate files:

- `data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv`
- `data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv`

Integration audit:

- `docs/audits/m7_zigzag_direction_integration_20260531.md`
- `tmp/causal_regen_20260516/zigzag_m7_direction_integration_20260531/summary.json`

ZigZag action model zoo:

- `docs/audits/zigzag_action_model_zoo_20260531.md`
- `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531/zigzag_action_model_zoo_summary.json`

ZigZag second-stage comparison:

- `docs/audits/zigzag_second_stage_retrain_20260531.md`
- `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/zigzag_second_stage_retrain_all_summary.json`

Regime3 CryptoMamba h6 future-context sidecar:

- Active artifact:
  `data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531`
- Active report:
  `data/ensemble/reports/regime3_cryptomamba_pred_h6_nocurrent_20260531_report.json`
- Previous docs-rolled-64 backup:
  `data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531_docsrolled64_backup_20260531`
- Feature pack: `all_sanitized`
- Feature count: `128`
- Model: CryptoMamba C-Block Merge, h6 future Regime3 prediction.
- 2026 OOS metrics: balanced accuracy `0.672556`, OVR AUC `0.843823`,
  transition AUC `0.695492`.
- Contract: current Regime3 probabilities are used only for target/evaluation,
  not as model inputs. `teacher_*`, `m7_*`, `a5dir_*`, Regime4, target, label,
  future, realized PnL, ZigZag, and wave columns remain forbidden inputs.

Layer 2 direction feature generators (legacy `dir3_*` artifact prefix):

- `dir3_retrieval`: `data/ensemble/supervised/omega1_dir3_retrieval_20260531`
- `dir3_retrieval` audit:
  `tmp/causal_regen_20260516/omega1_dir3_retrieval_20260531/dir3_retrieval_audit.json`
- `dir3_cycle`: `data/ensemble/supervised/omega1_dir3_cycle_20260531`
- `dir3_cycle` audit:
  `tmp/causal_regen_20260516/omega1_dir3_cycle_20260531/dir3_cycle_audit.json`
- `dir3_chartcnn`: `data/ensemble/supervised/omega1_dir3_chartcnn_20260531`
- `dir3_patch`: `data/ensemble/supervised/omega1_dir3_patch_20260531`
- `dir3_duet`: `data/ensemble/supervised/omega1_dir3_duet_20260531`
- Remaining-generator audit:
  `tmp/causal_regen_20260516/omega1_dir3_remaining_20260531/dir3_remaining_audit.json`
- Retrieval/cycle combined parent/meta probe:
  `tmp/causal_regen_20260516/omega1_dir3_combined_meta_probe_20260531/combined_meta_probe_summary.json`
- Remaining-generator parent/meta probe:
  `tmp/causal_regen_20260516/omega1_dir3_remaining_20260531/remaining_meta_probe_summary.json`
- `dir3_vsnlstm`: `data/ensemble/supervised/omega1_dir3_vsnlstm_20260531`
- `dir3_lpatchtst`: `data/ensemble/supervised/omega1_dir3_lpatchtst_20260531`
- `dir3_xtrend`: `data/ensemble/supervised/omega1_dir3_xtrend_20260531`
- Financial-paper-inspired audit:
  `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/dir3_finpaper_audit.json`
- Financial-paper-inspired parent/meta probe:
  `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/finpaper_meta_probe_summary.json`
- `dir3_cryptomamba`: `data/ensemble/supervised/omega1_dir3_cryptomamba_20260531`
- CryptoMamba direction audit:
  `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`
- CryptoMamba direction parent/meta probe:
  `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/cryptomamba_meta_probe_summary.json`
- Top2 full sweep:
  `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_sweep_summary.json`
- Top2 full-sweep parent/meta probe:
  `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_meta_probe_summary.json`
- `dir3_patch_full`: `data/ensemble/supervised/omega1_dir3_patch_full_20260531`
- `dir3_vsnlstm_full`: `data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531`

## Omega1 Architecture Feature Registry

This contract tracks all processed feature families that may
enter any Omega1 architecture layer. A feature can be approved for one consumer
layer and still be blocked from another. Do not interpret teacher approval as
global approval, and do not interpret non-teacher status as exclusion from the
overall Omega1 architecture.

Current consumer-layer categories:

- `teacher_generation`: retired. Existing teacher artifacts are historical
  only and are not allowed active Omega1 inputs.
- `parent_policy`: primary action / entry / meta-policy models.
- `risk_sizing_exit`: sizing, leverage/notional, TP/SL, exit, veto, cooldown,
  and risk-template layers.
- `diagnostics_only`: analysis and monitoring only.
- `research_only`: not active until a new promotion audit is written.

These consumer categories are permissions, not training-stage names. The
training-stage authority is the Omega1 Layer Contract above.

Default rule: every processed feature must be listed explicitly with an allowed
consumer layer. Broad prefix selection is forbidden.

### Architecture-Approved Processed Features

| Family | Columns | Allowed Consumers | Notes |
| --- | --- | --- | --- |
| AI / TSFM risk context | `ai_adverse_risk`, `ai_reward_risk`, `ai_vol_regime_pct`, `tide_vol_zscore` | `parent_policy`, `risk_sizing_exit` | Risk/volatility context only, not direct direction owner. |
| Chronos uncertainty / large-move context | `chronos_atr14_upside_band_ewm3`, `chronos_atr14_width_ewm6`, `chronos_atr14_width`, `chronos_atr14_large_move_score`, `chronos_realized_vol24_width`, `chronos_realized_vol24_large_move_score` | `risk_sizing_exit`, `diagnostics_only` | Uncertainty/range context. |
| Regime3 current context | `regime3_current_sensitive_wide24_bull_prob`, `regime3_current_sensitive_wide24_bear_prob`, `regime3_current_sensitive_wide24_chop_prob`, `regime3_current_sensitive_wide24_confidence`, `regime3_current_sensitive_wide24_entropy`, `regime3_current_sensitive_wide24_margin` | `parent_policy`, `risk_sizing_exit` | Current market-structure context. |
| Regime3 h6 stability / risk sidecar | `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, `regime3_churn_h6_risk_score` | `risk_sizing_exit`, `diagnostics_only` | Transition/churn risk context. |
| Regime3 CryptoMamba h6 future-context sidecar | `regime3_cmamba_h6_future_bull_prob`, `regime3_cmamba_h6_future_bear_prob`, `regime3_cmamba_h6_future_chop_prob`, `regime3_cmamba_h6_confidence`, `regime3_cmamba_h6_transition_prob`, `regime3_cmamba_h6_stability_score` | `parent_policy`, `risk_sizing_exit` | Prediction generated at `t`; exact-column `future` exception only. |
| Split-local current context | `cvp_regime`, `regime_trending` | `parent_policy`, `risk_sizing_exit` | Current-bar context. |
| M7 quantile-risk context | `m7_q10`, `m7_q90`, `m7_qwidth` | `risk_sizing_exit` | Non-target quantile-risk context only. |
| M7 ZigZag direction context | `m7_zigzag_cat_fl`, `m7_zigzag_cat_up`, `m7_zigzag_cat_dn`, `m7_zigzag_cat_confidence`, `m7_zigzag_cat_side_edge`, `m7_zigzag_cat_trade_prob`, `m7_zigzag_xgb_fl`, `m7_zigzag_xgb_up`, `m7_zigzag_xgb_dn`, `m7_zigzag_xgb_confidence`, `m7_zigzag_xgb_side_edge`, `m7_zigzag_xgb_trade_prob` | `parent_policy`, `risk_sizing_exit` | Retrained on active ZigZag 3-class labels; ordinal action columns blocked. |
| DIR3 retrieval direction context | `dir3_retrieval_h6_fl_prob`, `dir3_retrieval_h6_up_prob`, `dir3_retrieval_h6_dn_prob`, `dir3_retrieval_h6_confidence`, `dir3_retrieval_h6_side_edge`, `dir3_retrieval_h6_trade_prob`, `dir3_retrieval_h6_neighbor_edge_mean`, `dir3_retrieval_h6_neighbor_edge_q25`, `dir3_retrieval_h6_neighbor_edge_q75`, `dir3_retrieval_h6_regime_consensus`, `dir3_retrieval_h6_similarity_score` | `parent_policy`, `risk_sizing_exit`, `diagnostics_only` | Layer 2 standalone direction generator despite legacy prefix. Do not feed into teacher generation without an OOF stacking contract. |
| DIR3 cycle/session context | `dir3_cycle_h6_fl_prob`, `dir3_cycle_h6_up_prob`, `dir3_cycle_h6_dn_prob`, `dir3_cycle_h6_confidence`, `dir3_cycle_h6_side_edge`, `dir3_cycle_h6_trade_prob`, `dir3_cycle_h6_group_support` | `diagnostics_only` | Did not improve combined parent/meta probe; retain for analysis only. |
| DIR3 chart-CNN context | `dir3_chartcnn_h6_fl_prob`, `dir3_chartcnn_h6_up_prob`, `dir3_chartcnn_h6_dn_prob`, `dir3_chartcnn_h6_confidence`, `dir3_chartcnn_h6_side_edge`, `dir3_chartcnn_h6_trade_prob` | `diagnostics_only` | Light Conv1D chart-pattern probe. Standalone and combined probes underperformed Omega1 core. |
| DIR3 patch direction context | `dir3_patch_h6_fl_prob`, `dir3_patch_h6_up_prob`, `dir3_patch_h6_dn_prob`, `dir3_patch_h6_confidence`, `dir3_patch_h6_side_edge`, `dir3_patch_h6_trade_prob` | `parent_policy`, `risk_sizing_exit`, `diagnostics_only` | Best DIR3 batch candidate. Combined probe improved core bacc and proxy WR. |
| DIR3 DUET-style context | `dir3_duet_h6_fl_prob`, `dir3_duet_h6_up_prob`, `dir3_duet_h6_dn_prob`, `dir3_duet_h6_confidence`, `dir3_duet_h6_side_edge`, `dir3_duet_h6_trade_prob` | `parent_policy`, `diagnostics_only` | Correlation-cluster compression probe. Standalone was useful, but combined WR trailed `dir3_patch`; keep below patch. |
| DIR3 VSN-LSTM context | `dir3_vsnlstm_h6_fl_prob`, `dir3_vsnlstm_h6_up_prob`, `dir3_vsnlstm_h6_dn_prob`, `dir3_vsnlstm_h6_confidence`, `dir3_vsnlstm_h6_side_edge`, `dir3_vsnlstm_h6_trade_prob` | `parent_policy`, `risk_sizing_exit`, `diagnostics_only` | Oxford benchmark-inspired variable-selection LSTM. Improved Omega1 core probe; compare against `dir3_patch` in downstream PnL ablation. |
| DIR3 Lightweight PatchTST context | `dir3_lpatchtst_h6_fl_prob`, `dir3_lpatchtst_h6_up_prob`, `dir3_lpatchtst_h6_dn_prob`, `dir3_lpatchtst_h6_confidence`, `dir3_lpatchtst_h6_side_edge`, `dir3_lpatchtst_h6_trade_prob` | `parent_policy`, `diagnostics_only` | Oxford/PatchTST-inspired sequence encoder. Weaker than VSN-LSTM and HGB patch in this run. |
| DIR3 X-Trend context | `dir3_xtrend_h6_fl_prob`, `dir3_xtrend_h6_up_prob`, `dir3_xtrend_h6_dn_prob`, `dir3_xtrend_h6_confidence`, `dir3_xtrend_h6_side_edge`, `dir3_xtrend_h6_trade_prob` | `diagnostics_only` | X-Trend-inspired context-set retrieval. Did not improve the Omega1 core probe in this implementation. |
| DIR3 CryptoMamba direction context | `dir3_cryptomamba_h6_fl_prob`, `dir3_cryptomamba_h6_up_prob`, `dir3_cryptomamba_h6_dn_prob`, `dir3_cryptomamba_h6_confidence`, `dir3_cryptomamba_h6_side_edge`, `dir3_cryptomamba_h6_trade_prob` | `parent_policy`, `diagnostics_only` | Regime3 CryptoMamba C-Block architecture retargeted to ZigZag direction. Small core-probe lift; below `dir3_patch` and `dir3_vsnlstm`. |
| Teacher outputs | `teacher_hgb_p_cash`, `teacher_hgb_p_long`, `teacher_hgb_p_short`, `teacher_hgb_confidence`, `teacher_hgb_side_edge`, `teacher_hgb_uncertainty`, `teacher_hgb_risk_veto_score`, `teacher_mamba_p_cash`, `teacher_mamba_p_long`, `teacher_mamba_p_short`, `teacher_mamba_confidence`, `teacher_mamba_side_edge`, `teacher_mamba_uncertainty`, `teacher_mamba_risk_veto_score` | `research_only` | Retired by user decision. Do not use in active Omega1 parent/risk/final-policy inputs. |

## Retired Teacher Generation Consumer Subset

Teacher generation is retired for active Omega1 modeling. The lists below are
kept only to preserve historical audit context for old HGB/Mamba teacher
artifacts. They are not active input permissions.

### AI / TSFM Risk Context

- `ai_adverse_risk`
- `ai_reward_risk`
- `ai_vol_regime_pct`
- `tide_vol_zscore`

### Chronos Uncertainty / Large-Move Context

- `chronos_atr14_upside_band_ewm3`
- `chronos_atr14_width_ewm6`
- `chronos_atr14_width`
- `chronos_atr14_large_move_score`
- `chronos_realized_vol24_width`
- `chronos_realized_vol24_large_move_score`

### Regime3 Current Context

- `regime3_current_sensitive_wide24_bull_prob`
- `regime3_current_sensitive_wide24_bear_prob`
- `regime3_current_sensitive_wide24_chop_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_entropy`
- `regime3_current_sensitive_wide24_margin`

### Regime3 h6 Stability / Risk Sidecar

- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`

### Regime3 CryptoMamba h6 Future-Context Sidecar

These are prediction-time sidecar outputs generated from information available
at timestamp `t`. They are allowed only by exact column name; broad `future`
feature selection remains forbidden.

- `regime3_cmamba_h6_future_bull_prob`
- `regime3_cmamba_h6_future_bear_prob`
- `regime3_cmamba_h6_future_chop_prob`
- `regime3_cmamba_h6_confidence`
- `regime3_cmamba_h6_transition_prob`
- `regime3_cmamba_h6_stability_score`

Not included as teacher inputs:

- `regime3_cmamba_h6_future_pred_id`
- `regime3_cmamba_h6_future_pred_name`

### Split-Local Current Context

- `cvp_regime`
- `regime_trending`

### M7 Quantile-Risk Context

- `m7_q10`
- `m7_q90`
- `m7_qwidth`

### M7 ZigZag Direction Context

These direction features are produced by models retrained on the active ZigZag
3-class action label, then integrated into the M7 namespace with an exact
timestamp contract.

- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`

Not included as teacher inputs:

- `m7_zigzag_cat_action`
- `m7_zigzag_xgb_action`

Reason: the action columns are ordinal encodings of class decisions. Omega1
teacher uses calibrated probability/edge/context fields instead.

## M7 Usage Status

The new M7+ZigZag CSV files preserve legacy M7 columns so old experiments can
still be audited. Preservation is not usage approval.

### Retired Omega1 Teacher Inputs

The previous M7 teacher-generation inputs were only the explicit subset listed
below. Teacher generation is now retired for active Omega1 modeling:

- `m7_q10`
- `m7_q90`
- `m7_qwidth`
- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`

No other `m7_*` column is implicitly approved.

### Risk Candidates Only

These may be considered only in downstream parent/risk/final-policy ablations
with explicit provenance and no teacher-generation feedback:

- `m7_quality_pred`
- `m7_hold_pred`

Current caveat: `m7_quality_pred` and `m7_hold_pred` previously aliased
target-family columns in some artifacts, so they remain blocked from active
teacher generation.

### Legacy Source / Retired Columns

These legacy source columns are preserved for audit but are not approved direct
model inputs.

- `m7_tp_offset`
- `m7_sl_offset`
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_tp_price`
- `m7_sl_price`
- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_target_quality`
- `m7_target_hold`
- `m7_tail_risk`

Retired experimental clean-risk columns:

- `m7_clean_entry_long_ret`
- `m7_clean_entry_short_ret`
- `m7_clean_entry_long_abs_dist`
- `m7_clean_entry_short_abs_dist`
- `m7_clean_entry_mid_ret`
- `m7_clean_entry_band_width`
- `m7_clean_tp_ret`
- `m7_clean_sl_ret`
- `m7_clean_tp_abs_dist`
- `m7_clean_sl_abs_dist`
- `m7_clean_tp_sl_width`
- `m7_clean_tp_sl_rr`

### Historical / Excluded M7 Columns

These must not be required, fabricated, aliased, or silently backfilled in
Omega1 active/live inputs:

- `m7_trend_xgb_dn`
- `m7_trend_xgb_fl`
- `m7_trend_xgb_up`
- `m7_mtl_dn`
- `m7_mtl_fl`
- `m7_mtl_up`
- `m7_quant_dn`
- `m7_quant_fl`
- `m7_quant_up`
- `m7_q50`
- `m7_confidence`
- `m7_action`
- `m7_size`
- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_expected_ret`
- `m7_composite_score`

## New M7 Direction Candidates

These columns were added to fill the missing direction axis in downstream
parent/meta-policy tests.

The probability/edge/confidence fields below remain valid Layer 2 direction
context for parent/risk tests. They are no longer teacher-generation inputs
because teacher features are retired. The ordinal `*_action` fields remain
blocked.

### CatBoost ZigZag Action Candidate

Source model: `alpha_catboost_action_master_like`.

- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_action`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`

### Trend-XGB ZigZag Action Candidate

Source model: `trend_xgb_like_xgb`.

- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_action`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`

Class mapping:

- `fl`: ZigZag CASH probability
- `up`: ZigZag LONG probability
- `dn`: ZigZag SHORT probability

## Retired Teacher Outputs

Teacher outputs are retired by user decision. Historical artifacts remain for
audit, but `teacher_*` and `teacher_oof_*` must not be used in active Omega1
parent/risk/final-policy inputs.

### HGB Teacher

- `teacher_hgb_p_cash`
- `teacher_hgb_p_long`
- `teacher_hgb_p_short`
- `teacher_hgb_confidence`
- `teacher_hgb_side_edge`
- `teacher_hgb_uncertainty`
- `teacher_hgb_risk_veto_score`

Current M7 ZigZag HGB artifact:

- `tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_20260531`
- Label source: `zigzag_action`
- Feature count: `37`
- 2026 OOS label-probe: bacc `0.5637`, OVR AUC `0.7689`

### Mamba Teacher Candidate

- `teacher_mamba_p_cash`
- `teacher_mamba_p_long`
- `teacher_mamba_p_short`
- `teacher_mamba_confidence`
- `teacher_mamba_side_edge`
- `teacher_mamba_uncertainty`
- `teacher_mamba_risk_veto_score`

Current M7 ZigZag Mamba smoke artifact:

- `tmp/causal_regen_20260516/omega1_mamba_teacher_m7zigzag_smoke_20260531`
- Scope: 1 epoch smoke for contract validation, not final selection
- Label source: `zigzag_action`
- Feature count: `127` total = `37` explicit second-stage + `90` base context
- 2026 OOS label-probe: bacc `0.5712`, OVR AUC `0.7639`


## Context-Only / Hold / Fail Families

These families are not active direct action owners.

### Context-Only

- `regime3_all_context`
- `regime3_risk_context`
- `regime3_current_context`
- TimesNet anchor/session context

### Hold / Fail

- PatchTSMixer binary
- PatchTST
- DLinear
- Chronos q50-sign / direction output
- `m7_direction_legacy`
- `m7_all_nonp0`
- `ai_direction_legacy`
- `m7_unsup_risk_context`

## Hard Exclusions

These are not allowed in Omega1 active paths unless a new versioned contract is
created first.

- Regime4 families:
  - `clean_regime4_state24_sticky090_v2_*`
  - `clean_regime4_2024_unsup_v1_*`
  - `clean_regime_2024_unsup_v4_*`
  - `regime4_pred_*`
- `regime3_pred_*`
- `teacher_*` and `teacher_oof_*` as active Omega1 inputs
- `a5dir_*` as teacher/AI/M7/regime-generation inputs
- `label`, `target`, `future`, `pnl`, `action_score` columns
- `zigzag_soft_*` as inputs; these are target labels only
- `wave3_action`; this active label is retired
- `tp_sl_action_score`; this label source is retired for Omega1

## Usage Rules

- M7 ZigZag direction candidates are for downstream parent/meta-policy tests
  first.
- Do not feed `m7_zigzag_*` or `dir3_*` into teacher generation. Teacher
  generation is retired. Current standalone `dir3_*` features are Layer 2 OOS
  direction features despite the legacy prefix.
- Regime3 CryptoMamba confidence decoder audit on 2025->2026 did not justify a
  promoted decoded class/id transform. Use
  `regime3_cmamba_h6_confidence`, `regime3_cmamba_h6_transition_prob`, and
  `regime3_cmamba_h6_stability_score` as reliability/risk context, not as a
  hard replacement for raw probability outputs.
- If a feature changes status, update the Change Log and the relevant section
  in this document.

## Current Directional Feature Set

`teacher_*` is retired. The current Omega1 directional feature set is limited
to Layer 2 OOS direction/regime-context features below.

## Confirmed Omega1 Direction Head

As of 2026-06-02, the Omega1 Direction Head is fixed to the
`core_plus_tsfm_chronos + volatility_pca06` input contract below.

- Script:
  `scripts/train_omega1_direction_head_volatility_pca_20260602.py`
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06`
- Model:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/volatility_pca06_omega1_direction_volpca.cbm`
- PCA / feature contract:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/volatility_pca06_omega1_direction_volpca_contract.joblib`
- Target:
  `zigzag_action`
- Output columns:
  `omega1_dir_volpca_p_cash`, `omega1_dir_volpca_p_long`,
  `omega1_dir_volpca_p_short`, `omega1_dir_volpca_confidence`,
  `omega1_dir_volpca_side_edge`, `omega1_dir_volpca_trade_prob`,
  `omega1_dir_volpca_action`
- 2025 OOF output:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/training_features_2025_volatility_pca06_omega1_direction_volpca_oof_20260602.csv`
- 2026 score output:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/training_features_2026_rebuilt_volatility_pca06_omega1_direction_volpca_20260602.csv`
- 2026 OOS metrics:
  BAcc `0.6052`, OVR AUC `0.7917`, proxy WR `66.27%`, proxy trades `13245`.

Confirmed input features, 61 total:

- `dir3_vsnlstm_h6_fl_prob`
- `dir3_vsnlstm_h6_up_prob`
- `dir3_vsnlstm_h6_dn_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_fl_prob`
- `dir3_patch_h6_up_prob`
- `dir3_patch_h6_dn_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`
- `ai_dir_edge`
- `ai_dir_p_up`
- `ai_dir_p_down`
- `ai_dir_p_flat`
- `ai_dir_entropy`
- `patchtst_median`
- `patchtst_regime_sim`
- `ai_adverse_risk`
- `ai_reward_risk`
- `ai_vol_regime_pct`
- `tide_vol_raw`
- `tide_vol_zscore`
- `ai_flow_pressure`
- `ai_flow_exhaustion`
- `ai_flow_flip_prob`
- `ai_flow_slope`
- `dlinear_smf_ema`
- `dlinear_smf_slope`
- `ai_anchor_revert_prob`
- `ai_anchor_overheat`
- `ai_anchor_trend_escape_prob`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`
- `chronos_h6_q10`
- `chronos_h6_q50`
- `chronos_h6_q90`
- `chronos_h6_width`
- `chronos_h6_mean`
- `chronos_unc_atr14_q10`
- `chronos_unc_atr14_q50`
- `chronos_unc_atr14_q90`
- `chronos_unc_atr14_width`
- `chronos_unc_atr14_mean`
- `chronos_unc_atr14_width_ewm3`
- `chronos_unc_atr14_width_ewm6`
- `chronos_unc_rv24_q10`
- `chronos_unc_rv24_q50`
- `chronos_unc_rv24_q90`
- `chronos_unc_rv24_width`
- `chronos_unc_rv24_mean`
- `chronos_unc_rv24_width_ewm3`
- `chronos_unc_rv24_width_ewm6`
- `pca_volatility_01`
- `pca_volatility_02`
- `pca_volatility_03`
- `pca_volatility_04`
- `pca_volatility_05`
- `pca_volatility_06`

Volatility PCA source contract:

- PCA policy: only the explicit volatility context group is compressed. The
  55 `core_plus_tsfm_chronos` features stay raw because core-group PCA ablation
  did not beat the confirmed candidate.
- OOF rule: PCA must be fit only inside each expanding 2025 OOF fold.
- Final scoring rule: PCA must be fit on 2025 only, then applied to 2026 by
  exact timestamp frame construction. No 2026 fit, refit, or validation-ranked
  transform is allowed.
- PCA components: `6`
- 2025 final explained variance sum: `0.7563`
- Source columns:
  - `log_return`
  - `volatility_z`
  - `bb_width`
  - `bb_width_z`
  - `garman_klass_vol`
  - `realized_vol_ratio`
  - `rogers_satchell_vol`
  - `parkinson_vol`
  - `bb_width_pct_rank_288`
  - `atr_pct_rank_288`
  - `compression_score`
  - `compression_release_up`
  - `compression_release_down`
  - `garch_vol_z`
  - `jump_flag`
  - `jump_z`
  - `evt_tail_flag`
  - `evt_excess_z`
  - `squeeze_power`
  - `long_squeeze_risk`
  - `short_squeeze_risk`
  - `crowding_pressure`
  - `crowded_long_unwind_risk`
  - `crowded_short_squeeze_risk`

Use restrictions:

- TSFM/Chronos standalone variants are not direction owners. They are approved
  here only as additive context inside the confirmed
  `core_plus_tsfm_chronos + volatility_pca06` Direction Head.
- Do not replace `vsnlstm`, `patch`, TSFM role, Chronos h6, or Chronos
  uncertainty groups with PCA in the active Direction Head. Core-group PCA
  tests did not beat the fixed contract and are diagnostics only.
- `teacher_*`, `teacher_oof_*`, Regime4, previous TP/SL action labels,
  `wave3_action`, realized future PnL/target columns, and silent feature aliases
  remain forbidden.
- A downstream model may consume the Direction Head OOF/score outputs, but must
  not feed them back into AI/TSFM, M7, Regime3, or other upstream feature
  generators.

### Primary Direction Candidates

- `dir3_vsnlstm_h6_fl_prob`
- `dir3_vsnlstm_h6_up_prob`
- `dir3_vsnlstm_h6_dn_prob`
- `dir3_vsnlstm_h6_confidence`
- `dir3_vsnlstm_h6_side_edge`
- `dir3_vsnlstm_h6_trade_prob`
- `dir3_patch_h6_fl_prob`
- `dir3_patch_h6_up_prob`
- `dir3_patch_h6_dn_prob`
- `dir3_patch_h6_confidence`
- `dir3_patch_h6_side_edge`
- `dir3_patch_h6_trade_prob`

### Secondary Direction Candidates

- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`
- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`
- `dir3_duet_h6_fl_prob`
- `dir3_duet_h6_up_prob`
- `dir3_duet_h6_dn_prob`
- `dir3_duet_h6_confidence`
- `dir3_duet_h6_side_edge`
- `dir3_duet_h6_trade_prob`
- `dir3_cryptomamba_h6_fl_prob`
- `dir3_cryptomamba_h6_up_prob`
- `dir3_cryptomamba_h6_dn_prob`
- `dir3_cryptomamba_h6_confidence`
- `dir3_cryptomamba_h6_side_edge`
- `dir3_cryptomamba_h6_trade_prob`
- `dir3_retrieval_h6_fl_prob`
- `dir3_retrieval_h6_up_prob`
- `dir3_retrieval_h6_dn_prob`
- `dir3_retrieval_h6_confidence`
- `dir3_retrieval_h6_side_edge`
- `dir3_retrieval_h6_trade_prob`
- `dir3_retrieval_h6_neighbor_edge_mean`
- `dir3_retrieval_h6_neighbor_edge_q25`
- `dir3_retrieval_h6_neighbor_edge_q75`
- `dir3_retrieval_h6_regime_consensus`
- `dir3_retrieval_h6_similarity_score`

### Regime-Direction Context

- `regime3_current_sensitive_wide24_bull_prob`
- `regime3_current_sensitive_wide24_bear_prob`
- `regime3_current_sensitive_wide24_chop_prob`
- `regime3_current_sensitive_wide24_confidence`
- `regime3_current_sensitive_wide24_margin`
- `regime3_cmamba_h6_future_bull_prob`
- `regime3_cmamba_h6_future_bear_prob`
- `regime3_cmamba_h6_future_chop_prob`
- `regime3_cmamba_h6_confidence`

### Diagnostics-Only Direction Features

- `dir3_lpatchtst_h6_fl_prob`
- `dir3_lpatchtst_h6_up_prob`
- `dir3_lpatchtst_h6_dn_prob`
- `dir3_lpatchtst_h6_confidence`
- `dir3_lpatchtst_h6_side_edge`
- `dir3_lpatchtst_h6_trade_prob`
- `dir3_xtrend_h6_fl_prob`
- `dir3_xtrend_h6_up_prob`
- `dir3_xtrend_h6_dn_prob`
- `dir3_xtrend_h6_confidence`
- `dir3_xtrend_h6_side_edge`
- `dir3_xtrend_h6_trade_prob`
- `dir3_cycle_h6_fl_prob`
- `dir3_cycle_h6_up_prob`
- `dir3_cycle_h6_dn_prob`
- `dir3_cycle_h6_confidence`
- `dir3_cycle_h6_side_edge`
- `dir3_cycle_h6_trade_prob`
- `dir3_chartcnn_h6_fl_prob`
- `dir3_chartcnn_h6_up_prob`
- `dir3_chartcnn_h6_dn_prob`
- `dir3_chartcnn_h6_confidence`
- `dir3_chartcnn_h6_side_edge`
- `dir3_chartcnn_h6_trade_prob`

## Omega1 ZigZag-Only MoE Risk Redesign

Latest contract-compliant risk/execution parameter redesign:

- Script:
  `scripts/eval_alpha7_zigzag_moe_risk_param_sweep_20260601.py`
- Source supervised model artifact:
  `tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_20260601`
- Redesign artifact:
  `tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_risk_redesign_20260601`
- Selected candidate:
  `balanced_rr19_pc0.55_fc0.50_edge0.04_rc0.80_b0.75_r0.90_c0.90`

Selected execution parameters:

- Template `balanced_rr19`: notional `0.45`, leverage `2.0`,
  take-profit `0.026`, stop-loss `0.014`, max-hold `72`, cooldown `6`.
- Primary confidence `0.55`, fallback confidence `0.50`, active edge
  `0.04`, router min confidence `0.80`.
- Expert notional scales: bull `0.75`, bear `0.90`, chop `0.90`.

Results:

- Validation Cost3: PnL `+41.34%`, MDD `-5.61%`, trades `339`,
  trades/day `3.69`, WR `51.03%`.
- 2026 OOS Cost3: PnL `+5.58%`, MDD `-8.52%`, trades `211`,
  trades/day `3.61`, WR `44.55%`.
- Monthly validation Cost3: 2025-10 `+17.15%`, 2025-11 `+1.27%`,
  2025-12 `+12.11%`.
- Monthly 2026 OOS Cost3: 2026-01 `+3.62%`, 2026-02 `+4.99%`.

Interpretation:

- This replaces the earlier ZigZag-only default runtime
  `mid_pc0.55_fc0.50_edge0.08_rc0.80_b0.85_r1.15_c0.90`, which had 2026 OOS
  Cost3 `-13.82%`.
- The supervised label contract did not change. The gain came from making the
  execution template more compatible with ZigZag action labels: smaller
  notional, lower TP/SL distances, shorter max-hold, and lower expert scales.
- This is an Omega1-compatible research candidate, but not yet an active/live
  promotion candidate until monthly stability and walk-forward checks pass.

## Omega1 Regime3 Expert-Local Direction + Quality Research Result

2026-06-02 research branch:

- Script:
  `scripts/train_omega1_regime3_routed_expert_direction_quality_20260602.py`
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602`

Design tested:

- Regime3 Current Router is applied before direction ownership.
- Bull/bear/chop each own a separate Direction Head and Quality Head.
- Direction Head target: `zigzag_action`.
- Quality Head target: `zigzag_action`.
- Quality Head is a 3-class expert-local second-opinion/calibration model,
  not an SL/TP compatibility label.
- Global `omega1_dir_volpca_*` outputs are not expert inputs in this branch.
- Expert base inputs use the same 61-feature
  `core_plus_tsfm_chronos + volatility_pca06` contract.
- Quality Head inputs add expanding-OOF Direction probabilities/action and
  Regime3 route confidence/margin.

Best result:

- Variant: `soft_floor_0p00`
- Selected 2025 OOF quality threshold: `0.45`
- Direction-only 2026 OOS bacc/AUC/proxy WR/trades:
  `0.5983 / 0.7910 / 65.97% / 13463`
- Quality-filtered 2026 OOS bacc/AUC/proxy WR/trades:
  `0.5832 / 0.7220 / 66.44% / 12276`

Decision:

- Not promoted.
- Active Direction Head remains the global `omega1_dir_volpca_*`
  `core_plus_tsfm_chronos + volatility_pca06` model because its 2026 OOS
  reference is stronger on bacc/AUC/trade retention:
  `0.6052 / 0.7917 / 66.27% / 13245`.
- The expert-local Quality layer marginally improves proxy WR but loses too
  much bacc/AUC and cuts trades below the active baseline.

Follow-up Cost replay:

- Script:
  `scripts/eval_omega1_regime3_expertdq_risk_replay_20260602.py`
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_expertdq_risk_replay_20260602`
- Execution template used for replay:
  `balanced_rr19`, notional `0.45`, leverage `2.0`, TP `0.026`, SL `0.014`,
  max-hold `72`, cooldown `6`, expert scales bull `0.75`, bear `0.90`,
  chop `0.90`.
- Best OOS replay row:
  `soft_floor_0p10`, OOS Cost3 `+8.29%`, MDD `-7.86%`, trades `211`,
  WR `54.03%`.
- Active common-window OOS Cost3 reference:
  `+4.51%`, MDD `-8.69%`, trades `211`, WR `46.92%`.
- Blocking issue:
  `soft_floor_0p10` validation Cost3 is `-2.19%` with MDD `-18.46%`,
  while the active validation Cost3 remains `+41.34%` with MDD `-5.61%`.
- Contract decision:
  not promoted. The Cost replay does not override the active Direction Head
  contract because the improvement is OOS-only and validation is unstable.

### Omega1 CatBoost ExpertDQ Quality Head Input Contract Decision

2026-06-02 decision:

- This section documents the Omega1 CatBoost ExpertDQ baseline only.
- Omega1.1 is not this branch. Omega1.1 is the TabM ExpertDQ contract:
  `docs/model_contracts/omega1_1_tabm_expertdq_20260602_contract.md`.
- CatBoost Quality Head input contract is fixed to the existing 70-feature
  contract for historical Omega1 baseline comparison.
- This is the only accepted Omega1 CatBoost ExpertDQ Quality Head input
  contract unless a new explicitly named CatBoost baseline contract is opened.
- Do not add compatibility aliases, fallback prefixes, or silent feature
  expansion to the Omega1 CatBoost ExpertDQ baseline path.

The fixed 70 inputs are:

- 61 Direction Head base inputs:
  `core_plus_tsfm_chronos + volatility_pca06`.
- 7 expanding-OOF Direction outputs:
  `direction_p_cash`, `direction_p_long`, `direction_p_short`,
  `direction_confidence`, `direction_side_edge`, `direction_trade_prob`,
  `direction_action`.
- 2 Regime3 router calibration fields:
  `router_confidence`, `router_margin`.

Quality Head ablation result:

- Script:
  `scripts/test_omega1_expertdq_quality_feature_groups_20260602.py`
- Artifact:
  `tmp/causal_regen_20260516/omega1_expertdq_quality_feature_groups_20260602`
- Ranking:
  `tmp/causal_regen_20260516/omega1_expertdq_quality_feature_groups_20260602/ranking.csv`
- Best tested Quality input variant:
  `baseline_contract`, OOS Cost3 `+5.54%`, MDD `-9.41%`, trades `211`,
  WR `52.13%`.

Rejected for this contract:

- Raw volume/flow context expansion.
- Raw funding context expansion.
- Raw session context expansion.
- Raw liquidity/execution/spread proxy expansion.
- Raw volatility context expansion beyond the already contracted
  `volatility_pca06` fields.
- `baseline_plus_regime_risk`, `baseline_plus_all_context`, and other broad
  context-expansion Quality variants.

Reason:

- The 70-feature baseline contract was the best Quality-only ablation result.
- Broad context additions increased OOS fragility and mostly worsened Cost3
  replay.
- Contract drift would make Omega1 comparisons ambiguous; missing or renamed
  Quality inputs must fail fast and be fixed at the data/model source.
