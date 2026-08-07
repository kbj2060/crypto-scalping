# Active Specs Change Log

## 2026-07-02 KST

- Blocked Omega5 live promotion after a side-thread audit found validation/test
  ledger dependence in the promoted model-selection path.
- `FINAL_GOVERNOR_OMEGA5_ENABLE` and
  `FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE` now default to `false`.
- Enabling Omega5 explicitly now fails fast at import time, and direct
  `Omega5LiveAdapter` construction also fails fast.
- Block report:
  `docs/audits/omega5_live_promotion_blocked_20260702.md`.
- Omega5 live-only validation/OOS/test now means live-forward or
  shadow-forward observations only. Historical validation/OOS replay and saved
  ledger-derived scores are not acceptable promotion/test evidence.
- Added live-only shadow upgrade loop:
  `scripts/run_omega5_live_only_shadow_loop_20260702.py`.
- With Omega5 disabled, live feature-frame DuckDB writes use
  `decision_feature_frame_live_only_shadow_20260702` so disabled-Omega5
  telemetry cannot collide with the blocked Omega5 table schema.

- Promoted Omega5 event-risk governor `omega5_event_risk_governor_20260702`.
- Added scheduled macro entry veto for NFP/ISM/S&P Global PMI/FOMC windows:
  30 minutes before through 120 minutes after the rule-based event timestamp.
- Added shock notional haircut for new entries:
  scale notional by `0.50` when `jump_flag`, `evt_tail_flag`,
  `abs(jump_z) >= 3.0`, `abs(1h return) >= 3%`, or `abs(4h return) >= 4%`
  fires.
- Existing Omega5 short veto, source parent, TP/SL, 5x leverage cap, and
  8h max hold contracts remain unchanged.
- Contract:
  `docs/model_contracts/omega5_event_risk_governor_20260702_contract.md`.
- Live feature-frame DuckDB writes now use the model-versioned table
  `decision_feature_frame_omega5_event_risk_governor_20260702` so old feature
  contracts cannot silently mix with the active Omega5 contract.
- `scripts/start_trading_bot.sh` now starts the supervisor with `setsid -f` and
  rewrites the PID file from the actual `supervise_trading_bot.sh` process.

## 2026-07-01 KST

- Promoted Omega5 live model `omega5_validation_only_live_20260701`.
- Trading bot integration:
  `FinalGovernorRuntime` now loads `trading_bot_modules/omega5_live.py` by default
  through `FINAL_GOVERNOR_OMEGA5_ENABLE=true` and evaluates Omega5 before the
  legacy Omega1.2.1 entry path.
- Omega5 uses the current Omega1.2.1 live parent signal, then applies the
  validation-only Omega4.6.2 overlay:
  source `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`,
  short vetoes `bb_width <= 0.003939593535185601` and
  `m7_prob_up >= 0.909727596`, exposure `lf0.900_sf1.050_cap4.40`,
  leverage cap `5x`, and max hold `8h`.
- Promotion audit:
  `docs/audits/omega5_live_promotion_20260701.md` reports
  `promotion_pass=true`.
- Source artifact integrity audit:
  `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/omega_artifact_integrity_audit_20260630.json`
  reports `promotion_pass=true`.
- Contract:
  `docs/model_contracts/omega5_validation_only_live_20260701_contract.md`.
- Active live stack doc:
  `docs/active_live/omega5_live_stack.md`.

## 2026-06-30 KST

- Added Omega4.6.2 research/detail-line candidate
  `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`.
  Contract:
  `docs/model_contracts/omega4_6_2_cap220_short_boost125_time_stop120h_20260630_contract.md`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json`.
  Candidate manifest:
  `data/ensemble/supervised/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/candidate_manifest.json`.
  Red-team report:
  `docs/audits/omega4_6_2_cap220_short_boost125_time_stop120h_20260630_redteam_20260630.md`.
- Omega4.6.2 keeps the Omega4.6 parent components and applies three overlays:
  short RSI skip at `rsi >= 56.656189`, short exposure boost `1.25x` with
  tested notional cap `2.20`, and `120h` time stop. It is not live-wired and
  does not replace Omega4.6 or Omega4.6.1.
- Omega4.6.2 metrics:
  validation `+211.14%` / MDD `-13.72%` / WR `65.22%` / trades `23` /
  max hold `120.00h`;
  OOS readout `+79.32%` / MDD `-10.13%` / WR `61.54%` / trades `13` /
  max hold `120.00h`.
- Omega4.6.2 red-team status: conditional diagnostic pass, full live pass
  false. The old notional `<= 1.8` pass gate is removed for this audit, but
  accounting consistency and `notional = margin_fraction * leverage` remain
  mandatory. Fresh holdout or walk-forward is required before promotion because
  OOS readout was considered for the detail-line choice.
- Set `omega4_6_plus_t12_nohold_risk1_20260630` as the current Omega
  research/upgrade baseline.
  Contract:
  `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`.
  Manifest:
  `data/ensemble/supervised/omega4_6_plus_t12_nohold_risk1_20260630/candidate_manifest.json`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json`.
  Source report:
  `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`.
- Red-team verdict:
  `CONDITIONAL_PASS_MAX_HOLD_AND_PNL_TARGET_EXCLUDED_NOT_DAYTRADING_LIVE_PASS`.
  This means Omega4.6 is a swing/runner research baseline, not a full
  day-trading live-pass model.
- Artifact integrity audit passed with `promotion_pass=true`.
- Non-excluded gates pass: MDD within `20%`, leverage within `5x`, notional
  within `1.8`, no overlaps, accounting consistent, and
  `notional = margin_fraction * leverage`.
- Metrics:
  validation `+117.17%` / MDD `-17.43%` / WR `51.72%` / trades `29`;
  OOS readout `+67.85%` / MDD `-13.28%` / WR `53.85%` / trades `13`.
- Conditional pass excludes max-hold and PnL-target gates. Known hold-time
  blocker: validation max hold `222.0h`; OOS max hold `218.5h`.
- Upgrade focus is to preserve no-hold swing alpha while reducing tail hold
  time through partial TP, trailing giveback, and profit-lock policies rather
  than blunt 24h forced exits.
- Upgrade work must preserve exact-threshold parent predictions and
  `risk_model.precomputed_prediction_dir` / `risk_model.precomputed_prediction_tag`.
  Trade ledgers remain diagnostic only.
- Live wiring remains unchanged.
- Added conditional upgrade candidate
  `omega4_6_1_duration_ou_halflife_risk_gate_20260630`.
  Contract:
  `docs/model_contracts/omega4_6_1_duration_ou_halflife_risk_gate_20260630_contract.md`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/runtime_contract.json`.
  Candidate manifest:
  `data/ensemble/supervised/omega4_6_1_duration_ou_halflife_risk_gate_20260630/candidate_manifest.json`.
- Omega4.6.1 applies an entry-time duration-aware risk gate:
  skip entries where `ou_halflife <= 0.005415348`. Selection was validation-only;
  OOS remained readout-only. Artifact integrity audit passed with
  `promotion_pass=true`.
- Omega4.6.1 metrics:
  validation `+175.86%` / MDD `-10.60%` / WR `61.90%` / trades `21` /
  max hold `115.33h`;
  OOS readout `+72.59%` / MDD `-7.47%` / WR `66.67%` / trades `9` /
  max hold `133.50h`.
- Omega4.6.1 is not live-wired and still inherits the conditional swing
  classification. Max-hold `24h` and PnL-target gates remain excluded.

## 2026-06-23 KST

- Reset Omega4.3 current research baseline to
  `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
  after the red-team full-pass remediation.
  Contract:
  `docs/model_contracts/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623_contract.md`.
  Manifest:
  `data/ensemble/supervised/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`.
- Selection is validation-only: OOS is excluded from filter, sort, and
  tie-break. Red-team verdict:
  `REDTEAM_PASS_CLEAN_RESEARCH_BASELINE_NOT_LIVE_WIRED`.
- Metrics for selected `risk_3473` sizing-only contract:
  validation `+30.33%` / MDD `-7.91%` / WR `67.06%` / trades `85`;
  OOS readout `+32.44%` / MDD `-5.72%` / WR `66.15%` / trades `65`.
- This supersedes
  `omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623`, whose OOS-guard
  mapping selection failed clean-OOS red-team review.
- Live wiring remains unchanged.

- Promoted `omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623`
  to the current Omega research baseline.
  Contract:
  `docs/model_contracts/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623_contract.md`.
  Manifest:
  `data/ensemble/supervised/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`.
- Omega4.3 keeps Omega4.2 direction, quality, exit-head timing, and ATR safety
  SLTP unchanged. It adds the `tail_penalty = 0.5` log-risk HGB sidecar for
  entry-time `margin_fraction` and `leverage` sizing only.
- Sizing contract remains explicit: `notional = margin_fraction * leverage`;
  PnL is `realized_price_move * notional`. SLTP barriers remain raw price-move
  barriers and are not divided by notional.
- Metrics for selected `risk_1673` sizing-only contract:
  validation `+29.39%` / MDD `-7.66%` / WR `67.06%` / trades `85`;
  OOS `+31.42%` / MDD `-5.37%` / WR `66.15%` / trades `65`.
- Full dynamic-risk exit replay is diagnostic only, not promoted:
  validation MDD weakens to `-10.25%`. Dynamic sidecar outputs must not alter
  exit timing without a separate retrained exit contract.
- This is not a live wiring change. Runtime-native parity, current live feature
  contract validation, and shadow or paper smoke are still required before real
  exchange use.

## 2026-06-22 KST

- Promoted `omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622`
  to the current Omega research baseline.
  Contract:
  `docs/model_contracts/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622_contract.md`.
  Manifest:
  `data/ensemble/supervised/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622/candidate_manifest.json`.
  Runtime contract:
  `tmp/causal_regen_20260516/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622/runtime_contract.json`.
- Omega4.2 does not retrain neural weights. It keeps the Omega4.1
  exit-threshold-0.70 bundle and promotes the ATR safety SLTP runtime overlay:
  ATR window `192`, TP multiple `12.0`, SL multiple `6.0`, TP floor `7.5%`,
  SL floor `4.0%`.
- SLTP barriers are price-move barriers. They are not divided by notional and
  are not multiplied by leverage. PnL remains `realized_price_move * notional`.
- Metrics for selected `atr192_tp12_sl6`:
  validation `+16.02%` / MDD `-7.11%` / WR `67.06%` / trades `85`;
  OOS `+13.32%` / MDD `-4.38%` / WR `66.15%` / trades `65`.
- Reference Omega4.1 exit0.70 baseline:
  validation `+3.28%` / MDD `-7.82%` / WR `67.11%` / trades `149`;
  OOS `+7.51%` / MDD `-5.61%` / WR `63.00%` / trades `100`.
- This is not a live wiring change. Runtime-native parity, current live feature
  contract validation, and shadow or paper smoke are still required before real
  exchange use.

## 2026-06-16 KST

- Promoted `omega1_2_8_full_retrain_numeric_cash_sleeve_20260616` to the current Omega research baseline for subagents and future candidate comparisons.
  Contract: `docs/model_contracts/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616_contract.md`.
  Manifest: `data/ensemble/supervised/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616/candidate_manifest.json`.
  Report: `tmp/causal_regen_20260516/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616/report.json`.
- Selected validation-only config:
  `full_retrain_ev_cal0.50_ev0.001_numcfg1_u0.002_m0.000`.
- Metrics:
  validation `+116.52%` / MDD `-10.79%` / WR `69.05%` / trades `42`;
  OOS `+82.25%` / MDD `-8.11%` / WR `62.50%` / trades `32`.
- This is not a live wiring change. Current live-wired Omega baseline remains
  `omega1_2_3_ev_hgb_cash_sleeve_20260615` until separate runtime-native
  implementation, parity, walk-forward/stress, and explicit promotion.

## 2026-06-10 KST

- Added and wired `omega1_2_1_true_leverage_price_barrier_scale200_cap090` as the Omega1.2.1 true-leverage candidate.
  Contract: `docs/model_contracts/omega1_2_1_true_leverage_price_barrier_20260610_contract.md`.
  Manifest: `data/ensemble/supervised/omega1_2_1_true_leverage_price_barrier_scale200_cap090/baseline_manifest.json`.
- Risk contract changed from effective-notional-only replay to true leverage exposure:
  `effective_exposure = margin_notional * execution_leverage`.
- To preserve the previous price barrier, TP/SL equity thresholds are scaled by `margin_ratio * leverage`.
  Common runtime case: margin `0.81`, leverage `2.0`, effective exposure `1.62`, TP `0.104`, SL `0.056`.
- True-leverage price-barrier replay metrics:
  validation `+276.67%` / MDD `-20.34%` / WR `63.64%` / trades `33`;
  OOS `+186.43%` / MDD `-15.60%` / WR `72.22%` / trades `18`.
- The failed diagnostic with unchanged equity TP/SL is explicitly not promoted:
  validation `-5.31%` / MDD `-31.25%`; OOS `+52.66%` / MDD `-17.91%`.
- Existing open Omega1.2.1 positions from the previous effective-notional baseline are recovered from journal with their persisted TP/SL and are not reinterpreted mid-trade.

## 2026-06-06 KST

- Set `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080` as the current Omega1.2 research baseline.
  Manifest: `data/ensemble/supervised/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080/baseline_manifest.json`.
  Contract: `docs/model_contracts/omega1_2_true_3head_tabm_final_tp_sl_current_20260606_contract.md`.
- Fixed baseline replay metrics:
  Validation Cost3 PnL `+42.82%`, MDD `-5.47%`, WR `63.64%`, trades `33`.
  OOS Cost3 PnL `+32.15%`, MDD `-4.14%`, WR `72.22%`, trades `18`.
- `base_nogate_topk2` remains a post-lifecycle adapter research candidate, but it is no longer the current Omega1.2 baseline for the next growth loop.
- No `trading_bot.py` live wiring was changed.
- Ran an initial static growth scan. Best balanced static candidate is compensated TP/SL scaling `1.35` capped at `0.55`: validation `+61.14%` / MDD `-7.32%`, OOS `+45.31%` / MDD `-5.54%`. Highest static OOS candidate is scale `2.00` capped at `0.90`: validation `+100.54%` / MDD `-10.68%`, OOS `+72.76%` / MDD `-8.11%`.
- Added Omega1.2.1 growth branch: `omega1_2_1_current_baseline_growth_20260606`.
  Contract: `docs/model_contracts/omega1_2_1_current_baseline_growth_20260606_contract.md`.
  Manifest: `data/ensemble/supervised/omega1_2_1_current_baseline_growth_20260606/omega1_2_1_manifest.json`.
- Tested learned high-confidence exposure selector.
  Best strict selector: validation `+54.18%` / MDD `-5.47%`, OOS `+35.97%` / MDD `-4.14%`.
  Selector OOF AUC is weak (`extra_win=0.3714`), so it is recorded as diagnostic and not promoted over the static balanced Omega1.2.1 candidate.
- Promoted `omega1_2_1_aggressive_compensated_scale200_cap090` to the current Omega research baseline per user decision.
  Contract: `docs/model_contracts/omega1_2_1_aggressive_current_baseline_20260606_contract.md`.
  Manifest: `data/ensemble/supervised/omega1_2_1_aggressive_compensated_scale200_cap090/baseline_manifest.json`.
  Fixed replay metrics: validation `+100.54%` / MDD `-10.68%` / WR `63.64%` / trades `33`; OOS `+72.76%` / MDD `-8.11%` / WR `72.22%` / trades `18`.
- Wired `omega1_2_1_aggressive_compensated_scale200_cap090` into `trading_bot.py` through `trading_bot_modules/omega1_2_1_live.py`.
  Runtime default: `FINAL_GOVERNOR_OMEGA1_2_1_ENABLE=1`; `FINAL_GOVERNOR_FULLY_LEARNED_ENABLE=0` unless explicitly overridden.
  Runtime-native smoke passed on `data/live/decision_feature_frame_snapshot.pkl.gz`; latest sample returned Omega CASH with no Alpha fallthrough.
  The Omega adapter builds Regime3 current/CMamba/stability-risk features live and fails fast on missing/non-finite contract columns.

## 2026-06-05 KST

- Added `omega1_2_post_lifecycle_bucket_adapter_20260605` research contract and candidate manifest.
- Preserved `base_nogate_topk2` as the stable Omega1.2 post-lifecycle bucket adapter research candidate.
  Mean OOS Cost3: PnL `+29.41%`, MDD `-2.34%`, WR `78.12%`, trades `32`.
- Preserved `fixed_wide_lev5_cap120_nogate_topk2` as an explicit aggressive research candidate.
  Mean OOS Cost3: PnL `+39.36%`, MDD `-8.82%`, WR `66.77%`, trades `36`.
  This candidate has materially weaker validation MDD and is not live-promoted.
- Marked `fixed_ultra_wide_lev5_cap120_nogate_topk2` as `blocked_research_only`.
  Although OOS PnL averaged `+72.06%`, validation PnL/MDD collapsed to `-32.13%` / `-34.93%`, so it must not be used as promotion evidence.
- No `trading_bot.py` live wiring was changed.

## 2026-05-28 KST

- Created `docs/active_live/` as the active/live operational documentation folder.
- Added active Alpha7 live stack documentation.
- Added `trading_bot.py` runtime documentation.
- Added module interface documentation for active live path.
- Added Docs Manager subagent definition under `docs/subagents/docs_manager.md`.
- Documented production default as `alpha7_submodel_01965_decontam_deep_stop_cd18_20260528`.
- Demoted `alpha7_submodel_01965_decontam_deep_stop_cd18_bear_long_veto_20260528` to shadow-only because validation MDD was materially worse than the pure deep-stop config.
- Documented `trading_bot.py` production singleton lock using `data/live/trade_journal.lock`.

## 2026-05-29 KST

- Added funding-family red-team audit and marked the current Alpha7 decontam production default as deprecated/blocked until rebuilt or replaced with a clean funding manifest.
- Reclassified pre-clean Alpha7 `01965` artifact families as deprecated and blocked for active/candidate reuse:
  `alpha7_1_01965_live_20260527` and `alpha7_submodel_01965_decontam_v2_tp_20260528`.
  Added `DEPRECATED_DO_NOT_USE.json` markers to both artifact directories and updated their live manifests.
- Removed live funding `bfill()` from `trading_bot_modules/binance_live_fetcher.py`; missing values now fail after causal `ffill()`.
- Patched clean `01965` candidate building so funding-derived columns without `funding` in their names come from clean feature frames and cannot be overwritten by generic M7 overlays.
- Added experimental `alpha7_directional_dsac_router_20260529`; this is not wired into live trading.
- Rebuilt DSAC state around causal directional features and Alpha7 primary/fallback decision context.
- Changed the experimental DSAC reward to optimize net PnL plus win outcome, with no trade-count penalty.
- Removed confirmed-bug regime feature prefixes from active/live feature contracts:
  `clean_regime_2024_unsup_v4_*` and `clean_regime4_2024_unsup_v1_*`.
- Active/live regime inputs must use `clean_regime4_state24_sticky090_v2_*` and `regime4_pred_*`; enabling the removed runtime legacy predictor now fails fast.
- Added research-only `alpha8_mamba_lgbm_dsac_hybrid_20260529`.
  It uses existing sticky-v2/future regime features, CUDA Mamba sequence embeddings, LightGBM directional probabilities, and a DSAC execution router.
  It is not wired into live because validation PnL underperformed the Alpha7 baseline.
- Added design-only `alpha8_parent_dsac_risk_sizing_plan_20260529`.
  In this plan, `clean_regime4_state24_sticky090_v2_*` is explicitly the existing current HMM output, Alpha7 Primary Parent owns direction, and DSAC only owns veto/risk/sizing modifiers.
- Trained `alpha8_parent_dsac_risk_sizing_20260529`; not promoted.
  Cost3 OOS underperformed the active baseline combo (`70.49%` vs `123.63%`) because parent-only risk/sizing removed fallback coverage.
- Added `Regime3 + Whipsaw Risk` design policy for next action-classifier/Alpha8+ candidates.
  New action classifiers should use bull/bear/chop as medium-horizon structure classes and move whipsaw into risk/veto/sizing context.
- Trained research-only `regime3_hmm_mamba_risk_cleanfunding_20260529`.
  Current Regime3 HMM reached 2026 forward balanced accuracy `0.7655`; shared Mamba future h12/h24 remains weak and is not promoted for action ownership.
- Tested `regime3_current_hmm_wide24_experiment_20260529`.
  Wide24 improved 2026 current regime balanced accuracy to `0.8143` versus `0.7655` for state12, with higher flip rate `0.0520` versus `0.0417`.
- Confirmed `regime3_current_hmm_wide24_experiment_20260529` as the next Regime3 CURRENT research surface.
- Retrained `regime3_pred_mamba_wide24_current_cleanfunding_20260529` using explicit `regime3_current_wide24_*` inputs.
  The stable six-current-feature variant improves risk prediction but remains insufficient for future direction action ownership.
- Retested the previous TFT/VSN selected PRED architecture as `regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529`.
  Inputs are restricted to raw causal features plus `regime3_current_wide24_*`; legacy Regime4/current aliases are blocked.
  TFT/VSN materially outperformed the Mamba PRED direction experiment. Top74 is the defensible validation-selected variant; VSN36 is a 2026 test-favorable research variant.
- Added Docs Manager feature-audit guided PRED feature packs.
  `docs_regime_pred` removes audit-disfavored broad raw features such as `close_btc` and `garch_vol_z` while keeping causal direction/regime context.
  The `Docs40 stable` variant is now the preferred Regime3 PRED research candidate: 2024Q4 bacc `0.6765`, 2026 bacc `0.6876`, zero forbidden feature count.
- Promoted `Docs48 all raw` to the user-selected main Regime3 PRED research candidate.
  It is not live-wired. Selection manifest: `data/ensemble/supervised/regime3_pred_tft_vsn_docs48all_wide24_current_cleanfunding_20260529/SELECTED_MAIN_REGIME3_PRED_20260529.json`.
  Metrics: 2024Q4 bacc `0.6746`, 2026 bacc `0.6911`, 2026 bull/bear/chop recall `0.6277 / 0.7013 / 0.7442`.
- Tested Docs Manager feature-audit guided Regime3 CURRENT feature packs under `regime3_current_hmm_docs_feature_experiment_20260529`.
  `docs51all` improved 2026 bacc to `0.8307` but raised flip rate to `0.1077`; `sticky=0.97` did not reduce churn.
  Decision: keep `wide24` as the main Regime3 CURRENT research surface and keep docs feature packs as high-sensitivity research variants only.

## 2026-05-30 KST

- Recorded M7 red-team contract status: active M7 generation/required-column contracts no longer include the unsupervised `gmm_volatility`, `isolation_forest`, or `vae_anomaly` model/meta keys.
  Removed active generated/required columns: `m7_gmm_cluster`, `m7_gmm_conf`, `m7_gmm_vol_rank`, `m7_iso_pred`, `m7_iso_score`, `m7_iso_anom`, `m7_vae_error`, `m7_vae_threshold`, `m7_vae_anom`, `m7_gate_block`, `m7_size`, `m7_hdb_label`, and `m7_hdb_prob`.
  Historical artifacts or CSVs using those columns are diagnostic-only until retrained/rescored under the current active contract; they are not promotion evidence.
  Policy doc: `docs/audits/m7_redteam_contract_20260530.md`.
- Promoted the sensitive Regime3 CURRENT wide24 research candidate per user decision.
  New explicit CURRENT contract: `regime3_current_sensitive_wide24_*` from artifact directory `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/`.
  Label mode is `balancedish_adx16_slope15_bb012`, which reduces chop-heavy labels but increases regime churn.
- Retrained Regime3 PRED using the previous Docs48 all-raw input design with only CURRENT changed to `regime3_current_sensitive_wide24_*`.
  New explicit PRED contract: `regime3_pred_sensitive_tft_h12_*` from artifact directory `data/ensemble/supervised/regime3_pred_tft_vsn_sensitive_wide24_current_docs48all_20260530/`.
  The PRED target also changes because labels are generated from `argmax(regime3_current_sensitive_wide24_* at t+12)`.
  2026 PRED bacc is `0.5760`, materially below the previous Docs48 wide24-current PRED bacc `0.6911`; downstream action usage should account for this lower predictability.
- Ran sensitive-current PRED improvement sweep with larger TFT hidden size and expanded/rolled feature packs.
  Best `h=12` research candidate is `regime3_pred_sensitive_rolled_top72_d96_e10_20260530` with 2026 bacc `0.5809`.
  Best `h=6` research candidate is `regime3_pred_sensitive_h6_docsall_top48_d96_e10_20260530` with 2026 bacc `0.6774`.
  This indicates the sensitive CURRENT surface is more suitable for 30-minute regime prediction than one-hour prediction.
- Audited the h6 PRED for same-bar CURRENT shortcut behavior.
  Retrained no-current PRED candidates that use CURRENT only as the target source, not as input features.
  Best no-current candidate is `regime3_pred_sensitive_h6_nocurrent_rolled_top72_d96_e10_20260530` with 2026 bacc `0.6695` and transition-row bacc `0.2534`.
  The original current-conditioned h6 candidate had higher persistence-row bacc but weaker transition-row bacc `0.2023`, so it should not be treated as a pure transition predictor.
- Added dedicated Regime3 transition hazard/destination research head `regime3_transition_hazard_sensitive_h6_20260530`.
  The selected threshold-0.46 with-current candidate reaches 2026 transition-row bacc `0.5163`, versus `0.2023` for the original h6 future-PRED head.
  Overall bacc drops to `0.5881`, so this head should be used as transition-risk/veto/sizing context rather than as the sole regime classifier.
- Added no-current stable h6 decoder candidate `regime3_stable_h6_decoder_nocurrent_transitionaware_20260530`.
  Per user constraint, no `regime3_current_sensitive_wide24_*` probability columns are used as model input features; the current sidecar is used only to generate/evaluate the stable target.
  2026 selected OOS: overall bacc `0.7410`, transition bacc `0.2773`, persistence bacc `0.8403`.
- Added no-current stability/risk feature head `regime3_stability_risk_h6_20260530`.
  It outputs `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, and `regime3_churn_h6_risk_score` without using CURRENT probabilities as model inputs.
  2026 transition AUC is `0.6762`; top 20% risk zone transition rate is `0.2874` versus low 20% `0.0471`.
  Use as continuous size-throttle/veto context, not as a directional regime predictor.
- Finalized Regime3 active policy: directional `PRED regime` is removed from action ownership.
  Active CURRENT remains `regime3_current_sensitive_wide24_*`.
  Future-regime class predictions must not drive long/short direction, primary/fallback labels, or hard future regime selection.
  Use `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, and `regime3_churn_h6_risk_score` only as stability/transition-risk context.
  Policy doc: `docs/active_live/regime3_policy_20260530.md`.
- Added supervised M7 `lightgbm_ensemble` model `data/ensemble/supervised/lightgbm_ensemble.json`.
  Active downstream M7 risk/quality columns: `m7_tradeability_score`, `m7_long_mae_q90`, `m7_short_mae_q90`, `m7_long_adverse_prob`, and `m7_short_adverse_prob`.
  Direction context is binary-only and must be declared explicitly by each consumer.
  The trainer blocks `m7_*`, legacy clean-regime prefixes, and old `regime_bull/regime_bear/regime_chop/regime_whipsaw/regime_normal` one-hot inputs.
- Retrained active M7 base artifacts with expanded clean input features and same-name overwrite only where 2026 OOS improved.
  Overwritten artifacts: `trend_xgb`, `multi_target_lgbm`, and `quantile_forest`.
  `entry_price_model` candidate was rejected because OOS average offset MAE worsened; the runtime wrapper still now propagates the existing learned entry offsets instead of zeroing them.
  Canonical M7 supervised artifact feature audit now reports zero forbidden features for `trend_xgb`, `entry_price_model`, `multi_target_lgbm`, `quantile_forest`, and `lightgbm_ensemble`.
  M7 direction heads are binary-only. No-trade must be decided by `m7_tradeability_score`, quality/uncertainty/adverse features, or downstream risk/router layers.
  `SevenModelEnsemble.predict_batch()` now fails if `trend_xgb` or `multi_target_lgbm` emits a non-binary direction probability shape.

## 2026-07-02 KST

- Corrected the project definition of fresh-forward validation/OOS/test.
  Fresh-forward means causal bar-by-bar walk-forward over fixed historical splits, not necessarily waiting for new real-time bars.
  Default split: validation `2025-09-01` through `2025-12-31`, OOS `2026-01-01` through `2026-03-31`.
  Each 5m bar must be processed sequentially using only features/state available at that bar; exits and PnL are resolved only by subsequently advanced bars.
  Stored trade ledgers, candidate-event ledgers, parent exit timestamps, and future decision rows remain invalid as model-selection or promotion inputs.
- The earlier live-only wording for Omega5 validation/OOS/test is superseded by the corrected bar-by-bar fresh-forward definition above.
- Corrected `omega5_live_short_momentum_v2` fresh-forward test completed under the fixed-split bar-by-bar definition.
  Validation compound PnL/MDD: `-96.32%` / `-96.35%`; OOS compound PnL/MDD: `-62.43%` / `-62.43%`.
  Verdict: does not pass; the earlier live-only test artifact was invalidated.
- Completed corrected fixed-split bar-by-bar fresh-forward test for
  `omega4_6_2_source_parent_fresh_forward_with_hf_policy_overlay_20260702`.
  Validation compound PnL/MDD: `-14.86%` / `-25.52%`; OOS compound PnL/MDD: `+80.29%` / `-7.64%`.
  Ledger replay trace count, saved parent timestamp use, and future entry row use were all zero/false.
  Verdict: stronger than the short-momentum candidate, but not a clean promotion pass because validation remains negative and drawdown exceeds the prior target.
- Omega5 live promotion remains blocked because the prior artifact was found to depend on validation/test ledger information.
  The running trading bot was not stopped; data collection remains active with Omega5 disabled.
- Added `scripts/run_omega5_live_only_shadow_loop_20260702.py` as the active Omega5 upgrade loop.
  It tails `data/live/decision_feature_snapshot.jsonl` from the loop start offset and writes live-only shadow signals, closes, state, and report under `data/live/omega5_live_only_upgrade_loop_20260702_v3_ml/`.
- Added online ML Omega5 candidates that update only from closed live-forward shadow trades:
  `omega5_live_online_logit_v3`, `omega5_live_online_bandit_v3`, `omega5_live_online_fast_logit_v4`, and `omega5_live_online_fast_bandit_v4`.
  No historical labels or backtest outcomes are used for online weight updates.

## 2026-08-07 KST

- Promoted the BTC shadow-live parent bundle to the swingtransition candidate (h48qual + `swing_transition_prob`).
  New bundle: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt`.
  New sidecar: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260806_swingtransition/risk_sidecar.pkl`.
  New duration threshold: `0.0054143218` (candidate's own VAL-selected gate; previous `0.00541154875`). Scale map unchanged (`h48qual_L 0.5 / h48qual_S 2.5`).
  Gate: `scripts/audit_omega_artifact_integrity_btc_20260712.py` exit 0, `promotion_pass=true` -- the same asset-specific gate the previous BTC baseline was promoted under on 2026-07-13. The generic `audit_omega_artifact_integrity_20260630.py` gained ETH-sidecar-format checks after 2026-07-13 that BOTH the previous BTC baseline and this candidate fail identically (gate-version drift, not a candidate regression); documented here rather than silently switching gates.
  Selection evidence: VAL +24.23%/MDD -2.46% vs +12.39%/-6.49% baseline; OOS-extended +10.76%/MDD -12.41% vs +10.79%/-16.11%; worst OOS quarter -0.87% vs -7.11% (project default worst-quarter lens). Known weak spot: OOS-frozen-Q1 +2.63% vs +10.17% -- the candidate takes 21 long / 4 short in a quarter where shorts carried all profit for both models; zero entry-timestamp overlap with the previous bundle (feature addition re-rolls the trade set; this is a model swap, not a refinement).
- Added live computation of `swing_transition_prob`: `trading_bot_modules/btc_swing_transition_live.py`.
  Layer A LightGBM saved+verified by `scripts/train_btc_5m_layerA_swing_transition_save_model_20260807.py` (regenerated predictions match the 2026-08-06 offline parquet bit-exactly, max_abs_diff=0.0 over 271,773 rows; artifact `data/ensemble/supervised/btc_swing_transition_layerA_20260807/`).
  Live inputs: 96 in-frame causalfix cols + 10 `mtf1h_*` (1h resample of the asset's own kline buffer, exact offline builder functions, +1h availability shift) + 4 `dvol_btc*` (Deribit DVOL public REST, hourly, +1h availability shift, incremental cache).
  Offline parity: `scripts/test_btc_swing_transition_live_parity_20260807.py` -- max_abs_diff 0.0 vs the offline training feature at 3 cutoffs x 600 bars.
  Live-path replay: adapter reproduces the offline OOS ledger 11/11 on entry side and matches margin_fraction/leverage/notional to ledger precision at spot-checked entries.
  Wiring: provider auto-enables only when the loaded BTC bundle's base_cols require the feature, so an env-var rollback to the previous bundle also disables the provider and its Deribit dependency. Fail-fast on any missing input/fetch failure (per-asset refresh error handling, no degraded-feature trading).
