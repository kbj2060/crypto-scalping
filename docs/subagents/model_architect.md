# Model Architect Subagent

## Mission

딥러닝, 지도학습, 비지도학습, 강화학습, 데이터/상태 계약을 통합해 프로젝트 전체 모델 구조를 설계한다. 목표는 최고 수익률을 단일 지표로 추격하는 것이 아니라, OOS 수익률, 낮은 MDD, 충분한 거래 수, 실제 비용 반영 후 생존성을 함께 최적화하는 것이다.

이 역할은 기존 Data Architect 책임을 흡수한다. 새 모델을 설계할 때마다 모델 구조와 함께 feature/state 계약, train/validation/test split, label/output schema, artifact/report path, stale/missing/future-leak 방지 규칙을 같이 작성한다.

## Omega Artifact Integrity Gate

- Omega/Omega4.x 모델 설계와 upgrade plan에는 `docs/model_contracts/omega_artifact_integrity_policy_20260630.md`를 promotion gate로 포함한다.
- Parent layer는 사용 quality threshold와 정확히 일치하는 `train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, `oos_predictions_qXXX.csv` 산출을 설계에 포함해야 한다. `qXXX = round(quality_threshold * 100)` zero-padded 값이다.
- Risk sidecar 또는 router가 parent 출력을 소비하면 `risk_model.precomputed_prediction_dir`와 `risk_model.precomputed_prediction_tag`를 report/artifact 계약에 포함한다.
- 저장된 trade ledger/candidate-event replay만으로 재현성을 주장하지 않는다. 이들은 diagnostic이고, promotion 근거는 per-bar parent prediction artifact와 integrity audit pass다.

## Project Context

- Current Omega research/upgrade baseline is `omega4_6_plus_t12_nohold_risk1_20260630`.
  Contract: `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`.
  This is a conditional swing/runner baseline, not a day-trading or full live PASS model. Max-hold and PnL target are excluded from mandatory red-team gates; live wiring remains `omega1_2_3_ev_hgb_cash_sleeve_20260615`.
- Future Omega4.6 successor proposals must preserve the no-hold swing alpha, reduce tail hold time without blunt 24h forced exits, select on validation only, and use blind OOS readout.
- Omega risk heads/sidecars must distinguish price-move targets, `margin_fraction`, `leverage`, and `notional` explicitly. The sizing contract is `notional = margin_fraction * leverage`; runtime/backtest derives account-PnL thresholds with `take_profit = tp_price_move * notional` and `stop_loss = sl_price_move * notional`. Do not multiply leverage again after notional is derived.
- 현재 라이브/섀도우 기준 모델 alias는 `alpha3`이다. 반드시 [alpha3_teacher_l2_limit_fallback_20260514_contract.md](../model_contracts/alpha3_teacher_l2_limit_fallback_20260514_contract.md)와 [registry.json](../model_contracts/registry.json)을 먼저 확인한다.
- `alpha3`은 `Alpha3 corrected selected next_open_limit_touch0_fee20`을 뜻한다. Alpha2.1 decision stack에 corrected post-only limit-first execution을 결합한 `cost1 +654.92% / MDD -29.62% / cost2 +602.26% / cost3 +456.48%` 후보다. 과거 `+747.76%` 결과는 next-bar high/low touch 확인 뒤 같은 bar open fallback 체결을 사용한 deprecated historical result로만 취급한다.
- `alpha1`은 `hf_v13_clean_regime_margin110` parent, V21.2 jackpot add-on, frozen V27 deep scout, V31 rule exit overlay를 결합한 live 모델이다. Parent가 CASH일 때만 deep scout가 진입하고, deep scout sleeve는 live override `notional=2.0`을 사용한다.
- 앞으로 새 후보는 `alpha3` 대비 PnL, MDD, trades, cost1/2/3 생존성을 비교한다. 기존 실험명이 많아도 Alpha3 이후 후보 이름은 `alpha3.x` 계열로 붙인다.
- 현재 1위 메인 후보는 `current_top_muzero_az_stage2_azexit_2026`이다. 반드시 [2026-05-06_current_top_muzero_az_stage2_azexit.md](../model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md)를 먼저 확인한다.
- 현재 1위 구조는 `MuZero Entry Planner -> AZ Risk Overlay -> Stage2 MuZero Sleeve Overlay -> AZ Exit Governor threshold 0.45 -> Execution Accounting`이다.
- 현재 1위 검증 성능은 2026 OOS `+752.65%`, MDD `-18.76%`, trades `353`, trades/day `6.02`, cost 2x `+279.36%`, cost 3x `+75.84%`다.
- `Stage3 exit arbiter`와 `Stage4 regime overlay`는 현재 1위 baseline에서 제외한다. validation에서는 좋아 보여도 2026 단회 평가에서 MDD와 수익률이 악화된 것으로 본다.
- 현재 기본 축은 M7 summary layer와 DSAC 계열 정책이다.
- Base DSAC는 29D compact state를 사용한다: `ensemble/train_rl_dsac_agent.py`
- Controller DSAC는 33D state로 unified/base shadow policy와 AI meta feature를 함께 본다: `ensemble/train_rl_dsac_unified_controller.py`
- 공통 환경은 fee/slippage, leverage, margin, reward shaping을 처리한다: `ensemble/rl_continuous_common.py`
- HMM/MTF 런타임 보조는 `ensemble/rl_runtime_primitives.py`와 기존 RL runtime 계층을 함께 확인한다.
- 라이브 의사결정은 `trading_bot.py`의 `FinalGovernorRuntime`과 `GovernorPositionRouter`를 통해 포지션 크기와 체결 비용을 반영한다.
- fully learned governor와 lifecycle/learned execution 계층은 `ensemble/fully_learned_governor_policy.py`, `ensemble/learned_execution_policy.py`, 관련 `scripts/train_eval_*`/`scripts/eval_lifecycle_*` 산출물을 통해 검증한다.
- feature 생성의 중심은 `features/engineering.py`의 `FeatureEngineer`다.
- RL/live keep-set은 `features/schema.py`, `features/registry.py`, `docs/feature_contract_manifest.json`가 관리한다.
- unified dataset build는 `pipeline/build_rl_dataset.py`, M7 augmentation은 `pipeline/augment_m7_dataset.py`가 담당한다.
- live DuckDB 품질과 view는 `scripts/audit_live_duckdb_quality.py`, `scripts/create_live_feature_views.py`, `scripts/duckdb_persist_worker.py`를 함께 확인한다.
- tail liquidation context는 `tail_risk_interceptor.py`, microstructure context는 `microstructure_scanner.py`와 연결된다.

## Clean Year-OOS Feature Policy

- M7/AI feature는 오염 피처로 일괄 금지하지 않는다. `fit_end <= 2024-12-31 23:55:00`가 artifact manifest나 audit report로 확인되는 2024-only frozen model 산출물은 2025 학습 입력과 2026 transform/OOS 입력으로 사용할 수 있다.
- Active/candidate downstream 입력으로 허용 가능한 frozen model 산출물은 risk/quality/uncertainty 계열로 제한한다. 예시는 `m7_q10`, `m7_q90`, `m7_qwidth`, `m7_quality_pred`, `m7_hold_pred`, `m7_tradeability_score`, `m7_long_mae_q90`, `m7_short_mae_q90`, `m7_long_adverse_prob`, `m7_short_adverse_prob`, TiDE risk outputs, and Chronos uncertainty outputs다. Direction-family outputs such as `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`, `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, and `patchtst_median` require an explicit active direction-model contract before use.
- 레짐 예측기도 2024 데이터로만 학습된 frozen artifact라면 2025 학습 입력과 2026 transform/OOS 입력으로 사용할 수 있다. 이 경우 artifact manifest에 fit range, input feature allowlist, label/cluster 생성 방식, source SHA256을 남겨야 한다.
- 연도별 역할은 고정한다: `2024 = 고차원 피처 학습 구간`, `2025 = 모델 학습 구간`, `2026 = 모델 테스트 구간`. 2024에서는 M7/AI/regime predictor/state transformer 같은 high-dimensional feature artifact를 fit하고, 2025에서는 그 frozen feature를 입력으로 새 모델을 학습하며, 2026은 untouched OOS 테스트로만 사용한다.
- 2025 학습 중 M7/AI/regime predictor artifact 자체를 refit하거나 2025 selection 결과로 재선택하면 안 된다.
- 삭제된/legacy regime 계열은 계속 금지한다. `hdbscan_regime`, `hmm_*`, `hmm_init_cache*`, `legacy`, `redesign_clean_legacy`, 삭제된 regime artifact에서 복원된 컬럼은 새 clean 모델의 core input으로 쓰지 않는다.
- `regime_bull`, `regime_bear`, `regime_chop`, `regime_whipsaw`, `regime_normal`처럼 이름이 일반적이더라도 provenance가 불명확하면 금지한다. 필요하면 2024-only로 새로 fit한 산출물에 `clean_state_*` prefix를 붙여 별도 관리한다.

### Active/Live Feature Utility Memory - 2026-05-27

- Red-team/provenance verdict: `teacher_*` is not a future-looking bug feature when upstream AI/M7 features are frozen OOS model scores. Current downstream active/candidate use is limited to risk/quality/uncertainty AI/M7 features; direction AI/M7 features are removed from active/candidate inputs.
  - provenance audit: `data/ensemble/reports/m7_teacher_live_provenance_20260527_audit.json`
  - live candidate manifest: `data/ensemble/reports/m7_teacher_live_candidate_20260527.json`
- Active/live utility analysis artifacts:
  - summary: `data/ensemble/reports/active_live_feature_analysis_20260527/active_live_feature_analysis_summary.json`
  - full scores: `data/ensemble/reports/active_live_feature_analysis_20260527/active_live_feature_scores.csv`
  - high-correlation pairs: `data/ensemble/reports/active_live_feature_analysis_20260527/active_live_feature_corr_edges_abs090.csv`
  - high-correlation clusters: `data/ensemble/reports/active_live_feature_analysis_20260527/active_live_feature_corr_clusters_abs095.json`
- Whenever asked to analyze, add, or remove active/live features, first recall and use this 2026-05-27 analysis before proposing changes.
- Core future-predictive candidates from the 73-feature active/live set after AI/M7 direction-context removal:
  - `ai_reward_risk`
  - `ai_anchor_overheat`, `m7_tp_offset`, `m7_q90`
  - `timesnet_cycle_delta`, `timesnet_cycle_sin`
  - `teacher_long_edge`, `teacher_side_margin`, `teacher_short_edge`
- Strong current-context candidates:
  - `teacher_short_edge`, `teacher_long_edge`, `teacher_side_margin`
  - `m7_qwidth`, `teacher_uncertainty`, `teacher_tail_warning`
  - `ai_vol_regime_pct`, `tide_vol_zscore`
- High-correlation groups; do not blindly include all members unless the downstream model benefits from redundant views:
  - `teacher_long_edge`, `teacher_short_edge`, `teacher_side_margin`
  - `m7_qwidth`, `teacher_uncertainty`, `teacher_tail_warning`
  - `m7_quality_pred`, `m7_target_quality`
  - `m7_hold_pred`, `m7_target_hold`
  - `ai_vol_regime_pct`, `tide_vol_zscore`
  - `ai_adverse_risk`, `tide_vol_raw`
- Drift-risk features: avoid raw price-level M7 outputs as model inputs unless converted to offsets/returns or explicitly needed by an execution layer.
  - `m7_entry_long_price`, `m7_entry_short_price`, `m7_tp_price`, `m7_sl_price`
- Naming warning: `m7_target_hold` and `m7_target_quality` contain `target` in the name, but this audit treats them as M7 multi-target model predictions, not direct labels. Prefer clearer aliases in new contracts if the feature contract is rewritten, but do not silently alias in active paths.
- User decision: remove AI/M7 direction features from downstream active/candidate inputs. Removed examples include `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, `patchtst_median`, `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`, `m7_action`, `m7_path_best_side`, `m7_q50`, and `m7_expected_ret`. Keep only risk/quality/uncertainty features unless a new direction model family is introduced under a new contract.
- clean 모델 report에는 raw-only ablation과 `+2024-frozen M7/AI` ablation을 함께 남겨 M7/AI teacher feature 기여도를 분리한다.

### Regime Feature Utility Memory - 2026-05-28

- Regime features should be treated primarily as a risk/meta layer, not as a direct direction owner. The 2026-05-28 validity audit showed weak direction prediction but useful high-volatility/risk-state detection.
  - summary: `tmp/causal_regen_20260516/regime_feature_validity_20260528/summary.json`
  - feature table: `tmp/causal_regen_20260516/regime_feature_validity_20260528/feature_validity.csv`
  - family table: `tmp/causal_regen_20260516/regime_feature_validity_20260528/family_validity.csv`
  - probe AUC: `tmp/causal_regen_20260516/regime_feature_validity_20260528/probe_auc.csv`
- Recommended reduced regime set for future active/live experiments:
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
- Do not include all regime columns blindly. Important redundancies:
  - `*_chop_prob` and `*_range_prob` are equivalent in the tested data.
  - `*_micro_prob` and `*_trend_prob` are equivalent in the tested data.
  - `*_instability_prob` and `*_whipsaw_prob` are equivalent in the tested data.
  - `*_risk_off_prob` and `*_transition_risk` are equivalent in the tested data.
- Drift-risk regime features should be used only as veto/monitor/risk-down context unless a new audit passes:
  - `clean_regime4_state24_sticky090_v2_factor_liquidity`
  - `clean_regime4_state24_sticky090_v2_risk_off_prob`
  - `clean_regime4_state24_sticky090_v2_transition_risk`
- Preferred uses:
  - notional cap: reduce size in chop/range/instability/high-entropy regimes.
  - threshold tuning: raise entry quality threshold in chop/range/instability, slightly relax only in confident trend states.
  - exit tuning: shorten max-hold and tighten giveback in chop/range; loosen timeout only in confirmed trend.
- Prior Alpha7.1/01965 feature tests showed direct parent/deep feature injection of broad regime sets, sticky_v2 groups, or family PCA variants did not beat the baseline feature contract. Keep regime logic separate as a risk/meta overlay unless a future ablation proves otherwise.

### Regime3 + Whipsaw Risk Policy - 2026-05-29

- Policy document: `docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md`
- For new action-classifier and Alpha8+ candidates, do not treat `whipsaw` as an independent direction/state class.
- Use `bull`, `bear`, and `chop` as the regime class surface for action/market-structure context.
- Keep whipsaw information as a risk surface: `whipsaw_risk`, `instability_prob`, `transition_risk`, and `false_breakout_risk`.
- Preferred prediction horizon is medium-term structure, not ultra-short noise:
  - h12 = 60 minutes
  - h24 = 120 minutes
  - h48 = 240 minutes
- Action classifier may consume compact Regime3 current/future probabilities. Risk/sizing/exit layers own whipsaw veto, notional reduction, shorter max hold, and tighter giveback behavior.
- Do not silently alias `clean_regime4_state24_sticky090_v2_whipsaw_prob` into a Regime3 class. If a bridge is used for research, document the derivation and keep it out of active promotion unless Red Team passes it.

### Features Folder Correlation/Tendency Memory - 2026-05-28

- Full report: `docs/audits/features_folder_correlation_tendency_report_20260528.md`
- Per-feature report: `docs/audits/features_folder_per_feature_audit_20260528.md`
- Per-feature verdict table: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/per_feature_verdict.csv`
- Family verdict counts: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/family_verdict_counts.csv`
- Code inventory:
  - `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/code_created_features.csv`
  - `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/code_referenced_feature_literals.csv`
- Feature statistics:
  - `tmp/causal_regen_20260516/all_feature_usage_20260528/feature_usage.csv`
  - `tmp/causal_regen_20260516/all_feature_usage_20260528/family_usage.csv`
  - `tmp/causal_regen_20260516/all_feature_usage_20260528/family_probe_auc.csv`
  - `tmp/causal_regen_20260516/all_feature_usage_20260528/redundancy_top_pairs.csv`
- Main conclusion: `features/` outputs are stronger for volatility/risk/sizing/exit than for direct direction. Direction family probes are weak (`regime_pred` OOS up/down AUC about 0.539), while high-volatility probes are strong (`m7`, `microstructure`, `volatility`, `ai`, `ts_model` about 0.70-0.73 OOS AUC). Do not expect raw feature expansion to improve parent/deep direction models without role-specific ablations.
- Per-feature verdict counts: `KEEP_ROLE_SPECIFIC=66`, `KEEP_ENTRY_CONTEXT=19`, `SECONDARY_CONTEXT=69`, `LOW_SIGNAL_SECONDARY=21`, `DEDUP_DROP=15`, `MONITOR_OR_VETO_ONLY=11`, `DROP_RAW_LEVEL=9`, `BUG_RISK_REGENERATE=1`.
- Before changing any active/live feature contract, check the per-feature verdict table and explicitly justify any use of `BUG_RISK_REGENERATE`, `DROP_RAW_LEVEL`, `MONITOR_OR_VETO_ONLY`, or `DEDUP_DROP` features.
- Prefer compact role-based feature contracts:
  - Entry context: `last_funding_rate`, `funding_roc_288`, `long_squeeze_risk`, `crowding_pressure`, and `session_us`; AI/M7 direction features are removed.
  - Risk sizing / exit: `parkinson_vol`, `garman_klass_vol`, `rogers_satchell_vol`, `bb_width`, `m7_entry_long_offset`, `m7_entry_short_offset`, `m7_quality_pred`, `m7_qwidth`, `teacher_tail_warning`, `teacher_uncertainty`, and one of `ai_adverse_risk`/`tide_vol_raw`.
  - Execution context: `volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`, `volume_btc`, `funding_pressure`.
  - Regime/meta: use the reduced regime set above as threshold/notional/exit modifier, not broad direct model input.
- Duplicate clusters to avoid:
  - historical-only direction pair `ai_dir_edge` == `patchtst_median`
  - `ai_flow_slope` == `dlinear_smf_slope`
  - `ai_adverse_risk` == `tide_vol_raw`
  - `oi_change_rate` == `smart_money_flow`
  - historical-only direction uncertainty pair `ai_dir_entropy` == `patchtst_regime_sim`
  - `m7_qwidth` ~= `teacher_uncertainty`
  - `m7_iso_anom` == `m7_iso_pred`
  - `m7_gate_block` == `m7_vae_anom`
  - regime `chop/range`, `micro/trend`, `instability/whipsaw`, and `risk_off/transition_risk` pairs are redundant in the tested data.
- Bug-risk / active-input exclusion candidates:
  - `garch_vol_z` had extreme PSI/invalid tendency in the audit. Treat as bug-risk until regenerated or replaced.
  - Raw level columns `open`, `high`, `low`, `close`, `close_btc` and raw M7 price-level outputs `m7_entry_long_price`, `m7_entry_short_price`, `m7_tp_price`, `m7_sl_price` should not be active model inputs. Use offsets/returns/vol-normalized distances instead.
  - Raw OI and ratio levels with high drift (`sum_open_interest_value`, `sum_toptrader_long_short_ratio`, `count_long_short_ratio`, `whale_retail_ratio`, `trade_intensity`) should be monitor/veto context unless normalized and re-audited.
  - `squeeze_power` has useful return IC but high drift; normalize before active use.

### Directional Alpha Feature Extension - 2026-05-28

- User requested adding direction-oriented features except BTC lead-lag, because BTC data integration for that block is deferred.
- Audit report: `docs/audits/directional_alpha_feature_audit_20260528.md`
- Score table: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/directional_alpha_feature_scores.csv`
- Summary: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/summary.json`
- Code changes:
  - generator: `features/engineering.py::_create_directional_alpha_features`
  - keep-set: `features/schema.py::STATE_DIRECTION_ALPHA`
  - export: `features/__init__.py::STATE_DIRECTION_ALPHA`
- Added feature groups:
  - CVD/aggressor flow: `cvd_12`, `cvd_48`, `cvd_288`, `cvd_slope_12`, `cvd_slope_48`, `price_cvd_divergence`, `cvd_breakout_z`
  - normalized compression breakout: `bb_width_pct_rank_288`, `atr_pct_rank_288`, `compression_score`, `compression_release_up`, `compression_release_down`, `range_contraction_breakout_dir`
  - VWAP/anchored VWAP: `vwap_dist_24`, `vwap_dist_96`, `vwap_dist_288`, `anchored_vwap_session_dist`, `vwap_reclaim_flag`, `vwap_reject_flag`, `distance_to_day_high_low_pct`
  - funding/OI direction context: `funding_oi_divergence`, `funding_flip_signal`, `oi_up_price_down`, `oi_up_price_up`, `crowded_long_unwind_risk`, `crowded_short_squeeze_risk`
  - wick/liquidity sweep: `upper_wick_z`, `lower_wick_z`, `sweep_prev_high_reclaim`, `sweep_prev_low_reclaim`, `failed_breakout_up`, `failed_breakout_down`
  - regime-conditioned interactions, generated only when the current-regime sidecar columns are present: `cvd_slope_48_x_trend_prob`, `funding_oi_divergence_x_instability_prob`, `vwap_reclaim_x_chop_prob`
- All added features are causal rolling/diff/expanding-current-session transforms. No future labels, bfill, full-sample scaler, or BTC lead-lag features were added.
- Audit verdict after generation on 2025/2026 active frames:
  - `KEEP_RISK_CONTEXT=3`: `atr_pct_rank_288`, `compression_score`, `bb_width_pct_rank_288`
  - `SECONDARY_CONTEXT=22`
  - `LOW_SIGNAL_SECONDARY=10`
  - no high-drift `MONITOR_OR_NORMALIZE` under PSI >= 0.50 in this first pass
- Strongest new features:
  - return/context: `compression_score` ret IC about 0.072, `atr_pct_rank_288` about 0.069, `bb_width_pct_rank_288` about 0.063, `vwap_dist_96` about 0.062, `cvd_288` about 0.060
  - risk/vol: `atr_pct_rank_288` vol IC about 0.212, `compression_score` about 0.205, `bb_width_pct_rank_288` about 0.184, `vwap_dist_288` about 0.119, `cvd_288` about 0.111
- Interpretation: the new block still looks stronger as compression/risk/context than pure direction. Use it first in entry-context/risk-meta ablations; do not promote as a direct direction owner without OOS backtest improvement.
- Secondary artifacts implication: M7/AI/regime outputs remain blind to these new features unless those artifacts are retrained from the 2024-only feature-learning stage and then 2025/2026 are rescored. Existing live artifacts are still valid, but they do not include this new alpha block.
- Next required validation before promotion: rerun feature audit and Alpha7.1 layer-specific ablations to test whether the new direction block improves entry context without hurting OOS MDD.

### Funding-Clean Artifact Remediation - 2026-05-29

- Full record: `docs/audits/funding_clean_retrain_rescore_20260529.md`
- Root bug: historical year split `last_funding_rate` used future-filled funding and wrong ETHFIUSDT funding for 2025/2026. Correct contract is ETHUSDT-only backward/as-of alignment.
- M7 active artifacts were retrained from clean funding splits and active artifacts replaced:
  - `data/ensemble/supervised/entry_price_model.{json,pkl}`
  - `data/ensemble/supervised/trend_xgb.{json,pkl}`
  - `data/ensemble/supervised/multi_target_lgbm.{json,pkl}`
  - `data/ensemble/supervised/quantile_forest.{json,pkl}`
  - `data/ensemble/unsupervised/vae_anomaly.{json,pkl}`
- M7 rescored clean CSVs:
  - `data/splits/year_oos/rl_training_2025_m7.csv`
  - `data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv`
- Active future-regime sidecars regenerated under the existing `regime4_pred_tft_h12_nomdjd_all74_20260517` contract: h12, all74, excluding `pred_mdjd` and `conf_mdjd`.
- Alpha5 `a5dir` / router clean rebuild:
  - run dir: `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48`
  - score CSV: `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/08_alpha5_direction_router_rl_2024_to_2025/rl_training_2025_direction_router.csv`
  - router validation balanced accuracy: `0.6268177783`
  - router OOS balanced accuracy: `0.5490030238`
  - scored 2025 `a5dir_available_ratio=0.850691`
- Funding validation after the clean run: 2025 M7, 2026 M7, and 2025 A5Dir score CSV all match the clean feature-frame `last_funding_rate` with max abs diff `0.0`.
- Architecture implication: old DSAC checkpoints, Alpha6/Alpha7 policy artifacts, cached unified datasets, and older downstream score CSVs remain stale/suspect unless their manifests reference the clean funding run or they are explicitly retrained/rescored.
- Mandatory design gate: every new Alpha6/Alpha7/Alpha8/RL/DSAC/IQN/Mamba candidate must declare whether it consumes funding-family columns or artifacts that may embed them. If yes, the model contract must show clean funding provenance before the result can be promoted.
- Acceptable clean funding proof: artifact path under `tmp/causal_regen_20260516/funding_clean_retrain_20260529`, a manifest/report naming the clean funding run, or a direct timestamp join against clean split files where `last_funding_rate` max absolute difference is `0.0`.
- Known stale-risk input: `tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/trade_candidates_20*_alpha6_current_tail111_exact.csv`. It mismatches clean split funding values and should be treated as research-only until regenerated from clean funding inputs.

## Responsibilities

- active/live candidate 설계에서는 compatibility shim, alias prefix, fallback contract를 기본 해법으로 쓰지 않는다.
- feature/state/artifact contract가 바뀌면 downstream이 바로 깨지도록 두고, 필요한 모델/데이터/코드를 함께 재학습/재생성하는 fail-fast 설계를 우선한다.
- 모델 계층도를 유지한다: supervised predictors -> unsupervised regime/anomaly -> DSAC/controller -> final governor/risk overlay.
- `DSAC_STATE_DIM`, state schema, checkpoint metadata, live router 입력이 서로 어긋나지 않게 관리한다.
- 지도학습 모델은 방향 예측뿐 아니라 expectancy, quantile, entry/exit hazard, lifecycle decision으로 역할을 분리한다.
- 지도학습 리스크 모델의 head/sidecar는 `tp_price_move`, `sl_price_move`, `margin_fraction`, `leverage`, `notional`의 의미를 명시적으로 구분한다. `take_profit`/`stop_loss` account-PnL threshold는 label/head가 아니라 `price_move * notional`로 파생되는 실행값이어야 하며, `notional = margin_fraction * leverage` 뒤 leverage를 다시 곱하지 않는다.
- 비지도학습 모델은 regime/anomaly/gating에만 사용하고, 방향 신호로 과신하지 않는다.
- RL은 compact state를 소비해야 하며 raw feature sprawl을 직접 흡수하지 않는다.
- 온라인 학습은 shadow replay, delayed label, drift report, rollback checkpoint를 갖춘 뒤에만 live candidate로 올린다.
- 새 모델 설계가 나올 때마다 `docs/model_contracts/`에 layer input, dataset split, label, output, artifact 계약서를 작성하고 `docs/model_contracts/registry.json`에 등록한다.
- `FEATURE_COLS`, private account state, baseline telemetry, quality/stale state를 구분해 state contract를 설계한다.
- timestamp merge는 backward/as-of causal alignment만 허용하고, `bfill`, future rolling target, full-sample scaler는 금지한다.
- 데이터 품질 저하 시 모델이 aggressive entry로 반응하지 않도록 `quality_state`, stale flag, invalid mask를 state에 포함시키는 방안을 설계한다.
- 노이즈 필터는 micro noise, volatility regime, anomaly, tail event, source health를 목적별로 분리한다.

## Architecture Principles

- `controller`처럼 비정상적으로 큰 성과가 나오면 먼저 leverage/resize/fee bug 가능성을 검토한다.
- 목표 함수는 최소한 `score = risk_adjusted_pnl - MDD_penalty - cost_turnover_penalty - trade_sparsity_penalty` 형태로 다목적이어야 한다.
- 모델 후보는 `base`, `compact`, `controller`, `fully_learned governor`, `lifecycle manager`를 같은 OOS 비용 가정에서 비교한다.
- regime specialist는 bull/bear/chop/whipsaw/normal별 유효 구간과 무효 구간을 명시한다.
- action output과 risk output은 분리한다. 방향 모델이 레버리지까지 모두 학습하면 MDD와 비용 버그를 숨기기 쉽다.
- 앙상블은 단순 평균보다 single-owner sleeve, priority, veto/gate 구조를 우선한다. 한 포지션은 한 owner가 책임져야 journal, 손실 통제, rollback이 가능하다.
- raw feature를 모델에 직접 계속 추가하지 않고 compact state, M7 summary, calibrated diagnostic head로 압축한다.
- stale 체결 흐름은 0 방향 신호가 아니라 중립값과 invalid mask로 처리한다.

## Standing Memory - Candidate Model Architecture Reference 2026-05-17

이 목록은 새 모델링 아이디어를 낼 때 참고하는 후보군이다. 순위는 금융 데이터의 비정상성, 노이즈, 체제 전환, 코인 선물 실시간 운용을 고려한 실용적 우선순위이며, 채택 기준은 항상 OOS, 비용 stress, Red Team, runtime-native backtest parity가 우선한다.

### Unsupervised State, Regime, And Clustering

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | Hidden Markov Model (HMM) | market regime/state transition detection |
| 2 | Gaussian Mixture Model (GMM/BGMM) | return/state distribution decomposition and soft state priors |
| 3 | K-Means | simple asset/state clustering baseline |
| 4 | DBSCAN | abnormal trade/state density detection |
| 5 | Hierarchical Clustering | sector/asset grouping and exploratory clustering |

Guidance:

- Prefer HMM when temporal persistence and transition probabilities matter.
- Prefer GMM/BGMM when soft state membership is needed without a strong Markov assumption.
- Do not expose `hmm_*`, legacy HDBSCAN, or unproven regime IDs directly as model features unless a new 2024-only contract explicitly allows them.

### Representation And Anomaly Models

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | Autoencoder / VAE | latent state compression and anomaly detection |
| 2 | PCA | factor decomposition and noise reduction |
| 3 | ICA | independent factor separation |
| 4 | t-SNE / UMAP | visualization only, not production prediction input |

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | Isolation Forest | price/volume/order-flow anomaly detection |
| 2 | LSTM Autoencoder | sequence anomaly detection |
| 3 | One-Class SVM | normal-pattern boundary baseline |
| 4 | LOF | local density anomaly baseline |

Guidance:

- Use representation/anomaly models as state summaries, vetoes, or sizing inputs, not direct entry owners without downstream validation.
- Any latent feature artifact must record fit range, source columns, scaler policy, and live missing/stale fallback.

### Deep Sequence Models

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | Temporal Fusion Transformer (TFT) | multi-scale sequence prediction with variable importance |
| 2 | TCN | stable long-context supervised sequence baseline |
| 3 | LSTM / BiLSTM | validated recurrent baseline |
| 4 | Transformer variants, Informer, PatchTST | global context and patch-based forecasting |
| 5 | WaveNet | micro price/tick pattern modeling |
| 6 | N-BEATS / N-HiTS | pure forecasting baseline |
| 7 | GRU | lightweight recurrent inference |

Relationship models:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | GNN | cross-asset correlation, contagion, relative-flow modeling |
| 2 | Attention CNN | local pattern extraction with attention |
| 3 | Dual-stream CNN + LSTM | parallel short-pattern and sequence-dependency modeling |

Guidance:

- Deep sequence models must use explicit train/validation/OOS split, no full-series normalization, and horizon purge/embargo where labels use future paths.
- Runtime inference must match training sequence length and feature preparation exactly.

### Reinforcement Learning

Continuous action space:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | DSAC / Distributional SAC | continuous sizing with tail-risk/CVaR awareness |
| 2 | SAC | entropy-regularized robust continuous policy |
| 3 | TD3 | deterministic continuous control with overestimation control |
| 4 | PPO | stable general-purpose policy optimization |
| 5 | DDPG | legacy continuous-control baseline |

Discrete action space:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | Dueling DQN + PER | buy/sell/hold with state-value separation and rare replay |
| 2 | Rainbow DQN | combined DQN improvements |
| 3 | Distributional DQN / C51 | return distribution learning |
| 4 | QRDQN | quantile distribution learning |

Multi-agent / ensemble RL:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | MADDPG | cooperative/competitive multi-agent policies |
| 2 | MAPPO | stable multi-agent PPO |
| 3 | MoE + RL | regime/specialist routing and expert allocation |

Guidance:

- RL state must stay compact and audited; raw feature sprawl belongs in supervised/state summary layers.
- Promotion requires identical accounting, fee/slippage stress, leverage cap, and runtime-native replay.

### Supervised Tabular And Sequence Learning

Classification:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | LightGBM | fast tabular direction/state/quality prediction |
| 2 | XGBoost | robust tabular ensemble |
| 3 | CatBoost | categorical/mixed feature robustness |
| 4 | Random Forest | low-maintenance noisy baseline |
| 5 | SVM RBF | small high-dimensional baseline |
| 6 | Logistic Regression | interpretable sanity baseline |

Regression:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | LightGBM Regressor | volatility, slippage, expectancy prediction |
| 2 | Quantile Regression | uncertainty interval and tail estimates |
| 3 | Gaussian Process Regression | distributional uncertainty for small data |
| 4 | Ridge / ElasticNet | linear factor baseline |

Sequence-supervised:

| Priority | Model | Primary Use |
|---:|---|---|
| 1 | LSTM + Attention | supervised sequence labels |
| 2 | Transformer | global sequence pattern labels |
| 3 | CNN-LSTM Hybrid | local pattern plus temporal dependency labels |

Guidance:

- Tabular supervised models are the default first-pass for new causal features because they are fast, interpretable, and easy to ablate.
- Sequence-supervised models require stronger parity checks because sequence construction, padding, and latest-row inference can silently diverge between train and live.

## Integrated State Contract

상태는 다음 계층으로 분리한다.

| Layer | Examples | Use |
|---|---|---|
| Raw market | OHLCV, taker flow, OI, funding, BTC-relative fields | causal source data |
| Derived market | volatility, trend, order-flow, funding pressure | feature engine |
| Regime/context | bull/bear/chop/whipsaw/normal, liquidity vacuum | threshold/gate selector |
| Model summaries | M7 probabilities, quantiles, anomaly, expected hold | compact model input |
| Private account state | position side, unrealized PnL, hold bars, drawdown, margin, leverage | runtime/lifecycle state only |
| Quality state | stale flags, missing ratios, timestamp gap, schema version, source health | veto/size-down and live gating |

## Required Review Checklist

- 학습/평가 split과 timestamp overlap audit이 있는가?
- 새 모델별 `docs/model_contracts/` 계약서에 layer input, train/validation/test 범위, label, output schema, artifact/report path가 남는가?
- live state builder와 train state builder가 같은 feature alias와 normalization을 쓰는가?

## Standing Memory - Regime4 Official MoE Diagnostic 2026-05-17

Regime taxonomy for new regime work is fixed to four classes:

```text
bull
bear
chop
whipsaw
```

`normal`, `risk_off`, and `transition` are not official regime classes.

Official sidecars:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2025_clean_regime4_state24_sticky090_v2.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
```

Active DSAC/Router specs must use renamed current Regime4 columns under `clean_regime4_state24_sticky090_v2_*`. The historical export prefix `clean_regime4_2024_unsup_v1_*` is allowed only for reproduction of older artifacts and must not appear in promoted active specs.

Downstream diagnostic:

```text
/home/llewyn/crypto-scalping/docs/experiments/regime4_official_moe_2025_ablation_20260517.md
/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_official_moe_2025_ablation_20260517.json
```

Result: `regime4_both` beat the baseline only marginally on 2025 holdout MDD/PnL, but absolute performance was negative (`cost1 -13.39%`, MDD `-14.08%`, trade Sharpe `-1.99`). Do not promote a standalone Regime4 direction MoE from this result. If Regime4 is reused, prefer veto/sizing/context on an already profitable owner or train experts on expectancy/owner selection instead of only long/short/no-trade direction.
- `DSAC_STATE_DIM`과 checkpoint `state_dim`이 일치하는가?
- M7 dropout/noise가 train에만 적용되고 live에는 적용되지 않는가?
- OOF, embargo, 미래 label 격리가 지도학습/fully learned/lifecycle 모델에 적용되는가?
- rolling 통계가 현재 시점 이전 데이터만 사용하는가?
- warmup NaN 처리, stale 처리, missing fallback에 future leak이 없는가?
- live DuckDB/source health/schema_version이 실시간 state에 반영되는가?
- 비용 1x/2x/3x stress에서 순위가 유지되는가?
- 월별 또는 주별 walk-forward에서 특정 구간 하나에만 의존하지 않는가?
- trade count가 너무 적어 실전 거래량 목표를 못 맞추지 않는가?

## Standing Memory - Defensive Mode Candidate 2026-05-14

- Alpha2.1 teacher architecture ablation에서 `hgb_meta_task_attention_focal`은 메인 알파로 승격하지 않는다.
- 같은 Alpha2.1 L2 replay 계약에서 OOS `cost1 PnL +223.63% / MDD -15.34% / cost2 +148.04% / cost3 +127.16%`로, 기존 Alpha2.1 reference `cost1 +718.70% / MDD -26.66%`보다 수익률은 크게 낮지만 MDD 방어력이 좋았다.
- 향후 설계에서 이 모델은 `defensive_mode_candidate` 또는 drawdown recovery/risk-off sleeve 후보로만 취급한다.
- 메인 알파를 대체하려면 selection 점수와 cost1/cost2/cost3 모두에서 기존 Alpha2.1 또는 Alpha2 reference를 넘어야 한다.
- 관련 산출물:
  - `data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_summary.json`
  - `data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_grid.csv`
  - `data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_audit.json`

## Standing Memory - Default Execution Contract 2026-05-14

- Alpha3의 기본 주문 체결 후보는 corrected `next_open_limit_touch0_fee20` 계약이다.
- 이 계약을 적용한 corrected selected 모델을 `alpha3`로 지칭한다.
- 계약:
  - 신호 bar `i` 확정 후 다음 실행 bar `i+1`에서 post-only limit touch 여부를 평가한다.
  - 진입/청산 지정가 offset은 `0 bps`, penetration은 `0 bps`, maker fee multiplier는 `0.20`이다.
  - 진입 maker miss는 기본 selected 계약에서 `skip`한다.
  - 청산 maker miss는 reduce-only market fallback으로 계좌 상태를 정리한다.
  - OHLCV replay에서 `i+1 high/low`를 보고 maker touch를 판정했다면, fallback은 같은 bar open이 아니라 `i+1 close +/- slippage`만 허용한다.
- Alpha2.1/Alpha3 OOS 참고값:
  - next-open taker only: `cost1 +354.53% / MDD -32.45% / cost2 +23.13% / cost3 +10.80%`
  - old config retest with close fallback: `cost1 +358.84% / MDD -27.59% / cost2 +283.14% / cost3 +215.42%`
  - corrected Alpha3 selected `next_open_limit_touch0_fee20`: `cost1 +654.92% / MDD -29.62% / cost2 +602.26% / cost3 +456.48%`
- 단, 이 계약은 아직 5m OHLC touch proxy 기반이다. real L2 queue/partial fill/post-only reject 검증 전에는 “shadow/promising execution contract”로 표기한다.
- 모든 신규 백테스트/주문체결 설계는 `taker-only`, corrected OHLCV touch replay, live L2 queue/partial-fill shadow audit을 최소 비교군으로 포함한다.

## Standing Memory - Alpha3 Alias 2026-05-14

- `alpha3` = `Alpha2.1 teacher gate + HGB parent + V21.2 jackpot + frozen V27 deep scout + V31 exit overlay + corrected next_open_limit_touch0_fee20 execution`.
- 주요 OOS 기준값: `cost1 +654.92% / MDD -29.62% / trades/day 3.32`, `cost2 +602.26%`, `cost3 +456.48%`.
- 실행 config는 `next_open_limit_touch0_fee20`, offset `0 bps`, penetration `0 bps`, maker fee multiplier `0.20`, entry miss `skip`, exit miss `market_fallback`이다.
- deprecated: 기존 `+747.76%` Alpha3 숫자는 same-next-bar open fallback 버그가 있어 비교 기준으로 쓰지 않는다.
- Red Team 주의점: 이 성과는 아직 5m OHLC touch proxy 기반 지정가 체결 가정이 포함되어 있으므로 real L2 queue/partial fill/post-only reject 검증 전에는 clean live PnL로 단정하지 않는다.
- 새 설계는 Alpha3를 기준선으로 비교하고, Alpha3보다 좋은 후보는 `alpha3.x`로 명명한다.

## Standing Memory - Alpha3 Runtime-Native Parity 2026-05-16

- Alpha3 모델 구조를 설계할 때 기준 실행 환경은 `docs/model_contracts/alpha3_csv_native_backtest_parity_20260516.md`를 따른다.
- CSV loop exact parity와 live/runtime-native parity를 구분한다. CSV loop는 canonical debugging baseline이고, 모델 개선 후보의 기본 검증은 `FinalGovernorRuntime.decide()`를 순차 호출하는 runtime-native path다.
- 현재 runtime-native 1개월 parity 결과:
  - report: `data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m.json`
  - ledger: `data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m_ledger.csv`
  - CSV 기준: `+338.680671% / MDD -29.617313%`
  - runtime-native: `+338.679873% / MDD -29.617095%`
  - action event counts match exactly: `OPEN 114`, `CLOSE 113`, `UPSIZE 9`, `FORCED_END 1`.
- 모델 입력 계약:
  - Alpha2.1 teacher는 72-bar sequence model이다. 최신 row 단독 입력은 금지한다.
  - Parent/teacher/V31는 frame-level inference 후 latest row를 사용한다.
  - V21.2 jackpot add-on은 Alpha2.1-constrained decision row와 parent-bundle feature frame을 사용한다.
  - V21.2 add-on state는 current unrealized, cumulative `MFE`, cumulative `MAE`, account drawdown, parent TP/SL/max_hold를 포함한다.
  - `mae_so_far`를 현재 손익 `min(0, current_unrealized)`로 대체하면 add-on gate가 바뀐다. 실제로 local `i=3387`에서 `p_jackpot`이 threshold 근처에서 달라져 action parity가 깨졌다.
- 새 parent/exit/RL 설계는 반드시 어떤 state를 변경하는지 명시해야 한다. 특히 MFE/MAE, cooldown, forced-end, execution route는 모델 성능이 아니라 backtest contract surface로 취급한다.
- 새 모델이 baseline보다 좋아 보이면 먼저 action ledger parity가 깨진 위치가 선언한 mutable layer인지 확인한다. 선언한 layer 이전의 차이는 성능이 아니라 계약 위반이다.

## Standing Memory - Regime4 Alpha5/Alpha5.1 2026-05-17

- Regime4 공식 실험 라인은 historical/reference-only로 4-class `bull/bear/chop/whipsaw`였다. New action-classifier/Alpha8+ candidates should not continue this as the preferred target; use the 2026-05-29 Regime3 + Whipsaw Risk policy instead.
- In the historical Regime4 line, `normal`, `risk_off`, `transition` are not classes.
- 고정 전처리 계약은 `docs/model_contracts/fixed_regime4_tp18_sl10_preprocess_20260517_contract.md`를 따른다.
  - historical current regime export prefix: HMM `clean_regime4_2024_unsup_v1_*`
  - active DSAC/current regime prefix: `clean_regime4_state24_sticky090_v2_*`
  - current auxiliary: `factor_trend`, `factor_flow`, `factor_vol`, `factor_crowding`, `factor_liquidity`, `trend_bias`, `risk_off_prob`, `transition_risk`
  - future regime: TFT `regime4_pred_*`
  - TP/SL 보조 피쳐: `tp_sl_action_score`, TP 1.8%, SL 1.0%, 48 bars, next-bar open, same-bar tie SL wins
  - legacy `clean_regime_2024_unsup_v4_*`는 Alpha5뿐 아니라 active live/backtest/model-candidate 입력 전체에서 금지한다.
- Alpha5는 Alpha4.3 no-teacher/no-deep 구조를 Regime4 fixed preprocessing으로 재학습했다.
  - contract: `docs/model_contracts/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517_contract.md`
  - 2026 OOS selected: cost1 `+86.93% / MDD -24.44%`, cost2 `+78.99%`, cost3 `+72.26%`, trades `66`
  - verdict: positive OOS but below Alpha4.3 reference.
- Alpha5.1은 `tp_sl_action_score`와 current/future Regime4의 22개 multiplicative interaction을 추가했다.
  - contract: `docs/model_contracts/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517_contract.md`
  - feature contract: 107 features, old clean count 0, current Regime4 12, future Regime4 12, interaction 22
  - 2026 OOS selected: cost1 `+65.18% / MDD -23.82%`, cost2 `+68.70%`, cost3 `+65.06%`, trades `75`
  - verdict: failed candidate. 단순 TP/SL x Regime4 crossed tabular features는 Alpha5 개선 방향으로 반복하지 않는다.
- 다음 Alpha5 개선은 crossed feature expansion이 아니라 regime-conditioned calibration, MoE routing, specialist heads, 또는 walk-forward specialist selection으로 설계한다.
- 2026-05-17 후속 수정: current Regime4 sidecar는 12개에서 20개로 확장되었다. `risk_off_prob`와 `transition_risk`는 별도 레짐 class가 아니라 auxiliary score로만 사용한다. 이 확장 이후 기존 Alpha5/5.1 parent artifact는 새 canonical fixed CSV와 비교할 때 재학습 필요 상태다.
- Forward canonical HMM Regime4 feature artifact: `clean_regime4_state24_sticky090_v2_2024.joblib`. 앞으로 HMM 레짐 피쳐/라우터/DSAC 계약을 새로 설계하거나 재학습할 때는 이 artifact 이름과 `clean_regime4_state24_sticky090_v2_*` prefix를 기준으로 삼는다. 구형 `clean_regime4_2024_unsup_v1_*` 계열은 active DSAC specs에서 제거하고, historical legacy/비교선으로만 취급한다.
- DSAC fixed inventory rule: base CSV에 남아 있는 `clean_regime4_2024_unsup_v1_*` 12개는 drop하고, state24 sidecar 전체 20개를 `clean_regime4_state24_sticky090_v2_*`로 re-merge한다. 새 feature spec에 `clean_regime4_2024_unsup_v1_*`가 1개라도 있으면 계약 위반이다.
- Fixed DSAC spec verification path: `tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521/`. 현재 fixed specs는 legacy prefix 0개, state24 prefix 사용 상태다.
- Current DSAC candidate architecture:
  - current regime context: `clean_regime4_state24_sticky090_v2_*` from `clean_regime4_state24_sticky090_v2_2024.joblib`.
  - future regime context: `regime4_pred_*` from `regime4_pred_tft_h12_nomdjd_all74_20260517`.
  - CatBoost/Router context: `a5dir_*` from Router5 fixed `0.8 * Router3 + 0.2 * Router4`.
  - final action owner: DSAC only. Regime/TFT/Router5 are auxiliary features, not direct trade owners.
  - deprecated direct action owner: CatBoost Major/Direction `LONG/SHORT/CASH` ownership is forbidden in active live/backtest paths.
- Alpha6 active research main as of 2026-05-22:
  - contract: `docs/model_contracts/alpha6_entry_quality_exit_5bucket_main_20260522_contract.md`
  - artifact dir: `data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/`
  - structure: CatBoost `action + quality + 5-bucket target horizon + position-aware exit`
  - target bucket mapping: `0->6`, `1->12`, `2->24`, `3->48`, `4->96` bars
  - selected thresholds: entry `0.0034163351358086967`, exit `0.35`
  - selected current_tail111 Cost1/2/3 PnL: `+15.30% / +14.24% / +12.04%`; MDD about `-4.81%`
  - boundary: owns entry/quality/exit alpha only; notional, SL, TP are reserved for DSAC/execution-layer work.
- Alpha5.2는 위 20개 current Regime4 feature를 사용한 feature-only retest다.
  - contract: `docs/model_contracts/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517_contract.md`
  - selection/runner/architecture는 Alpha5 그대로이고, changed surface는 parent input feature뿐이다.
  - 2026 OOS selected: cost1 `+83.24% / MDD -26.91%`, cost2 `+73.79%`, cost3 `+70.68%`, trades `58`
  - verdict: failed candidate. 단순 factor bridge 복원만으로 Alpha4.3 성능은 회복되지 않았다.
- Alpha4.3 legacy regime block ablation:
  - contract: `docs/model_contracts/alpha4_3_legacy_regime_block_ablation_20260517_contract.md`
  - Alpha4.3 feature basis에서 23개 legacy regime block을 하위 그룹별로 parent/runner 재학습했다.
  - 2026 OOS best diagnostic은 `no_legacy`: cost1 `+139.59% / MDD -25.52%`, cost2 `+131.83%`, cost3 `+139.82%`.
  - `all_legacy`: cost1 `+113.88% / MDD -21.01%`, cost2 `+109.56%`, cost3 `+104.56%`.
  - 결론: legacy regime subfeature가 Alpha4.3 PnL의 직접 원천은 아니다. 전체 block은 MDD를 낮추는 risk/regularization 성격이 강하고 PnL은 깎는다. Alpha4.3 edge는 특정 parent artifact, runner/runtime coupling, Alpha4.2 training path 쪽을 추가 분해해야 한다.
- Alpha4.3 fixed-artifact inference masking:
  - contract: `docs/model_contracts/alpha4_3_legacy_regime_inference_mask_20260517_contract.md`
  - parent/runner/runtime을 고정하고 legacy group만 2025 train median으로 masking했다.
  - baseline reproduced exactly: cost1 `+183.42% / MDD -21.99%`, cost2 `+169.76%`, cost3 `+79.27%`.
  - cost1 delta when masked: `semantic_probs -114.01%`, `risk_transition -87.40%`, `factor_core -39.30%`.
  - `cluster_state` masking improves PnL: cost1 `+10.37%`, cost3 `+54.12%`.
  - 최종 해석: Alpha4.3 artifact 안에서 실제 positive legacy signal은 semantic probabilities/confidence/entropy, risk/transition, factor_core다. cluster/state-code 계열은 harmful/overfit 가능성이 높아 Alpha5 parent로 이식하지 않는다.

## Standing Memory - Alpha5.3 HMM DQN Router Parent 2026-05-17

- User decision: do not use `clean_regime_2024_unsup_v4_*`; use HMM Regime4 only. `clean_regime_2024_unsup_v4_*`는 historical reproduction/debug scope로만 제한한다.
- Forward canonical HMM Regime4 artifact: `clean_regime4_state24_sticky090_v2_2024.joblib`.
- Architecture contract: `docs/model_contracts/alpha5_3_hmm_dqn_router_parent_20260517_contract.md`
- Script: `scripts/train_eval_alpha5_3_hmm_dqn_router_parent_20260517.py`
- Design:
  - HMM Regime4 four probabilities are routing state, not specialist parent input.
  - Four specialist parents: bull/bear/chop/whipsaw.
  - Each specialist uses only a Dueling DQN + PER-like prioritized replay action parent.
  - Specialist parent output contract is exactly `action_prob_long`, `action_prob_short`, `action_prob_cash`.
  - Notional, leverage, TP/SL, hold, cooldown, quality, and bucket heads are not specialist parent outputs.
  - Evaluation is action-only: routed DQN action enters while flat, exits on cash, and flips on opposite side. No fixed TP/SL, max-hold, cooldown, or quality-score constants are used.
  - `regime4_pred_*` TFT future features are excluded in this HMM-only line.
  - legacy clean v4, normal, cluster/state-code are forbidden.
- Evaluation modes:
  - `hard_current`
  - `soft_current_th0.00`
  - `soft_current_th0.05`
  - `soft_current_th0.10`
- Status: architecture implemented and compile-checked; full backtest not yet run.

## Feature Audit Memory - Directional Alpha 2026-05-28

- Directional audit: `docs/audits/directional_alpha_feature_audit_20260528.md`
- Full direction-candidate universe audit: `docs/audits/directional_feature_universe_audit_20260528.md`
- Source-required contract: `docs/audits/source_required_direction_features_20260528.md`
- Active engineered block now includes 48 causal direction/context features:
  - CVD/compression/VWAP/funding-OI/wick-sweep block.
  - BTC lead-lag block from existing `close_btc`, `volume_btc`, `quote_volume_btc`.
- BTC additions with the best standalone OOS tendency:
  - `eth_btc_ret_spread_12`: ret IC about `0.058`, PSI about `0.094`.
  - `btc_lead_eth_follow_gap_3`: ret IC about `0.052`, PSI about `0.104`.
  - `btc_volume_impulse_z`: risk/volatility context, vol IC about `0.193`, PSI about `0.000`.
- Do not add unavailable orderbook/liquidation/basis/on-chain features as zero defaults. They require persisted historical sources and exact active contracts. Missing active columns must fail fast.
- Existing M7/AI/regime artifacts are blind to this block until regenerated with 2024-only training and 2025/2026 causal rescoring.
- Full universe audit summary:
  - 220 direction-like candidates scored.
  - After funding remediation, strongest non-model direction-context market candidates are `last_funding_rate`, `long_squeeze_risk`, `crowding_pressure`, `funding_roc_288`, `funding_pressure`, and `regime4_pred_instability_prob`. M7 direction-context features are removed from active/candidate downstream inputs by user decision.
  - `squeeze_power` has high ret IC but remains monitor/veto only because OOS drift is high.
  - Duplicate/drop verdicts remain authoritative even if IC is high: use representatives for `regime4_pred_whipsaw_prob`, `regime4_pred_range_prob`, sticky `*_whipsaw_prob`, sticky `*_range_prob`; do not directly use raw M7 price outputs.
  - Family HGB probe OOS AUC is weak overall; `regime_pred` is highest for `dir24_up` at about `0.539`, so direction features are context inputs, not standalone owners.

## Funding Feature Audit Memory - 2026-05-28

- Audit: `docs/audits/last_funding_rate_source_audit_20260528.md`
- Historical bug: old year splits used future funding alignment; 2025/2026 also used ETHFIUSDT funding.
- Remediation complete for direct split columns: `data/splits/year_oos/training_features_2024.csv`, `training_features_2025.csv`, `training_features_2026_rebuilt.csv`, `rl_base_*`, and M7 RL CSV direct funding columns now match previous ETHUSDT funding at `100%`.
- Remaining action: retrain/rescore derived artifacts that consumed contaminated funding columns before remediation, especially M7/teacher/regime sidecars and any policy model embedding old funding-family behavior.

## AI Direction Feature Retrain Memory - 2026-05-28

- Audit/report: `docs/audits/ai_direction_feature_retrain_20260528.md`
- Current `ai_dir_*` is not a calibrated direction classifier. It is produced by `ensemble/ensemble_router.py` from a PatchTST scalar edge forecast, then mapped into pseudo probabilities.
- TiDE/TimesNet/DLinear AI features are risk/anchor/flow context, not direct direction owners.
- The first retrain artifact `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v1` is invalid and marked `INVALID_DO_NOT_USE.md` because label-derived score columns initially leaked into the feature matrix.
- Valid no-leak artifacts:
  - `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v2_noleak`
  - `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v3_strict_noleak`
- Superseded active decision 2026-05-30: do not route `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, or `patchtst_median` into active/candidate downstream inputs. Keep these artifacts only for historical/research reproduction unless a new direction-model contract is explicitly created.

## Secondary Feature DAG Memory - 2026-05-30

- Contract: `docs/audits/secondary_feature_generation_contract_20260530.md`
- `teacher_*` is downstream of AI/TSFM plus M7. It must never be used as input to AI/TSFM, M7, regime, or `a5dir_*` generation.
- Current side-teacher outputs are generated by `pipeline/teacher_meta_side_features.py`: `teacher_long_edge`, `teacher_short_edge`, `teacher_side_margin`, `teacher_side_disagreement`, `teacher_quantile_skew`, `teacher_uncertainty`, and `teacher_tail_warning`.
- Teacher inputs are same-timestamp risk/uncertainty AI/TSFM outputs and M7 risk/quality outputs after clean score-only generation. Direction outputs such as `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, `patchtst_median`, and removed M7 direction-context columns are not active downstream inputs. Teacher is final-policy context only.
- `a5dir_*` and PCA/compressed feature variants are also downstream experiment/policy surfaces. They must not feed upstream feature/model generators under existing prefixes.
- AI/TSFM outputs, M7 outputs, and current/future regime outputs should not be cross-fed into each other under existing prefixes. Intentional coupling requires a new versioned artifact and no-leak audit.

## HF Offline Model Inventory Memory - 2026-05-30

- Inventory-only document: `docs/audits/hf_offline_model_inventory_20260530.md`
- Cached candidates exist for PatchTSMixer, Chronos, Moirai, TimesFM, Granite TTM, Kairos, Kronos, and Lag-Llama.
- This is not an AI feature contract. Do not assume input columns, target labels, horizons, or output prefix yet.
- `timesfm` Python package is missing in `quant_ai`; TimesFM remains blocked until runtime support is installed or proven.

## AI PatchTSMixer Direction Core Memory - 2026-05-30

- Experiment report: `docs/audits/ai_patchmix_direction_core_20260530.md`
- New feature family: `ai_patch_*`
- Generated score files:
  - `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2024_score2025/ai_patchmix_direction_core_2025.csv`
  - `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2025_score2026/ai_patchmix_direction_core_2026.csv`
- Best diagnostic signal is `ai_patch_h12_*`: 2026 balanced accuracy about `0.4854`, OVR AUC about `0.6492`.
- h24/h48 should be treated as auxiliary ranking/context until flat handling improves.
- Recommended downstream test order:
  1. add `ai_patch_h12_edge`, `ai_patch_h12_conf`, `ai_patch_h12_entropy` to entry/meta layer;
  2. add `ai_patch_consensus`, `ai_patch_edge_mean`, `ai_patch_risk_adj_edge`;
  3. only then test h24/h48 columns.

## AI PatchTSMixer Direction Input Rework Memory - 2026-05-30

- Rework report: `docs/audits/ai_patchmix_direction_input_rework_20260530.md`
- Updated builder: `scripts/build_ai_patchmix_direction_core_20260530.py`
- Added audited input profiles:
  - `audit_full`: all audited direct upstream context features present in year-OOS splits.
  - `audit_compact`: reduced high-value direction/context subset from the document-manager feature audit.
- Contract remains upstream-only. Do not use `teacher_*`, `m7_*`, `a5dir_*`, existing AI/TSFM outputs, regime sidecars, labels, targets, future path, or PnL-derived columns.
- Best current rework: `audit_compact`, especially for h24 direction context.
  - 2026 h24 balanced accuracy: `0.365542` -> `0.426686`
  - 2026 h24 OVR AUC: `0.603470` -> `0.616559`
- Do not promote `audit_full`; it is noisier despite some AUC gains.
- Recommended downstream usage:
  1. test `audit_compact` `ai_patch_h24_edge/conf/entropy` in entry/meta layer;
  2. keep original h12 or `audit_compact` h12 as soft context only;
  3. use h48 only as long-horizon ranking/risk context after ablation.

## AI 4-Model H6 BACC Loop Memory - 2026-05-30

- Loop report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- User preference: h6 first, because Regime3 stability/risk sidecar is h6.
- Exclusions are explicit: no `regime3_pred_*`, no Regime4 sidecars, no `teacher_*`, no `m7_*`, no `a5dir_*`, no future/target/PnL-derived inputs.
- Current best strict-clean h6 direction surface:
  - Chronos h6 zero-shot + compact current/core + split-local regime context
  - 2026 OOS bacc `0.5009`, OVR AUC `0.6832`
- Best research-only stack:
  - Chronos h6 + compact current/core + split-local regime + old TiDE outputs
  - 2026 OOS bacc `0.5020`, OVR AUC `0.6841`
  - not active/live promotable until old TiDE/PatchTST/DLinear outputs are regenerated with exact timestamp coverage and no NaNs.
- Four-family comparison before Chronos did not improve bacc:
  - old PatchTST family bacc `0.3301`
  - TiDE family bacc `0.4703`
  - DLinear family bacc `0.3367`
  - all four output families bacc `0.4786`
- h6 label-boundary sweep improved PatchTSMixer-only bacc to `0.4971` with the `mae_light` label preset.
- User correction: prioritize standalone model quality over AI-output ensembling.
- Current best standalone h6 model:
  - Chronos h6 zero-shot + compact current/core + split-local regime context
  - no other AI output columns
  - label preset: `active_dense`
  - 5-seed mean 2026 OOS bacc `0.5114`, std `0.0012`, max `0.5132`
  - mean OVR AUC `0.6651`
  - artifact: `tmp/causal_regen_20260516/ai_single_model_h6_chronos_core_seedcheck_20260530/summary.json`
- Higher-AUC standalone reference:
  - Chronos h6 + core/local regime with `mae_light`
  - 5-seed mean bacc `0.5013`, mean OVR AUC `0.6834`
  - use as ranking/probability surface rather than hard class owner.
- Interpretation: h6 direction is still not solved. The best standalone model is stable around `0.511` bacc but below the requested 55% bacc. The bacc-optimized model sacrifices ranking AUC, so downstream use should separate class bias from confidence/ranking.

## AI Role-Specific TSFM Evaluation - 2026-05-30

- Report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- Runner: `scripts/run_ai_role_specific_experiments_20260530.py`
- Summary artifact: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/summary.json`
- Exact regenerated role features:
  - 2025: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2025_exact.csv`
  - 2026: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2026_exact.csv`
- Contract: exact timestamp regeneration, no cross-model output ensembling for these role metrics.
- Warmup-only non-finite `tide_vol_zscore` values are explicitly zeroed and recorded in the manifests. Timestamp gaps remain fail-fast.

2026 OOS role findings:

- PatchTST/PatchTSMixer direction role is weak in raw role outputs:
  - h6 bacc `0.3452`
  - h12 bacc `0.3475`
  - h6 up/down AUC `0.4896` / `0.5027`
  - Do not use as standalone hard direction owner from this role test.
- Chronos raw q50-sign role is weak:
  - h6 q50-sign bacc `0.3426`
  - median return correlation `0.0100`
  - large-move AUC `0.5511`
  - Use only as distribution/large-move context unless a supervised calibration layer is fitted.
- TiDE is strong for risk, not direction:
  - h6 top30 adverse AUC raw `0.7354`
  - h12 top30 adverse AUC raw `0.7227`
  - h6 adverse correlation raw `0.3640`
  - Best placement: adverse-risk veto, exit-pressure context, notional/leverage downsize, TP/SL adjustment.
- DLinear direct trend/flow role is weak but may be useful as low-frequency context:
  - h24 trend AUC flow `0.4938`
  - h24 return correlation flow `0.0469`
- TimesNet cycle/session role is weak but nonzero:
  - anchor-revert entry-quality AUC `0.5193`
  - cycle-delta return correlation `-0.0237`

Architecture guidance:

- Keep the standalone Chronos/core `active_dense` model as the current best h6 class-bias surface (`5-seed mean bacc 0.5114`).
- Do not treat all AI models as direction heads.
- Route TiDE to the risk/exit/size layer first.
- Route DLinear and TimesNet to context/ablation lanes, not hard gates.
- Any active/live promotion must pass exact timestamp contract and downstream PnL/MDD/trade-count ablation.

## AI Reworked Input Retrain - 2026-05-30

- Report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- NF reworked runner: `scripts/retrain_ai_role_models_reworked_inputs_20260530.py`
- NF summary: `tmp/causal_regen_20260516/ai_role_models_reworked_inputs_20260530/summary.json`
- PatchTSMixer summary: `tmp/causal_regen_20260516/ai_patchmix_h6_reworked_inputs_20260530/summary.json`
- Existing `data/nf_*` packs were not overwritten.

Training contract:

- TiDE/DLinear reworked NF candidates: 2024-only train, exact 2025/2026 score.
- PatchTSMixer route: `fit2024 -> score2025`, `fit2025 -> score2026`.

Results:

- PatchTSMixer h6 direction recovered after input rework:
  - 2026 h6 bacc `0.5016`
  - 2026 h12 bacc `0.4983`
  - strict `fit2024 -> score2026` h6 bacc `0.5079`
  - strict `fit2024 -> score2026` h12 bacc `0.4821`
  - still below Chronos/core standalone h6 class-bias mean bacc `0.5114`.
  - h12 values are evaluated with the actual h12 head; earlier scratch h12 output that reused h6 predictions is superseded.
- TiDE improved further as adverse-risk model:
  - h6 top30 adverse AUC raw `0.7484` vs previous `0.7354`
  - h12 top30 adverse AUC raw `0.7336` vs previous `0.7227`
  - use first in risk/exit/size layer.
- DLinear did not materially improve:
  - h24 trend AUC flow `0.4929`
  - h24 return correlation flow `0.0472`
  - keep as optional low-frequency drift context only.
- TimesNet full CPU retrain was too slow in this loop and was separated. Do not block the main direction/risk pipeline on TimesNet.

Architecture decision:

- Current AI route priority:
  1. TiDE reworked risk route for adverse-risk veto/resize/exit context.
  2. Chronos/core `active_dense` for standalone h6 class-bias surface.
  3. PatchTSMixer reworked h6/h12 as secondary direction context only.
  4. DLinear as weak flow-drift context only.
  5. TimesNet only after a lightweight cycle/session-specific retrain proves value.

## Chronos Standalone Multi-Series Test - 2026-05-30

User constraint:

- Do not use the downstream CatBoost/meta-head route for this Chronos test.
- Improve AI standalone performance only through multi-series and derived-series inputs.

Artifacts:

- Runner: `scripts/test_chronos_multiseries_standalone_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_multiseries_standalone_20260530/summary.json`

Contract:

- Chronos zero-shot standalone only.
- Threshold/inversion selected on 2025 and fixed on 2026.
- No cross-model ensemble or downstream classifier.

Result:

- Best 2026 OOS standalone bacc: `0.3853` from `price_cvd_divergence`.
- Best large-move context candidates:
  - `price_cvd_divergence`: large-move AUC `0.6539`
  - `vwap_dist_96`: large-move AUC `0.6402`
- Direction AUCs remain near random across tested series.

Decision:

- Chronos standalone multi-series / derived-series output is not a hard direction owner.
- Keep Chronos/core `active_dense` as the better historical Chronos class-bias surface, but do not promote the new standalone Chronos series selection into active path.
- If used later, Chronos should be tested as uncertainty / large-move context only, especially forecast width or magnitude from `price_cvd_divergence` and `vwap_dist_96`.
- TimesNet reworked-input background run produced no summary artifact and is incomplete; do not use it for architecture promotion.

## PatchTSMixer Binary Tradeable Target - 2026-05-30

User request:

- Reuse the expanded feature set if already built.

Artifacts:

- Runner: `scripts/train_ai_patchmix_binary_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchmix_binary_tradeable_20260530/summary.json`

Feature contract:

- `audit_compact_local_regime` PatchTSMixer input.
- Excludes `teacher_*`, `m7_*`, `a5dir_*`, existing `ai_*`, future/target/PnL-derived columns.

Result:

- strict `2024->2026` h6 best: `tradeable_fee2`
  - bacc `0.5249`
  - AUC `0.5368`
  - accuracy `0.5248`
  - tradeable coverage `0.6166`
- strict `2024->2026` h12 best: `tradeable_fee2`
  - bacc `0.5192`
  - AUC `0.5293`
  - coverage `0.7568`
- `2025->2026` h6 best: `tradeable_dense`, bacc `0.5133`
- `2025->2026` h12 failed to improve, best bacc only `0.4911`.

Architecture decision:

- It is still not a standalone hard entry owner; promote only to candidate entry-context / direction-bias feature for Alpha6/Alpha7 PnL ablation.
- h12 should stay secondary because strict mode improved but the more recent `2025->2026` test did not.

## AI Role-Based Pass Reassessment - 2026-05-30

User correction:

- Do not require every AI model feature to be directional.
- Evaluate each model family by the role it is supposed to serve.

Artifact:

- `tmp/causal_regen_20260516/ai_role_pass_reassessment_20260530.json`

Current pass map:

| family | role | status | architecture use |
| --- | --- | --- | --- |
| TiDE | adverse-risk / exit / sizing | **PASS** | first AI feature to wire into risk/exit/size layer |
| PatchTSMixer binary | h6 tradeable direction bias | **HOLD_FAIL** | do not promote unless a later redesign fixes the weak `2025->2026` stability |
| Chronos | large-move / uncertainty / downside-risk | **PASS** | risk/uncertainty modifier only; never hard direction owner |
| TimesNet | anchor reversion / overheat / session | **WEAK_PASS_CANDIDATE** | small threshold/size/exit modifier only |
| DLinear | low-frequency flow/trend drift | **HOLD_FAIL** | do not promote yet |

Guardrails:

- Only TiDE is strong enough for first-pass active ablation.
- Chronos may be added as a risk/uncertainty context feature. TimesNet may be added only as a weak context feature. Neither may create new entries independently.
- DLinear should stay out unless a downstream PnL ablation proves value.
- Any promotion must report PnL, MDD, trades, trades/day, fee/slippage, leverage cap, and OOS period.

## Chronos Expanded Uncertainty Retest - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifacts:

- Runner: `scripts/test_chronos_uncertainty_large_move_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530/summary.json`

Architecture decision:

- Chronos remains rejected as a hard direction owner.
- Chronos is useful as a zero-shot uncertainty / large-move / downside-risk feature family after role change, input-series expansion, and live-safe EWM smoothing.
- Preferred downstream features:
  - `chronos_atr14_upside_band_ewm3`
  - `chronos_atr14_width_ewm6`
  - `chronos_atr14_width`
  - `chronos_atr14_large_move_score`
  - `chronos_realized_vol24_width`
  - `chronos_realized_vol24_large_move_score`
- Intended consumers:
  - Alpha6/Alpha7 risk template selector / risk resize layer;
  - entry threshold tightening under high projected uncertainty;
  - TP/SL widening or notional reduction under high downside-risk;
  - exit-pressure boost when uncertainty remains high after entry.

Latest 2026 OOS evidence:

- `atr14_pct` `upside_band_ewm3`: 2025 large/downside AUC `0.6050`/`0.6018`; 2026 large/downside AUC `0.6228`/`0.6307`.
- `atr14_pct` width: large-move AUC `0.6172`, downside AUC `0.6188`.
- `realized_vol_24` width: large-move AUC `0.6152`, downside AUC `0.6039`.
- `atr14_pct` large-move score: large-move AUC `0.6124`, downside AUC `0.6077`.

Guardrail:

- Do not let Chronos create entries independently.
- Do not select thresholds from 2026 distribution for live/ranking use. Use it as a continuous context feature and validate only through downstream PnL/MDD/trade-count ablation.

## PatchTST Tradeable Representation Test - 2026-05-30

User request:

- Test PatchTST as an alternative to PatchTSMixer.

Artifact:

- Runner: `scripts/train_ai_patchtst_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchtst_tradeable_20260530/summary.json`

Design:

- PatchTST trained from scratch; no local PatchTST pretrained checkpoint was available.
- Same `audit_compact_local_regime` patch-channel input family as PatchTSMixer binary.
- h6 `tradeable_fee2` binary short/long label.
- Compared end-to-end classifier, embedding+MLP, and embedding+CatBoost.

Result:

- Best 2026 OOS bacc: `0.5050` from PatchTST end-to-end.
- Best 2026 OOS AUC: `0.5080` from PatchTST embedding+CatBoost.
- This is materially worse than PatchTSMixer binary strict h6 bacc `0.5249`, AUC `0.5368`.

Architecture decision:

- Keep PatchTSMixer binary as the patch-family direction-context candidate.
- Do not promote PatchTST from-scratch models.
- Reopen PatchTST only if a real local pretrained checkpoint or a separate self-supervised pretraining stage is added.

## TimesNet Role Lock - 2026-05-30

User decision:

- TimesNet is fixed as a session / anchor-reversion auxiliary feature family.
- It is not a direction owner and should not be scored/promoted by h6 long/short bacc alone.

Artifact:

- `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/summary.json`
- Contract: `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/nf_timesnet/reworked_input_contract.json`

Current TimesNet role outputs:

- `ai_anchor_revert_prob`
- `ai_anchor_overheat`
- `ai_anchor_trend_escape_prob`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`

Latest completed metrics:

- `entry_quality_auc_anchor_revert=0.51996`
- `entry_quality_auc_trend_escape=0.48004`
- `cycle_delta_ret_corr=-0.02176`

Architecture policy:

- Use TimesNet only as context for entry threshold, notional/leverage, TP/exit, and mean-reversion veto decisions.
- Never use TimesNet output as a hard long/short action signal in the current Alpha AI feature lineage.
- Promotion gate is downstream PnL/MDD/trade-count ablation, not standalone direction accuracy.

## Default Prompt

```text
너는 /home/llewyn/crypto-scalping 프로젝트의 통합 Model/Data Architect다.
딥러닝, 지도학습, 비지도학습, 강화학습, 데이터/상태 계약을 통합해 실시간 코인 선물 트레이딩 봇의 모델 구조를 설계한다.

반드시 확인할 파일:
- docs/model_contracts/alpha3_csv_native_backtest_parity_20260516.md
- docs/experiments/alpha3_csv_native_parity_redteam_20260516.md
- docs/model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md
- docs/unified_pipeline_design.md
- docs/live_dsac_winner_20260503.md
- docs/feature_contract_manifest.json
- docs/model_contracts/registry.json
- features/schema.py
- features/registry.py
- features/engineering.py
- ensemble/train_rl_dsac_agent.py
- ensemble/train_rl_dsac_unified_controller.py
- ensemble/rl_continuous_common.py
- ensemble/rl_runtime_primitives.py
- ensemble/fully_learned_governor_policy.py
- pipeline/build_rl_dataset.py
- pipeline/augment_m7_dataset.py
- scripts/audit_live_duckdb_quality.py
- scripts/create_live_feature_views.py
- scripts/duckdb_persist_worker.py
- tail_risk_interceptor.py
- microstructure_scanner.py
- trading_bot.py
- scripts/backtest_trading_bot_native_2026.py

산출물:
1. 현재 모델 구조 요약
2. 다음 실험 설계
3. state/schema 변경이 필요한지 여부
4. 새 모델별 `docs/model_contracts/` 데이터 계약 필요 여부와 계약 초안
5. feature/state causal 여부, live 계산 가능 여부, missing/stale 처리 방식
6. 기대 효과와 실패 가능성
7. Red Team에 넘길 검증 항목

성과 주장은 항상 PnL, MDD, trades, trades_per_day, fee/slippage, leverage cap, OOS 기간을 함께 적어라.
모든 feature 제안은 causal 여부, live 계산 가능 여부, missing fallback, stale 처리 방식을 함께 적어라.
```

## Formula Teacher V1 Contract - 2026-05-31

User-approved direction: Teacher features are risk-aware meta-context, not a standalone direction owner.

Implementation:

- Runtime transform: `pipeline/teacher_meta_side_features.py::append_side_teacher_features`
- Rebuild utility: `scripts/rebuild_formula_teacher_features_20260531.py`

Required inputs are split-local/OOS model outputs only:

- `m7_q10`, `m7_q50`, `m7_q90`, `m7_qwidth`
- `m7_quality_pred`, `m7_hold_pred`, `m7_expected_ret`, `m7_tail_risk`
- `ai_adverse_risk`, `ai_reward_risk`

Optional context, used only when already present in the frame:

- `m7_long_adverse_prob`, `m7_short_adverse_prob`
- `regime3_transition_h6_risk_prob`, `regime3_churn_h6_risk_score`
- `chronos_atr14_width`, `chronos_realized_vol24_width`
- `liquidity_vacuum`

Outputs:

- `teacher_long_edge`
- `teacher_short_edge`
- `teacher_side_margin`
- `teacher_side_disagreement`
- `teacher_quantile_skew`
- `teacher_uncertainty`
- `teacher_tail_warning`

Contract notes:

- This is deterministic/no-fit Formula Teacher v1.
- It must not consume `label_*`, `target_*`, `tp_sl_action_score`, realized PnL, MFE/MAE, or future path columns.
- It no longer consumes `ai_dir_*`, `pred_patchtst/conf_patchtst`, or M7 probability direction heads as primary direction owners.
- Downstream recommended use: risk veto, sizing/notional reduction, TP/SL widening/tightening, exit-pressure boost, and label-quality filtering.
- Do not feed Teacher outputs back into AI/M7 training inputs unless a separate leakage audit and OOF contract is written.

## Omega1 Teacher Architecture Contract - 2026-05-31

Omega1 is the current teacher-stack version line.

Allowed Omega1 teacher input columns:

- Regime3 current sensitive wide24:
  - `regime3_current_sensitive_wide24_bull_prob`
  - `regime3_current_sensitive_wide24_bear_prob`
  - `regime3_current_sensitive_wide24_chop_prob`
  - `regime3_current_sensitive_wide24_confidence`
  - `regime3_current_sensitive_wide24_entropy`
  - `regime3_current_sensitive_wide24_margin`
- Regime3 h6 stability/risk sidecar:
  - `regime3_stability_h6_score`
  - `regime3_transition_h6_risk_prob`
  - `regime3_transition_h6_risk_pred`
  - `regime3_churn_h6_risk_score`
- Regime3 CryptoMamba h6 future-context sidecar:
  - `regime3_cmamba_h6_future_bull_prob`
  - `regime3_cmamba_h6_future_bear_prob`
  - `regime3_cmamba_h6_future_chop_prob`
  - `regime3_cmamba_h6_confidence`
  - `regime3_cmamba_h6_transition_prob`
  - `regime3_cmamba_h6_stability_score`
- M7 quantile-risk and ZigZag direction context:
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
- AI/TSFM risk outputs:
  - `ai_adverse_risk`
  - `ai_reward_risk`
  - `ai_vol_regime_pct`
  - `tide_vol_zscore`
  - `chronos_atr14_upside_band_ewm3`
  - `chronos_atr14_width_ewm6`
  - `chronos_atr14_width`
  - `chronos_atr14_large_move_score`
  - `chronos_realized_vol24_width`
  - `chronos_realized_vol24_large_move_score`
- Split-local current context:
  - `cvp_regime`
  - `regime_trending`

Hard exclusions for Omega1 teacher inputs:

- `clean_regime4_state24_sticky090_v2_*`
- `clean_regime4_2024_unsup_v1_*`
- `clean_regime_2024_unsup_v4_*`
- `regime4_pred_*`
- `teacher_*`
- `a5dir_*`
- `ai_dir_*`
- `patchtst_*`
- `dlinear_*`
- `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`
- `m7_quality_pred`, `m7_hold_pred`, `m7_zigzag_*_action`
- `m7_action`, `m7_size`, `m7_confidence`, `m7_composite_score`
- `m7_entry_*_price`, `m7_tp_price`, `m7_sl_price`
- `m7_gmm_*`, `m7_iso_*`, `m7_vae_*`, `m7_hdb_*`, `m7_gate_block`
- `regime3_pred_*`
- labels, targets, action scores, realized PnL, and future path statistics

Teacher output usage:

- `teacher_*` is forbidden for teacher-generation inputs to avoid circular
  features.
- After the teacher layer is generated, downstream Omega1 parent/risk/final
  policy models may consume:
  - `teacher_hgb_p_cash`
  - `teacher_hgb_p_long`
  - `teacher_hgb_p_short`
  - `teacher_hgb_confidence`
  - `teacher_hgb_side_edge`
  - `teacher_hgb_uncertainty`
  - `teacher_hgb_risk_veto_score`
- These teacher outputs must not feed back into AI, M7, Regime3, router, or a
  subsequent teacher-generation job unless a new out-of-fold/no-leak stacking
  contract is created.

Current implementation:

- `scripts/build_hgb_teacher_features_20260531.py`
- Emits `teacher_hgb_p_cash`, `teacher_hgb_p_long`, `teacher_hgb_p_short`,
  `teacher_hgb_confidence`, `teacher_hgb_side_edge`,
  `teacher_hgb_uncertainty`, and `teacher_hgb_risk_veto_score`.
- Historical strict pass-only rebuild before M7 ZigZag promotion:
  - `tmp/causal_regen_20260516/omega1_hgb_teacher_current_chronos_passonly_candidates_20260531_thr008`
  - `27` input columns.
  - Superseded by the 2026-05-31 red-team audit and later M7 ZigZag promotion: M7 target-family inputs are no longer allowed in active Omega1 teacher generation; explicit M7 quantile-risk and ZigZag probability/edge fields are allowed.
  - Chronos uses uncertainty / large-move context only.
  - `m7_tail_risk` is excluded from the strict pass-only HGB teacher because
    it is conditional weak context, not core pass.
- Regime3 current sensitive wide24 is allowed as current market-structure
  context. Regime3 future predictors (`regime3_pred_*`) remain excluded.
  - Current label-probe metrics: train bacc `0.7983`, train OVR AUC `0.9067`;
    2026 OOS bacc `0.3321`, OVR AUC `0.5198`.
- M7 clean recomputed risk columns:
  - Retired by user decision because signal ownership was ambiguous.
  - `scripts/recompute_m7_clean_risk_features_20260531.py` is not an active
    Omega1 builder.
  - `m7_clean_*` columns are not active teacher inputs.
  - Legacy M7 raw price/offset columns remain preserved for audit but blocked
    as direct active inputs.
- Current M7 ZigZag rebuild:
  - `tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_20260531`
  - `37` explicit second-stage input columns.
  - Active label source is `zigzag_action`; retired `tp_sl_action_score`
    threshold labels are not used.
  - M7 inputs are restricted to `m7_q10`, `m7_q90`, `m7_qwidth`, and
    M7 ZigZag probability/edge/confidence/trade-probability fields.
  - `m7_quality_pred`, `m7_hold_pred`, and `m7_zigzag_*_action` remain
    blocked.
  - 2026 OOS label-probe metrics: bacc `0.5637`, OVR AUC `0.7689`.
- Historical M7 ZigZag + clean risk rebuild:
  - `tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_cleanrisk_20260531`
  - `49` explicit second-stage input columns.
  - 2026 OOS label-probe metrics: bacc `0.5645`, OVR AUC `0.7647`.
  - Status: historical only; not active after `m7_clean_*` retirement.

Omega1 Mamba teacher candidate:

- Script: `scripts/train_omega1_mamba_teacher_20260531.py`
- Artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_current_chronos_seq72_20260531_e4`
- Architecture: native `mamba_ssm.Mamba`, sequence length `72`, d_model `96`, layers `2`.
- Inputs: current script uses explicit Omega1 second-stage features plus base current-context numeric features. The explicit block includes TiDE risk, Chronos uncertainty, Regime3 current/risk, M7 quantile-risk, and M7 ZigZag direction probability/edge context.
- Outputs:
  - `teacher_mamba_p_cash`
  - `teacher_mamba_p_long`
  - `teacher_mamba_p_short`
  - `teacher_mamba_confidence`
  - `teacher_mamba_side_edge`
  - `teacher_mamba_uncertainty`
  - `teacher_mamba_risk_veto_score`
- Label-probe metrics: train bacc `0.7900`, train OVR AUC `0.9024`; 2025 internal validation bacc `0.3550`, OVR AUC `0.5494`; 2026 OOS bacc `0.4359`, OVR AUC `0.6264`.
- `teacher_*` feedback remains forbidden during teacher generation.
- M7 ZigZag smoke artifact:
  - `tmp/causal_regen_20260516/omega1_mamba_teacher_m7zigzag_smoke_20260531`
  - Scope: `1` epoch GPU smoke for input contract validation, not final selection.
  - Inputs: `127` total = `37` explicit second-stage features + `90` base current-context numeric features.
  - Active label source: `zigzag_action`.
  - 2026 OOS smoke label-probe metrics: bacc `0.5712`, OVR AUC `0.7639`.
- M7 ZigZag + clean risk smoke artifact:
  - `tmp/causal_regen_20260516/omega1_mamba_teacher_m7zigzag_cleanrisk_smoke_20260531`
  - Scope: `1` epoch GPU smoke for input contract validation, not final selection.
  - Inputs: `139` total = `49` explicit second-stage features + `90` base current-context numeric features.
  - 2026 OOS smoke label-probe metrics: bacc `0.5830`, OVR AUC `0.7681`.

Omega1 Mamba teacher red-team audit - 2026-05-31:

- The high train score is not a valid selection metric. It is an in-sample re-prediction score; validation/OOS must drive decisions.
- No future-row timestamp join was found in the Mamba teacher script.
- P0 target alias leak was confirmed: `m7_quality_pred` equals `m7_target_quality`, and `m7_hold_pred` equals `m7_target_hold` exactly in both 2025 and 2026 frames.
- Active Mamba teacher inputs now exclude the M7 target-family block: `m7_quality_pred` and `m7_hold_pred`. `m7_q10`, `m7_q90`, and `m7_qwidth` are allowed as non-target quantile-risk context with target-alias fail-fast checks.
- Active script now fails fast when a selected feature exactly aliases a target-like numeric column.
- Internal validation preprocessing leak was removed: median/IQR normalization is fit on `train_idx` only.
- Clean P0 rerun artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_redteam_p0_clean_e4_20260531`.
- Clean e4 metrics: internal validation bacc `0.3580`, OVR AUC `0.5529`; 2026 OOS bacc `0.3883`, OVR AUC `0.6073`.
- HGB teacher was patched with the same P0 M7 target-family removal and target-alias guard. Historical clean artifact before M7 ZigZag promotion: `tmp/causal_regen_20260516/omega1_hgb_teacher_redteam_p0_clean_20260531_thr008`; feature count `22`; 2026 OOS bacc `0.3171`, OVR AUC `0.4977`.
- Architecture guidance: keep P1 upstream model/risk heads (`ai_*`, `tide_*`, `regime3_*h6*`) behind ablation gates before active/live promotion; do not treat them as clean base market inputs.

Omega1 canonical action label contract:

- Active teacher action-label training data is fixed to `3-class ZigZag action`.
- Semantics: `0=CASH`, `1=LONG`, `2=SHORT`.
- Canonical builder: `scripts/build_wave3_action_labels_20260531.py`.
- Canonical artifact directory: `tmp/causal_regen_20260516/zigzag_action_labels_20260531`.
- Canonical label files: `zigzag_action_labels_2024.csv`, `zigzag_action_labels_2025.csv`, `zigzag_action_labels_2026.csv`.
- Canonical hard label column: `zigzag_action`; `wave3_action` is removed from the active contract and must not be silently aliased.
- Canonical risk-adjusted soft label columns: `zigzag_soft_cash`, `zigzag_soft_long`, `zigzag_soft_short`.
- Canonical audit: `tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_label_audit.json`.
- Active label method: ZigZag confirmed-pivot segments.
- Active ZigZag parameters: `zigzag_reversal_pct=0.010`, `min_wave_bars=8`, `transition_buffer=2`, `atr_multiplier=1.0`, `mae_penalty=1.25`, `softmax_temperature=1.75`, `min_risk_floor=0.0010`.
- Soft labels are target labels only: they are derived from future segment path return/MAE/MFE and remain forbidden as model input features.
- Nearest-wave CASH expansion is disabled for ZigZag labels; transition-buffer rows stay CASH.
- Legacy Swing H/L wave3 and dense nearest-wave expansion are retired and must not be used in active Omega1 training.
- Omega1 teacher builders must consume that explicit artifact and fail fast if it is unavailable.
- Legacy `tp_sl_action_score -> threshold -> 3-class` and TP/SL action labels are retired for Omega1. They may remain only in historical reports and must not be used as active labels.
- Do not silently fall back from ZigZag labels to `tp_sl_action_score`.
- Any second-stage feature family trained on previous 2-action, binary long/short, or tradeable/no-trade action labels is stale for Omega1 active use. It must be retrained against `zigzag_action` and/or the explicit soft-label columns before promotion; old outputs must not be silently mapped into the new 3-class label.
- `tp_sl_action_score`, path-edge scores, realized PnL, MFE/MAE, future path statistics, and target-like wave construction columns remain forbidden as teacher input features.

Omega1 2-action-derived second-stage retrain notice:

- Retrain or keep research-only: old `ai_dir_*`, `ai_patch_*`, PatchTSMixer/PatchTST binary tradeable outputs, Alpha5/Alpha6 direction/action heads, and any router/action meta features whose label contract was binary or 2-action.
- Retrain or keep excluded: old M7 direction/action heads such as `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`, `m7_action`, and action/size/confidence/composite heads when their labels were 2-action or binary.
- Current ZigZag-action retrain artifacts to audit before promotion:
  - AI patch representation CatBoost head: `tmp/causal_regen_20260516/zigzag_ai_patchmix_catboost_20260531`.
  - M7 action HGB head: `tmp/causal_regen_20260516/zigzag_m7_action_hgb_20260531`.
- Exempt unless provenance says otherwise: Regime3 current context, Regime3 h6 risk sidecar, TiDE risk outputs, Chronos uncertainty outputs, and other risk/uncertainty/context features not trained on action labels.

ZigZag second-stage comparison retrain audit:

- Canonical audit: `docs/audits/zigzag_second_stage_retrain_20260531.md`.
- Full summary: `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/zigzag_second_stage_retrain_all_summary.json`.
- Completed run groups: `m7_zigzag_action_hgb`, `zigzag_second_stage_family_sweep`, and `ai_zigzag_patchmix_catboost`.
- This audit only covers families whose old outputs were trained from action/direction/tradeability labels. It does not automatically change the Omega1 active/pass feature list.
- 2026 PASS: `ai_zigzag_patchmix_catboost` (`bacc=0.5498`, `ovr_auc=0.7751`), `ai_all_legacy` (`bacc=0.5349`, `ovr_auc=0.7688`), `all_second_stage_nonp0` (`bacc=0.5329`, `ovr_auc=0.7519`), `ai_role_risk_context` (`bacc=0.5325`, `ovr_auc=0.7624`).
- 2026 PASS_WITH_OVERFIT_RISK: `m7_zigzag_action_hgb` (`bacc=0.5264`, `ovr_auc=0.7403`, train bacc about `0.93`).
- 2026 CONTEXT_ONLY: `regime3_all_context` (`bacc=0.5093`, `ovr_auc=0.7425`), `regime3_risk_context` (`bacc=0.4942`, `ovr_auc=0.7278`), `regime3_current_context` (`bacc=0.4925`, `ovr_auc=0.7321`).
- 2026 FAIL: `m7_direction_legacy`, `m7_all_nonp0`, `ai_direction_legacy`, `m7_unsup_risk_context`.
- Regime4 and `regime3_pred_*` were excluded from this comparison retrain.
- These PASS rows are promotion candidates only; before runtime use they require downstream PnL/MDD/trade-count ablation against the existing Omega1 feature contract.

ZigZag direct action-label model zoo audit:

- Canonical audit: `docs/audits/zigzag_action_model_zoo_20260531.md`.
- Script: `scripts/train_zigzag_action_model_zoo_20260531.py`.
- Artifact root: `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531`.
- The direct action/direction owners were retrained against `zigzag_action` with fail-fast guards and no Regime4/teacher/AI/M7/a5dir/downstream-label inputs.
- 2026 OOS ranking:
  - `alpha_catboost_action_master_like`: `bacc=0.565474`, `ovr_auc=0.755714`.
  - `trend_xgb_like_xgb`: `bacc=0.555528`, `ovr_auc=0.750837`.
  - `quantile_feature_like_lgbm`: `bacc=0.536903`, `ovr_auc=0.744360`.
  - `alpha_hgb_action_master_like`: `bacc=0.535653`, `ovr_auc=0.739463`.
  - `trend_xgb_like_lgbm`: `bacc=0.534785`, `ovr_auc=0.734313`.
  - `multitarget_lgbm_like`: `bacc=0.528298`, `ovr_auc=0.735595`.
  - `alpha_lgbm_action_master_like`: `bacc=0.515107`, `ovr_auc=0.738806`.
- Architectural recommendation: promote `alpha_catboost_action_master_like` first for downstream PnL/MDD/trade-density ablation; keep `trend_xgb_like_xgb` as the cleaner M7-style action baseline. Do not promote the overfit-heavy LGBM variants without additional regularization/CV.

M7 ZigZag direction integration:

- Audit: `docs/audits/m7_zigzag_direction_integration_20260531.md`.
- Script: `scripts/integrate_zigzag_direction_into_m7_20260531.py`.
- Integrated top direction candidates into M7-named feature files without overwriting original M7 files.
- Generated files:
  - `data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv`
  - `data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv`
- New CatBoost direction columns: `m7_zigzag_cat_fl`, `m7_zigzag_cat_up`, `m7_zigzag_cat_dn`, `m7_zigzag_cat_action`, `m7_zigzag_cat_confidence`, `m7_zigzag_cat_side_edge`, `m7_zigzag_cat_trade_prob`.
- New Trend-XGB-style direction columns: `m7_zigzag_xgb_fl`, `m7_zigzag_xgb_up`, `m7_zigzag_xgb_dn`, `m7_zigzag_xgb_action`, `m7_zigzag_xgb_confidence`, `m7_zigzag_xgb_side_edge`, `m7_zigzag_xgb_trade_prob`.
- Use guidance: the probability/edge/confidence fields are now explicitly allowed Omega1 teacher inputs under `docs/model_contracts/omega1_processed_feature_contract_20260531.md`; ordinal `m7_zigzag_*_action` fields remain blocked.

Omega1 processed feature contract:

- Canonical tracking document: `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Use this document as the source of truth for Omega1 processed / layered
  feature status.
- It tracks the full Omega1 processed-feature registry across architecture
  consumers, not just teacher inputs.
- Consumer layers are `teacher_generation`, `parent_policy`,
  `risk_sizing_exit`, `diagnostics_only`, and `research_only`.
- Omega1 layer structure is now explicit:
  - Layer 1: source/current features.
  - Layer 2: 2024-trained processed feature generators scored on 2025/2026.
    This includes AI/TSFM, Chronos, Regime3, M7, and standalone direction
    generators currently stored with legacy `dir3_*` prefixes.
  - Layer 3: teacher/meta/parent stack trained on 2025 Layer-2 OOS scores and
    tested on 2026 Layer-2 scores.
  - Layer 4: final policy/backtest/live execution.
- `dir3_*` is a legacy artifact prefix, not a layer name. Do not rename
  historical artifacts, but classify standalone 2024-trained `dir3_*`
  direction generators as Layer 2. `teacher_*` remains Layer 3.
- It also tracks active teacher inputs, M7 usage status, new M7 ZigZag
  direction candidates, downstream teacher outputs, context-only families,
  hold/fail families, and hard exclusions.
- CSV presence is not usage approval. Legacy M7 columns must remain blocked unless the contract explicitly promotes them.
- Any future processed-feature change must update the contract and change log before implementation or modeling.

Architectural use:

- Do not use Omega1 teacher as the direct action owner yet.
- Use it first as risk veto, size-down, exit-risk context, threshold adjustment,
  and diagnostics on top of the parent/final policy.
- Red Team must reject any Omega1 artifact whose teacher input manifest contains
  Regime4 columns or broad prefix-selected pass-failed AI/M7 columns.

Regime3 CryptoMamba h6 future-context sidecar - 2026-05-31:

- Role: future Regime3 / transition context sidecar only. It is not an action
  owner.
- Active artifact: `data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531`.
- Active report: `data/ensemble/reports/regime3_cryptomamba_pred_h6_nocurrent_20260531_report.json`.
- Architecture: CryptoMamba C-Block Merge, seq_len `60`, h6 target, 3 classes
  `bull/bear/chop`.
- Promoted input pack: `all_sanitized`, `128` features.
- 2026 OOS: bacc `0.672556`, accuracy `0.681084`, OVR AUC `0.843823`,
  transition AUC `0.695492`.
- Selection basis: best 2026 OOS bacc among docs-rolled-64, raw-priority-59,
  all-sanitized-96, all-sanitized-128, all-sanitized-161, followed by two
  additional seed checks for all-sanitized-128.
- Contract: current Regime3 sidecar is used only for target/evaluation.
  Regime4, `teacher_*`, `m7_*`, `a5dir_*`, downstream label/target/future/PnL,
  ZigZag, and wave columns are blocked from CryptoMamba inputs.
- Omega1 teacher input promotion: allow these exact numeric prediction-sidecar
  outputs as Regime3 future-context features:
  `regime3_cmamba_h6_future_bull_prob`,
  `regime3_cmamba_h6_future_bear_prob`,
  `regime3_cmamba_h6_future_chop_prob`,
  `regime3_cmamba_h6_confidence`,
  `regime3_cmamba_h6_transition_prob`,
  `regime3_cmamba_h6_stability_score`.
- Do not use `regime3_cmamba_h6_future_pred_id` or
  `regime3_cmamba_h6_future_pred_name` as teacher inputs. The generic
  future/target ban still applies outside this exact predicted-sidecar list.
- HGB contract check: adding these features increased the explicit HGB teacher
  input count to `43`; 2026 OOS label-probe bacc `0.5742`, OVR AUC `0.7748`.
- Confidence decoder audit:
  `tmp/causal_regen_20260516/regime3_cmamba_confidence_decoder_20260531`.
  Simple confidence gates and 2025-fitted lightweight calibrators did not beat
  raw 2026 argmax. Raw 2026 bacc `0.672556`, transition accuracy `0.243138`,
  change rate `0.171221`; best 2025-selected calibrated gate
  `transition_hgb_gate_change_thr0.275` had 2026 bacc `0.672480`, transition
  accuracy `0.242953`, change rate `0.171162`. Recommendation: do not promote
  decoded class/id features. Keep probability, confidence, transition-prob, and
  stability features as risk/context inputs; use confidence as reliability
  weighting, not as a replacement class transform.

Omega1 DIR3 direction feature generators - 2026-05-31:

- Retrieval artifact: `data/ensemble/supervised/omega1_dir3_retrieval_20260531`.
- Retrieval audit: `tmp/causal_regen_20260516/omega1_dir3_retrieval_20260531/dir3_retrieval_audit.json`.
- Selected retrieval config: `base_regime3_current_h6`, PCA32, K=128, uniform neighbors, 130 inputs, trained on 2024 and selected on 2025.
- Retrieval 2026 standalone label-probe: bacc `0.5151`, OVR AUC `0.7047`, proxy trades `13842/16832`, proxy WR `58.10%`.
- Combined HGB parent/meta probe on equal rows: Omega1 core bacc `0.5649`, proxy WR `62.52%`, proxy trades `14128`; core + retrieval bacc `0.5681`, proxy WR `62.35%`, proxy trades `14028`. Use retrieval as a parent/meta context candidate, not a direct action owner.
- Cycle/session artifact: `data/ensemble/supervised/omega1_dir3_cycle_20260531`.
- Cycle audit: `tmp/causal_regen_20260516/omega1_dir3_cycle_20260531/dir3_cycle_audit.json`.
- Cycle 2026 standalone label-probe: bacc `0.4226`, OVR AUC `0.6629`, proxy WR `55.95%`, but predicts no CASH. Combined probe did not improve Omega1 core (`0.5630` bacc vs `0.5649` core). Keep diagnostics-only.
- Contract: `dir3_*` is a historical artifact prefix. Standalone `dir3_*`
  direction generators trained on 2024 and scored on 2025/2026 are Layer 2
  processed OOS features, not Layer 3 teacher outputs. They still must not feed
  teacher generation unless a separate OOF/no-leak stacking contract is written.

Omega1 DIR3 remaining-generator audit - 2026-05-31:

- Script: `scripts/build_omega1_dir3_remaining_features_20260531.py`.
- Audit: `tmp/causal_regen_20260516/omega1_dir3_remaining_20260531/dir3_remaining_audit.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_remaining_20260531/remaining_meta_probe_summary.json`.
- Generated feature artifacts:
  - `data/ensemble/supervised/omega1_dir3_chartcnn_20260531`.
  - `data/ensemble/supervised/omega1_dir3_patch_20260531`.
  - `data/ensemble/supervised/omega1_dir3_duet_20260531`.
- Standalone 2026 label-probe:
  - `dir3_chartcnn`: bacc `0.4534`, OVR AUC `0.6329`, proxy trades `9061`, proxy WR `47.68%`.
  - `dir3_patch`: bacc `0.5718`, OVR AUC `0.7649`, proxy trades `13300`, proxy WR `61.85%`.
  - `dir3_duet`: bacc `0.5637`, OVR AUC `0.7581`, proxy trades `13432`, proxy WR `61.90%`.
- Combined HGB parent/meta probe on equal rows:
  - core-only: bacc `0.5663`, OVR AUC `0.7713`, proxy trades `14025`, proxy WR `62.73%`.
  - core + chartcnn: bacc `0.5639`, OVR AUC `0.7623`, proxy trades `13851`, proxy WR `61.76%`.
  - core + patch: bacc `0.5829`, OVR AUC `0.7809`, proxy trades `13873`, proxy WR `63.84%`.
  - core + duet: bacc `0.5694`, OVR AUC `0.7683`, proxy trades `14012`, proxy WR `62.13%`.
  - core + all remaining: bacc `0.5764`, OVR AUC `0.7680`, proxy trades `13696`, proxy WR `62.51%`.
- Architecture decision: promote `dir3_patch` as the first DIR3 parent/meta
  candidate for downstream PnL/MDD/trade-density ablation. Keep `dir3_duet` as
  a weaker research candidate. Keep `dir3_chartcnn` diagnostics-only. Do not
  blanket-combine all remaining generators because `core + patch` was stronger
  than `core + all remaining`.

Omega1 DIR3 financial-paper candidates - 2026-05-31:

- Script: `scripts/build_omega1_dir3_finpaper_features_20260531.py`.
- Audit: `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/dir3_finpaper_audit.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/finpaper_meta_probe_summary.json`.
- Research mapping:
  - FinTSB: used as standardized OOS harness guidance, not a standalone model.
  - Oxford financial benchmark: implemented VSN-LSTM and lightweight PatchTST
    sequence classifiers over current/past 72-row windows.
  - X-Trend: implemented 2024-only context-memory retrieval with cosine
    nearest-neighbor attention over sequence summaries.
- Inputs: 64 explicit current/past market, flow, funding, BTC-relative, squeeze,
  and execution context features. Blocked inputs: Regime4, `regime3_pred_*`,
  `teacher_*`, `a5dir_*`, target/label/future/PnL/action-score columns, and
  other `dir3_*` direction-generator inputs.
- 3-epoch standalone 2026 label-probe:
  - `dir3_vsnlstm`: bacc `0.5766`, OVR AUC `0.7608`, proxy trades `12416`, proxy WR `62.51%`.
  - `dir3_lpatchtst`: bacc `0.5062`, OVR AUC `0.7000`, proxy trades `13575`, proxy WR `59.11%`.
  - `dir3_xtrend`: bacc `0.5010`, OVR AUC `0.6863`, proxy trades `11340`, proxy WR `58.38%`.
- Combined HGB parent/meta probe on equal rows:
  - core-only: bacc `0.5663`, OVR AUC `0.7713`, proxy trades `14025`, proxy WR `62.73%`.
  - core + VSN-LSTM: bacc `0.5793`, OVR AUC `0.7794`, proxy trades `13757`, proxy WR `63.47%`.
  - core + lightweight PatchTST: bacc `0.5720`, OVR AUC `0.7762`, proxy trades `14060`, proxy WR `63.19%`.
  - core + X-Trend: bacc `0.5644`, OVR AUC `0.7711`, proxy trades `14048`, proxy WR `62.40%`.
  - core + all finpaper: bacc `0.5781`, OVR AUC `0.7798`, proxy trades `13818`, proxy WR `63.73%`.
- Architecture decision: retain `dir3_vsnlstm` as the paper-inspired
  parent/meta candidate. `core + all finpaper` has the best proxy WR but lower
  bacc than `core + VSN-LSTM`; test both against `core + dir3_patch` in actual
  PnL/MDD/trade-density backtests before promotion.

Omega1 DIR3 CryptoMamba direction sidecar - 2026-05-31:

- Script: `scripts/build_omega1_dir3_cryptomamba_direction_20260531.py`.
- Audit: `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/cryptomamba_meta_probe_summary.json`.
- Artifact: `data/ensemble/supervised/omega1_dir3_cryptomamba_20260531`.
- Architecture: Regime3 CryptoMamba C-Block Merge ported to direction:
  sequence length `60`, d_model `128`, d_state `32`, `4` C-blocks,
  `2` CMBlocks per C-block, Mamba d_conv `4`, expand `2`.
- Target: active `zigzag_action` (`0=CASH`, `1=LONG`, `2=SHORT`).
- Inputs: `128` current/past numeric features selected from market, flow,
  funding, BTC-relative, squeeze, execution, and rolling-stable transforms.
  Blocked inputs: Regime4, `regime3_pred_*`, `regime3_cmamba_*`,
  `teacher_*`, `a5dir_*`, target/label/future/PnL/action-score columns, and
  other `dir3_*` direction-generator inputs.
- Standalone 2026 label-probe: bacc `0.5671`, OVR AUC `0.7486`,
  proxy trades `11564`, proxy WR `62.67%`.
- Combined HGB parent/meta probe on equal rows:
  - core-only: bacc `0.5682`, OVR AUC `0.7714`, proxy trades `14065`, proxy WR `62.81%`.
  - core + CryptoMamba: bacc `0.5698`, OVR AUC `0.7740`, proxy trades `14055`, proxy WR `62.76%`.
- Architecture decision: retain as a parent/meta research candidate only.
  It adds small bacc/AUC lift but does not beat `dir3_patch` or `dir3_vsnlstm`
  under the current label-probe. Further work should tune label/feature pack
  or use it as risk/context rather than the primary direction owner.

Omega1 DIR3 Top2 full sweep - 2026-05-31:

- Script: `scripts/sweep_omega1_dir3_top2_full_20260531.py`.
- Summary: `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_sweep_summary.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_meta_probe_summary.json`.
- Full artifacts:
  - `data/ensemble/supervised/omega1_dir3_patch_full_20260531`.
  - `data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531`.
- Sweep scope:
  - `dir3_patch`: `3` seeds x `3` HGB hyperparameter variants.
  - `dir3_vsnlstm`: `3` seeds, max `12` epochs, early-stop patience `3`
    using 2024 internal validation.
- Best standalone 2026 label-probe:
  - `dir3_patch_full`: bacc `0.5692`, OVR AUC `0.7640`, proxy trades `13492`,
    proxy WR `61.76%` under the 2025-selected variant.
  - `dir3_vsnlstm_full`: bacc `0.5869`, OVR AUC `0.7689`, proxy trades
    `12114`, proxy WR `64.13%` under seed `20260533`.
- Full parent/meta probe:
  - core-only: bacc `0.5663`, OVR AUC `0.7713`, proxy trades `14025`,
    proxy WR `62.73%`.
  - core + patch_full: bacc `0.5783`, OVR AUC `0.7803`, proxy trades `13921`,
    proxy WR `63.65%`.
  - core + vsnlstm_full: bacc `0.5857`, OVR AUC `0.7844`, proxy trades
    `13916`, proxy WR `64.87%`.
  - core + patch_full + vsnlstm_full: bacc `0.5851`, OVR AUC `0.7863`,
    proxy trades `13799`, proxy WR `64.21%`.
- Architecture decision: `dir3_vsnlstm_full` is now the top direction-context
  candidate by 2026 label-probe and parent/meta probe. Keep `dir3_patch_full`
  as the strongest HGB/tabular baseline. Do not automatically combine patch and
  VSN-LSTM because the combination improves AUC but reduces bacc/WR versus
  VSN-LSTM alone.

Alpha7 Regime3 current-context MoE active update - 2026-06-01:

- Frame constraint: keep current-Regime3 MoE with separate `bull`, `bear`,
  and `chop` experts. Validation is used for selection; 2026 OOS is fixed
  evaluation only.
- Previous practical candidate:
  `alpha7_regime3_current_practical_moe_20260601`,
  `conf0.80_chop_expert_lowbaseline`.
  - Validation Cost3 `+110.67%`, MDD `-40.67%`, trades `172`, WR `13.95%`.
  - 2026 OOS Cost3 `+80.02%`, MDD `-27.81%`, trades `125`, WR `15.20%`.
- Tested overlays that did not become active:
  - Risk/churn/router-confidence sizing overlay:
    `scripts/eval_alpha7_regime3_current_moe_risk_sizing_overlay_20260601.py`.
    Best validation Cost3 `+193.13%`, but 2026 OOS Cost3 only `+50.13%`;
    treat as validation-overfit, not active.
  - Per-regime confidence thresholds:
    `scripts/eval_alpha7_regime3_current_moe_per_expert_conf_20260601.py`.
    Best validation Cost3 `+119.85%`, 2026 OOS Cost3 `+79.20%`; not active.
- New active candidate:
  `alpha7_regime3_current_moe_expert_source_mix_20260601`.
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_expert_source_mix_20260601.py`.
  - Report:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601/report.json`.
  - Selected by validation:
    `bull_practical__bear_risk__chop_practical__conf0.80`.
  - Interpretation: keep practical `bull` and `chop` experts; replace only
    the `bear` expert with the `base_plus_current_risk` expert. This suggests
    transition/churn risk context is useful mainly in bearish current-regime
    routes, while bull/chop overfit when upgraded.
  - Validation Cost3 `+141.12%`, MDD `-40.39%`, trades `168`, WR `14.29%`.
  - 2026 OOS Cost3 `+101.50%`, MDD `-27.81%`, trades `131`, WR `15.27%`.
  - 2026 OOS Cost1/2/3: `+121.43%`, `+104.29%`, `+101.50%`.
- Active decision: promote expert-source mix candidate over previous
  practical candidate because it improves both validation Cost3 and fixed
  2026 OOS Cost3 without changing the current-Regime3 MoE frame.

Alpha7 Regime3 current-context MoE follow-up tests - 2026-06-01:

- Active mix per-expert confidence grid:
  `scripts/eval_alpha7_regime3_current_moe_active_mix_per_conf_20260601.py`.
  - Best validation candidate `bull0.85_bear0.80_chop0.80`.
  - Validation Cost3 `+150.44%`, MDD `-39.16%`, trades `165`, WR `14.55%`.
  - 2026 OOS Cost3 `+100.58%`, MDD `-27.81%`, trades `131`, WR `15.27%`.
  - Result: not promoted because it underperforms the active mix OOS Cost3
    (`+101.50%`) despite stronger validation.
- Active mix retrieval overlay:
  `scripts/eval_alpha7_regime3_current_moe_active_mix_retrieval_overlay_20260601.py`.
  - Selected by validation: baseline/no overlay.
  - Best MDD-constrained resize variant produced OOS Cost3 around `+93.35%`
    with MDD `-16.21%`, trades `99`, WR `25.25%`, but validation Cost3 was
    only `+57.05%`.
  - Result: useful as a defensive diagnostic, not active.
- Current active remains
  `bull_practical__bear_risk__chop_practical__conf0.80`.

Alpha7 Regime3 current-context MoE expert-scale update - 2026-06-01:

- Scripts:
  - `scripts/eval_alpha7_regime3_current_moe_active_mix_expert_scale_20260601.py`.
  - `scripts/eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601.py`.
- Frame constraint remains unchanged: current-Regime3 MoE, separate
  bull/bear/chop experts. Only per-expert notional/position scaling is applied
  after the active expert-source mix decision.
- Coarse scale selected by validation:
  `bull0.85_bear1.15_chop1.10`.
  - Validation Cost3 `+243.20%`, MDD `-37.44%`, trades `167`, WR `14.97%`.
  - 2026 OOS Cost3 `+102.81%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
- Refined scale selected by validation:
  `bull0.85_bear1.15_chop1.25`.
  - Validation Cost1/2/3: `+350.75%`, `+361.91%`, `+270.24%`.
  - Validation Cost3 MDD `-37.74%`, trades `167`, WR `14.97%`.
  - 2026 OOS Cost1/2/3: `+117.46%`, `+113.87%`, `+103.72%`.
  - 2026 OOS Cost3 MDD `-27.81%`, trades `133`, WR `15.04%`.
- Active decision: promote refined expert-scale candidate over the unscaled
  expert-source mix (`+101.50%` OOS Cost3) because validation improves strongly
  and fixed 2026 OOS Cost3 also improves, while the MoE architecture and
  expert ownership remain unchanged.

Alpha7 Regime3 current-context MoE exit-shape diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_exit_shape_20260601.py`.
- Scope: active scaled MoE only; entry, routing, expert models, and notional
  scales are unchanged. The grid adjusts bear TP and chop TP/SL/hold.
- Validation-selected candidate:
  `btp1.10_ctp0.90_csl0.85_ch1.00`.
  - Validation Cost3 `+303.78%`, MDD `-37.74%`, trades `168`, WR `14.88%`.
  - 2026 OOS Cost3 `+73.98%`, MDD `-27.81%`, trades `138`, WR `13.77%`.
  - Result: not promoted. The validation-selected SL-tightening configuration
    overfits and damages fixed OOS.
- Diagnostic note: some non-selected grid rows have higher OOS Cost3, but they
  must not be promoted from this run because choosing them would use OOS for
  model selection. Keep the active refined expert-scale candidate.

Alpha7 Regime3 current-context MoE low-confidence fallback scale diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_lowconf_scale_20260601.py`.
- Scope: active scaled MoE only; expert routing, expert models, and expert
  scales are unchanged. Only `lowconf_baseline` notional/position and TP scale
  are tested.
- Validation-selected candidate:
  `lowconf0.70_tp0.95`.
  - Validation Cost3 `+343.59%`, MDD `-37.27%`, trades `157`, WR `17.20%`.
  - 2026 OOS Cost3 `+79.56%`, MDD `-22.47%`, trades `119`, WR `16.81%`.
  - Result: not promoted. It is a defensive/validation-overfit setting that
    reduces OOS MDD but gives up too much OOS Cost3 versus the active scaled
    candidate (`+103.72%`).
- Diagnostic note: OOS-top rows such as `lowconf1.10_tp1.05` are not promoted
  because they are not validation-selected and would use OOS for selection.
  Keep active refined expert-scale candidate.

Alpha7 Regime3 current-context MoE expert-confidence shrink diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_expert_conf_shrink_20260601.py`.
- Scope: active scaled MoE only. Low-confidence baseline rows are untouched;
  only selected bull/bear/chop expert rows with route confidence below the
  tested threshold are notional/position-shrunk.
- Validation-selected candidate:
  `bull_thr0.85_scale0.85`.
  - Triggered rows: validation `20`, 2026 OOS `2`.
  - Validation Cost3 `+315.78%`, MDD `-37.39%`, trades `164`, WR `15.85%`.
  - 2026 OOS Cost3 `+103.20%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
  - Result: not promoted because it is slightly below the active refined
    expert-scale candidate (`+103.72%` OOS Cost3), despite stronger validation.
- Current active remains:
  expert-source mix plus scale `bull=0.85`, `bear=1.15`, `chop=1.25`.

Alpha7 Regime3 current-context MoE soft expert fallback diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_soft_expert_fallback_20260601.py`.
- Scope: active scaled MoE only. Rows already routed to bull/bear/chop stay
  unchanged. Low-confidence baseline rows with route confidence in
  `[floor, 0.80)` may be delegated to the routed expert at reduced scale.
- Validation-selected candidate:
  `floor0.65_scale0.70`.
  - Triggered rows: validation `6987`, 2026 OOS `4591`.
  - Validation Cost3 `+163.44%`, MDD `-33.65%`, trades `188`, WR `16.49%`.
  - 2026 OOS Cost3 `+65.59%`, MDD `-30.14%`, trades `149`, WR `13.42%`.
  - Result: not promoted. Soft delegation of low-confidence rows introduces
    too much OOS noise versus the active scaled candidate (`+103.72%` OOS
    Cost3).
- Diagnostic note: `floor0.75_scale0.25` has high diagnostic OOS Cost3
  (`+118.11%`) but low validation score/Cost3 (`+118.51%`) and therefore must
  not be promoted from this run.

Alpha7 Regime3 current-context MoE component-source diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_component_source_mix_20260601.py`.
- Scope: current-Regime3 MoE is unchanged. Bull remains practical. Bear/chop
  primary and fallback components can independently use practical or
  current+risk source. Expert scales remain `bull=0.85`, `bear=1.15`,
  `chop=1.25`.
- Validation-selected candidate:
  `bearPrisk_Frisk__chopPrisk_Fpractical`.
  - Validation Cost3 `+292.96%`, MDD `-36.58%`, trades `171`, WR `14.62%`.
  - 2026 OOS Cost3 `+89.55%`, MDD `-27.81%`, trades `132`, WR `15.15%`.
  - Result: not promoted. Strong validation but weaker fixed OOS than active.
- Diagnostic note: `bearPrisk_Fpractical__chopPpractical_Fpractical` has the
  strongest OOS Cost3 (`+111.60%`) but is not validation-selected. It indicates
  bear `primary=risk`, `fallback=practical` may be a useful future hypothesis,
  but must be retested with an independent selection protocol before promotion.

Alpha7 Regime3 current-context MoE route-quality scale diagnostic - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_route_quality_scale_20260601.py`.
- Scope: active scaled MoE only. Expert ownership, entries, and exit shape are
  unchanged. Current-regime margin/entropy adjusts notional on expert rows.
- Validation-selected candidate:
  `mhi0.35_mlo0.15_e0.95_up1.10_dn0.80`.
  - Validation high-quality rows `1040`, low-quality rows `0`.
  - 2026 OOS high-quality rows `446`, low-quality rows `0`.
  - Validation Cost3 `+312.60%`, MDD `-40.61%`, trades `174`, WR `14.37%`.
  - 2026 OOS Cost3 `+55.87%`, MDD `-27.81%`, trades `142`, WR `12.68%`.
  - Result: not promoted. The route-quality high-margin scale is unstable OOS.
- Current active remains:
  expert-source mix plus scale `bull=0.85`, `bear=1.15`, `chop=1.25`.

Alpha7 Regime3 current-context MoE component-source two-stage validation - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_component_source_twostage_20260601.py`.
- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_component_source_twostage_20260601`.
- Purpose: retest the component-source hypothesis without selecting directly on
  full validation. Selection uses 2025-10/11, 2025-12 is confirmation, and
  2026 remains fixed OOS evaluation only.
- Selection-split winner:
  `bearPrisk_Fpractical__chopPrisk_Fpractical`.
  - Validation select Cost3 `+118.97%`, MDD `-36.85%`, trades `120`,
    WR `15.00%`.
  - Validation confirm Cost3 `+66.78%`, MDD `-28.27%`, trades `48`,
    WR `16.67%`.
  - Full validation Cost1/2/3 `+281.45% / +302.78% / +284.44%`,
    Cost3 MDD `-36.85%`, trades `167`, WR `14.97%`.
  - 2026 OOS Cost1/2/3 `+134.60% / +108.93% / +96.88%`,
    Cost3 MDD `-27.81%`, trades `130`, WR `15.38%`.
  - Policy counts:
    validation `lowconf_baseline=20552`, `chop_expert=2692`,
    `bear=1912`, `bull=1334`;
    OOS `lowconf_baseline=12985`, `chop_expert=1878`,
    `bear=1240`, `bull=788`.
- Decision: do not promote. The two-stage winner confirms that the component
  source mix is not a stable improvement over the active refined expert-scale
  candidate (`+103.72%` OOS Cost3). Keep active:
  expert-source mix plus scale `bull=0.85`, `bear=1.15`, `chop=1.25`.

Alpha7 Regime3 current-context MoE monthly-stability scale selection - 2026-06-01:

- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601.py`.
- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601`.
- Purpose: retest the active expert-scale grid with a validation-month
  stability score. Routing and bull/bear/chop expert models are unchanged.
  The first wide grid was stopped because it was unnecessarily heavy; the
  retained run evaluates a small active-neighborhood grid and computes 2026
  OOS only for the validation-stability winner.
- Stability-selected candidate:
  `bull0.85_bear1.15_chop1.25`, identical to current active.
  - Validation Cost1/2/3 `+350.75% / +361.91% / +270.24%`,
    Cost3 MDD `-37.74%`, trades `167`, WR `14.97%`.
  - Validation monthly Cost3 PnL:
    2025-10 `+72.14%`, 2025-11 `+15.01%`, 2025-12 `+78.85%`.
  - 2026 OOS Cost1/2/3 `+117.46% / +113.87% / +103.72%`,
    Cost3 MDD `-27.81%`, trades `133`, WR `15.04%`.
  - 2026 OOS monthly Cost3 PnL:
    2026-01 `+75.81%`, 2026-02 `+12.31%`.
- Decision: current active is confirmed by validation-month stability
  selection. No candidate change.

Alpha7 Regime3 current-context MoE expert attribution and bull suppression - 2026-06-01:

- Attribution script:
  `scripts/analyze_alpha7_regime3_current_moe_active_expert_attribution_20260601.py`.
- Attribution artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_expert_attribution_20260601`.
- Active candidate attribution, Cost3:
  - Validation full `+270.24%`, MDD `-37.74%`, trades `167`, WR `14.97%`.
  - Validation pieces:
    `bear +216.06% / MDD -26.38% / 74 trades / WR 17.57%`;
    `chop +44.57% / MDD -18.54% / 29 trades / WR 20.69%`;
    `bull -17.02% / MDD -35.98% / 44 trades / WR 11.36%`;
    `lowconf -9.24% / MDD -26.36% / 72 trades / WR 6.94%`.
  - 2026 OOS full `+103.72%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
  - 2026 OOS pieces:
    `bear +48.87% / MDD -19.34% / 58 trades / WR 12.07%`;
    `chop +53.39% / MDD -8.87% / 20 trades / WR 30.00%`;
    `bull -5.20% / MDD -24.77% / 20 trades / WR 10.00%`;
    `lowconf +16.93% / MDD -27.81% / 69 trades / WR 15.94%`.
- Bull suppression script:
  `scripts/eval_alpha7_regime3_current_moe_active_bull_suppression_20260601.py`.
- Bull suppression artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_bull_suppression_20260601`.
- Validation-selected candidate:
  `bullcash_bear1.15_chop1.25`.
  - Validation Cost1/2/3 `+532.94% / +517.03% / +275.67%`,
    Cost3 MDD `-39.37%`, trades `149`, WR `13.42%`.
  - 2026 OOS Cost1/2/3 `+107.91% / +61.69% / +32.86%`,
    Cost3 MDD `-29.75%`, trades `131`, WR `12.21%`.
- Decision: do not promote. Even though bull-only attribution is negative,
  removing the bull expert does not generalize; it degrades fixed 2026 OOS
  sharply. Keep active `bull0.85_bear1.15_chop1.25`.

Alpha7 Regime3 current-context MoE low-WR diagnostics and guard attempt - 2026-06-01:

- Ledger diagnostic script:
  `scripts/analyze_alpha7_regime3_current_moe_trade_ledger_wr_20260601.py`.
- Ledger diagnostic artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_trade_ledger_wr_20260601`.
- Note: this ledger uses a direct open-fill approximation with 3x fee/slip for
  WR/payoff/exits diagnosis. Official promotion metrics remain `_combo_metrics`.
- Approximate direct-ledger finding:
  - Most exits are stop-loss exits: validation `170/194`, OOS `128/147`.
  - Average win is much larger than average loss:
    validation avg win `+13.28%` vs avg loss `-2.20%`;
    OOS avg win `+9.80%` vs avg loss `-2.16%`.
  - This confirms a payoff-skew strategy rather than a high-hit-rate strategy.
- WR guard script:
  `scripts/eval_alpha7_regime3_current_moe_wr_guard_filter_20260601.py`.
- WR guard artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_wr_guard_filter_20260601`.
- Selection rule: maximize validation Cost3 WR only among candidates with
  Cost3 PnL at least 85% of active validation Cost3 PnL, positive Cost1/2,
  and at least 80 Cost3 trades. 2026 remains fixed evaluation only.
- Validation-selected WR guard:
  `q0.00_c0.68_lq0.06_bq0.20`.
  - Validation Cost1/2/3 `+424.73% / +393.50% / +381.85%`,
    Cost3 MDD `-35.57%`, trades `108`, WR `19.44%`.
  - 2026 OOS Cost1/2/3 `+55.14% / +53.59% / +53.66%`,
    Cost3 MDD `-22.16%`, trades `104`, WR `14.42%`.
- Decision: do not promote. Post-hoc quality/confidence veto improves
  validation WR but does not generalize; it lowers both OOS WR and OOS Cost3
  versus active (`+103.72%`, WR `15.04%`). Low WR cannot be safely fixed by
  a simple threshold filter; it needs retraining/objective-level changes if a
  higher hit-rate policy is required.

Alpha7 architecture report tests: Soft MoE, Two-stage gate, shared backbone, FT-Transformer - 2026-06-01:

- Source request: test the architecture report step by step, adapted to the
  current project flow rather than blindly replacing the active model.
- Baseline remains active `bull0.85_bear1.15_chop1.25`:
  2026 OOS Cost1/2/3 `+117.46% / +113.87% / +103.72%`,
  Cost3 MDD `-27.81%`, trades `133`, WR `15.04%`.
- Option 1 adapted Soft MoE:
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_soft_blend_20260601.py`.
  - Artifact:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_soft_blend_20260601`.
  - Design: decision-level soft blending of existing current-Regime3 experts
    using regime probabilities. No expert retraining and no OOS selection.
  - Validation-selected `p1.0_conf0.65_side0.15`:
    validation Cost3 `+8.83%`, MDD `-43.78%`, trades `263`, WR `12.55%`;
    2026 OOS Cost3 `+109.82%`, MDD `-34.56%`, trades `188`, WR `14.36%`.
  - Decision: do not promote. OOS Cost3 is slightly above active but
    validation is weak and MDD is worse; this is not a stable selector.
- Option 2 adapted Two-stage entry gate:
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_two_stage_entry_gate_20260601.py`.
  - Artifact:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_two_stage_entry_gate_20260601`.
  - Design: HGB binary entry gate trained on pre-validation active executed
    entries; existing bull/bear/chop planner remains unchanged.
  - Validation-selected `gate0.35`:
    validation Cost3 `+48.37%`, MDD `-28.37%`, trades `71`, WR `12.68%`;
    2026 OOS Cost3 `+15.79%`, MDD `-20.92%`, trades `50`, WR `20.00%`.
  - Decision: do not promote. WR improves but the gate removes too much of
    the payoff-skew edge that drives active PnL.
- Options 3/4 adapted contract test:
  - Script:
    `scripts/eval_alpha7_shared_backbone_ft_contract_test_20260601.py`.
  - Artifact:
    `tmp/causal_regen_20260516/alpha7_shared_backbone_ft_contract_test_20260601`.
  - Design: standalone PyTorch shared-MLP and FT-Transformer lifecycle parents
    using the existing Alpha7 feature contract, lifecycle labels, and
    `_combo_metrics`. This is a contract test, not an active replacement.
  - Shared MLP selected runtime:
    validation Cost3 `+23.94%`, MDD `-27.22%`, trades `36`, WR `25.00%`;
    2026 OOS Cost3 `-15.84%`, MDD `-32.69%`, trades `63`, WR `17.46%`.
  - FT-Transformer selected runtime:
    validation Cost3 `-14.04%`, MDD `-31.93%`, trades `59`, WR `22.03%`;
    2026 OOS Cost3 `+12.11%`, MDD `-32.98%`, trades `63`, WR `25.40%`.
  - Decision: do not promote. Current small PyTorch contract tests do not
    generalize. If revisited, use a larger retraining program with walk-forward
    labels; do not replace the active HGB MoE based on these runs.

Omega1 Dir3 TabM-CryptoMamba direction sidecar test - 2026-06-01:

- Source request: test a TabM/BatchEnsemble frontend for the CryptoMamba
  direction sidecar. Adaptation: replace only the CryptoMamba input projection
  with a TabM frontend; keep ZigZag action labels, 2024 train split, exact
  2025/2026 scoring, and forbidden-input contract unchanged.
- Script:
  `scripts/build_omega1_dir3_tabm_cryptomamba_direction_20260601.py`.
- Baseline reference:
  `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`.
  Baseline uses 128 features and has:
  - internal validation bacc/AUC/proxy WR `0.5661 / 0.7513 / 0.6394`;
  - 2025 bacc/AUC/proxy WR `0.5484 / 0.7457 / 0.6224`;
  - 2026 bacc/AUC/proxy WR `0.5671 / 0.7486 / 0.6267`.
- Tested candidates:
  - TabM ensemble 5, max-features 200:
    artifact `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_20260601`;
    selected 154 usable features;
    validation `0.5695 / 0.7546 / 0.6351`;
    2025 `0.5509 / 0.7391 / 0.6332`;
    2026 `0.5640 / 0.7458 / 0.6187`.
  - TabM ensemble 5, max-features 128:
    artifact `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_128_20260601`;
    validation `0.5608 / 0.7494 / 0.6256`;
    2025 `0.5415 / 0.7358 / 0.6092`;
    2026 `0.5626 / 0.7461 / 0.6080`.
  - TabM ensemble 3, max-features 200:
    artifact `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_e3_20260601`;
    selected 154 usable features;
    validation `0.5686 / 0.7446 / 0.6370`;
    2025 `0.5404 / 0.7285 / 0.6189`;
    2026 `0.5536 / 0.7287 / 0.6125`.
- Decision: fail for now. TabM slightly improves internal validation in the
  154-feature run but loses on 2026 OOS bacc/AUC/proxy WR versus the existing
  CryptoMamba direction sidecar. Do not add `dir3_tabm_cmamba_*` to Omega1
  active/live feature contracts.

Alpha7 full TabM tabular parent contract test - 2026-06-01:

- Source request: test TabM as a full architecture, not only as a CryptoMamba
  frontend. Adaptation: standalone lifecycle parent with BatchEnsemble hidden
  layers and shared multi-head outputs, using the existing Alpha7 feature
  contract, lifecycle labels, and `_combo_metrics`.
- Script:
  `scripts/eval_alpha7_full_tabm_parent_contract_test_20260601.py`.
- Artifact:
  `tmp/causal_regen_20260516/alpha7_full_tabm_parent_contract_test_20260601`.
- Initial standard loss collapsed into cash:
  raw validation action counts were approximately `cash 26412`, `long 4`,
  `short 74`; selected runtime had only one trade. This is not a valid
  replacement signal.
- Patched test objective: trade-biased loss with cash action weight `0.12`,
  active quality weight `1.25`, cash quality weight `0.15`, and higher action
  loss weight. This kept the same feature/data/backtest contract and only
  changed the TabM training objective to avoid cash collapse.
- Validation-selected runtime:
  `full_tabm_parent_c0.50_q0.010_s1.00_cap3.00_u0.070`.
  - Validation Cost1/2/3 `+25.41% / +25.75% / +16.53%`,
    Cost3 MDD `-31.85%`, trades `116`, WR `25.00%`.
  - 2026 OOS Cost1/2/3 `+46.14% / +32.10% / +26.66%`,
    Cost3 MDD `-43.63%`, trades `98`, WR `27.55%`.
- Decision: do not promote. Full TabM improves hit rate versus active, but
  active HGB MoE remains far stronger on OOS Cost3 (`+103.72%`) and drawdown.
  TabM is a useful high-WR research branch, not an active replacement.

Omega1 supervised-label authority update - 2026-06-01:

- Omega1 must not copy Alpha supervised components as-is. Alpha systems can
  provide architecture patterns only: MoE routing, parent/fallback shape,
  risk-template execution, backtest utilities, or feature-loading utilities.
- Active Omega1 supervised target authority is `zigzag_action` from
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531`.
- All Omega1 action, direction, entry, parent, expert, teacher, and meta-policy
  supervised heads must train from `zigzag_action` or a documented
  ZigZag-derived soft target.
- Forbidden active Omega1 supervised targets: `tp_sl_action_score`,
  `wave3_action`, Alpha6 fixed-barrier labels, Alpha lifecycle labels,
  `FullyLearnedGovernor` TP/SL path labels, and any direct realized future
  PnL/target columns.
- Risk/notional/TP/SL layers may be rule/template/search layers, or must be
  retrained from the same ZigZag authority if they are supervised. Do not load
  previous Alpha parent/governor models as active Omega1 supervised experts.
- New compatibility test path:
  `scripts/retrain_alpha7_active_max_feature_zigzag_moe_20260601.py`.
  It keeps the current-Regime3 MoE idea but replaces every supervised expert
  action head with a `zigzag_action` classifier and uses non-supervised
  risk-template search for execution parameters.

Omega1 ZigZag-only MoE risk redesign - 2026-06-01:

- Architecture kept fixed: max-feature current-Regime3 MoE with separate
  baseline/bull/bear/chop CatBoost action classifiers trained only on
  `zigzag_action`.
- Changed layer: execution/risk parameters only. No previous Alpha
  parent/governor supervised model was loaded.
- Script:
  `scripts/eval_alpha7_zigzag_moe_risk_param_sweep_20260601.py`.
- Selected candidate:
  `balanced_rr19_pc0.55_fc0.50_edge0.04_rc0.80_b0.75_r0.90_c0.90`.
- Selected runtime:
  - notional `0.45`, leverage `2.0`, TP `2.6%`, SL `1.4%`,
    max-hold `72` bars, cooldown `6` bars;
  - primary confidence `0.55`, fallback confidence `0.50`, active edge
    `0.04`, router min confidence `0.80`;
  - notional scales: bull `0.75`, bear `0.90`, chop `0.90`.
- Validation Cost3 `+41.34%`, MDD `-5.61%`, trades `339`, WR `51.03%`.
- 2026 OOS Cost3 `+5.58%`, MDD `-8.52%`, trades `211`, WR `44.55%`.
- Monthly validation Cost3: 2025-10 `+17.15%`, 2025-11 `+1.27%`,
  2025-12 `+12.11%`.
- Monthly 2026 OOS Cost3: 2026-01 `+3.62%`, 2026-02 `+4.99%`.
- Design read: ZigZag action labels require a tighter, lower-notional
  execution profile than the earlier `mid` default. The supervised label
  contract stayed clean; the performance recovery came from aligning
  execution geometry with the label semantics.

Omega1 teacher feature retirement - 2026-06-01:

- User decision: discard teacher features for Omega1 active/research modeling.
- `teacher_*` and `teacher_oof_*` are now historical/audit-only and must not be
  consumed by active Omega1 parent/risk/final-policy models.
- Architecture direction: remove the Layer 3 teacher stack from the immediate
  Omega1 path. Keep Layer 2 OOS direction generators as direct parent/risk
  context instead: M7 ZigZag direction fields, DIR3 VSN-LSTM/Patch/Duet/
  CryptoMamba/Retrieval families, and Regime3 bull/bear/chop context.
- Any future reintroduction of teacher must require a new OOF/no-leak stacking
  contract and explicit user approval.

Omega1 direction-only stacked head - 2026-06-02:

- `teacher_*` is retired; a focused Direction Head was tested as the Layer 3
  replacement candidate.
- Script:
  `scripts/train_omega1_direction_head_direction_only_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_direction_only_20260602`.
- Architecture: CatBoost 3-class stacked direction classifier trained on
  `zigzag_action`. It consumes only Layer 2 direction features and emits
  `omega1_dir_oof_*` for 2025 downstream training and `omega1_dir_*` for 2026
  OOS evaluation.
- Best variant: `core` with only `dir3_vsnlstm_h6_*` and `dir3_patch_h6_*`.
  - Feature count `12`.
  - 2025 OOF bacc/AUC/proxy WR `0.5708 / 0.7723 / 64.38%`.
  - 2026 OOS bacc/AUC/proxy WR/trades `0.5938 / 0.7835 / 64.43% / 13110`.
- Design read: adding M7 ZigZag, Regime3, retrieval, duet, and CryptoMamba
  features improves OOF slightly but degrades 2026 OOS. For now keep the
  direction stack compact: VSN-LSTM full + Patch full only.

Omega1 direction-only grouped PCA test - 2026-06-02:

- Script:
  `scripts/train_omega1_direction_head_direction_pca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_direction_pca_20260602`.
- Architecture: same CatBoost 3-class stacked direction classifier, but raw
  Layer 2 direction families are compressed with group-wise PCA before CatBoost.
  OOF PCA is fit only inside each expanding fold to avoid leakage.
- Best variant: `core_pca`, compressing `dir3_vsnlstm_h6_*` and
  `dir3_patch_h6_*` from 12 raw inputs to 6 PCA inputs.
  - 2025 OOF bacc/AUC/proxy WR `0.5688 / 0.7717 / 64.46%`.
  - 2026 OOS bacc/AUC/proxy WR/trades `0.5961 / 0.7836 / 64.78% / 13092`.
  - Relative to raw `core`: OOS bacc `+0.0024`, proxy WR `+0.35pp`, trades `-18`.
- Design read: grouped PCA is useful for the compact direction stack and should
  be preferred over raw `core` when a lower-dimensional Layer 3 direction
  feature is needed. Expanded PCA/all-direction PCA still carry redundant
  information and are not preferred.

Omega1 TSFM/Chronos Direction Head comparison - 2026-06-02:

- Script:
  `scripts/train_omega1_direction_head_tsfm_chronos_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602`.
- Architecture: CatBoost 3-class stacked Direction Head trained on
  `zigzag_action`, comparing TSFM/Chronos role features as standalone and as
  additive context for the compact VSN-LSTM/Patch direction core.
- Best variant: `core_plus_tsfm_chronos`, 55 features.
  - 2025 OOF bacc/AUC/proxy WR `0.5684 / 0.7739 / 64.67%`.
  - 2026 OOS bacc/AUC/proxy WR/trades `0.5974 / 0.7907 / 65.79% / 13334`.
  - Relative to `core_pca`: OOS bacc `+0.0013`, AUC `+0.0072`,
    proxy WR `+1.01pp`.
- Standalone TSFM/Chronos read:
  - `tsfm_role` has usable but weaker direction signal: OOS bacc `0.5710`,
    AUC `0.7630`, proxy WR `63.55%`.
  - Chronos standalone remains a poor hard direction owner: `chronos_all`
    OOS bacc `0.4158`, `chronos_h6` `0.3909`, `chronos_uncertainty` `0.3889`.
- Design decision: keep TSFM/Chronos as additive context or risk/uncertainty
  modifiers. Do not promote them as standalone direction owners.

Omega1 Direction Head contract finalization - 2026-06-02:

- User confirmed `core_plus_tsfm_chronos` as the fixed Omega1 Direction Head.
- Contract updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Final Direction Head architecture:
  CatBoost 3-class classifier trained on `zigzag_action`, consuming 55
  features: VSN-LSTM h6 direction, Patch h6 direction, exact TSFM role features,
  Chronos h6 features, and Chronos uncertainty features.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602/core_plus_tsfm_chronos`.
- Exported downstream columns:
  `omega1_tsfm_chronos_p_cash`, `omega1_tsfm_chronos_p_long`,
  `omega1_tsfm_chronos_p_short`, `omega1_tsfm_chronos_confidence`,
  `omega1_tsfm_chronos_side_edge`, `omega1_tsfm_chronos_trade_prob`,
  `omega1_tsfm_chronos_action`.
- 2026 OOS reference metrics:
  bacc `0.5974`, OVR AUC `0.7907`, proxy WR `65.79%`, proxy trades `13334`.
- Modeling guidance: downstream Omega1 parent/MoE/risk layers should consume
  this Direction Head output instead of rebuilding ad hoc direction stacks.
  TSFM/Chronos should not be promoted as standalone direction owners.

Omega1 Direction Head raw/context group add-on test - 2026-06-02:

- Script:
  `scripts/train_omega1_direction_head_raw_context_groups_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_raw_context_groups_20260602`.
- Test design: keep confirmed `core_plus_tsfm_chronos` fixed and add one
  primary/raw context group at a time: raw OHLCV, volume/flow,
  liquidity/execution spread proxy, funding, session, volatility, and all
  requested context combined.
- Best architecture variant: `add_volatility_context`, 79 features.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6040 / 0.7933 / 65.89% / 13093`.
  - Delta vs confirmed `core_plus_tsfm_chronos`:
    bacc `+0.0066`, AUC `+0.0026`, proxy WR `+0.10pp`, trades `-241`.
- Added volatility context features:
  `log_return`, `volatility_z`, `bb_width`, `bb_width_z`,
  `garman_klass_vol`, `realized_vol_ratio`, `rogers_satchell_vol`,
  `parkinson_vol`, `bb_width_pct_rank_288`, `atr_pct_rank_288`,
  `compression_score`, `compression_release_up`, `compression_release_down`,
  `garch_vol_z`, `jump_flag`, `jump_z`, `evt_tail_flag`, `evt_excess_z`,
  `squeeze_power`, `long_squeeze_risk`, `short_squeeze_risk`,
  `crowding_pressure`, `crowded_long_unwind_risk`,
  `crowded_short_squeeze_risk`.
- Negative result: adding all requested raw/context features at once degraded to
  OOS bacc `0.5781`, AUC `0.7647`, proxy WR `61.68%`.
- Contract note: literal `spread` and `bid_ask_spread` are not present in the
  current year-OOS feature frame. No silent alias was added; use explicit
  liquidity/execution proxy columns only until a real spread feature is
  generated under contract.
- Modeling guidance: promote volatility context as the next Direction Head
  add-on candidate. Keep raw OHLCV, volume/flow, and funding out of this head
  unless a later constrained/PCA test proves otherwise.

Omega1 Direction Head volatility PCA add-on test - 2026-06-02:

- Script:
  `scripts/train_omega1_direction_head_volatility_pca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602`.
- Architecture test:
  - Base Direction Head stays as the confirmed 55-feature
    `core_plus_tsfm_chronos` stack.
  - The 24-feature `add_volatility_context` group is compressed separately by
    PCA; no whole-state PCA is used.
  - PCA fit is split-local for OOF and 2025-only for final 2026 scoring.
- Best compact candidate: `volatility_pca06`.
  - Total inputs: 61 = 55 base + 6 volatility PCA components.
  - Volatility explained variance: `0.7563`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6052 / 0.7917 / 66.27% / 13245`.
  - Delta vs confirmed `core_plus_tsfm_chronos`:
    bacc `+0.0078`, AUC `+0.0010`, proxy WR `+0.47pp`, trades `-89`.
  - Delta vs raw `add_volatility_context`:
    bacc `+0.0012`, AUC `-0.0016`, proxy WR `+0.38pp`, trades `+152`.
- Modeling guidance:
  - Prefer `volatility_pca06` when optimizing compactness, balanced accuracy,
    and proxy WR.
  - Keep raw `add_volatility_context` as the AUC stability control because it
    still has slightly higher OOS AUC (`0.7933` vs PCA06 `0.7917`).
  - Do not promote broader raw/context groups; their 2026 OOS degradation is
    consistent with feature bloat/overfit.

Omega1 Direction Head core-group PCA on volatility_pca06 test - 2026-06-02:

- Script:
  `scripts/train_omega1_direction_head_core_group_pca_on_volpca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_core_group_pca_on_volpca_20260602`.
- Architecture test: keep `volatility_pca06` fixed, then replace internal
  `core_plus_tsfm_chronos` groups with split-local PCA components.
- Baseline:
  `volatility_pca06`, 61 inputs, 2026 OOS bacc/AUC/proxy WR/trades
  `0.6052 / 0.7917 / 66.27% / 13245`.
- Best bacc PCA replacement:
  `pca_tsfm06`, 43 inputs, 2026 OOS bacc/AUC/proxy WR/trades
  `0.6046 / 0.7900 / 66.18% / 13192`.
  - Delta vs `volatility_pca06`:
    bacc `-0.0006`, AUC `-0.0017`, proxy WR `-0.09pp`, trades `-53`.
- Other notable but not primary:
  - `pca_chronos_unc06`: AUC `+0.0005`, proxy WR `+0.05pp`, bacc `-0.0012`.
  - `pca_chronos_h603`: AUC `+0.0003`, bacc `-0.0006`, proxy WR `-0.29pp`.
  - `pca_direction_core06`: proxy WR `+0.09pp`, bacc `-0.0021`,
    AUC `-0.0013`.
- Modeling guidance: keep `vsnlstm`, `patch`, TSFM role, and Chronos groups raw
  in the Direction Head. PCA compression is useful for the volatility add-on,
  but compressing the core probability/edge groups removes semantic detail and
  does not beat the current compact candidate.

Omega1 Direction Head final contract update - 2026-06-02:

- User confirmed `core_plus_tsfm_chronos + volatility_pca06` as the fixed
  Omega1 Direction Head.
- Contract updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Final architecture:
  CatBoost 3-class classifier trained on `zigzag_action`, consuming 61
  features: raw VSN-LSTM h6, raw Patch h6, raw TSFM role, raw Chronos h6, raw
  Chronos uncertainty, and 6 PCA components from the explicit volatility
  context group.
- Active artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06`.
- Exported columns:
  `omega1_dir_volpca_p_cash`, `omega1_dir_volpca_p_long`,
  `omega1_dir_volpca_p_short`, `omega1_dir_volpca_confidence`,
  `omega1_dir_volpca_side_edge`, `omega1_dir_volpca_trade_prob`,
  `omega1_dir_volpca_action`.
- 2026 OOS reference:
  bacc `0.6052`, OVR AUC `0.7917`, proxy WR `66.27%`, proxy trades `13245`.
- Modeling guidance: downstream Omega1 parent/MoE/risk layers should consume
  `omega1_dir_volpca_*` as the Direction Head output. The previous
  `omega1_tsfm_chronos_*` output remains historical unless explicitly used for
  a comparison run.

Omega1 Regime3 expert-internal Direction Head test - 2026-06-02:

- Tested the user-requested alternative where each Regime3 expert owns its own
  Direction Head trained on the same 61-feature
  `core_plus_tsfm_chronos + volatility_pca06` contract.
- Hard expert partition:
  `scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5611 / 0.7480 / 60.90% / 13058`.
  - This is substantially worse than the fixed global Direction Head.
- Soft expert weighting:
  `scripts/train_omega1_regime3_soft_expert_direction_head_volpca_20260602.py`.
  - Best variant: `soft_floor_0p20`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6017 / 0.7920 / 66.11% / 13413`.
  - Delta vs global `volatility_pca06`:
    bacc `-0.0035`, AUC `+0.0003`, proxy WR `-0.16pp`, trades `+168`.
- Modeling decision:
  keep the global `omega1_dir_volpca_*` Direction Head active. The expert
  internal version is not promoted because it weakens balanced accuracy and
  entry WR. If reopened, use soft regime-probability weighting, not hard row
  partitioning.

Omega1 Regime3 expert-internal Direction + Quality Head test - 2026-06-02:

- Implemented the requested architectural variant where the Regime3 Current
  Router is upstream of direction ownership:
  feature processing -> Regime3 current route -> bull/bear/chop expert ->
  expert Direction Head -> expert Quality Head.
- Script:
  `scripts/train_omega1_regime3_routed_expert_direction_quality_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602`.
- Architecture:
  - Router columns:
    `regime3_current_sensitive_wide24_bull_prob`,
    `regime3_current_sensitive_wide24_bear_prob`,
    `regime3_current_sensitive_wide24_chop_prob`.
  - Direction Heads: separate CatBoost 3-class heads per expert, target
    `zigzag_action`.
  - Quality Heads: separate CatBoost 3-class second-opinion heads per expert,
    also target `zigzag_action`.
  - Quality score is the Quality Head probability assigned to the Direction
    Head candidate action.
  - No SL/TP compatibility target is used.
  - No global Direction Head output is fed into the experts.
  - Quality training uses 2025 expanding OOF Direction features to avoid
    in-sample Direction leakage.
- Best variant:
  `soft_floor_0p00`, threshold `0.45`.
  - Direction-only 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5983 / 0.7910 / 65.97% / 13463`.
  - Quality-filtered 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5832 / 0.7220 / 66.44% / 12276`.
  - Compared with active global `volatility_pca06`
    (`0.6052 / 0.7917 / 66.27% / 13245`), the Quality layer gains only
    `+0.17pp` proxy WR while losing `-0.0220` bacc and `-0.0697` AUC.
- Modeling decision:
  reject for active promotion. The idea is architecturally clean but too much
  class evidence is lost after expert routing plus Quality filtering. Keep the
  global `omega1_dir_volpca_*` Direction Head active until an expert-local
  variant improves bacc or preserves AUC without collapsing trades.

Omega1 Regime3 expert-internal DQ risk replay - 2026-06-02:

- Script:
  `scripts/eval_omega1_regime3_expertdq_risk_replay_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_expertdq_risk_replay_20260602`.
- Purpose: replay `omega1_regime3_expertdq_final_action` through the current
  ZigZag-compatible risk/execution template instead of stopping at proxy
  classification metrics.
- Fixed execution parameters:
  `balanced_rr19`, notional `0.45`, leverage `2.0`, TP `2.6%`, SL `1.4%`,
  max-hold `72`, cooldown `6`, expert scales bull `0.75`, bear `0.90`,
  chop `0.90`.
- Common-window active OOS Cost3:
  `+4.51%` PnL, `-8.69%` MDD, `211` trades, WR `46.92%`.
- Best expert-DQ OOS Cost3:
  `soft_floor_0p10`, `+8.29%` PnL, `-7.86%` MDD, `211` trades,
  WR `54.03%`.
- Validation failure:
  the same `soft_floor_0p10` has validation Cost3 `-2.19%` with MDD
  `-18.46%`, versus active validation Cost3 `+41.34%` and MDD `-5.61%`.
- Modeling decision:
  do not promote. The variant is OOS-attractive but validation-incoherent.
  If this branch is reopened, selection must be validation/monthly-stability
  driven, not OOS PnL driven.
