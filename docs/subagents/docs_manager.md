# Docs Manager Subagent

## Mission

The Docs Manager maintains the project documentation folder so a developer can read the docs and safely modify the trading bot, model stack, or module interfaces without rediscovering hidden contracts from source code.

This agent owns concise operational documentation, not long experiment reports. It must keep current active specs aligned with code and artifacts.

## Primary Paths

- `docs/active_live/`
- `docs/subagents/README.md`
- `docs/subagents/agent_registry.json`
- active model notes under `docs/alpha*_*.md` when referenced by the live path
- `trading_bot.py` only for reading contracts unless specifically assigned a code change
- model artifact manifests/summaries under `data/ensemble/supervised/*`

## Required Documents

The active documentation set is:

- `docs/active_live/README.md`
- `docs/active_live/alpha7_live_stack.md`
- `docs/active_live/trading_bot_runtime.md`
- `docs/active_live/module_interfaces.md`
- `docs/active_live/change_log.md`

## Update Triggers

Update `docs/active_live/` in the same work item when any of these change:

- active model ID, model version, runtime config path, or artifact path,
- active primary/fallback/deep-alpha routing order,
- feature contract, required prefix, forbidden prefix, or fail-fast behavior,
- `trading_bot.py` process lock, ledger, dashboard, or state file behavior,
- module function/class signatures used by active live path,
- exchange execution, Binance account sync, or dry-run/testnet/mainnet behavior,
- DuckDB storage path or writer ownership.

## Rules

- Do not document alias, compatibility shim, or silent fallback as an acceptable active behavior.
- If a contract mismatch should fail, write the expected error condition explicitly.
- Separate historical experiment results from active live behavior.
- When legacy regime feature rows appear in audit/model documents, label them as historical/reference-only and not active inputs.
- Confirmed-bug regime feature prefixes are not usable active features:
  - `clean_regime_2024_unsup_v4_*`
  - `clean_regime4_2024_unsup_v1_*`
- Active/live regime docs must name `clean_regime4_state24_sticky090_v2_*` and `regime4_pred_*` as the allowed regime surfaces.
- New action-classifier/regime redesign docs must follow `docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md`: bull/bear/chop are direction/structure classes; whipsaw is risk/veto/sizing context, not an action-regime class.
- If a code/model change creates Regime3 columns, document exact column names, horizon, derivation, provenance, and fail-fast behavior. Do not document silent Regime4-to-Regime3 aliasing as active behavior.
- Prefer exact file paths, model IDs, function names, env vars, and JSON keys.
- Keep docs short enough to be read during implementation.
- If code and docs disagree, treat that as a bug and report it before editing unrelated logic.
- Omega risk docs must state that TP/SL heads output price moves, not account-PnL thresholds. Document `tp_price_move`, `sl_price_move`, `margin_fraction`, `leverage`, and `notional` with explicit meanings, then derive runtime fields with `notional = margin_fraction * leverage`, `take_profit = tp_price_move * notional`, and `stop_loss = sl_price_move * notional`.
- If leverage is fixed, document the fixed value and derive notional from margin. Do not document any path that multiplies leverage again after notional is derived.
- Do not describe `long_take_profit`/`short_take_profit` or `long_stop_loss`/`short_stop_loss` account-threshold heads as a new active contract. If they appear, mark them historical-only or a blocker until regenerated.
- Omega/Omega4.x upgrade, baseline, and live-candidate docs must reference `docs/model_contracts/omega_artifact_integrity_policy_20260630.md` and the required audit `scripts/audit_omega_artifact_integrity_20260630.py`.
- Document exact-threshold parent prediction artifacts as required promotion evidence: `train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, and `oos_predictions_qXXX.csv`. Also document `risk_model.precomputed_prediction_dir` and `risk_model.precomputed_prediction_tag` for any risk sidecar that consumes parent outputs.
- Do not document saved trade ledgers or candidate-event replays as a replacement for parent prediction artifacts. They are diagnostic-only unless a separate historical reproduction path is explicitly opened.

## Feature Audit Memory - 2026-05-28

- Detailed per-feature audit: `docs/audits/features_folder_per_feature_audit_20260528.md`
- Prior summary audit: `docs/audits/features_folder_correlation_tendency_report_20260528.md`
- Directional alpha extension audit: `docs/audits/directional_alpha_feature_audit_20260528.md`
- Full direction-candidate universe audit: `docs/audits/directional_feature_universe_audit_20260528.md`
- Per-feature verdict CSV: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/per_feature_verdict.csv`
- Directional alpha feature CSV: `tmp/causal_regen_20260516/directional_alpha_feature_audit_20260528/directional_alpha_feature_scores.csv`
- Family verdict counts: `tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/family_verdict_counts.csv`
- Source inventory: `tmp/causal_regen_20260516/features_folder_code_inventory_20260528/`
- When active feature contracts change, check this audit first and document any use of:
  - `BUG_RISK_REGENERATE`
  - `DROP_RAW_LEVEL`
  - `MONITOR_OR_VETO_ONLY`
  - `DEDUP_DROP`
- Active/live docs must not promote broad raw feature expansion. Feature use should be layer-specific:
  - entry context,
  - risk sizing / TP-SL / exit,
  - execution context,
  - regime/meta overlay.
- Direct active input exclusions recorded by the audit:
  - confirmed-bug regime prefixes `clean_regime_2024_unsup_v4_*` and `clean_regime4_2024_unsup_v1_*`,
  - `garch_vol_z` until regenerated or replaced,
  - raw OHLC/`close_btc`,
  - raw M7 price outputs `m7_entry_long_price`, `m7_entry_short_price`, `m7_tp_price`, `m7_sl_price`.
- The 2026-05-28 directional alpha block adds causal CVD/compression/VWAP/funding-OI/wick-sweep plus BTC lead-lag features. Existing M7/AI/regime artifacts do not automatically consume those new inputs; if active docs claim they do, require 2024-only artifact retraining and 2025/2026 rescoring evidence.
- Source-required direction features are tracked in `docs/audits/source_required_direction_features_20260528.md`. Orderbook, real tick CVD, liquidation cluster, spot/perp basis, side-specific OI, and on-chain flow features must not be zero-filled into active inputs; persist historical/live sources first and fail fast on missing exact columns.
- Direction feature universe audit scored 220 candidates. Preserve prior `DEDUP_DROP`, `DROP_RAW_LEVEL`, and `MONITOR_OR_VETO_ONLY` verdicts when writing active/live feature contracts; high IC alone is not enough to promote a feature.
- AI direction retrain audit: `docs/audits/ai_direction_feature_retrain_20260528.md`
  - Current `ai_dir_*` is pseudo-probability output from PatchTST scalar edge, not a calibrated classifier.
  - Valid experimental outputs use `ai_dir_v2_*` under `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v2_noleak` and `..._v3_strict_noleak`.
  - `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v1` is explicitly invalid due label-score leakage.
  - Superseded active decision 2026-05-30: do not document `ai_dir_v2_*`, `pred_patchtst`, `conf_patchtst`, or `patchtst_median` as active/live inputs. Keep them historical/research-only unless a new direction-model contract is created.
- Funding source audit: `docs/audits/last_funding_rate_source_audit_20260528.md`
  - Historical bug: old year splits front-filled future funding, and 2025/2026 used ETHFIUSDT funding.
  - Active split CSVs and direct RL CSV funding columns were regenerated/patched to ETHUSDT-only backward-asof funding and validated at `100%` previous-ETHUSDT match.
  - Do not treat old M7/teacher/regime/policy artifacts as clean merely because direct CSV funding columns were patched; retrain or rescore artifacts that consumed contaminated funding inputs.
- Funding-derived artifact remediation: `docs/audits/funding_clean_retrain_rescore_20260529.md`
  - M7 active artifacts were retrained and 2025/2026 M7 CSVs rescored from clean funding splits.
  - `regime4_pred_tft_h12_nomdjd_all74_20260517` was regenerated while preserving the h12/all74/`pred_mdjd`+`conf_mdjd` exclusion contract.
  - Alpha5 `a5dir` / router was rebuilt under `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48`.
  - Clean router score CSV: `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/08_alpha5_direction_router_rl_2024_to_2025/rl_training_2025_direction_router.csv`.
  - Older DSAC, Alpha6, Alpha7, and cached unified outputs remain suspect unless their manifest points to this clean run or they are explicitly retrained/rescored.
- Funding feature red-team follow-up: `docs/audits/funding_feature_redteam_20260529.md`
  - Funding-derived columns without `funding` in the name, such as `squeeze_power`, squeeze/crowding derivatives, and artifacts trained or scored from those inputs, require the same clean funding provenance.
  - Current `alpha7_submodel_01965_decontam_v2_tp_20260528` active default is deprecated/blocked until rebuilt or replaced with a clean funding manifest.
  - Artifact directories with `DEPRECATED_DO_NOT_USE.json` or manifest status `deprecated_do_not_use_active_or_candidate` must be documented as blocked for active runtime, candidate baseline, parent/fallback block, sidecar source, Alpha8 baseline, and promotion evidence.
  - Current blocked Alpha7 examples are `data/ensemble/supervised/alpha7_1_01965_live_20260527` and `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528`.
- Mandatory funding-clean documentation rule:
  - Any active/live or promotion candidate that uses `last_funding_rate`, `funding_*`, `mta_funding`, `ou_funding_z`, squeeze/crowding features, or artifacts trained/scored from those inputs must document clean funding provenance.
  - Required proof is either an artifact/input path under the clean funding remediation run, a manifest that names the clean run, or a direct comparison to clean split `last_funding_rate` with `max_abs_diff == 0.0`.
  - Do not describe older Alpha6/Alpha7/Alpha8 candidate CSVs or downstream policy artifacts as clean unless this proof exists.
  - Known stale-risk example: `tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/trade_candidates_20*_alpha6_current_tail111_exact.csv` mismatches clean funding split values and must be documented as research-only/stale until regenerated.

## M7 Red-Team Contract Memory - 2026-05-30

- Policy doc: `docs/audits/m7_redteam_contract_20260530.md`
- Current status: active M7 generation/required-column contracts no longer include unsupervised GMM / Isolation Forest / VAE model keys or derived columns. Funding-clean provenance alone is not promotion evidence for historical artifacts that still contain removed M7 columns.
- Removed active model/meta keys: `gmm_volatility`, `isolation_forest`, `vae_anomaly`.
- Allowed downstream M7 columns after direction-context removal: `m7_q10`, `m7_q90`, `m7_qwidth`, `m7_quality_pred`, `m7_hold_pred`, `m7_tradeability_score`, `m7_long_mae_q90`, `m7_short_mae_q90`, `m7_long_adverse_prob`, `m7_short_adverse_prob`.
- M7 direction context is binary-only (`DOWN` / `UP`) where it is explicitly used. Do not reintroduce a separate no-trade direction class or any compatibility mapping for removed flat-axis fields.
- Active M7 `lightgbm_ensemble` artifact: `data/ensemble/supervised/lightgbm_ensemble.json`; trainer: `ensemble/supervised/train_lightgbm_ensemble.py`. Its feature contract blocks `m7_*`, legacy clean-regime prefixes, and old `regime_bull/regime_bear/regime_chop/regime_whipsaw/regime_normal` one-hot inputs.
- Active M7 base artifacts retrained and overwritten only on 2026 OOS improvement: `trend_xgb`, `multi_target_lgbm`, `quantile_forest`. `entry_price_model` retrain candidate was rejected; runtime offset propagation bug was patched in `ensemble/seven_model_ensemble.py`.
- Conditional weak meta/context only: `m7_tail_risk`, `m7_tp_offset`, `m7_tp_price`, `m7_entry_long_price`, `m7_entry_short_price`, `m7_target_hold`, `m7_target_quality`. If entry prices are used, recompute entry offsets; do not reuse scored offset columns.
- Any reported M7 backtest using removed `m7_gmm_*`, `m7_iso_*`, `m7_vae_*`, `m7_gate_block`, `m7_size`, or `m7_hdb_*` columns is diagnostic-only until retrained/rescored under the current active contract.
- M7 artifact metadata must include training data path/hash, clean funding run ID, scaler contract, threshold contract, and scored CSV hash. Missing metadata is a blocker.
- For historical artifacts that used removed columns, require retraining/rescoring and retrain/re-evaluate DSAC, Alpha7, and Alpha8 candidates that consume those M7 artifacts. Do not add alias, fallback prefix, fabricated compatibility columns, or silent corrections.

## Omega1.2 Exit-Feature Lifecycle Baseline Memory - 2026-06-05

- New research baseline: `omega1_2_exit_feature_lifecycle_baseline_20260604`.
- Contract: `docs/model_contracts/omega1_2_exit_feature_lifecycle_baseline_20260604_contract.md`.
- Manifest: `data/ensemble/supervised/omega1_2_exit_feature_lifecycle_baseline_20260604/baseline_manifest.json`.
- Source artifact: `tmp/causal_regen_20260516/omega1_2_mamba_sac_lifecycle_controller_20260604_mid600_e800_noresize_noreverse_edge002_q075_seed260604`.
- Baseline rule: the 3-head TabM Exit Head is feature-only. It may provide `threehead_exit_p_hold_feature_only`, `threehead_exit_p_exit_feature_only`, and `threehead_exit_edge_feature_only` to the lifecycle controller, but it must not directly trigger `exit_prob >= threshold -> immediate exit`.
- Selected lifecycle controller: discrete Mamba offline SAC-style controller, `quality_threshold=0.75`, `seq_len=64`, `max_train_entries=600`, `steps=800`, `min_action_edge=0.002`, `disable_resize=true`, `disable_reverse=true`.
- OOS Cost3 baseline: PnL `+16.0740%`, MDD `-5.3960%`, WR `65.625%`, trades `32`.
- This is a research baseline, not a live promotion. Do not wire to `trading_bot.py` without runtime-native parity, current live feature-contract validation, and comparison against the prior Omega1.2 final TP/SL baseline.

## Omega1.2 Current Final TP/SL Baseline Memory - 2026-06-06

- Current Omega1.2 research baseline: `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`.
- Contract: `docs/model_contracts/omega1_2_true_3head_tabm_final_tp_sl_current_20260606_contract.md`.
- Manifest: `data/ensemble/supervised/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080/baseline_manifest.json`.
- Source artifact: `tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`.
- Template: TP `0.026`, SL `0.014`, notional `0.405`, leverage `2.0`, max-hold `0`, cooldown `0`, Cost3 multiplier `3.0`.
- Validation replay: PnL `+42.8226%`, MDD `-5.4716%`, WR `63.6364%`, trades `33`.
- OOS replay: PnL `+32.1456%`, MDD `-4.1352%`, WR `72.2222%`, trades `18`.
- `base_nogate_topk2` remains a post-lifecycle bucket-adapter research candidate, not the current Omega1.2 baseline for new growth work.
- Future growth candidates must reproduce this baseline first, compare validation and OOS, and avoid legacy aliases/fallback prefixes/compatibility feature shims.
- Initial static growth scan: compensated TP/SL scaling is viable; raw notional-only scaling is rejected. Balanced candidate scale/cap `1.35/0.55` gives validation `+61.14%` / MDD `-7.32%` and OOS `+45.31%` / MDD `-5.54%`. More aggressive scale/cap `2.00/0.90` gives validation `+100.54%` / MDD `-10.68%` and OOS `+72.76%` / MDD `-8.11%`.
- This is now the previous Omega1.2 research baseline. The active live baseline is `omega1_2_1_aggressive_compensated_scale200_cap090`.

## Omega1.2.1 Growth Branch Memory - 2026-06-06

- Growth branch: `omega1_2_1_current_baseline_growth_20260606`.
- Contract: `docs/model_contracts/omega1_2_1_current_baseline_growth_20260606_contract.md`.
- Manifest: `data/ensemble/supervised/omega1_2_1_current_baseline_growth_20260606/omega1_2_1_manifest.json`.
- Parent baseline: `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`.
- Static balanced candidate: `omega1_2_1_balanced_compensated_exposure_scale135_cap055`, validation `+61.14%` / MDD `-7.32%`, OOS `+45.31%` / MDD `-5.54%`.
- Static aggressive candidate: `omega1_2_1_aggressive_compensated_exposure_scale200_cap090`, validation `+100.54%` / MDD `-10.68%`, OOS `+72.76%` / MDD `-8.11%`.
- Learned selector script: `scripts/train_eval_omega1_2_1_exposure_selector_20260606.py`.
- Learned selector report: `tmp/causal_regen_20260516/omega1_2_1_exposure_selector_20260606/report.json`.
- Best learned selector: `omega1_2_1_learned_extra_win_top40_scale200_cap090`, validation `+54.18%` / MDD `-5.47%`, OOS `+35.97%` / MDD `-4.14%`.
- Learned selector is not promoted over static balanced because OOF win AUC is weak (`extra_win=0.3714`). Treat it as diagnostic until better high-confidence features are added.

## Omega1.2.1 Aggressive Current Baseline Memory - 2026-06-06

- Current Omega live baseline: `omega1_2_1_aggressive_compensated_scale200_cap090`.
- Contract: `docs/model_contracts/omega1_2_1_aggressive_current_baseline_20260606_contract.md`.
- Manifest: `data/ensemble/supervised/omega1_2_1_aggressive_compensated_scale200_cap090/baseline_manifest.json`.
- Parent baseline: `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`.
- Transform: compensated TP/SL + exposure scale, scale `2.0`, notional cap `0.90`.
- Validation replay: PnL `+100.5427%`, MDD `-10.6777%`, WR `63.6364%`, trades `33`.
- OOS replay: PnL `+72.7600%`, MDD `-8.1082%`, WR `72.2222%`, trades `18`.
- User accepted MDD near `-10%` as acceptable for this baseline.
- This is wired into `trading_bot.py` through `trading_bot_modules/omega1_2_1_live.py`.
- Runtime default: `FINAL_GOVERNOR_OMEGA1_2_1_ENABLE=1`; `FINAL_GOVERNOR_FULLY_LEARNED_ENABLE=0` unless explicitly overridden.
- Omega CASH is terminal and must not fall through to Alpha7 or legacy sleeves.

## Omega1.2.1 TP Runner Baseline Red-Team Block - 2026-06-13

- Deprecated artifact: `omega1_2_1_tp_runner_only_baseline_20260612`.
- Audit doc: `docs/audits/omega1_2_1_tp_runner_baseline_redteam_20260613.md`.
- Audit report: `tmp/causal_regen_20260516/omega1_2_1_tp_runner_baseline_redteam_audit_20260613/report.json`.
- Verdict: `deprecated_do_not_use_active_or_candidate`.
- The reported OOS `+205.92%` is OOS-mined research output, not clean holdout evidence.
- Primary blockers: TP-runner/config selection used 2026 OOS, TP/SL checks were close-threshold rather than intrabar barrier, execution assumed next-bar-open maker limit fills, and ledger prices did not record actual accounting fills.
- Do not use this model, `tp_runner_meta_selector_20260610`, or derived 2026-06-13 time-decay/lifecycle-selector experiments as active baseline, promotion evidence, or clean comparison target unless retrained/reselected without 2026 OOS and re-evaluated on a fresh untouched holdout.

## Secondary Feature Provenance Memory - 2026-05-30

- Detailed contract: `docs/audits/secondary_feature_generation_contract_20260530.md`
- Critical dependency rule: `teacher_*` is downstream of AI/TSFM plus M7 outputs. It must never be used to generate AI/TSFM, M7, regime, or `a5dir_*` artifacts.
- Current Alpha8 teacher columns are generated by `pipeline/teacher_meta_side_features.py`:
  - `teacher_long_edge`
  - `teacher_short_edge`
  - `teacher_side_margin`
  - `teacher_side_disagreement`
  - `teacher_quantile_skew`
  - `teacher_uncertainty`
  - `teacher_tail_warning`
- Certified teacher builders:
  - `scripts/build_certified_teacher_features_2025.py`
  - `scripts/build_certified_teacher_features_2026.py`
- Valid teacher inputs are same-timestamp risk/uncertainty AI/TSFM outputs and only M7 risk/quality outputs allowed or conditionally allowed by `docs/audits/m7_redteam_contract_20260530.md` after clean score-only generation. Direction outputs such as `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, `patchtst_median`, and removed M7 direction-context columns are not active downstream inputs. Teacher is final-policy context, not an upstream feature source.
- `a5dir_*` is also downstream. It may be final policy/router context, but not input to AI/TSFM, M7, teacher, or regime generation.
- AI/TSFM outputs, M7 outputs, and current/future regime outputs should not be cross-fed into each other under existing prefixes. If a future experiment intentionally couples these layers, require a new versioned artifact and no-leak audit.
- PCA/compressed feature variants are downstream experiment artifacts only. Do not use them as upstream sources for AI/TSFM, M7, teacher, regime, or router generation.
- If a future model intentionally changes this dependency DAG, require a new versioned prefix/artifact and a new no-leak audit. Do not reuse existing prefixes with changed semantics.

## HF Offline Model Inventory Memory - 2026-05-30

- Inventory-only document: `docs/audits/hf_offline_model_inventory_20260530.md`
- No new HF AI feature contract has been selected yet. Input features, objective, horizon, output prefix, and downstream consumer are still undecided.
- Cached candidates include PatchTSMixer, Chronos, Moirai, TimesFM, Granite TTM, Kairos, Kronos, and Lag-Llama families.
- Runtime package status in `quant_ai`: `transformers`, `torch`, `chronos`, `gluonts`, and `uni2ts` are available; `timesfm` package is not available even though TimesFM model caches exist.
- Do not document generated `ai_hf_*` features as active or candidate inputs until a separate feature/target contract is approved.

## AI PatchTSMixer Direction Core Memory - 2026-05-30

- Experiment report: `docs/audits/ai_patchmix_direction_core_20260530.md`
- Generator: `scripts/build_ai_patchmix_direction_core_20260530.py`
- Runner: `scripts/run_ai_patchmix_direction_core_20260530.sh`
- New output family: `ai_patch_*`
- Outputs:
  - `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2024_score2025/ai_patchmix_direction_core_2025.csv`
  - `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2025_score2026/ai_patchmix_direction_core_2026.csv`
- Input contract: clean upstream market-derived features only. `teacher_*`, `m7_*`, `a5dir_*`, existing AI/TSFM output families, labels, targets, future, realized, and PnL columns are forbidden.
- 2026 diagnostic: h12 is strongest (`balanced_accuracy=0.4854`, `OVR_AUC=0.6492`). h24/h48 have ranking signal by AUC but weak flat handling. Treat as entry/meta context first, not standalone direction owner.

## AI PatchTSMixer Direction Input Rework Memory - 2026-05-30

- Rework report: `docs/audits/ai_patchmix_direction_input_rework_20260530.md`
- Updated generator: `scripts/build_ai_patchmix_direction_core_20260530.py`
- Added runners:
  - `scripts/run_ai_patchmix_direction_core_audit_v2_20260530.sh`
  - `scripts/run_ai_patchmix_direction_core_audit_compact_20260530.sh`
- The rework used the document-manager feature audit to add only upstream, live-computable direct features. It still forbids `teacher_*`, `m7_*`, `a5dir_*`, existing AI/TSFM outputs, regime sidecars, labels, targets, future path, and PnL-derived columns.
- `audit_full` is not promoted because it improved some AUC values but added noise and weakened balanced accuracy.
- `audit_compact` is the current best AI input rework candidate for h24 direction context:
  - 2026 h24 balanced accuracy: `0.365542` -> `0.426686`
  - 2026 h24 OVR AUC: `0.603470` -> `0.616559`
- h12 remains close to the original baseline. Keep h12 as entry/meta context and ablate before using it as a hard direction owner.
- h48 remains secondary/ranking context only.

## AI 4-Model H6 BACC Loop Memory - 2026-05-30

- Loop report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- Primary horizon changed to h6 to align with the current Regime3 stability/risk sidecar.
- `regime3_pred_*` and Regime4 sidecars are excluded from this loop.
- Current best h6 direction candidate:
  - strict-clean: Chronos h6 zero-shot + compact current/core + split-local regime context
  - 2026 OOS bacc: `0.5009`
  - 2026 OOS OVR AUC: `0.6832`
- Best research-only h6 stack:
  - Chronos h6 + compact current/core + split-local regime + old TiDE outputs
  - 2026 OOS bacc: `0.5020`
  - 2026 OOS OVR AUC: `0.6841`
  - not active/live promotable because the old PatchTST/TiDE/DLinear combo CSV has timestamp gaps/NaNs and required median fill in the research comparison.
- Existing PatchTST/TiDE/DLinear output-family heads did not beat the strict-clean Chronos/core stack. TiDE remains risk/exit context unless regenerated under the current fail-fast timestamp contract.
- User correction: prioritize standalone model quality over AI-output ensembling.
- Current best standalone h6 candidate:
  - Chronos h6 zero-shot + compact current/core + split-local regime context
  - label preset: `active_dense`
  - 5-seed mean 2026 OOS bacc: `0.5114`
  - 5-seed std bacc: `0.0012`
  - max single-seed bacc: `0.5132`
  - mean OVR AUC: `0.6651`
  - artifact: `tmp/causal_regen_20260516/ai_single_model_h6_chronos_core_seedcheck_20260530/summary.json`
- Higher-AUC standalone reference:
  - Chronos h6 + core/local regime with `mae_light`
  - 5-seed mean bacc: `0.5013`
  - mean OVR AUC: `0.6834`
  - better ranking surface, weaker class bacc.
- `bacc >= 55%` was not reached in this loop.

## AI Role-Specific TSFM Evaluation - 2026-05-30

- Report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- Runner: `scripts/run_ai_role_specific_experiments_20260530.py`
- Output: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/summary.json`
- Regenerated exact timestamp artifacts:
  - `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2025_exact.csv`
  - `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2026_exact.csv`
- Contract: exact timestamp TSFM regeneration; no cross-model output ensembling for role metrics.
- Manifest note: `tide_vol_zscore` warmup non-finite values are explicitly zeroed and recorded. Timestamp gaps remain fail-fast.

Role-specific decisions:

- PatchTST/PatchTSMixer raw direction output is not a hard direction owner from this test:
  - 2026 h6 bacc `0.3452`
  - 2026 h12 bacc `0.3475`
- Chronos raw q50-sign output is not a hard direction owner:
  - 2026 h6 bacc `0.3426`
  - large-move AUC `0.5511`
- TiDE should be documented as a risk/exit/size candidate:
  - 2026 h6 top30 adverse-risk AUC raw `0.7354`
  - 2026 h12 top30 adverse-risk AUC raw `0.7227`
- DLinear should be documented as low-frequency trend/flow context only:
  - 2026 h24 trend AUC flow `0.4938`
  - h24 return correlation flow `0.0469`
- TimesNet should be documented as weak cycle/session context only:
  - anchor-revert entry-quality AUC `0.5193`

Documentation implication:

- Keep the current best standalone h6 direction candidate as Chronos/core `active_dense` from the seed-check artifact.
- Do not document PatchTST, raw Chronos q50 sign, DLinear, or TimesNet as live hard-entry owners unless a downstream policy ablation proves PnL/MDD/trade-count improvement.
- TiDE may be promoted only as a risk-side input candidate after active-path backtest ablation.

## AI Reworked Input Retrain - 2026-05-30

- Report: `docs/audits/ai_4model_h6_bacc_loop_20260530.md`
- Reworked NF runner: `scripts/retrain_ai_role_models_reworked_inputs_20260530.py`
- NF summary artifact: `tmp/causal_regen_20260516/ai_role_models_reworked_inputs_20260530/summary.json`
- PatchTSMixer summary artifact: `tmp/causal_regen_20260516/ai_patchmix_h6_reworked_inputs_20260530/summary.json`
- Existing `data/nf_*` live packs were not overwritten.

Documented result:

- TiDE reworked input retrain is the strongest AI change:
  - h6 adverse-risk AUC raw `0.7484`
  - h12 adverse-risk AUC raw `0.7336`
- PatchTSMixer reworked inputs improved the h6/h12 class surface:
  - h6 bacc `0.5016`
  - h12 bacc `0.4983`
  - strict `fit2024 -> score2026` h6 bacc `0.5079`
  - strict `fit2024 -> score2026` h12 bacc `0.4821`
  - still not the primary hard direction owner.
  - h12 values are evaluated with the actual h12 head; earlier scratch h12 output that reused h6 predictions is superseded.
- DLinear remains weak:
  - h24 trend AUC flow `0.4929`
  - h24 return correlation flow `0.0472`
- TimesNet full CPU retrain is deferred because the loop was too slow.

Docs implication:

- Active/live specs should not promote reworked AI packs until downstream PnL/MDD/trade-count ablation passes.
- TiDE is the first candidate to test in the risk/exit/size layer.
- PatchTSMixer is secondary entry context.
- DLinear/TimesNet are not active hard gates.

## Chronos Standalone Multi-Series Test - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifacts:

- Runner: `scripts/test_chronos_multiseries_standalone_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_multiseries_standalone_20260530/summary.json`

Recorded contract:

- Chronos was tested as a standalone zero-shot AI model only.
- No downstream CatBoost/meta layer and no ensemble were used.
- Threshold/inversion selection used 2025 only and was fixed for 2026.

Recorded result:

- Best standalone 2026 OOS bacc was `0.3853` from `price_cvd_divergence`.
- `price_cvd_divergence` and `vwap_dist_96` produced useful large-move AUC (`0.6539`, `0.6402`) but poor direction bacc.

Docs decision:

- Do not document this Chronos standalone multi-series output as active/live direction owner.
- It may be documented only as an experimental uncertainty / large-move context candidate after a downstream PnL/MDD/trade-count ablation.
- TimesNet reworked-input run is incomplete because no summary artifact was produced; it should not be listed as a completed model result.

## PatchTSMixer Binary Tradeable Target - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifacts:

- Runner: `scripts/train_ai_patchmix_binary_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchmix_binary_tradeable_20260530/summary.json`
- Log: `tmp/causal_regen_20260516/ai_patchmix_binary_tradeable_20260530_run.log`

Recorded contract:

- Uses the existing expanded PatchTSMixer `audit_compact_local_regime` input contract.
- Binary target excludes neutral/flat bars and learns `short` vs `long` only.
- No legacy alias/fallback feature contract was added.

Recorded result:

- strict `2024->2026` h6 `tradeable_fee2`: bacc `0.5249`, AUC `0.5368`, coverage `0.6166`.
- strict `2024->2026` h12 `tradeable_fee2`: bacc `0.5192`, AUC `0.5293`, coverage `0.7568`.

Docs decision:

- Document binary tradeable PatchTSMixer h6 as an experimental Alpha6/Alpha7 direction-context candidate.
- Do not document it as active/live hard-entry owner until PnL/MDD/trade-count ablation passes.

## AI Role-Based Pass Reassessment - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifact:

- `tmp/causal_regen_20260516/ai_role_pass_reassessment_20260530.json`

Recorded decision:

- AI features are no longer judged by h6 direction bacc alone.
- Model families are judged by intended role:
  - TiDE: risk/exit/sizing
  - PatchTSMixer binary: direction-bias context
  - Chronos: large-move/uncertainty context
  - TimesNet: anchor/session modifier
  - DLinear: low-frequency flow/trend context

Status:

- TiDE: `PASS`
- PatchTSMixer binary: `HOLD_FAIL`
- Chronos: `PASS` for uncertainty / large-move / downside-risk context
- TimesNet: `WEAK_PASS_CANDIDATE`
- DLinear: `HOLD_FAIL`

Docs rule:

- Only TiDE may be treated as a strong AI candidate.
- Chronos/TimesNet must be documented as context/modifier candidates only.
- None of those candidates may be documented as a hard live entry owner before PnL/MDD/trade-count ablation passes.

## Chronos Expanded Uncertainty Retest - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifacts:

- Runner: `scripts/test_chronos_uncertainty_large_move_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530/summary.json`

Recorded decision:

- Chronos was retested with expanded input series after changing its role from direction owner to uncertainty / large-move / downside-risk context.
- Live-safe EWM smoothing was added to Chronos score outputs; it uses current and past Chronos outputs only.
- Do not document Chronos as a hard long/short direction owner.
- Preferred active-candidate feature names for later downstream ablation:
  - `chronos_atr14_upside_band_ewm3`
  - `chronos_atr14_width_ewm6`
  - `chronos_atr14_width`
  - `chronos_atr14_large_move_score`
  - `chronos_realized_vol24_width`
  - `chronos_realized_vol24_large_move_score`
- Latest 2026 OOS evidence:
  - `atr14_pct` `upside_band_ewm3`: 2025 large/downside AUC `0.6050`/`0.6018`; 2026 large/downside AUC `0.6228`/`0.6307`.
  - `atr14_pct` width large-move AUC `0.6172`, downside AUC `0.6188`.
  - `realized_vol_24` width large-move AUC `0.6152`, downside AUC `0.6039`.
- Active docs must describe these as risk/uncertainty modifiers only: threshold tightening, notional reduction, TP/SL widening, or exit-pressure boost.

## PatchTST Tradeable Representation Test - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Artifacts:

- Runner: `scripts/train_ai_patchtst_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchtst_tradeable_20260530/summary.json`

Recorded result:

- PatchTST end-to-end h6 `tradeable_fee2`: bacc `0.5050`, AUC `0.5054`.
- PatchTST embedding+MLP: bacc `0.5009`, AUC `0.5002`.
- PatchTST embedding+CatBoost: bacc `0.5046`, AUC `0.5080`.
- All variants underperform PatchTSMixer binary strict h6 bacc `0.5249`, AUC `0.5368`.

Docs decision:

- Do not document PatchTST as an active/live candidate.
- PatchTST may be revisited only with a local pretrained checkpoint or a separate self-supervised pretraining artifact.

## TimesNet Role Lock - 2026-05-30

Audit report:

- `docs/audits/ai_4model_h6_bacc_loop_20260530.md`

Completed artifact:

- `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/summary.json`

Recorded decision:

- TimesNet is fixed as a session / anchor-reversion auxiliary feature family.
- Do not document TimesNet as a hard long/short direction owner.
- Current valid TimesNet outputs:
  - `ai_anchor_revert_prob`
  - `ai_anchor_overheat`
  - `ai_anchor_trend_escape_prob`
  - `timesnet_cycle_sin`
  - `timesnet_cycle_cos`
  - `timesnet_cycle_delta`
- Intended downstream use:
  - threshold adjustment around anchor reversion;
  - notional/leverage reduction under overheat;
  - shorter TP/faster exit in reversion regimes;
  - mean-reversion veto under trend-escape risk.
- Latest role metrics:
  - `entry_quality_auc_anchor_revert=0.51996`
  - `entry_quality_auc_trend_escape=0.48004`
  - `cycle_delta_ret_corr=-0.02176`
- This is not active/live-promoted until a downstream PnL/MDD/trade-count ablation passes.

## Default Prompt

```text
너는 /home/llewyn/crypto-scalping 프로젝트의 Docs Manager다.
목표는 docs 폴더, 특히 active/live 코드 경로를 바로 개발 가능한 명세로 유지하는 것이다.

반드시 확인할 문서:
- docs/active_live/README.md
- docs/active_live/alpha7_live_stack.md
- docs/active_live/trading_bot_runtime.md
- docs/active_live/module_interfaces.md
- docs/active_live/change_log.md

작업 규칙:
- 코드나 모델 로직이 바뀌면 관련 active spec도 같은 변경 세트에서 갱신한다.
- active path에는 alias/compat/fallback prefix를 문서상 허용하지 않는다.
- 레거시 레짐 버그 피쳐(`clean_regime_2024_unsup_v4_*`, `clean_regime4_2024_unsup_v1_*`)는 문서에서 historical/reference-only, not active input으로 표기한다.
- Regime3 active policy는 `docs/active_live/regime3_policy_20260530.md`를 따른다. `regime3_pred_*` future-class 피쳐는 active action/direction owner에서 제거된 것으로 기록하고, 안정성/전환위험 피쳐(`regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, `regime3_churn_h6_risk_score`)만 veto/size/leverage/TP/SL/hold 조정 context로 문서화한다.
- funding-family 입력이나 derived artifact를 문서화할 때 clean funding provenance를 반드시 기록한다. 증거가 없으면 active/live/promotable로 쓰지 않는다.
- feature/state/artifact contract mismatch는 fail-fast로 기록한다.
- 산출물은 변경된 문서 목록, 갱신한 계약, 남은 문서 부채다.
```

## Formula Teacher V1 Documentation Note - 2026-05-31

Runtime/code update:

- `pipeline/teacher_meta_side_features.py` now defines strict Formula Teacher v1.
- `pipeline/certified_teacher_meta_features.py` delegates to the same transform to avoid divergent teacher formulas.
- `scripts/rebuild_formula_teacher_features_20260531.py` regenerates candidate CSVs and writes `formula_teacher_v1_audit.json`.

Active documentation policy:

- Teacher is a risk/meta-context compressor, not an active long/short direction owner.
- Required inputs are OOS model outputs only: M7 quantile/risk/quality and AI risk/reward outputs.
- Forbidden inputs: labels, targets, action scores, future-path statistics, realized PnL.
- Active docs should describe `teacher_uncertainty`, `teacher_tail_warning`, and `teacher_side_disagreement` as risk controls first.
- `teacher_long_edge` and `teacher_short_edge` are allowed as diagnostics or label-quality context; avoid using them as direct entry action heads without a new audit.

## Omega1 Teacher Contract - 2026-05-31

Canonical audit/spec:

- `docs/audits/omega1_teacher_contract_20260531.md`

User decision:

- The current version line is Omega1.
- Omega1 teacher inputs are pass-only explicit columns from Regime3 current sensitive wide24 + Regime3 h6 stability/risk sidecar + TiDE risk outputs + Chronos uncertainty outputs + a narrow M7 subset. Broad prefix selection is forbidden.
- Active M7 teacher inputs are restricted to `m7_q10`, `m7_q90`, `m7_qwidth`, plus ZigZag-retrained M7 direction probability/edge fields (`m7_zigzag_cat_fl/up/dn/confidence/side_edge/trade_prob` and `m7_zigzag_xgb_fl/up/dn/confidence/side_edge/trade_prob`). `m7_zigzag_*_action`, `m7_quality_pred`, and `m7_hold_pred` remain blocked.
- Experimental `m7_clean_*` recomputed risk/execution context was retired by user decision because the signal ownership was ambiguous. It is removed from active teacher inputs and from the M7 ZigZag CSVs.
- Regime4 must be removed from Omega1 teacher inputs entirely.
- Omega1 active action-label training data is fixed to `3-class ZigZag action`.
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
- Omega1 teacher builders must use that explicit artifact and fail fast if it is missing.
- Legacy `tp_sl_action_score -> threshold -> 3-class` and TP/SL action labels are retired for Omega1. They may remain only in historical reports and must not be used as active labels.
- Silent fallback from ZigZag labels to `tp_sl_action_score` is forbidden.
- Any second-stage feature family trained on previous 2-action, binary long/short, or tradeable/no-trade action labels is stale for Omega1 active use. It must be retrained against `zigzag_action` and/or the explicit soft-label columns before promotion; do not silently map old outputs into the new 3-class label.

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
- Full summary: `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531/zigzag_action_model_zoo_summary.json`.
- Tested direct action/direction label owners: Trend-XGB-style, M7 multitarget LGBM-style, quantile-feature proxy, Alpha HGB action master, Alpha LGBM action master, and Alpha CatBoost action master.
- 2026 best practical candidate: `alpha_catboost_action_master_like` (`bacc=0.565474`, `ovr_auc=0.755714`).
- 2026 best Trend-XGB-style direct retrain: `trend_xgb_like_xgb` (`bacc=0.555528`, `ovr_auc=0.750837`).
- LGBM variants have strong train/score gaps and require stricter regularization or time-series CV before active/live promotion.
- The audit is a promotion-candidate comparison only and does not automatically change the Omega1 active feature list.

M7 ZigZag direction integration:

- Audit: `docs/audits/m7_zigzag_direction_integration_20260531.md`.
- Integration script: `scripts/integrate_zigzag_direction_into_m7_20260531.py`.
- Generated files:
  - `data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv`
  - `data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv`
- Added M7-named direction columns:
  - `m7_zigzag_cat_fl`, `m7_zigzag_cat_up`, `m7_zigzag_cat_dn`, `m7_zigzag_cat_action`, `m7_zigzag_cat_confidence`, `m7_zigzag_cat_side_edge`, `m7_zigzag_cat_trade_prob`
  - `m7_zigzag_xgb_fl`, `m7_zigzag_xgb_up`, `m7_zigzag_xgb_dn`, `m7_zigzag_xgb_action`, `m7_zigzag_xgb_confidence`, `m7_zigzag_xgb_side_edge`, `m7_zigzag_xgb_trade_prob`
- Source models: `alpha_catboost_action_master_like` and `trend_xgb_like_xgb`, scored as `2024 train -> 2025 score` and `2025 train -> 2026 score`.
- The probability/edge/confidence fields are now explicitly allowed Omega1 teacher inputs by `docs/model_contracts/omega1_processed_feature_contract_20260531.md`. The ordinal `m7_zigzag_*_action` fields remain blocked.

Omega1 processed feature contract:

- Canonical tracking document: `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- This document is the source of truth for the full Omega1 processed / layered
  feature registry, not only teacher inputs.
- Omega1 layer structure:
  - Layer 1: source/current features.
  - Layer 2: 2024-trained processed feature generators scored on 2025/2026,
    including AI/TSFM, Chronos, Regime3, M7, and standalone direction
    generators stored with legacy `dir3_*` prefixes.
  - Layer 3: teacher/meta/parent stack trained on 2025 Layer-2 OOS scores and
    tested on 2026 Layer-2 scores.
  - Layer 4: final policy/backtest/live execution.
- `dir3_*` is a legacy artifact prefix, not a layer name. Do not rename
  historical artifacts; classify standalone 2024-trained `dir3_*` direction
  generators as Layer 2. `teacher_*` remains Layer 3.
- Teacher input features are only the `teacher_generation` consumer subset.
  Omega1 also tracks features for `parent_policy`, `risk_sizing_exit`,
  `diagnostics_only`, and `research_only`.
- A feature can be approved for one consumer layer and blocked from another.
  Do not treat teacher approval as global approval, and do not treat
  non-teacher status as exclusion from the whole Omega1 architecture.
- It separately tracks architecture-approved processed features, active teacher
  inputs, M7 usage status, new M7 ZigZag direction candidates, teacher outputs,
  context-only families, hold/fail families, and hard exclusions.
- Legacy M7 columns may remain present in CSV artifacts for audit compatibility, but the contract does not treat them as approved Omega1 inputs.
- Any processed-feature addition/removal/promotion/demotion must update that contract first, including its change log.

Omega1 forbidden regime prefixes:

- `clean_regime4_state24_sticky090_v2_*`
- `clean_regime4_2024_unsup_v1_*`
- `clean_regime_2024_unsup_v4_*`
- `regime4_pred_*`

Omega1 implementation note:

- `scripts/build_hgb_teacher_features_20260531.py` is now the Omega1 HGB teacher builder.
- It must fail if any Regime4 column is selected as an input.
- It must not select broad `ai_*`, `m7_*`, `patchtst_*`, `dlinear_*`, or legacy diagnostic prefixes.
- Current Omega1 core inputs:
  - `cvp_regime`, `regime_trending`
  - `ai_adverse_risk`, `ai_reward_risk`, `ai_vol_regime_pct`, `tide_vol_zscore`
  - `chronos_atr14_upside_band_ewm3`, `chronos_atr14_width_ewm6`, `chronos_atr14_width`, `chronos_atr14_large_move_score`, `chronos_realized_vol24_width`, `chronos_realized_vol24_large_move_score`
  - `regime3_current_sensitive_wide24_bull_prob`, `regime3_current_sensitive_wide24_bear_prob`, `regime3_current_sensitive_wide24_chop_prob`, `regime3_current_sensitive_wide24_confidence`, `regime3_current_sensitive_wide24_entropy`, `regime3_current_sensitive_wide24_margin`
  - `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, `regime3_churn_h6_risk_score`
  - `m7_q10`, `m7_q90`, `m7_qwidth`
  - `m7_zigzag_cat_fl`, `m7_zigzag_cat_up`, `m7_zigzag_cat_dn`, `m7_zigzag_cat_confidence`, `m7_zigzag_cat_side_edge`, `m7_zigzag_cat_trade_prob`
  - `m7_zigzag_xgb_fl`, `m7_zigzag_xgb_up`, `m7_zigzag_xgb_dn`, `m7_zigzag_xgb_confidence`, `m7_zigzag_xgb_side_edge`, `m7_zigzag_xgb_trade_prob`
- Excluded from Omega1 active teacher until separately promoted: `ai_dir_*`, `patchtst_*`, `dlinear_*`, `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`, `m7_quality_pred`, `m7_hold_pred`, `m7_zigzag_*_action`, `m7_gmm_*`, `m7_iso_*`, `m7_vae_*`, `m7_hdb_*`, `m7_gate_block`, raw M7 price outputs, and M7 action/size/confidence/composite outputs.
- Its current output prefix remains `teacher_hgb_*`; use the audit `version=omega1` to identify the contract.
- `teacher_hgb_*` outputs are allowed as downstream Omega1 parent/risk/final-policy inputs only after the teacher layer is generated.
- `teacher_*` remains forbidden as an input to AI, M7, Regime3, router, and teacher-generation jobs to prevent circular features.
- Current allowed downstream teacher outputs: `teacher_hgb_p_cash`, `teacher_hgb_p_long`, `teacher_hgb_p_short`, `teacher_hgb_confidence`, `teacher_hgb_side_edge`, `teacher_hgb_uncertainty`, `teacher_hgb_risk_veto_score`.

Docs policy:

- When documenting or reviewing Omega1, do not reintroduce Regime4 as an input family.
- Regime4 may remain in historical reports only; those reports are not Omega1 active specs.
- When documenting Omega1 features, record the intended consumer layer:
  `teacher_generation`, `parent_policy`, `risk_sizing_exit`,
  `diagnostics_only`, or `research_only`.
- Do not collapse the whole Omega1 feature registry into teacher inputs.
  Teacher is only one consumer layer in the architecture.

Omega1 Chronos pass-only rebuild:

- Artifact: `tmp/causal_regen_20260516/omega1_hgb_teacher_current_chronos_passonly_candidates_20260531_thr008`
- Feature count: `27`
- Added Chronos pass context by exact timestamp join from `tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530`.
- Added Regime3 current sensitive wide24 context by exact timestamp join from `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530`.
- Superseded by the 2026-05-31 red-team audit: M7 target-family inputs are no longer allowed in active Omega1 teacher generation.
- `m7_tail_risk` is conditional weak context and is not included in this strict pass-only Omega1 HGB teacher rebuild.
- Label-probe metrics: train bacc `0.7983`, train OVR AUC `0.9067`; 2026 OOS bacc `0.3321`, OVR AUC `0.5198`.

Omega1 HGB teacher with M7 ZigZag inputs:

- Artifact: `tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_20260531`
- Script: `scripts/build_hgb_teacher_features_20260531.py`
- Active label source: `zigzag_action` from `tmp/causal_regen_20260516/zigzag_action_labels_20260531`; no `tp_sl_action_score` thresholding.
- Input count: `37` explicit second-stage features.
- Added active M7 inputs: `m7_q10`, `m7_q90`, `m7_qwidth`, plus M7 ZigZag probability/edge/confidence/trade-probability fields. `m7_quality_pred`, `m7_hold_pred`, and `m7_zigzag_*_action` remain blocked.
- Exact joins: Regime3 h6, Regime3 current, Chronos, M7 ZigZag, and ZigZag label.
- Label-probe metrics: train bacc `0.6986`, train OVR AUC `0.8662`; 2026 OOS bacc `0.5637`, OVR AUC `0.7689`.

Omega1 Mamba teacher candidate:

- Script: `scripts/train_omega1_mamba_teacher_20260531.py`
- Artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_current_chronos_seq72_20260531_e4`
- Architecture: native `mamba_ssm.Mamba`, 72-step sequence, 2 layers, d_model `96`.
- Inputs: 27 Omega1 second-stage features plus 90 base current-context numeric features; total `117`.
- Outputs: `teacher_mamba_p_cash`, `teacher_mamba_p_long`, `teacher_mamba_p_short`, `teacher_mamba_confidence`, `teacher_mamba_side_edge`, `teacher_mamba_uncertainty`, `teacher_mamba_risk_veto_score`.
- Teacher feedback prevention remains active: `teacher_*` is forbidden during teacher generation.
- Label-probe metrics: train bacc `0.7900`, train OVR AUC `0.9024`; 2025 internal validation bacc `0.3550`, OVR AUC `0.5494`; 2026 OOS bacc `0.4359`, OVR AUC `0.6264`.

Omega1 Mamba teacher M7 ZigZag smoke:

- Artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_m7zigzag_smoke_20260531`
- Script: `scripts/train_omega1_mamba_teacher_20260531.py`
- Smoke scope: `1` epoch only, GPU path and input contract validation, not a final model.
- Active label source: `zigzag_action`.
- Input count: `127` total = `37` explicit second-stage features + `90` base current-context numeric features.
- 2026 OOS smoke label-probe metrics: bacc `0.5712`, OVR AUC `0.7639`.

Omega1 Mamba teacher red-team audit - 2026-05-31:

- Trigger: train metric `bacc=0.7900` / `OVR AUC=0.9024` looked too high.
- Verdict: no evidence that exact timestamp joins pulled future rows. Regime3/Chronos joins use exact `timestamp` with one-to-one validation.
- Confirmed P0 contract violation: `m7_quality_pred == m7_target_quality` and `m7_hold_pred == m7_target_hold` exactly in both 2025 and 2026 candidate frames.
- Patch applied in `scripts/train_omega1_mamba_teacher_20260531.py`: remove `m7_quality_pred` and `m7_hold_pred` from active Mamba teacher inputs. Later contract update re-allowed `m7_q10`, `m7_q90`, and `m7_qwidth` as non-target quantile-risk context after target-alias fail-fast checks.
- Patch also adds target-alias fail-fast guard and fits median/IQR normalization on `train_idx` only, not full 2025 including internal validation.
- Clean rerun artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_redteam_p0_clean_e4_20260531`.
- Clean e4 metrics: train in-sample bacc `0.8321`, OVR AUC `0.9216` (still in-sample/overfit; do not use for selection); internal validation bacc `0.3580`, OVR AUC `0.5529`; 2026 OOS bacc `0.3883`, OVR AUC `0.6073`.
- HGB teacher was patched with the same P0 M7 target-family removal and target-alias guard. Historical clean artifact before M7 ZigZag promotion: `tmp/causal_regen_20260516/omega1_hgb_teacher_redteam_p0_clean_20260531_thr008`; feature count `22`; 2026 OOS bacc `0.3171`, OVR AUC `0.4977`.
- Remaining audit risk: `ai_*`, `tide_*`, and `regime3_*h6*` are upstream model/risk-head outputs and should be isolated in ablations before active/live promotion. `sig_*`, `evt_*`, `execution_quality`, and `liquidity_vacuum` remain current-bar source-audit candidates.

Regime3 CryptoMamba h6 future-context sidecar - 2026-05-31:

- Script: `scripts/train_regime3_cryptomamba_pred_20260531.py`.
- Active artifact: `data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531`.
- Active report: `data/ensemble/reports/regime3_cryptomamba_pred_h6_nocurrent_20260531_report.json`.
- Promoted feature pack: `all_sanitized`, `128` inputs.
- Retest summary: `tmp/causal_regen_20260516/regime3_cryptomamba_feature_sweep_20260531/all_sanitized128_seed_retest_summary.json`.
- Selected seed run: `tmp/causal_regen_20260516/regime3_cryptomamba_feature_sweep_20260531/all_sanitized128_seed20260533`.
- 2026 OOS metrics: bacc `0.672556`, accuracy `0.681084`, OVR AUC `0.843823`, transition AUC `0.695492`.
- Previous active docs-rolled-64 artifact was backed up to `data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531_docsrolled64_backup_20260531`.
- Contract notes: current Regime3 probabilities are target/evaluation-only, not model inputs. Regime4, `teacher_*`, `m7_*`, `a5dir_*`, downstream label/target/future/PnL/ZigZag/wave columns remain blocked as inputs.
- Omega1 teacher input promotion: explicit numeric outputs are now allowed as
  second-stage features:
  `regime3_cmamba_h6_future_bull_prob`,
  `regime3_cmamba_h6_future_bear_prob`,
  `regime3_cmamba_h6_future_chop_prob`,
  `regime3_cmamba_h6_confidence`,
  `regime3_cmamba_h6_transition_prob`,
  `regime3_cmamba_h6_stability_score`.
- `regime3_cmamba_h6_future_pred_id` and
  `regime3_cmamba_h6_future_pred_name` remain excluded. Broad `future` token
  allowance remains forbidden; this is an exact-column prediction-sidecar
  exception only.
- HGB teacher contract check with the new features:
  `tmp/causal_regen_20260516/omega1_hgb_teacher_cmamba_contract_check_20260531`;
  input count `43`; 2026 OOS label-probe bacc `0.5742`, OVR AUC `0.7748`.
- Confidence decoder/calibration audit:
  `tmp/causal_regen_20260516/regime3_cmamba_confidence_decoder_20260531`.
  Raw argmax remains the selected class transform. The best 2025-fitted
  calibrated gate (`transition_hgb_gate_change_thr0.275`) slightly improved
  2025 selection score but did not improve 2026 OOS (`bacc=0.672480` vs raw
  `0.672556`). Do not create a promoted decoded-class sidecar from this audit;
  preserve the explicit numeric probability/confidence/transition/stability
  features already listed in the Omega1 contract.

Omega1 DIR3 direction feature generators - 2026-05-31:

- Retrieval artifact: `data/ensemble/supervised/omega1_dir3_retrieval_20260531`.
- Retrieval audit: `tmp/causal_regen_20260516/omega1_dir3_retrieval_20260531/dir3_retrieval_audit.json`.
- Cycle artifact: `data/ensemble/supervised/omega1_dir3_cycle_20260531`.
- Cycle audit: `tmp/causal_regen_20260516/omega1_dir3_cycle_20260531/dir3_cycle_audit.json`.
- Combined meta-probe: `tmp/causal_regen_20260516/omega1_dir3_combined_meta_probe_20260531/combined_meta_probe_summary.json`.
- Retrieval is retained as a parent/meta candidate: core-only 2026 bacc `0.5649`, proxy WR `62.52%`, proxy trades `14128`; core + retrieval bacc `0.5681`, proxy WR `62.35%`, proxy trades `14028`.
- Cycle is diagnostics-only: standalone bacc `0.4226`, proxy WR `55.95%`, trades `16897`, and combined probe did not improve core.
- Documentation rule: `dir3_*` is a historical artifact prefix. Standalone
  `dir3_*` direction generators trained on 2024 and scored on 2025/2026 are
  Layer 2 processed OOS features, not Layer 3 teacher outputs. They still must
  not feed teacher generation unless a separate OOF/no-leak stacking contract
  is written.

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
- Documentation decision: update the Omega1 processed-feature contract so
  `dir3_patch` is a parent/meta candidate, `dir3_duet` is a weaker research
  parent/meta candidate, and `dir3_chartcnn` remains diagnostics-only.

Omega1 DIR3 financial-paper candidates - 2026-05-31:

- Script: `scripts/build_omega1_dir3_finpaper_features_20260531.py`.
- Audit: `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/dir3_finpaper_audit.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531/finpaper_meta_probe_summary.json`.
- Source mapping:
  - FinTSB contributed the standardized OOS evaluation style: bacc, macro F1,
    OVR AUC, proxy trades, proxy trade rate, and proxy WR.
  - Oxford financial benchmark inspired VSN-LSTM and lightweight PatchTST
    sequence candidates.
  - X-Trend inspired context-set nearest-regime attention using 2024-only
    memory and 2025/2026 OOS scoring.
- Generated artifacts:
  - `data/ensemble/supervised/omega1_dir3_vsnlstm_20260531`.
  - `data/ensemble/supervised/omega1_dir3_lpatchtst_20260531`.
  - `data/ensemble/supervised/omega1_dir3_xtrend_20260531`.
- Standalone 2026 label-probe:
  - `dir3_vsnlstm`: bacc `0.5766`, OVR AUC `0.7608`, proxy trades `12416`, proxy WR `62.51%`.
  - `dir3_lpatchtst`: bacc `0.5062`, OVR AUC `0.7000`, proxy trades `13575`, proxy WR `59.11%`.
  - `dir3_xtrend`: bacc `0.5010`, OVR AUC `0.6863`, proxy trades `11340`, proxy WR `58.38%`.
- Combined parent/meta probe:
  - core-only: bacc `0.5663`, proxy WR `62.73%`, proxy trades `14025`.
  - core + VSN-LSTM: bacc `0.5793`, proxy WR `63.47%`, proxy trades `13757`.
  - core + lightweight PatchTST: bacc `0.5720`, proxy WR `63.19%`, proxy trades `14060`.
  - core + X-Trend: bacc `0.5644`, proxy WR `62.40%`, proxy trades `14048`.
  - core + all finpaper: bacc `0.5781`, proxy WR `63.73%`, proxy trades `13818`.
- Documentation decision: mark `dir3_vsnlstm` as the first paper-inspired
  parent/meta candidate. Keep `dir3_lpatchtst` research-only below VSN-LSTM.
  Keep `dir3_xtrend` diagnostics-only in this implementation. Compare
  `core + VSN-LSTM`, `core + all finpaper`, and `core + dir3_patch` in real
  PnL/MDD/trade-density backtests before any active promotion.

Omega1 DIR3 CryptoMamba direction sidecar - 2026-05-31:

- Script: `scripts/build_omega1_dir3_cryptomamba_direction_20260531.py`.
- Audit: `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/cryptomamba_meta_probe_summary.json`.
- Artifact: `data/ensemble/supervised/omega1_dir3_cryptomamba_20260531`.
- Model: Regime3 CryptoMamba C-Block Merge architecture retargeted to
  `zigzag_action` direction labels.
- Inputs: `128` explicit current/past numeric features; no Regime4,
  `regime3_pred_*`, `regime3_cmamba_*`, `teacher_*`, `a5dir_*`,
  target/label/future/PnL/action-score, or other `dir3_*`
  direction-generator inputs.
- 2026 standalone label-probe: bacc `0.5671`, OVR AUC `0.7486`,
  proxy trades `11564`, proxy WR `62.67%`.
- Combined parent/meta probe: core-only bacc `0.5682`, proxy WR `62.81%`;
  core + CryptoMamba bacc `0.5698`, proxy WR `62.76%`.
- Documentation decision: keep `dir3_cryptomamba` as a parent/meta research
  candidate. It provides a small bacc/AUC lift but is currently below
  `dir3_patch` and `dir3_vsnlstm`.

Omega1 DIR3 Top2 full sweep - 2026-05-31:

- Script: `scripts/sweep_omega1_dir3_top2_full_20260531.py`.
- Summary: `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_sweep_summary.json`.
- Parent/meta probe: `tmp/causal_regen_20260516/omega1_dir3_top2_full_sweep_20260531/top2_full_meta_probe_summary.json`.
- Full artifacts:
  - `data/ensemble/supervised/omega1_dir3_patch_full_20260531`.
  - `data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531`.
- Sweep scope:
  - `dir3_patch`: `3` seeds x `3` HGB settings.
  - `dir3_vsnlstm`: `3` seeds, max `12` epochs, patience `3`.
- Best standalone 2026 label-probe:
  - `dir3_patch_full`: bacc `0.5692`, OVR AUC `0.7640`, proxy trades `13492`, proxy WR `61.76%`.
  - `dir3_vsnlstm_full`: bacc `0.5869`, OVR AUC `0.7689`, proxy trades `12114`, proxy WR `64.13%`.
- Full parent/meta probe:
  - core-only: bacc `0.5663`, proxy WR `62.73%`, proxy trades `14025`.
  - core + patch_full: bacc `0.5783`, proxy WR `63.65%`, proxy trades `13921`.
  - core + vsnlstm_full: bacc `0.5857`, proxy WR `64.87%`, proxy trades `13916`.
  - core + patch_full + vsnlstm_full: bacc `0.5851`, proxy WR `64.21%`, proxy trades `13799`.
- Documentation decision: mark `dir3_vsnlstm_full` as the current top
  direction-context candidate and keep `dir3_patch_full` as the strongest
  tabular/HGB baseline. Combination is not automatically promoted because it
  trades away bacc/WR for AUC.

Alpha7 Regime3 current-context MoE active update - 2026-06-01:

- Scope: heartbeat loop for current-Regime3 MoE. The architecture remains
  separate `bull`, `bear`, and `chop` experts with validation-only selection
  and fixed 2026 OOS evaluation.
- Previous practical candidate:
  `tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601`.
  - Validation Cost3 `+110.67%`, MDD `-40.67%`, trades `172`, WR `13.95%`.
  - 2026 OOS Cost3 `+80.02%`, MDD `-27.81%`, trades `125`, WR `15.20%`.
- Non-promoted tests:
  - `tmp/causal_regen_20260516/alpha7_regime3_current_moe_risk_sizing_overlay_20260601`:
    validation improved but OOS Cost3 fell to `+50.13%`.
  - `tmp/causal_regen_20260516/alpha7_regime3_current_moe_per_expert_conf_20260601`:
    validation Cost3 `+119.85%`, OOS Cost3 `+79.20%`.
- Promoted candidate:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601`.
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_expert_source_mix_20260601.py`.
  - Report:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601/report.json`.
  - Decisions:
    `validation_decisions.csv`, `oos_2026_decisions.csv`.
  - Selected candidate:
    `bull_practical__bear_risk__chop_practical__conf0.80`.
  - Validation Cost1/2/3:
    `+193.07%`, `+187.12%`, `+141.12%`; Cost3 MDD `-40.39%`,
    trades `168`, WR `14.29%`.
  - 2026 OOS Cost1/2/3:
    `+121.43%`, `+104.29%`, `+101.50%`; Cost3 MDD `-27.81%`,
    trades `131`, WR `15.27%`.
- Tracking note: active Alpha7 Regime3-current MoE candidate should now be the
  expert-source mix, not the older all-practical expert candidate.

Alpha7 Regime3 current-context MoE follow-up tests - 2026-06-01:

- `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_per_conf_20260601`
  tested per-expert confidence thresholds on top of the active mix.
  - Best validation candidate: `bull0.85_bear0.80_chop0.80`.
  - Validation Cost3 `+150.44%`, MDD `-39.16%`, trades `165`, WR `14.55%`.
  - 2026 OOS Cost3 `+100.58%`, MDD `-27.81%`, trades `131`, WR `15.27%`.
  - Not promoted; active mix OOS Cost3 is still higher at `+101.50%`.
- `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_retrieval_overlay_20260601`
  tested retrieval confirmation/veto/resize on top of the active mix.
  - Validation selected no overlay.
  - Defensive resize diagnostics can reduce OOS MDD to about `-16.21%` with
    OOS Cost3 around `+93.35%`, but validation Cost3 is weak.
  - Not promoted.
- Active Alpha7 Regime3-current MoE remains:
  `alpha7_regime3_current_moe_expert_source_mix_20260601`,
  selected candidate `bull_practical__bear_risk__chop_practical__conf0.80`.

Alpha7 Regime3 current-context MoE expert-scale update - 2026-06-01:

- Tested per-expert notional/position scaling on top of the active expert-source
  mix without changing routing or expert models.
- Artifacts:
  - `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_20260601`.
  - `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601`.
- Coarse validation-selected scale:
  `bull0.85_bear1.15_chop1.10`.
  - Validation Cost3 `+243.20%`, MDD `-37.44%`, trades `167`, WR `14.97%`.
  - 2026 OOS Cost3 `+102.81%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
- Refined validation-selected scale:
  `bull0.85_bear1.15_chop1.25`.
  - Validation Cost1/2/3:
    `+350.75%`, `+361.91%`, `+270.24%`.
  - 2026 OOS Cost1/2/3:
    `+117.46%`, `+113.87%`, `+103.72%`.
  - 2026 OOS Cost3 MDD `-27.81%`, trades `133`, WR `15.04%`.
- Active Alpha7 Regime3-current MoE should now be tracked as:
  expert-source mix `bull_practical__bear_risk__chop_practical__conf0.80`
  plus expert scale `bull=0.85`, `bear=1.15`, `chop=1.25`.

Alpha7 Regime3 current-context MoE exit-shape diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_exit_shape_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_exit_shape_20260601.py`.
- The validation-selected exit shape was
  `btp1.10_ctp0.90_csl0.85_ch1.00`.
  - Validation Cost3 `+303.78%`, MDD `-37.74%`, trades `168`, WR `14.88%`.
  - 2026 OOS Cost3 `+73.98%`, MDD `-27.81%`, trades `138`, WR `13.77%`.
- Decision: do not promote. The selected exit-shape overlay is a validation
  overfit. Keep the current active candidate:
  expert-source mix plus scale `bull=0.85`, `bear=1.15`, `chop=1.25`.

Alpha7 Regime3 current-context MoE low-confidence fallback scale diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_lowconf_scale_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_lowconf_scale_20260601.py`.
- Validation-selected candidate:
  `lowconf0.70_tp0.95`.
  - Validation Cost3 `+343.59%`, MDD `-37.27%`, trades `157`, WR `17.20%`.
  - 2026 OOS Cost3 `+79.56%`, MDD `-22.47%`, trades `119`, WR `16.81%`.
- Decision: do not promote. It improves validation and OOS MDD but lowers OOS
  Cost3 too much. OOS-best rows are diagnostic only and must not be selected
  because 2026 OOS is fixed evaluation, not a selection set.

Alpha7 Regime3 current-context MoE expert-confidence shrink diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_expert_conf_shrink_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_expert_conf_shrink_20260601.py`.
- Validation-selected candidate:
  `bull_thr0.85_scale0.85`.
  - Triggered rows: validation `20`, 2026 OOS `2`.
  - Validation Cost3 `+315.78%`, MDD `-37.39%`, trades `164`, WR `15.85%`.
  - 2026 OOS Cost3 `+103.20%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
- Decision: do not promote. It is close but still below the current active
  OOS Cost3 `+103.72%`.

Alpha7 Regime3 current-context MoE soft expert fallback diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_soft_expert_fallback_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_soft_expert_fallback_20260601.py`.
- Validation-selected candidate:
  `floor0.65_scale0.70`.
  - Triggered rows: validation `6987`, 2026 OOS `4591`.
  - Validation Cost3 `+163.44%`, MDD `-33.65%`, trades `188`, WR `16.49%`.
  - 2026 OOS Cost3 `+65.59%`, MDD `-30.14%`, trades `149`, WR `13.42%`.
- Decision: do not promote. The low-confidence soft expert fallback adds too
  much OOS noise. A diagnostic OOS-best row exists but is not selected because
  OOS is not a selection set.

Alpha7 Regime3 current-context MoE component-source diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_component_source_mix_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_component_source_mix_20260601.py`.
- Validation-selected candidate:
  `bearPrisk_Frisk__chopPrisk_Fpractical`.
  - Validation Cost3 `+292.96%`, MDD `-36.58%`, trades `171`, WR `14.62%`.
  - 2026 OOS Cost3 `+89.55%`, MDD `-27.81%`, trades `132`, WR `15.15%`.
- Decision: do not promote.
- Diagnostic-only OOS observation:
  `bearPrisk_Fpractical__chopPpractical_Fpractical` reaches OOS Cost3
  `+111.60%`, but it is not validation-selected and therefore cannot be
  promoted from this run.

Alpha7 Regime3 current-context MoE route-quality scale diagnostic - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_route_quality_scale_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_scaled_route_quality_scale_20260601.py`.
- Validation-selected candidate:
  `mhi0.35_mlo0.15_e0.95_up1.10_dn0.80`.
  - Validation high-quality rows `1040`, low-quality rows `0`.
  - 2026 OOS high-quality rows `446`, low-quality rows `0`.
  - Validation Cost3 `+312.60%`, MDD `-40.61%`, trades `174`, WR `14.37%`.
  - 2026 OOS Cost3 `+55.87%`, MDD `-27.81%`, trades `142`, WR `12.68%`.
- Decision: do not promote. Current-regime margin/entropy high-quality scaling
  does not generalize on fixed 2026 OOS.

Alpha7 Regime3 current-context MoE component-source two-stage validation - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_component_source_twostage_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_component_source_twostage_20260601.py`.
- Selection protocol:
  - 2025-10/11: candidate selection.
  - 2025-12: confirmation.
  - 2026: fixed OOS evaluation only.
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
- Decision: do not promote. It is below the current active refined
  expert-scale candidate (`+103.72%` OOS Cost3). Current active remains
  `bull=0.85`, `bear=1.15`, `chop=1.25` on the promoted expert-source mix.

Alpha7 Regime3 current-context MoE monthly-stability scale selection - 2026-06-01:

- Artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601`.
- Script:
  `scripts/eval_alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601.py`.
- Selection rule: validation-month stability score over a small
  active-neighborhood expert-scale grid. 2026 OOS is evaluated only after
  validation selection.
- Selected candidate:
  `bull0.85_bear1.15_chop1.25`, the same as current active.
  - Validation Cost1/2/3 `+350.75% / +361.91% / +270.24%`,
    Cost3 MDD `-37.74%`, trades `167`, WR `14.97%`.
  - Validation monthly Cost3 PnL:
    2025-10 `+72.14%`, 2025-11 `+15.01%`, 2025-12 `+78.85%`.
  - 2026 OOS Cost1/2/3 `+117.46% / +113.87% / +103.72%`,
    Cost3 MDD `-27.81%`, trades `133`, WR `15.04%`.
  - 2026 OOS monthly Cost3 PnL:
    2026-01 `+75.81%`, 2026-02 `+12.31%`.
- Decision: no promotion needed because the stability selector confirms the
  existing active candidate.

Alpha7 Regime3 current-context MoE expert attribution and bull suppression - 2026-06-01:

- Attribution artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_expert_attribution_20260601`.
- Attribution script:
  `scripts/analyze_alpha7_regime3_current_moe_active_expert_attribution_20260601.py`.
- Active candidate Cost3 attribution:
  - Validation full `+270.24%`, MDD `-37.74%`, trades `167`, WR `14.97%`.
  - Validation pieces:
    `bear +216.06%`, `chop +44.57%`, `bull -17.02%`,
    `lowconf -9.24%`.
  - 2026 OOS full `+103.72%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
  - 2026 OOS pieces:
    `bear +48.87%`, `chop +53.39%`, `bull -5.20%`,
    `lowconf +16.93%`.
- Bull suppression artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_bull_suppression_20260601`.
- Bull suppression script:
  `scripts/eval_alpha7_regime3_current_moe_active_bull_suppression_20260601.py`.
- Validation-selected candidate:
  `bullcash_bear1.15_chop1.25`.
  - Validation Cost1/2/3 `+532.94% / +517.03% / +275.67%`,
    Cost3 MDD `-39.37%`, trades `149`, WR `13.42%`.
  - 2026 OOS Cost1/2/3 `+107.91% / +61.69% / +32.86%`,
    Cost3 MDD `-29.75%`, trades `131`, WR `12.21%`.
- Decision: do not promote. The attribution is useful, but direct bull
  suppression is validation-biased and damages fixed 2026 OOS.

Alpha7 Regime3 current-context MoE low-WR diagnostics and guard attempt - 2026-06-01:

- Ledger diagnostic artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_trade_ledger_wr_20260601`.
- Ledger diagnostic script:
  `scripts/analyze_alpha7_regime3_current_moe_trade_ledger_wr_20260601.py`.
- Diagnostic note: direct open-fill ledger is for payoff/exits analysis only;
  promotion still uses official `_combo_metrics`.
- Approximate direct-ledger finding:
  - Stop-loss exits dominate: validation `170/194`, OOS `128/147`.
  - Average win is much larger than average loss:
    validation `+13.28%` vs `-2.20%`, OOS `+9.80%` vs `-2.16%`.
  - The current active policy is a payoff-skew strategy, not a high-WR policy.
- WR guard artifact:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_wr_guard_filter_20260601`.
- WR guard script:
  `scripts/eval_alpha7_regime3_current_moe_wr_guard_filter_20260601.py`.
- Validation-selected guard:
  `q0.00_c0.68_lq0.06_bq0.20`.
  - Validation Cost1/2/3 `+424.73% / +393.50% / +381.85%`,
    Cost3 MDD `-35.57%`, trades `108`, WR `19.44%`.
  - 2026 OOS Cost1/2/3 `+55.14% / +53.59% / +53.66%`,
    Cost3 MDD `-22.16%`, trades `104`, WR `14.42%`.
- Decision: do not promote. Threshold-based WR repair is validation-overfit
  and does not beat active OOS (`+103.72%`, WR `15.04%`).

Alpha7 architecture report execution log - 2026-06-01:

- The architecture report was tested in project-compatible order: low-risk
  overlays first, then small standalone PyTorch contract tests. No active/live
  feature aliases or fallback contracts were added.
- Current active reference:
  `tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601`;
  active candidate `bull0.85_bear1.15_chop1.25`;
  2026 OOS Cost3 `+103.72%`, MDD `-27.81%`, trades `133`, WR `15.04%`.
- Soft MoE test:
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_soft_blend_20260601.py`.
  - Report:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_soft_blend_20260601/report.json`.
  - Selected `p1.0_conf0.65_side0.15`.
  - Validation Cost1/2/3 `+49.81% / +26.17% / +8.83%`;
    validation Cost3 MDD `-43.78%`, trades `263`, WR `12.55%`.
  - 2026 OOS Cost1/2/3 `+147.73% / +103.49% / +109.82%`;
    OOS Cost3 MDD `-34.56%`, trades `188`, WR `14.36%`.
  - Decision: research-only, not promoted because validation is unstable and
    drawdown is worse than active.
- Two-stage entry gate test:
  - Script:
    `scripts/eval_alpha7_regime3_current_moe_two_stage_entry_gate_20260601.py`.
  - Report:
    `tmp/causal_regen_20260516/alpha7_regime3_current_moe_two_stage_entry_gate_20260601/report.json`.
  - Selected `gate0.35`.
  - Validation Cost1/2/3 `+68.81% / +63.83% / +48.37%`;
    validation Cost3 MDD `-28.37%`, trades `71`, WR `12.68%`.
  - 2026 OOS Cost1/2/3 `+19.70% / +17.49% / +15.79%`;
    OOS Cost3 MDD `-20.92%`, trades `50`, WR `20.00%`.
  - Decision: not promoted. It improves hit rate but destroys payoff capture.
- Shared-backbone / FT-Transformer contract test:
  - Script:
    `scripts/eval_alpha7_shared_backbone_ft_contract_test_20260601.py`.
  - Report:
    `tmp/causal_regen_20260516/alpha7_shared_backbone_ft_contract_test_20260601/report.json`.
  - Shared MLP: validation Cost3 `+23.94%`, OOS Cost3 `-15.84%`.
  - FT-Transformer: validation Cost3 `-14.04%`, OOS Cost3 `+12.11%`.
  - Decision: not promoted. These small PyTorch replacements do not match the
    current HGB MoE. Keep active candidate unchanged.

Omega1 Dir3 TabM-CryptoMamba direction sidecar test - 2026-06-01:

- Tested the TabM report as a contained sidecar experiment, not as an active
  model replacement. The implementation only changes the CryptoMamba input
  frontend to a BatchEnsemble/TabM projection and writes separate
  `dir3_tabm_cmamba_*` outputs.
- Script:
  `scripts/build_omega1_dir3_tabm_cryptomamba_direction_20260601.py`.
- Baseline existing CryptoMamba sidecar:
  `tmp/causal_regen_20260516/omega1_dir3_cryptomamba_20260531/dir3_cryptomamba_audit.json`.
  Baseline 2026 bacc/AUC/proxy WR `0.5671 / 0.7486 / 0.6267`.
- Candidate reports:
  - `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_20260601/dir3_tabm_cryptomamba_audit.json`
    (`ensemble_size=5`, max-features 200, actual 154 features):
    2026 bacc/AUC/proxy WR `0.5640 / 0.7458 / 0.6187`.
  - `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_128_20260601/dir3_tabm_cryptomamba_audit.json`
    (`ensemble_size=5`, 128 features):
    2026 bacc/AUC/proxy WR `0.5626 / 0.7461 / 0.6080`.
  - `tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_e3_20260601/dir3_tabm_cryptomamba_audit.json`
    (`ensemble_size=3`, max-features 200, actual 154 features):
    2026 bacc/AUC/proxy WR `0.5536 / 0.7287 / 0.6125`.
- Decision: not promoted and not added to active Omega1 feature contract.
  The 154-feature TabM run slightly improved internal validation but did not
  beat the existing CryptoMamba direction sidecar on fixed 2026 OOS.

Alpha7 full TabM tabular parent contract test - 2026-06-01:

- Tested full TabM as its own lifecycle parent, not just as a front-end.
- Script:
  `scripts/eval_alpha7_full_tabm_parent_contract_test_20260601.py`.
- Report:
  `tmp/causal_regen_20260516/alpha7_full_tabm_parent_contract_test_20260601/report.json`.
- Contract:
  - Existing Alpha7 feature list from active clean parent.
  - Existing lifecycle label builder.
  - Existing `_combo_metrics` validation/OOS evaluation.
  - 2026 OOS evaluated only after validation runtime selection.
- Standard TabM loss collapsed into near-all-cash decisions, so the final run
  uses a trade-biased loss to make the test meaningful while keeping the same
  feature and backtest contract.
- Selected runtime:
  `full_tabm_parent_c0.50_q0.010_s1.00_cap3.00_u0.070`.
  - Validation Cost1/2/3 `+25.41% / +25.75% / +16.53%`;
    Cost3 MDD `-31.85%`, trades `116`, WR `25.00%`.
  - 2026 OOS Cost1/2/3 `+46.14% / +32.10% / +26.66%`;
    Cost3 MDD `-43.63%`, trades `98`, WR `27.55%`.
- Decision: not promoted. It is a higher-WR research branch, but it does not
  beat active `bull0.85_bear1.15_chop1.25` OOS Cost3 `+103.72%`.

Omega1 supervised-label authority update - 2026-06-01:

- User clarified the Omega1 contract: previous Alpha models may be borrowed
  only as architecture references, not reused as active supervised components.
- Active Omega1 supervised target is now canonicalized to:
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531`,
  column `zigzag_action`, classes `0=CASH`, `1=LONG`, `2=SHORT`.
- The canonical contract was updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Active Omega1 supervised heads must not train from `tp_sl_action_score`,
  `wave3_action`, fixed-barrier Alpha6 labels, Alpha lifecycle labels, or
  `FullyLearnedGovernor` TP/SL path labels.
- Alpha code utilities may still be used for frame loading, validation/OOS
  splitting, exact timestamp joins, and backtest metric calculation, but any
  supervised head in an Omega1 active experiment must be retrained on
  `zigzag_action` or an explicitly documented ZigZag-derived soft target.
- New max-feature MoE ZigZag-only experiment script:
  `scripts/retrain_alpha7_active_max_feature_zigzag_moe_20260601.py`.
  Despite the `alpha7` filename lineage, it is an Omega1-compatible test:
  it uses max-feature current-Regime3 MoE structure but replaces all
  supervised action heads with `zigzag_action` classifiers.

Omega1 ZigZag-only MoE risk redesign - 2026-06-01:

- User clarified that after unifying supervised heads on `zigzag_action`,
  remaining execution parameters should be redesigned instead of reusing
  previous Alpha governor/risk labels.
- Script:
  `scripts/eval_alpha7_zigzag_moe_risk_param_sweep_20260601.py`.
- Source supervised model artifact:
  `tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_20260601`.
- Risk redesign artifact:
  `tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_risk_redesign_20260601`.
- Selected candidate:
  `balanced_rr19_pc0.55_fc0.50_edge0.04_rc0.80_b0.75_r0.90_c0.90`.
- Selected runtime parameters:
  - template `balanced_rr19`: notional `0.45`, leverage `2.0`,
    TP `0.026`, SL `0.014`, max-hold `72`, cooldown `6`;
  - primary confidence `0.55`, fallback confidence `0.50`, active edge
    `0.04`, router min confidence `0.80`;
  - expert notional scales: bull `0.75`, bear `0.90`, chop `0.90`.
- Validation Cost3: PnL `+41.34%`, MDD `-5.61%`, trades `339`,
  WR `51.03%`.
- 2026 OOS Cost3: PnL `+5.58%`, MDD `-8.52%`, trades `211`,
  WR `44.55%`.
- Monthly validation Cost3: 2025-10 `+17.15%`, 2025-11 `+1.27%`,
  2025-12 `+12.11%`.
- Monthly 2026 OOS Cost3: 2026-01 `+3.62%`, 2026-02 `+4.99%`.
- Status: contract-compliant research candidate. It fixes the previous
  ZigZag-only default runtime OOS Cost3 `-13.82%`, but it is not active/live
  promoted until monthly stability and walk-forward checks pass.

Omega1 teacher feature retirement - 2026-06-01:

- User decision: discard teacher features for Omega1 active/research modeling.
- `teacher_*` and `teacher_oof_*` are now historical/audit-only and must not be
  used in active Omega1 parent, risk, final-policy, AI, M7, Regime3, router, or
  teacher-generation inputs.
- Omega1 should use Layer 1/2 features directly. Directional feature work should
  focus on approved Layer 2 direction contexts such as M7 ZigZag probability/edge
  fields and standalone `dir3_*` OOS direction generators.
- Canonical contract updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.

Omega1 direction-only stacked head - 2026-06-02:

- User requested a Direction Head using only directional Layer 2 features after
  retiring `teacher_*`.
- Script:
  `scripts/train_omega1_direction_head_direction_only_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_direction_only_20260602`.
- Target: `zigzag_action`; forbidden inputs include `teacher_*`, `teacher_oof_*`,
  Regime4, `a5dir_*`, target/future/PnL/action-score columns.
- Tested variants:
  - `core`: `dir3_vsnlstm_full + dir3_patch_full`, 12 features.
  - `expanded`: core + M7 ZigZag + Regime3 current + Regime3 CryptoMamba, 33 features.
  - `all_direction`: expanded + DIR3 duet + DIR3 CryptoMamba + DIR3 retrieval, 56 features.
- 2025 output is expanding time-series OOF (`omega1_dir_oof_*`); 2026 output is
  final model score (`omega1_dir_*`).
- Best by 2026 OOS bacc: `core`.
  - 2025 OOF: bacc `0.5708`, OVR AUC `0.7723`, proxy WR `64.38%`, OOF coverage `65.00%`.
  - 2026 OOS: bacc `0.5938`, OVR AUC `0.7835`, proxy WR `64.43%`, proxy trades `13110`.
- Expanded/all variants slightly improved OOF but lost on 2026 OOS, so do not
  prefer broad direction-feature stacking yet.

Omega1 direction-only grouped PCA test - 2026-06-02:

- User requested PCA dimensionality reduction on the three direction-head
  variants (`core`, `expanded`, `all_direction`).
- Script:
  `scripts/train_omega1_direction_head_direction_pca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_direction_pca_20260602`.
- Method: group-wise `StandardScaler + PCA`; OOF folds fit PCA only on each
  fold's training prefix, and 2026 OOS uses PCA fit on 2025 only.
- Tested variants:
  - `core_pca`: 12 raw features -> 6 PCA features.
  - `expanded_pca`: 33 raw features -> 17 PCA features.
  - `all_direction_pca`: 56 raw features -> 29 PCA features.
- Best by 2026 OOS bacc: `core_pca`.
  - 2025 OOF: bacc `0.5688`, OVR AUC `0.7717`, proxy WR `64.46%`.
  - 2026 OOS: bacc `0.5961`, OVR AUC `0.7836`, proxy WR `64.78%`,
    proxy trades `13092`.
  - Delta vs raw `core`: OOS bacc `+0.0024`, AUC `+0.0000`,
    proxy WR `+0.35pp`, trades `-18`.
- Design read: grouped PCA slightly improves OOS stability for the compact
  direction stack while reducing dimensionality. Broad PCA variants still do not
  beat compact `core_pca`, so `core_pca` is the preferred direction-only
  compressed candidate.

Omega1 TSFM/Chronos Direction Head comparison - 2026-06-02:

- User requested the same Direction Head comparison for TSFM and Chronos
  feature families.
- Script:
  `scripts/train_omega1_direction_head_tsfm_chronos_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602`.
- Target: `zigzag_action`; no `teacher_*`, no Regime4.
- Tested variants:
  - `tsfm_role`: 24 TSFM role features from the exact TSFM artifact.
  - `chronos_h6`: 5 Chronos h6 direction-surface features.
  - `chronos_uncertainty`: 14 Chronos uncertainty features derived from
    `atr14_pct` and `realized_vol_24` forecast distributions with causal EWM
    width transforms.
  - `chronos_all`: 19 Chronos features.
  - `tsfm_chronos`: 43 TSFM+Chronos features.
  - `core_plus_tsfm`, `core_plus_chronos`, `core_plus_tsfm_chronos`.
- Best by 2026 OOS bacc: `core_plus_tsfm_chronos`, 55 features.
  - 2025 OOF: bacc `0.5684`, OVR AUC `0.7739`, proxy WR `64.67%`.
  - 2026 OOS: bacc `0.5974`, OVR AUC `0.7907`, proxy WR `65.79%`,
    proxy trades `13334`.
  - Delta vs `core_pca`: OOS bacc `+0.0013`, AUC `+0.0072`,
    proxy WR `+1.01pp`.
- TSFM/Chronos standalone variants do not beat the compact direction stack:
  `tsfm_role` OOS bacc `0.5710`; `chronos_all` OOS bacc `0.4158`.
- Design read: use TSFM/Chronos as additive context on top of the compact
  VSN-LSTM/Patch direction core, not as standalone direction owners.

Omega1 Direction Head contract finalization - 2026-06-02:

- User confirmed `core_plus_tsfm_chronos` as the Omega1 Direction Head input
  contract.
- Contract updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Confirmed Direction Head artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602/core_plus_tsfm_chronos`.
- Confirmed input: 55 features = VSN-LSTM h6 direction, Patch h6 direction,
  exact TSFM role features, Chronos h6 features, and Chronos uncertainty
  features.
- Confirmed outputs:
  `omega1_tsfm_chronos_p_cash`, `omega1_tsfm_chronos_p_long`,
  `omega1_tsfm_chronos_p_short`, `omega1_tsfm_chronos_confidence`,
  `omega1_tsfm_chronos_side_edge`, `omega1_tsfm_chronos_trade_prob`,
  `omega1_tsfm_chronos_action`.
- 2025 OOF:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602/core_plus_tsfm_chronos/training_features_2025_core_plus_tsfm_chronos_omega1_tsfm_chronos_oof_20260602.csv`.
- 2026 score:
  `tmp/causal_regen_20260516/omega1_direction_head_tsfm_chronos_20260602/core_plus_tsfm_chronos/training_features_2026_rebuilt_core_plus_tsfm_chronos_omega1_tsfm_chronos_20260602.csv`.
- Contract restriction: TSFM/Chronos remain additive context inside this
  Direction Head. They are not standalone direction owners.

Omega1 Direction Head raw/context group add-on test - 2026-06-02:

- User requested grouped tests adding raw market OHLCV, volume/spread,
  execution context, funding/session/volatility raw and primary engineered
  features to the confirmed Direction Head.
- Script:
  `scripts/train_omega1_direction_head_raw_context_groups_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_raw_context_groups_20260602`.
- Baseline:
  `core_plus_tsfm_chronos`, 55 features, 2026 OOS bacc `0.5974`,
  OVR AUC `0.7907`, proxy WR `65.79%`, proxy trades `13334`.
- Best add-on:
  `add_volatility_context`, 79 total features, 24 added volatility/context
  features.
  - Added: `log_return`, `volatility_z`, `bb_width`, `bb_width_z`,
    `garman_klass_vol`, `realized_vol_ratio`, `rogers_satchell_vol`,
    `parkinson_vol`, `bb_width_pct_rank_288`, `atr_pct_rank_288`,
    `compression_score`, `compression_release_up`, `compression_release_down`,
    `garch_vol_z`, `jump_flag`, `jump_z`, `evt_tail_flag`, `evt_excess_z`,
    `squeeze_power`, `long_squeeze_risk`, `short_squeeze_risk`,
    `crowding_pressure`, `crowded_long_unwind_risk`,
    `crowded_short_squeeze_risk`.
  - 2026 OOS: bacc `0.6040`, OVR AUC `0.7933`, proxy WR `65.89%`,
    proxy trades `13093`.
  - Delta vs baseline: bacc `+0.0066`, AUC `+0.0026`, proxy WR `+0.10pp`,
    trades `-241`.
- Other groups:
  - `add_session_context`: tiny bacc/AUC gain, neutral WR.
  - `add_liquidity_execution_spread_proxy`: tiny bacc gain but worse AUC/WR.
  - `add_funding_context`, `add_raw_market_ohlcv`, `add_volume_flow`, and
    `add_all_requested_context` degraded OOS.
- Missing literal spread columns:
  `spread`, `bid_ask_spread`. No alias/fallback was added; spread-like testing
  used only existing explicit liquidity/execution proxy columns and records the
  missing names in the report.
- Design read: add volatility context to the Direction Head candidate. Do not
  bulk-add all raw/context features; broad context overfits and hurts 2026 OOS.

Omega1 Direction Head volatility PCA add-on test - 2026-06-02:

- User requested PCA compression for the `add_volatility_context` variant.
- Script:
  `scripts/train_omega1_direction_head_volatility_pca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602`.
- Test design:
  - Keep the confirmed 55-feature `core_plus_tsfm_chronos` Direction Head
    inputs raw.
  - Compress only the 24 volatility/context add-on features with split-local
    PCA.
  - OOF PCA is fit inside each expanding fold only; final PCA is fit on 2025
    only before scoring 2026.
- Raw volatility reference:
  `add_volatility_context`, 79 features, 2026 OOS bacc/AUC/proxy WR/trades
  `0.6040 / 0.7933 / 65.89% / 13093`.
- Best PCA variant:
  `volatility_pca06`, 61 total features, 6 volatility PCA components,
  explained variance `0.7563`.
  - 2025 OOF bacc/AUC/proxy WR:
    `0.5703 / 0.7749 / 64.78%`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6052 / 0.7917 / 66.27% / 13245`.
  - Delta vs `core_plus_tsfm_chronos`:
    bacc `+0.0078`, AUC `+0.0010`, proxy WR `+0.47pp`, trades `-89`.
  - Delta vs raw `add_volatility_context`:
    bacc `+0.0012`, AUC `-0.0016`, proxy WR `+0.38pp`, trades `+152`.
- Other PCA variants:
  - `volatility_pca08`: OOS bacc/AUC/proxy WR/trades
    `0.6044 / 0.7924 / 66.18% / 13217`.
  - `volatility_pca16`: `0.6038 / 0.7915 / 66.03% / 13129`.
  - `volatility_pca04`: `0.6010 / 0.7918 / 66.03% / 13243`.
  - `volatility_pca12`: `0.6003 / 0.7906 / 65.85% / 13117`.
- Design read: `volatility_pca06` is the best compact add-on by OOS bacc and
  proxy WR. Raw volatility remains a useful AUC control, but PCA06 is the
  preferred compact candidate if the Direction Head contract is expanded.

Omega1 Direction Head core-group PCA on volatility_pca06 test - 2026-06-02:

- User requested PCA tests for the internal `core_plus_tsfm_chronos` groups
  after fixing `volatility_pca06` as the volatility add-on.
- Script:
  `scripts/train_omega1_direction_head_core_group_pca_on_volpca_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_core_group_pca_on_volpca_20260602`.
- Baseline:
  `volatility_pca06`, 61 features, 2026 OOS bacc/AUC/proxy WR/trades
  `0.6052 / 0.7917 / 66.27% / 13245`.
- Tested PCA replacements:
  `vsnlstm`, `patch`, `tsfm_role`, `chronos_h6`, `chronos_uncertainty`,
  `vsnlstm+patch`, `tsfm+chronos`, and all-core light/mid grouped PCA.
- Best PCA replacement:
  `pca_tsfm06`, 43 features, replaces 24 TSFM role features with 6 PCA
  components while keeping other core groups raw and volatility as PCA06.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6046 / 0.7900 / 66.18% / 13192`.
  - Delta vs `volatility_pca06`:
    bacc `-0.0006`, AUC `-0.0017`, proxy WR `-0.09pp`, trades `-53`.
- Only narrow secondary gains:
  - `pca_chronos_h603` slightly improved AUC vs `volatility_pca06`
    (`+0.0003`) but reduced bacc and proxy WR.
  - `pca_chronos_unc06` improved AUC (`+0.0005`) and proxy WR (`+0.05pp`)
    but reduced bacc (`-0.0012`).
  - `pca_direction_core06` improved proxy WR (`+0.09pp`) but reduced bacc and
    AUC.
- Design read: do not PCA-compress the confirmed core groups for the primary
  Direction Head. Their raw probability/edge semantics carry useful signal.
  Keep PCA only on the volatility context add-on unless a later objective
  explicitly prioritizes compactness over bacc/AUC.

Omega1 Direction Head final contract update - 2026-06-02:

- User confirmed `core_plus_tsfm_chronos + volatility_pca06` as the fixed
  Omega1 Direction Head feature contract.
- Contract updated:
  `docs/model_contracts/omega1_processed_feature_contract_20260531.md`.
- Active Direction Head script:
  `scripts/train_omega1_direction_head_volatility_pca_20260602.py`.
- Active artifact:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06`.
- Active model:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/volatility_pca06_omega1_direction_volpca.cbm`.
- Active contract object:
  `tmp/causal_regen_20260516/omega1_direction_head_volatility_pca_20260602/volatility_pca06/volatility_pca06_omega1_direction_volpca_contract.joblib`.
- Active input: 61 features = 55 raw `core_plus_tsfm_chronos` features plus
  `pca_volatility_01` ... `pca_volatility_06`.
- Output prefix:
  `omega1_dir_volpca_*`.
- 2026 OOS reference:
  BAcc `0.6052`, OVR AUC `0.7917`, proxy WR `66.27%`, proxy trades `13245`.
- Contract note: core groups stay raw. Only the volatility context group is
  compressed. No alias/fallback or core-group PCA replacement is active.

Omega1 Regime3 expert-internal Direction Head test - 2026-06-02:

- User requested moving the confirmed Direction Head feature contract inside
  each Regime3 expert instead of using one global Direction Head before the
  expert layer.
- Hard-split script:
  `scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py`.
- Soft-weight script:
  `scripts/train_omega1_regime3_soft_expert_direction_head_volpca_20260602.py`.
- Common input contract per expert:
  61 features = raw `core_plus_tsfm_chronos` 55 + `volatility_pca06`.
- Router:
  Regime3 current context
  (`regime3_current_sensitive_wide24_bull_prob`,
  `regime3_current_sensitive_wide24_bear_prob`,
  `regime3_current_sensitive_wide24_chop_prob`).
- Hard-split result:
  `tmp/causal_regen_20260516/omega1_regime3_expert_direction_head_volpca_20260602`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5611 / 0.7480 / 60.90% / 13058`.
  - Delta vs global `volatility_pca06`:
    bacc `-0.0441`, AUC `-0.0436`, proxy WR `-5.37pp`.
- Best soft-weight result:
  `soft_floor_0p20` in
  `tmp/causal_regen_20260516/omega1_regime3_soft_expert_direction_head_volpca_20260602`.
  - 2026 OOS bacc/AUC/proxy WR/trades:
    `0.6017 / 0.7920 / 66.11% / 13413`.
  - Delta vs global `volatility_pca06`:
    bacc `-0.0035`, AUC `+0.0003`, proxy WR `-0.16pp`, trades `+168`.
- Design read:
  expert-internal Direction Heads are viable as a research branch, but they do
  not beat the fixed global Direction Head. Hard row partitioning is especially
  weak; soft regime-probability weighting is much closer but still lower on
  OOS bacc and proxy WR. Keep global `omega1_dir_volpca_*` active for now.

Omega1 Regime3 expert-internal Direction + Quality Head test - 2026-06-02:

- User requested the full expert-internal design: feature processing -> Regime3
  Current Router -> bull/bear/chop expert -> expert-local Direction Head ->
  expert-local Quality Head.
- Script:
  `scripts/train_omega1_regime3_routed_expert_direction_quality_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602`.
- Contract details:
  - Direction Head target: `zigzag_action`.
  - Quality Head target: `zigzag_action`.
  - Quality is a 3-class second-opinion/calibration head, not an SL/TP
    compatibility label and not a binary quality label.
  - Global `omega1_dir_volpca_*` outputs are not used as expert inputs.
  - Per-expert base input remains the fixed 61-feature
    `core_plus_tsfm_chronos + volatility_pca06` contract.
  - Quality input adds OOF Direction probabilities/action plus Regime3 router
    confidence and margin.
- Best DQ variant by filtered OOS bacc:
  `soft_floor_0p00`, selected 2025 OOF quality threshold `0.45`.
  - Direction-only 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5983 / 0.7910 / 65.97% / 13463`.
  - Quality-filtered 2026 OOS bacc/AUC/proxy WR/trades:
    `0.5832 / 0.7220 / 66.44% / 12276`.
  - Delta vs active global `volatility_pca06`:
    filtered bacc `-0.0220`, AUC `-0.0697`, proxy WR `+0.17pp`,
    trades `-969`.
- Other variants:
  - `soft_floor_0p20`: filtered OOS
    `0.5830 / 0.7225 / 66.23% / 12285`.
  - `hard_floor_0p00`: filtered OOS
    `0.5318 / 0.6686 / 60.39% / 11131`.
- Decision:
  do not promote. Quality filtering raises proxy WR only marginally while
  materially degrading bacc/AUC and reducing trade count. Hard expert
  partition remains weak. Keep the active global `omega1_dir_volpca_*`
  Direction Head and record expert-DQ as a research branch only.

Omega1 Regime3 expert-internal DQ risk replay - 2026-06-02:

- User requested Cost1/2/3 replay after the expert-DQ classification result.
- Script:
  `scripts/eval_omega1_regime3_expertdq_risk_replay_20260602.py`.
- Artifact:
  `tmp/causal_regen_20260516/omega1_regime3_expertdq_risk_replay_20260602`.
- Replay setup:
  - Use expert-DQ `final_action` as action owner.
  - Apply the current Omega1 ZigZag risk template:
    `balanced_rr19`, notional `0.45`, leverage `2.0`, TP `0.026`,
    SL `0.014`, max-hold `72`, cooldown `6`.
  - Apply active expert scales: bull `0.75`, bear `0.90`, chop `0.90`.
  - Compare on exact timestamp intersection with the active decision set.
- Active common-window OOS Cost3:
  PnL `+4.51%`, MDD `-8.69%`, trades `211`, WR `46.92%`.
- Best OOS expert-DQ replay:
  `soft_floor_0p10`, OOS Cost3 PnL `+8.29%`, MDD `-7.86%`, trades `211`,
  WR `54.03%`, delta vs active common-window Cost3 `+3.77pp`.
- Validation check:
  `soft_floor_0p10` validation Cost3 PnL `-2.19%`, MDD `-18.46%`,
  trades `333`, WR `48.05%`, while active validation Cost3 is `+41.34%`,
  MDD `-5.61%`, trades `339`, WR `51.03%`.
- Decision:
  do not promote. The OOS-only lift is not acceptable because validation
  collapses badly. Treat this as an OOS diagnostic showing possible 2026
  fit, not a stable active candidate.

## Omega2 Exit-Head-Free Cash Sleeve Memory - 2026-06-09

- User requested recording the current exit-head-free model as Omega2.
- New contract:
  `docs/model_contracts/omega2_exit_head_free_cash_sleeve_20260609_contract.md`.
- Omega2 model id:
  `omega2_label_atr1_h24_hgb_cash_sleeve_thr055`.
- Status:
  `research_baseline_candidate_not_live_promoted`.
- Parent baseline:
  `omega1_2_1_aggressive_compensated_scale200_cap090`.
- Multiseed artifact:
  `tmp/causal_regen_20260516/omega1_2_1_cash_alpha43_multiseed_20260608`.
- Exit feature ablation artifact:
  `tmp/causal_regen_20260516/omega1_2_1_cash_alpha43_exit_feature_ablation_20260609`.
- Architecture:
  preserve Omega1.2.1 aggressive primary; call HGB cash sleeve only when
  primary action is `CASH`.
- Cash sleeve label/model:
  `label_atr1_h24`, triple barrier `atr_mult=1.0`, `max_hold=24`,
  `min_barrier=0.0035`, `HistGradientBoostingClassifier`, confidence
  threshold `0.55`.
- Risk:
  fallback TP `0.026`, SL `0.014`, notional `0.30`, leverage metadata `2.0`,
  max hold `192`, Cost3 accounting.
- Metrics, 12 seeds:
  validation mean/median PnL `+109.91% / +110.41%`; OOS mean/median PnL
  `+95.61% / +99.92%`; OOS range `+82.25%` to `+104.50%`; OOS worst MDD
  `-8.33%`; OOS mean WR `61.84%`; mean trades `45.17`.
- Exit Head policy:
  parent artifact may still contain an auxiliary/shared Exit Head from
  TabM training, but Omega2 must not feed `exit_head_*` into the cash sleeve,
  must not use Exit Head as runtime exit/veto/risk selector, and must not add
  aliases or compatibility fallbacks for Exit Head columns.
- Direct ablation:
  adding Exit Head entry-risk features reduced OOS mean PnL from `+95.61%`
  to `+92.00%`, reduced OOS median PnL from `+99.92%` to `+92.67%`, and
  worsened OOS minimum from `+82.25%` to `+70.70%`.
- Forbidden active Omega2 cash-sleeve inputs:
  `clean_regime4_*`, `regime4_pred_*`, `teacher_*`, `tp_sl_action_score`,
  and `exit_head_*`.
- Docs Manager instruction:
  treat Omega2 as the current documented research baseline candidate, not a
  live promotion. Do not update `docs/active_live/` until the user explicitly
  requests live promotion and Red Team passes runtime-native parity.

## Omega2.1 HGB 12-Seed Cash Sleeve Memory - 2026-06-09

- New frozen research candidate:
  `omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055`.
- Contract:
  `docs/model_contracts/omega2_1_hgb_12seed_cash_sleeve_20260609_contract.md`.
- Manifest:
  `data/ensemble/supervised/omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055/candidate_manifest.json`.
- Frozen bundle:
  `data/ensemble/supervised/omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055/omega2_1_hgb_12seed_cash_sleeve.joblib`.
- Runtime scorer:
  `trading_bot_modules/omega2_1_cash_sleeve.py`.
- Freeze/verify script:
  `scripts/freeze_omega2_1_hgb_12seed_cash_sleeve_20260609.py`.
- Verification report:
  `tmp/causal_regen_20260516/omega2_1_cash_sleeve_freeze_verify_20260609/report.json`.
- Architecture:
  preserve Omega1.2.1 aggressive primary; call the 12-seed HGB ensemble cash
  sleeve only when primary action is `CASH`.
- Feature contract:
  exact 42 Omega-only features, exact order from manifest/joblib, fail-fast on
  missing/non-finite/forbidden columns.
- Forbidden active Omega2.1 cash-sleeve inputs:
  `clean_regime4_*`, `regime4_pred_*`, `teacher_*`, `exit_head_*`,
  and `tp_sl_action_score`.
- Label/model:
  `label_atr1_h24`, triple barrier `atr_mult=1.0`, `max_hold=24`,
  `min_barrier=0.0035`, 12 `HistGradientBoostingClassifier` seeds, averaged
  class probabilities, threshold `0.55`.
- Risk:
  fallback TP `0.026`, SL `0.014`, notional `0.30`, leverage metadata `2.0`,
  max hold `192`, Cost3 accounting.
- Selection evidence:
  validation OOF PnL `+111.959707%`; OOS full-train PnL `+102.611483%`,
  MDD `-8.108171%`, WR `60.975610%`, trades `41`.
- Frozen parity:
  OOS PnL `+102.611482864%`, MDD `-8.108170709%`, WR `60.975610%`,
  trades `41`, fallback entries `23`, primary takeovers `12`.
- Status:
  `frozen_research_candidate_not_live_promoted`. Current live baseline remains
  `omega1_2_1_aggressive_compensated_scale200_cap090` until explicit live
  promotion is requested.

## Omega2.1 HGB 12-Seed Cash Sleeve Red-Team Update - 2026-06-14

- Audited model:
  `omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055`.
- Audit doc:
  `docs/audits/omega2_1_hgb_12seed_cash_sleeve_redteam_20260614.md`.
- Audit report:
  `tmp/causal_regen_20260516/omega2_1_hgb_12seed_redteam_audit_20260614/report.json`.
- Updated status:
  `deprecated_historical_reference_only_accounting_invalid_true_leverage`.
- What passed:
  manifest/bundle feature contract matched; 42 frozen features contain no
  forbidden columns; runtime adapter is fail-fast.
- What failed:
  legacy replay treated `notional_exposure` as effective exposure while storing
  `leverage=2.0`; current Omega accounting requires
  `effective_exposure = notional * leverage`.
- Legacy OOS:
  `+102.611483%`, MDD `-8.108171%`, WR `60.975610%`, trades `41`.
- Corrected true-leverage OOS:
  `+33.877901%`, MDD `-23.976364%`, WR `43.410853%`, trades `129`.
- Instruction:
  do not live-promote this artifact and do not use the legacy `+102.61%` result
  as promotion evidence. Rebuild under true-leverage accounting if reusing this
  architecture.

## Omega1.2.1 TP Runner Clean Repair Memory - 2026-06-13

- Deprecated model:
  `omega1_2_1_tp_runner_only_baseline_20260612` remains blocked for active,
  candidate, clean-OOS comparison, and promotion evidence.
- Clean repair model:
  `omega1_2_1_tp_runner_clean_repair_20260613`.
- Contract:
  `docs/model_contracts/omega1_2_1_tp_runner_clean_repair_20260613_contract.md`.
- Manifest:
  `data/ensemble/supervised/omega1_2_1_tp_runner_clean_repair_20260613/baseline_manifest.json`.
- Repair script:
  `scripts/repair_omega1_2_1_tp_runner_clean_baseline_20260613.py`.
- Repair artifact:
  `tmp/causal_regen_20260516/omega1_2_1_tp_runner_clean_repair_20260613/report.json`.
- Fixed blockers:
  OOS is no longer used for TP-runner config selection, active runtime no
  longer depends on the OOS-mined `tp_runner_meta_selector_20260610` bundle,
  and repair accounting uses next-open taker entry plus intrabar high/low
  price-barrier exits.
- Runtime TP-runner contract:
  deterministic `mom3_quality`, `quality_min=0.70`, `momentum_min=0.0`,
  `extend_mult=1.75`, `floor_frac=0.75`, `max_extensions=2`.
- Clean repair metrics:
  validation `+160.22%` PnL, MDD `-27.64%`, WR `59.46%`, trades `37`;
  OOS `+85.70%` PnL, MDD `-15.64%`, WR `66.67%`, trades `18`.
- Caveat:
  clean no-runner accounting OOS was `+120.07%`, so the repaired TP-runner is
  a shadow-required candidate, not proven superior OOS alpha.

## Omega1.2.2 TP Runner Cash Sleeve Memory - 2026-06-15

- Current next Omega version:
  `omega1_2_2_tp_runner_cash_sleeve_20260615`.
- Base model:
  `omega1_2_1_tp_runner_clean_repair_20260613`.
- Contract:
  `docs/model_contracts/omega1_2_2_tp_runner_cash_sleeve_20260615_contract.md`.
- Manifest:
  `data/ensemble/supervised/omega1_2_2_tp_runner_cash_sleeve_20260615/candidate_manifest.json`.
- Training/eval script:
  `scripts/train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615.py`.
- Report:
  `tmp/causal_regen_20260516/omega1_2_2_tp_runner_cash_sleeve_20260615/report.json`.
- Structure:
  preserve Omega1.2.1 TP-runner clean-repair primary; train and execute the HGB
  sleeve only on parent-CASH rows; close sleeve by `fallback_primary_takeover`
  when primary becomes active.
- Red Team policy:
  PnL and OOS lift are diagnostics only. FAIL is limited to logical defects,
  data/feature contract violations, forbidden feature leakage, missing train/test
  CASH rows, or failed sleeve candidate generation.
- Red Team result:
  `redteam_pass=true`, `redteam_blockers=[]`,
  status `redteam_pass_shadow_candidate_not_live_wired`.
- Metrics:
  validation `+172.46%` PnL, MDD `-26.54%`, WR `60.87%`, trades `46`;
  OOS `+86.26%` PnL, MDD `-15.64%`, WR `60.00%`, trades `40`.
- Live wiring:
  not connected to `trading_bot.py`; live wiring requires a separate runtime
  implementation and parity check.

## Omega1.2.3 EV-HGB Cash Sleeve Memory - 2026-06-15

- Current next Omega version:
  `omega1_2_3_ev_hgb_cash_sleeve_20260615`.
- Supersedes:
  `omega1_2_2_tp_runner_cash_sleeve_20260615`.
- Base model:
  `omega1_2_1_tp_runner_clean_repair_20260613`.
- Contract:
  `docs/model_contracts/omega1_2_3_ev_hgb_cash_sleeve_20260615_contract.md`.
- Manifest:
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/candidate_manifest.json`.
- Upgrade sweep:
  `scripts/train_eval_omega1_2_3_cash_sleeve_upgrade_20260615.py`.
- Walk-forward:
  `scripts/walkforward_omega1_2_3_ev_hgb_cash_sleeve_20260615.py`.
- Robust selected config:
  HGB long/short EV regressors, risk `base_tp026_sl014_n0405_h192`,
  `min_edge=0.002`, `ev_min=0.002`.
- Rejected point-best config:
  `ev_min=0.004` had stronger point OOS fallback-only PnL but improved only
  `2/4` monthly walk-forward folds, so do not promote it as the robust default.
- Full OOS robust config:
  combo PnL `+91.89%`, MDD `-15.64%`, WR `61.76%`, trades `34`;
  fallback-only PnL `+3.33%`, trades `16`, WR `56.25%`, PF `1.33`.
- Monthly walk-forward robust config:
  selected `ev_min=0.002` improved `3/4` folds, total combo delta `+6.10p`,
  total fallback-only PnL points `+4.06p`.
- Live wiring:
  connected to `trading_bot.py` through
  `trading_bot_modules/omega1_2_3_cash_sleeve.py` and the Omega1.2.1 CASH
  branch; live bundle is
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib`.

## Omega4.6 Conditional Swing Baseline Memory - 2026-06-30

- Current Omega research baseline:
  `omega4_6_plus_t12_nohold_risk1_20260630`.
- Contract:
  `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`.
- Manifest:
  `data/ensemble/supervised/omega4_6_plus_t12_nohold_risk1_20260630/candidate_manifest.json`.
- Runtime contract:
  `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json`.
- Report:
  `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`.
- Artifact audit:
  `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/omega_artifact_integrity_audit_20260630.json`.
- Structure:
  `h48qual > zig075` conditional swing/runner baseline rebuilt from exact
  threshold parent prediction artifacts.
- Metrics:
  validation `+117.17%` PnL, MDD `-17.43%`, WR `51.72%`, trades `29`;
  OOS readout `+67.85%` PnL, MDD `-13.28%`, WR `53.85%`, trades `13`.
- Baseline rule:
  this is not a day-trading/full live PASS. Max-hold and PnL target are
  excluded from mandatory gates; non-excluded gates pass and artifact integrity
  audit is `promotion_pass=true`.
- Live boundary:
  this is not wired to `trading_bot.py`. The live-wired Omega baseline remains
  `omega1_2_3_ev_hgb_cash_sleeve_20260615`.
- Upgrade priority:
  preserve no-hold swing alpha, reduce tail hold time through exit/partial
  profit/trailing-giveback policy, use validation-only selection with blind OOS
  readout, and preserve exact-threshold parent prediction artifacts.

## Omega1.2.1 True Leverage Baseline Audit Memory - 2026-06-13

- Audited model:
  `omega1_2_1_true_leverage_price_barrier_scale200_cap090`.
- Audit doc:
  `docs/audits/omega1_2_1_true_leverage_baseline_redteam_20260613.md`.
- Audit report:
  `tmp/causal_regen_20260516/omega1_2_1_true_leverage_baseline_redteam_audit_20260613/report.json`.
- Verdict:
  `audited_research_candidate_not_clean_untouched_oos`.
- Original reported preserve-price-barrier replay:
  validation `+276.67%`, MDD `-20.34%`, WR `63.64%`, trades `33`;
  OOS `+186.43%`, MDD `-15.60%`, WR `72.22%`, trades `18`.
- Clean intrabar/taker replay:
  validation `+49.16%`, MDD `-33.16%`, WR `46.67%`, trades `45`;
  OOS `+120.07%`, MDD `-15.64%`, WR `65.00%`, trades `20`.
- Feature audit:
  direct forbidden-feature leak into `decision` or runner `state` was not found,
  but source frames still contain legacy forbidden columns. Do not treat source
  frame presence as active model consumption, but remove such columns from new
  active research paths where possible.
- Intrabar audit:
  original validation ledger had `23/33` earlier high/low barrier touches and
  original OOS ledger had `11/18`; original `+186.43%` must not be cited as
  clean untouched OOS.
