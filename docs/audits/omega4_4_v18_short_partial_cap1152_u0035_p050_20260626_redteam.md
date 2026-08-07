# Omega4.4 v18 Short Partial Lifecycle Overlay Red-Team Audit

- Verdict: `REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED`
- Red-team reproduction pass: `True`
- Clean-OOS promotion pass: `False`
- Model id: `omega4_4_v18_short_partial_cap1152_u0035_p050_20260626`
- Base model: `omega4_4_v18_baseline_20260624`
- Audit JSON: `tmp/causal_regen_20260516/omega4_4_v18_short_partial_cap1152_u0035_p050_20260626/redteam_report.json`

## Metrics

| Split | PnL | MDD | Trades | WR | Overlay hits | Log-risk utility |
|---|---:|---:|---:|---:|---:|---:|
| baseline_validation | 35.8538 | -14.1696 | 41 | 0.5366 | 0 | 0.169373 |
| candidate_validation | 35.9479 | -12.0643 | 41 | 0.5366 | 5 | 0.170065 |
| baseline_oos_readout | 43.2312 | -10.7585 | 34 | 0.6765 | 0 | 0.290538 |
| candidate_oos_readout | 48.5644 | -8.1922 | 35 | 0.7143 | 4 | 0.327097 |

## Blockers

- No hard contract or reproduction blockers.

## Promotion Blockers

- WARNING `clean_oos_promotion_blocker`: candidate was chosen after OOS diagnostic comparison; fresh holdout/walk-forward is required before clean-OOS promotion

## Key Passes

- PASS `artifact_presence`: missing=[]
- PASS `base_model_id`: base_model_id=omega4_4_v18_baseline_20260624
- PASS `base_redteam_full_pass`: base_verdict=REDTEAM_PASS_FULL_PROMOTABLE pass=True
- PASS `candidate_manifest_model_id`: model_id=omega4_4_v18_short_partial_cap1152_u0035_p050_20260626
- PASS `promotion_manifest_model_id`: model_id=omega4_4_v18_short_partial_cap1152_u0035_p050_20260626
- PASS `runtime_contract_model_id`: model_id=omega4_4_v18_short_partial_cap1152_u0035_p050_20260626
- PASS `runtime_fail_fast_required`: fail_fast_required=True
- PASS `base_runtime_not_mutated`: base_runtime_model_id=omega4_4_v18_baseline_20260624
- PASS `base_manifest_redteam_pass`: base_manifest_redteam_pass=True
- PASS `overlay_enabled`: value=True expected=True
- PASS `overlay_mode`: value='short_aged_profit_partial_deleverage' expected='short_aged_profit_partial_deleverage'
- PASS `overlay_side`: value='short' expected='short'
- PASS `overlay_side_value`: value=-1 expected=-1
- PASS `overlay_cap_bars`: value=1152 expected=1152
- PASS `overlay_min_unrealized_price_move`: value=0.035 expected=0.035
- PASS `overlay_partial_fraction`: value=0.5 expected=0.5
- PASS `overlay_fires_once_per_position`: value=True expected=True
- PASS `overlay_result_variant`: value='short_partial_cap1152_u0.035_p0.50' expected='short_partial_cap1152_u0.035_p0.50'
- PASS `overlay_result_mode`: value='partial_deleverage' expected='partial_deleverage'
- PASS `overlay_result_side`: value=-1 expected=-1
- PASS `overlay_result_cap_bars`: value=1152 expected=1152
- PASS `overlay_result_min_unreal`: value=0.035 expected=0.035
- PASS `overlay_result_partial_fraction`: value=0.5 expected=0.5
- PASS `manifest_validation_pnl_matches_report`: manifest=35.94793621231054 report=35.94793621231054
- PASS `manifest_validation_mdd_matches_report`: manifest=-12.064342008727124 report=-12.064342008727124
- PASS `manifest_validation_trades_matches_report`: manifest=41 report=41
- PASS `manifest_validation_wr_matches_report`: manifest=0.5365853658536586 report=0.5365853658536586
- PASS `manifest_validation_overlay_hits_matches_report`: manifest=5 report=5
- PASS `manifest_validation_log_risk_utility_matches_report`: manifest=0.17006530289584862 report=0.17006530289584862
- PASS `manifest_oos_pnl_matches_report`: manifest=48.56443287813712 report=48.56443287813712
- PASS `manifest_oos_mdd_matches_report`: manifest=-8.192179045485325 report=-8.192179045485325
- PASS `manifest_oos_trades_matches_report`: manifest=35 report=35
- PASS `manifest_oos_wr_matches_report`: manifest=0.7142857142857143 report=0.7142857142857143
- PASS `manifest_oos_overlay_hits_matches_report`: manifest=4 report=4
- PASS `manifest_oos_log_risk_utility_matches_report`: manifest=0.3270968211621471 report=0.3270968211621471
- PASS `validation_pnl_improves`: candidate=35.94793621231054 baseline=35.853831530265
- PASS `validation_mdd_improves`: candidate=-12.064342008727124 baseline=-14.16964911682127
- PASS `oos_pnl_improves_diagnostic`: candidate=48.56443287813712 baseline=43.2312455386217
- PASS `oos_mdd_improves_diagnostic`: candidate=-8.192179045485325 baseline=-10.758485734455137
- PASS `validation_overlay_hits_positive`: hits=5
- PASS `oos_overlay_hits_positive`: hits=4
- PASS `ledger_notional_math_exact`: max_error=0.0
- PASS `validation_partial_done_matches_hits`: partial_done=5 hits=5
- PASS `oos_partial_done_matches_hits`: partial_done=4 hits=4
- PASS `selection_oos_informed_declared`: selection={'policy': 'user-approved balanced validation/OOS diagnostic candidate from overlay sweep', 'selection_oos_informed': True, 'clean_oos_holdout_available_for_this_candidate': False, 'note': 'This candidate may pass contract/reproduction audit but cannot claim clean OOS promotion without a fresh holdout or walk-forward confirmation.'}
