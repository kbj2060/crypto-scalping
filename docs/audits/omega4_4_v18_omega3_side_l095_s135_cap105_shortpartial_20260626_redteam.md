# Omega4.4 v18 Omega3 Exposure Transfer Red-Team Audit

- Verdict: `REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED`
- Red-team reproduction pass: `True`
- Clean-OOS promotion pass: `False`
- Model id: `omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626`
- Variant: `side_l0p95_s1p35_cap1p05_shortpartial`
- Audit JSON: `tmp/causal_regen_20260516/omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626/redteam_report.json`

## Metrics

| Split | PnL | MDD | Trades | WR | Avg notional | Overlay hits | Log-risk utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_validation | 35.8538 | -14.1696 | 41 | 0.5366 | 0.7595 | 0 | 0.169373 |
| candidate_validation | 49.9022 | -13.8272 | 41 | 0.5610 | 0.8853 | 5 | 0.218622 |
| baseline_oos_readout | 43.2312 | -10.7585 | 34 | 0.6765 | 0.7931 | 0 | 0.290538 |
| candidate_oos_readout | 62.7677 | -10.4708 | 35 | 0.7143 | 0.9347 | 4 | 0.386488 |

## Blockers

- No hard contract or reproduction blockers.

## Promotion Blockers

- WARNING `clean_oos_promotion_blocker`: candidate was chosen after OOS diagnostic comparison; fresh holdout/walk-forward is required before clean-OOS promotion

## Key Passes

- PASS `artifact_presence`: missing=[]
- PASS `ledger_presence`: validation=True oos=True
- PASS `base_model_id`: base_model_id=omega4_4_v18_baseline_20260624
- PASS `source_model_id`: source_model_id=omega3_aggressive_compensated_scale200_cap090_20260618
- PASS `base_redteam_full_pass`: base_verdict=REDTEAM_PASS_FULL_PROMOTABLE pass=True
- PASS `candidate_manifest_model_id`: model_id=omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626
- PASS `promotion_manifest_model_id`: model_id=omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626
- PASS `runtime_contract_model_id`: model_id=omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626
- PASS `runtime_fail_fast_required`: fail_fast_required=True
- PASS `base_runtime_not_mutated`: base_runtime_model_id=omega4_4_v18_baseline_20260624
- PASS `base_manifest_redteam_pass`: base_manifest_redteam_pass=True
- PASS `fine_sweep_variant_found`: variant=side_l0p95_s1p35_cap1p05_shortpartial
- PASS `risk_remap_enabled`: risk_remap={'enabled': True, 'source_idea': 'borrow Omega3 aggressive exposure while preserving Omega4.4 risk score ordering', 'mode': 'side_scaled', 'scale': 1.0, 'cap_notional': 1.05, 'fixed_notional': 0.0, 'long_scale': 0.95, 'short_scale': 1.35, 'leverage': 2.0, 'notional_math': 'notional = margin_fraction * leverage', 'side_scaled_formula': 'notional = min(base_margin_fraction * base_leverage * side_scale, cap_notional)', 'margin_formula': 'margin_fraction = notional / leverage', 'sltp_contract': 'ATR safety TP/SL remains a price-move barrier before PnL conversion; leverage is not multiplied twice.', 'runtime_must_fail_on_missing_contract': True}
- PASS `risk_remap_leverage_fixed_2`: leverage=2.0
- PASS `risk_remap_notional_contract`: notional_math=notional = margin_fraction * leverage
- PASS `sltp_no_double_leverage`: sltp_contract=ATR safety TP/SL remains a price-move barrier before PnL conversion; leverage is not multiplied twice.
- PASS `overlay_enabled`: value=True expected=True
- PASS `overlay_mode`: value='short_aged_profit_partial_deleverage' expected='short_aged_profit_partial_deleverage'
- PASS `overlay_side_value`: value=-1 expected=-1
- PASS `overlay_cap_bars`: value=1152 expected=1152
- PASS `overlay_min_unrealized_price_move`: value=0.035 expected=0.035
- PASS `overlay_partial_fraction`: value=0.5 expected=0.5
- PASS `overlay_fires_once_per_position`: value=True expected=True
- PASS `manifest_validation_pnl_matches_fine_sweep`: manifest=49.90218803641435 fine=49.90218803641435
- PASS `manifest_validation_mdd_matches_fine_sweep`: manifest=-13.827224476407384 fine=-13.827224476407384
- PASS `manifest_validation_trades_matches_fine_sweep`: manifest=41 fine=41
- PASS `manifest_validation_wr_matches_fine_sweep`: manifest=0.5609756097560976 fine=0.5609756097560976
- PASS `manifest_validation_avg_notional_matches_fine_sweep`: manifest=0.8852855978457046 fine=0.8852855978457046
- PASS `manifest_validation_avg_margin_fraction_matches_fine_sweep`: manifest=0.4426427989228523 fine=0.4426427989228523
- PASS `manifest_validation_avg_leverage_matches_fine_sweep`: manifest=2.0 fine=2.0
- PASS `manifest_validation_overlay_hits_matches_fine_sweep`: manifest=5 fine=5
- PASS `manifest_validation_log_risk_utility_matches_fine_sweep`: manifest=0.21862174559310354 fine=0.21862174559310354
- PASS `manifest_oos_pnl_matches_fine_sweep`: manifest=62.767678274328716 fine=62.767678274328716
- PASS `manifest_oos_mdd_matches_fine_sweep`: manifest=-10.470835745611339 fine=-10.470835745611339
- PASS `manifest_oos_trades_matches_fine_sweep`: manifest=35 fine=35
- PASS `manifest_oos_wr_matches_fine_sweep`: manifest=0.7142857142857143 fine=0.7142857142857143
- PASS `manifest_oos_avg_notional_matches_fine_sweep`: manifest=0.9346747848244374 fine=0.9346747848244374
- PASS `manifest_oos_avg_margin_fraction_matches_fine_sweep`: manifest=0.4673373924122187 fine=0.4673373924122187
- PASS `manifest_oos_avg_leverage_matches_fine_sweep`: manifest=2.0 fine=2.0
- PASS `manifest_oos_overlay_hits_matches_fine_sweep`: manifest=4 fine=4
- PASS `manifest_oos_log_risk_utility_matches_fine_sweep`: manifest=0.3864881994069224 fine=0.3864881994069224
- PASS `validation_pnl_improves`: candidate=49.90218803641435 baseline=35.853831530265
- PASS `oos_pnl_improves_diagnostic`: candidate=62.767678274328716 baseline=43.2312455386217
- PASS `validation_mdd_improves_for_strict_promotion`: candidate=-13.827224476407384 baseline=-14.16964911682127
- PASS `oos_mdd_improves_diagnostic`: candidate=-10.470835745611339 baseline=-10.758485734455137
- PASS `ledger_notional_math_exact`: max_error=0.0
- PASS `validation_partial_done_matches_hits`: partial_done=5 hits=5
- PASS `oos_partial_done_matches_hits`: partial_done=4 hits=4
- PASS `selection_oos_informed_declared`: selection={'policy': 'refined strict validation/OOS diagnostic winner from Omega3 exposure transfer sweep', 'selection_oos_informed': True, 'clean_oos_holdout_available_for_this_candidate': False, 'note': 'This candidate may pass contract/reproduction audit but cannot claim clean OOS promotion without a fresh holdout or walk-forward confirmation.'}
