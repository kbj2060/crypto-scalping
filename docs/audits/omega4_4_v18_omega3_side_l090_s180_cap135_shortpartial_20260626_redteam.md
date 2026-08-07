# Omega4.4 v18 Omega3 Exposure Transfer Red-Team Audit

- Verdict: `REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED`
- Red-team reproduction pass: `True`
- Clean-OOS promotion pass: `False`
- Model id: `omega4_4_v18_omega3_side_l090_s180_cap135_shortpartial_20260626`
- Variant: `side_l0p90_s1p80_cap1p35_shortpartial`
- Audit JSON: `tmp/causal_regen_20260516/omega4_4_v18_omega3_side_l090_s180_cap135_shortpartial_20260626/redteam_report.json`

## Metrics

| Split | PnL | MDD | Trades | WR | Avg notional | Overlay hits | Log-risk utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_validation | 35.8538 | -14.1696 | 41 | 0.5366 | 0.7595 | 0 | 0.169373 |
| candidate_validation | 62.7919 | -17.6393 | 41 | 0.5610 | 1.0754 | 5 | 0.200191 |
| baseline_oos_readout | 43.2312 | -10.7585 | 34 | 0.6765 | 0.7931 | 0 | 0.290538 |
| candidate_oos_readout | 71.9975 | -13.5714 | 29 | 0.6897 | 1.2075 | 4 | 0.384867 |

## Blockers

- No hard contract or reproduction blockers.

## Promotion Blockers

- WARNING `validation_mdd_improves_for_strict_promotion`: candidate=-17.639305433768914 baseline=-14.16964911682127
- WARNING `oos_mdd_improves_diagnostic`: candidate=-13.571350217596335 baseline=-10.758485734455137
- WARNING `clean_oos_promotion_blocker`: candidate was chosen after OOS diagnostic comparison; fresh holdout/walk-forward is required before clean-OOS promotion

## Key Passes

- PASS `artifact_presence`: missing=[]
- PASS `ledger_presence`: validation=True oos=True
- PASS `base_model_id`: base_model_id=omega4_4_v18_baseline_20260624
- PASS `source_model_id`: source_model_id=omega3_aggressive_compensated_scale200_cap090_20260618
- PASS `base_redteam_full_pass`: base_verdict=REDTEAM_PASS_FULL_PROMOTABLE pass=True
- PASS `candidate_manifest_model_id`: model_id=omega4_4_v18_omega3_side_l090_s180_cap135_shortpartial_20260626
- PASS `promotion_manifest_model_id`: model_id=omega4_4_v18_omega3_side_l090_s180_cap135_shortpartial_20260626
- PASS `runtime_contract_model_id`: model_id=omega4_4_v18_omega3_side_l090_s180_cap135_shortpartial_20260626
- PASS `runtime_fail_fast_required`: fail_fast_required=True
- PASS `base_runtime_not_mutated`: base_runtime_model_id=omega4_4_v18_baseline_20260624
- PASS `base_manifest_redteam_pass`: base_manifest_redteam_pass=True
- PASS `fine_sweep_variant_found`: variant=side_l0p90_s1p80_cap1p35_shortpartial
- PASS `risk_remap_enabled`: risk_remap={'enabled': True, 'source_idea': 'borrow Omega3 aggressive exposure while preserving Omega4.4 risk score ordering', 'mode': 'side_scaled', 'scale': 1.0, 'cap_notional': 1.35, 'fixed_notional': 0.0, 'long_scale': 0.9, 'short_scale': 1.8, 'leverage': 2.0, 'notional_math': 'notional = margin_fraction * leverage', 'side_scaled_formula': 'notional = min(base_margin_fraction * base_leverage * side_scale, cap_notional)', 'margin_formula': 'margin_fraction = notional / leverage', 'sltp_contract': 'ATR safety TP/SL remains a price-move barrier before PnL conversion; leverage is not multiplied twice.', 'runtime_must_fail_on_missing_contract': True}
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
- PASS `manifest_validation_pnl_matches_fine_sweep`: manifest=62.79188028207658 fine=62.79188028207658
- PASS `manifest_validation_mdd_matches_fine_sweep`: manifest=-17.639305433768914 fine=-17.639305433768914
- PASS `manifest_validation_trades_matches_fine_sweep`: manifest=41 fine=41
- PASS `manifest_validation_wr_matches_fine_sweep`: manifest=0.5609756097560976 fine=0.5609756097560976
- PASS `manifest_validation_avg_notional_matches_fine_sweep`: manifest=1.0754364556866143 fine=1.0754364556866143
- PASS `manifest_validation_avg_margin_fraction_matches_fine_sweep`: manifest=0.5377182278433071 fine=0.5377182278433071
- PASS `manifest_validation_avg_leverage_matches_fine_sweep`: manifest=2.0 fine=2.0
- PASS `manifest_validation_overlay_hits_matches_fine_sweep`: manifest=5 fine=5
- PASS `manifest_validation_log_risk_utility_matches_fine_sweep`: manifest=0.2001911850926682 fine=0.2001911850926682
- PASS `manifest_oos_pnl_matches_fine_sweep`: manifest=71.997512214847 fine=71.997512214847
- PASS `manifest_oos_mdd_matches_fine_sweep`: manifest=-13.571350217596335 fine=-13.571350217596335
- PASS `manifest_oos_trades_matches_fine_sweep`: manifest=29 fine=29
- PASS `manifest_oos_wr_matches_fine_sweep`: manifest=0.6896551724137931 fine=0.6896551724137931
- PASS `manifest_oos_avg_notional_matches_fine_sweep`: manifest=1.2074976208182269 fine=1.2074976208182269
- PASS `manifest_oos_avg_margin_fraction_matches_fine_sweep`: manifest=0.6037488104091134 fine=0.6037488104091134
- PASS `manifest_oos_avg_leverage_matches_fine_sweep`: manifest=2.0 fine=2.0
- PASS `manifest_oos_overlay_hits_matches_fine_sweep`: manifest=4 fine=4
- PASS `manifest_oos_log_risk_utility_matches_fine_sweep`: manifest=0.38486709748033526 fine=0.38486709748033526
- PASS `validation_pnl_improves`: candidate=62.79188028207658 baseline=35.853831530265
- PASS `oos_pnl_improves_diagnostic`: candidate=71.997512214847 baseline=43.2312455386217
- PASS `ledger_notional_math_exact`: max_error=0.0
- PASS `validation_partial_done_matches_hits`: partial_done=5 hits=5
- PASS `oos_partial_done_matches_hits`: partial_done=4 hits=4
- PASS `selection_oos_informed_declared`: selection={'policy': 'mdd18 high-growth research candidate from Omega3 exposure fine sweep', 'selection_oos_informed': True, 'clean_oos_holdout_available_for_this_candidate': False, 'note': 'This candidate may pass contract/reproduction audit but cannot claim clean OOS promotion without a fresh holdout or walk-forward confirmation.'}
