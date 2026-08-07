# Omega 4.4 v18 + Omega3 exposure transfer (side_l0p95_s1p35_cap1p05_shortpartial) Contract - 2026-06-26

## Status

- Model id: `omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626`
- Base model: `omega4_4_v18_baseline_20260624`
- Source model: `omega3_aggressive_compensated_scale200_cap090_20260618`
- Variant: `side_l0p95_s1p35_cap1p05_shortpartial`
- Status: `omega4_4_v18_omega3_exposure_transfer_research_candidate_not_live_wired`
- Red-team verdict: `REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED`

## Risk Remap Contract

```json
{
  "enabled": true,
  "source_idea": "borrow Omega3 aggressive exposure while preserving Omega4.4 risk score ordering",
  "mode": "side_scaled",
  "scale": 1.0,
  "cap_notional": 1.05,
  "fixed_notional": 0.0,
  "long_scale": 0.95,
  "short_scale": 1.35,
  "leverage": 2.0,
  "notional_math": "notional = margin_fraction * leverage",
  "side_scaled_formula": "notional = min(base_margin_fraction * base_leverage * side_scale, cap_notional)",
  "margin_formula": "margin_fraction = notional / leverage",
  "sltp_contract": "ATR safety TP/SL remains a price-move barrier before PnL conversion; leverage is not multiplied twice.",
  "runtime_must_fail_on_missing_contract": true
}
```

## Lifecycle Overlay Contract

```json
{
  "enabled": true,
  "source_family": "omega1_2_1_horizon_short_cap_fine_20260612",
  "source_idea": "short_cap1760_min0.035 aged profitable short lifecycle guard",
  "mode": "short_aged_profit_partial_deleverage",
  "side": "short",
  "side_value": -1,
  "cap_bars": 1152,
  "bar_interval_minutes_assumption": 5,
  "cap_duration_days_assumption": 4.0,
  "min_unrealized_price_move": 0.035,
  "partial_fraction": 0.5,
  "fires_once_per_position": true,
  "execution_timing": "On each in-position bar after MFE/MAE update and before standard TP/SL/exit-head checks, if a short has held at least cap_bars and current directional price move is at least min_unrealized_price_move, execute one partial close for partial_fraction of current notional; the remaining notional continues under the existing exit head and ATR safety TP/SL.",
  "cash_update_contract": "closed_fraction_cash *= 1 + raw_partial_price_move * closed_notional, less execution fees",
  "remaining_position_contract": "remaining_notional = prior_notional * (1 - partial_fraction)",
  "sltp_contract": "TP/SL price-move barrier locations are unchanged by the partial de-risk event"
}
```

## Selection Caveat

This candidate was selected after a validation/OOS diagnostic fine sweep.
It cannot claim clean-OOS promotion until a fresh holdout or walk-forward confirmation is run.

## Metrics

| Split | PnL | MDD | WR | Trades | Avg notional | Overlay hits | Log-risk utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline validation | `+35.8538%` | `-14.1696%` | `0.5366` | `41` | `0.7595` | `0` | `0.169373` |
| Candidate validation | `+49.9022%` | `-13.8272%` | `0.5610` | `41` | `0.8853` | `5` | `0.218622` |
| Baseline OOS readout | `+43.2312%` | `-10.7585%` | `0.6765` | `34` | `0.7931` | `0` | `0.290538` |
| Candidate OOS readout | `+62.7677%` | `-10.4708%` | `0.7143` | `35` | `0.9347` | `4` | `0.386488` |

## Artifacts

- Runtime contract: `tmp/causal_regen_20260516/omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626/runtime_contract.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626/promotion_manifest.json`
- Candidate manifest: `data/ensemble/supervised/omega4_4_v18_omega3_side_l095_s135_cap105_shortpartial_20260626/candidate_manifest.json`
- Fine sweep report: `tmp/causal_regen_20260516/omega44_v18_omega3_refined_side_scale_sweep_20260626/report.json`
