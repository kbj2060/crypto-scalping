# Omega4.4 v18 Short Partial Lifecycle Overlay Contract - 2026-06-26

## Status

- Model id: `omega4_4_v18_short_partial_cap1152_u0035_p050_20260626`
- Base model: `omega4_4_v18_baseline_20260624`
- Source variant: `short_partial_cap1152_u0.035_p0.50`
- Status: `omega4_4_v18_lifecycle_overlay_research_candidate_not_live_wired`
- Red-team verdict: `REDTEAM_PASS_RESEARCH_CANDIDATE_CLEAN_OOS_PROMOTION_BLOCKED`

## Overlay Contract

```json
{
  "enabled": true,
  "source_family": "omega1_2_1_horizon_short_cap_fine_20260612",
  "source_idea": "short_cap1760_min0.035 aged profitable short lifecycle guard",
  "source_variant": "short_partial_cap1152_u0.035_p0.50",
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

This candidate was chosen as a balanced validation/OOS diagnostic candidate after the overlay sweep.
It cannot claim clean-OOS promotion until a fresh holdout or walk-forward confirmation is run.

## Metrics

| Split | PnL | MDD | WR | Trades | Overlay hits | Log-risk utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline validation | `+35.8538%` | `-14.1696%` | `0.5366` | `41` | `0` | `0.169373` |
| Candidate validation | `+35.9479%` | `-12.0643%` | `0.5366` | `41` | `5` | `0.170065` |
| Baseline OOS readout | `+43.2312%` | `-10.7585%` | `0.6765` | `34` | `0` | `0.290538` |
| Candidate OOS readout | `+48.5644%` | `-8.1922%` | `0.7143` | `35` | `4` | `0.327097` |

## Artifacts

- Runtime contract: `tmp/causal_regen_20260516/omega4_4_v18_short_partial_cap1152_u0035_p050_20260626/runtime_contract.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_4_v18_short_partial_cap1152_u0035_p050_20260626/promotion_manifest.json`
- Candidate manifest: `data/ensemble/supervised/omega4_4_v18_short_partial_cap1152_u0035_p050_20260626/candidate_manifest.json`
- Overlay report: `tmp/causal_regen_20260516/omega4_4_v18_short_aged_profit_overlay_full_replay_20260625/report.json`
