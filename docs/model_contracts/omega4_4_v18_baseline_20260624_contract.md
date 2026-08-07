# Omega4.4 v18 Baseline Contract - 2026-06-24

## Status

- Model id: `omega4_4_v18_baseline_20260624`
- Display version: `Omega 4.4 v18 live-like dynamic leverage baseline`
- Status: `omega4_4_v18_research_baseline_not_live_wired`
- Role: high-exposure live-like dynamic leverage research baseline
- Red-team verdict: `REDTEAM_PASS_FULL_PROMOTABLE`

## Lineage

1. Parent: top-down best parent, q0.70, exit threshold 0.75.
2. Risk sidecar: HGB, side-split, parent-output features, dynamic leverage.
3. Selection: validation-only log-risk with validation MDD >= -16.00 and validation average notional in [0.75, 0.90].
4. OOS is readout only and is not used for mapping selection.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

- Quality threshold: `0.70`
- Exit threshold: `0.75`
- Max hold bars: `0`
- Cooldown bars: `0`
- Full dynamic-risk exit replay: promoted readout, `exit_sizing_input_mode=actual`

## Selected Risk Mapping

```json
{
  "min_scale": 1.0,
  "max_scale": 2.0,
  "temp": 1.7,
  "floor": 0.3,
  "cap": 0.4,
  "long_scale": 0.75,
  "short_scale": 1.25,
  "leverage_min": 1.75,
  "leverage_max": 2.5,
  "leverage_temp": 1.35,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 0.95,
  "short_leverage_scale": 1.05
}
```

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Leverage | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation sizing-only | `+39.43%` | `-11.12%` | `59.09%` | `44` | `0.7682` | `0.3562` | `2.1458` | `0.198157` |
| OOS sizing-only readout | `+36.65%` | `-10.74%` | `66.67%` | `36` | `0.7916` | `0.3637` | `2.1653` | `0.241201` |
| Validation full replay | `+35.85%` | `-14.17%` | `53.66%` | `41` | `0.7595` | `0.3532` | `2.1397` | `0.169373` |
| OOS full replay readout | `+43.23%` | `-10.76%` | `67.65%` | `34` | `0.7931` | `0.3645` | `2.1654` | `0.290538` |

## Artifacts

- Runtime contract: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/runtime_contract.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/promotion_manifest.json`
- Candidate manifest: `data/ensemble/supervised/omega4_4_v18_baseline_20260624/candidate_manifest.json`
- Source report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/risk_sidecar.pkl`
- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623/true_3head_tabm_bundle.pt`
