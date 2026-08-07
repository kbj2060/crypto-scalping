# Omega4.3.1 Validation-Only TabM Tail-Downside Risk Guard Test - 2026-06-23

## Status

- Model id: `omega4_3_1_valonly_tabm_tail_downside_guard_20260623`
- Status: `rejected_research_test_not_upgrade`
- Base model: `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Manifest: `data/ensemble/supervised/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/runtime_contract.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/risk_sidecar.pkl`
- Red-team report: `docs/audits/omega4_3_1_valonly_tabm_tail_downside_guard_redteam_20260623.md`

This test replaced the HGB score regressor and HGB q10 downside regressor with
side-split TabM score/q10 multi-output regressors. Parent direction, quality,
exit-head timing, and raw ATR safety SLTP were unchanged.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected variant: `risk_4866`

```json
{
  "min_scale": 0.95,
  "max_scale": 1.65,
  "temp": 1.35,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.85,
  "short_scale": 1.0,
  "leverage_min": 1.0,
  "leverage_max": 2.5,
  "leverage_temp": 1.35,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 0.9,
  "short_leverage_scale": 1.1
}
```

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| TabM validation | `+19.07%` | `-6.85%` | `67.06%` | `85` | `0.4451` | `0.2361` | `1.8199` | `0.152264` |
| Baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |
| TabM OOS readout | `+17.17%` | `-4.61%` | `66.15%` | `65` | `0.4017` | `0.2261` | `1.7194` | `0.150114` |

## Conclusion

Rejected as an upgrade. The TabM sidecar passed selection-hygiene checks, but
it reduced validation PnL from `+30.33%` to `+19.07%` and validation log-risk utility from `0.205548` to `0.152264`.
