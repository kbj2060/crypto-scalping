# Omega4.3.1 Validation-Only cMamba Tail-Downside Risk Guard Test - 2026-06-23

## Status

- Model id: `omega4_3_1_valonly_cmamba_tail_downside_guard_20260623`
- Status: `rejected_research_test_not_upgrade`
- Base model: `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Manifest: `data/ensemble/supervised/omega4_3_1_valonly_cmamba_tail_downside_guard_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_1_valonly_cmamba_tail_downside_guard_20260623/runtime_contract.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_1_valonly_cmamba_tail_downside_guard_20260623/risk_sidecar.pkl`
- Red-team report: `docs/audits/omega4_3_1_valonly_cmamba_tail_downside_guard_redteam_20260623.md`

This test replaced the HGB score regressor and HGB q10 downside regressor with
side-split cMamba-style causal gated-conv score/q10 regressors. Parent direction,
quality, exit-head timing, and raw ATR safety SLTP were unchanged.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected variant: `risk_3681`

```json
{
  "min_scale": 0.95,
  "max_scale": 1.45,
  "temp": 1.35,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.55,
  "short_scale": 1.25,
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
| cMamba validation | `+17.45%` | `-6.87%` | `67.06%` | `85` | `0.4283` | `0.2268` | `1.7888` | `0.134584` |
| Baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |
| cMamba OOS readout | `+16.31%` | `-6.61%` | `66.15%` | `65` | `0.3894` | `0.2163` | `1.6797` | `0.136916` |

## Conclusion

Rejected as an upgrade. The cMamba sidecar passed selection-hygiene checks, but
it reduced validation PnL from `+30.33%` to `+17.45%` and validation log-risk utility from `0.205548` to `0.134584`.
