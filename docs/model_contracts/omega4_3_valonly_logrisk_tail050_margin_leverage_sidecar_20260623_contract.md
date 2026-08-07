# Omega4.3 Validation-Only Log-Risk Margin+Leverage Sidecar Contract - 2026-06-23

## Status

- Model id: `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Status: `current_omega_research_baseline_not_live_wired`
- Red-team verdict: `REDTEAM_PASS_CLEAN_RESEARCH_BASELINE_NOT_LIVE_WIRED`
- Manifest: `data/ensemble/supervised/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`
- Red-team report: `docs/audits/omega4_3_valonly_logrisk_sidecar_redteam_20260623.md`

This candidate replaces OOS-guarded mapping selection with validation-only
selection. It keeps Omega4.2 parent direction, quality, exit-head timing, and
ATR safety SLTP unchanged. The sidecar owns sizing only.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
```

SLTP remains raw price-move based:

```text
tp_price_move = clip(max(0.075, atr_pct_192 * 12.0), 0.0, 0.22)
sl_price_move = clip(max(0.040, atr_pct_192 * 6.0), 0.0, 0.12)
```

Selected variant: `risk_3473`

```json
{
  "min_scale": 0.85,
  "max_scale": 1.65,
  "temp": 2.1,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.75,
  "short_scale": 1.1,
  "leverage_min": 1.25,
  "leverage_max": 3.0,
  "leverage_temp": 1.7,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 0.75,
  "short_leverage_scale": 1.25
}
```

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` |
| OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` |

## Live Wiring

No live wiring was changed. Runtime-native parity and shadow/paper smoke remain
required before exchange use.
