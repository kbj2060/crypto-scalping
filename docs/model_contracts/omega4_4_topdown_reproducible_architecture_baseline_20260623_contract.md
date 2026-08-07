# Omega4.4 Top-Down Reproducible Architecture Baseline - 2026-06-23

## Status

- Model id: `omega4_4_topdown_reproducible_architecture_baseline_20260623`
- Display version: `Omega 4.4 top-down reproducible architecture baseline`
- Status: `current_omega_reproducible_architecture_baseline_not_live_wired`
- Role: reproducible architecture baseline, not the Omega4.3 performance champion replacement
- Red-team verdict: `REDTEAM_PASS_CLEAN_RESEARCH_REPRODUCIBLE_ARCHITECTURE_BASELINE_NOT_PERFORMANCE_CHAMPION`

## Lineage

Omega4.4 promotes the top-down candidate as a clean structural baseline:

1. Parent retrain: `same_as_direction + terminal_giveback + epochs2 + train15k + exit15k + q0.70`
2. Exit threshold selected for this parent: `0.75`
3. Risk sidecar: HGB, side-split, parent-output features, dynamic leverage, validation-only log-risk, `tail_penalty=0.5`

Source candidate: `omega4_3_topdown_best_parent_exit075_valonly_logrisk_tail050_20260623`

## Artifacts

- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623/true_3head_tabm_bundle.pt`
- Parent report: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- Risk report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623/report.json`
- Manifest: `data/ensemble/supervised/omega4_4_topdown_reproducible_architecture_baseline_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_4_topdown_reproducible_architecture_baseline_20260623/runtime_contract.json`

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
- Full dynamic-risk exit replay: diagnostic only, not promoted

## Selected Risk Mapping

```json
{
  "min_scale": 0.95,
  "max_scale": 1.65,
  "temp": 2.1,
  "floor": 0.06,
  "cap": 0.28,
  "long_scale": 0.75,
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

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Leverage | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation sizing-only | `+19.10%` | `-6.97%` | `59.09%` | `44` | `0.4330` | `0.2389` | `1.7761` | `0.148370` |
| OOS sizing-only readout | `+22.21%` | `-5.55%` | `66.67%` | `36` | `0.4567` | `0.2471` | `1.8103` | `0.187005` |

## Performance Champion Reference

Omega4.3 remains the performance champion reference:

- `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Validation `+30.33%`, OOS `+32.44%`

Omega4.4 is intentionally tracked as the reproducible architecture baseline for future top-down improvements.
