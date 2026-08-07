# Omega4.3 Log-Risk Margin+Leverage Sidecar Contract - 2026-06-23

## Status

- Model id: `omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623`
- Status: `current_omega_research_baseline_not_live_wired`
- Manifest: `data/ensemble/supervised/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`
- Source sidecar run: `omega4_2_trade_risk_sidecar_20260622_v7b_parent_side_hgb_mae050_margin_leverage_logrisk_tb020_tp050_liq012`
- Source report copy: `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/source_report.json`

Omega4.3 promotes the `tail_penalty = 0.5` log-risk margin+leverage sidecar as
the current Omega research baseline. It does not retrain the Omega4.2 neural
parent. Direction, quality, exit-head timing, and ATR safety SLTP remain the
Omega4.2 contract.

## Lineage

- Base runtime model: `omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622`
- Base bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070/true_3head_tabm_bundle.pt`
- Risk sidecar artifact: `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/risk_sidecar.pkl`
- Risk sidecar training report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v7b_parent_side_hgb_mae050_margin_leverage_logrisk_tb020_tp050_liq012/report.json`
- Promotion type: sizing-only risk sidecar overlay on Omega4.2.

## Risk Sidecar Training Data

- Model kind: `hgb`
- Feature mode: `parent_outputs`
- Side split long/short model: `True`
- Dynamic leverage: `True`
- Target: `trade net_per_notional from Omega 4.2 replay`
- Target MAE penalty: `0.5`
- Rows: `242` trade-level rows from Omega4.2 replay.
- Target mean / p50 / p75: `-0.008331` / `0.004762` / `0.016537`.

The sidecar selection objective is `log_risk`, with:

```json
{
  "tail_budget": 0.02,
  "tail_penalty": 0.5,
  "liquidation_buffer": 0.12,
  "liquidation_penalty": 0.25
}
```

## Runtime Contract

- Quality threshold: `0.70`
- Exit-head threshold: `0.70`
- Max hold bars: `0`
- Cooldown bars: `0`
- ATR window: `192` bars
- TP multiple / floor / cap: `12.0` / `0.075` / `0.22`
- SL multiple / floor / cap: `6.0` / `0.04` / `0.12`

SLTP remains a raw price-move barrier:

```text
tp_price_move = clip(max(0.075, atr_pct_192 * 12.0), 0.0, 0.22)
sl_price_move = clip(max(0.040, atr_pct_192 * 6.0), 0.0, 0.12)
```

Risk sizing is explicit:

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
```

The sidecar may change `margin_fraction` and `leverage`; it must not move the
TP/SL price barriers. Do not divide TP/SL price moves by notional.

## Selected Mapping

Selected variant: `risk_1673`

```json
{
  "min_scale": 0.75,
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

## Selected Metrics

Sizing-only contract, selected on validation log-risk utility:

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+29.39%` | `-7.66%` | `67.06%` | `85` | `0.5297` | `0.2237` | `2.2574` | `0.204769` |
| OOS | `+31.42%` | `-5.37%` | `66.15%` | `65` | `0.5382` | `0.2272` | `2.2699` | `0.258542` |

Reference Omega4.2 sizing baseline:

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Lev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+16.02%` | `-7.07%` | `67.06%` | `85` | `0.3875` | `2.0` |
| OOS | `+13.32%` | `-4.34%` | `66.15%` | `65` | `0.3894` | `2.0` |

## Full-Replay Diagnostic

Full replay with dynamic risk values fed back into exit simulation is recorded
only as a diagnostic and is not the promoted runtime contract.

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `+30.25%` | `-10.25%` | `68.35%` | `79` | `0.190557` |
| OOS | `+32.77%` | `-5.38%` | `66.10%` | `59` | `0.268701` |

Validation MDD weakens to `-10.25%` in full replay, so dynamic risk values are
not allowed to alter the active exit timing without separate exit retraining and
promotion.

## Live Wiring

No live runtime wiring was changed. Before real exchange use, run
runtime-native parity, current live feature-contract validation, and shadow or
paper smoke. Contract mismatches must fail fast; do not add aliases, fallback
prefixes, or compatibility shims on the active path.
