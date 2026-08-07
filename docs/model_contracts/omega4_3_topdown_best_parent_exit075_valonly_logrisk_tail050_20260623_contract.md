# Omega4.3 Top-Down Best Parent Exit0.75 Test - 2026-06-23

## Status

- Model id: `omega4_3_topdown_best_parent_exit075_valonly_logrisk_tail050_20260623`
- Status: `topdown_improved_research_candidate_not_current_upgrade`
- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623/true_3head_tabm_bundle.pt`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- Manifest: `data/ensemble/supervised/omega4_3_topdown_best_parent_exit075_valonly_logrisk_tail050_20260623/candidate_manifest.json`
- Red-team report: `docs/audits/omega4_3_topdown_best_parent_exit075_redteam_20260623.md`

## Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected risk variant: `risk_5241`

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

## Result

Improved versus failed full retrain: validation `+4.83% -> +19.10%`, OOS `-10.23% -> +22.21%`. Still below current baseline validation `+30.33%` and OOS `+32.44%`.
