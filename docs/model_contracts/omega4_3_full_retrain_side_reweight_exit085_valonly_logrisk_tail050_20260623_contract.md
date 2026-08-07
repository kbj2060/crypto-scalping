# Omega4.3 Full-Retrain Side-Reweight Exit0.85 Test - 2026-06-23

## Status

- Model id: `omega4_3_full_retrain_side_reweight_exit085_valonly_logrisk_tail050_20260623`
- Status: `improved_research_candidate_not_current_upgrade`
- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_full_retrain_side_reweight_l065_s135_e2_fulltrain_fullexit_q070_20260623/true_3head_tabm_bundle.pt`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v11_full_retrain_side_reweight_l065_s135_exit085_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- Manifest: `data/ensemble/supervised/omega4_3_full_retrain_side_reweight_exit085_valonly_logrisk_tail050_20260623/candidate_manifest.json`
- Red-team report: `docs/audits/omega4_3_full_retrain_side_reweight_exit085_redteam_20260623.md`

## Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected risk variant: `risk_4895`

```json
{
  "min_scale": 0.95,
  "max_scale": 1.65,
  "temp": 1.35,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.55,
  "short_scale": 1.35,
  "leverage_min": 1.0,
  "leverage_max": 2.0,
  "leverage_temp": 1.0,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 1.0,
  "short_leverage_scale": 1.0
}
```

## Result

Improved versus the failed full retrain: validation `+4.83% -> +12.23%`, OOS `-10.23% -> +13.48%`. Still below current Omega4.3 baseline validation `+30.33%` and OOS `+32.44%`.
