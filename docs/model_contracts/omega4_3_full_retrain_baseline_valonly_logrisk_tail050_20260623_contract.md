# Omega4.3 Full-Retrain Baseline Test - 2026-06-23

## Status

- Model id: `omega4_3_full_retrain_baseline_valonly_logrisk_tail050_20260623`
- Status: `rejected_research_test_not_upgrade`
- Reference baseline: `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_full_retrain_baseline_e2_fulltrain_fullexit_q070_20260623/true_3head_tabm_bundle.pt`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v9_full_retrain_baseline_parent_e2_fulltrain_fullexit_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- Manifest: `data/ensemble/supervised/omega4_3_full_retrain_baseline_valonly_logrisk_tail050_20260623/candidate_manifest.json`
- Red-team report: `docs/audits/omega4_3_full_retrain_baseline_valonly_logrisk_tail050_redteam_20260623.md`

## Training Contract

- Parent direction labels: `zigzag_action_labels_parent72_loose_20260620`
- Quality labels: `same_as_direction`
- Exit labels: `entry_label_terminal_giveback`
- Parent train rows: full train split, no `max_train_rows` cap
- Exit samples: full generated exit dataset, no `max_exit_samples` cap
- Risk sidecar: HGB, side split, parent output features, dynamic leverage
- Selection: validation-only log-risk, `tail_penalty=0.5`, OOS readout only

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected risk variant: `risk_1683`

```json
{
  "min_scale": 0.75,
  "max_scale": 1.65,
  "temp": 2.1,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.55,
  "short_scale": 1.25,
  "leverage_min": 1.25,
  "leverage_max": 3.0,
  "leverage_temp": 1.7,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 0.75,
  "short_leverage_scale": 1.25
}
```

## Result

Rejected. The full retrain did not reproduce the current Omega4.3 baseline. Sizing-only validation was `+4.83%` versus current baseline `+30.33%`, and OOS was `-10.23%` versus `+32.44%`.
