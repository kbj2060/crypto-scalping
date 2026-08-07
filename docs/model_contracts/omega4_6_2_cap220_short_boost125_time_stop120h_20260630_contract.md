# Omega 4.6.2 cap220 Detail-Line Contract - 2026-06-30

## Status

- Model id: `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`
- Display version: `Omega 4.6.2 cap220 detail-line candidate`
- Base model: `omega4_6_plus_t12_nohold_risk1_20260630`
- Status: `omega4_6_2_research_detail_line_candidate_not_live_wired`
- Classification: `conditional_diagnostic_pass_full_live_fail_fresh_holdout_required`
- Runtime contract: `tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json`
- Candidate manifest: `data/ensemble/supervised/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/candidate_manifest.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/promotion_manifest.json`
- Diagnostic source: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/report.json`
- Red-team report: `docs/audits/omega4_6_2_cap220_short_boost125_time_stop120h_20260630_redteam_20260630.md`

This is an Omega4.6.2 research/detail-line candidate. It does not replace the
Omega4.6 baseline or the Omega4.6.1 upgrade candidate, and it is not live-wired.

## Rule Stack

Entry gate:

```text
if side == SHORT and rsi >= 56.656189:
    skip entry
```

Exposure overlay:

```text
long_factor = 0.90
short_factor = 1.25
tested_notional_cap = 2.20
leverage_cap = 5.0
```

Lifecycle overlay:

```text
max_hold_hours = 120.0
```

The overlays change entry filtering, exposure sizing, and forced time-stop exit
only. They do not retrain the Omega4.6 parent components.

## Metrics

| Split | PnL | MDD | WR | Trades | Long | Short | Max Hold | Avg Hold | Max Lev | Max Notional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+211.14%` | `-13.72%` | `65.22%` | `23` | `6` | `17` | `120.00h` | `63.27h` | `5.00x` | `1.9176` |
| OOS readout | `+79.32%` | `-10.13%` | `61.54%` | `13` | `1` | `12` | `120.00h` | `67.79h` | `5.00x` | `2.1967` |

Baseline Omega4.6 reference:

| Split | PnL | MDD | WR | Trades | Max Hold | Max Notional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+117.17%` | `-17.43%` | `51.72%` | `29` | `222.00h` | `1.8000` |
| OOS readout | `+67.85%` | `-13.28%` | `53.85%` | `13` | `218.50h` | `1.8000` |

Delta versus Omega4.6: validation `+93.98pp`,
OOS readout `+11.47pp`; max hold improves by
`-102.00h` validation and
`-98.50h` OOS.

## Red-Team Scope

- Conditional diagnostic pass: `true`
- Full live pass: `false`
- Artifact integrity promotion pass: `true`
- Selection scope: `cap220 balanced diagnostic candidate; OOS readout considered for detail-line choice, so no clean-OOS promotion claim`

The old notional `<= 1.8` pass gate is not part of this red-team condition.
The futures accounting contract still remains mandatory:
`notional = margin_fraction * leverage`; `PnL = realized_price_move * notional`.

Full-live blockers:

- Validation and OOS max hold remain above `24h`.
- OOS PnL remains below `100%`.
- Runtime-native replay is not yet available for this overlay stack.
- Because OOS readout was considered for the detail-line choice, fresh holdout
  or walk-forward is required before any clean promotion claim.

## Artifacts

- Ranking: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/borrowed_upgrade_ranking.csv`
- Validation ledger: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/validation_short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h_ledger.csv`
- OOS ledger: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/oos_short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h_ledger.csv`
- Validation chart: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/charts/omega4_6_cap220_120h_validation_trade_chart.png`
- OOS chart: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/charts/omega4_6_cap220_120h_oos_trade_chart.png`
- Max-hold sensitivity: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/max_hold_sensitivity_short_boost125_cap220.csv`
