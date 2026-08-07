# Omega 4.6.2 cap220 Detail-Line Red-Team Audit - 2026-06-30

## Verdict

- Model id: `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`
- Variant: `short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h`
- Verdict: `CONDITIONAL_DIAGNOSTIC_PASS_FULL_LIVE_FAIL_FRESH_HOLDOUT_REQUIRED`
- Conditional diagnostic pass: `true`
- Full live pass: `false`
- Source audit JSON: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h_redteam_audit_20260630.json`
- Source audit MD: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h_redteam_audit_20260630.md`

This is an Omega4.6.2 research/detail-line candidate, not a live-wired model.
The candidate passes the conditional diagnostic gate after removing the old
notional `<= 1.8` red-team pass condition, while still requiring exact
`notional = margin_fraction * leverage` consistency.

## Conditional Gate Checks

| Check | Pass |
| --- | ---: |
| `artifact_integrity_pass` | `true` |
| `validation_mdd_lte_20_abs` | `true` |
| `oos_mdd_lte_20_abs` | `true` |
| `validation_leverage_lte_5` | `true` |
| `oos_leverage_lte_5` | `true` |
| `validation_no_overlap` | `true` |
| `oos_no_overlap` | `true` |
| `validation_accounting_consistent` | `true` |
| `oos_accounting_consistent` | `true` |
| `validation_notional_contract_consistent` | `true` |
| `oos_notional_contract_consistent` | `true` |

## Full Live Checks

| Check | Pass |
| --- | ---: |
| `validation_max_hold_lte_24h` | `false` |
| `oos_max_hold_lte_24h` | `false` |
| `validation_pnl_gte_100pct` | `true` |
| `oos_pnl_gte_100pct` | `false` |
| `runtime_native_replay_available` | `false` |
| `clean_oos_selection_claim_allowed` | `false` |

## Metrics

| Split | PnL | MDD | WR | Trades | Max Hold | Max Lev | Max Notional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+211.14%` | `-13.72%` | `65.22%` | `23` | `120.00h` | `5.00x` | `1.9176` |
| OOS readout | `+79.32%` | `-10.13%` | `61.54%` | `13` | `120.00h` | `5.00x` | `2.1967` |

## Required Before Promotion

- Fresh holdout or walk-forward test because OOS readout was considered for this detail-line choice.
- Runtime-native replay parity.
- Max-hold remediation if the day-trading `24h` gate remains mandatory.
- OOS PnL improvement above `100%` if the `100%` target remains mandatory.
