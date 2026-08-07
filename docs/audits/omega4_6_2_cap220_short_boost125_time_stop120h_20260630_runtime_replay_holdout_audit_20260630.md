# Omega 4.6.2 Runtime Replay / Holdout Audit - 2026-06-30

## Scope

- Model id: `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`
- Variant: `short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h`
- Source report: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/report.json`
- Source audit: `tmp/causal_regen_20260516/omega4_6_borrowed_upgrade_notional_cap220_20260630/short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h_redteam_audit_20260630.json`

## 1. Runtime-Native Replay

Verdict: `FAIL`

Checked evidence:

- Runtime policy requires replay through `trading_bot.FinalGovernorRuntime.decide()` sequentially.
- `runtime_contract.json` has `runtime_native_replay_available = false`.
- `candidate_manifest.json` has `runtime_native_replay_available = false`.
- Source red-team audit has `runtime_native_replay_available = false`.
- Exact Omega4.6.2 model directory contains only contract/manifest files; no `runtime_native_decisions.csv`, `runtime_native_trade_journal.csv`, or native replay report was found.
- No exact `omega4_6_2` / `short_boost125_cap220` runtime wiring was found in `trading_bot.py`, `quant/**/*.py`, or relevant `scripts/backtest_*` / runtime scripts.
- The source evaluator declares this work as ledger-level diagnostics and says it does not promote a runtime contract.

Conclusion: runtime-native replay is still incomplete.

## 2. OOS Readout / Fresh Holdout

Verdict: `FAIL_FOR_CLEAN_OOS_PROMOTION`

Source report selection scope: `validation_only; OOS readout only`

| Candidate | Variant | Val PnL | Val monthly min | Score | OOS PnL | OOS MDD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Script validation-only winner | `short_rsi_skip_ge_56p656189__none__time_stop_120h` | `+215.19%` | `+38.24%` | `274.59` | `+67.38%` | `-10.06%` |
| Omega4.6.2 cap220 | `short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h` | `+211.14%` | `+36.99%` | `268.67` | `+79.32%` | `-10.13%` |

Differences, cap220 minus validation-only winner:

- Validation PnL: `-4.05pp`
- Validation monthly min: `-1.25pp`
- Selection score: `-5.92`
- OOS PnL: `+11.93pp`

Conclusion: Omega4.6.2 cap220 was not the validation-only winner. Its promotion
as a detail-line candidate depends on the stronger OOS readout, so it cannot
claim clean-OOS promotion without a fresh holdout or walk-forward confirmation.
No exact fresh holdout/walk-forward artifact for this model was found.

## Overall

- Runtime-native replay available: `false`
- Clean-OOS promotion claim allowed: `false`
- Fresh holdout/walk-forward required: `true`

Status: `BLOCKED_FOR_FULL_PROMOTION_UNTIL_RUNTIME_NATIVE_REPLAY_AND_FRESH_HOLDOUT_OR_WALKFORWARD`
