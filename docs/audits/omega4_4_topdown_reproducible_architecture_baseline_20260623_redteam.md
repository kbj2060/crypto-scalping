# Omega4.4 Top-Down Reproducible Architecture Baseline Audit - 2026-06-23

## Verdict

`omega4_4_topdown_reproducible_architecture_baseline_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_REPRODUCIBLE_ARCHITECTURE_BASELINE_NOT_PERFORMANCE_CHAMPION`**

Omega4.4 promotes the top-down retrained stack as the reproducible architecture baseline. This does not overwrite the Omega4.3 performance champion.

## Promotion Scope

- Promoted role: reproducible architecture baseline
- Not promoted as: live model, production model, or performance champion
- Source candidate: `omega4_3_topdown_best_parent_exit075_valonly_logrisk_tail050_20260623`
- Source top-down audit: `docs/audits/omega4_3_topdown_best_parent_exit075_redteam_20260623.md`

## Contract Checks

- Parent bundle exists: `True`
- Risk sidecar exists: `True`
- Quality threshold is `0.70`: `True`
- Exit threshold is `0.75`: `True`
- Selection scope is validation-only: `True`
- Notional contract declared: `True`
- Notional-scaled SLTP disabled: `True`
- Omega4.3 performance champion not overwritten: `True`

## Metrics

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation sizing-only | `+19.10%` | `-6.97%` | `59.09%` | `44` | `0.148370` |
| OOS sizing-only readout | `+22.21%` | `-5.55%` | `66.67%` | `36` | `0.187005` |

## Notes

- OOS is a readout only and was not used to select the risk mapping.
- Full replay remains diagnostic only.
- Future Omega4.4 improvements should compare against both this reproducible baseline and the Omega4.3 performance champion.
