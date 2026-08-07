# Clean Base Deep Gated Gross V2 Red Team Approval

Decision: `APPROVED_AS_PROMOTE_CANDIDATE`

## Scope

- Experiment: `clean_base_deep_gated_gross_v2`
- Report: `data/ensemble/reports/clean_base_deep_gated_gross_v2_2026.json`
- Ledger: `data/ensemble/reports/clean_base_deep_gated_gross_v2_ledger.csv`
- Model: `data/ensemble/supervised/clean_base_deep_gated_gross_v2/deep_gated_gross.pkl`

## Result

| Metric | Value |
|---|---:|
| OOS PnL 1x | `796.058286%` |
| OOS MDD 1x | `-24.945437%` |
| Trades/day 1x | `6.152542` |
| Avg notional 1x | `3.535537` |
| Cost2 PnL | `67.008088%` |
| Cost3 PnL | `0.000000%` |

Selected config: `dgg_high3.6_mid3.0_def3.0_h-0.0060_m-0.0120_adv99.000_c30.00`

Deep bucket usage:

- HIGH: `324`
- MID: `20`
- DEFENSIVE: `19`

## Audit Findings

- Accounting identity: `PASS`
- Ledger final PnL equals report PnL: `PASS`
- Fee identity: `PASS`
- Non-finite numeric cells: `0`
- Negative notional: `0`
- Gross/net cap violations: `0`
- Exit after lifecycle core exit: `0`
- Runtime future-return use: `FALSE`
- OOS threshold selection: `FALSE`

## Approval Conditions

- Approved as a research promote candidate and live-shadow candidate.
- Cost3 survival is capital preservation by disabling trades, not positive trading alpha under 3x costs.
- Cost2 survives with positive PnL but has high drawdown, so live deployment should keep hard kill-switch and audit logging enabled.
