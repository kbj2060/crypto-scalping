# Clean Base Exposure Sweep 2026

Date: 2026-05-06 KST

Verdict: `aggressive_shadow_only_do_not_promote`

## Purpose

Test whether the clean base policy can lift OOS PnL by increasing `notional_mult`, `max_notional`, and leverage metadata without retraining or replacing the clean base entry owner.

This experiment does not overwrite production or baseline artifacts.

## Artifacts

- Script: `scripts/sweep_clean_base_exposure_2026.py`
- JSON report: `data/ensemble/reports/clean_base_exposure_sweep_2026.json`
- Validation grid CSV: `data/ensemble/reports/clean_base_exposure_sweep_2026.csv`
- Source contract: `docs/model_contracts/clean_base_policy_2026_contract.md`

## Key Accounting Note

In the current canonical `backtest_no_limit_exit` path, PnL is primarily driven by `notional_exposure`.

`leverage` is used as a decision/output field and as exit-model context, but it does not by itself multiply PnL unless notional also increases. Therefore, raising leverage alone is not equivalent to raising true exchange exposure in this backtest.

## Validation Selection

Grid:

```text
notional_mult: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0
max_notional: 3.6, 5.0, 7.5, 10.0, 12.5
leverage_mult: 1.0, 1.5, 2.0, 3.0
leverage_cap: 10.0
```

Validation-selected candidates:

| Selector | Candidate |
|---|---|
| Base reference | `nm1.50_maxn3.6_levm1.00_levcap10.0` |
| Max validation PnL | `nm3.50_maxn3.6_levm1.00_levcap10.0` |
| Balanced validation score | `nm3.50_maxn3.6_levm1.00_levcap10.0` |
| Red-team constrained | `nm1.50_maxn3.6_levm1.00_levcap10.0` |

The high-PnL validation candidate increases average notional but reduces validation trade coverage:

| Candidate | Validation PnL | MDD | Trades | Trades/day | Avg notional |
|---|---:|---:|---:|---:|---:|
| Base | `553.610081%` | `-12.656922%` | `695` | `11.394091` | `0.685455` |
| Aggressive | `827.024107%` | `-25.270486%` | `258` | `4.229749` | `1.230117` |

## OOS Results

Canonical OOS, 2026:

| Candidate | PnL | MDD | Trades | Trades/day | Avg notional | Avg leverage |
|---|---:|---:|---:|---:|---:|---:|
| Base | `177.329809%` | `-17.759665%` | `363` | `6.187500` | `0.600263` | `1.581454` |
| Aggressive validation max | `208.569834%` | `-33.772054%` | `73` | `1.244318` | `0.995015` | `1.576416` |
| Best OOS diagnostic, not promotable | `213.948275%` | `-36.589986%` | `79` | `1.346591` | `0.880429` | `1.578527` |

Cost stress:

| Candidate | Cost 1x | Cost 2x | Cost 3x |
|---|---:|---:|---:|
| Base | `177.329809%` | `92.254878%` | `-7.969395%` |
| Aggressive validation max | `208.569834%` | `198.127654%` | `165.480386%` |
| Best OOS diagnostic, not promotable | `213.948275%` | `210.294729%` | `177.355138%` |

Realistic replay for the aggressive validation max:

| Metric | Value |
|---|---:|
| PnL | `153.191786%` |
| MDD | `-28.450725%` |
| Trades | `42` |
| Trades/day | `0.715909` |
| Liquidations | `0` |
| Partial-fill events | `42` |

## Red-Team Interpretation

Increasing exposure improves canonical OOS PnL from `177.33%` to `208.57%` for the validation-selected aggressive candidate, and cost 3x becomes positive. However, the tradeoff is large:

- MDD worsens from `-17.76%` to `-33.77%`.
- Trades/day collapses from `6.19` to `1.24`.
- Realistic replay reduces PnL to `153.19%` and trades/day to `0.72`.
- The red-team constrained selector rejects the aggressive candidate and keeps the base.

Decision invariant audit passed for both base and aggressive candidates, so this is not an arithmetic invalidity. It is a risk/coverage failure.

## Decision

Do not promote the aggressive exposure candidate.

It may be used as a shadow-only research branch if the objective changes from stable scalping to lower-frequency, higher-conviction exposure. For the current goal of high return, low MDD, and rich trade volume, simple exposure scaling is not the right next production move.

## Next Recommendation

If higher PnL is required, build a conditional exposure booster instead of globally raising notional:

```text
base entry owner preserved
notional boost only on validation-calibrated high-edge buckets
boost range limited, e.g. 1.0x to 1.6x
no boost during daily DD, loss cooldown, high funding, low liquidity, or tail-risk states
trades/day floor >= 5.5
MDD gate no worse than clean base unless explicitly accepted for shadow
cost 1x/2x/3x reported
realistic replay required
```

