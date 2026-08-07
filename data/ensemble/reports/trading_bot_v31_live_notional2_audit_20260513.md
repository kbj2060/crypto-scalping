# Trading Bot V31 Live Notional 2.0 Audit

Date: 2026-05-13 KST

## Change

`trading_bot.py` keeps V31 as the live main deep scout path and applies a live-only runtime override:

```text
FINAL_GOVERNOR_V31_DEEP_NOTIONAL = 2.0
```

This changes only the V31 `deep_alpha` sleeve opened when the parent policy is CASH.

## Unchanged

- Parent model: `hf_v13_clean_regime_margin110_20260511`
- Parent LONG/SHORT notional/leverage/TP/SL/max-hold behavior
- V21.2 jackpot add-on behavior for parent-owned positions
- Frozen V27 deep scout entry model
- V31 rule exit overlay formulas
- Next-bar fill contract

## Backtest Basis

Sensitivity report:

- `data/ensemble/reports/v31_deep_scout_notional_sensitivity_20260513.json`
- `data/ensemble/reports/v31_deep_scout_notional_sensitivity_20260513.csv`

Selected live override:

```text
deep scout notional = 2.0
cost1 PnL = +361.19%
cost1 MDD = -31.74%
cost2 PnL = +88.74%
cost3 PnL = +0.58%
```

## Red Team Notes

- This is an aggressive live/shadow setting.
- Cost3 survives only marginally in the sensitivity test.
- Parent positions are not changed; only the CASH-sleeve V31 deep scout exposure is increased.
