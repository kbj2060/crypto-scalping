# Main Governor Trend5x Design

Last updated: 2026-05-03 KST

## Current Main Stack

The current main governor stack is:

1. High-Conviction Sniper Sleeve 5x
2. Bull/Bear Trend Sleeve 5x
3. W/N/C Microstructure Sleeve 5x

The Day Trader Controller is removed from this governor path. It is not an on/off branch in the main evaluator.

The old name `Legacy Leverage Sniper` is operationally misleading. It is still active and should be treated as the high-conviction sniper sleeve, not as a deprecated module.

## Exposure

All three active sleeves use 5x notional/leverage in the current main profile.

| Sleeve | Regimes | Notional exposure | Leverage | Role |
| --- | --- | ---: | ---: | --- |
| High-Conviction Sniper Sleeve | all admitted sniper signals | 5.0 | 5.0 | Highest-priority sparse entries |
| Bull/Bear Trend Sleeve | bull, bear | up to 5.0 | 5.0 | Trend continuation in directional regimes |
| W/N/C Microstructure Sleeve | whipsaw, normal, chop | up to 5.0 | 5.0 | Short-horizon microstructure entries |

`notional_exposure=5.0` means the trade is sized to 5x account equity notional. `leverage=5.0` means that notional is carried with 5x leverage, so the margin fraction is approximately `notional / leverage = 1.0` before any sleeve-specific scaling.

## Flat Router Priority

When the account is flat, the governor chooses the first eligible sleeve in this order:

1. High-Conviction Sniper Sleeve 5x
2. Bull/Bear Trend Sleeve 5x
3. W/N/C Microstructure Sleeve 5x
4. Cash

This priority is intentional. The Trend Sleeve did not replace the Sniper Sleeve. In 2026 OOS, removing the sniper sleeve reduced PnL from `+45,377` to `+13,952`.

## Position Ownership

Only one sleeve owns an open position.

| Owner | Hold/close manager |
| --- | --- |
| `sniper` | Sniper actor close/hold logic |
| `trend` | Trend stop-loss, take-profit, trailing stop, max-hold, opposite-probability exit |
| `micro` | Micro stop-loss, take-profit, trailing stop, max-hold, opposite-probability exit |

Same-position resize is not active in the current main direct-sleeve path. The latest main 2026 OOS report has `resize_events=0`.

## Main 2026 OOS Result

Report:

- `tmp/governor_no_day_sniper_trend_micro_c68_g34_trend5x_2026.json`

Metrics:

| Metric | Value |
| --- | ---: |
| PnL | `+45,377` |
| MDD | `-13.59%` |
| Trades/day | `18.12` |
| Win rate | `72.44%` |
| Sniper entries | `108` |
| Trend entries | `400` |
| Micro entries | `554` |

## Repro Command

The main defaults are now encoded in `scripts/eval_governor_microstructure_wnc_oos_2026.py`, so this command reproduces the main stack without passing model paths:

```bash
python scripts/eval_governor_microstructure_wnc_oos_2026.py \
  --report-out data/ensemble/reports/governor_main_sniper_trend5x_micro5x_oos_2026.json
```

Equivalent explicit command:

```bash
python scripts/eval_governor_microstructure_wnc_oos_2026.py \
  --micro-model data/ensemble/supervised/microstructure_wnc_sleeve_realistic_5x.pkl \
  --trend-model data/ensemble/supervised/trend_bull_bear_sleeve_v1_c68_g34_notional5_leverage5.pkl \
  --sniper-notional-exposure 5.0 \
  --sniper-leverage 5.0 \
  --report-out data/ensemble/reports/governor_main_sniper_trend5x_micro5x_oos_2026.json
```

## Known Risk

The main stack is cost-sensitive.

| Stress case | PnL | MDD |
| --- | ---: | ---: |
| Base cost | `+45,377` | `-13.59%` |
| Fee 2x | `+620` | `-17.37%` |
| Slippage 2x | `+7,267` | `-17.05%` |
| Fee + slippage 2x | `-37` | `-40.92%` |

The next improvement should be a cost-aware defensive profile, not removal of the sniper sleeve.
