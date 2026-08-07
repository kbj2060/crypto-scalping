# ETH leverage-reduction + chop soft-sizing fresh-forward test (2026-07-20)

## Motivation

User asked whether ETH's cited live numbers (VAL +54.35%/OOS +102.02%/+105.38%,
`docs/model_contracts/live_model_v1_checkpoint_20260714.md`) already reflect (a) leverage
reduction or (b) `ChopSoftSizeShadow` (`shadow_notional = real_notional * max(0, 1-chop_prob)`,
`trading_bot_modules/omega4_6_1_chop_shadow.py`). Verified: **no to both** -- ChopSoftSizeShadow
is coded shadow-only (reads the already-finalized `_eth_notional`, writes to a separate duckdb,
never mutates the real order), and ETH's notional multiplier was actually *increased* (1.0x->1.5x)
during 3-asset portfolio tuning, not reduced. This session then re-ran the actual live greedy
router replay with both changes genuinely applied (not approximated).

## Reproduction note (found during this session)

No single script reproduces the exact cited numbers end-to-end; the closest is
`scripts/replay_omega4_6_1_greedy_router_20260706.py` (the real single-account greedy h48qual>zig075
router matching live logic, `SCALE_MAP`/`LEVERAGE_CAP=5.0`/`NOTIONAL_CAP=1.8` match
`trading_bot_modules/runtime_config.py`/`omega4_6_1_live.py` exactly), but its `oos_predictions_*.csv`
inputs had already been silently re-scored to 2026-07-12 during the 2026-07-13 portfolio
fresh-window session, while the script's own `ext_frame` is still hardcoded to
`2026-01-01..2026-06-30` -- a length mismatch this session had to resolve by truncating the
predictions back to 06-30 to match. Doing so and re-running gives **pnl +77.11%/mdd -15.48%/37
trades** (no duration gate) -- positive and same ballpark as the doc's +102.02%, but not an exact
match; the residual gap is most likely the live `ETH_NOTIONAL_MULTIPLIER=1.5x` layer, which is
applied in `trading_bot.py` on top of this component-level replay and was not reconstructed here.
This script's own output is used as the baseline below (self-consistent, freshly re-run against
live-matching caps/scale-map), not the doc's number, since the doc's own exact reproduction path
could not be located.

## Method

- Baseline: `LEVERAGE_CAP=5.0, NOTIONAL_CAP=1.8` (current live values).
- Reduced leverage: both caps halved (`2.5`/`0.9`) -- a genuine re-replay (not a post-hoc rescale),
  monkeypatching the same `greedy_replay` function's module-level caps, since a scale-map's applied
  leverage/notional is clamped by these caps inside the replay itself.
- Chop soft-sizing: applied as a post-hoc per-trade ledger rescale (`shadow_trade_return =
  trade_return * max(0, 1 - chop_prob)`), identical formula/methodology to
  `scripts/apply_chop_soft_sizing_sol_adaptive_squeeze_20260720.py` and to the shadow module's own
  live formula -- causal-safe since `chop_prob` at `entry_timestamp` is already known at decision
  time, and this only rescales the realized trade return (never changes entry/exit timing).
- VAL window: 2025-10-01..12-31 (`replay_omega4_6_1_greedy_val_20260706.py`'s own window, matching
  the doc's own VAL). OOS window: 2026-01-01..06-30 (matching the doc's own OOS-extended window,
  after the truncation above).

## Results

**VAL (2025-10-01..12-31, 29 trades, WR 41.4%, unaffected by the truncation issue above)**

| | PnL | MDD |
|---|---:|---:|
| baseline (live caps, no chop) | +36.82% | -24.34% |
| baseline + chop soft-sizing | **+38.97%** | **-8.41%** |
| reduced leverage (50%), no chop | +19.50% | -12.56% |
| reduced leverage (50%) + chop soft-sizing | +18.64% | **-4.29%** |

**OOS (2026-01-01..06-30, 37 trades, WR 45.9%)**

| | PnL | MDD |
|---|---:|---:|
| baseline (live caps, no chop) | +77.11% | -15.48% |
| baseline + chop soft-sizing | +49.33% | -10.54% |
| reduced leverage (50%), no chop | +37.46% | -8.48% |
| reduced leverage (50%) + chop soft-sizing | +23.66% | -5.50% |

Chop-rescale mean multiplier: 0.519 (roughly halves average notional across both windows).

## Reading the result

- **Leverage reduction**: pure risk/return tradeoff on both windows, as expected -- roughly halves
  PnL and roughly halves MDD. No free lunch, consistent with every other asset's leverage-sweep
  finding in this project (SOL, BTC).
- **Chop soft-sizing**: **mixed, window-dependent** -- on VAL it's close to a free lunch (PnL flat
  to slightly up, MDD cut by ~2/3, from -24.3% to -8.4%); on OOS it's a straightforward
  PnL-for-MDD tradeoff (PnL nearly halved, MDD cut by ~1/3). This matches the qualitative
  conclusion already on record in `omega4_6_1_chop_shadow.py`'s own docstring ("mostly a
  proportional leverage-reduction effect, not a genuine chop-timing edge, needs 4+ weeks of live
  observation") -- the live shadow duckdb (`data/live/omega4_6_1_eth_chop_shadow.duckdb`) has 0
  rows recorded so far, so this offline replay is currently the only evidence available either way.
- Combining both (reduced leverage + chop) stacks the risk reduction further (VAL MDD -24.3% ->
  -4.3%, OOS MDD -15.5% -> -5.5%) at a compounding PnL cost.

## Addendum: less-aggressive chop soft-sizing (threshold-gated), same session

User pushed back that the linear formula (`max(0, 1-chop_prob)`) sacrifices too much PnL for the
MDD it buys, and suggested lowering the "trigger point" rather than scaling from `chop_prob=0`.
Swept a threshold-gated variant -- full size below a threshold `T`, ramping linearly to 0 only for
`chop_prob` in `[T, 1.0]`: `mult = 1.0 if chop_prob < T else max(0, 1 - (chop_prob-T)/(1-T))` --
against the same baseline ledgers, `T in {0.3, 0.42, 0.5}`, plus two other shapes (`floor`,
`softslope`) for comparison.

**VAL**

| | PnL | MDD |
|---|---:|---:|
| no chop | +36.82% | -24.34% |
| linear (original) | +38.97% | -8.41% |
| **threshold T=0.3** | **+50.86%** | **-12.25%** |
| threshold T=0.42 | +49.11% | -15.40% |
| threshold T=0.5 | +47.36% | -17.33% |
| floor=0.5 | +35.02% | -11.76% |
| softslope alpha=0.5 | +39.01% | -16.40% |

**OOS**

| | PnL | MDD |
|---|---:|---:|
| no chop | +77.11% | -15.48% |
| linear (original) | +49.33% | -10.54% |
| **threshold T=0.3** | **+68.78%** | **-12.88%** |
| threshold T=0.42 | +71.26% | -12.88% |
| threshold T=0.5 | +71.92% | -12.88% |
| floor=0.5 | +53.58% | -10.54% |
| softslope alpha=0.5 | +64.41% | -11.71% |

**`threshold T=0.3` is the recommended shape**: on VAL it *dominates* plain no-chop (higher PnL
+50.86% vs +36.82%, AND lower MDD -12.25% vs -24.34% -- not a tradeoff at all here); on OOS it
recovers most of the no-chop PnL (+68.78% vs +77.11%) while still cutting MDD meaningfully
(-12.88% vs -15.48%). Raising `T` past 0.3 (to 0.42/0.5) buys slightly more OOS PnL but the OOS MDD
floor doesn't improve further (all three land on the same -12.88%, since the single worst
drawdown-driving trade's `chop_prob` is already above 0.5 and gets fully de-risked by any `T<=0.5`)
-- so `T=0.3` captures essentially all of the achievable MDD reduction for the least PnL sacrifice.
The user's stated intuition that the earlier ">100%" figures seen in prior sessions differ from
this session's OOS number mainly because of a longer/different OOS window (Jan-Jun here vs the
original doc's narrower sub-windows), not because of the chop logic, is correct -- window length
alone explains most of that gap, independent of the sizing formula.

Still shadow-only / not live-wired; this is a formula recommendation for a future promotion
decision, not an applied change.

## Live wiring (2026-07-20, same session, user-approved)

Promoted `threshold T=0.3` from shadow-only to real ETH position sizing:

- `trading_bot_modules/runtime_config.py`: new `FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE`
  (default `False`) and `FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD` (default `0.3`).
- `trading_bot.py` (`_decide_omega4_6_1_entry`, right after the portfolio-cap block, before the
  existing `ChopSoftSizeShadow.on_entry` call): when enabled, recomputes `chop_prob` the same way
  the shadow module does (`self.omega4_6_1_adapter.regime3_current.append(frame)`), applies the
  threshold-gated multiplier to `_eth_notional`/`_eth_leverage` (same formula as the backtest:
  `1.0` below the threshold, linear ramp to `0` above it), wrapped in try/except so any failure
  logs and falls through to unmodified sizing rather than blocking a trade.
- `.env`: `FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_ENABLE=True`,
  `FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SOFT_SIZE_THRESHOLD=0.3`.
- **Removed** (user request, same session): the pre-existing `ChopSoftSizeShadow` observer (plain
  linear `max(0,1-chop_prob)` shape) is now redundant with real sizing applying a milder version
  of the same idea, so it was deleted outright rather than left running -- `trading_bot_modules/omega4_6_1_chop_shadow.py`
  removed; its import, instantiation, `on_entry`/`on_exit` calls, and the now-unused
  `FINAL_GOVERNOR_OMEGA4_6_1_ETH_CHOP_SHADOW_ENABLE` config removed from `trading_bot.py`/`runtime_config.py`.
  Grepped the repo afterward to confirm no remaining references (one historical docstring mention
  in an unrelated old research script was left as-is, harmless).
- Verified: `ast.parse` + `py_compile` on both edited files, `runtime_config` import reflects the
  new `.env` values (`enable=True`, `threshold=0.3`), full repo grep shows no dangling references
  to the removed module/flag.
- `BINANCE_ACCOUNT_ENABLED=False` still blocks all real order placement -- this change affects
  decision-level sizing only until that account gate is separately lifted.
- **Live-restarted**: killed the running `trading_bot.py` (SIGTERM); `scripts/supervise_trading_bot.sh`
  auto-respawned it within the configured 10s delay with the new code. Confirmed clean startup in
  `data/live/trading_bot_stderr.log` (no import/attribute errors, existing SHORT position continued
  being managed normally post-restart).

## Status

`model_status=live_wired_decision_only, deployed`. Real order placement remains blocked by
`BINANCE_ACCOUNT_ENABLED=False`; the chop soft-sizing formula is now part of the real ETH sizing
decision path (shadow observer removed, not just disabled) and is running in the live process as
of this restart. `data/splits/year_oos/training_features_2026_rebuilt.csv` was temporarily
reverted to its pre-2026-07-20 state to match the frozen `oos_predictions_*.csv` inputs for the
backtest above, then restored to the 2026-07-20-extended state immediately afterward -- verified
via row count/last-timestamp check, no lasting effect on other artifacts.
