# BTC/SOL Low-Cost Tuning Sweep - 2026-07-13

Status: `research_diagnostic_not_live_wired`. Follows the same cheap-tuning methodology already
applied to ETH this session (`docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md`):
duration-gate on/off and a per-asset notional multiplier grid, solo (no portfolio interaction),
COST_MULT=1.0 real-cost convention, on the fully-regenerated/verified 2026-01-01..2026-07-12 data.
Does not change anything about `trading_bot.py` or the underlying trained models -- pure backtest
research to decide whether cheap parameter tuning (as opposed to a full model upgrade/retrain)
closes the gap between BTC/SOL's solo OOS performance and ETH's.

## Configuration swept

For BTC and SOL independently (solo, `enabled_assets=(asset,)`):
- Duration gate (`ou_halflife`): on (own original threshold) vs off (`-999.0`).
- That asset's own notional multiplier: 1.0x, 1.25x, 1.5x, 2.0x, 2.5x (mirrors the grid used for
  ETH's earlier 1.0-3.0x sweep, narrowed).

Script: scratchpad `test_btc_sol_lowcost_tuning_sweep.py`. Raw results:
`/tmp/btc_sol_lowcost_tuning_sweep_results.json`.

## BTC results

| config | VAL pnl | VAL mdd | OOSext pnl | OOSext mdd | trades | wr | Q1 pnl | Q1 mdd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate ON, 1.0x | 5.44% | -12.40% | 11.58% | -16.79% | 32 | 37.5% | 6.86% | -16.79% |
| gate ON, 1.25x | 6.30% | -15.33% | 13.13% | -20.78% | 32 | 37.5% | 7.71% | -20.78% |
| gate ON, 1.5x | 6.96% | -18.18% | 14.10% | -24.67% | 32 | 37.5% | 8.21% | -24.67% |
| gate ON, 2.0x | 7.69% | -23.69% | 14.26% | -32.14% | 32 | 37.5% | 8.14% | -32.14% |
| gate ON, 2.5x | 7.63% | -28.95% | 12.12% | -39.19% | 32 | 37.5% | 6.69% | -39.19% |
| gate OFF, 1.0x | 6.69% | -12.11% | 10.52% | -16.46% | 31 | 35.5% | 6.21% | -16.46% |
| gate OFF, 1.25x | 7.90% | -14.97% | 11.81% | -20.38% | 31 | 35.5% | 6.92% | -20.38% |
| gate OFF, 1.5x | 8.91% | -17.77% | 12.52% | -24.21% | 31 | 35.5% | 7.28% | -24.21% |
| gate OFF, 2.0x | 10.36% | -23.18% | 12.21% | -31.57% | 31 | 35.5% | 6.96% | -31.57% |
| gate OFF, 2.5x | 11.03% | -28.34% | 9.68% | -38.53% | 31 | 35.5% | 5.30% | -38.53% |

**Gate makes almost no difference for BTC** (gate off is marginally worse on trade count and WR,
essentially flat on PnL) -- unlike ETH and SOL, where disabling the gate clearly helped OOS.
**Multiplier is not a useful lever either**: OOS PnL peaks around 2.0x (~14%) and MDD roughly
triples from 1.0x to 2.5x (-16.8%→-39.2%) for a peak PnL gain of only ~3 points, then PnL declines
again at 2.5x -- pure notional scaling is not amplifying a real edge here, it's mostly amplifying
variance. Every single cell in this grid is far below ETH's solo OOS range (70-102%).

**Recommendation: no meaningful improvement available from this cheap tuning for BTC.** Keep the
existing config (gate on, 1.0x multiplier) as the practical choice, since the small PnL gains
elsewhere in the grid don't justify their MDD cost. BTC's weakness relative to ETH/SOL appears to
be a genuine signal-quality gap, not a tunable-parameter gap -- closing it would require deeper
model work (features, re-scoring quality thresholds, or new training), not cheap knob-turning.

## SOL results

| config | VAL pnl | VAL mdd | OOSext pnl | OOSext mdd | trades | wr | Q1 pnl | Q1 mdd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate ON, 1.0x | 96.74% | -13.64% | 12.56% | -20.10% | 52 | 38.5% | 21.96% | -17.71% |
| gate ON, 1.25x | 129.47% | -16.83% | 14.29% | -24.87% | 52 | 38.5% | 27.15% | -21.73% |
| gate ON, 1.5x | 166.09% | -19.94% | 15.41% | -29.52% | 52 | 38.5% | 32.14% | -25.59% |
| gate ON, 2.0x | 251.90% | -25.91% | 15.75% | -38.37% | 52 | 38.5% | 41.45% | -32.87% |
| gate ON, 2.5x | 355.57% | -31.57% | 13.64% | -46.58% | 52 | 38.5% | 49.64% | -39.58% |
| gate OFF, 1.0x | 32.23% | -24.98% | 28.56% | -19.43% | 59 | 39.0% | 22.90% | -17.03% |
| gate OFF, 1.25x | 39.49% | -30.70% | 34.59% | -23.76% | 59 | 39.0% | 28.11% | -21.00% |
| gate OFF, 1.5x | 46.24% | -36.17% | 39.98% | -27.91% | 59 | 39.0% | 33.02% | -24.85% |
| gate OFF, 2.0x | 57.83% | -46.33% | 48.56% | -35.64% | 59 | 39.0% | 41.80% | -32.19% |
| gate OFF, 2.5x | 66.37% | -55.40% | 53.77% | -42.67% | 59 | 39.0% | 48.96% | -39.05% |

**Gate-off clearly dominates gate-on for SOL, exactly the same pattern already found for ETH and
the 3-asset stress test**: at 1.0x, gate-off beats gate-on on OOS PnL AND OOS MDD simultaneously
(+28.56%/-19.43% vs +12.56%/-20.10%) while validation PnL is much *lower* for gate-off (32.23% vs
96.74%) -- the gate-on validation numbers are the same kind of overfit-to-validation artifact
already diagnosed for ETH/the 3-asset run, not a real edge.

**Multiplier, on the gate-off branch, is a genuine (if noisier) dial**: OOS PnL rises monotonically
1.0x→2.5x (28.56%→53.77%) with validation also rising monotonically (32%→66%, no sign inversion --
unlike ETH's multiplier sweep, which flipped validation negative above ~2.5x). MDD does grow
substantially though (-19.4%→-42.7%). Following the same "moderate pick over max-PnL pick"
principle used for ETH (1.5x chosen over the PnL-maximizing 3.0x): **recommend gate-off + 1.5x**
for SOL -- OOSext +39.98%/mdd -27.91%, Q1 +33.02%/mdd -24.85%, roughly 3x SOL's untuned OOS PnL
without pushing MDD into the -40%+ range the higher multipliers show.

## Overall conclusion

Cheap tuning **works for SOL** (same gate-off pattern as ETH, ~3x solo OOS PnL improvement at a
reasonable MDD) but **does not meaningfully help BTC** (gate is a non-factor, multiplier just
trades PnL for MDD with no net edge amplification). BTC remains well below ETH/SOL even after this
sweep -- the gap there looks structural, not parametric. Recommend: adopt gate-off + 1.5x for SOL
if/when wiring is revisited (mirrors the existing
`FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF` flag, which already covers the gate half;
would need a new SOL-specific multiplier env var, analogous to ETH's, for the multiplier half --
not implemented this session, kept as a clean follow-up since the user asked for the sweep/decision
first). Keep BTC's config as-is; treat "improve BTC" as a model-quality problem for a future,
separate research effort, not a tuning problem.
