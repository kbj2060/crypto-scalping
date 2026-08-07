# SOL/BTC chop soft-sizing transfer check -- negative result (2026-07-20)

## Motivation

ETH's newly live-wired chop soft-sizing (threshold-gated, T=0.3;
`docs/model_contracts/eth_leverage_chop_softsize_fresh_forward_20260720.md`) was a clear win on
VAL (dominates no-chop outright) and a reasonable tradeoff on OOS. Tested whether the same formula
transfers to SOL (v2 adaptive_squeeze, live) and BTC (v1, live), using each asset's own live
regime3-wide24 `chop_prob` source and their own frozen VAL/OOS ledgers
(`tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720/`,
`tmp/causal_regen_20260516/btc_final_scale_map_20260708/`) -- a pure post-hoc ledger rescale
(`trade_return * mult(chop_prob, T)`), same causal-safe methodology as the ETH test.

## Method

Threshold-gated multiplier: `mult = 1.0 if chop_prob < T else max(0, 1-(chop_prob-T)/(1-T))`.
Swept `T in {0.2, 0.3, 0.4, 0.5, 0.6}` against each asset's own VAL ledger (selection basis, per
this project's no-peeking convention), OOS reported for context only, not for selection.

## Results

**SOL** (VAL 42 trades, OOS 59 trades)

| T | VAL PnL/MDD | OOS PnL/MDD (context only) |
|---|---:|---:|
| none | +16.75% / -26.29% | +57.94% / -21.35% |
| 0.2 | +16.75% / -23.18% | +40.06% / -19.15% |
| **0.4 (best VAL)** | **+18.49% / -25.93%** | +46.92% / -20.48% |
| 0.6 | +16.88% / -26.29% | +49.86% / -20.99% |

**BTC** (VAL 16 trades, OOS 30 trades)

| T | VAL PnL/MDD | OOS PnL/MDD (context only) |
|---|---:|---:|
| none | +7.45% / -11.93% | +22.69% / -15.88% |
| **0.2 (best VAL)** | **+2.66% / -11.03%** | +24.93% / -11.67% |
| 0.3 | +2.05% / -11.26% | +25.92% / -12.57% |
| 0.6 | +2.57% / -11.93% | +24.14% / -15.88% |

## Conclusion: does not transfer, do not apply

- **BTC: reject.** Every threshold roughly halves-or-worse VAL PnL for essentially no MDD benefit
  (best case -11.93%->-11.03%). OOS looks attractive at every threshold, but selecting on OOS after
  VAL rejects the idea would be exactly the look-ahead/selection-bias pattern this project's own
  Fresh-Forward discipline exists to prevent. Not applied.
- **SOL: inconclusive, not worth adopting.** Best VAL threshold (T=0.4) gives a small PnL gain
  (+16.75%->+18.49%) with negligible MDD improvement (-26.29%->-25.93%) -- nothing like ETH's clean
  win -- while OOS gives up a meaningful chunk of PnL (+57.94%->+46.92%). Not a compelling
  trade either way; added complexity isn't justified. Not applied.

This is consistent with this project's repeated finding that ETH-derived constants/techniques do
not automatically transfer to SOL/BTC (see the funding-divisor magic-number case and the failed
BTC adaptive_squeeze fresh-forward test, both same general pattern) -- chop soft-sizing is another
instance of the same lesson, this time on the risk-sizing side rather than the feature side.

## Status

`model_status=research_negative_result_not_adopted`. No `trading_bot.py`/`.env`/live config changes
for SOL or BTC. ETH's own chop soft-sizing (T=0.3) remains live as documented separately.
