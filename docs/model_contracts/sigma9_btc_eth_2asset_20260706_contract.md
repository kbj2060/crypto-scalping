# Sigma9 — BTC+ETH 2-Asset Book (scoped-down cross-sectional test, FAILED to beat ETH-Sigma6 alone)

Status: `research_negative_result_not_adopted`

Last updated: 2026-07-06 KST

Lineage: user asked whether Sigma6 is the best possible architecture; the highest-ranked next
idea proposed was multi-asset cross-sectional trend-following (diversify away from Sigma6's
single-asset chop-vulnerability). Data-availability check found only BTCUSDT and ETHUSDT have
local kline data, and funding-rate/OI/top-trader metrics only exist for ETHUSDT
(`binance_data/funding_rate/`, `binance_data/metrics/` are ETH-only). A broad cross-sectional
universe is not implementable locally, so this was scoped down to a **2-asset BTC+ETH sleeve
book**: apply the exact Sigma3/Sigma6 pipeline to BTC as a second, independent leg, and test
whether blending it with ETH-Sigma6 improves the book on a risk-adjusted basis.

## Design

- **BTC dataset** (`scripts/build_1h_trendscan_dataset_btc_20260706.py`): same 1h resample +
  `compute_features()` + trend-scanning labels (windows [3,6,12,24,36,48], threshold=2.5) as
  Sigma3, sourced from raw `binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv`. **28 features**
  (vs ETH's 38) because funding/OI/top-trader/close_btc-cross columns don't exist for BTC —
  `compute_features()` skips those branches automatically when the source columns are absent.
- **BTC signal**: 5-seed HGB ensemble, identical hyperparameters to Sigma3
  (`scripts/train_sigma9_btc_1h_ensemble_20260706.py`), trained 2024-01..2025-06, tape from
  2025-06-25 on.
- **BTC execution**: Sigma6's exact trend-follower barrier mechanics (trail 5xATR, hard stop,
  min-profit-arm 2xATR, max_hold 144, cooldown 3) but **`reg_mode="none"`** — no regime filter,
  because Regime3's HMM/CryptoMamba sidecars were trained on ETH-only features and have no BTC
  equivalent. This is an explicit, honest limitation, not an oversight.
- **Combined book**: 50/50 capital-weighted sleeves — `combined_equity(t) = 0.5*eth_equity(t) +
  0.5*btc_equity(t)`, each sleeve compounding independently at its own best VAL config and full
  margin (0.30), as if each half of the book is run by its own strategy.
  Scripts: `scripts/run_sigma9_btc_standalone_20260706.py`,
  `scripts/run_sigma9_combined_book_20260706.py`.

## Result 1: BTC standalone (no regime filter) is weak

VAL 2025-07..12, best of a 24-point (threshold x leverage x sl_atr) grid:

| Config | VAL cost1 | MDD | WR | trades |
|---|---|---|---|---|
| **BTC best (thr=0.60, lev=2, sl=1.5)** | **+16.6%** | -9.6% | 39.3% | 56 |
| ETH-Sigma6 (thr=0.70, lev=3, not_chop+stab, sl=2.5) | +34.3% | -14.2% | 37.5% | 32 |
| ETH-Sigma6 (thr=0.70, lev=4, not_chop+stab, sl=2.5) | +71.1% | -15.9% | 43.2% | 37 |

Every higher-leverage BTC config lost money on VAL (down to -39.0% at lev4/thr0.45) — the same
failure mode as ungated Sigma5 (trend-following without a chop filter bleeds). BTC's edge here is
real but much smaller than ETH's regime-gated edge, and confirms the regime filter (not the
signal or execution style) is what makes Sigma6 work.

## Result 2: blending BTC into the book made risk-adjusted return WORSE, not better

| Book | VAL cost1 | VAL MDD | return/MDD |
|---|---|---|---|
| ETH-Sigma6 lev3 alone | +34.3% | -14.2% | 2.42 |
| **50/50 ETH-lev3 + BTC** | **+25.7%** | **-11.6%** | 2.22 |
| ETH-Sigma6 lev4 alone | +71.1% | -15.9% | 4.47 |
| **50/50 ETH-lev4 + BTC** | **+44.1%** | **-13.1%** | 3.37 |

MDD did shrink in both blends (14.2%→11.6%, 15.9%→13.1%), but return shrank proportionally more
(34.3%→25.7% = -25%, 71.1%→44.1% = -38%) because BTC's own drag (weaker edge, higher trade
frequency without a chop filter) outweighs the diversification benefit. Return-per-unit-MDD is
worse for both blends than for ETH-Sigma6 alone.

## Conclusion: do not adopt; OOS not spent

The 2-asset diversification hypothesis, as implementable with the data actually available, does
NOT beat ETH-Sigma6 alone on VAL. Root cause: BTC lacks a working regime filter (no Regime3 HMM
exists for it), so its standalone edge is too weak and noisy to be a good diversifier — adding it
dilutes returns faster than it cuts drawdown. Per the one-shot OOS discipline, since VAL did not
improve, **the reserved 2026-03-02..06-30 OOS window was NOT scored for this idea** (no point
spending a one-shot look on a config that already lost on validation).

**Recommendation**: keep ETH-Sigma6 alone as the strongest generalizing result. If multi-asset
diversification is revisited later, it needs either (a) a BTC-specific regime model (would require
building a second HMM/CryptoMamba sidecar from BTC-only features — nontrivial new work, not
attempted here), or (b) sourcing a broader asset universe so cross-sectional ranking/breadth
(the original idea) becomes possible rather than a 2-asset blend.

## Honesty checklist

- `fresh_forward_bar_by_bar`: n/a (this is a VAL-only backtest, no OOS scored).
- `trade_ledgers_used_as_input`: false — decisions computed causally from tapes each run.
- No lookahead: BTC dataset uses the same causal 1h resample/label conventions as Sigma3 (verified
  by code reuse, not re-derived).
- OOS window peek count: unchanged (still 4 prior instances from Sigma3/4/5/6/7/8) — this result
  did not add a 5th peek since VAL failed the pre-registered "improve before touching OOS" gate.
