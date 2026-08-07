# ETH HMM Confluence Labels v2 — Experiment Result

## Changes from v1

- Removed chop/range-reversal candidates.
- Replaced one-bar OR confluence with HMM persistence → pullback touch → directional reclaim.
- Added cost-aware structural stops, bounded risk targets, and causal next-open transition exits.
- Added net-R, path-quality, and three-class labels.
- Added a train/validation-only parameter search and a separate meta-filter search.

## Locked base policy

The 2025-only search selected a 6-bar HMM window with 4 matching states, mean class probability 0.60, 6-bar pullback, 0.30 ATR stop buffer, 1.75R target, transition entry risk at most 0.45, 72-bar horizon, and transition exit threshold 0.75.

Diagnostic non-overlapping results:

| Split | Trades | Win rate | Compounded return | MDD |
|---|---:|---:|---:|---:|
| Train | 132 | 47.7% | +21.4% | -12.4% |
| Validation | 67 | 49.3% | +28.8% | -6.6% |
| OOS first touch | 60 | 28.3% | -20.2% | -23.0% |
| Fresh diagnostic | 52 | 21.2% | -23.7% | -23.6% |

The base policy therefore fails out of sample and is not promotable.

## Locked meta-filter

A bounded 2025-only search over logistic success classification and net-R regression selected ridge net-R regression (`alpha=1.0`) with a train-score median cutoff.

| Split | Selected trades | Win rate | Compounded return |
|---|---:|---:|---:|
| Train | 71 | 60.6% | +54.3% |
| Validation | 30 | 56.7% | +24.6% |
| OOS first touch | 33 | 30.3% | -11.4% |
| Fresh diagnostic | 23 | 30.4% | -6.8% |

The meta-filter also fails out of sample. A train/validation-selected 3% drawdown circuit breaker reduced but did not reverse the diagnostic losses (OOS -9.1%, fresh -7.4%).

## Conclusion

The sequential formulation materially improves 2025 development results but does not generalize after January 2026. No further threshold tuning against the consumed 2026 periods is valid. The current HMM plus available VWMA/VPVR/OI/funding proxies is insufficient evidence for a profitable promotion candidate. A new untouched forward period and materially new causal information—such as liquidation concentration or a retrained walk-forward regime model—are required for a valid next test.

## V3 microstructure follow-up

A final development-only follow-up added unused causal proxies for position hunting and liquidity: top-trader long/short ratios, taker flow, CVD, VWAP distance/reclaim, compression release, OI/funding divergence, wick/sweep signals, and liquidity vacuum. Ridge regularization and the selection quantile were chosen through four chronological folds inside 2025 before loading 2026.

The locked V3 policy produced +135.1% on the combined 2025 development sequence, but remained negative on both consumed 2026 diagnostics: -13.5% on January–March and -16.1% afterward. This confirms that adding these proxies does not repair the forward regime break. V3 is explicitly marked `promotion_eligible=false`.
