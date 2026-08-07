# ETH HMM Confluence Meta-Label Contract v1

## Purpose

This research path creates supervised setup-quality labels for ETHUSDT 5-minute bars. The current Regime3 HMM is causal decision-time context; it does not supply the trade direction label and future regime predictions never own direction.

## Decision-time context

- Exact HMM artifact: `regime3_current_sensitive_hmm_wide24_2024.joblib`.
- Soft outputs: bull/bear/chop probabilities, confidence, margin, and entropy.
- HMM thresholds are fitted only on 2025-01-01 through 2025-08-31.
- High transition or churn risk vetoes a setup. Low-confidence rows route to `uncertain` and emit no candidate.
- Rolling VWMA, ATR, swing levels, and VPVR use only data available at the decision bar. VPVR for row `i` uses bars `[i-window, i-1]`.
- RSI divergence is exposed only after the second pivot has been confirmed by right-hand bars.

## Candidate routing

- `bull`: long trend-pullback candidates only.
- `bear`: short trend-pullback candidates only.
- `chop`: long or short range-reversal candidates only.
- Direction still requires price structure plus a confluence of VWMA/VPVR, multi-timeframe trend, RSI/divergence, volume/OI/funding, wick/reclaim, or liquidity-trap evidence.
- Target and stop are frozen at the decision bar from causal structure. Invalid geometry and insufficient reward-to-risk are rejected before labeling.

## Outcome contract

- Entry is the next 5-minute bar open with adverse slippage.
- Trend horizon is 96 bars; range horizon is 24 bars.
- Outcomes are `TP`, `SL`, `TIMEOUT`, or `AMBIGUOUS`.
- If target and stop touch in the same bar without tick ordering, the label is invalid (`AMBIGUOUS`); no optimistic or stop-first assumption is made.
- The binary training target is `label_success = 1` only when TP is first and net return after entry/exit fees, slippage, and actual ETH funding is positive.
- Labels that require bars beyond a dataset or split boundary are censored and invalid.

## Split and integrity contract

- Train: through 2025-08-31 23:55 UTC.
- Validation: 2025-09-01 through 2025-12-31 23:55 UTC.
- OOS: 2026-01-01 through 2026-03-31 23:55 UTC.
- Fresh diagnostic: 2026-04-01 through 2026-07-20 00:00 UTC.
- No saved trade ledger, saved parent exit timestamp, or future row is used to create an entry.
- Input hashes, HMM identity, fitted route thresholds, label parameters, and causal-integrity flags are recorded in `report.json`.
- Non-overlapping trade CSVs and their performance are diagnostic only. They are not model-promotion evidence.

## Asset contract

Version 1 is ETHUSDT-only. BTC or SOL must supply an exact per-asset HMM and matching sidecars; no fallback or cross-asset artifact substitution is permitted.
