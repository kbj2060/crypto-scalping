# BTC Multi-Asset TFT Research Contract (2026-08-06)

Status: research-only. This does not modify the live stack or create a promotion candidate.

The model consumes a decision-time tensor `[batch, lookback, asset, feature]`, with BTC selected
by an explicit per-row asset index. Asset features cover OHLCV, funding, OI, and other causal
market-state fields. A separate `[batch, lookback, global_feature]` tensor is reserved for
Bitcoin-wide on-chain and options inputs; these fields must already encode their publication delay
before entering the model. Missing declared features are a contract error, not a zero-filled alias.

The supervised targets are (1) the 0.35% reversal-floor, 3-bar-minimum aggressive zigzag regime,
(2) the causal triple-barrier entry action, and (3) an aggressive-zigzag transition-risk target:
hold, exit-to-cash, or flip-direction inside the next 12 bars. The first transition target is
an exit-risk auxiliary signal, not an instruction to place or close an order. The triple-barrier
entry action comes from
`scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py`: decide at bar `t`, enter at
`t+1` open, and resolve the long/short TP-before-SL race over its fixed horizon. The TFT has
separate regime, entry, exit-risk, and ordered future log-return quantile heads.
Neither target may be used as a future feature, decision input, or test-selection criterion.

Interpretability artifacts are the VSN weights by time/asset/feature and the target asset's
cross-asset attention weights. They are diagnostic only and cannot override fail-fast feature
availability or fresh-forward evaluation rules.

Selection remains validation-only. Any candidate must pass the repository's fresh-forward,
single-position, price-move-to-notional accounting and Omega artifact-integrity gates before it
can be considered for promotion.
