# Deep Entry Owner V1 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: GRU sequence encoder over 5m market/AI features.
- Unsupervised layer: KMeans state over deep embeddings and directional heads.
- Supervised heads: long/short expectancy and adverse-risk regressors.
- Execution: standalone long/short entry owner, one position at a time, fixed horizon, gross cap 3.6.

## Runtime Invariants

- Forbidden event candidate labels/sides/margins are excluded.
- Future prices are used only for train/validation labels, never runtime features.
- OOS threshold selection is forbidden.
- fee/slippage are charged on entry and exit.
