# ETH Full-History Oracle Strategy Label Contract v1

## Purpose

This dataset is a hindsight training target. It is not an OOS strategy result and cannot be promoted directly. Future OHLC is intentionally used only to determine target actions and outcomes; all stored HMM and technical features remain decision-time causal.

## Action space

- Action: `SKIP`, `LONG`, or `SHORT`.
- Entry: next 5-minute bar open with adverse slippage.
- Stop distance: 0.50, 0.75, 1.00, or 1.50 times causal ATR192.
- Reward: 1.00R, 1.50R, 2.00R, or 3.00R.
- Maximum holding period: 12, 24, 48, or 96 bars.
- Exposure is reported per unit notional. No leverage or margin assumption is embedded in the label.

Each decision row therefore evaluates 128 executable trade actions plus `SKIP`.

## Execution

- Fees are charged on entry and exit.
- Adverse slippage is charged on entry and exit.
- Actual ETHUSDT funding settlements between entry and exit are included.
- A stop gap exits at the observed open.
- A favorable target gap exits at the frozen target.
- If TP and SL are both touched in one bar and open ordering does not resolve them, that action is invalid rather than optimistically ordered.
- A timeout exits at the configured future bar open.

## Oracle targets

`oracle_local_*` describes the best positive standalone action at each row. `oracle_dp_*` describes the globally selected non-overlapping path.

The dynamic program maximizes:

`sum(log(1 + net_return_per_notional))`

At every row it compares skipping one bar with every valid action followed by the already optimal value at that action's exit index. The reconstructed forward path can hold at most one position.

## HMM role

The exact current Regime3 bull/bear/chop probabilities, confidence, margin, entropy, six-bar persistence counts, transition risk, and churn risk are stored as causal features. HMM state never constrains the hindsight oracle action.

## Dataset integrity

- No train/validation/OOS split is used to create or select oracle labels.
- `future_rows_used_for_label=true` is required and expected.
- `future_rows_used_for_entry_features=false` is required.
- Stored trade ledgers and parent exit timestamps are not inputs.
- The last maximum-horizon rows are right-censored; initial ATR warmup rows are invalid.
- A later predictive model must introduce its own chronological train/validation/OOS split.
