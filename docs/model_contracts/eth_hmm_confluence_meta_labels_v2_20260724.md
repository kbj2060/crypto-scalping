# ETH HMM Sequential Pullback Meta-Label Contract v2

## Scope

V2 is an ETHUSDT 5-minute research label path. It removes the failed V1 chop/range-reversal route and emits only sequential trend-pullback candidates under the exact current Regime3 HMM artifact.

## Candidate state machine

1. The bull or bear route must persist for a configured count inside a trailing HMM window.
2. The matching soft probability must meet its trailing mean threshold.
3. VWMA288 must slope in the trade direction.
4. Price must touch the causal VWMA100/VPVR anchor during a trailing pullback window.
5. A strong candle must close back through the anchor in the trend direction.
6. Transition risk must be below the entry veto threshold.
7. Adjacent repeated true states emit only one trigger.

All rolling state ends at the decision row. Entry remains the next bar open with adverse slippage.

## Risk and outcome labels

- Stop: pullback-window extreme plus an ATR buffer.
- Target: fixed risk multiple selected on 2025 train/validation only.
- A candidate is rejected if its initial price risk is less than four times the modeled round-trip fee and slippage.
- An optional transition-risk exit is observed at a completed bar and executed at the next bar open.
- Barrier outcomes are `TP`, `SL`, `TIMEOUT`, `REGIME_EXIT`, and invalid `AMBIGUOUS`.
- Additional targets are `label_net_r`, `label_mfe_r`, `label_mae_r`, `label_path_quality`, and three-class `label_class`.

## Selection and evaluation

- Parameter search reads 2025 train and validation only.
- A parameter set must have at least 180 valid train labels and 100 valid validation labels.
- Both splits must have positive candidate mean return and positive non-overlapping diagnostic compounded return.
- The robust selection score is the smaller of train and validation mean return.
- The selected policy is printed and frozen before any 2026 input is loaded.
- 2026 OOS and fresh results are diagnostic after the first touch and cannot be used to retune the same test claim.

## Integrity

`fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`, and `future_rows_used_for_entry=false` are mandatory report fields. Stored trade CSVs are diagnostic outputs only.

