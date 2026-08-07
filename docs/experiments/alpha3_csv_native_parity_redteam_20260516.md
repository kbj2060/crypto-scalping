# Alpha3 CSV vs Native Parity Red Team Audit

Last updated: 2026-05-16 KST

## Verdict

`pass_for_csv_loop_parity_baseline`

This audit does not promote the synthetic maker-fill Alpha3 execution model to live trading. It verifies that the new parity harness can reproduce the canonical CSV Alpha3 one-month ledger exactly, so future layer experiments can be compared under one frozen environment.

Live promotion remains blocked until real L2 queue/fill evidence validates the synthetic `next_open_limit_touch0_fee20` maker assumptions.

## Scope

- Workspace: `/home/llewyn/crypto-scalping`
- Baseline ledger: `data/ensemble/reports/alpha3_csv_canonical_aligned_native_1m_20260516_ledger_corrected.csv`
- Parity report: `data/ensemble/reports/alpha3_runtime_native_backtest_live_frame_alpha3_csv_loop_exact_20260516_1m.json`
- Parity ledger: `data/ensemble/reports/alpha3_runtime_native_backtest_live_frame_alpha3_csv_loop_exact_20260516_1m_ledger.csv`
- Backtest range: `2026-01-25 07:15:00` through `2026-02-24 07:10:00`
- Full-frame indices: `6999` through `15638`
- Rows in sliced test window: `8640`

## Final Parity Result

The parity harness now matches the corrected canonical CSV ledger at action and full ledger level.

| Check | Reference | Candidate | Diff |
|---|---:|---:|---:|
| Action events | `237` | `237` | `0` |
| Ledger rows | `8648` | `8648` | `0` |
| Final cash | `4.3868067144959` | `4.3868067144959` | `0.0` |
| Final equity log | `4.440568861658766` | `4.440568861658766` | `0.0` |
| PnL | `+338.68067144958997%` | `+338.68067144958997%` | `0.0%` |
| MDD | `-29.617312952777574%` | `-29.617312952777574%` | `0.0%` |
| Trades | `114` | `114` | `0` |

Event counts also match:

```json
{
  "HOLD": 7523,
  "COOLDOWN": 888,
  "OPEN": 114,
  "CLOSE": 113,
  "UPSIZE": 9,
  "FORCED_END": 1
}
```

## What Was Actually Wrong

### 1. CSV canonical and live-native were not the same simulator

The large gap was not one single model failure. The CSV baseline and the live-native runner were executing different contracts:

- CSV canonical computes parent, teacher, and V27 decision frames over the full 2026 evaluation frame before slicing the one-month window.
- CSV canonical then resets position/accounting state at the aligned start timestamp.
- The native runtime loop calls `FinalGovernorRuntime.decide()` sequentially and updates live-style router/lifecycle state each bar.
- The two paths had different state transition order, mark-PnL accounting, add-on feature context, and forced-end indexing.

Consequence: comparing CSV PnL directly against an early native replay produced false deltas. The early native one-month run was about `+101.59%`, while the corrected CSV one-month baseline was `+338.68%`.

### 2. Sequence-model inference contract was underspecified

The Alpha2.1 teacher is a 72-bar sequence model. A latest-row-only live call can build a sequence equivalent to many zero rows plus the latest row, not the true `[t-71, ..., t]` lookback.

The full layer contract now requires:

```text
teacher_input(t) = normalized sequence over rows max(0, t-71) ... t
teacher_decision(t) = predict_frame(frame_until_t).iloc[-1]
forbidden: predict_latest(frame.tail(1)) for sequence models
```

This was an integration contract gap. It can change parent/teacher acceptance and therefore entries, exits, and cooldown timing.

### 3. Entry price accounting double-counted slippage in native mark/close math

The native replay stored maker-filled entry prices, then some mark/close paths treated those prices as if they still needed entry-side synthetic slippage. That double-counts entry friction for real route fills.

Fix direction:

- Preserve `entry_execution_liquidity` per lot.
- Treat `signal_immediate_maker_limit`, market fallback, and exchange/live route prices as real executed prices.
- Apply synthetic entry slippage only when the entry price is a raw decision/mark proxy.

This correction moved native-style parity much closer to the CSV path, but was not sufficient by itself because other contracts still differed.

### 4. V21.2 jackpot add-on used a different feature frame

CSV V21.2 add-on evaluation uses:

```text
_feature_frame(df_slice, parent_bundle, alpha2_decisions_slice, i, state)
```

The runtime-native path initially used the live adapter context, prepared frame, and different decision source. That changed jackpot accept/reject timing and resize count.

Parity fix direction:

- Precompute canonical parent decisions, Alpha2.1 decisions, and V27 q over the full frame.
- Slice them to the same one-month window.
- Feed the V21.2 add-on with the same `_feature_frame` inputs as CSV canonical.

### 5. Forced-end indexing differed

CSV canonical loops over `range(0, len(df) - 2)` and forces the remaining position out at `len(df) - 1`.

The native sequential loop can easily evaluate the last comparable bar as a normal bar and then force exit at a different absolute timestamp if the stop index is interpreted as an execution bar instead of a data-window end.

This is a classic off-by-one replay bug. It affects final forced exit price, final equity log, and sometimes the last action event.

### 6. Diagnostic ledger format was initially not equivalent

After action events and final PnL matched, the full ledger still differed in non-trading rows and close-row display fields:

- `UPSIZE` reason string: `v21_2_jackpot_add` vs canonical `v21_add_on`.
- `CLOSE` row `equity` and `realized_pnl_frac` were logged with different display semantics.
- `HOLD` vs `COOLDOWN` labels after parent closes differed even when no trade action changed.
- `UPSIZE` followed by same-bar duplicate `HOLD` needed to reuse the updated mark.

These were logging/parity issues, not strategy changes. They were fixed so future diff tooling can identify the true first behavioral divergence.

## Evidence Trail

Progression on the same one-month window:

| Run | PnL | MDD | Trades | Interpretation |
|---|---:|---:|---:|---|
| Early native live-frame replay | `+101.5907%` | `-33.3971%` | `122` | Different execution/accounting/state/decision contracts |
| Native exec+state parity | `+299.5257%` | `-27.7258%` | `112` | Major accounting/state gaps reduced |
| Full parity fix2 | `+313.0410%` | `-28.0623%` | `113` | Decision/add-on parity improved |
| Full parity fix4 | `+305.2356%` | `-29.7803%` | `114` | Still not exact; resize path differed |
| Experimental fix5 | `+210.8024%` | `-29.6295%` | `118` | Incomplete; one-shot reject alone exposed missing cumulative MAE state |
| CSV loop exact parity | `+338.6807%` | `-29.6173%` | `114` | Matches corrected canonical ledger |
| Runtime-native after MAE/forced-end fix | `+338.6799%` | `-29.6171%` | `114` | `FinalGovernorRuntime.decide()` action events match CSV; residual PnL diff `-0.0008%p` |

The `fix5` one-shot reject change was not wrong by itself. CSV canonical does consume a V21.2 add-on attempt after reject. The failure was that runtime-native V21.2 add-on state still passed current-only MAE instead of cumulative `mae_so_far`, so the reject gate fired on the wrong score. After cumulative MAE was added to runtime state, the missing add-on at local `i=3387` matched CSV.

## Runtime-Native Fix Log

These fixes are now part of the Alpha3 live/runtime parity contract and must not be removed during future model work:

1. Full-frame latest-row inference:
   Parent, Alpha2.1 teacher, and V31 sequence inference must run over the available frame and consume the latest row. Sequence models must never receive a single latest row padded by zeros.

2. Router mark state before decision:
   The native harness must set `meta_router.cur_equity` and `meta_router.peak_equity` from current mark equity before calling `FinalGovernorRuntime.decide()`. V21.2 add-on `drawdown_abs` depends on this.

3. V21.2 add-on input contract:
   Runtime add-on scoring must use the canonical parent-bundle feature frame, the Alpha2.1-constrained decision row, cumulative `MFE`, cumulative `MAE`, current unrealized PnL, and the original parent TP/SL/max_hold.

4. V21.2 one-shot semantics:
   A V21.2 add-on pass, miss, or reject consumes the add-on attempt for that parent position.

5. V31 cooldown order:
   Deep cooldown decrements before V31 entry evaluation. If decrement reaches zero, V31 can enter on that same bar.

6. Forced-end indexing:
   Runtime-native forced final close must use the canonical `stop` index. Using `stop + 1` changes the last forced exit price and final PnL.

Current runtime-native evidence:

- Report: `data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m.json`
- Ledger: `data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m_ledger.csv`
- Action events: `237` vs CSV `237`
- First action diff: `null`
- PnL: `+338.67987283290336%` vs CSV `+338.68067144958997%`
- Difference: `-0.000798616686648046%p`

## Red Team Blockers

These blockers remain for live promotion:

- `synthetic_ohlcv_touch_fill_not_real_l2_queue_fill`
- `maker_fill_probability_not_validated_by_forward_l2_snapshots`
- `csv_loop_parity_is_not_live_exchange_promotion`
- `live_feature_recompute_vs_precomputed_csv_features_still_requires_shadow_monitoring`

## Required Guardrails Going Forward

Any Alpha3 parent, exit, RL, or execution experiment must:

1. Declare the mutable layer.
2. Freeze every non-target layer.
3. Reproduce the baseline row in the same run.
4. Emit a ledger with `i`, timestamp, event, owner, reason, route, cash, equity, unrealized, realized PnL, and notional.
5. Compare action events against the frozen baseline ledger before interpreting candidate PnL.
6. Stop on the first action/PnL divergence unless the divergence is the declared experimental change.
7. Report whether execution uses synthetic OHLCV touch fills or real L2 queue evidence.

## Promotion Interpretation

The exact parity result is a debugging and research baseline. It means future tests can now isolate one layer under a frozen loop.

It does not mean:

- `+338.68%` is live-promotable PnL.
- OHLCV high/low touch equals real maker queue fill.
- The live bot can ignore feature recomputation drift.
- The old `+747.76%` open-fallback baseline is valid.

For live deployment, the next required gate is runtime-native shadow replay with real feature recomputation and real/sufficient L2 fill statistics.
