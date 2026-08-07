# Alpha3 CSV-Native Backtest Parity Protocol

Last updated: 2026-05-16 KST

## Purpose

This protocol records the exact frozen backtest environment used to align Alpha3 CSV canonical behavior with the native parity harness.

It exists so future Alpha3 experiments can change one declared layer while keeping data, model outputs, execution, accounting, and ledger semantics fixed.

## Canonical One-Month Baseline

Reference:

- Report: `data/ensemble/reports/alpha3_csv_canonical_aligned_native_1m_20260516.json`
- Ledger: `data/ensemble/reports/alpha3_csv_canonical_aligned_native_1m_20260516_ledger_corrected.csv`

Aligned window:

- Full-frame start index: `6999`
- Full-frame stop index: `15638`
- Start timestamp: `2026-01-25 07:15:00`
- Stop timestamp: `2026-02-24 07:10:00`
- Rows: `8640`

Baseline cost1:

```json
{
  "pnl": 338.68067144958997,
  "mdd": -29.617312952777574,
  "trades": 114,
  "wr": 0.3157894736842105,
  "deep_entries": 75,
  "long_entries": 65,
  "short_entries": 49,
  "avg_notional": 2.1901128532086536,
  "runner_actions": {
    "v21_entry": 39,
    "v21_reject": 15,
    "deep_entry": 75,
    "v21_add_on": 9
  },
  "route_counts": {
    "signal_immediate_maker_limit": 236,
    "forced_end_market": 1
  }
}
```

## Required Command

```bash
venv/bin/python scripts/backtest_alpha3_runtime_native_20260515.py \
  --start-index 6999 \
  --end-index 15638 \
  --alpha3-csv-loop-parity \
  --alpha3-csv-execution-parity \
  --alpha3-csv-state-parity \
  --alpha3-csv-decision-parity \
  --alpha3-csv-cooldown-parity-env \
  --report-out data/ensemble/reports/alpha3_runtime_native_backtest_live_frame_alpha3_csv_loop_exact_20260516_1m.json \
  --ledger-out data/ensemble/reports/alpha3_runtime_native_backtest_live_frame_alpha3_csv_loop_exact_20260516_1m_ledger.csv \
  --compare-csv-ledger data/ensemble/reports/alpha3_csv_canonical_aligned_native_1m_20260516_ledger_corrected.csv \
  --progress 0
```

## Passing Result

The run must report:

```json
{
  "action_events_match": true,
  "first_action_diff": null,
  "reference_action_events": 237,
  "candidate_action_events": 237,
  "final_cash_diff": 0.0,
  "final_equity_diff": 0.0,
  "final_pnl_diff_pct": 0.0
}
```

A stricter full-ledger check must also pass for:

- `i`
- `timestamp`
- `event`
- `pos`
- `owner`
- `reason`
- `route`
- `cash`
- `equity`
- `unrealized`
- `realized_pnl_frac`
- `realized_pnl_pct`
- `notional`

with numeric tolerance no larger than `5e-9`.

## Frozen Loop Semantics

The parity mode intentionally runs the canonical CSV loop. It does not call the live bot as the source of truth for strategy behavior.

Frozen decisions:

- HGB parent decision frame is computed over the full 2026 evaluation frame.
- Alpha2.1 teacher decision frame is computed over the full 2026 evaluation frame.
- V27 q values are computed over the full 2026 evaluation frame.
- The one-month window is sliced only after those full-frame outputs are built.

Frozen state:

- Position/accounting state starts flat at the aligned start timestamp.
- Parent and deep cooldown state follows canonical CSV semantics.
- V21.2 add-on state is one-shot per parent position.
- Deep scout can enter only when parent path is cash and deep cooldown allows it.

Frozen execution:

- Config: `next_open_limit_touch0_fee20`
- Entry miss: `skip`
- Exit miss: `market_fallback`
- Maker fee multiplier: `0.20`
- Fee: `0.0005`
- Slip: `0.0002`
- Route label for maker fills: `signal_immediate_maker_limit`
- Forced final route: `forced_end_market`

Frozen accounting:

- Entry fee is paid at open/add-on.
- Position mark uses CSV exit-side slippage only.
- Close row logs pre-close mark unrealized and post-close cash.
- Forced-end row logs final cash plus the canonical final mark/equity display.
- Same-bar V21.2 add-on emits `UPSIZE` and then a duplicate `HOLD` row at the same `i`.

## Known Non-Live Assumptions

This protocol is not a live promotion protocol.

Known non-live assumptions:

- Maker fills are simulated from OHLCV high/low touch, not real L2 queue position.
- The feature frame uses precomputed CSV features; live recomputation can drift.
- The parity mode uses a canonical CSV loop, not the full `FinalGovernorRuntime.decide()` path.
- The one-month window uses full-frame sequence context before slicing.

## Live Bot Application

As of 2026-05-16, `trading_bot.py` applies the live-compatible parts of this protocol by default when Alpha3 canonical mode is enabled:

- Parent decisions use full-frame inference and consume the latest row.
- Alpha2.1 teacher decisions use full-frame 72-bar sequence inference and consume the latest row.
- V31 scout decisions use the latest sequence from the full available frame.
- V21.2/V31 active-position exits use CSV-style gross mark PnL with exit-side slippage.
- Parent and deep cooldowns preserve the canonical CSV Alpha3 cooldown behavior.
- V21.2 jackpot add-on uses the canonical parent-bundle feature frame and the Alpha2.1-constrained parent decision row, not a parent-only row.
- V21.2 jackpot add-on state includes cumulative `MFE` and cumulative `MAE`; do not replace `mae_so_far` with `min(0, current_unrealized)`.
- V21.2 jackpot add-on remains a one-shot evaluation per parent position. A reject is still a consumed add-on attempt.
- V31 deep cooldown decrements before entry evaluation; when the decrement reaches zero, the same bar can evaluate V31 entry.
- Live execution remains `signal_close_next_open` with post-only maker routing, entry miss skip, and reduce-only exit market fallback where configured.
- Runtime logs include `alpha3_csv_native_backtest_parity_20260516_live`, `mark_contract`, and `cooldown_contract` fields.

This does not copy the offline CSV loop into live trading. Live still cannot know future high/low touch outcomes; the exchange adapter submits post-only maker orders and records actual route/fill metadata.

## Runtime-Native Parity Check

The live-runtime backtest path must also reproduce the one-month CSV action ledger before candidate results are interpreted. This path calls `trading_bot.FinalGovernorRuntime.decide()` sequentially, while using the frozen Alpha3 execution/accounting parity switches.

Current passing command:

```bash
venv/bin/python scripts/backtest_alpha3_runtime_native_20260515.py \
  --start-index 6999 \
  --end-index 15638 \
  --accelerated-cache \
  --alpha3-csv-execution-parity \
  --alpha3-csv-state-parity \
  --alpha3-csv-mark-parity \
  --alpha3-csv-cooldown-parity-env \
  --report-out data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m.json \
  --ledger-out data/ensemble/reports/alpha3_runtime_native_trading_bot_logic_after_mae_forced_fix_fast_20260516_1m_ledger.csv \
  --progress 0
```

Passing result as of 2026-05-16:

```json
{
  "first_action_diff": null,
  "reference_action_events": 237,
  "candidate_action_events": 237,
  "reference_event_counts": {"OPEN": 114, "CLOSE": 113, "UPSIZE": 9, "FORCED_END": 1},
  "candidate_event_counts": {"OPEN": 114, "CLOSE": 113, "UPSIZE": 9, "FORCED_END": 1},
  "reference_pnl_pct": 338.68067144958997,
  "candidate_pnl_pct": 338.67987283290336,
  "pnl_diff_pct": -0.000798616686648046,
  "reference_mdd_pct": 29.617312952777574,
  "candidate_mdd_pct": 29.617095389040106
}
```

The remaining difference is accounting-level floating-point/display drift, not action-path drift. If a future candidate changes `OPEN`, `CLOSE`, `UPSIZE`, or `FORCED_END` count before its declared mutable layer is reached, the candidate result is invalid.

Runtime-native harness invariants:

- `meta_router.cur_equity` and `meta_router.peak_equity` must be updated from current mark equity before each `decide()` call.
- Active V21.2/V31 marks use the CSV gross mark with exit-side slippage.
- Forced final close uses the same `stop` index as the canonical CSV ledger, not `stop + 1`.
- The runtime state file and router state file must be isolated per run.

## Failure Handling

If a candidate run fails this protocol:

1. Do not interpret candidate PnL.
2. Record the first differing action event.
3. Record whether the first mismatch is timestamp, event, side, owner, reason, route, notional, cash, or equity.
4. If the mismatch is not the declared mutable layer, fix the harness before rerunning the candidate.
5. If the mismatch is the declared mutable layer, compare candidate delta only after baseline reproduction is shown in the same report.

## Intended Use

Use this parity mode for:

- Parent replacement A/B tests that need a frozen downstream Alpha3 environment.
- Exit-layer tests that need identical entries and execution assumptions.
- RL exit-owner experiments that must prove the action/PnL delta starts at the RL layer.
- Debugging backtest/live discrepancies at action-ledger resolution.

Do not use this parity mode alone for:

- Live promotion.
- Real exchange fill claims.
- L2 queue fill claims.
- Validating the deprecated `+747.76%` open-fallback baseline.
