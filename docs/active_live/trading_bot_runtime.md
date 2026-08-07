# Trading Bot Runtime

Last updated: 2026-07-01 KST

## Entry Point

Active bot entry point:

- `trading_bot.py`

The bot is normally launched with the `quant_ai` conda environment. Production default paths are used when no `TRADE_JOURNAL_PATH`, `DASHBOARD_STATE_PATH`, or model override environment variables are set.

## Process Lock

`trading_bot.py` now acquires an exclusive process lock at startup.

Default lock path:

- `data/live/trade_journal.lock`

The lock path is derived from `TRADE_JOURNAL_PATH`. This allows isolated shadow bots to run only when they use isolated ledgers. A second bot using the same production journal exits immediately with code `2`.

Reason:

- DuckDB files such as `data/live/microstructure.duckdb` and `data/live/tail_risk.duckdb` are single-writer.
- Duplicate production bots caused lock conflicts and inconsistent dashboard/Telegram behavior.

## Core Live Files

| File | Purpose |
|---|---|
| `data/live/trade_journal.jsonl` | Production trade open/close journal. Dashboard position should be derived from this and runtime state. |
| `data/live/position_accounting_audit.jsonl` | Accounting-level audit rows for open, close, resize, costs, exposure, leverage, and realized PnL. |
| `data/live/dashboard_state.json` | Full dashboard state. |
| `data/live/dashboard_state_governor.json` | Compact governor/dashboard state. |
| `data/live/final_governor_runtime_state.json` | Runtime owner state, cooldowns, and active position context. |
| `data/live/pending_next_open_intent.json` | Pending next-open intent when scheduling is used. |
| `data/live/microstructure.duckdb` | Microstructure one-minute data written by `MicrostructureScanner`. |
| `data/live/tail_risk.duckdb` | Tail-risk/liquidation data written by `TailRiskInterceptor`. |

Shadow or testnet launchers must set their own isolated versions of these paths. They must not share production DuckDB or production journals.

## Decision Flow

```mermaid
sequenceDiagram
    participant Loop as Live loop
    participant Prep as Feature prep
    participant Gov as FinalGovernorRuntime
    participant PM as Position manager
    participant Ledger as Journal/Audit
    participant Dash as Dashboard state

    Loop->>Prep: completed 5m bars + live context
    Prep->>Gov: prepared frame with AI, regime, micro/tail features
    Gov->>Gov: Parent signal build (Omega1.2.1 by default, Omega4.6.2 source parent when explicitly enabled)
    Gov->>Gov: Omega5 validation-only live overlay
    Gov->>Gov: Omega5 CASH/veto terminates without fallthrough
    Gov->>PM: action, exposure, fraction, leverage, info, regime
    PM->>Ledger: OPEN/CLOSE/RESIZE/HOLD audit rows
    PM->>Dash: position, unrealized PnL, TP/SL, decision trace
```

## Position Signal Semantics

`FinalGovernorRuntime.decide()` returns a decision tuple used by downstream position management:

- `action`: integer action. Current convention is `0=CASH/HOLD`, `1=LONG`, `2=SHORT` for learned governors.
- `exposure`: target notional exposure.
- `fraction`: target position fraction/margin fraction.
- `execution_leverage`: leverage used to map notional to margin fraction.
- `info`: decision metadata, including `position_signal`, `source`, `reason`, model details, TP/SL/max-hold, confidence, and audit fields.
- `regime`: current regime label.

Important `position_signal` values:

- `LONG_ENTRY`
- `SHORT_ENTRY`
- `HOLD_LONG`
- `HOLD_SHORT`
- `EXIT`
- `REDUCE`
- `HOLD`

Do not infer position state only from Telegram messages. The production source of truth is the production journal and dashboard state.

## Next-Bar Timing Contract

Active runtime uses completed-bar signal and next-bar execution:

- Decision uses completed 5m bar `i`.
- Execution is evaluated on next bar `i+1`.
- Late execution beyond the configured delay may be skipped.
- Current in-progress candle must not be used as a decision feature row.

## Current Live Process

As of this documentation update:

- active Omega5 stack: blocked
- blocked model stack: `omega5_event_risk_governor_20260702`
- block report: `docs/audits/omega5_live_promotion_blocked_20260702.md`
- Omega5 source parent switch: `FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE=false`
- legacy Omega1.2.1 adapter: still loadable for the separate Omega1.2.1 path, but forbidden as an Omega5 parent substitute.
- default Alpha7 fully-learned path: disabled while Omega1.2.1 is enabled.
- Omega5 source model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`.
- Omega5 feature-frame DuckDB table: `decision_feature_frame_omega5_event_risk_governor_20260702`.
- latest promotion block: `docs/audits/omega5_live_promotion_blocked_20260702.md`.
- latest backtest parity audit: `docs/audits/omega5_live_backtest_parity_20260701.md` with `LIVE_BACKTEST_PARITY_BLOCKED`.

The Omega5 path is not active. Setting `FINAL_GOVERNOR_OMEGA5_ENABLE=1` fails fast because the promoted selection path is blocked by validation/test ledger dependence. Source backtest PnL must not be claimed for live promotion until Omega5 is rebuilt without ledger contamination and passes fresh holdout or walk-forward validation.

PID values are ephemeral; always verify with:

```bash
ps -eo pid,ppid,lstart,etime,cmd | rg 'python trading_bot.py|run_alpha7'
```

## Operational Checks

Use these checks before debugging model behavior:

```bash
ps -eo pid,ppid,lstart,etime,cmd | rg 'python trading_bot.py|run_alpha7'
tail -n 5 data/live/trade_journal.jsonl
tail -n 40 logs/trading_bot*.log
cat data/live/trade_journal.lock
```

If `trading_bot.py` appears more than once with the same production paths, kill the duplicate first. Do not interpret model decisions while two production writers are active.
