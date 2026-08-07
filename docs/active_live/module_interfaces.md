# Module Interfaces

Last updated: 2026-06-06 KST

This document records the active/live module contracts needed to modify or connect code without reverse-engineering the entire repository.

## `trading_bot.py`

### Process lock

Function:

- `_acquire_trading_bot_process_lock() -> None`

Inputs:

- `TRADE_JOURNAL_PATH`, default `data/live/trade_journal.jsonl`
- `TRADING_BOT_PROCESS_LOCK_PATH`, optional explicit lock path

Behavior:

- Acquires an exclusive non-blocking `fcntl` lock.
- Writes current PID into the lock file.
- Exits with code `2` if another process already holds the same lock.

### Main runtime class

Class:

- `FinalGovernorRuntime`

Key responsibilities:

- Load active model artifacts.
- Prepare regime/AI/tp-sl score features.
- Validate fully learned feature contracts.
- Run Omega1.2.1 live adapter when enabled.
- Run Alpha7 primary, fallback parent, and optional V31 deep-alpha fallthrough only when the Omega path is disabled or explicitly bypassed.
- Apply runtime config overlays.
- Return a position decision tuple to the position manager.

Important methods:

- `_load_fully_learned_runtime_config(path: str) -> dict[str, object]`
- `_apply_fully_learned_v31_runtime_config() -> None`
- `_apply_fully_learned_tp_sl_action_score(frame: pd.DataFrame) -> pd.DataFrame`
- `_fully_learned_decision_frame(frame, bundle=None) -> tuple[pd.DataFrame, pd.DataFrame] | None`
- `_fully_learned_contract_ok(frame: pd.DataFrame, policy: dict) -> tuple[bool, dict[str, object]]`
- `_apply_fully_learned_runtime_config(dec: pd.Series) -> pd.Series`
- `_fully_learned_latest_decision(frame: pd.DataFrame) -> pd.Series | None`
- `_decide_omega1_2_1_entry(frame: pd.DataFrame, *, regime: str, raw_regime: str) -> tuple[int, float, float, float, dict, str]`
- `_manage_omega1_2_1_position(*, meta_router, current_price: float, regime: str) -> tuple[int, float, float, float, dict, str]`

Fail-fast expectations:

- Missing runtime config path: `RuntimeError`.
- Runtime config `model_id` mismatch against `FINAL_GOVERNOR_ALPHA7_MODEL_ID`: `RuntimeError`.
- Missing required runtime config keys: `RuntimeError`.
- Missing model artifact or TP/SL score artifact: `RuntimeError`.
- Missing critical model features: block/contract failure, not silent aliasing.

## `trading_bot_modules/omega1_2_1_live.py`

Class:

- `Omega121LiveAdapter`

Runtime use:

- Builds Regime3 current HMM, Regime3 CMamba h6 sidecar, and Regime3 stability/risk features from the live 5m frame.
- Routes to the frozen Omega1.2 true 3-head TabM expert: `bull`, `bear`, or `chop`.
- Uses Direction + Quality outputs to choose CASH/LONG/SHORT.
- Applies the aggressive compensated risk transform: notional `min(base_notional * 2.0, 0.90)`, leverage `2.0`, TP/SL scaled by the realized notional ratio.

Contract:

- CUDA is required for the CMamba live path.
- Missing or non-finite Regime3/TabM input columns raise `RuntimeError`.
- Omega CASH is terminal; active code must not silently fall through to Alpha7 or legacy sleeves.

## `ensemble/fully_learned_governor_policy.py`

Imported by `trading_bot.py`:

- `FULLY_LEARNED_ACTION_CASH`
- `prepare_features`
- `predict_policy_frame`

Runtime use:

- `prepare_features(...)` converts the live frame into the model feature matrix using the artifact feature list.
- `predict_policy_frame(...)` returns action, side, quality, confidence, notional, leverage, TP, SL, and max-hold decision columns.

Contract:

- Feature list comes from the loaded policy artifact.
- Strict mode must not silently create missing active model features.
- If feature contract changes, retrain/regenerate artifact and update active specs.

## `microstructure_scanner.py`

Class:

- `MicrostructureScanner`

Active storage:

- env `QUANT_MICRO_DB_PATH`
- default `data/live/microstructure.duckdb`
- table `microstructure_1m`

Important method:

- `_db_insert(bucket_ts: datetime, sig: dict) -> None`

Important signal keys:

- `obi`
- `taker_buy_ratio`
- `nif_whale`
- `nif_retail`
- `eai`
- `oi_delta_pct`
- `funding_rate`
- `kelly_mult`
- `signal_bias`
- `data_stale`
- `depth_connected`
- `trade_connected`
- `poll_connected`
- `recent_trade_count_5m`
- `recent_trade_notional_5m`
- `recent_whale_count_5m`
- `valid_taker_flow`
- `valid_nif`
- `warmup_30m_ready`

Operational constraint:

- DuckDB is single-writer. Do not run multiple production bots against the same DB path.

## `tail_risk_interceptor.py`

Active storage:

- default `data/live/tail_risk.duckdb`

Purpose:

- Liquidation/tail-risk context used by live monitoring and dashboard summaries.

Operational constraint:

- Same DuckDB single-writer rule applies.

## `trading_bot_modules/live_io.py`

Imported helpers:

- `_append_jsonl`
- `_append_jsonl_many`
- `_atomic_write_json`
- `_file_age_sec`
- `_read_json_safe`

Use:

- Atomic dashboard state writes.
- JSONL append-only ledger/event files.
- File freshness checks for dashboard health.

Contract:

- Journal files are append-only unless explicitly reset/archived.
- Dashboard JSON is replace-on-write through atomic writes.

## `trading_bot_modules/position_accounting.py`

Imported helpers:

- `_accounting_equity_from_history`
- `_build_position_accounting_audit_row`
- `_price_return_frac`
- `_safe_float`

Use:

- Rebuild/validate equity and position accounting from journal history.
- Create audit rows for OPEN/CLOSE/RESIZE with costs, notional exposure, leverage, and realized PnL.

Required accounting behavior:

- Same-side resize must charge fee/slippage on delta notional.
- Closing a position must apply exit fee/slippage on leveraged notional exposure.
- Synthetic execution fields must remain explicit when exchange execution is disabled.

## `trading_bot_modules/binance_execution.py`

Class:

- `BinanceFuturesExecutionAdapter`

Current active setting:

- Main production bot has exchange execution disabled unless env flags enable it.

Relevant env flags:

- `BINANCE_ACCOUNT_ENABLED`
- `BINANCE_POSITION_SYNC_ENABLED`
- `BINANCE_EXECUTION_ENABLED`
- `BINANCE_EXECUTION_DRY_RUN`
- `BINANCE_EXECUTION_REQUIRE_TESTNET`

Contract:

- Shadow/testnet launchers must not silently enable mainnet execution.
- Execution audit path must be isolated when running non-production bots.

## Dashboard

Process:

- `dashboard/server.py --host 0.0.0.0 --port 8787`

Reads:

- `data/live/dashboard_state.json`
- `data/live/dashboard_state_governor.json`
- production journal/audit files as configured by dashboard code.

Contract:

- Dashboard displays production state unless launched/configured for an isolated shadow path.
- Telegram messages are not a source of truth for dashboard position state.
