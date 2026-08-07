# Multi-Asset (SOL/BTC) Real Execution Wiring Plan - 2026-07-12

Status: `plan_not_yet_implemented`. Decision confirmed with user: **ETH's existing real execution
path (FinalGovernorRuntime-based, gate-on, 1.0x) stays untouched.** Only SOL and BTC get new real
execution, using their existing shadow-config candidates (already artifact-integrity-audited
today: SOL zig075 q070, BTC h48qual q055 -- NOT this session's CURRENT_BASELINE tuning, which was
derived from a different, "native"-replay-engine implementation than what trading_bot.py's real
adapters use).

## What already exists and needs zero changes

- `Omega461LiveAdapter` (`trading_bot_modules/omega4_6_1_live.py`) -- already asset-parameterizable
  via `components_override`/bundle/sidecar/threshold/scale_map constructor args. The SOL/BTC shadow
  instances already prove this works.
- `BinanceFuturesExecutionAdapter` (`trading_bot_modules/binance_execution.py`) -- already
  generic/per-symbol via `self.symbol` from its own `BinanceLiveFetcher`. Safe to instantiate 3x.
- `GovernorPositionRouter` + `_bootstrap_virtual_router(router, live_state_path)` -- already used
  per shadow asset with its own state file (`data/ensemble/omega4_6_1_shadow_{asset}_state.json`).
- Dashboard schema (`asset_decisions`/`asset_states` keyed by asset) -- already supports per-asset
  display; just needs real (not shadow-only) data populated into it.
- NEW: `trading_bot_modules/portfolio_risk.py` (built today) -- `PortfolioRiskManager` gives each
  asset a fixed, non-competing notional budget (mirrors the validated `prealloc` cap_mode from the
  backtest research; avoids the v2 hard-reject pathology and the v3 shared-budget starvation
  pathology). Not yet called from anywhere.

## Operational preconditions to verify before touching trading_bot.py (must check, not assume)

1. **`BINANCE_EXECUTION_SYMBOL` and `BINANCE_ACCOUNT_SYMBOL` env vars must be unset/empty** in
   whatever `.env`/deployment config is actually used. Both are single process-wide overrides that
   would silently force every new SOL/BTC adapter (and fetcher) to use ETH's symbol if set. This
   is an environment check, not a code change -- confirm with whoever manages the live deployment
   config before enabling anything.
2. **`FINAL_GOVERNOR_RUNTIME_STATE_PATH` is irrelevant to this plan** since we are NOT
   instantiating `FinalGovernorRuntime` for SOL/BTC (see next section) -- no collision risk here.
3. **Binance API rate-limit budget**: 3 independent ccxt clients (ETH's existing one + 2 new ones)
   share one API key's exchange-side weight limit. ccxt's own rate limiter is per-client, not
   shared, so nothing in-process throttles the combined request rate across all 3. Should be
   monitored/staggered once live, not solved in code up front, but worth flagging to whoever
   operates this.

## Design: lightweight per-asset managers, NOT 3x FinalGovernorRuntime

`FinalGovernorRuntime` is a monolith that loads many ETH-specific, unrelated sleeves (macro trend,
fully-learned policy, Omega5, regime4 HMM/TFT, execution-policy bundle, microstructure/event
sleeves) and has a single shared `runtime_state_path` file with no per-instance override hook.
Instantiating it 3x would triple-load all of that irrelevant machinery and collide on that shared
state file. **Do not do this.**

Instead, for SOL and BTC, build a small new `Omega461AssetExecutor` (new class, e.g. in
`trading_bot_modules/omega4_6_1_asset_executor.py`) that wraps exactly the primitives the existing
shadow loop already proves work per-asset:

```
Omega461AssetExecutor(asset, symbol, adapter: Omega461LiveAdapter,
                       router: GovernorPositionRouter,   # via _bootstrap_virtual_router, own state file
                       fetcher: BinanceLiveFetcher,       # own instance, own symbol
                       executor: BinanceFuturesExecutionAdapter,  # NEW -- wraps `fetcher`
                       risk: PortfolioRiskManager)         # shared single instance across all assets
```

Its per-cycle method mirrors `_refresh_omega461_shadow_asset` almost exactly (same decide_entry/
evaluate_exit/router._update_pos sequence), with these changes at the exact points the research
identified:

- Before calling `router._update_pos(...)`, compute `target_exposure` the same way the shadow path
  already does, then call `approved_exposure = risk.scale_to_budget(asset, target_exposure)`.
- If `approved_exposure < risk.config.min_notional`: skip opening this cycle (log it), matching
  the shadow path's existing skip semantics.
- Otherwise call `await executor.execute_to_target(final_action=action, target_exposure=approved_exposure,
  target_exec_leverage=leverage, current_price=price, timestamp_kst=timestamp_kst, decision_info=active_info)`
  -- this is the ONE call that doesn't exist anywhere in the shadow path today (confirmed by
  research: no `execute_to_target`/`_submit_market_order`/`BinanceFuturesExecutionAdapter` call
  anywhere inside `_refresh_omega461_shadow_asset`).
- Replace the hardcoded `"shadow_only": True` markers (today at `trading_bot.py:14640`, `13339`,
  `13371-13374`, `13434-13435`) with values derived from `executor.status()` / the real
  `execute_to_target` result, so real vs. shadow rows are distinguishable per-asset rather than by
  a hardcoded literal.
- Trade journal rows: reuse `_omega461_shadow_decorate_trade_row`'s pattern of adding explicit
  `asset`/`symbol`/`account_symbol` fields (confirmed real ETH rows today have NO symbol field at
  all -- implicit single-asset -- so real SOL/BTC rows MUST be explicitly tagged or they'd be
  indistinguishable from ETH rows in the shared `data/live/trade_journal.jsonl`).

## Concurrency model: stay in the same process (Option A from the research)

`trading_bot.py` acquires a single exclusive process-wide lock (`fcntl.flock`) at import time
(`trading_bot.py:29-58`) -- a second `python trading_bot.py` process would immediately exit.
Running SOL/BTC executors inside the SAME process via `asyncio.gather`, exactly like
`_refresh_omega461_shadow_assets` already does today, avoids this entirely and reuses a pattern
already proven in production for the decision-making half of this. This is the recommended path
over running 3 separate OS processes (which would require namespacing ~6 different env-var-driven
file paths and fragmenting the trade journal/dashboard).

## Gating (must default OFF)

Following the existing codebase convention (`FINAL_GOVERNOR_OMEGA4_6_1_ENABLE`,
`BINANCE_EXECUTION_ENABLED` both default `False`), add new env vars, all defaulting `False`:

- `SOL_BTC_REAL_EXECUTION_ENABLE` (or per-asset: `SOL_REAL_EXECUTION_ENABLE`,
  `BTC_REAL_EXECUTION_ENABLE`, if independent go-live timing per asset is wanted)
- Reuses `BINANCE_EXECUTION_ENABLED`/`BINANCE_EXECUTION_DRY_RUN` for the underlying
  `BinanceFuturesExecutionAdapter` instances (already-existing gates), but each new asset's
  executor should ALSO check its own asset-level enable flag before ever calling
  `execute_to_target`, independent of whether ETH's flag is on.

Implementing this plan means: even after landing the code, nothing places a SOL/BTC order unless
someone explicitly sets the new env var(s) to true, exactly mirroring how ETH's real path works
today.

## Order of implementation (proposed)

1. `trading_bot_modules/portfolio_risk.py` -- **done** (this doc's writing session).
2. `trading_bot_modules/omega4_6_1_asset_executor.py` -- new `Omega461AssetExecutor` class per the
   design above. Pure new code, not wired into `trading_bot.py` yet -- can be written and reviewed
   in isolation.
3. `trading_bot.py` changes:
   a. Extend the existing `omega461_shadow_assets` context construction (~13579-13625) to also
      build a `BinanceFuturesExecutionAdapter` per asset and an `Omega461AssetExecutor` wrapping it,
      gated behind the new env var(s) (default off, so this is inert until enabled).
   b. Add a single shared `PortfolioRiskManager` instance constructed once, passed to each
      `Omega461AssetExecutor`.
   c. Modify `_refresh_omega461_shadow_asset` (or add a new sibling function) to call the real
      execution path when the asset's executor is present/enabled, falling back to today's
      shadow-only behavior otherwise.
   d. Update trade-journal-row tagging and dashboard `shadow_only` flag population to reflect real
      vs. shadow per-asset.
4. Manual review of the diff against this plan before enabling any new env var in any real
   deployment config.

Step 3 touches `trading_bot.py` directly -- real, currently-running trading infrastructure -- and
is the highest-blast-radius part of this plan. Recommend implementing it as its own reviewable
change, not bundled silently with other work, even though every new code path defaults to
disabled.
