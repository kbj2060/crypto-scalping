# Live Model v1 Checkpoint - ETH / SOL / BTC - 2026-07-14

Status: `live_decision_only` (real order placement still blocked -- `BINANCE_ACCOUNT_ENABLED=False`
fails closed for all three assets, confirmed via `_ready()` in `trading_bot_modules/binance_execution.py`).
This doc freezes the CURRENT live-wired configuration for each asset as **v1** -- the first
formally-versioned live model generation for this multi-asset setup. Future tuning/model-upgrade
work on any asset should be labeled v2, v3, etc. and compared back against this doc, so it's clear
what changed and why.

All three share the same underlying architecture
(`omega4_6_1_duration_ou_halflife_risk_gate_20260630`, `trading_bot_modules/omega4_6_1_live.py`)
but differ in which frozen parent bundle they route to and their own tuning.

## ETH v1

- **Component**: h48qual + zig075 router (`Omega461LiveAdapter` with both components, priority
  `("h48qual", "zig075")`), real path active via `FinalGovernorRuntime` (not the shadow loop).
- **Parent bundles**:
  - h48qual: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt`
  - zig075: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt`
- **Tuning**: duration gate OFF (`FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF=True`), notional
  multiplier **1.5x** (`FINAL_GOVERNOR_OMEGA4_6_1_ETH_NOTIONAL_MULTIPLIER=1.5`), portfolio-cap
  opt-in enabled (`FINAL_GOVERNOR_OMEGA4_6_1_ETH_PORTFOLIO_CAP_ENABLE=True`) but the cap itself is
  currently `uncapped` (`FINAL_GOVERNOR_PORTFOLIO_TOTAL_NOTIONAL_CAP=uncapped`), share weight 0.5
  if a cap is ever set.
- **Solo backtest (post data-pipeline-fix, real-cost COST_MULT=1.0, fully reproduced 2026-07-13)**:
  validation +54.35%/mdd -35.48%; oos_extended +102.02%/mdd -33.25%/mtm -37.45%/40 trades/wr 45.0%;
  oos_frozen_q1 +105.38%/mdd -33.25%/25 trades/wr 48.0%.
- **Live status**: real execution path pre-existing/active since before this session; only
  real-order placement is blocked (account disabled). Position opened 2026-07-07, still held as of
  this checkpoint (exit governed by TP/SL or the learned exit head, not by any fixed time limit --
  see `Omega461LiveAdapter.evaluate_exit`).

## SOL v1

- **Component**: zig075 only, via the shadow-asset loop (`OMEGA4_6_1_SHADOW_ASSET_CONFIG["sol"]`).
- **Parent bundle**: `tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707/true_3head_tabm_bundle.pt`
- **Risk sidecar**: `tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707/risk_sidecar.pkl`
- **Own quality_threshold**: 0.70. **Own (pre-tuning) duration_threshold**: 0.0055208323.
  **scale_map**: `{zig075_L: 1.0, zig075_S: 2.0}`.
- **Tuning**: duration gate OFF (shared flag with BTC,
  `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_DURATION_GATE_OFF=True`), notional multiplier **1.5x**
  (`FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER=1.5`, SOL-only -- no BTC equivalent, see
  BTC section). Found via `docs/model_contracts/btc_sol_lowcost_tuning_sweep_20260713.md`.
- **Solo backtest (same pipeline/cost convention as ETH)**: validation +46.24%/mdd -36.17%;
  oos_extended +39.98%/mdd -27.91%/59 trades/wr 39.0%; oos_frozen_q1 +33.02%/mdd -24.85%.
- **Live status**: real-execution plumbing enabled
  (`FINAL_GOVERNOR_OMEGA4_6_1_SOL_BTC_REAL_EXECUTION_ENABLE=True`), still blocked by account
  disable. Active shadow trading since 2026-07-11 16:45 -- 9 journal events (4 open/close cycles +
  1 open) as of this checkpoint, all going through cleanly.

## BTC v1

- **Component**: h48qual only, via the shadow-asset loop (`OMEGA4_6_1_SHADOW_ASSET_CONFIG["btc"]`).
- **Parent bundle**: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708/true_3head_tabm_bundle.pt`
- **Risk sidecar**: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708/risk_sidecar.pkl`
- **Own quality_threshold**: 0.55. **Own (pre-tuning) duration_threshold**: 0.00541154875.
  **scale_map**: `{h48qual_L: 0.5, h48qual_S: 2.5}`.
- **Tuning**: duration gate OFF (shared flag with SOL, same env var as above). **No notional
  multiplier** -- the low-cost tuning sweep found multiplier scaling doesn't help BTC (PnL stays
  flat ~10-14% while MDD grows unfavorably at every multiplier tried); kept at the implicit 1.0x
  default. BTC's gap to ETH/SOL is treated as a genuine signal-quality question, not a tuning one
  -- see `docs/model_contracts/btc_sol_lowcost_tuning_sweep_20260713.md` for the full grid.
- **Solo backtest (same pipeline/cost convention as ETH, gate off, 1.0x)**: validation
  +6.69%/mdd -12.11%; oos_extended +10.52%/mdd -16.46%/31 trades/wr 35.5%; oos_frozen_q1
  +6.21%/mdd -16.46%.
- **Live status**: real-execution plumbing enabled (same shared flag as SOL), still blocked by
  account disable. Zero journal events as of this checkpoint -- BTC's own candidate-event rate is
  the sparsest of the three (~31 events/180 days in backtest, ~1 every 5.8 days), and shadow
  logging has only been active ~2.5 days at time of writing, so zero events so far is
  statistically unremarkable (not confirmed as a bug; would warrant investigation if still zero
  after ~1-2 weeks).

## Shared infrastructure (applies to all three)

- `PortfolioRiskManager` (`trading_bot_modules/portfolio_risk.py`): shares eth/btc/sol = 0.5/0.3/0.2,
  `total_notional_cap` currently `uncapped` in `.env` (this session's own cap-sweep research found
  cap=3.0 dominates uncapped on OOS risk-adjusted terms for the concurrent portfolio -- current
  `.env` setting does not reflect that finding; flagged, not changed).
- Data pipeline: `data/training_features_5m.csv` -> `data/splits/year_oos/training_features_2026_rebuilt.csv`
  -> regime3 wide24 overlay -> per-asset parent re-scoring -- entire chain regenerated and verified
  reproducible end-to-end this session (2026-07-13), all three Omega Artifact Integrity Promotion
  Gate audits passing `promotion_pass: true` as of that regeneration.
- `data/live/microstructure.duckdb`'s `decision_feature_frame_live_only_shadow_20260702` logging
  table: schema-migration bug fixed 2026-07-13 (was silently failing every write for 11+ days).
- Real order placement: blocked for all three assets by `BINANCE_ACCOUNT_ENABLED=False`
  (`_ready()` check in `BinanceFuturesExecutionAdapter` requires a live account connection that
  never gets built while this is False) -- this is the actual master safety gate right now, not
  any of the per-asset execution-enable flags above.

## Purpose of this checkpoint

This is the reference point for "v1" of each asset's live-wired Omega4.6.1 model. Any future
change -- a new duration-gate threshold, a different multiplier, a retrained parent bundle, a new
feature, a different quality threshold -- should be tracked as v2 (or later) and diffed against
this doc, so it's always clear what the live bot was actually running at a given point in time and
why a later version changed.
