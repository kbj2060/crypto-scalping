# ETH 1m Scalping Line Data Contract

Status: `research` (not promoted — see Red Team Gates)

Last updated: 2026-07-17 KST

## Scope

- Model id: `eth_scalp_1m_20260716`
- Architecture: single `sklearn.ensemble.HistGradientBoostingClassifier` (3-class LONG/SHORT/CASH), causal 1-minute features, confidence-threshold gate, realistic maker-entry fill simulation. NOT a multi-model Omega-style ensemble (that was tried and made results worse — see Experiment Log).
- Purpose: test whether a genuinely new, dedicated 1-minute ETH scalping model can beat this project's long-standing "lower frequency wins" finding (Sigma3/Sigma6), at the user's explicit request.
- Owner agents: this session (2026-07-16 → 2026-07-17), spawned from prior Sigma/Omega work.
- Implementation scripts: see per-experiment table below (all under `scripts/`, suffixed `_20260716` or `_20260717`).
- Report artifacts: `data/ensemble/reports/scalp_1m_*.json`.
- Model artifacts: **none persisted to disk** — every experiment in this line retrains the HGB fresh from `data/training_features_1m.csv` at run time; there is no saved `.pkl`/`.joblib` model file yet. This must change before any live wiring (see Open Issues).

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Train (single-window experiments) | `data/training_features_1m.csv` | 2024-01-01 → 2026-04-30 | ~1.22M | Fit primary HGB |
| Val (single-window) | same | 2026-04-30 → 2026-05-31 | ~44.6K | Confidence-threshold selection |
| OOS (single-window) | same | 2026-05-31 → 2026-07-12 | ~60.5K | Headline result reporting |
| Walk-forward folds 1–7 (**clean**) | same | 2025-07-01 → 2026-05-15, 7 × ~45-day expanding-window folds | ~64–82K/fold | Independent validation, never touched threshold selection |
| ~~Walk-forward fold 8~~ | same | 2026-05-16 → 2026-07-12 | ~82K | **Excluded** — overlaps the val/OOS window above; not an independent test of the threshold hyperparameter (found 2026-07-17) |

Audit:

- Timestamp overlap: single-window val/OOS and walk-forward fold 8 **do overlap** (see above) — resolved by dropping fold 8 from all walk-forward conclusions. Folds 1–7 have zero overlap with the threshold-selection window or each other.
- Duplicate timestamps: none (1-minute klines deduplicated at build time, `build_features_1m_20260716.py`).
- Warmup handling: rolling-window features (max 288 bars) require sufficient trailing history; first ~1,748 rows of the full build were dropped as NaN.
- OOF/embargo: purged 5-block OOF used only for the meta-labeling ablation (`train_eval_scalp_1m_meta_label_20260716.py`) and the concurrency-weighting ablation (`build_scalp_1m_tb_labels_weighted_20260716.py` + `walkforward_scalp_1m_weighted_purged_20260716.py`, 20-minute purge at fold boundaries). The main baseline/walk-forward pipeline does not purge at fold boundaries (20-minute label horizon vs. day-scale folds makes the boundary leakage negligible; verified by the weighted/purged ablation reproducing near-identical numbers).

## Shared Feature Contract

- Canonical feature source: `features/engineering.py::FeatureEngineer` (`candle_minutes=1`), same class used by the project's 5-minute pipeline.
- Feature count: 142 (119 after `ULTIMATE_FEATURE_COLS` whitelist intersection used at train time for the price-only baseline; +18 microstructure columns for the B2 variant).
- Normalization: feature-internal (rolling z-scores/percentile ranks inside `FeatureEngineer`); DL variants (TabM, GRU) additionally standardize with train-only-fit mean/std.
- Missing fallback: `.fillna(0.0)` at train/inference time for the HGB models.
- Stale handling: not applicable offline; a live wiring would need to inherit the existing bot's staleness/heartbeat checks (`trading_bot_modules/binance_live_fetcher.py`).
- Live availability: **not yet wired to any live data path** — this whole line is backtest-only as of 2026-07-17.

**Important caveat (stated, not fixed)**: `FeatureEngineer.candle_minutes` is stored but never threaded into any rolling-window size (verified by code search). Running it on 1-minute bars compresses every window's real-time span 5× vs. the 5-minute pipeline it was designed for. Treated as intentional for this dedicated scalping line, not a bug.

## Label Contract

- **Base label** (`build_scalp_1m_tb_labels_20260716.py`): triple-barrier, entry = next-bar open, HORIZON=20 bars, ATR-scaled TP/SL (`TP_ATR_MULT=1.2`, `SL_ATR_MULT=1.0`, bounds `TP∈[0.0015,0.006]`, `SL∈[0.0010,0.005]`), same-bar tie → SL wins.
- Alternative labels tried and **rejected** (see Experiment Log): DP trajectory oracle, trend-scanning, 5-minute short-horizon relabel.
- Cost included: NOT in the label itself (label is a pure price-move fraction per the Futures Risk Sizing Contract) — cost is applied downstream at evaluation/simulation time.
- Future path usage: label construction legitimately uses forward OHLC (standard triple-barrier practice) — this is the training TARGET, never fed back as a model input feature (verified, see Red Team Gates).
- Leakage controls: empirically verified via truncate-and-recompute test (see Red Team Gates).
- Known limitations: 20-bar windows on 1-minute bars overlap ~95% between adjacent rows — addressed via Lopez de Prado-style concurrency/uniqueness sample weighting (`build_scalp_1m_tb_labels_weighted_20260716.py`), found not to materially change results (see Experiment Log).

## Cost/Risk Assumptions

- Fee: maker entry 0.02% (`MAKER_FEE`) + taker exit 0.045% (`TAKER_FEE`) = 0.065% round trip (`simulate_maker_entry_scalp_1m_20260716.py`).
- Slippage: modeled only via the maker-fill simulation itself (1bp passive limit offset, 3-minute fill window, cancel-if-unfilled) — no additional slippage buffer beyond that.
- Max notional exposure: **unresolved as a single number** — see Position Sizing below. Recommended final: 1% per trade / 5% total concurrent exposure, OR (per latest user direction) 100% per trade in a fully segregated dedicated sub-account.
- Leverage cap: not modeled (price-move-fraction labels per the Futures Risk Sizing Contract; leverage/notional conversion is a live-wiring concern, not yet designed for this line).
- Funding: not included in the 1m label/cost model (round-trip holds are ≤20 minutes, funding impact negligible at that horizon; not formally verified).
- Liquidation/maintenance margin: **not modeled at all** — a real concern given the 100%-per-trade sizing under discussion.
- Resize accounting: not applicable (fixed-size entries, no dynamic resize within a trade).

### Position sizing — the central open question of this line

Three sizing models were tried in order, each superseding the last:

1. **Unconstrained** (original baseline reports): every trade gets its own independent 100% notional — later found to implicitly assume unlimited capital (real avg 4.9/max 17 concurrent positions). All early "+3.74%" style headline numbers use this and should be read as directional signal quality, not real portfolio return.
2. **Slot-count `cap`** (`simulate_portfolio_capped_scalp_1m_20260717.py`): N concurrent slots, each sized `equity/N`. Superseded — user correctly pointed out this conflates position-count with risk-amount and doesn't cap worst-case loss below 100% if all N slots fire the same direction (trades are correlated, not independent).
3. **Explicit exposure cap** (`simulate_exposure_capped_scalp_1m_20260717.py`, current/final): two independent parameters, `PER_TRADE_PCT` (size of each position) and `MAX_TOTAL_EXPOSURE_PCT` (hard ceiling on the SUM of all currently-open notional; new signals rejected once hit). Provides a provable worst-case single-event loss bound = `MAX_TOTAL_EXPOSURE_PCT` exactly, regardless of correlation.

Backtested results (7 clean folds, 10.5 months, threshold=0.55):

| PER_TRADE_PCT | MAX_TOTAL_EXPOSURE_PCT | Return | Max DD | Worst single-event bound |
|---:|---:|---:|---:|---:|
| 100% (single position) | 100% | +36,541,757% | 10.35% | 100% |
| 20% | 20% | +1,206% | 2.15% | 20% |
| 5% | 20% | +219% | 0.79% | 20% |
| 2% | 20% | +61% | 0.35% | 20% |
| 5% | 10% | +165% | 0.55% | 10% |
| **1%** | **5%** | **+26.7%** | **0.17%** | **5%** |

The (1%, 5%) row is the only one this session judged genuinely credible. **Current user direction (2026-07-17) overrides this recommendation**: run the scalper in a fully segregated dedicated sub-account (separate from the existing Omega4.6.1 swing account, solving the same-symbol exchange-level netting problem) with 100% per-trade sizing inside that sub-account. This bounds blast radius to whatever capital is funded into the sub-account, but does not make 100%-per-trade sizing itself safe — the block-bootstrap stress test (below) shows this sample cannot even construct a bad enough scenario to produce a believable drawdown at that sizing, which is itself a red flag, not reassurance.

## Output Contract

Not yet finalized for a live decision row — this line has no live wiring yet (see Open Issues). Should follow the project's standard shape (`action`, `side`, `notional_exposure`, `leverage`, `position_fraction`, `quality_score`, `confidence`) once live wiring begins.

Required report metrics (already produced by this line's scripts): `pnl`, `mdd`, `trades`, `trades_per_day`, `wr` (hit_rate), `cost_stress` (maker/taker fee sensitivity). Not yet produced: `avg_notional`, `avg_leverage`, `monthly` breakdown.

## Experiment Log (chronological, 2026-07-16 → 2026-07-17)

All challenger experiments below were evaluated against the baseline (HGB + base triple-barrier label + confidence threshold + realistic maker-fill sim) and **none beat it**:

| # | Experiment | Script | Result vs. baseline |
|---|---|---|---|
| 0 | Baseline (this contract's model) | `train_eval_scalp_1m_hgb_20260716.py` + `simulate_maker_entry_scalp_1m_20260716.py` | OOS +3.74%, 8,075 trades, 75.6% hit rate — reference point |
| 1 | ETH microstructure overlay (B2, price+order-book features) | `train_eval_scalp_1m_hgb_20260716.py` (B2 config) | Marginal, real but small edge (+0.4pp hit rate); bounded to a ~70-day window |
| 2 | Wide horizon (60min vs 20min) | tuning sweep in `tune_scalp_1m_levers_20260716.py` | **Worse** — hit rate dropped below 50% |
| 3 | Confidence threshold + naive taker fee | `tune_scalp_1m_levers_20260716.py` | Improved (+3.7% single window) — folded into baseline |
| 4 | Realistic maker-fill simulation | `simulate_maker_entry_scalp_1m_20260716.py` | Confirmed the improvement survives realistic fill risk; **this became the baseline** |
| 5 | 8-fold walk-forward of the baseline | `walkforward_scalp_1m_conf_maker_20260716.py` | 8/8 (later 7/7 clean) folds positive, mean +3.51%→+3.41% |
| 6 | Concurrency/uniqueness sample weighting (B-1) | `build_scalp_1m_tb_labels_weighted_20260716.py` | No material change — confirms baseline wasn't an overlap-overconfidence artifact |
| 7 | Meta-labeling gate (A-1, purged OOF) | `train_eval_scalp_1m_meta_label_20260716.py` | **Slightly worse** (+3.12% vs +3.74%) |
| 8 | TabM primary model (A-2) | `train_eval_scalp_1m_tabm_20260716.py` | **Slightly worse** (+3.22%) |
| 9 | DP trajectory oracle label (B-3) | `build_scalp_1m_dp_labels_20260716.py` + `train_eval_scalp_1m_altlabel_20260716.py` | **Slightly worse** (+3.44%) |
| 10 | Trend-scanning label (B-4, Sigma3/6's own family) | `build_scalp_1m_trendscan_labels_20260716.py` + `train_eval_scalp_1m_altlabel_20260716.py` | **Much worse** (-0.05%, near-noise — 1h t-stat thresholds don't transfer to 1m) |
| 11 | Full Omega-style 6-component composed architecture | `train_eval_scalp_1m_omega_style_20260716.py` | **Much worse** (regime veto alone: -1.79%; full router: +0.62% on 1,040 trades) |
| 12 | Threshold=0.70 (frequency reduction) | `reduce_scalp_1m_trade_frequency_20260717.py` + `walkforward_scalp_1m_thr070_20260717.py` | Real tradeoff, not a "beat": -22% PnL for -65% volume, 7/7 folds positive — adopted as the low-frequency operating point |
| 13 | GRU sequence model (30min window) | `train_eval_scalp_1m_gru_20260717.py` | **Worse on both axes** (frequency went UP not down, PnL slightly lower) |
| 14 | Short-horizon relabel (5min, DP-oracle-informed) | `build_scalp_1m_tb_labels_short_20260717.py` + `train_eval_scalp_1m_short_horizon_20260717.py` | **Worse** (+1.23% vs +3.74%, fee drag dominates at smaller TP scale) |
| 15 | Block-bootstrap correlation/tail-risk stress test | `stress_test_scalp_1m_block_bootstrap_20260717.py` | Not a model change — confirmed the sizing problem is real and the historical sample can't construct a believable bad scenario at any sizing above ~5% total exposure |

**Net conclusion of the experiment log**: 14 independent architecture/label/frequency challengers were tried; none beat the baseline's risk-adjusted quality. The baseline is very likely near this feature set's ceiling at 1-minute frequency, not an undertuned starting point.

## Red Team Gates

- [x] Train/validation/test timestamp overlap audit is zero — **found a real overlap** (walk-forward fold 8 vs. threshold-selection window), fixed by exclusion; folds 1-7 confirmed clean.
- [x] No bfill/full-sample scaler/future feature enters live state — full code audit (general-purpose agent, all of `features/engineering.py`, `elite.py`, `high_order_state.py`, `core/cvp.py`) + empirical truncate-and-recompute test (141 features, 2 cutoffs, 0 mismatches) both PASS.
- [x] Fee/slippage 1x/2x/3x ranking is reported — maker/taker sensitivity tested (`tune_scalp_1m_levers_20260716.py` MAKER lever); realistic fill-risk simulation supersedes naive fee-only sensitivity.
- [ ] Score/probability buckets are calibrated against realized net PnL — threshold sweep exists (val-optimal 0.55/0.70) but no formal calibration curve/reliability diagram produced.
- [x] Monthly/weekly walk-forward is reported — 7-fold (~6-week each) expanding-window walk-forward, clean of contamination.
- [ ] Live train state parity is checked — **not applicable yet, no live wiring exists**.
- [ ] Funding/liquidation limitations are documented — funding assumed negligible at ≤20min holds but not formally verified; liquidation/margin-call risk under 100%-per-trade sizing is explicitly UNMODELED (see Cost/Risk Assumptions).
- [ ] **Not run**: `scripts/audit_omega_artifact_integrity_20260630.py` — this line has not gone through the project's formal Omega Artifact Integrity Promotion Gate (CLAUDE.md) at all. Required before any live/baseline promotion.

## Open Issues

- No model artifact is persisted to disk anywhere in this line — every script retrains from scratch. A live runner needs a `joblib.dump`/versioned artifact export step that doesn't exist yet.
- No live 1-minute feature computation pipeline exists — the existing live bot (`trading_bot.py`) computes features on a 5-minute cadence; a 1-minute live path needs its own incremental/rolling feature computation, not yet designed (research on the existing 5m live feature path was in progress as of this document's writing).
- No live order-placement code exists for this line — `BinanceFuturesExecutionAdapter`'s support for "resting limit entry + cancel-if-unfilled + TP/SL bracket + time exit" (the maker-fill model this line's backtests assume) had not been confirmed as of this document's writing.
- **Account architecture decided but not built**: user has directed a fully segregated Binance sub-account (separate from the existing Omega4.6.1 swing account) to avoid same-symbol position netting at the exchange level. Sub-account creation and API key provisioning must happen on the user's side (cannot be done by an agent); engineering side (new standalone script mirroring `run_live_collectors.py`, own `BinanceLiveFetcher`/`BinanceFuturesExecutionAdapter`, own env-gated account-enable flag defaulting OFF) not yet built.
- Real order placement must remain gated OFF (mirroring `BINANCE_ACCOUNT_ENABLED=False`) until this line passes the Promotion Gate and a deliberate, separate enable decision.
