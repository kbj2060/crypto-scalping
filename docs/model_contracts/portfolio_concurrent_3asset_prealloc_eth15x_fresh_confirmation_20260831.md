# Portfolio Concurrent 3-Asset — Prealloc Cap + ETH 1.5x Fresh Confirmation - 2026-08-31

Status: `research_diagnostic_not_live_wired`. Answers the standing caveat left open by
[`portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md`](portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md)
("combining the ETH multiplier with the v4 `prealloc` cap design has not been tested") and by
[`docs/eth_cross_symbol_exposure_cap_design_20260831.md`](../eth_cross_symbol_exposure_cap_design_20260831.md)
(A4 design doc), on data extended well past the 2026-01-01..06-30 window every v1-v4/duration-gate/
ETH-multiplier design choice in that chain was selected on, and past the 07-01..07-12 window the
first fresh-window confirmation used. **This session's timing decision (recorded in the A4 design
doc) was an explicit, user-approved exception to the project's normal 09-30 single-touch-OOS
rule** for the 2026-07-01..09-30 window -- this exposure is valid only for this A4 confirmation and
must not be treated as license to reuse 07-01..08-30 data for other axes.

## Configuration

Both configs share the CURRENT_BASELINE knobs (`--duration-gate off --eth-notional-multiplier 1.5
--btc-notional-multiplier 1.0 --sol-notional-multiplier 1.0`), differing only in the portfolio cap:

- **Config A (`config_A_prealloc_cap3`)**: `cap_mode=prealloc`, `total_notional_cap=3.0`,
  `asset_shares={eth: 0.5, btc: 0.3, sol: 0.2}`. `total_notional_cap=3.0` is **not** a new sweep --
  it is reused as-is from the already-swept grid in
  [`portfolio_concurrent_3asset_v4_prealloc_20260712.md`](portfolio_concurrent_3asset_v4_prealloc_20260712.md)
  (whose entire results table is reported at this cap value) and re-surfaced as the worked example
  in [`portfolio_concurrent_3asset_gate_off_cap_sweep_20260712.md`](portfolio_concurrent_3asset_gate_off_cap_sweep_20260712.md)'s
  own recommendation section. Per this project's rule against carrying forward an un-swept
  parameter to the next stage, 3.0 is the one point in that prior grid already anointed as the
  reference by both docs -- not a freshly-invented value.
- **Config B (`config_B_uncapped_current_baseline`)**: `cap_mode=scale`, `total_notional_cap=None`
  (uncapped) -- byte-identical to the existing CURRENT_BASELINE config.
- **same_direction_notional_cap**: not used in either config, per the user's 2026-08-31 decision
  recorded in the A4 design doc ("prealloc 지분 상한만으로 충분, 기존 결론 유지").

Scripts: `scripts/replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py` (new; mirrors
`replay_portfolio_fresh_window_20260713.py`'s monkeypatch of `eth_retest.load_frame_current`'s
hardcoded end-date literal and its `_replay_concurrent_entry_floor` lower-bound-floor variant of
`_replay_concurrent`, generalized to run both configs across 3 splits each in one pass). Reuses
`replay_portfolio_concurrent_3asset_native_20260712`'s `_replay_concurrent`/`_build_world`
unmodified. Artifacts: `tmp/causal_regen_20260516/portfolio_prealloc_eth15x_fresh_confirmation_20260831/`.

## Data extension

Reconstructed the 07-13 fresh-window-extension procedure
([`portfolio_concurrent_3asset_fresh_window_confirmation_20260713.md`](portfolio_concurrent_3asset_fresh_window_confirmation_20260713.md))
for ETH/SOL/BTC end-to-end (raw klines/funding/metrics -> features -> regime3 wide24 overlay ->
direction/quality labels -> frozen parent-bundle re-scoring), extended through late August instead
of 07-12, via public Binance REST/data.binance.vision endpoints only (no live-account client
touched). No retraining anywhere -- every model step is frozen-artifact re-scoring only.

**All three assets' features/labels/predictions now cover a uniform range through
`2026-08-30 23:55:00`** (not literally "today" 08-31 -- see "Issues encountered" below for why):

| asset | before this session | after this session |
|---|---|---|
| ETH (`training_features_2026_rebuilt.csv`) | 2026-08-19 23:55:00 | **2026-08-30 23:55:00** |
| BTC (`btc_features_2026.csv`) | 2026-08-01 17:40:00 | **2026-08-30 23:55:00** |
| SOL (`sol_features_2026.csv`) | 2026-07-21 11:45:00 | **2026-08-30 23:55:00** |

Raw klines were actually extended further (ETH/BTC/SOL all to `2026-08-31 11:30:00`, the moment
the pipeline ran), but the daily OI/long-short-ratio metrics archive (data.binance.vision) only
publishes through the previous day, so the features/labels/prediction chain was deliberately capped
at what that archive can support (see below) -- "as recent as possible" per the task's own framing,
constrained by a genuine data-availability boundary rather than wall-clock "now".

OOS world: `2026-01-01 00:00:00 .. 2026-08-30 23:55:00`, n=69,696 bars (all three assets, common
timestamp intersection). Validation world unchanged: `2025-10-01 00:00:00 .. 2025-12-31 23:25:00`,
n=26,490 bars. Parent re-scoring nonzero-action rates on the extended OOS frame: ETH h48qual 0.6%,
ETH zig075 6.4%, SOL zig075 21.0%, BTC h48qual 17.1%.

## Results

| config | split | PnL | realized MDD | MTM MDD | trades | WR |
|---|---|---:|---:|---:|---:|---:|
| A (prealloc cap=3.0) | validation | 6.23% | -27.78% | -33.35% | 88 | 38.64% |
| A (prealloc cap=3.0) | oos_extended (2026-01-01..08-30) | 70.74% | -47.09% | -53.29% | 148 | 42.57% |
| A (prealloc cap=3.0) | **fresh_window (2026-07-01..08-30)** | **18.68%** | **-13.71%** | **-18.25%** | **24** | **62.50%** |
| B (uncapped, CURRENT_BASELINE) | validation | 22.90% | -36.71% | -44.42% | 88 | 38.64% |
| B (uncapped, CURRENT_BASELINE) | oos_extended (2026-01-01..08-30) | 143.53% | -58.33% | -65.92% | 148 | 43.92% |
| B (uncapped, CURRENT_BASELINE) | **fresh_window (2026-07-01..08-30)** | **18.93%** | **-22.69%** | **-29.13%** | **24** | **66.67%** |

Trade counts are identical between A and B on every split (88/148/24) -- confirms, again, that
`prealloc` never rejects a candidate the uncapped baseline would have taken; it only scales notional
down when an asset's own fixed share is exceeded (this replay's `notional_scaled_events` for config
A: validation eth=29/sol=25/btc=6 of 88; oos_extended eth=36/sol=34/btc=21 of 148; fresh_window
eth=6/sol=1/btc=0 of 24; mean scale ratio by split ~0.67 (validation) / ~0.69 (oos_extended) /
~0.68 (fresh_window), min scale ratio ~0.51-0.55). WR differs slightly between A
and B despite identical trade timing/count (fresh_window: 15/24 wins under A vs 16/24 under B)
because each trade's `trade_return` is a fraction of the shared cash pool at that trade's own entry
time, and the pool's realized path diverges between the two cap regimes -- same caveat this whole
line of work has carried since `portfolio_concurrent_3asset_native_20260712.md` (shared-ledger,
not dedicated-capital, per-asset/per-trade numbers).

Per-asset trade aggregates (fresh_window, shared-ledger, non-dedicated-capital):

| config | asset | trades | WR | sum trade return |
|---|---|---:|---:|---:|
| A | eth | 8 | 50.0% | 0.1632 |
| A | sol | 11 | 63.6% | 0.0150 |
| A | btc | 5 | 80.0% | 0.1963 |
| B | eth | 8 | 62.5% | 0.1949 |
| B | sol | 11 | 63.6% | -0.0579 |
| B | btc | 5 | 80.0% | 0.2289 |

## Cap impact: config A vs config B

| split | ΔPnL (A-B, pp) | ΔMDD realized (pp, +=better) | ΔMTM MDD (pp, +=better) | MDD(A)/MDD(B) |
|---|---:|---:|---:|---:|
| validation | -16.67 | +8.93 | +11.07 | 75.7% |
| oos_extended | -72.79 | +11.24 | +12.63 | 80.7% |
| **fresh_window** | **-0.25** | **+8.97** | **+10.88** | **60.4%** |

**The cap's cost/benefit trade-off is dramatically more favorable in the fresh window than in the
heavily-peeked historical windows.** In validation/oos_extended, capping cuts realized MDD by only
~19-24% relative while costing 70-75% of the uncapped PnL. In the fresh window specifically --
2026-07-01 through 08-30, the genuinely least-inspected data this whole design line has touched --
the cap cuts realized MDD by ~40% relative (-13.71% vs -22.69%) and ~37% on a mark-to-market basis
(-18.25% vs -29.13%), for essentially **zero PnL cost** (18.68% vs 18.93%, a 0.25pp difference on a
24-trade sample). This is the first genuinely fresh-forward evidence this design line has produced
that the `prealloc` cap's structural benefit (bounding worst-case same-direction stacking) is not
just "a different point on the same risk/return trade-off" but, on unseen data, closer to a
risk reduction the strategy gets close to for free.

**Caveat on the fresh-window read**: 24 trades over ~2 months is still a small sample (same
trade-count-scarcity caveat this whole model family carries throughout the project), and the
07-01..08-30 window itself is short and was not chosen to be representative of any particular
regime. Directionally positive and now on genuinely fresh data, not a statistically powered
confirmation.

## Concurrency diagnostics

Diagnostics are identical between config A and config B (the cap only rescales notional, not
entry/exit timing), computed over the full OOS world (69,696 bars) with entries gated by each
split's floor/cutoff:

| split | max concurrent | % bars 2+ open | % bars 3 open | eth&sol bars | eth&btc bars | sol&btc bars |
|---|---:|---:|---:|---:|---:|---:|
| validation | 3 | 91.12% | 78.63% | 21,108 | 21,879 | 22,808 |
| oos_extended | 3 | 95.22% | 82.45% | 58,110 | 58,909 | 64,277 |
| fresh_window (raw, full 69,696-bar world) | 3 | 24.997% | 23.538% | 16,478 | 16,700 | 17,054 |

**The raw fresh_window percentages above are diluted and not the meaningful number**: because
`fresh_window` gates entries at `2026-07-01`, none of the 52,129 pre-07-01 bars (2026-01-01..06-30,
almost exactly matching that period's own bar count) ever have a fresh-window position open, so they
report as "0 concurrent" and drag the whole-world percentage down. Restricting to the 17,567 bars
that are actually inside the active fresh window (2026-07-01..08-30) gives the real concentration
during the period that matters for this confirmation:

| fresh window, active bars only (n=17,567) | value |
|---|---:|
| % bars 2+ open | **99.17%** |
| % bars 3 open | **93.40%** |
| eth&sol overlap | 93.80% |
| eth&btc overlap | 95.07% |
| sol&btc overlap | 97.08% |

**This is higher than the original v1-uncapped baseline's headline stat** (87-88% bars 2+ open,
64-69% all-3-open, over 2026-01-01..06-30, per
[`portfolio_concurrent_3asset_native_20260712.md`](portfolio_concurrent_3asset_native_20260712.md)).
The cross-symbol simultaneous-exposure problem A4 exists to address has, if anything, gotten *more*
pronounced in the freshest ~2 months of data, not less -- reinforcing rather than weakening the case
for a portfolio-level cap.

## Issues encountered

1. **Metrics-vintage / future-leak integrity risk (found and fixed before extending)**: the
   2026-08-23 audit
   ([`eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md`](../../experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md))
   found and fixed a real causal-integrity bug -- Binance's daily metrics archive changed its
   timestamp-bucket label convention (bucket-start vs. bucket-end) at some point, and the raw
   `sum_open_interest_value`/`sum_toptrader_long_short_ratio`/`count_long_short_ratio` merge paths
   in `update_features.py` / `build_{sol,btc}_raw_frame_*.py` were never patched at the source, only
   the *existing* data was patched after the fact. Naively re-running the 07-13 procedure on new
   dates would have silently reintroduced a systematic ~5-minute future-reference into exactly the
   newest, most decision-relevant tail. Fixed by: refreshing the corrected
   `data/TOTAL_{ETH,BTC,SOL}USDT_metrics_2024_2026.csv` reference (+5min bucket-label correction
   baked in, via the already-existing, now env-parameterized
   `download_eth_binance_metrics_archive_20260823.py`) and re-running the existing whole-file
   gate-then-patch fix scripts (`fix_eth_canonical_2026_oi_futureleak_20260823.py`,
   `fix_btcsol_metrics_vintage_20260823.py`, both unmodified) after each asset's full recompute, so
   the 08-23 correction naturally extends to the new tail. This is a deviation from a literal
   reading of "reuse the 07-13 procedure" -- it reuses 07-13's procedure plus the 08-23 fixes that
   postdate it, since skipping them would have violated CLAUDE.md's Fresh-Forward causal-
   availability rule for this session's own new data.
2. **Metrics-archive publication lag forced a non-"today" cutoff**: the daily OI/long-short-ratio
   archive only publishes through the previous day, so features/labels/predictions were capped at
   `2026-08-30 23:55:00` even though raw klines reached `2026-08-31 11:30:00` -- extending features
   past what the corrected metrics reference could support left ~90-138 cells unfixable (exceeding
   the fix scripts' deliberate 9-hour `merge_asof` tolerance), correctly triggering their fail-fast
   guards rather than silently leaving a gap.
3. **Pandas datetime64-to-CSV serialization quirk, discovered and root-caused mid-session**: the
   first attempt capped the cutoff at exactly `2026-08-31 00:00:00` (midnight). Whichever dataframe
   had that exact-midnight timestamp as its *last* row got that one row serialized by `to_csv()` as
   a bare `"2026-08-31"` (no time component) while every other row kept the full
   `"YYYY-MM-DD HH:MM:SS"` -- reproduced independently through two different, unmodified
   pre-existing scripts (`fix_eth_canonical_2026_oi_futureleak_20260823.py`'s own write, then
   `build_omega4_6_1_extended_parent_predictions_20260706.py` failing to re-read that file). Patching
   every downstream script's `to_csv()` call individually was impractical (dozens of scripts touch
   these files); root-caused instead to "never let the cutoff itself land exactly on midnight" and
   shifted it to `cutoff - 5 minutes` (`2026-08-30 23:55:00`), which fully avoided the trigger for
   the rest of the pipeline. A parallel, narrower fix (`pd.to_datetime(..., format="mixed")` plus
   loosening one `merge(how=...)` from `left` to `inner` with a ≤1-row-dropped tolerance in
   `build_omega4_6_1_extended_parent_predictions_20260706.py`) was independently applied to the
   same file during this session; verified compatible and redundant-but-harmless with the
   non-midnight-cutoff fix, since the trigger condition it defends against no longer occurs upstream.
4. **New scripts written this session** (all new, no existing scripts modified):
   `scripts/build_eth_raw_frame_and_extend_canonical_20260831.py` (ETH raw-frame + full
   FeatureEngineer recompute + existing-first safe merge into the canonical file -- needed because,
   per direct code reading, `update_features.py` only ever writes the much narrower/shorter
   `data/training_features_5m.csv`, never `training_features_2026_rebuilt.csv`; the 2026-07-30
   integrity audit independently confirmed there is no single canonical writer script for that
   file), `scripts/truncate_features_to_metrics_safe_cutoff_20260831.py` (caps a features file to
   the metrics-archive-supportable range, with backup),
   `scripts/replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py` (this doc's replay),
   and three `scripts/ops/run_a4_fresh_extension_pipeline*.sh` driver scripts (the resume/resume2/
   resume3 iterations reflect the debugging above; resume3 is the one that ran to completion).
5. No other pipeline-integrity issues surfaced. The regime3 wide24 overlay's pre-existing,
   already-diagnosed `ou_halflife`/`garch_vol_z` byte-match warning (formula drift, not a bug --
   documented since 2026-07-13) reproduced as expected and was not re-investigated, per the task's
   instructions.

## Flags (enforced by the replay script itself, confirmed present in report.json)

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `promotion_grade=false`

## Caveats

- Same modeling caveats as the whole v1-v4/CURRENT_BASELINE/fresh-window-confirmation line: shared-
  cash-pool per-asset numbers are not a dedicated-capital replay; new positions size off realized
  cash only (ignore other sleeves' unrealized PnL); not a promotion artifact; no live wiring; no
  `trading_bot.py` / `portfolio_risk.py` changes.
- This session's 2026-07-01..08-30 exposure is a deliberate, user-approved, one-time exception to
  the project's 09-30 single-touch-OOS rule, valid only for this A4 confirmation -- other axes must
  not treat this window as newly "clean" for their own purposes, and this window itself should be
  treated as spent for any future A4-adjacent work.
- The `same_direction_notional_cap` question and the ETH 1.5x multiplier's own justification were
  both already decided by the user on 2026-08-31 (recorded in the A4 design doc) prior to this run;
  this doc does not re-litigate either.
- Cap level (3.0) and asset shares (50/30/20) were reused from the prior grid, not re-swept on the
  new data -- if a different cap level or share split is ever wanted, that is a new sweep, not
  implied by this result.
