# ETH 1-Minute Scalping Model Design — Microstructure DuckDB Analysis (2026-07-18)

## 0. Executive verdict

The live `microstructure_1m` order-book data contains a **real, statistically robust,
contrarian flow signal** — but its magnitude (0.3–2 bps per event) is **3–10x below the
round-trip cost floor** (maker-maker ≈ 4 bps, taker ≈ 9 bps). Three independent analyses
(univariate IC, tail-conditioned returns, multivariate walk-forward) all agree.

**A standalone "enter/exit within minutes" 1m scalping model cannot be made profitable from
this data**, and this is now the 4th independent confirmation of that conclusion in this
project (post-leak baseline, BTC-excluded retest, DeepScalp-PnL, and this study — each with a
different architecture).

**The design that IS supported by the data**: a 1m-cadence *execution & timing overlay* on top
of the proven low-frequency edge (Omega4.6.1 / Sigma6). The model decides every minute, but
what it decides is *when/how* to execute and whether to veto — not independent round trips.
The measured 0.3–2 bps contrarian edge is exactly execution-improvement-sized.

## 1. Data audit (what exists in data/live/microstructure.duckdb)

| table | rows | span | notes |
|---|---|---|---|
| `microstructure_1m` (ETH) | 94.5k | 2026-05-03 → live | 1m grid; 99.9% quality (stale 0.03%, valid_taker_flow 99.98%); 75 days |
| `orderbook_decision_snapshots` (ETH) | 11.4k | 2026-05-13 → live | L2 depth (20 lvl) summaries, irregular ~2s-throttled at decisions — too sparse for a 1m grid; use for spread/fill calibration |
| `microstructure_1m_btc` / `_sol` | ~4.6k each | 2026-07-14 → live | started too recently to model |
| `decision_feature_frame*` | small | — | live decision logging, not training data |

Feature groups in `microstructure_1m`: order-book imbalance (`obi`), taker flow
(`taker_buy_ratio`), whale/retail net inflow (`nif_whale`, `nif_retail`), `spoofing_score`,
`eai`, `oi_delta_pct`, funding, and shadow-book signals (`shadow_toxicity_score`,
`shadow_absorption_score`, `shadow_queue_bias`, `shadow_queue_collapse`, regime tag/conf).

## 2. Causality contract (must-follow for any consumer of this table)

From `microstructure_scanner.py::_scan_loop`: the row labeled `ts=T` is inserted at wall-clock
**T+60..75s** and its rolling windows include data up to the write moment (~15s past the T+1min
boundary). Therefore:

- At a decision on minute boundary `D`, the newest safely usable row is **`ts = D − 2min`**.
- Using `ts = D − 1min` at decision `D` is a look-ahead (row not yet written / includes future
  seconds). `join_microstructure_1m_20260716.py`'s backward-asof join has exactly this ~15s
  optimism — any future re-use must add the extra 1-minute shift.
- Live is *fresher* than the backtest contract (10s in-memory cache), so the contract is
  conservative in the right direction.

All three analysis scripts below implement this contract (`AVAIL_SHIFT_MIN = 2`).

## 3. Evidence

Scripts (all runnable, venv python):
- `scripts/analyze_microstructure_edge_20260718.py` — daily-block Spearman IC, 8 horizons.
  Report: `data/ensemble/reports/microstructure_edge_ic_20260718.csv`
- `scripts/analyze_microstructure_tails_20260718.py` — tail-conditioned forward returns.
- `scripts/walkforward_micro_scalp_20260718.py` — HGB walk-forward prototype (73 causal
  features: micro + derived rolling/z + pure-ETH price features; no BTC anywhere).

### 3.1 The signal is real…
- `taker_buy_ratio` (dev from 0.5): IC = **−0.027 @ 1m, t = −7.6** over 70 daily blocks; same
  sign on 81% of days. `nif_retail` t = −6.9, `nif_whale` t = −6.0, `queue_bias_m15` t = −5.1,
  `obi_m15` @ 15m t = −3.1. All **contrarian**: heavy taker buying / whale inflow / bid-heavy
  book → *negative* forward returns. Classic 1m flow mean-reversion.

### 3.2 …but too small to trade alone
- Decile long-short spread of the best single feature: **~0.35 bps @ 1m**.
- Extreme 2%-tail of a 6-feature contrarian composite: **+0.7–1.4 bps @ 5–15m** (daily t = 0.9,
  54% positive days — not even significant at the tail).
- Multivariate HGB walk-forward (6 folds, expanding ≥30d train / 7d test, 2026-06-03→07-15 OOS):
  - mean test IC **+0.014** (5/6 folds positive) — predictability exists;
  - 5m hold, per-fold quantile thresholds: **negative in 6/6 folds at every cost tier**
    (taker 9bps, maker-taker 6.5bps, even optimistic maker-maker 4bps). Gross edge of selected
    trades ≈ 0–2 bps/trade.
  - 15m hold, absolute pred thresholds (5/8/12 bps): still net negative overall
    (best config −5.7% total, 50% positive folds, unstable).

### 3.3 Cost floor (why 1–15m round trips can't work here)
ETHUSDT perp: taker 4.5 bps/side, maker 2 bps/side (VIP0, no BNB discount), spread ~0.1–0.4 bps.
Round trip 4–9 bps vs. harvestable signal 0.3–2 bps. The gap is structural, not a tuning issue.

## 4. Recommended design: 1m-cadence execution overlay ("MicroExec v1")

Decisions every minute — but the decision space is execution, not independent positions:

```
        ┌────────────────────────────────────────────────┐
        │  Layer 1 (edge source, unchanged):             │
        │  Omega4.6.1 live stack / Sigma6 — direction,   │
        │  sizing, TP/SL, holds hours+                   │
        └───────────────┬────────────────────────────────┘
                        │ intent: "open LONG within next K minutes"
        ┌───────────────▼────────────────────────────────┐
        │  Layer 2: MicroExec v1 (1m cadence, this data) │
        │  every minute, using micro row ts ≤ D−2min:    │
        │   a) TIMING  — contrarian composite says       │
        │      "taker-sell extreme now" → execute the     │
        │      buy now; else wait (bounded by K)          │
        │   b) VETO/DELAY — shadow_toxicity_score high,  │
        │      spoofing_score spike, queue_collapse,     │
        │      data_stale → hold off this minute         │
        │   c) MAKER PLACEMENT — obi/queue_bias choose   │
        │      join-bid vs cross-spread; L2 snapshots    │
        │      calibrate expected queue fill             │
        └────────────────────────────────────────────────┘
```

- Value proposition: entry/exit improvement of ~1–3 bps per execution is exactly the size of
  the measured signal, and on Omega4.6.1's ~24 trades/6mo at 0.9 notional even 2 bps/side ≈
  small but *free* and risk-reducing (veto of toxic entries is the bigger win).
- Success criterion (measurable, causal): paired replay of Layer-1 entries with/without the
  overlay on the 75-day window — overlay must improve realized entry price vs. arrival mid on
  average, and never delay past the intent window K.
- Components: contrarian composite = expanding-z average of {−tbr_dev, −nif_whale, −nif_retail,
  −obi}; veto set = {shadow_toxicity_score z>2, spoofing_score spike, shadow_queue_collapse=1,
  data_stale}. No trained model needed for v1 — the composite is analysis-validated; a trained
  ranker is a v2 option once overlay telemetry accumulates.

### 4.1 Implementation + paired-replay gate result (added 2026-07-18)

Implemented: `trading_bot_modules/micro_exec_overlay.py` (shared live/replay logic:
`prepare_overlay_frame`, `decide_minute`, `MicroExecConfig`) and
`scripts/replay_micro_exec_overlay_20260718.py` (dense paired replay: every valid minute ×
both sides = 86,493 intents/side over 71 days; both arms priced at the bar-open of their
execution minute; report `data/ensemble/reports/micro_exec_overlay_replay_20260718.json`).

Two composites were compared: the 6-component set above with 15m-smoothed terms (long-only
positive, shorts flat — smoothed terms too sluggish for timing) and a fast 4-component set
{−tbr_dev, −nif_whale, −nif_retail, −obi} (positive on BOTH sides, simpler). **Fast set
adopted as the module default.** Final numbers at the recommended operating point
(exec_z=0.8 ≈ top-20% minutes, K=10..15):

| config | long | short | notes |
|---|---|---|---|
| z≥0.8, K=10 | +0.25 bps (t=0.70) | +0.15 bps (t=1.14) | forced-at-deadline ~50% |
| z≥0.8, K=15 | +0.39 bps (t=0.83) | +0.19 bps (t=1.09) | forced ~37%, mean delay ~9m |

- Random-wait control: ≈ 0.00 bps everywhere (harness sound; gains are signal, not drift).
- Per *acted* execution (signal fired before deadline) the gain is ≈ +1 bps — matching the
  tail-analysis prediction. Veto rate 0.95% of minutes; deadline always honored by
  construction (overlay can only delay ≤ K, never cancel).
- **Gate verdict: POSITIVE-EXPECTED BUT UNDERPOWERED.** Sign is right on both sides across
  all z≤1.28 configs and the effect size matches the independent IC/tail measurements, but
  daily-block t ≈ 0.7–1.1 — 75 days cannot certify a ~0.3 bps/intent effect (t≥2 would need
  roughly 4–8x more days). Certification bar for touching live execution is NOT met yet.
- Recommended next step: wire as **shadow telemetry only** (log `decide_minute` outcomes next
  to real Layer-1 entries, no effect on orders, default-off env flag) and re-run this replay
  when the window reaches ~150+ days. Do not enable live effect before a significant re-test.

### 4.2 v1.5 maker-placement extension + gate result (added 2026-07-18)

Implemented: `PlacementConfig` + `choose_placement()` in `trading_bot_modules/micro_exec_overlay.py`
and `scripts/replay_micro_exec_placement_20260718.py` (report
`data/ensemble/reports/micro_exec_placement_replay_20260718.json`). Policy: the contrarian
composite modulates limit-order *patience* — side·score ≥ +1.0 with momentum running → cross
the spread; side·score ≤ −0.5 → rest deeper by 0.5×15m-range; otherwise join top-of-book;
veto minutes suspend the order; deadline always takers. Fill simulation is deliberately
pessimistic (strict trade-through of best-bid−1-tick line, fill at limit, touch never fills,
unfilled intents pay the full taker chase at D+K). Book calibration from
`orderbook_decision_snapshots`: spread median **0.053 bps** (≈1 tick), best-level depth
≈ $195k — economics are fee-differential dominated (maker 2.0 vs taker 4.5 bps).

Results (86,493 intents/side, 71 days, vs taker-at-decision baseline):

| arm | long | short | maker fill | notes |
|---|---|---|---|---|
| naive always-join (K=any) | **+1.28 bps (t=61)** | **+1.34 bps (t=64)** | 100% | fee saving 2.5 − adverse selection ≈1.2 |
| adaptive v1.5 (K=15) | +1.36 bps (t=30) | +1.41 bps (t=30) | 99% | |
| adaptive − naive (paired) | **+0.086 bps (t=2.48)** | +0.062 bps (t=1.53) | — | the true incremental value of adaptivity |

**Interpretation — critical caveat**: the live bot's execution layer ALREADY routes entries as
post-only GTX limit at best bid (alpha14 router, `binance_execution.py::_route_order`,
`BINANCE_EXECUTION_ALPHA14_ROUTER_ENABLE` default-on, entry offset 0, 2s wait, entry market
fallback off). So the certified +1.3 bps/side maker-vs-taker gain is largely *already captured*
in live; what this study adds is (a) the first 71-day empirical certification of that
economics under a pessimistic fill rule, and (b) the **adaptive increment (+0.06–0.09 bps/side,
significant on longs)**, which is NOT in the live router today. Integration point when
desired: make the static `maker_entry_offset_bps` env param dynamic per-order via
`choose_placement()` (composite score + 15m range + momentum + veto). Given the increment's
size, this follows the same shadow-first discipline as 4.1 — not wired into live in this
session.

## 5. What would reopen the standalone-1m question (do NOT reopen without these)

1. **Richer raw data**: the 1m aggregates destroy the alpha that lives at second scale. Needed:
   full-depth L2 diffs + trade tape at ≤1s (the current recorder keeps 1m summaries and 2s
   throttled snapshots only). Months of it.
2. **Fee tier**: at VIP0 the cost floor alone exceeds the total measurable edge. BNB discount +
   volume tier or maker rebate program changes the arithmetic; below ~2 bps round trip the
   measured tails start to clear.
3. Both together, and even then queue-position modeling (fill realism) is the make-or-break —
   the optimistic maker-maker assumption is *already* negative here.

## 6. Discipline notes

- No numbers in this doc come from saved ledgers or non-causal joins; walk-forward is
  fresh, bar-by-bar, expanding-train. `fresh_forward_bar_by_bar=true`,
  `trade_ledgers_used_as_input=false`, `future_rows_used_for_entry=false`.
- 75 days is short; every result here is dev/research-grade. Nothing here is a promotion
  candidate, and the overlay must pass its own paired-replay gate before touching live config.
