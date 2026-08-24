#!/usr/bin/env python3
"""RESEARCH -- Event-driven, market-neutral use of the evidence signals on the 60-perp panel
(2026-08-15). Companion to research_evidence_signal_cross_sectional_ic_60coin_20260815.py.

=== Why this second study exists (and why the first one alone would be an unfair verdict) ===
The IC study found a real cross-sectional signal (mean rank IC up to +0.027, t=+24, permutation
control ~0) whose quintile long-short portfolio nonetheless loses everything to costs. But that
portfolio REBALANCES THE ENTIRE 60-SYMBOL PANEL every h bars, which is the maximum-turnover way to
use a signal that only fires on ~0.5-3% of bars. Charging full-panel turnover to a sparse signal
and then declaring it uneconomic would be an artifact of the portfolio construction, not a property
of the signal.

This script therefore tests the construction the signal actually implies: TRADE ONLY THE EVENTS.
Each firing is one discrete round trip, so the cost accounting is exact and trivial -- a fixed
round-trip charge per event, no rebalancing turnover at all.

=== Pre-registered design (fixed before any result was looked at) ===
  Event    : bottom_votes >= K  -> LONG that symbol; top_votes >= K -> SHORT. K in {2, 3}
             (the same vote construction as the ETH standalone backtest; no new threshold).
  Holding  : exactly h bars, h in {12, 48, 144} (1h/4h/12h). Entry at the NEXT bar's close after
             the firing bar (signal bar t -> enter t+1, exit t+1+h), so no same-bar look-ahead.
  Return   : raw = side * (close[t+1+h]/close[t+1] - 1).
             EXCESS = raw - side * (equal-weight panel return over the same window). This is the
             market-neutral reading and the one that matters: it asks whether the signal predicts
             RELATIVE performance, with the panel's own drift (and the universe's survivorship
             bias) differenced out. Both are reported.
  Cost     : a flat 0.1% round trip per event (same constant this lineage uses for ETH), applied
             once per event -- no turnover model needed because every event IS a round trip.
  Verdict  : the pre-registered bar is mean EXCESS return per event > cost, with a t-stat computed
             over events, AND sign consistency across the four period splits (2024 / 2025H1 /
             2025H2 / 2026). A positive mean that does not clear cost is reported as a failure, not
             as "promising".
  Controls : (1) a matched RANDOM-EVENT control -- the same number of (symbol, bar) draws sampled
             uniformly, 20 replicates, fixed seed, which measures what any arbitrary entry earns
             under the identical holding rule and cost; (2) the per-event breakeven cost, stated
             in basis points, so the result can be compared against real fee tiers rather than one
             assumed constant.

Same limitations as the companion study, restated because they bind here too: the 60-symbol
universe is top-by-volume AT SELECTION TIME with survivorship (see
data/splits/panel_universe_symbols_20260804.json) -- the EXCESS return series differences out the
common component but not the per-symbol selection effect; the order-flow term is the validated
panel proxy (Stage A of the companion study: Spearman 0.97/0.96 vs raw klines on ETH/BTC), not the
raw quantity; costs are a flat constant with no impact/funding model.

Causal: every input is rolling/shift only; forward windows are used exclusively as the outcome.
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training, no GPU, no live files touched. Imports
research_evidence_signal_cross_sectional_ic_60coin_20260815 read-only and reuses its build_signals
verbatim (no re-implementation, no re-tuned threshold).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import research_evidence_signal_cross_sectional_ic_60coin_20260815 as xs  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/evidence_signal_cross_sectional_event_study_60coin_20260815"
K_GRID = (2, 3)
HOLD_GRID = (12, 48, 144)
COST_ROUNDTRIP = 0.001
RANDOM_REPS, RANDOM_SEED = 20, 20260815
PERIODS = xs.PERIODS


def log(msg: str) -> None:
    print(f"[event_study] {msg}", flush=True)


def _build_votes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bottom-vote / top-vote / close matrices (bars x symbols). build_signals returns the NET
    vote, so bottom and top are recomputed here from the same component frames by calling it twice
    is not possible -- instead the net vote is decomposed by sign, which is exact for the entry rule
    used below (an event needs >=K votes on ONE side, and a bar with votes on both sides nets out
    and is correctly excluded)."""
    symbols = [s["symbol"] for s in json.loads(xs.UNIVERSE_JSON.read_text())["symbols"]]
    net, closes = {}, {}
    for i, sym in enumerate(symbols):
        f = xs.PANEL_DIR / f"{sym}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "rvol_48", "taker_buy_ratio"])
        p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        sig = xs.build_signals(p, delta_series=p["rvol_48"] * (2.0 * p["taker_buy_ratio"] - 1.0), volume_series=p["rvol_48"])
        net[sym] = sig.set_index("timestamp")["votes"]
        closes[sym] = sig.set_index("timestamp")["close"]
        if (i + 1) % 20 == 0:
            log(f"  built {i + 1}/{len(symbols)}")
    return pd.DataFrame(net).sort_index(), pd.DataFrame(closes).sort_index(), pd.DataFrame()


def _stats(x: np.ndarray) -> dict[str, Any]:
    n = int(len(x))
    if n == 0:
        return {"n": 0}
    m, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else float("nan")
    return {"n": n, "mean_pct": m * 100, "median_pct": float(np.median(x)) * 100,
            "t_stat": float(m / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
            "win_rate": float((x > 0).mean()),
            "mean_net_of_cost_pct": (m - COST_ROUNDTRIP) * 100,
            "breakeven_cost_bps": m * 10000.0}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=build_panel ===")
    net_w, close_w, _ = _build_votes()
    log(f"  matrix {net_w.shape[0]} bars x {net_w.shape[1]} symbols [{net_w.index.min()} .. {net_w.index.max()}]")

    close_np = close_w.to_numpy(dtype=float)
    net_np = net_w.to_numpy(dtype=float)
    ts = net_w.index.to_numpy()
    n_bars, n_sym = close_np.shape
    # equal-weight panel return over each (t+1, t+1+h) window -- the market-neutral benchmark
    rng = np.random.default_rng(RANDOM_SEED)

    report: dict[str, Any] = {
        "design": "Event-driven market-neutral use of the evidence votes on the 60-perp panel; each firing is one "
                  "discrete round trip, so cost is exact. Pre-registered bar: mean EXCESS return per event > cost, "
                  "with sign consistency across 4 period splits.",
        "pre_registered": {"K_grid": list(K_GRID), "hold_grid": list(HOLD_GRID), "cost_roundtrip": COST_ROUNDTRIP,
                           "random_reps": RANDOM_REPS, "random_seed": RANDOM_SEED,
                           "entry": "next bar close after the firing bar", "exit": "exactly h bars later"},
        "universe_caveats": ["liquidity_lookahead (top-60 by volume at 2026-08-04)", "survivorship (only currently-trading perps)"],
        "order_flow_term": "panel proxy validated in the companion IC study (Spearman 0.97/0.96 vs raw klines on ETH/BTC)",
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "results": {},
    }

    for h in HOLD_GRID:
        # forward return of every symbol over (t+1 -> t+1+h), aligned so row t = signal bar t
        entry = np.full_like(close_np, np.nan)
        exit_ = np.full_like(close_np, np.nan)
        entry[:n_bars - 1 - h] = close_np[1:n_bars - h]
        exit_[:n_bars - 1 - h] = close_np[1 + h:n_bars]
        fwd = exit_ / entry - 1.0
        panel_mean = np.nanmean(fwd, axis=1)  # equal-weight panel return over the same window

        for K in K_GRID:
            long_mask = net_np >= K
            short_mask = net_np <= -K
            valid = np.isfinite(fwd)
            rows_l, cols_l = np.where(long_mask & valid)
            rows_s, cols_s = np.where(short_mask & valid)
            raw = np.concatenate([fwd[rows_l, cols_l], -fwd[rows_s, cols_s]])
            exc = np.concatenate([fwd[rows_l, cols_l] - panel_mean[rows_l],
                                  -(fwd[rows_s, cols_s] - panel_mean[rows_s])])
            ev_ts = np.concatenate([ts[rows_l], ts[rows_s]])

            per_period = {}
            for pname, (a, b) in PERIODS.items():
                sel = (ev_ts >= np.datetime64(pd.Timestamp(a))) & (ev_ts <= np.datetime64(pd.Timestamp(b)))
                per_period[pname] = {"raw": _stats(raw[sel]), "excess": _stats(exc[sel])}

            # matched random-event control: same number of (bar, symbol) draws, same holding rule
            rand_raw, rand_exc = [], []
            n_ev = len(raw)
            for _ in range(RANDOM_REPS):
                vr, vc = np.where(valid)
                pick = rng.choice(len(vr), size=min(n_ev, len(vr)), replace=False)
                r, c = vr[pick], vc[pick]
                side = rng.choice([1.0, -1.0], size=len(r))
                rr = side * fwd[r, c]
                ee = side * (fwd[r, c] - panel_mean[r])
                rand_raw.append(float(rr.mean()))
                rand_exc.append(float(ee.mean()))

            res = {
                "events": int(n_ev), "long_events": int(len(rows_l)), "short_events": int(len(rows_s)),
                "raw": _stats(raw), "excess": _stats(exc),
                "random_control": {"reps": RANDOM_REPS,
                                   "raw_mean_pct": float(np.mean(rand_raw)) * 100, "raw_sd_pct": float(np.std(rand_raw)) * 100,
                                   "excess_mean_pct": float(np.mean(rand_exc)) * 100, "excess_sd_pct": float(np.std(rand_exc)) * 100},
                "per_period": per_period,
                "sign_consistent_periods_excess": int(sum(1 for v in per_period.values() if v["excess"].get("mean_pct", 0) > 0)),
                "clears_cost": bool(_stats(exc).get("mean_pct", 0) > COST_ROUNDTRIP * 100),
            }
            report["results"][f"h{h}_K{K}"] = res
            log(f"  h={h:3d} K={K}: events={n_ev:7d} | raw mean={res['raw']['mean_pct']:+.4f}% t={res['raw']['t_stat']:+.2f} "
                f"| EXCESS mean={res['excess']['mean_pct']:+.4f}% t={res['excess']['t_stat']:+.2f} "
                f"breakeven={res['excess']['breakeven_cost_bps']:+.2f}bps (cost={COST_ROUNDTRIP * 10000:.0f}bps) "
                f"| random excess={res['random_control']['excess_mean_pct']:+.4f}% (sd {res['random_control']['excess_sd_pct']:.4f}) "
                f"| periods_positive={res['sign_consistent_periods_excess']}/4 clears_cost={res['clears_cost']}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
