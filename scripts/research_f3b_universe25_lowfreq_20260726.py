#!/usr/bin/env python3
"""F3-B-25: cross-sectional momentum, low-frequency rebalance, 25-coin universe.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S5.1.2
(committed 87f92ae, before this script produced any output). Grid, windows,
cost model and gates are fixed there. Do not edit them after seeing results.

F3-B (3 assets, daily rebalance) was killed on cost. F3-B-LF (3 assets, low-freq
rebalance) flipped several variants net-positive but 0/9 cleared t>3 -- not
because there was no effect, but because a 3-name cross-section carries ~145
bps/day of idiosyncratic vol against a 7-9 bps/day effect, needing 6-13 years
of history to resolve at only 1.2 years available. This widens the cross-
section to the 25-coin universe locked in f3b_universe25_selection_20260726.json
(chosen by a performance-blind liquidity/listing-age rule, committed before any
return was computed) to shrink that idiosyncratic noise via basket averaging.

  positions   rank all eligible coins by k-day log return; long the top
              quintile (5 names) / short the bottom quintile (5 names), equal
              weight within each leg, 50% gross long / 50% gross short
  grid        k in {14,30,60} days x rebalance in {3,7,14} days = 9 variants
              (identical to F3-B-LF, so results are attributable to the
              universe widening alone, not a second simultaneous change)
  eligibility a coin enters the day's ranking only once it has >=k trading
              days of history since its own onboard date (ACEUSDT and
              1000BONKUSDT onboard Dec 2023, so this affects only the first
              few weeks of the k=60 grid cells -- not a separate registered
              choice, just the k-day lookback's own precondition)
  cost        FLAT: full round-trip on 100% gross notional at every rebalance
              (matches F3-B/F3-B-LF's conservative convention exactly, and
              does not scale with basket size, so it stays comparable to the
              3-asset run). TURNOVER: cost proportional to the fraction of the
              10 leg-slots (5 long + 5 short, 1/5 gross each) that actually
              changed at that rebalance.
  gate        exploration (2024-01-01..2025-08-31) net_cost1_flat > 0 AND
              day-block bootstrap t > 3
  val/oos     2025-09..12 / 2026-01..03, entered only if the gate passes
  promotion   OOS DSR >= 0.95 AND PBO <= 0.25
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.selection_stats import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    sharpe,
)

K_GRID = [14, 30, 60]
REBAL_GRID = [3, 7, 14]
N_LEGS = 5  # top/bottom quintile of a 25-coin universe
COST1_ROUNDTRIP_BPS = 10.0
COST3_ROUNDTRIP_BPS = 30.0

EXPL_START = pd.Timestamp("2024-01-01", tz="UTC")
EXPL_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31", tz="UTC")
GATE_T = 3.0


def load_universe() -> tuple[pd.DataFrame, list[str]]:
    sel = json.loads((ROOT / "data/ensemble/metrics/f3b_universe25_selection_20260726.json").read_text())
    universe = sel["universe"]
    closes = {}
    for sym in universe:
        f = ROOT / f"binance_data/klines_daily/{sym}/{sym}-1d-api.csv"
        df = pd.read_csv(f, parse_dates=["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        closes[sym] = df.set_index("timestamp")["close"]
    price_df = pd.DataFrame(closes).sort_index()
    return price_df, universe


def day_block_bootstrap_tstat(returns: np.ndarray, n_boot: int = 3000, seed: int = 20260726) -> dict | None:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 30:
        return None
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(r, size=r.size, replace=True).mean() for _ in range(n_boot)])
    observed = float(np.mean(r))
    se = float(np.std(boot))
    return {"observed_mean": observed, "boot_se": se, "t_stat": float(observed / se) if se > 1e-12 else None}


def run_variant(price_df: pd.DataFrame, universe: list[str], k: int, rebal: int) -> pd.DataFrame:
    """Daily gross PnL and cost (flat + turnover) for one (k, rebal) variant.

    A coin is eligible for ranking on day i only if its price k trading days
    earlier is not NaN (i.e. it was already listed and has a full lookback) --
    this is what naturally excludes ACE/1000BONK from the very first few weeks
    without any special-casing.
    """
    log_px = np.log(price_df)
    past_ret = log_px - log_px.shift(k)
    daily_ret = log_px.diff()

    n = len(price_df)
    gross = np.zeros(n)
    cost_flat = np.zeros(n)
    cost_turn = np.zeros(n)
    cur_long: set[str] = set()
    cur_short: set[str] = set()
    leg_cost = COST1_ROUNDTRIP_BPS / 1e4

    first = int(np.argmax(past_ret.notna().sum(axis=1) >= 2 * N_LEGS))

    for i in range(first, n):
        if cur_long:
            gross[i] = (
                0.5 * np.mean([daily_ret[a].iloc[i] for a in cur_long if pd.notna(daily_ret[a].iloc[i])])
                - 0.5 * np.mean([daily_ret[a].iloc[i] for a in cur_short if pd.notna(daily_ret[a].iloc[i])])
            )

        if (i - first) % rebal:
            continue
        row = past_ret.iloc[i].dropna()
        if len(row) < 2 * N_LEGS:
            continue
        ranked = row.sort_values()
        new_short = set(ranked.index[:N_LEGS])
        new_long = set(ranked.index[-N_LEGS:])

        n_changed_slots = len(cur_long - new_long) + len(cur_short - new_short)
        cost_flat[i] = 2 * leg_cost
        cost_turn[i] = (n_changed_slots / (2 * N_LEGS)) * 2 * leg_cost
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(
        {"gross": gross, "cost_flat": cost_flat, "cost_turn": cost_turn}, index=price_df.index
    ).iloc[first:]


def window_stats(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> dict:
    w = frame.loc[(frame.index <= end) if start is None else (frame.index >= start) & (frame.index <= end)]
    if len(w) < 30:
        return {"n_days": int(len(w)), "insufficient": True}

    cost3_scale = COST3_ROUNDTRIP_BPS / COST1_ROUNDTRIP_BPS
    net_flat = (w["gross"] - w["cost_flat"]).to_numpy()
    net_turn = (w["gross"] - w["cost_turn"]).to_numpy()
    net_flat3 = (w["gross"] - w["cost_flat"] * cost3_scale).to_numpy()

    mu, sd = float(net_flat.mean()), float(net_flat.std(ddof=1))
    days_for_t3 = float((GATE_T * sd / mu) ** 2) if mu > 0 and sd > 0 else None

    return {
        "n_days": int(len(w)),
        "n_rebalances": int((w["cost_flat"] > 0).sum()),
        "daily_sd_net_cost1_flat": sd,
        "days_needed_for_t3": days_for_t3,
        "years_needed_for_t3": (days_for_t3 / 365.0) if days_for_t3 else None,
        "gross_mean_daily": float(w["gross"].mean()),
        "cost_flat_mean_daily": float(w["cost_flat"].mean()),
        "cost_turn_mean_daily": float(w["cost_turn"].mean()),
        "net_cost1_flat_mean_daily": mu,
        "net_cost1_turn_mean_daily": float(net_turn.mean()),
        "net_cost3_flat_mean_daily": float(net_flat3.mean()),
        "annualized_net_cost1_flat_pct": mu * 365 * 100.0,
        "daily_sharpe_net_cost1_flat": sharpe(net_flat),
        "bootstrap_net_cost1_flat": day_block_bootstrap_tstat(net_flat),
        "bootstrap_net_cost1_turn": day_block_bootstrap_tstat(net_turn),
    }


def passes_gate(stats: dict) -> bool:
    if stats.get("insufficient"):
        return False
    boot = stats.get("bootstrap_net_cost1_flat")
    t = boot.get("t_stat") if boot else None
    return bool(stats["net_cost1_flat_mean_daily"] > 0 and t is not None and t > GATE_T)


def main() -> None:
    price_df, universe = load_universe()
    report = {
        "stage": "F3-B-25",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S5.1.2 (commit 87f92ae)",
        "universe": universe,
        "n_universe": len(universe),
        "n_legs_each_side": N_LEGS,
        "grid": {"k_days": K_GRID, "rebalance_days": REBAL_GRID},
        "primary_cost_model": "FLAT (100% gross round trip every rebalance)",
        "gate": f"exploration net_cost1_flat > 0 AND bootstrap t > {GATE_T}",
        "data_coverage": {
            "n_days": int(len(price_df)),
            "min_date": str(price_df.index.min().date()),
            "max_date": str(price_df.index.max().date()),
        },
    }

    frames, rows = {}, []
    for k, rebal in itertools.product(K_GRID, REBAL_GRID):
        f = run_variant(price_df, universe, k, rebal)
        frames[(k, rebal)] = f
        expl = window_stats(f, EXPL_START, EXPL_END)
        rows.append({"k": k, "rebalance_days": rebal, "exploration": expl, "gate_pass": passes_gate(expl)})
    report["exploration_grid"] = rows

    expl_sharpes = np.array(
        [r["exploration"].get("daily_sharpe_net_cost1_flat", 0.0) for r in rows], dtype=np.float64
    )
    sr_std = float(np.std(expl_sharpes, ddof=1))
    report["noise_floor"] = {
        "n_variants": len(rows),
        "trial_sharpe_std": sr_std,
        "expected_max_daily_sharpe": expected_max_sharpe(len(rows), sr_std),
        "best_observed_daily_sharpe": float(np.max(expl_sharpes)),
    }

    passing = [r for r in rows if r["gate_pass"]]
    report["n_variants_passing_exploration"] = len(passing)

    if not passing:
        report["verdict"] = (
            f"KILLED at exploration -- 0/{len(rows)} variants clear net_cost1_flat > 0 with t > {GATE_T}. "
            "Per the pre-registration, val/OOS are not run."
        )
    else:
        best = max(passing, key=lambda r: r["exploration"]["net_cost1_flat_mean_daily"])
        key = (best["k"], best["rebalance_days"])
        f = frames[key]
        val = window_stats(f, VAL_START, VAL_END)
        oos = window_stats(f, OOS_START, OOS_END)

        oos_net = (f["gross"] - f["cost_flat"]).loc[
            (f.index >= OOS_START) & (f.index <= OOS_END)
        ].to_numpy()
        dsr = deflated_sharpe_ratio(oos_net, expl_sharpes)

        net_df = pd.DataFrame(
            {f"k{k}_r{rb}": frames[(k, rb)]["gross"] - frames[(k, rb)]["cost_flat"]
             for k, rb in itertools.product(K_GRID, REBAL_GRID)}
        ).dropna()
        net_df = net_df.loc[net_df.index <= OOS_END]
        pbo = pbo_cscv(net_df.to_numpy(), n_splits=10)

        report["selected_variant"] = {"k": key[0], "rebalance_days": key[1]}
        report["val"] = val
        report["oos"] = oos
        report["oos_deflated_sharpe"] = dsr
        report["pbo_cscv"] = pbo
        promoted = bool(dsr.get("passes_95") and pbo["pbo"] <= 0.25)
        report["promotion_pass"] = promoted
        report["verdict"] = (
            "PROMOTION CRITERIA MET" if promoted
            else "gate passed at exploration but promotion criteria (OOS DSR >= 0.95 AND PBO <= 0.25) not met"
        )

    out = Path("data/ensemble/metrics/f3b_universe25_lowfreq_20260726.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
