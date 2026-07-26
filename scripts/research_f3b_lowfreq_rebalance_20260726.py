#!/usr/bin/env python3
"""F3-B-LF: cross-sectional momentum with low-frequency rebalancing.

F3-B (2026-07-19) killed daily-rebalanced cross-sectional momentum, but the kill
reason was cost, not absence of signal: gross return was positive at every
lookback and rose with k, while a flat 20bps/day rebalance charge swamped it.
This tests the follow-up the F3-B write-up left open -- rebalance less often,
pay the charge less often, and see whether the signal survives being held stale.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S5.1
(committed 4af3626, before this script was written). Grid, windows, cost model
and gates are fixed there. Do not edit them after seeing output.

  grid       k in {14,30,60} days x rebalance in {3,7,14} days = 9 variants
  universe   ETH/BTC/SOL only -- expanding it is a separate experiment
  cost       cost1 = 10bps round trip per leg (primary), cost3 = 30bps
  gate       exploration net_cost1 > 0 AND day-block bootstrap t > 3
  val/OOS    2025-09..12 / 2026-01..03, entered only if the gate passes
  promotion  OOS DSR >= 0.95 AND PBO <= 0.25

Signal convention is inherited unchanged from research_f3b_cross_sectional_
momentum_20260719.py so the numbers stay comparable: rank 3 assets by k-day log
return, long the top / short the bottom at half notional each (dollar neutral).

Two cost models are reported. FLAT repeats F3-B's conservative assumption --
both legs pay a full round trip at every rebalance even if the ranking did not
move. TURNOVER charges only the legs that actually change. FLAT is the primary
gate so that lowering the rebalance frequency cannot be confused with quietly
loosening the cost model.
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

KLINE_5M = {
    "ETHUSDT": "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
    "BTCUSDT": "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
    "SOLUSDT": "binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv",
}
ASSETS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

K_GRID = [14, 30, 60]
REBAL_GRID = [3, 7, 14]
COST1_ROUNDTRIP_BPS = 10.0
COST3_ROUNDTRIP_BPS = 30.0

EXPL_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31", tz="UTC")
# Deliberately left untouched by the pre-registered gate; reported only if OOS passes.
BONUS_START, BONUS_END = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-07-20", tz="UTC")

GATE_T = 3.0


def load_daily_close(asset: str) -> pd.Series:
    df = pd.read_csv(KLINE_5M[asset], usecols=["timestamp", "close"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("ts")["close"].resample("1D").last().dropna()


def day_block_bootstrap_tstat(returns: np.ndarray, n_boot: int = 3000, seed: int = 20260726) -> dict | None:
    """Bootstrap t-stat of the mean. Inherited from the F3-B script unchanged."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 30:
        return None
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(r, size=r.size, replace=True).mean() for _ in range(n_boot)])
    observed = float(np.mean(r))
    se = float(np.std(boot))
    return {
        "observed_mean": observed,
        "boot_se": se,
        "t_stat": float(observed / se) if se > 1e-12 else None,
    }


def run_variant(price_df: pd.DataFrame, k: int, rebal: int) -> pd.DataFrame:
    """Daily gross PnL and per-model cost for one (k, rebal) variant.

    The decision at the close of a rebalance day uses only k-day past returns
    known at that close; the resulting position earns each following day's
    return until the next rebalance. Costs land on the rebalance day itself.
    """
    log_px = np.log(price_df)
    past_ret = log_px - log_px.shift(k)
    daily_ret = log_px.diff()

    valid = ~past_ret.isna().any(axis=1).to_numpy()
    if not valid.any():
        raise ValueError(f"k={k} leaves no valid rows")
    first = int(np.argmax(valid))

    n = len(price_df)
    gross = np.zeros(n)
    cost_flat = np.zeros(n)
    cost_turn = np.zeros(n)
    cur_long = cur_short = None
    leg_cost = COST1_ROUNDTRIP_BPS / 1e4

    for i in range(first, n):
        # PnL of the position carried into today, realised on today's move.
        if cur_long is not None:
            gross[i] = 0.5 * daily_ret[cur_long].iloc[i] - 0.5 * daily_ret[cur_short].iloc[i]

        if (i - first) % rebal:
            continue
        row = past_ret.iloc[i]
        if row.isna().any():
            continue
        new_long, new_short = row.idxmax(), row.idxmin()
        if new_long == new_short:
            continue
        n_changed = int(new_long != cur_long) + int(new_short != cur_short)
        cost_flat[i] = 2 * leg_cost          # F3-B's conservative assumption
        cost_turn[i] = n_changed * leg_cost  # only the legs that actually move
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(
        {"gross": gross, "cost_flat": cost_flat, "cost_turn": cost_turn},
        index=price_df.index,
    ).iloc[first:]


def window_stats(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> dict:
    w = frame.loc[(frame.index <= end) if start is None else (frame.index >= start) & (frame.index <= end)]
    if len(w) < 30:
        return {"n_days": int(len(w)), "insufficient": True}

    cost3_scale = COST3_ROUNDTRIP_BPS / COST1_ROUNDTRIP_BPS
    net_flat = (w["gross"] - w["cost_flat"]).to_numpy()
    net_turn = (w["gross"] - w["cost_turn"]).to_numpy()
    net_flat3 = (w["gross"] - w["cost_flat"] * cost3_scale).to_numpy()

    # How much history the t>3 gate would need at this effect size and vol.
    # Separates "no effect" from "test too weak to see one" -- with 3 assets the
    # daily cross-sectional vol dwarfs the effect, so this is usually decisive.
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
        "net_cost1_flat_mean_daily": float(net_flat.mean()),
        "net_cost1_turn_mean_daily": float(net_turn.mean()),
        "net_cost3_flat_mean_daily": float(net_flat3.mean()),
        "annualized_net_cost1_flat_pct": float(net_flat.mean() * 365 * 100.0),
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
    report = {
        "stage": "F3-B-LF",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S5.1 (commit 4af3626)",
        "grid": {"k_days": K_GRID, "rebalance_days": REBAL_GRID},
        "primary_cost_model": "FLAT (both legs pay a full round trip every rebalance)",
        "gate": f"exploration net_cost1_flat > 0 AND bootstrap t > {GATE_T}",
    }

    closes = {a: load_daily_close(a) for a in ASSETS}
    idx = closes[ASSETS[0]].index
    for a in ASSETS[1:]:
        idx = idx.intersection(closes[a].index)
    price_df = pd.DataFrame({a: closes[a].reindex(idx) for a in ASSETS}).dropna().sort_index()
    report["data_coverage"] = {
        "n_days": int(len(price_df)),
        "min_date": str(price_df.index.min().date()),
        "max_date": str(price_df.index.max().date()),
        "note": (
            "SOLUSDT starts 2024-06-01, so the common index begins there rather than the "
            "2024-01 written in the pre-registration. Same constraint the original F3-B ran "
            "under; forced by data availability, not by choice."
        ),
    }

    frames, rows = {}, []
    for k, rebal in itertools.product(K_GRID, REBAL_GRID):
        f = run_variant(price_df, k, rebal)
        frames[(k, rebal)] = f
        expl = window_stats(f, None, EXPL_END)
        rows.append({"k": k, "rebalance_days": rebal, "exploration": expl, "gate_pass": passes_gate(expl)})

    report["exploration_grid"] = rows

    # Noise floor for a 9-variant search, per the pre-registration.
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
        # Pre-registered tie-break: highest exploration net return under the primary cost model.
        best = max(passing, key=lambda r: r["exploration"]["net_cost1_flat_mean_daily"])
        key = (best["k"], best["rebalance_days"])
        f = frames[key]
        val = window_stats(f, VAL_START, VAL_END)
        oos = window_stats(f, OOS_START, OOS_END)

        oos_net = (f["gross"] - f["cost_flat"]).loc[
            (f.index >= OOS_START) & (f.index <= OOS_END)
        ].to_numpy()
        dsr = deflated_sharpe_ratio(oos_net, expl_sharpes)

        # Variants start on different dates (k=60 needs a longer warm-up than
        # k=14), so align on the date index and drop the ragged head.
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
        if promoted:
            report["bonus_unpeeked_2026_04_07"] = window_stats(f, BONUS_START, BONUS_END)

    out = Path("data/ensemble/metrics/f3b_lowfreq_rebalance_20260726.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
