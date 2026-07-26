#!/usr/bin/env python3
"""Cash-and-carry spot-perp delta-neutral basis trade.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S8
(committed f8a17ea, before this script produced any output). Grid, windows,
cost model and gates are fixed there. Do not edit them after seeing results.

Mechanism: long spot + short perp, equal notional, delta-neutral. Only entered
when funding is positive (the reverse trade needs spot short/borrow
infrastructure this repo doesn't model -- asymmetric by design, not an
oversight). Removes the price risk that killed F3-A's naked funding-carry bet;
the only P&L here is (1) funding accrual and (2) basis (perp-spot spread)
drift, both computed directly from real spot+perp price data.

  variants     always-on (0 free parameters: enter once, hold continuously,
               exit once at window end, regardless of funding sign) vs.
               gated-7d / gated-30d (flat whenever the trailing N-day mean
               funding turns negative, to avoid paying it -- but pays extra
               open/close cost on every toggle)
  pnl/day      funding_accrual (realized funding prints while positioned) +
               basis_pnl (short-perp P&L from basis narrowing/widening,
               computed from actual spot & perp closes, not assumed away)
  cost         10bps round trip PER LEG (spot leg + perp leg), charged on
               every open and every close (2 legs x 10bps = 20bps per toggle)
  gate         exploration daily net return mean > 0 AND day-block bootstrap
               t > 3 (continuous holding, so day-blocked like F3-B, not
               event-blocked like F5)
  val/oos      2025-09..12 / 2026-01..03, entered only if the gate passes
  promotion    OOS DSR >= 0.95 AND PBO <= 0.25
"""

from __future__ import annotations

import glob
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

FUNDING_DIR = ROOT / "data/research/funding_extracted"
PERP_5M = {a: ROOT / f"binance_data/klines/{a}/{a}-5m-api.csv" for a in ("ETHUSDT", "BTCUSDT", "SOLUSDT")}
SPOT_5M = {a: ROOT / f"binance_data/klines_spot/{a}/{a}-5m-spot-api.csv" for a in ("ETHUSDT", "BTCUSDT", "SOLUSDT")}
ASSETS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]

LEG_COST_BPS = 10.0  # round trip per leg, matches F3-A/F3-B/F5 convention
GATE_T = 3.0

EXPL_START = pd.Timestamp("2024-01-01", tz="UTC")
EXPL_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31", tz="UTC")


def load_funding(asset: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(FUNDING_DIR / asset / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(subset=["calc_time"])
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[["ts", "last_funding_rate"]]


def load_daily_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("ts")["close"].resample("1D").last().dropna()


def build_asset_daily(asset: str) -> pd.DataFrame:
    """Daily frame with spot close, perp close, basis (perp-spot)/spot, and
    that day's realized funding sum (3 prints/day, sign as received when short
    perp -- positive funding paid by longs to shorts, so a short-perp holder
    RECEIVES it directly, no sign flip needed)."""
    spot = load_daily_close(SPOT_5M[asset]).rename("spot")
    perp = load_daily_close(PERP_5M[asset]).rename("perp")
    funding = load_funding(asset)
    funding_daily = funding.set_index("ts")["last_funding_rate"].resample("1D").sum().rename("funding_sum")

    df = pd.concat([spot, perp, funding_daily], axis=1).dropna(subset=["spot", "perp"])
    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["funding_sum"] = df["funding_sum"].fillna(0.0)
    return df


def run_variant(daily: pd.DataFrame, gate_window_days: int | None) -> pd.DataFrame:
    """gate_window_days=None -> always-on. Otherwise: positioned only while the
    trailing gate_window_days mean funding_sum is > 0 (computed causally on
    prior days only, current day's own funding not included in its own gate)."""
    n = len(daily)
    spot = daily["spot"].to_numpy()
    perp = daily["perp"].to_numpy()
    funding_sum = daily["funding_sum"].to_numpy()

    if gate_window_days is None:
        positioned = np.ones(n, dtype=bool)
    else:
        trailing_mean = (
            pd.Series(funding_sum).shift(1).rolling(gate_window_days, min_periods=gate_window_days).mean()
        )
        positioned = (trailing_mean > 0).to_numpy()
        positioned = np.nan_to_num(positioned, nan=0.0).astype(bool)

    daily_return = np.zeros(n)
    cost = np.zeros(n)
    was_positioned = False

    for i in range(n):
        is_positioned = bool(positioned[i])
        if is_positioned and not was_positioned:
            cost[i] += 2 * LEG_COST_BPS / 1e4  # open: spot leg + perp leg
        if (not is_positioned) and was_positioned:
            cost[i] += 2 * LEG_COST_BPS / 1e4  # close
        if is_positioned:
            daily_return[i] = funding_sum[i]  # received (short perp, funding>0)
            if was_positioned and i > 0:
                # long spot + short perp, equal notional: exact price P&L is
                # spot_return - perp_return (no small-basis approximation).
                spot_ret = (spot[i] - spot[i - 1]) / spot[i - 1]
                perp_ret = (perp[i] - perp[i - 1]) / perp[i - 1]
                daily_return[i] += spot_ret - perp_ret
            # opening day: no prior-day price move attributable to a position
            # that didn't exist yet.
        was_positioned = is_positioned

    return pd.DataFrame({"gross": daily_return, "cost": cost}, index=daily.index)


def window_stats(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    w = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if len(w) < 30:
        return {"n_days": int(len(w)), "insufficient": True}
    net = (w["gross"] - w["cost"]).to_numpy()
    mu, sd = float(net.mean()), float(net.std(ddof=1))
    days_for_t3 = float((GATE_T * sd / mu) ** 2) if mu > 0 and sd > 0 else None
    boot = None
    if len(net) >= 30:
        rng = np.random.default_rng(20260726)
        boot_means = np.array([rng.choice(net, size=len(net), replace=True).mean() for _ in range(3000)])
        se = float(np.std(boot_means))
        boot = {"observed_mean": mu, "boot_se": se, "t_stat": float(mu / se) if se > 1e-12 else None}
    return {
        "n_days": int(len(w)),
        "n_days_positioned": int(w["gross"].astype(bool).sum()),
        "gross_mean_daily": float(w["gross"].mean()),
        "cost_mean_daily": float(w["cost"].mean()),
        "net_mean_daily": mu,
        "net_sd_daily": sd,
        "days_needed_for_t3": days_for_t3,
        "years_needed_for_t3": (days_for_t3 / 365.0) if days_for_t3 else None,
        "annualized_net_pct": mu * 365 * 100.0,
        "sharpe_like": sharpe(net),
        "bootstrap": boot,
    }


def funding_level_by_window(per_asset_daily: dict[str, pd.DataFrame]) -> dict:
    """Diagnostic: is the funding LEVEL itself stationary across windows, or is
    any exploration-window edge just a snapshot of a since-decayed regime?"""
    windows = {"exploration": (EXPL_START, EXPL_END), "val": (VAL_START, VAL_END), "oos": (OOS_START, OOS_END)}
    out: dict = {}
    for a, daily in per_asset_daily.items():
        out[a] = {}
        for name, (s, e) in windows.items():
            w = daily.loc[(daily.index >= s) & (daily.index <= e)]
            out[a][name] = {
                "funding_mean_bps_per_day": float(w["funding_sum"].mean() * 1e4) if len(w) else None,
                "basis_mean_bps": float(w["basis"].mean() * 1e4) if len(w) else None,
            }
    return out


def passes_gate(stats: dict) -> bool:
    if stats.get("insufficient"):
        return False
    t = stats.get("bootstrap", {}).get("t_stat") if stats.get("bootstrap") else None
    return bool(stats["net_mean_daily"] > 0 and t is not None and t > GATE_T)


def main() -> None:
    report = {
        "stage": "basis-carry",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S8 (commit f8a17ea)",
        "variants": ["always_on", "gated_7d", "gated_30d"],
        "leg_cost_bps_roundtrip": LEG_COST_BPS,
        "gate": f"exploration net_mean_daily > 0 AND bootstrap t > {GATE_T}",
    }

    per_asset_daily = {a: build_asset_daily(a) for a in ASSETS}
    report["data_coverage"] = {
        a: {"n_days": int(len(d)), "min_date": str(d.index.min().date()), "max_date": str(d.index.max().date())}
        for a, d in per_asset_daily.items()
    }

    variant_defs = {"always_on": None, "gated_7d": 7, "gated_30d": 30}
    per_asset_frames = {}
    for a, daily in per_asset_daily.items():
        per_asset_frames[a] = {name: run_variant(daily, w) for name, w in variant_defs.items()}

    # Equal-weighted 3-asset portfolio per variant (primary decision series).
    portfolio_frames = {}
    for name in variant_defs:
        combined = pd.concat(
            [per_asset_frames[a][name] for a in ASSETS], axis=1,
            keys=ASSETS,
        )
        gross = combined.xs("gross", axis=1, level=1).mean(axis=1)
        cost = combined.xs("cost", axis=1, level=1).mean(axis=1)
        portfolio_frames[name] = pd.DataFrame({"gross": gross, "cost": cost}).dropna()

    rows = []
    for name in variant_defs:
        expl = window_stats(portfolio_frames[name], EXPL_START, EXPL_END)
        rows.append({"variant": name, "exploration": expl, "gate_pass": passes_gate(expl)})
    report["exploration_grid"] = rows
    report["per_asset_exploration"] = {
        a: {name: window_stats(per_asset_frames[a][name], EXPL_START, EXPL_END) for name in variant_defs}
        for a in ASSETS
    }
    report["funding_level_by_window"] = funding_level_by_window(per_asset_daily)

    expl_sharpes = np.array([r["exploration"].get("sharpe_like", 0.0) for r in rows], dtype=np.float64)
    sr_std = float(np.std(expl_sharpes, ddof=1))
    report["noise_floor"] = {
        "n_variants": len(rows),
        "trial_sharpe_std": sr_std,
        "expected_max_sharpe": expected_max_sharpe(len(rows), sr_std),
        "best_observed_sharpe": float(np.max(expl_sharpes)),
        "note": "always_on has zero free parameters; clearing this floor there is stronger evidence than a gated variant clearing it.",
    }

    passing = [r for r in rows if r["gate_pass"]]
    report["n_variants_passing_exploration"] = len(passing)

    if not passing:
        report["verdict"] = (
            f"KILLED at exploration -- 0/{len(rows)} variants clear net_mean_daily > 0 with t > {GATE_T}. "
            "Per the pre-registration, val/OOS are not run."
        )
    else:
        best = max(passing, key=lambda r: r["exploration"]["net_mean_daily"])
        name = best["variant"]
        f = portfolio_frames[name]
        val = window_stats(f, VAL_START, VAL_END)
        oos = window_stats(f, OOS_START, OOS_END)

        oos_net = (f["gross"] - f["cost"]).loc[(f.index >= OOS_START) & (f.index <= OOS_END)].to_numpy()
        dsr = deflated_sharpe_ratio(oos_net, expl_sharpes)

        net_df = pd.DataFrame(
            {n: portfolio_frames[n]["gross"] - portfolio_frames[n]["cost"] for n in variant_defs}
        ).dropna()
        net_df = net_df.loc[net_df.index <= OOS_END]
        pbo = pbo_cscv(net_df.to_numpy(), n_splits=10) if len(net_df) >= 30 else {"pbo": None}

        report["selected_variant"] = name
        report["val"] = val
        report["oos"] = oos
        report["oos_deflated_sharpe"] = dsr
        report["pbo_cscv"] = pbo
        promoted = bool(dsr.get("passes_95") and pbo.get("pbo") is not None and pbo["pbo"] <= 0.25)
        report["promotion_pass"] = promoted
        report["verdict"] = (
            "PROMOTION CRITERIA MET" if promoted
            else "gate passed at exploration but promotion criteria (OOS DSR >= 0.95 AND PBO <= 0.25) not met"
        )

    out = Path("data/ensemble/metrics/basis_carry_20260726.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
