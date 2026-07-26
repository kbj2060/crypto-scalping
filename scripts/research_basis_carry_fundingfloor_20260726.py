#!/usr/bin/env python3
"""Cash-and-carry basis trade, redesigned with a funding-level floor filter.

PRE-REGISTERED in docs/mechanical_trading_research_synthesis_20260726.md S8.4
(committed 612b054, before this script produced any output). Grid, windows,
cost model and gates are fixed there. Do not edit them after seeing results.

Direct follow-up to research_basis_carry_20260726.py (S8.3): the always-on
variant's exploration edge (t=18.09) collapsed in OOS because the funding
RATE LEVEL itself decayed non-stationarily (ETH 2.72->1.11->0.01 bps/day, SOL
went negative). This redesign steps aside exactly when that decay is already
visible in the trailing average, rather than staying blindly positioned.

Two contamination guards, since S8.3 already looked at 2026-01-01..03-31 and
knows the always-on variant failed there:
  1. Floor thresholds are fixed ex-ante round numbers (0.5/1.0/2.0 bps/day,
     chosen from the economics of the 20bps-per-toggle cost), NOT fit to this
     data's distribution.
  2. Promotion is judged ONLY on 2026-04-01..06-30 -- a window never opened
     this session (funding data itself ends 2026-06-30). 2026-01-03 is
     reported for context only and is NOT used for the promotion decision.

  filter    causal 90-day trailing mean funding (shift(1), no lookahead) must
            exceed theta to hold the position; otherwise flat
  variants  always_on (S8.3's baseline, theta=-inf i.e. no filter) +
            theta in {0.5, 1.0, 2.0} bps/day = 4 variants
  cost      10bps round trip per leg, charged on every open/close toggle
  gate      exploration (2024-01..2025-08) daily net mean > 0 AND
            day-block bootstrap t > 3
  promotion FRESH holdout (2026-04..06) DSR >= 0.95 AND PBO <= 0.25
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core.selection_stats import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    sharpe,
)
from research_basis_carry_20260726 import (  # noqa: E402
    ASSETS,
    LEG_COST_BPS,
    build_asset_daily,
)

GATE_T = 3.0
FLOOR_GRID_BPS_PER_DAY = [0.5, 1.0, 2.0]
FILTER_WINDOW_DAYS = 90

EXPL_START = pd.Timestamp("2024-01-01", tz="UTC")
EXPL_END = pd.Timestamp("2025-08-31", tz="UTC")
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
# Already examined in S8.3 -- context only, never a promotion basis here.
CONTEXT_OOS_START, CONTEXT_OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31", tz="UTC")
# Genuinely fresh this session -- the only window promotion is judged on.
FRESH_OOS_START, FRESH_OOS_END = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")


def run_variant(daily: pd.DataFrame, floor_bps_per_day: float | None) -> pd.DataFrame:
    """floor_bps_per_day=None -> always-on (S8.3 baseline, no filter)."""
    n = len(daily)
    spot = daily["spot"].to_numpy()
    perp = daily["perp"].to_numpy()
    funding_sum = daily["funding_sum"].to_numpy()

    if floor_bps_per_day is None:
        positioned = np.ones(n, dtype=bool)
    else:
        theta = floor_bps_per_day / 1e4
        trailing_mean = (
            pd.Series(funding_sum).shift(1).rolling(FILTER_WINDOW_DAYS, min_periods=FILTER_WINDOW_DAYS).mean()
        )
        positioned = (trailing_mean > theta).to_numpy()
        positioned = np.nan_to_num(positioned, nan=0.0).astype(bool)

    daily_return = np.zeros(n)
    cost = np.zeros(n)
    was_positioned = False

    for i in range(n):
        is_positioned = bool(positioned[i])
        if is_positioned and not was_positioned:
            cost[i] += 2 * LEG_COST_BPS / 1e4
        if (not is_positioned) and was_positioned:
            cost[i] += 2 * LEG_COST_BPS / 1e4
        if is_positioned:
            daily_return[i] = funding_sum[i]
            if was_positioned and i > 0:
                spot_ret = (spot[i] - spot[i - 1]) / spot[i - 1]
                perp_ret = (perp[i] - perp[i - 1]) / perp[i - 1]
                daily_return[i] += spot_ret - perp_ret
        was_positioned = is_positioned

    return pd.DataFrame({"gross": daily_return, "cost": cost}, index=daily.index)


def window_stats(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    w = frame.loc[(frame.index >= start) & (frame.index <= end)]
    if len(w) < 30:
        return {"n_days": int(len(w)), "insufficient": True}
    net = (w["gross"] - w["cost"]).to_numpy()
    mu, sd = float(net.mean()), float(net.std(ddof=1))
    days_for_t3 = float((GATE_T * sd / mu) ** 2) if mu > 0 and sd > 0 else None
    rng = np.random.default_rng(20260726)
    boot_means = np.array([rng.choice(net, size=len(net), replace=True).mean() for _ in range(3000)])
    se = float(np.std(boot_means))
    boot = {"observed_mean": mu, "boot_se": se, "t_stat": float(mu / se) if se > 1e-12 else None}
    return {
        "n_days": int(len(w)),
        "n_days_positioned": int((w["gross"] != 0).sum()),
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


def passes_gate(stats: dict) -> bool:
    if stats.get("insufficient"):
        return False
    t = stats.get("bootstrap", {}).get("t_stat") if stats.get("bootstrap") else None
    return bool(stats["net_mean_daily"] > 0 and t is not None and t > GATE_T)


def main() -> None:
    report = {
        "stage": "basis-carry-fundingfloor",
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "preregistration": "docs/mechanical_trading_research_synthesis_20260726.md S8.4 (commit 612b054)",
        "variants": ["always_on"] + [f"floor_{t}bps" for t in FLOOR_GRID_BPS_PER_DAY],
        "leg_cost_bps_roundtrip": LEG_COST_BPS,
        "filter_window_days": FILTER_WINDOW_DAYS,
        "context_window_already_seen": {"start": str(CONTEXT_OOS_START.date()), "end": str(CONTEXT_OOS_END.date()),
                                         "note": "S8.3 already reported this window's outcome -- reported here for context only, NOT a promotion basis"},
        "fresh_holdout_window": {"start": str(FRESH_OOS_START.date()), "end": str(FRESH_OOS_END.date()),
                                  "note": "Never opened this session -- the only window promotion is judged on"},
    }

    per_asset_daily = {a: build_asset_daily(a) for a in ASSETS}
    variant_defs: dict[str, float | None] = {"always_on": None}
    for t in FLOOR_GRID_BPS_PER_DAY:
        variant_defs[f"floor_{t}bps"] = t

    per_asset_frames = {a: {name: run_variant(d, w) for name, w in variant_defs.items()} for a, d in per_asset_daily.items()}

    portfolio_frames = {}
    for name in variant_defs:
        combined = pd.concat([per_asset_frames[a][name] for a in ASSETS], axis=1, keys=ASSETS)
        gross = combined.xs("gross", axis=1, level=1).mean(axis=1)
        cost = combined.xs("cost", axis=1, level=1).mean(axis=1)
        portfolio_frames[name] = pd.DataFrame({"gross": gross, "cost": cost}).dropna()

    rows = []
    for name in variant_defs:
        expl = window_stats(portfolio_frames[name], EXPL_START, EXPL_END)
        rows.append({"variant": name, "exploration": expl, "gate_pass": passes_gate(expl)})
    report["exploration_grid"] = rows

    expl_sharpes = np.array([r["exploration"].get("sharpe_like", 0.0) for r in rows], dtype=np.float64)
    sr_std = float(np.std(expl_sharpes, ddof=1))
    report["noise_floor"] = {
        "n_variants": len(rows),
        "trial_sharpe_std": sr_std,
        "expected_max_sharpe": expected_max_sharpe(len(rows), sr_std),
        "best_observed_sharpe": float(np.max(expl_sharpes)),
    }

    passing = [r for r in rows if r["gate_pass"]]
    report["n_variants_passing_exploration"] = len(passing)

    # Report VAL + both OOS windows for every variant that passed exploration,
    # for full transparency, but promotion is decided on the fresh window only.
    for r in rows:
        name = r["variant"]
        f = portfolio_frames[name]
        r["val"] = window_stats(f, VAL_START, VAL_END)
        r["context_oos_2026_01_03_already_seen"] = window_stats(f, CONTEXT_OOS_START, CONTEXT_OOS_END)
        r["fresh_oos_2026_04_06"] = window_stats(f, FRESH_OOS_START, FRESH_OOS_END)

    if not passing:
        report["verdict"] = (
            f"KILLED at exploration -- 0/{len(rows)} variants clear net_mean_daily > 0 with t > {GATE_T}. "
            "Fresh holdout not used for promotion (none of the variants qualify)."
        )
    else:
        best = max(passing, key=lambda r: r["exploration"]["net_mean_daily"])
        name = best["variant"]
        f = portfolio_frames[name]

        fresh_net = (f["gross"] - f["cost"]).loc[(f.index >= FRESH_OOS_START) & (f.index <= FRESH_OOS_END)].to_numpy()
        dsr = deflated_sharpe_ratio(fresh_net, expl_sharpes)

        net_df = pd.DataFrame(
            {n: portfolio_frames[n]["gross"] - portfolio_frames[n]["cost"] for n in variant_defs}
        ).dropna()
        net_df = net_df.loc[net_df.index <= FRESH_OOS_END]
        pbo = pbo_cscv(net_df.to_numpy(), n_splits=10) if len(net_df) >= 30 else {"pbo": None}

        report["selected_variant"] = name
        report["promotion_basis_fresh_oos_deflated_sharpe"] = dsr
        report["promotion_basis_pbo_cscv"] = pbo
        promoted = bool(dsr.get("passes_95") and pbo.get("pbo") is not None and pbo["pbo"] <= 0.25)
        report["promotion_pass"] = promoted
        report["verdict"] = (
            "PROMOTION CRITERIA MET (fresh holdout)" if promoted
            else "gate passed at exploration but promotion criteria on the FRESH holdout (DSR>=0.95 AND PBO<=0.25) not met"
        )

    out = Path("data/ensemble/metrics/basis_carry_fundingfloor_20260726.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
