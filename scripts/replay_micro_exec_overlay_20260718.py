"""Paired replay gate for MicroExec v1 (docs/micro_scalp_1m_design_20260718.md section 4).

For a dense grid of synthetic Layer-1 intents (every minute with valid overlay data, both
sides), compare the entry price achieved by the overlay (wait up to K minutes for a
contrarian-extreme minute, veto-aware, forced at deadline) against immediate execution at the
intent minute. Both arms execute at the bar-open of their execution minute (same price proxy),
so the comparison isolates timing. improvement_bps = side * (baseline - exec) / baseline * 1e4.

Controls:
  - random-wait arm (execute at uniform random minute in [0, K]) must come out ~0: proves any
    overlay gain is signal, not drift/grid artifact.
Significance: daily-block t-stat (intents within a day are heavily overlapping/correlated).

Causality: overlay frame is indexed by first-usable decision minute (ts+2min inside
prepare_overlay_frame). The intent at minute D may execute at D (using the row written at
D-1..D-45s) — same information a live caller of the scanner cache would have, or staler.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from trading_bot_modules.micro_exec_overlay import prepare_overlay_frame  # noqa: E402
import analyze_microstructure_edge_20260718 as base  # noqa: E402

OUT_JSON = os.path.join(_ROOT, "data", "ensemble", "reports", "micro_exec_overlay_replay_20260718.json")

EXEC_Z = [0.8, 1.28, 1.64]
DEADLINES = [5, 10, 15]
RNG = np.random.default_rng(42)


def first_passage_exec(cond: np.ndarray, valid_forced: np.ndarray, deadline: int) -> np.ndarray:
    """For each intent index i, first offset j in [0, deadline) with cond[i+j]; else deadline.

    Returns j array (len == len(cond) - deadline); -1 where the forced bar is invalid.
    """
    n = len(cond) - deadline
    j_exec = np.full(n, deadline, dtype=np.int64)
    unset = np.ones(n, dtype=bool)
    for j in range(deadline):
        hit = unset & cond[j:j + n]
        j_exec[hit] = j
        unset &= ~hit
    j_exec[unset & ~valid_forced[deadline:deadline + n]] = -1
    return j_exec


def daily_t(x: pd.Series) -> tuple[float, float, int]:
    d = x.groupby(x.index.date).mean()
    if len(d) < 10 or d.std(ddof=1) == 0:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)


def main() -> None:
    micro = base.load_micro()
    ov = prepare_overlay_frame(micro)
    kl = pd.read_csv(base.KLINES, parse_dates=["timestamp"],
                     usecols=["timestamp", "open"]).set_index("timestamp").sort_index()

    # Continuous minute grid over the overlap window.
    t0 = max(ov.index.min(), kl.index.min())
    t1 = min(ov.index.max(), kl.index.max())
    grid = pd.date_range(t0, t1, freq="1min")
    open_px = kl["open"].reindex(grid)
    score = ov["score"].reindex(grid)
    veto = ov["veto"].reindex(grid).fillna(False).to_numpy(bool)
    valid_px = open_px.notna().to_numpy()
    print(f"grid: {len(grid):,} minutes  {t0} -> {t1}  "
          f"px_cov={valid_px.mean():.1%} score_cov={score.notna().mean():.1%}")

    results = []
    max_k = max(DEADLINES)
    score_np = score.to_numpy()
    for side_name, side in [("long", 1.0), ("short", -1.0)]:
        s_side = side * score_np
        for ez, k in itertools.product(EXEC_Z, DEADLINES):
            cond = np.nan_to_num(s_side, nan=-np.inf) >= ez
            cond &= ~veto
            cond &= valid_px
            j = first_passage_exec(cond, valid_px, k)
            n = len(j)
            # intents: require valid baseline px, defined score at intent, resolvable exec bar
            intent_ok = valid_px[:n] & np.isfinite(score_np[:n]) & (j >= 0)
            idx = np.nonzero(intent_ok)[0]
            base_px = open_px.to_numpy()[idx]
            exec_px = open_px.to_numpy()[idx + j[idx]]
            imp = side * (base_px - exec_px) / base_px * 1e4
            imp_s = pd.Series(imp, index=grid[idx])
            mean_d, t_d, ndays = daily_t(imp_s)

            # random-wait control (same intents, uniform j in [0,k], skip invalid px)
            j_rand = RNG.integers(0, k + 1, size=len(idx))
            ok_r = valid_px[idx + j_rand]
            imp_r = side * (base_px[ok_r] - open_px.to_numpy()[idx + j_rand][ok_r]) / base_px[ok_r] * 1e4
            mean_r, t_r, _ = daily_t(pd.Series(imp_r, index=grid[idx[ok_r]]))

            waited = j[idx] > 0
            forced = j[idx] == k
            row = {
                "side": side_name, "exec_z": ez, "deadline_min": k,
                "n_intents": int(len(idx)),
                "improve_mean_bps": round(float(np.mean(imp)), 3),
                "improve_daily_mean_bps": round(mean_d, 3),
                "improve_daily_t": round(t_d, 2), "n_days": ndays,
                "pct_improved": round(float(np.mean(imp > 0)), 3),
                "pct_waited": round(float(np.mean(waited)), 3),
                "pct_forced_deadline": round(float(np.mean(forced)), 3),
                "mean_delay_min": round(float(np.mean(j[idx])), 2),
                "p05_bps": round(float(np.percentile(imp, 5)), 2),
                "p95_bps": round(float(np.percentile(imp, 95)), 2),
                "random_wait_mean_bps": round(mean_r, 3),
                "random_wait_t": round(t_r, 2),
            }
            results.append(row)
            print(f"{side_name:5s} z>={ez:<4} K={k:<2} n={len(idx):>6,} "
                  f"mean={row['improve_mean_bps']:+.2f}bps t={t_d:+.2f} "
                  f"improved={row['pct_improved']:.0%} waited={row['pct_waited']:.0%} "
                  f"forced={row['pct_forced_deadline']:.0%} delay={row['mean_delay_min']:.1f}m "
                  f"| rand={mean_r:+.2f}bps(t{t_r:+.1f})")

    veto_rate = float(pd.Series(veto, index=grid)[score.notna()].mean())
    report = {
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "window_utc": [str(t0), str(t1)],
        "n_grid_minutes": int(len(grid)),
        "veto_rate": round(veto_rate, 4),
        "price_proxy": "1m bar open of execution minute (both arms identical proxy)",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "future_rows_used_for_entry": False,
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nveto rate on valid minutes: {veto_rate:.2%}")
    print(f"saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
