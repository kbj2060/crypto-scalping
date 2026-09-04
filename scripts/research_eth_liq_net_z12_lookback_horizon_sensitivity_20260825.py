#!/usr/bin/env python3
"""ETH -- liq_net_z_12 lookback/horizon PARAMETER SENSITIVITY check, 2026-08-25.

NOT a parameter search for a better config to deploy -- this repo has already paid for that
mistake many times over (Elliott/Gann "tune it and it'll pass" hypothesis directly refuted with
real data, eth_omega461_atr_tpsl_floor_recalibration_closed_20260815.md's 6 failed recalibration
attempts, eth_dashboard7_loosened_threshold_lift_rejected_20260824.md's threshold-loosening
rejection -- see eth_tuning_hypothesis_directly_refuted_elliott_gann_harmonic_20260824.md for the
running count). This script runs the FULL grid ONCE, reports EVERY cell (no cherry-picking, no
re-running with a narrower grid to chase a better-looking number), and does NOT change the
deployed NET_WIN=12/TRAIL_WIN=2880/h in {1,3,12} config regardless of what any individual cell
shows.

The question is narrower than "which config is best": is the already-reverified §12 result (5m/
15m PASS, 1h a near-miss with consistent cross-window sign, see eth_liquidation_s13_s14_early_
peek_status_20260825.md's 2026-08-25 addendum) a fragile artifact of these exact three numbers,
or does it hold up across a neighborhood of nearby lookback/horizon choices? A signal whose
significance evaporates the moment a lookback is nudged by a few minutes is a materially weaker
kind of finding than one that is stable across reasonable neighbors -- this is a robustness
read, not a selection procedure.

Same data (tail_risk_1m valid-since 2026-07-18 15:03 UTC, 37-day window), same shift_z
(verbatim from research_eth_weekly_oi_growth_hong_yogo_cheap_gate_20260824.py), same formula
structure as /tmp/.../scratchpad/s12_liq_net_z12_direction_validation.py -- only the NET_WIN/
TRAIL_WIN/horizon grid is new.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRATCH = "/tmp/claude-1000/-home-kbj20-crypto-scalping/47944e2e-6ddc-4b8e-b592-a7ad73f6eb75/scratchpad"
N_PERM = 2000
MIN_SHIFT = 8

NET_WIN_GRID = [6, 12, 24]            # deployed = 12 (minutes)
TRAIL_WIN_GRID = [1440, 2880, 5760]   # deployed = 2880 (1d / 2d / 4d, minutes)
H_GRID = {"5m": 1, "15m": 3, "1h": 12, "2h": 24}  # deployed tests {5m,15m,1h}; 2h added for this check
DEPLOYED = (12, 2880)


def shift_z(x: pd.Series, y: pd.Series, seed: int = 20260825) -> tuple[float, float, int]:
    """Verbatim from research_eth_weekly_oi_growth_hong_yogo_cheap_gate_20260824.py::shift_z."""
    d = pd.concat([x, y], axis=1).dropna().to_numpy()
    n = len(d)
    if n < 40:
        return float("nan"), float("nan"), n
    obs = spearmanr(d[:, 0], d[:, 1]).statistic
    rng = np.random.default_rng(seed)
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_PERM)
    null = np.array([spearmanr(np.roll(d[:, 0], s), d[:, 1]).statistic for s in shifts])
    return obs, (obs - null.mean()) / null.std(ddof=1), n


# ---- load tail_risk_1m, same quality filter as §12 ----
tr = pd.read_csv(f"{SCRATCH}/tail_risk_1m_export2.csv")
tr["ts"] = pd.to_datetime(tr["ts"], utc=True)
tr = tr[(tr["valid_liq_stream"] == True) & (tr["ws_stale"] != True)].copy()  # noqa: E712
tr = tr.drop_duplicates("ts").sort_values("ts").set_index("ts")
full_idx = pd.date_range(tr.index.min(), tr.index.max(), freq="1min", tz="UTC")
tr = tr.reindex(full_idx)
long_ = tr["long_usd_1m"].fillna(0.0)
short_ = tr["short_usd_1m"].fillna(0.0)
total = long_ + short_

# ---- ETH 5m price + forward log-returns at every horizon in the grid ----
px = pd.read_csv(f"{SCRATCH}/eth_5m_jul15_now.csv", parse_dates=["timestamp"])
px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
px = px.sort_values("timestamp").reset_index(drop=True)
close = px["close"]
for hname, h in H_GRID.items():
    px[f"fwd_{hname}"] = np.log(close.shift(-h) / close)

n_cells = len(NET_WIN_GRID) * len(TRAIL_WIN_GRID) * len(H_GRID)
print(f"liq_net_z_12 LOOKBACK/HORIZON SENSITIVITY -- same 37-day window as §12, "
      f"{len(NET_WIN_GRID)}x{len(TRAIL_WIN_GRID)}x{len(H_GRID)}={n_cells} cells, ALL reported, "
      f"no re-optimization\ndeployed reference = NET_WIN=12min TRAIL_WIN=2880min\n{'=' * 100}")

n_pass = 0
n_total = 0
for net_win in NET_WIN_GRID:
    net_minp = max(1, int(round(0.8 * net_win)))
    net_12 = long_.rolling(net_win, min_periods=net_minp).sum() - short_.rolling(net_win, min_periods=net_minp).sum()
    for trail_win in TRAIL_WIN_GRID:
        trail_minp = max(1, int(round(0.8 * trail_win)))
        trail_mean = total.rolling(trail_win, min_periods=trail_minp).mean()
        eps = 0.01 * trail_mean
        z = (net_12 / (trail_mean + eps)).rename("z")
        z_df = z.reset_index().rename(columns={"index": "ts"})
        merged = pd.merge_asof(px[["timestamp"]], z_df, left_on="timestamp", right_on="ts", direction="backward")
        col = f"z_{net_win}_{trail_win}"
        px[col] = merged["z"].to_numpy()

        tag = "  <== DEPLOYED" if (net_win, trail_win) == DEPLOYED else ""
        print(f"\nNET_WIN={net_win:>3}min  TRAIL_WIN={trail_win:>5}min{tag}")
        for hname in H_GRID:
            d = px[[col, f"fwd_{hname}"]].dropna()
            ic, z_stat, n = shift_z(d[col], d[f"fwd_{hname}"])
            passed = (not np.isnan(z_stat)) and abs(z_stat) >= 2 and abs(ic) >= 0.025
            n_total += 1
            n_pass += int(passed)
            zfmt = "n/a" if np.isnan(z_stat) else f"{z_stat:+.2f}"
            print(f"  h={hname:>3}: IC={ic:+.4f}  shift-z={zfmt}  (n={n})  [{'PASS' if passed else 'fail'}]")

print(f"\n{'=' * 100}\n{n_pass}/{n_total} cells PASS -- reported in full. No cell selected for "
      f"redeployment; deployed config (12, 2880, h={{1,3,12}}) is unchanged by this check.\n{'=' * 100}")
