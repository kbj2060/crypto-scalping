"""ETH weekly OI-growth direction cheap-gate (Hong & Yogo 2012 port), pre-registered.

Design locked in docs/experiments/eth_candidate_weekly_oi_growth_hong_yogo_cheap_gate_20260824.md
before touching joined data. Sign fixed a priori (+, Hong-Yogo). Standard costs only
(5bp/leg -- no fee-discount scenarios, per user policy 2026-08-24). BTC/SOL replication
is report-only, never used for the verdict.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
N_PERM = 2000
MIN_SHIFT = 8
COST_PER_LEG = 5e-4  # 5bp
Z_WIN, Z_MINP = 26, 24
HALF_BOUNDARY = pd.Timestamp("2025-05-05")  # Monday
ASSETS = {
    "ETH": ("data/TOTAL_ETHUSDT_metrics_2024_2026.csv", "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"),
    "BTC": ("data/TOTAL_BTCUSDT_metrics_2024_2026.csv", "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"),
    "SOL": ("data/TOTAL_SOLUSDT_metrics_2024_2026.csv", "binance_data/klines/SOLUSDT/SOLUSDT-5m-api.csv"),
}


def weekly_frame(metrics_path: str, klines_path: str) -> pd.DataFrame:
    m = pd.read_csv(ROOT / metrics_path, usecols=["create_time", "sum_open_interest"],
                    parse_dates=["create_time"]).sort_values("create_time")
    k = pd.read_csv(ROOT / klines_path, usecols=["timestamp", "close"],
                    parse_dates=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    assert m["create_time"].max() >= pd.Timestamp("2026-08-18"), f"metrics stale: {m['create_time'].max()}"
    assert k["timestamp"].max() >= pd.Timestamp("2026-08-18"), f"klines stale: {k['timestamp'].max()}"

    start = max(m["create_time"].min(), k["timestamp"].min()).normalize()
    end = min(m["create_time"].max(), k["timestamp"].max())
    mondays = pd.date_range(start + pd.offsets.Week(weekday=0), end, freq="W-MON")
    grid = pd.DataFrame({"t": mondays})
    grid = pd.merge_asof(grid, m.rename(columns={"create_time": "ts"}), left_on="t", right_on="ts",
                         direction="backward", tolerance=pd.Timedelta("30min"))
    grid = pd.merge_asof(grid, k.rename(columns={"timestamp": "ts_k"}), left_on="t", right_on="ts_k",
                         direction="backward", tolerance=pd.Timedelta("30min"))
    grid = grid.set_index("t")
    grid["oi_g4"] = np.log(grid["sum_open_interest"] / grid["sum_open_interest"].shift(4))
    grid["oi_g1"] = np.log(grid["sum_open_interest"] / grid["sum_open_interest"].shift(1))
    grid["fwd_1w"] = np.log(grid["close"].shift(-1) / grid["close"])
    return grid


def shift_z(x: pd.Series, y: pd.Series, seed: int = 20260824) -> tuple[float, float, int]:
    d = pd.concat([x, y], axis=1).dropna().to_numpy()
    n = len(d)
    if n < 40:
        return float("nan"), float("nan"), n
    obs = spearmanr(d[:, 0], d[:, 1]).statistic
    rng = np.random.default_rng(seed)
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_PERM)
    null = np.array([spearmanr(np.roll(d[:, 0], s), d[:, 1]).statistic for s in shifts])
    return obs, (obs - null.mean()) / null.std(ddof=1), n


def econ(sub: pd.DataFrame, col: str) -> dict:
    z = (sub[col] - sub[col].rolling(Z_WIN, min_periods=Z_MINP).mean()) / \
        sub[col].rolling(Z_WIN, min_periods=Z_MINP).std().replace(0.0, np.nan)
    pos = pd.Series(0.0, index=sub.index)
    pos[z >= 1.0] = 1.0
    pos[z <= -1.0] = -1.0
    pos[z.isna() | sub["fwd_1w"].isna()] = 0.0
    legs = pos.diff().abs().fillna(pos.abs())
    pnl = float((pos * sub["fwd_1w"]).sum() - legs.sum() * COST_PER_LEG)
    fwd = sub["fwd_1w"].dropna()
    bench = max(fwd.sum(), -fwd.sum()) - 2 * COST_PER_LEG
    return {"n_active": int((pos != 0).sum()), "n_weeks": int(sub["fwd_1w"].notna().sum()),
            "pnl": pnl, "bench": float(bench), "inc": pnl - float(bench)}


def main() -> None:
    for asset, (mp, kp) in ASSETS.items():
        try:
            g = weekly_frame(mp, kp)
        except (FileNotFoundError, AssertionError) as exc:
            print(f"\n### {asset}: SKIPPED ({exc})")
            continue
        contam = spearmanr(*g[["oi_g4", "close"]].dropna().to_numpy().T).statistic
        halves = {"H1": g[g.index < HALF_BOUNDARY], "H2": g[g.index >= HALF_BOUNDARY]}
        print(f"\n### {asset} | weeks={g['fwd_1w'].notna().sum()} "
              f"({g.index.min().date()}~{g.index.max().date()}) contam(oi_g4,close)={contam:+.3f}"
              f"{' EXCLUDED' if abs(contam) >= 0.5 else ''}")
        for col in ["oi_g4", "oi_g1"]:
            ic, zf, n = shift_z(g[col], g["fwd_1w"])
            parts = []
            for hname, h in halves.items():
                hic, _, hn = shift_z(h[col], h["fwd_1w"])
                d = h[[col, "fwd_1w"]].dropna()
                hic_raw = spearmanr(d[col], d["fwd_1w"]).statistic if len(d) >= 20 else float("nan")
                parts.append(f"{hname} IC={hic_raw:+.3f}(n={len(d)})")
            print(f"  {col}: full IC={ic:+.4f} shift-z={zf:+.2f} (n={n}) | " + " ".join(parts))
        e_full = econ(g, "oi_g4")
        e_h = {h: econ(sub, "oi_g4") for h, sub in halves.items()}
        print(f"  ECON oi_g4 |z|>=1 5bp/leg: full pnl={e_full['pnl'] * 100:+.2f}% bench={e_full['bench'] * 100:+.2f}% "
              f"inc={e_full['inc'] * 100:+.2f}% (active {e_full['n_active']}/{e_full['n_weeks']}wk) | "
              + " ".join(f"{h}: inc={e['inc'] * 100:+.2f}%" for h, e in e_h.items()))
        if asset == "ETH":
            d = {h: hh[["oi_g4", "fwd_1w"]].dropna() for h, hh in halves.items()}
            ics = {h: spearmanr(v["oi_g4"], v["fwd_1w"]).statistic for h, v in d.items()}
            ic, zf, _ = shift_z(g["oi_g4"], g["fwd_1w"])
            crit1 = ic > 0 and abs(zf) >= 2
            crit2 = all(v > 0 for v in ics.values())
            crit3 = all(e["inc"] > 0 for e in e_h.values())
            print(f"\n  VERDICT (pre-registered, ETH oi_g4): "
                  f"[1] IC>0 & |z|>=2: {crit1} | [2] both halves IC>0: {crit2} "
                  f"({ics['H1']:+.3f}/{ics['H2']:+.3f}) | [3] econ inc>0 both halves: {crit3} "
                  f"-> {'PASS' if crit1 and crit2 and crit3 else 'REJECTED'}")


if __name__ == "__main__":
    main()
