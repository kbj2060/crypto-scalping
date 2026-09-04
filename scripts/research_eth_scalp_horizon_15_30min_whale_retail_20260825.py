"""Extends research_eth_microstructure_scalp_horizon_screen_20260824.py (1/3/5/10/15-minute
native-1m direction screen) with a 30-minute horizon (not in the original [1,3,5,10,15] list) and
whale_position_score (not in the original RAW_FEATURES) -- user asked specifically "what about
15min and 30min" after the 1h/4h three-way comparison (nif_retail vs nif_whale vs
whale_position_score). Everything else (windows, permutation test, economic gate, thresholds) is
copied VERBATIM from the parent script; nif_whale/nif_retail are re-run alongside the new pieces
in the SAME pass so the reproduced 15-min numbers are directly comparable, not pulled from a
possibly-different-vintage CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KLINES_1M_MAIN = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
KLINES_1M_FILL = ROOT / "tmp/eth_scalp_horizon_screen_20260824/ethusdt_1m_fill_20260731_20260817.csv"
MICRO_DB_PATH = ROOT / "data/live/microstructure.duckdb"
N_PERM = 2000
MIN_SHIFT = 320  # minutes -- same real-time decorrelation buffer as the parent screens
Z_WINDOW = 1440  # 24h of 1m bars
COST_BP = 10.0
CONTAM_MAX = 0.5

RAW_FEATURES = ["nif_whale", "nif_retail", "whale_position_score"]
HORIZONS = [15, 30]  # minutes -- 15 reproduces the parent scalp screen's cell, 30 is the new gap

SPLITS = {
    "TRAIN": ("2026-05-03", "2026-07-31"),
    "VAL": ("2026-08-01", "2026-08-16"),
}


def build_frame() -> pd.DataFrame:
    main = pd.read_csv(KLINES_1M_MAIN, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    fill = pd.read_csv(KLINES_1M_FILL, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    klines = pd.concat([main, fill], ignore_index=True)
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-25") & (klines["timestamp"] <= "2026-08-18")]
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=1)

    con = duckdb.connect(str(MICRO_DB_PATH), read_only=True)
    micro = con.execute(f"SELECT ts, {', '.join(RAW_FEATURES)} FROM microstructure_1m ORDER BY ts").fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), micro.sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("1min"),
    )
    for h in HORIZONS:
        frame[f"fwd_{h}"] = frame["close"].shift(-h) / frame["close"] - 1.0
    return frame


def circular_shift_z(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = 20260825) -> tuple[float, float]:
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 500 or n <= 2 * MIN_SHIFT:
        return float("nan"), float("nan")
    obs = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=n_perm)
    null = np.empty(n_perm)
    for i, s in enumerate(shifts):
        null[i] = spearmanr(np.roll(x, s), y).statistic
    return obs, (obs - null.mean()) / null.std(ddof=1)


def economic_gate(sub: pd.DataFrame, col: str, h: int, ic_sign: float) -> tuple[float, float, int]:
    z = (sub[col] - sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()) / \
        sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).std().replace(0.0, np.nan)
    fwd = sub[f"fwd_{h}"]
    pnl, gross_sum, i, idx = 0.0, 0.0, 0, sub.index.to_list()
    n_trades = 0
    while i < len(idx):
        zi, fi = z.get(idx[i]), fwd.get(idx[i])
        if pd.notna(zi) and abs(zi) >= 1.0 and pd.notna(fi):
            trade_ret = ic_sign * np.sign(zi) * fi
            pnl += trade_ret - COST_BP / 1e4
            gross_sum += trade_ret
            n_trades += 1
            i += h
        else:
            i += 1
    always_long = fwd.iloc[::h].dropna().sum()
    bench = max(always_long, -always_long)
    gross_bp_per_trade = (gross_sum / n_trades) * 1e4 if n_trades else float("nan")
    return pnl - bench, gross_bp_per_trade, n_trades


def main() -> None:
    frame = build_frame()
    print(f"joined frame: {len(frame)} bars, coverage: "
          + ", ".join(f"{f}={frame[f].notna().mean()*100:.1f}%" for f in RAW_FEATURES) + "\n")

    rows = []
    for split, (start, end) in SPLITS.items():
        sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        for feat in RAW_FEATURES:
            valid_n = sub[feat].notna().sum()
            contam = spearmanr(*sub[[feat, "close"]].dropna().to_numpy().T).statistic if valid_n > 500 else float("nan")
            for h in HORIZONS:
                ic, z = circular_shift_z(sub[feat].to_numpy(dtype=float), sub[f"fwd_{h}"].to_numpy(dtype=float))
                rows.append({"split": split, "feature": feat, "h": h, "ic": ic, "z": z, "contam": contam, "n": valid_n})

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["feature", "h"], columns="split", values=["ic", "z"], aggfunc="first")
    contam_map = res[res["split"] == "TRAIN"].set_index("feature")["contam"].to_dict()
    n_map = res[res["split"] == "TRAIN"].set_index("feature")["n"].to_dict()

    print(f"{'feature':<24}{'h(min)':>7}{'n_train':>9}{'contam':>8}{'IC_tr':>9}{'z_tr':>7}{'IC_val':>9}{'z_val':>7}  verdict")
    survivors = []
    for (feat, h), r in piv.iterrows():
        contam = contam_map.get(feat, float("nan"))
        ic_tr, z_tr = r[("ic", "TRAIN")], r[("z", "TRAIN")]
        ic_val, z_val = r[("ic", "VAL")], r[("z", "VAL")]
        if pd.notna(contam) and abs(contam) >= CONTAM_MAX:
            verdict = "EXCLUDED(contam)"
        elif pd.isna(ic_tr) or pd.isna(ic_val):
            verdict = "insufficient"
        elif np.sign(ic_tr) != np.sign(ic_val):
            verdict = "fail:sign"
        elif abs(z_tr) < 2 or abs(z_val) < 2:
            verdict = "fail:z"
        else:
            verdict = "PASS_STATS"
            survivors.append((feat, h, np.sign(ic_tr)))
        print(f"{feat:<24}{h:>7}{n_map.get(feat,0):>9d}{contam:>8.3f}{ic_tr:>9.4f}{z_tr:>7.2f}{ic_val:>9.4f}{z_val:>7.2f}  {verdict}")

    print(f"\nstatistical survivors: {len(survivors)}/{len(RAW_FEATURES) * len(HORIZONS)} cells")
    for feat, h, sign in survivors:
        for split, (start, end) in SPLITS.items():
            sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
            inc, gross_bp, n_tr = economic_gate(sub, feat, h, sign)
            tag = "TRAIN" if split == "TRAIN" else "VAL  "
            print(f"  ECONOMIC {feat} h={h}m [{tag}]: inc={inc*100:+.3f}% gross={gross_bp:+.2f}bp/trade n={n_tr}")


if __name__ == "__main__":
    main()
