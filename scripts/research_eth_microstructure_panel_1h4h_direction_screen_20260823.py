"""ETH -- microstructure_1m panel 1h/4h direction screen (방식2 cheap-gate).

Pre-registered design (locked before touching any joined feature-return data):
docs/experiments/eth_candidate_microstructure_panel_1h4h_direction_screen_20260823.md

9 fixed features x 2 horizons (h=12/48 5m bars = 1h/4h) = 18 cells.
Spearman IC + circular-shift permutation null (N=2000, preserves autocorrelation of
overlapping forward-return windows). TRAIN/VAL only; OOS (2026-08-17+) reserved untouched.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KLINES_PATH = "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MICRO_DB_PATH = "data/live/microstructure.duckdb"
N_PERM = 2000
Z_WINDOW = 288  # 24h of 5m bars, for the two z-scored features and the economic-gate rule
COST_BP = 10.0
CONTAM_MAX = 0.5

RAW_FEATURES = ["obi", "taker_buy_ratio", "spoofing_score", "nif_whale", "nif_retail",
                "shadow_toxicity_score", "shadow_queue_collapse"]
Z_FEATURES = ["eai", "oi_delta_pct"]
ALL_FEATURES = RAW_FEATURES + Z_FEATURES
HORIZONS = [12, 48]

SPLITS = {
    "TRAIN": ("2026-05-03", "2026-07-31"),
    "VAL": ("2026-08-01", "2026-08-16"),
}


def build_frame() -> pd.DataFrame:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    klines = klines.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    klines = klines[(klines["timestamp"] >= "2026-04-25") & (klines["timestamp"] <= "2026-08-18")]
    klines["bar_close_time"] = klines["timestamp"] + pd.Timedelta(minutes=5)

    con = duckdb.connect(MICRO_DB_PATH, read_only=True)
    micro = con.execute(
        f"SELECT ts, {', '.join(ALL_FEATURES)} FROM microstructure_1m ORDER BY ts"
    ).fetchdf()
    con.close()
    # KST tz-aware -> UTC naive (the klines CSV is UTC-naive; verified against Binance API bars).
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), micro.sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("5min"),
    )
    for col in Z_FEATURES:
        mu = frame[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
        sd = frame[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).std()
        frame[f"{col}_z"] = (frame[col] - mu) / sd.replace(0.0, np.nan)
    for h in HORIZONS:
        frame[f"fwd_{h}"] = frame["close"].shift(-h) / frame["close"] - 1.0
    return frame


def canonical_col(feature: str) -> str:
    return f"{feature}_z" if feature in Z_FEATURES else feature


def circular_shift_z(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = 20260823) -> tuple[float, float]:
    """Observed Spearman IC and its z vs a circular-shift null (shifts >= 64 bars away from 0
    so shifted series can't trivially overlap the true alignment)."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 500:
        return float("nan"), float("nan")
    obs = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    shifts = rng.integers(64, n - 64, size=n_perm)
    null = np.empty(n_perm)
    for i, s in enumerate(shifts):
        null[i] = spearmanr(np.roll(x, s), y).statistic
    return obs, (obs - null.mean()) / null.std(ddof=1)


def economic_gate(sub: pd.DataFrame, col: str, h: int, ic_sign: float) -> float:
    """TRAIN-IC-sign-fixed threshold rule: trailing-288 z of the canonical feature; when
    |z|>=1, position = ic_sign * sign(z); hold h bars non-overlapping; 10bp round-trip cost.
    Returns net-PnL increment (frac) vs max(always_long, always_short) on the same rows."""
    z = (sub[col] - sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()) / \
        sub[col].rolling(Z_WINDOW, min_periods=Z_WINDOW).std().replace(0.0, np.nan)
    fwd = sub[f"fwd_{h}"]
    pnl, i, idx = 0.0, 0, sub.index.to_list()
    n_trades = 0
    while i < len(idx):
        zi, fi = z.get(idx[i]), fwd.get(idx[i])
        if pd.notna(zi) and abs(zi) >= 1.0 and pd.notna(fi):
            pnl += ic_sign * np.sign(zi) * fi - COST_BP / 1e4
            n_trades += 1
            i += h
        else:
            i += 1
    always_long = fwd.iloc[::h].dropna().sum()
    bench = max(always_long, -always_long)
    return pnl - bench


def main() -> None:
    frame = build_frame()
    print(f"joined frame: {len(frame)} bars, indicator coverage "
          f"{frame['obi'].notna().mean() * 100:.1f}%\n")

    rows = []
    for split, (start, end) in SPLITS.items():
        sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        for feat in ALL_FEATURES:
            col = canonical_col(feat)
            contam = spearmanr(*sub[[col, "close"]].dropna().to_numpy().T).statistic if sub[col].notna().sum() > 500 else float("nan")
            for h in HORIZONS:
                ic, z = circular_shift_z(sub[col].to_numpy(dtype=float), sub[f"fwd_{h}"].to_numpy(dtype=float))
                rows.append({"split": split, "feature": feat, "h": h, "ic": ic, "z": z, "contam": contam})

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["feature", "h"], columns="split", values=["ic", "z"], aggfunc="first")
    contam_map = res[res["split"] == "TRAIN"].set_index("feature")["contam"].to_dict()

    print(f"{'feature':<24}{'h':>4}{'contam':>8}{'IC_tr':>9}{'z_tr':>7}{'IC_val':>9}{'z_val':>7}  verdict")
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
        print(f"{feat:<24}{h:>4}{contam:>8.3f}{ic_tr:>9.4f}{z_tr:>7.2f}{ic_val:>9.4f}{z_val:>7.2f}  {verdict}")

    print(f"\nstatistical survivors: {len(survivors)}/18 cells")
    for feat, h, sign in survivors:
        col = canonical_col(feat)
        incs = {}
        for split, (start, end) in SPLITS.items():
            sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
            incs[split] = economic_gate(sub, col, h, sign)
        both_pos = all(v > 0 for v in incs.values())
        print(f"  ECONOMIC {feat} h={h}: TRAIN inc={incs['TRAIN'] * 100:+.2f}% VAL inc={incs['VAL'] * 100:+.2f}% -> {'PASS' if both_pos else 'FAIL'}")

    if not survivors:
        print("\nOVERALL: REJECTED -- no cell passed the pre-registered statistical criteria (2&3).")


if __name__ == "__main__":
    main()
