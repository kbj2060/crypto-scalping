"""ETH -- microstructure_1m scalp-horizon (1/3/5/10/15-minute) direction screen.

Pre-registered design (locked before touching any joined feature-return data):
docs/experiments/eth_candidate_microstructure_scalp_horizon_screen_20260824.md

Extends scripts/research_eth_microstructure_panel_1h4h_direction_screen_20260823.py's exact
methodology (9 fixed features, Spearman IC + circular-shift permutation null, TRAIN/VAL only,
OOS reserved) to native 1-minute horizons instead of 5-minute-bar h=12/48 (1h/4h).

Structurally different from the invalidated eth_scalp_1m_20260716 line: forward returns are
computed from ETHUSDT's own 1m klines (self-contained, no BTC-derived features at all), so the
BTC-5m-bar semantic-availability leak that invalidated that line cannot occur here by construction.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/kbj20/crypto-scalping")
KLINES_1M_MAIN = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
KLINES_1M_FILL = ROOT / "tmp/eth_scalp_horizon_screen_20260824/ethusdt_1m_fill_20260731_20260817.csv"
MICRO_DB_PATH = ROOT / "data/live/microstructure.duckdb"
N_PERM = 2000
MIN_SHIFT = 320  # minutes -- same real-time decorrelation buffer as the parent 1h/4h screen (64 x 5m bars)
Z_WINDOW = 1440  # 24h of 1m bars (parent screen's 288 x 5m bars = same 24h)
COST_BP = 10.0
CONTAM_MAX = 0.5

RAW_FEATURES = ["obi", "taker_buy_ratio", "spoofing_score", "nif_whale", "nif_retail",
                "shadow_toxicity_score", "shadow_queue_collapse"]
Z_FEATURES = ["eai", "oi_delta_pct"]
ALL_FEATURES = RAW_FEATURES + Z_FEATURES
HORIZONS = [1, 3, 5, 10, 15]

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
    micro = con.execute(
        f"SELECT ts, {', '.join(ALL_FEATURES)} FROM microstructure_1m ORDER BY ts"
    ).fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)

    frame = pd.merge_asof(
        klines.sort_values("bar_close_time"), micro.sort_values("ts"),
        left_on="bar_close_time", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("1min"),
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


def circular_shift_z(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = 20260824) -> tuple[float, float]:
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
    """Same threshold-crossing rule as the parent screen: trailing-1440 z of the canonical
    feature; when |z|>=1, position = ic_sign * sign(z); hold h bars non-overlapping; COST_BP
    round-trip. Returns (net-PnL increment vs bench, mean gross bp/trade before cost, n_trades)."""
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
    res.to_csv(ROOT / "tmp/eth_scalp_horizon_screen_20260824/full_ic_results.csv", index=False)
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

    print(f"\nstatistical survivors: {len(survivors)}/{len(ALL_FEATURES) * len(HORIZONS)} cells")
    econ_rows = []
    for feat, h, sign in survivors:
        col = canonical_col(feat)
        for split, (start, end) in SPLITS.items():
            sub = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
            inc, gross_bp, n_tr = economic_gate(sub, col, h, sign)
            econ_rows.append({"feature": feat, "h": h, "split": split, "inc": inc, "gross_bp_per_trade": gross_bp, "n_trades": n_tr})
    econ_df = pd.DataFrame(econ_rows)
    if not econ_df.empty:
        econ_df.to_csv(ROOT / "tmp/eth_scalp_horizon_screen_20260824/economic_gate_results.csv", index=False)
        for feat, h, sign in survivors:
            tr = econ_df[(econ_df.feature == feat) & (econ_df.h == h) & (econ_df.split == "TRAIN")].iloc[0]
            va = econ_df[(econ_df.feature == feat) & (econ_df.h == h) & (econ_df.split == "VAL")].iloc[0]
            both_pos = tr["inc"] > 0 and va["inc"] > 0
            print(f"  ECONOMIC {feat} h={h}: TRAIN inc={tr['inc']*100:+.3f}% (gross {tr['gross_bp_per_trade']:+.2f}bp/trade, n={tr['n_trades']:.0f}) "
                  f"VAL inc={va['inc']*100:+.3f}% (gross {va['gross_bp_per_trade']:+.2f}bp/trade, n={va['n_trades']:.0f}) -> {'PASS' if both_pos else 'FAIL'}")

    if not survivors:
        print(f"\nOVERALL: REJECTED -- no cell passed the pre-registered statistical criteria (sign agreement + |z|>=2 both splits).")


if __name__ == "__main__":
    main()
