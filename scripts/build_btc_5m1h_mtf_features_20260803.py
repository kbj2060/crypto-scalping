"""
Merge BTC 1h trend-scan/RSI/volatility overlay onto the existing 5m base feature
frame (data/splits/year_oos/btc_features_2024_2026.csv), for the new BTC
multi-timeframe (5m+1h) architecture feature candidate pool.

Overlay source: tmp/causal_regen_20260516/sigma9_1h_btc_full_20260801/sigma9_btc_1h_full_{year}.parquet
(the post-lookahead-fix ETH-identical resample_1h/compute_features pipeline).

Lookahead guard: an hourly row at `timestamp` (bar OPEN time, since resample_1h
labels bars by their open) is only known once that hour has fully closed, i.e.
at `timestamp + 1h`. We merge_asof each 5m row against `available_at = timestamp
+ 1h` (direction='backward'), so a 5m bar can only see 1h features from an hour
that has already closed strictly before it -- mirrors the +1h shift fix applied
to run_window() regime merges (see project memory: run-window same-bar lookahead).
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_5M_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
OVERLAY_1H_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_full_20260801"
METRICS4_DIR = ROOT / "data/splits/year_oos_metrics4_btc_20260802"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"

OVERLAY_COLS = [
    "ts_action", "ts_t_value", "ts_opt_L",   # 1h trend-scan
    "rsi_14",                                 # 1h RSI
    "rvol_6", "rvol_12", "rvol_24", "rvol_48", "atr_pct", "bb_width", "bb_pos",  # 1h volatility
]

# 2026-08-02 metrics4 experiment columns (previously tested standalone with h48qual
# and found to hurt that classifier's calibration -- see project memory
# project-btc-metrics4-features-standalone-weak-20260802 -- re-added here at user's
# explicit request for this new architecture; re-evaluate empirically, don't assume
# the prior negative result transfers).
METRICS4_COLS = [
    "sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio",
    "taker_vol_ratio_z", "count_toptrader_ratio_z",
    "toptrader_count_size_divergence", "sig_whale", "sig_oi_divergence",
]


def load_metrics4() -> pd.DataFrame:
    parts = [pd.read_csv(p, usecols=["timestamp"] + METRICS4_COLS, low_memory=False)
              for p in sorted(METRICS4_DIR.glob("btc_features_*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_1h_overlay() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(OVERLAY_1H_DIR.glob("sigma9_btc_1h_full_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    missing = [c for c in OVERLAY_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"1h overlay missing expected columns: {missing}")
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at"] + OVERLAY_COLS].copy()
    df = df.rename(columns={c: f"mtf1h_{c}" for c in OVERLAY_COLS})
    return df.sort_values("available_at").reset_index(drop=True)


def main():
    base = pd.read_csv(BASE_5M_PATH, low_memory=False)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    base = base.sort_values("timestamp").reset_index(drop=True)

    overlay = load_1h_overlay()

    merged = pd.merge_asof(
        base,
        overlay,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )

    metrics4 = load_metrics4()
    before = len(merged)
    merged = merged.merge(metrics4, on="timestamp", how="left")
    if len(merged) != before:
        raise SystemExit(f"metrics4 merge changed row count: {before} -> {len(merged)}")
    n_metrics4_missing = merged[METRICS4_COLS].isna().any(axis=1).sum()
    print(f"metrics4 merge: {before - n_metrics4_missing}/{before} rows matched")

    # Lookahead check: for every merged row, the 1h data used must have become
    # available strictly before (or at) the 5m bar's own timestamp.
    matched = merged["available_at"].notna()
    violations = (merged.loc[matched, "available_at"] > merged.loc[matched, "timestamp"]).sum()
    if violations:
        raise SystemExit(f"LOOKAHEAD VIOLATION: {violations} rows have 1h data from the future")

    n_unmatched = (~matched).sum()
    print(f"base rows: {len(base)}, matched 1h overlay: {matched.sum()}, unmatched (pre-warmup): {n_unmatched}")
    print(f"lookahead check passed: 0 violations across {matched.sum()} matched rows")

    sample = merged.loc[matched, ["timestamp", "available_at"]].sample(min(5, matched.sum()), random_state=0)
    print("sample matches (5m timestamp vs 1h available_at used):")
    print(sample.to_string(index=False))

    merged = merged.drop(columns=["available_at"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={merged.shape}")


if __name__ == "__main__":
    main()
