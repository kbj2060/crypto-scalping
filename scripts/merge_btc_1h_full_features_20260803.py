"""
Replace the narrow 1h overlay (11 trend-scan/RSI/vol columns) with the FULL
1h FeatureEngineer feature set (146 cols, built by
build_btc_1h_full_features_20260803.py), merged onto the regime3-enriched
5m execution frame with the same causal availability-delay convention
(available_at = 1h bar open + 1h) validated earlier this session.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_2024_2026.parquet"
FULL_1H_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_5m1hfull_regime3_2024_2026.parquet"

# columns already present identically-named in the 5m base -- keep only the 1h
# TIMESTAMP + everything else, prefixed, to avoid collisions
OLD_OVERLAY_COLS = [
    "mtf1h_ts_action", "mtf1h_ts_t_value", "mtf1h_ts_opt_L", "mtf1h_rsi_14",
    "mtf1h_rvol_6", "mtf1h_rvol_12", "mtf1h_rvol_24", "mtf1h_rvol_48",
    "mtf1h_atr_pct", "mtf1h_bb_width", "mtf1h_bb_pos",
]


def main():
    base = pd.read_parquet(BASE_PATH).sort_values("timestamp").reset_index(drop=True)
    base = base.drop(columns=OLD_OVERLAY_COLS)

    full1h = pd.read_csv(FULL_1H_PATH, low_memory=False)
    full1h["timestamp"] = pd.to_datetime(full1h["timestamp"])
    full1h["available_at"] = full1h["timestamp"] + pd.Timedelta(hours=1)
    drop_from_1h = {"timestamp"}
    rename = {c: f"mtf1hfull_{c}" for c in full1h.columns if c not in drop_from_1h and c != "available_at"}
    full1h = full1h.rename(columns=rename).drop(columns=["timestamp"])
    full1h = full1h.sort_values("available_at").reset_index(drop=True)

    merged = pd.merge_asof(base, full1h, left_on="timestamp", right_on="available_at", direction="backward")

    matched = merged["available_at"].notna()
    violations = (merged.loc[matched, "available_at"] > merged.loc[matched, "timestamp"]).sum()
    if violations:
        raise SystemExit(f"LOOKAHEAD VIOLATION: {violations} rows")
    print(f"matched {matched.sum()}/{len(merged)} rows, 0 lookahead violations")

    merged = merged.drop(columns=["available_at"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={merged.shape}")


if __name__ == "__main__":
    main()
