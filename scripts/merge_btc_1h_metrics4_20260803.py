"""Merge 1h-native metrics4 (7 cols) onto the reverted 156-feature base
(5m + narrow 1h overlay + regime3), causal availability-delay merge."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_2024_2026.parquet"
M4_1H_PATH = ROOT / "data/splits/year_oos/btc_features_1h_metrics4_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_regime3_metrics4at1h_2024_2026.parquet"

COLS = ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio",
        "taker_vol_ratio_z", "count_toptrader_ratio_z", "toptrader_count_size_divergence",
        "sig_whale", "sig_oi_divergence"]


def main():
    base = pd.read_parquet(BASE_PATH).sort_values("timestamp").reset_index(drop=True)
    m4 = pd.read_csv(M4_1H_PATH, usecols=["timestamp"] + COLS, low_memory=False)
    m4["timestamp"] = pd.to_datetime(m4["timestamp"])
    m4["available_at"] = m4["timestamp"] + pd.Timedelta(hours=1)
    m4 = m4.rename(columns={c: f"mtf1h_m4_{c}" for c in COLS}).drop(columns=["timestamp"]).sort_values("available_at")

    merged = pd.merge_asof(base, m4, left_on="timestamp", right_on="available_at", direction="backward")
    matched = merged["available_at"].notna()
    violations = (merged.loc[matched, "available_at"] > merged.loc[matched, "timestamp"]).sum()
    if violations:
        raise SystemExit(f"LOOKAHEAD VIOLATION: {violations}")
    print(f"matched {matched.sum()}/{len(merged)}, 0 violations")
    merged = merged.drop(columns=["available_at"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={merged.shape}")


if __name__ == "__main__":
    main()
