"""Re-merge the causality-fixed 1h overlay (ts_action/ts_t_value/ts_opt_L
corrected, other 1h cols regenerated identically) onto the regime3-enriched
5m base frame, replacing the old lookahead-contaminated mtf1h_ columns."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_2024_2026.parquet"
CAUSAL_1H_PATH = ROOT / "data/splits/year_oos/btc_1h_trendscan_causal_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_CAUSALFIX_2024_2026.parquet"

OLD_MTF1H_COLS = ["mtf1h_ts_action", "mtf1h_ts_t_value", "mtf1h_ts_opt_L", "mtf1h_rsi_14",
                   "mtf1h_rvol_6", "mtf1h_rvol_12", "mtf1h_rvol_24", "mtf1h_rvol_48",
                   "mtf1h_atr_pct", "mtf1h_bb_width", "mtf1h_bb_pos"]
NEW_COLS = ["ts_action", "ts_t_value", "ts_opt_L", "rsi_14", "rvol_6", "rvol_12", "rvol_24",
            "rvol_48", "atr_pct", "bb_width", "bb_pos"]


def main():
    base = pd.read_parquet(BASE_PATH).sort_values("timestamp").reset_index(drop=True)
    base = base.drop(columns=OLD_MTF1H_COLS)

    overlay = pd.read_parquet(CAUSAL_1H_PATH, columns=["timestamp"] + NEW_COLS)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    overlay["available_at"] = overlay["timestamp"] + pd.Timedelta(hours=1)
    overlay = overlay.rename(columns={c: f"mtf1h_{c}" for c in NEW_COLS}).drop(columns=["timestamp"])
    overlay = overlay.sort_values("available_at").reset_index(drop=True)

    merged = pd.merge_asof(base, overlay, left_on="timestamp", right_on="available_at", direction="backward")
    matched = merged["available_at"].notna()
    violations = (merged.loc[matched, "available_at"] > merged.loc[matched, "timestamp"]).sum()
    if violations:
        raise SystemExit(f"LOOKAHEAD VIOLATION: {violations}")
    print(f"matched {matched.sum()}/{len(merged)}, 0 availability-delay violations")
    merged = merged.drop(columns=["available_at"])
    merged.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={merged.shape}")


if __name__ == "__main__":
    main()
