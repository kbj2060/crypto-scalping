"""
Merge BTC regime3 CURRENT surface (bull/bear/chop_prob, confidence, entropy,
margin) into the 5m+1h+metrics4 feature frame, as Layer 2 context.

regime3 supersedes the deprecated regime4 (data contamination) per user
correction 2026-08-03. Source: data/ensemble/supervised/
btc_regime3_current_hmm_sensitive_wide24_20260708/ (CURRENT surface, already
bar-timestamp-aligned -- contemporaneous merge, same convention as the 5m
base features, NOT the +1h-availability-delay pattern used for the 1h
trend-scan overlay).
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
REGIME3_DIR = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_2024_2026.parquet"

REGIME3_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]


def main():
    base = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)

    parts = [pd.read_csv(p, usecols=["timestamp"] + REGIME3_COLS, low_memory=False)
             for p in sorted(REGIME3_DIR.glob("btc_features_*_regime3_current_sensitive_hmm_wide24.csv"))]
    reg = pd.concat(parts, ignore_index=True)
    reg["timestamp"] = pd.to_datetime(reg["timestamp"])
    reg = reg.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    before = len(base)
    merged = base.merge(reg, on="timestamp", how="left")
    if len(merged) != before:
        raise SystemExit(f"regime3 merge changed row count: {before} -> {len(merged)}")
    n_missing = merged[REGIME3_COLS].isna().any(axis=1).sum()
    print(f"regime3 merge: {before - n_missing}/{before} rows matched, {n_missing} missing")

    merged.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={merged.shape}")


if __name__ == "__main__":
    main()
