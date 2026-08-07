"""Stage 0 of the BTC deep-feature-encoder plan (see
docs/btc_new_architecture_session_summary_20260804.md and the approved plan for context).

Builds one unified raw panel = causalfix_final's 99 model columns (114 raw incl.
excluded OHLCV/mtf1h_ts_* ) + the 13 Regime3 wide24 input columns not already present
in causalfix_final (STATE7_COLS + STATE12_COLS's RAW5_COLS + breakout_strength) +
Deribit DVOL's 6 derived columns + CoinMetrics on-chain's 6 derived columns.

The 13 novel regime3-input columns are computed via the *same* functions the live
Regime3 wide24 sidecar itself uses (_with_raw_state12, which chains _with_raw_state7),
reused unchanged from scripts/retrain_clean_regime_hmm_raw_state12_20260517.py, applied
to the same funding_clean_splits_20260528 source tape the live sidecar was built from
-- not reimplemented. DVOL and on-chain parquets are already causal
(available_at delay + merge_asof backward) and are joined as-is.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import STATE12_COLS, _with_raw_state12  # noqa: E402

CAUSALFIX_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
STATE_SOURCE_PATHS = [
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
]
DVOL_PATH = ROOT / "data/splits/year_oos/btc_dvol_features_20260804.parquet"
ONCHAIN_PATH = ROOT / "data/splits/year_oos/btc_onchain_features_20260804.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"

NOVEL_STATE_COLS = list(STATE12_COLS) + ["breakout_strength"]  # 12 + 1 = 13


def build_state_columns() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in STATE_SOURCE_PATHS]
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    enriched = _with_raw_state12(raw)
    enriched["breakout_strength"] = pd.to_numeric(raw["breakout_strength"], errors="coerce").fillna(0.0)
    return enriched[["timestamp"] + NOVEL_STATE_COLS]


def main() -> None:
    base = pd.read_parquet(CAUSALFIX_PATH).sort_values("timestamp").reset_index(drop=True)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    n0 = len(base)

    state_cols = build_state_columns()
    merged = base.merge(state_cols, on="timestamp", how="left", validate="one_to_one")
    missing_state = merged[NOVEL_STATE_COLS].isna().any(axis=1).sum()
    print(f"state cols merged: {missing_state} / {len(merged)} rows have any NaN state col")

    dvol = pd.read_parquet(DVOL_PATH)
    dvol["timestamp"] = pd.to_datetime(dvol["timestamp"], utc=True)
    dvol_cols = [c for c in dvol.columns if c != "timestamp"]
    merged = merged.merge(dvol, on="timestamp", how="left", validate="one_to_one")
    print(f"dvol cols merged: {merged[dvol_cols].isna().any(axis=1).sum()} / {len(merged)} rows have any NaN dvol col")

    onchain = pd.read_parquet(ONCHAIN_PATH)
    onchain["timestamp"] = pd.to_datetime(onchain["timestamp"], utc=True)
    onchain_cols = [c for c in onchain.columns if c != "timestamp"]
    merged = merged.merge(onchain, on="timestamp", how="left", validate="one_to_one")
    print(f"onchain cols merged: {merged[onchain_cols].isna().any(axis=1).sum()} / {len(merged)} rows have any NaN onchain col")

    assert len(merged) == n0, f"row count changed during merge: {n0} -> {len(merged)}"

    merged.to_parquet(OUT_PATH, index=False)
    new_cols = NOVEL_STATE_COLS + dvol_cols + onchain_cols
    print(f"\nwrote {OUT_PATH}")
    print(f"base cols: {base.shape[1]}, new cols added: {len(new_cols)}, total: {merged.shape[1]}")
    print(f"new columns: {new_cols}")


if __name__ == "__main__":
    main()
