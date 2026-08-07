"""One-off verification (2026-07-06): re-run today's FeatureEngineer().process() on the raw
input columns already stored in training_features_2026_rebuilt.csv, and diff the recomputed
96 non-regime3 base_cols (Omega4.6.1's h48qual+zig075 parent feature contract) against the
values already stored in that CSV. This tests whether the feature-computation CODE currently
in the repo reproduces the CSV's own recorded values on the same raw inputs -- i.e. whether
there has been code-level drift since the CSV/backtests were built, independent of any
live-data-availability concerns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer

CSV_PATH = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
H_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt"
Z_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt"


def main() -> None:
    bh = torch.load(H_BUNDLE, map_location="cpu", weights_only=False)
    bz = torch.load(Z_BUNDLE, map_location="cpu", weights_only=False)
    base_cols = sorted(set(bh["base_cols"]) | set(bz["base_cols"]))
    check_cols = [c for c in base_cols if not c.startswith("regime3_current_sensitive_wide24_")]

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Use a chunk near the end (2026-06-01..06-30) with enough leading warmup rows for
    # rolling-window features (elite.py windows go up to ~288 bars = 1 day at 5m).
    warmup_bars = 3000
    window_mask = (df["timestamp"] >= pd.Timestamp("2026-06-01"))
    first_idx = int(np.argmax(window_mask.to_numpy()))
    start_idx = max(0, first_idx - warmup_bars)
    chunk = df.iloc[start_idx:].reset_index(drop=True)
    print(f"chunk rows={len(chunk)} range={chunk['timestamp'].iloc[0]}..{chunk['timestamp'].iloc[-1]}")

    eth_raw_cols = [
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote",
        "sum_open_interest_value", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "last_funding_rate",
    ]
    missing_raw = [c for c in eth_raw_cols if c not in chunk.columns]
    if missing_raw:
        raise RuntimeError(f"missing raw input columns in CSV: {missing_raw}")
    eth_df = chunk[eth_raw_cols].copy()
    btc_df = chunk[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy()

    fe = FeatureEngineer()
    recomputed = fe.process(eth_df, btc_df)
    recomputed["timestamp"] = pd.to_datetime(recomputed["timestamp"])
    recomputed = recomputed.sort_values("timestamp").reset_index(drop=True)

    # Compare only the tail (post warmup) against the CSV's own stored values for the same
    # timestamps, restricted to June 2026.
    compare_mask = recomputed["timestamp"] >= pd.Timestamp("2026-06-01")
    recomputed_cmp = recomputed[compare_mask].reset_index(drop=True)
    csv_cmp = df[(df["timestamp"] >= pd.Timestamp("2026-06-01")) & (df["timestamp"] <= chunk["timestamp"].iloc[-1])].reset_index(drop=True)
    n = min(len(recomputed_cmp), len(csv_cmp))
    recomputed_cmp = recomputed_cmp.iloc[:n]
    csv_cmp = csv_cmp.iloc[:n]
    assert (recomputed_cmp["timestamp"].to_numpy() == csv_cmp["timestamp"].to_numpy()).all(), "timestamp alignment mismatch"
    print(f"comparing n={n} rows over {compare_mask.sum()} candidate rows")

    missing_in_recomputed = [c for c in check_cols if c not in recomputed_cmp.columns]
    print("base_cols missing from recomputed output:", missing_in_recomputed)

    results = []
    for c in check_cols:
        if c not in recomputed_cmp.columns:
            continue
        a = pd.to_numeric(recomputed_cmp[c], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(csv_cmp[c], errors="coerce").to_numpy(dtype=np.float64)
        both_finite = np.isfinite(a) & np.isfinite(b)
        if both_finite.sum() == 0:
            results.append((c, np.nan, np.nan, 0, "no_finite_overlap"))
            continue
        diff = np.abs(a[both_finite] - b[both_finite])
        denom = np.maximum(np.abs(b[both_finite]), 1e-8)
        rel = diff / denom
        results.append((c, float(diff.max()), float(rel.max()), int(both_finite.sum()), ""))

    results.sort(key=lambda r: -(r[2] if np.isfinite(r[2]) else -1))
    print("\nTop 20 by max relative diff:")
    for c, dmax, relmax, cnt, note in results[:20]:
        print(f"  {c:45s} max_abs_diff={dmax:.6g} max_rel_diff={relmax:.6g} n={cnt} {note}")

    bad = [r for r in results if np.isfinite(r[2]) and r[2] > 1e-3]
    print(f"\ncolumns with max_rel_diff > 1e-3: {len(bad)} / {len(results)}")
    for c, dmax, relmax, cnt, note in bad:
        print(f"  DRIFT: {c:45s} max_abs_diff={dmax:.6g} max_rel_diff={relmax:.6g} n={cnt}")


if __name__ == "__main__":
    main()
