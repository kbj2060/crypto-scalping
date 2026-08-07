"""Phase 2 item 5 of the improvement roadmap: a re-runnable (not one-off) feature-contract drift
watch. `verify_live_feature_pipeline_parity_20260706.py` proved code-level parity ONCE, against a
fixed historical CSV. This script instead diffs against whatever is the CURRENT live bot state --
`data/live/decision_feature_frame_snapshot.pkl.gz`'s `frame` -- so it can be rerun at any point
going forward (e.g. from a cron/supervisor hook) to catch drift between the feature-computation
code and what the live bot is actually using RIGHT NOW, not just as of 2026-07-06.

The snapshot's `frame` already contains both the raw OHLCV/BTC input columns AND the live bot's own
already-computed feature values (everything the adapter saw immediately before Omega4.6.1's own
regime3 append). This script recomputes the same 96 non-regime3 base_cols (Omega4.6.1's h48qual+
zig075 parent feature contract) via `FeatureEngineer().process()` on those same raw inputs, and
diffs the recompute against the frame's own stored values. Read-only; does not touch
trading_bot.py or any live state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

SNAPSHOT_PATH = ROOT / "data/live/decision_feature_frame_snapshot.pkl.gz"
H_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt"
Z_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt"
OUT_PATH = ROOT / "data/ensemble/omega4_6_1_feature_drift_report.json"
DRIFT_TOLERANCE = 1e-3
COMPARE_TAIL_ROWS = 200  # rest of the snapshot is warmup for rolling-window features


def main() -> int:
    if not SNAPSHOT_PATH.exists():
        raise RuntimeError(f"no live snapshot found at {SNAPSHOT_PATH} -- bot may never have run "
                            f"Omega4.6.1 yet, or the snapshot path changed")

    bh = torch.load(H_BUNDLE, map_location="cpu", weights_only=False)
    bz = torch.load(Z_BUNDLE, map_location="cpu", weights_only=False)
    base_cols = sorted(set(bh["base_cols"]) | set(bz["base_cols"]))
    check_cols = [c for c in base_cols if not c.startswith("regime3_current_sensitive_wide24_")]

    import pickle
    import gzip
    with gzip.open(SNAPSHOT_PATH, "rb") as f:
        snapshot = pickle.load(f)
    frame = snapshot["frame"].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    snapshot_ts = snapshot.get("created_at", "unknown")
    print(f"snapshot created_at={snapshot_ts} rows={len(frame)} range={frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)

    if len(frame) < COMPARE_TAIL_ROWS + 50:
        raise RuntimeError(f"snapshot only has {len(frame)} rows -- not enough warmup to safely "
                            f"compare the tail {COMPARE_TAIL_ROWS} rows; rerun once the live bot "
                            f"has accumulated more history")

    eth_raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                    "trades", "taker_buy_base", "taker_buy_quote",
                    "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                    "count_long_short_ratio", "last_funding_rate"]
    missing_raw = [c for c in eth_raw_cols if c not in frame.columns]
    if missing_raw:
        raise RuntimeError(f"missing raw input columns in live snapshot: {missing_raw}")
    eth_df = frame[eth_raw_cols].copy()
    btc_df = frame[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy()

    fe = FeatureEngineer()
    recomputed = fe.process(eth_df, btc_df)
    recomputed["timestamp"] = pd.to_datetime(recomputed["timestamp"])
    recomputed = recomputed.sort_values("timestamp").reset_index(drop=True)

    tail_ts = frame["timestamp"].iloc[-COMPARE_TAIL_ROWS:]
    compare_mask = recomputed["timestamp"].isin(tail_ts.to_numpy())
    recomputed_cmp = recomputed[compare_mask].reset_index(drop=True)
    live_cmp = frame[frame["timestamp"].isin(tail_ts.to_numpy())].reset_index(drop=True)
    n = min(len(recomputed_cmp), len(live_cmp))
    recomputed_cmp, live_cmp = recomputed_cmp.iloc[:n], live_cmp.iloc[:n]
    if not (recomputed_cmp["timestamp"].to_numpy() == live_cmp["timestamp"].to_numpy()).all():
        raise RuntimeError("timestamp alignment mismatch between recomputed and live snapshot tail")
    print(f"comparing n={n} tail rows", flush=True)

    missing_in_recomputed = [c for c in check_cols if c not in recomputed_cmp.columns]
    if missing_in_recomputed:
        print(f"WARN base_cols missing from recomputed output: {missing_in_recomputed}", flush=True)

    results = []
    for c in check_cols:
        if c not in recomputed_cmp.columns or c not in live_cmp.columns:
            continue
        a = pd.to_numeric(recomputed_cmp[c], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(live_cmp[c], errors="coerce").to_numpy(dtype=np.float64)
        both_finite = np.isfinite(a) & np.isfinite(b)
        if both_finite.sum() == 0:
            results.append({"col": c, "max_abs_diff": None, "max_rel_diff": None, "n": 0, "note": "no_finite_overlap"})
            continue
        diff = np.abs(a[both_finite] - b[both_finite])
        denom = np.maximum(np.abs(b[both_finite]), 1e-8)
        rel = diff / denom
        results.append({"col": c, "max_abs_diff": float(diff.max()), "max_rel_diff": float(rel.max()), "n": int(both_finite.sum()), "note": ""})

    results.sort(key=lambda r: -(r["max_rel_diff"] if r["max_rel_diff"] is not None else -1))
    print("\nTop 10 by max relative diff:", flush=True)
    for r in results[:10]:
        print(f"  {r['col']:45s} max_abs_diff={r['max_abs_diff']} max_rel_diff={r['max_rel_diff']} n={r['n']} {r['note']}", flush=True)

    bad = [r for r in results if r["max_rel_diff"] is not None and r["max_rel_diff"] > DRIFT_TOLERANCE]
    status = "PASS" if not bad else "DRIFT_DETECTED"
    print(f"\n=== {status}: {len(bad)} / {len(results)} columns exceed rel-diff tolerance {DRIFT_TOLERANCE} ===", flush=True)
    for r in bad:
        print(f"  DRIFT: {r['col']:45s} max_abs_diff={r['max_abs_diff']:.6g} max_rel_diff={r['max_rel_diff']:.6g} n={r['n']}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "checked_at": pd.Timestamp.now().isoformat(), "snapshot_created_at": str(snapshot_ts),
        "snapshot_range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        "status": status, "n_columns_checked": len(results), "n_drifted": len(bad),
        "tolerance": DRIFT_TOLERANCE, "results": results,
    }, indent=2))
    print(f"\nWrote {OUT_PATH}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
