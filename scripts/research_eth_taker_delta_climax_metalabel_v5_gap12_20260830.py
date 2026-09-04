#!/usr/bin/env python3
"""v5 of taker_delta_z_climax meta-labeling: widen CLUSTER_GAP_MERGE from v4's 3 bars (15min) to
12 bars (60min) -- everything else (HORIZON=24, ATR_HIT_MULT=2.0, touch-only hit, 23 Tier0
features, TabPFN, Fresh-Forward split) is byte-for-byte identical to v4
(research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py), which this script imports from
directly wherever the logic doesn't depend on CLUSTER_GAP_MERGE.

Motivation (2026-08-30, user-directed): v4's trailing-stop cost-gate design (SL=2.0/ARM=1.5/
Trail=0.2xATR) passed VAL/OOS (scratchpad dual-verify + standard engine) but FAILED its single
HOLDOUT touch (avg_trade -0.98bp, see eth_taker_delta_climax_trailing_stop_costgate_breakthrough_
20260830.md) -- that card is spent. User asked whether raising the |delta_z| EXTREMITY threshold
could shrink candidate volume; VAL/OOS-only re-check showed this HURTS OOS economics monotonically
(4.75->-1.17bp as threshold rises 2.0->3.0). A DIFFERENT lever -- widening CLUSTER_GAP_MERGE (a
pure deduplication window, not a signal-strength filter) -- was tested instead and, unlike the
threshold change, IMPROVED VAL/OOS trailing-stop economics while also cutting candidate volume:

    GAP   candidates(VAL+OOS)  hit_rate  VAL avg_trade  OOS avg_trade  VAL+OOS avg_trade
    3(v4)        2,114           54.2%      +4.31bp        +4.75bp         +4.49bp
    12            1,540           58.9%      +8.73bp        +8.60bp         +8.68bp   <- this script
    24            1,213           59.3%      +7.81bp       +10.34bp         +8.90bp

12 was chosen over 24 for its tighter VAL/OOS agreement (more robust-looking); both were VAL/OOS-
only checks using the UNCHANGED trailing-stop config with a plain economics simulation
(scratchpad research_taker_cluster_gap_merge_val_oos_only_20260830.py), not yet re-run through
the full labeled-features + TabPFN AUC pipeline -- that is what THIS script does.

Why the anchor is expected to survive widening: cluster_dedup picks the single MOST-EXTREME-
delta_z bar per cluster (causal, price-blind) -- widening the merge window only means fewer,
larger clusters, each still anchored at its own true local peak; it does not discard genuine
independent events (verified against a real 2024-06-16 burst example, see chat/render_gap_merge_
explainer_20260830.py). This is fundamentally different from raising ATR_HIT_MULT or the raw
z-score cutoff, which discards real (if less extreme) fires outright.

HOLDOUT (2026-04-01~) is intentionally NOT evaluated in this script. taker_delta_z_climax's
single HOLDOUT touch was already spent by v4 -- whether/how a v5 candidate should ever be
holdout-checked (same window again vs. wait for new future data) is a methodology question to
raise with the user, not something to decide unilaterally inside a labeling script.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
    compute_permutation_importance,
    load_klines,
    random_bar_baseline,
    run_tabpfn_panel,
)

OUT_DIR = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_v5_gap12_20260830"
REPORT_DIR = ROOT / "tmp/eth_taker_delta_climax_metalabel_v5_gap12_20260830"

START = pd.Timestamp("2024-01-01")
HORIZON = 24            # unchanged from v4
ATR_HIT_MULT = 2.0      # unchanged from v4
CLUSTER_GAP_MERGE = 12  # v5 CHANGE: was 3 in v4 -- see module docstring

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")  # boundary only, HOLDOUT itself not evaluated here


def log(msg: str) -> None:
    print(f"[taker_v5_gap12] {msg}", flush=True)


def cluster_dedup_v5(idx: np.ndarray, delta_z_at_idx: np.ndarray, most_negative: bool) -> np.ndarray:
    """Identical logic to v4's cluster_dedup, parameterized on CLUSTER_GAP_MERGE=12 instead of
    the module-level constant=3 hardcoded in the v4 script (which is why this can't just be
    imported and reused as-is)."""
    order = np.argsort(idx)
    idx_sorted = idx[order]
    dz_sorted = delta_z_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > CLUSTER_GAP_MERGE:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "dz": dz_sorted})
    keep = df.loc[df.groupby("cluster")["dz"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["dz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(klines: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    assert len(sig) == len(indicator_frame), "row count mismatch between compute_signals and indicator_frame"
    assert (sig["timestamp"].to_numpy() == indicator_frame["timestamp"].to_numpy()).all(), "timestamp misalignment"

    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    n = len(sig)
    rows = []
    delta_z_all = indicator_frame["delta_z"].to_numpy()
    for side, col in [("bottom", "bottom_taker_delta_z_climax"), ("top", "top_taker_delta_z_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (sig["timestamp"].to_numpy()[idx] >= np.datetime64(START))]
        idx_before_dedup = len(idx)
        idx = cluster_dedup_v5(idx, delta_z_all[idx], most_negative=(side == "bottom"))
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after cluster-anchor dedup (gap={CLUSTER_GAP_MERGE})")
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        touched = pred_dir_ret >= ATR_HIT_MULT * atr_pct[idx]
        hit = touched
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "pred_dir_ret": pred_dir_ret,
            "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines...")
    klines = load_klines()
    log(f"{len(klines)} bars loaded")

    log("building Tier0-style indicator frame...")
    indicator_frame = build_indicator_frame(klines)

    log(f"building taker_delta_z_climax v5 fires+features (CLUSTER_GAP_MERGE={CLUSTER_GAP_MERGE})...")
    fires = build_fires_and_features(klines, indicator_frame)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    log("running random-bar continuous-sign baseline check...")
    baseline = random_bar_baseline(indicator_frame, klines)
    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal hit rate: {fire_hit_rate:.4f} vs all-bar baseline: "
        f"{baseline['all_bar_continuous_sign_hit_rate']:.4f} (lift {fire_hit_rate / baseline['all_bar_continuous_sign_hit_rate']:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)} (HOLDOUT deliberately not split out/evaluated)")

    fires.to_csv(OUT_DIR / "eth_5m_taker_delta_climax_metalabel_v5_gap12_features.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "taker_delta_z_climax", "adopted_version": "v5_gap12_candidate",
        "status": "exploratory_candidate_holdout_not_yet_decided",
        "change_from_v4": f"CLUSTER_GAP_MERGE 3->{CLUSTER_GAP_MERGE} bars, all else identical",
        "v4_reference": {"val_auc": 0.622, "oos_auc": 0.608, "holdout_auc": 0.650,
                          "trailing_stop_val_oos_avg_trade_bp": 4.49, "trailing_stop_holdout_avg_trade_bp": -0.98},
        "n_fires_total": int(len(fires)),
        "random_bar_baseline": baseline, "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_all_bar_baseline": fire_hit_rate / baseline["all_bar_continuous_sign_hit_rate"],
        "val": val_result, "oos": oos_result, "permutation_importance_val": perm_importance,
        "trailing_stop_val_oos_reference": {
            "config": "SL=2.0/ARM=1.5/Trail=0.2xATR (unchanged from v4)",
            "val_avg_trade_bp": 8.73, "oos_avg_trade_bp": 8.60, "val_oos_avg_trade_bp": 8.68,
            "note": "from prior VAL/OOS-only scratchpad economics check, not re-run here",
        },
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
