#!/usr/bin/env python3
"""v2 candidates for short_term_return_z meta-labeling: widen CLUSTER_GAP_MERGE from v1's 3 bars
(15min) to 6 and 12 bars -- everything else (HORIZON=12, ATR_HIT_MULT=1.75, touch-only hit, 23
Tier0 features, TabPFN, Fresh-Forward split) is identical to v1
(research_eth_short_term_return_z_metalabel_tabpfn_20260829.py).

Motivation (2026-08-30, user-directed): mirrors the taker_delta_z_climax v5 experiment
(CLUSTER_GAP_MERGE 3->12), which found AUC and trailing-stop economics improved together and the
result then SURVIVED taker's single holdout touch. A VAL/OOS-only trailing-stop economics
pre-check on short_term_return_z (scratchpad research_str_z_cluster_gap_merge_val_oos_only_
20260830.py) found a MUCH weaker effect than taker's (roughly +10% at best vs taker's ~+93%),
close to noise given the smaller per-window sample sizes (~400-500) -- unlike taker, this is NOT
expected to be a clear win, and both gap=6 (this pre-check's best VAL+OOS point) and gap=12
(matching taker's chosen value, also close) are run here to see whether the FULL AUC pipeline
(not just the plain trailing-stop simulation) shows anything more decisive.

HOLDOUT (2026-04-01~) is intentionally NOT evaluated in this script. short_term_return_z's
existing v1 (gap=3) design already used its single holdout touch this session and SURVIVED
(+3.70bp, eth_short_term_return_z_trailing_stop_costgate_confirmed_20260830.md) -- whether a v2
gap-widened variant should get a fresh look at that SAME window is a separate decision to raise
with the user, not something to decide unilaterally inside a labeling script.

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
    run_tabpfn_panel,
)
# random_bar_baseline is NOT imported from the taker script -- that one hardcodes taker's own
# module-level HORIZON=24 internally (not a parameter). short_term_return_z's own v1 script
# defines its OWN copy against HORIZON=12 for exactly this reason -- reuse THAT one, not taker's.
from research_eth_short_term_return_z_metalabel_tabpfn_20260829 import random_bar_baseline  # noqa: E402

REPORT_DIR = ROOT / "tmp/eth_short_term_return_z_metalabel_v2_gap_sweep_20260830"

START = pd.Timestamp("2024-01-01")
HORIZON = 12          # unchanged from v1
ATR_HIT_MULT = 1.75   # unchanged from v1
GAPS_TO_TRY = [6, 12]

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")  # boundary only, HOLDOUT itself not evaluated here


def log(msg: str) -> None:
    print(f"[str_z_v2_gap] {msg}", flush=True)


def cluster_dedup_gap(idx: np.ndarray, rz_at_idx: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    order = np.argsort(idx)
    idx_sorted = idx[order]
    rz_sorted = rz_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "rz": rz_sorted})
    keep = df.loc[df.groupby("cluster")["rz"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["rz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(indicator_frame: pd.DataFrame, sig: pd.DataFrame, gap: int) -> pd.DataFrame:
    high, low, close = sig["high"].to_numpy(), sig["low"].to_numpy(), sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    ret3_z_all = indicator_frame["ret3_z"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_short_term_return_z"), ("top", "top_short_term_return_z")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (sig["timestamp"].to_numpy()[idx] >= np.datetime64(START))]
        idx_before = len(idx)
        idx = cluster_dedup_gap(idx, ret3_z_all[idx], most_negative=(side == "bottom"), gap=gap)
        log(f"  {side}: {idx_before} raw fires -> {len(idx)} after cluster-anchor dedup (gap={gap})")
        entry = close[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        hit = pred_dir_ret >= ATR_HIT_MULT * atr_pct[idx]
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
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_for_gap(gap: int, klines, indicator_frame, sig, baseline) -> dict:
    log(f"\n{'='*70}\nGAP={gap} bars ({gap*5}min)\n{'='*70}")
    fires = build_fires_and_features(indicator_frame, sig, gap)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna")

    fire_hit_rate = float(fires["hit"].mean())
    log(f"fired-signal hit rate: {fire_hit_rate:.4f} vs all-bar baseline: "
        f"{baseline['all_bar_continuous_sign_hit_rate']:.4f} (lift {fire_hit_rate / baseline['all_bar_continuous_sign_hit_rate']:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, VAL n={len(val)}, OOS n={len(oos)} (HOLDOUT not evaluated)")

    out_dir = ROOT / f"data/labels/eth_5m_short_term_return_z_metalabel_v2_gap{gap}_20260830"
    out_dir.mkdir(parents=True, exist_ok=True)
    fires.to_csv(out_dir / f"eth_5m_short_term_return_z_metalabel_v2_gap{gap}_features.csv", index=False)

    log(f"=== GAP={gap} VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, f"gap{gap}-VAL")
    log(f"GAP={gap} VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log(f"=== GAP={gap} OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, f"gap{gap}-OOS")
    log(f"GAP={gap} OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"GAP={gap} top permutation features:")
    for row in perm_importance["importances"][:6]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f}")

    return {
        "gap_bars": gap, "n_fires_total": int(len(fires)), "fired_signal_hit_rate": fire_hit_rate,
        "lift_vs_all_bar_baseline": fire_hit_rate / baseline["all_bar_continuous_sign_hit_rate"],
        "val": val_result, "oos": oos_result, "permutation_importance_val": perm_importance,
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading klines...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    baseline = random_bar_baseline(indicator_frame, klines)

    report = {
        "signal": "short_term_return_z", "adopted_version": "v1 (gap=3)",
        "v1_reference": {"val_auc": 0.674, "oos_auc": 0.649, "holdout_auc": 0.643,
                          "trailing_stop_val_oos_avg_trade_bp": 12.33, "trailing_stop_holdout_avg_trade_bp": 3.70},
        "candidates": {},
    }
    for gap in GAPS_TO_TRY:
        report["candidates"][f"gap{gap}"] = run_for_gap(gap, klines, indicator_frame, sig, baseline)

    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
