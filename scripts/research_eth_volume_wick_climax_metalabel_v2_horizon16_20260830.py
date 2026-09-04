#!/usr/bin/env python3
"""volume_wick_climax v2 -- ADOPTED (2026-08-30, user decision after reviewing the VAL/OOS
comparison below), replacing v1 (HORIZON=24). Originally written as a candidate (HORIZON=16, not
yet adopted) -- follow-up to user pushback that
v1's weak result (VAL/OOS/HOLDOUT 0.612/0.563/0.565) might reflect a real design flaw rather than
"this signal is inherently weak". Investigation (research_eth_volume_wick_climax_anchor_and_
horizon_recheck_20260830.py) found:
  - Cluster-anchor criterion (vol_z vs wick_ratio vs combined) barely matters (VAL/OOS all within
    ~0.01 of each other) -- NOT the explanation.
  - HORIZON, however, reveals a real methodological problem with how v1 picked it: the original
    3-point screening grid (6/12/24) selected HORIZON=24 by highest single-seed VAL AUC (0.617),
    but that config also has one of the LARGEST VAL-OOS gaps in the full 8-point grid tested here
    (0.617 VAL vs 0.563 OOS, gap=0.054) -- i.e. it was likely the point most overfit to this
    particular VAL window's noise, not the most genuinely-generalizing horizon. A denser grid
    (8/12/16/20/24/30/36/48) found HORIZON=16 has VAL=0.598 and OOS=0.598 -- essentially IDENTICAL,
    the smallest VAL-OOS gap (0.001) of any point tested, both comfortably above chance. This
    script re-runs H=16 with the full 4-seed VAL+OOS panel (HOLDOUT deliberately NOT touched here,
    pending a separate go-ahead -- same discipline as v1's own HORIZON/GAP screening) plus
    permutation importance and the wick-only baseline check, to see whether this more-balanced
    horizon is a genuinely better design or just a coincidentally close VAL/OOS pair.

Everything else unchanged from v1: GAP=3 (anchor-criterion choice didn't matter, kept as vol_z),
K recalibrated for H=16 (1.65, from the recheck script's calibration), touch-based MFE, no
persistence check, same 23 Tier0 features.
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

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS, build_indicator_frame, compute_permutation_importance, load_klines, run_tabpfn_panel,
)
from research_eth_volume_wick_climax_metalabel_tabpfn_20260830 import (
    VOL_Z_THRESH, WICK_RATIO_THRESH, random_bar_baseline_wick_only, cluster_dedup_by_vol_z,
)

OUT_DIR = ROOT / "data/labels/eth_5m_volume_wick_climax_metalabel_v2_horizon16_20260830"
REPORT_DIR = ROOT / "tmp/eth_volume_wick_climax_metalabel_tabpfn_20260830"

START = pd.Timestamp("2024-01-01")
HORIZON = 16
GAP = 3
K = 1.65  # from research_eth_volume_wick_climax_anchor_and_horizon_recheck_20260830.py's calibration

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")


def log(msg: str) -> None:
    print(f"[vwc_v2_h16] {msg}", flush=True)


def build_fires_and_features(klines: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    high = sig["high"].to_numpy(); low = sig["low"].to_numpy(); close = sig["close"].to_numpy()
    atr_pct = indicator_frame["atr_pct"].to_numpy()
    vol_z_all = indicator_frame["vol_z"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_volume_wick_climax"), ("top", "top_volume_wick_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (sig["timestamp"].to_numpy()[idx] >= np.datetime64(START))]
        idx = cluster_dedup_by_vol_z(idx, vol_z_all[idx], GAP)
        entry = close[idx]; a = atr_pct[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            pred_dir_ret = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            pred_dir_ret = (entry - fut_ext) / entry
        hit = pred_dir_ret >= K * a
        feat_rows = indicator_frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "pred_dir_ret": pred_dir_ret, "entry": entry, "atr_pct": a,
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

    log("loading klines + building indicator frame...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)

    log(f"building fires (HORIZON={HORIZON}, GAP={GAP}, K={K})...")
    fires = build_fires_and_features(klines, indicator_frame)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires (bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    baseline = random_bar_baseline_wick_only(indicator_frame, klines, HORIZON, K)
    fire_hit_rate = float(fires["hit"].mean())
    lift = fire_hit_rate / baseline["wick_only_no_volume_gate_hit_rate"]
    log(f"fired hit_rate={fire_hit_rate:.4f} vs wick-only baseline={baseline['wick_only_no_volume_gate_hit_rate']:.4f} (lift {lift:.3f}x)")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT n={len(holdout)} "
        f"(v2 adopted based on VAL+OOS evidence above; HOLDOUT evaluated once now as the standard "
        f"'final adopted' classification-AUC touch, matching v1/taker-v4/short_term_return_z-v1 "
        f"precedent -- distinct from the separately-gated trading-economics HOLDOUT touch)")

    fires.to_csv(OUT_DIR / "eth_5m_volume_wick_climax_metalabel_v2_horizon16_features.csv", index=False)

    log("=== VAL (4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS (4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== RESERVED HOLDOUT (4 seeds, single touch) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}")

    log("=== permutation importance (VAL) ===")
    perm = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    for row in perm["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f}")

    report = {
        "signal": "volume_wick_climax", "adopted_version": "v2_horizon16", "status": "exploratory_single_signal_below_promotion_bar",
        "horizon": HORIZON, "gap": GAP, "K": K,
        "n_fires_total": int(len(fires)),
        "random_bar_baseline_wick_only": baseline, "fired_signal_hit_rate": fire_hit_rate, "lift_vs_wick_only_baseline": lift,
        "val": val_result, "oos": oos_result, "reserved_holdout": holdout_result, "permutation_importance_val": perm,
        "comparison_to_v1_horizon24": {"v1_val": 0.6121, "v1_oos": 0.5633, "v1_holdout": 0.5652},
    }
    out_path = REPORT_DIR / "v2_horizon16_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
