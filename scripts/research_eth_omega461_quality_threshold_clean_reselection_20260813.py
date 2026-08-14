#!/usr/bin/env python3
"""Diagnostic ONLY (not promotion/live-candidate): clean re-selection of h48qual/zig075
`quality_threshold` on data that the original OOS-primary selection never scored against.

## Background

`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` (the script that produced the
live-deployed h48qual quality_threshold=0.50 and zig075 quality_threshold=0.75) sorts candidate
thresholds by `key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True`
(its line ~1173) and its report.json stores the result under a SINGLE key, `ranking_by_oos_pnl`
(no `ranking_by_validation_pnl` sibling exists in the actual deployed h48qual/zig075 report.json
files -- independently confirmed by opening them directly). Its "oos" frame is
`EVAL_CSV = tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/
trade_candidates_2026_alpha6_current_tail111_exact.csv`, which covers EXACTLY 2026-01-01..02-28
(independently confirmed by reading its timestamp column) -- the same "precious, look-once" OOS
window this sub-project's other 2026-08-13 experiments have been treating as clean.

Opening the two deployed `quality_threshold_ranking.csv` files directly shows the VAL-optimal
threshold differs a lot from the deployed (OOS-cherry-picked) one:
  h48qual: deployed q=0.50 (VAL pnl=+4.58%, OOS pnl=+10.65%, OOS rank 1)
           vs VAL-optimal q=0.35 (VAL pnl=+22.47%, OOS pnl=+3.29%)
  zig075:  deployed q=0.75 (VAL pnl=+11.09%, OOS pnl=+14.77%, OOS rank 1)
           vs VAL-optimal q=0.55 (VAL pnl=+13.37%, OOS pnl=-4.37%)

## What this script does (no retraining, no GPU)

Recomputes `final_action` at every candidate quality_threshold directly from the ALREADY-COMPUTED,
threshold-independent columns (`dir_action`, `quality_for_action`) already sitting in the existing
frozen-bundle prediction file
`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/{h48qual,zig075}/oos_predictions_qXXX.csv`
(built 2026-07-06 by `retest_omega4_6_1_extended_oos_20260706.py` via a forward-only pass of the
FROZEN parent bundle over 2026-01-01..07-12 -- no lookahead, no retraining; verified here purely by
reapplying `dir_action != 0 and quality_for_action >= threshold -> final_action = dir_action`, the
exact same rule as `train_omega1_regime3_routed_expert_direction_quality_20260602._prediction_output`
and the live `omega4_6_1_live.py` gate). It then computes PnL/MDD/trades/wr with the SAME
fixed-BASE_TEMPLATE-sizing methodology as the original `quality_threshold_ranking.csv`
(`omega._to_fixed_decisions` + `omega._metrics`, unchanged, `BASE_TEMPLATE["max_hold"/"cooldown"]`
zeroed exactly as the original script does at its line 302-303) -- but ONLY on
2026-03-01..07-12 ("fresh" window), which the original OOS-primary sort never saw (it only ever
scored 01-01..02-28) and which is NOT the sub-project's usual VAL (2025-10-01..12-31, left alone
and untouched by this script; its already-computed values are copied from the existing
`quality_threshold_ranking.csv` as ground truth, not recomputed here).

## Purpose / non-purpose

Diagnostic: "if quality_threshold had been chosen VAL-only (as this project's own convention
requires) instead of OOS-primary, would the answer differ, and does the VAL-only answer generalize
better or worse than the deployed OOS-cherry-picked one on data neither ever saw during selection?"

This is NOT a promotion, live-candidate, or baseline-replacement claim. Per the Fresh-Forward rule,
this replays STORED per-bar model predictions (frozen bundle, causal, no lookahead) through a
single bar-by-bar walk, exactly like the artifact this whole sub-project already treats as a
legitimate (non-live, research-grade) OOS score -- it carries the same
research/diagnostic status as the original `quality_threshold_ranking.csv`, no better, no worse.
No live file is read or written by this script.

Caveat on "freshness": 2026-03-01..06-30 was separately glanced at (in aggregate, portfolio-level,
NOT per-quality-threshold) by unrelated 2026-08-13 experiments (multislot capacity/MFE-gating
research). It was NEVER used to select `quality_threshold` specifically -- that is the only
contamination vector this script targets. Only 2026-07-01..07-12 is untouched by literally
everything tonight; data does not extend further at the time of writing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

EXT_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_DIR = ROOT / "tmp/research_20260813/omega461_quality_threshold_clean_reselection"

SPENT_OOS_START, SPENT_OOS_END = "2026-01-01", "2026-02-28"   # original selection's OOS window
FRESH_START, FRESH_END = "2026-03-01", "2026-07-12"           # never scored by the original sort
PREFIX = "omega1_regime3_expertdq_"

ORIG_RANKING_CSV = {
    "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/quality_threshold_ranking.csv",
    "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/quality_threshold_ranking.csv",
}
COMPONENTS = {
    "h48qual": {
        "pred_csv": EXT_PRED_DIR / "h48qual" / "oos_predictions_q050.csv",
        "deployed_q": 0.50,
    },
    "zig075": {
        "pred_csv": EXT_PRED_DIR / "zig075" / "oos_predictions_q075.csv",
        "deployed_q": 0.75,
    },
}


def load_ohlc_2026() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2026, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame[["timestamp", "open", "high", "low", "close"]]


def recompute_final_action(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Threshold-independent columns (dir_action, quality_for_action) already exist in the
    frozen-bundle prediction file; this only reapplies the gate rule at a new threshold -- no
    model inference, no retraining."""
    out = pred.copy()
    dir_action = pd.to_numeric(out[f"{PREFIX}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    qual = pd.to_numeric(out[f"{PREFIX}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    final_action = np.where((dir_action != 0) & (qual >= float(threshold)), dir_action, 0)
    out[f"{PREFIX}final_action"] = final_action.astype(np.int64)
    out[f"{PREFIX}quality_threshold"] = float(threshold)
    return out


def score_window(pred_full: pd.DataFrame, ohlc: pd.DataFrame, start: str, end: str, threshold: float, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    pred = pred_full[(pred_full["timestamp"] >= start) & (pred_full["timestamp"] <= end)].reset_index(drop=True)
    common = set(pred["timestamp"]) & set(ohlc["timestamp"])
    pred = pred[pred["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
    frame = ohlc[ohlc["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
    if len(pred) == 0 or not frame["timestamp"].equals(pred["timestamp"]):
        raise RuntimeError(f"alignment failure for window {start}..{end} (pred={len(pred)}, frame={len(frame)})")
    dec_src = recompute_final_action(pred, threshold)
    dec = omega._to_fixed_decisions(dec_src, oof=False)
    m = omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)
    m["rows"] = len(frame)
    return m


def main() -> int:
    # Match the original quality_threshold_ranking.csv methodology exactly: fixed BASE_TEMPLATE
    # sizing (notional=0.45, leverage=2.0, take_profit=0.026, stop_loss=0.014), max_hold/cooldown
    # zeroed (train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:302-303),
    # cost_mult=3.0 (that script's --cost-mult default, used throughout its evaluation calls).
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    fee, slip = omega._load_fee_slip()
    cost_mult = 3.0

    ohlc = load_ohlc_2026()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, cfg in COMPONENTS.items():
        orig = pd.read_csv(ORIG_RANKING_CSV[name])
        grid = sorted(orig["quality_threshold"].astype(float).unique().tolist())
        pred_full = pd.read_csv(cfg["pred_csv"])
        pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"])
        pred_min, pred_max = pred_full["timestamp"].min(), pred_full["timestamp"].max()
        print(f"[{name}] prediction file covers {pred_min}..{pred_max}, grid={grid}", flush=True)

        for q in grid:
            val_row = orig[np.isclose(orig["quality_threshold"].astype(float), q)].iloc[0]
            spent_oos_row = orig[np.isclose(orig["quality_threshold"].astype(float), q)].iloc[0]
            fresh_m = score_window(pred_full, ohlc, FRESH_START, FRESH_END, q, fee=fee, slip=slip, cost_mult=cost_mult)
            resweep_spent_m = score_window(pred_full, ohlc, SPENT_OOS_START, SPENT_OOS_END, q, fee=fee, slip=slip, cost_mult=cost_mult)
            row = {
                "component": name,
                "quality_threshold": q,
                "is_deployed": bool(np.isclose(q, cfg["deployed_q"])),
                "val_pnl_original_csv": float(val_row["validation_pnl"]),
                "val_mdd_original_csv": float(val_row["validation_mdd"]),
                "val_trades_original_csv": int(val_row["validation_trades"]),
                "spent_oos_pnl_original_csv": float(spent_oos_row["oos_pnl"]),
                "spent_oos_pnl_resweep_selfcheck": float(resweep_spent_m["pnl"]),
                "spent_oos_trades_resweep_selfcheck": int(resweep_spent_m["trades"]),
                "fresh_pnl": float(fresh_m["pnl"]),
                "fresh_mdd": float(fresh_m["mdd"]),
                "fresh_trades": int(fresh_m["trades"]),
                "fresh_wr": float(fresh_m["wr"]),
                "fresh_rows": int(fresh_m["rows"]),
            }
            all_rows.append(row)
            print(f"  q={q:.2f} deployed={row['is_deployed']}  VAL(orig)={row['val_pnl_original_csv']:+7.2f}%  "
                  f"spentOOS(orig)={row['spent_oos_pnl_original_csv']:+7.2f}%  spentOOS(resweep-selfcheck)={row['spent_oos_pnl_resweep_selfcheck']:+7.2f}%  "
                  f"FRESH(03-01..07-12)={row['fresh_pnl']:+7.2f}% n={row['fresh_trades']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "clean_reselection_grid.csv", index=False)

    for name in COMPONENTS:
        sub = df[df["component"] == name].copy()
        deployed = sub[sub["is_deployed"]].iloc[0]
        val_optimal = sub.sort_values("val_pnl_original_csv", ascending=False).iloc[0]
        fresh_optimal = sub.sort_values("fresh_pnl", ascending=False).iloc[0]
        selfcheck_max_abs_err = float((sub["spent_oos_pnl_original_csv"] - sub["spent_oos_pnl_resweep_selfcheck"]).abs().max())
        summary[name] = {
            "deployed_threshold": float(deployed["quality_threshold"]),
            "deployed_val_pnl": float(deployed["val_pnl_original_csv"]),
            "deployed_spent_oos_pnl": float(deployed["spent_oos_pnl_original_csv"]),
            "deployed_fresh_pnl": float(deployed["fresh_pnl"]),
            "deployed_fresh_trades": int(deployed["fresh_trades"]),
            "val_optimal_threshold": float(val_optimal["quality_threshold"]),
            "val_optimal_val_pnl": float(val_optimal["val_pnl_original_csv"]),
            "val_optimal_fresh_pnl": float(val_optimal["fresh_pnl"]),
            "val_optimal_fresh_trades": int(val_optimal["fresh_trades"]),
            "fresh_optimal_threshold": float(fresh_optimal["quality_threshold"]),
            "fresh_optimal_fresh_pnl": float(fresh_optimal["fresh_pnl"]),
            "would_clean_val_only_selection_match_deployed": bool(np.isclose(float(val_optimal["quality_threshold"]), float(deployed["quality_threshold"]))),
            "resweep_selfcheck_max_abs_pnl_diff_pp_vs_original_csv": selfcheck_max_abs_err,
        }

    report = {
        "purpose": "diagnostic_only_not_promotion",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_on_input": "replays STORED per-bar frozen-bundle predictions (causal, no lookahead), same class as the original quality_threshold_ranking.csv this sub-project already treats as research/diagnostic-grade -- NOT a live/promotion score.",
        "spent_oos_window": [SPENT_OOS_START, SPENT_OOS_END],
        "fresh_window": [FRESH_START, FRESH_END],
        "fresh_window_caveat": "2026-03-01..06-30 was separately glanced at (portfolio-level, not per-quality-threshold) by other 2026-08-13 experiments (multislot capacity/MFE gating); only 2026-07-01..07-12 is untouched by everything tonight. Neither sub-range was ever used to select quality_threshold specifically -- that is the only contamination vector in scope here.",
        "cost_mult": cost_mult,
        "fee": fee,
        "slip": slip,
        "base_template": dict(omega.BASE_TEMPLATE),
        "summary": summary,
    }
    (OUT_DIR / "clean_reselection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_DIR / 'clean_reselection_grid.csv'}")
    print(f"wrote {OUT_DIR / 'clean_reselection_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
