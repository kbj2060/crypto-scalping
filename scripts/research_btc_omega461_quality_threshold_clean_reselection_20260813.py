#!/usr/bin/env python3
"""Diagnostic ONLY (not promotion/live-candidate): clean re-selection of BTC h48qual
`quality_threshold` on data that the original OOS-primary selection never scored against.

BTC analog of `research_eth_omega461_quality_threshold_clean_reselection_20260813.py`
(read first -- this mirrors its methodology exactly, adapted for BTC's single-component
deployed model).

## Background

`train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806.py` (the
script that produced the live-deployed BTC h48qual bundle) sorts candidate thresholds by
`key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True` (its line 755)
-- the same OOS-primary-sort pattern already found and scoped for ETH. Its "oos" frame covers
EXACTLY 2026-01-01..2026-07-12 16:50 (independently confirmed by reading the deployed bundle's
own `oos_predictions_q055.csv` timestamp column).

Deployed value: quality_threshold=0.55, component=h48qual (confirmed via
`trading_bot_modules/runtime_config.py` `OMEGA4_6_1_SHADOW_ASSET_CONFIG["btc"]`, cross-checked
against the bundle dir name
`tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/`).

Opening the deployed `quality_threshold_ranking.csv` directly shows something DIFFERENT from
both ETH cases: q=0.55 is simultaneously the VAL-optimal AND the OOS-primary pick (all 5
candidates -0.40/0.45/0.50/0.55/0.60- have NEGATIVE oos_pnl on the original OOS window; q=0.55 is
the least-negative OOS AND clearly the best VAL, +10.29% vs next best +3.65%). So for BTC, unlike
ETH's zig075, the OOS-primary sort bug did NOT change which threshold got selected -- a
VAL-only-sorted selection would have picked the exact same q=0.55. This script still checks
whether that pick holds up on a genuinely fresh window, per the same policy this sub-project
applies to ETH.

## What this script does (no retraining, no GPU)

Recomputes `final_action` at every candidate quality_threshold directly from the ALREADY-COMPUTED,
threshold-independent columns (`dir_action`, `quality_for_action`) already sitting in the existing
frozen-bundle prediction file
`tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806/oos_predictions_q055.csv`
(built 2026-08-06 by `infer_btc_h48qual_predictions_freshforward_ext_swingtransition_20260806.py`
via a forward-only pass of the FROZEN deployed bundle over 2026-01-01..2026-08-01 -- no lookahead,
no retraining, confirmed by that script's own docstring and by reading its code directly: it loads
`true_3head_tabm_bundle.pt` and runs pure forward inference, no training loop). It then reapplies
`dir_action != 0 and quality_for_action >= threshold -> final_action = dir_action`, the exact same
rule the parent training script and the live gate use.

IMPORTANT module-aliasing finding (specific to BTC, did not apply to ETH's own script): the shared
`train_eval_omega1_2_tabm_3head_20260603.py` (`parent`) hardcodes
`import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega` at ITS OWN top level, so
`parent._to_decisions(...)` ALWAYS resolves to the ETH diffusion-risk module's
`_to_fixed_decisions`/`BASE_TEMPLATE`, regardless of whether the caller is the ETH or BTC training
script. The BTC original training script calls `parent._to_decisions(...)` for its ranking sweep
(its lines 726-727) and separately its OWN BTC omega module's `_metrics`/`_load_fee_slip` (its
lines 728-729, 626) for PnL/fee/slip. Its own `omega.BASE_TEMPLATE["max_hold"] = 0` /
`["cooldown"] = 0` (its lines 130-131) zero the BTC module's OWN BASE_TEMPLATE dict, which is a
DIFFERENT object from the one `parent._to_decisions` actually reads (the ETH module's) -- so that
zeroing likely never took effect on the original ranking sweep's decisions. To faithfully
reproduce what the ORIGINAL script's `parent._to_decisions` calls actually executed, this script
imports the ETH diffusion-risk module directly for `_to_fixed_decisions` (exactly what
`parent._to_decisions` resolves to) and does NOT zero its max_hold/cooldown, and separately imports
BTC's own diffusion-risk module for `_metrics`/`_load_fee_slip`/fee/slip (exactly what the BTC
script's own `omega.*` calls used). BASE_TEMPLATE notional/leverage/take_profit/stop_loss/
EXPERT_SCALES are identical dicts between the two modules (verified by direct comparison), so this
only matters for max_hold/cooldown. The resweep self-check below (comparing this script's
recomputed spent-OOS pnl against the original ranking CSV's stored oos_pnl for the SAME threshold)
is the empirical check on this fidelity, exactly as ETH's script already does -- ETH's own
self-check gap was 8-13pp (attributed there to the extended-prediction file being a SEPARATE
re-inference pass, not a literal replay of the original in-process oos_src), so a nonzero gap here
is expected and not by itself evidence of a mistake.

## Purpose / non-purpose

Diagnostic: "does BTC's deployed quality_threshold=0.55 hold up on 2026-07-13..08-01, a window the
original OOS-primary sort never scored against?" This is NOT a promotion, live-candidate, or
baseline-replacement claim. Per the Fresh-Forward rule, this replays STORED per-bar model
predictions (frozen bundle, causal, no lookahead) through a single bar-by-bar walk -- same
research/diagnostic status as the original `quality_threshold_ranking.csv`, no better, no worse.
No live file is read or written by this script.

## Freshness

Original OOS-primary selection scored 2026-01-01..2026-07-12 16:50 only (confirmed from the
deployed bundle's own `oos_predictions_q055.csv`). 2026-07-13..2026-08-01 was never scored by that
selection and is the fresh window used here. Data does not extend further at the time of writing
(freshforward_ext file's max timestamp is 2026-08-01 17:40).
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega_decisions  # noqa: E402  (ETH module -- what parent._to_decisions actually resolves to)
import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as omega  # noqa: E402  (BTC's own module -- _metrics/_load_fee_slip/fee/slip)

DEPLOYED_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition"
EXT_PRED_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806"
BASE_2026 = ROOT / "data/splits/year_oos/btc_features_2026.csv"
OUT_DIR = ROOT / "tmp/research_20260813/btc_omega461_quality_threshold_clean_reselection"

SPENT_OOS_START, SPENT_OOS_END = "2026-01-01", "2026-07-12"   # original selection's OOS window
FRESH_START, FRESH_END = "2026-07-13", "2026-08-01"           # never scored by the original sort
PREFIX = "omega1_regime3_expertdq_"

ORIG_RANKING_CSV = {
    "h48qual": DEPLOYED_DIR / "quality_threshold_ranking.csv",
}
COMPONENTS = {
    "h48qual": {
        "pred_csv": EXT_PRED_DIR / "oos_predictions_q055.csv",
        "deployed_q": 0.55,
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
    dec = omega_decisions._to_fixed_decisions(dec_src, oof=False)
    m = omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)
    m["rows"] = len(frame)
    return m


def main() -> int:
    # Match the original quality_threshold_ranking.csv methodology: decisions via
    # parent._to_decisions (== omega_decisions._to_fixed_decisions, the ETH module, NOT zeroed --
    # see module-aliasing note in the file docstring), metrics via BTC's own omega._metrics,
    # cost_mult=3.0 (BTC script's --cost-mult default, used throughout its evaluation calls).
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
                  f"FRESH(07-13..08-01)={row['fresh_pnl']:+7.2f}% n={row['fresh_trades']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "clean_reselection_grid.csv", index=False)

    for name in COMPONENTS:
        sub = df[df["component"] == name].copy()
        deployed = sub[sub["is_deployed"]].iloc[0]
        val_optimal = sub.sort_values("val_pnl_original_csv", ascending=False).iloc[0]
        oos_primary_optimal = sub.sort_values(["spent_oos_pnl_original_csv", "val_pnl_original_csv"], ascending=False).iloc[0]
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
            "oos_primary_optimal_threshold": float(oos_primary_optimal["quality_threshold"]),
            "would_val_only_selection_match_deployed": bool(np.isclose(float(val_optimal["quality_threshold"]), float(deployed["quality_threshold"]))),
            "would_oos_primary_selection_match_deployed": bool(np.isclose(float(oos_primary_optimal["quality_threshold"]), float(deployed["quality_threshold"]))),
            "fresh_optimal_threshold": float(fresh_optimal["quality_threshold"]),
            "fresh_optimal_fresh_pnl": float(fresh_optimal["fresh_pnl"]),
            "would_clean_val_only_selection_survive_fresh": bool(np.isclose(float(val_optimal["quality_threshold"]), float(fresh_optimal["quality_threshold"]))),
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
        "cost_mult": cost_mult,
        "fee": fee,
        "slip": slip,
        "base_template_used_for_decisions": dict(omega_decisions.BASE_TEMPLATE),
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
