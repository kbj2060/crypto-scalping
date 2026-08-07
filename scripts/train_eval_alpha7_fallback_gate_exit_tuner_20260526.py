#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _active,
    _combine_primary_fallback,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
FALLBACK_PARENT = BASELINE.fallback_parent
FALLBACK_SUMMARY = BASELINE.fallback_summary
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_gate_exit_tuner_20260526"


def _gate_mask(frame: pd.DataFrame, dec: pd.DataFrame, gate_name: str, p1: float, p2: float, p3: float) -> np.ndarray:
    if gate_name == "none":
        return _active(dec)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    active = _active(dec)
    cur_range = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_range_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cur_chop = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_chop_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cur_whip = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_whipsaw_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cur_trend = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_trend_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cur_inst = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_instability_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_range = pd.to_numeric(frame["regime4_pred_range_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_chop = pd.to_numeric(frame["regime4_pred_chop_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_whip = pd.to_numeric(frame["regime4_pred_whipsaw_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_trend = pd.to_numeric(frame["regime4_pred_trend_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pred_inst = pd.to_numeric(frame["regime4_pred_instability_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fund_pressure = pd.to_numeric(frame["funding_pressure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    crowd = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_factor_crowding"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    flow_exhaust = pd.to_numeric(frame["ai_flow_exhaustion"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    flow_pressure = pd.to_numeric(frame["ai_flow_pressure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    net_taker = pd.to_numeric(frame["net_taker_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    cur_score = cur_range + cur_chop + cur_whip - cur_trend - cur_inst
    pred_score = pred_range + pred_chop + pred_whip - pred_trend - pred_inst
    side_long = side > 0
    side_short = side < 0

    if gate_name == "current_mr":
        allow = (cur_score >= p1) & (cur_trend <= p2)
    elif gate_name == "predicted_mr":
        allow = (pred_score >= p1) & (pred_trend <= p2)
    elif gate_name == "current_or_pred_mr":
        allow = ((cur_score >= p1) | (pred_score >= p2)) & (cur_inst <= p3)
    elif gate_name == "crowding_revert":
        allow = ((side_long & (fund_pressure <= -p1)) | (side_short & (fund_pressure >= p1))) & (crowd >= p2)
    elif gate_name == "flow_exhaust_revert":
        allow = ((side_long & (flow_pressure <= -p1)) | (side_short & (flow_pressure >= p1))) & (flow_exhaust >= p2) & (np.abs(net_taker) <= p3)
    elif gate_name == "hybrid_revert":
        allow = (((cur_score >= p1) | (pred_score >= p2)) & (crowd >= p3))
    else:
        raise ValueError(f"unknown gate_name={gate_name}")
    return active & allow


def _apply_gate_exit(frame: pd.DataFrame, dec: pd.DataFrame, gate_name: str, p1: float, p2: float, p3: float, tp_scale: float, sl_scale: float, hold_cap: int) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    allow = _gate_mask(frame, out, gate_name, p1, p2, p3)
    block = active & (~allow)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score", "confidence"]] = 0
    out.loc[block, "leverage"] = 1.0
    keep = allow
    if np.any(keep):
        out.loc[keep, "take_profit"] = pd.to_numeric(out.loc[keep, "take_profit"], errors="coerce").clip(lower=1e-4) * float(tp_scale)
        out.loc[keep, "stop_loss"] = pd.to_numeric(out.loc[keep, "stop_loss"], errors="coerce").clip(lower=1e-4) * float(sl_scale)
        if int(hold_cap) > 0:
            capped = np.minimum(pd.to_numeric(out.loc[keep, "max_hold_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64), int(hold_cap))
            out.loc[keep, "max_hold_bars"] = np.maximum(capped, 1)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune runtime-native gate + exit overlays on Alpha7 current fallback.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    baseline_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, fallback_eval),
            fee=fee,
            slip=slip,
        )
    )

    gate_grid = [
        ("none", [(0.0, 0.0, 0.0)]),
        ("current_mr", [(0.30, 0.35, 0.0), (0.40, 0.30, 0.0)]),
        ("current_or_pred_mr", [(0.30, 0.30, 0.30), (0.40, 0.40, 0.25)]),
        ("crowding_revert", [(0.20, 0.20, 0.0), (0.30, 0.25, 0.0)]),
        ("hybrid_revert", [(0.30, 0.30, 0.15), (0.40, 0.40, 0.20)]),
    ]
    tp_scales = [0.65, 0.80, 1.00]
    sl_scales = [0.85, 1.00]
    hold_caps = [24, 48, 0]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for gate_name, params_list in gate_grid:
        for p1, p2, p3 in params_list:
            for tp_scale in tp_scales:
                for sl_scale in sl_scales:
                    for hold_cap in hold_caps:
                        tuned_val = _apply_gate_exit(val_df, fallback_val, gate_name, p1, p2, p3, tp_scale, sl_scale, hold_cap)
                        tuned_eval = _apply_gate_exit(eval_df, fallback_eval, gate_name, p1, p2, p3, tp_scale, sl_scale, hold_cap)
                        val_final = _combine_primary_fallback(primary_val, tuned_val)
                        eval_final = _combine_primary_fallback(primary_eval, tuned_eval)
                        val_metrics = _compact_costs(
                            _metrics(
                                val_df,
                                parent_for_features=ref_parent,
                                runner=noop_runner,
                                runner_cfg=noop_cfg,
                                dec=val_final,
                                fee=fee,
                                slip=slip,
                            )
                        )
                        eval_metrics = _compact_costs(
                            _metrics(
                                eval_df,
                                parent_for_features=ref_parent,
                                runner=noop_runner,
                                runner_cfg=noop_cfg,
                                dec=eval_final,
                                fee=fee,
                                slip=slip,
                            )
                        )
                        val_fallback_rows = int((~_active(primary_val) & _active(tuned_val)).sum())
                        eval_fallback_rows = int((~_active(primary_eval) & _active(tuned_eval)).sum())
                        row = {
                            "gate_name": gate_name,
                            "p1": float(p1),
                            "p2": float(p2),
                            "p3": float(p3),
                            "tp_scale": float(tp_scale),
                            "sl_scale": float(sl_scale),
                            "hold_cap": int(hold_cap),
                            "selection_score": float(_score(val_metrics)),
                            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                            "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                            "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                            "val_fallback_rows": val_fallback_rows,
                            "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                            "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                            "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                            "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                            "oos_fallback_rows": eval_fallback_rows,
                            "delta_vs_baseline": float(eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
                        }
                        rows.append(row)
                        if best is None or row["selection_score"] > best["selection_score"]:
                            best = row
    assert best is not None
    grid_path = args.out_dir / "grid.csv"
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(grid_path, index=False)
    report = {
        "model_id": "alpha7_fallback_gate_exit_tuner_20260526",
        "design": "Primary and current fallback models are unchanged. Validation selects a runtime-native gate plus TP/SL/max_hold override only for current fallback rows, then fixes that overlay for 2026 OOS.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "current_fallback_combo_metrics": baseline_metrics,
        },
        "best_by_selection": best,
        "artifacts": {"grid": str(grid_path)},
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best": best,
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
