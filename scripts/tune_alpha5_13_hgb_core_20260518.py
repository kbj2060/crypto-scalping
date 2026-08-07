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

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_5_lgbm_supervised_parent_20260518 import _decide_actions, _predict_proba_3  # noqa: E402
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _backtest_barrier, _direction_metrics  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.tune_alpha5_9_hgb_action_master_20260518 import _fit_hgb, _hgb_specs  # noqa: E402


MODEL_ID = "alpha5_13_hgb_core_tuned_20260518"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_core_tuned_20260518"


def _x(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols)


def _feature_cols(train_raw: pd.DataFrame, eval_raw: pd.DataFrame, available: set[str]) -> list[str]:
    cols = _alpha4_mapped_features(train_raw, eval_raw, include_future=False)
    return [c for c in cols if c in available]


def _filter_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    action = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    keep = pd.to_numeric(frame["label_train_keep"], errors="coerce").fillna(0).to_numpy(np.int8) == 1
    selected = pd.to_numeric(frame["regime_trade_selected"], errors="coerce").fillna(0).to_numpy(np.int8) == 1
    tp_first = pd.to_numeric(frame["meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.int8) == 1
    is_cash = action == 0
    is_trade = action != 0
    if name == "base":
        return keep
    if name == "cash_plus_tp":
        return keep & (is_cash | (is_trade & tp_first))
    if name == "cash_plus_selected":
        return keep & (is_cash | (is_trade & selected))
    raise ValueError(name)


def _weights(frame: pd.DataFrame, mask: np.ndarray, mode: str) -> np.ndarray:
    base = pd.to_numeric(frame.loc[mask, "label_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    tp_first = pd.to_numeric(frame.loc[mask, "meta_tp_first"], errors="coerce").fillna(0).to_numpy(np.float64)
    selected = pd.to_numeric(frame.loc[mask, "regime_trade_selected"], errors="coerce").fillna(0).to_numpy(np.float64)
    action = pd.to_numeric(frame.loc[mask, "label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    if mode == "uniform":
        return np.ones(len(base), dtype=np.float64)
    if mode == "current":
        return np.clip(base, 1e-4, None)
    if mode == "tp_boost":
        w = np.clip(base, 1e-4, None) * (1.0 + 1.5 * tp_first) * (1.0 + 0.35 * selected)
        w[action == 0] *= 0.85
        return w
    raise ValueError(mode)


def _eval(frame: pd.DataFrame, actions: np.ndarray, labels: np.ndarray, *, fee: float, slip: float, exposure: float, max_hold: int) -> dict[str, Any]:
    bt = {
        f"cost{m}": _backtest_barrier(
            frame,
            actions,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            unit_exposure=float(exposure),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 20:
        score = -1e6 + float(c1["pnl"])
    else:
        score = (
            float(c1["pnl"])
            + 0.50 * float(c2["pnl"])
            + 0.25 * float(c3["pnl"])
            + 14.0 * float(dm["balanced_trade_precision"])
            + 8.0 * float(dm["trade_precision"])
            - 0.35 * abs(float(c1["mdd"]))
            - max(0.0, 0.20 - float(dm["coverage"])) * 12.0
            - max(0.0, float(c1["trades_per_day"]) - 3.5) * 2.0
        )
    return {"backtest": bt, "direction": dm, "score": float(score)}


def _grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Tune Alpha5.13 standalone HGB on regime4_core with stricter label filters.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--filters", default="base,cash_plus_tp,cash_plus_selected")
    p.add_argument("--weight-modes", default="current,tp_boost,uniform")
    p.add_argument("--prob-thresholds", default="0.80,0.85,0.90,0.93,0.95,0.97")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12,0.16")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=51401)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet")
    val_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet")
    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    x_val = _x(val_df, cols)
    x_oos = _x(oos_df, cols)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    filters = [x.strip() for x in str(args.filters).split(",") if x.strip()]
    weight_modes = [x.strip() for x in str(args.weight_modes).split(",") if x.strip()]
    specs = _hgb_specs()
    total = len(filters) * len(weight_modes) * len(specs)
    done = 0
    rows: list[dict[str, Any]] = []

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "feature_count": len(cols),
        "filters": filters,
        "weight_modes": weight_modes,
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for fi, filter_name in enumerate(filters):
        mask = _filter_mask(train_df, filter_name)
        fit_df = train_df.loc[mask].reset_index(drop=True)
        x_train = _x(fit_df, cols)
        y_train = pd.to_numeric(fit_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
        print(json.dumps({
            "stage": "filter_ready",
            "filter": filter_name,
            "rows": int(len(fit_df)),
            "action_counts": {str(k): int(v) for k, v in fit_df["label_action"].value_counts().sort_index().to_dict().items()},
        }, ensure_ascii=False), flush=True)
        for wi, weight_mode in enumerate(weight_modes):
            w_train = _weights(train_df, mask, weight_mode)
            for si, spec in enumerate(specs):
                done += 1
                print(json.dumps({"stage": "fit", "done": done, "total": total, "filter": filter_name, "weight_mode": weight_mode, "hgb": spec.name}, ensure_ascii=False), flush=True)
                model = _fit_hgb(x_train, y_train, w_train, spec, int(args.seed + fi * 100 + wi * 10 + si))
                val_proba = _predict_proba_3(model, x_val)
                oos_proba = _predict_proba_3(model, x_oos)

                best_val: dict[str, Any] | None = None
                for prob in _grid(args.prob_thresholds):
                    for margin in _grid(args.margin_thresholds):
                        val_actions = _decide_actions(val_proba, prob, margin)
                        ev = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                        cand = {"prob": prob, "margin": margin, "actions": val_actions, **ev}
                        if best_val is None or float(cand["score"]) > float(best_val["score"]):
                            best_val = cand
                assert best_val is not None
                oos_actions = _decide_actions(oos_proba, float(best_val["prob"]), float(best_val["margin"]))
                oos_eval = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                artifact = args.out_dir / f"{filter_name}_{weight_mode}_{spec.name}_alpha5_13_hgb_core.joblib"
                joblib.dump({
                    "model_id": MODEL_ID,
                    "model": model,
                    "feature_cols": cols,
                    "filter": filter_name,
                    "weight_mode": weight_mode,
                    "hgb": {"name": spec.name},
                    "decision": {"prob": float(best_val['prob']), "margin": float(best_val['margin'])},
                }, artifact)
                row = {
                    "filter": filter_name,
                    "weight_mode": weight_mode,
                    "hgb": {"name": spec.name},
                    "feature_count": len(cols),
                    "fit_rows": int(len(fit_df)),
                    "validation": {k: v for k, v in best_val.items() if k != "actions"},
                    "oos": oos_eval,
                    "artifact": str(artifact),
                }
                rows.append(row)
                print(json.dumps({
                    "stage": "candidate",
                    "filter": filter_name,
                    "weight_mode": weight_mode,
                    "hgb": spec.name,
                    "fit_rows": len(fit_df),
                    "val_score": best_val["score"],
                    "val_dir": best_val["direction"],
                    "val_cost1": best_val["backtest"]["cost1"],
                    "oos_score": oos_eval["score"],
                    "oos_dir": oos_eval["direction"],
                    "oos_cost1": oos_eval["backtest"]["cost1"],
                }, ensure_ascii=False, default=_json_default), flush=True)

    best = max(rows, key=lambda r: float(r["validation"]["score"]))
    summary = {"model_id": MODEL_ID, "experiments": rows, "best": best, "top10": sorted(rows, key=lambda r: float(r["validation"]["score"]), reverse=True)[:10]}
    summary_path = args.out_dir / "alpha5_13_hgb_core_tuned_summary.json"
    grid_path = args.out_dir / "alpha5_13_hgb_core_tuned_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "filter": r["filter"],
            "weight_mode": r["weight_mode"],
            "hgb_name": r["hgb"]["name"],
            "feature_count": r["feature_count"],
            "fit_rows": r["fit_rows"],
            "val_score": r["validation"]["score"],
            "val_prob": r["validation"]["prob"],
            "val_margin": r["validation"]["margin"],
            "val_trade_precision": r["validation"]["direction"]["trade_precision"],
            "val_balanced_trade_precision": r["validation"]["direction"]["balanced_trade_precision"],
            "val_coverage": r["validation"]["direction"]["coverage"],
            "val_cost1_pnl": r["validation"]["backtest"]["cost1"]["pnl"],
            "val_cost1_mdd": r["validation"]["backtest"]["cost1"]["mdd"],
            "oos_score": r["oos"]["score"],
            "oos_trade_precision": r["oos"]["direction"]["trade_precision"],
            "oos_balanced_trade_precision": r["oos"]["direction"]["balanced_trade_precision"],
            "oos_coverage": r["oos"]["direction"]["coverage"],
            "oos_cost1_pnl": r["oos"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_mdd": r["oos"]["backtest"]["cost1"]["mdd"],
            "oos_cost1_trades": r["oos"]["backtest"]["cost1"]["trades"],
            "artifact": r["artifact"],
        }
        for r in rows
    ]).sort_values("val_score", ascending=False).to_csv(grid_path, index=False)
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_path),
        "best": {
            "filter": best["filter"],
            "weight_mode": best["weight_mode"],
            "hgb": best["hgb"]["name"],
            "fit_rows": best["fit_rows"],
            "val_score": best["validation"]["score"],
            "oos_score": best["oos"]["score"],
            "oos_cost1": best["oos"]["backtest"]["cost1"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
