#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import EVAL_CSV, FORBIDDEN_PREFIXES, TRAIN_CSV  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "alpha8_lgbm_entry_filter_combo_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


FEATURE_PREFIXES = (
    "clean_regime4_state24_sticky090_v2_",
    "regime4_pred_",
    "teacher_",
)
FEATURE_EXACT = (
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "atr14_pct",
    "volatility_z",
    "rsi14",
    "vwap_dist",
)


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in {"timestamp", "open", "high", "low", "close"}:
            continue
        if str(col).startswith(FORBIDDEN_PREFIXES):
            raise RuntimeError(f"forbidden legacy regime feature selected: {col}")
        if col in FEATURE_EXACT or str(col).startswith(FEATURE_PREFIXES):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any() and vals.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def _build_x(frame: pd.DataFrame, dec: pd.DataFrame, *, cols: list[str], origin: np.ndarray) -> pd.DataFrame:
    x = frame.reindex(columns=cols).copy()
    for col in ("side", "notional_exposure", "leverage", "take_profit", "stop_loss", "max_hold_bars", "quality_score", "confidence"):
        x[f"dec_{col}"] = _num(dec, col)
    x["origin_primary"] = (origin == 1).astype(float)
    x["origin_fallback"] = (origin == 2).astype(float)
    x["side_x_confidence"] = x["dec_side"] * x["dec_confidence"]
    x["side_x_quality"] = x["dec_side"] * x["dec_quality_score"]
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _origin(primary_dec: pd.DataFrame, fallback_dec: pd.DataFrame) -> np.ndarray:
    p = _active(primary_dec)
    f = _active(fallback_dec)
    out = np.zeros(len(primary_dec), dtype=np.int8)
    out[p] = 1
    out[(~p) & f] = 2
    return out


def _path_label(frame: pd.DataFrame, dec: pd.DataFrame, *, mode: str) -> np.ndarray:
    close = _close(frame)
    side = _num(dec, "side").astype(np.int64)
    active = _active(dec)
    notional = np.maximum(_num(dec, "notional_exposure"), 0.0)
    tp = np.maximum(_num(dec, "take_profit"), 0.0)
    sl = np.maximum(np.abs(_num(dec, "stop_loss")), 0.0)
    hold = np.maximum(_num(dec, "max_hold_bars", 1.0).astype(np.int64), 1)
    y = np.zeros(len(frame), dtype=np.int8)
    for i in np.flatnonzero(active):
        entry_i = min(i + 1, len(close) - 1)
        end_i = min(entry_i + int(hold[i]), len(close) - 1)
        if end_i <= entry_i:
            continue
        entry = float(close[entry_i])
        path = close[entry_i : end_i + 1]
        raw = (path - entry) / max(entry, 1e-12)
        pnl_path = raw * float(side[i]) * float(notional[i])
        hit_tp = np.flatnonzero(pnl_path >= float(tp[i]))
        hit_sl = np.flatnonzero(pnl_path <= -float(sl[i]))
        tp_first = bool(len(hit_tp) and (not len(hit_sl) or int(hit_tp[0]) <= int(hit_sl[0])))
        final_pos = float(pnl_path[-1]) > 0.0
        if mode == "tp_first":
            y[i] = int(tp_first)
        elif mode == "tp_or_final_positive":
            y[i] = int(tp_first or (not len(hit_sl) and final_pos))
        else:
            raise ValueError(f"unknown label mode: {mode}")
    return y


def _veto_by_prob(dec: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    veto = _active(out) & (prob < float(threshold))
    if np.any(veto):
        for col, value in (
            ("action", 0),
            ("side", 0),
            ("notional_exposure", 0.0),
            ("position_fraction", 0.0),
            ("take_profit", 0.0),
            ("stop_loss", 0.0),
            ("max_hold_bars", 0),
            ("cooldown_bars", 0),
        ):
            out.loc[veto, col] = value
        out.loc[veto, "leverage"] = 1.0
    return out


def _scale_notional(dec: pd.DataFrame, mult: float, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    if np.any(active):
        notional = np.minimum(np.maximum(_num(out, "notional_exposure") * float(mult), 0.0), float(cap))
        leverage = np.maximum(_num(out, "leverage", 1.0), 1e-8)
        out.loc[active, "notional_exposure"] = notional[active]
        out.loc[active, "position_fraction"] = notional[active] / leverage[active]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--notional-mults", default="1.0")
    ap.add_argument("--notional-cap", type=float, default=5.0)
    ap.add_argument("--label-mode", choices=["tp_first", "tp_or_final_positive"], default="tp_or_final_positive")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_train = _predict_scaled(primary, train_df, primary_rt).reset_index(drop=True)
    f_train = _predict_scaled(fallback, train_df, fallback_rt).reset_index(drop=True)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    f_val = _predict_scaled(fallback, val_df, fallback_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt).reset_index(drop=True)
    combo_train = _combine_primary_fallback(p_train, f_train).reset_index(drop=True)
    combo_val = _combine_primary_fallback(p_val, f_val).reset_index(drop=True)
    combo_eval = _combine_primary_fallback(p_eval, f_eval).reset_index(drop=True)

    feature_cols = _feature_cols(train_all)
    train_origin = _origin(p_train, f_train)
    val_origin = _origin(p_val, f_val)
    eval_origin = _origin(p_eval, f_eval)
    train_active = _active(combo_train)
    y_train_all = _path_label(train_df, combo_train, mode=str(args.label_mode))
    x_train_all = _build_x(train_df, combo_train, cols=feature_cols, origin=train_origin)
    x_train = x_train_all.loc[train_active].reset_index(drop=True)
    y_train = y_train_all[train_active]
    if int(np.unique(y_train).size) < 2:
        raise RuntimeError("entry filter training labels have fewer than 2 classes")

    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.4,
        reg_lambda=2.0,
        random_state=80529,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    joblib.dump({"model_id": MODEL_ID, "model": model, "feature_cols": feature_cols}, OUT_DIR / "entry_filter_lgbm.pkl")

    x_val = _build_x(val_df, combo_val, cols=feature_cols, origin=val_origin)
    x_eval = _build_x(eval_df, combo_eval, cols=feature_cols, origin=eval_origin)
    val_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float64)
    eval_prob = np.asarray(model.predict_proba(x_eval)[:, 1], dtype=np.float64)
    evaluator = OfficialCost3()
    baseline_val = evaluator(val_df, combo_val)
    baseline_oos = evaluator(eval_df, combo_eval)

    rows: list[dict[str, Any]] = []
    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]
    notional_mults = [float(x) for x in str(args.notional_mults).split(",") if x.strip()]
    for t in thresholds:
        for nm in notional_mults:
            val_dec = _scale_notional(_veto_by_prob(combo_val, val_prob, t), nm, float(args.notional_cap))
            val = evaluator(val_df, val_dec)
            rows.append(
                {
                    "threshold": t,
                    "notional_mult": nm,
                    "notional_cap": float(args.notional_cap),
                    "score": float(val["pnl"]) + 150.0 * float(val["wr"]) - 0.25 * abs(float(val["mdd"])),
                    **{f"val_{k}": v for k, v in val.items()},
                }
            )
            print(json.dumps({"stage": "val", "threshold": t, "notional_mult": nm, "val": val}, ensure_ascii=False), flush=True)
    val_rank = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    val_rank.to_csv(OUT_DIR / "validation_thresholds.csv", index=False)

    oos_rows: list[dict[str, Any]] = []
    for _, r in val_rank.head(5).iterrows():
        t = float(r["threshold"])
        nm = float(r["notional_mult"])
        oos_dec = _scale_notional(_veto_by_prob(combo_eval, eval_prob, t), nm, float(r["notional_cap"]))
        oos = evaluator(eval_df, oos_dec)
        oos_rows.append(
            {
                "threshold": t,
                "notional_mult": nm,
                "notional_cap": float(r["notional_cap"]),
                **{f"val_{k[4:]}": v for k, v in r.items() if str(k).startswith("val_")},
                **{f"oos_{k}": v for k, v in oos.items()},
            }
        )
        print(json.dumps({"stage": "oos", "threshold": t, "notional_mult": nm, "oos": oos}, ensure_ascii=False), flush=True)
    oos_rank = pd.DataFrame(oos_rows).sort_values(["oos_pnl", "oos_wr"], ascending=False).reset_index(drop=True)
    oos_rank.to_csv(OUT_DIR / "oos_thresholds.csv", index=False)
    best = oos_rank.iloc[0].to_dict() if len(oos_rank) else {}
    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 combo-preserving LightGBM entry meta-filter. Direction remains Alpha7 primary/fallback; filter can only veto low-probability entries.",
        "label_mode": str(args.label_mode),
        "train_split": "2025 before 2025-10-01",
        "selection_split": "2025Q4 threshold selection",
        "oos_split": "2026 eval",
        "baseline": {"val_cost3": baseline_val, "oos_cost3": baseline_oos},
        "train_active_rows": int(train_active.sum()),
        "train_positive_rate": float(np.mean(y_train)),
        "feature_count": int(len(feature_cols) + 12),
        "target_hit": bool(best and float(best.get("oos_pnl", 0.0)) >= 200.0 and float(best.get("oos_wr", 0.0)) >= 0.50),
        "best": best,
        "audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_prefix_count": 0,
            "no_oos_training": True,
            "live_wired": False,
        },
        "artifacts": {
            "model": str(OUT_DIR / "entry_filter_lgbm.pkl"),
            "validation_thresholds": str(OUT_DIR / "validation_thresholds.csv"),
            "oos_thresholds": str(OUT_DIR / "oos_thresholds.csv"),
            "summary": str(OUT_DIR / "summary.json"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT_DIR / "summary.json"), "target_hit": summary["target_hit"], "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
