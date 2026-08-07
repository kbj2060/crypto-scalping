#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_cash_region_dsac_fallback_selector_20260526 import (  # noqa: E402
    _decision_reward,
    _safe_col,
)


LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_live_20260526"
PRIMARY_PARENT = LIVE_DIR / "primary_parent.pkl"
PRIMARY_SUMMARY = LIVE_DIR / "primary_summary.json"
FALLBACK_PARENT = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
FALLBACK_SUMMARY = LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_cash_region_hgb_fallback_veto_20260526"
MODEL_ID = "alpha7_cash_region_hgb_fallback_veto_20260526"


def _state_matrix(frame: pd.DataFrame, primary: pd.DataFrame, fallback: pd.DataFrame) -> np.ndarray:
    cols: list[np.ndarray] = []
    market_cols = [
        "smart_money_flow",
        "ofi_acceleration",
        "funding_pressure",
        "crowding_pressure",
        "liquidity_vacuum",
        "trade_intensity",
        "ai_dir_edge",
        "ai_dir_entropy",
        "ai_adverse_risk",
        "tide_vol_zscore",
        "patchtst_regime_sim",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_chop_prob",
        "regime4_pred_confidence",
        "regime4_pred_trend_prob",
        "regime4_pred_whipsaw_prob",
        "tp_sl_action_score",
    ]
    for c in market_cols:
        cols.append(_safe_col(frame, c))
    for c in [
        "action",
        "side",
        "quality_score",
        "confidence",
        "notional_exposure",
        "leverage",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
    ]:
        cols.append(_safe_col(primary, c))
    for c in ["quality_score", "confidence", "notional_exposure", "leverage", "take_profit", "stop_loss", "max_hold_bars"]:
        cols.append(_safe_col(fallback, c))
    x = np.column_stack(cols).astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = _safe_col(dec, "action").astype(np.int64)
    side = _safe_col(dec, "side").astype(np.int64)
    return (action != 0) & (side != 0)


def _build_veto_dataset(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    pnl_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = _safe_col(frame, "close").astype(np.float64)
    x_all = _state_matrix(frame, primary, fallback)
    mask = (~_active(primary)) & _active(fallback)
    idx = np.flatnonzero(mask)
    if idx.size < 30:
        raise RuntimeError("fallback veto dataset too small")

    x = x_all[idx]
    y = np.zeros(len(idx), dtype=np.int64)
    w = np.zeros(len(idx), dtype=np.float32)
    for j, i in enumerate(idx):
        r = float(_decision_reward(close, int(i), fallback.iloc[int(i)], fee=fee, slip=slip))
        allow = int(r > float(pnl_threshold))
        y[j] = allow
        w[j] = float(np.clip(abs(r), 0.03, 2.5))
    return x, y, w, idx


def _apply_veto(primary: pd.DataFrame, fallback: pd.DataFrame, allow_mask: np.ndarray) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    pa = _active(primary)
    fa = _active(fallback)
    for i in range(len(out)):
        if pa[i]:
            continue
        if not fa[i]:
            continue
        if not bool(allow_mask[i]):
            continue
        out.iloc[i] = fallback.iloc[i]
    return out


def _selection_score(cost3: dict[str, Any], baseline_cost3: dict[str, Any]) -> float:
    pnl = float(cost3["pnl"])
    mdd = float(cost3["mdd"])
    trades = int(cost3["trades"])
    base_trades = int(baseline_cost3["trades"])
    return float(pnl / max(abs(mdd), 1e-12) + 0.02 * (trades - base_trades))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    f_train = _predict_scaled(fallback_parent, train_df, fallback_rt)
    f_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    f_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    baseline_val = _combo_metrics(val_df, _combine_primary_fallback(p_val, f_val))
    baseline_eval = _combo_metrics(eval_df, _combine_primary_fallback(p_eval, f_eval))

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: HistGradientBoostingClassifier | None = None

    for pnl_threshold in [0.0, 0.0005, 0.0010, 0.0015]:
        x_train, y_train, w_train, _idx_train = _build_veto_dataset(
            train_df,
            p_train,
            f_train,
            fee=0.0005,
            slip=0.0002,
            pnl_threshold=pnl_threshold,
        )
        if len(np.unique(y_train)) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=False,
            random_state=260526,
        )
        model.fit(x_train, y_train, sample_weight=w_train)

        x_val_all = _state_matrix(val_df, p_val, f_val)
        x_eval_all = _state_matrix(eval_df, p_eval, f_eval)
        pv = model.predict_proba(x_val_all)
        pe = model.predict_proba(x_eval_all)
        classes = list(np.asarray(model.classes_, dtype=np.int64))
        allow_idx = classes.index(1)
        val_allow_prob = pv[:, allow_idx]
        eval_allow_prob = pe[:, allow_idx]

        for prob_floor in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            allow_val = val_allow_prob >= float(prob_floor)
            allow_eval = eval_allow_prob >= float(prob_floor)
            dec_val = _apply_veto(p_val, f_val, allow_val)
            dec_eval = _apply_veto(p_eval, f_eval, allow_eval)
            m_val = _combo_metrics(val_df, dec_val)["cost3"]
            m_eval = _combo_metrics(eval_df, dec_eval)["cost3"]
            row = {
                "pnl_threshold": float(pnl_threshold),
                "prob_floor": float(prob_floor),
                "val_cost3_pnl": float(m_val["pnl"]),
                "val_cost3_mdd": float(m_val["mdd"]),
                "val_cost3_trades": int(m_val["trades"]),
                "oos_cost3_pnl": float(m_eval["pnl"]),
                "oos_cost3_mdd": float(m_eval["mdd"]),
                "oos_cost3_trades": int(m_eval["trades"]),
                "oos_cost3_wr": float(m_eval["wr"]),
                "selection_score": _selection_score(m_val, baseline_val["cost3"]),
            }
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
                best_model = model

    if best is None or best_model is None:
        raise RuntimeError("no valid HGB veto candidate found")

    # Recompute with best config for summary artifacts.
    x_train, y_train, w_train, idx_train = _build_veto_dataset(
        train_df,
        p_train,
        f_train,
        fee=0.0005,
        slip=0.0002,
        pnl_threshold=float(best["pnl_threshold"]),
    )
    # best_model already trained on the same threshold loop, keep consistent and avoid retrain drift.
    x_val_all = _state_matrix(val_df, p_val, f_val)
    x_eval_all = _state_matrix(eval_df, p_eval, f_eval)
    pv = best_model.predict_proba(x_val_all)
    pe = best_model.predict_proba(x_eval_all)
    classes = list(np.asarray(best_model.classes_, dtype=np.int64))
    allow_idx = classes.index(1)
    allow_val = pv[:, allow_idx] >= float(best["prob_floor"])
    allow_eval = pe[:, allow_idx] >= float(best["prob_floor"])
    dec_train = _apply_veto(p_train, f_train, best_model.predict_proba(_state_matrix(train_df, p_train, f_train))[:, allow_idx] >= float(best["prob_floor"]))
    dec_val = _apply_veto(p_val, f_val, allow_val)
    dec_eval = _apply_veto(p_eval, f_eval, allow_eval)

    m_train = _combo_metrics(train_df, dec_train)
    m_val = _combo_metrics(val_df, dec_val)
    m_eval = _combo_metrics(eval_df, dec_eval)

    joblib.dump(
        {
            "model_id": MODEL_ID,
            "selector": best_model,
            "state_dim": int(x_train.shape[1]),
            "best_pnl_threshold": float(best["pnl_threshold"]),
            "best_prob_floor": float(best["prob_floor"]),
            "classes": [int(c) for c in classes],
        },
        OUT_DIR / "cash_region_hgb_veto_selector.pkl",
    )
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=False).to_csv(OUT_DIR / "grid.csv", index=False)

    primary_cash_eval = ~_active(p_eval)
    fallback_active_eval = _active(f_eval)
    summary = {
        "model_id": MODEL_ID,
        "design": "Cash-region-only HGB veto model on current fallback. Primary is unchanged; fallback entries are selectively blocked.",
        "state_dim": int(x_train.shape[1]),
        "train_samples_fallback_rows": int(len(idx_train)),
        "train_label_distribution": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().to_dict().items()},
        "best_config": best,
        "train_metrics": m_train["cost3"],
        "val_metrics": m_val["cost3"],
        "oos_metrics": m_eval["cost3"],
        "baseline_val_metrics": baseline_val["cost3"],
        "baseline_oos_metrics": baseline_eval["cost3"],
        "delta_vs_baseline": {
            "val_cost3_pnl": float(m_val["cost3"]["pnl"] - baseline_val["cost3"]["pnl"]),
            "val_cost3_trades": int(m_val["cost3"]["trades"] - baseline_val["cost3"]["trades"]),
            "oos_cost3_pnl": float(m_eval["cost3"]["pnl"] - baseline_eval["cost3"]["pnl"]),
            "oos_cost3_trades": int(m_eval["cost3"]["trades"] - baseline_eval["cost3"]["trades"]),
        },
        "allow_usage_eval": {
            "primary_cash_rows": int(primary_cash_eval.sum()),
            "fallback_active_rows": int(fallback_active_eval.sum()),
            "allow_rows_all": int(np.sum(allow_eval)),
            "allow_rows_primary_cash": int(np.sum(allow_eval & primary_cash_eval)),
            "allow_rows_primary_cash_and_fb_active": int(np.sum(allow_eval & primary_cash_eval & fallback_active_eval)),
        },
        "artifacts": {
            "selector_ckpt": str((OUT_DIR / "cash_region_hgb_veto_selector.pkl").relative_to(ROOT)),
            "grid": str((OUT_DIR / "grid.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
