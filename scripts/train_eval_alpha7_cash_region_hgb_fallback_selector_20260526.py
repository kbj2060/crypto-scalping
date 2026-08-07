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
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    SPLIT_TS,
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_cash_region_dsac_fallback_selector_20260526 import (  # noqa: E402
    ACTION_FB0,
    ACTION_FB1,
    ACTION_SKIP,
    _compose_final_decisions,
    _decision_reward,
    _extract_runtime,
    _safe_col,
    _state_matrix,
)


OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_cash_region_hgb_fallback_selector_20260526"
S1_PARENT = ROOT / "data/ensemble/supervised/alpha7_v2_only_high_turnover_s1_live_20260526/primary_parent.pkl"
S1_CAND_SUMMARY = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_high_turnover_rebuild_20260526/t0015_c015_h030_s6/summary.json"
MODEL_ID = "alpha7_cash_region_hgb_fallback_selector_20260526"


def _build_cash_supervised(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    fb0: pd.DataFrame,
    fb1: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    non_skip_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = _safe_col(frame, "close").astype(np.float64)
    x_all = _state_matrix(frame, primary, fb0, fb1)
    primary_cash = (_safe_col(primary, "action").astype(np.int64) == 0) | (_safe_col(primary, "side").astype(np.int64) == 0)
    cash_idx = np.flatnonzero(primary_cash)
    if cash_idx.size < 10:
        raise RuntimeError("cash-region supervised dataset too small")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    ws: list[float] = []

    for i in cash_idx:
        i_int = int(i)
        r0 = 0.0
        r1 = _decision_reward(close, i_int, fb0.iloc[i_int], fee=fee, slip=slip) - float(non_skip_penalty)
        r2 = _decision_reward(close, i_int, fb1.iloc[i_int], fee=fee, slip=slip) - float(non_skip_penalty)
        rewards = np.asarray([r0, r1, r2], dtype=np.float64)
        best = int(np.argmax(rewards))
        best_v = float(rewards[best])
        second_v = float(np.partition(rewards, -2)[-2])
        label = int(best if best_v > 0.0 else ACTION_SKIP)
        weight = float(np.clip(abs(best_v - second_v), 0.05, 2.5))

        xs.append(x_all[i_int])
        ys.append(label)
        ws.append(weight)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ws, dtype=np.float32),
        cash_idx.astype(np.int64),
    )


def _predict_actions(
    model: HistGradientBoostingClassifier,
    states: np.ndarray,
    *,
    margin: float,
    prob_floor: float,
) -> np.ndarray:
    proba_part = model.predict_proba(states)
    full = np.zeros((states.shape[0], 3), dtype=np.float64)
    for j, cls in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        full[:, int(cls)] = proba_part[:, j]

    skip_prob = full[:, ACTION_SKIP]
    non = full[:, 1:]
    non_idx = np.argmax(non, axis=1) + 1
    non_prob = np.max(non, axis=1)
    act = np.where(
        ((non_prob - skip_prob) >= float(margin)) & (non_prob >= float(prob_floor)),
        non_idx,
        ACTION_SKIP,
    )
    return act.astype(np.int64)


def _selection_score(cost3: dict[str, Any], baseline_cost3: dict[str, Any]) -> float:
    pnl = float(cost3["pnl"])
    mdd = float(cost3["mdd"])
    trades = int(cost3["trades"])
    base_trades = int(baseline_cost3["trades"])
    return float(pnl / max(abs(mdd), 1e-12) + 0.025 * (trades - base_trades))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_all = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    fb0_parent = joblib.load(FALLBACK_PARENT)
    fb1_parent = joblib.load(S1_PARENT)

    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fb0_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    fb1_rt = _extract_runtime(S1_CAND_SUMMARY)

    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    fb0_train = _predict_scaled(fb0_parent, train_df, fb0_rt)
    fb0_val = _predict_scaled(fb0_parent, val_df, fb0_rt)
    fb0_eval = _predict_scaled(fb0_parent, eval_df, fb0_rt)

    fb1_train = _predict_scaled(fb1_parent, train_df, fb1_rt)
    fb1_val = _predict_scaled(fb1_parent, val_df, fb1_rt)
    fb1_eval = _predict_scaled(fb1_parent, eval_df, fb1_rt)

    baseline_val = _combo_metrics(val_df, _combine_primary_fallback(p_val, fb0_val))
    baseline_eval = _combo_metrics(eval_df, _combine_primary_fallback(p_eval, fb0_eval))

    x_train, y_train, w_train, train_cash_idx = _build_cash_supervised(
        train_df,
        p_train,
        fb0_train,
        fb1_train,
        fee=0.0005,
        slip=0.0002,
        non_skip_penalty=0.0008,
    )
    x_val_all = _state_matrix(val_df, p_val, fb0_val, fb1_val)
    x_eval_all = _state_matrix(eval_df, p_eval, fb0_eval, fb1_eval)

    model = HistGradientBoostingClassifier(
        max_iter=280,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        early_stopping=False,
        random_state=260526,
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for margin in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        for prob_floor in [0.34, 0.38, 0.42, 0.46, 0.50, 0.55]:
            act_val = _predict_actions(model, x_val_all, margin=margin, prob_floor=prob_floor)
            act_eval = _predict_actions(model, x_eval_all, margin=margin, prob_floor=prob_floor)
            dec_val = _compose_final_decisions(p_val, fb0_val, fb1_val, act_val)
            dec_eval = _compose_final_decisions(p_eval, fb0_eval, fb1_eval, act_eval)
            m_val = _combo_metrics(val_df, dec_val)["cost3"]
            m_eval = _combo_metrics(eval_df, dec_eval)["cost3"]
            row = {
                "margin": float(margin),
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

    assert best is not None
    act_val = _predict_actions(model, x_val_all, margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))
    act_eval = _predict_actions(model, x_eval_all, margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))
    dec_val = _compose_final_decisions(p_val, fb0_val, fb1_val, act_val)
    dec_eval = _compose_final_decisions(p_eval, fb0_eval, fb1_eval, act_eval)
    m_train = _combo_metrics(train_df, _compose_final_decisions(p_train, fb0_train, fb1_train, _predict_actions(model, _state_matrix(train_df, p_train, fb0_train, fb1_train), margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))))
    m_val = _combo_metrics(val_df, dec_val)
    m_eval = _combo_metrics(eval_df, dec_eval)

    joblib.dump(
        {
            "model_id": MODEL_ID,
            "selector": model,
            "state_dim": int(x_train.shape[1]),
            "best_margin": float(best["margin"]),
            "best_prob_floor": float(best["prob_floor"]),
            "classes": [int(c) for c in np.asarray(model.classes_, dtype=np.int64).tolist()],
        },
        OUT_DIR / "cash_region_hgb_selector.pkl",
    )
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=False).to_csv(OUT_DIR / "grid.csv", index=False)

    primary_cash_eval = (_safe_col(p_eval, "action").astype(np.int64) == 0) | (_safe_col(p_eval, "side").astype(np.int64) == 0)
    summary = {
        "model_id": MODEL_ID,
        "design": "Cash-region-only HGB fallback selector (skip / current alpha43 fallback / s1 fallback), val-selected margin/prob_floor.",
        "state_dim": int(x_train.shape[1]),
        "train_samples_cash_rows": int(len(train_cash_idx)),
        "train_label_distribution": {
            str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().to_dict().items()
        },
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
        "action_usage_eval_all_rows": {
            "skip": int(np.sum(act_eval == ACTION_SKIP)),
            "fallback_alpha43": int(np.sum(act_eval == ACTION_FB0)),
            "fallback_s1": int(np.sum(act_eval == ACTION_FB1)),
        },
        "action_usage_eval_cash_rows": {
            "cash_rows": int(primary_cash_eval.sum()),
            "skip_on_cash": int(np.sum((act_eval == ACTION_SKIP) & primary_cash_eval)),
            "fb0_on_cash": int(np.sum((act_eval == ACTION_FB0) & primary_cash_eval)),
            "fb1_on_cash": int(np.sum((act_eval == ACTION_FB1) & primary_cash_eval)),
        },
        "artifacts": {
            "selector_ckpt": str((OUT_DIR / "cash_region_hgb_selector.pkl").relative_to(ROOT)),
            "grid": str((OUT_DIR / "grid.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
