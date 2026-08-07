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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_cash_region_dsac_fallback_selector_20260526 import (  # noqa: E402
    ACTION_FB0,
    ACTION_FB1,
    ACTION_SKIP,
    _decision_reward,
    _extract_runtime,
    _safe_col,
    _state_matrix,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402


MODEL_ID = "alpha7_01543_primary_cash_fallback_hgb_20260527"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01543_primary_cash_fallback_hgb_20260527"
BEST_JSON = ROOT / "tmp/causal_regen_20260516/alpha3_1_alpha6_alpha7_combo_loop_20260527/best.json"
S1_PARENT = ROOT / "data/ensemble/supervised/alpha7_v2_only_high_turnover_s1_live_20260526/primary_parent.pkl"
S1_CAND_SUMMARY = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_high_turnover_rebuild_20260526/t0015_c015_h030_s6/summary.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _load_01543_config() -> dict[str, Any]:
    best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    cfg = dict(best["best_validation_selected"]["config"])
    if cfg.get("name") != "01543_random_alpha7_primary":
        raise RuntimeError(f"unexpected validation-selected config: {cfg.get('name')}")
    if cfg.get("source") != "alpha7_primary":
        raise RuntimeError(f"01543 source must be alpha7_primary, got {cfg.get('source')}")
    return cfg


def _load_train_val_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = loop._merge_state24(_read(v31.DEFAULT_TRAIN), loop.alpha3_full.SIDE_CLEAN4_2025)
    eval_df = loop._merge_state24(_read(v31.DEFAULT_EVAL), loop.alpha3_full.SIDE_CLEAN4_2026)
    a7_train = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    a7_eval = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_all = loop._augment_with_alpha7_features(train_all, a7_train)
    eval_df = loop._augment_with_alpha7_features(eval_df, a7_eval)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return train_df, val_df, eval_df.reset_index(drop=True)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _compose(primary: pd.DataFrame, fb0: pd.DataFrame, fb1: pd.DataFrame, selector_action: np.ndarray) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    fb0 = fb0.reset_index(drop=True)
    fb1 = fb1.reset_index(drop=True)
    selector_action = np.asarray(selector_action, dtype=np.int64)
    primary_cash = ~_active(out)
    for action, source in ((ACTION_FB0, fb0), (ACTION_FB1, fb1)):
        mask = primary_cash & (selector_action == action) & _active(source)
        for col in source.columns:
            if col in out.columns:
                out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _build_cash_supervised(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    fb0: pd.DataFrame,
    fb1: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    non_skip_penalty: float,
    min_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = _safe_col(frame, "close").astype(np.float64)
    states = _state_matrix(frame, primary, fb0, fb1)
    primary_cash = ~_active(primary)
    cash_idx = np.flatnonzero(primary_cash)
    if cash_idx.size < 1000:
        raise RuntimeError(f"too few 01543 primary-cash rows: {cash_idx.size}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ws: list[float] = []
    raw_rewards: list[tuple[float, float, float]] = []
    for i in cash_idx:
        i_int = int(i)
        r0 = 0.0
        r1 = _decision_reward(close, i_int, fb0.iloc[i_int], fee=fee, slip=slip) - float(non_skip_penalty)
        r2 = _decision_reward(close, i_int, fb1.iloc[i_int], fee=fee, slip=slip) - float(non_skip_penalty)
        rewards = np.asarray([r0, r1, r2], dtype=np.float64)
        best = int(np.argmax(rewards))
        best_v = float(rewards[best])
        second_v = float(np.partition(rewards, -2)[-2])
        label = int(best if best_v > float(min_edge) else ACTION_SKIP)
        weight = float(np.clip(abs(best_v - second_v), 0.03, 2.5))
        xs.append(states[i_int])
        ys.append(label)
        ws.append(weight)
        raw_rewards.append((r0, r1, r2))
    y = np.asarray(ys, dtype=np.int64)
    meta = {
        "cash_rows": int(cash_idx.size),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "mean_rewards_skip_fb0_fb1": np.asarray(raw_rewards, dtype=np.float64).mean(axis=0).tolist(),
        "non_skip_penalty": float(non_skip_penalty),
        "min_edge": float(min_edge),
    }
    return np.asarray(xs, dtype=np.float32), y, np.asarray(ws, dtype=np.float32), cash_idx, meta


def _predict_actions(model: HistGradientBoostingClassifier, states: np.ndarray, *, margin: float, prob_floor: float) -> np.ndarray:
    proba_part = model.predict_proba(states)
    full = np.zeros((states.shape[0], 3), dtype=np.float64)
    for j, cls in enumerate(np.asarray(model.classes_, dtype=np.int64)):
        full[:, int(cls)] = proba_part[:, j]
    skip_prob = full[:, ACTION_SKIP]
    non = full[:, 1:]
    non_idx = np.argmax(non, axis=1) + 1
    non_prob = np.max(non, axis=1)
    return np.where(((non_prob - skip_prob) >= float(margin)) & (non_prob >= float(prob_floor)), non_idx, ACTION_SKIP).astype(np.int64)


def _backtest(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    stack: dict[str, Any],
    q: np.ndarray,
    cfg_01543: dict[str, Any],
) -> dict[str, Any]:
    guard = loop._guard(cfg_01543)
    overlay = loop._overlay(stack["overlay"], cfg_01543)
    return loop.backtest_signal_limit_exit_guard(
        frame,
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        dec,
        overlay,
        loop._default_limit_cfg(),
        guard,
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=3.0,
    )


def _selection_score(c3: dict[str, Any]) -> float:
    trades = int(c3.get("trades", 0))
    if trades < 20:
        return -1e9 + float(c3.get("pnl", 0.0))
    return float(c3["pnl"]) + 2.0 * float(c3["mdd"]) + 40.0 * float(c3["wr"]) - 0.03 * trades


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_01543 = _load_01543_config()
    stack = loop._load_stack()
    train_df, val_df, eval_df = _load_train_val_eval_frames()

    a7_primary = joblib.load(loop.ALPHA7_LIVE_DIR / "primary_parent.pkl")
    fb0_parent = joblib.load(FALLBACK_PARENT)
    fb1_parent = joblib.load(S1_PARENT)
    loop._assert_parent_contract(a7_primary, train_df, name="alpha7_primary")
    loop._assert_parent_contract(fb0_parent, train_df, name="alpha7_fallback_alpha43")
    loop._assert_parent_contract(fb1_parent, train_df, name="alpha7_high_turnover_s1")

    p_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fb0_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    fb1_rt = _extract_runtime(S1_CAND_SUMMARY)

    p_train0 = _predict_scaled(a7_primary, train_df, p_rt)
    p_val0 = _predict_scaled(a7_primary, val_df, p_rt)
    p_eval0 = _predict_scaled(a7_primary, eval_df, p_rt)
    p_train = loop._apply_decision_mods(p_train0, cfg_01543)
    p_val = loop._apply_decision_mods(p_val0, cfg_01543)
    p_eval = loop._apply_decision_mods(p_eval0, cfg_01543)

    fb0_train = _predict_scaled(fb0_parent, train_df, fb0_rt)
    fb0_val = _predict_scaled(fb0_parent, val_df, fb0_rt)
    fb0_eval = _predict_scaled(fb0_parent, eval_df, fb0_rt)
    fb1_train = _predict_scaled(fb1_parent, train_df, fb1_rt)
    fb1_val = _predict_scaled(fb1_parent, val_df, fb1_rt)
    fb1_eval = _predict_scaled(fb1_parent, eval_df, fb1_rt)

    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    baseline_train = _backtest(train_df, p_train, stack=stack, q=train_q, cfg_01543=cfg_01543)
    baseline_val = _backtest(val_df, p_val, stack=stack, q=val_q, cfg_01543=cfg_01543)
    baseline_eval = _backtest(eval_df, p_eval, stack=stack, q=eval_q, cfg_01543=cfg_01543)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: HistGradientBoostingClassifier | None = None
    best_payload: dict[str, Any] | None = None
    for non_skip_penalty, min_edge in (
        (0.0004, 0.0),
        (0.0008, 0.0004),
        (0.0012, 0.0008),
        (0.0020, 0.0015),
    ):
            x_train, y_train, w_train, train_cash_idx, label_meta = _build_cash_supervised(
                train_df,
                p_train,
                fb0_train,
                fb1_train,
                fee=float(stack["fee"]),
                slip=float(stack["slip"]),
                non_skip_penalty=float(non_skip_penalty),
                min_edge=float(min_edge),
            )
            model = HistGradientBoostingClassifier(
                max_iter=360,
                learning_rate=0.035,
                max_leaf_nodes=15,
                l2_regularization=0.15,
                early_stopping=False,
                random_state=270527,
            )
            model.fit(x_train, y_train, sample_weight=w_train)
            x_val_all = _state_matrix(val_df, p_val, fb0_val, fb1_val)
            x_eval_all = _state_matrix(eval_df, p_eval, fb0_eval, fb1_eval)
            for margin in (0.00, 0.04, 0.08, 0.14):
                for prob_floor in (0.44, 0.52, 0.62):
                    act_val = _predict_actions(model, x_val_all, margin=margin, prob_floor=prob_floor)
                    act_eval = _predict_actions(model, x_eval_all, margin=margin, prob_floor=prob_floor)
                    dec_val = _compose(p_val, fb0_val, fb1_val, act_val)
                    dec_eval = _compose(p_eval, fb0_eval, fb1_eval, act_eval)
                    m_val = _backtest(val_df, dec_val, stack=stack, q=val_q, cfg_01543=cfg_01543)
                    m_eval = _backtest(eval_df, dec_eval, stack=stack, q=eval_q, cfg_01543=cfg_01543)
                    row = {
                        "non_skip_penalty": float(non_skip_penalty),
                        "min_edge": float(min_edge),
                        "margin": float(margin),
                        "prob_floor": float(prob_floor),
                        "selection_score": float(_selection_score(m_val)),
                        "val_pnl": float(m_val["pnl"]),
                        "val_mdd": float(m_val["mdd"]),
                        "val_wr": float(m_val["wr"]),
                        "val_trades": int(m_val["trades"]),
                        "oos_pnl": float(m_eval["pnl"]),
                        "oos_mdd": float(m_eval["mdd"]),
                        "oos_wr": float(m_eval["wr"]),
                        "oos_trades": int(m_eval["trades"]),
                        "val_added_fb0": int(np.sum((act_val == ACTION_FB0) & (~_active(p_val)))),
                        "val_added_fb1": int(np.sum((act_val == ACTION_FB1) & (~_active(p_val)))),
                        "oos_added_fb0": int(np.sum((act_eval == ACTION_FB0) & (~_active(p_eval)))),
                        "oos_added_fb1": int(np.sum((act_eval == ACTION_FB1) & (~_active(p_eval)))),
                        "label_meta": label_meta,
                    }
                    rows.append(row)
                    if best is None or row["selection_score"] > best["selection_score"]:
                        best = row
                        best_model = model
                        best_payload = {
                            "x_train_shape": list(x_train.shape),
                            "train_cash_rows": int(len(train_cash_idx)),
                            "label_meta": label_meta,
                        }
    assert best is not None and best_model is not None and best_payload is not None

    x_train_all = _state_matrix(train_df, p_train, fb0_train, fb1_train)
    x_val_all = _state_matrix(val_df, p_val, fb0_val, fb1_val)
    x_eval_all = _state_matrix(eval_df, p_eval, fb0_eval, fb1_eval)
    act_train = _predict_actions(best_model, x_train_all, margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))
    act_val = _predict_actions(best_model, x_val_all, margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))
    act_eval = _predict_actions(best_model, x_eval_all, margin=float(best["margin"]), prob_floor=float(best["prob_floor"]))
    final_train = _compose(p_train, fb0_train, fb1_train, act_train)
    final_val = _compose(p_val, fb0_val, fb1_val, act_val)
    final_eval = _compose(p_eval, fb0_eval, fb1_eval, act_eval)
    train_metrics = _backtest(train_df, final_train, stack=stack, q=train_q, cfg_01543=cfg_01543)
    val_metrics = _backtest(val_df, final_val, stack=stack, q=val_q, cfg_01543=cfg_01543)
    eval_metrics = _backtest(eval_df, final_eval, stack=stack, q=eval_q, cfg_01543=cfg_01543)

    model_path = OUT_DIR / "cash_fallback_selector.pkl"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "primary_config": cfg_01543,
            "selector": best_model,
            "best": best,
            "candidate_actions": {"skip": ACTION_SKIP, "fallback_alpha43": ACTION_FB0, "fallback_high_turnover_s1": ACTION_FB1},
            **best_payload,
        },
        model_path,
    )
    pd.DataFrame(rows).sort_values(["selection_score", "oos_pnl"], ascending=[False, False]).to_csv(OUT_DIR / "grid.csv", index=False)
    final_val.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    final_eval.to_csv(OUT_DIR / "oos_decisions.csv", index=False)
    summary = {
        "model_id": MODEL_ID,
        "design": "01543_random_alpha7_primary as primary, then cash-region-only HGB selector retrained on 2025 pre-Q4 primary-CASH rows to choose skip/current alpha43 fallback/high-turnover-s1 fallback.",
        "primary": {
            "name": cfg_01543["name"],
            "config": cfg_01543,
        },
        "fallback_candidates": ["alpha7_current_alpha43_fallback", "alpha7_high_turnover_s1_primary"],
        "best": best,
        "training": best_payload,
        "baseline_01543_primary_only": {
            "train": baseline_train,
            "val": baseline_val,
            "oos": baseline_eval,
        },
        "with_retrained_cash_fallback": {
            "train": train_metrics,
            "val": val_metrics,
            "oos": eval_metrics,
        },
        "delta_vs_01543_primary_only": {
            "val_pnl": float(val_metrics["pnl"] - baseline_val["pnl"]),
            "val_trades": int(val_metrics["trades"] - baseline_val["trades"]),
            "oos_pnl": float(eval_metrics["pnl"] - baseline_eval["pnl"]),
            "oos_trades": int(eval_metrics["trades"] - baseline_eval["trades"]),
        },
        "artifacts": {
            "model": model_path,
            "grid": OUT_DIR / "grid.csv",
            "validation_decisions": OUT_DIR / "validation_decisions.csv",
            "oos_decisions": OUT_DIR / "oos_decisions.csv",
        },
        "audit": {
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed",
            "oos_used_for_selection": False,
            "primary_cash_only_training": True,
            "compat_alias_added": False,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(OUT_DIR / "summary.json"),
                "baseline_oos_pnl": baseline_eval["pnl"],
                "fallback_oos_pnl": eval_metrics["pnl"],
                "delta_oos_pnl": eval_metrics["pnl"] - baseline_eval["pnl"],
                "fallback_oos_trades": eval_metrics["trades"],
                "best": {k: best[k] for k in ("non_skip_penalty", "min_edge", "margin", "prob_floor", "selection_score")},
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
