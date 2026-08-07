#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.eval_alpha7_zigzag_moe_risk_param_sweep_20260601 import (
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    RISK_TEMPLATES,
    _candidate_decisions,
    _load_feature_cols,
    _models,
    _probas,
)
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import ROUTER_NAME, _load_frames_max
from scripts.retrain_alpha7_active_max_feature_zigzag_moe_20260601 import _feature_frame
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _route_conf, _route_id, _score


MODEL_ID = "omega1_zigzag_direction_sl_tp_entry_quality_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_zigzag_direction_sl_tp_entry_quality_20260601"
QUALITY_COLS = [
    "candidate_side",
    "candidate_notional",
    "candidate_leverage",
    "candidate_take_profit",
    "candidate_stop_loss",
    "candidate_max_hold",
    "candidate_cooldown",
]


@njit(parallel=True)
def _tp_first_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    row_idx: np.ndarray,
    sides: np.ndarray,
    notionals: np.ndarray,
    tps: np.ndarray,
    sls: np.ndarray,
    holds: np.ndarray,
) -> np.ndarray:
    n = len(row_idx)
    out = np.zeros(n, dtype=np.int64)
    m = len(close)
    for r in prange(n):
        i = int(row_idx[r])
        side = int(sides[r])
        notional = float(notionals[r])
        tp = float(tps[r])
        sl = float(sls[r])
        hold = int(holds[r])
        if i + 1 >= m or hold <= 0 or side == 0:
            continue
        entry = float(close[i])
        if entry <= 0.0:
            continue
        end = min(m, i + 1 + hold)
        for j in range(i + 1, end):
            if side > 0:
                fav = (float(high[j]) / entry - 1.0) * notional
                adv = (float(low[j]) / entry - 1.0) * notional
            else:
                fav = (1.0 - float(low[j]) / entry) * notional
                adv = (1.0 - float(high[j]) / entry) * notional
            tp_hit = fav >= tp
            sl_hit = adv <= -sl
            if tp_hit and not sl_hit:
                out[r] = 1
                break
            if sl_hit:
                break
    return out


def _quality_training_frame(df: pd.DataFrame, feature_cols: list[str], *, max_rows: int = 280_000) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    fit = df[df["timestamp"] < SPLIT_TS - pd.Timedelta(minutes=5 * 288)].reset_index(drop=True)
    if len(fit) < 20_000:
        raise RuntimeError(f"too few quality fit rows: {len(fit)}")

    rows: list[np.ndarray] = []
    sides: list[np.ndarray] = []
    t_names: list[str] = []
    for name, template in RISK_TEMPLATES.items():
        for side in (1, -1):
            rows.append(np.arange(len(fit), dtype=np.int64))
            sides.append(np.full(len(fit), side, dtype=np.int64))
            t_names.extend([name] * len(fit))
    row_idx = np.concatenate(rows)
    side_arr = np.concatenate(sides)
    template_names = np.asarray(t_names, dtype=object)
    notional = np.asarray([float(RISK_TEMPLATES[str(t)]["notional"]) for t in template_names], dtype=np.float64)
    leverage = np.asarray([float(RISK_TEMPLATES[str(t)]["leverage"]) for t in template_names], dtype=np.float64)
    tp = np.asarray([float(RISK_TEMPLATES[str(t)]["take_profit"]) for t in template_names], dtype=np.float64)
    sl = np.asarray([float(RISK_TEMPLATES[str(t)]["stop_loss"]) for t in template_names], dtype=np.float64)
    hold = np.asarray([int(RISK_TEMPLATES[str(t)]["max_hold"]) for t in template_names], dtype=np.int64)
    cooldown = np.asarray([int(RISK_TEMPLATES[str(t)]["cooldown"]) for t in template_names], dtype=np.float64)

    y = _tp_first_labels(
        fit["close"].to_numpy(dtype=np.float64),
        fit["high"].to_numpy(dtype=np.float64),
        fit["low"].to_numpy(dtype=np.float64),
        row_idx,
        side_arr,
        notional,
        tp,
        sl,
        hold,
    )

    rng = np.random.default_rng(6061101)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0:
        raise RuntimeError("entry quality labels have no positive examples")
    n_pos = min(len(pos), max_rows // 2)
    n_neg = min(len(neg), max_rows - n_pos)
    keep = np.concatenate([rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(keep)

    base_x = _feature_frame(fit, feature_cols).iloc[row_idx[keep]].reset_index(drop=True)
    x = base_x.copy()
    x["candidate_side"] = side_arr[keep].astype(np.float64)
    x["candidate_notional"] = notional[keep]
    x["candidate_leverage"] = leverage[keep]
    x["candidate_take_profit"] = tp[keep]
    x["candidate_stop_loss"] = sl[keep]
    x["candidate_max_hold"] = hold[keep].astype(np.float64)
    x["candidate_cooldown"] = cooldown[keep]
    meta = {
        "fit_rows": int(len(fit)),
        "expanded_rows": int(len(y)),
        "sampled_rows": int(len(keep)),
        "positive_rate_full": float(np.mean(y)),
        "positive_rate_sample": float(np.mean(y[keep])),
        "sample_positive": int(np.sum(y[keep] == 1)),
        "sample_negative": int(np.sum(y[keep] == 0)),
    }
    return x, y[keep].astype(np.int64), meta


def _load_or_train_quality(train_all: pd.DataFrame, feature_cols: list[str]) -> tuple[CatBoostClassifier, dict[str, Any]]:
    model_path = OUT_DIR / "entry_quality_sl_tp_compat.cbm"
    summary_path = OUT_DIR / "entry_quality_summary.json"
    if model_path.exists() and summary_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model, json.loads(summary_path.read_text(encoding="utf-8"))
    x, y, meta = _quality_training_frame(train_all, feature_cols)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=550,
        depth=6,
        learning_rate=0.045,
        l2_leaf_reg=10.0,
        random_seed=6061102,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=False,
        thread_count=6,
    )
    model.fit(x, y)
    pred = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    summary = {
        "model_id": MODEL_ID,
        "label_role": "entry_quality_sl_tp_compatibility",
        "not_action_label": True,
        "target_definition": "1 when the candidate side/template reaches template TP before template SL inside max_hold; ambiguous same-bar TP/SL is treated as failure.",
        "feature_count": int(x.shape[1]),
        "feature_cols": list(x.columns),
        "label_meta": meta,
        "train_prob_mean": float(np.mean(pred)),
        "train_prob_p50": float(np.quantile(pred, 0.50)),
        "train_prob_p90": float(np.quantile(pred, 0.90)),
        "model_path": str(model_path),
    }
    model.save_model(str(model_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return model, summary


def _quality_frame(df: pd.DataFrame, feature_cols: list[str], dec: pd.DataFrame) -> pd.DataFrame:
    x = _feature_frame(df, feature_cols).reset_index(drop=True)
    out = x.copy()
    out["candidate_side"] = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_notional"] = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_leverage"] = pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_take_profit"] = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_stop_loss"] = pd.to_numeric(dec["stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_max_hold"] = pd.to_numeric(dec["max_hold_bars"], errors="raise").to_numpy(dtype=np.float64)
    out["candidate_cooldown"] = pd.to_numeric(dec["cooldown_bars"], errors="raise").to_numpy(dtype=np.float64)
    return out


def _apply_entry_quality(model: CatBoostClassifier, df: pd.DataFrame, feature_cols: list[str], dec: pd.DataFrame, *, min_quality: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = np.asarray(_active(out), dtype=bool)
    q = np.zeros(len(out), dtype=np.float64)
    if np.any(active):
        x = _quality_frame(df, feature_cols, out)
        q = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        block = active & (q < float(min_quality))
        out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
        out.loc[block, "leverage"] = 1.0
    out["entry_quality_prob"] = q
    out["entry_quality_min"] = float(min_quality)
    return out


def _template_quality_probs(model: CatBoostClassifier, df: pd.DataFrame, feature_cols: list[str], template: dict[str, float | int]) -> dict[int, np.ndarray]:
    base = _feature_frame(df, feature_cols).reset_index(drop=True)
    out: dict[int, np.ndarray] = {}
    for side in (1, -1):
        x = base.copy()
        x["candidate_side"] = float(side)
        x["candidate_notional"] = float(template["notional"])
        x["candidate_leverage"] = float(template["leverage"])
        x["candidate_take_profit"] = float(template["take_profit"])
        x["candidate_stop_loss"] = float(template["stop_loss"])
        x["candidate_max_hold"] = float(template["max_hold"])
        x["candidate_cooldown"] = float(template["cooldown"])
        out[side] = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return out


def _apply_entry_quality_cached(dec: pd.DataFrame, quality_probs: dict[int, np.ndarray], *, min_quality: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = np.asarray(_active(out), dtype=bool)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    q = np.zeros(len(out), dtype=np.float64)
    long_mask = side > 0
    short_mask = side < 0
    q[long_mask] = quality_probs[1][long_mask]
    q[short_mask] = quality_probs[-1][short_mask]
    block = active & (q < float(min_quality))
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[block, "leverage"] = 1.0
    out["entry_quality_prob"] = q
    out["entry_quality_min"] = float(min_quality)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_max()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    feature_cols = _load_feature_cols()
    quality_model, quality_summary = _load_or_train_quality(train_all, feature_cols)
    models = _models()
    val_probas = _probas(models, val_df, feature_cols)
    oos_probas = _probas(models, eval_df, feature_cols)
    val_route_id = _route_id(val_df, ROUTER_NAME)
    oos_route_id = _route_id(eval_df, ROUTER_NAME)
    val_route_conf = _route_conf(val_df, ROUTER_NAME)
    oos_route_conf = _route_conf(eval_df, ROUTER_NAME)
    val_quality_probs = {name: _template_quality_probs(quality_model, val_df, feature_cols, template) for name, template in RISK_TEMPLATES.items()}
    oos_quality_probs = {name: _template_quality_probs(quality_model, eval_df, feature_cols, template) for name, template in RISK_TEMPLATES.items()}

    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    grids = itertools.product(
        ["scalp_rr20", "balanced_rr19", "mid_rr20", "safe_rr20", "trend_rr25"],
        [0.55, 0.65],
        [0.50],
        [0.08],
        [0.80],
        [0.45, 0.55, 0.65],
        [0.75, 0.90],
        [0.90, 1.05],
        [0.75, 0.90],
    )
    total = 5 * 2 * 1 * 1 * 3 * 2 * 2 * 2
    for seen, (template_name, primary_conf, fallback_conf, min_edge, route_min_conf, min_quality, bull, bear, chop) in enumerate(grids, start=1):
        template = RISK_TEMPLATES[template_name]
        routed = _candidate_decisions(
            val_probas,
            val_route_id,
            val_route_conf,
            template=template,
            primary_conf=primary_conf,
            fallback_conf=fallback_conf,
            min_edge=min_edge,
            route_min_conf=route_min_conf,
        )
        gated = _apply_entry_quality_cached(routed, val_quality_probs[template_name], min_quality=min_quality)
        val_dec = _apply_scale(gated, bull=bull, bear=bear, chop=chop)
        costs = _combo_metrics(val_df, val_dec)
        key = f"{template_name}_pc{primary_conf:.2f}_edge{min_edge:.2f}_rc{route_min_conf:.2f}_q{min_quality:.2f}_b{bull:.2f}_r{bear:.2f}_c{chop:.2f}"
        rows.append(
            {
                "candidate": key,
                "template": template_name,
                "primary_conf": float(primary_conf),
                "fallback_conf": float(fallback_conf),
                "min_edge": float(min_edge),
                "route_min_conf": float(route_min_conf),
                "entry_quality_min": float(min_quality),
                "bull_scale": float(bull),
                "bear_scale": float(bear),
                "chop_scale": float(chop),
                "score": float(_score(costs)),
                "validation": costs,
                "validation_policy_counts": {str(k): int(v) for k, v in val_dec.get("router_expert", pd.Series(dtype=object)).value_counts().to_dict().items()},
            }
        )
        if len(payload) < 50:
            payload[key] = val_dec
        if seen % 40 == 0:
            print(json.dumps({"progress": seen, "total": total, "best_score_so_far": max(float(r["score"]) for r in rows)}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    template = RISK_TEMPLATES[str(selected["template"])]
    oos_routed = _candidate_decisions(
        oos_probas,
        oos_route_id,
        oos_route_conf,
        template=template,
        primary_conf=float(selected["primary_conf"]),
        fallback_conf=float(selected["fallback_conf"]),
        min_edge=float(selected["min_edge"]),
        route_min_conf=float(selected["route_min_conf"]),
    )
    oos_gated = _apply_entry_quality_cached(oos_routed, oos_quality_probs[str(selected["template"])], min_quality=float(selected["entry_quality_min"]))
    selected_oos = _apply_scale(oos_gated, bull=float(selected["bull_scale"]), bear=float(selected["bear_scale"]), chop=float(selected["chop_scale"]))
    oos_costs = _combo_metrics(eval_df, selected_oos)
    selected_val = payload.get(str(selected["candidate"]))
    if selected_val is None:
        selected_val = _apply_scale(
            _apply_entry_quality_cached(
                _candidate_decisions(
                    val_probas,
                    val_route_id,
                    val_route_conf,
                    template=template,
                    primary_conf=float(selected["primary_conf"]),
                    fallback_conf=float(selected["fallback_conf"]),
                    min_edge=float(selected["min_edge"]),
                    route_min_conf=float(selected["route_min_conf"]),
                ),
                val_quality_probs[str(selected["template"])],
                min_quality=float(selected["entry_quality_min"]),
            ),
            bull=float(selected["bull_scale"]),
            bear=float(selected["bear_scale"]),
            chop=float(selected["chop_scale"]),
        )
    selected_val.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": r["candidate"],
                "template": r["template"],
                "primary_conf": r["primary_conf"],
                "fallback_conf": r["fallback_conf"],
                "min_edge": r["min_edge"],
                "route_min_conf": r["route_min_conf"],
                "entry_quality_min": r["entry_quality_min"],
                "bull_scale": r["bull_scale"],
                "bear_scale": r["bear_scale"],
                "chop_scale": r["chop_scale"],
                "score": r["score"],
                **_flatten("val", r["validation"]),
                "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            }
            for r in rows
        ]
    ).to_csv(OUT_DIR / "ranking_validation_only.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Omega1 split-head test: Direction/Action is loaded from ZigZag-only max-feature current-Regime3 MoE; Entry Quality is a separate SL/TP-compatibility binary CatBoost head trained only as an execution filter; Risk remains template/search based.",
        "label_contract": {
            "direction_action_label": "zigzag_action",
            "entry_quality_label": "template-specific TP-first-before-SL compatibility for candidate side; not used as direction/action target",
            "forbidden_as_action_labels": ["tp_sl_action_score", "wave3_action", "alpha_lifecycle", "fully_learned_governor_path_labels"],
        },
        "feature_count": len(feature_cols),
        "overlay": overlay,
        "quality_summary": quality_summary,
        "selected": {
            **{k: selected[k] for k in ["candidate", "template", "primary_conf", "fallback_conf", "min_edge", "route_min_conf", "entry_quality_min", "bull_scale", "bear_scale", "chop_scale"]},
            "validation": selected["validation"],
            "oos": oos_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val.get("router_expert", pd.Series(dtype=object)).value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos.get("router_expert", pd.Series(dtype=object)).value_counts().to_dict().items()},
        },
        "top_grid": rows[:30],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking_validation_only": str(OUT_DIR / "ranking_validation_only.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
            "entry_quality_model": str(OUT_DIR / "entry_quality_sl_tp_compat.cbm"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
