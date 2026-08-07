#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import (
    OUT_DIR as MAX_TP_SL_OUT_DIR,
    ROUTER_NAME,
    _assert_no_forbidden,
    _load_frames_max,
    _max_feature_cols,
)
from scripts.train_alpha7_regime3_expert_moe_20260601 import (
    EXPERT_NAMES,
    _active,
    _flatten,
    _route_conf,
    _route_id,
    _score,
)


MODEL_ID = "alpha7_active_max_feature_zigzag_moe_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_20260601"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
LABEL_2025 = LABEL_DIR / "zigzag_action_labels_2025.csv"

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


RISK_TEMPLATES: dict[str, dict[str, float | int]] = {
    "scalp": {"notional": 0.35, "leverage": 3.0, "take_profit": 0.018, "stop_loss": 0.010, "max_hold": 48, "cooldown": 3},
    "mid": {"notional": 0.55, "leverage": 3.0, "take_profit": 0.030, "stop_loss": 0.016, "max_hold": 96, "cooldown": 6},
    "swing": {"notional": 0.80, "leverage": 3.0, "take_profit": 0.050, "stop_loss": 0.024, "max_hold": 288, "cooldown": 12},
    "wide": {"notional": 0.80, "leverage": 2.0, "take_profit": 0.090, "stop_loss": 0.035, "max_hold": 288, "cooldown": 12},
}


def _read_zigzag_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    if df["timestamp"].duplicated().any():
        dup = df.loc[df["timestamp"].duplicated(), "timestamp"].head(10).tolist()
        raise RuntimeError(f"duplicate zigzag label timestamps: {dup}")
    invalid = sorted(set(pd.to_numeric(df["zigzag_action"], errors="raise").astype(int)) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"invalid zigzag_action classes: {invalid}")
    return df.sort_values("timestamp").reset_index(drop=True)


def _attach_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    before = len(frame)
    out = frame.merge(labels, on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError("zigzag label join changed row count")
    missing = out["zigzag_action"].isna()
    if bool(missing.any()):
        bad = out.loc[missing, "timestamp"]
        raise RuntimeError(f"missing zigzag labels for frame timestamps: {bad.head(20).tolist()}")
    out["zigzag_action"] = out["zigzag_action"].astype(np.int64)
    return out


def _class_weights(y: np.ndarray) -> list[float]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=3).astype(np.float64)
    if np.any(counts <= 0):
        raise RuntimeError(f"zigzag train split missing class: {counts.tolist()}")
    inv = np.sqrt(np.max(counts) / counts)
    return [float(x) for x in inv]


def _fit_action_model(train_df: pd.DataFrame, feature_cols: list[str], *, seed: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "zigzag_action.cbm"
    summary_path = out_dir / "summary.json"
    if model_path.exists() and summary_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return {"model": model, "summary": json.loads(summary_path.read_text(encoding="utf-8"))}
    fit = train_df[train_df["timestamp"] < SPLIT_TS].reset_index(drop=True)
    if len(fit) < 2000:
        raise RuntimeError(f"too few rows for zigzag action model: {len(fit)}")
    x = _feature_frame(fit, feature_cols)
    y = fit["zigzag_action"].to_numpy(dtype=np.int64)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=650,
        depth=6,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        random_seed=int(seed),
        class_weights=_class_weights(y),
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    model.fit(x, y)
    model.save_model(str(model_path))
    pred = model.predict(x).reshape(-1).astype(np.int64)
    summary = {
        "model_id": MODEL_ID,
        "label_source": str(LABEL_2025),
        "label_column": "zigzag_action",
        "feature_count": len(feature_cols),
        "train_rows": int(len(fit)),
        "label_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "pred_counts": {str(k): int(v) for k, v in pd.Series(pred).value_counts().sort_index().to_dict().items()},
        "class_weights": _class_weights(y),
        "model_path": str(model_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return {"model": model, "summary": summary}


def _feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    allowed = {"side_hint"}
    unexpected = [c for c in missing if c not in allowed]
    if unexpected:
        raise RuntimeError(f"missing non-derivable feature columns: {unexpected[:20]}")
    out = df.copy()
    if "side_hint" in missing:
        out["side_hint"] = 0.0
    return out[feature_cols]


def _predict_action_decision(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_conf: float,
    min_edge: float,
    template: dict[str, float | int],
) -> pd.DataFrame:
    model: CatBoostClassifier = bundle["model"]
    proba = np.asarray(model.predict_proba(_feature_frame(df, feature_cols)), dtype=np.float64)
    if proba.shape[1] != 3:
        raise RuntimeError(f"expected 3 zigzag classes, got {proba.shape}")
    active_prob = np.maximum(proba[:, ACTION_LONG], proba[:, ACTION_SHORT])
    side = np.where(proba[:, ACTION_LONG] >= proba[:, ACTION_SHORT], 1, -1).astype(np.int64)
    conf = np.max(proba, axis=1)
    edge = active_prob - proba[:, ACTION_CASH]
    active = (active_prob >= float(min_conf)) & (edge >= float(min_edge))
    action = np.where(active, np.where(side > 0, ACTION_LONG, ACTION_SHORT), ACTION_CASH).astype(np.int64)
    side = np.where(active, side, 0).astype(np.int64)
    notional = float(template["notional"])
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, notional, 0.0),
            "leverage": np.where(active, float(template["leverage"]), 1.0),
            "position_fraction": np.where(active, notional, 0.0),
            "take_profit": np.where(active, float(template["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(template["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(template["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(template["cooldown"]), 0).astype(np.int64),
            "quality_score": active_prob.astype(np.float64),
            "confidence": conf.astype(np.float64),
        }
    )
    return out


def _combine_primary_fallback_zigzag(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    mask = (~_active(out)) & _active(fallback)
    for col in fallback.columns:
        out.loc[mask, col] = fallback.loc[mask, col].to_numpy()
    return out


def _side_constrained(dec: pd.DataFrame, *, expert: str) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    if expert == "bull":
        block = active & (side < 0)
    elif expert == "bear":
        block = active & (side > 0)
    else:
        block = np.zeros(len(out), dtype=bool)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


def _route_decision(expert_dec: dict[str, pd.DataFrame], base_dec: pd.DataFrame, route: np.ndarray, conf: np.ndarray, *, route_min_conf: float) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    cols = list(base_dec.columns)
    selected = route.copy()
    selected[conf < float(route_min_conf)] = 3
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = selected == idx
        out.loc[mask, cols] = expert_dec[expert].loc[mask, cols].to_numpy()
    out["router_expert"] = np.where(selected == 0, "bull", np.where(selected == 1, "bear", np.where(selected == 2, "chop_expert", "lowconf_baseline")))
    out["router_confidence"] = conf
    out["router_min_conf"] = float(route_min_conf)
    return out


def _train_pair(name: str, train_all: pd.DataFrame, feature_cols: list[str], seed: int) -> dict[str, Any]:
    primary = _fit_action_model(train_all, feature_cols, seed=seed, out_dir=OUT_DIR / name / "primary_zigzag")
    fallback = _fit_action_model(train_all, feature_cols, seed=seed + 1, out_dir=OUT_DIR / name / "fallback_zigzag")
    return {"primary": primary, "fallback": fallback}


def _predict_pair(
    pair: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    primary_conf: float,
    fallback_conf: float,
    min_edge: float,
    template: dict[str, float | int],
) -> pd.DataFrame:
    primary_dec = _predict_action_decision(pair["primary"], df, feature_cols, min_conf=primary_conf, min_edge=min_edge, template=template)
    fallback_dec = _predict_action_decision(pair["fallback"], df, feature_cols, min_conf=fallback_conf, min_edge=min_edge, template=template)
    return _combine_primary_fallback_zigzag(primary_dec, fallback_dec)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_max()
    labels = _read_zigzag_labels(LABEL_2025)
    train_all = _attach_labels(train_all, labels)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    feature_cols = _max_feature_cols(train_all, eval_df)
    _assert_no_forbidden(feature_cols)
    if "zigzag_action" in feature_cols:
        raise RuntimeError("zigzag label leaked into feature columns")

    base = _train_pair("baseline_max", train_all, feature_cols, seed=6064100)
    route_train = _route_id(train_all, ROUTER_NAME)
    experts: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(EXPERT_NAMES):
        mask = route_train == idx
        expert_train = train_all.loc[mask].reset_index(drop=True)
        experts[expert] = _train_pair(f"{expert}_max", expert_train, feature_cols, seed=6064200 + idx * 10)
        summaries[expert] = {
            "rows": int(mask.sum()),
            "primary": experts[expert]["primary"]["summary"],
            "fallback": experts[expert]["fallback"]["summary"],
        }

    val_route = _route_id(val_df, ROUTER_NAME)
    oos_route = _route_id(eval_df, ROUTER_NAME)
    val_route_conf = _route_conf(val_df, ROUTER_NAME)
    oos_route_conf = _route_conf(eval_df, ROUTER_NAME)

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for template_name, template in {k: RISK_TEMPLATES[k] for k in ["mid", "swing"]}.items():
        for primary_conf, fallback_conf, min_edge, route_min_conf in itertools.product(
            [0.55, 0.65],
            [0.50],
            [0.08],
            [0.80],
        ):
            baseline_val = _predict_pair(
                base,
                val_df,
                feature_cols,
                primary_conf=primary_conf,
                fallback_conf=fallback_conf,
                min_edge=min_edge,
                template=template,
            )
            baseline_oos = _predict_pair(
                base,
                eval_df,
                feature_cols,
                primary_conf=primary_conf,
                fallback_conf=fallback_conf,
                min_edge=min_edge,
                template=template,
            )
            val_experts: dict[str, pd.DataFrame] = {}
            oos_experts: dict[str, pd.DataFrame] = {}
            for expert, pair in experts.items():
                val_experts[expert] = _side_constrained(
                    _predict_pair(
                        pair,
                        val_df,
                        feature_cols,
                        primary_conf=primary_conf,
                        fallback_conf=fallback_conf,
                        min_edge=min_edge,
                        template=template,
                    ),
                    expert=expert,
                )
                oos_experts[expert] = _side_constrained(
                    _predict_pair(
                        pair,
                        eval_df,
                        feature_cols,
                        primary_conf=primary_conf,
                        fallback_conf=fallback_conf,
                        min_edge=min_edge,
                        template=template,
                    ),
                    expert=expert,
                )
            routed_val = _route_decision(val_experts, baseline_val, val_route, val_route_conf, route_min_conf=route_min_conf)
            routed_oos = _route_decision(oos_experts, baseline_oos, oos_route, oos_route_conf, route_min_conf=route_min_conf)
            for bull, bear, chop in itertools.product([0.85, 1.00], [1.00, 1.15], [0.90, 1.10]):
                val_dec = _apply_scale(routed_val, bull=bull, bear=bear, chop=chop)
                val_costs = _combo_metrics(val_df, val_dec)
                key = f"{template_name}_pc{primary_conf:.2f}_fc{fallback_conf:.2f}_edge{min_edge:.2f}_rc{route_min_conf:.2f}_b{bull:.2f}_r{bear:.2f}_c{chop:.2f}"
                row = {
                    "candidate": key,
                    "template": template_name,
                    "primary_conf": float(primary_conf),
                    "fallback_conf": float(fallback_conf),
                    "min_edge": float(min_edge),
                    "route_min_conf": float(route_min_conf),
                    "bull_scale": float(bull),
                    "bear_scale": float(bear),
                    "chop_scale": float(chop),
                    "score": float(_score(val_costs)),
                    "validation": val_costs,
                    "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                }
                rows.append(row)
                if len(rows) <= 20000:
                    payload[key] = (val_dec, _apply_scale(routed_oos, bull=bull, bear=bear, chop=chop))

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    if selected["candidate"] not in payload:
        raise RuntimeError("selected candidate was not retained in payload")
    selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    oos_costs = _combo_metrics(eval_df, selected_oos_dec)
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": r["candidate"],
                "template": r["template"],
                "primary_conf": r["primary_conf"],
                "fallback_conf": r["fallback_conf"],
                "min_edge": r["min_edge"],
                "route_min_conf": r["route_min_conf"],
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
        "design": "Max-feature current-Regime3 MoE with all supervised action heads trained from zigzag_action only. Risk sizing/TP/SL are non-supervised execution templates selected on validation.",
        "label_contract": {
            "label_path": str(LABEL_2025),
            "label_column": "zigzag_action",
            "classes": {"0": "cash", "1": "long", "2": "short"},
            "no_tp_sl_path_labels": True,
            "no_zigzag_feature_input": True,
        },
        "feature_contract": {
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
        },
        "overlay": overlay,
        "risk_templates": RISK_TEMPLATES,
        "summaries": {"baseline": {"primary": base["primary"]["summary"], "fallback": base["fallback"]["summary"]}, "experts": summaries},
        "selected": {
            **{k: selected[k] for k in ["candidate", "template", "primary_conf", "fallback_conf", "min_edge", "route_min_conf", "bull_scale", "bear_scale", "chop_scale"]},
            "validation": selected["validation"],
            "oos": oos_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()},
        },
        "top_grid": rows[:20],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking_validation_only": str(OUT_DIR / "ranking_validation_only.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
            "previous_tp_sl_max_report": str(MAX_TP_SL_OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"], "feature_count": len(feature_cols)}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
