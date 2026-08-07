#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_distilled_tree_parent_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_distilled_tree_parent_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_distilled_tree_parent_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_distilled_tree_parent_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_distilled_tree_parent_20260513_grid.csv"


@dataclass(frozen=True)
class Runtime:
    name: str
    model_key: str
    confidence: float
    require_same_side: bool
    veto_only: bool


def _runtime_grid() -> list[Runtime]:
    rows: list[Runtime] = []
    for key in ("lgbm_quant", "lgbm_conservative", "catboost"):
        if key == "catboost" and CatBoostClassifier is None:
            continue
        for conf in (0.34, 0.40, 0.46, 0.52, 0.58):
            rows.append(Runtime(f"{key}_veto_same_c{conf:.2f}", key, conf, True, True))
            rows.append(Runtime(f"{key}_veto_any_active_c{conf:.2f}", key, conf, False, True))
    return rows


def _fit_models(x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    params_common = dict(
        objective="multiclass",
        num_class=3,
        n_estimators=260,
        learning_rate=0.035,
        max_depth=-1,
        num_leaves=47,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.03,
        reg_lambda=0.16,
        random_state=20260513,
        n_jobs=-1,
        verbose=-1,
    )
    models: dict[str, Any] = {
        "lgbm_quant": LGBMClassifier(**params_common, path_smooth=4.0, extra_trees=True),
        "lgbm_conservative": LGBMClassifier(**{**params_common, "n_estimators": 180, "num_leaves": 31, "min_child_samples": 260, "path_smooth": 8.0, "extra_trees": True}),
    }
    if CatBoostClassifier is not None:
        models["catboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostClassifier(
                loss_function="MultiClass",
                iterations=260,
                learning_rate=0.035,
                depth=6,
                l2_leaf_reg=8.0,
                random_seed=20260513,
                verbose=False,
                allow_writing_files=False,
            ),
        )
    fitted: dict[str, Any] = {}
    for key, model in models.items():
        print(f"[{MODEL_ID}] fitting {key}", flush=True)
        if key.startswith("catboost"):
            model.fit(x, y, catboostclassifier__sample_weight=sample_weight)
        else:
            model.fit(x, y, sample_weight=sample_weight)
        fitted[key] = model
    return fitted


def _predict_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    if p.shape[1] == 3:
        return p
    out = np.zeros((len(x), 3), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", getattr(model[-1], "classes_", [0, 1, 2])), dtype=int)
    for j, cls in enumerate(classes):
        out[:, int(cls)] = p[:, j]
    return out


def _tree_decisions(teacher: pd.DataFrame, proba: np.ndarray, rt: Runtime) -> pd.DataFrame:
    out = teacher.copy()
    teacher_side = out["side"].astype(int).to_numpy()
    teacher_action = out["action"].astype(int).to_numpy()
    teacher_active = (teacher_action != ACTION_CASH) & (teacher_side != 0)
    pred_action = np.argmax(proba, axis=1).astype(np.int64)
    pred_conf = np.max(proba, axis=1)
    pred_side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0))
    keep = teacher_active & (pred_conf >= float(rt.confidence)) & (pred_action != ACTION_CASH)
    if rt.require_same_side:
        keep &= pred_side == teacher_side
    # Conservative parent swap: do not create new parent entries where original parent was CASH.
    out.loc[~keep, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[~keep, "leverage"] = 1.0
    out.loc[:, "tree_parent_confidence"] = pred_conf
    out.loc[:, "tree_parent_action"] = pred_action
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.20 * c3["pnl"] - 0.20 * abs(c1["mdd"]))


def _metrics(df: pd.DataFrame, q: np.ndarray, decisions: pd.DataFrame, parent: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, base: dict[str, Any]) -> dict[str, Any]:
    return {
        f"cost{mult}": alpha1.backtest_alpha1(df, parent, jackpot_model, add_cfg, q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=decisions)
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading alpha1 artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    x_train = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    x_val = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    x_eval = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    y = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    active = y != ACTION_CASH
    weights = np.where(active, 2.8, 0.55).astype(np.float64)
    q = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    weights += np.where(active, np.clip(q, 0.0, 2.0) * 0.35, 0.0)
    models = _fit_models(x_train, y, weights)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_proba = {k: _predict_proba(m, x_val) for k, m in models.items()}
    eval_proba = {k: _predict_proba(m, x_eval) for k, m in models.items()}
    rows: list[dict[str, Any]] = []
    selected: Runtime | None = None
    best_score = -1e18
    for rt in _runtime_grid():
        dec = _tree_decisions(val_dec, val_proba[rt.model_key], rt)
        vm = _metrics(val, val_q, dec, parent, jackpot_model, add_cfg, base)
        score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
        row = {**asdict(rt), "score": score, "val_pnl": vm["cost1"]["pnl"], "val_mdd": vm["cost1"]["mdd"], "val_trades": vm["cost1"]["trades"], "val_deep_entries": vm["cost1"]["deep_entries"], "val_c2_pnl": vm["cost2"]["pnl"], "val_c3_pnl": vm["cost3"]["pnl"]}
        rows.append(row)
        if score > best_score:
            best_score = score
            selected = rt
    assert selected is not None
    experiments = []
    for name, dec in (
        ("alpha1", eval_dec),
        (f"distilled_tree::{selected.name}", _tree_decisions(eval_dec, eval_proba[selected.model_key], selected)),
    ):
        metrics = _metrics(eval_df, eval_q, dec, parent, jackpot_model, add_cfg, base)
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "distilled_tree_parent.pkl"
    joblib.dump({"model_id": MODEL_ID, "models": models, "feature_cols": feature_cols, "selected_config": asdict(selected), "design": "Tree-distilled conservative parent veto. Original parent CASH remains CASH; V27 scout opportunity is preserved. Tree can only keep or veto original parent active entries."}, model_path)
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    best = max(experiments, key=lambda e: e["score"])
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    if best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("distilled_tree_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] < alpha1.ALPHA1_BASELINE["cost2"]["pnl"]:
        warnings.append("distilled_tree_cost2_below_alpha1")
    if best["metrics"]["cost3"]["pnl"] < alpha1.ALPHA1_BASELINE["cost3"]["pnl"]:
        warnings.append("distilled_tree_cost3_below_alpha1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] and best["metrics"]["cost2"]["pnl"] >= alpha1.ALPHA1_BASELINE["cost2"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "cash_preserving": True,
        "new_parent_entries_allowed_in_teacher_cash": False,
        "v27_deep_scout_preserved": True,
        "selected_config": asdict(selected),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "LightGBM/CatBoost distilled parent veto over alpha1 parent active decisions. It does not create new parent entries and keeps parent CASH bars available for V27 deep scout.",
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
