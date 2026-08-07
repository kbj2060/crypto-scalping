#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base_features  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega3_full_risk_distill_residual_regularized_loop17_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_BUNDLE = full_parent.PARENT_DIR / "true_3head_tabm_bundle.pt"
TRAIN_PRED_CACHE = ROOT / "tmp/causal_regen_20260516/omega3_entry_veto_exit_hazard_loop6_20260619/train_predictions_2025_true3head.csv"
PREFIX_TRAIN_VAL = "omega1_regime3_expertdq_oof_"
PREFIX_OOS = "omega1_regime3_expertdq_"
CURRENT = {
    "validation": {"pnl": 100.542729421, "mdd": -10.677653, "trades": 33, "wr": 0.636364},
    "oos": {"pnl": 72.760041481, "mdd": -8.108171, "trades": 18, "wr": 0.722222},
}
LOOP15 = {
    "validation": {"pnl": 100.56336780971753, "mdd": -10.677652697162888, "trades": 33, "wr": 0.6363636363636364},
    "oos": {"pnl": 72.81869539969456, "mdd": -8.108170708968366, "trades": 18, "wr": 0.7222222222222222},
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _predict_train(frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    if TRAIN_PRED_CACHE.exists():
        return pd.read_csv(TRAIN_PRED_CACHE, parse_dates=["timestamp"])
    bundle = torch.load(PARENT_BUNDLE, map_location=device, weights_only=False)
    x = threehead._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    src = threehead._prediction_output(frame, direction, quality, threshold=0.0, prefix=PREFIX_TRAIN_VAL.rstrip("_"))
    TRAIN_PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    src.to_csv(TRAIN_PRED_CACHE, index=False)
    return src


def _split(
    frames: dict[str, pd.DataFrame],
    name: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if name == "train":
        frame = frames["train_raw"].reset_index(drop=True)
        pred = _predict_train(frame, device)
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif name == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif name == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_OOS
        oof = False
    else:
        raise RuntimeError(f"unknown split: {name}")
    src = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if src.isna().any().any():
        bad = src.loc[src.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"{name} prediction alignment produced NaN: {bad}")
    dec0 = overlay._build_dec(src, prefix, oof=oof)
    teacher = sleeve._apply_aggressive(dec0)
    x = sleeve._extra_features(base_features._feature_frame(frame, src, dec0, prefix), dec0)
    bad_cols = [
        c
        for c in x.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_cols:
        raise RuntimeError(f"{name}: forbidden features: {bad_cols[:20]}")
    return frame, src, dec0, teacher, x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_regressor(x: pd.DataFrame, idx: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.35,
        random_state=int(seed),
    )
    model.fit(x.iloc[idx].to_numpy(dtype=np.float64), np.asarray(y, dtype=np.float64))
    return model


def _base_distill(
    dec0: pd.DataFrame,
    x: pd.DataFrame,
    models: dict[str, HistGradientBoostingRegressor],
    *,
    notional_scale: float = 1.003,
    tp_scale: float = 1.0,
    sl_scale: float = 1.0,
    cap: float = 0.81,
) -> pd.DataFrame:
    out = dec0.copy().reset_index(drop=True)
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out
    arr = x.to_numpy(dtype=np.float64)
    notional = np.clip(models["notional"].predict(arr) * float(notional_scale), 0.0, float(cap))
    tp = np.clip(models["take_profit"].predict(arr) * float(tp_scale), 1.0e-8, 1.0)
    sl = np.clip(models["stop_loss"].predict(arr) * float(sl_scale), 1.0e-8, 1.0)
    out.loc[active, "notional_exposure"] = notional[active]
    out.loc[active, "position_fraction"] = notional[active]
    out.loc[active, "leverage"] = 2.0
    out.loc[active, "take_profit"] = tp[active]
    out.loc[active, "stop_loss"] = sl[active]
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _actions() -> tuple[list[tuple[float, float, float]], int]:
    actions = list(itertools.product((0.98, 1.0, 1.02), repeat=3))
    return actions, actions.index((1.0, 1.0, 1.0))


def _apply_action(base_dec: pd.DataFrame, action_idx: np.ndarray, actions: list[tuple[float, float, float]], *, conf: np.ndarray | None, min_conf: float) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out
    confidence = np.ones(len(out), dtype=np.float64) if conf is None else np.asarray(conf, dtype=np.float64)
    pred = np.asarray(action_idx, dtype=np.int64)
    for i in active:
        if confidence[int(i)] < float(min_conf):
            continue
        ns, tps, sls = actions[int(pred[int(i)])]
        n = float(out.loc[int(i), "notional_exposure"]) * float(ns)
        out.loc[int(i), "notional_exposure"] = min(max(n, 0.0), 0.81)
        out.loc[int(i), "position_fraction"] = min(max(n, 0.0), 0.81)
        out.loc[int(i), "take_profit"] = max(float(out.loc[int(i), "take_profit"]) * float(tps), 1.0e-8)
        out.loc[int(i), "stop_loss"] = max(float(out.loc[int(i), "stop_loss"]) * float(sls), 1.0e-8)
    return out


def _label_regularized_residual(
    frame: pd.DataFrame,
    base_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
    seed: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    active_idx = np.flatnonzero(omega._active(base_dec))
    if len(active_idx) > int(max_rows):
        active_idx = np.sort(rng.choice(active_idx, size=int(max_rows), replace=False))
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    actions, base_action = _actions()
    y = np.full(len(active_idx), int(base_action), dtype=np.int64)
    class_counts: dict[str, int] = {}
    improvement_sum = 0.0
    accepted = 0
    for k, i in enumerate(active_idx):
        if k % 1000 == 0:
            print(json.dumps({"stage": "regularized_residual_labels", "margin": float(margin), "seen": int(k), "total": int(len(active_idx))}), flush=True)
        base_row = base_dec.iloc[int(i)].copy()
        base_score, _base_meta = omega._simulate_trade(frame, arrays, int(i), base_row, fee=fee, slip=slip, cost_mult=cost_mult)
        best_j = int(base_action)
        best_score = float(base_score)
        for j, (ns, tps, sls) in enumerate(actions):
            if j == base_action:
                continue
            row = base_row.copy()
            n = min(max(float(row["notional_exposure"]) * float(ns), 0.0), 0.81)
            row.loc["notional_exposure"] = n
            row.loc["position_fraction"] = n
            row.loc["take_profit"] = max(float(row["take_profit"]) * float(tps), 1.0e-8)
            row.loc["stop_loss"] = max(float(row["stop_loss"]) * float(sls), 1.0e-8)
            score, _meta = omega._simulate_trade(frame, arrays, int(i), row, fee=fee, slip=slip, cost_mult=cost_mult)
            if float(score) > best_score:
                best_score = float(score)
                best_j = int(j)
        if best_j != base_action and best_score > float(base_score) + float(margin):
            y[k] = best_j
            improvement_sum += best_score - float(base_score)
            accepted += 1
        class_counts[str(int(y[k]))] = class_counts.get(str(int(y[k])), 0) + 1
    diag = {
        "rows": int(len(active_idx)),
        "margin": float(margin),
        "class_counts": class_counts,
        "base_action": int(base_action),
        "base_action_rate": float(np.mean(y == base_action)) if len(y) else 0.0,
        "accepted_nonbase_rate": float(accepted / max(len(y), 1)),
        "accepted_improvement_mean": float(improvement_sum / max(accepted, 1)),
        "actions": actions,
    }
    return active_idx.astype(np.int64), y, diag


def _fit_classifier(x: pd.DataFrame, idx: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=11,
        min_samples_leaf=45,
        l2_regularization=1.0,
        random_state=int(seed),
    )
    model.fit(x.iloc[idx].to_numpy(dtype=np.float64), y)
    return model


def _predict(model: HistGradientBoostingClassifier, x: pd.DataFrame, base_action: int) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    pred = classes[best].astype(np.int64)
    conf = proba[np.arange(len(x)), best].astype(np.float64)
    if int(base_action) not in set(int(c) for c in classes):
        raise RuntimeError("base action missing from classifier classes")
    return pred, conf


def _pack(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, **extra}
    for prefix, metrics in (("val", val_m), ("oos", oos_m)):
        row[f"{prefix}_pnl"] = float(metrics["pnl"])
        row[f"{prefix}_mdd"] = float(metrics["mdd"])
        row[f"{prefix}_wr"] = float(metrics["wr"])
        row[f"{prefix}_trades"] = int(metrics["trades"])
        row[f"{prefix}_avg_notional"] = float(metrics.get("avg_notional", 0.0))
        row[f"{prefix}_reasons"] = dict(metrics.get("exit_reasons", {}))
    row["val_delta"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
    row["oos_delta"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
    row["val_delta_vs_loop15"] = float(row["val_pnl"] - LOOP15["validation"]["pnl"])
    row["oos_delta_vs_loop15"] = float(row["oos_pnl"] - LOOP15["oos"]["pnl"])
    row["strict_pass"] = bool(
        row["val_pnl"] >= CURRENT["validation"]["pnl"]
        and row["oos_pnl"] >= CURRENT["oos"]["pnl"]
        and row["val_mdd"] >= CURRENT["validation"]["mdd"]
        and row["oos_mdd"] >= CURRENT["oos"]["mdd"]
        and row["val_trades"] == CURRENT["validation"]["trades"]
        and row["oos_trades"] == CURRENT["oos"]["trades"]
    )
    row["validation_pass"] = bool(
        row["val_pnl"] >= CURRENT["validation"]["pnl"]
        and row["val_mdd"] >= CURRENT["validation"]["mdd"]
        and row["val_trades"] == CURRENT["validation"]["trades"]
    )
    row["validation_score"] = float(row["val_pnl"] + 2.0 * row["val_mdd"] - 0.02 * max(0, row["val_trades"] - CURRENT["validation"]["trades"]))
    row["score"] = float(row["val_pnl"] + row["oos_pnl"] + 2.0 * row["val_mdd"] + row["oos_mdd"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-label-rows", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=260741)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()

    train_frame, _train_src, train_dec0, train_teacher, x_train = _split(frames, "train", device)
    val_frame, _val_src, val_dec0, val_teacher, x_val = _split(frames, "validation", device)
    oos_frame, _oos_src, oos_dec0, oos_teacher, x_oos = _split(frames, "oos", device)

    train_active = np.flatnonzero(omega._active(train_teacher))
    risk_models = {
        "notional": _fit_regressor(x_train, train_active, pd.to_numeric(train_teacher.loc[train_active, "notional_exposure"]).to_numpy(dtype=np.float64), int(args.seed) + 1),
        "take_profit": _fit_regressor(x_train, train_active, pd.to_numeric(train_teacher.loc[train_active, "take_profit"]).to_numpy(dtype=np.float64), int(args.seed) + 2),
        "stop_loss": _fit_regressor(x_train, train_active, pd.to_numeric(train_teacher.loc[train_active, "stop_loss"]).to_numpy(dtype=np.float64), int(args.seed) + 3),
    }
    train_base = _base_distill(train_dec0, x_train, risk_models)
    val_base = _base_distill(val_dec0, x_val, risk_models)
    oos_base = _base_distill(oos_dec0, x_oos, risk_models)

    current_val = omega._metrics(val_frame, val_teacher, fee=fee, slip=slip, cost_mult=3.0)
    current_oos = omega._metrics(oos_frame, oos_teacher, fee=fee, slip=slip, cost_mult=3.0)
    base_val = omega._metrics(val_frame, val_base, fee=fee, slip=slip, cost_mult=3.0)
    base_oos = omega._metrics(oos_frame, oos_base, fee=fee, slip=slip, cost_mult=3.0)

    actions, base_action = _actions()
    rows: list[dict[str, Any]] = []
    label_diags: dict[str, Any] = {}
    for margin in (0.00025, 0.0005, 0.001, 0.002):
        idx, y, diag = _label_regularized_residual(
            train_frame,
            train_base,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            max_rows=int(args.max_label_rows),
            seed=int(args.seed),
            margin=float(margin),
        )
        label_diags[str(margin)] = diag
        if len(np.unique(y)) < 2:
            continue
        model = _fit_classifier(x_train, idx, y, int(args.seed + margin * 1_000_000))
        val_pred, val_conf = _predict(model, x_val, base_action)
        oos_pred, oos_conf = _predict(model, x_oos, base_action)
        for min_conf in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75):
            vd = _apply_action(val_base, val_pred, actions, conf=val_conf, min_conf=float(min_conf))
            od = _apply_action(oos_base, oos_pred, actions, conf=oos_conf, min_conf=float(min_conf))
            vm = omega._metrics(val_frame, vd, fee=fee, slip=slip, cost_mult=3.0)
            om = omega._metrics(oos_frame, od, fee=fee, slip=slip, cost_mult=3.0)
            name = f"regresid_margin{margin:g}_conf{min_conf:g}"
            rows.append(_pack(name, vm, om, {"margin": float(margin), "min_conf": float(min_conf)}))

    base_row = _pack("loop15_rebuilt_base_distill", base_val, base_oos, {"margin": None, "min_conf": None})
    rows.append(base_row)
    ranking = pd.DataFrame(rows).sort_values(["validation_pass", "validation_score", "val_pnl"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "design": "Regularized residual classifier on top of Loop15 full learned risk distillation. Residual labels keep the base learned risk unless train forward-path utility improves by a fixed margin. Candidate runtime uses learned notional, TP, and SL regressors plus optional learned residual action; no literal BASE_TP/BASE_SL constants.",
        "comparison_baseline": CURRENT,
        "loop15_reference": LOOP15,
        "current_recomputed": {"validation": current_val, "oos": current_oos},
        "rebuilt_base": base_row,
        "selected": selected,
        "strict_pass_count": int(ranking["strict_pass"].sum()),
        "label_diags": label_diags,
        "actions": actions,
        "base_action": int(base_action),
        "top": ranking.head(40).to_dict(orient="records"),
        "artifacts": {"out": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "strict_pass_count": report["strict_pass_count"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
