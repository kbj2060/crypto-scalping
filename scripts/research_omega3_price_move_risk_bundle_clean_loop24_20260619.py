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
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_omega3_full_distill_residual_regularized_loop17_20260619 as loop17  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega3_price_move_risk_bundle_clean_loop24_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CURRENT = {
    "validation": {"pnl": 100.542729421, "mdd": -10.677653, "trades": 33, "wr": 0.636364},
    "oos": {"pnl": 72.760041481, "mdd": -8.108171, "trades": 18, "wr": 0.722222},
}
RISK_KEYS = ("notional", "tp_price_move", "sl_price_move")
FORBIDDEN_FEATURE_PREFIXES = ("teacher_", "regime4_pred_", "clean_regime4_", "clean_regime_2024_unsup_v4_")
FORBIDDEN_FEATURE_NAMES = {"tp_sl_action_score"}


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


def _reject_forbidden_features(cols: list[str]) -> None:
    bad = [
        str(c)
        for c in cols
        if str(c) in FORBIDDEN_FEATURE_NAMES or any(str(c).startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    ]
    if bad:
        raise RuntimeError(f"forbidden risk feature columns: {bad[:40]}")


def _fit_regressor(x: pd.DataFrame, idx: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.45,
        random_state=int(seed),
    )
    model.fit(x.iloc[idx].to_numpy(dtype=np.float64), np.asarray(y, dtype=np.float64))
    return model


def _label_frame(teacher: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    active = np.flatnonzero(omega._active(teacher))
    notional = pd.to_numeric(teacher.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    good = np.isfinite(notional) & (notional > 1.0e-9)
    idx = active[good]
    n = notional[good]
    tp = pd.to_numeric(teacher.loc[idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64)
    sl = pd.to_numeric(teacher.loc[idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    labels = pd.DataFrame(
        {
            "notional": n,
            "tp_price_move": np.clip(tp / np.maximum(n, 1.0e-9), 1.0e-8, 1.0),
            "sl_price_move": np.clip(sl / np.maximum(n, 1.0e-9), 1.0e-8, 1.0),
        }
    )
    bounds = {
        key: {
            "min": float(labels[key].quantile(0.001)),
            "max": float(labels[key].quantile(0.999)),
            "mean": float(labels[key].mean()),
            "median": float(labels[key].median()),
        }
        for key in RISK_KEYS
    }
    diag = {
        "rows": int(len(labels)),
        "notional_min": float(labels["notional"].min()),
        "notional_max": float(labels["notional"].max()),
        "tp_price_move_median": float(labels["tp_price_move"].median()),
        "sl_price_move_median": float(labels["sl_price_move"].median()),
        "bounds": bounds,
    }
    return idx.astype(np.int64), labels, diag


def _predict_risk(
    x: pd.DataFrame,
    models: dict[str, HistGradientBoostingRegressor],
    bounds: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    arr = x.to_numpy(dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for key in RISK_KEYS:
        pred = np.asarray(models[key].predict(arr), dtype=np.float64)
        lo = float(bounds[key]["min"])
        hi = float(bounds[key]["max"])
        out[key] = np.clip(pred, lo, hi)
    return out


def _apply_candidate(
    dec0: pd.DataFrame,
    x: pd.DataFrame,
    models: dict[str, HistGradientBoostingRegressor],
    bounds: dict[str, dict[str, float]],
    *,
    notional_scale: float,
    tp_scale: float,
    sl_scale: float,
    cap: float,
) -> pd.DataFrame:
    out = dec0.copy().reset_index(drop=True)
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out
    pred = _predict_risk(x, models, bounds)
    notional = np.clip(pred["notional"] * float(notional_scale), 0.0, float(cap))
    tp_move = np.clip(pred["tp_price_move"] * float(tp_scale), 1.0e-8, 1.0)
    sl_move = np.clip(pred["sl_price_move"] * float(sl_scale), 1.0e-8, 1.0)
    out.loc[active, "notional_exposure"] = notional[active]
    out.loc[active, "position_fraction"] = notional[active]
    out.loc[active, "leverage"] = 2.0
    out.loc[active, "take_profit"] = tp_move[active] * notional[active]
    out.loc[active, "stop_loss"] = sl_move[active] * notional[active]
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _row(candidate: str, cfg: dict[str, float], val_m: dict[str, Any], oos_m: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": candidate, **cfg}
    for prefix, metrics in (("val", val_m), ("oos", oos_m)):
        if metrics is None:
            continue
        row[f"{prefix}_pnl"] = float(metrics["pnl"])
        row[f"{prefix}_mdd"] = float(metrics["mdd"])
        row[f"{prefix}_wr"] = float(metrics["wr"])
        row[f"{prefix}_trades"] = int(metrics["trades"])
        row[f"{prefix}_avg_notional"] = float(metrics.get("avg_notional", 0.0))
        row[f"{prefix}_reasons"] = dict(metrics.get("exit_reasons", {}))
    row["validation_pass"] = bool(
        row["val_pnl"] >= CURRENT["validation"]["pnl"]
        and row["val_mdd"] >= CURRENT["validation"]["mdd"]
        and row["val_trades"] == CURRENT["validation"]["trades"]
    )
    row["validation_score"] = float(
        row["val_pnl"]
        + 2.0 * row["val_mdd"]
        - 0.05 * max(0, row["val_trades"] - CURRENT["validation"]["trades"])
        - 0.10 * abs(row["val_trades"] - CURRENT["validation"]["trades"])
    )
    if oos_m is not None:
        row["val_delta"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
        row["oos_delta"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
        row["strict_pass"] = bool(
            row["validation_pass"]
            and row["oos_pnl"] >= CURRENT["oos"]["pnl"]
            and row["oos_mdd"] >= CURRENT["oos"]["mdd"]
            and row["oos_trades"] == CURRENT["oos"]["trades"]
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260901)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()

    train_frame, _train_src, train_dec0, train_teacher, x_train = loop17._split(frames, "train", device)
    val_frame, _val_src, val_dec0, val_teacher, x_val = loop17._split(frames, "validation", device)
    oos_frame, _oos_src, oos_dec0, oos_teacher, x_oos = loop17._split(frames, "oos", device)
    feature_cols = list(x_train.columns)
    _reject_forbidden_features(feature_cols)

    train_idx, train_labels, label_diag = _label_frame(train_teacher)
    models = {
        "notional": _fit_regressor(x_train, train_idx, train_labels["notional"].to_numpy(dtype=np.float64), int(args.seed) + 1),
        "tp_price_move": _fit_regressor(x_train, train_idx, train_labels["tp_price_move"].to_numpy(dtype=np.float64), int(args.seed) + 2),
        "sl_price_move": _fit_regressor(x_train, train_idx, train_labels["sl_price_move"].to_numpy(dtype=np.float64), int(args.seed) + 3),
    }
    bounds = dict(label_diag["bounds"])
    current_val = omega._metrics(val_frame, val_teacher, fee=fee, slip=slip, cost_mult=3.0)
    current_oos_reference = omega._metrics(oos_frame, oos_teacher, fee=fee, slip=slip, cost_mult=3.0)

    rows: list[dict[str, Any]] = []
    for notional_scale in (0.997, 1.0, 1.003, 1.006, 1.01):
        for tp_scale in (0.98, 1.0, 1.02, 1.04):
            for sl_scale in (0.98, 1.0, 1.02):
                for cap in (0.79, 0.81, 0.83):
                    cfg = {
                        "notional_scale": float(notional_scale),
                        "tp_scale": float(tp_scale),
                        "sl_scale": float(sl_scale),
                        "cap": float(cap),
                    }
                    vd = _apply_candidate(val_dec0, x_val, models, bounds, **cfg)
                    vm = omega._metrics(val_frame, vd, fee=fee, slip=slip, cost_mult=3.0)
                    rows.append(_row(f"pmove_ns{notional_scale:g}_tps{tp_scale:g}_sls{sl_scale:g}_cap{cap:g}", cfg, vm, None))
    ranking = pd.DataFrame(rows).sort_values(["validation_pass", "validation_score", "val_pnl"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "validation_ranking.csv", index=False)

    selected = ranking.iloc[0].to_dict()
    selected_cfg = {
        "notional_scale": float(selected["notional_scale"]),
        "tp_scale": float(selected["tp_scale"]),
        "sl_scale": float(selected["sl_scale"]),
        "cap": float(selected["cap"]),
    }
    # OOS is evaluated only after validation-only candidate lock.
    selected_val_dec = _apply_candidate(val_dec0, x_val, models, bounds, **selected_cfg)
    selected_oos_dec = _apply_candidate(oos_dec0, x_oos, models, bounds, **selected_cfg)
    selected_val = omega._metrics(val_frame, selected_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    selected_oos = omega._metrics(oos_frame, selected_oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    selected_final = _row(str(selected["candidate"]), selected_cfg, selected_val, selected_oos)

    bundle = {
        "model_id": MODEL_ID,
        "base_model_id": "omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608",
        "risk_contract": {
            "type": "price_move_heads",
            "keys": list(RISK_KEYS),
            "conversion": {
                "take_profit": "tp_price_move * notional",
                "stop_loss": "sl_price_move * notional",
                "notional": "predicted_notional * selected.notional_scale clipped by selected.cap",
            },
            "leverage": 2.0,
            "margin_fraction": "notional / leverage",
        },
        "feature_cols": feature_cols,
        "forbidden_feature_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "forbidden_feature_names": sorted(FORBIDDEN_FEATURE_NAMES),
        "models": models,
        "risk_label_bounds": bounds,
        "selected_config": selected_cfg,
        "selection_protocol": {
            "model_fit_split": "train",
            "candidate_selection_split": "validation",
            "final_holdout_split": "oos",
            "oos_used_for_selection": False,
        },
    }
    bundle_path = OUT_DIR / "omega3_price_move_risk_bundle_clean_loop24.joblib"
    joblib.dump(bundle, bundle_path)

    report = {
        "model_id": MODEL_ID,
        "design": "Clean price-move risk bundle. Train regressors predict notional, tp_price_move, and sl_price_move from train teacher outputs. Validation only selects scale/cap config. OOS is evaluated once after candidate lock. Account TP/SL conversion is take_profit=tp_price_move*notional and stop_loss=sl_price_move*notional.",
        "current_recomputed_reference": {"validation": current_val, "oos": current_oos_reference},
        "label_diag": label_diag,
        "feature_contract": {
            "feature_count": int(len(feature_cols)),
            "forbidden_feature_audit": {"passed": True},
            "risk_keys": list(RISK_KEYS),
        },
        "selection_protocol": {
            "model_fit_split": "train",
            "candidate_selection_split": "validation",
            "final_holdout_split": "oos",
            "oos_used_for_selection": False,
        },
        "selected_by_validation": selected_final,
        "validation_top10_no_oos": ranking.head(10).to_dict(orient="records"),
        "artifacts": {"out": str(OUT_DIR), "ranking": str(OUT_DIR / "validation_ranking.csv"), "bundle": str(bundle_path)},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "bundle": str(bundle_path), "selected": selected_final}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
