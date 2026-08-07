#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_quality_replay_ridge_20260620 as label_probe  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_frozen_direction_quality_replay_hgb_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"

SPLIT_TS = pd.Timestamp("2025-10-01")
LABEL_CLIP_Q = (0.01, 0.99)
THRESHOLDS = [-0.002, 0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]
MAX_FEATURES = 180


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(type(obj).__name__)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["timestamp"], low_memory=False).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _load_split(raw: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    if split == "train":
        parent, prefix = label_probe._parent_predictions("train")
        frame = raw[raw["timestamp"] < SPLIT_TS].reset_index(drop=True)
        oof = True
    elif split == "validation":
        parent, prefix = label_probe._parent_predictions("validation")
        frame = raw[raw["timestamp"] >= SPLIT_TS].reset_index(drop=True)
        oof = True
    elif split == "oos":
        parent, prefix = label_probe._parent_predictions("oos")
        frame = raw.reset_index(drop=True)
        oof = False
    else:
        raise RuntimeError(split)
    aligned = label_probe._align(frame, parent, prefix)
    return aligned, parent, prefix, oof


def _feature_cols(train: pd.DataFrame, val: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    banned_tokens = ("target", "future", "label", "pnl", "zigzag", "wave3", "timestamp")
    banned_prefixes = ("clean_regime4_", "regime4_pred_", "teacher_", "teacher_oof_")
    banned_exact = {"tp_sl_action_score", "execution_quality", "parent_old_quality", "parent_old_final_action"}
    common = [c for c in train.columns if c in val.columns and c in oos.columns]
    cols: list[str] = []
    for col in common:
        low = col.lower()
        if col in banned_exact:
            continue
        if any(tok in low for tok in banned_tokens):
            continue
        if any(col.startswith(prefix) for prefix in banned_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(val[col]) and pd.api.types.is_numeric_dtype(oos[col]):
            cols.append(col)
    if len(cols) > MAX_FEATURES:
        var = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).var().sort_values(ascending=False)
        cols = list(var.head(MAX_FEATURES).index)
    if not cols:
        raise RuntimeError("no quality feature columns")
    return cols


def _fit_quality_model(train: pd.DataFrame, cols: list[str]) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    active = pd.to_numeric(train["parent_dir_action"], errors="coerce").fillna(0).astype(int) != omega.ACTION_CASH
    y_raw = pd.to_numeric(train.loc[active, "quality_target_net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    lo, hi = np.quantile(y_raw, LABEL_CLIP_Q)
    y = np.clip(y_raw, lo, hi)
    x = train.loc[active, cols].replace([np.inf, -np.inf], np.nan)
    sample_weight = 1.0 + np.clip(np.abs(y_raw), 0.0, 0.03) * 25.0
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=160,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        min_samples_leaf=35,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=260620,
    )
    model.fit(x, y, sample_weight=sample_weight)
    pred = model.predict(x)
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 3 and np.std(pred) > 0 and np.std(y) > 0 else 0.0
    diag = {
        "active_rows": int(active.sum()),
        "clip_q01_q99": [float(lo), float(hi)],
        "train_corr_active": corr,
        "n_iter": int(getattr(model, "n_iter_", 0)),
    }
    return model, diag


def _predict(df: pd.DataFrame, cols: list[str], model: HistGradientBoostingRegressor) -> np.ndarray:
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    out = model.predict(x).astype(np.float64)
    action = pd.to_numeric(df["parent_dir_action"], errors="coerce").fillna(0).astype(int).to_numpy()
    out[action == omega.ACTION_CASH] = -999.0
    return out


def _replace_quality(parent_src: pd.DataFrame, prefix: str, q: np.ndarray, threshold: float) -> pd.DataFrame:
    out = parent_src.copy()
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    final = action.copy()
    final[(action != omega.ACTION_CASH) & (q < float(threshold))] = omega.ACTION_CASH
    for suffix in ("quality_p_cash", "quality_p_long", "quality_p_short", "quality_for_action"):
        out[f"{prefix}{suffix}"] = q
    out[f"{prefix}quality_threshold"] = float(threshold)
    out[f"{prefix}final_action"] = final
    return out


def _metric_row(name: str, threshold: float, val_m: dict[str, Any], oos_m: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": name,
        "threshold": float(threshold),
        "validation_pnl": val_m["pnl"],
        "validation_mdd": val_m["mdd"],
        "validation_trades": val_m["trades"],
        "validation_wr": val_m["wr"],
        "oos_pnl": oos_m["pnl"],
        "oos_mdd": oos_m["mdd"],
        "oos_trades": oos_m["trades"],
        "oos_wr": oos_m["wr"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(label_probe.RAW_2025)
    raw_2026 = _read(label_probe.RAW_2026)
    train, train_parent, train_prefix, _ = _load_split(raw_2025, "train")
    val, val_parent, val_prefix, val_oof = _load_split(raw_2025, "validation")
    oos, oos_parent, oos_prefix, oos_oof = _load_split(raw_2026, "oos")

    train = pd.concat([train, label_probe._replay_quality_labels(train)], axis=1)
    val = pd.concat([val, label_probe._replay_quality_labels(val)], axis=1)
    oos = pd.concat([oos, label_probe._replay_quality_labels(oos)], axis=1)
    cols = _feature_cols(train, val, oos)
    model, model_diag = _fit_quality_model(train, cols)
    val_q = _predict(val, cols, model)
    oos_q = _predict(oos, cols, model)
    fee, slip = omega._load_fee_slip()

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        val_src = _replace_quality(val_parent, val_prefix, val_q, threshold)
        oos_src = _replace_quality(oos_parent, oos_prefix, oos_q, threshold)
        val_dec = omega._to_fixed_decisions(val_src, oof=val_oof)
        oos_dec = omega._to_fixed_decisions(oos_src, oof=oos_oof)
        val_m = omega._metrics(val.reset_index(drop=True), val_dec, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = omega._metrics(oos.reset_index(drop=True), oos_dec, fee=fee, slip=slip, cost_mult=3.0)
        name = f"frozen_dir_quality_thr_{threshold:.4f}".replace("-", "m").replace(".", "p")
        reports[name] = {"validation": val_m, "oos": oos_m}
        rows.append(_metric_row(name, threshold, val_m, oos_m))

    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"])), reverse=True)
    best_threshold = float(rows[0]["threshold"])
    best_val_src = _replace_quality(val_parent, val_prefix, val_q, best_threshold)
    best_oos_src = _replace_quality(oos_parent, oos_prefix, oos_q, best_threshold)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    best_val_src.to_csv(OUT_DIR / "validation_predictions_2025_frozen_direction_quality_replay.csv", index=False)
    best_oos_src.to_csv(OUT_DIR / "oos_predictions_2026_frozen_direction_quality_replay.csv", index=False)
    joblib.dump({"model": model, "columns": cols, "model_diag": model_diag}, OUT_DIR / "quality_replay_hgb.joblib")
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen q080 parent direction. Train only an external quality regression head on barrier-replay cost-included net return, then replace quality_for_action/final_action before the existing fixed TP/SL replay.",
        "quality_label_contract": {
            "side_source": "q080 parent_dir_action; train uses existing in-sample parent predictions from Jan-Sep 2025",
            "target": "quality_target_net_return",
            "replay": "same lightweight close-path fixed TP/SL barrier replay as the step-1 quality probe",
            "label_clip_q": LABEL_CLIP_Q,
        },
        "model_diag": model_diag,
        "label_audit": {
            "train_rows": int(len(train)),
            "train_active_rows": int((train["parent_dir_action"].astype(int) != omega.ACTION_CASH).sum()),
            "train_positive_rate": float((train["quality_target_net_return"] > 0.0).mean()),
            "validation_positive_rate": float((val["quality_target_net_return"] > 0.0).mean()),
            "oos_positive_rate": float((oos["quality_target_net_return"] > 0.0).mean()),
        },
        "prediction_audit": {
            "validation_pred_quantiles": {str(q): float(np.quantile(val_q[np.isfinite(val_q) & (val_q > -1.0)], q)) for q in [0.1, 0.5, 0.9]},
            "oos_pred_quantiles": {str(q): float(np.quantile(oos_q[np.isfinite(oos_q) & (oos_q > -1.0)], q)) for q in [0.1, 0.5, 0.9]},
        },
        "ranking_by_validation_pnl": rows,
        "results": reports,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "quality_model": str(OUT_DIR / "quality_replay_hgb.joblib"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top5": rows[:5], "model_diag": model_diag}, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
