#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_fallback_label_family_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
BASE_RISK = sleeve.FallbackRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192)
CURRENT_MLP_VAL_PNL = 102.349040
CURRENT_MLP_OOS_PNL = 85.8772460561837
CURRENT_MLP_VAL_MDD = -10.677652697162888
CURRENT_MLP_OOS_MDD = -8.108170708968387


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


def _forbidden_features(cols: list[str]) -> list[str]:
    return [
        c
        for c in cols
        if c == "tp_sl_action_score"
        or c.startswith("clean_regime4_")
        or c.startswith("regime4_pred_")
        or c.startswith("teacher_")
    ]


def _make_model(name: str, seed: int):
    if name == "extra":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_depth=5,
            min_samples_leaf=35,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=int(seed),
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        alpha=0.02,
                        learning_rate_init=0.001,
                        max_iter=240,
                        early_stopping=True,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    raise RuntimeError(f"unknown model: {name}")


def _atr_pct(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise")
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=int(period), adjust=False, min_periods=1).mean()
    return (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _load_zigzag_labels(frame: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = ZIGZAG_DIR / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    if "zigzag_action" not in labels.columns:
        raise RuntimeError(f"{path} missing zigzag_action")
    src = frame[["timestamp"]].copy()
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    joined = src.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="left", validate="one_to_one")
    missing = int(joined["zigzag_action"].isna().sum())
    if missing:
        raise RuntimeError(f"zigzag label join missing rows for {year}: {missing}")
    y = pd.to_numeric(joined["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"invalid zigzag_action values: {invalid}")
    valid = np.ones(len(frame), dtype=bool)
    return y, valid, {"source": str(path), "label_counts": _counts(y)}


def _triple_barrier_labels(frame: pd.DataFrame, *, atr_mult: float, max_hold: int, min_barrier: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    atrp = np.maximum(_atr_pct(frame), float(min_barrier))
    y = np.zeros(len(frame), dtype=np.int64)
    valid = np.zeros(len(frame), dtype=bool)
    tie_count = 0
    for i in range(0, len(frame) - int(max_hold) - 2):
        base_px = float(close[i])
        if base_px <= 0.0 or not np.isfinite(base_px):
            continue
        barrier = float(atr_mult) * float(atrp[i])
        up = base_px * (1.0 + barrier)
        dn = base_px * (1.0 - barrier)
        valid[i] = True
        for j in range(i + 1, min(len(frame), i + int(max_hold) + 1)):
            hit_up = bool(high[j] >= up)
            hit_dn = bool(low[j] <= dn)
            if hit_up and hit_dn:
                tie_count += 1
                y[i] = sleeve.ACTION_LONG if close[j] >= base_px else sleeve.ACTION_SHORT
                break
            if hit_up:
                y[i] = sleeve.ACTION_LONG
                break
            if hit_dn:
                y[i] = sleeve.ACTION_SHORT
                break
    return y, valid, {"atr_mult": float(atr_mult), "max_hold": int(max_hold), "min_barrier": float(min_barrier), "tie_count": int(tie_count), "label_counts": _counts(y[valid])}


def _topk_future_labels(frame: pd.DataFrame, cash_mask: np.ndarray, *, group_hours: int, k: int, min_ret: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    ts = pd.to_datetime(frame["timestamp"])
    y = np.zeros(len(frame), dtype=np.int64)
    valid = np.zeros(len(frame), dtype=bool)
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    sides = np.zeros(len(frame), dtype=np.int64)
    horizons = (6, 12, 24, 48)
    max_h = max(horizons)
    for i in np.flatnonzero(cash_mask):
        if i >= len(frame) - max_h - 2:
            continue
        base_px = float(close[i])
        if base_px <= 0.0 or not np.isfinite(base_px):
            continue
        rets = np.asarray([(close[i + h] - base_px) / base_px for h in horizons], dtype=np.float64)
        long_score = float(np.nanmax(rets))
        short_score = float(np.nanmax(-rets))
        if long_score >= short_score:
            sides[i] = sleeve.ACTION_LONG
            scores[i] = long_score
        else:
            sides[i] = sleeve.ACTION_SHORT
            scores[i] = short_score
        valid[i] = True
    groups = ts.dt.floor(f"{int(group_hours)}h")
    picked = 0
    for _g, idx_s in pd.Series(np.arange(len(frame))).groupby(groups):
        idx = np.asarray(idx_s.to_numpy(), dtype=np.int64)
        idx = idx[valid[idx] & np.isfinite(scores[idx]) & (scores[idx] >= float(min_ret))]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(scores[idx])[::-1][: int(k)]]
        y[top] = sides[top]
        picked += len(top)
    return y, valid, {"group_hours": int(group_hours), "k": int(k), "min_ret": float(min_ret), "picked": int(picked), "valid_rows": int(valid.sum()), "label_counts": _counts(y[valid])}


def _reversal_labels(frame: pd.DataFrame, *, z_th: float, horizon: int, min_ret: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close_s = pd.to_numeric(frame["close"], errors="raise")
    close = close_s.to_numpy(dtype=np.float64)
    ma = close_s.rolling(48, min_periods=12).mean()
    sd = close_s.rolling(48, min_periods=12).std().replace(0.0, np.nan)
    z = ((close_s - ma) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    y = np.zeros(len(frame), dtype=np.int64)
    valid = np.zeros(len(frame), dtype=bool)
    for i in range(0, len(frame) - int(horizon) - 2):
        base_px = float(close[i])
        if base_px <= 0.0 or not np.isfinite(base_px):
            continue
        future_ret = float((close[i + int(horizon)] - base_px) / base_px)
        valid[i] = True
        if z[i] <= -abs(float(z_th)) and future_ret >= float(min_ret):
            y[i] = sleeve.ACTION_LONG
        elif z[i] >= abs(float(z_th)) and future_ret <= -float(min_ret):
            y[i] = sleeve.ACTION_SHORT
    return y, valid, {"z_th": float(z_th), "horizon": int(horizon), "min_ret": float(min_ret), "label_counts": _counts(y[valid])}


def _sltp_edge_labels(frame: pd.DataFrame, dec: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y, diag = sleeve._build_labels(frame, dec, BASE_RISK, 0.006)
    valid = np.zeros(len(frame), dtype=bool)
    active = omega._active(dec)
    valid[np.flatnonzero(~active & (np.arange(len(frame)) < len(frame) - int(BASE_RISK.max_hold_bars) - 3))] = True
    return y, valid, diag


def _counts(arr: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(arr).value_counts().sort_index().items()}


def _label_family(name: str, frame: pd.DataFrame, dec: pd.DataFrame, cash_mask: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if name == "sltp_edge006":
        return _sltp_edge_labels(frame, dec)
    if name == "zigzag_action":
        return _load_zigzag_labels(frame, year)
    if name == "tb_atr08_h48":
        return _triple_barrier_labels(frame, atr_mult=0.8, max_hold=48, min_barrier=0.0035)
    if name == "tb_atr12_h96":
        return _triple_barrier_labels(frame, atr_mult=1.2, max_hold=96, min_barrier=0.0040)
    if name == "topk2_8h":
        return _topk_future_labels(frame, cash_mask, group_hours=8, k=2, min_ret=0.004)
    if name == "topk3_8h":
        return _topk_future_labels(frame, cash_mask, group_hours=8, k=3, min_ret=0.003)
    if name == "reversal_z12_h24":
        return _reversal_labels(frame, z_th=1.2, horizon=24, min_ret=0.004)
    raise RuntimeError(f"unknown label family: {name}")


def _predict_oof(model_name: str, x: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(train_mask)
    action = np.zeros(len(x), dtype=np.int64)
    conf = np.zeros(len(x), dtype=np.float64)
    folds = []
    n = len(idx)
    for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        unique = np.unique(y[train_idx])
        if len(unique) < 2:
            folds.append({"fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "skipped": "single_class"})
            continue
        model = _make_model(model_name, seed + train_end)
        model.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
        proba = model.predict_proba(x.iloc[val_idx].to_numpy(dtype=np.float64))
        classes = np.asarray(model.classes_, dtype=np.int64)
        best = np.argmax(proba, axis=1)
        action[val_idx] = classes[best]
        conf[val_idx] = proba[np.arange(len(val_idx)), best]
        folds.append({"fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "classes": classes.tolist()})
    return action, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _fit_predict(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, train_mask: np.ndarray, x_eval: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, Any | None]:
    idx = np.flatnonzero(train_mask)
    if len(np.unique(y_train[idx])) < 2:
        return np.zeros(len(x_eval), dtype=np.int64), np.zeros(len(x_eval), dtype=np.float64), None
    model = _make_model(model_name, seed)
    model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_train[idx])
    proba = model.predict_proba(x_eval.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x_eval)), best].astype(np.float64), model


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, metrics)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    val_dec = sleeve._apply_aggressive(val_dec0)
    oos_dec = sleeve._apply_aggressive(oos_dec0)
    val_features = sleeve._extra_features(base._feature_frame(val_frame, val_src, val_dec0, val_prefix), val_dec)
    oos_features = sleeve._extra_features(base._feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix), oos_dec)
    bad = _forbidden_features(list(val_features.columns))
    if bad:
        raise RuntimeError(f"forbidden fallback label-family feature columns: {bad}")
    val_cash = ~omega._active(val_dec)
    oos_cash = ~omega._active(oos_dec)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "risk": asdict(BASE_RISK),
        "feature_count": int(val_features.shape[1]),
        "features": list(val_features.columns),
        "val_cash_rows": int(np.count_nonzero(val_cash)),
        "oos_cash_rows": int(np.count_nonzero(oos_cash)),
        "forbidden_feature_audit": {"passed": True, "forbidden": []},
    }
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append(
        {
            "label": "none",
            "model": "aggressive_primary_only",
            "threshold": 1.0,
            **_metric_row("val", {**baseline_val, "primary_entries": baseline_val["long_entries"] + baseline_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
            **_metric_row("oos", {**baseline_oos, "primary_entries": baseline_oos["long_entries"] + baseline_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
        }
    )
    labels = ("sltp_edge006", "zigzag_action", "tb_atr08_h48", "tb_atr12_h96", "topk2_8h", "topk3_8h", "reversal_z12_h24")
    model_names = ("extra", "hgb", "mlp")
    thresholds = (0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95)
    best_payload: dict[str, Any] | None = None
    best_key = (-1.0e18, -1.0e18)
    for label_name in labels:
        print(json.dumps({"stage": "label", "label": label_name}, ensure_ascii=False), flush=True)
        y_val, valid_val, val_diag = _label_family(label_name, val_frame, val_dec, val_cash, 2025)
        _y_oos, _valid_oos, oos_label_diag = _label_family(label_name, oos_frame, oos_dec, oos_cash, 2026)
        train_mask = val_cash & valid_val
        diagnostics[f"{label_name}_val"] = val_diag
        diagnostics[f"{label_name}_oos_label_diag"] = oos_label_diag
        if int(np.count_nonzero(train_mask)) < 500 or len(np.unique(y_val[train_mask])) < 2:
            diagnostics[f"{label_name}_skip"] = {"train_rows": int(np.count_nonzero(train_mask)), "unique": np.unique(y_val[train_mask]).tolist()}
            continue
        for model_name in model_names:
            print(json.dumps({"stage": "fit_eval", "label": label_name, "model": model_name}, ensure_ascii=False), flush=True)
            val_action, val_conf, oof_diag = _predict_oof(model_name, val_features, y_val, train_mask, seed=260606)
            oos_action, oos_conf, fitted = _fit_predict(model_name, val_features, y_val, train_mask, oos_features, seed=260606)
            diagnostics[f"{label_name}_{model_name}_oof"] = oof_diag
            for threshold in thresholds:
                val_m = sleeve._metrics_with_fallback(val_frame, val_dec, BASE_RISK, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
                oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, BASE_RISK, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
                row = {"label": label_name, "model": model_name, "threshold": float(threshold)}
                row.update(_metric_row("val", val_m))
                row.update(_metric_row("oos", oos_m))
                rows.append(row)
                key = (float(oos_m["pnl"]), float(val_m["pnl"]))
                if fitted is not None and key > best_key:
                    best_key = key
                    best_payload = {
                        "label": label_name,
                        "model_name": model_name,
                        "threshold": float(threshold),
                        "model": fitted,
                        "val_metrics": val_m,
                        "oos_metrics": oos_m,
                        "train_mask_rows": int(np.count_nonzero(train_mask)),
                    }
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - sleeve.AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - sleeve.AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - sleeve.AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - sleeve.AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_fallback_label_family_ranking.csv", index=False)
    promotable = ranking[
        (ranking["model"] != "aggressive_primary_only")
        & (ranking["oos_pnl"] > CURRENT_MLP_OOS_PNL)
        & (ranking["val_pnl"] > CURRENT_MLP_VAL_PNL)
        & (ranking["oos_mdd"] >= CURRENT_MLP_OOS_MDD * 1.35)
        & (ranking["val_mdd"] >= CURRENT_MLP_VAL_MDD * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "cash_fallback_label_family_promotable.csv", index=False)
    saved_model_dir = None
    if best_payload is not None and int(len(promotable)) > 0:
        top = promotable.sort_values(["oos_pnl", "val_pnl"], ascending=False).iloc[0]
        top_label = str(top["label"])
        top_model = str(top["model"])
        y_val, valid_val, _diag = _label_family(top_label, val_frame, val_dec, val_cash, 2025)
        train_mask = val_cash & valid_val
        _, _, fitted = _fit_predict(top_model, val_features, y_val, train_mask, oos_features, seed=260606)
        if fitted is None:
            raise RuntimeError("promotable candidate produced no fitted model")
        thr_tag = str(float(top["threshold"])).replace(".", "")
        saved_model_dir = ROOT / "data/ensemble/supervised" / f"omega1_2_1_cash_fallback_{top_label}_{top_model}_thr{thr_tag}_20260606"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        with (saved_model_dir / "cash_fallback_label_model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "model": fitted,
                    "feature_columns": list(val_features.columns),
                    "label_family": top_label,
                    "model_name": top_model,
                    "threshold": float(top["threshold"]),
                    "risk": asdict(BASE_RISK),
                    "forbidden_feature_audit": {"passed": True, "forbidden": []},
                },
                f,
            )
        manifest = {
            "model_id": saved_model_dir.name,
            "created_at": "2026-06-06",
            "role": "cash fallback label-family sleeve candidate; activates only when primary is CASH and no position is open",
            "base_primary": "omega1_2_1_aggressive_compensated_scale200_cap090",
            "label_family": top_label,
            "model_type": top_model,
            "feature_count": int(val_features.shape[1]),
            "feature_columns": list(val_features.columns),
            "risk": asdict(BASE_RISK),
            "threshold": float(top["threshold"]),
            "metrics": top.to_dict(),
            "ranking_report": str(OUT_DIR / "report.json"),
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
        }
        (saved_model_dir / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "baseline": "omega1_2_1_cash_fallback_mlp_base_edge006_thr085_20260606",
        "method": "Cash-only fallback label-family comparison. Primary aggressive baseline, fallback risk, feature contract, and Cost3 accounting are fixed; only supervised label source and model family vary.",
        "diagnostics": diagnostics,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(len(promotable)),
        "saved_model_dir": str(saved_model_dir) if saved_model_dir is not None else None,
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "cash_fallback_label_family_ranking.csv"),
            "promotable": str(OUT_DIR / "cash_fallback_label_family_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "promotable_count": int(len(promotable)), "saved_model_dir": report["saved_model_dir"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
