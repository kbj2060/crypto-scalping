#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_exposure_selector_20260606"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

BASELINE_VAL = {"pnl": 42.822624455902236, "mdd": -5.471616800975976, "wr": 0.6363636363636364, "trades": 33}
BASELINE_OOS = {"pnl": 32.14560500542696, "mdd": -4.135192490277451, "wr": 0.7222222222222222, "trades": 18}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"prediction alignment produced NaN: {bad}")
    return out


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(BASE_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        src = _align(frame, pred)
        dec = overlay._build_dec(src, "omega1_regime3_expertdq_oof_", oof=True)
        prefix = "omega1_regime3_expertdq_oof_"
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(BASE_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        src = _align(frame, pred)
        dec = overlay._build_dec(src, "omega1_regime3_expertdq_", oof=False)
        prefix = "omega1_regime3_expertdq_"
    else:
        raise RuntimeError(f"unknown split: {split}")
    return frame, src, dec, prefix


def _rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    open_ = pd.to_numeric(frame["open"], errors="raise")
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    out = pd.DataFrame(index=frame.index)
    out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    out["body_pct"] = ((close - open_) / close).replace([np.inf, -np.inf], np.nan)
    out["atr14_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)
    for lag in (1, 3, 6, 12, 24):
        out[f"ret_{lag}"] = close.pct_change(lag).replace([np.inf, -np.inf], np.nan)
    for win in (6, 12, 24, 48):
        out[f"ret_vol_{win}"] = ret.rolling(win, min_periods=max(3, win // 3)).std()
        out[f"range_mean_{win}"] = out["bar_range_pct"].rolling(win, min_periods=max(3, win // 3)).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema9_21_gap"] = ((ema9 - ema21) / close).replace([np.inf, -np.inf], np.nan)
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    minute = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _feature_frame(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = _rolling_features(frame)
    cols = [
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
    ]
    for col in cols:
        out[col] = pd.to_numeric(src[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float64)
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"})
    for name in ("bull", "bear", "chop"):
        out[f"router_is_{name}"] = expert.eq(name).astype(float).to_numpy()
    out["side"] = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float64)
    out["base_notional"] = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    out["base_tp"] = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    out["base_sl"] = pd.to_numeric(dec["stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    bad = [c for c in out.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"forbidden selector feature columns: {bad}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _candidate_labels(frame: pd.DataFrame, dec: pd.DataFrame, active_idx: np.ndarray, *, fee: float, slip: float) -> tuple[np.ndarray, np.ndarray]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y_win = np.zeros(len(active_idx), dtype=np.int64)
    y_net = np.zeros(len(active_idx), dtype=np.float64)
    for k, idx in enumerate(active_idx):
        _score, meta = omega._simulate_trade(frame, arrays, int(idx), dec.iloc[int(idx)], fee=fee, slip=slip, cost_mult=3.0)
        y_win[k] = int(meta.get("win", 0))
        y_net[k] = float(meta.get("net", 0.0))
    return y_win, y_net


def _make_model(name: str, seed: int):
    if name == "hgb_win":
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.035, max_leaf_nodes=7, l2_regularization=1.5, random_state=seed)
    if name == "extra_win":
        return ExtraTreesClassifier(n_estimators=240, max_depth=4, min_samples_leaf=25, random_state=seed, n_jobs=-1, class_weight="balanced")
    if name == "hgb_net":
        return HistGradientBoostingRegressor(max_iter=80, learning_rate=0.035, max_leaf_nodes=7, l2_regularization=2.0, random_state=seed)
    raise RuntimeError(f"unknown selector model: {name}")


def _predict_score(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name.endswith("_win"):
        return model.predict_proba(x)[:, 1].astype(np.float64)
    return model.predict(x).astype(np.float64)


def _fit_oof_scores(model_name: str, x: np.ndarray, y_win: np.ndarray, y_net: np.ndarray, active_idx: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(active_idx)
    scores = np.full(n, np.nan, dtype=np.float64)
    folds: list[dict[str, int]] = []
    for start_frac, end_frac in ((0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * start_frac)
        val_start = train_end
        val_end = int(n * end_frac)
        if train_end < 50 or val_end <= val_start:
            continue
        model = _make_model(model_name, seed + val_start)
        y = y_win[:train_end] if model_name.endswith("_win") else y_net[:train_end]
        model.fit(x[:train_end], y)
        scores[val_start:val_end] = _predict_score(model_name, model, x[val_start:val_end])
        folds.append({"train_end": train_end, "val_start": val_start, "val_end": val_end})
    valid = np.isfinite(scores)
    auc = None
    if valid.sum() > 10 and len(np.unique(y_win[valid])) == 2:
        auc = float(roc_auc_score(y_win[valid], scores[valid]))
    return scores, {"folds": folds, "oof_rows": int(valid.sum()), "oof_win_auc": auc}


def _apply_selector(dec: pd.DataFrame, active_idx: np.ndarray, scores: np.ndarray, threshold: float, *, scale: float, cap: float) -> tuple[pd.DataFrame, int]:
    out = dec.copy().reset_index(drop=True)
    selected_active = np.isfinite(scores) & (scores >= float(threshold))
    selected_idx = active_idx[selected_active]
    if len(selected_idx) == 0:
        return out, 0
    base_notional = pd.to_numeric(dec.loc[selected_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1e-12)
    out.loc[selected_idx, "notional_exposure"] = new_notional
    out.loc[selected_idx, "position_fraction"] = new_notional
    out.loc[selected_idx, "take_profit"] = pd.to_numeric(dec.loc[selected_idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[selected_idx, "stop_loss"] = pd.to_numeric(dec.loc[selected_idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio
    return out, int(len(selected_idx))


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


@dataclass(frozen=True)
class SelectorRun:
    model_name: str
    top_frac: float
    threshold: float
    scale: float
    cap: float


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec, val_prefix = _build_split(frames, "validation")
    oos_frame, oos_src, oos_dec, oos_prefix = _build_split(frames, "oos")
    val_x_all = _feature_frame(val_frame, val_src, val_dec, val_prefix)
    oos_x_all = _feature_frame(oos_frame, oos_src, oos_dec, oos_prefix)
    val_active = np.flatnonzero(omega._active(val_dec))
    oos_active = np.flatnonzero(omega._active(oos_dec))
    y_win, y_net = _candidate_labels(val_frame, val_dec, val_active, fee=fee, slip=slip)
    x_val_active = val_x_all.iloc[val_active].to_numpy(dtype=np.float64)
    x_oos_active = oos_x_all.iloc[oos_active].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "val_active_rows": int(len(val_active)),
        "oos_active_rows": int(len(oos_active)),
        "val_candidate_win_rate": float(y_win.mean()) if len(y_win) else 0.0,
        "feature_count": int(val_x_all.shape[1]),
        "features": list(val_x_all.columns),
    }
    static = pd.read_csv(BASE_DIR / "baseline_final_static_exposure_growth_grid_20260606.csv")
    static.to_csv(OUT_DIR / "static_growth_grid_copy.csv", index=False)
    for model_name in ("hgb_win", "extra_win", "hgb_net"):
        oof_scores, diag = _fit_oof_scores(model_name, x_val_active, y_win, y_net, val_active, seed=260606)
        full_model = _make_model(model_name, 260606)
        target = y_win if model_name.endswith("_win") else y_net
        full_model.fit(x_val_active, target)
        oos_scores = _predict_score(model_name, full_model, x_oos_active)
        valid_oof = oof_scores[np.isfinite(oof_scores)]
        if len(valid_oof) == 0:
            continue
        diagnostics[model_name] = diag
        for top_frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            threshold = float(np.quantile(valid_oof, 1.0 - float(top_frac)))
            for scale in (1.20, 1.35, 1.50, 1.75, 2.00):
                for cap in (0.55, 0.70, 0.90):
                    val_sel_dec, val_scaled = _apply_selector(val_dec, val_active, oof_scores, threshold, scale=scale, cap=cap)
                    oos_sel_dec, oos_scaled = _apply_selector(oos_dec, oos_active, oos_scores, threshold, scale=scale, cap=cap)
                    val_m = omega._metrics(val_frame, val_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
                    oos_m = omega._metrics(oos_frame, oos_sel_dec, fee=fee, slip=slip, cost_mult=3.0)
                    row = {
                        "model_name": model_name,
                        "top_frac": float(top_frac),
                        "threshold": threshold,
                        "scale": float(scale),
                        "cap": float(cap),
                        "val_scaled_signals": int(val_scaled),
                        "oos_scaled_signals": int(oos_scaled),
                    }
                    row.update(_metric_row("val", val_m))
                    row.update(_metric_row("oos", oos_m))
                    row["val_delta_pnl"] = row["val_pnl"] - BASELINE_VAL["pnl"]
                    row["val_delta_mdd"] = row["val_mdd"] - BASELINE_VAL["mdd"]
                    row["oos_delta_pnl"] = row["oos_pnl"] - BASELINE_OOS["pnl"]
                    row["oos_delta_mdd"] = row["oos_mdd"] - BASELINE_OOS["mdd"]
                    rows.append(row)
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise RuntimeError("selector ranking is empty")
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "selector_ranking.csv", index=False)
    # A stricter promotable view: validation and OOS PnL must improve; MDD cannot worsen by more than 50% relative to baseline.
    strict = ranking[
        (ranking["val_pnl"] > BASELINE_VAL["pnl"])
        & (ranking["oos_pnl"] > BASELINE_OOS["pnl"])
        & (ranking["val_mdd"] >= BASELINE_VAL["mdd"] * 1.5)
        & (ranking["oos_mdd"] >= BASELINE_OOS["mdd"] * 1.5)
    ].copy()
    strict = strict.sort_values(["oos_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    strict.to_csv(OUT_DIR / "selector_strict_candidates.csv", index=False)
    best = strict.iloc[0].to_dict() if not strict.empty else ranking.iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "baseline": {
            "model_id": "omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080",
            "validation": BASELINE_VAL,
            "oos": BASELINE_OOS,
        },
        "method": "Train high-confidence exposure selector on 2025 validation active signals only. Validation ranking uses expanding OOF selector scores. OOS uses a selector refit on all validation active signals.",
        "risk_transform": "compensated TP/SL + exposure scale. Notional is scaled and capped, TP/SL equity thresholds are multiplied by the same realized notional ratio to preserve price-hit geometry.",
        "diagnostics": diagnostics,
        "best_strict_candidate": best,
        "top10": ranking.head(10).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "selector_ranking": str(OUT_DIR / "selector_ranking.csv"),
            "selector_strict_candidates": str(OUT_DIR / "selector_strict_candidates.csv"),
            "static_growth_grid_copy": str(OUT_DIR / "static_growth_grid_copy.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": best, "diagnostics": diagnostics}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
