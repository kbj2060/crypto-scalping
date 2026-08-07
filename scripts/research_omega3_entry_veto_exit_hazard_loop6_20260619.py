#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


MODEL_ID = "omega3_entry_veto_exit_hazard_loop6_20260619"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_BUNDLE = full_parent.PARENT_DIR / "true_3head_tabm_bundle.pt"
TRAIN_PRED_CACHE = OUT_DIR / "train_predictions_2025_true3head.csv"
PREFIX_TRAIN_VAL = "omega1_regime3_expertdq_oof_"
PREFIX_OOS = "omega1_regime3_expertdq_"
CURRENT = {
    "validation": {"pnl": 100.542729421, "mdd": -10.677653, "trades": 33, "wr": 0.636364},
    "oos": {"pnl": 72.760041481, "mdd": -8.108171, "trades": 18, "wr": 0.722222},
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


def _predict_frame(bundle: dict[str, Any], frame: pd.DataFrame, prefix: str, device: torch.device) -> pd.DataFrame:
    x = threehead._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: threehead._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    return threehead._prediction_output(frame, direction, quality, threshold=0.0, prefix=prefix.rstrip("_"))


def _read_split(frames: dict[str, pd.DataFrame], split: str, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, bool]:
    if split == "train":
        frame = frames["train_raw"].reset_index(drop=True)
        if TRAIN_PRED_CACHE.exists():
            pred = pd.read_csv(TRAIN_PRED_CACHE, parse_dates=["timestamp"])
        else:
            bundle = torch.load(PARENT_BUNDLE, map_location=device, weights_only=False)
            pred = _predict_frame(bundle, frame, PREFIX_TRAIN_VAL, device)
            TRAIN_PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(TRAIN_PRED_CACHE, index=False)
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_TRAIN_VAL
        oof = True
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred = pd.read_csv(full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"])
        prefix = PREFIX_OOS
        oof = False
    else:
        raise RuntimeError(f"unknown split: {split}")
    src = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if src.isna().any().any():
        bad = src.loc[src.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"{split} prediction alignment produced NaN: {bad}")
    dec0 = overlay._build_dec(src, prefix, oof=oof)
    features = sleeve._extra_features(base_features._feature_frame(frame, src, dec0, prefix), dec0)
    bad_cols = [
        c
        for c in features.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_cols:
        raise RuntimeError(f"{split} forbidden features: {bad_cols[:20]}")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame, src, dec0, features, prefix, oof


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _fill_price(arrays: dict[str, np.ndarray], idx: int, side: int, slip_eff: float, *, entry: bool) -> float:
    px = float(arrays["open"][int(np.clip(idx, 0, len(arrays["open"]) - 1))])
    if side > 0:
        return px * (1.0 + slip_eff if entry else 1.0 - slip_eff)
    return px * (1.0 - slip_eff if entry else 1.0 + slip_eff)


def _raw_unreal(arrays: dict[str, np.ndarray], entry_price: float, side: int, idx: int, slip_eff: float) -> float:
    px = float(arrays["close"][int(idx)])
    if side > 0:
        return float((px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12))
    return float((entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12))


def _entry_labels(
    frame: pd.DataFrame,
    dec0: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    horizon: int,
    max_rows: int,
    seed: int,
    mae_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    notionals = np.asarray([0.0, 0.45, 0.75, 1.05, 1.35], dtype=np.float64)
    arrays = _arrays(frame)
    active = np.flatnonzero(omega._active(dec0))
    if len(active) > max_rows:
        active = np.sort(np.random.default_rng(seed).choice(active, size=max_rows, replace=False))
    slip_eff = float(slip) * float(cost_mult)
    fee_eff = float(fee) * float(cost_mult)
    y = np.zeros(len(active), dtype=np.int64)
    label_u = np.zeros(len(active), dtype=np.float64)
    for k, i in enumerate(active):
        if k % 1000 == 0:
            print(json.dumps({"stage": "entry_labels", "seen": int(k), "total": int(len(active))}), flush=True)
        side = int(dec0.iloc[int(i)]["side"])
        entry_i = min(int(i) + 1, len(frame) - 1)
        end = min(entry_i + int(horizon), len(frame) - 2)
        entry = _fill_price(arrays, entry_i, side, slip_eff, entry=True)
        raw = np.asarray([_raw_unreal(arrays, entry, side, j, slip_eff) for j in range(entry_i, end + 1)], dtype=np.float64)
        if len(raw) == 0:
            continue
        best = int(np.argmax(raw - float(mae_penalty) * np.maximum(0.0, -raw)))
        exit_raw = float(raw[best])
        mae = float(np.min(raw[: best + 1]))
        best_j = 0
        best_u = 0.0
        for j, n in enumerate(notionals[1:], start=1):
            net = exit_raw * n - fee_eff * n * 2.0
            u = net - float(mae_penalty) * max(0.0, -mae * n) - 0.0025 * max(0.0, n - 1.05)
            if u > best_u:
                best_u = float(u)
                best_j = int(j)
        y[k] = best_j
        label_u[k] = best_u
    diag = {
        "rows": int(len(active)),
        "skip_rate": float(np.mean(y == 0)) if len(y) else 0.0,
        "label_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "notional_buckets": notionals.tolist(),
        "utility_mean": float(np.mean(label_u)) if len(label_u) else 0.0,
    }
    return active.astype(np.int64), y, diag


def _fit_entry_model(x_train: pd.DataFrame, idx: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    if len(np.unique(y)) < 2:
        raise RuntimeError("entry risk labels are single-class")
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=12,
        l2_regularization=0.8,
        min_samples_leaf=25,
        random_state=seed,
    )
    model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y)
    return model


def _predict_entry(model: HistGradientBoostingClassifier, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x)), best].astype(np.float64)


def _apply_entry_risk(dec0: pd.DataFrame, bucket: np.ndarray, conf: np.ndarray, *, min_conf: float, cap: float) -> pd.DataFrame:
    notionals = np.asarray([0.0, 0.45, 0.75, 1.05, 1.35], dtype=np.float64)
    out = dec0.copy().reset_index(drop=True)
    active = np.flatnonzero(omega._active(out))
    b = np.asarray(bucket, dtype=np.int64)
    c = np.asarray(conf, dtype=np.float64)
    for i in active:
        if c[int(i)] < float(min_conf):
            n = 0.0
        else:
            n = float(notionals[int(np.clip(b[int(i)], 0, len(notionals) - 1))])
        n = min(n, float(cap))
        if n <= 0.0:
            out.loc[int(i), ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss"]] = [omega.ACTION_CASH, 0, 0.0, 0.0, 0.0, 0.0]
        else:
            out.loc[int(i), "notional_exposure"] = n
            out.loc[int(i), "position_fraction"] = n / 2.0
            out.loc[int(i), "leverage"] = 2.0
            out.loc[int(i), "take_profit"] = 0.0
            out.loc[int(i), "stop_loss"] = 0.0
            out.loc[int(i), "max_hold_bars"] = 0
            out.loc[int(i), "cooldown_bars"] = 0
    return out


def _position_feature_row(base_x: pd.DataFrame, i: int, *, side: int, hold: int, unreal: float, mfe: float, mae: float, notional: float) -> dict[str, float]:
    row = base_x.iloc[int(i)]
    giveback = max(0.0, float(mfe) - float(unreal))
    out: dict[str, float] = {
        "side": float(side),
        "hold_bars": float(hold),
        "hold_log1p": float(np.log1p(max(hold, 0))),
        "unreal": float(unreal),
        "mfe": float(mfe),
        "mae": float(mae),
        "giveback": float(giveback),
        "giveback_ratio": float(giveback / max(abs(mfe), 1e-8)) if mfe > 0 else 0.0,
        "notional": float(notional),
        "unreal_per_notional": float(unreal / max(notional, 1e-8)),
    }
    for col in (
        "tabm_quality_for_action",
        "tabm_router_confidence",
        "tabm_router_margin",
        "tabm_dir_confidence",
        "tabm_dir_side_edge",
        "tabm_dir_trade_prob",
        "atr14_pct",
        "bar_range_pct",
        "ema9_21_gap",
    ):
        out[col] = float(row.get(col, 0.0))
    return out


def _collect_exit_rows(
    frame: pd.DataFrame,
    dec_entry: pd.DataFrame,
    base_x: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_trades: int,
    horizon: int,
    label_fwd: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    arrays = _arrays(frame)
    active = np.flatnonzero(omega._active(dec_entry))
    slip_eff = float(slip) * float(cost_mult)
    rows: list[dict[str, float]] = []
    trades = 0
    for i in active:
        if trades >= int(max_trades):
            break
        side = int(dec_entry.iloc[int(i)]["side"])
        notional = float(dec_entry.iloc[int(i)]["notional_exposure"])
        if side == 0 or notional <= 0:
            continue
        entry_i = min(int(i) + 1, len(frame) - 2)
        entry = _fill_price(arrays, entry_i, side, slip_eff, entry=True)
        end = min(entry_i + int(horizon), len(frame) - 2)
        mfe = 0.0
        mae = 0.0
        trades += 1
        for j in range(entry_i, end + 1):
            raw = _raw_unreal(arrays, entry, side, j, slip_eff)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            f_end = min(j + int(label_fwd), len(frame) - 2)
            f_raw = np.asarray([_raw_unreal(arrays, entry, side, k, slip_eff) * notional for k in range(j, f_end + 1)], dtype=np.float64)
            if len(f_raw) <= 1:
                edge = 0.0
                y_exit = 1
                worst = float(f_raw[0]) if len(f_raw) else 0.0
            else:
                mfe_path = np.maximum.accumulate(np.maximum(f_raw, mfe))
                giveback = np.maximum(0.0, mfe_path - f_raw)
                utility = f_raw - 0.35 * giveback - 0.00004 * np.arange(len(f_raw))
                edge = float(np.max(utility[1:]) - utility[0])
                y_exit = int((j - entry_i) >= 2 and edge <= 0.0015)
                worst = float(np.min(f_raw))
            rows.append({**_position_feature_row(base_x, j, side=side, hold=j - entry_i, unreal=unreal, mfe=mfe, mae=mae, notional=notional), "y_exit": y_exit, "edge_hold": edge, "future_worst": worst})
    df = pd.DataFrame(rows)
    if df.empty or df["y_exit"].nunique() < 2:
        raise RuntimeError("exit hazard dataset is empty or single-class")
    diag = {
        "rows": int(len(df)),
        "trades": int(trades),
        "positive_rate": float(df["y_exit"].mean()),
        "edge_mean": float(df["edge_hold"].mean()),
    }
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0), diag


def _train_exit_models(df: pd.DataFrame, seed: int, *, fast: bool = False) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingRegressor, list[str], dict[str, Any]]:
    drop = {"y_exit", "edge_hold", "future_worst"}
    cols = [c for c in df.columns if c not in drop]
    x = df[cols].to_numpy(dtype=np.float64)
    y = df["y_exit"].astype(int).to_numpy()
    max_iter = 70 if fast else 160
    reg_iter = 55 if fast else 120
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=35, l2_regularization=1.0, random_state=seed)
    edge = HistGradientBoostingRegressor(max_iter=reg_iter, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=35, l2_regularization=1.0, random_state=seed + 1)
    worst = HistGradientBoostingRegressor(max_iter=reg_iter, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=35, l2_regularization=1.0, random_state=seed + 2)
    clf.fit(x, y)
    edge.fit(x, df["edge_hold"].to_numpy(dtype=np.float64))
    worst.fit(x, df["future_worst"].to_numpy(dtype=np.float64))
    return clf, edge, worst, cols, {"feature_count": int(len(cols)), "positive_rate": float(y.mean())}


def _simulate(
    frame: pd.DataFrame,
    dec_entry: pd.DataFrame,
    base_x: pd.DataFrame,
    clf: HistGradientBoostingClassifier,
    edge: HistGradientBoostingRegressor,
    worst: HistGradientBoostingRegressor,
    feature_cols: list[str],
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    p_min: float,
    edge_max: float,
    worst_min: float,
    min_hold: int,
    exit_eval_stride: int,
) -> dict[str, Any]:
    arrays = _arrays(frame)
    active = omega._active(dec_entry)
    slip_eff = float(slip) * float(cost_mult)
    fee_eff = float(fee) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry = 0.0
    entry_i = 0
    entry_eq = 1.0
    notional = 0.0
    mfe = 0.0
    mae = 0.0
    trades: list[float] = []
    reasons: dict[str, int] = {}
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            raw = _raw_unreal(arrays, entry, pos, i, slip_eff)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            hold = int(i) - int(entry_i)
            reason = ""
            if hold >= int(min_hold) and (hold - int(min_hold)) % max(int(exit_eval_stride), 1) == 0:
                feat = _position_feature_row(base_x, i, side=pos, hold=hold, unreal=unreal, mfe=mfe, mae=mae, notional=notional)
                x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
                p = float(clf.predict_proba(x)[0, list(clf.classes_).index(1)])
                e = float(edge.predict(x)[0])
                w = float(worst.predict(x)[0])
                if p >= float(p_min) and e <= float(edge_max) and w >= float(worst_min):
                    reason = "hazard_exit"
            if reason:
                exit_px = _fill_price(arrays, i, pos, slip_eff, entry=False)
                raw_exit = (exit_px - entry) / max(entry, 1e-12) if pos > 0 else (entry - exit_px) / max(entry, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trades.append(cash / max(entry_eq, 1e-12) - 1.0)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                continue
        else:
            peak = max(peak, cash)
            mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec_entry.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        n = float(row.get("notional_exposure", 0.0) or 0.0)
        if side == 0 or n <= 0:
            continue
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry = _fill_price(arrays, entry_i, side, slip_eff, entry=True)
        entry_eq = cash
        cash -= cash * fee_eff * n
        pos = side
        notional = n
        mfe = 0.0
        mae = 0.0
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
    if pos != 0:
        exit_px = _fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry) / max(entry, 1e-12) if pos > 0 else (entry - exit_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades.append(cash / max(entry_eq, 1e-12) - 1.0)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    t = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(t)),
        "wr": float(np.mean(t > 0.0)) if len(t) else 0.0,
        "avg_trade": float(np.mean(t) * 100.0) if len(t) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def _row(name: str, vm: dict[str, Any], om: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": name, **extra}
    for prefix, m in (("val", vm), ("oos", om)):
        for k in ("pnl", "mdd", "trades", "wr", "avg_trade", "long_entries", "short_entries"):
            row[f"{prefix}_{k}"] = m[k]
        row[f"{prefix}_reasons"] = m["exit_reasons"]
    row["val_delta_vs_current"] = float(row["val_pnl"] - CURRENT["validation"]["pnl"])
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - CURRENT["oos"]["pnl"])
    row["validation_only_score"] = float(row["val_pnl"] + row["val_mdd"] - 0.05 * max(0, row["val_trades"] - CURRENT["validation"]["trades"]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry-label-rows", type=int, default=8000)
    ap.add_argument("--max-exit-trades", type=int, default=700)
    ap.add_argument("--seed", type=int, default=260624)
    ap.add_argument("--fast-grid", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()

    train_frame, _train_src, train_dec0, x_train, _p, _o = _read_split(frames, "train", device)
    val_frame, _val_src, val_dec0, x_val, _vp, _vo = _read_split(frames, "validation", device)
    oos_frame, _oos_src, oos_dec0, x_oos, _op, _oo = _read_split(frames, "oos", device)

    idx, y, entry_label_diag = _entry_labels(
        train_frame,
        train_dec0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        horizon=144,
        max_rows=int(args.max_entry_label_rows),
        seed=int(args.seed),
        mae_penalty=0.85,
    )
    entry_model = _fit_entry_model(x_train, idx, y, int(args.seed))
    train_bucket, train_conf = _predict_entry(entry_model, x_train)
    val_bucket, val_conf = _predict_entry(entry_model, x_val)
    oos_bucket, oos_conf = _predict_entry(entry_model, x_oos)

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    conf_grid = (0.0,) if bool(args.fast_grid) else (0.0, 0.35, 0.45, 0.55)
    p_grid = (0.20, 0.35) if bool(args.fast_grid) else (0.20, 0.35, 0.50, 0.65, 0.75)
    edge_grid = (0.0080, 0.0150) if bool(args.fast_grid) else (0.0005, 0.0015, 0.0030, 0.0080, 0.0150)
    worst_grid = (-0.40,) if bool(args.fast_grid) else (-0.50, -0.30, -0.18, -0.12, -0.08)
    hold_grid = (2,) if bool(args.fast_grid) else (2, 6, 12)
    stride_grid = (12,) if bool(args.fast_grid) else (1, 3, 6, 12)
    for conf_min in conf_grid:
        train_entry = _apply_entry_risk(train_dec0, train_bucket, train_conf, min_conf=conf_min, cap=1.35)
        val_entry = _apply_entry_risk(val_dec0, val_bucket, val_conf, min_conf=conf_min, cap=1.35)
        oos_entry = _apply_entry_risk(oos_dec0, oos_bucket, oos_conf, min_conf=conf_min, cap=1.35)
        exit_df, exit_diag = _collect_exit_rows(
            train_frame,
            train_entry,
            x_train,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            max_trades=int(args.max_exit_trades),
            horizon=192,
            label_fwd=48,
        )
        clf, edge, worst, cols, exit_model_diag = _train_exit_models(exit_df, int(args.seed) + int(conf_min * 1000) + 11, fast=bool(args.fast_grid))
        reports[f"conf{conf_min:g}"] = {"exit_dataset": exit_diag, "exit_model": exit_model_diag}
        for p_min in p_grid:
            for edge_max in edge_grid:
                for worst_min in worst_grid:
                    for min_hold in hold_grid:
                        for stride in stride_grid:
                            vm = _simulate(val_frame, val_entry, x_val, clf, edge, worst, cols, fee=fee, slip=slip, cost_mult=3.0, p_min=p_min, edge_max=edge_max, worst_min=worst_min, min_hold=min_hold, exit_eval_stride=stride)
                            om = _simulate(oos_frame, oos_entry, x_oos, clf, edge, worst, cols, fee=fee, slip=slip, cost_mult=3.0, p_min=p_min, edge_max=edge_max, worst_min=worst_min, min_hold=min_hold, exit_eval_stride=stride)
                            rows.append(_row(f"conf{conf_min:g}_p{p_min:g}_e{edge_max:g}_w{worst_min:g}_mh{min_hold}_st{stride}", vm, om, {"entry_conf_min": conf_min, "p_min": p_min, "edge_max": edge_max, "worst_min": worst_min, "min_hold": min_hold, "exit_eval_stride": stride}))

    ranking = pd.DataFrame(rows).sort_values(["validation_only_score", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Train-only learned risk management. Entry model predicts SKIP/notional bucket; in-position hazard model exits using current PnL path and market/parent features. Candidate decisions use no TP/SL price barrier thresholds.",
        "comparison_baseline": CURRENT,
        "entry_label_diag": entry_label_diag,
        "entry_model_classes": [int(x) for x in entry_model.classes_],
        "exit_reports": reports,
        "selected_by_validation": ranking.iloc[0].to_dict(),
        "best_oos_diagnostic": ranking.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).iloc[0].to_dict(),
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv"), "train_pred_cache": str(TRAIN_PRED_CACHE)},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected_by_validation"], "best_oos": report["best_oos_diagnostic"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
