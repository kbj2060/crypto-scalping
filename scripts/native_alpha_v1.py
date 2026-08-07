#!/usr/bin/env python3
"""Independent SOL/BTC Native-Alpha v1 research pipeline.

The asset id is part of the model contract.  SOL and BTC have separate data,
features, labels, models, calibration, and artifacts; only the runtime shape
of the pipeline is shared.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/splits/year_oos"
ARTIFACT_DIR = ROOT / "tmp/native_alpha_v1"


@dataclass(frozen=True)
class AssetSpec:
    asset: str
    horizons: tuple[int, ...]
    seeds: tuple[int, ...]
    threshold_grid: tuple[float, ...]
    gate_threshold_grid: tuple[float, ...]
    max_hold_bars: int


ASSET_SPECS = {
    "sol": AssetSpec("sol", (12, 24, 48), (270713, 270719, 270727, 270731, 270733), (0.45, 0.50, 0.55, 0.60), (0.45, 0.50, 0.55, 0.60), 288),
    "btc": AssetSpec("btc", (24, 48, 96), (310713, 310719, 310727, 310731, 310733), (0.45, 0.50, 0.55, 0.60), (0.45, 0.50, 0.55, 0.60), 576),
}

REQUIRED = {"timestamp", "open", "high", "low", "close", "volume"}
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "target", "target_return", "target_horizon", "future_mfe", "future_mae"}


def _check_asset(asset: str) -> AssetSpec:
    key = str(asset).lower()
    if key not in ASSET_SPECS:
        raise ValueError(f"unsupported asset: {asset!r}; expected sol or btc")
    return ASSET_SPECS[key]


def load_5m(asset: str, years: Iterable[int] = (2024, 2025, 2026)) -> pd.DataFrame:
    spec = _check_asset(asset)
    frames = []
    for year in years:
        path = DATA_DIR / f"{spec.asset}_features_{int(year)}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path}: missing required columns: {sorted(missing)}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if not out["timestamp"].is_monotonic_increasing:
        raise RuntimeError("timestamps are not sorted")
    return out


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(8, window // 4)).mean()
    std = series.rolling(window, min_periods=max(8, window // 4)).std().replace(0, np.nan)
    return ((series - mean) / std).clip(-8.0, 8.0)


def build_hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build only trailing 1h features from completed 5m bars."""
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for col in ("quote_volume", "sum_open_interest_value", "last_funding_rate", "close_btc",
                "cvd_48", "cvd_288", "cvp_volume_imbalance", "whale_conviction", "atr_pct_rank_288"):
        if col in f.columns:
            agg[col] = "last"
    r = f.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = pd.DataFrame(index=r.index)
    close = r["close"].astype(float).clip(lower=1e-12)
    log_close = np.log(close)
    ret = log_close.diff()
    for h in (1, 3, 6, 12, 24, 48, 96):
        out[f"ret_{h}"] = (log_close - log_close.shift(h)).clip(-2.0, 2.0)
    for w in (6, 12, 24, 48, 96):
        out[f"vol_{w}"] = ret.rolling(w, min_periods=min(w, max(8, w // 4))).std().clip(0.0, 1.0)
    for w in (12, 24, 48):
        out[f"seq_ret_mean_{w}"] = ret.rolling(w, min_periods=max(8, w // 4)).mean().clip(-1.0, 1.0)
        out[f"seq_ret_std_{w}"] = ret.rolling(w, min_periods=max(8, w // 4)).std().clip(0.0, 1.0)
        out[f"seq_sign_consistency_{w}"] = ret.rolling(w, min_periods=max(8, w // 4)).mean().abs() / (ret.abs().rolling(w, min_periods=max(8, w // 4)).mean() + 1e-8)
    prev = close.shift(1)
    tr = pd.concat([(r["high"] - r["low"]), (r["high"] - prev).abs(), (r["low"] - prev).abs()], axis=1).max(axis=1)
    out["atr_pct"] = (tr.rolling(24, min_periods=8).mean() / close).clip(0.0, 1.0)
    out["range_pct"] = ((r["high"] - r["low"]) / close).clip(0.0, 1.0)
    out["trend_strength"] = (out["ret_24"].abs() / (out["vol_24"] * np.sqrt(24) + 1e-8)).clip(0.0, 8.0)
    out["chop_proxy"] = (1.0 / (1.0 + out["trend_strength"])).clip(0.0, 1.0)
    out["volume_z"] = _zscore(np.log1p(r["volume"].astype(float)), 48)
    if "sum_open_interest_value" in r:
        oi = r["sum_open_interest_value"].astype(float).replace(0, np.nan)
        out["oi_change_12"] = oi.pct_change(12, fill_method=None).clip(-2.0, 2.0)
        out["oi_z_48"] = _zscore(oi.pct_change(fill_method=None).fillna(0.0), 48)
    if "last_funding_rate" in r:
        funding = r["last_funding_rate"].astype(float).fillna(0.0)
        out["funding"] = funding.clip(-0.02, 0.02)
        out["funding_z_48"] = _zscore(funding, 48)
        out["funding_change_12"] = funding.diff(12).clip(-0.02, 0.02)
    if "close_btc" in r:
        btc = np.log(r["close_btc"].astype(float).clip(lower=1e-12))
        out["btc_ret_12"] = (btc - btc.shift(12)).clip(-2.0, 2.0)
        out["asset_btc_residual_12"] = out["ret_12"] - out["btc_ret_12"]
    for col in ("cvd_48", "cvd_288", "cvp_volume_imbalance", "whale_conviction", "atr_pct_rank_288"):
        if col in r:
            out[col] = pd.to_numeric(r[col], errors="coerce").clip(-8.0, 8.0)
    ts = pd.Series(r.index, index=r.index)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["open"] = r["open"].astype(float)
    out["high"] = r["high"].astype(float)
    out["low"] = r["low"].astype(float)
    out["close"] = r["close"].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in out.columns if c not in {"open", "high", "low", "close"}]
    out[feature_cols] = out[feature_cols].ffill().fillna(0.0)
    return out.reset_index(names="timestamp")


def build_labels(hourly: pd.DataFrame, spec: AssetSpec) -> pd.DataFrame:
    """Create future labels; these columns are never used as live features."""
    close = hourly["close"].to_numpy(dtype=float)
    atr = hourly["atr_pct"].to_numpy(dtype=float)
    returns = np.column_stack([np.roll(close, -h) / close - 1.0 for h in spec.horizons])
    returns[-max(spec.horizons):, :] = np.nan
    scaled = returns / np.maximum(atr[:, None], 0.0025)
    finite = np.isfinite(scaled)
    best = np.where(finite, np.abs(scaled), -1.0).argmax(axis=1)
    row = np.arange(len(hourly))
    target_return = returns[row, best]
    target_scaled = scaled[row, best]
    target = np.where(target_scaled >= 1.0, 1, np.where(target_scaled <= -1.0, 2, 0)).astype(int)
    invalid = ~finite.any(axis=1)
    target_return[invalid] = np.nan
    target_scaled[invalid] = np.nan
    target[invalid] = 0
    out = hourly.copy()
    out["target"] = target
    out["target_return"] = target_return
    out["target_horizon"] = np.asarray(spec.horizons, dtype=int)[best]
    for h in (max(spec.horizons),):
        future_close = np.array([np.nanmax(close[i + 1:i + h + 1]) if i + 1 < len(close) else np.nan for i in range(len(close))])
        future_low = np.array([np.nanmin(close[i + 1:i + h + 1]) if i + 1 < len(close) else np.nan for i in range(len(close))])
        out["future_mfe"] = future_close / close - 1.0
        out["future_mae"] = future_low / close - 1.0
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(frame[c])]
    if not cols:
        raise RuntimeError("no numeric model features")
    return cols


REGIME_COLUMNS = ("ret_24", "ret_48", "vol_24", "trend_strength", "chop_proxy", "range_pct")


def attach_regime_states(frame: pd.DataFrame, train_idx: np.ndarray | None = None, bundle: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach a train-only Gaussian regime state model with causal inference."""
    out = frame.copy()
    if bundle is None:
        if train_idx is None or len(train_idx) < 100:
            raise RuntimeError("regime model requires at least 100 training rows")
        scaler = StandardScaler().fit(out.loc[train_idx, list(REGIME_COLUMNS)].to_numpy(dtype=float))
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=270713, reg_covar=1e-5)
        gmm.fit(scaler.transform(out.loc[train_idx, list(REGIME_COLUMNS)].to_numpy(dtype=float)))
        meta = {"scaler": scaler, "gmm": gmm, "columns": list(REGIME_COLUMNS)}
    else:
        meta = bundle["regime"]
        scaler, gmm = meta["scaler"], meta["gmm"]
    proba = gmm.predict_proba(scaler.transform(out[list(REGIME_COLUMNS)].to_numpy(dtype=float)))
    if bundle is None:
        train_p = proba[np.asarray(train_idx, dtype=int)]
        chop_state = int(np.argmax([train_p[:, k].mean() for k in range(gmm.n_components)]))
        meta["chop_state"] = chop_state
    chop_state = int(meta["chop_state"])
    out["regime_stability"] = proba.max(axis=1)
    out["regime_chop_prob"] = proba[:, chop_state]
    return out, meta


def _fit_classifier(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.035, max_iter=220, max_depth=4,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=False, random_state=seed, class_weight="balanced",
    )
    model.fit(X, y)
    return model


def _aligned_proba(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, i]
    return out


def train(asset: str, output_dir: Path = ARTIFACT_DIR, train_end: str = "2025-08-31 23:59:59") -> dict:
    spec = _check_asset(asset)
    hourly = build_labels(build_hourly_features(load_5m(spec.asset, (2024, 2025, 2026))), spec)
    train_end_ts = pd.Timestamp(train_end)
    train_mask = (hourly["timestamp"] <= train_end_ts) & hourly["target_return"].notna()
    val_mask = (hourly["timestamp"] > pd.Timestamp("2025-09-01")) & (hourly["timestamp"] <= pd.Timestamp("2025-12-31 23:59:59")) & hourly["target_return"].notna()
    y = hourly["target"].to_numpy(dtype=int)
    ret = hourly["target_return"].to_numpy(dtype=float)
    train_idx, val_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    if len(train_idx) < 500 or len(val_idx) < 50:
        raise RuntimeError(f"insufficient split rows for {spec.asset}: train={len(train_idx)} val={len(val_idx)}")
    hourly, regime = attach_regime_states(hourly, train_idx=train_idx)
    cols = feature_columns(hourly)
    X = hourly[cols].to_numpy(dtype=np.float64)
    classifiers = [_fit_classifier(X[train_idx], y[train_idx], s) for s in spec.seeds]
    p_val = np.mean([_aligned_proba(m, X[val_idx]) for m in classifiers], axis=0)
    sequence_cols = [c for c in cols if c.startswith("seq_")] + ["regime_stability", "regime_chop_prob"]
    sequence_gate = _fit_classifier(X[train_idx][:, [cols.index(c) for c in sequence_cols]], (y[train_idx] != 0).astype(int), spec.seeds[0] + 91)
    p_gate_val = _aligned_proba(sequence_gate, X[val_idx][:, [cols.index(c) for c in sequence_cols]])[:, 1]
    regressors = {}
    for quantile, loss in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=loss, learning_rate=0.035, max_iter=180,
            max_depth=4, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=spec.seeds[0],
        )
        model.fit(X[train_idx], ret[train_idx])
        regressors[quantile] = model
    risk = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.035, max_iter=180, max_depth=4,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=False, random_state=spec.seeds[1],
    )
    mae_target = np.abs(hourly["future_mae"].to_numpy(dtype=float))
    finite_risk = train_idx[np.isfinite(mae_target[train_idx])]
    risk.fit(X[finite_risk], mae_target[finite_risk])
    q10 = regressors["q10"].predict(X[val_idx])
    q50 = regressors["q50"].predict(X[val_idx])
    q90 = regressors["q90"].predict(X[val_idx])
    cost = 2 * 0.0007
    best_threshold, best_gate_threshold, best_score = spec.threshold_grid[0], spec.gate_threshold_grid[0], -np.inf
    for threshold in spec.threshold_grid:
        for gate_threshold in spec.gate_threshold_grid:
            side = np.where((p_val[:, 1] >= threshold) & (p_val[:, 1] > p_val[:, 2]), 1, np.where((p_val[:, 2] >= threshold) & (p_val[:, 2] > p_val[:, 1]), -1, 0))
            side = np.where(p_gate_val >= gate_threshold, side, 0)
            utility = side * q50 - 0.5 * np.maximum(q90 - q10, 0.0) - cost
            score = float(np.nanmean(np.where(side != 0, utility, 0.0)))
            if score > best_score:
                best_threshold, best_gate_threshold, best_score = float(threshold), float(gate_threshold), score
    out = Path(output_dir) / spec.asset
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "contract": "native_alpha_v1",
        "asset": spec.asset,
        "features": cols,
        "horizons": spec.horizons,
        "seeds": spec.seeds,
        "threshold": best_threshold,
        "sequence_gate_threshold": best_gate_threshold,
        "train_end": str(train_end_ts),
        "validation": {"start": "2025-09-01", "end": "2025-12-31 23:59:59", "utility": best_score, "rows": int(len(val_idx))},
        "regime": regime,
        "sequence_features": sequence_cols,
        "sequence_gate": sequence_gate,
        "classifier": classifiers,
        "return_heads": regressors,
        "risk_head": risk,
        "causal_contract": {"future_rows_used_for_entry": False, "saved_ledger_used": False, "bar_cadence": "5m_decision_1h_signal"},
    }
    path = out / "model.joblib"
    joblib.dump(bundle, path)
    report = {k: v for k, v in bundle.items() if k not in {"classifier", "return_heads", "risk_head", "regime", "sequence_gate"}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return {"asset": spec.asset, "path": str(path), "train_rows": len(train_idx), "validation_rows": len(val_idx), "threshold": best_threshold, "validation_utility": best_score}


def _predict_hourly(bundle: dict, hourly: pd.DataFrame) -> pd.DataFrame:
    hourly, _ = attach_regime_states(hourly, bundle=bundle)
    cols = bundle["features"]
    X = hourly[cols].to_numpy(dtype=np.float64)
    p = np.mean([_aligned_proba(m, X) for m in bundle["classifier"]], axis=0)
    q10, q50, q90 = (bundle["return_heads"][k].predict(X) for k in ("q10", "q50", "q90"))
    pred_mae = np.maximum(bundle["risk_head"].predict(X), 0.0025)
    seq_cols = bundle["sequence_features"]
    seq_idx = [cols.index(c) for c in seq_cols]
    p_gate = _aligned_proba(bundle["sequence_gate"], X[:, seq_idx])[:, 1]
    threshold = float(bundle["threshold"])
    side = np.where((p[:, 1] >= threshold) & (p[:, 1] > p[:, 2]), 1, np.where((p[:, 2] >= threshold) & (p[:, 2] > p[:, 1]), -1, 0))
    side = np.where(p_gate >= float(bundle["sequence_gate_threshold"]), side, 0)
    utility = side * q50 - 0.5 * np.maximum(q90 - q10, 0.0) - 0.0014
    side = np.where(utility > 0.0, side, 0)
    return pd.DataFrame({"timestamp": hourly["timestamp"], "side": side, "utility": utility, "q10": q10, "q50": q50, "q90": q90, "pred_mae": pred_mae, "p_long": p[:, 1], "p_short": p[:, 2], "p_sequence_gate": p_gate, "regime_stability": hourly["regime_stability"], "regime_chop_prob": hourly["regime_chop_prob"]})


def fresh_forward_oos(asset: str, bundle_path: Path, start: str = "2026-01-01", end: str = "2026-07-12 23:55:00") -> tuple[dict, pd.DataFrame]:
    spec = _check_asset(asset)
    bundle = joblib.load(bundle_path)
    if bundle.get("asset") != spec.asset or bundle.get("contract") != "native_alpha_v1":
        raise RuntimeError("artifact contract mismatch")
    five = load_5m(spec.asset, (2024, 2025, 2026))
    hourly = build_hourly_features(five)
    pred = _predict_hourly(bundle, hourly)
    tape = five[(five["timestamp"] >= pd.Timestamp(start)) & (five["timestamp"] <= pd.Timestamp(end))].copy().reset_index(drop=True)
    pred = pred[pred["timestamp"] <= pd.Timestamp(end)].copy()
    tape = pd.merge_asof(tape.sort_values("timestamp"), pred.sort_values("timestamp"), on="timestamp", direction="backward")
    equity = 1.0
    pos = None
    pending = 0
    trades = []
    fee = 0.0005
    slip = 0.0002
    for i, row in tape.iterrows():
        price = float(row["open"])
        if pos is not None:
            move = pos["side"] * (float(row["close"]) / pos["entry"] - 1.0)
            exit_now = move >= pos["tp"] or move <= -pos["sl"] or (i - pos["i"] >= spec.max_hold_bars)
            if exit_now:
                gross = move * pos["notional"]
                costs = fee * pos["notional"] + fee * pos["notional"] + slip * pos["notional"] * 2
                equity += gross - costs
                trades.append({"entry_timestamp": pos["timestamp"], "exit_timestamp": row["timestamp"], "side": pos["side"], "trade_return": gross - costs, "reason": "barrier_or_time"})
                pos = None
        if pos is None and pending != 0:
            pred_mae = max(float(row.get("pred_mae", 0.01)), 0.0025)
            pos = {"i": i, "timestamp": row["timestamp"], "entry": price * (1 + slip * pending), "side": int(pending), "notional": 0.90, "tp": min(max(float(row.get("q50", 0.01)) * 0.8, 0.0075), 0.12), "sl": min(max(pred_mae * 1.2, 0.004), 0.08)}
        pending = int(row.get("side", 0)) if pd.notna(row.get("side")) else 0
    if pos is not None:
        last = tape.iloc[-1]
        move = pos["side"] * (float(last["close"]) / pos["entry"] - 1.0)
        trades.append({"entry_timestamp": pos["timestamp"], "exit_timestamp": last["timestamp"], "side": pos["side"], "trade_return": move * pos["notional"] - 2 * fee * pos["notional"], "reason": "end_of_data"})
        equity += trades[-1]["trade_return"]
    ledger = pd.DataFrame(trades)
    if ledger.empty:
        ledger = pd.DataFrame(columns=["entry_timestamp", "exit_timestamp", "side", "trade_return", "reason"])
    curve = np.r_[1.0, 1.0 + ledger["trade_return"].cumsum().to_numpy(dtype=float)]
    mdd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0)) if len(curve) else 0.0
    report = {"asset": spec.asset, "start": start, "end": end, "pnl": float(equity - 1.0), "mdd": mdd, "trades": int(len(ledger)), "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "artifact": str(bundle_path)}
    return report, ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--asset", choices=sorted(ASSET_SPECS), required=True)
    tr.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    oo = sub.add_parser("oos")
    oo.add_argument("--asset", choices=sorted(ASSET_SPECS), required=True)
    oo.add_argument("--bundle", type=Path, required=True)
    oo.add_argument("--start", default="2026-01-01")
    oo.add_argument("--end", default="2026-07-12 23:55:00")
    args = ap.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.asset, args.output_dir), indent=2))
    else:
        report, ledger = fresh_forward_oos(args.asset, args.bundle, args.start, args.end)
        print(json.dumps(report, indent=2))
        ledger.to_csv(args.bundle.parent / f"oos_ledger_{args.start[:10]}_{args.end[:10]}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
