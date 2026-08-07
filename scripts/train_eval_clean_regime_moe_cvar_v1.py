#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.high_order_state import add_high_order_state_features


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "clean_regime_moe_cvar_v1_20260511"
CLEAN_PREFIX = "clean_regime_v6_"

DEFAULT_STATE_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_TRAIN_2025 = ROOT / "data/splits/year_oos/rl_training_2025_m7.csv"
DEFAULT_EVAL_2026 = ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime_moe_cvar_v1_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime_moe_cvar_v1_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_regime_moe_cvar_v1_20260511_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_regime_moe_cvar_v1_20260511_contract.md"

BANNED_FRAGMENTS = (
    "legacy",
    "hdb",
    "hmm",
    "patchtst",
    "timesnet",
    "dlinear",
    "tide",
    "target",
    "future",
    "realized",
    "trade_pnl",
    "cash_after",
    "label",
    "edge",
)
BANNED_EXACT = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}


@dataclass(frozen=True)
class RuntimeConfig:
    threshold: float
    gap: float
    max_notional: float
    min_notional: float
    leverage: float
    max_hold_bars: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    cooldown_bars: int
    cvar_adverse_cap: float
    clean_conf_floor: float
    risk_off_cap: float
    candidate_stride: int


@dataclass
class Position:
    side: int
    signal_idx: int
    entry_idx: int
    entry_price: float
    notional: float
    leverage: float
    prob: float
    gap: float
    predicted_adverse: float
    clean_state: str
    peak_raw: float = 0.0


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


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = add_high_order_state_features(df)
    return df.reset_index(drop=True)


def _safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _factor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    def n(col: str, scale: float = 1.0) -> np.ndarray:
        return np.tanh(_safe_series(frame, col).to_numpy(dtype=float) / max(scale, 1e-12))

    trend = (
        0.28 * n("mtf_trend_1h", 0.0010)
        + 0.28 * n("mtf_trend_4h", 0.0007)
        + 0.18 * n("cross_scale_curvature", 1.0)
        + 0.14 * n("breakout_strength", 1.0)
        - 0.12 * n("mean_reversion_z", 2.0)
    )
    flow = (
        0.26 * n("net_taker_ratio", 1.0)
        + 0.22 * n("smart_money_flow", 1.0)
        + 0.18 * n("taker_acceleration", 1.0)
        + 0.18 * n("ofi_acceleration", 1.0)
        + 0.16 * n("whale_retail_ratio", 2.0)
    )
    vol = (
        0.27 * np.abs(n("volatility_z", 2.0))
        + 0.23 * np.abs(n("garch_vol_z", 2.0))
        + 0.20 * np.abs(n("rogers_satchell_vol", 0.01))
        + 0.14 * np.abs(n("bb_width_z", 2.0))
        + 0.16 * np.abs(n("liquidity_vacuum", 1.0))
    )
    crowd = (
        0.28 * n("funding_pressure", 1.0)
        + 0.22 * n("funding_abs", 0.01)
        + 0.18 * n("funding_price_divergence", 1.0)
        + 0.18 * n("crowding_pressure", 1.0)
        + 0.14 * n("long_squeeze_risk", 1.0)
    )
    liquidity = (
        0.34 * np.abs(n("amihud_illiquidity_z", 2.0))
        + 0.26 * np.abs(n("liquidity_vacuum", 1.0))
        + 0.22 * np.abs(n("cvp_volume_imbalance", 2.0))
        + 0.18 * np.maximum(-n("execution_quality", 1.0), 0.0)
    )
    btc = 0.55 * n("btc_corr_60", 1.0) + 0.45 * n("eth_btc_ratio_change", 0.01)
    trend_bias = np.clip(0.58 * trend + 0.28 * flow + 0.14 * btc, -1.0, 1.0)
    risk_off = np.clip(0.42 * vol + 0.33 * liquidity + 0.25 * np.abs(crowd), 0.0, 1.0)

    bull = _sigmoid(2.8 * trend_bias + 0.85 * flow - 0.75 * risk_off)
    bear = _sigmoid(-2.8 * trend_bias - 0.85 * flow - 0.55 * risk_off)
    chop = _sigmoid(-2.1 * np.abs(trend_bias) + 1.15 * vol + 0.70 * liquidity - 0.20 * np.abs(flow))
    whipsaw = _sigmoid(1.35 * vol + 0.95 * liquidity + 0.55 * np.abs(flow) - 0.85 * np.abs(trend_bias))
    normal = _sigmoid(1.10 - 0.85 * vol - 0.75 * liquidity - 0.45 * np.abs(crowd) - 0.25 * np.abs(trend_bias))
    probs = np.vstack([bull, bear, chop, whipsaw, normal]).T
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    out = pd.DataFrame(index=frame.index)
    out[f"{CLEAN_PREFIX}factor_trend"] = trend
    out[f"{CLEAN_PREFIX}factor_flow"] = flow
    out[f"{CLEAN_PREFIX}factor_vol"] = vol
    out[f"{CLEAN_PREFIX}factor_crowding"] = crowd
    out[f"{CLEAN_PREFIX}factor_liquidity"] = liquidity
    out[f"{CLEAN_PREFIX}factor_btc"] = btc
    out[f"{CLEAN_PREFIX}trend_bias"] = trend_bias
    out[f"{CLEAN_PREFIX}risk_off_prob"] = risk_off
    for k, name in enumerate(("bull", "bear", "chop", "whipsaw", "normal")):
        out[f"{CLEAN_PREFIX}{name}_prob"] = probs[:, k]
    out[f"{CLEAN_PREFIX}confidence"] = probs.max(axis=1)
    out[f"{CLEAN_PREFIX}entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1) / math.log(probs.shape[1])
    out[f"{CLEAN_PREFIX}state_code"] = probs.argmax(axis=1)
    return out


def _is_forbidden_feature(col: str) -> bool:
    lower = str(col).lower()
    if lower.startswith(CLEAN_PREFIX):
        return False
    if lower.startswith("_") or lower in BANNED_EXACT:
        return True
    if lower.startswith(("m7_", "pred_", "conf_", "ai_")):
        return True
    if "regime" in lower:
        return True
    return any(fragment in lower for fragment in BANNED_FRAGMENTS)


def _numeric_common(frames: list[pd.DataFrame], *, include_clean: bool = True) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    out: list[str] = []
    for col in sorted(common):
        if _is_forbidden_feature(col):
            continue
        if not include_clean and str(col).startswith(CLEAN_PREFIX):
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            out.append(str(col))
    return out


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    data = {
        c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in cols
    }
    return pd.DataFrame(data, index=frame.index)


def _fit_state_model(y2024: pd.DataFrame, state_cols: list[str]) -> dict[str, Any]:
    x = _matrix(y2024, state_cols).to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=min(10, max(2, len(state_cols) - 1)), random_state=611)
    xz = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(x)))
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=611, batch_size=4096, n_init=12, max_iter=320)
    kmeans.fit(xz)
    return {"feature_cols": state_cols, "imputer": imputer, "scaler": scaler, "pca": pca, "kmeans": kmeans}


def _append_clean_regime(frame: pd.DataFrame, state_model: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    factors = _factor_frame(out)
    for col in factors.columns:
        out[col] = factors[col].to_numpy(dtype=float)
    x = _matrix(out, list(state_model["feature_cols"])).to_numpy(dtype=np.float32)
    xz = state_model["pca"].transform(state_model["scaler"].transform(state_model["imputer"].transform(x)))
    dist = state_model["kmeans"].transform(xz)
    labels = state_model["kmeans"].predict(xz).astype(int)
    inv = -dist / np.clip(np.std(dist, axis=1, keepdims=True), 1e-6, None)
    inv -= inv.max(axis=1, keepdims=True)
    prob = np.exp(inv)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    out[f"{CLEAN_PREFIX}cluster"] = labels
    for k in range(prob.shape[1]):
        out[f"{CLEAN_PREFIX}cluster_prob_{k}"] = prob[:, k]
    out[f"{CLEAN_PREFIX}cluster_confidence"] = prob.max(axis=1)
    return out


def _label_frame(frame: pd.DataFrame, horizon: int, fee: float, slip: float) -> pd.DataFrame:
    out = frame.copy()
    open_px = pd.to_numeric(out["open"], errors="coerce").ffill().to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(dtype=float)
    n = len(out)
    long_edge = np.full(n, -999.0)
    short_edge = np.full(n, -999.0)
    long_mae = np.zeros(n)
    short_mae = np.zeros(n)
    last_ret = np.zeros(n)
    cost = 2.0 * (float(fee) + float(slip))
    for i in range(0, n - horizon - 1):
        entry = open_px[i + 1]
        if entry <= 0.0:
            continue
        hi = np.nanmax(high[i + 1 : i + horizon + 1])
        lo = np.nanmin(low[i + 1 : i + horizon + 1])
        last = close[i + horizon]
        long_mfe = hi / entry - 1.0
        short_mfe = entry / max(lo, 1e-12) - 1.0
        long_edge[i] = long_mfe - cost
        short_edge[i] = short_mfe - cost
        long_mae[i] = max(0.0, 1.0 - lo / entry)
        short_mae[i] = max(0.0, hi / entry - 1.0)
        last_ret[i] = last / entry - 1.0
    y = np.zeros(n, dtype=int)
    y[(long_edge > 0.0042) & (long_edge > short_edge * 1.05)] = 1
    y[(short_edge > 0.0042) & (short_edge > long_edge * 1.05)] = 2
    out["_label"] = y
    out["_long_edge"] = long_edge
    out["_short_edge"] = short_edge
    out["_long_adverse"] = long_mae
    out["_short_adverse"] = short_mae
    out["_future_last_ret"] = last_ret
    return out.iloc[: n - horizon - 1].copy()


def _feature_analysis(fit: pd.DataFrame, candidate_cols: list[str], out_path: Path, max_features: int) -> tuple[list[str], list[dict[str, Any]]]:
    y = fit["_label"].astype(int).to_numpy()
    x = _matrix(fit, candidate_cols)
    sample_n = min(len(x), 40000)
    if sample_n < len(x):
        rng = np.random.default_rng(611)
        idx = np.sort(rng.choice(len(x), size=sample_n, replace=False))
        xs = x.iloc[idx]
        ys = y[idx]
    else:
        xs = x
        ys = y
    try:
        mi = mutual_info_classif(xs, ys, random_state=611, discrete_features=False)
    except Exception:
        mi = np.zeros(len(candidate_cols), dtype=float)
    rows = []
    for col, val in zip(candidate_cols, mi):
        s = pd.to_numeric(fit[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "mutual_info": float(val),
                "non_null_ratio": float(s.notna().mean()),
                "std": float(s.std(skipna=True) or 0.0),
                "family": "clean_regime" if col.startswith(CLEAN_PREFIX) else ("m7" if col.startswith("m7_") else "base"),
            }
        )
    rows = sorted(rows, key=lambda r: (r["feature"].startswith(CLEAN_PREFIX), r["mutual_info"]), reverse=True)
    selected: list[str] = []
    for col in candidate_cols:
        if col.startswith(CLEAN_PREFIX):
            selected.append(col)
    for row in rows:
        col = str(row["feature"])
        if col in selected:
            continue
        if float(row["mutual_info"]) > 0.0:
            selected.append(col)
        if len(selected) >= int(max_features):
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return selected, rows


def _fit_classifier(train: pd.DataFrame, cols: list[str], seed: int) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.038,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=int(seed),
        ),
    )
    model.fit(_matrix(train, cols), train["_label"].astype(int).to_numpy())
    return model


def _fit_regressor(train: pd.DataFrame, cols: list[str], target: str, seed: int) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=int(seed),
        ),
    )
    model.fit(_matrix(train, cols), pd.to_numeric(train[target], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return model


def _classes(model: Any) -> list[int]:
    clf = getattr(model, "named_steps", {}).get("histgradientboostingclassifier", model)
    return [int(c) for c in getattr(clf, "classes_", [])]


def _predict_model_proba(model: Any, frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[int]]:
    return np.asarray(model.predict_proba(_matrix(frame, cols)), dtype=float), _classes(model)


def _prob(proba: np.ndarray, classes: list[int], cls: int) -> np.ndarray:
    if cls not in classes:
        return np.zeros(len(proba), dtype=float)
    return np.asarray(proba[:, classes.index(cls)], dtype=float)


def _fit_moe(fit: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    global_model = _fit_classifier(fit, cols, 611)
    experts: dict[int, Any] = {}
    cluster_col = f"{CLEAN_PREFIX}cluster"
    for cluster in sorted(pd.to_numeric(fit[cluster_col], errors="coerce").dropna().astype(int).unique().tolist()):
        sub = fit[pd.to_numeric(fit[cluster_col], errors="coerce").fillna(-1).astype(int) == int(cluster)]
        if len(sub) >= 1800 and sub["_label"].nunique() >= 2:
            experts[int(cluster)] = _fit_classifier(sub, cols, 700 + int(cluster))
    return {"global_model": global_model, "experts": experts}


def _predict_moe(moe: dict[str, Any], frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[int]]:
    global_proba, global_classes = _predict_model_proba(moe["global_model"], frame, cols)
    out = np.zeros((len(frame), 3), dtype=float)
    for cls in (0, 1, 2):
        out[:, cls] = _prob(global_proba, global_classes, cls)
    cluster = pd.to_numeric(frame[f"{CLEAN_PREFIX}cluster"], errors="coerce").fillna(-1).astype(int).to_numpy()
    for c, model in dict(moe.get("experts", {}) or {}).items():
        idx = np.flatnonzero(cluster == int(c))
        if idx.size == 0:
            continue
        p, classes = _predict_model_proba(model, frame.iloc[idx], cols)
        ep = np.zeros((idx.size, 3), dtype=float)
        for cls in (0, 1, 2):
            ep[:, cls] = _prob(p, classes, cls)
        conf = pd.to_numeric(frame.iloc[idx].get(f"{CLEAN_PREFIX}cluster_prob_{int(c)}", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.clip(0.20 + 0.48 * conf, 0.20, 0.68)
        out[idx] = out[idx] * (1.0 - w[:, None]) + ep * w[:, None]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out, [0, 1, 2]


def _raw_ret(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _state_name(row: pd.Series) -> str:
    probs = {name: float(row.get(f"{CLEAN_PREFIX}{name}_prob", 0.0) or 0.0) for name in ("bull", "bear", "chop", "whipsaw", "normal")}
    return max(probs, key=probs.get)


def _backtest(
    frame: pd.DataFrame,
    proba: np.ndarray,
    risk_long: np.ndarray,
    risk_short: np.ndarray,
    cfg: RuntimeConfig,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    cost_side = float(fee) + float(slip)
    equity = 1.0
    peak = 1.0
    min_equity = 1.0
    pos: Position | None = None
    last_exit = -100000
    trade_id = 0
    ledger: list[dict[str, Any]] = []
    block_counts: dict[str, int] = {}
    for i in range(0, len(frame) - 1):
        close = float(frame.iloc[i]["close"])
        next_open = float(frame.iloc[i + 1]["open"])
        if pos is not None:
            raw = _raw_ret(pos.side, pos.entry_price, close)
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            peak = max(peak, mark)
            min_equity = min(min_equity, mark)
            exit_reason = ""
            if raw <= -cfg.stop_loss:
                exit_reason = "stop_loss"
            elif raw >= cfg.take_profit:
                exit_reason = "take_profit"
            elif pos.peak_raw >= cfg.trailing_stop * 1.15 and raw <= pos.peak_raw - cfg.trailing_stop:
                exit_reason = "trailing_stop"
            elif i - pos.entry_idx >= cfg.max_hold_bars:
                exit_reason = "max_hold"
            if exit_reason:
                realized = _raw_ret(pos.side, pos.entry_price, next_open)
                exit_cost = pos.notional * cost_side
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - exit_cost)
                peak = max(peak, equity)
                min_equity = min(min_equity, equity)
                ledger.append(
                    {
                        "trade_id": trade_id,
                        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
                        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
                        "exit_time": str(frame.iloc[i + 1]["timestamp"]),
                        "entry_idx": int(pos.entry_idx),
                        "exit_idx": int(i + 1),
                        "side": "LONG" if pos.side > 0 else "SHORT",
                        "action": "trade",
                        "sleeve": "clean_regime_moe_cvar_v1",
                        "clean_state": pos.clean_state,
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(next_open),
                        "notional": float(pos.notional),
                        "leverage": float(pos.leverage),
                        "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
                        "probability": float(pos.prob),
                        "gap": float(pos.gap),
                        "predicted_adverse": float(pos.predicted_adverse),
                        "realized_raw": float(realized),
                        "entry_fee_cash": float(pos.notional * cost_side),
                        "exit_fee_cash": float(exit_cost),
                        "trade_pnl_pct": float((gross - pos.notional * cost_side - exit_cost) * 100.0),
                        "cash_after": float(equity),
                        "blocked": False,
                        "stop_reason": exit_reason,
                    }
                )
                trade_id += 1
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars:
            continue
        if i % max(1, cfg.candidate_stride) != 0:
            continue
        long_p = float(proba[i, 1])
        short_p = float(proba[i, 2])
        no_p = float(proba[i, 0])
        side = 1 if long_p >= short_p else -1
        p = long_p if side > 0 else short_p
        alt = short_p if side > 0 else long_p
        gap = p - max(alt, 0.35 * no_p)
        row = frame.iloc[i]
        clean_conf = float(row.get(f"{CLEAN_PREFIX}confidence", 0.0) or 0.0)
        risk_off = float(row.get(f"{CLEAN_PREFIX}risk_off_prob", 0.0) or 0.0)
        trend_bias = float(row.get(f"{CLEAN_PREFIX}trend_bias", 0.0) or 0.0)
        adverse = float(risk_long[i] if side > 0 else risk_short[i])
        reason = ""
        if p < cfg.threshold:
            reason = "probability_below_threshold"
        elif gap < cfg.gap:
            reason = "gap_below_threshold"
        elif clean_conf < cfg.clean_conf_floor:
            reason = "clean_confidence_below_floor"
        elif risk_off > cfg.risk_off_cap:
            reason = "risk_off_cap"
        elif adverse > cfg.cvar_adverse_cap:
            reason = "cvar_adverse_cap"
        elif side * trend_bias < -0.42 and max(row.get(f"{CLEAN_PREFIX}chop_prob", 0.0), row.get(f"{CLEAN_PREFIX}whipsaw_prob", 0.0)) < 0.45:
            reason = "direction_state_conflict"
        if reason:
            block_counts[reason] = block_counts.get(reason, 0) + 1
            continue
        edge_scale = ((p - cfg.threshold) / max(1.0 - cfg.threshold, 1e-9)) ** 0.70
        state_scale = np.clip(0.78 + 0.42 * clean_conf - 0.58 * risk_off, 0.25, 1.20)
        risk_scale = np.clip(1.0 - adverse / max(cfg.cvar_adverse_cap, 1e-9) * 0.55, 0.25, 1.0)
        notional = cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * state_scale * risk_scale
        entry_cost = notional * cost_side
        equity *= max(0.0, 1.0 - entry_cost)
        min_equity = min(min_equity, equity)
        pos = Position(
            side=side,
            signal_idx=i,
            entry_idx=i + 1,
            entry_price=next_open,
            notional=float(np.clip(notional, cfg.min_notional, cfg.max_notional)),
            leverage=float(cfg.leverage),
            prob=p,
            gap=gap,
            predicted_adverse=adverse,
            clean_state=_state_name(row),
        )
    if pos is not None:
        i = len(frame) - 1
        exit_price = float(frame.iloc[i]["close"])
        realized = _raw_ret(pos.side, pos.entry_price, exit_price)
        exit_cost = pos.notional * cost_side
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - exit_cost)
        min_equity = min(min_equity, equity)
        ledger.append(
            {
                "trade_id": trade_id,
                "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
                "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
                "exit_time": str(frame.iloc[i]["timestamp"]),
                "entry_idx": int(pos.entry_idx),
                "exit_idx": int(i),
                "side": "LONG" if pos.side > 0 else "SHORT",
                "action": "trade",
                "sleeve": "clean_regime_moe_cvar_v1",
                "clean_state": pos.clean_state,
                "entry_price": float(pos.entry_price),
                "exit_price": float(exit_price),
                "notional": float(pos.notional),
                "leverage": float(pos.leverage),
                "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
                "probability": float(pos.prob),
                "gap": float(pos.gap),
                "predicted_adverse": float(pos.predicted_adverse),
                "realized_raw": float(realized),
                "entry_fee_cash": float(pos.notional * cost_side),
                "exit_fee_cash": float(exit_cost),
                "trade_pnl_pct": float((gross - pos.notional * cost_side - exit_cost) * 100.0),
                "cash_after": float(equity),
                "blocked": False,
                "stop_reason": "end",
            }
        )
    ledger.append(
        {
            "trade_id": -1,
            "timestamp": str(frame.iloc[-1]["timestamp"]),
            "entry_time": "",
            "exit_time": "",
            "entry_idx": int(len(frame) - 1),
            "exit_idx": int(len(frame) - 1),
            "side": "COVERAGE",
            "action": "coverage_end",
            "sleeve": "clean_regime_moe_cvar_v1",
            "clean_state": "",
            "entry_price": np.nan,
            "exit_price": np.nan,
            "notional": 0.0,
            "leverage": 0.0,
            "margin_fraction": 0.0,
            "probability": 0.0,
            "gap": 0.0,
            "predicted_adverse": 0.0,
            "realized_raw": 0.0,
            "entry_fee_cash": 0.0,
            "exit_fee_cash": 0.0,
            "trade_pnl_pct": 0.0,
            "cash_after": float(equity),
            "blocked": True,
            "stop_reason": "coverage_end",
        }
    )
    trades = [r for r in ledger if r["action"] == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_equity / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "long_entries": int(sum(r["side"] == "LONG" for r in trades)),
        "short_entries": int(sum(r["side"] == "SHORT" for r in trades)),
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_notional": float(np.max([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": block_counts,
        "ledger": ledger,
    }


def _score(result: dict[str, Any]) -> float:
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    trades = int(result["trades"])
    if trades < 20:
        return -1e9 + pnl
    return float(pnl + 0.04 * min(trades, 180) + 1.8 * pnl / max(mdd, 1.0) - max(0.0, mdd - 16.0) * 3.5)


def _grid() -> list[RuntimeConfig]:
    configs: list[RuntimeConfig] = []
    for threshold in (0.44, 0.48, 0.52, 0.56):
        for gap in (0.04, 0.08, 0.12):
            for cvar_cap in (0.010, 0.014, 0.018, 0.024):
                for max_n in (1.0, 1.6, 2.4, 3.2):
                    configs.append(
                        RuntimeConfig(
                            threshold=threshold,
                            gap=gap,
                            max_notional=max_n,
                            min_notional=0.25,
                            leverage=5.0,
                            max_hold_bars=36,
                            stop_loss=0.012,
                            take_profit=0.034,
                            trailing_stop=0.010,
                            cooldown_bars=2,
                            cvar_adverse_cap=cvar_cap,
                            clean_conf_floor=0.23,
                            risk_off_cap=0.94,
                            candidate_stride=6,
                        )
                    )
    return configs


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _audit(report: dict[str, Any], feature_cols: list[str], eval_frame: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    bad_cols = [c for c in feature_cols if _is_forbidden_feature(c)]
    if bad_cols:
        blocking.append("forbidden_feature_cols:" + ",".join(bad_cols[:20]))
    if not any(c.startswith(CLEAN_PREFIX) for c in feature_cols):
        blocking.append("missing_clean_regime_features")
    if int(report["data_audit"]["train_eval_overlap"]) != 0:
        blocking.append("train_eval_timestamp_overlap")
    ledger_ts = pd.to_datetime(ledger["timestamp"], errors="coerce")
    if ledger_ts.max() < pd.to_datetime(eval_frame["timestamp"], errors="coerce").max():
        blocking.append("ledger_does_not_cover_eval_window")
    for key, metrics in report["metrics"].items():
        if not np.isfinite(float(metrics["pnl"])) or not np.isfinite(float(metrics["mdd"])):
            blocking.append(f"{key}_nonfinite_metric")
        if float(metrics["max_margin_fraction"]) > 1.0 + 1e-12:
            blocking.append(f"{key}_margin_fraction_gt_1")
        if int(metrics["trades"]) <= 0:
            warnings.append(f"{key}_no_trades")
    return {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "legacy_regime_columns_absent_from_model_inputs": not bad_cols,
            "clean_regime_v6_features_present": any(c.startswith(CLEAN_PREFIX) for c in feature_cols),
            "train_eval_timestamp_overlap_zero": int(report["data_audit"]["train_eval_overlap"]) == 0,
            "next_bar_open_execution": True,
            "fee_and_slippage_charged_on_entry_and_exit": True,
            "ledger_covers_full_eval_window": "ledger_does_not_cover_eval_window" not in blocking,
        },
        "feature_audit": {
            "feature_count": len(feature_cols),
            "clean_regime_feature_count": len([c for c in feature_cols if c.startswith(CLEAN_PREFIX)]),
            "forbidden_feature_cols": bad_cols,
        },
    }


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# Clean Regime MoE CVaR V1",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Architecture: 2024-only clean regime/state encoder + supervised global/cluster MoE entry policy + side-specific CVaR adverse-risk critic + cost-stressed selection.",
        f"- Audit: `{audit['status']}`",
        f"- Blocking: `{audit['blocking']}`",
        f"- Fit: `{report['data']['fit_range'][0]}` to `{report['data']['fit_range'][1]}`",
        f"- Selection: `{report['data']['selection_range'][0]}` to `{report['data']['selection_range'][1]}`",
        f"- Holdout: `{report['data']['holdout_range'][0]}` to `{report['data']['holdout_range'][1]}`",
        f"- OOS: `{report['data']['oos_range'][0]}` to `{report['data']['oos_range'][1]}`",
        "",
        "## Cost1 OOS",
        f"- PnL: `{c1['pnl']}`",
        f"- MDD: `{c1['mdd']}`",
        f"- Trades: `{c1['trades']}`",
        f"- Trades/day: `{c1['trades_per_day']}`",
        "",
        "## Feature Contract",
        f"- Selected features: `{len(report['feature_contract']['selected_features'])}`",
        f"- Clean regime features: `{len([c for c in report['feature_contract']['selected_features'] if c.startswith(CLEAN_PREFIX)])}`",
        "- Legacy `regime_*`, `cvp_regime`, `regime_trending`, and label/future/accounting columns are blocked from model inputs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state-2024", type=Path, default=DEFAULT_STATE_2024)
    p.add_argument("--train-2025", type=Path, default=DEFAULT_TRAIN_2025)
    p.add_argument("--eval-2026", type=Path, default=DEFAULT_EVAL_2026)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y2024 = _load_csv(args.state_2024)
    y2025 = _load_csv(args.train_2025)
    y2026 = _load_csv(args.eval_2026)

    state_cols = _numeric_common([y2024, y2025, y2026], include_clean=False)
    state_model = _fit_state_model(y2024, state_cols)
    y2025s = _append_clean_regime(y2025, state_model)
    y2026s = _append_clean_regime(y2026, state_model)
    materialized_2025 = args.out_dir / "clean_regime_v6_2025.csv"
    materialized_2026 = args.out_dir / "clean_regime_v6_2026.csv"
    y2025s.to_csv(materialized_2025, index=False)
    y2026s.to_csv(materialized_2026, index=False)

    labeled = _label_frame(y2025s, int(args.horizon_bars), float(args.fee), float(args.slip))
    fit = labeled[labeled["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = labeled[(labeled["timestamp"] >= pd.Timestamp("2025-09-01")) & (labeled["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = labeled[labeled["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    if fit.empty or selection.empty or holdout.empty:
        raise ValueError("empty fit/selection/holdout split")

    candidate_cols = _numeric_common([fit, selection, holdout, y2026s], include_clean=True)
    feature_analysis_path = args.report_out.with_name(args.report_out.stem + "_feature_analysis.csv")
    feature_cols, feature_rows = _feature_analysis(fit, candidate_cols, feature_analysis_path, int(args.max_features))
    bad = [c for c in feature_cols if _is_forbidden_feature(c)]
    if bad:
        raise ValueError("forbidden selected feature columns: " + ",".join(bad[:20]))

    moe = _fit_moe(fit, feature_cols)
    risk_long = _fit_regressor(fit, feature_cols, "_long_adverse", 801)
    risk_short = _fit_regressor(fit, feature_cols, "_short_adverse", 802)

    selection_proba, _ = _predict_moe(moe, selection, feature_cols)
    holdout_proba, _ = _predict_moe(moe, holdout, feature_cols)
    eval_proba, _ = _predict_moe(moe, y2026s, feature_cols)
    selection_risk_long = np.asarray(risk_long.predict(_matrix(selection, feature_cols)), dtype=float)
    selection_risk_short = np.asarray(risk_short.predict(_matrix(selection, feature_cols)), dtype=float)
    holdout_risk_long = np.asarray(risk_long.predict(_matrix(holdout, feature_cols)), dtype=float)
    holdout_risk_short = np.asarray(risk_short.predict(_matrix(holdout, feature_cols)), dtype=float)
    eval_risk_long = np.asarray(risk_long.predict(_matrix(y2026s, feature_cols)), dtype=float)
    eval_risk_short = np.asarray(risk_short.predict(_matrix(y2026s, feature_cols)), dtype=float)

    rows: list[dict[str, Any]] = []
    best_cfg: RuntimeConfig | None = None
    best_score = -1e18
    best_selection: dict[str, Any] | None = None
    for cfg in _grid():
        r1 = _backtest(selection, selection_proba, selection_risk_long, selection_risk_short, cfg, fee=args.fee, slip=args.slip)
        r2 = _backtest(selection, selection_proba, selection_risk_long, selection_risk_short, cfg, fee=args.fee * 2.0, slip=args.slip * 2.0)
        r3 = _backtest(selection, selection_proba, selection_risk_long, selection_risk_short, cfg, fee=args.fee * 3.0, slip=args.slip * 3.0)
        score = 0.50 * _score(r1) + 0.30 * _score(r2) + 0.20 * _score(r3)
        if r2["pnl"] < 0:
            score -= abs(float(r2["pnl"])) * 2.0
        if r3["pnl"] < 0:
            score -= abs(float(r3["pnl"])) * 3.5
        row = {"score": float(score), **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r1).items()}}
        row["selection_cost2_pnl"] = r2["pnl"]
        row["selection_cost3_pnl"] = r3["pnl"]
        rows.append(row)
        if score > best_score:
            best_score = float(score)
            best_cfg = cfg
            best_selection = r1
    if best_cfg is None or best_selection is None:
        raise RuntimeError("no config selected")

    holdout_result = _backtest(holdout, holdout_proba, holdout_risk_long, holdout_risk_short, best_cfg, fee=args.fee, slip=args.slip)
    metrics: dict[str, Any] = {}
    ledger_paths: dict[str, str] = {}
    last_ledger: pd.DataFrame | None = None
    for mult in (1, 2, 3):
        result = _backtest(y2026s, eval_proba, eval_risk_long, eval_risk_short, best_cfg, fee=args.fee * mult, slip=args.slip * mult)
        key = f"cost{mult}"
        metrics[key] = _compact(result)
        ledger_path = args.report_out.with_name(args.report_out.stem + f"_{key}_ledger.csv")
        ledger_df = pd.DataFrame(result["ledger"])
        ledger_df.to_csv(ledger_path, index=False)
        ledger_paths[key] = str(ledger_path)
        last_ledger = ledger_df

    model_payload = {
        "model_id": MODEL_ID,
        "state_model": state_model,
        "moe": moe,
        "risk_long_model": risk_long,
        "risk_short_model": risk_short,
        "feature_cols": feature_cols,
        "selected_config": asdict(best_cfg),
        "clean_prefix": CLEAN_PREFIX,
        "forbidden_policy": "block legacy regime, target/future/label/accounting, raw OHLC, and generated price target columns",
    }
    model_path = args.out_dir / "clean_regime_moe_cvar_v1.pkl"
    joblib.dump(model_payload, model_path)
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "2024-only clean regime/state encoder, high-order feature refresh, supervised cluster MoE entry policy, side-specific CVaR adverse-risk critic, next-bar-open execution.",
        "data": {
            "state_2024": str(args.state_2024),
            "train_2025": str(args.train_2025),
            "eval_2026": str(args.eval_2026),
            "materialized_2025": str(materialized_2025),
            "materialized_2026": str(materialized_2026),
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(y2026s["timestamp"].iloc[0]), str(y2026s["timestamp"].iloc[-1])],
        },
        "data_audit": {
            "state_rows_2024": int(len(y2024)),
            "fit_rows": int(len(fit)),
            "selection_rows": int(len(selection)),
            "holdout_rows": int(len(holdout)),
            "eval_rows": int(len(y2026s)),
            "train_eval_overlap": _overlap(fit, y2026s) + _overlap(selection, y2026s) + _overlap(holdout, y2026s),
        },
        "artifacts": {
            "model": str(model_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
            "feature_analysis": str(feature_analysis_path),
            "selection_grid": str(grid_path),
            "ledgers": ledger_paths,
        },
        "feature_contract": {
            "state_fit_feature_count": len(state_cols),
            "candidate_feature_count": len(candidate_cols),
            "selected_features": feature_cols,
            "top_feature_analysis": feature_rows[:40],
        },
        "selected_config": asdict(best_cfg),
        "selection_score": best_score,
        "selection_result": _compact(best_selection),
        "holdout_result": _compact(holdout_result),
        "metrics": metrics,
    }
    audit = _audit(report, feature_cols, y2026s, last_ledger if last_ledger is not None else pd.DataFrame())
    report["audit"] = audit
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_contract(args.contract_out, report, audit)
    print(json.dumps({"status": audit["status"], "metrics": metrics, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
