from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ID = "certified_teacher_regime_moe_v1"
CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"

NON_FEATURES = {
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
    adverse_cap: float
    clean_conf_floor: float
    transition_risk_cap: float
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
    entry_cost: float
    peak_raw: float = 0.0


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def merge_by_timestamp(base: pd.DataFrame, add: pd.DataFrame, *, suffix: str = "") -> pd.DataFrame:
    out = base.copy()
    cols = [c for c in add.columns if c != "timestamp" and c not in out.columns]
    if not cols:
        return out
    merged = out.merge(add[["timestamp"] + cols], on="timestamp", how="left", suffixes=("", suffix))
    return merged.sort_values("timestamp").reset_index(drop=True)


def merge_teacher_sources(base: pd.DataFrame, ai: pd.DataFrame | None, m7: pd.DataFrame | None) -> pd.DataFrame:
    out = base.copy()
    if ai is not None:
        ai_cols = [
            c for c in ai.columns
            if c == "timestamp"
            or c.startswith("ai_")
            or c.startswith("patchtst_")
            or c.startswith("tide_")
            or c.startswith("timesnet_")
            or c.startswith("dlinear_")
            or c in {"pred_patchtst", "conf_patchtst"}
        ]
        out = merge_by_timestamp(out, ai[ai_cols])
    if m7 is not None:
        m7_cols = [c for c in m7.columns if c == "timestamp" or c.startswith("m7_")]
        out = merge_by_timestamp(out, m7[m7_cols])
    return out


def _safe(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _tanh(frame: pd.DataFrame, col: str, scale: float = 1.0) -> np.ndarray:
    return np.tanh(_safe(frame, col).to_numpy(dtype=float) / max(scale, 1e-12))


def clean_regime_factors(frame: pd.DataFrame) -> pd.DataFrame:
    trend = (
        0.26 * _tanh(frame, "mtf_trend_1h", 0.0010)
        + 0.24 * _tanh(frame, "mtf_trend_4h", 0.0007)
        + 0.16 * _tanh(frame, "m7_q50", 0.0030)
        + 0.14 * _tanh(frame, "ai_dir_edge", 1.0)
        + 0.12 * _tanh(frame, "breakout_strength", 1.0)
        - 0.08 * _tanh(frame, "mean_reversion_z", 2.0)
    )
    flow = (
        0.27 * _tanh(frame, "net_taker_ratio", 1.0)
        + 0.22 * _tanh(frame, "smart_money_flow", 1.0)
        + 0.18 * _tanh(frame, "taker_acceleration", 1.0)
        + 0.18 * _tanh(frame, "ofi_acceleration", 1.0)
        + 0.15 * _tanh(frame, "ai_flow_pressure", 1.0)
    )
    vol = (
        0.22 * np.abs(_tanh(frame, "volatility_z", 2.0))
        + 0.18 * np.abs(_tanh(frame, "garch_vol_z", 2.0))
        + 0.18 * np.abs(_tanh(frame, "bb_width_z", 2.0))
        + 0.18 * np.abs(_tanh(frame, "m7_qwidth", 0.01))
        + 0.14 * np.abs(_tanh(frame, "tide_vol_zscore", 2.0))
        + 0.10 * np.abs(_tanh(frame, "ai_vol_regime_pct", 1.0))
    )
    crowd = (
        0.25 * _tanh(frame, "funding_pressure", 1.0)
        + 0.22 * _tanh(frame, "funding_abs", 0.01)
        + 0.18 * _tanh(frame, "funding_price_divergence", 1.0)
        + 0.18 * _tanh(frame, "crowding_pressure", 1.0)
        + 0.17 * _tanh(frame, "long_squeeze_risk", 1.0)
    )
    liquidity = (
        0.30 * np.abs(_tanh(frame, "amihud_illiquidity_z", 2.0))
        + 0.24 * np.abs(_tanh(frame, "liquidity_vacuum", 1.0))
        + 0.22 * np.abs(_tanh(frame, "cvp_volume_imbalance", 2.0))
        + 0.14 * np.maximum(-_tanh(frame, "execution_quality", 1.0), 0.0)
        + 0.10 * np.abs(_tanh(frame, "ai_flow_exhaustion", 1.0))
    )
    ai_disagreement = np.clip(_safe(frame, "ai_dir_entropy", 0.0).to_numpy(dtype=float), 0.0, 1.0)
    m7_uncertainty = np.clip(_safe(frame, "m7_qwidth", 0.0).to_numpy(dtype=float) / 0.02, 0.0, 1.0)
    risk_off = np.clip(0.34 * vol + 0.28 * liquidity + 0.18 * np.abs(crowd) + 0.12 * ai_disagreement + 0.08 * m7_uncertainty, 0.0, 1.0)
    trend_bias = np.clip(0.56 * trend + 0.30 * flow + 0.14 * _tanh(frame, "btc_corr_60", 1.0), -1.0, 1.0)
    transition = np.clip(0.38 * risk_off + 0.24 * ai_disagreement + 0.20 * np.abs(_tanh(frame, "ai_flow_flip_prob", 1.0)) + 0.18 * m7_uncertainty, 0.0, 1.0)

    bull = 1.0 / (1.0 + np.exp(np.clip(-2.8 * trend_bias - 0.7 * flow + 0.7 * risk_off, -40, 40)))
    bear = 1.0 / (1.0 + np.exp(np.clip(2.8 * trend_bias + 0.7 * flow + 0.5 * risk_off, -40, 40)))
    chop = 1.0 / (1.0 + np.exp(np.clip(2.0 * np.abs(trend_bias) - 1.0 * vol - 0.55 * liquidity, -40, 40)))
    whipsaw = 1.0 / (1.0 + np.exp(np.clip(-1.3 * vol - 0.95 * transition - 0.55 * liquidity, -40, 40)))
    normal = 1.0 / (1.0 + np.exp(np.clip(-1.2 + 0.7 * vol + 0.6 * liquidity + 0.45 * np.abs(crowd), -40, 40)))
    probs = np.vstack([bull, bear, chop, whipsaw, normal]).T
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    out = pd.DataFrame(index=frame.index)
    for name, values in {
        "factor_trend": trend,
        "factor_flow": flow,
        "factor_vol": vol,
        "factor_crowding": crowd,
        "factor_liquidity": liquidity,
        "trend_bias": trend_bias,
        "risk_off_prob": risk_off,
        "transition_risk": transition,
    }.items():
        out[f"{CLEAN_PREFIX}{name}"] = values
    for k, name in enumerate(("bull", "bear", "chop", "whipsaw", "normal")):
        out[f"{CLEAN_PREFIX}{name}_prob"] = probs[:, k]
    out[f"{CLEAN_PREFIX}confidence"] = probs.max(axis=1)
    out[f"{CLEAN_PREFIX}entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1) / math.log(probs.shape[1])
    out[f"{CLEAN_PREFIX}state_code"] = probs.argmax(axis=1)
    return out


def clean_regime_fit_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "mtf_trend_1h", "mtf_trend_4h", "m7_q50", "m7_qwidth", "m7_expected_ret", "m7_confidence",
        "ai_dir_edge", "ai_dir_entropy", "ai_adverse_risk", "ai_flow_pressure", "ai_flow_flip_prob",
        "net_taker_ratio", "smart_money_flow", "taker_acceleration", "ofi_acceleration",
        "volatility_z", "bb_width_z", "garch_vol_z", "tide_vol_zscore", "amihud_illiquidity_z",
        "funding_pressure", "funding_abs", "crowding_pressure", "liquidity_vacuum",
    ]
    return [c for c in candidates if c in frame.columns]


def fit_clean_regime_predictor(frame_2024: pd.DataFrame) -> dict[str, Any]:
    cols = clean_regime_fit_columns(frame_2024)
    if len(cols) < 6:
        raise ValueError("not enough clean regime fit columns")
    x = matrix(frame_2024, cols)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    xz = pipe.fit_transform(x)
    model = MiniBatchKMeans(n_clusters=5, random_state=704, batch_size=4096, n_init=12, max_iter=320)
    model.fit(xz)
    return {"feature_cols": cols, "preprocess": pipe, "model": model}


def append_clean_regime(frame: pd.DataFrame, regime: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    factors = clean_regime_factors(out)
    for col in factors.columns:
        out[col] = factors[col].to_numpy(dtype=float)
    cols = list(regime["feature_cols"])
    xz = regime["preprocess"].transform(matrix(out, cols))
    dist = regime["model"].transform(xz)
    labels = regime["model"].predict(xz).astype(int)
    inv = -dist / np.clip(np.std(dist, axis=1, keepdims=True), 1e-6, None)
    inv -= inv.max(axis=1, keepdims=True)
    prob = np.exp(inv)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    out[f"{CLEAN_PREFIX}cluster"] = labels
    out[f"{CLEAN_PREFIX}cluster_confidence"] = prob.max(axis=1)
    for k in range(prob.shape[1]):
        out[f"{CLEAN_PREFIX}cluster_prob_{k}"] = prob[:, k]
    return out


def forbidden_feature(col: str) -> bool:
    lower = col.lower()
    if lower.startswith(CLEAN_PREFIX):
        return False
    if col in NON_FEATURES:
        return True
    if lower.startswith("_"):
        return True
    if lower.startswith(("future", "target", "label")) or "future" in lower or "realized" in lower:
        return True
    if lower in {"regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal", "regime_trending", "cvp_regime"}:
        return True
    if "legacy" in lower or "regime_v2" in lower or "hdb" in lower or lower.startswith("hmm_"):
        return True
    if lower in {"trade_pnl_pct", "cash_after", "entry_fee_cash", "exit_fee_cash"}:
        return True
    return False


def candidate_feature_cols(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    cols: list[str] = []
    for col in sorted(common):
        if forbidden_feature(col):
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            cols.append(col)
    return cols


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for c in cols
        },
        index=frame.index,
    )


def label_frame(frame: pd.DataFrame, horizon: int, fee: float, slip: float) -> pd.DataFrame:
    out = frame.copy()
    open_px = pd.to_numeric(out["open"], errors="coerce").ffill().to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(dtype=float)
    n = len(out)
    cost = 2.0 * (float(fee) + float(slip))
    long_edge = np.full(n, -999.0)
    short_edge = np.full(n, -999.0)
    long_adverse = np.zeros(n)
    short_adverse = np.zeros(n)
    hold_eff = np.zeros(n)
    for i in range(0, n - horizon - 1):
        entry = open_px[i + 1]
        if entry <= 0:
            continue
        hi = float(np.nanmax(high[i + 1 : i + horizon + 1]))
        lo = float(np.nanmin(low[i + 1 : i + horizon + 1]))
        last = float(close[i + horizon])
        long_edge[i] = hi / entry - 1.0 - cost
        short_edge[i] = entry / max(lo, 1e-12) - 1.0 - cost
        long_adverse[i] = max(0.0, 1.0 - lo / entry)
        short_adverse[i] = max(0.0, hi / entry - 1.0)
        hold_eff[i] = abs(last / entry - 1.0) / max(max(long_edge[i], short_edge[i], 0.0), 1e-9)
    y = np.zeros(n, dtype=int)
    y[(long_edge > 0.0040) & (long_edge > short_edge * 1.04)] = 1
    y[(short_edge > 0.0040) & (short_edge > long_edge * 1.04)] = 2
    out["_label"] = y
    out["_long_edge"] = long_edge
    out["_short_edge"] = short_edge
    out["_long_adverse"] = long_adverse
    out["_short_adverse"] = short_adverse
    out["_hold_efficiency"] = hold_eff
    return out.iloc[: n - horizon - 1].copy()


def feature_analysis(fit: pd.DataFrame, cols: list[str], max_features: int) -> tuple[list[str], list[dict[str, Any]]]:
    x = matrix(fit, cols)
    y = fit["_label"].astype(int).to_numpy()
    sample_n = min(len(x), 50000)
    if sample_n < len(x):
        rng = np.random.default_rng(704)
        idx = np.sort(rng.choice(len(x), size=sample_n, replace=False))
        xs = x.iloc[idx]
        ys = y[idx]
    else:
        xs, ys = x, y
    try:
        mi = mutual_info_classif(xs, ys, random_state=704, discrete_features=False)
    except Exception:
        mi = np.zeros(len(cols), dtype=float)
    rows = []
    selected = [c for c in cols if c.startswith(CLEAN_PREFIX)]
    for col, val in zip(cols, mi):
        family = "clean_regime" if col.startswith(CLEAN_PREFIX) else "m7" if col.startswith("m7_") else "ai" if col.startswith(("ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")) or col in {"pred_patchtst", "conf_patchtst"} else "market"
        rows.append({"feature": col, "mutual_info": float(val), "family": family})
    for row in sorted(rows, key=lambda r: r["mutual_info"], reverse=True):
        col = str(row["feature"])
        if col not in selected and float(row["mutual_info"]) > 0.0:
            selected.append(col)
        if len(selected) >= max_features:
            break
    return selected[:max_features], sorted(rows, key=lambda r: r["mutual_info"], reverse=True)


def train_moe(fit: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    model = _fit_classifier(fit, cols, 704)
    experts: dict[int, Any] = {}
    cluster = pd.to_numeric(fit[f"{CLEAN_PREFIX}cluster"], errors="coerce").fillna(-1).astype(int)
    for c in sorted(cluster.unique().tolist()):
        sub = fit[cluster == int(c)]
        if int(c) >= 0 and len(sub) >= 1500 and sub["_label"].nunique() >= 2:
            experts[int(c)] = _fit_classifier(sub, cols, 820 + int(c))
    risk_long = _fit_regressor(fit, cols, "_long_adverse", 901)
    risk_short = _fit_regressor(fit, cols, "_short_adverse", 902)
    return {"global_model": model, "experts": experts, "risk_long": risk_long, "risk_short": risk_short, "feature_cols": cols}


def _fit_classifier(fit: pd.DataFrame, cols: list[str], seed: int) -> Any:
    clf = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.040,
        max_leaf_nodes=31,
        l2_regularization=0.10,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=seed,
    )
    model = make_pipeline(SimpleImputer(strategy="median"), clf)
    model.fit(matrix(fit, cols), fit["_label"].astype(int).to_numpy())
    return model


def _fit_regressor(fit: pd.DataFrame, cols: list[str], target: str, seed: int) -> Any:
    reg = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=0.08,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=seed,
    )
    model = make_pipeline(SimpleImputer(strategy="median"), reg)
    model.fit(matrix(fit, cols), pd.to_numeric(fit[target], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return model


def predict_moe(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = list(bundle["feature_cols"])
    proba = _predict_proba_3(bundle["global_model"], frame, cols)
    cluster = pd.to_numeric(frame[f"{CLEAN_PREFIX}cluster"], errors="coerce").fillna(-1).astype(int).to_numpy()
    for c, model in dict(bundle.get("experts", {}) or {}).items():
        idx = np.flatnonzero(cluster == int(c))
        if idx.size == 0:
            continue
        ep = _predict_proba_3(model, frame.iloc[idx], cols)
        conf_col = f"{CLEAN_PREFIX}cluster_prob_{int(c)}"
        conf = pd.to_numeric(frame.iloc[idx].get(conf_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.clip(0.20 + 0.48 * conf, 0.20, 0.68)
        proba[idx] = proba[idx] * (1.0 - w[:, None]) + ep * w[:, None]
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    risk_long = np.asarray(bundle["risk_long"].predict(matrix(frame, cols)), dtype=float)
    risk_short = np.asarray(bundle["risk_short"].predict(matrix(frame, cols)), dtype=float)
    return proba, risk_long, risk_short


def _predict_proba_3(model: Any, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix(frame, cols)), dtype=float)
    clf = getattr(model, "named_steps", {}).get("histgradientboostingclassifier", model)
    classes = [int(c) for c in getattr(clf, "classes_", [])]
    out = np.zeros((len(frame), 3), dtype=float)
    for cls in (0, 1, 2):
        if cls in classes:
            out[:, cls] = raw[:, classes.index(cls)]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def runtime_grid() -> list[RuntimeConfig]:
    out: list[RuntimeConfig] = []
    for threshold in (0.42, 0.46, 0.50, 0.54):
        for gap in (0.04, 0.08, 0.12):
            for adverse in (0.010, 0.014, 0.020):
                for max_n in (1.0, 1.8, 2.6):
                    out.append(RuntimeConfig(threshold, gap, max_n, 0.25, 5.0, 36, 0.012, 0.034, 0.010, 2, adverse, 0.22, 0.94, 6))
    return out


def backtest(frame: pd.DataFrame, proba: np.ndarray, risk_long: np.ndarray, risk_short: np.ndarray, cfg: RuntimeConfig, *, fee: float, slip: float) -> dict[str, Any]:
    cost_side = float(fee) + float(slip)
    equity = 1.0
    peak = 1.0
    min_equity = 1.0
    pos: Position | None = None
    last_exit = -100000
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
            reason = ""
            if raw <= -cfg.stop_loss:
                reason = "stop_loss"
            elif raw >= cfg.take_profit:
                reason = "take_profit"
            elif pos.peak_raw >= cfg.trailing_stop * 1.15 and raw <= pos.peak_raw - cfg.trailing_stop:
                reason = "trailing_stop"
            elif i - pos.entry_idx >= cfg.max_hold_bars:
                reason = "max_hold"
            if reason:
                realized = _raw_ret(pos.side, pos.entry_price, next_open)
                exit_cost = pos.notional * cost_side
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - exit_cost)
                peak = max(peak, equity)
                min_equity = min(min_equity, equity)
                ledger.append(_ledger_row(frame, pos, i + 1, next_open, realized, gross, exit_cost, equity, reason, len(ledger)))
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars or i % max(1, cfg.candidate_stride) != 0:
            continue
        long_p, short_p, no_p = float(proba[i, 1]), float(proba[i, 2]), float(proba[i, 0])
        side = 1 if long_p >= short_p else -1
        p = long_p if side > 0 else short_p
        alt = short_p if side > 0 else long_p
        gap = p - max(alt, 0.35 * no_p)
        row = frame.iloc[i]
        clean_conf = float(row.get(f"{CLEAN_PREFIX}confidence", 0.0) or 0.0)
        transition = float(row.get(f"{CLEAN_PREFIX}transition_risk", 0.0) or 0.0)
        trend_bias = float(row.get(f"{CLEAN_PREFIX}trend_bias", 0.0) or 0.0)
        adverse = float(risk_long[i] if side > 0 else risk_short[i])
        reason = ""
        if p < cfg.threshold:
            reason = "probability_below_threshold"
        elif gap < cfg.gap:
            reason = "gap_below_threshold"
        elif clean_conf < cfg.clean_conf_floor:
            reason = "clean_confidence_below_floor"
        elif transition > cfg.transition_risk_cap:
            reason = "transition_risk_cap"
        elif adverse > cfg.adverse_cap:
            reason = "adverse_cap"
        elif side * trend_bias < -0.42:
            reason = "direction_state_conflict"
        if reason:
            block_counts[reason] = block_counts.get(reason, 0) + 1
            continue
        edge_scale = ((p - cfg.threshold) / max(1.0 - cfg.threshold, 1e-9)) ** 0.70
        state_scale = np.clip(0.78 + 0.42 * clean_conf - 0.48 * transition, 0.25, 1.20)
        risk_scale = np.clip(1.0 - adverse / max(cfg.adverse_cap, 1e-9) * 0.55, 0.25, 1.0)
        notional = float(np.clip(cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * state_scale * risk_scale, cfg.min_notional, cfg.max_notional))
        equity *= max(0.0, 1.0 - notional * cost_side)
        min_equity = min(min_equity, equity)
        pos = Position(side, i, i + 1, next_open, notional, cfg.leverage, p, gap, adverse, _state_name(row), notional * cost_side)
    if pos is not None:
        i = len(frame) - 1
        exit_price = float(frame.iloc[i]["close"])
        realized = _raw_ret(pos.side, pos.entry_price, exit_price)
        exit_cost = pos.notional * cost_side
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - exit_cost)
        min_equity = min(min_equity, equity)
        ledger.append(_ledger_row(frame, pos, i, exit_price, realized, gross, exit_cost, equity, "end", len(ledger)))
    ledger.append({"trade_id": -1, "timestamp": str(frame.iloc[-1]["timestamp"]), "action": "coverage_end", "side": "COVERAGE", "cash_after": float(equity), "stop_reason": "coverage_end"})
    trades = [r for r in ledger if r.get("action") == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_equity / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": block_counts,
        "ledger": ledger,
    }


def _raw_ret(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _state_name(row: pd.Series) -> str:
    probs = {name: float(row.get(f"{CLEAN_PREFIX}{name}_prob", 0.0) or 0.0) for name in ("bull", "bear", "chop", "whipsaw", "normal")}
    return max(probs, key=probs.get)


def _ledger_row(frame: pd.DataFrame, pos: Position, exit_idx: int, exit_price: float, realized: float, gross: float, exit_cost: float, equity: float, reason: str, trade_id: int) -> dict[str, Any]:
    return {
        "trade_id": int(trade_id),
        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
        "exit_time": str(frame.iloc[exit_idx]["timestamp"]),
        "entry_idx": int(pos.entry_idx),
        "exit_idx": int(exit_idx),
        "side": "LONG" if pos.side > 0 else "SHORT",
        "action": "trade",
        "sleeve": MODEL_ID,
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
        "entry_fee_cash": float(pos.entry_cost),
        "exit_fee_cash": float(exit_cost),
        "trade_pnl_pct": float((gross - pos.entry_cost - exit_cost) * 100.0),
        "cash_after": float(equity),
        "blocked": False,
        "stop_reason": reason,
    }


def score(result: dict[str, Any]) -> float:
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    trades = int(result["trades"])
    if trades < 20:
        return -1e9 + pnl
    return float(pnl + 0.05 * min(trades, 220) + 1.6 * pnl / max(mdd, 1.0) - max(0.0, mdd - 16.0) * 4.0)


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
