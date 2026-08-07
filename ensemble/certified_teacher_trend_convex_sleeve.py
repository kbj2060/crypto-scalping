from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


MODEL_ID = "certified_teacher_trend_convex_sleeve_v5"

CONTRACTS = {
    "CONVEX_TREND_72": {
        "max_hold_bars": 72,
        "stop_loss": 0.010,
        "take_profit": 0.080,
        "trail_activate": 0.020,
        "trailing_stop": 0.012,
    },
    "CONVEX_TREND_144": {
        "max_hold_bars": 144,
        "stop_loss": 0.012,
        "take_profit": 0.140,
        "trail_activate": 0.030,
        "trailing_stop": 0.018,
    },
}


@dataclass(frozen=True)
class SleeveConfig:
    top_k_per_day: int
    min_event_score: float
    min_pred_edge_pct: float
    max_notional: float
    min_notional: float
    min_gap_bars: int
    max_pred_adverse_pct: float
    leverage: float = 5.0


@dataclass
class Position:
    side: int
    family: str
    signal_idx: int
    entry_idx: int
    entry_price: float
    notional: float
    leverage: float
    expected_pct: float
    adverse_pct: float
    event_score: float
    peak_raw: float = 0.0


def small_grid() -> list[SleeveConfig]:
    return [
        SleeveConfig(1, 0.62, 0.08, 1.0, 0.18, 24, 1.40),
        SleeveConfig(2, 0.62, 0.10, 0.9, 0.16, 18, 1.25),
        SleeveConfig(2, 0.70, 0.08, 1.0, 0.18, 24, 1.40),
        SleeveConfig(3, 0.70, 0.12, 0.8, 0.14, 18, 1.15),
        SleeveConfig(2, 0.78, 0.05, 1.1, 0.20, 30, 1.60),
        SleeveConfig(1, 0.70, 0.00, 1.2, 0.20, 36, 1.80),
    ]


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in cols}, index=frame.index)


def feature_cols(frames: list[pd.DataFrame], clean_prefix: str) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    banned = {"timestamp", "open", "high", "low", "close", "regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal", "regime_trending", "cvp_regime"}
    cols: list[str] = []
    for col in sorted(common):
        lower = col.lower()
        if col in banned or lower.startswith("_") or "future" in lower or "target" in lower or "label" in lower or "realized" in lower or "cash_after" in lower:
            continue
        if ("regime" in lower and not lower.startswith(clean_prefix)) or "hdb" in lower or lower.startswith("hmm_") or "legacy" in lower:
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            cols.append(col)
    return cols


def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _sigmoid(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def append_event_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ai_dir = _col(out, "ai_dir_p_up") - _col(out, "ai_dir_p_down")
    m7_dir = _col(out, "m7_trend_xgb_up") - _col(out, "m7_trend_xgb_dn")
    q_skew = (_col(out, "m7_q90") + _col(out, "m7_q50") + _col(out, "m7_expected_ret")) - _col(out, "m7_q10").abs()
    trend_bias = _col(out, "clean_regime_2024_unsup_v4_trend_bias")
    bull_bear = _col(out, "clean_regime_2024_unsup_v4_bull_prob") - _col(out, "clean_regime_2024_unsup_v4_bear_prob")
    market = (
        2.4 * _col(out, "breakout_strength")
        + 35.0 * _col(out, "mtf_trend_1h")
        + 70.0 * _col(out, "mtf_trend_4h")
        + 4.0 * _col(out, "dlinear_smf_slope")
        + 0.55 * _col(out, "ai_flow_pressure")
        + 0.40 * (_col(out, "ai_anchor_trend_escape_prob") - 0.50)
    )
    confidence = (
        0.55 * _col(out, "m7_confidence")
        + 0.45 * _col(out, "conf_patchtst")
        + 1.40 * _col(out, "clean_regime_2024_unsup_v4_confidence")
        - 0.45 * _col(out, "clean_regime_2024_unsup_v4_transition_risk")
        - 0.20 * _col(out, "clean_regime_2024_unsup_v4_entropy")
        - 0.08 * _col(out, "amihud_illiquidity_z").clip(lower=0.0)
    )
    long_raw = 1.35 * ai_dir + 1.15 * m7_dir + 1.30 * trend_bias + 0.80 * bull_bear + market + confidence
    short_raw = -1.35 * ai_dir - 1.15 * m7_dir - 1.30 * trend_bias - 0.80 * bull_bear - market + confidence
    out["event_score_long"] = _sigmoid(long_raw)
    out["event_score_short"] = _sigmoid(short_raw)
    out["event_direction_margin"] = (out["event_score_long"] - out["event_score_short"]).abs()
    return out


def _raw(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def replay_convex(frame: pd.DataFrame, idx: int, side: int, contract: dict[str, float], *, fee: float, slip: float) -> tuple[float, float, float, str]:
    if idx + 1 >= len(frame):
        return -999.0, 999.0, 0.0, "no_next_bar"
    entry = float(frame.iloc[idx + 1]["open"])
    if entry <= 0.0:
        return -999.0, 999.0, 0.0, "bad_entry"
    peak_raw = -999.0
    adverse = 0.0
    exit_idx = min(idx + int(contract["max_hold_bars"]), len(frame) - 1)
    reason = "max_hold"
    for j in range(idx + 1, min(idx + int(contract["max_hold_bars"]) + 1, len(frame))):
        raw = _raw(side, entry, float(frame.iloc[j]["close"]))
        peak_raw = max(peak_raw, raw)
        adverse = max(adverse, max(0.0, -raw))
        if raw <= -float(contract["stop_loss"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "stop_loss"
            break
        if raw >= float(contract["take_profit"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "loose_take_profit"
            break
        if peak_raw >= float(contract["trail_activate"]) and raw <= peak_raw - float(contract["trailing_stop"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "convex_trailing_stop"
            break
    px = float(frame.iloc[exit_idx]["open"] if exit_idx < len(frame) - 1 else frame.iloc[exit_idx]["close"])
    net = (_raw(side, entry, px) - 2.0 * (fee + slip)) * 100.0
    convexity = (peak_raw - adverse) * 100.0
    return float(net), float(adverse * 100.0), float(convexity), reason


def build_event_candidates(frame: pd.DataFrame, cols: list[str], side: int, *, fee: float, slip: float, label: bool, min_seed_score: float = 0.42, row_stride: int = 1) -> pd.DataFrame:
    score_col = "event_score_long" if side > 0 else "event_score_short"
    base_idx = np.arange(0, len(frame) - 1, max(1, int(row_stride)), dtype=np.int32)
    scores = pd.to_numeric(frame.iloc[base_idx][score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    idx_values = base_idx[scores >= float(min_seed_score)]
    if len(idx_values) == 0:
        return pd.DataFrame()
    base = matrix(frame, cols).iloc[idx_values].reset_index(drop=True)
    rows = []
    for family, contract in CONTRACTS.items():
        part = base.copy()
        part["_idx"] = idx_values
        part["cand_family"] = family
        part["cand_side"] = float(side)
        part["cand_max_hold_bars"] = float(contract["max_hold_bars"])
        part["cand_stop_loss"] = float(contract["stop_loss"])
        part["cand_take_profit"] = float(contract["take_profit"])
        part["cand_trail_activate"] = float(contract["trail_activate"])
        part["cand_trailing_stop"] = float(contract["trailing_stop"])
        part["cand_event_score"] = pd.to_numeric(frame.iloc[idx_values][score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if label:
            vals = [replay_convex(frame, int(i), side, contract, fee=fee, slip=slip) for i in idx_values]
            part["target_net_pct"] = [v[0] for v in vals]
            part["target_adverse_pct"] = [v[1] for v in vals]
            part["target_convexity_pct"] = [v[2] for v in vals]
            part["target_exit_reason"] = [v[3] for v in vals]
        rows.append(part)
    return pd.concat(rows, axis=0, ignore_index=True)


def model_cols(cands: pd.DataFrame) -> list[str]:
    return [c for c in cands.columns if c not in {"_idx", "cand_family", "target_net_pct", "target_adverse_pct", "target_convexity_pct", "target_exit_reason"}]


def fit_scorer(cands: pd.DataFrame, cols: list[str], *, seed: int, max_rows: int) -> dict[str, Any]:
    train = cands
    if len(train) > max_rows:
        train = train.sample(max_rows, random_state=seed)
    x = matrix(train, cols)
    y_edge = pd.to_numeric(train["target_net_pct"], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
    y_adv = pd.to_numeric(train["target_adverse_pct"], errors="coerce").fillna(999.0).to_numpy(dtype=float)
    y_conv = pd.to_numeric(train["target_convexity_pct"], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
    params = dict(max_iter=240, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=30, early_stopping=False)
    edge = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **params))
    adv = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="quantile", quantile=0.70, random_state=seed + 1, **params))
    conv = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="squared_error", random_state=seed + 2, **params))
    edge.fit(x, y_edge)
    adv.fit(x, y_adv)
    conv.fit(x, y_conv)
    return {"edge": edge, "adverse": adv, "convexity": conv, "cols": cols}


def predict_scorer(model: dict[str, Any], cands: pd.DataFrame) -> pd.DataFrame:
    if cands.empty:
        return cands.copy()
    out = cands.copy()
    x = matrix(out, list(model["cols"]))
    out["pred_edge_pct"] = np.asarray(model["edge"].predict(x), dtype=float)
    out["pred_adverse_pct"] = np.asarray(model["adverse"].predict(x), dtype=float)
    out["pred_convexity_pct"] = np.asarray(model["convexity"].predict(x), dtype=float)
    out["rank_score"] = out["pred_edge_pct"] + 0.35 * out["pred_convexity_pct"] - 0.55 * out["pred_adverse_pct"] + 0.20 * out["cand_event_score"]
    return out


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
