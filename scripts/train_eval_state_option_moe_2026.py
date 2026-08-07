#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/state_option_moe_2026"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/state_option_moe_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/state_option_moe_2026_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/state_option_moe_2026_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/state_option_moe_2026.md"

BASELINE_REFERENCE = {
    "pnl": 752.648580357841,
    "mdd": -18.755787211251405,
    "trades": 353,
    "trades_per_day": 6.017045454545455,
    "avg_leverage": 1.5960290252000644,
    "cost_2x_pnl": 279.3638542719212,
    "cost_3x_pnl": 75.840036609542,
}

RESEARCH_REFERENCE = {
    "pnl": 210.53568634188076,
    "mdd": -18.015154932248112,
    "trades_per_day": 6.0,
    "cost_3x_pnl": 0.0,
}

FEATURE_COLS = [
    "log_return",
    "rogers_satchell_vol",
    "garch_vol_z",
    "volatility_z",
    "jump_flag",
    "jump_z",
    "wick_ratio",
    "hurst_48",
    "amihud_illiquidity_z",
    "liquidity_vacuum",
    "execution_quality",
    "ofti",
    "net_taker_ratio",
    "taker_acceleration",
    "trade_intensity",
    "big_trade_ratio",
    "ofi_acceleration",
    "sum_open_interest_value",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "oi_change_rate",
    "long_squeeze_risk",
    "crowding_pressure",
    "whale_retail_ratio",
    "whale_conviction",
    "last_funding_rate",
    "funding_abs",
    "funding_pressure",
    "funding_roc_288",
    "funding_price_divergence",
    "ou_funding_z",
    "ou_halflife",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "breakout_strength",
    "m7_action",
    "m7_hold_pred",
    "m7_expected_ret",
    "m7_confidence",
    "m7_qwidth",
    "m7_tp_offset",
    "m7_sl_offset",
    "pred_patchtst",
    "conf_patchtst",
    "patchtst_median",
    "patchtst_confidence",
    "tide_vol_raw",
    "tide_vol_zscore",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "ai_flow_flip_prob",
    "ai_flow_slope",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "regime_persistence",
    "cross_scale_curvature",
    "evt_tail_flag",
    "evt_excess_z",
    "evt_candidate_side",
    "evt_candidate_label",
    "evt_cost_hurdle",
    "evt_side_margin",
]

OPTION_INPUT_COLS = ["option_side", "option_notional", "option_hold", "state_token", "state_distance"]


@dataclass(frozen=True)
class OptionSpec:
    option_id: str
    side: int
    notional: float
    hold_bars: int
    exit_mode: str = "time"


@dataclass(frozen=True)
class SelectorConfig:
    name: str
    lambda_cvar: float
    lambda_cost: float
    lambda_turnover: float
    prob_large_loss_block: float
    min_utility: float
    cost_3x_required: bool
    max_daily_trades: int
    loss_cooldown_bars: int


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _split_train_validation(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    split = pd.Timestamp(split_date)
    return df.loc[ts < split].reset_index(drop=True), df.loc[ts >= split].reset_index(drop=True)


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    return [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])]


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(float(len(df)) / 288.0, 1.0)
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_num(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _prices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    fill_col = "open" if "open" in df.columns else "close"
    fill = pd.to_numeric(df[fill_col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return close.to_numpy(dtype=np.float64), fill.to_numpy(dtype=np.float64)


def _catalog() -> list[OptionSpec]:
    rows: list[OptionSpec] = []
    for side_name, side in (("long", 1), ("short", -1)):
        for notional in (0.5, 1.0, 1.5, 2.0, 2.8, 3.6):
            for hold in (6, 12, 24, 48):
                rows.append(OptionSpec(f"{side_name}_n{notional:.1f}_h{hold}", side, float(notional), int(hold)))
    return rows


def _exec_return(close: np.ndarray, fill: np.ndarray, option: OptionSpec, *, fee: float, slip: float, cost_mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    h = int(option.hold_bars)
    out = np.full(n, np.nan, dtype=np.float64)
    raw = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    mfe = np.full(n, np.nan, dtype=np.float64)
    if n <= h + 2:
        return out, raw, mae, mfe
    idx = np.arange(0, n - h - 1, dtype=np.int64)
    entry_px = fill[idx + 1]
    exit_px = fill[idx + h + 1]
    s = int(option.side)
    slip_m = float(slip) * float(cost_mult)
    if s > 0:
        entry_exec = entry_px * (1.0 + slip_m)
        exit_exec = exit_px * (1.0 - slip_m)
        gross = (exit_exec - entry_exec) / np.maximum(entry_exec, 1e-12)
        raw_gross = (exit_px - entry_px) / np.maximum(entry_px, 1e-12)
    else:
        entry_exec = entry_px * (1.0 - slip_m)
        exit_exec = exit_px * (1.0 + slip_m)
        gross = (entry_exec - exit_exec) / np.maximum(entry_exec, 1e-12)
        raw_gross = (entry_px - exit_px) / np.maximum(entry_px, 1e-12)
    out[idx] = gross * float(option.notional) - 2.0 * float(fee) * float(cost_mult) * float(option.notional)
    raw[idx] = raw_gross * float(option.notional)

    try:
        windows = np.lib.stride_tricks.sliding_window_view(close[1:], window_shape=h)
        windows = windows[: len(idx)]
        if s > 0:
            path_ret = (windows - entry_exec[:, None]) / np.maximum(entry_exec[:, None], 1e-12) * float(option.notional)
        else:
            path_ret = (entry_exec[:, None] - windows) / np.maximum(entry_exec[:, None], 1e-12) * float(option.notional)
        mae[idx] = np.nanmin(path_ret, axis=1)
        mfe[idx] = np.nanmax(path_ret, axis=1)
    except Exception:
        mae[idx] = np.minimum(out[idx], 0.0)
        mfe[idx] = np.maximum(out[idx], 0.0)
    return out, raw, mae, mfe


def _option_labels(df: pd.DataFrame, options: list[OptionSpec], *, fee: float, slip: float) -> dict[str, dict[str, np.ndarray]]:
    close, fill = _prices(df)
    labels: dict[str, dict[str, np.ndarray]] = {}
    for opt in options:
        pnl1, raw1, mae1, mfe1 = _exec_return(close, fill, opt, fee=fee, slip=slip, cost_mult=1.0)
        pnl2, _raw2, _mae2, _mfe2 = _exec_return(close, fill, opt, fee=fee, slip=slip, cost_mult=2.0)
        pnl3, _raw3, _mae3, _mfe3 = _exec_return(close, fill, opt, fee=fee, slip=slip, cost_mult=3.0)
        labels[opt.option_id] = {
            "pnl1": pnl1,
            "pnl2": pnl2,
            "pnl3": pnl3,
            "raw1": raw1,
            "mae1": mae1,
            "mfe1": mfe1,
        }
    return labels


def _fit_state_tokenizer(train_x: pd.DataFrame, val_x: pd.DataFrame, eval_x: pd.DataFrame, *, n_tokens: int, seed: int) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x_train = pipe.fit_transform(train_x)
    kmeans = MiniBatchKMeans(n_clusters=int(n_tokens), random_state=int(seed), batch_size=4096, n_init=5)
    kmeans.fit(x_train)

    def transform(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        z = pipe.transform(x)
        token = kmeans.predict(z).astype(np.int16)
        centers = kmeans.cluster_centers_[token]
        dist = np.linalg.norm(z - centers, axis=1).astype(np.float32)
        return token, dist

    tr_t, tr_d = transform(train_x)
    va_t, va_d = transform(val_x)
    ev_t, ev_d = transform(eval_x)
    meta = {
        "n_tokens": int(n_tokens),
        "train_token_counts": {str(i): int(np.sum(tr_t == i)) for i in range(int(n_tokens))},
    }
    return {"preprocess": pipe, "kmeans": kmeans}, {
        "train_token": tr_t,
        "train_distance": tr_d,
        "val_token": va_t,
        "val_distance": va_d,
        "eval_token": ev_t,
        "eval_distance": ev_d,
    }, meta


def _candidate_matrix(
    base_x: pd.DataFrame,
    tokens: np.ndarray,
    distances: np.ndarray,
    labels: dict[str, dict[str, np.ndarray]],
    options: list[OptionSpec],
    row_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, list[str]]:
    frames: list[pd.DataFrame] = []
    ys = {"pnl1": [], "pnl2": [], "pnl3": [], "mae1": [], "raw1": []}
    opt_ids: list[str] = []
    row_ids: list[np.ndarray] = []
    base_part = base_x.iloc[row_idx].reset_index(drop=True)
    for oi, opt in enumerate(options):
        part = base_part.copy()
        part["option_side"] = float(opt.side)
        part["option_notional"] = float(opt.notional)
        part["option_hold"] = float(opt.hold_bars)
        part["state_token"] = tokens[row_idx].astype(float)
        part["state_distance"] = distances[row_idx].astype(float)
        frames.append(part)
        lab = labels[opt.option_id]
        for key in ys:
            ys[key].append(lab[key][row_idx])
        opt_ids.extend([opt.option_id] * len(row_idx))
        row_ids.append(row_idx.astype(np.int64))
    x = pd.concat(frames, ignore_index=True)
    y = {k: np.concatenate(v).astype(np.float64) for k, v in ys.items()}
    rows = np.concatenate(row_ids).astype(np.int64)
    finite = np.isfinite(y["pnl1"]) & np.isfinite(y["pnl3"]) & np.isfinite(y["mae1"])
    x = x.loc[finite].reset_index(drop=True)
    y = {k: v[finite] for k, v in y.items()}
    rows = rows[finite]
    opt_ids = list(np.asarray(opt_ids, dtype=object)[finite])
    return x, y, rows, opt_ids


def _sample_rows(n: int, *, stride: int, seed: int, max_rows: int) -> np.ndarray:
    idx = np.arange(0, max(0, n - 50), max(1, int(stride)), dtype=np.int64)
    if len(idx) > int(max_rows):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_rows), replace=False))
    return idx


def _fit_critics(x: pd.DataFrame, y: dict[str, np.ndarray], *, seed: int, max_iter: int) -> dict[str, Any]:
    params = dict(max_iter=int(max_iter), learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.03, random_state=int(seed))
    y_large = (y["pnl1"] <= -0.025).astype(np.int8)
    critics = {
        "q05": HistGradientBoostingRegressor(loss="quantile", quantile=0.05, **params),
        "q50": HistGradientBoostingRegressor(loss="quantile", quantile=0.50, **params),
        "q95": HistGradientBoostingRegressor(loss="quantile", quantile=0.95, **params),
        "cost3": HistGradientBoostingRegressor(loss="squared_error", **params),
        "mae": HistGradientBoostingRegressor(loss="squared_error", **params),
        "large_loss": HistGradientBoostingClassifier(max_iter=int(max_iter), learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.03, random_state=int(seed)),
    }
    critics["q05"].fit(x, y["pnl1"])
    critics["q50"].fit(x, y["pnl1"])
    critics["q95"].fit(x, y["pnl1"])
    critics["cost3"].fit(x, y["pnl3"])
    critics["mae"].fit(x, np.abs(np.minimum(y["mae1"], 0.0)))
    critics["large_loss"].fit(x, y_large)
    return critics


def _predict_option_cube(
    critics: dict[str, Any],
    base_x: pd.DataFrame,
    tokens: np.ndarray,
    distances: np.ndarray,
    options: list[OptionSpec],
) -> dict[str, np.ndarray]:
    n = len(base_x)
    m = len(options)
    out = {k: np.full((n, m), np.nan, dtype=np.float32) for k in ("q05", "q50", "q95", "cost3", "mae", "prob_large_loss")}
    row_idx = np.arange(n, dtype=np.int64)
    for j, opt in enumerate(options):
        part = base_x.copy()
        part["option_side"] = float(opt.side)
        part["option_notional"] = float(opt.notional)
        part["option_hold"] = float(opt.hold_bars)
        part["state_token"] = tokens[row_idx].astype(float)
        part["state_distance"] = distances[row_idx].astype(float)
        out["q05"][:, j] = critics["q05"].predict(part).astype(np.float32)
        out["q50"][:, j] = critics["q50"].predict(part).astype(np.float32)
        out["q95"][:, j] = critics["q95"].predict(part).astype(np.float32)
        out["cost3"][:, j] = critics["cost3"].predict(part).astype(np.float32)
        out["mae"][:, j] = critics["mae"].predict(part).astype(np.float32)
        proba = critics["large_loss"].predict_proba(part)
        if proba.shape[1] == 1:
            out["prob_large_loss"][:, j] = float(critics["large_loss"].classes_[0] == 1)
        else:
            cls = list(critics["large_loss"].classes_)
            out["prob_large_loss"][:, j] = proba[:, cls.index(1)].astype(np.float32) if 1 in cls else 0.0
    return out


def _selector_grid() -> list[SelectorConfig]:
    rows: list[SelectorConfig] = []
    for cvar in (0.0, 0.4, 0.8, 1.2):
        for cost in (0.0, 0.4, 0.8):
            for turnover in (0.0, 0.25):
                for prob in (0.55, 0.95, 1.01):
                    for min_u in (-0.0300, -0.0200, -0.0100, 0.0000):
                        for need_c3 in (True, False):
                            for max_trades in (12, 16):
                                name = (
                                    f"cv{cvar:.1f}_co{cost:.1f}_to{turnover:.2f}_p{prob:.2f}_"
                                    f"u{min_u:.4f}_c3{int(need_c3)}_mt{max_trades}"
                                )
                                rows.append(SelectorConfig(name, cvar, cost, turnover, prob, min_u, need_c3, max_trades, 24))
    return rows


def _select_options(pred: dict[str, np.ndarray], options: list[OptionSpec], cfg: SelectorConfig, *, fee: float) -> dict[str, np.ndarray]:
    q05 = pred["q05"].astype(np.float64)
    q50 = pred["q50"].astype(np.float64)
    c3 = pred["cost3"].astype(np.float64)
    prob = pred["prob_large_loss"].astype(np.float64)
    notionals = np.asarray([o.notional for o in options], dtype=np.float64)[None, :]
    turnover = 2.0 * float(fee) * notionals
    utility = (
        q50
        - float(cfg.lambda_cvar) * np.maximum(-q05, 0.0)
        - float(cfg.lambda_cost) * np.maximum(q50 - c3, 0.0)
        - float(cfg.lambda_turnover) * turnover
    )
    mask = prob > float(cfg.prob_large_loss_block)
    if bool(cfg.cost_3x_required):
        mask |= c3 <= 0.0
    utility = np.where(mask, -1e9, utility)
    best_idx = np.argmax(utility, axis=1).astype(np.int16)
    best_u = utility[np.arange(len(best_idx)), best_idx]
    side = np.asarray([options[int(j)].side for j in best_idx], dtype=np.int8)
    notional = np.asarray([options[int(j)].notional for j in best_idx], dtype=np.float64)
    hold = np.asarray([options[int(j)].hold_bars for j in best_idx], dtype=np.int16)
    option_id = np.asarray([options[int(j)].option_id for j in best_idx], dtype=object)
    cash = best_u < float(cfg.min_utility)
    side[cash] = 0
    notional[cash] = 0.0
    hold[cash] = 0
    option_id[cash] = "CASH"
    return {
        "side": side,
        "notional": notional,
        "hold": hold,
        "option_idx": best_idx,
        "option_id": option_id,
        "utility": best_u.astype(np.float64),
    }


def _day_codes(df: pd.DataFrame) -> np.ndarray:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("D").astype("int64").to_numpy()
    return (np.arange(len(df), dtype=np.int64) // 288).astype(np.int64)


def _fill_price(fill: np.ndarray, idx: int, side: int, *, entry: bool, slip: float) -> float:
    px = float(fill[int(np.clip(idx, 0, len(fill) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def backtest_selected_options(
    df: pd.DataFrame,
    selected: dict[str, np.ndarray],
    *,
    fee: float,
    slip: float,
    max_daily_trades: int,
    loss_cooldown_bars: int,
    daily_loss_limit: float = 0.025,
    daily_dd_limit: float = 0.025,
    global_dd_cut: float = 0.12,
    global_dd_mult: float = 0.45,
    emit_ledger: bool = False,
) -> dict[str, Any]:
    close, fill = _prices(df)
    day_codes = _day_codes(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    hold_bars = 0
    notional = 0.0
    leverage = 1.0
    option_id = ""
    utility = 0.0
    peak_unreal = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    blocks: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    loss_streak = 0
    loss_cooldown_left = 0
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def block(reason: str) -> None:
        blocks[reason] = blocks.get(reason, 0) + 1

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, hold_bars, trades, wins
        nonlocal loss_streak, loss_cooldown_left, daily_trades, peak_unreal
        exit_idx = min(i + 1, len(df) - 1)
        exit_price = _fill_price(fill, exit_idx, pos, entry=False, slip=slip)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        pnl_frac = raw * notional - 2.0 * float(fee) * notional
        cash = before * max(0.0, 1.0 + pnl_frac)
        trades += 1
        daily_trades += 1
        is_win = pnl_frac > 0.0
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(loss_cooldown_bars))
        exits[reason] = exits.get(reason, 0) + 1
        if emit_ledger:
            ledger.append({
                "entry_idx": int(entry_idx),
                "exit_idx": int(i),
                "side": "LONG" if pos > 0 else "SHORT",
                "option_id": str(option_id),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "position_fraction": float(min(notional / max(leverage, 1e-12), 1.0)),
                "hold_bars": int(i - entry_idx),
                "pnl_frac": float(pnl_frac),
                "pnl_pct": float(pnl_frac * 100.0),
                "equity_before": float(before),
                "equity_after": float(cash),
                "exit_reason": str(reason),
                "utility": float(utility),
            })
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        hold_bars = 0
        peak_unreal = 0.0

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unreal = max(peak_unreal, unreal)
            age = i - entry_idx
            if hold_bars > 0 and age >= hold_bars:
                close_position(i, "option_time_exit")
                continue
            if unreal <= -0.035:
                close_position(i, "hard_stop")
                continue
            if peak_unreal >= 0.025 and unreal <= peak_unreal - 0.014:
                close_position(i, "trailing_giveback")
                continue
            continue

        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(max_daily_trades):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(daily_loss_limit)):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(daily_dd_limit)):
            block("daily_dd_lock")
            continue

        side = int(selected["side"][i])
        if side == 0:
            block("cash")
            continue
        n = float(selected["notional"][i])
        if account_dd >= float(global_dd_cut):
            n *= float(global_dd_mult)
        n = float(np.clip(n, 0.0, 3.6))
        if n <= 1e-12:
            block("zero_notional")
            continue
        pos = side
        entry_idx = i
        entry_equity = cash
        entry_price = _fill_price(fill, min(i + 1, len(df) - 1), pos, entry=True, slip=slip)
        notional = n
        leverage = max(1.0, min(float(n), 3.6))
        hold_bars = int(selected["hold"][i])
        option_id = str(selected["option_id"][i])
        utility = float(selected["utility"][i])
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage

    if pos != 0:
        close_position(len(df) - 2, "end_of_data")
    final_eq = cash
    return {
        "pnl": float((final_eq - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "avg_notional": float(notional_sum / max(long_entries + short_entries, 1)),
        "avg_leverage": float(leverage_sum / max(long_entries + short_entries, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "entry_blocks": blocks,
        "exits": exits,
        "ledger": ledger if emit_ledger else [],
    }


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics.get(k)
        for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries")
    }


def _audit_decisions(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "notional_nonnegative": bool(metrics.get("avg_notional", 0.0) >= 0.0),
        "trades_positive": bool(int(metrics.get("trades", 0)) > 0),
        "mdd_finite": bool(np.isfinite(float(metrics.get("mdd", 0.0)))),
        "pnl_finite": bool(np.isfinite(float(metrics.get("pnl", 0.0)))),
        "passed": bool(
            int(metrics.get("trades", 0)) > 0
            and np.isfinite(float(metrics.get("mdd", 0.0)))
            and np.isfinite(float(metrics.get("pnl", 0.0)))
            and float(metrics.get("avg_notional", 0.0)) >= 0.0
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _experiment_doc(report: dict[str, Any]) -> str:
    cand = report["oos"]["candidate"]
    c2 = report["oos"]["cost_2x"]
    c3 = report["oos"]["cost_3x"]
    decision = report["promotion_gate"]
    return f"""# State Option MoE 2026

Status: `{decision['decision']}`

## Summary

This experiment implements `state_option_moe_2026`, a state-tokenized option selector that is structurally different from the current clean-base / MuZero-AZ / Lifecycle family.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `{cand['pnl']:.6f}%` |
| MDD 1x | `{cand['mdd']:.6f}%` |
| Trades/day | `{cand['trades_per_day']:.6f}` |
| Avg leverage | `{cand['avg_leverage']:.6f}` |
| PnL 2x | `{c2['pnl']:.6f}%` |
| PnL 3x | `{c3['pnl']:.6f}%` |

## Selected Config

`{report['selected_config']['name']}`

## Gate

- Baseline PnL gate: `{decision['baseline_pnl_gate']}`
- Baseline MDD gate: `{decision['baseline_mdd_gate']}`
- Trades/day gate: `{decision['trades_per_day_gate']}`
- Cost 3x survival: `{decision['cost_3x_survival']}`
- Invariant audit: `{decision['invariant_audit_passed']}`

## Artifacts

- Report: `{report['artifacts']['report']}`
- Grid: `{report['artifacts']['grid']}`
- Ledger: `{report['artifacts']['ledger']}`
- Model dir: `{report['artifacts']['model_dir']}`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_full = _read(Path(args.train_csv))
    eval_df = _read(Path(args.eval_csv))
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    feature_cols = [c for c in FEATURE_COLS if c in train_full.columns]
    train_x = _feature_frame(train_df, feature_cols)
    val_x = _feature_frame(val_df, feature_cols)
    eval_x = _feature_frame(eval_df, feature_cols)
    options = _catalog()

    tokenizer, token_data, token_meta = _fit_state_tokenizer(
        train_x,
        val_x,
        eval_x,
        n_tokens=int(args.state_tokens),
        seed=int(args.seed),
    )
    train_labels = _option_labels(train_df, options, fee=float(args.fee), slip=float(args.slip))
    train_rows = _sample_rows(len(train_df), stride=int(args.train_row_stride), seed=int(args.seed), max_rows=int(args.max_base_train_rows))
    cand_x, cand_y, _rows, _opts = _candidate_matrix(
        train_x,
        token_data["train_token"],
        token_data["train_distance"],
        train_labels,
        options,
        train_rows,
    )
    if len(cand_x) > int(args.max_candidate_rows):
        rng = np.random.default_rng(int(args.seed))
        keep = np.sort(rng.choice(np.arange(len(cand_x)), size=int(args.max_candidate_rows), replace=False))
        cand_x = cand_x.iloc[keep].reset_index(drop=True)
        cand_y = {k: v[keep] for k, v in cand_y.items()}

    critics = _fit_critics(cand_x, cand_y, seed=int(args.seed), max_iter=int(args.max_iter))
    val_pred = _predict_option_cube(critics, val_x, token_data["val_token"], token_data["val_distance"], options)
    eval_pred = _predict_option_cube(critics, eval_x, token_data["eval_token"], token_data["eval_distance"], options)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: SelectorConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _selector_grid():
        sel = _select_options(val_pred, options, cfg, fee=float(args.fee))
        val_1x = backtest_selected_options(
            val_df,
            sel,
            fee=float(args.fee),
            slip=float(args.slip),
            max_daily_trades=int(cfg.max_daily_trades),
            loss_cooldown_bars=int(cfg.loss_cooldown_bars),
        )
        val_3x = backtest_selected_options(
            val_df,
            sel,
            fee=float(args.fee) * 3.0,
            slip=float(args.slip) * 3.0,
            max_daily_trades=int(cfg.max_daily_trades),
            loss_cooldown_bars=int(cfg.loss_cooldown_bars),
        )
        row = {
            **asdict(cfg),
            "val_pnl": val_1x["pnl"],
            "val_mdd": val_1x["mdd"],
            "val_trades": val_1x["trades"],
            "val_trades_per_day": val_1x["trades_per_day"],
            "val_avg_leverage": val_1x["avg_leverage"],
            "val_cost3_pnl": val_3x["pnl"],
            "val_cost3_mdd": val_3x["mdd"],
        }
        grid_rows.append(row)
        score = (
            float(val_1x["pnl"])
            + 10.0 * float(val_1x["mdd"])
            + 6.0 * float(val_1x["trades_per_day"])
            + 0.35 * float(val_3x["pnl"])
            - 100.0 * float(max(0.0, 5.5 - float(val_1x["trades_per_day"])))
        )
        if val_3x["pnl"] <= 0.0:
            score -= 250.0
        if float(val_1x["trades_per_day"]) < 4.0:
            score -= 150.0
        if score > selected_score:
            selected_score = float(score)
            selected_cfg = cfg
            selected_val = {"candidate": _compact(val_1x), "cost_3x": _compact(val_3x), "score": float(score)}

    assert selected_cfg is not None
    eval_sel = _select_options(eval_pred, options, selected_cfg, fee=float(args.fee))
    oos_1x = backtest_selected_options(
        eval_df,
        eval_sel,
        fee=float(args.fee),
        slip=float(args.slip),
        max_daily_trades=int(selected_cfg.max_daily_trades),
        loss_cooldown_bars=int(selected_cfg.loss_cooldown_bars),
        emit_ledger=True,
    )
    oos_2x = backtest_selected_options(
        eval_df,
        eval_sel,
        fee=float(args.fee) * 2.0,
        slip=float(args.slip) * 2.0,
        max_daily_trades=int(selected_cfg.max_daily_trades),
        loss_cooldown_bars=int(selected_cfg.loss_cooldown_bars),
    )
    oos_3x = backtest_selected_options(
        eval_df,
        eval_sel,
        fee=float(args.fee) * 3.0,
        slip=float(args.slip) * 3.0,
        max_daily_trades=int(selected_cfg.max_daily_trades),
        loss_cooldown_bars=int(selected_cfg.loss_cooldown_bars),
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, model_dir / "state_encoder.pkl")
    joblib.dump(critics, model_dir / "option_critics.pkl")
    selector_payload = {
        "model_id": "state_option_moe_2026",
        "selected_config": asdict(selected_cfg),
        "feature_cols": feature_cols,
        "option_input_cols": OPTION_INPUT_COLS,
        "options": [asdict(o) for o in options],
        "token_meta": token_meta,
    }
    joblib.dump(selector_payload, model_dir / "option_selector.pkl")
    _write_csv(Path(args.grid), grid_rows)
    _write_csv(Path(args.ledger), list(oos_1x.get("ledger", [])))
    oos_1x = {k: v for k, v in oos_1x.items() if k != "ledger"}
    audit = _audit_decisions(oos_1x)
    promotion = {
        "baseline_pnl_gate": bool(float(oos_1x["pnl"]) >= BASELINE_REFERENCE["pnl"]),
        "baseline_mdd_gate": bool(float(oos_1x["mdd"]) >= BASELINE_REFERENCE["mdd"]),
        "trades_per_day_gate": bool(float(oos_1x["trades_per_day"]) >= BASELINE_REFERENCE["trades_per_day"]),
        "avg_leverage_gate": bool(1.2 <= float(oos_1x["avg_leverage"]) <= 2.2),
        "cost_2x_survival": bool(float(oos_2x["pnl"]) > 0.0),
        "cost_3x_survival": bool(float(oos_3x["pnl"]) > 0.0),
        "invariant_audit_passed": bool(audit["passed"]),
    }
    promotion["research_gate"] = bool(
        float(oos_1x["pnl"]) >= RESEARCH_REFERENCE["pnl"]
        and float(oos_1x["mdd"]) >= RESEARCH_REFERENCE["mdd"]
        and float(oos_1x["trades_per_day"]) >= RESEARCH_REFERENCE["trades_per_day"]
        and float(oos_3x["pnl"]) > 0.0
        and audit["passed"]
    )
    promotion["decision"] = "promote" if all(
        promotion[k]
        for k in (
            "baseline_pnl_gate",
            "baseline_mdd_gate",
            "trades_per_day_gate",
            "avg_leverage_gate",
            "cost_2x_survival",
            "cost_3x_survival",
            "invariant_audit_passed",
        )
    ) else ("iterate" if promotion["research_gate"] else "reject")

    report = {
        "model_id": "state_option_moe_2026",
        "contract": str(ROOT / "docs/model_contracts/state_option_moe_2026_contract.md"),
        "data": {
            "train_csv": str(Path(args.train_csv)),
            "eval_csv": str(Path(args.eval_csv)),
            "train_range": _range(train_df),
            "validation_range": _range(val_df),
            "oos_range": _range(eval_df),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "oos_rows": int(len(eval_df)),
            "feature_cols": feature_cols,
        },
        "training": {
            "state_tokens": int(args.state_tokens),
            "candidate_train_rows": int(len(cand_x)),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
        },
        "selected_config": asdict(selected_cfg),
        "validation": selected_val,
        "oos": {
            "candidate": _compact(oos_1x),
            "cost_2x": _compact(oos_2x),
            "cost_3x": _compact(oos_3x),
        },
        "baseline_reference": BASELINE_REFERENCE,
        "research_reference": RESEARCH_REFERENCE,
        "audit": audit,
        "promotion_gate": promotion,
        "artifacts": {
            "model_dir": str(model_dir),
            "state_encoder": str(model_dir / "state_encoder.pkl"),
            "option_critics": str(model_dir / "option_critics.pkl"),
            "option_selector": str(model_dir / "option_selector.pkl"),
            "report": str(Path(args.report)),
            "grid": str(Path(args.grid)),
            "ledger": str(Path(args.ledger)),
            "doc": str(Path(args.doc)),
        },
        "artifact_sha256": {
            "state_encoder": _sha256(model_dir / "state_encoder.pkl"),
            "option_critics": _sha256(model_dir / "option_critics.pkl"),
            "option_selector": _sha256(model_dir / "option_selector.pkl"),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(_experiment_doc(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate State Option MoE 2026.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    p.add_argument("--state-tokens", type=int, default=64)
    p.add_argument("--train-row-stride", type=int, default=3)
    p.add_argument("--max-base-train-rows", type=int, default=24000)
    p.add_argument("--max-candidate-rows", type=int, default=450000)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "model_id": report["model_id"],
        "selected_config": report["selected_config"]["name"],
        "oos": report["oos"],
        "promotion_gate": report["promotion_gate"],
        "report": report["artifacts"]["report"],
    }, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
