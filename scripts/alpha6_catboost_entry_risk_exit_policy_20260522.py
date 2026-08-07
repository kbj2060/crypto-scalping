#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_LABEL_CACHE_DIR,
    DEFAULT_SPEC_DIR,
    PolicyConfig,
    _apply_bucket_preset,
    _build_lifecycle_labels,
    _days,
    _feature_matrix,
    _fill_price,
    _fit_models,
    _json_default,
    _label_frame,
    _load_or_build_lifecycle_labels,
    _predict_policy,
    _read_feature_frame,
    _read_spec,
    _score,
)


MODEL_ID = "alpha6_catboost_entry_risk_exit_policy_20260522"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_entry_risk_exit_policy_20260522"
DEFAULT_EXIT_CACHE_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_exit_dataset_cache_20260522"

EXIT_CONTEXT_COLS = [
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "nif_retail",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "signal_bias",
    "shadow_toxicity_score",
    "shadow_queue_collapse",
    "shadow_absorption_score",
    "shadow_queue_bias",
    "market_state_2024_unsup_v5_risk_off_prob",
    "market_state_2024_unsup_v5_trend_prob",
    "clean_regime4_state24_sticky090_v2_instability_prob",
    "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    "clean_regime4_state24_sticky090_v2_confidence",
    "clean_regime4_state24_sticky090_v2_trend_prob",
    "regime4_pred_instability_prob",
    "regime4_pred_whipsaw_prob",
]

EXIT_STATE_FEATURES = [
    "side",
    "notional",
    "ret",
    "hold_frac",
    "remaining_frac",
    "dist_tp_pct",
    "dist_sl_pct",
    "mae",
    "mfe",
    "tp_pct",
    "sl_pct",
    "current_atr_pct",
    "entry_atr_pct",
    "dist_tp_atr",
    "dist_sl_atr",
    "tp_atr",
    "sl_atr",
    "ret_to_mae",
    "ret_to_mfe",
    "giveback",
    "giveback_ratio",
    "side_obi",
    "side_obi_delta",
    "side_taker_buy_ratio",
    "side_taker_buy_ratio_delta",
    "side_nif_whale",
    "side_nif_whale_delta",
    "side_eai",
    "side_eai_delta",
    "side_oi_delta_pct",
    "side_funding_rate",
    "risk_off_prob",
    "whipsaw_prob",
    "regime_confidence",
]


def _safe_frame_value(frame: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
    if col not in frame.columns:
        return float(default)
    val = pd.to_numeric(frame[col], errors="coerce").iloc[int(np.clip(idx, 0, len(frame) - 1))]
    if pd.isna(val) or not np.isfinite(float(val)):
        return float(default)
    return float(val)


def _exit_state_vec(
    frame: pd.DataFrame,
    *,
    side: int,
    entry_idx: int,
    current_idx: int,
    entry_px: float,
    px: float,
    notional: float,
    tp: float,
    sl: float,
    hold: int,
    ref_horizon: int,
    mae: float,
    mfe: float,
) -> np.ndarray:
    eps = 1e-9
    raw = (px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px) / max(entry_px, 1e-12)
    ret = raw * float(notional)
    current_atr = max(_safe_frame_value(frame, "atr14_pct", current_idx, 0.003), eps)
    entry_atr = max(_safe_frame_value(frame, "atr14_pct", entry_idx, current_atr), eps)
    dist_tp = float(tp - raw)
    dist_sl = float(raw + sl)
    giveback = max(0.0, float(mfe) - max(float(ret), 0.0))
    obi = _safe_frame_value(frame, "obi", current_idx, 0.0)
    entry_obi = _safe_frame_value(frame, "obi", entry_idx, obi)
    taker = _safe_frame_value(frame, "taker_buy_ratio", current_idx, 0.5) - 0.5
    entry_taker = _safe_frame_value(frame, "taker_buy_ratio", entry_idx, taker + 0.5) - 0.5
    nif_whale = _safe_frame_value(frame, "nif_whale", current_idx, 0.0)
    entry_nif_whale = _safe_frame_value(frame, "nif_whale", entry_idx, nif_whale)
    eai = _safe_frame_value(frame, "eai", current_idx, 0.0)
    entry_eai = _safe_frame_value(frame, "eai", entry_idx, eai)
    risk_off = max(
        _safe_frame_value(frame, "market_state_2024_unsup_v5_risk_off_prob", current_idx, 0.0),
        _safe_frame_value(frame, "clean_regime4_state24_sticky090_v2_instability_prob", current_idx, 0.0),
        _safe_frame_value(frame, "regime4_pred_instability_prob", current_idx, 0.0),
    )
    whipsaw = max(
        _safe_frame_value(frame, "clean_regime4_state24_sticky090_v2_whipsaw_prob", current_idx, 0.0),
        _safe_frame_value(frame, "regime4_pred_whipsaw_prob", current_idx, 0.0),
    )
    return np.asarray(
        [
            float(side),
            float(notional),
            float(ret),
            float(hold) / max(float(ref_horizon), 1.0),
            float(max(ref_horizon - hold, 0)) / max(float(ref_horizon), 1.0),
            dist_tp,
            dist_sl,
            float(mae),
            float(mfe),
            float(tp),
            float(sl),
            current_atr,
            entry_atr,
            dist_tp / current_atr,
            dist_sl / current_atr,
            float(tp) / current_atr,
            float(sl) / current_atr,
            float(ret) / max(float(mae), eps),
            float(ret) / max(float(mfe), eps),
            giveback,
            giveback / max(float(mfe), eps),
            float(side) * obi,
            float(side) * (obi - entry_obi),
            float(side) * taker,
            float(side) * (taker - entry_taker),
            float(side) * nif_whale,
            float(side) * (nif_whale - entry_nif_whale),
            float(side) * eai,
            float(side) * (eai - entry_eai),
            float(side) * _safe_frame_value(frame, "oi_delta_pct", current_idx, 0.0),
            float(side) * _safe_frame_value(frame, "funding_rate", current_idx, 0.0),
            risk_off,
            whipsaw,
            _safe_frame_value(frame, "clean_regime4_state24_sticky090_v2_confidence", current_idx, 0.0),
        ],
        dtype=np.float64,
    )


def _build_exit_dataset(
    frame: pd.DataFrame,
    x_all: np.ndarray,
    valid: np.ndarray,
    y: dict[str, np.ndarray],
    cfg: PolicyConfig,
    *,
    max_samples: int,
    step: int,
    cost_mult: float,
    min_net_edge: float,
    weight_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    terminal_samples_skipped = 0
    rng = np.random.default_rng(42)
    trade_idx = np.flatnonzero(y["action"] != 0)
    if max_samples > 0 and len(trade_idx) > max_samples:
        trade_idx = rng.choice(trade_idx, size=int(max_samples), replace=False)
        trade_idx.sort()
    for j in trade_idx:
        idx = int(valid[j])
        side = 1 if int(y["action"][j]) == 1 else -1
        notional = float(cfg.notional_buckets[int(np.clip(y["notional"][j], 0, len(cfg.notional_buckets) - 1))])
        atr = float(pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).iloc[idx])
        tp = float(np.clip(cfg.tp_atr_buckets[int(y["take_profit"][j])] * atr, cfg.tp_min, cfg.tp_max))
        sl = float(np.clip(cfg.sl_atr_buckets[int(y["stop_loss"][j])] * atr, cfg.sl_min, cfg.sl_max))
        horizon = min(int(cfg.max_train_horizon_bars), len(frame) - idx - 2)
        if horizon <= 1:
            continue
        entry = close[idx]
        side_ret = (close[idx : idx + horizon + 1] / max(entry, 1e-12) - 1.0) * side
        terminal = horizon
        for k in range(1, horizon + 1):
            if side > 0:
                if high[idx + k] >= entry * (1.0 + tp) or low[idx + k] <= entry * (1.0 - sl):
                    terminal = k
                    break
            else:
                if low[idx + k] <= entry * (1.0 - tp) or high[idx + k] >= entry * (1.0 + sl):
                    terminal = k
                    break
        for k in range(1, terminal, max(1, int(step))):
            cur_path = side_ret[: k + 1] * notional
            fut_path = side_ret[k : terminal + 1] * notional
            if len(fut_path) == 0:
                continue
            cur_ret = float(cur_path[-1])
            mae = max(0.0, -float(np.min(cur_path)))
            mfe = max(0.0, float(np.max(cur_path)))
            future_best = float(np.max(fut_path))
            future_adverse = max(0.0, cur_ret - float(np.min(fut_path)))
            cur_atr = max(float(pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).iloc[idx + k]), 1e-9)
            round_trip_cost = 2.0 * (float(cfg.fee) + float(cfg.slip)) * float(notional) * float(cost_mult)
            close_score = cur_ret - round_trip_cost
            continue_score = future_best - cfg.adverse_penalty * future_adverse - cfg.hold_penalty * (terminal - k) / max(float(horizon), 1.0)
            dynamic_hurdle = max(float(min_net_edge), 0.10 * cur_atr * float(notional))
            margin = close_score - continue_score - dynamic_hurdle
            close_label = int(margin >= 0.0)
            state = _exit_state_vec(
                frame,
                side=side,
                entry_idx=idx,
                current_idx=idx + k,
                entry_px=entry,
                px=close[idx + k],
                notional=notional,
                tp=tp,
                sl=sl,
                hold=k,
                ref_horizon=horizon,
                mae=mae,
                mfe=mfe,
            )
            rows.append(np.concatenate([x_all[idx + k], state]))
            labels.append(close_label)
            weights.append(float(np.clip(abs(margin) * float(weight_scale) + 0.25, 0.25, 6.0)))
        terminal_samples_skipped += 1
    if not rows:
        raise RuntimeError("empty exit dataset")
    meta = {
        "samples": int(len(rows)),
        "close_rate": float(np.mean(labels)),
        "trade_entries_used": int(len(trade_idx)),
        "state_dim": int(len(EXIT_STATE_FEATURES)),
        "state_features": EXIT_STATE_FEATURES,
        "cost_mult": float(cost_mult),
        "min_net_edge": float(min_net_edge),
        "weight_scale": float(weight_scale),
        "step": int(step),
        "terminal_samples_skipped": int(terminal_samples_skipped),
    }
    return np.vstack(rows), np.asarray(labels, dtype=np.int64), np.asarray(weights, dtype=np.float64), meta


def _exit_cache_path(
    *,
    cache_dir: Path,
    variant: str,
    train: pd.DataFrame,
    cfg: PolicyConfig,
    max_samples: int,
    step: int,
    bucket_preset: str,
    cost_mult: float,
    min_net_edge: float,
    weight_scale: float,
) -> Path:
    key_payload = {
        "version": "alpha6-exit-dataset-v4",
        "variant": str(variant),
        "rows": int(len(train)),
        "ts0": str(train["timestamp"].iloc[0]) if len(train) else "",
        "ts1": str(train["timestamp"].iloc[-1]) if len(train) else "",
        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        "max_samples": int(max_samples),
        "step": int(step),
        "bucket_preset": str(bucket_preset),
        "cost_mult": float(cost_mult),
        "min_net_edge": float(min_net_edge),
        "weight_scale": float(weight_scale),
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{variant}_{digest}.joblib"


def _load_or_build_exit_dataset(
    frame: pd.DataFrame,
    x_all: np.ndarray,
    valid: np.ndarray,
    y: dict[str, np.ndarray],
    cfg: PolicyConfig,
    *,
    cache_dir: Path | None,
    variant: str,
    max_samples: int,
    step: int,
    bucket_preset: str,
    cost_mult: float,
    min_net_edge: float,
    weight_scale: float,
    disable_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = None
    if cache_dir is not None and not disable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _exit_cache_path(
            cache_dir=cache_dir,
            variant=variant,
            train=frame,
            cfg=cfg,
            max_samples=max_samples,
            step=step,
            bucket_preset=bucket_preset,
            cost_mult=cost_mult,
            min_net_edge=min_net_edge,
            weight_scale=weight_scale,
        )
        if cache_path.exists():
            payload = joblib.load(cache_path)
            meta = dict(payload["meta"])
            meta["cache_hit"] = True
            meta["cache_path"] = str(cache_path)
            return payload["x"], payload["y"], payload["w"], meta
    try:
        x_exit, y_exit, w_exit, meta = _build_exit_dataset(
            frame,
            x_all,
            valid,
            y,
            cfg,
            max_samples=max_samples,
            step=step,
            cost_mult=cost_mult,
            min_net_edge=min_net_edge,
            weight_scale=weight_scale,
        )
    except RuntimeError as exc:
        if "empty exit dataset" not in str(exc):
            raise
        x_exit = np.empty((0, x_all.shape[1] + len(EXIT_STATE_FEATURES)), dtype=np.float64)
        y_exit = np.empty((0,), dtype=np.int64)
        w_exit = np.empty((0,), dtype=np.float64)
        meta = {
            "samples": 0,
            "close_rate": 0.0,
            "trade_entries_used": 0,
            "state_dim": int(len(EXIT_STATE_FEATURES)),
            "state_features": EXIT_STATE_FEATURES,
            "cost_mult": float(cost_mult),
            "min_net_edge": float(min_net_edge),
            "weight_scale": float(weight_scale),
            "step": int(step),
            "terminal_samples_skipped": 0,
            "empty_dataset": True,
        }
    meta = dict(meta)
    meta["cache_hit"] = False
    if cache_path is not None:
        joblib.dump({"x": x_exit, "y": y_exit, "w": w_exit, "meta": meta}, cache_path)
        meta["cache_path"] = str(cache_path)
    return x_exit, y_exit, w_exit, meta


def _fit_exit_model(x: np.ndarray, y: np.ndarray, w: np.ndarray, args: argparse.Namespace) -> CatBoostClassifier:
    if np.unique(y).size < 2:
        return _ConstantExitModel(int(y[0]) if len(y) else 0)  # type: ignore[return-value]
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "iterations": int(args.exit_iterations),
        "learning_rate": float(args.exit_learning_rate),
        "depth": int(args.exit_depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(args.seed + 777),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    model = CatBoostClassifier(**params)
    model.fit(Pool(x, y, weight=w))
    return model


class _ConstantExitModel:
    def __init__(self, cls: int) -> None:
        self.cls = int(cls)
        self.classes_ = np.asarray([self.cls], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1), dtype=np.float64)


def _exit_close_prob(model: CatBoostClassifier, x_row: np.ndarray, state: np.ndarray) -> float:
    probs = model.predict_proba(np.concatenate([x_row, state])[None, :])[0]
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return 0.0
    return float(probs[int(np.flatnonzero(classes == 1)[0])])


def _backtest_entry_exit(
    frame: pd.DataFrame,
    x_val: np.ndarray,
    entry_dec: pd.DataFrame,
    exit_model: CatBoostClassifier,
    *,
    entry_threshold: float,
    exit_threshold: float,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    entry_exit_feedback: float,
    feedback_window_bars: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    exposure = 0.0
    hold = 0
    tp = sl = 0.0
    mae = mfe = 0.0
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    entry_feedback_blocks = 0
    exit_model_close_bars: list[int] = []
    exits: dict[str, int] = {}
    exposure_sum = 0.0

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, row: pd.Series) -> None:
        nonlocal side, entry, entry_equity, exposure, hold, tp, sl, cash, exposure_sum, long_entries, short_entries, mae, mfe
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        exposure = float(np.clip(row.notional, 0.01, 2.0))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        tp = float(max(row.take_profit, 1e-4))
        sl = float(max(row.stop_loss, 1e-4))
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, tp, sl, exposure, trades, wins, mae, mfe
        if fill_px is None:
            fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        tp = sl = exposure = mae = mfe = 0.0

    for i in range(len(frame) - 2):
        row = entry_dec.iloc[i]
        if feedback_window_bars > 0:
            exit_model_close_bars = [j for j in exit_model_close_bars if i - j <= int(feedback_window_bars)]
        feedback_rate = len(exit_model_close_bars) / max(float(feedback_window_bars), 1.0)
        dynamic_entry_threshold = float(entry_threshold) + float(entry_exit_feedback) * feedback_rate
        desired = int(row.action) if float(row.quality_score) >= dynamic_entry_threshold else 0
        if desired == 0 and int(row.action) != 0 and float(row.quality_score) >= float(entry_threshold):
            entry_feedback_blocks += 1
        if side != 0:
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + tp)
                sl_hit = low[i] <= entry * (1.0 - sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - sl) * (1.0 - slip))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + tp) * (1.0 - slip))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - sl) * (1.0 - slip))
            else:
                tp_hit = low[i] <= entry * (1.0 - tp)
                sl_hit = high[i] >= entry * (1.0 + sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + sl) * (1.0 + slip))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - tp) * (1.0 + slip))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + sl) * (1.0 + slip))
            if side != 0 and hold >= int(min_exit_hold):
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=max(i - hold, 0),
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    notional=exposure,
                    tp=tp,
                    sl=sl,
                    hold=hold,
                    ref_horizon=int(state_horizon),
                    mae=mae,
                    mfe=mfe,
                )
                if _exit_close_prob(exit_model, x_val[i], state) >= float(exit_threshold):
                    exit_model_closes += 1
                    exit_model_close_bars.append(i)
                    exit_pos(i, "exit_model")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1, row)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "exit_model_closes": int(exit_model_closes),
        "entry_feedback_blocks": int(entry_feedback_blocks),
        "exits": exits,
    }


def _entry_threshold_grid(dec: pd.DataFrame, n: int) -> np.ndarray:
    active = dec.loc[dec["action"] != 0, "quality_score"].to_numpy(dtype=np.float64)
    active = active[np.isfinite(active)]
    if active.size == 0:
        return np.array([np.inf])
    return np.unique(np.quantile(active, np.linspace(0.10, 0.995, int(n))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 CatBoost entry/risk model with separate exit layer.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="stable48_global_pca32")
    ap.add_argument("--iterations", type=int, default=700)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2-leaf-reg", type=float, default=6.0)
    ap.add_argument("--exit-iterations", type=int, default=500)
    ap.add_argument("--exit-learning-rate", type=float, default=0.045)
    ap.add_argument("--exit-depth", type=int, default=6)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--exit-step", type=int, default=1)
    ap.add_argument("--exit-max-trades", type=int, default=12000)
    ap.add_argument("--exit-cost-mult", type=float, default=3.0)
    ap.add_argument("--exit-min-net-edge", type=float, default=0.00010)
    ap.add_argument("--exit-weight-scale", type=float, default=80.0)
    ap.add_argument("--entry-thresholds", type=int, default=40)
    ap.add_argument("--exit-threshold-grid", default="0.45,0.55,0.65,0.75,0.85")
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--entry-exit-feedback", type=float, default=0.0005)
    ap.add_argument("--entry-feedback-window-bars", type=int, default=288)
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--bucket-preset", choices=["full", "fast"], default="fast")
    ap.add_argument("--label-cache-dir", type=Path, default=DEFAULT_LABEL_CACHE_DIR)
    ap.add_argument("--exit-cache-dir", type=Path, default=DEFAULT_EXIT_CACHE_DIR)
    ap.add_argument("--disable-label-cache", action="store_true")
    ap.add_argument("--disable-exit-cache", action="store_true")
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _apply_bucket_preset(PolicyConfig(), str(args.bucket_preset))
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0
    feat, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), EXIT_CONTEXT_COLS)
    frame = feat.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame[frame["dataset_split"].astype(str).str.lower().ne("train")].copy()
    if args.smoke:
        train = train.iloc[: min(len(train), 5000)].copy()
        val = val.iloc[: min(len(val), 3000)].copy()
        args.iterations = min(args.iterations, 20)
        args.exit_iterations = min(args.exit_iterations, 20)
        args.entry_thresholds = min(args.entry_thresholds, 8)
        args.stride_bars = max(args.stride_bars, 6)
        args.exit_max_trades = min(args.exit_max_trades, 1000)
    x_train_all, x_val, model_features, pipe = _feature_matrix(
        train,
        val,
        present,
        use_pca=use_pca,
        pca_components=int(spec.get("extra_pca_components") or 0),
    )
    valid, y, label_meta = _load_or_build_lifecycle_labels(
        train,
        cfg,
        stride_bars=args.stride_bars,
        batch_size=args.batch_size,
        cache_dir=args.label_cache_dir,
        variant=args.variant,
        bucket_preset=str(args.bucket_preset),
        disable_cache=bool(args.disable_label_cache),
    )
    x_entry = x_train_all[valid]
    print(
        f"[alpha6-v3] variant={args.variant} train_rows={len(train)} val_rows={len(val)} labels={len(valid)} features={len(model_features)} use_pca={use_pca}",
        flush=True,
    )
    entry_models = _fit_models(x_entry, y, args)
    x_exit, y_exit, w_exit, exit_meta = _load_or_build_exit_dataset(
        train,
        x_train_all,
        valid,
        y,
        cfg,
        cache_dir=args.exit_cache_dir,
        variant=args.variant,
        max_samples=int(args.exit_max_trades),
        step=int(args.exit_step),
        bucket_preset=str(args.bucket_preset),
        cost_mult=float(args.exit_cost_mult),
        min_net_edge=float(args.exit_min_net_edge),
        weight_scale=float(args.exit_weight_scale),
        disable_cache=bool(args.disable_exit_cache),
    )
    print(f"[alpha6-v3] exit_samples={len(y_exit)} close_rate={np.mean(y_exit):.3f}", flush=True)
    exit_model = _fit_exit_model(x_exit, y_exit, w_exit, args)
    dec = _predict_policy(entry_models, x_val, val, cfg)
    exit_thresholds = [float(x.strip()) for x in str(args.exit_threshold_grid).split(",") if x.strip()]
    rows = []
    best: dict[str, Any] | None = None
    for eth in _entry_threshold_grid(dec, args.entry_thresholds):
        for xth in exit_thresholds:
            bt = {
                f"cost{m}": _backtest_entry_exit(
                    val,
                    x_val,
                    dec,
                    exit_model,
                    entry_threshold=float(eth),
                    exit_threshold=float(xth),
                    fee=cfg.fee * m,
                    slip=cfg.slip * m,
                    min_exit_hold=int(args.min_exit_hold),
                    state_horizon=int(cfg.max_train_horizon_bars),
                    entry_exit_feedback=float(args.entry_exit_feedback),
                    feedback_window_bars=int(args.entry_feedback_window_bars),
                )
                for m in (1, 2, 3)
            }
            score = _score(bt["cost1"], bt["cost2"], bt["cost3"])
            row = {
                "entry_threshold": float(eth),
                "exit_threshold": float(xth),
                "score": float(score),
                "pnl": float(bt["cost1"]["pnl"]),
                "mdd": float(bt["cost1"]["mdd"]),
                "trades": int(bt["cost1"]["trades"]),
                "trades_per_day": float(bt["cost1"]["trades_per_day"]),
                "wr": float(bt["cost1"]["wr"]),
                "long_entries": int(bt["cost1"]["long_entries"]),
                "short_entries": int(bt["cost1"]["short_entries"]),
                "avg_notional": float(bt["cost1"]["avg_notional"]),
                "exit_model_closes": int(bt["cost1"]["exit_model_closes"]),
                "entry_feedback_blocks": int(bt["cost1"].get("entry_feedback_blocks", 0)),
                "exits": json.dumps(bt["cost1"]["exits"], sort_keys=True),
            }
            rows.append(row)
            if best is None or row["score"] > best["summary"]["score"]:
                best = {"summary": row, "backtest": bt}
    assert best is not None
    prefix = args.out_dir / args.variant
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(f"{prefix}_threshold_grid.csv", index=False)
    pred = val[["timestamp", "open", "high", "low", "close", "label_action"]].copy()
    for col in dec.columns:
        pred[col] = dec[col].to_numpy()
    pred.to_csv(f"{prefix}_val_entry_predictions.csv", index=False)
    artifact = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "config": cfg,
        "feature_cols": present,
        "model_features": model_features,
        "missing_features": missing,
        "use_pca": use_pca,
        "pipeline": pipe,
        "entry_models": entry_models,
        "exit_model": exit_model,
        "exit_meta": exit_meta,
        "exit_state_features": EXIT_STATE_FEATURES,
    }
    joblib.dump(artifact, f"{prefix}_bundle.joblib")
    summary = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "label_meta": label_meta,
        "exit_meta": exit_meta,
        "entry_label_distribution": entry_models["label_distribution"],
        "raw_feature_count": int(len(present)),
        "missing_features": missing,
        "model_feature_count": int(len(model_features)),
        "use_pca": bool(use_pca),
        "best": best["summary"],
        "best_backtest": best["backtest"],
        "params": vars(args),
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
