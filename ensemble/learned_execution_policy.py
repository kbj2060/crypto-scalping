from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


FEATURE_COLS = [
    "source_id",
    "side",
    "macro_momentum",
    "macro_abs_momentum",
    "log_return",
    "volatility_z",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "liquidity_vacuum",
    "execution_quality",
    "jump_z",
    "evt_tail_flag",
    "evt_excess_z",
    "funding_abs",
    "funding_pressure",
    "funding_price_divergence",
    "long_squeeze_risk",
    "crowding_pressure",
    "smart_money_flow",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "whale_conviction",
    "m7_gate_block",
    "m7_tail_risk",
    "m7_expected_ret",
    "m7_composite_score",
    "m7_confidence",
    "m7_qwidth",
    "patchtst_pred",
    "patchtst_confidence",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "trade_intensity",
    "big_trade_ratio",
    "whale_retail_ratio",
    "squeeze_power",
    "breakout_strength",
    "regime_bull_id",
    "regime_bear_id",
    "regime_chop_id",
    "regime_whipsaw_id",
    "regime_normal_id",
]


@dataclass(frozen=True)
class LearnedExecutionConfig:
    notional_buckets: tuple[float, ...] = (0.75, 1.00, 1.35, 1.75, 2.25, 3.00, 3.50)
    leverage_buckets: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    take_profit_buckets: tuple[float, ...] = (0.025, 0.050, 0.100, 0.250, 0.750, 1.250)
    max_hold_buckets: tuple[int, ...] = (12, 24, 48, 96, 288, 864, 3000)
    stop_loss: float = 0.030
    max_train_horizon_bars: int = 864
    fee: float = 0.0005
    slip: float = 0.0002
    adverse_penalty: float = 0.40
    horizon_penalty: float = 0.004
    size_penalty: float = 0.006
    leverage_bonus: float = 0.002
    min_notional: float = 0.50
    max_margin_fraction: float = 1.0
    tail_notional: float = 3.0
    tail_take_profit: float = 1.25
    tail_return_threshold: float = 0.75
    tail_prob_threshold: float = 0.22


@dataclass(frozen=True)
class LearnedExecutionDecision:
    source: str
    side: int
    notional_exposure: float
    leverage: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    position_fraction: float
    quality_score: float
    confidence: float

    def to_risk_decision(self) -> dict[str, Any]:
        side = float(np.sign(self.side))
        return {
            "allow_entry": bool(side != 0.0 and self.notional_exposure > 0.0),
            "effective_action": side,
            "position_fraction": float(self.position_fraction),
            "leverage": float(self.leverage),
            "notional_exposure": float(self.notional_exposure),
            "target_notional_exposure": float(self.notional_exposure),
            "allow_resize": False,
            "resize_notional_delta": 0.0,
            "block_reason": "",
            "exit_reason": "",
            "resize_reason": "learned_execution_policy",
            "take_profit": float(self.take_profit),
            "stop_loss": float(self.stop_loss),
            "max_hold_bars": int(self.max_hold_bars),
            "quality_score": float(self.quality_score),
            "confidence": float(self.confidence),
        }

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _feature_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    side: np.ndarray,
    macro_momentum: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    source_id = 1.0 if str(source).lower() == "macro" else 2.0 if str(source).lower() == "sniper" else 0.0
    out["source_id"] = float(source_id)
    out["side"] = np.asarray(side, dtype=np.float64)
    out["macro_momentum"] = np.asarray(macro_momentum, dtype=np.float64)
    out["macro_abs_momentum"] = np.abs(out["macro_momentum"].to_numpy(dtype=np.float64))
    for col in FEATURE_COLS:
        if col in out.columns:
            continue
        if col.endswith("_id") and col.startswith("regime_"):
            name = col.replace("regime_", "").replace("_id", "")
            raw_col = f"regime_{name}"
            out[col] = pd.to_numeric(frame[raw_col], errors="coerce") if raw_col in frame.columns else 0.0
            continue
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            out[col] = 0.0
    return out.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan)


def prepare_execution_features(
    frame: pd.DataFrame,
    *,
    source: str,
    side: int | np.ndarray,
    macro_momentum: float | np.ndarray,
) -> pd.DataFrame:
    n = len(frame)
    side_arr = np.full(n, int(side), dtype=np.float64) if np.isscalar(side) else np.asarray(side, dtype=np.float64)
    mom_arr = np.full(n, float(macro_momentum), dtype=np.float64) if np.isscalar(macro_momentum) else np.asarray(macro_momentum, dtype=np.float64)
    return _feature_frame(frame, source=source, side=side_arr, macro_momentum=mom_arr)


def _classes_to_bucket(model: Any, x: pd.DataFrame, buckets: tuple[float, ...]) -> tuple[float, float]:
    proba = model.predict_proba(x)[0]
    classes = np.asarray(model.classes_, dtype=int)
    vals = np.asarray([buckets[int(c)] for c in classes], dtype=np.float64)
    value = float(np.sum(proba * vals))
    confidence = float(np.max(proba))
    return value, confidence


def predict_learned_execution(
    bundle: Mapping[str, Any],
    row: pd.DataFrame | pd.Series,
    *,
    source: str,
    side: int,
    macro_momentum: float,
) -> LearnedExecutionDecision:
    if isinstance(row, pd.Series):
        frame = row.to_frame().T
    else:
        frame = row.tail(1).copy()
    cfg = LearnedExecutionConfig(**dict(bundle.get("config", {})))
    x = prepare_execution_features(frame, source=source, side=int(side), macro_momentum=float(macro_momentum))
    notional, c1 = _classes_to_bucket(bundle["notional_model"], x, cfg.notional_buckets)
    leverage, c2 = _classes_to_bucket(bundle["leverage_model"], x, cfg.leverage_buckets)
    take_profit, c3 = _classes_to_bucket(bundle["take_profit_model"], x, cfg.take_profit_buckets)
    hold_raw, c4 = _classes_to_bucket(bundle["max_hold_model"], x, tuple(float(v) for v in cfg.max_hold_buckets))
    quality = float(bundle["quality_model"].predict(x)[0]) if "quality_model" in bundle else 0.0
    tail_prob = 0.0
    if "tail_model" in bundle:
        classes = list(getattr(bundle["tail_model"], "classes_", []))
        proba = bundle["tail_model"].predict_proba(x)[0]
        if 1 in classes:
            tail_prob = float(proba[classes.index(1)])
        elif len(proba):
            tail_prob = float(proba[-1])
    if tail_prob >= float(cfg.tail_prob_threshold):
        notional = max(float(notional), float(cfg.tail_notional))
        leverage = max(float(leverage), max(cfg.leverage_buckets))
        take_profit = max(float(take_profit), float(cfg.tail_take_profit))
    leverage = float(np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets)))
    notional = float(np.clip(notional, cfg.min_notional, max(cfg.notional_buckets)))
    position_fraction = float(np.clip(notional / max(leverage, 1e-8), 0.0, cfg.max_margin_fraction))
    notional = float(position_fraction * leverage)
    return LearnedExecutionDecision(
        source=str(source),
        side=int(np.sign(side)),
        notional_exposure=notional,
        leverage=leverage,
        take_profit=float(np.clip(take_profit, min(cfg.take_profit_buckets), max(cfg.take_profit_buckets))),
        stop_loss=float(cfg.stop_loss),
        max_hold_bars=int(round(hold_raw)),
        position_fraction=position_fraction,
        quality_score=quality,
        confidence=float(max(np.mean([c1, c2, c3, c4]), tail_prob)),
    )


def _first_hit(path: np.ndarray, tp: float, stop: float, max_hold: int) -> int:
    m = min(int(max_hold), len(path))
    if m <= 1:
        return 0
    p = path[:m]
    hit = np.flatnonzero((p >= float(tp)) | (p <= -abs(float(stop))))
    return int(hit[0]) if hit.size else int(m - 1)


def _label_batch(
    future_ret: np.ndarray,
    cfg: LearnedExecutionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = future_ret.shape[0]
    notional_y = np.zeros(n, dtype=np.int64)
    leverage_y = np.zeros(n, dtype=np.int64)
    tp_y = np.zeros(n, dtype=np.int64)
    hold_y = np.zeros(n, dtype=np.int64)
    quality_y = np.zeros(n, dtype=np.float64)
    tail_y = np.zeros(n, dtype=np.int64)
    cost = float(cfg.fee + cfg.slip) * 2.0
    for i in range(n):
        best_score = -1e18
        best = (0, 0, 0, 0, 0.0)
        raw = future_ret[i]
        if not np.isfinite(raw).any():
            continue
        tail_y[i] = int(float(np.nanmax(raw) * float(cfg.tail_notional)) >= float(cfg.tail_return_threshold))
        for ni, notional in enumerate(cfg.notional_buckets):
            path = np.nan_to_num(raw * float(notional), nan=0.0, posinf=0.0, neginf=0.0)
            for ti, tp in enumerate(cfg.take_profit_buckets):
                for hi, hold in enumerate(cfg.max_hold_buckets):
                    exit_i = _first_hit(path, float(tp), float(cfg.stop_loss), int(hold))
                    pnl = float(path[exit_i] - cost * float(notional))
                    adverse = max(0.0, -float(np.min(path[: exit_i + 1])))
                    horizon_frac = float(exit_i + 1) / max(float(cfg.max_train_horizon_bars), 1.0)
                    for li, leverage in enumerate(cfg.leverage_buckets):
                        margin = float(notional) / max(float(leverage), 1e-8)
                        if margin > float(cfg.max_margin_fraction) + 1e-9:
                            continue
                        liq_buffer = 0.72 / max(float(leverage), 1.0)
                        liq_penalty = 2.5 * max(0.0, adverse - liq_buffer)
                        score = (
                            pnl
                            - float(cfg.adverse_penalty) * adverse
                            - float(cfg.horizon_penalty) * horizon_frac
                            - float(cfg.size_penalty) * (float(notional) / max(cfg.notional_buckets)) ** 2
                            - liq_penalty
                            + float(cfg.leverage_bonus) * (float(leverage) / max(cfg.leverage_buckets)) * max(pnl, 0.0)
                        )
                        if score > best_score:
                            best_score = score
                            best = (ni, li, ti, hi, score)
        notional_y[i], leverage_y[i], tp_y[i], hold_y[i], quality_y[i] = best
    return notional_y, leverage_y, tp_y, hold_y, quality_y, tail_y


def build_execution_training_set(
    frame: pd.DataFrame,
    *,
    close: np.ndarray,
    macro_signal: np.ndarray,
    macro_momentum: np.ndarray,
    cfg: LearnedExecutionConfig,
    candidate_stride_bars: int = 6,
    batch_size: int = 1024,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    n = len(frame)
    max_h = int(cfg.max_train_horizon_bars)
    valid = np.flatnonzero((np.asarray(macro_signal) != 0) & np.isfinite(macro_momentum))
    valid = valid[(valid > 0) & (valid < n - max_h - 1)]
    if candidate_stride_bars > 1:
        valid = valid[valid % int(candidate_stride_bars) == 0]
    if valid.size == 0:
        raise ValueError("no execution training candidates")
    horizons = np.arange(1, max_h + 1, dtype=np.int64)
    sides = np.sign(np.asarray(macro_signal[valid], dtype=np.float64)).astype(np.int64)
    x = prepare_execution_features(
        frame.iloc[valid].reset_index(drop=True),
        source="macro",
        side=sides,
        macro_momentum=np.asarray(macro_momentum[valid], dtype=np.float64),
    )
    ys = {
        "notional": np.zeros(valid.size, dtype=np.int64),
        "leverage": np.zeros(valid.size, dtype=np.int64),
        "take_profit": np.zeros(valid.size, dtype=np.int64),
        "max_hold": np.zeros(valid.size, dtype=np.int64),
        "quality": np.zeros(valid.size, dtype=np.float64),
        "tail": np.zeros(valid.size, dtype=np.int64),
    }
    close = np.asarray(close, dtype=np.float64)
    for start in range(0, valid.size, int(batch_size)):
        end = min(start + int(batch_size), valid.size)
        idx = valid[start:end]
        fut_idx = idx[:, None] + horizons[None, :]
        entry = close[idx][:, None]
        fut = close[fut_idx]
        ret = (fut / np.maximum(entry, 1e-12) - 1.0) * sides[start:end, None]
        y = _label_batch(ret, cfg)
        ys["notional"][start:end] = y[0]
        ys["leverage"][start:end] = y[1]
        ys["take_profit"][start:end] = y[2]
        ys["max_hold"][start:end] = y[3]
        ys["quality"][start:end] = y[4]
        ys["tail"][start:end] = y[5]
    meta = {
        "candidates": int(valid.size),
        "candidate_stride_bars": int(candidate_stride_bars),
        "max_train_horizon_bars": int(max_h),
        "source": "macro",
    }
    return x, ys, meta


def _classifier(random_state: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            early_stopping=False,
            random_state=int(random_state),
        ),
    )


def _regressor(random_state: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            early_stopping=False,
            random_state=int(random_state),
        ),
    )


def train_learned_execution_policy(
    x: pd.DataFrame,
    y: Mapping[str, np.ndarray],
    *,
    cfg: LearnedExecutionConfig,
    random_state: int = 42,
) -> dict[str, Any]:
    notional_model = _classifier(random_state)
    leverage_model = _classifier(random_state + 1)
    take_profit_model = _classifier(random_state + 2)
    max_hold_model = _classifier(random_state + 3)
    quality_model = _regressor(random_state + 4)
    weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64)), 0.02, 1.0)
    notional_model.fit(x, y["notional"], histgradientboostingclassifier__sample_weight=weights)
    leverage_model.fit(x, y["leverage"], histgradientboostingclassifier__sample_weight=weights)
    take_profit_model.fit(x, y["take_profit"], histgradientboostingclassifier__sample_weight=weights)
    max_hold_model.fit(x, y["max_hold"], histgradientboostingclassifier__sample_weight=weights)
    quality_model.fit(x, y["quality"], histgradientboostingregressor__sample_weight=weights)
    tail_model = None
    if np.unique(y["tail"]).size >= 2:
        tail_model = _classifier(random_state + 5)
        tail_model.fit(x, y["tail"], histgradientboostingclassifier__sample_weight=np.maximum(weights, 0.10))
    bundle = {
        "model_type": "learned_execution_policy_v1",
        "feature_cols": list(FEATURE_COLS),
        "config": asdict(cfg),
        "notional_model": notional_model,
        "leverage_model": leverage_model,
        "take_profit_model": take_profit_model,
        "max_hold_model": max_hold_model,
        "quality_model": quality_model,
        "label_distribution": {
            "notional": pd.Series(y["notional"]).value_counts().sort_index().to_dict(),
            "leverage": pd.Series(y["leverage"]).value_counts().sort_index().to_dict(),
            "take_profit": pd.Series(y["take_profit"]).value_counts().sort_index().to_dict(),
            "max_hold": pd.Series(y["max_hold"]).value_counts().sort_index().to_dict(),
            "quality_mean": float(np.mean(y["quality"])),
            "quality_p95": float(np.quantile(y["quality"], 0.95)),
            "tail": pd.Series(y["tail"]).value_counts().sort_index().to_dict(),
        },
    }
    if tail_model is not None:
        bundle["tail_model"] = tail_model
    return bundle
