from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional import guard for lean runtimes.
    CatBoostClassifier = None


TARGET_REGIMES = ("whipsaw", "normal", "chop")

BASE_FEATURES = [
    "smart_money_flow",
    "oi_change_rate",
    "taker_acceleration",
    "log_return",
    "whale_retail_ratio",
    "net_taker_ratio",
    "trade_intensity",
    "big_trade_ratio",
    "volatility_z",
    "rsi",
    "wick_ratio",
    "garman_klass_vol",
    "amihud_illiquidity_z",
    "cvp_volume_imbalance",
    "cvp_cluster_position",
    "long_squeeze_risk",
    "funding_price_divergence",
    "ofi_acceleration",
    "whale_conviction",
    "funding_pressure",
    "regime_persistence",
    "cross_scale_curvature",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "hour_cos",
]

ROLLING_FEATURES = [
    "smart_money_flow",
    "taker_acceleration",
    "net_taker_ratio",
    "trade_intensity",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
    "ofi_acceleration",
    "whale_conviction",
]

NON_FEATURE_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "evt_candidate_label",
    "evt_candidate_side",
    "evt_long_score",
    "evt_short_score",
    "evt_candidate_quality",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}

LEAKY_FEATURE_PREFIXES = ("evt_",)
LEAKY_FEATURE_FRAGMENTS = ("candidate",)


@dataclass(frozen=True)
class MicrostructureSleeveConfig:
    entry_confidence: float = 0.42
    entry_gap: float = 0.16
    max_hold_bars: int = 24
    stop_loss: float = 0.0055
    take_profit: float = 0.0120
    trailing_stop: float = 0.0055
    max_notional_exposure: float = 5.0
    min_notional_exposure: float = 0.40
    max_leverage: float = 5.0
    cooldown_bars: int = 1
    whipsaw_notional_mult: float = 1.20
    chop_notional_mult: float = 1.00
    normal_notional_mult: float = 1.00
    fee: float = 0.0005
    slippage: float = 0.0002
    portfolio_soft_drawdown: float = 0.20
    portfolio_hard_drawdown: float = 0.35
    portfolio_min_drawdown_scale: float = 0.55


@dataclass
class MicrostructureSleeveBacktest:
    total_return_pct: float
    mdd_pct: float
    trades: int
    win_rate: float
    trades_per_day: float
    long_entries: int
    short_entries: int
    regime_entries: dict[str, int]
    avg_notional_exposure: float
    max_notional_exposure: float
    config: dict[str, Any]
    trade_ledger: list[dict[str, Any]]

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicrostructureSleeveDecision:
    allow_entry: bool
    side: str
    confidence: float
    probability_gap: float
    notional_exposure: float
    leverage: float
    position_fraction: float
    regime: str
    block_reason: str
    source: str = "microstructure_wnc_sleeve_v2"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def to_risk_decision(self) -> dict[str, Any]:
        direction = 1.0 if self.side == "LONG" else (-1.0 if self.side == "SHORT" else 0.0)
        return {
            "allow_entry": bool(self.allow_entry),
            "effective_action": float(direction if self.allow_entry else 0.0),
            "position_fraction": float(self.position_fraction),
            "leverage": float(self.leverage),
            "notional_exposure": float(self.notional_exposure),
            "target_notional_exposure": float(self.notional_exposure),
            "allow_resize": False,
            "resize_notional_delta": 0.0,
            "block_reason": str(self.block_reason),
            "exit_reason": "",
            "resize_reason": "microstructure_sleeve_no_resize",
            "sizing_reason": self.source,
            "source": self.source,
            "regime": self.regime,
            "confidence": float(self.confidence),
            "probability_gap": float(self.probability_gap),
        }


class WeightedProbabilityEnsemble:
    """Small joblib-safe probability ensemble with class-aligned averaging."""

    def __init__(self, models: list[Any], weights: list[float], classes: tuple[int, ...] = (0, 1, 2)) -> None:
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        self.models = list(models)
        self.weights = [float(w) for w in weights]
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        denom = 0.0
        class_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        for model, weight in zip(self.models, self.weights):
            if weight <= 0.0:
                continue
            p = np.asarray(model.predict_proba(x), dtype=np.float64)
            model_classes = [int(c) for c in getattr(model, "classes_", self.classes_)]
            for j, cls in enumerate(model_classes):
                if cls in class_to_idx:
                    out[:, class_to_idx[cls]] += float(weight) * p[:, j]
            denom += float(weight)
        out /= max(denom, 1e-12)
        row_sum = out.sum(axis=1, keepdims=True)
        return out / np.maximum(row_sum, 1e-12)


def required_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "evt_candidate_label",
                "regime_whipsaw",
                "regime_normal",
                "regime_chop",
            ]
            + BASE_FEATURES
        )
    )


def _raw_regime_series(frame: pd.DataFrame) -> pd.Series:
    cols = [f"regime_{name}" for name in ("whipsaw", "normal", "chop", "bull", "bear")]
    present = [c for c in cols if c in frame.columns]
    if not present:
        return pd.Series(["normal"] * len(frame), index=frame.index)
    vals = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = [c.replace("regime_", "") for c in present]
    arr = vals.to_numpy(dtype=float)
    idx = np.argmax(arr, axis=1)
    out = pd.Series([labels[int(i)] for i in idx], index=frame.index)
    out.loc[vals.sum(axis=1) <= 0.0] = "normal"
    return out.astype(str).str.lower()


def _is_model_feature(col: str) -> bool:
    if col in NON_FEATURE_COLUMNS:
        return False
    lower = str(col).lower()
    if lower in QUARANTINED_REGIME_V2_FEATURES:
        return False
    if any(lower.startswith(prefix) for prefix in LEAKY_FEATURE_PREFIXES):
        return False
    if any(fragment in lower for fragment in LEAKY_FEATURE_FRAGMENTS):
        return False
    return True


def prepare_microstructure_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out.sort_values("timestamp", inplace=True)
        out.reset_index(drop=True, inplace=True)
    for col in BASE_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    for col in ROLLING_FEATURES:
        out[f"{col}_d1"] = out[col].diff().fillna(0.0)
        out[f"{col}_r3"] = out[col].rolling(3, min_periods=1).mean()
        out[f"{col}_r6"] = out[col].rolling(6, min_periods=1).mean()
    feature_cols = [c for c in out.columns if _is_model_feature(c)]
    return out, feature_cols


def _minimal_feature_cols(feature_cols: list[str]) -> list[str]:
    allowed = set(BASE_FEATURES)
    for col in ROLLING_FEATURES:
        allowed.update({f"{col}_d1", f"{col}_r3", f"{col}_r6"})
    return [c for c in feature_cols if c in allowed]


def train_microstructure_classifier(
    train_df: pd.DataFrame,
    *,
    feature_mode: str = "minimal",
    model_mode: str = "hgb",
) -> tuple[Any, list[str], dict[str, Any]]:
    frame, feature_cols = prepare_microstructure_frame(train_df)
    if feature_mode == "minimal":
        feature_cols = _minimal_feature_cols(feature_cols)
    elif feature_mode != "full":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    regimes = _raw_regime_series(frame)
    mask = regimes.isin(TARGET_REGIMES)
    y = pd.to_numeric(frame.loc[mask, "evt_candidate_label"], errors="coerce").fillna(0).astype(int)
    x = frame.loc[mask, feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    y_arr = y.to_numpy(np.int64)
    model_configs: list[dict[str, Any]]
    if model_mode == "hgb":
        model_configs = [
            {
                "name": "hgb_v1",
                "weight": 1.0,
                "params": {
                    "max_iter": 140,
                    "learning_rate": 0.055,
                    "max_leaf_nodes": 31,
                    "l2_regularization": 0.015,
                    "random_state": 7,
                    "class_weight": "balanced",
                },
            }
        ]
        model = HistGradientBoostingClassifier(**model_configs[0]["params"])
        model.fit(x, y_arr)
    elif model_mode == "ensemble":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is required for model_mode='ensemble'")
        model_specs = [
            (
                "hgb_wide",
                1.0,
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.040,
                    max_leaf_nodes=47,
                    l2_regularization=0.010,
                    random_state=101,
                    class_weight="balanced",
                ),
                {
                    "max_iter": 220,
                    "learning_rate": 0.040,
                    "max_leaf_nodes": 47,
                    "l2_regularization": 0.010,
                    "random_state": 101,
                    "class_weight": "balanced",
                },
            ),
            (
                "hgb_fast",
                1.0,
                HistGradientBoostingClassifier(
                    max_iter=140,
                    learning_rate=0.060,
                    max_leaf_nodes=63,
                    l2_regularization=0.035,
                    random_state=102,
                    class_weight="balanced",
                ),
                {
                    "max_iter": 140,
                    "learning_rate": 0.060,
                    "max_leaf_nodes": 63,
                    "l2_regularization": 0.035,
                    "random_state": 102,
                    "class_weight": "balanced",
                },
            ),
            (
                "cat_smooth",
                2.0,
                CatBoostClassifier(
                    iterations=420,
                    depth=6,
                    learning_rate=0.045,
                    l2_leaf_reg=4.0,
                    loss_function="MultiClass",
                    auto_class_weights="Balanced",
                    random_seed=104,
                    verbose=False,
                    allow_writing_files=False,
                ),
                {
                    "iterations": 420,
                    "depth": 6,
                    "learning_rate": 0.045,
                    "l2_leaf_reg": 4.0,
                    "loss_function": "MultiClass",
                    "auto_class_weights": "Balanced",
                    "random_seed": 104,
                },
            ),
            (
                "cat_deep",
                2.0,
                CatBoostClassifier(
                    iterations=320,
                    depth=7,
                    learning_rate=0.055,
                    l2_leaf_reg=7.0,
                    loss_function="MultiClass",
                    auto_class_weights="Balanced",
                    random_seed=105,
                    verbose=False,
                    allow_writing_files=False,
                ),
                {
                    "iterations": 320,
                    "depth": 7,
                    "learning_rate": 0.055,
                    "l2_leaf_reg": 7.0,
                    "loss_function": "MultiClass",
                    "auto_class_weights": "Balanced",
                    "random_seed": 105,
                },
            ),
        ]
        fitted_models: list[Any] = []
        weights: list[float] = []
        model_configs = []
        for name, weight, estimator, params in model_specs:
            estimator.fit(x, y_arr)
            fitted_models.append(estimator)
            weights.append(float(weight))
            model_configs.append({"name": name, "weight": float(weight), "params": params})
        model = WeightedProbabilityEnsemble(fitted_models, weights)
    else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")
    summary = {
        "train_rows": int(len(frame)),
        "train_target_rows": int(mask.sum()),
        "target_regimes": list(TARGET_REGIMES),
        "target_counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "feature_count": int(len(feature_cols)),
        "feature_mode": str(feature_mode),
        "model_mode": str(model_mode),
        "model": type(model).__name__,
        "model_configs": model_configs,
    }
    return model, feature_cols, summary


def predict_microstructure_proba(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    frame, _ = prepare_microstructure_frame(df)
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    x = frame[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    proba = model.predict_proba(x)
    classes = [int(c) for c in list(model.classes_)]
    return frame, proba, classes


def _class_prob(proba: np.ndarray, classes: list[int], idx: int, cls: int) -> float:
    return float(proba[idx, classes.index(cls)]) if cls in classes else 0.0


def _drawdown_scale(equity: float, peak: float, cfg: MicrostructureSleeveConfig) -> float:
    dd = max(0.0, 1.0 - float(equity) / max(float(peak), 1e-12))
    if dd >= cfg.portfolio_hard_drawdown:
        return 0.0
    if dd <= cfg.portfolio_soft_drawdown:
        return 1.0
    pressure = np.clip(
        (dd - cfg.portfolio_soft_drawdown) / max(cfg.portfolio_hard_drawdown - cfg.portfolio_soft_drawdown, 1e-8),
        0.0,
        1.0,
    )
    return float(1.0 - pressure * (1.0 - cfg.portfolio_min_drawdown_scale))


def microstructure_sleeve_decision(
    *,
    row: dict[str, Any] | pd.Series,
    long_prob: float,
    short_prob: float,
    equity: float = 1.0,
    peak_equity: float = 1.0,
    cfg: MicrostructureSleeveConfig | None = None,
) -> MicrostructureSleeveDecision:
    """Convert model probabilities into a governor-compatible entry decision."""
    cfg = cfg or MicrostructureSleeveConfig()
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
    regime = "normal"
    for name in ("whipsaw", "chop", "normal", "bull", "bear"):
        try:
            if float(row_dict.get(f"regime_{name}", 0.0) or 0.0) > 0.5:
                regime = name
                break
        except Exception:
            continue
    side_int = 1 if float(long_prob) > float(short_prob) else -1
    confidence = float(max(float(long_prob), float(short_prob)))
    gap = float(abs(float(long_prob) - float(short_prob)))
    if regime not in TARGET_REGIMES:
        return MicrostructureSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "regime_not_covered")
    if confidence < cfg.entry_confidence:
        return MicrostructureSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "confidence_below_threshold")
    if gap < cfg.entry_gap:
        return MicrostructureSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "probability_gap_below_threshold")

    regime_mult = (
        cfg.whipsaw_notional_mult
        if regime == "whipsaw"
        else cfg.chop_notional_mult
        if regime == "chop"
        else cfg.normal_notional_mult
    )
    conf_scale = ((confidence - cfg.entry_confidence) / max(1.0 - cfg.entry_confidence, 1e-8)) ** 0.70
    dd_scale = _drawdown_scale(float(equity), float(peak_equity), cfg)
    notional = cfg.min_notional_exposure + (cfg.max_notional_exposure - cfg.min_notional_exposure) * conf_scale
    notional = float(np.clip(notional * regime_mult * dd_scale, 0.0, cfg.max_notional_exposure))
    if notional < cfg.min_notional_exposure:
        return MicrostructureSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "notional_below_min")
    leverage = float(max(1.0, cfg.max_leverage))
    position_fraction = float(notional / max(leverage, 1e-8))
    return MicrostructureSleeveDecision(
        allow_entry=True,
        side="LONG" if side_int == 1 else "SHORT",
        confidence=confidence,
        probability_gap=gap,
        notional_exposure=notional,
        leverage=leverage,
        position_fraction=position_fraction,
        regime=regime,
        block_reason="",
    )


def backtest_microstructure_sleeve(
    df: pd.DataFrame,
    proba: np.ndarray,
    classes: list[int],
    cfg: MicrostructureSleeveConfig | None = None,
) -> MicrostructureSleeveBacktest:
    cfg = cfg or MicrostructureSleeveConfig()
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(float)
    regimes = _raw_regime_series(df).to_numpy()
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(range(len(df)))
    allowed = np.isin(regimes, TARGET_REGIMES)

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    entry_ts = None
    notional = 0.0
    best_unrealized = 0.0
    last_exit_idx = -10**9
    wins = 0
    trades: list[dict[str, Any]] = []
    regime_entries = {r: 0 for r in TARGET_REGIMES}
    side_entries = {1: 0, -1: 0}
    exposure_samples: list[float] = []

    entry_cost = float(cfg.fee + cfg.slippage)
    exit_cost = float(cfg.fee + cfg.slippage)
    for i, price in enumerate(close):
        if not np.isfinite(price) or price <= 0.0:
            continue
        if pos:
            unrealized = float(pos * (price / max(entry_price, 1e-12) - 1.0))
            best_unrealized = max(best_unrealized, unrealized)
            long_p = _class_prob(proba, classes, i, 1)
            short_p = _class_prob(proba, classes, i, 2)
            exit_reason = ""
            if unrealized <= -cfg.stop_loss:
                exit_reason = "stop"
            elif unrealized >= cfg.take_profit:
                exit_reason = "take_profit"
            elif best_unrealized >= cfg.trailing_stop * 1.15 and unrealized <= best_unrealized - cfg.trailing_stop:
                exit_reason = "trailing"
            elif i - entry_idx >= int(cfg.max_hold_bars):
                exit_reason = "max_hold"
            elif pos == 1 and short_p >= cfg.entry_confidence + 0.10:
                exit_reason = "opposite_short"
            elif pos == -1 and long_p >= cfg.entry_confidence + 0.10:
                exit_reason = "opposite_long"
            if exit_reason:
                pnl_frac = float(notional * unrealized - notional * exit_cost)
                equity *= max(0.02, 1.0 + pnl_frac)
                wins += int(pnl_frac > 0.0)
                trades.append(
                    {
                        "entry_ts": str(entry_ts),
                        "exit_ts": str(timestamps.iloc[i]),
                        "side": "LONG" if pos == 1 else "SHORT",
                        "entry_price": float(entry_price),
                        "exit_price": float(price),
                        "notional_exposure": float(notional),
                        "hold_bars": int(i - entry_idx),
                        "pnl_frac": float(pnl_frac),
                        "pnl_pct": float(pnl_frac * 100.0),
                        "exit_reason": exit_reason,
                        "regime": str(regimes[entry_idx]),
                    }
                )
                pos = 0
                notional = 0.0
                last_exit_idx = i

        if pos == 0 and allowed[i] and (i - last_exit_idx) >= int(cfg.cooldown_bars):
            long_p = _class_prob(proba, classes, i, 1)
            short_p = _class_prob(proba, classes, i, 2)
            side = 1 if long_p > short_p else -1
            confidence = max(long_p, short_p)
            gap = abs(long_p - short_p)
            if confidence >= cfg.entry_confidence and gap >= cfg.entry_gap:
                reg = str(regimes[i])
                regime_mult = (
                    cfg.whipsaw_notional_mult
                    if reg == "whipsaw"
                    else cfg.chop_notional_mult
                    if reg == "chop"
                    else cfg.normal_notional_mult
                )
                conf_scale = ((confidence - cfg.entry_confidence) / max(1.0 - cfg.entry_confidence, 1e-8)) ** 0.70
                dd_scale = _drawdown_scale(equity, peak, cfg)
                target_notional = (cfg.min_notional_exposure + (cfg.max_notional_exposure - cfg.min_notional_exposure) * conf_scale)
                target_notional = float(np.clip(target_notional * regime_mult * dd_scale, 0.0, cfg.max_notional_exposure))
                if target_notional >= cfg.min_notional_exposure:
                    equity *= max(0.02, 1.0 - target_notional * entry_cost)
                    pos = side
                    entry_price = float(price * (1.0 + cfg.slippage if side == 1 else 1.0 - cfg.slippage))
                    entry_idx = i
                    entry_ts = timestamps.iloc[i]
                    notional = target_notional
                    best_unrealized = 0.0
                    regime_entries[reg] = int(regime_entries.get(reg, 0)) + 1
                    side_entries[side] = int(side_entries.get(side, 0)) + 1
                    exposure_samples.append(float(target_notional))
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1e-12) - 1.0)

    if pos:
        price = float(close[-1])
        unrealized = float(pos * (price / max(entry_price, 1e-12) - 1.0))
        pnl_frac = float(notional * unrealized - notional * exit_cost)
        equity *= max(0.02, 1.0 + pnl_frac)
        wins += int(pnl_frac > 0.0)
        trades.append(
            {
                "entry_ts": str(entry_ts),
                "exit_ts": str(timestamps.iloc[-1]),
                "side": "LONG" if pos == 1 else "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(price),
                "notional_exposure": float(notional),
                "hold_bars": int(len(close) - 1 - entry_idx),
                "pnl_frac": float(pnl_frac),
                "pnl_pct": float(pnl_frac * 100.0),
                "exit_reason": "final_close",
                "regime": str(regimes[entry_idx]),
            }
        )

    if len(timestamps) >= 2:
        elapsed_days = max((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400.0, 1e-8)
    else:
        elapsed_days = 1.0
    trade_count = len(trades)
    return MicrostructureSleeveBacktest(
        total_return_pct=float((equity - 1.0) * 100.0),
        mdd_pct=float(mdd * 100.0),
        trades=int(trade_count),
        win_rate=float(wins / max(trade_count, 1)),
        trades_per_day=float(trade_count / elapsed_days),
        long_entries=int(side_entries[1]),
        short_entries=int(side_entries[-1]),
        regime_entries={k: int(v) for k, v in regime_entries.items()},
        avg_notional_exposure=float(np.mean(exposure_samples) if exposure_samples else 0.0),
        max_notional_exposure=float(max(exposure_samples) if exposure_samples else 0.0),
        config=asdict(cfg),
        trade_ledger=trades,
    )
