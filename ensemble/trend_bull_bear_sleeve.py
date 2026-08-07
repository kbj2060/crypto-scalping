from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
try:
    from catboost import CatBoostClassifier
except ImportError:  # Optional in live runtime when the trend sleeve is disabled.
    CatBoostClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier

from ensemble.microstructure_wnc_sleeve import (
    BASE_FEATURES,
    ROLLING_FEATURES,
    WeightedProbabilityEnsemble,
    _raw_regime_series,
    prepare_microstructure_frame,
)


TARGET_REGIMES = ("bull", "bear")


@dataclass(frozen=True)
class TrendSleeveConfig:
    entry_confidence: float = 0.42
    entry_gap: float = 0.18
    max_hold_bars: int = 36
    stop_loss: float = 0.010
    take_profit: float = 0.030
    trailing_stop: float = 0.010
    max_notional_exposure: float = 3.0
    min_notional_exposure: float = 0.40
    max_leverage: float = 5.0
    cooldown_bars: int = 1
    bull_notional_mult: float = 1.0
    bear_notional_mult: float = 1.0
    fee: float = 0.0005
    slippage: float = 0.0002
    portfolio_soft_drawdown: float = 0.16
    portfolio_hard_drawdown: float = 0.30
    portfolio_min_drawdown_scale: float = 0.60


@dataclass(frozen=True)
class TrendSleeveDecision:
    allow_entry: bool
    side: str
    confidence: float
    probability_gap: float
    notional_exposure: float
    leverage: float
    position_fraction: float
    regime: str
    block_reason: str
    source: str = "trend_bull_bear_sleeve_v1"

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
            "resize_reason": "trend_sleeve_no_resize",
            "sizing_reason": self.source,
            "source": self.source,
            "regime": self.regime,
            "confidence": float(self.confidence),
            "probability_gap": float(self.probability_gap),
        }


def _feature_cols(all_cols: list[str], mode: str) -> list[str]:
    if mode == "full":
        return list(all_cols)
    if mode == "minimal":
        allowed = set(BASE_FEATURES)
        for col in ROLLING_FEATURES:
            allowed.update({f"{col}_d1", f"{col}_r3", f"{col}_r6"})
        return [c for c in all_cols if c in allowed]
    if mode == "no_m7_all":
        return [c for c in all_cols if not c.startswith("m7_")]
    raise ValueError(f"unknown feature_mode: {mode}")


def train_trend_classifier(
    train_df: pd.DataFrame,
    *,
    feature_mode: str = "full",
) -> tuple[WeightedProbabilityEnsemble, list[str], dict[str, Any]]:
    frame, all_cols = prepare_microstructure_frame(train_df)
    feature_cols = _feature_cols(all_cols, feature_mode)
    regimes = _raw_regime_series(frame)
    mask = regimes.isin(TARGET_REGIMES)
    y = pd.to_numeric(frame.loc[mask, "evt_candidate_label"], errors="coerce").fillna(0).astype(int).to_numpy(np.int64)
    x = frame.loc[mask, feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(np.float32)
    specs = [
        (
            "hgb_trend_wide",
            1.0,
            HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.038,
                max_leaf_nodes=63,
                l2_regularization=0.020,
                random_state=211,
                class_weight="balanced",
            ),
        ),
        (
            "hgb_trend_fast",
            1.0,
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.055,
                max_leaf_nodes=47,
                l2_regularization=0.035,
                random_state=212,
                class_weight="balanced",
            ),
        ),
    ]
    if CatBoostClassifier is not None:
        specs.extend(
            [
                (
                    "cat_trend_smooth",
                    2.0,
                    CatBoostClassifier(
                        iterations=460,
                        depth=6,
                        learning_rate=0.040,
                        l2_leaf_reg=5.0,
                        loss_function="MultiClass",
                        auto_class_weights="Balanced",
                        random_seed=213,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
                (
                    "cat_trend_deep",
                    2.0,
                    CatBoostClassifier(
                        iterations=340,
                        depth=7,
                        learning_rate=0.052,
                        l2_leaf_reg=8.0,
                        loss_function="MultiClass",
                        auto_class_weights="Balanced",
                        random_seed=214,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    models: list[Any] = []
    weights: list[float] = []
    model_configs: list[dict[str, Any]] = []
    for name, weight, estimator in specs:
        estimator.fit(x, y)
        models.append(estimator)
        weights.append(float(weight))
        model_configs.append({"name": name, "weight": float(weight), "params": estimator.get_params()})
    model = WeightedProbabilityEnsemble(models, weights)
    summary = {
        "train_rows": int(len(frame)),
        "train_target_rows": int(mask.sum()),
        "target_regimes": list(TARGET_REGIMES),
        "target_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "feature_count": int(len(feature_cols)),
        "feature_mode": str(feature_mode),
        "model": type(model).__name__,
        "model_configs": model_configs,
    }
    return model, feature_cols, summary


def predict_trend_proba(
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


def class_prob(proba: np.ndarray, classes: list[int], idx: int, cls: int) -> float:
    return float(proba[idx, classes.index(cls)]) if cls in classes else 0.0


def _drawdown_scale(equity: float, peak: float, cfg: TrendSleeveConfig) -> float:
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


def trend_sleeve_decision(
    *,
    row: dict[str, Any] | pd.Series,
    long_prob: float,
    short_prob: float,
    no_trade_prob: float = 0.0,
    equity: float = 1.0,
    peak_equity: float = 1.0,
    cfg: TrendSleeveConfig | None = None,
) -> TrendSleeveDecision:
    cfg = cfg or TrendSleeveConfig()
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
    regime = "normal"
    for name in ("bull", "bear", "whipsaw", "chop", "normal"):
        try:
            if float(row_dict.get(f"regime_{name}", 0.0) or 0.0) > 0.5:
                regime = name
                break
        except Exception:
            continue
    if regime not in TARGET_REGIMES:
        return TrendSleeveDecision(False, "NONE", 0.0, 0.0, 0.0, 0.0, 0.0, regime, "regime_not_covered")
    if regime == "bull":
        side = "LONG"
        confidence = float(long_prob)
        gap = float(long_prob - max(float(short_prob), 0.35 * float(no_trade_prob)))
        regime_mult = cfg.bull_notional_mult
    else:
        side = "SHORT"
        confidence = float(short_prob)
        gap = float(short_prob - max(float(long_prob), 0.35 * float(no_trade_prob)))
        regime_mult = cfg.bear_notional_mult
    if confidence < cfg.entry_confidence:
        return TrendSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "confidence_below_threshold")
    if gap < cfg.entry_gap:
        return TrendSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "probability_gap_below_threshold")
    conf_scale = ((confidence - cfg.entry_confidence) / max(1.0 - cfg.entry_confidence, 1e-8)) ** 0.70
    dd_scale = _drawdown_scale(float(equity), float(peak_equity), cfg)
    notional = cfg.min_notional_exposure + (cfg.max_notional_exposure - cfg.min_notional_exposure) * conf_scale
    notional = float(np.clip(notional * regime_mult * dd_scale, 0.0, cfg.max_notional_exposure))
    if notional < cfg.min_notional_exposure:
        return TrendSleeveDecision(False, "NONE", confidence, gap, 0.0, 0.0, 0.0, regime, "notional_below_min")
    leverage = float(max(1.0, cfg.max_leverage))
    position_fraction = float(notional / max(leverage, 1e-8))
    return TrendSleeveDecision(True, side, confidence, gap, notional, leverage, position_fraction, regime, "")
