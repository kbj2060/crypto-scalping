from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
EXPERT_NAMES = ("bull", "bear", "chop")


def _safe_col(x: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in x.columns:
        return np.full(len(x), float(default), dtype=np.float64)
    arr = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)
    return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))


def regime_router_weights(x: pd.DataFrame, *, temperature: float = 1.0, floor: float = 0.03) -> np.ndarray:
    bull = np.clip(_safe_col(x, f"{CLEAN_PREFIX}bull_prob"), 0.0, 1.0)
    bear = np.clip(_safe_col(x, f"{CLEAN_PREFIX}bear_prob"), 0.0, 1.0)
    chop = np.clip(_safe_col(x, f"{CLEAN_PREFIX}chop_prob"), 0.0, 1.0)
    whipsaw = np.clip(_safe_col(x, f"{CLEAN_PREFIX}whipsaw_prob"), 0.0, 1.0)
    normal = np.clip(_safe_col(x, f"{CLEAN_PREFIX}normal_prob"), 0.0, 1.0)
    trend_bias = np.clip(_safe_col(x, f"{CLEAN_PREFIX}trend_bias"), -1.0, 1.0)
    risk_off = np.clip(_safe_col(x, f"{CLEAN_PREFIX}risk_off_prob"), 0.0, 1.0)

    raw = np.column_stack(
        [
            bull + 0.25 * np.maximum(trend_bias, 0.0),
            bear + 0.20 * risk_off + 0.25 * np.maximum(-trend_bias, 0.0),
            chop + 0.50 * whipsaw + 0.20 * normal,
        ]
    ).astype(np.float64)
    raw = np.clip(raw, float(floor), None)
    temp = max(float(temperature), 1e-6)
    if abs(temp - 1.0) > 1e-9:
        raw = np.power(raw, temp)
    return raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)


def expand_proba(model: Any, x: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    local = np.asarray(model.predict_proba(x), dtype=np.float64)
    local_classes = np.asarray(getattr(model, "classes_", classes), dtype=int)
    out = np.zeros((len(x), len(classes)), dtype=np.float64)
    for j, cls in enumerate(local_classes):
        where = np.flatnonzero(classes == int(cls))
        if where.size:
            out[:, int(where[0])] = local[:, j]
    return out


@dataclass
class RegimeMoEActionModel:
    experts: dict[str, Any]
    classes_: np.ndarray
    temperature: float = 1.0
    floor: float = 0.03

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        weights = regime_router_weights(x, temperature=self.temperature, floor=self.floor)
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        for j, name in enumerate(EXPERT_NAMES):
            out += weights[:, [j]] * expand_proba(self.experts[name], x, self.classes_)
        out = np.clip(out, 1e-12, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


@dataclass
class RegimeMoEQualityModel:
    experts: dict[str, Any]
    temperature: float = 1.0
    floor: float = 0.03

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        weights = regime_router_weights(x, temperature=self.temperature, floor=self.floor)
        pred = np.column_stack([np.asarray(self.experts[name].predict(x), dtype=np.float64) for name in EXPERT_NAMES])
        return np.sum(weights * pred, axis=1)
