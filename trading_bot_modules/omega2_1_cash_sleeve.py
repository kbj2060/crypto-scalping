from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_ID = "omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055"
ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2
FORBIDDEN_FEATURE_PREFIXES = (
    "clean_regime4_",
    "regime4_pred_",
    "teacher_",
    "exit_head_",
)
FORBIDDEN_FEATURE_NAMES = {"tp_sl_action_score"}


@dataclass(frozen=True)
class Omega21CashSleeveDecision:
    action: int
    confidence: float
    probabilities: tuple[float, float, float]
    threshold: float
    trace: dict[str, Any]


class Omega21CashSleeve:
    """Fail-fast 12-seed HGB ensemble scorer for Omega2.1 cash sleeve rows."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        self.bundle = joblib.load(self.artifact_path)
        model_id = str(self.bundle.get("model_id", ""))
        if model_id != MODEL_ID:
            raise RuntimeError(f"unexpected Omega2.1 model id: {model_id}")
        self.feature_cols = list(self.bundle["feature_cols"])
        self.threshold = float(self.bundle["threshold"])
        self.models = list(self.bundle["models"])
        self._reject_forbidden(self.feature_cols)
        if not self.models:
            raise RuntimeError("Omega2.1 cash sleeve bundle has no models")

    @staticmethod
    def _reject_forbidden(cols: list[str]) -> None:
        bad = [
            col
            for col in cols
            if col in FORBIDDEN_FEATURE_NAMES
            or any(col.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
        ]
        if bad:
            raise RuntimeError(f"Omega2.1 forbidden feature columns: {bad[:40]}")

    def _frame_from_row(self, row: pd.Series | dict[str, Any]) -> pd.DataFrame:
        if isinstance(row, pd.Series):
            raw = pd.DataFrame([row.to_dict()])
        else:
            raw = pd.DataFrame([dict(row)])
        missing = [col for col in self.feature_cols if col not in raw.columns]
        extra_forbidden = [
            col
            for col in raw.columns
            if col in FORBIDDEN_FEATURE_NAMES
            or any(str(col).startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
        ]
        if missing:
            raise RuntimeError(f"Omega2.1 missing feature columns: {missing[:40]}")
        if extra_forbidden:
            raise RuntimeError(f"Omega2.1 forbidden supplied columns: {extra_forbidden[:40]}")
        frame = raw[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            bad = frame.columns[frame.isna().any()].tolist()
            raise RuntimeError(f"Omega2.1 non-finite feature columns: {bad[:40]}")
        return frame

    @staticmethod
    def _classes_to_proba(model: Any, proba: np.ndarray) -> np.ndarray:
        out = np.zeros((len(proba), 3), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        for j, cls in enumerate(classes):
            cls_i = int(cls)
            if 0 <= cls_i <= 2:
                out[:, cls_i] = proba[:, j]
        return out

    def predict_proba_frame(self, features: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_cols if col not in features.columns]
        if missing:
            raise RuntimeError(f"Omega2.1 missing feature columns: {missing[:40]}")
        self._reject_forbidden(list(features.columns))
        x = features[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        if x.isna().any().any():
            bad = x.columns[x.isna().any()].tolist()
            raise RuntimeError(f"Omega2.1 non-finite feature columns: {bad[:40]}")
        arr = x.to_numpy(dtype=np.float64)
        probs = [self._classes_to_proba(model, model.predict_proba(arr)) for model in self.models]
        return np.stack(probs, axis=0).mean(axis=0)

    def decide(self, row: pd.Series | dict[str, Any]) -> Omega21CashSleeveDecision:
        proba = self.predict_proba_frame(self._frame_from_row(row))[0]
        raw_action = int(np.argmax(proba))
        confidence = float(proba[raw_action])
        action = raw_action if confidence >= self.threshold else ACTION_CASH
        return Omega21CashSleeveDecision(
            action=action,
            confidence=confidence,
            probabilities=(float(proba[0]), float(proba[1]), float(proba[2])),
            threshold=self.threshold,
            trace={
                "model_id": MODEL_ID,
                "artifact_path": str(self.artifact_path),
                "raw_action": raw_action,
                "accepted": bool(action != ACTION_CASH),
            },
        )
