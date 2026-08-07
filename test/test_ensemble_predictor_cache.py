from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot_modules.ensemble_predictor import EnsemblePredictor


@dataclass
class _Prediction:
    median: np.ndarray
    confidence: np.ndarray


class _Model:
    def __init__(self):
        self.calls = []

    def predict(self, frame, horizon):
        self.calls.append(horizon)
        values = np.arange(horizon, dtype=np.float32)[None, :]
        return _Prediction(values, np.ones_like(values))


def _predictor_without_models():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor._feature_cache = {}
    predictor._active_feature_frame_key = None
    predictor._prediction_cache_key = None
    predictor._prediction_cache = {}
    return predictor


def test_prediction_cache_reuses_larger_horizon_for_same_frame():
    predictor = _predictor_without_models()
    model = _Model()
    frame = pd.DataFrame([{"timestamp": "2026-01-01", "close": 100.0}])

    six, first_hit, _ = predictor._predict_cached("model", model, frame, horizon=6)
    three, second_hit, _ = predictor._predict_cached("model", model, frame, horizon=3)

    assert first_hit is False
    assert second_hit is True
    assert model.calls == [6]
    assert six.median.shape == (1, 6)
    assert three.median.shape == (1, 3)


def test_prediction_cache_is_invalidated_when_latest_bar_changes():
    predictor = _predictor_without_models()
    model = _Model()
    first = pd.DataFrame([{"timestamp": "2026-01-01", "close": 100.0}])
    second = pd.DataFrame([{"timestamp": "2026-01-01 00:05", "close": 101.0}])

    predictor._predict_cached("model", model, first, horizon=3)
    _, cache_hit, _ = predictor._predict_cached("model", model, second, horizon=3)

    assert cache_hit is False
    assert model.calls == [3, 3]
