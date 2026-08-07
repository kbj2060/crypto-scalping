from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class DuelingDQNConfig:
    input_dim: int
    hidden_dim: int = 256
    action_dim: int = 3
    dropout: float = 0.05
    temperature: float = 0.18


class DuelingQNetwork(nn.Module):
    def __init__(self, cfg: DuelingDQNConfig) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(int(cfg.input_dim), int(cfg.hidden_dim)),
            nn.LayerNorm(int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim) // 2),
            nn.LayerNorm(int(cfg.hidden_dim) // 2),
            nn.SiLU(),
        )
        mid = int(cfg.hidden_dim) // 2
        self.value = nn.Sequential(nn.Linear(mid, mid), nn.SiLU(), nn.Linear(mid, 1))
        self.advantage = nn.Sequential(nn.Linear(mid, mid), nn.SiLU(), nn.Linear(mid, int(cfg.action_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        value = self.value(h)
        adv = self.advantage(h)
        return value + adv - adv.mean(dim=1, keepdim=True)


class DQNActionModel:
    """Sklearn-like action model wrapper for fully_learned_governor_policy."""

    classes_ = np.asarray([0, 1, 2], dtype=np.int64)

    def __init__(
        self,
        *,
        state_dict: dict[str, torch.Tensor],
        config: dict[str, Any],
        medians: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        feature_cols: list[str],
    ) -> None:
        self.config = dict(config)
        self.feature_cols = list(feature_cols)
        self.medians = np.asarray(medians, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.state_dict = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in state_dict.items()}

    def _matrix(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            arr = x.reindex(columns=self.feature_cols).to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.asarray(x, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, self.medians)
        return (arr - self.mean) / np.maximum(self.std, 1e-6)

    def _model(self) -> DuelingQNetwork:
        cached = getattr(self, "_cached_model", None)
        if cached is not None:
            return cached
        cfg = DuelingDQNConfig(
            input_dim=int(self.config["input_dim"]),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            action_dim=int(self.config.get("action_dim", 3)),
            dropout=float(self.config.get("dropout", 0.05)),
            temperature=float(self.config.get("temperature", 0.18)),
        )
        model = DuelingQNetwork(cfg)
        model.load_state_dict(self.state_dict)
        model.eval()
        self._cached_model = model
        return model

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_cached_model", None)
        return state

    def q_values(self, x: pd.DataFrame | np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
        arr = self._matrix(x)
        model = self._model()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arr), int(batch_size)):
                xb = torch.from_numpy(arr[start : start + int(batch_size)])
                outs.append(model(xb).cpu().numpy())
        return np.vstack(outs).astype(np.float64)

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        q = self.q_values(x)
        temp = max(float(self.config.get("temperature", 0.18)), 1e-4)
        z = q / temp
        z = z - np.max(z, axis=1, keepdims=True)
        p = np.exp(z)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        trade_min_prob = float(self.config.get("trade_min_prob", 0.0))
        trade_margin = float(self.config.get("trade_margin", -1e9))
        if trade_min_prob > 0.0 or trade_margin > -1e8:
            best = np.argmax(p, axis=1)
            best_trade_prob = np.maximum(p[:, 1], p[:, 2])
            cash_prob = p[:, 0]
            block = (best != 0) & ((best_trade_prob < trade_min_prob) | ((best_trade_prob - cash_prob) < trade_margin))
            if np.any(block):
                p = p.copy()
                p[block, 0] = 1.0
                p[block, 1:] = 0.0
        return p

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def metadata(self) -> dict[str, Any]:
        return {"model": "DuelingDQNActionModel", "config": dict(self.config), "feature_cols": list(self.feature_cols)}


def make_action_model(
    model: DuelingQNetwork,
    *,
    config: DuelingDQNConfig,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: list[str],
) -> DQNActionModel:
    return DQNActionModel(
        state_dict=model.state_dict(),
        config=asdict(config),
        medians=medians,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
    )
