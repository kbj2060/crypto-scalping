"""Odyssey ETH live core -- self-contained 3-Head TabM inference primitives.

Vendored (not imported) from the training scripts that originally defined them, so that
Odyssey's live/shadow decision path never has to import a training script to get a handful
of inference-time symbols:
  - `ThreeHeadTabM`/`ThreeHeadConfig`/`POS_COLS`/`_standardize_apply` <- scripts/
    train_eval_omega1_2_tabm_3head_20260603.py
  - `EXPERT_NAMES`/`ROUTE_COLS`/`_route_id` <- scripts/
    train_omega1_regime3_expert_direction_head_volpca_20260602.py
  - `_atr_pct` <- scripts/eval_omega4_1_atr_safety_sltp_20260622.py
  - ETH `BASE_TEMPLATE`/`EXPERT_SCALES`/`FEE_RATE`/`SLIP_RATE` <- scripts/
    train_eval_omega1_2_tabm_diffusion_risk_20260603.py

Importing any one symbol from those training scripts drags in their full module (each
importing several sibling training scripts in turn, plus catboost/extra sklearn estimators
never touched at inference time) -- see docs/experiments/eth_odyssey_live_cleanroom_
dependency_rewrite_20260816.md for the traced dependency graph that motivated this file.

Two behavior fixes vs. the original live consumer (trading_bot_modules/omega4_6_1_live.py),
both intentional -- see that module's `_Component.entry_decision`/`_build_model` for the
originals being fixed:
  1. Model reconstruction always uses `ThreeHeadConfig(**payload["config"])` (the artifact's
     own recorded config), never a module-global config singleton. The original entry-decision
     path silently ignored `payload["config"]` and always rebuilt from a global default that
     only "worked" because every bundle so far happened to share it.
  2. Model construction (`build_model`) is a separate step from prediction (`predict_proba`) so
     callers can cache the built model once per component instead of rebuilding it (a fresh
     nn.Module + load_state_dict) on every single decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# =====================================================================================================
# scripts/train_eval_omega1_2_tabm_3head_20260603.py -- POS_COLS / ThreeHeadConfig / ThreeHeadTabM /
# _standardize_apply, copied verbatim (zero external references in the original).
# =====================================================================================================

POS_COLS = [
    "pos_side",
    "pos_hold_bars",
    "pos_unrealized",
    "pos_mfe",
    "pos_mae",
    "pos_giveback",
    "pos_dist_to_tp",
    "pos_dist_to_sl",
    "pos_notional",
    "pos_leverage",
    "pos_exposure",
    "pos_tp",
    "pos_sl",
]


@dataclass(frozen=True)
class ThreeHeadConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8
    exit_loss_weight: float = 1.15
    quality_loss_weight: float = 0.80


class ThreeHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: ThreeHeadConfig) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_head = nn.Linear(int(cfg.hidden), 3)
        self.quality_head = nn.Linear(int(cfg.hidden), 3)
        self.exit_head = nn.Linear(int(cfg.hidden), 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "direction": self.direction_head(h),
            "quality": self.quality_head(h),
            "exit": self.exit_head(h),
        }


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("3-head TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 3-head inference matrix")
    return out.astype(np.float32)


def build_model(payload: dict[str, Any], *, device: torch.device) -> ThreeHeadTabM:
    """Reconstruct a ThreeHeadTabM from a saved bundle payload -- always from the payload's own
    recorded `config`, never a module-global default (fix #1, see module docstring)."""
    cfg = ThreeHeadConfig(**dict(payload["config"]))
    model = ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: ThreeHeadTabM, x: pd.DataFrame, scaler: dict[str, Any], *, device: torch.device) -> dict[str, np.ndarray]:
    """Ensemble-averaged softmax probabilities for an already-built (and ideally cached) model
    -- callers should build the model once per component via `build_model` and reuse it across
    calls (fix #2, see module docstring), rather than rebuilding it on every decision."""
    x_np = _standardize_apply(x, scaler)
    chunks: dict[str, list[np.ndarray]] = {"direction": [], "quality": [], "exit": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


# =====================================================================================================
# scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py -- EXPERT_NAMES / ROUTE_COLS /
# _route_id, copied verbatim (zero external references in the original).
# =====================================================================================================

EXPERT_NAMES = ["bull", "bear", "chop"]

ROUTE_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]


def _route_id(frame: pd.DataFrame) -> np.ndarray:
    values = frame[ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return np.argmax(values, axis=1).astype(np.int64)


# =====================================================================================================
# scripts/eval_omega4_1_atr_safety_sltp_20260622.py -- _atr_pct, copied verbatim. This is the only
# symbol the live path ever used from that 292-line file (confirmed via grep of the live consumer).
# =====================================================================================================


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    out = atr / np.maximum(close, 1.0e-12)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite ATR percent")
    return out


# =====================================================================================================
# scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py (ETH) -- BASE_TEMPLATE / EXPERT_SCALES /
# FEE_RATE / SLIP_RATE, copied verbatim. SOL/BTC's sibling copies of these two dicts are intentionally
# NOT vendored here: the live Odyssey4 shadow never used them (see dependency-graph doc), and this
# module is ETH-only by design.
# =====================================================================================================

BASE_TEMPLATE = {
    "notional": 0.45,
    "leverage": 2.0,
    "take_profit": 0.026,
    "stop_loss": 0.014,
    "max_hold": 72,
    "cooldown": 6,
}
EXPERT_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90}
FEE_RATE = 0.0005
SLIP_RATE = 0.0002


def resolve_expert_scale_key(expert: str) -> str:
    """EXPERT_SCALES uses `"chop_expert"` while EXPERT_NAMES/routing use `"chop"` -- the original
    live code re-derived this remap ad hoc at each call site (fix #3, see module docstring); this
    is that remap pinned down as one function so it can't be silently forgotten at a new call site."""
    return "chop_expert" if expert == "chop" else expert


def load_fee_slip() -> tuple[float, float]:
    return float(FEE_RATE), float(SLIP_RATE)
