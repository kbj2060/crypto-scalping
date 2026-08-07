"""Live BTC CryptoMamba h6 future-regime entry-time gate.

Backtest-validated (docs/model_contracts, 2026-07-22 session): entry-time-only filter
(checked once at entry decision, never re-checked intra-trade -- the intra-trade/continuous
variant was already found to cause catastrophic whipsaw for ETH's own CryptoMamba signal).
Skips a BTC entry when the model's +6bar directional prediction disagrees with the entry side.

VAL: PnL +7.45%->+7.98%, MDD unchanged (-11.93%). OOS: PnL +12.59%->+26.79%,
MDD -15.88%->-14.05%. Both splits improve or hold steady -- passes fresh-forward VAL-then-OOS
discipline. Not applied to SOL/ETH: SOL's own backtest of the same filter made VAL and OOS worse
on both PnL and MDD (rejected).

CUDA required (mamba_ssm has no CPU fallback, confirmed earlier this session).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_regime3_cryptomamba_pred_20260531 import (  # noqa: E402
    CryptoMambaRegimePred,
    _add_volume_features,
)

_MIN_ROLLING_BUFFER = 288 + 60  # rolling(288, min_periods=48) window + model's own seq_len

_DEFAULT_MODEL_PATH = (
    ROOT
    / "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
    / "regime3_cryptomamba_pred_btc_h6_nocurrent_20260721_2024.pt"
)


class BtcCmambaEntryGate:
    def __init__(self, *, model_path: str | Path = _DEFAULT_MODEL_PATH, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("BtcCmambaEntryGate requires CUDA (mamba_ssm has no CPU fallback)")
        self.device = torch.device(device)
        payload: dict[str, Any] = torch.load(Path(model_path), map_location=self.device, weights_only=False)
        self.feature_cols: list[str] = list(payload["feature_cols"])
        medians = payload["feature_medians"]
        self.feature_medians = np.asarray([float(medians[c]) for c in self.feature_cols], dtype=np.float64)
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.seq_len = int(payload["seq_len"])
        self.classes: list[str] = list(payload["classes"])
        model = CryptoMambaRegimePred(
            n_features=len(self.feature_cols),
            seq_len=self.seq_len,
            d_model=int(payload["d_model"]),
            n_cblocks=int(payload["cblocks"]),
            n_cmblocks=int(payload["cmblocks"]),
            d_state=int(payload["d_state"]),
            dropout=0.0,
        ).to(self.device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.model = model

    @torch.no_grad()
    def direction_signal(self, frame: pd.DataFrame) -> int:
        """+1 if the model favors long over the next `horizon` bars, -1 if short, 0 if chop.

        `frame` must be the live processed feature frame with at least
        288+seq_len trailing rows available (causal: rolling features only use already-closed bars,
        no lookahead)."""
        if len(frame) < _MIN_ROLLING_BUFFER:
            raise RuntimeError(
                "BtcCmambaEntryGate unavailable: insufficient history "
                f"({len(frame)} < {_MIN_ROLLING_BUFFER})"
            )
        enriched = _add_volume_features(frame)
        tail = enriched[self.feature_cols].tail(self.seq_len)
        raw = tail.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        med = self.feature_medians[np.newaxis, :]
        filled = np.where(np.isnan(raw), med, raw)
        x = (filled - self.scaler_mean[np.newaxis, :]) / np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)[np.newaxis, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits = self.model(xb)
        pred_id = int(torch.argmax(logits, dim=-1).item())
        label = self.classes[pred_id]
        if label == "bull":
            return 1
        if label == "bear":
            return -1
        return 0
