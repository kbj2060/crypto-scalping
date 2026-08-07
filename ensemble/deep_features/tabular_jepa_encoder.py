"""Small JEPA-style self-supervised encoder for tabular financial time series.

Stage B of the BTC deep-feature plan (see
C:\\Users\\kbj20\\.claude\\plans\\unified-napping-lighthouse.md). Motivation for a
latent-prediction (JEPA) objective over raw-value reconstruction (VIME/MAE-style):
financial tabular features are noisy, and reconstructing that noise in input space
wastes encoder capacity; predicting the (EMA) target encoder's own embedding of the
masked span is more robust to input noise (cf. I-JEPA; CF-JEPA, arXiv:2606.07031).
A temporal-contrastive auxiliary term (TS2Vec-style) is added to encourage
regime-persistence-aware structure, matching this repo's existing Regime3 design.

Deliberately kept small (short window, shallow transformer, low embedding dim) --
the repo's prior large sequence-transformer attempts (PatchTST, cross-asset panel
transformer) passed VAL and failed OOS, a capacity/overfitting failure mode this
guards against by pretraining unsupervised on the full history and keeping the
model tiny.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class JEPAConfig:
    n_features: int
    window: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_mult: int = 2
    dropout: float = 0.1
    embed_dim: int = 24
    mask_min_frac: float = 0.25
    mask_max_frac: float = 0.5
    ema_decay: float = 0.996
    contrastive_temp: float = 0.2
    contrastive_weight: float = 0.5


class WindowDataset(Dataset):
    """Causal sliding windows over a standardized (N, F) feature matrix.

    Sample i covers rows [i - window + 1, i], i.e. never looks forward. timestamps
    is only carried through for bookkeeping (embeddings are emitted keyed by the
    window's last timestamp).
    """

    def __init__(self, features: np.ndarray, timestamps: np.ndarray, window: int):
        assert features.ndim == 2
        self.features = features.astype(np.float32)
        self.timestamps = timestamps
        self.window = window
        self.valid_idx = np.arange(window - 1, len(features))

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        end = self.valid_idx[idx]
        start = end - self.window + 1
        return torch.from_numpy(self.features[start:end + 1]), end


class _SeqEncoder(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.window, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.pos_emb[:, : x.shape[1]]
        h = self.encoder(h)
        return self.norm(h)


class TabularJEPAEncoder(nn.Module):
    """Owns context encoder (trained), EMA target encoder (no grad), mask token,
    and a small predictor. `forward_pretrain` returns the two self-supervised
    losses; `encode` is the frozen-inference path used to emit deep features.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = _SeqEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.n_features) * 0.02)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.readout = nn.Linear(cfg.d_model, cfg.embed_dim)

    @torch.no_grad()
    def _update_target_ema(self):
        d = self.cfg.ema_decay
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(d).add_(cp.data, alpha=1 - d)

    def _random_mask_span(self, batch: int, window: int, device) -> torch.Tensor:
        frac = torch.empty(batch, device=device).uniform_(self.cfg.mask_min_frac, self.cfg.mask_max_frac)
        span_len = (frac * window).long().clamp(min=2, max=window - 2)
        latest_start = window - span_len
        start = (torch.rand(batch, device=device) * latest_start.float()).long()
        idx = torch.arange(window, device=device).unsqueeze(0).expand(batch, -1)
        mask = (idx >= start.unsqueeze(1)) & (idx < (start + span_len).unsqueeze(1))
        return mask

    def forward_pretrain(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, F). Returns (loss_jepa, loss_contrastive)."""
        b, l, _ = x.shape
        device = x.device

        mask_a = self._random_mask_span(b, l, device)
        mask_b = self._random_mask_span(b, l, device)

        x_ctx_a = torch.where(mask_a.unsqueeze(-1), self.mask_token.expand(b, l, -1), x)
        x_ctx_b = torch.where(mask_b.unsqueeze(-1), self.mask_token.expand(b, l, -1), x)

        ctx_a = self.context_encoder(x_ctx_a)
        ctx_b = self.context_encoder(x_ctx_b)
        with torch.no_grad():
            tgt = self.target_encoder(x)

        pred_a = self.predictor(ctx_a)
        pred_b = self.predictor(ctx_b)
        loss_jepa = 0.5 * (
            F.smooth_l1_loss(pred_a[mask_a], tgt[mask_a].detach())
            + F.smooth_l1_loss(pred_b[mask_b], tgt[mask_b].detach())
        )

        anchor = self.readout(ctx_a[:, -1])
        positive = self.readout(ctx_b[:, -1].detach())
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.T / self.cfg.contrastive_temp
        labels = torch.arange(b, device=device)
        loss_contrastive = F.cross_entropy(logits, labels)

        return loss_jepa, loss_contrastive

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F), no masking. Returns (B, embed_dim) deep feature for the
        window's last timestep -- causal, frozen, used for downstream eval."""
        z = self.context_encoder(x)
        return self.readout(z[:, -1])
