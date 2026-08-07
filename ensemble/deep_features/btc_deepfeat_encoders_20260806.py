"""Three deep-feature encoder architectures for the new BTC standalone model line, all trained
supervised against the zigzag risk-adjusted soft label (see btc_deepfeat_dataset_20260806.py) as
the teacher target -- unlike the earlier self-supervised JEPA encoder
(ensemble/deep_features/tabular_jepa_encoder.py, closed 2026-08-04), these learn embeddings
directly from the soft-label objective, not from a masked-reconstruction pretext task.

- cnn_seq:      Conv1d over the time axis, raw 113-dim feature channels (a standard TCN-style
                stack). Tests whether temporal locality alone (no category structure) is useful.
- cnn_category: per-category linear projection at each timestep, then Conv1d over the category
                axis (cross-category mixing) followed by Conv1d over the time axis. Tests whether
                explicit category grouping helps versus the flat feature-channel approach above.
- transformer:  standard supervised TransformerEncoder (no self-supervised pretraining/masking),
                pooled at the window's last (current) timestep.

All three expose the same interface: forward(x) -> (logits[B,3], embedding[B,embed_dim]), so a
single training script can drive any of them via --arch.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from btc_deepfeat_tabm_head_20260806 import TabMEnsembleHead  # noqa: E402

import torch
import torch.nn as nn


class SequenceCNNEncoder(nn.Module):
    def __init__(self, n_features: int, embed_dim: int = 32, channels: tuple[int, ...] = (64, 64, 32), kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Linear(in_ch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B,F,T)
        h = self.conv(h)
        h = self.pool(h).squeeze(-1)  # (B,C)
        return self.embed(h)


class _CategoryProjection(nn.Module):
    """Splits the flat (category-ordered) feature vector into per-category blocks and projects
    each block to a shared cat_dim with its own linear layer (block-diagonal, not shared weights
    -- categories have unrelated units/scales)."""

    def __init__(self, category_sizes: list[int], cat_dim: int):
        super().__init__()
        self.sizes = list(category_sizes)
        self.projs = nn.ModuleList([nn.Linear(s, cat_dim) for s in self.sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,T,n_cat,cat_dim)
        splits = torch.split(x, self.sizes, dim=-1)
        proj = [p(s) for p, s in zip(self.projs, splits)]
        return torch.stack(proj, dim=2)


class CategoryCNNEncoder(nn.Module):
    def __init__(
        self,
        category_sizes: list[int],
        cat_dim: int = 16,
        cat_hidden: int = 24,
        time_channels: tuple[int, ...] = (32, 32),
        embed_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = _CategoryProjection(category_sizes, cat_dim)
        self.cat_conv = nn.Sequential(
            nn.Conv1d(cat_dim, cat_hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cat_hidden, cat_hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.cat_pool = nn.AdaptiveAvgPool1d(1)
        time_layers: list[nn.Module] = []
        in_ch = cat_hidden
        for out_ch in time_channels:
            time_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.time_conv = nn.Sequential(*time_layers)
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Linear(in_ch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        cat = self.proj(x)  # (B,T,n_cat,cat_dim)
        n_cat, cat_dim = cat.shape[2], cat.shape[3]
        cat = cat.permute(0, 1, 3, 2).reshape(b * t, cat_dim, n_cat)
        h = self.cat_conv(cat)  # (B*T, cat_hidden, n_cat)
        h = self.cat_pool(h).squeeze(-1)  # (B*T, cat_hidden)
        h = h.reshape(b, t, -1).transpose(1, 2)  # (B, cat_hidden, T)
        h = self.time_conv(h)
        h = self.time_pool(h).squeeze(-1)  # (B,C)
        return self.embed(h)


class SupervisedTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.25,
        embed_dim: int = 32,
        max_len: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * ffn_mult, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.embed = nn.Linear(d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :t]
        h = self.input_dropout(h)
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1]  # representation of the window's current (last) bar
        return self.embed(pooled)


class DeepFeatModel(nn.Module):
    """Wraps an encoder with a prediction head trained END-TO-END with the encoder (not a
    frozen-embedding downstream model). `head_type="linear"` is a plain nn.Linear direction head
    + optional nn.Linear quality head (the original baseline: val 67.1%/OOS 64.8% direction acc).
    `head_type="tabm"` replaces both with a single TabMEnsembleHead (see
    btc_deepfeat_tabm_head_20260806.py -- BatchEnsemble-style per-expert input scale/bias + shared
    MLP trunk, matching this repo's established ThreeHeadTabM convention), still trained jointly
    with the encoder via backprop, not as a separate frozen-embedding stage.
    forward() returns (logits, quality_pred, embedding)."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int = 3, head_dropout: float = 0.2, quality_head: bool = False, head_type: str = "linear"):
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type
        if head_type == "linear":
            self.head_dropout = nn.Dropout(head_dropout)
            self.head = nn.Linear(embed_dim, n_classes)
            self.quality_head = nn.Linear(embed_dim, 1) if quality_head else None  # attribute name kept for checkpoint backward-compat
        elif head_type == "tabm":
            self.tabm_head = TabMEnsembleHead(embed_dim, n_experts=8, hidden=64, n_layers=2, dropout=head_dropout, n_classes=n_classes, quality_head=quality_head)
        else:
            raise ValueError(f"unknown head_type: {head_type!r}, expected 'linear' or 'tabm'")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        if self.head_type == "linear":
            h = self.head_dropout(emb)
            logits = self.head(h)
            quality = self.quality_head(h).squeeze(-1) if self.quality_head is not None else torch.zeros(x.shape[0], device=x.device)
        else:
            logits, quality = self.tabm_head(emb)
        return logits, quality, emb


ARCHES = ("cnn_seq", "cnn_category", "transformer")


def build_model(
    arch: str,
    n_features: int,
    category_sizes: list[int],
    embed_dim: int = 32,
    *,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    ffn_mult: int = 2,
    dropout: float = 0.25,
    quality_head: bool = False,
    head_type: str = "linear",
) -> DeepFeatModel:
    if arch == "cnn_seq":
        encoder = SequenceCNNEncoder(n_features, embed_dim=embed_dim)
    elif arch == "cnn_category":
        encoder = CategoryCNNEncoder(category_sizes, embed_dim=embed_dim)
    elif arch == "transformer":
        encoder = SupervisedTransformerEncoder(
            n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ffn_mult=ffn_mult, dropout=dropout, embed_dim=embed_dim,
        )
    else:
        raise ValueError(f"unknown arch: {arch!r}, expected one of {ARCHES}")
    return DeepFeatModel(encoder, embed_dim=embed_dim, n_classes=3, quality_head=quality_head, head_type=head_type)
