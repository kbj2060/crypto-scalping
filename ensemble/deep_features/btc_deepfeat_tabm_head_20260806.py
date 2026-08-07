"""TabM-style ensemble downstream head, used on top of the FROZEN transformer deep-feature
embedding (32-dim, from btc_deepfeat_encoders_20260806.SupervisedTransformerEncoder) in place of
the plain nn.Linear direction/quality heads baked into DeepFeatModel.

Matches this repo's established TabM convention (see ThreeHeadTabM in
scripts/train_eval_omega1_2_tabm_3head_20260603.py): an efficient ensemble-of-virtual-experts via
per-expert learned input scale/bias (BatchEnsemble-style), a SHARED MLP trunk applied per-expert,
and per-expert output heads averaged into the final prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TabMEnsembleHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_experts: int = 8,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 3,
        quality_head: bool = True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.expert_scale = nn.Parameter(torch.ones(n_experts, in_dim))
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, in_dim))

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        self.direction_head = nn.Linear(hidden, n_classes)
        self.quality_head = nn.Linear(hidden, 1) if quality_head else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d = x.shape
        x_expert = x.unsqueeze(1) * self.expert_scale.unsqueeze(0) + self.expert_bias.unsqueeze(0)  # (B, E, D)
        x_flat = x_expert.reshape(b * self.n_experts, d)
        h = self.trunk(x_flat).reshape(b, self.n_experts, -1)  # (B, E, hidden)

        logits_per_expert = self.direction_head(h)  # (B, E, n_classes)
        logits = logits_per_expert.mean(dim=1)

        if self.quality_head is not None:
            quality_per_expert = self.quality_head(h).squeeze(-1)  # (B, E)
            quality = quality_per_expert.mean(dim=1)
        else:
            quality = torch.zeros(b, device=x.device)
        return logits, quality
