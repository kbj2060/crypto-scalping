"""
Standalone Forecasting MacroHFT v2.1 (Pure Temporal & Multi-Horizon)
================================================================================
- 목적: TFT 및 파운데이션 모델(Chronos, Lag-Llama 등)과의 앙상블을 위한 순수 예측 모델
- 수정: Static(정적) 변수 개념을 완전히 제거하고 모든 피처를 시계열(Temporal)로 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
from typing import List, Dict
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════
@dataclass
class MacroHFTConfig:
    input_window: int = 64
    forecast_horizon: int = 6        
    target_col: str = 'target_ret_6'
    d_model: int = 32
    n_head: int = 4
    n_layers: int = 2
    proj_dim: int = 16
    dropout: float = 0.3
    num_quantiles: int = 5
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # 🌟 [Fix] Static 피처 관련 변수 완전 삭제, 오직 Temporal만 남김
    num_temporal_features: int = 0 
    
    learning_rate: float = 3e-5
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 20
    direction_loss_weight: float = 8.0
    device: str = 'auto'
    model_dir: str = 'data/macrohft'
    use_amp: bool = True

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.use_amp = False

# (2. Core Internal Modules 코드는 이전과 동일하게 유지 - MultiScaleCNN, RotaryEmbedding, QuantTransformerBackbone, QuantileCriticHead)
class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        small_dim, large_dim = hidden_dim // 4, hidden_dim // 2
        self.conv1 = nn.Conv1d(input_dim, small_dim, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv1d(input_dim, small_dim, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv1d(input_dim, large_dim, kernel_size=7, padding=3, padding_mode='replicate')
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.activation(self.bn(out)).transpose(1, 2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x, cos, sin):
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return (x * cos) + (torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1) * sin)

class QuantTransformerBackbone(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.ms_cnn = MultiScaleCNN(state_dim, hidden_dim)
        self.rope = RotaryEmbedding(hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout, 'gelu', True, True),
            num_layers=n_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.ms_cnn(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        cos, sin = self.rope(x)
        x = self.dropout(apply_rotary_pos_emb(x, cos, sin))
        
        mask = torch.zeros(T+1, T+1, device=x.device)
        mask[1:, 1:] = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        return self.layer_norm(self.transformer(x, mask=mask))[:, 0, :]

class QuantileCriticHead(nn.Module):
    def __init__(self, d_model, num_quantiles=32):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, num_quantiles))

    def forward(self, x):
        x = torch.clamp(x, min=-10.0, max=10.0)
        raw_out = self.net(x)
        
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            fallback = torch.zeros_like(x[..., :self.num_quantiles]) + x.mean(dim=-1, keepdim=True).clamp(-5, 5)
            raw_out = fallback.detach() + (x * 0).requires_grad_()
        
        mid_idx = self.num_quantiles // 2
        center = raw_out[..., mid_idx:mid_idx+1]
        
        lower_deltas = F.softplus(raw_out[..., :mid_idx].flip(-1).clamp(-20, 20)) + 1e-6
        upper_deltas = F.softplus(raw_out[..., mid_idx+1:].clamp(-20, 20)) + 1e-6
        
        lower = center - torch.cumsum(lower_deltas, dim=-1).flip(-1)
        upper = center + torch.cumsum(upper_deltas, dim=-1)
        
        quantiles = torch.cat([lower, center, upper], dim=-1)
        return quantiles

# ════════════════════════════════════════════════════════════════
# 3. Main Standalone Ensemble Model
# ════════════════════════════════════════════════════════════════
class ForecastingMacroHFT(nn.Module):
    def __init__(self, config: MacroHFTConfig):
        super().__init__()
        self.config = config
        
        self.backbone = QuantTransformerBackbone(config.num_temporal_features, config.d_model, config.n_layers, config.n_head, config.dropout)
        
        # 🌟 [Fix] info_encoder 삭제 및 fusion_layer 입력 차원 축소
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.d_model, config.proj_dim), 
            nn.LayerNorm(config.proj_dim), 
            nn.SiLU(), 
            nn.Dropout(config.dropout)
        )
        
        self.horizon_fc = nn.Linear(config.proj_dim, config.forecast_horizon * config.proj_dim)
        self.forecast_head = QuantileCriticHead(config.proj_dim, config.num_quantiles)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.41)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # 🌟 [Fix] 입력 파라미터에서 state_info 완전히 제거
    def forward(self, state_seq):
        context = self.backbone(state_seq)                
        proj_context = self.fusion_layer(context) # context만 단독 사용
        
        B = proj_context.shape[0]
        horizon_h = self.horizon_fc(proj_context).view(B, self.config.forecast_horizon, self.config.proj_dim)
        quantiles = self.forecast_head(horizon_h)      
        return quantiles