"""
MacroHFT Network v5.0 - DISCRETE LEVERAGE
===========================================
- d_model=256, n_layers=6, n_head=8
- Projection Layer 유지
- Actor 출력: 행동 logits 15개 (방향 3 × 레버리지 5)
- Critic: Quantile Regression (단조 분위수)
- FiLM Conditioning 유지
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import config  # config import 필요

HAS_MAMBA = False

# ----------------------------------------------------------------------
# FiLM Layer (변경 없음)
# ----------------------------------------------------------------------
class FiLMLayer(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        with torch.no_grad():
            self.film_gen[-1].bias[:d_model].fill_(1.0)

    def forward(self, h, cond):
        params = self.film_gen(cond)
        gamma, beta = params.chunk(2, dim=-1)
        if torch.isnan(gamma).any() or torch.isinf(gamma).any() or \
           torch.isnan(beta).any() or torch.isinf(beta).any():
            return h
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * h + beta

# ----------------------------------------------------------------------
# Transformer Block with FiLM
# ----------------------------------------------------------------------
class TransformerFiLMBlock(nn.Module):
    def __init__(self, d_model, n_head, cond_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.film1 = FiLMLayer(d_model, cond_dim)
        self.film2 = FiLMLayer(d_model, cond_dim)

    def forward(self, x, cond, mask=None):
        x_mod = self.film1(x, cond)
        x = x + self.attn(self.norm1(x_mod), self.norm1(x_mod), self.norm1(x_mod), attn_mask=mask)[0]
        x_mod2 = self.film2(x, cond)
        x = x + self.ffn(self.norm2(x_mod2))
        return x

# ----------------------------------------------------------------------
# Quantile Critic Head (단조 분위수)
# ----------------------------------------------------------------------
class QuantileCriticHead(nn.Module):
    def __init__(self, d_model, num_quantiles=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_quantiles)
        )
        
    def forward(self, x):
        raw_out = self.net(x)
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            return raw_out
        first_q = raw_out[..., 0:1]
        deltas = F.softplus(raw_out[..., 1:]) + 1e-6
        deltas = torch.clamp(deltas, max=1e4)
        return torch.cat([first_q, first_q + torch.cumsum(deltas, dim=-1)], dim=-1)

# ----------------------------------------------------------------------
# MacroHFT Network Main (Discrete Action)
# ----------------------------------------------------------------------
class MacroHFTNetwork(nn.Module):
    EXPECTED_INFO_DIM = 11

    def __init__(self, state_dim, action_dim, info_dim=11,
                 d_model=256, n_head=8, n_layers=6, proj_dim=256, dropout=0.1,
                 num_quantiles=32):
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.num_quantiles = num_quantiles

        # Condition Encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(info_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Input Embedding
        self.embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # Backbone (Transformer)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerFiLMBlock(d_model, n_head, d_model, dropout))

        # Projection Layer (복원)
        self.projection = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU()
        )

        # Heads
        self.final_norm = nn.LayerNorm(d_model)
        # 🔥 Actor: 이산 행동 logits (15)
        self.actor_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, action_dim)   # action_dim = 15
        )
        self.critic_head = QuantileCriticHead(proj_dim, num_quantiles)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state_seq, state_info, states=None):
        B, T, _ = state_seq.shape
        if state_info.dim() == 3:
            state_info = state_info.squeeze(1)

        cond = self.condition_encoder(state_info)
        x = self.embedding(state_seq)
        x = x + self.pos_encoder[:, :T, :]

        for layer in self.layers:
            x = layer(x, cond)

        context = self.final_norm(x[:, -1, :])
        proj_context = self.projection(context)          # [B, proj_dim]

        # 🔥 행동 logits (15)
        action_logits = self.actor_head(proj_context)    # [B, 15]

        quantiles = self.critic_head(proj_context)
        value_mean = quantiles.mean(dim=-1, keepdim=True)

        # 🔥 반환값: action_logits, value_mean, quantiles, proj_context
        return action_logits, value_mean, quantiles, proj_context

# ----------------------------------------------------------------------
# Expert Classes (동일 아키텍처, 레이어 수만 차등)
# ----------------------------------------------------------------------
class TrendExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=6, **kwargs)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=4, **kwargs)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=5, **kwargs)