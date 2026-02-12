"""
MacroHFT Network v3.5 SOTA
==========================
1. Backbone: Mamba-FiLM Hybrid (속도/장기기억 O(N))
2. Critic: Distributional Quantile Regression (리스크 관리)
3. Adapter: FiLM (Feature-wise Linear Modulation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common import config

# Try importing Mamba (없으면 Transformer로 Fallback)
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️ Mamba not found. Falling back to Transformer.")

# ==============================================================================
# 1. Components (Mamba & FiLM)
# ==============================================================================

class FiLMLayer(nn.Module):
    """Condition(Market/Pos) -> Scale & Shift Features"""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )
        # Zero-Init
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        with torch.no_grad():
            self.film_gen[-1].bias[:d_model].fill_(1.0)

    def forward(self, h, cond):
        params = self.film_gen(cond)
        gamma, beta = params.chunk(2, dim=-1)
        # [FIX] NaN/Inf 체크 - 문제 발생 시 skip
        if torch.isnan(gamma).any() or torch.isinf(gamma).any() or \
           torch.isnan(beta).any() or torch.isinf(beta).any():
            return h
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * h + beta

class MambaFiLMBlock(nn.Module):
    """
    SOTA: Mamba Block with FiLM Modulation
    x_t = Mamba(Norm(FiLM(x_{t-1}, cond))) + x_{t-1}
    """
    def __init__(self, d_model, cond_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba-ssm not installed")
            
        self.film = FiLMLayer(d_model, cond_dim)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x, cond):
        x_mod = self.film(x, cond)
        out = self.mamba(self.norm(x_mod))
        return x + out

# (Fallback용 Transformer Block - 기존 코드 재사용)
class TransformerFiLMBlock(nn.Module):
    def __init__(self, d_model, n_head, cond_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model), nn.Dropout(dropout)
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

# ==============================================================================
# 2. Distributional Critic Head (D-PPO) - 단조 분위수 보장
# ==============================================================================

class QuantileCriticHead(nn.Module):
    """
    Predicts N Quantiles (τ_1, ..., τ_N) of the return distribution
    Output: (Batch, N_Quantiles) with monotonicity guarantee.
    """
    def __init__(self, d_model, num_quantiles=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_quantiles)
        )
        
    def forward(self, x):
        raw_out = self.net(x)
        # [FIX] NaN 발생 시 원본 반환 (fallback)
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            return raw_out
        first_q = raw_out[..., 0:1]
        deltas = F.softplus(raw_out[..., 1:]) + 1e-6
        # [FIX] 누적 합산 시 inf 방지 clamp
        deltas = torch.clamp(deltas, max=1e4)
        quantiles = torch.cat([first_q, first_q + torch.cumsum(deltas, dim=-1)], dim=-1)
        return quantiles

# ==============================================================================
# [제안 2] 보상 분포 예측 헤드 (Distributional Reward) - 단조 보장
# ==============================================================================
class RewardDistributionHead(nn.Module):
    """
    입력 상태(context)에 대한 즉각 보상 r의 분위수 예측
    출력: (Batch, N_Quantiles) with monotonicity.
    """
    def __init__(self, d_model, num_quantiles=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_quantiles)
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        raw_out = self.net(x)
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            return raw_out
        first_q = raw_out[..., 0:1]
        deltas = F.softplus(raw_out[..., 1:]) + 1e-6
        deltas = torch.clamp(deltas, max=1e4)
        return torch.cat([first_q, first_q + torch.cumsum(deltas, dim=-1)], dim=-1)

# ==============================================================================
# 3. MacroHFT Network v3.5 Main
# ==============================================================================

class MacroHFTNetwork(nn.Module):
    EXPECTED_INFO_DIM = 11

    def __init__(self, state_dim, action_dim, info_dim=11,
                 d_model=128, n_head=4, n_layers=4, dropout=0.1,
                 proj_dim=128):  # [추가] 통합 context 차원
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        cond_dim = d_model
        
        # Config Load
        self.use_mamba = getattr(config, 'USE_MAMBA', True) and HAS_MAMBA
        self.num_quantiles = getattr(config, 'NUM_QUANTILES', 32)
        
        self.reward_head = RewardDistributionHead(d_model, self.num_quantiles)

        # 1. Condition Encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(info_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # 2. Input Embedding
        self.embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # 3. Backbone Layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if self.use_mamba:
                self.layers.append(MambaFiLMBlock(
                    d_model, cond_dim,
                    d_state=getattr(config, 'MAMBA_D_STATE', 16),
                    d_conv=getattr(config, 'MAMBA_D_CONV', 4)
                ))
            else:
                self.layers.append(TransformerFiLMBlock(d_model, n_head, cond_dim, dropout))

        # 4. Heads
        self.final_norm = nn.LayerNorm(d_model)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Linear(d_model, action_dim)
        )
        self.critic_head = QuantileCriticHead(d_model, self.num_quantiles)
        
        # [핵심] Projection Layer - 모든 Expert의 context를 동일 차원으로 변환
        self.projection = nn.Linear(d_model, proj_dim)
        nn.init.orthogonal_(self.projection.weight)  # 직교 초기화 (선택)
        nn.init.zeros_(self.projection.bias)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
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

        # [변경] 원본 context (d_model 차원)
        raw_context = self.final_norm(x[:, -1, :])
        
        # [변경] 통합 차원으로 projection
        unified_context = self.projection(raw_context)  # (B, proj_dim)
        
        logits = self.actor_head(raw_context)           # Actor는 원본 사용 (성능 유지)
        quantiles = self.critic_head(raw_context)       # Critic도 원본 사용
        value_mean = quantiles.mean(dim=-1, keepdim=True)
        reward_quantiles = self.reward_head(raw_context)
        
        # [변경] 통합된 context를 반환 (다른 expert와 차원 일치)
        return logits, value_mean, None, None, None, quantiles, reward_quantiles, unified_context

# Expert Classes (상속)
class TrendExpert(MacroHFTNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d_model=256, n_layers=4)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d_model=128, n_layers=2)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d_model=192, n_layers=3)