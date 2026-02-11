"""
MacroHFT Network v3 — Paper-Faithful Implementation
=====================================================
논문: MacroHFT (KDD 2024) + FiLM (Perez 2017) + Multi-Scale Attention (MTS 2025)

핵심 업그레이드:
1. FiLM Conditional Adapter — 매 Transformer 레이어에서 condition(포지션+전략+시장)이
   hidden state를 scale/shift로 변조. 논문 Eq(1)~(3)의 핵심 혁신.
2. Condition Encoder — 포지션을 이산 Embedding으로, 전략 점수를 별도 MLP로 인코딩.
3. Multi-Scale Temporal Attention — 단기(local window) + 장기(full range) 적응적 융합.
4. Dueling Actor Head — V(s) + A(s,a) - mean(A) 구조로 포지션 의존적 행동 분리.

호환성: forward() 시그니처 유지 → ppo_agent.py 수정 불필요
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# 1. Building Blocks
# ==============================================================================

class SwiGLU(nn.Module):
    """SwiGLU Activation (Shazeer 2020)"""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al. 2017)
    h_out = gamma(c) * h + beta(c)
    Zero-Init: 초기에 identity (gamma=1, beta=0)
    """
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
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * h + beta


# ==============================================================================
# 2. Condition Encoder (논문 Eq.1)
# ==============================================================================

class ConditionEncoder(nn.Module):
    """c = psi_3(P_t) + psi_2(s2_lt)"""
    def __init__(self, info_dim, cond_dim):
        super().__init__()
        self.position_embed = nn.Embedding(3, cond_dim)
        context_input_dim = info_dim - 1
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, info):
        pos_val = info[:, 0]
        context = info[:, 1:]
        pos_idx = (pos_val + 1).long().clamp(0, 2)
        pos_emb = self.position_embed(pos_idx)
        ctx_emb = self.context_encoder(context)
        return pos_emb + ctx_emb


# ==============================================================================
# 3. FiLM-Conditioned Transformer Block
# ==============================================================================

class FiLMTransformerBlock(nn.Module):
    """Pre-Norm Transformer + FiLM at Attention and FFN"""
    def __init__(self, d_model, n_head, cond_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(SwiGLU(d_model), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)
        self.film_attn = FiLMLayer(d_model, cond_dim)
        self.film_ffn = FiLMLayer(d_model, cond_dim)

    def forward(self, x, cond, mask=None):
        x_mod = self.film_attn(x, cond)
        x_norm = self.norm1(x_mod)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
        x_mod = self.film_ffn(x, cond)
        x_norm = self.norm2(x_mod)
        x = x + self.ffn(x_norm)
        return x


# ==============================================================================
# 4. Multi-Scale Temporal Attention
# ==============================================================================

class MultiScaleTemporalAttention(nn.Module):
    """Short-range + Long-range with adaptive gating"""
    def __init__(self, d_model, n_head, cond_dim, window_size=10, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.short_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm_short = nn.LayerNorm(d_model)
        self.long_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm_long = nn.LayerNorm(d_model)
        self.scale_gate = nn.Sequential(nn.Linear(cond_dim, d_model), nn.Sigmoid())

    def _create_local_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0.0
        return mask

    def forward(self, x, cond):
        B, T, D = x.shape
        local_mask = self._create_local_mask(T, x.device)
        x_short = self.norm_short(x)
        h_short, _ = self.short_attn(x_short, x_short, x_short, attn_mask=local_mask)
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        x_long = self.norm_long(x)
        h_long, _ = self.long_attn(x_long, x_long, x_long, attn_mask=causal_mask)
        gate = self.scale_gate(cond).unsqueeze(1)
        fused = gate * h_short + (1 - gate) * h_long
        return x + fused


# ==============================================================================
# 5. FiLM + Multi-Scale Combined Block
# ==============================================================================

class FiLMMultiScaleBlock(nn.Module):
    """MultiScaleAttention + FiLM-FFN"""
    def __init__(self, d_model, n_head, cond_dim, window_size=10, dropout=0.1):
        super().__init__()
        self.ms_attn = MultiScaleTemporalAttention(d_model, n_head, cond_dim, window_size, dropout)
        self.film_ffn = FiLMLayer(d_model, cond_dim)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(SwiGLU(d_model), nn.Dropout(dropout))

    def forward(self, x, cond, mask=None):
        x = self.ms_attn(x, cond)
        x_mod = self.film_ffn(x, cond)
        x_norm = self.norm_ffn(x_mod)
        x = x + self.ffn(x_norm)
        return x


# ==============================================================================
# 6. Dueling Actor Head
# ==============================================================================

class DuelingActorHead(nn.Module):
    """base_preference + (advantage - mean_advantage)"""
    def __init__(self, d_model, action_dim):
        super().__init__()
        self.base_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )

    def forward(self, latent):
        base = self.base_stream(latent)
        advantage = self.advantage_stream(latent)
        return base + (advantage - advantage.mean(dim=-1, keepdim=True))


# ==============================================================================
# 7. MacroHFTNetwork v3 (메인)
# ==============================================================================

class MacroHFTNetwork(nn.Module):
    """
    MacroHFT v3: FiLM + Multi-Scale + Dueling
    forward() 6-tuple 반환 유지 → ppo_agent.py 호환
    """
    EXPECTED_INFO_DIM = 11

    def __init__(self, state_dim, action_dim, info_dim=11,
                 d_model=128, n_head=4, n_layers=2, dropout=0.1, ms_window=10):
        super().__init__()
        self.d_model = d_model
        cond_dim = d_model

        # A. Condition Encoder
        self.condition_encoder = ConditionEncoder(info_dim, cond_dim)

        # B. Input Processing
        self.embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)

        # C. FiLM-Conditioned Transformer Layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(FiLMTransformerBlock(d_model, n_head, cond_dim, dropout))
            else:
                self.layers.append(FiLMMultiScaleBlock(d_model, n_head, cond_dim, ms_window, dropout))

        # D. Output Heads
        self.final_norm = nn.LayerNorm(d_model)
        self.actor_head = DuelingActorHead(d_model, action_dim)
        self.critic_mean = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )
        self.last_gate_mean = 0.5

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state_seq, state_info, states=None, temperature=None):
        B, T, _ = state_seq.shape
        if state_info.dim() == 3:
            state_info = state_info.squeeze(1)

        cond = self.condition_encoder(state_info)
        x = self.embedding(state_seq)
        x = x + self.pos_encoder[:, :T, :]

        for layer in self.layers:
            x = layer(x, cond)

        context = self.final_norm(x[:, -1, :])
        logits = self.actor_head(context)
        value = self.critic_mean(context)
        self.last_gate_mean = 0.5

        return logits, value, None, None, None, self.last_gate_mean


# ==============================================================================
# 8. Expert Specialization
# ==============================================================================

class TrendExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, dropout=0.2):
        super().__init__(state_dim, action_dim, info_dim,
                        d_model=256, n_head=8, n_layers=4, dropout=dropout, ms_window=20)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, dropout=0.1):
        super().__init__(state_dim, action_dim, info_dim,
                        d_model=128, n_head=4, n_layers=2, dropout=dropout, ms_window=5)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, dropout=0.05):
        super().__init__(state_dim, action_dim, info_dim,
                        d_model=192, n_head=6, n_layers=3, dropout=dropout, ms_window=10)

# Backward Compatibility
XLSTMNetwork = MacroHFTNetwork