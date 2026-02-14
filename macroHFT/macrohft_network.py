"""
MacroHFT Network v6.0 - Ultimate Fusion (CNN + RoPE + Transformer + Quantile)
================================================================================
- MultiScaleCNN: 다양한 주기 패턴 포착
- RoPE: 상대적 위치 인코딩 (시계열에 최적화)
- Transformer Backbone: 장기 의존성 학습
- 정보 융합: Concat (안정성)
- Actor: 이산 행동 logits
- Critic: Quantile Regression (분포형 가치)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import config
from common.fusion_transformer import QuantTransformerBackbone  # 이미 구현된 고성능 백본

# ----------------------------------------------------------------------
# Quantile Critic Head (기존과 동일, 안정성 강화)
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
        # NaN/Inf 방지
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            return torch.zeros_like(raw_out)
        first_q = raw_out[..., 0:1]
        deltas = F.softplus(raw_out[..., 1:]) + 1e-6
        deltas = torch.clamp(deltas, max=100.0)  # 발산 방지
        return torch.cat([first_q, first_q + torch.cumsum(deltas, dim=-1)], dim=-1)


# ----------------------------------------------------------------------
# MacroHFT Network Main (Fusion 버전)
# ----------------------------------------------------------------------
class MacroHFTNetwork(nn.Module):
    EXPECTED_INFO_DIM = 11

    def __init__(self, state_dim, action_dim, info_dim=11,
                 d_model=256,          # 표현력 확보 (VRAM 허용 시)
                 n_head=4,
                 n_layers=4,
                 proj_dim=128,
                 dropout=0.1,
                 num_quantiles=32):
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.num_quantiles = num_quantiles

        # ---------- 1. 시계열 백본 (CNN + RoPE + Transformer) ----------
        # mode='ppo'로 설정하여 미래 마스킹 없이 작동
        self.backbone = QuantTransformerBackbone(
            state_dim=state_dim,
            hidden_dim=d_model,
            n_layers=n_layers,
            n_heads=n_head,
            dropout=dropout,
            mode='ppo'          # RoPE 사용, causal mask 없음
        )

        # ---------- 2. 정보(계좌 상태, 전략 점수) 인코더 ----------
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2)
        )

        # ---------- 3. 융합 레이어 (시계열 특징 + 정보 특징) ----------
        fusion_dim = d_model + (d_model // 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # ---------- 4. 헤드 ----------
        self.actor_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, action_dim)
        )
        self.critic_head = QuantileCriticHead(proj_dim, num_quantiles)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.41)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state_seq, state_info, states=None):
        """
        Args:
            state_seq: (B, T, state_dim)  – 차트 데이터
            state_info: (B, info_dim)      – 포지션, 전략 점수 등
        Returns:
            action_logits: (B, action_dim)
            value_mean: (B, 1)
            quantiles: (B, num_quantiles)
            proj_context: (B, proj_dim)   – 필요시 사용 (예: ICM)
        """
        B, T, _ = state_seq.shape
        if state_info.dim() == 3:
            state_info = state_info.squeeze(1)

        # 1. 시계열 백본 -> context 벡터 (CLS 토큰)
        context, _, _ = self.backbone(state_seq)  # (B, d_model)

        # 2. 정보 인코딩
        info_emb = self.info_encoder(state_info)  # (B, d_model//2)

        # 3. 융합 (Concat)
        fused = torch.cat([context, info_emb], dim=-1)
        proj_context = self.fusion_layer(fused)   # (B, proj_dim)

        # 4. 행동 및 가치 산출
        action_logits = self.actor_head(proj_context)
        quantiles = self.critic_head(proj_context)
        value_mean = quantiles.mean(dim=-1, keepdim=True)

        return action_logits, value_mean, quantiles, proj_context


# ----------------------------------------------------------------------
# Expert Classes (전문가별 특성 유지)
# ----------------------------------------------------------------------
class TrendExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        # 추세장: 더 깊은 층으로 장기 패턴 학습
        super().__init__(state_dim, action_dim, info_dim, n_layers=4, **kwargs)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        # 변동성장: 빠른 반응을 위해 얕은 층
        super().__init__(state_dim, action_dim, info_dim, n_layers=2, **kwargs)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=3, **kwargs)