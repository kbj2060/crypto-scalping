"""
MacroHFT Network v6.0 - Ultimate Fusion (CNN + RoPE + Transformer + Quantile)
================================================================================
- MultiScaleCNN: 다양한 주기 패턴 포착
- RoPE: 상대적 위치 인코딩 (시계열에 최적화)
- Transformer Backbone: 장기 의존성 학습
- 정보 융합: Concat (안정성)
- Actor: 이산 행동 logits (LayerNorm + Dropout)
- Critic: Quantile Regression (분포형 가치, first_q softplus)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import config
from macroMFT.fusion_transformer import QuantTransformerBackbone  # MambaBackbone 제거

# ----------------------------------------------------------------------
# Quantile Critic Head (first_q를 softplus로 양수 보장)
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
        # 🔥 1단계: 입력 클리핑 (원천 차단)
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        raw_out = self.net(x)
        
        # 🔥 2단계: NaN 감지 시 안전 복구 (0이 아닌 입력 기반 복구)
        if torch.isnan(raw_out).any() or torch.isinf(raw_out).any():
            logger.warning("⚠️ Quantile head NaN detected - using input fallback")
            # 입력 x의 평균으로 복구 (그레디언트 흐름 유지)
            fallback = torch.zeros_like(x[:, :self.num_quantiles]) + x.mean(dim=-1, keepdim=True).clamp(-5, 5)
            raw_out = fallback.detach() + (x * 0).requires_grad_()  # 그레디언트 경로 유지
        
        # 🔥 3단계: 출력 안정화 (소프트 클리핑)
        first_q = F.softplus(raw_out[..., 0:1].clamp(-20, 20))  # softplus overflow 방지
        deltas = F.softplus(raw_out[..., 1:].clamp(-20, 20)) + 1e-6
        deltas = torch.clamp(deltas, min=1e-6, max=50.0)  # 상한 강화
        
        quantiles = torch.cat([first_q, first_q + torch.cumsum(deltas, dim=-1)], dim=-1)
        
        # 추가: 출력 검증 (디버깅용)
        if torch.isnan(quantiles).any():
            logger.error("❌ Quantile output still NaN after recovery!")
        
        return quantiles


# ----------------------------------------------------------------------
# MacroHFT Network Main
# ----------------------------------------------------------------------
class MacroHFTNetwork(nn.Module):
    EXPECTED_INFO_DIM = 11

    def __init__(self, state_dim, action_dim, info_dim=11,
                 d_model=256,          # 표현력 확보
                 n_head=4,
                 n_layers=2,
                 proj_dim=128,
                 dropout=0.2,
                 num_quantiles=32):
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.num_quantiles = num_quantiles

        # ---------- 1. 시계열 백본 (CNN + RoPE + Transformer) ----------
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

        # ---------- 3. 융합 레이어 ----------
        fusion_dim = d_model + (d_model // 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # ---------- 4. 헤드 ----------
        # Actor: logits 출력 (LogSoftmax 없음)
        self.actor_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
            nn.LayerNorm(proj_dim),
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
        B, T, _ = state_seq.shape
        if state_info.dim() == 3:
            state_info = state_info.squeeze(1)

        context, _, _ = self.backbone(state_seq)          # (B, d_model)
        info_emb = self.info_encoder(state_info)          # (B, d_model//2)
        fused = torch.cat([context, info_emb], dim=-1)
        proj_context = self.fusion_layer(fused)           # (B, proj_dim)

        action_logits = self.actor_head(proj_context)     # (B, action_dim)
        quantiles = self.critic_head(proj_context)        # (B, num_quantiles)
        value_mean = quantiles.mean(dim=-1, keepdim=True)

        return action_logits, value_mean, quantiles, proj_context


# ----------------------------------------------------------------------
# Expert Classes (전문가별 특성 유지)
# ----------------------------------------------------------------------
class TrendExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=3, **kwargs)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=2, **kwargs)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim=11, **kwargs):
        super().__init__(state_dim, action_dim, info_dim, n_layers=2, **kwargs)