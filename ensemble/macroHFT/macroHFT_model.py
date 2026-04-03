"""
Standalone Forecasting MacroHFT v2.5 (Architectural Overhaul)
================================================================================
v2.4 → v2.5 아키텍처 개선:

  [ARCH-1] CLS 토큰 제거 → Temporal Attention Pooling
           BERT식 CLS는 금융 시계열에서 최근 타임스텝 정보를 희석함.
           학습 가능한 쿼리 벡터 + recency_bias로 최근 스텝을 자연스럽게 강조.

  [ARCH-2] Transformer → ResidualCNN 순서 변경
           CNN이 먼저 뭉개면 Transformer가 볼 세부 정보 손실.
           Transformer(전역) 먼저 → ResidualCNN(로컬 보강, skip-connection).

  [ARCH-3] Autoregressive Horizon Decoder (GRU 기반)
           단순 Linear 분할 대신 GRU Cell로 h=1→2→...→6 순차 생성.
           각 호라이즌이 이전 호라이즌의 hidden state를 조건으로 예측.

  [ARCH-4] Causal Mask 누수 수정
           CLS 제거로 자연스럽게 해결. 순수 causal masking만 적용.

  v2.4 유지: GroupNorm, RoPE, NaN fallback, DropPath, 타겟 정규화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MacroHFTConfig:
    input_window: int = 64
    forecast_horizon: int = 6
    target_col: str = 'target_ret_6'

    d_model: int = 128              # [ARCH-Fix] 256 -> 512 (표현력 증대)
    n_head: int = 4                 # [ARCH-Fix] 64 -> 8 (head_dim 64 확보)
    n_layers: int = 1
    proj_dim: int = 64
    dropout: float = 0.3            # [Fix] 0.2 -> 0.1 (언더피팅 완화)
    drop_path_rate: float = 0.1

    decoder_hidden: int = 32
    recency_bias: float = 0.05

    num_outputs: int = 1
    num_features: int = 35

    learning_rate: float = 1e-4    # [Fix] 5e-6 -> 1e-4
    batch_size: int = 128
    max_epochs: int = 500
    patience: int = 100

    direction_loss_weight: float = 2.0 # [Fix] 3.0 -> 1.5
    large_move_weight: float = 1.5
    sharpe_loss_weight: float = 0.0
    sharpe_warmup_epochs: int = 100

    weight_decay: float = 0.01      # [Fix] 5e-3 -> 1e-3
    grad_clip: float = 0.5
    warmup_epochs: int = 30
    lr_scheduler: str = 'cosine'
    restart_period: int = 100          
    restart_mult: float = 2.0          
    min_lr: float = 1e-6            # [Fix] 3e-6 -> 1e-6
    use_ema: bool = True
    ema_decay: float = 0.996
    accumulation_steps: int = 1

    device: str = 'auto'
    model_dir: str = 'data/macroHFT'
    use_amp: bool = False

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.use_amp = False


# ════════════════════════════════════════════════════════════════
# 2. BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rt = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * torch.floor(rt + keep_prob) / keep_prob


class InputProjection(nn.Module):
    """피처 → d_model. 2-layer MLP로 비선형성 확보."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# [ARCH-2] Transformer 이후 CNN: skip-connection으로 원본 보존
class ResidualCNN(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        mid = hidden_dim // 2
        self.conv3 = nn.Conv1d(hidden_dim, mid, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv5 = nn.Conv1d(hidden_dim, mid, kernel_size=5, padding=2, padding_mode='replicate')
        
        num_groups = min(8, hidden_dim)
        while hidden_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.norm  = nn.GroupNorm(num_groups, hidden_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = torch.cat([self.conv3(h), self.conv5(h)], dim=1) # Shape: [B, hidden_dim, T]
        h = self.drop(self.act(self.norm(h))).transpose(1, 2)
        return x + self.proj(h)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin, n_heads):
    B, T, d = x.shape
    hd = d // n_heads
    xh = x.view(B, T, n_heads, hd)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return ((xh * cos) + (rotate_half(xh) * sin)).view(B, T, d)


class TFLayerDP(nn.Module):
    def __init__(self, layer, drop_prob):
        super().__init__()
        self.layer = layer
        self.dp    = DropPath(drop_prob)

    def forward(self, x, src_mask=None):
        out = self.layer(x, src_mask=src_mask)
        return x + self.dp(out - x)


# [ARCH-1] Temporal Attention Pooling
class TemporalAttentionPooling(nn.Module):
    """
    학습 가능한 쿼리 + recency bias로 시퀀스를 1벡터로 압축.
    최근 타임스텝에 지수적으로 더 집중 (recency_bias > 0).
    """
    def __init__(self, hidden_dim: int, recency_bias: float = 0.05):
        super().__init__()
        self.query        = nn.Linear(hidden_dim, 1, bias=False)
        self.recency_bias = recency_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        scores = self.query(x).squeeze(-1)           # [B, T]
        if self.recency_bias > 0:
            pos    = torch.linspace(0, 1, T, device=x.device)
            scores = scores + self.recency_bias * pos
        weights = F.softmax(scores, dim=-1)          # [B, T]
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

class HorizonDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_outputs, forecast_horizon):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_outputs = num_outputs
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),           # 0.2→0.3
            nn.Linear(hidden_dim, num_outputs * forecast_horizon),
            # Tanh 제거 — BCE with logits는 raw logit이 필요
        )
        # 마지막 Linear의 weight를 작게 초기화 → 초기 logit이 0 근처
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, context):
        out = self.head(context)
        return out.view(-1, self.forecast_horizon, self.num_outputs)


# [ARCH-3] Autoregressive Horizon Decoder
class AutoregressiveHorizonDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_quantiles, forecast_horizon):
        super().__init__()
        assert num_quantiles % 2 == 1
        self.forecast_horizon = forecast_horizon
        self.mid_idx          = num_quantiles // 2
        self.context_proj     = nn.Linear(input_dim, hidden_dim)
        self.gru_cell         = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_quantiles),
        )
        # [B3 Fix] 예측 출력을 다시 입력 공간으로 매핑
        self.output_proj = nn.Linear(num_quantiles, input_dim)

    def _monotonic(self, raw):
        if torch.isnan(raw).any() or torch.isinf(raw).any():
            raw = raw.clone()
            raw[~torch.isfinite(raw)] = 0.0
        c  = raw[..., self.mid_idx:self.mid_idx + 1]
        ld = F.softplus(raw[..., :self.mid_idx].flip(-1)) + 1e-4
        ud = F.softplus(raw[..., self.mid_idx + 1:])      + 1e-4
        lo = c - torch.cumsum(ld, dim=-1).flip(-1)
        hi = c + torch.cumsum(ud, dim=-1)
        return torch.cat([lo, c, hi], dim=-1)

    def forward(self, context):
        h = torch.tanh(self.context_proj(context))
        step_input = context
        outs = []
        for _ in range(self.forecast_horizon):
            h = self.gru_cell(step_input, h)
            raw = self.head(h)
            pred = self._monotonic(raw)
            outs.append(pred.unsqueeze(1))
            # [B3 Fix] 다음 호라이즌을 위한 진정한 Auto-regressive 입력
            step_input = self.output_proj(pred)
        return torch.cat(outs, dim=1)


# ════════════════════════════════════════════════════════════════
# 3. BACKBONE
# ════════════════════════════════════════════════════════════════
class MacroHFTBackbone(nn.Module):
    """
    처리 순서: InputProj → RoPE → Transformer → ResidualCNN → TemporalAttnPool
    """
    def __init__(self, state_dim, hidden_dim, n_layers, n_heads,
                 dropout, drop_path_rate, recency_bias):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads

        self.input_proj = InputProjection(state_dim, hidden_dim, dropout * 0.5)
        self.rope       = RotaryEmbedding(self.head_dim)
        self.dropout    = nn.Dropout(dropout)

        dpr = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]

        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,
            )

        self.tf_layers    = nn.ModuleList([TFLayerDP(make_layer(), dpr[i]) for i in range(n_layers)])
        self.residual_cnn = ResidualCNN(hidden_dim, dropout=0.15)
        self.norm         = nn.LayerNorm(hidden_dim)
        self.pooling      = TemporalAttentionPooling(hidden_dim, recency_bias)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        cos, sin = self.rope(T, x.device)
        x = self.dropout(apply_rope(x, cos, sin, self.n_heads))

        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for layer in self.tf_layers:
            x = layer(x, src_mask=mask)

        x = self.residual_cnn(x)
        return self.pooling(self.norm(x))


# ════════════════════════════════════════════════════════════════
# 4. MAIN MODEL
# ════════════════════════════════════════════════════════════════
class ForecastingMacroHFT(nn.Module):
    def __init__(self, config: MacroHFTConfig):
        super().__init__()
        self.config  = config
        self.backbone = MacroHFTBackbone(
            state_dim=config.num_features, hidden_dim=config.d_model,
            n_layers=config.n_layers, n_heads=config.n_head,
            dropout=config.dropout, drop_path_rate=config.drop_path_rate,
            recency_bias=config.recency_bias,
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model, config.proj_dim),
            nn.LayerNorm(config.proj_dim), nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.decoder = HorizonDecoder(
            input_dim=config.proj_dim,
            hidden_dim=config.decoder_hidden,
            num_outputs=config.num_outputs,
            forecast_horizon=config.forecast_horizon,
        )
        
        # [수정] 전체 레이어를 1차적으로 초기화
        self.apply(self._init_weights)
        
        # [수정] 디코더의 마지막 출력 레이어만 다시 0에 가깝게 초기화하여 로짓 폭발 방지
        nn.init.xavier_uniform_(self.decoder.head[-1].weight, gain=0.01)
        if self.decoder.head[-1].bias is not None:
            nn.init.zeros_(self.decoder.head[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.41)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            for n, p in m.named_parameters():
                if 'weight' in n: nn.init.orthogonal_(p)
                elif 'bias' in n: nn.init.zeros_(p)

    def forward(self, state_seq):
        # [수정] state_seq의 로컬 Z-score 정규화 삭제. 
        # (이미 글로벌 정규화가 되어 들어오므로, 여기서 윈도우 텐서 단위로 정규화하면
        # binary 특성들이 e-6로 나뉘어 무한대로 튀는 Blow-up 현상이 발생합니다.)
        
        if self.training:
            B, T, F = state_seq.shape
            noise = torch.randn_like(state_seq) * 0.05
            state_seq = state_seq + noise

            # 타임스텝 랜덤 드롭 15%
            tm = (torch.rand(B, T, 1, device=state_seq.device) > 0.15).float()
            state_seq = state_seq * tm
            
            # 피처 랜덤 드롭 15%
            fm = (torch.rand(B, 1, F, device=state_seq.device) > 0.15).float()
            state_seq = state_seq * fm

        ctx = self.backbone(state_seq)
        ctx = self.fusion(ctx)
        return self.decoder(ctx)
class DirectionalLoss(nn.Module):
    def __init__(self, large_move_weight=2.0, label_smoothing=0.1, noise_threshold=0.0):
        super().__init__()
        self.large_move_weight = large_move_weight
        self.label_smoothing = label_smoothing
        # [혁신 수정] 0.2 시그마 미만의 움직임은 철저히 무시하는 임계값
        self.noise_threshold = noise_threshold 

    def set_sharpe_weight(self, w): pass

    def forward(self, predictions, targets):
        labels = (targets > 0).float()
        # 라벨 스무딩으로 모델의 과도한 확신 방지
        labels_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        logits = predictions[:, :, 0]  # [B, H]

        # 1. Base BCE Loss
        bce = F.binary_cross_entropy_with_logits(logits, labels_smooth, reduction='none')

        # 2. 노이즈 필터링 마스크 (Noise Masking)
        # 타겟의 절대값이 임계치보다 작으면(방향성이 없는 노이즈 구간) 역전파를 아예 차단
        target_abs = torch.abs(targets)
        valid_mask = (target_abs > self.noise_threshold).float()

        # 3. 큰 움직임 가중치 (Large Move Weight)
        is_large = (target_abs > 1.0).float()
        w = 1.0 + (self.large_move_weight - 1.0) * is_large

        # 최종 Loss: 노이즈는 0으로 만들고 큰 변동성 구간에만 집중
        masked_loss = bce * w * valid_mask
        
        # 유효한(valid) 캔들들에 대해서만 평균을 구함 (전체 B로 나누면 gradient가 너무 작아짐)
        valid_count = valid_mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / valid_count

        # 진단용 메트릭 계산 (이때는 전체 데이터 대상 정확도 확인)
        pred_dir = (logits > 0).float()
        actual_dir = (targets > 0).float()
        acc = (pred_dir == actual_dir).float().mean()

        return loss, {
            'bce_loss': loss.item(),
            'sharpe_loss': 0.0,
            'direction_accuracy': acc.item(),
            'pred_pos_ratio': pred_dir.mean().item(),
        }

