"""
Signal Module: Enhanced Temporal Fusion Transformer for 5-min ETH Day Trading

[IDEA 1] Regime-Conditioned Dual-Path TFT
[IDEA 2] Payoff-Weighted Loss
[IDEA 4] Attention Mask for Non-Stationary Regimes
[IDEA 7] Multi-Horizon Ensemble (별도 클래스 추가)
"""

import os
import copy
import json
import math
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from torch.optim.swa_utils import AveragedModel, SWALR   # [SWA] 추가

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. CONFIG (수정)
# ════════════════════════════════════════════════════════════════
@dataclass
class TFTConfig:
    """TFT 하이퍼파라미터 — 논문 최적값 기반 + 5분봉 튜닝."""

    # ── 입력/출력 ──
    input_window: int = 48           # 48 × 5min = 4시간 룩백
    forecast_horizon: int = 6        # 6 × 5min = 30분 예측
    num_features: int = 57           # ULTIMATE_FEATURE_COLS(30) + meta(5)
    target_col: str = 'target_ret_12'   # [IDEA 7] 여러 타겟 중 선택

    # ── 모델 구조 ──
    hidden_size: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float =  0.2

    # ── 새로 추가: Label Smoothing ──
    label_smoothing: float = 0.05      # 예측 과신 방지

    # ── Variable Selection Network ──
    num_static_features: int = 3     # session_asia/europe/us (현재 세션)
    num_temporal_features: int = 54  # 나머지 시계열 피처

    # ── Quantile 예측 (트레이딩 최적화) ──
    quantiles: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])

    # ── 학습 기본 ──
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = 1.0 

    # [IDEA 2] Payoff-Weighted Loss 파라미터 (기존 direction_loss_weight 등 제거)
    loss_delta: float = 0.05
    loss_power: float = 3
    loss_wrong_penalty: float = 5.0
    loss_direction_boost: float = 5.0      # ← 추가 (3.0부터 시작, 5.0까지 올려봐)

    # ── LR 스케줄러 ──
    warmup_epochs: int = 20
    lr_scheduler: str = 'onecycle'
    min_lr: float = 1e-7

    # ── EMA ──
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ── Mixed Precision ──
    use_amp: bool = True

    # ── Gradient Accumulation ──
    accumulation_steps: int = 1

    # ── 재현성 ──
    seed: int = 42

    # ── 로깅 ──
    log_dir: str = 'logs/tensorboard'
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10

    # ── SWA (Stochastic Weight Averaging) ──
    use_swa: bool = True
    swa_start_epoch: int = 300   # 150 에포크부터 SWA 시작 (max_epochs=200 기준)
    swa_lr: float = 5e-5

    # ── 기타 ──
    device: str = 'auto'
    model_dir: str = 'data/tft'

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.use_amp = False


# ════════════════════════════════════════════════════════════════
# 1.5. EMA MODEL (동일)
# ════════════════════════════════════════════════════════════════
class EMAModel:
    """
    Exponential Moving Average of model parameters.
    학습 중 파라미터의 이동 평균을 유지하여 일반화 성능 향상.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """EMA 파라미터를 모델에 적용 (추론/검증 시)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """원래 파라미터 복원 (학습 재개 시)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ════════════════════════════════════════════════════════════════
# 2. DATASET (수정: regime_break 포함)
# ════════════════════════════════════════════════════════════════
class TFTDataset(Dataset):
    """슬라이딩 윈도우 데이터셋. regime_break를 temporal에 포함."""

    def __init__(self, config: TFTConfig, df: pd.DataFrame,  feature_cols: List[str]):
        self.config = config
        self.target_col = config.target_col
        self.static_cols = ['session_asia', 'session_europe', 'session_us']
        self.temporal_cols = [c for c in feature_cols if c not in self.static_cols]

        self.temporal_data = df[self.temporal_cols].values.astype(np.float32)
        self.static_data = df[self.static_cols].values.astype(np.float32)
        self.target_data = df[self.target_col].values.astype(np.float32)

        self.total_len = config.input_window + config.forecast_horizon
        self.n_samples = len(df) - self.total_len + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        t_start = idx
        t_end = idx + self.config.input_window
        f_end = t_end + self.config.forecast_horizon

        temporal = torch.tensor(self.temporal_data[t_start:t_end])
        static = torch.tensor(self.static_data[t_end - 1])
        target = torch.tensor(self.target_data[t_end:f_end])

        return {'temporal': temporal, 'static': static, 'target': target}


# ════════════════════════════════════════════════════════════════
# 3. MODEL COMPONENTS (InterpretableMultiHeadAttention 수정: 마스킹 개선)
# ════════════════════════════════════════════════════════════════
class GatedLinearUnit(nn.Module):
    """GLU: σ(Wx + b) ⊙ (Vx + c)"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """GRN — 스킵 커넥션 + GLU + LayerNorm."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj else x
        hidden = self.fc1(x)
        if self.context_fc and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)
        return self.layer_norm(residual + hidden)


class VariableSelectionNetwork(nn.Module):
    """VSN — 변수 중요도 자동 학습."""
    def __init__(self, num_vars: int, hidden_size: int, dropout: float = 0.1,
                 context_size: int = None):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])
        self.selection_grn = GatedResidualNetwork(
            num_vars * hidden_size, hidden_size, num_vars, dropout,
            context_size=context_size
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        has_time = x.dim() == 3
        var_outputs = []
        for i in range(self.num_vars):
            vi = x[:, :, i:i+1] if has_time else x[:, i:i+1]
            var_outputs.append(self.var_grns[i](vi))
        var_outputs = torch.stack(var_outputs, dim=-2)

        if has_time:
            B, T, V, H = var_outputs.shape
            flat = var_outputs.reshape(B, T, V * H)
        else:
            B, V, H = var_outputs.shape
            flat = var_outputs.reshape(B, V * H)

        if context is not None and has_time:
            ctx = context.unsqueeze(1).expand(-1, T, -1)
            weights = self.softmax(self.selection_grn(flat, ctx))
        else:
            weights = self.softmax(self.selection_grn(flat, context))

        weights = weights.unsqueeze(-1)
        selected = (var_outputs * weights).sum(dim=-2)
        return selected, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """TFT 해석 가능 MHA — Values를 헤드 간 공유. [IDEA 4] 마스킹 개선."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, self.d_k)
        self.W_o = nn.Linear(self.d_k, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, token_mask=None):
        """
        mask: causal mask (lower triangular)
        token_mask: [B, T] where 1 means token is allowed to attend
        """
        B, T, _ = query.shape
        Q = self.W_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, T, T]

        # causal mask
        if mask is not None:
            fill_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(mask == 0, fill_value)

        # [IDEA 4] token mask: for each query, mask out keys not in same regime
        if token_mask is not None:
            # token_mask: [B, T] -> [B, 1, 1, T] for broadcasting over heads and queries
            token_mask_exp = token_mask[:, None, None, :]  # [B, 1, 1, T]
            fill_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(token_mask_exp == 0, fill_value)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        V_exp = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, T, d_k]
        context = torch.matmul(attn_weights, V_exp).mean(dim=1)    # [B, T, d_k]
        output = self.W_o(context)
        avg_attn = attn_weights.mean(dim=1)
        return output, avg_attn


# ════════════════════════════════════════════════════════════════
# 4. TFT CORE MODEL (Regime-Conditioned Dual-Path)
# ════════════════════════════════════════════════════════════════
class TemporalFusionTransformer(nn.Module):
    """
    Enhanced TFT with:
        - Dual GRU paths (momentum / reversion) [IDEA 1]
        - Regime gating via static features
        - Attention masking with regime_break [IDEA 4]
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        self.temporal_vsn = VariableSelectionNetwork(
            config.num_temporal_features, H, config.dropout, context_size=H)
        self.static_encoder = nn.Sequential(
            nn.Linear(config.num_static_features, H), nn.ReLU(), nn.Linear(H, H))
        self.static_context_enrichment = GatedResidualNetwork(H, H, H, config.dropout)
        self.static_context_state_h = GatedResidualNetwork(H, H, H, config.dropout)

        # [IDEA 1] Dual GRU encoders
        self.gru_momentum = nn.GRU(
            input_size=H, hidden_size=H, num_layers=config.lstm_layers,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.gru_reversion = nn.GRU(
            input_size=H, hidden_size=H, num_layers=config.lstm_layers,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)

        # [IDEA 1 수정] Regime classifier 제거 (허스트로 대체)
        # self.regime_classifier = nn.Sequential(...)  # ← 제거
        
        # ★ 새로운 방식: 허스트 기반 게이팅
        # feature_cols를 나중에 설정 (TFTSignalModel에서)
        self.feature_cols = None  # 나중에 설정됨
        self.hurst_idx = None
        self.regime_trending_idx = None

        self.post_lstm_gate = GatedLinearUnit(H, H)
        self.post_lstm_norm = nn.LayerNorm(H)

        self.static_enrichment = GatedResidualNetwork(H, H, H, config.dropout, context_size=H)
        self.multihead_attn = InterpretableMultiHeadAttention(H, config.attention_heads, config.dropout)
        self.post_attn_gate = GatedLinearUnit(H, H)
        self.post_attn_norm = nn.LayerNorm(H)

        self.pos_ff = GatedResidualNetwork(H, H, H, config.dropout)
        self.pos_ff_gate = GatedLinearUnit(H, H)
        self.pos_ff_norm = nn.LayerNorm(H)

        self.horizon_fc = nn.Linear(H, config.forecast_horizon * H)
        self.quantile_heads = nn.ModuleList([nn.Linear(H, 1) for _ in config.quantiles])

    def set_feature_indices(self, feature_cols: List[str]):
        """
        피처 컬럼 순서를 설정하고 허스트 인덱스 찾기
        
        Args:
            feature_cols: temporal 피처 리스트 (static 제외)
        """
        self.feature_cols = feature_cols
        
        # hurst_48 인덱스 찾기 (우선순위 1)
        if 'hurst_48' in feature_cols:
            self.hurst_idx = feature_cols.index('hurst_48')
            logger.info(f"✅ Hurst-based gating enabled: hurst_48 at index {self.hurst_idx}")
        
        # regime_trending 인덱스 찾기 (우선순위 2)
        elif 'regime_trending' in feature_cols:
            self.regime_trending_idx = feature_cols.index('regime_trending')
            logger.info(f"✅ Regime-based gating enabled: regime_trending at index {self.regime_trending_idx}")
        
        else:
            logger.warning("⚠️ No hurst_48 or regime_trending found - using uniform gating")

    def forward(self, temporal: torch.Tensor, static: torch.Tensor):
        """
        temporal: [B, T, F]  (T = input_window)
        static:   [B, S]
        
        수정사항:
        - regime_classifier 제거
        - 허스트 지수 또는 regime_trending을 temporal에서 직접 추출
        - 연속/이진 게이팅 모두 지원
        """
        B = temporal.shape[0]
        T = temporal.shape[1]
        H = self.config.hidden_size

        static_emb = self.static_encoder(static)
        cs_e = self.static_context_enrichment(static_emb)
        cs_h = self.static_context_state_h(static_emb)

        # Variable selection
        selected, var_weights = self.temporal_vsn(temporal, cs_e)

        # Initial hidden state for GRUs (from static context)
        h0 = cs_h.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()

        # [IDEA 1] Dual GRU forward
        out_m, _ = self.gru_momentum(selected, h0)   # [B, T, H]
        out_r, _ = self.gru_reversion(selected, h0)

        # ════════════════════════════════════════════════════════════════
        # [IDEA 1 수정] Hurst-based Regime Gating
        # ════════════════════════════════════════════════════════════════
        
        # 방법 1: hurst_48 사용 (연속 값, 더 정확)
        if self.hurst_idx is not None:
            # 마지막 타임스텝의 허스트 지수
            hurst = temporal[:, -1, self.hurst_idx]  # [B]
            
            # Soft gating: sigmoid로 부드럽게 전환
            # H=0.5 기준, 5배 스케일링 (sharp transition)
            # H=0.6 → momentum_gate=0.88
            # H=0.5 → momentum_gate=0.50
            # H=0.4 → momentum_gate=0.12
            momentum_gate = torch.sigmoid(5.0 * (hurst - 0.5))  # [B]
            reversion_gate = 1.0 - momentum_gate
            
            # [B, 1, 1] 형태로 unsqueeze하여 [B, T, H]와 브로드캐스팅
            gate0 = momentum_gate.unsqueeze(-1).unsqueeze(-1)   # [B, 1, 1]
            gate1 = reversion_gate.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # 방법 2: regime_trending 사용 (이진 값, 빠름)
        elif self.regime_trending_idx is not None:
            # 마지막 타임스텝의 regime_trending (0 or 1)
            regime_trending = temporal[:, -1, self.regime_trending_idx]  # [B]
            
            # Hard gating: 0 또는 1
            gate0 = regime_trending.unsqueeze(-1).unsqueeze(-1)   # [B, 1, 1]
            gate1 = (1.0 - regime_trending).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # 방법 3: 허스트 없으면 균등 게이팅 (fallback)
        else:
            # 50:50 블렌딩
            gate0 = torch.ones(B, 1, 1, device=temporal.device) * 0.5
            gate1 = torch.ones(B, 1, 1, device=temporal.device) * 0.5
        
        # Blend GRU outputs
        gru_out = gate0 * out_m + gate1 * out_r  # [B, T, H]
        
        # ════════════════════════════════════════════════════════════════

        gated = self.post_lstm_gate(gru_out)
        temporal_feat = self.post_lstm_norm(gated + selected)

        # Static enrichment
        cs_exp = cs_e.unsqueeze(1).expand(-1, temporal_feat.shape[1], -1)
        enriched = self.static_enrichment(temporal_feat, cs_exp)

        # [IDEA 4] Prepare token mask from regime_break feature
        # (Optional - 현재는 스킵, 나중에 추가 가능)
        token_mask = None

        # Attention with causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=temporal.device)).unsqueeze(0)  # [1, T, T]
        attn_out, attn_w = self.multihead_attn(enriched, enriched, enriched,
                                                mask=causal_mask, token_mask=token_mask)
        attn_out = self.post_attn_norm(self.post_attn_gate(attn_out) + enriched)

        ff_out = self.pos_ff(attn_out)
        ff_out = self.pos_ff_norm(self.pos_ff_gate(ff_out) + attn_out)

        last_h = ff_out[:, -1, :]
        horizon_h = self.horizon_fc(last_h).view(B, self.config.forecast_horizon, H)

        q_preds = torch.cat([qh(horizon_h) for qh in self.quantile_heads], dim=-1)
        return q_preds, attn_w, var_weights


# ════════════════════════════════════════════════════════════════
# 5. CUSTOM LOSS (Payoff-Weighted Loss) [IDEA 2]
# ════════════════════════════════════════════════════════════════
class PayoffWeightedLoss(nn.Module):
    """
    트레이딩 최적화 손실 함수:
        - Huber loss base
        - 중요도 가중치: |target|^power (큰 움직임 강조)
        - 방향 틀렸을 때 페널티 배수
    """
    def __init__(self, quantiles: List[float], delta=0.01, power=1.5, wrong_penalty=3.0, label_smoothing=0.0, direction_boost=5.0):
        super().__init__()
        self.quantiles = quantiles
        self.delta = delta
        self.power = power
        self.wrong_penalty = wrong_penalty
        self.label_smoothing = label_smoothing
        self.direction_boost = direction_boost  # ← 추가
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles)//2

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        predictions: [B, horizon, num_quantiles]
        targets:     [B, horizon]
        """
        pred_median = predictions[:, :, self.median_idx]  # [B, H]

        # Label smoothing 적용 (타겟을 살짝 노이즈 추가)
        if self.label_smoothing > 0 and self.training:
            noise = torch.randn_like(targets) * targets.std() * self.label_smoothing
            target_smooth = targets + noise

            # 방향 보존
            original_sign = torch.sign(targets)
            flipped_mask = (original_sign != torch.sign(target_smooth))
            target_smooth = torch.where(
                flipped_mask,
                original_sign * torch.abs(target_smooth),
                target_smooth
            )
        else:
            target_smooth = targets

        # Huber loss
        base_loss = F.huber_loss(pred_median, target_smooth, reduction='none', delta=self.delta)

        # Importance weight: large moves matter more
        importance = torch.abs(targets) ** self.power

        # Direction penalty
        wrong_dir = (torch.sign(pred_median) != torch.sign(target_smooth)).float()

        # ★★ 새 공식: 방향 틀렸을 때 exponential penalty
        # wrong_dir=1일 때: exp(direction_boost) = e^5 = 148배 페널티
        # wrong_dir=0일 때: exp(0) = 1배 (정상)
        direction_factor = torch.exp(self.direction_boost * wrong_dir)

        loss = (base_loss * importance * direction_factor).mean()

        with torch.no_grad():
            dir_acc = (torch.sign(pred_median) == torch.sign(target_smooth)).float().mean().item()

        return loss, {
            'payoff_loss': loss.item(),
            'direction_accuracy': dir_acc,
        }


# ════════════════════════════════════════════════════════════════
# 6. HIGH-LEVEL WRAPPER (TFTSignalModel 수정)
# ════════════════════════════════════════════════════════════════
class TFTSignalModel:
    """TFT 모델의 학습/예측/저장/로드 래퍼."""

    def __init__(self, config: TFTConfig = None):
        self.config = config
        self.model = TemporalFusionTransformer(config)
        self.model.to(config.device)
        self.ema = None
        self.feature_cols = None
        self.target_col = config.target_col
        self.scaler_params = {}
        self.target_scaler = None
        
    # ── SEED ──
    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── LR SCHEDULER ──
    def _create_scheduler(self, optimizer, steps_per_epoch: int):
        cfg = self.config
        total_steps = steps_per_epoch * cfg.max_epochs
        warmup_steps = steps_per_epoch * cfg.warmup_epochs

        if cfg.lr_scheduler == 'cosine':
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(cfg.min_lr / cfg.learning_rate,
                          0.5 * (1.0 + math.cos(math.pi * progress)))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        elif cfg.lr_scheduler == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.learning_rate, total_steps=total_steps,
                pct_start=cfg.warmup_epochs / cfg.max_epochs, anneal_strategy='cos',
                final_div_factor=cfg.learning_rate / max(cfg.min_lr, 1e-8))

        else:  # 'plateau'
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=7, min_lr=cfg.min_lr)
    
    # ── CHECKPOINT ──
    def _save_checkpoint(self, tag: str):
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        
        # 모델 state_dict만 저장 (가볍게)
        torch.save(self.model.state_dict(), path)
        
        # 메타데이터 별도 저장
        meta_path = os.path.join(self.config.model_dir, f'tft_{tag}_meta.json')
        meta = {
            'feature_cols': self.feature_cols,
            'scaler_params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in self.scaler_params.items()},
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"체크포인트 저장: {path}")

    def _save_full_checkpoint(self, tag: str, epoch, optimizer, scheduler, scaler,
                              best_val_loss, patience_counter, global_step):
        """학습 재개를 위한 전체 상태 저장."""
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'global_step': global_step,
            'feature_cols': self.feature_cols,
            'scaler_params': self.scaler_params,
            'config': self.config.__dict__
        }
        torch.save(state, path)
        logger.info(f"Full 체크포인트 저장: {path}")

    def _load_checkpoint(self, tag: str):
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트 없음: {path}")
            
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        logger.info(f"모델 로드 완료: {path}")
        
        # 메타데이터 로드
        meta_path = os.path.join(self.config.model_dir, f'tft_{tag}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_cols = meta['feature_cols']
            self.scaler_params = {k: np.array(v) if isinstance(v, list) else v 
                                 for k, v in meta['scaler_params'].items()}

    def _load_full_checkpoint(self, path: str, optimizer, scheduler, scaler):
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트 없음: {path}")
            
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.ema and checkpoint.get('ema_state_dict'):
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            
        self.feature_cols = checkpoint['feature_cols']
        self.scaler_params = checkpoint['scaler_params']
        
        return checkpoint

    def _normalize(self, train_df, val_df, cols):
        """Z-score 정규화 (train 기준). target_col도 정규화."""
        train_df = train_df.copy()
        if val_df is not None:
            val_df = val_df.copy()

        # temporal 피처 정규화
        mean = train_df[cols].mean()
        std = train_df[cols].std().replace(0, 1.0)   # ← Series.replace 는 정상 동작
        train_df[cols] = (train_df[cols] - mean) / std
        if val_df is not None:
            val_df[cols] = (val_df[cols] - mean) / std
        self.scaler_params = {'mean': mean.values, 'std': std.values}

        # 타겟 정규화 (target_col은 cols에 포함되지 않을 수 있음)
        target = self.target_col
        t_mean = train_df[target].mean()
        t_std = train_df[target].std()
        if t_std == 0:
            t_std = 1.0
        train_df[target] = (train_df[target] - t_mean) / t_std
        if val_df is not None:
            val_df[target] = (val_df[target] - t_mean) / t_std
        self.target_scaler = (t_mean, t_std)

        return train_df, val_df

    def fit(self, cfg: TFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], resume_from=None):
        """
        학습 수행.
        
        Args:
            train_df, val_df: 학습/검증 데이터
            feature_cols: 피처 컬럼 리스트 (static 포함)
            target_col: 타겟 컬럼명
        """
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in feature_cols if c not in static_cols]

        cfg.num_static_features = len(static_cols)      # 3
        cfg.num_temporal_features = len(temporal_cols)  # 27 (실제 개수!)
        cfg.num_features = len(feature_cols)

        # ★ 허스트 인덱스 설정 (중요!)

        self.config = cfg
        self.feature_cols = feature_cols
        self.target_col = cfg.target_col

        self.model = TemporalFusionTransformer(cfg)
        self.model.to(cfg.device)
        self.model.set_feature_indices(temporal_cols)  # ← 추가!
        
        # 정규화
        train_norm, val_norm = self._normalize(train_df, val_df, temporal_cols)

        # Dataset/Loader
        train_dataset = TFTDataset(cfg, train_norm, feature_cols)
        val_dataset = TFTDataset(cfg, val_norm, feature_cols)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        steps_per_epoch = max(len(train_loader) // cfg.accumulation_steps, 1)
        scheduler = self._create_scheduler(optimizer, steps_per_epoch)
        is_step_scheduler = cfg.lr_scheduler in ('cosine', 'onecycle')

        # [IDEA 2] 새로운 손실 함수 사용
        criterion = PayoffWeightedLoss(
            cfg.quantiles,
            delta=cfg.loss_delta,
            power=cfg.loss_power,
            wrong_penalty=cfg.loss_wrong_penalty,
            label_smoothing=cfg.label_smoothing,
            direction_boost=cfg.loss_direction_boost
        )

        if cfg.use_ema:
            self.ema = EMAModel(self.model, decay=cfg.ema_decay)

        # [SWA] 초기화
        swa_model = None
        swa_scheduler = None
        if cfg.use_swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
            logger.info(f"SWA 활성화: 시작 에포크 {cfg.swa_start_epoch}, LR {cfg.swa_lr}")

        scaler = GradScaler(enabled=cfg.use_amp)

        # TensorBoard
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_dir = os.path.join(cfg.log_dir, run_name)
        writer = SummaryWriter(log_dir=tb_dir)
        config_text = json.dumps(
            {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
            indent=2, default=str)
        writer.add_text('config', f'```\\n{config_text}\\n```', 0)
        logger.info(f"TensorBoard: tensorboard --logdir {cfg.log_dir}")

        # Resume
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0

        if resume_from and os.path.exists(resume_from):
            ckpt = self._load_full_checkpoint(resume_from, optimizer, scheduler, scaler)
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            patience_counter = ckpt['patience_counter']
            global_step = ckpt['global_step']
            logger.info(f"학습 이어하기: epoch {start_epoch}부터 재개")

        history = {'train_loss': [], 'val_loss': [], 'val_direction_acc': [], 'learning_rate': []}

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"학습 시작: {len(train_dataset)} samples, device={cfg.device}, "
                    f"AMP={cfg.use_amp}, EMA={cfg.use_ema}, params={param_count:,}")
        writer.add_scalar('info/param_count', param_count, 0)

        for epoch in range(start_epoch, cfg.max_epochs):
            # ── Train ──
            self.model.train()
            epoch_loss = 0.0
            epoch_dir_correct = 0
            epoch_dir_total = 0
            optimizer.zero_grad()


            for step, batch in enumerate(train_loader):
                temporal = batch['temporal'].to(cfg.device)
                static = batch['static'].to(cfg.device)
                target = batch['target'].to(cfg.device)

                # [IDEA] Mixup Augmentation (에포크 50 이후부터만 50% 확률)
                if epoch >= 50 and np.random.rand() < 0.5:  # ← epoch 조건 추가
                    lam = np.random.beta(0.2, 0.2)
                    index = torch.randperm(temporal.size(0)).to(temporal.device)
                    temporal = lam * temporal + (1 - lam) * temporal[index]
                    static = lam * static + (1 - lam) * static[index]
                    target = lam * target + (1 - lam) * target[index]

                with autocast(enabled=cfg.use_amp, device_type='cuda'):
                    preds, _, _ = self.model(temporal, static)
                    loss, loss_dict = criterion(preds, target)
                    loss = loss / cfg.accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % cfg.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if self.ema:
                        self.ema.update(self.model)
                    
                    # [SWA] 스케줄러 분기
                    if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                        # SWA는 에포크 단위로 스텝하지만, 여기서는 batch step에서 아무것도 안함
                        # 실제 step()은 에포크 끝에서 호출
                        pass
                    elif is_step_scheduler:
                        scheduler.step()

                    global_step += 1

                    if global_step % cfg.log_every_n_steps == 0:
                        lr_now = optimizer.param_groups[0]['lr']
                        writer.add_scalar('step/train_loss', loss.item() * cfg.accumulation_steps, global_step)
                        writer.add_scalar('step/payoff_loss', loss_dict['payoff_loss'], global_step)
                        writer.add_scalar('step/lr', lr_now, global_step)
                        gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                        writer.add_scalar('step/grad_norm', gn, global_step)
                        if cfg.use_amp:
                            writer.add_scalar('step/amp_scale', scaler.get_scale(), global_step)

                epoch_loss += loss.item() * cfg.accumulation_steps

                with torch.no_grad():
                    mid = cfg.quantiles.index(0.5)
                    pd_dir = torch.sign(preds[:, 0, mid])
                    ad_dir = torch.sign(target[:, 0])
                    epoch_dir_correct += (pd_dir == ad_dir).sum().item()
                    epoch_dir_total += target.shape[0]

            # ── SWA Update (Epoch End) ──
            if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                logger.info(f"Epoch {epoch+1}/{cfg.max_epochs} - [SWA Update] LR: {swa_scheduler.get_last_lr()[0]:.6f}")

            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            train_dir_acc = epoch_dir_correct / max(epoch_dir_total, 1)

            # ── Validate (EMA) ──
            if self.ema:
                self.ema.apply_shadow(self.model)
            val_loss, val_dir_acc, val_ld = self._validate(val_loader, criterion)
            if self.ema:
                self.ema.restore(self.model)

            if not is_step_scheduler:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_direction_acc'].append(val_dir_acc)
            history['learning_rate'].append(current_lr)

            # TensorBoard epoch
            writer.add_scalars('epoch/loss', {'train': avg_train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('epoch/dir_acc', {'train': train_dir_acc, 'val': val_dir_acc}, epoch)
            writer.add_scalar('epoch/val_payoff_loss', val_ld['payoff_loss'], epoch)
            writer.add_scalar('epoch/lr', current_lr, epoch)

            if (epoch + 1) % 5 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'params/{name}', param.data, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'grads/{name}', param.grad, epoch)

            is_best = val_loss < best_val_loss
            logger.info(
                f"Epoch {epoch+1}/{cfg.max_epochs} | "
                f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Dir(T/V): {train_dir_acc:.1%}/{val_dir_acc:.1%} | "
                f"LR: {current_lr:.2e} | "
                f"{'★ best' if is_best else f'patience {patience_counter+1}/{cfg.patience}'}")

            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best')
                self._save_full_checkpoint('best_full', epoch, optimizer, scheduler,
                                          scaler, best_val_loss, patience_counter, global_step)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_full_checkpoint(f'epoch_{epoch+1}_full', epoch, optimizer,
                                          scheduler, scaler, best_val_loss,
                                          patience_counter, global_step)

        writer.close()
        
        # [SWA] 학습 종료 후 저장
        if cfg.use_swa and swa_model is not None:
             # BN update (LayerNorm이라 필수 아닐 수 있으나 권장)
             # torch.optim.swa_utils.update_bn(train_loader, swa_model, device=cfg.device)
             
             if not os.path.exists(cfg.model_dir):
                os.makedirs(cfg.model_dir, exist_ok=True)
                
             swa_path = os.path.join(cfg.model_dir, 'tft_swa.pt')
             torch.save(swa_model.state_dict(), swa_path)
             logger.info(f"SWA 모델 저장 완료: {swa_path}")

        self._load_checkpoint('best')
        if self.ema:
            self.ema.apply_shadow(self.model)

        logger.info(f"학습 완료. Best val loss: {best_val_loss:.6f}")
        logger.info(f"TensorBoard: tensorboard --logdir {cfg.log_dir}")
        return history

    @torch.no_grad()
    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        agg_ld = {'payoff_loss': 0.0, 'direction_accuracy': 0.0}
        mid = self.config.quantiles.index(0.5)

        for batch in val_loader:
            temporal = batch['temporal'].to(self.config.device)
            static = batch['static'].to(self.config.device)
            target = batch['target'].to(self.config.device)

            with autocast(enabled=self.config.use_amp, device_type='cuda'):
                preds, _, _ = self.model(temporal, static)
                loss, ld = criterion(preds, target)

            total_loss += loss.item()
            for k, v in ld.items():
                agg_ld[k] += v
            # ld['direction_accuracy'] is batch mean, but we want exact count for total accuracy
            # so we recalculate:
            correct += (torch.sign(preds[:, 0, mid]) == torch.sign(target[:, 0])).sum().item()
            total += target.shape[0]

        n = max(len(val_loader), 1)
        return total_loss / n, correct / max(total, 1), {k: v / n for k, v in agg_ld.items()}

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """예측 수행 및 결과 반환."""
        assert self.model is not None, "모델이 학습되지 않았습니다."
        assert self.feature_cols is not None, "feature_cols가 설정되지 않았습니다."
        
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow(self.model)
        
        df_norm = df.copy()
        
        # Temporal 피처 정규화
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in self.feature_cols if c not in static_cols]
        
        # scaler_params 적용
        mean = self.scaler_params['mean']
        std = self.scaler_params['std']
        df_norm[temporal_cols] = (df_norm[temporal_cols] - mean) / std

        # 더미 타겟 (예측 시엔 사용 안 함)
        if self.target_col not in df.columns:
            df_norm[self.target_col] = 0.0
        
        dataset = TFTDataset(self.config, df_norm, self.feature_cols)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_preds = []
        all_attn = []
        all_vars = []

        cfg = self.config
        mid_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles)//2

        for batch in loader:
            temporal = batch['temporal'].to(cfg.device)
            static = batch['static'].to(cfg.device)

            with torch.no_grad(), autocast(enabled=cfg.use_amp, device_type='cuda'):
                preds, attn, var_w = self.model(temporal, static)
                
            all_preds.append(preds.cpu().numpy())
            all_attn.append(attn.cpu().numpy())
            all_vars.append(var_w.cpu().numpy())

        if self.ema:
            self.ema.restore(self.model)

        if not all_preds:
            return {}

        preds_arr = np.concatenate(all_preds, axis=0)  # (N, H, Q)
        attn_arr = np.concatenate(all_attn, axis=0)
        vars_arr = np.concatenate(all_vars, axis=0)

        # ★★★ Confidence 계산 수정 (Overflow 방지) ★★★
        q10_idx = cfg.quantiles.index(0.1)
        q90_idx = cfg.quantiles.index(0.9)
        
        # ★ Spread 계산 (역정규화 BEFORE)
        # 현재: 역정규화 후 spread 계산 → 스케일 왜곡
        # 수정: 역정규화 전에 spread 계산
        
        # 정규화된 상태에서 spread 계산
        spread_normalized = np.clip(
            preds_arr[:, :, q90_idx] - preds_arr[:, :, q10_idx],
            1e-6, 10.0  # ← 범위 확대
        )
        
        # Confidence: spread가 작을수록 확신
        # 분산 기반: spread의 표준편차로 정규화
        spread_std = spread_normalized.std() + 1e-6
        spread_norm = spread_normalized / spread_std
        confidence = 1.0 / (1.0 + spread_norm)  # 0.5 ~ 1.0 범위
        
        # 예측값 역정규화 (target_scaler 사용)
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            t_mean, t_std = self.target_scaler
            preds_arr = preds_arr * t_std + t_mean
        
        return {
            'quantiles': preds_arr,                    # 전체 quantile 예측
            'median_pred': preds_arr[:, :, mid_idx],   # 중앙값 (0.5 quantile)
            'attention': attn_arr,                     # Attention weights
            'variable_importance': vars_arr,           # 변수 중요도
            'confidence': confidence,                  # ★ 수정됨
            'direction_prob': (preds_arr > 0).mean(axis=-1),  # 상승 확률
        }


    @classmethod
    def load(cls, path: str):
        """저장된 모델 로드."""
        # Config 로드
        model_dir = os.path.dirname(path)
        tag = os.path.basename(path).replace('tft_', '').replace('.pt', '')
        meta_path = os.path.join(model_dir, f'tft_{tag}_meta.json')
        
        config = TFTConfig()
        feature_cols = None
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            # dict to config
            for k, v in meta['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)
            feature_cols = meta['feature_cols']
        
        instance = cls(config)
        instance.model = TemporalFusionTransformer(config)
        instance.model.to(config.device)
        
        # Weights 로드
        checkpoint = torch.load(path, map_location=config.device)
        if 'model_state_dict' in checkpoint:
            instance.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            instance.model.load_state_dict(checkpoint)
            
        if feature_cols:
            instance.feature_cols = feature_cols
            
        if os.path.exists(meta_path):
             with open(meta_path, 'r') as f:
                meta = json.load(f)
             instance.scaler_params = {k: np.array(v) if isinstance(v, list) else v 
                                  for k, v in meta['scaler_params'].items()}

        if instance.config.use_ema:
             instance.ema = EMAModel(instance.model, decay=instance.config.ema_decay)
        
        return instance


# ════════════════════════════════════════════════════════════════
# [IDEA 7] Multi-Horizon Ensemble
# ════════════════════════════════════════════════════════════════
class TFTEnsemble:
    """
    여러 horizon에 특화된 TFT 모델들을 앙상블.
    """
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        """
        model_paths: 각 모델의 체크포인트 경로 리스트
        weights: 앙상블 가중치 (기본: [0.5, 0.3, 0.2])
        """
        self.models = [TFTSignalModel.load(p) for p in model_paths]
        self.weights = weights if weights is not None else [0.5, 0.3, 0.2]
        if len(self.models) != len(self.weights):
             # 가중치 개수 안맞으면 균등하게
             self.weights = [1.0/len(self.models)] * len(self.models)
             logger.warning("가중치 개수 불일치 -> 균등 가중치 적용")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        앙상블 예측 (median prediction의 첫 스텝).
        Returns:
            앙상블된 예측값 배열 (길이는 가장 짧은 horizon에 맞춰짐)
        """
        preds = []
        for m in self.models:
            result = m.predict(df)
            if not result: continue
            # 첫 번째 horizon의 median 예측 사용
            preds.append(result['median_pred'][:, 0])

        if not preds: return np.array([])
        
        # 길이 맞추기 (가장 짧은 예측 수)
        min_len = min(len(p) for p in preds)
        aligned = [p[-min_len:] for p in preds]

        ensemble = np.zeros(min_len)
        for w, p in zip(self.weights, aligned):
            ensemble += w * p
        return ensemble

    def predict_with_confidence(self, df: pd.DataFrame) -> Dict:
        """
        각 모델의 confidence도 함께 반환 (단순 평균).
        """
        preds = []
        confs = []
        for m in self.models:
            result = m.predict(df)
            if not result: continue
            preds.append(result['median_pred'][:, 0])
            confs.append(result['confidence'][:, 0])

        if not preds: return {}

        min_len = min(len(p) for p in preds)
        aligned_preds = [p[-min_len:] for p in preds]
        aligned_confs = [c[-min_len:] for c in confs]

        ensemble_pred = np.zeros(min_len)
        for w, p in zip(self.weights, aligned_preds):
            ensemble_pred += w * p

        # Confidence 앙상블 (가중 평균 or 단순 평균)
        ensemble_conf = np.zeros(min_len)
        norm_weights = np.array(self.weights) / sum(self.weights)
        for w, c in zip(norm_weights, aligned_confs):
            ensemble_conf += w * c

        return {
            'ensemble_pred': ensemble_pred,
            'ensemble_confidence': ensemble_conf,
            'individual_preds': aligned_preds,
            'individual_confs': aligned_confs,
        }