"""
Signal Module: Enhanced Temporal Fusion Transformer for 5-min ETH Day Trading

논문 기반 설계:
    1. Adaptive TFT (arXiv:2509.10542, 2025)
       - Volatility Rate 변환 (가격 → 수익률) — 비정상성 제거
       - 서브시리즈 패턴 기반 카테고리화 개념 차용
    2. ADE-TFT (Heliyon, 2024)
       - hidden_size=8 레이어에서 최고 성능 → 깊은 GRN 구조
    3. Multi-Crypto TFT (MDPI Systems, 2025)
       - On-chain + Technical 통합, z-score 기반 트레이딩 시그널 생성
    4. LSTM→GPT-2 비교 (Symmetry, 2025.12)
       - TFT가 비정상/레짐 전환이 잦은 자산(SOL)에서 최고 성능
       - 40+ 기술 지표 멀티변량 입력 시 VSN의 자동 변수 선택이 핵심

창의적 수정 (수익률 최적화):
    - Directional Asymmetric Loss: 방향 예측 정확도에 가중치 부여
    - Regime-Conditioned Attention: 시장 레짐 피처를 static enrichment에 주입
    - Multi-horizon 구조이되 트레이딩에 최적화된 horizon 설계
    - Confidence Score 출력: 예측 불확실성을 RL Agent에 전달

학습 인프라:
    - TensorBoard 로깅 (loss, 방향정확도, LR, gradient norm, 파라미터 분포)
    - Warmup + Cosine Annealing LR 스케줄러
    - EMA (Exponential Moving Average) 모델
    - Mixed Precision (AMP) 지원
    - Gradient Accumulation 지원
    - 학습 이어하기 (Resume): optimizer/scheduler/scaler/epoch 전체 저장
    - 시드 고정을 통한 재현성 보장

사용법:
    from core.tft_model import TFTSignalModel, TFTConfig

    config = TFTConfig(max_epochs=100, lr_scheduler='cosine')
    model = TFTSignalModel(config)
    model.fit(train_df, val_df, feature_cols)
    predictions = model.predict(test_df)

    # 학습 이어하기
    model.fit(train_df, val_df, feature_cols, resume_from='models/tft/tft_epoch_50_full.pt')
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

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════
@dataclass
class TFTConfig:
    """TFT 하이퍼파라미터 — 논문 최적값 기반 + 5분봉 튜닝."""

    # ── 입력/출력 ──
    input_window: int = 12           # 48 × 5min = 4시간 룩백
    forecast_horizon: int = 6        # 6 × 5min = 30분 예측
    num_features: int = 35           # ULTIMATE_FEATURE_COLS(30) + meta(5)
    target_col: str = 'target_cumret_6'   # Volatility Rate 타겟

    # ── 모델 구조 ──
    hidden_size: int = 32
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.3

    # ── Variable Selection Network ──
    num_static_features: int = 3     # session_asia/europe/us (현재 세션)
    num_temporal_features: int = 32  # 나머지 시계열 피처

    # ── Quantile 예측 (트레이딩 최적화) ──
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # ── 학습 기본 ──
    learning_rate: float = 3e-5
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    # ── 커스텀 손실 가중치 ──
    direction_loss_weight: float = 0.5   # 0.5 → 1.0으로 올려주세요
    large_move_weight: float = 2.0

    # ── LR 스케줄러 ──
    warmup_epochs: int = 5           # 초반 5 에폭 warmup (발산 방지)
    lr_scheduler: str = 'cosine'     # 'cosine' | 'plateau' | 'onecycle'
    min_lr: float = 1e-6

    # ── EMA ──
    use_ema: bool = True
    ema_decay: float = 0.999

    # ── Mixed Precision ──
    use_amp: bool = True

    # ── Gradient Accumulation ──
    accumulation_steps: int = 1      # >1이면 effective batch = batch_size * steps

    # ── 재현성 ──
    seed: int = 42

    # ── 로깅 ──
    log_dir: str = 'runs/tft'
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10

    # ── 기타 ──
    device: str = 'auto'
    model_dir: str = 'models/tft'

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.use_amp = False


# ════════════════════════════════════════════════════════════════
# 1.5. EMA MODEL
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
# 2. DATASET
# ════════════════════════════════════════════════════════════════
class TFTDataset(Dataset):
    """슬라이딩 윈도우 데이터셋."""

    def __init__(self, df: pd.DataFrame, config: TFTConfig, feature_cols: List[str], target_col: str):
        self.config = config
        self.static_cols = ['session_asia', 'session_europe', 'session_us']
        self.temporal_cols = [c for c in feature_cols if c not in self.static_cols]

        self.temporal_data = df[self.temporal_cols].values.astype(np.float32)
        self.static_data = df[self.static_cols].values.astype(np.float32)
        self.target_data = df[target_col].values.astype(np.float32)

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
# 3. MODEL COMPONENTS
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
    """TFT 해석 가능 MHA — Values를 헤드 간 공유."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, self.d_k)
        self.W_o = nn.Linear(self.d_k, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
      B, T, _ = query.shape
      Q = self.W_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
      K = self.W_k(key).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
      V = self.W_v(value)

      scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
      if mask is not None:
          # Use a fill value that is safe for the current dtype (especially for float16)
          fill_value = -torch.finfo(scores.dtype).max
          scores = scores.masked_fill(mask == 0, fill_value)

      attn_weights = F.softmax(scores, dim=-1)
      attn_weights = self.dropout(attn_weights)

      V_exp = V.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
      context = torch.matmul(attn_weights, V_exp).mean(dim=1)
      output = self.W_o(context)
      avg_attn = attn_weights.mean(dim=1)
      return output, avg_attn


# ════════════════════════════════════════════════════════════════
# 4. TFT CORE MODEL
# ════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    Enhanced TFT for 5-min ETH Trading.
    Input → VSN → LSTM Encoder → Static Enrichment → MHA → GRN → Quantile Output
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

        # self.lstm_encoder = nn.LSTM(
        #     input_size=H, hidden_size=H, num_layers=config.lstm_layers,
        #     batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.gru_encoder = nn.GRU(
            input_size=H, hidden_size=H, num_layers=config.lstm_layers,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
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

    def forward(self, temporal: torch.Tensor, static: torch.Tensor):
        B = temporal.shape[0]
        H = self.config.hidden_size

        static_emb = self.static_encoder(static)
        cs_e = self.static_context_enrichment(static_emb)
        cs_h = self.static_context_state_h(static_emb)
        selected, var_weights = self.temporal_vsn(temporal, cs_e)

        h0 = cs_h.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()
        gru_out, hidden = self.gru_encoder(selected, h0)  # h0는 static context에서 생성

        gated = self.post_lstm_gate(gru_out)
        temporal_feat = self.post_lstm_norm(gated + selected)

        cs_exp = cs_e.unsqueeze(1).expand(-1, temporal_feat.shape[1], -1)
        enriched = self.static_enrichment(temporal_feat, cs_exp)

        T = enriched.shape[1]
        mask = torch.tril(torch.ones(T, T, device=temporal.device)).unsqueeze(0)
        attn_out, attn_w = self.multihead_attn(enriched, enriched, enriched, mask=mask)
        attn_out = self.post_attn_norm(self.post_attn_gate(attn_out) + enriched)

        ff_out = self.pos_ff(attn_out)
        ff_out = self.pos_ff_norm(self.pos_ff_gate(ff_out) + attn_out)

        last_h = ff_out[:, -1, :]
        horizon_h = self.horizon_fc(last_h).view(B, self.config.forecast_horizon, H)

        q_preds = torch.cat([qh(horizon_h) for qh in self.quantile_heads], dim=-1)
        return q_preds, attn_w, var_weights


# ════════════════════════════════════════════════════════════════
# 5. CUSTOM LOSS
# ════════════════════════════════════════════════════════════════

# DirectionalQuantileLoss v3 — 방향 정확도를 직접 최적화
# TFT_model.py의 기존 DirectionalQuantileLoss를 이것으로 교체


class DirectionalQuantileLoss(nn.Module):
    """
    트레이딩 최적화 손실 함수 v3.

    v2 문제점:
        - Quantile Loss가 "거리"를 줄이는 데 집중 → 0 근처로 수렴
        - Direction penalty가 non-differentiable (torch.sign → gradient 0)
        - 결과: val loss 줄어도 방향 정확도는 안 오름

    v3 변경점:
        1. Differentiable Direction Loss: sign() 대신 tanh() 기반 soft direction
           → gradient가 실제로 흐르면서 방향 예측을 직접 학습
        2. Profit-weighted Loss: 방향 맞으면 보상, 틀리면 페널티
           → 트레이딩 수익과 직접 연동
        3. Large move: quantile loss에 직접 가중치 (v2와 동일)
    """

    def __init__(self, quantiles: List[float], direction_weight: float = 1.0,
                 large_move_weight: float = 2.0, sharpness: float = 20.0):
        super().__init__()
        self.quantiles = quantiles
        self.direction_weight = direction_weight
        self.large_move_weight = large_move_weight
        self.sharpness = sharpness  # tanh 기울기 — 클수록 sign()에 가까움
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions: (B, horizon, num_quantiles)
            targets:     (B, horizon)
        """
        targets_exp = targets.unsqueeze(-1)  # (B, H, 1)

        # ══════════════════════════════════════
        # 1. Weighted Quantile Loss (거리 최적화)
        # ══════════════════════════════════════
        q_losses = []
        for i, q in enumerate(self.quantiles):
            err = targets_exp - predictions[:, :, i:i+1]
            q_losses.append(torch.max(q * err, (q - 1) * err))
        q_loss_per_sample = torch.cat(q_losses, dim=-1).mean(dim=-1)  # (B, H)

        # Large move 가중치
        move_mag = targets.abs()
        threshold = torch.quantile(move_mag.flatten(), 0.8)
        sample_weight = torch.where(
            move_mag > threshold,
            torch.tensor(self.large_move_weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )
        weighted_quantile_loss = (sample_weight * q_loss_per_sample).mean()

        # ══════════════════════════════════════
        # 2. Differentiable Direction Loss (방향 최적화)
        # ══════════════════════════════════════
        median_pred = predictions[:, :, self.median_idx]  # (B, H)

        # soft sign: tanh(x * sharpness) → -1 ~ +1 (미분 가능)
        # sharpness가 클수록 sign()에 가까워지지만 gradient가 살아있음
        soft_pred_dir = torch.tanh(median_pred * self.sharpness)
        soft_actual_dir = torch.tanh(targets * self.sharpness)

        # 방향 일치도: 둘 다 같은 부호면 +1, 다르면 -1
        # direction_agreement = soft_pred * soft_actual → 맞으면 양수, 틀리면 음수
        direction_agreement = soft_pred_dir * soft_actual_dir

        # 방향이 틀릴수록 loss가 큼 (1 - agreement) / 2 → 맞으면 0, 틀리면 1
        direction_loss = ((1.0 - direction_agreement) / 2.0).mean()

        # ══════════════════════════════════════
        # 3. Profit-aligned Loss (수익 최적화)
        # ══════════════════════════════════════
        # 예측 방향대로 포지션을 잡았을 때의 (음수) 수익
        # = -sign(pred) * actual → 맞으면 음수(good), 틀리면 양수(bad)
        # soft version으로 gradient 확보
        neg_profit = -soft_pred_dir * targets
        profit_loss = neg_profit.mean()

        # ══════════════════════════════════════
        # Total Loss
        # ══════════════════════════════════════
        # quantile: 크기 정확도 (RL Agent에게 정확한 예측값 제공)
        # direction: 방향 정확도 (51% → 55% 목표)
        # profit: 실제 수익과 연동 (큰 움직임에서 맞추는 게 중요)
        total = (weighted_quantile_loss
                + self.direction_weight * direction_loss
                + 0.1 * profit_loss)

        # 모니터링용 방향 정확도 (hard sign, 로그용)
        with torch.no_grad():
            hard_pred = torch.sign(median_pred)
            hard_actual = torch.sign(targets)
            valid = (hard_actual != 0)
            if valid.sum() > 0:
                hard_dir_acc = (hard_pred[valid] == hard_actual[valid]).float().mean().item()
            else:
                hard_dir_acc = 0.5

        return total, {
            'quantile_loss': weighted_quantile_loss.item(),
            'direction_loss': direction_loss.item(),
            'profit_loss': profit_loss.item(),
            'direction_accuracy': hard_dir_acc,
        }

# ════════════════════════════════════════════════════════════════
# 6. HIGH-LEVEL WRAPPER
# ════════════════════════════════════════════════════════════════

class TFTSignalModel:
    """TFT 모델의 학습/예측/저장/로드 래퍼."""

    def __init__(self, config: TFTConfig = None):
        self.config = config or TFTConfig()
        self.model = None
        self.ema = None
        self.feature_cols = None
        self.target_col = config.target_col 
        self.scaler_params = {}

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

    # ── FIT ──
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
            feature_cols: List[str] = None, resume_from: str = None) -> Dict:
        cfg = self.config
        self._set_seed(cfg.seed)

        if feature_cols is None:
            from core.feature_engineering import ULTIMATE_FEATURE_COLS
            feature_cols = ULTIMATE_FEATURE_COLS
        self.feature_cols = feature_cols

        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in feature_cols if c not in static_cols]
        cfg.num_static_features = len(static_cols)
        cfg.num_temporal_features = len(temporal_cols)
        cfg.num_features = len(feature_cols)

        train_df, val_df = self._normalize(train_df, val_df, temporal_cols)

        train_df = train_df.dropna(subset=[cfg.target_col])
        val_df = val_df.dropna(subset=[cfg.target_col])

        if val_df is None:
            split_idx = int(len(train_df) * 0.8)
            val_df = train_df.iloc[split_idx:].copy()
            train_df = train_df.iloc[:split_idx].copy()

        train_dataset = TFTDataset(train_df, cfg, feature_cols, self.target_col)
        val_dataset = TFTDataset(val_df, cfg, feature_cols, self.target_col)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=0)

        # 모델 초기화
        self.model = TemporalFusionTransformer(cfg)
        self.model.to(cfg.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        steps_per_epoch = max(len(train_loader) // cfg.accumulation_steps, 1)
        scheduler = self._create_scheduler(optimizer, steps_per_epoch)
        is_step_scheduler = cfg.lr_scheduler in ('cosine', 'onecycle')

        criterion = DirectionalQuantileLoss(
            cfg.quantiles, cfg.direction_loss_weight, cfg.large_move_weight)

        if cfg.use_ema:
            self.ema = EMAModel(self.model, decay=cfg.ema_decay)

        scaler = GradScaler(enabled=cfg.use_amp)

        # TensorBoard
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_dir = os.path.join(cfg.log_dir, run_name)
        writer = SummaryWriter(log_dir=tb_dir)
        config_text = json.dumps(
            {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
            indent=2, default=str)
        writer.add_text('config', f'```\n{config_text}\n```', 0)
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
                    if is_step_scheduler:
                        scheduler.step()

                    global_step += 1

                    if global_step % cfg.log_every_n_steps == 0:
                        lr_now = optimizer.param_groups[0]['lr']
                        writer.add_scalar('step/train_loss', loss.item() * cfg.accumulation_steps, global_step)
                        writer.add_scalar('step/quantile_loss', loss_dict['quantile_loss'], global_step)
                        writer.add_scalar('step/direction_loss', loss_dict['direction_loss'], global_step)
                        writer.add_scalar('step/profit_loss', loss_dict['profit_loss'], global_step)
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
            writer.add_scalar('epoch/val_quantile', val_ld['quantile_loss'], epoch)
            writer.add_scalar('epoch/val_dir_penalty', val_ld['direction_loss'], epoch)
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
        self._load_checkpoint('best')
        if self.ema:
            self.ema.apply_shadow(self.model)

        logger.info(f"학습 완료. Best val loss: {best_val_loss:.6f}")
        logger.info(f"TensorBoard: tensorboard --logdir {cfg.log_dir}")
        return history

    # ── VALIDATE ──
    @torch.no_grad()
    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        agg_ld = {'quantile_loss': 0.0, 'direction_loss': 0.0, 'profit_loss': 0.0, 'direction_accuracy': 0.0}
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
            correct += (torch.sign(preds[:, 0, mid]) == torch.sign(target[:, 0])).sum().item()
            total += target.shape[0]

        n = max(len(val_loader), 1)
        return total_loss / n, correct / max(total, 1), {k: v / n for k, v in agg_ld.items()}

    # ── PREDICT ──
    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        assert self.model is not None, "모델이 학습되지 않았습니다."
        df_norm = self._apply_normalization(df)
        # ★ _apply_normalization이 scaler_params에 있는 모든 컬럼을 정규화하므로
        #    target_col(log_return)도 이미 1회 정규화됨 — 추가 정규화 불필요

        dataset = TFTDataset(df_norm, self.config, self.feature_cols, target_col=self.target_col)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        all_p, all_a, all_v = [], [], []
        for batch in loader:
            t = batch['temporal'].to(self.config.device)
            s = batch['static'].to(self.config.device)
            with autocast(enabled=self.config.use_amp, device_type='cuda'):
                p, a, v = self.model(t, s)
            all_p.append(p.cpu().float().numpy())
            all_a.append(a.cpu().float().numpy())
            all_v.append(v.cpu().float().numpy())

        qp = np.concatenate(all_p)
        attn = np.concatenate(all_a)
        vi = np.concatenate(all_v)

        # ★ 역변환: 정규화된 예측값 → 원래 스케일
        if hasattr(self, 'target_scaler'):
            t_m, t_s = self.target_scaler
            qp = qp * t_s + t_m

        mid = self.config.quantiles.index(0.5)
        q10 = self.config.quantiles.index(0.1)
        q90 = self.config.quantiles.index(0.9)
        iqr = np.clip(qp[:, :, q90] - qp[:, :, q10], 1e-8, None)

        return {
            'median_pred': qp[:, :, mid],
            'quantile_preds': qp,
            'confidence': 1.0 / (1.0 + iqr * 100),
            'direction_prob': (qp > 0).mean(axis=-1),
            'attention': attn,
            'var_importance': vi,
        }


    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        result = self.predict(df)
        latest = {k: v[-1] if isinstance(v, np.ndarray) else v for k, v in result.items()}
        q10 = self.config.quantiles.index(0.1)
        q90 = self.config.quantiles.index(0.9)
        return {
            'predicted_returns': latest['median_pred'],
            'confidence': latest['confidence'],
            'direction_prob': latest['direction_prob'],
            'risk_range': np.stack([
                latest['quantile_preds'][:, q10], latest['quantile_preds'][:, q90]], axis=-1),
        }

    # ── NORMALIZATION ──
    def _normalize(self, train_df, val_df, temporal_cols):
        """Z-score 정규화 (학습셋 통계 기준) — 타겟 별도 처리."""
        train_df = train_df.copy()
        self.scaler_params = {}
        target_col = self.config.target_col

        # ★ 타겟 컬럼은 별도로 정규화 — temporal 루프에서 제외
        normalize_cols = [c for c in temporal_cols if c != target_col]

        for col in normalize_cols:
            m, s = train_df[col].mean(), train_df[col].std()
            if s == 0 or pd.isna(s):
                s = 1.0
            self.scaler_params[col] = (m, s)
            train_df[col] = (train_df[col] - m) / s

        # 타겟 정규화 (원본 값 기준으로 1회만)
        if target_col in train_df.columns:
            t_m = train_df[target_col].mean()
            t_s = train_df[target_col].std()
            if t_s == 0 or pd.isna(t_s):
                t_s = 1.0
            self.target_scaler = (t_m, t_s)
            # 피처로 쓰이는 것도, 타겟으로 쓰이는 것도 동일한 스케일
            self.scaler_params[target_col] = (t_m, t_s)
            train_df[target_col] = (train_df[target_col] - t_m) / t_s
        else:
            self.target_scaler = (0.0, 1.0)

        if val_df is not None:
            val_df = val_df.copy()
            for col in normalize_cols:
                m, s = self.scaler_params[col]
                val_df[col] = (val_df[col] - m) / s
            if target_col in val_df.columns:
                t_m, t_s = self.target_scaler
                val_df[target_col] = (val_df[target_col] - t_m) / t_s

        return train_df, val_df


    def _apply_normalization(self, df):
        df = df.copy()
        for col, (m, s) in self.scaler_params.items():
            if col in df.columns:
                df[col] = (df[col] - m) / s
        return df

    # ── CHECKPOINTS ──
    def _save_full_checkpoint(self, tag, epoch, optimizer, scheduler, scaler,
                              best_val_loss, patience_counter, global_step):
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if self.config.use_amp else None,
            'ema_state': self.ema.state_dict() if self.ema else None,
            'config': self.config,
            'scaler_params': self.scaler_params,
            'feature_cols': self.feature_cols,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'global_step': global_step,
            'target_scaler': self.target_scaler,
        }, path)
        logger.debug(f"Full checkpoint: {path}")

    def _load_full_checkpoint(self, path, optimizer, scheduler, scaler):
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if self.config.use_amp and ckpt.get('scaler_state'):
            scaler.load_state_dict(ckpt['scaler_state'])
        if self.ema and ckpt.get('ema_state'):
            self.ema.load_state_dict(ckpt['ema_state'])
        self.scaler_params = ckpt['scaler_params']
        self.feature_cols = ckpt['feature_cols']
        self.target_scaler = ckpt['target_scaler']
        logger.info(f"Full checkpoint 로드: {path} (epoch {ckpt['epoch']})")
        return ckpt

    def _save_checkpoint(self, tag):
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'ema_state': self.ema.state_dict() if self.ema else None,
            'config': self.config,
            'scaler_params': self.scaler_params,
            'target_scaler': self.target_scaler,
            'feature_cols': self.feature_cols,
        }, path)

    def _load_checkpoint(self, tag):
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state'])
            if self.ema and ckpt.get('ema_state'):
                self.ema.load_state_dict(ckpt['ema_state'])
            self.scaler_params = ckpt['scaler_params']
            self.feature_cols = ckpt['feature_cols']
            self.target_scaler = ckpt['target_scaler']
            logger.info(f"체크포인트 로드: {path}")

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(self.config.model_dir, 'tft_final.pt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'ema_state': self.ema.state_dict() if self.ema else None,
            'config': self.config,
            'scaler_params': self.scaler_params,
            'target_scaler': self.target_scaler,
            'feature_cols': self.feature_cols,
        }, path)
        logger.info(f"모델 저장: {path}")

    @classmethod
    def load(cls, path: str) -> 'TFTSignalModel':
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        config = ckpt['config']
        inst = cls(config)
        inst.model = TemporalFusionTransformer(config)
        inst.model.load_state_dict(ckpt['model_state'])
        if config.use_ema and ckpt.get('ema_state'):
            inst.ema = EMAModel(inst.model, decay=config.ema_decay)
            inst.ema.load_state_dict(ckpt['ema_state'])
            inst.ema.apply_shadow(inst.model)
        inst.model.to(config.device)
        inst.model.eval()
        inst.scaler_params = ckpt['scaler_params']
        inst.feature_cols = ckpt['feature_cols']
        inst.target_scaler = ckpt['target_scaler']
        logger.info(f"모델 로드: {path}")
        return inst