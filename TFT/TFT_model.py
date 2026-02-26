"""
Signal Module: Temporal Fusion Transformer for 5-min ETH Day Trading

복원 버전 - 52% / 53.1% 고확신 달성했던 설정 기반
+ Dual GRU (momentum/reversion) with Hurst gating 유지
- BCE direction loss 제거 (로짓 오용 문제)
- Mixup 제거 (시계열 파괴)
- Volatility targeting 제거 (불필요한 복잡도)
- Quantiles 11개 → 5개 복원
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
from torch.optim.swa_utils import AveragedModel, SWALR
from collections import defaultdict

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════
@dataclass
class TFTConfig:
    training: bool = True
    """TFT 하이퍼파라미터 — 52%/53.1% 달성 설정 복원."""
    # ── 입력/출력 ──
    input_window: int = 64           # 48 × 5min = 4시간 룩백
    forecast_horizon: int = 6        # 6 × 5min = 30분 예측
    target_col: str = 'target_ret_6'

    # ── 모델 구조 ──
    hidden_size: int = 32
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.2            # ★ 0.3 → 0.2 복원

    # ── Variable Selection Network ──
    num_features: int = 35
    num_static_features: int = 3
    num_temporal_features: int = field(init=False)

    # ── Quantile 예측 (5개로 복원) ──
    quantiles: List[float] = field(default_factory=lambda: 
        [0.1, 0.3, 0.5, 0.7, 0.9])  # ★ 11개 → 5개 복원

    # ── 학습 기본 ──
    learning_rate: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 100               # ★ 20 → 50 복원
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    # ── Loss 파라미터 ──
    direction_loss_weight: float = 8.0   # direction penalty 비중
    large_move_weight: float = 6.0       # 큰 움직임 가중치

    # ── LR 스케줄러 ──
    warmup_epochs: int = 20
    lr_scheduler: str = 'cosine'     # ★ onecycle → cosine 복원
    min_lr: float = 1e-7

    # ── EMA ──
    use_ema: bool = True
    ema_decay: float = 0.999         # ★ 0.9999 → 0.999 복원

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

    # ── SWA ──
    use_swa: bool = True
    swa_start_epoch: int = 300
    swa_lr: float = 5e-5

    # ── 기타 ──
    device: str = 'auto'
    model_dir: str = 'data/tft'

    def __post_init__(self):
        # num_temporal_features는 항상 num_features - num_static_features
        self.num_temporal_features = self.num_features - self.num_static_features
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.use_amp = False


# ════════════════════════════════════════════════════════════════
# 1.5. EMA MODEL
# ════════════════════════════════════════════════════════════════
class EMAModel:
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
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
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
# 2. DATASET (단순화: volatility 필드 제거)
# ════════════════════════════════════════════════════════════════
class TFTDataset(Dataset):
    """슬라이딩 윈도우 데이터셋."""
    def __init__(self, config: TFTConfig, df: pd.DataFrame, feature_cols: List[str]):
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
# 3. MODEL COMPONENTS
# ════════════════════════════════════════════════════════════════
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
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
# 4. TFT CORE MODEL (Dual GRU + Hurst gating 유지)
# ════════════════════════════════════════════════════════════════
class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        self.training = config.training
        H = config.hidden_size

        self.temporal_vsn = VariableSelectionNetwork(
            config.num_temporal_features, H, config.dropout, context_size=H)
        self.static_encoder = nn.Sequential(
            nn.Linear(config.num_static_features, H), nn.ReLU(), nn.Linear(H, H))
        self.static_context_enrichment = GatedResidualNetwork(H, H, H, config.dropout)
        self.static_context_state_h = GatedResidualNetwork(H, H, H, config.dropout)

        # ★ Dual GRU 유지 (momentum + reversion)
        self.gru_momentum = nn.GRU(
            input_size=H, hidden_size=H, num_layers=config.lstm_layers,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.gru_reversion = nn.GRU(
            input_size=H, hidden_size=H, num_layers=config.lstm_layers,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)

        self.feature_cols = None
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
        """Dual GRU gating에 사용할 피처 인덱스 설정."""
        self.feature_cols = feature_cols
        
        if 'hurst_48' in feature_cols:
            self.hurst_idx = feature_cols.index('hurst_48')
            logger.info(f"✅ Hurst-based gating: hurst_48 at index {self.hurst_idx}")
        elif 'regime_trending' in feature_cols:
            self.regime_trending_idx = feature_cols.index('regime_trending')
            logger.info(f"✅ Regime-based gating: regime_trending at index {self.regime_trending_idx}")
        else:
            logger.warning("⚠️ No hurst/regime feature → uniform 50/50 gating")

    def forward(self, temporal: torch.Tensor, static: torch.Tensor):
        B, T, _ = temporal.shape
        H = self.config.hidden_size
        
        # SOTA 퀀트 증강 기법 (학습(Training) 중에만 작동)
        if self.training:
            # [기법 1] Volatility-Adaptive Noise (변동성 비례 노이즈)
            # 과거 48봉의 '실제 변동성(std)'을 계산하여, 그 변동성의 5%만큼만 노이즈를 주입
            local_std = temporal.std(dim=1, keepdim=True) + 1e-6
            noise_scale = 0.05  # 실제 변동성의 5% 수준
            temporal = temporal + torch.randn_like(temporal) * local_std * noise_scale

            # [기법 2] Time-Step Masking (시간축 블라인드)
            # 전체 캔들 중 5% 확률로 캔들의 정보 전체를 가려버림 (과적합 원천 차단)
            mask_prob = 0.05 
            # 피처(F)는 유지하고 시간(T)을 통째로 끄기 위해 (B, T, 1) 형태의 마스크 생성
            time_mask = (torch.rand(B, T, 1, device=temporal.device) > mask_prob).float()
            
            # 마스킹된 부분의 스케일 붕괴를 막기 위한 보정 (Dropout의 표준 원리)
            temporal = (temporal * time_mask) / (1.0 - mask_prob)
        
        static_emb = self.static_encoder(static)
        cs_e = self.static_context_enrichment(static_emb)
        cs_h = self.static_context_state_h(static_emb)

        selected, var_weights = self.temporal_vsn(temporal, cs_e)

        h0 = cs_h.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()

        out_m, _ = self.gru_momentum(selected, h0)
        out_r, _ = self.gru_reversion(selected, h0)

        # ★ Hurst-based Regime Gating
        if self.hurst_idx is not None:
            hurst = temporal[:, -1, self.hurst_idx]
            momentum_gate = torch.sigmoid(5.0 * (hurst - 0.5))
            gate0 = momentum_gate.unsqueeze(-1).unsqueeze(-1)
            gate1 = (1.0 - momentum_gate).unsqueeze(-1).unsqueeze(-1)
        elif self.regime_trending_idx is not None:
            regime = temporal[:, -1, self.regime_trending_idx]
            gate0 = regime.unsqueeze(-1).unsqueeze(-1)
            gate1 = (1.0 - regime).unsqueeze(-1).unsqueeze(-1)
        else:
            gate0 = torch.ones(B, 1, 1, device=temporal.device) * 0.5
            gate1 = torch.ones(B, 1, 1, device=temporal.device) * 0.5

        gru_out = gate0 * out_m + gate1 * out_r

        gated = self.post_lstm_gate(gru_out)
        temporal_feat = self.post_lstm_norm(gated + selected)

        cs_exp = cs_e.unsqueeze(1).expand(-1, T, -1)
        enriched = self.static_enrichment(temporal_feat, cs_exp)

        causal_mask = torch.tril(torch.ones(T, T, device=temporal.device)).unsqueeze(0)
        attn_out, attn_w = self.multihead_attn(enriched, enriched, enriched, mask=causal_mask)
        attn_out = self.post_attn_norm(self.post_attn_gate(attn_out) + enriched)

        ff_out = self.pos_ff(attn_out)
        ff_out = self.pos_ff_norm(self.pos_ff_gate(ff_out) + attn_out)

        last_h = ff_out[:, -1, :]
        horizon_h = self.horizon_fc(last_h).view(B, self.config.forecast_horizon, H)

        q_preds = torch.cat([qh(horizon_h) for qh in self.quantile_heads], dim=-1)
        return q_preds, attn_w, var_weights


# ════════════════════════════════════════════════════════════════
# 5. LOSS: Directional Quantile Loss v3 (52%/53.1% 달성 버전 복원)
# ════════════════════════════════════════════════════════════════
# [TFT_model.py 의 5. LOSS 부분 전체 교체]

# [TFT_model.py 의 DirectionalQuantileLoss 클래스 전체 교체]

class DirectionalQuantileLoss(nn.Module):
    """
    기울기 소실(Vanishing Gradient)을 완벽히 제거한 최적화된 손실 함수.
    큰 움직임(Large Moves)에 대한 패널티를 선형적으로 증폭시킴.
    """
    def __init__(self, quantiles: List[float],
                 direction_weight: float = 5.0,     # 🚨 방향성 틀림에 대한 가중치 대폭 상향
                 large_move_weight: float = 5.0,    # 🚨 큰 움직임 가중치 상향
                 sharpe_weight: float = 0.5):
        super().__init__()
        self.quantiles = quantiles
        self.direction_weight = direction_weight
        self.large_move_weight = large_move_weight
        self.sharpe_weight = sharpe_weight
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        B, H, Q = predictions.shape
        
        # ── 1. Quantile Loss (기본 체급) ──
        targets_exp = targets.unsqueeze(-1).expand_as(predictions)
        errors = targets_exp - predictions
        
        quantile_tensor = torch.tensor(
            self.quantiles, device=predictions.device, dtype=predictions.dtype
        ).unsqueeze(0).unsqueeze(0)
        
        quantile_loss = torch.max(
            quantile_tensor * errors,
            (quantile_tensor - 1) * errors
        )
        
        move_size = torch.abs(targets)
        move_size_exp = move_size.unsqueeze(-1)
        
        # 상위 움직임에 가중치 부여
        threshold = move_size.median() 
        large_mask = (move_size_exp > threshold).float()
        
        weights = 1.0 + (self.large_move_weight - 1.0) * large_mask
        weighted_ql = (quantile_loss * weights).mean()
        
        # ── 2. Non-Saturating Directional Penalty (핵심: 기울기 소실 해결) ──
        pred_median = predictions[:, :, self.median_idx]
        actual_sign = torch.sign(targets)
        
        # 🚨 예측값과 정답 부호가 다를 경우(곱이 음수)에만 양수 패널티 반환 (ReLU)
        # 거기에 실제 변동폭(move_size)을 곱해서 큰 움직임을 틀릴수록 막대한 그래디언트 발생!
        # (예: 3% 파동을 틀리면 0.1% 파동 틀린 것보다 30배 더 강한 전기충격 부여)
        wrong_dir_penalty = torch.relu(-pred_median * actual_sign) * move_size
        
        # 스케일 보정 (수익률 단위가 0.001 단위이므로 100배 곱해서 Loss 실효성 확보)
        direction_loss = wrong_dir_penalty.mean() * 100.0
        
        # ── 3. Sharpe Ratio Loss (부드러운 포지션 사이징) ──
        # Tanh 스케일러를 2000 -> 50으로 확 줄여서 그래디언트 생존 (2% 예측일때 0.76 도달)
        soft_position = torch.tanh(pred_median * 50.0)
        simulated_returns = soft_position * targets
        
        expected_return = simulated_returns.mean()
        var = simulated_returns.var(unbiased=False)
        volatility = torch.sqrt(torch.clamp(var, min=1e-8))
        
        sharpe_ratio = expected_return / volatility
        sharpe_loss = -sharpe_ratio * self.sharpe_weight
        
        total = weighted_ql + (self.direction_weight * direction_loss) + sharpe_loss
        
        with torch.no_grad():
            dir_acc = (torch.sign(pred_median) == actual_sign).float().mean().item()
        
        return total, {
            'quantile_loss': weighted_ql.item(),
            'direction_loss': direction_loss.item(),
            'sharpe_loss': sharpe_loss.item(),
            'direction_accuracy': dir_acc,
        }

# ════════════════════════════════════════════════════════════════
# 6. HIGH-LEVEL WRAPPER
# ════════════════════════════════════════════════════════════════
class TFTSignalModel:
    def __init__(self, config: TFTConfig = None):
        self.config = config
        self.model = TemporalFusionTransformer(config)
        self.model.to(config.device)
        self.ema = None
        self.feature_cols = None
        self.target_col = config.target_col
        self.scaler_params = {}
        self.target_scaler = None

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=7, min_lr=cfg.min_lr)

    def _save_checkpoint(self, tag: str):
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        torch.save(self.model.state_dict(), path)
        
        meta_path = os.path.join(self.config.model_dir, f'tft_{tag}_meta.json')
        meta = {
            'feature_cols': self.feature_cols,
            'scaler_params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in self.scaler_params.items()},
            'target_scaler': self.target_scaler,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"체크포인트 저장: {path}")

    def _save_full_checkpoint(self, tag: str, epoch, optimizer, scheduler, scaler,
                              best_val_loss, patience_counter, global_step):
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
            'target_scaler': self.target_scaler,
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
        
        meta_path = os.path.join(self.config.model_dir, f'tft_{tag}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_cols = meta['feature_cols']
            self.scaler_params = {k: np.array(v) if isinstance(v, list) else v 
                                 for k, v in meta['scaler_params'].items()}
            self.target_scaler = meta.get('target_scaler', None)

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
        self.target_scaler = checkpoint.get('target_scaler', None)
        
        return checkpoint

    def _normalize(self, train_df, val_df, cols):
        """피처만 정규화. 타겟은 원본 유지."""
        train_df = train_df.copy()
        if val_df is not None:
            val_df = val_df.copy()

        mean = train_df[cols].mean()
        std = train_df[cols].std().replace(0, 1.0)
        train_df[cols] = (train_df[cols] - mean) / std
        if val_df is not None:
            val_df[cols] = (val_df[cols] - mean) / std
        self.scaler_params = {'mean': mean.values, 'std': std.values}
        self.target_scaler = None  # 타겟 정규화 안 함
        return train_df, val_df

    def fit(self, cfg: TFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, 
            feature_cols: List[str], resume_from=None):
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in feature_cols if c not in static_cols]

        cfg.num_static_features = len(static_cols)
        cfg.num_temporal_features = len(temporal_cols)
        cfg.num_features = len(feature_cols)

        self.config = cfg
        self.feature_cols = feature_cols
        self.target_col = cfg.target_col

        self.model = TemporalFusionTransformer(cfg)
        self.model.to(cfg.device)
        self.model.set_feature_indices(temporal_cols)
        
        train_norm, val_norm = self._normalize(train_df, val_df, temporal_cols)

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

        # ★ 복원된 Loss 함수
        criterion = DirectionalQuantileLoss(
            cfg.quantiles,
            direction_weight=cfg.direction_loss_weight,
            large_move_weight=cfg.large_move_weight,
        )

        if cfg.use_ema:
            self.ema = EMAModel(self.model, decay=cfg.ema_decay)

        swa_model = None
        swa_scheduler = None
        if cfg.use_swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
            logger.info(f"SWA 활성화: epoch {cfg.swa_start_epoch}부터, LR {cfg.swa_lr}")

        scaler = GradScaler(enabled=cfg.use_amp)

        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_dir = os.path.join(cfg.log_dir, run_name)
        writer = SummaryWriter(log_dir=tb_dir)
        config_text = json.dumps(
            {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
            indent=2, default=str)
        writer.add_text('config', f'```\n{config_text}\n```', 0)

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
            logger.info(f"학습 이어하기: epoch {start_epoch}부터")

        history = {'train_loss': [], 'val_loss': [], 'val_direction_acc': [], 'learning_rate': []}

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"학습 시작: {len(train_dataset)} samples, device={cfg.device}, "
                    f"AMP={cfg.use_amp}, EMA={cfg.use_ema}, params={param_count:,}")

        mid = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2

        # ---------- Train Loop ----------
        for epoch in range(start_epoch, cfg.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_dir_correct = 0
            epoch_dir_total = 0
            optimizer.zero_grad()

            in_swa = cfg.use_swa and epoch >= cfg.swa_start_epoch

            for step, batch in enumerate(train_loader):
                temporal = batch['temporal'].to(cfg.device)
                static = batch['static'].to(cfg.device)
                target = batch['target'].to(cfg.device)

                # ★ Mixup 제거됨 (시계열에서 의미 없음)

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
                    
                    # ★ SWA 구간에서는 기본 scheduler skip
                    if not in_swa and is_step_scheduler:
                        scheduler.step()

                    global_step += 1

                    if global_step % cfg.log_every_n_steps == 0:
                        lr_now = optimizer.param_groups[0]['lr']
                        writer.add_scalar('step/train_loss', loss.item() * cfg.accumulation_steps, global_step)
                        writer.add_scalar('step/quantile_loss', loss_dict['quantile_loss'], global_step)
                        writer.add_scalar('step/direction_loss', loss_dict['direction_loss'], global_step)
                        writer.add_scalar('step/direction_acc', loss_dict['direction_accuracy'], global_step)
                        writer.add_scalar('step/lr', lr_now, global_step)

                epoch_loss += loss.item() * cfg.accumulation_steps

                with torch.no_grad():
                    pd_dir = torch.sign(preds[:, 0, mid])
                    ad_dir = torch.sign(target[:, 0])
                    epoch_dir_correct += (pd_dir == ad_dir).sum().item()
                    epoch_dir_total += target.shape[0]

            # SWA update
            if in_swa:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()

            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            train_dir_acc = epoch_dir_correct / max(epoch_dir_total, 1)

            if self.ema:
                self.ema.apply_shadow(self.model)
            val_loss, val_dir_acc, val_ld = self._validate(val_loader, criterion)
            if self.ema:
                self.ema.restore(self.model)

            if not is_step_scheduler and not in_swa:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_direction_acc'].append(val_dir_acc)
            history['learning_rate'].append(current_lr)

            writer.add_scalars('epoch/loss', {'train': avg_train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('epoch/dir_acc', {'train': train_dir_acc, 'val': val_dir_acc}, epoch)
            writer.add_scalar('epoch/lr', current_lr, epoch)

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
        
        if cfg.use_swa and swa_model is not None:
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
        agg_ld = {
            'quantile_loss': 0.0,
            'direction_loss': 0.0,
            'profit_loss': 0.0,
            'direction_accuracy': 0.0,
        }
        mid = self.config.quantiles.index(0.5) if 0.5 in self.config.quantiles else len(self.config.quantiles) // 2

        for batch in val_loader:
            temporal = batch['temporal'].to(self.config.device)
            static = batch['static'].to(self.config.device)
            target = batch['target'].to(self.config.device)

            with autocast(enabled=self.config.use_amp, device_type='cuda'):
                preds, _, _ = self.model(temporal, static)
                loss, ld = criterion(preds, target)

            total_loss += loss.item()
            for k in agg_ld.keys():
                agg_ld[k] += ld.get(k, 0.0)
            
            correct += (torch.sign(preds[:, 0, mid]) == torch.sign(target[:, 0])).sum().item()
            total += target.shape[0]

        n = max(len(val_loader), 1)
        return total_loss / n, correct / max(total, 1), {k: v / n for k, v in agg_ld.items()}

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        assert self.model is not None, "모델이 학습되지 않았습니다."
        assert self.feature_cols is not None, "feature_cols가 설정되지 않았습니다."
        
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow(self.model)
        
        df_norm = df.copy()
        
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in self.feature_cols if c not in static_cols]
        
        mean = self.scaler_params['mean']
        std = self.scaler_params['std']
        
        # ★ Shape 검증
        assert len(mean) == len(temporal_cols), \
            f"Scaler shape mismatch: {len(mean)} vs {len(temporal_cols)} features"
        
        df_norm[temporal_cols] = (df_norm[temporal_cols] - mean) / std

        if self.target_col not in df.columns:
            df_norm[self.target_col] = 0.0
        
        dataset = TFTDataset(self.config, df_norm, self.feature_cols)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_preds = []
        all_attn = []
        all_vars = []

        cfg = self.config
        mid_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2

        for batch in loader:
            temporal = batch['temporal'].to(cfg.device)
            static = batch['static'].to(cfg.device)

            with autocast(enabled=cfg.use_amp, device_type='cuda'):
                preds, attn, var_w = self.model(temporal, static)
                
            all_preds.append(preds.cpu().numpy())
            all_attn.append(attn.cpu().numpy())
            all_vars.append(var_w.cpu().numpy())

        if self.ema:
            self.ema.restore(self.model)

        if not all_preds:
            return {}

        preds_arr = np.concatenate(all_preds, axis=0)
        attn_arr = np.concatenate(all_attn, axis=0)
        vars_arr = np.concatenate(all_vars, axis=0)

        q10_idx = cfg.quantiles.index(0.1)
        q90_idx = cfg.quantiles.index(0.9)
        
        spread = np.clip(
            preds_arr[:, :, q90_idx] - preds_arr[:, :, q10_idx],
            1e-6, 10.0
        )
        spread_std = spread.std() + 1e-6
        confidence = 1.0 / (1.0 + spread / spread_std)

        return {
            'quantiles': preds_arr,
            'median_pred': preds_arr[:, :, mid_idx],
            'attention': attn_arr,
            'variable_importance': vars_arr,
            'confidence': confidence,
            'direction_prob': (preds_arr > 0).mean(axis=-1),
        }

    @classmethod
    def load(cls, path: str):
        model_dir = os.path.dirname(path)
        tag = os.path.basename(path).replace('tft_', '').replace('.pt', '')
        meta_path = os.path.join(model_dir, f'tft_{tag}_meta.json')
        
        config = TFTConfig()
        feature_cols = None
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            for k, v in meta['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)
            feature_cols = meta['feature_cols']
        
        instance = cls(config)
        instance.model = TemporalFusionTransformer(config)
        instance.model.to(config.device)
        
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
            instance.target_scaler = meta.get('target_scaler', None)

        if instance.config.use_ema:
            instance.ema = EMAModel(instance.model, decay=instance.config.ema_decay)
        
        return instance


# ════════════════════════════════════════════════════════════════
# Multi-Horizon Ensemble
# ════════════════════════════════════════════════════════════════
class TFTEnsemble:
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        self.models = [TFTSignalModel.load(p) for p in model_paths]
        self.weights = weights if weights is not None else [1.0/len(model_paths)] * len(model_paths)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for m in self.models:
            result = m.predict(df)
            if not result: continue
            preds.append(result['median_pred'][:, 0])

        if not preds: return np.array([])
        
        min_len = min(len(p) for p in preds)
        aligned = [p[-min_len:] for p in preds]

        ensemble = np.zeros(min_len)
        for w, p in zip(self.weights, aligned):
            ensemble += w * p
        return ensemble
