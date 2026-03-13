"""
Signal Module: TimeMachine (Mamba-based SSM) for Long-Term Forecasting
================================================================================
- 논문 반영: Quadruple-Mamba (Temporal -> Channel -> Temporal -> Channel) 아키텍처
- 특징: 트랜스포머의 O(N^2) 연산량을 O(N)으로 압축하여 초장기 시계열 패턴 인식
- 호환성: 기존 TFT 모델과 동일한 입출력 인터페이스 (Quantile Loss 적용)

Val Dir 개선 핵심:
1. Dropout 정규화 (SSM 블록 + 예측 헤드)
2. QuantileLoss + DirectionalLoss 혼합 → 방향성을 직접 학습
3. 예측 헤드 구조 개선 (GELU 음수 억제 제거, 다층 헤드)
4. 전체 시퀀스 풀링 (마지막 K스텝 한정 제거)
5. 학습 시 더 강한 augmentation
"""
import os, json, math, logging, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 0. UTILS & CONFIG
# ════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class TimeMachineConfig:
    input_window: int = 64           
    forecast_horizon: int = 6        
    target_col: str = 'target_ret_1' 
    
    hidden_size: int = 128            
    d_state: int = 32                
    d_conv: int = 4                  
    expand: int = 2                  
    
    num_features: int = 35
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]) 
    
    learning_rate: float = 1e-4
    batch_size: int = 128            
    max_epochs: int = 500
    patience: int = 50
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    warmup_epochs: int = 10
    lr_scheduler: str = 'cosine'
    min_lr: float = 1e-7
    use_ema: bool = True
    ema_decay: float = 0.999         
    use_amp: bool = True
    accumulation_steps: int = 2      
    seed: int = 42
    device: str = 'auto'
    model_dir: str = 'data/timemachine'
    
    # ── 정규화 하이퍼파라미터 (Val Dir 개선 핵심) ──
    training_noise_std: float = 0.1   # 0.05 → 0.1 (더 강한 augmentation)
    dropout: float = 0.15             # 신규: Dropout 비율
    dir_loss_weight: float = 0.3      # 신규: DirectionalLoss 비중 (0=off, 1=only dir)

    def __post_init__(self):
        if self.device == 'auto': self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu': self.use_amp = False


# ════════════════════════════════════════════════════════════════
# 1. CORE ARCHITECTURE
# ════════════════════════════════════════════════════════════════

class PureMambaBlock(nn.Module):
    """Mamba Selective SSM - PyTorch Native
    
    [개선] Dropout 추가: out_proj 이전에 적용하여 과적합 방지
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, bias=True,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.activation = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.B_proj = nn.Linear(d_state, self.d_inner, bias=False)
        self.C_proj = nn.Linear(d_state, self.d_inner, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # [신규]
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_softplus = torch.log(torch.expm1(dt_init))
            self.dt_proj.bias.data.copy_(inv_softplus)
            nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
            nn.init.xavier_uniform_(self.B_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.C_proj.weight, gain=0.5)

    def _selective_scan(self, x, dt, B_gate, C_gate):
        """EMA-style recurrence: h_t = decay * h_{t-1} + (1-decay) * inp_t"""
        B, L, D = x.shape
        decay = torch.exp(-dt)
        one_minus_decay = 1.0 - decay
        inp = x * B_gate
        
        chunk_size = 64
        n_chunks = (L + chunk_size - 1) // chunk_size
        outputs = []
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        for c in range(n_chunks):
            s = c * chunk_size
            e = min(s + chunk_size, L)
            d_c = decay[:, s:e, :]
            omd_c = one_minus_decay[:, s:e, :]
            inp_c = inp[:, s:e, :]
            C_c = C_gate[:, s:e, :]
            
            chunk_out = []
            for t in range(e - s):
                h = d_c[:, t, :] * h + omd_c[:, t, :] * inp_c[:, t, :]
                chunk_out.append(h)
            
            chunk_out = torch.stack(chunk_out, dim=1)
            outputs.append(chunk_out * C_c)
        
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        x_in = x_in.transpose(1, 2) 
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = x_in.transpose(1, 2)
        x_in = self.activation(x_in)
        
        x_proj = self.x_proj(x_in)
        dt_raw, B_mat, C_mat = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=4.0)
        
        B_gate = torch.sigmoid(self.B_proj(B_mat))
        C_gate = torch.sigmoid(self.C_proj(C_mat))
        
        orig_dtype = x_in.dtype
        y = self._selective_scan(
            x_in.float(), dt.float(), B_gate.float(), C_gate.float()
        ).to(orig_dtype)
        
        y = y * self.activation(z)
        y = self.dropout(y)  # [신규] Dropout before out_proj
        return self.out_proj(y) + residual


class QuadrupleMamba(nn.Module):
    def __init__(self, config: TimeMachineConfig):
        super().__init__()
        dp = config.dropout
        self.t_mamba_1 = PureMambaBlock(config.hidden_size, config.d_state, config.d_conv, config.expand, dp)
        self.t_mamba_2 = PureMambaBlock(config.hidden_size, config.d_state, config.d_conv, config.expand, dp)
        self.c_mamba_1 = PureMambaBlock(config.input_window, config.d_state, config.d_conv, expand=1, dropout=dp)
        self.c_mamba_2 = PureMambaBlock(config.input_window, config.d_state, config.d_conv, expand=1, dropout=dp)
        
        self.norm_t1 = nn.LayerNorm(config.hidden_size)
        self.norm_c1 = nn.LayerNorm(config.hidden_size)
        self.norm_t2 = nn.LayerNorm(config.hidden_size)
        self.norm_final = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.norm_t1(self.t_mamba_1(x))
        x = x.transpose(1, 2)
        x = self.c_mamba_1(x)
        x = x.transpose(1, 2)
        x = self.norm_c1(x)
        
        x = self.norm_t2(self.t_mamba_2(x))
        x = x.transpose(1, 2)
        x = self.c_mamba_2(x)
        x = x.transpose(1, 2)
        return self.norm_final(x)


class TimeMachineModel(nn.Module):
    def __init__(self, config: TimeMachineConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size
        
        self.feature_embed = nn.Sequential(
            nn.Linear(config.num_features, H),
            nn.LayerNorm(H),
            nn.Dropout(config.dropout),  # [신규]
        )
        self.time_machine = QuadrupleMamba(config)
        
        # [개선] 전체 시퀀스 풀링 (마지막 K스텝 한정 제거)
        self.pool_attn = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.Tanh(),
            nn.Linear(H // 4, 1),
        )
        
        # [개선] 예측 헤드: GELU 제거 → Dropout + 선형으로 음수 예측력 보존
        self.horizon_fc = nn.Sequential(
            nn.Linear(H, config.forecast_horizon * H),
            nn.Dropout(config.dropout),  # [신규]
        )
        
        # Quantile crossing 방지
        self.median_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(H // 2, 1),
        )
        n_upper = sum(1 for q in config.quantiles if q > 0.5)
        n_lower = sum(1 for q in config.quantiles if q < 0.5)
        self.upper_head = nn.Linear(H, n_upper) if n_upper > 0 else None
        self.lower_head = nn.Linear(H, n_lower) if n_lower > 0 else None

    def _attention_pool(self, x):
        """전체 시퀀스에서 어텐션 풀링 [B, T, H] → [B, H], [B, T]"""
        attn_scores = self.pool_attn(x).squeeze(-1)   # [B, T]
        attn_w = F.softmax(attn_scores, dim=-1)        # [B, T]
        pooled = torch.einsum('bt,bth->bh', attn_w, x) # [B, H]
        return pooled, attn_w

    def forward(self, temporal: torch.Tensor):
        B, T, _ = temporal.shape
        
        # [개선] 학습 시 augmentation 강화
        if self.training:
            local_std = temporal.std(dim=1, keepdim=True).clamp(min=1e-6)
            temporal = temporal + torch.randn_like(temporal) * local_std * self.config.training_noise_std
            
            # [신규] 시간축 마스킹 (일부 시점을 0으로): 과거 특정 시점에 과의존 방지
            if self.config.dropout > 0:
                time_mask = (torch.rand(B, T, 1, device=temporal.device) > self.config.dropout * 0.5).float()
                temporal = temporal * time_mask
            
        x = self.feature_embed(temporal)
        x = self.time_machine(x)
        
        context, attn_w = self._attention_pool(x)
        
        H = self.config.hidden_size
        horizon_h = self.horizon_fc(context).view(B, self.config.forecast_horizon, H)
        
        # Quantile crossing 방지
        median = self.median_head(horizon_h)
        
        parts = []
        if self.lower_head is not None:
            lo = F.softplus(self.lower_head(horizon_h))
            lo = torch.cumsum(lo.flip(dims=[-1]), dim=-1).flip(dims=[-1])
            parts.append(median - lo)
        parts.append(median)
        if self.upper_head is not None:
            up = F.softplus(self.upper_head(horizon_h))
            up = torch.cumsum(up, dim=-1)
            parts.append(median + up)
        
        out_quantiles = torch.cat(parts, dim=-1)
        
        # variable importance
        with torch.no_grad():
            w = self.feature_embed[0].weight
            vi = w.norm(dim=0)
            vi = vi / (vi.sum() + 1e-8)
            vi = vi.unsqueeze(0).expand(B, -1)
        
        return out_quantiles, attn_w, vi


# ════════════════════════════════════════════════════════════════
# 2. LOSS, WRAPPER & DATASET
# ════════════════════════════════════════════════════════════════

class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.register_buffer('q_t', torch.tensor(quantiles, dtype=torch.float32))
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        mask = ~torch.isnan(targets)
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        targets_clean = targets.masked_fill(~mask, 0.0)
        errors = targets_clean.unsqueeze(-1) - predictions
        q = self.q_t.to(device=predictions.device, dtype=predictions.dtype).view(1, 1, -1)
        loss = torch.max((q - 1) * errors, q * errors)
        loss = loss * mask.unsqueeze(-1).float()
        return loss.sum() / (mask.sum().clamp(min=1) * len(self.quantiles))


class DirectionalLoss(nn.Module):
    """[신규] 방향성 직접 학습 Loss
    
    중앙값 예측의 부호가 실제 타겟의 부호와 일치하도록 학습.
    Binary Cross Entropy를 사용하여 P(up) vs P(down)을 학습.
    """
    def __init__(self, median_idx: int):
        super().__init__()
        self.median_idx = median_idx
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        mask = ~torch.isnan(targets)
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # 중앙값 예측
        pred_med = predictions[:, :, self.median_idx]  # [B, F]
        
        # 실제 방향 (1=up, 0=down)
        target_dir = (targets > 0).float()
        target_dir = target_dir.masked_fill(~mask, 0.0)
        
        # logits = pred * scale (sigmoid 적용 없이 BCEWithLogitsLoss 사용 → AMP 안전)
        logits = pred_med * 5.0
        
        # BCEWithLogitsLoss: 내부에서 sigmoid + BCE를 수치적으로 안정하게 결합
        bce = F.binary_cross_entropy_with_logits(logits, target_dir, reduction='none')
        bce = bce * mask.float()
        return bce.sum() / mask.sum().clamp(min=1)


class CombinedLoss(nn.Module):
    """[신규] QuantileLoss + DirectionalLoss 혼합
    
    dir_weight=0이면 순수 QuantileLoss (기존과 동일)
    dir_weight=0.3이면 QLoss 70% + DirLoss 30% 혼합
    """
    def __init__(self, quantiles: List[float], dir_weight: float = 0.3):
        super().__init__()
        self.q_loss = QuantileLoss(quantiles)
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        self.d_loss = DirectionalLoss(median_idx)
        self.dir_weight = dir_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        ql = self.q_loss(predictions, targets)
        dl = self.d_loss(predictions, targets)
        return (1.0 - self.dir_weight) * ql + self.dir_weight * dl


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay, self.shadow, self.backup = decay, {}, {}
        for name, param in model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup: param.data.copy_(self.backup[name])
        self.backup = {}


class TimeMachineDataset(Dataset):
    def __init__(self, config: TimeMachineConfig, df: pd.DataFrame, feature_cols: List[str]):
        self.config, self.target_col = config, config.target_col
        self.temporal_cols = feature_cols
        self.temporal_data = np.nan_to_num(
            df[self.temporal_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.target_data = np.nan_to_num(
            df[self.target_col].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.n_samples = len(df) - (config.input_window + config.forecast_horizon) + 1

    def __len__(self): return max(0, self.n_samples)

    def __getitem__(self, idx):
        t_end = idx + self.config.input_window
        return {
            'temporal': torch.tensor(self.temporal_data[idx:t_end]),
            'target': torch.tensor(self.target_data[t_end:t_end + self.config.forecast_horizon])
        }


class TimeMachineSignalModel:
    def __init__(self, config: TimeMachineConfig = None):
        self.config, self.model = config, TimeMachineModel(config) if config else None
        if self.model: self.model.to(config.device)
        self.ema, self.feature_cols, self.scaler_params, self.target_scaler = None, None, {}, None

    def _create_scheduler(self, optimizer, steps_per_epoch: int):
        total_steps = steps_per_epoch * self.config.max_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        if self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: (
                    float(s) / max(1, warmup_steps) if s < warmup_steps
                    else max(
                        self.config.min_lr / self.config.learning_rate,
                        0.5 * (1.0 + math.cos(
                            math.pi * float(s - warmup_steps) / max(1, total_steps - warmup_steps)
                        ))
                    )
                )
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=self.config.min_lr
        )

    def _save_checkpoint(self, tag: str):
        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'timemachine_{tag}.pt'))
        meta = {
            'feature_cols': self.feature_cols,
            'scaler_params': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.scaler_params.items()},
            'target_scaler': self.target_scaler,
            'config': self.config.__dict__
        }
        with open(os.path.join(self.config.model_dir, f'timemachine_{tag}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_checkpoint(self, tag: str):
        path = os.path.join(self.config.model_dir, f'timemachine_{tag}.pt')
        ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    def fit(self, cfg: TimeMachineConfig, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feature_cols: List[str], resume_from=None):
        set_seed(cfg.seed)
        cfg.num_features = len(feature_cols)
        self.config, self.feature_cols = cfg, feature_cols
        
        if self.model is None or self.model.config.num_features != cfg.num_features:
            self.model = TimeMachineModel(cfg).to(cfg.device)

        if resume_from and os.path.exists(resume_from):
            logger.info(f"사전 가중치 로드 중: {resume_from}")
            ckpt = torch.load(resume_from, map_location=cfg.device, weights_only=True)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        
        # ── 정규화 (RobustScaler 방식: IQR 기반) ──
        # [개선] 기존 mean/std 대신 median/IQR → 극단값에 강건
        feat_data = train_df[feature_cols]
        median_vals = feat_data.median().fillna(0.0).replace([np.inf, -np.inf], 0.0)
        q75 = feat_data.quantile(0.75).fillna(1.0)
        q25 = feat_data.quantile(0.25).fillna(0.0)
        iqr = (q75 - q25).replace(0, 1.0).fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        self.scaler_params = {'mean': median_vals.values, 'std': iqr.values}
        
        target_mean = train_df[cfg.target_col].median()
        target_iqr = train_df[cfg.target_col].quantile(0.75) - train_df[cfg.target_col].quantile(0.25)
        if target_iqr == 0 or np.isnan(target_iqr) or np.isinf(target_iqr): target_iqr = 1.0
        if np.isnan(target_mean) or np.isinf(target_mean): target_mean = 0.0
        self.target_scaler = {'mean': float(target_mean), 'std': float(target_iqr)}
        
        train_norm, val_norm = train_df.copy(), val_df.copy()
        for dn in [train_norm, val_norm]:
            dn[feature_cols] = ((dn[feature_cols] - median_vals) / iqr).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
            dn[cfg.target_col] = (
                (dn[cfg.target_col] - self.target_scaler['mean']) / self.target_scaler['std']
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)

        feat_range = train_norm[feature_cols]
        logger.info(f"  정규화 후 피처 범위: [{feat_range.min().min():.2f}, {feat_range.max().max():.2f}]")
        logger.info(f"  정규화 후 피처 std 평균: {feat_range.std().mean():.3f}")

        train_loader = DataLoader(
            TimeMachineDataset(cfg, train_norm, feature_cols),
            batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TimeMachineDataset(cfg, val_norm, feature_cols),
            batch_size=cfg.batch_size, shuffle=False
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = self._create_scheduler(optimizer, max(len(train_loader) // cfg.accumulation_steps, 1))
        
        # [핵심 변경] CombinedLoss 사용 (Quantile + Directional)
        criterion = CombinedLoss(cfg.quantiles, dir_weight=cfg.dir_loss_weight)
        
        if cfg.use_ema: self.ema = EMAModel(self.model, decay=cfg.ema_decay)
        scaler = GradScaler('cuda') if (cfg.use_amp and cfg.device == 'cuda') else None
        
        best_val_loss, patience_counter = float('inf'), 0
        best_val_dir = 0.0  # [신규] 최고 Val Dir 추적
        val_rise_cnt, prev_val_loss = 0, float('inf')
        nan_epoch_count = 0
        
        history = {'train_loss': [], 'val_loss': [], 'val_dir': [], 'learning_rate': []}
        median_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2

        for epoch in range(cfg.max_epochs):
            in_warmup = (epoch < cfg.warmup_epochs)
            self.model.train()
            tl = tdc = tt = nan_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                with autocast(cfg.device, enabled=cfg.use_amp):
                    preds, _, _ = self.model(batch['temporal'].to(cfg.device))
                    target_dev = batch['target'].to(cfg.device)
                    loss = criterion(preds, target_dev) / cfg.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_steps += 1
                    optimizer.zero_grad()
                    continue
                
                if scaler:
                    scaler.scale(loss).backward()
                    if (step + 1) % cfg.accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        sb = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scaler.get_scale() == sb: scheduler.step()
                else:
                    loss.backward()
                    if (step + 1) % cfg.accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                if (step + 1) % cfg.accumulation_steps == 0 and self.ema:
                    self.ema.update(self.model)
                
                with torch.no_grad():
                    pred_med = preds[:, :, median_idx]
                    mask = ~torch.isnan(target_dev)
                    if mask.any():
                        correct = ((pred_med > 0) == (target_dev > 0)) & mask
                        tdc += correct.sum().item()
                        tt += mask.sum().item()
                tl += loss.item() * cfg.accumulation_steps

            total_steps = step + 1
            if nan_steps > 0:
                logger.warning(f"  ⚠️ Epoch {epoch+1}: NaN {nan_steps}/{total_steps} steps")
            if nan_steps == total_steps:
                nan_epoch_count += 1
                if nan_epoch_count >= 3:
                    logger.error("  🛑 3 에폭 연속 전체 NaN → 학습 중단")
                    break
            else:
                nan_epoch_count = 0

            # ── Validation ──
            self.model.eval()
            vl = vdc = vt = 0
            if self.ema: self.ema.apply_shadow(self.model)
            
            with torch.no_grad():
                for batch in val_loader:
                    with autocast(cfg.device, enabled=cfg.use_amp):
                        preds, _, _ = self.model(batch['temporal'].to(cfg.device))
                        target_dev = batch['target'].to(cfg.device)
                        v_loss = criterion(preds, target_dev)
                    
                    if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                        pred_med = preds[:, :, median_idx]
                        mask = ~torch.isnan(target_dev)
                        if mask.any():
                            correct = ((pred_med > 0) == (target_dev > 0)) & mask
                            vdc += correct.sum().item()
                            vt += mask.sum().item()
                            vl += v_loss.item() * mask.sum().item()
            
            avg_tl = tl / max((tt // cfg.forecast_horizon), 1) if tt > 0 else float('nan')
            ta = tdc / max(tt, 1)
            avg_vl = vl / max(vt, 1) if vt > 0 else float('nan')
            va = vdc / max(vt, 1)

            history['train_loss'].append(avg_tl if not math.isnan(avg_tl) else 0.0)
            history['val_loss'].append(avg_vl if not math.isnan(avg_vl) else 0.0)
            history['val_dir'].append(va)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            wtag = " [WARMUP]" if in_warmup else ""
            logger.info(
                f"Epoch {epoch+1:03d}/{cfg.max_epochs}{wtag} | "
                f"Train: {avg_tl:.4f} ({ta:.1%}) | "
                f"Val: {avg_vl:.4f} ({va:.1%}) | "
                f"AccGap: {ta-va:+.1%} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            if in_warmup:
                if epoch == cfg.warmup_epochs - 1:
                    best_val_loss = avg_vl if not math.isnan(avg_vl) else float('inf')
                    best_val_dir = va
                    prev_val_loss = best_val_loss
                    val_rise_cnt = 0
                    self._save_checkpoint('best')
                    logger.info(f"  ✅ [Warmup 완료] Best 저장 (Val: {avg_vl:.4f}, Dir: {va:.1%})")
                if self.ema: self.ema.restore(self.model)
                continue

            if math.isnan(avg_vl):
                if self.ema: self.ema.restore(self.model)
                continue

            val_rise_cnt = val_rise_cnt + 1 if avg_vl > prev_val_loss else 0
            prev_val_loss = avg_vl
            if val_rise_cnt >= 7:
                logger.warning(f"  🚨 Val Loss {val_rise_cnt}연속 상승")
                val_rise_cnt = 0

            # [개선] 복합 기준: Val Loss 개선 OR Val Dir 최고 갱신
            improved = False
            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                improved = True
            if va > best_val_dir + 0.002:  # 0.2%p 이상 개선
                best_val_dir = va
                improved = True
            
            if improved:
                patience_counter = 0
                self._save_checkpoint('best')
                logger.info(f"  🌟 [New Best] Val: {avg_vl:.4f} (Dir: {va:.1%})")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️ Patience: {patience_counter}/{cfg.patience}")
                
            if self.ema: self.ema.restore(self.model)
            if patience_counter >= cfg.patience:
                logger.info("🛑 Early Stopping.")
                break
                
        logger.info("✅ 학습 종료. Best 모델 롤백.")
        self._load_checkpoint('best')
        if self.ema: self.ema.apply_shadow(self.model) 
        return history

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.model.eval()
        df_norm = df.copy()
        df_norm[self.feature_cols] = (
            (df_norm[self.feature_cols] - self.scaler_params['mean']) / self.scaler_params['std']
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
        
        if self.config.target_col not in df.columns:
            df_norm[self.config.target_col] = 0.0
        
        all_preds, all_attn, all_vars = [], [], []
        for batch in DataLoader(
            TimeMachineDataset(self.config, df_norm, self.feature_cols),
            batch_size=self.config.batch_size, shuffle=False
        ):
            with autocast(self.config.device, enabled=self.config.use_amp):
                preds, attn, var_w = self.model(batch['temporal'].to(self.config.device))
            all_preds.append(preds.float().cpu().numpy())
            all_attn.append(attn.float().cpu().numpy())
            all_vars.append(var_w.float().cpu().numpy())
            
        if not all_preds: return {}
        preds_arr = np.concatenate(all_preds, axis=0)
        
        if self.target_scaler is not None:
            preds_arr = (preds_arr * self.target_scaler['std']) + self.target_scaler['mean']
            
        spread = np.clip(preds_arr[:, :, -1] - preds_arr[:, :, 0], 1e-6, 10.0)
        
        mid = self.config.quantiles.index(0.5) if 0.5 in self.config.quantiles else len(self.config.quantiles) // 2
        return {
            'quantiles': preds_arr,
            'median_pred': preds_arr[:, :, mid],
            'attention': np.concatenate(all_attn, axis=0),
            'variable_importance': np.concatenate(all_vars, axis=0),
            'confidence': 1.0 / (1.0 + spread / (spread.std() + 1e-6))
        }

    @classmethod
    def load(cls, path: str):
        meta_path = os.path.join(
            os.path.dirname(path),
            f"timemachine_{os.path.basename(path).replace('timemachine_', '').replace('.pt', '')}_meta.json"
        )
        with open(meta_path, 'r') as f: 
            meta = json.load(f)
            
        cfg = TimeMachineConfig()
        for k, v in meta['config'].items(): setattr(cfg, k, v)
        cfg.__post_init__() 
        instance = cls(cfg)
        
        ckpt = torch.load(path, map_location=cfg.device, weights_only=True)
        instance.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        
        instance.feature_cols = meta['feature_cols']
        instance.scaler_params = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in meta['scaler_params'].items()
        }
        instance.target_scaler = meta.get('target_scaler', {'mean': 0.0, 'std': 1.0})
        
        if cfg.use_ema: instance.ema = EMAModel(instance.model, decay=cfg.ema_decay)
        return instance