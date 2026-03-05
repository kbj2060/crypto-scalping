"""
Signal Module: Temporal Fusion Transformer for Pure Price Prediction
================================================================================
- 개편 1: 트레이딩 손실 함수(Sharpe, Directional) 제거 -> 순수 Quantile Loss 적용
- 개편 2: Sequence-to-Sequence 다중 스텝 예측 아키텍처 지원
"""
import os, copy, json, math, logging, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 0. UTILS
# ════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ════════════════════════════════════════════════════════════════
# 1. CONFIG (트레이딩 파라미터 제거)
# ════════════════════════════════════════════════════════════════
@dataclass
class TFTConfig:
    input_window: int = 64           
    forecast_horizon: int = 6        
    target_col: str = 'target_ret_1' # [개선 3] 1스텝 수익률을 시퀀스로 묶어 사용
    hidden_size: int = 64            
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.3
    num_features: int = 40  
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]) 
    learning_rate: float = 3e-5
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 20
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    warmup_epochs: int = 20
    lr_scheduler: str = 'cosine'
    min_lr: float = 1e-7
    use_ema: bool = True
    ema_decay: float = 0.999         
    use_amp: bool = True
    accumulation_steps: int = 1
    seed: int = 42
    device: str = 'auto'
    model_dir: str = 'data/tft'
    
    training_noise_std: float = 0.05 
    recent_bias_decay: float = 0.05  

    def __post_init__(self):
        if self.device == 'auto': self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu': self.use_amp = False

# ════════════════════════════════════════════════════════════════
# 2. INTERNAL MODULES
# ════════════════════════════════════════════════════════════════
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
    def forward(self, x): return torch.sigmoid(self.fc1(x)) * self.fc2(x)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None
    
    def forward(self, x):
        residual = self.skip_proj(x) if self.skip_proj else x
        hidden = self.glu(self.dropout(self.fc2(self.elu(self.fc1(x)))))
        return self.layer_norm(residual + hidden)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_vars, self.hidden_size = num_vars, hidden_size
        self.var_grns = nn.ModuleList([GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)])
        self.selection_grn = GatedResidualNetwork(num_vars * hidden_size, hidden_size, num_vars, dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        var_outputs = torch.stack([self.var_grns[i](x[:, :, i:i+1]) for i in range(self.num_vars)], dim=-2)
        flat = var_outputs.reshape(x.shape[0], x.shape[1], self.num_vars * self.hidden_size)
        weights = self.softmax(self.selection_grn(flat))
        return (var_outputs * weights.unsqueeze(-1)).sum(dim=-2), weights

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads, self.d_k = num_heads, hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.shape
        Q = self.W_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None: 
            if mask.dtype in [torch.float16, torch.float32, torch.float64]:
                scores = scores + mask
            else:
                scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
                
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(context), attn_weights.mean(dim=1)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        self.temporal_vsn = VariableSelectionNetwork(config.num_features, H, config.dropout)
        self.gru_momentum = nn.GRU(H, H, config.lstm_layers, batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.gru_reversion = nn.GRU(H, H, config.lstm_layers, batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)

        self.hurst_idx, self.regime_trending_idx = None, None
        self.custom_gate = nn.Linear(H, 1)

        self.post_lstm_gate, self.post_lstm_norm = GatedLinearUnit(H, H), nn.LayerNorm(H)
        self.multihead_attn = InterpretableMultiHeadAttention(H, config.attention_heads, config.dropout)
        self.post_attn_gate, self.post_attn_norm = GatedLinearUnit(H, H), nn.LayerNorm(H)
        self.pos_ff, self.pos_ff_gate, self.pos_ff_norm = GatedResidualNetwork(H, H, H, config.dropout), GatedLinearUnit(H, H), nn.LayerNorm(H)

        self.horizon_fc = nn.Linear(H, config.forecast_horizon * H)
        self.quantile_heads = nn.ModuleList([nn.Linear(H, 1) for _ in config.quantiles])

    def set_feature_indices(self, feature_cols: List[str]):
        if 'hurst_48' in feature_cols: self.hurst_idx = feature_cols.index('hurst_48')
        elif 'regime_trending' in feature_cols: self.regime_trending_idx = feature_cols.index('regime_trending')

    def forward(self, temporal: torch.Tensor):
        B, T, H = temporal.shape[0], temporal.shape[1], self.config.hidden_size
        
        if self.training:
            local_std = temporal.std(dim=1, keepdim=True) + 1e-6
            temporal = temporal + torch.randn_like(temporal) * local_std * self.config.training_noise_std
            time_mask = (torch.rand(B, T, 1, device=temporal.device) > 0.05).float()
            temporal = (temporal * time_mask) / 0.95
        
        selected, var_weights = self.temporal_vsn(temporal)
        
        h0 = torch.zeros(self.config.lstm_layers, B, H, device=temporal.device)
        out_m, _ = self.gru_momentum(selected, h0)
        out_r, _ = self.gru_reversion(selected, h0)

        if self.hurst_idx is not None:
            gate0 = torch.sigmoid(5.0 * temporal[:, -1, self.hurst_idx]).unsqueeze(-1).unsqueeze(-1)
        elif self.regime_trending_idx is not None:
            gate0 = torch.sigmoid(5.0 * temporal[:, -1, self.regime_trending_idx]).unsqueeze(-1).unsqueeze(-1)
        else:
            gate0 = torch.sigmoid(self.custom_gate(selected[:, -1, :])).unsqueeze(-1)

        gru_out = gate0 * out_m + (1.0 - gate0) * out_r
        temporal_feat = self.post_lstm_norm(self.post_lstm_gate(gru_out) + selected)
        
        causal_mask = torch.tril(torch.ones(T, T, device=temporal.device))
        idx = torch.arange(T, device=temporal.device)
        dist = idx.unsqueeze(0) - idx.unsqueeze(1)
        bias = -self.config.recent_bias_decay * dist.float()
        bias = bias.masked_fill(causal_mask == 0, float('-inf')).unsqueeze(0)
        
        attn_out, attn_w = self.multihead_attn(temporal_feat, temporal_feat, temporal_feat, mask=bias)
        
        attn_out = self.post_attn_norm(self.post_attn_gate(attn_out) + temporal_feat)
        ff_out = self.pos_ff_norm(self.pos_ff_gate(self.pos_ff(attn_out)) + attn_out)
        
        horizon_h = self.horizon_fc(ff_out[:, -1, :]).view(B, self.config.forecast_horizon, H)
        return torch.cat([qh(horizon_h) for qh in self.quantile_heads], dim=-1), attn_w, var_weights

# ════════════════════════════════════════════════════════════════
# 3. [개선 1] 순수 Quantile Loss (Pinball Loss) 도입
# ════════════════════════════════════════════════════════════════
class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions: (Batch, Horizon, Quantiles)
        # targets: (Batch, Horizon)
        errors = targets.unsqueeze(-1) - predictions
        q_tensor = torch.tensor(self.quantiles, device=predictions.device, dtype=predictions.dtype).view(1, 1, -1)
        
        # 순수 Pinball Loss (예측 오차에 대한 비대칭 페널티를 통해 분포를 추정)
        loss = torch.max((q_tensor - 1) * errors, q_tensor * errors)
        return loss.mean()

# ════════════════════════════════════════════════════════════════
# 4. WRAPPER & DATASET
# ════════════════════════════════════════════════════════════════
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

class TFTDataset(Dataset):
    def __init__(self, config: TFTConfig, df: pd.DataFrame, feature_cols: List[str]):
        self.config, self.target_col = config, config.target_col
        self.temporal_cols = feature_cols
        self.temporal_data = df[self.temporal_cols].values.astype(np.float32)
        self.target_data = df[self.target_col].values.astype(np.float32)
        self.n_samples = len(df) - (config.input_window + config.forecast_horizon) + 1

    def __len__(self): return max(0, self.n_samples)
    def __getitem__(self, idx):
        t_end = idx + self.config.input_window
        return {
            'temporal': torch.tensor(self.temporal_data[idx:t_end]),
            # [개선 3] 1스텝 타겟을 horizon 길이만큼 슬라이싱하여 다차원(S2S) 벡터 구성
            'target': torch.tensor(self.target_data[t_end:t_end + self.config.forecast_horizon])
        }

class TFTSignalModel:
    def __init__(self, config: TFTConfig = None):
        self.config, self.model = config, TemporalFusionTransformer(config) if config else None
        if self.model: self.model.to(config.device)
        self.ema, self.feature_cols, self.scaler_params, self.target_scaler = None, None, {}, None

    def _create_scheduler(self, optimizer, steps_per_epoch: int):
        total_steps = steps_per_epoch * self.config.max_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        if self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: float(s)/max(1, warmup_steps) if s < warmup_steps else max(self.config.min_lr/self.config.learning_rate, 0.5*(1.0+math.cos(math.pi*float(s-warmup_steps)/max(1, total_steps-warmup_steps)))))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=self.config.min_lr)

    def _save_checkpoint(self, tag: str):
        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'tft_{tag}.pt'))
        meta = {'feature_cols': self.feature_cols, 'scaler_params': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.scaler_params.items()}, 'target_scaler': self.target_scaler, 'config': self.config.__dict__}
        with open(os.path.join(self.config.model_dir, f'tft_{tag}_meta.json'), 'w') as f: json.dump(meta, f, indent=2)

    def _load_checkpoint(self, tag: str):
        path = os.path.join(self.config.model_dir, f'tft_{tag}.pt')
        ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    def fit(self, cfg: TFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], resume_from=None, warm_start_path=None):
        set_seed(cfg.seed)
        cfg.num_features = len(feature_cols)
        self.config, self.feature_cols = cfg, feature_cols
        
        if self.model is None or self.model.config.num_features != cfg.num_features:
            self.model = TemporalFusionTransformer(cfg).to(cfg.device)
        self.model.set_feature_indices(feature_cols)

        target_ckpt_path = resume_from or warm_start_path
        if target_ckpt_path and os.path.exists(target_ckpt_path):
            logger.info(f"사전 가중치 로드 중: {target_ckpt_path}")
            ckpt = torch.load(target_ckpt_path, map_location=cfg.device, weights_only=True)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        
        mean, std = train_df[feature_cols].mean(), train_df[feature_cols].std().replace(0, 1.0)
        self.scaler_params = {'mean': mean.values, 'std': std.values}
        
        target_mean = train_df[cfg.target_col].mean()
        target_std = train_df[cfg.target_col].std()
        if target_std == 0 or np.isnan(target_std): target_std = 1.0
        self.target_scaler = {'mean': target_mean, 'std': target_std}
        
        train_norm, val_norm = train_df.copy(), val_df.copy()
        train_norm[feature_cols], val_norm[feature_cols] = (train_norm[feature_cols] - mean) / std, (val_norm[feature_cols] - mean) / std
        
        train_norm[cfg.target_col] = (train_norm[cfg.target_col] - self.target_scaler['mean']) / self.target_scaler['std']
        val_norm[cfg.target_col] = (val_norm[cfg.target_col] - self.target_scaler['mean']) / self.target_scaler['std']

        train_loader = DataLoader(TFTDataset(cfg, train_norm, feature_cols), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TFTDataset(cfg, val_norm, feature_cols), batch_size=cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = self._create_scheduler(optimizer, max(len(train_loader) // cfg.accumulation_steps, 1))
        
        # [개선 1] 순수 Quantile Loss 사용
        criterion = QuantileLoss(cfg.quantiles)
        
        if cfg.use_ema: self.ema = EMAModel(self.model, decay=cfg.ema_decay)
        scaler = GradScaler('cuda') if cfg.use_amp else None
        start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'learning_rate': []}
        
        median_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2

        for epoch in range(start_epoch, cfg.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            train_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                with autocast('cuda', enabled=cfg.use_amp):
                    preds, _, _ = self.model(batch['temporal'].to(cfg.device))
                    loss = criterion(preds, batch['target'].to(cfg.device))
                    loss = loss / cfg.accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                    if (step + 1) % cfg.accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    loss.backward()
                    if (step + 1) % cfg.accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                if (step + 1) % cfg.accumulation_steps == 0 and self.ema: self.ema.update(self.model)
                epoch_loss += loss.item() * cfg.accumulation_steps
                train_steps += 1

            self.model.eval()
            val_loss, val_mae, total_val = 0.0, 0.0, 0
            if self.ema: self.ema.apply_shadow(self.model)
            with torch.no_grad():
                for batch in val_loader:
                    with autocast('cuda', enabled=cfg.use_amp):
                        preds, _, _ = self.model(batch['temporal'].to(cfg.device))
                        target_dev = batch['target'].to(cfg.device)
                        v_loss = criterion(preds, target_dev)
                    val_loss += v_loss.item()
                    
                    # MAE 기록용 (중간값 0.5 기준)
                    val_mae += F.l1_loss(preds[:, :, median_idx], target_dev).item()
                    total_val += 1
                    
            if self.ema: self.ema.restore(self.model)

            avg_train_loss = epoch_loss / max(train_steps, 1)
            val_loss /= max(total_val, 1)
            val_mae /= max(total_val, 1)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            logger.info(f"Epoch {epoch+1:03d}/{cfg.max_epochs} | Train Q-Loss: {avg_train_loss:.4f} | Val Q-Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
                self._save_checkpoint('best')
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("🛑 Early Stopping Triggered.")
                    break
                
        self._load_checkpoint('best')
        if self.ema: self.ema.apply_shadow(self.model) 
        return history

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.model.eval()
        df_norm = df.copy()
        df_norm[self.feature_cols] = (df_norm[self.feature_cols] - self.scaler_params['mean']) / self.scaler_params['std']
        if self.config.target_col not in df.columns: df_norm[self.config.target_col] = 0.0
        
        all_preds, all_attn, all_vars = [], [], []
        for batch in DataLoader(TFTDataset(self.config, df_norm, self.feature_cols), batch_size=self.config.batch_size, shuffle=False):
            with autocast('cuda', enabled=self.config.use_amp):
                preds, attn, var_w = self.model(batch['temporal'].to(self.config.device))
            all_preds.append(preds.cpu().numpy())
            all_attn.append(attn.cpu().numpy())
            all_vars.append(var_w.cpu().numpy())
            
        if not all_preds: return {}
        preds_arr = np.concatenate(all_preds, axis=0)
        
        if self.target_scaler is not None:
            preds_arr = (preds_arr * self.target_scaler['std']) + self.target_scaler['mean']
            
        spread = np.clip(
            preds_arr[:, :, self.config.quantiles.index(max(self.config.quantiles))] - 
            preds_arr[:, :, self.config.quantiles.index(min(self.config.quantiles))], 
            1e-6, 10.0
        )
        
        return {
            'quantiles': preds_arr,
            'median_pred': preds_arr[:, :, self.config.quantiles.index(0.5) if 0.5 in self.config.quantiles else len(self.config.quantiles) // 2],
            'attention': np.concatenate(all_attn, axis=0),
            'variable_importance': np.concatenate(all_vars, axis=0),
            'confidence': 1.0 / (1.0 + spread / (spread.std() + 1e-6))
        }

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(os.path.dirname(path), f"tft_{os.path.basename(path).replace('tft_', '').replace('.pt', '')}_meta.json"), 'r') as f: 
            meta = json.load(f)
            
        cfg = TFTConfig()
        for k, v in meta['config'].items(): setattr(cfg, k, v)
        cfg.__post_init__() 
        instance = cls(cfg)
        
        ckpt = torch.load(path, map_location=cfg.device, weights_only=True)
        instance.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        
        instance.feature_cols = meta['feature_cols']
        instance.scaler_params = {k: np.array(v) if isinstance(v, list) else v for k, v in meta['scaler_params'].items()}
        instance.target_scaler = meta.get('target_scaler', {'mean': 0.0, 'std': 1.0})
        
        if cfg.use_ema: instance.ema = EMAModel(instance.model, decay=cfg.ema_decay)
        return instance