"""
Signal Module: Temporal Fusion Transformer for 5-min ETH Day Trading (Standalone & Bug Fixed)
"""
import os, copy, json, math, logging, random
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

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════
@dataclass
class TFTConfig:
    input_window: int = 64           
    forecast_horizon: int = 6        
    target_col: str = 'target_ret_6'
    hidden_size: int = 32
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.3
    num_features: int = 35
    num_static_features: int = 3
    num_temporal_features: int = field(init=False)
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])  
    learning_rate: float = 3e-5
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 100               
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    direction_loss_weight: float = 8.0   
    large_move_weight: float = 6.0       
    warmup_epochs: int = 20
    lr_scheduler: str = 'cosine'     
    min_lr: float = 1e-7
    use_ema: bool = True
    ema_decay: float = 0.999         
    use_amp: bool = True
    accumulation_steps: int = 1
    seed: int = 42
    log_dir: str = 'logs/tensorboard'
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10
    use_swa: bool = True
    swa_start_epoch: int = 300
    swa_lr: float = 5e-5
    device: str = 'auto'
    model_dir: str = 'data/tft'

    def __post_init__(self):
        self.num_temporal_features = self.num_features - self.num_static_features
        if self.device == 'auto': self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu': self.use_amp = False

# ════════════════════════════════════════════════════════════════
# 2. INTERNAL MODULES (🔴 [Fix] 외부 모듈 의존성 제거 완벽 복원)
# ════════════════════════════════════════════════════════════════
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1, context_size: int = None):
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
        if self.context_fc and context is not None: hidden = hidden + self.context_fc(context)
        hidden = self.glu(self.dropout(self.fc2(self.elu(hidden))))
        return self.layer_norm(residual + hidden)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars: int, hidden_size: int, dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.num_vars, self.hidden_size = num_vars, hidden_size
        self.var_grns = nn.ModuleList([GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)])
        self.selection_grn = GatedResidualNetwork(num_vars * hidden_size, hidden_size, num_vars, dropout, context_size=context_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context=None):
        has_time = x.dim() == 3
        var_outputs = torch.stack([self.var_grns[i](x[:, :, i:i+1] if has_time else x[:, i:i+1]) for i in range(self.num_vars)], dim=-2)
        
        flat = var_outputs.reshape(x.shape[0], x.shape[1], self.num_vars * self.hidden_size) if has_time else var_outputs.reshape(x.shape[0], self.num_vars * self.hidden_size)
        weights = self.softmax(self.selection_grn(flat, context.unsqueeze(1).expand(-1, x.shape[1], -1) if has_time and context is not None else context))
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
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, min_val)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, T, -1)
        return self.W_o(context), attn_weights.mean(dim=1)

# ════════════════════════════════════════════════════════════════
# 3. TFT CORE MODEL
# ════════════════════════════════════════════════════════════════
class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        self.temporal_vsn = VariableSelectionNetwork(config.num_temporal_features, H, config.dropout, context_size=H)
        self.static_encoder = nn.Sequential(nn.Linear(config.num_static_features, H), nn.ReLU(), nn.Linear(H, H))
        self.static_context_enrichment = GatedResidualNetwork(H, H, H, config.dropout)
        self.static_context_state_h = GatedResidualNetwork(H, H, H, config.dropout)

        self.gru_momentum = nn.GRU(H, H, config.lstm_layers, batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
        self.gru_reversion = nn.GRU(H, H, config.lstm_layers, batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)

        self.hurst_idx, self.regime_trending_idx = None, None
        self.post_lstm_gate, self.post_lstm_norm = GatedLinearUnit(H, H), nn.LayerNorm(H)

        self.static_enrichment = GatedResidualNetwork(H, H, H, config.dropout, context_size=H)
        self.multihead_attn = InterpretableMultiHeadAttention(H, config.attention_heads, config.dropout)
        self.post_attn_gate, self.post_attn_norm = GatedLinearUnit(H, H), nn.LayerNorm(H)
        self.pos_ff, self.pos_ff_gate, self.pos_ff_norm = GatedResidualNetwork(H, H, H, config.dropout), GatedLinearUnit(H, H), nn.LayerNorm(H)

        self.horizon_fc = nn.Linear(H, config.forecast_horizon * H)
        self.quantile_heads = nn.ModuleList([nn.Linear(H, 1) for _ in config.quantiles])

    def set_feature_indices(self, feature_cols: List[str]):
        if 'hurst_48' in feature_cols: self.hurst_idx = feature_cols.index('hurst_48')
        elif 'regime_trending' in feature_cols: self.regime_trending_idx = feature_cols.index('regime_trending')

    def forward(self, temporal: torch.Tensor, static: torch.Tensor):
        B, T, H = temporal.shape[0], temporal.shape[1], self.config.hidden_size
        
        if self.training:
            local_std = temporal.std(dim=1, keepdim=True) + 1e-6
            temporal = temporal + torch.randn_like(temporal) * local_std * 0.05
            time_mask = (torch.rand(B, T, 1, device=temporal.device) > 0.05).float()
            temporal = (temporal * time_mask) / 0.95
        
        static_emb = self.static_encoder(static)
        cs_e, cs_h = self.static_context_enrichment(static_emb), self.static_context_state_h(static_emb)
        selected, var_weights = self.temporal_vsn(temporal, cs_e)
        h0 = cs_h.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()
        out_m, _ = self.gru_momentum(selected, h0)
        out_r, _ = self.gru_reversion(selected, h0)

        if self.hurst_idx is not None:
            momentum_gate = torch.sigmoid(5.0 * temporal[:, -1, self.hurst_idx])
            gate0 = momentum_gate.unsqueeze(-1).unsqueeze(-1)
        elif self.regime_trending_idx is not None:
            gate0 = temporal[:, -1, self.regime_trending_idx].unsqueeze(-1).unsqueeze(-1)
        else:
            gate0 = torch.ones(B, 1, 1, device=temporal.device) * 0.5

        gru_out = gate0 * out_m + (1.0 - gate0) * out_r
        temporal_feat = self.post_lstm_norm(self.post_lstm_gate(gru_out) + selected)
        enriched = self.static_enrichment(temporal_feat, cs_e.unsqueeze(1).expand(-1, T, -1))
        
        attn_out, attn_w = self.multihead_attn(enriched, enriched, enriched, mask=torch.tril(torch.ones(T, T, device=temporal.device)).unsqueeze(0))
        
        # 🔴 [Fix] 불필요한 중복 연산(computational graph duplication) 제거
        attn_out = self.post_attn_norm(self.post_attn_gate(attn_out) + enriched)
        ff_out = self.pos_ff(attn_out)
        ff_out = self.pos_ff_norm(self.pos_ff_gate(ff_out) + attn_out)
        
        horizon_h = self.horizon_fc(ff_out[:, -1, :]).view(B, self.config.forecast_horizon, H)
        return torch.cat([qh(horizon_h) for qh in self.quantile_heads], dim=-1), attn_w, var_weights

class DirectionalQuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float], direction_weight: float = 5.0, large_move_weight: float = 5.0, sharpe_weight: float = 0.5):
        super().__init__()
        self.quantiles, self.direction_weight, self.large_move_weight, self.sharpe_weight = quantiles, direction_weight, large_move_weight, sharpe_weight
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        errors = targets.unsqueeze(-1).expand_as(predictions) - predictions
        q_tensor = torch.tensor(self.quantiles, device=predictions.device, dtype=predictions.dtype).unsqueeze(0).unsqueeze(0)
        quantile_loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        
        move_size = torch.abs(targets)
        weights = 1.0 + (self.large_move_weight - 1.0) * (move_size.unsqueeze(-1) > move_size.median()).float()
        weighted_ql = (quantile_loss * weights).mean()
        
        pred_median = predictions[:, :, self.median_idx]
        actual_sign = torch.sign(targets)
        
        is_bearish = (targets < 0).float()
        asymmetric_multiplier = 1.0 + is_bearish * 1.0 
        wrong_dir_penalty = torch.relu(-pred_median * actual_sign) * move_size * asymmetric_multiplier
        
        # 🔴 [Fix] 과도한 스케일링(* 100.0) 제거하여 Quantile Loss 보존
        direction_loss = wrong_dir_penalty.mean() 
        
        soft_position = torch.tanh(pred_median * 50.0)
        simulated_returns = soft_position * targets
        sharpe_loss = -(simulated_returns.mean() / torch.sqrt(torch.clamp(simulated_returns.var(unbiased=False), min=1e-8))) * self.sharpe_weight
        
        return weighted_ql + (self.direction_weight * direction_loss) + sharpe_loss, {
            'quantile_loss': weighted_ql.item(), 'direction_loss': direction_loss.item(), 'direction_accuracy': (torch.sign(pred_median) == actual_sign).float().mean().item()}

# ════════════════════════════════════════════════════════════════
# 4. WRAPPER & DATASET (🔴 [Fix] 파일 누락 없이 전체 유지)
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

    def state_dict(self): return {'shadow': self.shadow, 'decay': self.decay}
    def load_state_dict(self, state_dict): self.shadow, self.decay = state_dict['shadow'], state_dict['decay']

class TFTDataset(Dataset):
    def __init__(self, config: TFTConfig, df: pd.DataFrame, feature_cols: List[str]):
        self.config, self.target_col = config, config.target_col
        self.static_cols = ['session_asia', 'session_europe', 'session_us']
        self.temporal_cols = [c for c in feature_cols if c not in self.static_cols]
        self.temporal_data = df[self.temporal_cols].values.astype(np.float32)
        self.static_data = df[self.static_cols].values.astype(np.float32)
        self.target_data = df[self.target_col].values.astype(np.float32)
        self.n_samples = len(df) - (config.input_window + config.forecast_horizon) + 1

    def __len__(self): return max(0, self.n_samples)
    def __getitem__(self, idx):
        t_end = idx + self.config.input_window
        return {
            'temporal': torch.tensor(self.temporal_data[idx:t_end]),
            'static': torch.tensor(self.static_data[t_end - 1]),
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트 없음: {path}")
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        meta_path = os.path.join(self.config.model_dir, f'tft_{tag}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_cols = meta['feature_cols']
            self.scaler_params = {k: np.array(v) if isinstance(v, list) else v for k, v in meta['scaler_params'].items()}

    def _load_full_checkpoint(self, path: str, optimizer, scheduler, scaler):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.ema and checkpoint.get('ema_state_dict'): self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.feature_cols, self.scaler_params = checkpoint['feature_cols'], checkpoint['scaler_params']
        return checkpoint

    def fit(self, cfg: TFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str], resume_from=None, warm_start_path=None):
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in feature_cols if c not in static_cols]
        cfg.num_static_features, cfg.num_temporal_features = len(static_cols), len(temporal_cols)
        self.config, self.feature_cols, self.model = cfg, feature_cols, TemporalFusionTransformer(cfg).to(cfg.device)
        self.model.set_feature_indices(temporal_cols)
        
        mean, std = train_df[temporal_cols].mean(), train_df[temporal_cols].std().replace(0, 1.0)
        self.scaler_params = {'mean': mean.values, 'std': std.values}
        train_norm, val_norm = train_df.copy(), val_df.copy()
        train_norm[temporal_cols], val_norm[temporal_cols] = (train_norm[temporal_cols] - mean) / std, (val_norm[temporal_cols] - mean) / std

        train_loader = DataLoader(TFTDataset(cfg, train_norm, feature_cols), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TFTDataset(cfg, val_norm, feature_cols), batch_size=cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = self._create_scheduler(optimizer, max(len(train_loader) // cfg.accumulation_steps, 1))
        criterion = DirectionalQuantileLoss(cfg.quantiles, cfg.direction_loss_weight, cfg.large_move_weight)
        
        if cfg.use_ema: self.ema = EMAModel(self.model, decay=cfg.ema_decay)
        scaler = GradScaler('cuda') if cfg.use_amp else None
        start_epoch, best_val_loss, patience_counter, global_step = 0, float('inf'), 0, 0

        if warm_start_path and os.path.exists(warm_start_path):
            ckpt = torch.load(warm_start_path, map_location=cfg.device)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))

        # 🟢 [Fix] Train 지표 기록을 위한 history 초기화
        history = {'train_loss': [], 'train_direction_acc': [], 'val_loss': [], 'val_direction_acc': [], 'learning_rate': []}
        
        for epoch in range(start_epoch, cfg.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_dir_acc = 0.0
            train_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                with autocast('cuda', enabled=cfg.use_amp):
                    preds, _, _ = self.model(batch['temporal'].to(cfg.device), batch['static'].to(cfg.device))
                    loss, loss_dict = criterion(preds, batch['target'].to(cfg.device))
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
                
                # 🟢 [Fix] Train Loss 및 Train Dir Acc 누적
                epoch_loss += loss.item() * cfg.accumulation_steps
                epoch_dir_acc += loss_dict['direction_accuracy']
                train_steps += 1

            self.model.eval()
            val_loss, dir_correct, total_val = 0.0, 0, 0
            if self.ema: self.ema.apply_shadow(self.model)
            with torch.no_grad():
                for batch in val_loader:
                    with autocast('cuda', enabled=cfg.use_amp):
                        preds, _, _ = self.model(batch['temporal'].to(cfg.device), batch['static'].to(cfg.device))
                        v_loss, v_dict = criterion(preds, batch['target'].to(cfg.device))
                    val_loss += v_loss.item()
                    dir_correct += (torch.sign(preds[:, 0, len(cfg.quantiles)//2]) == torch.sign(batch['target'].to(cfg.device)[:, 0])).sum().item()
                    total_val += preds.shape[0]
            if self.ema: self.ema.restore(self.model)

            # 🟢 [Fix] Train/Val 평균 계산
            avg_train_loss = epoch_loss / max(train_steps, 1)
            avg_train_acc = epoch_dir_acc / max(train_steps, 1)
            
            val_loss /= max(len(val_loader), 1)
            val_acc = dir_correct / max(total_val, 1)
            
            history['train_loss'].append(avg_train_loss)
            history['train_direction_acc'].append(avg_train_acc)
            history['val_loss'].append(val_loss)
            history['val_direction_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 🟢 [Fix] Logger에 Train Loss 및 Train Dir 포함
            logger.info(f"Epoch {epoch+1}/{cfg.max_epochs} | Train Loss: {avg_train_loss:.4f} | Train Dir: {avg_train_acc:.1%} | Val Loss: {val_loss:.4f} | Val Dir: {val_acc:.1%} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
                self._save_checkpoint('best')
                # 🌟 여기에 신기록 달성 로그 추가
                logger.info(f"  🌟 [New Best] Val Loss 갱신! 모델 저장 완료 (Patience 리셋)")
            else:
                patience_counter += 1
                # ⚠️ 여기에 Patience 카운팅 로그 추가
                logger.info(f"  ⚠️ Patience 카운트: {patience_counter} / {cfg.patience}")
                if patience_counter >= cfg.patience:
                    logger.info("🛑 Early Stopping Triggered. (학습 조기 종료)")
                    break
                
        self._load_checkpoint('best')
        if self.ema: self.ema.apply_shadow(self.model)
        return history

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.model.eval()
        df_norm = df.copy()
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in self.feature_cols if c not in static_cols]
        df_norm[temporal_cols] = (df_norm[temporal_cols] - self.scaler_params['mean']) / self.scaler_params['std']
        if self.config.target_col not in df.columns: df_norm[self.config.target_col] = 0.0
        
        all_preds, all_attn, all_vars = [], [], []
        for batch in DataLoader(TFTDataset(self.config, df_norm, self.feature_cols), batch_size=self.config.batch_size, shuffle=False):
            with autocast('cuda', enabled=self.config.use_amp):
                preds, attn, var_w = self.model(batch['temporal'].to(self.config.device), batch['static'].to(self.config.device))
            all_preds.append(preds.cpu().numpy())
            all_attn.append(attn.cpu().numpy())
            all_vars.append(var_w.cpu().numpy())
            
        if not all_preds: return {}
        preds_arr = np.concatenate(all_preds, axis=0)
        spread = np.clip(preds_arr[:, :, self.config.quantiles.index(0.9)] - preds_arr[:, :, self.config.quantiles.index(0.1)], 1e-6, 10.0)
        
        return {
            'quantiles': preds_arr,
            'median_pred': preds_arr[:, :, self.config.quantiles.index(0.5) if 0.5 in self.config.quantiles else len(self.config.quantiles) // 2],
            'attention': np.concatenate(all_attn, axis=0),
            'variable_importance': np.concatenate(all_vars, axis=0),
            'confidence': 1.0 / (1.0 + spread / (spread.std() + 1e-6))
        }

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(os.path.dirname(path), f"tft_{os.path.basename(path).replace('tft_', '').replace('.pt', '')}_meta.json"), 'r') as f: meta = json.load(f)
        cfg = TFTConfig()
        for k, v in meta['config'].items(): setattr(cfg, k, v)
        instance = cls(cfg)
        instance.model.load_state_dict(torch.load(path, map_location=cfg.device).get('model_state_dict', torch.load(path, map_location=cfg.device)))
        instance.feature_cols, instance.scaler_params = meta['feature_cols'], {k: np.array(v) if isinstance(v, list) else v for k, v in meta['scaler_params'].items()}
        if cfg.use_ema: instance.ema = EMAModel(instance.model, decay=cfg.ema_decay)
        return instance
