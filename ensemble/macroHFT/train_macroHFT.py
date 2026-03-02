"""
MacroHFT Signal Model 학습 스크립트
"""
import sys, os, argparse, logging, json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from macroHFT_model import ForecastingMacroHFT, MacroHFTConfig
from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacroHFTDataset(Dataset):
    def __init__(self, config: MacroHFTConfig, df: pd.DataFrame, temporal_cols: list):
        self.temporal_data = df[temporal_cols].values.astype(np.float32)
        self.target_data = df[config.target_col].values.astype(np.float32)
        self.window_size, self.horizon = config.input_window, config.forecast_horizon
        self.n_samples = len(df) - (self.window_size + self.horizon) + 1

    def __len__(self): return max(0, self.n_samples)
    def __getitem__(self, idx):
        t_end = idx + self.window_size
        return (torch.tensor(self.temporal_data[idx:t_end]), 
                torch.tensor(self.target_data[t_end:t_end + self.horizon]))

class MacroHFTQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9], direction_weight=5.0):
        super().__init__()
        self.quantiles = quantiles
        self.direction_weight = direction_weight
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def forward(self, preds, targets):
        targets_exp = targets.unsqueeze(-1).expand_as(preds)
        errors = targets_exp - preds
        q_tensor = torch.tensor(self.quantiles, device=preds.device).unsqueeze(0).unsqueeze(0)
        q_loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors).mean()
        
        pred_median = preds[:, :, self.median_idx]
        actual_sign = torch.sign(targets)
        
        is_bearish = (targets < 0).float()
        asymmetric_weight = 1.0 + is_bearish * 1.0
        
        wrong_dir_penalty = torch.relu(-pred_median * actual_sign) * torch.abs(targets)
        # 🔴 [Fix] 과도한 스케일링(* 100.0) 제거하여 Quantile Loss 보존
        dir_loss = (wrong_dir_penalty * asymmetric_weight).mean() 
        
        return q_loss + (self.direction_weight * dir_loss)

class MacroHFTSignalModel:
    def __init__(self, config: MacroHFTConfig = None):
        self.config, self.model, self.feature_cols, self.scaler_params = config, None, None, {}

    def _save_checkpoint(self, tag: str):
        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'macrohft_{tag}.pt'))
        meta = {'feature_cols': self.feature_cols, 'scaler_params': {k: v.tolist() for k, v in self.scaler_params.items()}, 'config': self.config.__dict__}
        with open(os.path.join(self.config.model_dir, f'macrohft_{tag}_meta.json'), 'w') as f: json.dump(meta, f, indent=2)

    def fit(self, cfg: MacroHFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]):
        temporal_cols = feature_cols
        cfg.num_temporal_features = len(temporal_cols)
        self.config, self.feature_cols, self.model = cfg, feature_cols, ForecastingMacroHFT(cfg).to(cfg.device)

        mean, std = train_df[temporal_cols].mean(), train_df[temporal_cols].std().replace(0, 1.0)
        self.scaler_params = {'mean': mean.values, 'std': std.values}
        
        train_norm, val_norm = train_df.copy(), val_df.copy()
        train_norm[temporal_cols] = (train_norm[temporal_cols] - mean) / std
        val_norm[temporal_cols] = (val_norm[temporal_cols] - mean) / std

        train_loader = DataLoader(MacroHFTDataset(cfg, train_norm, temporal_cols), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(MacroHFTDataset(cfg, val_norm, temporal_cols), batch_size=cfg.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)
        criterion = MacroHFTQuantileLoss(quantiles=cfg.quantiles, direction_weight=cfg.direction_loss_weight)
        scaler = GradScaler('cuda') if cfg.use_amp else None
        
        best_val_loss, patience_cnt = float('inf'), 0
        # 🟢 [Fix] Train 지표 기록을 위한 history 초기화
        history = {'train_loss': [], 'train_direction_acc': [], 'val_loss': [], 'val_direction_acc': []}
        
        for epoch in range(cfg.max_epochs):
            self.model.train()
            train_loss = 0.0
            train_dir_correct = 0
            train_total = 0
            
            for seq, info, tgt in train_loader:
                seq, info, tgt = seq.to(cfg.device), info.to(cfg.device), tgt.to(cfg.device)
                optimizer.zero_grad()
                with autocast('cuda', enabled=cfg.use_amp):
                    preds = self.model(seq, info)
                    loss = criterion(preds, tgt)
                if scaler:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                # 🟢 [Fix] Train Loss 및 Train Dir Acc 누적
                train_loss += loss.item()
                pred_median = preds[:, 0, criterion.median_idx]
                train_dir_correct += (torch.sign(pred_median) == torch.sign(tgt[:, 0])).sum().item()
                train_total += tgt.size(0)
            
            self.model.eval()
            val_loss, val_dir_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for seq, info, tgt in val_loader:
                    seq, info, tgt = seq.to(cfg.device), info.to(cfg.device), tgt.to(cfg.device)
                    preds = self.model(seq, info)
                    val_loss += criterion(preds, tgt).item()
                    val_dir_correct += (torch.sign(preds[:, 0, criterion.median_idx]) == torch.sign(tgt[:, 0])).sum().item()
                    val_total += tgt.size(0)
            
            # 🟢 [Fix] Train/Val 평균 계산
            avg_train_loss = train_loss / max(len(train_loader), 1)
            train_acc = train_dir_correct / max(train_total, 1)
            
            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_acc = val_dir_correct / max(val_total, 1)
            
            history['train_loss'].append(avg_train_loss)
            history['train_direction_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_direction_acc'].append(val_acc)
            
            # 🟢 [Fix] Logger에 Train Loss 및 Train Dir 포함
            logger.info(f"Epoch {epoch+1:03d}/{cfg.max_epochs} | Train Loss: {avg_train_loss:.4f} | Train Dir: {train_acc:.1%} | Val Loss: {avg_val_loss:.4f} | Val Dir: {val_acc:.1%}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss, patience_cnt = avg_val_loss, 0
                self._save_checkpoint('best')
                logger.info(f"  🌟 [New Best] Val Loss 갱신! 모델 저장 완료 (Patience 리셋)")
            else:
                patience_cnt += 1
                logger.info(f"  ⚠️ Patience 카운트: {patience_cnt} / {cfg.patience}")
                if patience_cnt >= cfg.patience:
                    logger.info("🛑 Early Stopping Triggered. (학습 조기 종료)")
                    break
                
        return history


def load_data(path: str = 'data/training_features_5m.csv', h: int = 6):
    df = pd.read_csv(path, parse_dates=['timestamp']).replace([np.inf, -np.inf], np.nan)
    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    tp = (df['high'] + df['low'] + df['close']) / 3.0
    future_tp_vol_sum = (tp * df['volume']).rolling(window=h).sum().shift(-h)
    future_vol_sum = df['volume'].rolling(window=h).sum().shift(-h)
    future_tp_avg = tp.rolling(window=h).mean().shift(-h)
    df[f'target_ret_{h}'] = (np.where(future_vol_sum == 0, future_tp_avg, future_tp_vol_sum / future_vol_sum.replace(0, np.nan)) / df['close']) - 1

    if 'cvp_poc_dist' not in df.columns:
        logger.info("📊 Clusters Volume Profile 피처 생성 중...")
        df = add_cvp_features(df, lookback=200, n_clusters=4, drop_strategy=False)

    df.dropna(inplace=True)

    # 🔴 [Fix] 결정론적 순서 유지 피처 병합
    combined_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns] + ['mtf_trend_1h', 'mtf_trend_4h']
    all_features = list(dict.fromkeys(combined_features))
    return df, all_features
    
def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end, val_end = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def evaluate_model(model_wrapper: MacroHFTSignalModel, test_df: pd.DataFrame):
    logger.info("\n테스트셋 평가 중...")
    cfg = model_wrapper.config
    
    if cfg.target_col not in test_df.columns:
        logger.warning(f"타겟 '{cfg.target_col}' 부재 → 평가 스킵")
        return {}
        
    temporal_cols = model_wrapper.feature_cols
    
    test_norm = test_df.copy()
    test_norm[temporal_cols] = (test_norm[temporal_cols] - model_wrapper.scaler_params['mean']) / model_wrapper.scaler_params['std']
    
    dataset = MacroHFTDataset(cfg, test_norm, temporal_cols)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    
    model_wrapper.model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for seq, info, tgt in loader:
            preds = model_wrapper.model(seq.to(cfg.device), info.to(cfg.device)) 
            all_preds.append(preds.cpu().numpy())
            all_targets.append(tgt.numpy())
            
    if not all_preds: return {}
        
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    
    mid_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2
    pred_1step = preds_arr[:, 0, mid_idx]
    actual_1step = targets_arr[:, 0]
    
    mae = np.mean(np.abs(pred_1step - actual_1step))
    rmse = np.sqrt(np.mean((pred_1step - actual_1step) ** 2))
    
    pred_dir = np.sign(pred_1step)
    actual_dir = np.sign(actual_1step)
    direction_acc = np.mean(pred_dir == actual_dir)
    
    threshold = np.percentile(np.abs(actual_1step), 80)
    large_mask = np.abs(actual_1step) > threshold
    large_dir_acc = np.mean(pred_dir[large_mask] == actual_dir[large_mask]) if large_mask.sum() > 0 else 0.0
    
    metrics = {
        'mae': float(mae), 'rmse': float(rmse),
        'direction_accuracy': float(direction_acc),
        'large_move_direction_acc': float(large_dir_acc),
        'test_samples': len(pred_1step)
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과 (MacroHFT)")
    logger.info("=" * 60)
    logger.info(f"  MAE:                     {mae:.6f}")
    logger.info(f"  RMSE:                    {rmse:.6f}")
    logger.info(f"  방향 정확도:              {direction_acc:.1%}")
    logger.info(f"  큰 움직임 방향 정확도:    {large_dir_acc:.1%}")
    logger.info("=" * 60)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    cfg = MacroHFTConfig()
    
    print("\n" + "=" * 80)
    print("🚀 MacroHFT Model 학습 시작 (TFT-Aligned Structure)")
    print("=" * 80)
    start_time = datetime.now()
    
    df, feature_cols = load_data(parser.parse_args().data, h=cfg.forecast_horizon)
    train_df, val_df, test_df = split_data(df)
    
    selected_features = auto_select_features(
        train_df, feature_cols,
        target_col=cfg.target_col,
        max_features=35,
        corr_threshold=0.85,
    )

    for sc in ['session_asia', 'session_europe', 'session_us']:
        if sc in df.columns and sc not in selected_features: selected_features.append(sc)
        
    logger.info(f"\n=== Config ===")
    logger.info(f"  Target:    {cfg.target_col}")
    logger.info(f"  Horizon:   {cfg.forecast_horizon}")
    logger.info(f"  Features:  {len(selected_features)}")
    logger.info(f"  Hidden:    {cfg.d_model}")
    logger.info(f"  Quantiles: {cfg.quantiles}")
    logger.info(f"  LR:        {cfg.learning_rate}")
    
    model = MacroHFTSignalModel(cfg)
    history = model.fit(cfg, train_df, val_df, selected_features)
    metrics = evaluate_model(model, test_df)
    
    results = {
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'selected_features': selected_features,
        'history': history,
        'test_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'elapsed': str(datetime.now() - start_time),
    }
    
    results_path = os.path.join(cfg.model_dir, 'macrohft_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)
    logger.info(f"결과 저장: {results_path}")

    print("\n" + "=" * 80)
    print(f"🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print("=" * 80)

if __name__ == '__main__': main()