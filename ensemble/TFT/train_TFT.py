"""
TFT Signal Model 학습 스크립트 (Look-ahead Bias 제거 & 결정론적 순서 유지)
"""
import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
import torch
from datetime import datetime
from TFT_model import TFTSignalModel, TFTConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path: str = 'data/training_features_5m.csv', h: int = 6):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

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

    # 🔴 [Fix] Set 자료형의 비결정론적 순서를 방지하기 위한 안전한 중복 제거
    combined_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns] + ['mtf_trend_1h', 'mtf_trend_4h']
    all_features = list(dict.fromkeys(combined_features))
    
    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 피처")
    return df, all_features
    
def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"  Train: {len(train_df):,}행")
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    return train_df, val_df, test_df

def evaluate_model(model: TFTSignalModel, test_df: pd.DataFrame):
    logger.info("\n테스트셋 평가 중...")
    cfg = model.config
    
    if cfg.target_col not in test_df.columns:
        logger.warning(f"타겟 '{cfg.target_col}' 부재 → 평가 스킵")
        return {}

    result = model.predict(test_df)
    if not result:
        logger.warning("예측 결과 없음")
        return {}
        
    median_pred = result['median_pred']
    actual = test_df[cfg.target_col].values

    n_samples = len(test_df) - (cfg.input_window + cfg.forecast_horizon) + 1

    pred_1step = median_pred[:, 0]
    actual_1step = np.array([actual[i + cfg.input_window] for i in range(n_samples)])

    valid_mask = ~np.isnan(actual_1step) & ~np.isnan(pred_1step)
    if valid_mask.sum() == 0:
        logger.warning("유효한 평가 샘플 없음")
        return {}
    
    actual_1step = actual_1step[valid_mask]
    pred_1step = pred_1step[valid_mask]
    
    mae = np.mean(np.abs(pred_1step - actual_1step))
    rmse = np.sqrt(np.mean((pred_1step - actual_1step) ** 2))

    pred_dir = np.sign(pred_1step)
    actual_dir = np.sign(actual_1step)
    direction_acc = np.mean(pred_dir == actual_dir)

    threshold = np.percentile(np.abs(actual_1step), 80)
    large_mask = np.abs(actual_1step) > threshold
    large_dir_acc = np.mean(pred_dir[large_mask] == actual_dir[large_mask]) if large_mask.sum() > 0 else 0.0

    confidence = result['confidence'][:, 0]
    if len(confidence) == len(valid_mask):
        confidence = confidence[valid_mask]
    high_conf_mask = confidence > np.median(confidence)
    high_conf_dir_acc = np.mean(pred_dir[high_conf_mask] == actual_dir[high_conf_mask]) if high_conf_mask.sum() > 0 else 0.0

    metrics = {
        'mae': float(mae), 'rmse': float(rmse),
        'direction_accuracy': float(direction_acc),
        'large_move_direction_acc': float(large_dir_acc),
        'high_confidence_direction_acc': float(high_conf_dir_acc),
        'test_samples': int(valid_mask.sum()),
    }

    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과")
    logger.info("=" * 60)
    logger.info(f"  MAE:                     {mae:.6f}")
    logger.info(f"  RMSE:                    {rmse:.6f}")
    logger.info(f"  방향 정확도:              {direction_acc:.1%}")
    logger.info(f"  큰 움직임 방향 정확도:    {large_dir_acc:.1%}")
    logger.info(f"  고확신 방향 정확도:       {high_conf_dir_acc:.1%}")
    logger.info("=" * 60)

    if 'variable_importance' in result:
        vi = result['variable_importance'].mean(axis=0)
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in model.feature_cols if c not in static_cols]

        if len(vi) == len(temporal_cols):
            imp_df = pd.DataFrame({'feature': temporal_cols, 'importance': vi}).sort_values('importance', ascending=False)
            logger.info("\n📊 Top 10 중요 피처:")
            for _, row in imp_df.head(10).iterrows():
                bar = '█' * int(row['importance'] * 100)
                logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='TFT Signal Model 학습')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-best-optuna', type=str, default=None)
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    args = parser.parse_args()

    _default_cfg = TFTConfig()
    
    print("\n" + "=" * 80)
    print("🚀 TFT Signal Model 학습 시작")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(args.data)
    train_df, val_df, test_df = split_data(df)

    if _default_cfg.target_col not in df.columns:
        df[_default_cfg.target_col] = (df['close'].shift(-_default_cfg.forecast_horizon) / df['close'] - 1)

    selected_features = auto_select_features(
        train_df, feature_cols,
        target_col=_default_cfg.target_col,
        max_features=_default_cfg.num_features,
        corr_threshold=0.85,
    )

    
    cfg = TFTConfig(num_features=len(selected_features))
    
    logger.info(f"\n=== Config ===")
    logger.info(f"  Target:    {cfg.target_col}")
    logger.info(f"  Horizon:   {cfg.forecast_horizon}")
    logger.info(f"  Features:  {len(selected_features)}")
    logger.info(f"  Hidden:    {cfg.hidden_size}")
    logger.info(f"  Quantiles: {cfg.quantiles}")
    logger.info(f"  LR:        {cfg.learning_rate}")
    logger.info(f"  Scheduler: {cfg.lr_scheduler}")

    model = TFTSignalModel(cfg)
    history = model.fit(cfg, train_df, val_df, selected_features, resume_from=args.resume, warm_start_path=args.resume_best_optuna)
    model._save_checkpoint('final')
    
    metrics = evaluate_model(model, test_df)

    results = {
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'selected_features': selected_features,
        'history': history,
        'test_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'elapsed': str(datetime.now() - start_time),
    }
    
    results_path = os.path.join(cfg.model_dir, 'training_results.json')
    os.makedirs(cfg.model_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"결과 저장: {results_path}")

    print("\n" + "=" * 80)
    print(f"🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print("=" * 80)

if __name__ == '__main__': main()