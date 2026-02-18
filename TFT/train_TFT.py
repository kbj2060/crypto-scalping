"""
TFT Signal Model 학습 스크립트

하이퍼파라미터는 TFT_model.py의 TFTConfig 클래스에서 수정하세요.

사용법:
    python TFT/train_TFT.py
    python TFT/train_TFT.py --resume models/tft/tft_epoch_50_full.pt
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from TFT_model import TFTSignalModel, TFTConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(path: str = 'data/training_features_5m.csv'):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])

    all_features = ULTIMATE_FEATURE_COLS.copy()

    missing = [c for c in all_features if c not in df.columns]
    if missing:
        logger.warning(f"누락 피처 (제외됨): {missing}")
        all_features = [c for c in all_features if c in df.columns]

    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 피처")
    logger.info(f"  ✓ 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
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
    result = model.predict(test_df)
    median_pred = result['median_pred']
    actual = test_df[cfg.target_col].values

    n_samples = len(test_df) - (cfg.input_window + cfg.forecast_horizon) + 1

    pred_1step = median_pred[:, 0]
    actual_1step = np.array([actual[i + cfg.input_window] for i in range(n_samples)])

    # NaN 제거
    valid_mask = ~np.isnan(actual_1step) & ~np.isnan(pred_1step)
    if valid_mask.sum() == 0:
        logger.warning("⚠️ 유효한 평가 샘플이 없습니다.")
        return { ... }  # 기본값 반환
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
    high_conf_mask = confidence > np.median(confidence)
    high_conf_dir_acc = np.mean(pred_dir[high_conf_mask] == actual_dir[high_conf_mask]) if high_conf_mask.sum() > 0 else 0.0

    returns = pred_dir * actual_1step
    cumulative_return = np.cumsum(returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(288 * 365)

    metrics = {
        'mae': float(mae), 'rmse': float(rmse),
        'direction_accuracy': float(direction_acc),
        'large_move_direction_acc': float(large_dir_acc),
        'high_confidence_direction_acc': float(high_conf_dir_acc),
        'simulated_sharpe_ratio': float(sharpe),
        'simulated_total_return': float(cumulative_return[-1]) if len(cumulative_return) > 0 else 0,
        'test_samples': n_samples,
    }

    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과")
    logger.info("=" * 60)
    logger.info(f"  MAE:                     {mae:.6f}")
    logger.info(f"  RMSE:                    {rmse:.6f}")
    logger.info(f"  방향 정확도:              {direction_acc:.1%}")
    logger.info(f"  큰 움직임 방향 정확도:    {large_dir_acc:.1%}")
    logger.info(f"  고확신 방향 정확도:       {high_conf_dir_acc:.1%}")
    logger.info(f"  시뮬레이션 Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  시뮬레이션 누적 수익률:   {cumulative_return[-1]:.4f}")
    logger.info("=" * 60)

    # 변수 중요도
    vi = result['var_importance'].mean(axis=(0, 1))
    static_cols = ['session_asia', 'session_europe', 'session_us']
    temporal_cols = [c for c in model.feature_cols if c not in static_cols]

    if len(vi) == len(temporal_cols):
        imp_df = pd.DataFrame({'feature': temporal_cols, 'importance': vi})
        imp_df = imp_df.sort_values('importance', ascending=False)
        logger.info("\n📊 Top 10 중요 피처:")
        for _, row in imp_df.head(20).iterrows():
            bar = '█' * int(row['importance'] * 100)
            logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    return metrics


def main():
    # Resume 경로만 argument로 받음
    parser = argparse.ArgumentParser(
        description='TFT Signal Model 학습 (하이퍼파라미터는 TFTConfig 클래스에서 수정)')
    parser.add_argument('--resume', type=str, default=None,
                        help='이어서 학습할 체크포인트 경로')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv',
                        help='학습 데이터 경로')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 TFT Signal Model 학습 시작")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(args.data)
    train_df, val_df, test_df = split_data(df)

    # ★ 자동 피처 선택 (train_df 기준으로 MI 계산)
    feature_cols = auto_select_features(
        train_df, 
        feature_cols,
        target_col='target_cumret_6',  # TFTConfig.target_col과 일치시킬 것
        max_features=20,               # temporal 피처 최대 20개
        corr_threshold=0.85,           # 상관계수 0.85 이상이면 중복 제거
        must_include=[
          'whale_retail_ratio', 'whale_conviction', 'funding_pressure', 'squeeze_power', 'oi_change_rate', 'momentum_regime', 'reversion_regime',
        ],
    )
    logger.info(f"선택된 피처 ({len(feature_cols)}개): {feature_cols}")

    config = TFTConfig(num_features=len(feature_cols))

    logger.info("\n" + "=" * 60)
    logger.info("📋 TFT Config")
    logger.info("=" * 60)
    logger.info(f"  Input Window:        {config.input_window}")
    logger.info(f"  Forecast Horizon:    {config.forecast_horizon}")
    logger.info(f"  Hidden Size:         {config.hidden_size}")
    logger.info(f"  Attention Heads:     {config.attention_heads}")
    logger.info(f"  LSTM Layers:         {config.lstm_layers}")
    logger.info(f"  Learning Rate:       {config.learning_rate}")
    logger.info(f"  Batch Size:          {config.batch_size}")
    logger.info(f"  Max Epochs:          {config.max_epochs}")
    logger.info(f"  Patience:            {config.patience}")
    logger.info(f"  LR Scheduler:        {config.lr_scheduler}")
    logger.info(f"  Warmup Epochs:       {config.warmup_epochs}")
    logger.info(f"  Direction Weight:    {config.direction_loss_weight}")
    logger.info(f"  Large Move Weight:   {config.large_move_weight}")
    logger.info(f"  Use EMA:             {config.use_ema}")
    logger.info(f"  Use AMP:             {config.use_amp}")
    logger.info(f"  Device:              {config.device}")
    logger.info("=" * 60)

    model = TFTSignalModel(config)
    history = model.fit(train_df, val_df, feature_cols)
    # 테스트셋 평가
    metrics = evaluate_model(model, test_df)

    # 모델 저장
    model.save()
    logger.info(f"\n모델 저장 완료: {config.model_dir}/tft_final.pt")

    # 결과 저장
    results = {
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'history': history,
        'test_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'elapsed': str(datetime.now() - start_time),
    }
    results_path = os.path.join(config.model_dir, 'training_results.json')
    os.makedirs(config.model_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"결과 저장: {results_path}")

    print("\n" + "=" * 80)
    print(f"🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print("=" * 80)


if __name__ == '__main__':
    main()
