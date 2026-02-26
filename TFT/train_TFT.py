"""
TFT Signal Model 학습 스크립트 (복원 버전)

52% / 53.1% 고확신 달성했던 설정 기반.
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
from TFT_model import TFTSignalModel, TFTConfig, TFTEnsemble

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# [train_TFT.py 의 load_data 함수 전체 교체]

def load_data(path: str = 'data/training_features_5m.csv', h: int = 6):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])

    # 🚨 1. 무한대 값 등 에러 유발 인자 사전 차단
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 🚨 2. MTF 피처 생성
    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    # 🚨 3. 미래 3봉 VWAP 타겟 (안전한 계산)
    logger.info("타겟 컬럼 생성: 미래 3봉 VWAP")
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = tp * df['volume']
    
    future_tp_vol_sum = tp_vol.rolling(window=3).sum().shift(-h)
    future_vol_sum = df['volume'].rolling(window=3).sum().shift(-h)
    future_tp_avg = tp.rolling(window=3).mean().shift(-h)
    
    # 거래량이 0인 구간은 일반 평균(tp_avg)으로 대체하여 NaN/Inf 원천 봉쇄
    future_vwap = np.where(future_vol_sum == 0, future_tp_avg, future_tp_vol_sum / future_vol_sum.replace(0, np.nan))
    
    df[f'target_ret_{h}'] = (future_vwap / df['close']) - 1

    if 'regime_break' not in df.columns:
        df['regime_break'] = 0.0

    # 🚨 4. 결측치(NaN) 완벽 제거 (이 한 줄이 없어서 Train이 NaN이 되었습니다!)
    df.dropna(inplace=True)

    all_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    
    for c in ['mtf_trend_1h', 'mtf_trend_4h']:
        if c not in all_features:
            all_features.append(c)

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

    # NaN 제거
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
    logger.info(f"  시뮬레이션 Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"  시뮬레이션 누적 수익률:   {cumulative_return[-1]:.4f}")
    logger.info("=" * 60)

    # 변수 중요도
    if 'variable_importance' in result:
        vi = result['variable_importance'].mean(axis=0)
        static_cols = ['session_asia', 'session_europe', 'session_us']
        temporal_cols = [c for c in model.feature_cols if c not in static_cols]

        if len(vi) == len(temporal_cols):
            imp_df = pd.DataFrame({'feature': temporal_cols, 'importance': vi})
            imp_df = imp_df.sort_values('importance', ascending=False)
            logger.info("\n📊 Top 10 중요 피처:")
            for _, row in imp_df.head(10).iterrows():
                bar = '█' * int(row['importance'] * 100)
                logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='TFT Signal Model 학습')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    args = parser.parse_args()

    # horizon / target / max_features → TFTConfig 기본값 사용
    _default_cfg = TFTConfig()

    print("\n" + "=" * 80)
    print("🚀 TFT Signal Model 학습 시작 (복원 버전)")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(args.data, _default_cfg.forecast_horizon)
    train_df, val_df, test_df = split_data(df)

    # ★ 타겟 선택 — TFTConfig.target_col 사용
    target_col = _default_cfg.target_col
    if target_col not in df.columns:
        logger.warning(f"{target_col} 없음 → 자동 생성")
        horizon = _default_cfg.forecast_horizon
        df[target_col] = (df['close'].shift(-horizon) / df['close'] - 1)

    # ★ 피처 선택 — TFTConfig.num_features 를 max_features 상한으로 사용
    selected_features = auto_select_features(
        train_df,
        feature_cols,
        target_col=target_col,
        max_features=_default_cfg.num_features,
        corr_threshold=0.85,
        must_include=[
            'whale_conviction', 'net_taker_ratio', 'oi_change_rate',
            'funding_z_score', 'hurst_48', 'regime_trending',
            'volatility_z', 'garman_klass_vol', 'mtf_trend_1h', 'mtf_trend_4h'
        ]
    )
    logger.info(f"선택된 피처 ({len(selected_features)}개): {selected_features}")

    # ★ 모델 설정 — TFTConfig 기본값 사용, num_features만 실제 선택 수로 덮어쓰기
    cfg = TFTConfig(num_features=len(selected_features))

    logger.info(f"\n=== Config ===")
    logger.info(f"  Target:    {cfg.target_col}")
    logger.info(f"  Horizon:   {cfg.forecast_horizon}")
    logger.info(f"  Features:  {len(selected_features)}")
    logger.info(f"  Hidden:    {cfg.hidden_size}")
    logger.info(f"  Quantiles: {cfg.quantiles}")
    logger.info(f"  LR:        {cfg.learning_rate}")
    logger.info(f"  Scheduler: {cfg.lr_scheduler}")

    # 학습
    model = TFTSignalModel(cfg)
    history = model.fit(cfg, train_df, val_df, selected_features, args.resume)
    
    model._save_checkpoint('final')
    metrics = evaluate_model(model, test_df)

    # 결과 저장
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


if __name__ == '__main__':
    main()
