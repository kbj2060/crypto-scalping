"""
TFT Signal Model 학습 스크립트 (Pure Prediction Edition)
================================================================================
- 타겟 구조: 단일 스칼라 -> 미래 h스텝 궤적(Sequence-to-Sequence)
- 평가 구조: 트레이딩 PnL 삭제 -> 통계적 분포 측정(CRPS, PICP, RMSE)
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
from core.feature_engineering import ULTIMATE_FEATURE_COLS, MUST_INCLUDE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(cfg: TFTConfig, path: str = 'data/training_features_5m.csv'):
    logger.info(f"데이터 로드: {path} (Horizon: {cfg.forecast_horizon})")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    # [개선 3] 1스텝 앞의 수익률(%)만 타겟으로 만듭니다. 
    # 모델 학습 시 데이터셋 슬라이싱으로 미래 h스텝 벡터를 자동으로 구성하게 됩니다.
    df[cfg.target_col] = df['close'].pct_change().shift(-1) * 100.0

    df.dropna(inplace=True)

    combined_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns] + ['mtf_trend_1h', 'mtf_trend_4h']
    all_features = list(dict.fromkeys(combined_features))
    
    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 피처")
    return df, all_features
    
def split_data_with_embargo(df: pd.DataFrame, cfg: TFTConfig, train_ratio=0.7, val_ratio=0.15):
    embargo = cfg.input_window + cfg.forecast_horizon
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_start = train_end + embargo
    val_end = int(n * (train_ratio + val_ratio))
    test_start = val_end + embargo
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"  Train: {len(train_df):,}행")
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    logger.info(f"  🔒 Embargo 갭 적용 완료 (각 구간별 {embargo} 캔들 삭제)")
    return train_df, val_df, test_df

def evaluate_model(model: TFTSignalModel, test_df: pd.DataFrame):
    logger.info("\n통계적 성능 (Price Prediction Metrics) 산출 중...")
    cfg = model.config
    
    if cfg.target_col not in test_df.columns:
        logger.warning(f"타겟 '{cfg.target_col}' 부재 → 평가 스킵")
        return {}

    result = model.predict(test_df)
    if not result: return {}
        
    # 결과 차원: (Samples, Horizon, Quantiles)
    median_pred = result['median_pred']         # (N, H)
    lower_pred = result['quantiles'][:, :, 0]   # 0.05 퀀타일
    upper_pred = result['quantiles'][:, :, -1]  # 0.95 퀀타일
    
    actual = test_df[cfg.target_col].values
    n_samples = len(test_df) - (cfg.input_window + cfg.forecast_horizon) + 1

    # [개선 2 & 3] 평가를 위해 실제 정답도 예측과 동일한 형태의 (N, H) 시퀀스 매트릭스로 변환
    actual_seqs = np.array([
        actual[i + cfg.input_window : i + cfg.input_window + cfg.forecast_horizon] 
        for i in range(n_samples)
    ])

    # 1. 평균 절대 오차 (MAE) 및 제곱근 오차 (RMSE) - 전체 Horizon 평균
    mae = np.mean(np.abs(median_pred - actual_seqs))
    rmse = np.sqrt(np.mean((median_pred - actual_seqs) ** 2))

    # 2. 방향성 지표 (보조용)
    pred_dir = np.sign(median_pred)
    actual_dir = np.sign(actual_seqs)
    direction_acc = np.mean(pred_dir == actual_dir)

    # 3. PICP (Prediction Interval Coverage Probability)
    # 모델이 예측한 90% 밴드(0.05~0.95) 안에 실제 값이 몇 % 확률로 들어왔는가? (이상적 = 90%)
    coverage_mask = (actual_seqs >= lower_pred) & (actual_seqs <= upper_pred)
    picp = np.mean(coverage_mask)

    # 4. CRPS Proxy (Continuous Ranked Probability Score 대용치)
    # 분포 예측의 전반적인 오류율 (낮을수록 좋음)
    errors = actual_seqs[..., None] - result['quantiles']
    q_tensor = np.array(cfg.quantiles).reshape(1, 1, -1)
    crps_proxy = np.mean(np.maximum((q_tensor - 1) * errors, q_tensor * errors))

    metrics = {
        'mae': float(mae), 
        'rmse': float(rmse),
        'crps_proxy': float(crps_proxy),
        'picp_90': float(picp),
        'direction_accuracy': float(direction_acc),
        'test_samples': int(n_samples),
    }

    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트셋 예측력 검증 결과 (Prediction Metrics)")
    logger.info("=" * 60)
    logger.info(f"  [정확도] MAE: {mae:.5f} | RMSE: {rmse:.5f}")
    logger.info(f"  [분포력] CRPS (Q-Loss 평균): {crps_proxy:.5f}")
    logger.info(f"  [신뢰성] PICP (90% 신뢰구간 커버리지): {picp:.1%} (이상적 수치: 90%)")
    logger.info(f"  [보조값] 방향 일치율: {direction_acc:.1%}")
    logger.info("=" * 60)

    if 'variable_importance' in result:
        vi = result['variable_importance'].mean(axis=(0, 1))
        temporal_cols = model.feature_cols
        if len(vi) == len(temporal_cols):
            imp_df = pd.DataFrame({'feature': temporal_cols, 'importance': vi}).sort_values('importance', ascending=False)
            logger.info("\n🔥 Top 10 예측 기여 피처:")
            for _, row in imp_df.head(10).iterrows():
                bar = '█' * int(row['importance'] * 100)
                logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description='TFT Signal Model 학습')
    parser.add_argument('--resume', type=str, default=None, help='사전 학습된 모델 가중치 경로 (.pt)')
    parser.add_argument('--resume-best-optuna', type=str, default=None)
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    args = parser.parse_args()

    _default_cfg = TFTConfig()
    
    print("\n" + "=" * 80)
    print("🚀 TFT Signal Model 학습 시작 (Pure Price Prediction Edition)")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(_default_cfg, args.data)
    train_df, val_df, test_df = split_data_with_embargo(df, _default_cfg)

    # 예측 모델이므로 불필요한 노이즈 시그널이 아닌, 순수 미시구조/매크로 피처가 선택되도록 유도
    selected_features = auto_select_features(
        train_df, feature_cols,
        target_col=_default_cfg.target_col,
        max_features=_default_cfg.num_features,
        corr_threshold=0.85,
        must_include=MUST_INCLUDE_FEATURES
    )

    cfg = TFTConfig(num_features=len(selected_features))
    
    logger.info(f"\n=== Config ===")
    logger.info(f"  Target:    {cfg.target_col} (1-step sequence mode)")
    logger.info(f"  Horizon:   {cfg.forecast_horizon} steps")
    logger.info(f"  Features:  {len(selected_features)}")
    logger.info(f"  Hidden:    {cfg.hidden_size}")
    logger.info(f"  LR:        {cfg.learning_rate}")

    model = TFTSignalModel(cfg)
    
    history = model.fit(
        cfg, train_df, val_df, selected_features,
        resume_from=args.resume,
        warm_start_path=args.resume_best_optuna
    )
    
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