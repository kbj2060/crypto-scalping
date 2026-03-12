"""
TimeMachine (SSM) Signal Model 학습 스크립트
================================================================================
- 타겟 구조: 미래 h스텝 궤적(Sequence-to-Sequence)
- 평가 구조: 통계적 분포 측정(CRPS, PICP, RMSE)
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
from model import TimeMachineSignalModel, TimeMachineConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS, MUST_INCLUDE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(cfg: TimeMachineConfig, path: str = 'data/training_features_5m.csv'):
    logger.info(f"데이터 로드: {path} (Horizon: {cfg.forecast_horizon})")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    df[cfg.target_col] = df['close'].pct_change().shift(-1) * 100.0
    df.dropna(inplace=True)

    combined_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns] + ['mtf_trend_1h', 'mtf_trend_4h']
    all_features = list(dict.fromkeys(combined_features))
    
    # ──────────────────────────────────────────────────
    # [FIX] 피처 열 단위 NaN/상수값 검증 및 제거
    # dropna(행 단위) 이후에도 특정 열이 전부 NaN이거나 상수인 경우 방어
    # ──────────────────────────────────────────────────
    valid_features = []
    dropped_features = []
    for col in all_features:
        col_data = df[col]
        # 열 전체가 NaN이거나, 유효 값이 너무 적거나, 분산이 0이면 제거
        if col_data.isna().all():
            dropped_features.append((col, 'all NaN'))
        elif col_data.nunique() <= 1:
            dropped_features.append((col, 'constant'))
        elif col_data.std() == 0 or np.isnan(col_data.std()):
            dropped_features.append((col, 'zero/NaN std'))
        else:
            valid_features.append(col)
    
    if dropped_features:
        logger.warning(f"  ⚠️ 제거된 피처 {len(dropped_features)}개:")
        for col, reason in dropped_features:
            logger.warning(f"    - {col}: {reason}")
    
    all_features = valid_features
    
    # [FIX] target 열에 NaN/Inf가 남아있으면 해당 행 제거
    target_invalid = df[cfg.target_col].isna() | np.isinf(df[cfg.target_col])
    if target_invalid.any():
        logger.warning(f"  ⚠️ 타겟 열 NaN/Inf {target_invalid.sum()}행 제거")
        df = df[~target_invalid].copy()

    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 유효 피처")
    
    # [개선] 데이터 품질 요약 로깅
    nan_counts = df[all_features].isna().sum()
    if nan_counts.any():
        logger.info(f"  ℹ️ 잔존 NaN 피처 수: {(nan_counts > 0).sum()}개, 총 NaN 셀: {nan_counts.sum()}")
    
    return df, all_features

    
def split_data_with_embargo(df: pd.DataFrame, cfg: TimeMachineConfig, train_ratio=0.7, val_ratio=0.15):
    embargo = cfg.input_window + cfg.forecast_horizon
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_start = train_end + embargo
    val_end = int(n * (train_ratio + val_ratio))
    test_start = val_end + embargo
    
    # [FIX] 분할 후 데이터가 충분한지 검증
    if val_start >= val_end:
        logger.warning(f"  ⚠️ Embargo({embargo}) 때문에 val 데이터가 부족합니다. embargo 줄임")
        val_start = train_end + cfg.forecast_horizon  # 최소 embargo
    if test_start >= n:
        logger.warning(f"  ⚠️ Embargo({embargo}) 때문에 test 데이터가 부족합니다. embargo 줄임")
        test_start = val_end + cfg.forecast_horizon
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"  Train: {len(train_df):,}행")
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    
    # 최소 데이터 수 경고
    min_required = cfg.input_window + cfg.forecast_horizon + 1
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if len(split_df) < min_required:
            logger.warning(f"  ⚠️ {name} 데이터({len(split_df)}행)가 "
                           f"최소 요구량({min_required}행)보다 적습니다!")
    
    return train_df, val_df, test_df


def evaluate_model(model: TimeMachineSignalModel, test_df: pd.DataFrame):
    logger.info("\n통계적 성능 (Price Prediction Metrics) 산출 중...")
    cfg = model.config
    
    if cfg.target_col not in test_df.columns: return {}

    result = model.predict(test_df)
    if not result: return {}
        
    median_pred = result['median_pred']         
    lower_pred = result['quantiles'][:, :, 0]   
    upper_pred = result['quantiles'][:, :, -1]  
    
    actual = test_df[cfg.target_col].values
    n_samples = len(test_df) - (cfg.input_window + cfg.forecast_horizon) + 1

    if n_samples <= 0:
        logger.warning("  ⚠️ 평가할 샘플이 없습니다.")
        return {}

    actual_seqs = np.array([
        actual[i + cfg.input_window : i + cfg.input_window + cfg.forecast_horizon] 
        for i in range(n_samples)
    ])

    # [FIX] 평가 시 NaN 방어
    valid_mask = ~(np.isnan(median_pred) | np.isnan(actual_seqs))
    if not valid_mask.any():
        logger.warning("  ⚠️ 유효한 예측/실제 값이 없습니다.")
        return {}

    mae = np.mean(np.abs(median_pred[valid_mask] - actual_seqs[valid_mask]))
    rmse = np.sqrt(np.mean((median_pred[valid_mask] - actual_seqs[valid_mask]) ** 2))
    direction_acc = np.mean(np.sign(median_pred[valid_mask]) == np.sign(actual_seqs[valid_mask]))

    coverage_mask = (actual_seqs >= lower_pred) & (actual_seqs <= upper_pred)
    picp = np.mean(coverage_mask[valid_mask])

    errors = actual_seqs[..., None] - result['quantiles']
    q_tensor = np.array(cfg.quantiles).reshape(1, 1, -1)
    crps_proxy = np.nanmean(np.maximum((q_tensor - 1) * errors, q_tensor * errors))

    metrics = {
        'mae': float(mae), 
        'rmse': float(rmse),
        'crps_proxy': float(crps_proxy),
        'picp_90': float(picp),
        'direction_accuracy': float(direction_acc),
        'test_samples': int(n_samples),
    }

    logger.info("\n" + "=" * 60)
    logger.info("📊 Mamba 예측력 검증 결과 (TimeMachine Metrics)")
    logger.info("=" * 60)
    logger.info(f"  [정확도] MAE: {mae:.5f} | RMSE: {rmse:.5f}")
    logger.info(f"  [분포력] CRPS (Q-Loss 평균): {crps_proxy:.5f}")
    logger.info(f"  [신뢰성] PICP (90% 신뢰구간): {picp:.1%} (이상적 수치: 90%)")
    logger.info(f"  [보조값] 방향 일치율: {direction_acc:.1%}")
    logger.info("=" * 60)

    return metrics

def main():
    parser = argparse.ArgumentParser(description='TimeMachine Model 학습')
    parser.add_argument('--resume', type=str, default=None, help='사전 학습 가중치')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    args = parser.parse_args()

    _default_cfg = TimeMachineConfig()
    
    # 💡 [추가] 재현성을 위해 시드 고정 (MacroHFT 로직)
    import random
    random.seed(_default_cfg.seed)
    np.random.seed(_default_cfg.seed)
    torch.manual_seed(_default_cfg.seed)
    
    print("\n" + "=" * 80)
    print("🚀 TimeMachine (Mamba SSM) 학습 시작")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(_default_cfg, args.data)
    train_df, val_df, test_df = split_data_with_embargo(df, _default_cfg)

    selected_features = auto_select_features(
        train_df, feature_cols,
        target_col=_default_cfg.target_col,
        max_features=_default_cfg.num_features,
        corr_threshold=0.85,
        must_include=MUST_INCLUDE_FEATURES
    )

    if not selected_features:
        logger.warning("  ⚠️ auto_select_features 결과가 비어있음. 전체 유효 피처 사용")
        selected_features = feature_cols[:_default_cfg.num_features]

    # 💡 [추가] Baseline GBM 로직 (피처 성능 진단)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        
        # GBM은 단일 시점 방향성만 테스트 (1스텝 앞 방향)
        X_train = train_df[selected_features].fillna(0).values
        y_train = (train_df[_default_cfg.target_col] > 0).astype(int).values
        X_val = val_df[selected_features].fillna(0).values
        y_val = (val_df[_default_cfg.target_col] > 0).astype(int).values
        
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, gb.predict(X_train))
        val_acc = accuracy_score(y_val, gb.predict(X_val))
        logger.info(f"📊 [Baseline GBM] Train Dir: {train_acc:.1%}, Val Dir: {val_acc:.1%} (피처 유효성 진단)")
    except Exception as e:
        logger.warning(f"Baseline GBM 진단 스킵: {e}")

    cfg = TimeMachineConfig(num_features=len(selected_features))
    
    logger.info(f"\n=== Config ===")
    logger.info(f"  Target:    {cfg.target_col}")
    logger.info(f"  Horizon:   {cfg.forecast_horizon} steps")
    logger.info(f"  Features:  {len(selected_features)}")
    logger.info(f"  Hidden:    {cfg.hidden_size}")
    logger.info(f"  Mamba Dim: d_state={cfg.d_state}, d_conv={cfg.d_conv}")
    logger.info(f"  Device:    {cfg.device} (AMP: {cfg.use_amp})")

    model = TimeMachineSignalModel(cfg)
    
    history = model.fit(
        cfg, train_df, val_df, selected_features,
        resume_from=args.resume
    )
    
    metrics = evaluate_model(model, test_df)

    elapsed = datetime.now() - start_time
    
    results = {
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'selected_features': selected_features,
        'history': history,
        'test_metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed.total_seconds(),
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
    logger.info(f"총 소요 시간: {elapsed}")

if __name__ == '__main__': 
    main()