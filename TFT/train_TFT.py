"""
TFT Signal Model 학습 스크립트

[IDEA 7] 앙상블 학습 지원 (--ensemble 플래그)
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

def load_data(path: str = 'data/training_features_5m.csv'):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])

    # [Fix] 타겟 컬럼 부재 시 자동 생성
    required_targets = ['target_ret_12', 'target_ret_6', 'target_cumret_6']
    if not all(col in df.columns for col in required_targets):
        logger.info("타겟 컬럼 일부 부재 -> 자동 생성 중...")
        # shift(-N)은 미래 N시점을 의미
        df['target_ret_6'] = (df['close'].shift(-6) / df['close'] - 1)
        df['target_ret_12'] = (df['close'].shift(-12) / df['close'] - 1)  # 1시간

        if 'target_cumret_6' not in df.columns:
             # 보통 cumret과 ret_6는 거의 동일 (여기서는 ret_6 사용)
             df['target_cumret_6'] = df['target_ret_6']
    
    # [Fix] regime_break 부재 시 0으로 채움
    if 'regime_break' not in df.columns:
        logger.warning("'regime_break' 컬럼 부재 -> 0으로 채움 (주의)")
        df['regime_break'] = 0.0

    all_features = ULTIMATE_FEATURE_COLS.copy()

    missing = [c for c in all_features if c not in df.columns]
    if missing:
        logger.warning(f"누락 피처 (제외됨): {missing}")
        all_features = [c for c in all_features if c in df.columns]

    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 피처")
    logger.info(f"  ✓ 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df, all_features

def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, 
               use_augmentation=False):  # ← 파라미터 추가
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # ★ 학습 데이터 증강 (선택적)
    if use_augmentation:
        try:
            from core.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            if hasattr(fe, 'augment_training_data'):
                train_df = fe.augment_training_data(train_df, noise_level=0.02)
                logger.info(f"  Train: {len(train_df):,}행 (증강 적용)")
            else:
                logger.warning("augment_training_data 메서드 없음 - 증강 스킵")
                logger.info(f"  Train: {len(train_df):,}행")
        except Exception as e:
            logger.warning(f"Data augmentation 실패: {e}")
            logger.info(f"  Train: {len(train_df):,}행")
    else:
        logger.info(f"  Train: {len(train_df):,}행")
    
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    return train_df, val_df, test_df

def train_single_model(config, train_df, val_df, feature_cols, resume):
    """단일 TFT 모델 학습 및 반환"""
    model = TFTSignalModel(config)
    history = model.fit(config, train_df, val_df, feature_cols, resume)
    return model, history

def evaluate_model(model: TFTSignalModel, test_df: pd.DataFrame):
    logger.info("\n테스트셋 평가 중...")
    cfg = model.config
    
    # 타겟 컬럼이 테스트 데이터에 있는지 확인
    if cfg.target_col not in test_df.columns:
        logger.warning(f"테스트셋에 타겟 컬럼 '{cfg.target_col}' 부재로 평가 스킵")
        return {}

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
    # confidence valid_mask 적용
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
    if 'variable_importance' in result:
        vi = result['variable_importance'].mean(axis=(0))
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
    parser.add_argument('--resume', type=str, default=None, help='이어서 학습할 체크포인트 경로')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv', help='학습 데이터 경로')
    parser.add_argument('--ensemble', action='store_true', help='앙상블용 3개 모델 학습')
    parser.add_argument('--horizon', type=int, default=6, choices=[1,3,6], help='예측 horizon (앙상블 아닐 때)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 TFT Signal Model 학습 시작" + (" (Ensemble Mode)" if args.ensemble else ""))
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(args.data)
    train_df, val_df, test_df = split_data(df)

    # 피처 선택 (공통)
    target_for_selection = 'target_ret_12'
    if target_for_selection not in df.columns:
         # 만약 target_cumret_6가 없다면 target_ret_6 시도
         if 'target_ret_12' in df.columns:
             target_for_selection = 'target_ret_12'
         else:
             logger.warning("타겟 컬럼(target_ret_12) 없음. 첫번째 타겟 사용.")
             # 임시로 첫번째 타겟 사용
             target_for_selection = [c for c in df.columns if 'target' in c][0]

    selected_features = auto_select_features(
        train_df,
        feature_cols,
        target_col=target_for_selection,
        max_features=25,
        corr_threshold=0.9,
        must_include=[
          # 스마트 머니 (3)
          'whale_conviction', 'smart_money_flow', 'big_trade_ratio', 
          
          # 펀딩 신호 (3)
          'long_squeeze_risk', 'short_squeeze_risk', 'funding_z_score', 'funding_price_divergence',
          
          # 오더 플로우 (3)
          'net_taker_ratio', 'oi_change_rate', 'ofi_acceleration', 'trade_intensity',
          
          # 레짐 감지 (3)
          'regime_trending', 'regime_mean_reverting', 'hurst_48', 'regime_break', 'fvg_dist',
          
          # 변동성 (2)
          'volatility_z', 'garman_klass_vol', 'vwap_dist', 'log_return',
          
          # 전략 메타 (1)
          'strategy_consensus'
      ]
    )
    logger.info(f"선택된 피처 ({len(selected_features)}개): {selected_features}")

    if args.ensemble:
        # [IDEA 7] 3개 horizon에 대한 모델 학습
        configs = {
            3: TFTConfig(target_col='target_ret_3'),
            6: TFTConfig(target_col='target_ret_6'),
            12: TFTConfig(target_col='target_ret_12')
          }
        
        # 타겟 컬럼 존재 확인
        for h, cfg in configs.items():
            if cfg.target_col not in df.columns:
                logger.error(f"Target column {cfg.target_col} not found in dataframe!")
                return

        model_paths = []
        for h, cfg in configs.items():
            logger.info(f"\n" + "="*40)
            logger.info(f"🔄 Training Horizon-{h} Model (Target: {cfg.target_col})")
            logger.info("="*40)
            
            cfg.num_features = len(selected_features)   # 피처 수 동기화
            
            # 모델 디렉토리 생성
            os.makedirs(cfg.model_dir, exist_ok=True)
            
            model, _ = train_single_model(cfg, train_df, val_df, selected_features, args.resume)
            
            # 저장
            path = os.path.join(cfg.model_dir, f'tft_horizon{h}_final.pt')
            
            # 수동 저장 (TFT_model.py에 save 메서드가 없으므로 직접 state dict 저장)
            state = {
                'model_state_dict': model.model.state_dict(),
                'feature_cols': selected_features,
                'scaler_params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in model.scaler_params.items()},
                'config': cfg.__dict__
            }
            if model.ema:
                state['ema_state_dict'] = model.ema.state_dict()
                
            torch.save(state, path)
            
            # 메타데이터 별도 저장
            meta_path = os.path.join(cfg.model_dir, f'tft_horizon{h}_final_meta.json')
            meta = {
                'feature_cols': selected_features,
                'scaler_params': state['scaler_params'],
                'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"모델 저장 완료: {path}")
            model_paths.append(path)
            
            # 개별 평가
            evaluate_model(model, test_df)

        logger.info(f"\n✅ 앙상블 학습 완료. 모델 경로: {model_paths}")
        
    else:
        # 단일 모델 학습 (기존 로직)
        # target_ret_X가 있으면 사용, 없으면 target_cumret_6 (기존 default)
        target = f'target_ret_{args.horizon}'
        if target not in df.columns:
            logger.warning(f"{target} 없음. target_cumret_6 사용.")
            target = 'target_cumret_6'
            
        cfg = TFTConfig(forecast_horizon=args.horizon, target_col=target)
        cfg.num_features = len(selected_features)
        
        logger.info(f"\n=== Training Single Model (Horizon={args.horizon}, Target={target}) ===")
        model, history = train_single_model(cfg, train_df, val_df, selected_features, args.resume)
        
        model._save_checkpoint('final')
        metrics = evaluate_model(model, test_df)

        # 결과 저장
        results = {
            'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
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
