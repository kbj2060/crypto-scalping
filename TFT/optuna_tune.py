"""
TFT 하이퍼파라미터 최적화 (Optuna)

우선순위별 실험:
    1순위 (데이터): input_window, forecast_horizon, target
    2순위 (모델): hidden_size, attention_heads, lstm_layers
    3순위 (손실): direction_loss_weight, large_move_weight

사용법:
    # 1순위만 실험 (데이터 관련)
    python TFT/optuna_tune.py --priority 1 --trials 30

    # 1+2순위 실험 (데이터 + 모델 구조)
    python TFT/optuna_tune.py --priority 2 --trials 50

    # 전체 실험 (1+2+3순위)
    python TFT/optuna_tune.py --priority 3 --trials 100

    # 특정 study 이어서 실행
    python TFT/optuna_tune.py --priority 2 --trials 30 --study-name my_study

    # 병렬 실행 (여러 터미널에서 동일 study-name 사용)
    python TFT/optuna_tune.py --priority 2 --trials 20 --study-name shared_study
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TFT.TFT_model import TFTSignalModel, TFTConfig
from core.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optuna 로그 레벨 조정 (너무 verbose하지 않도록)
optuna.logging.set_verbosity(optuna.logging.WARNING)


META_FEATURE_COLS = [
    'strategy_consensus', 'strategy_conviction', 'strategy_conflict',
    'momentum_regime', 'reversion_regime',
]


def load_and_split_data(data_path: str = 'data/training_features_5m.csv'):
    """데이터 로드 및 분할 (70/15/15)."""
    logger.info(f"데이터 로드: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    all_features = ULTIMATE_FEATURE_COLS.copy()
    available_meta = [c for c in META_FEATURE_COLS if c in df.columns]
    all_features.extend(available_meta)

    missing = [c for c in all_features if c not in df.columns]
    if missing:
        logger.warning(f"누락 피처 (제외됨): {missing}")
        all_features = [c for c in all_features if c in df.columns]

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"  Train: {len(train_df):,}행")
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    logger.info(f"  피처:  {len(all_features)}개")

    return train_df, val_df, test_df, all_features


def create_target_variants(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """
    타겟 변형 생성.
    
    Args:
        target_type: 'log_return' | 'avg_3' | 'direction'
    """
    df = df.copy()
    
    if target_type == 'log_return':
        # 기본 (이미 존재)
        pass
    elif target_type == 'avg_3':
        # 3봉 평균 수익률
        df['target'] = df['log_return'].rolling(window=3, min_periods=1).mean()
    elif target_type == 'direction':
        # 방향 분류 (-1, 0, 1)
        df['target'] = np.sign(df['log_return'])
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    return df


def objective(trial: optuna.Trial, priority: int, 
              train_df: pd.DataFrame, val_df: pd.DataFrame, 
              feature_cols: list) -> float:
    """
    Optuna objective 함수.
    
    Args:
        trial: Optuna trial
        priority: 1 (데이터만), 2 (데이터+모델), 3 (전체)
        train_df, val_df: 학습/검증 데이터
        feature_cols: 피처 컬럼 리스트
    
    Returns:
        validation loss (최소화 목표)
    """
    
    # ══════════════════════════════════════════════════════════
    # 1순위: 데이터 관련 하이퍼파라미터
    # ══════════════════════════════════════════════════════════
    input_window = trial.suggest_categorical('input_window', [48, 72, 96])
    forecast_horizon = trial.suggest_categorical('forecast_horizon', [1, 3, 6])
    target_type = trial.suggest_categorical('target_type', ['log_return', 'avg_3', 'direction'])
    
    # 타겟 변형 적용
    train_df_mod = create_target_variants(train_df, target_type)
    val_df_mod = create_target_variants(val_df, target_type)
    target_col = 'target' if target_type != 'log_return' else 'log_return'
    
    # ══════════════════════════════════════════════════════════
    # 2순위: 모델 구조 (priority >= 2일 때만)
    # ══════════════════════════════════════════════════════════
    if priority >= 2:
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2, 3])
    else:
        # 기본값 사용
        hidden_size = 64
        attention_heads = 4
        lstm_layers = 1
    
    # ══════════════════════════════════════════════════════════
    # 3순위: 손실 함수 (priority == 3일 때만)
    # ══════════════════════════════════════════════════════════
    if priority >= 3:
        direction_loss_weight = trial.suggest_float('direction_loss_weight', 0.1, 0.5, step=0.1)
        large_move_weight = trial.suggest_float('large_move_weight', 1.5, 3.0, step=0.5)
    else:
        # 기본값 사용
        direction_loss_weight = 0.3
        large_move_weight = 2.0
    
    # ══════════════════════════════════════════════════════════
    # Config 생성
    # ══════════════════════════════════════════════════════════
    config = TFTConfig(
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        num_features=len(feature_cols),
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        lstm_layers=lstm_layers,
        direction_loss_weight=direction_loss_weight,
        large_move_weight=large_move_weight,
        
        # 학습 설정 (빠른 실험을 위해 조정)
        max_epochs=50,  # Optuna 실험용으로 짧게
        patience=10,
        batch_size=256,
        learning_rate=3e-4,
        lr_scheduler='cosine',
        warmup_epochs=3,
        use_ema=True,
        use_amp=True,
        
        # 로깅 최소화
        log_dir=f'runs/optuna/trial_{trial.number}',
        log_every_n_steps=200,
        save_every_n_epochs=999,  # 저장 안 함
        
        model_dir=f'models/optuna/trial_{trial.number}',
        seed=42 + trial.number,  # 각 trial마다 다른 시드
    )
    
    # ══════════════════════════════════════════════════════════
    # 모델 학습
    # ══════════════════════════════════════════════════════════
    try:
        model = TFTSignalModel(config)
        
        # target_col 설정
        original_target = config.target_col
        config.target_col = target_col
        
        history = model.fit(train_df_mod, val_df_mod, feature_cols)
        
        # 최종 validation loss
        best_val_loss = min(history['val_loss'])
        
        # 방향 정확도도 기록 (참고용)
        best_val_dir_acc = max(history['val_direction_acc'])
        trial.set_user_attr('best_val_dir_acc', best_val_dir_acc)
        
        # Config 복원
        config.target_col = original_target
        
        logger.info(f"Trial {trial.number} 완료: val_loss={best_val_loss:.6f}, "
                   f"dir_acc={best_val_dir_acc:.1%}")
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} 실패: {e}")
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description='TFT Optuna 하이퍼파라미터 최적화')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv',
                       help='학습 데이터 경로')
    parser.add_argument('--priority', type=int, default=1, choices=[1, 2, 3],
                       help='실험 우선순위 (1=데이터만, 2=데이터+모델, 3=전체)')
    parser.add_argument('--trials', type=int, default=30,
                       help='실험 횟수')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Study 이름 (이어서 실행 시 동일 이름 사용)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_tft.db',
                       help='Optuna storage (SQLite DB 경로)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='최대 실행 시간 (초)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🔬 TFT Optuna 하이퍼파라미터 최적화")
    print("=" * 80)
    print(f"우선순위: {args.priority} ({'데이터' if args.priority == 1 else '데이터+모델' if args.priority == 2 else '전체'})")
    print(f"실험 횟수: {args.trials}")
    print(f"Storage: {args.storage}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 데이터 로드
    train_df, val_df, test_df, feature_cols = load_and_split_data(args.data)
    
    # Study 이름 생성
    if args.study_name is None:
        study_name = f"tft_priority{args.priority}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        study_name = args.study_name
    
    logger.info(f"\nStudy 이름: {study_name}")
    
    # Optuna Study 생성
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,  # 기존 study가 있으면 이어서 실행
        direction='minimize',  # validation loss 최소화
        sampler=sampler,
        pruner=pruner,
    )
    
    # 최적화 실행
    logger.info(f"\n최적화 시작 ({args.trials} trials)...\n")
    
    study.optimize(
        lambda trial: objective(trial, args.priority, train_df, val_df, feature_cols),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # ══════════════════════════════════════════════════════════
    # 결과 출력
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 최적화 결과")
    print("=" * 80)
    
    print(f"\n✅ 완료된 trials: {len(study.trials)}")
    print(f"✅ 최적 trial: {study.best_trial.number}")
    print(f"✅ 최적 validation loss: {study.best_value:.6f}")
    
    if 'best_val_dir_acc' in study.best_trial.user_attrs:
        print(f"✅ 최적 방향 정확도: {study.best_trial.user_attrs['best_val_dir_acc']:.1%}")
    
    print("\n📋 최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key:25s} = {value}")
    
    # ══════════════════════════════════════════════════════════
    # 중요도 분석
    # ══════════════════════════════════════════════════════════
    if len(study.trials) >= 10:
        print("\n📊 하이퍼파라미터 중요도:")
        try:
            importance = optuna.importance.get_param_importances(study)
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(imp * 50)
                print(f"  {param:25s} {imp:6.1%} {bar}")
        except Exception as e:
            logger.warning(f"중요도 계산 실패: {e}")
    
    # ══════════════════════════════════════════════════════════
    # 결과 저장
    # ══════════════════════════════════════════════════════════
    results_dir = 'results/optuna'
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'study_name': study_name,
        'priority': args.priority,
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_dir_acc': study.best_trial.user_attrs.get('best_val_dir_acc'),
        'timestamp': datetime.now().isoformat(),
        'elapsed': str(datetime.now() - start_time),
    }
    
    results_path = os.path.join(results_dir, f'{study_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n결과 저장: {results_path}")
    
    # ══════════════════════════════════════════════════════════
    # 시각화 (optuna-dashboard 안내)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📈 시각화")
    print("=" * 80)
    print(f"Optuna Dashboard 실행:")
    print(f"  optuna-dashboard {args.storage}")
    print(f"\n또는 Python에서:")
    print(f"  import optuna")
    print(f"  study = optuna.load_study(study_name='{study_name}', storage='{args.storage}')")
    print(f"  optuna.visualization.plot_optimization_history(study).show()")
    print(f"  optuna.visualization.plot_param_importances(study).show()")
    print("=" * 80)
    
    print(f"\n🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print("=" * 80)


if __name__ == '__main__':
    main()
