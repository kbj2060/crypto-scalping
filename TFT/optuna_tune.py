"""
TFT 하이퍼파라미터 최적화 (Optuna) - 피처 개수 자동 선택 + target_cumret 제거

우선순위별 실험:
    1순위 (데이터): input_window, forecast_horizon, max_features
    2순위 (모델): hidden_size, attention_heads, lstm_layers
    3순위 (손실): direction_loss_weight, large_move_weight

사용법:
    # 1순위만 실험 (데이터 관련)
    python TFT/optuna_tune.py --priority 1 --trials 30

    # 1+2순위 실험 (데이터 + 모델 구조)
    python TFT/optuna_tune.py --priority 2 --trials 50

    # 전체 실험 (1+2+3순위)
    python TFT/optuna_tune.py --priority 3 --trials 100
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
from core.feature_selector import auto_select_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_split_data(data_path: str = 'data/training_features_5m.csv'):
    """데이터 로드 및 분할 (70/15/15). target_cumret 생성 로직 제거"""
    logger.info(f"데이터 로드: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    # 필요한 타겟 컬럼만 생성 (target_cumret 제외)
    if 'target_ret_6' not in df.columns:
        df['target_ret_6'] = (df['close'].shift(-6) / df['close'] - 1)
    if 'target_ret_12' not in df.columns:
        df['target_ret_12'] = (df['close'].shift(-12) / df['close'] - 1)
    if 'target_ret_3' not in df.columns:
        df['target_ret_3'] = (df['close'].shift(-3) / df['close'] - 1)
    # regime_break는 그대로 유지
    if 'regime_break' not in df.columns:
        df['regime_break'] = 0.0

    # 사용 가능한 모든 피처
    all_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    missing = [c for c in ULTIMATE_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"누락 피처 (제외됨): {missing[:10]}...")

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"  Train: {len(train_df):,}행")
    logger.info(f"  Val:   {len(val_df):,}행")
    logger.info(f"  Test:  {len(test_df):,}행")
    logger.info(f"  전체 피처 후보: {len(all_features)}개")

    return train_df, val_df, test_df, all_features


def get_target_column(forecast_horizon: int) -> str:
    """forecast_horizon에 따라 타겟 컬럼명 반환 (target_ret_X만 사용)"""
    return f'target_ret_{forecast_horizon}'


def objective(trial: optuna.Trial, priority: int,
              train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list) -> float:
    """
    Optuna objective 함수. target_type 파라미터 제거.
    """
    # ══════════════════════════════════════════════════════════
    # 1순위: 데이터 관련 하이퍼파라미터
    # ══════════════════════════════════════════════════════════
    if priority >= 2:
        # 1순위 결과 고정 (이전 최적값)
        input_window = 48
        forecast_horizon = 3
        max_features = 15
    else:
        # priority 1일 때만 탐색 (여기서는 사용 안 함)
        input_window = trial.suggest_categorical('input_window', [48, 64, 96])
        forecast_horizon = trial.suggest_categorical('forecast_horizon', [1, 3, 6])
        max_features = trial.suggest_int('max_features', 10, 40, step=5)

    target_col = f'target_ret_{forecast_horizon}'   # horizon=3 → target_ret_3

    # 피처 선택 (max_features 고정)
    must_include = [
        'whale_conviction', 'net_taker_ratio', 'oi_change_rate',
        'funding_z_score', 'hurst_48', 'regime_trending',
        'volatility_z', 'garman_klass_vol'
    ]
    selected_features = auto_select_features(
        train_df,
        feature_cols,
        target_col=target_col,
        max_features=max_features,
        corr_threshold=0.85,
        must_include=must_include
    )

    # ══════════════════════════════════════════════════════════
    # 2순위: 모델 구조 (priority >= 2일 때만)
    # ══════════════════════════════════════════════════════════
    if priority >= 2:
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2])
    else:
        hidden_size = 32
        attention_heads = 8
        lstm_layers = 2

    # ══════════════════════════════════════════════════════════
    # 3순위: 손실 함수 (priority == 3일 때만)
    # ══════════════════════════════════════════════════════════
    if priority >= 3:
        direction_loss_weight = trial.suggest_float('direction_loss_weight', 1.0, 5.0, step=0.5)
        large_move_weight = trial.suggest_float('large_move_weight', 1.0, 3.0, step=0.5)
    else:
        direction_loss_weight = 1.0
        large_move_weight = 2.0

    # ══════════════════════════════════════════════════════════
    # Config 생성
    # ══════════════════════════════════════════════════════════
    config = TFTConfig(
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        lstm_layers=lstm_layers,
        direction_loss_weight=direction_loss_weight,
        large_move_weight=large_move_weight,
        num_features=len(selected_features),

        # 학습 설정 (빠른 실험용)
        max_epochs=50,
        patience=10,
        batch_size=256,
        learning_rate=3e-4,
        lr_scheduler='cosine',
        warmup_epochs=5,
        use_ema=True,
        use_amp=True,

        log_dir=f'logs/optuna/trial_{trial.number}',
        log_every_n_steps=200,
        save_every_n_epochs=999,
        model_dir=f'models/optuna/trial_{trial.number}',
        seed=42 + trial.number,
    )

    # ══════════════════════════════════════════════════════════
    # 모델 학습
    # ══════════════════════════════════════════════════════════
    try:
        model = TFTSignalModel(config)
        history = model.fit(
            cfg=config,
            train_df=train_df,
            val_df=val_df,
            feature_cols=selected_features,
            resume_from=None
        )

        best_val_loss = min(history['val_loss'])
        best_val_dir_acc = max(history['val_direction_acc'])
        trial.set_user_attr('best_val_dir_acc', best_val_dir_acc)

        logger.info(f"Trial {trial.number} 완료: val_loss={best_val_loss:.6f}, "
                   f"dir_acc={best_val_dir_acc:.1%}, features={len(selected_features)}")
        return best_val_dir_acc

    except Exception as e:
        logger.error(f"Trial {trial.number} 실패: {e}")
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description='TFT Optuna 하이퍼파라미터 최적화')
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    parser.add_argument('--priority', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_tft.db')
    parser.add_argument('--timeout', type=int, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔬 TFT Optuna 하이퍼파라미터 최적화 (피처 개수 자동 선택, target_cumret 제거)")
    print("=" * 80)
    print(f"우선순위: {args.priority}")
    print(f"실험 횟수: {args.trials}")
    print("=" * 80)

    start_time = datetime.now()
    train_df, val_df, test_df, feature_cols = load_and_split_data(args.data)

    if args.study_name is None:
        study_name = f"tft_priority{args.priority}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        study_name = args.study_name

    logger.info(f"\nStudy 이름: {study_name}")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    logger.info(f"\n최적화 시작 ({args.trials} trials)...\n")
    study.optimize(
        lambda trial: objective(trial, args.priority, train_df, val_df, feature_cols),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # 결과 출력
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

    # 중요도 분석 (생략 가능)

    # 결과 저장
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

    print(f"\n🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print("=" * 80)


if __name__ == '__main__':
    main()