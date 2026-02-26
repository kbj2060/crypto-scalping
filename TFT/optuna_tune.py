"""
TFT 하이퍼파라미터 최적화 (Optuna) - SOTA 데이터 증강 & VWAP 타겟 반영 버전

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
    """데이터 로드 및 분할 (70/15/15) - VWAP 타겟 및 MTF 피처 적용"""
    logger.info(f"데이터 로드: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    # 🚨 1. 무한대 값 등 에러 유발 인자 사전 차단
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 🚨 2. MTF (Multi-Timeframe) 피처 생성
    df['ema_1h'] = df['close'].ewm(span=12).mean()
    df['ema_4h'] = df['close'].ewm(span=48).mean()
    df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    # 🚨 3. 미래 N봉 VWAP(거래량 가중 평균) 타겟 동적 생성
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = tp * df['volume']

    # Optuna가 탐색할 수 있는 모든 horizon(1, 3, 6)에 대해 정답지 미리 생성
    for h in [1, 3, 6]:
        col_name = f'target_ret_{h}'
        if col_name not in df.columns:
            future_tp_vol_sum = tp_vol.rolling(window=h).sum().shift(-h)
            future_vol_sum = df['volume'].rolling(window=h).sum().shift(-h)
            future_tp_avg = tp.rolling(window=h).mean().shift(-h)
            
            future_vwap = np.where(
                future_vol_sum == 0, 
                future_tp_avg, 
                future_tp_vol_sum / future_vol_sum.replace(0, np.nan)
            )
            df[col_name] = (future_vwap / df['close']) - 1

    if 'regime_break' not in df.columns:
        df['regime_break'] = 0.0

    # 🚨 4. 결측치(NaN) 완벽 제거 (Train NaN 방지)
    df.dropna(inplace=True)

    # 사용 가능한 모든 피처
    all_features = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    
    # 생성된 MTF 피처 강제 편입
    for c in ['mtf_trend_1h', 'mtf_trend_4h']:
        if c not in all_features:
            all_features.append(c)

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
    """forecast_horizon에 따라 타겟 컬럼명 반환"""
    return f'target_ret_{forecast_horizon}'


def objective(trial: optuna.Trial, priority: int,
              train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list) -> float:
    """Optuna objective 함수."""
    
    # 🚀 [추가] Priority 0: 모든 것을 한꺼번에 섞어서 동시 최적화 (Joint Optimization)
    if priority == 0:
        input_window = trial.suggest_categorical('input_window', [48, 64, 96])
        forecast_horizon = trial.suggest_categorical('forecast_horizon', [1, 3, 6, 12])
        max_features = trial.suggest_int('max_features', 10, 40, step=5)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2])
        direction_loss_weight = trial.suggest_float('direction_loss_weight', 3.0, 10.0, step=1.0)
        large_move_weight = trial.suggest_float('large_move_weight', 3.0, 10.0, step=1.0)
        
    else:
        # 기존 우선순위 로직 (1, 2, 3순위 개별 탐색)
        if priority >= 2:
            input_window = 48
            forecast_horizon = 3
            max_features = 15
        else:
            input_window = trial.suggest_categorical('input_window', [48, 64, 96])
            forecast_horizon = trial.suggest_categorical('forecast_horizon', [1, 3, 6])
            max_features = trial.suggest_int('max_features', 10, 40, step=5)

        if priority >= 2:
            hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
            attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
            lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2])
        else:
            hidden_size = 32
            attention_heads = 4
            lstm_layers = 2

        if priority >= 3:
            direction_loss_weight = trial.suggest_float('direction_loss_weight', 3.0, 10.0, step=1.0)
            large_move_weight = trial.suggest_float('large_move_weight', 3.0, 10.0, step=1.0)
        else:
            direction_loss_weight = 5.0
            large_move_weight = 5.0

    target_col = get_target_column(forecast_horizon)

    # 🚨 피처 선택에 MTF 피처 필수 포함
    must_include = [
        'whale_conviction', 'net_taker_ratio', 'oi_change_rate',
        'funding_z_score', 'hurst_48', 'regime_trending',
        'volatility_z', 'garman_klass_vol',
        'mtf_trend_1h', 'mtf_trend_4h' 
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
    # 2순위: 모델 구조
    # ══════════════════════════════════════════════════════════
    if priority >= 2:
        # 데이터 증강(Masking & Noise)으로 인해 뇌 용량(hidden_size)을 약간 키워도 과적합 방어 가능
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
        attention_heads = trial.suggest_categorical('attention_heads', [2, 4, 8])
        lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2])
    else:
        hidden_size = 32
        attention_heads = 4
        lstm_layers = 2

    # ══════════════════════════════════════════════════════════
    # 3순위: 손실 함수 (선형 패널티 스케일에 맞춰 조정)
    # ══════════════════════════════════════════════════════════
    if priority >= 3:
        # 이제 기본 가중치가 5.0이므로 탐색 범위를 상향 조정
        direction_loss_weight = trial.suggest_float('direction_loss_weight', 3.0, 10.0, step=1.0)
        large_move_weight = trial.suggest_float('large_move_weight', 3.0, 10.0, step=1.0)
    else:
        direction_loss_weight = 5.0
        large_move_weight = 5.0

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

        # 🚨 학습 설정: 데이터 증강 기법 때문에 수렴이 느리므로 patience를 늘림
        dropout=0.2,                 # 마스킹 효과 고려
        weight_decay=1e-3,           # 이중 정규화 방지
        max_epochs=100,              # 빠른 실험이지만 충분한 에포크 보장
        patience=30,                 # 증강된 데이터 수렴을 기다리기 위해 증가
        batch_size=256,
        learning_rate=1e-4,          # 안정적인 학습
        lr_scheduler='cosine',
        warmup_epochs=5,
        use_ema=True,
        use_amp=True,

        log_dir=f'logs/optuna/trial_{trial.number}',
        log_every_n_steps=500,       # 로그 너무 자주 찍히는 것 방지
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
    parser.add_argument('--priority', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_tft.db')
    parser.add_argument('--timeout', type=int, default=None)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔬 TFT Optuna 하이퍼파라미터 최적화 (SOTA 데이터 증강 & VWAP 타겟 반영)")
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
        direction='maximize',  # 방향성 적중률 최대화
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
    print(f"✅ 최적 validation loss (목적함수값): {study.best_value:.6f}")
    if 'best_val_dir_acc' in study.best_trial.user_attrs:
        print(f"✅ 최적 방향 정확도: {study.best_trial.user_attrs['best_val_dir_acc']:.1%}")
    print("\n📋 최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key:25s} = {value}")

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