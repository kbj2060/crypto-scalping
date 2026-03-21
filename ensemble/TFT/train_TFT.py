"""
TFT TrendBrain 학습 스크립트 (Trend Brain Edition)
================================================================================
- 역할: 5분봉 16시간(192봉) 컨텍스트 → 향후 30분(6봉) VWAP 방향 예측
- 타겟: VWAP 기반 수익률 (pct_change 대비 조작/스파이크에 강건)
- 평가: CRPS, PICP (분포), direction_accuracy (보조), TrendSignal 출력 테스트
- 저장: data/tft/tft_best.pt  + data/tft/tft_best_meta.json
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

# ────────────────────────────────────────────────────────────────
# 기본 설정 — Trend Brain 특화
# ────────────────────────────────────────────────────────────────
TREND_INPUT_WINDOW   = 96    # 5m × 96 = 8시간 컨텍스트
TREND_HORIZON        = 1     # 단일 VWAP 타겟 (forecast_horizon=1)
TREND_TARGET_COL     = 'target_ret_12'  # 12봉(1시간) VWAP 수익률
TREND_MAX_FEATURES   = 36
TREND_MODEL_DIR      = 'data/tft'

# Trend Brain 특화 must_include (현재 데이터셋에 존재하는 피처)
TREND_MUST_INCLUDE = [
    'rsi', 'mtf_trend_1h', 'mtf_trend_4h',
    'hurst_48', 'regime_trending', 'hurst_change',
    'garman_klass_vol', 'volatility_z', 'bb_width_z',
    'smart_money_flow', 'whale_conviction', 'net_taker_ratio',
    'oi_change_rate', 'funding_z_score', 'btc_corr_60',
    # RL 핵심 5개만
    'regime_bull', 'regime_bear', 'regime_chop',
    'garch_vol_z', 'jump_flag',
]

# RL 학습 데이터에서 가져온 추가 피처 후보
RL_FEATURE_CANDIDATES = [
    # 앙상블 pred_*/conf_* 제거 (시간적 분포 이동으로 오버피팅 유발)
    # 엘리트 신호
    'sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze',
    'sig_garch_regime', 'sig_ou_mean_rev', 'sig_jump_rebound', 'sig_evt_tail',
    # 레짐 상태
    'regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal',
    # 변동성 모델
    'garch_vol', 'garch_vol_z', 'ou_funding_z', 'ou_halflife',
    'jump_flag', 'jump_z', 'evt_tail_flag', 'evt_excess_z',
    # 합성 알파
    'cada', 'mshd', 'fvci', 'wpad', 'fdlv',
    'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz',
]


def build_vwap_targets(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP 기반 미래 수익률 타겟 생성.

    VWAP = Volume-Weighted Average Price → 가격 스파이크에 강건,
    실제 체결 가격을 더 잘 반영한다 (optuna_tune.py 방식 동일).

    생성 컬럼: target_ret_1, target_ret_6  (%)
    """
    tp     = (df['high'] + df['low'] + df['close']) / 3.0
    tp_vol = tp * df['volume']

    for h in [1, 6, 12]:
        col = f'target_ret_{h}'
        if col not in df.columns:
            fut_tp_vol = tp_vol.rolling(window=h).sum().shift(-h)
            fut_vol    = df['volume'].rolling(window=h).sum().shift(-h)
            fut_tp     = tp.rolling(window=h).mean().shift(-h)
            fut_vwap   = np.where(
                fut_vol == 0,
                fut_tp,
                fut_tp_vol / fut_vol.replace(0, np.nan),
            )
            df[col] = (fut_vwap / df['close'] - 1.0) * 100.0   # % 단위
    return df


def load_data(cfg: TFTConfig, path: str = 'data/training_features_5m.csv'):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # MTF 피처 (없으면 생성)
    if 'mtf_trend_1h' not in df.columns:
        df['ema_1h'] = df['close'].ewm(span=12).mean()
        df['mtf_trend_1h'] = (df['close'] / df['ema_1h']) - 1
    if 'mtf_trend_4h' not in df.columns:
        df['ema_4h'] = df['close'].ewm(span=48).mean()
        df['mtf_trend_4h'] = (df['close'] / df['ema_4h']) - 1

    # regime_break (없으면 0)
    if 'regime_break' not in df.columns:
        df['regime_break'] = 0.0

    # VWAP 기반 타겟 생성
    df = build_vwap_targets(df)
    df.dropna(inplace=True)

    # 피처 후보: ULTIMATE_FEATURE_COLS + MTF + RL 특화 컬럼
    combined = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    for c in ['mtf_trend_1h', 'mtf_trend_4h']:
        if c in df.columns and c not in combined:
            combined.append(c)
    for c in RL_FEATURE_CANDIDATES:
        if c in df.columns and c not in combined:
            combined.append(c)
    all_features = list(dict.fromkeys(combined))

    logger.info(f"  ✓ {len(df):,}행, {len(all_features)}개 피처 후보, target={cfg.target_col}")
    return df, all_features


def split_data_with_embargo(df: pd.DataFrame, cfg: TFTConfig,
                             train_ratio: float = 0.70,
                             val_ratio:   float = 0.15):
    """시간 순서 유지 분할 + Embargo 갭 (데이터 누출 방지)."""
    embargo   = cfg.input_window + cfg.forecast_horizon
    n         = len(df)
    train_end = int(n * train_ratio)
    val_start = train_end + embargo
    val_end   = int(n * (train_ratio + val_ratio))
    test_start = val_end + embargo

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[val_start:val_end].copy()
    test_df  = df.iloc[test_start:].copy()

    logger.info(f"  Train: {len(train_df):,}행  |  Val: {len(val_df):,}행  |  Test: {len(test_df):,}행")
    logger.info(f"  🔒 Embargo {embargo}봉 적용")
    return train_df, val_df, test_df


def evaluate_model(model: TFTSignalModel, test_df: pd.DataFrame) -> dict:
    logger.info("\n📊 테스트셋 성능 산출 중...")
    cfg = model.config

    if cfg.target_col not in test_df.columns:
        logger.warning(f"타겟 '{cfg.target_col}' 부재 → 평가 스킵")
        return {}

    result = model.predict(test_df)
    if not result:
        return {}

    median_pred = result['median_pred']          # (N, H)
    lower_pred  = result['quantiles'][:, :, 0]  # 0.05
    upper_pred  = result['quantiles'][:, :, -1] # 0.95
    actual      = test_df[cfg.target_col].values
    n_samples   = len(test_df) - (cfg.input_window + cfg.forecast_horizon) + 1

    actual_seqs = np.array([
        actual[i + cfg.input_window : i + cfg.input_window + cfg.forecast_horizon]
        for i in range(n_samples)
    ])

    mae           = float(np.mean(np.abs(median_pred - actual_seqs)))
    rmse          = float(np.sqrt(np.mean((median_pred - actual_seqs) ** 2)))
    direction_acc = float(np.mean(np.sign(median_pred) == np.sign(actual_seqs)))
    picp          = float(np.mean((actual_seqs >= lower_pred) & (actual_seqs <= upper_pred)))

    errors    = actual_seqs[..., None] - result['quantiles']
    q_tensor  = np.array(cfg.quantiles).reshape(1, 1, -1)
    crps      = float(np.mean(np.maximum((q_tensor - 1) * errors, q_tensor * errors)))

    metrics = {
        'mae': mae, 'rmse': rmse, 'crps_proxy': crps,
        'picp_90': picp, 'direction_accuracy': direction_acc,
        'test_samples': n_samples,
    }

    logger.info("=" * 60)
    logger.info(f"  MAE: {mae:.5f}  |  RMSE: {rmse:.5f}")
    logger.info(f"  CRPS: {crps:.5f}  |  PICP(90%): {picp:.1%}  |  DirAcc: {direction_acc:.1%}")
    logger.info("=" * 60)

    if 'variable_importance' in result:
        vi = result['variable_importance'].mean(axis=(0, 1))
        if len(vi) == len(model.feature_cols):
            imp = sorted(zip(model.feature_cols, vi), key=lambda x: x[1], reverse=True)
            logger.info("🔥 Top 10 피처:")
            for feat, score in imp[:10]:
                logger.info(f"  {feat:30s} {score:.4f} {'█' * int(score * 100)}")

    # TrendSignal 출력 확인 (smoke test)
    try:
        ts = model.predict_from_df(test_df.tail(cfg.input_window + 50))
        if ts is not None:
            logger.info(f"\n✅ TrendSignal smoke test: dir={ts.trend_dir} str={ts.strength:.3f} rev={ts.rev_prob:.3f}")
        else:
            logger.warning("⚠️ TrendSignal smoke test: None 반환 (데이터 부족?)")
    except Exception as e:
        logger.warning(f"⚠️ TrendSignal smoke test 실패: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='TFT TrendBrain 학습')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-best-optuna', type=str, default=None)
    parser.add_argument('--data', type=str, default='data/training_features_5m.csv')
    parser.add_argument('--input-window', type=int, default=TREND_INPUT_WINDOW,
                        help=f'입력 캔들 수 (기본: {TREND_INPUT_WINDOW} = 8h)')
    parser.add_argument('--max-features', type=int, default=TREND_MAX_FEATURES)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    # ── Trend Brain 특화 Config ──────────────────────────────────
    base_cfg = TFTConfig(
        input_window     = args.input_window,
        forecast_horizon = TREND_HORIZON,
        target_col       = TREND_TARGET_COL,
        hidden_size      = args.hidden,
        num_features     = args.max_features,   # 피처 선택 후 갱신됨
        max_epochs       = args.epochs,
        patience         = args.patience,
        model_dir        = TREND_MODEL_DIR,
        # 안정적인 학습 설정
        learning_rate    = 3e-5,
        weight_decay     = 1e-3,
        dropout          = 0.45,
        batch_size       = 256,
        use_ema          = True,
        use_amp          = True,
        warmup_epochs    = 20,
    )

    print("\n" + "=" * 80)
    print("🚀 TFT TrendBrain 학습 (5m × 8h → 1h VWAP 방향)")
    print(f"   input_window={base_cfg.input_window}봉 | horizon={base_cfg.forecast_horizon} | target={base_cfg.target_col}")
    print("=" * 80)
    start_time = datetime.now()

    df, feature_cols = load_data(base_cfg, args.data)
    train_df, val_df, test_df = split_data_with_embargo(df, base_cfg)

    # 피처 선택 — Trend Brain 전용 must_include
    must = [c for c in TREND_MUST_INCLUDE if c in feature_cols]
    selected_features = auto_select_features(
        train_df, feature_cols,
        target_col     = base_cfg.target_col,
        max_features   = args.max_features,
        corr_threshold = 0.85,
        must_include   = must,
    )

    cfg = TFTConfig(
        input_window     = base_cfg.input_window,
        forecast_horizon = base_cfg.forecast_horizon,
        target_col       = base_cfg.target_col,
        hidden_size      = base_cfg.hidden_size,
        num_features     = len(selected_features),
        max_epochs       = base_cfg.max_epochs,
        patience         = base_cfg.patience,
        model_dir        = base_cfg.model_dir,
        learning_rate    = base_cfg.learning_rate,
        weight_decay     = base_cfg.weight_decay,
        dropout          = base_cfg.dropout,
        batch_size       = base_cfg.batch_size,
        use_ema          = base_cfg.use_ema,
        use_amp          = base_cfg.use_amp,
        warmup_epochs    = base_cfg.warmup_epochs,
    )

    logger.info(f"\n=== Trend Brain Config ===")
    logger.info(f"  target:   {cfg.target_col}")
    logger.info(f"  window:   {cfg.input_window}봉 ({cfg.input_window * 5 // 60}h)")
    logger.info(f"  horizon:  {cfg.forecast_horizon}스텝")
    logger.info(f"  features: {len(selected_features)}")
    logger.info(f"  hidden:   {cfg.hidden_size}")
    logger.info(f"  save:     {cfg.model_dir}/tft_best.pt")

    model = TFTSignalModel(cfg)
    history = model.fit(
        cfg, train_df, val_df, selected_features,
        resume_from    = args.resume,
        warm_start_path = args.resume_best_optuna,
    )

    metrics = evaluate_model(model, test_df)

    # 결과 저장
    os.makedirs(cfg.model_dir, exist_ok=True)
    results_path = os.path.join(cfg.model_dir, 'training_results.json')

    def _convert(obj):
        if isinstance(obj, (np.integer, int)):   return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, np.ndarray):           return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump({
            'config':            {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
            'selected_features': selected_features,
            'history':           history,
            'test_metrics':      metrics,
            'timestamp':         datetime.now().isoformat(),
            'elapsed':           str(datetime.now() - start_time),
        }, f, indent=2, default=_convert)

    logger.info(f"결과 저장: {results_path}")
    print("\n" + "=" * 80)
    print(f"🎉 완료! 소요 시간: {datetime.now() - start_time}")
    print(f"   모델: {cfg.model_dir}/tft_best.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()
