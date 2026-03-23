"""
XGBTrendBrain 학습 스크립트
================================================================================
LightGBM 3-class 분류기 (DOWN / FLAT / UP)

타겟:  Triple-Barrier 레이블 (de Prado 2018) — 5m 봉 기준
피처:  ULTIMATE_FEATURE_COLS + MTF 피처 → auto_select_features (상위 N개)
튜닝:  Optuna 50회 시행 (CPU, ~1시간)
저장:  data/trend_xgb/trend_xgb.pkl

실행:
    cd /home/llewyn/crypto-scalping
    python ensemble/trend_xgb/train_trend_xgb.py
"""

import os
import sys
import json
import logging
import argparse

import hashlib
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed

optuna.logging.set_verbosity(optuna.logging.WARNING)

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR     = os.path.dirname(_ENSEMBLE_DIR)
for _p in [_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from core.feature_selector import auto_select_features
from core.feature_engineering import ULTIMATE_FEATURE_COLS
from ensemble.trend_xgb.trend_xgb_model import compute_atr, make_triple_barrier_label, XGBTrendBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/training_features_5m.csv'
RL_DATA_PATH = 'data/rl_training_data_full.csv'
SAVE_PATH   = 'data/trend_xgb/trend_xgb.pkl'
MAX_FEATURES = 48      # auto_select_features 상한
N_TRIALS     = 50      # Optuna 시행 수
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15

# Triple-Barrier 파라미터 (5m 봉 기준)
ATR_WINDOW_5M = 14     # 14봉 ATR (= 70분)
ATR_MULT      = 0.8    # 장벽 = ATR × 0.8 (수수료 커버 + 의미있는 방향만 레이블)
MAX_HOLD_5M   = 12     # 최대 보유 12봉 = 1시간  ← target_ret_12와 동일 호라이즌

# RL CSV에서 오는 컬럼들
RL_SIG_COLS = [   # elite strategy signals (+ NewEliteSignalEngine 3종)
    'sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze',
    'sig_garch_regime', 'sig_ou_mean_rev', 'sig_jump_rebound', 'sig_evt_tail',
    'sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health',
]
RL_ALPHA_COLS = [  # 합성 알파 + 모델 파생
    'garch_vol_z', 'ou_funding_z', 'ou_halflife',
    'jump_flag', 'jump_z', 'evt_tail_flag', 'evt_excess_z',
    'cada', 'mshd', 'fvci', 'wpad', 'fdlv',
    'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz',
    'regime_bull', 'regime_bear', 'regime_chop', 'regime_whipsaw', 'regime_normal',
]

MUST_INCLUDE = [
    # 레짐/변동성 축
    'volatility_z', 'garman_klass_vol', 'bb_width_z',
    'hurst_48', 'regime_trending', 'hurst_change',
    # 추세 구조 축
    'ret_12', 'trend_accel', 'hh_count_24', 'hl_count_24',
    # 미시구조 축
    'net_taker_ratio', 'oi_change_rate', 'smart_money_flow', 'btc_corr_60',
    # AI 앙상블 결합 축
    'signal_timesfm', 'signal_chronos', 'signal_mdjd',
    # New Elite 핵심 축
    'sig_volume_confirm', 'sig_trend_health',
]


def _build_config_hash(max_features: int) -> str:
    cfg = {
        'atr_mult': ATR_MULT,
        'max_hold_5m': MAX_HOLD_5M,
        'max_features': int(max_features),
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'must_include': sorted(MUST_INCLUDE),
        'rl_sig_cols': sorted(RL_SIG_COLS),
    }
    payload = json.dumps(cfg, ensure_ascii=True, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────
# 데이터 로드 및 전처리
# ────────────────────────────────────────────────────────────────
def _add_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """추세 구조 전용 피처 생성 (MoE가 보지 않는 관점).

    - ret_12/24/48  : 다중 호라이즌 과거 수익률 (tanh 정규화)
    - hh_count_24   : 24봉 내 Higher-High 발생 횟수
    - hl_count_24   : 24봉 내 Higher-Low 발생 횟수
    - trend_accel   : 추세 기울기의 기울기 (가속/감속)
    """
    c = df['close']
    h = df['high']  if 'high'  in df.columns else c
    l = df['low']   if 'low'   in df.columns else c

    # 다중 호라이즌 수익률
    df['ret_12'] = np.tanh(c.pct_change(12) * 10)
    df['ret_24'] = np.tanh(c.pct_change(24) * 10)
    df['ret_48'] = np.tanh(c.pct_change(48) * 10)

    # Higher-High / Higher-Low 카운트 (24봉 롤링)
    h_shift = h.shift(1)
    l_shift = l.shift(1)
    hh = (h > h_shift).astype(float)
    hl = (l > l_shift).astype(float)
    df['hh_count_24'] = hh.rolling(24, min_periods=1).sum() / 24.0
    df['hl_count_24'] = hl.rolling(24, min_periods=1).sum() / 24.0

    # 추세 가속도: 단기 기울기 - 장기 기울기
    slope_12 = c.pct_change(12)
    slope_48 = c.pct_change(48) / 4   # 봉 수 차이 보정 → 봉당 기울기
    df['trend_accel'] = np.tanh((slope_12 - slope_48) * 20)

    return df


def _combine_pred_conf(df: pd.DataFrame) -> pd.DataFrame:
    """pred_x × conf_x → signal_x (확신도 가중 방향 신호, 1개로 결합).

    raw pred/conf 컬럼을 제거하고 signal_* 컬럼만 남겨 feature 수를 절반으로 줄이고
    다중공선성(pred↔conf)을 차단한다.
    """
    PRED_TO_CONF = {
        'pred_timesfm': 'conf_timesfm', 'pred_chronos': 'conf_chronos',
        'pred_ttm':     'conf_ttm',     'pred_patchtst': 'conf_patchtst',
        'pred_tide':    'conf_tide',    'pred_mdjd':    'conf_mdjd',
        'pred_ridge':   'conf_ridge',
    }
    created = []
    for p_col, c_col in PRED_TO_CONF.items():
        sig_col = p_col.replace('pred_', 'signal_')
        if p_col in df.columns and c_col in df.columns:
            df[sig_col] = df[p_col] * df[c_col]
            created.append(sig_col)
        elif p_col in df.columns:
            df[sig_col] = df[p_col]
            created.append(sig_col)
    # raw pred/conf 제거 (signal_*로 대체됨)
    drop_cols = [c for c in list(PRED_TO_CONF.keys()) + list(PRED_TO_CONF.values()) if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    if created:
        logger.info(f"  pred×conf 결합: {len(created)}개 signal_* 생성, raw {len(drop_cols)}개 제거")
    return df, created


def load_data(path: str, rl_path: str = RL_DATA_PATH):
    logger.info(f"데이터 로드: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── RL CSV 병합 (inner join on timestamp) ──────────────────
    if os.path.exists(rl_path):
        logger.info(f"RL 데이터 병합: {rl_path}")
        df_rl = pd.read_csv(rl_path, parse_dates=['timestamp'])
        df_rl.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 중복 컬럼(close 등) _rl suffix 붙은 것 제거
        rl_extra = [c for c in df_rl.columns if c not in ('timestamp',) and c not in df.columns]
        df = df.merge(df_rl[['timestamp'] + rl_extra], on='timestamp', how='inner')
        logger.info(f"  ✓ 병합 후: {len(df):,}행, {len(df.columns)}개 컬럼")
    else:
        logger.warning(f"RL 데이터 없음: {rl_path} (pred_*/sig_* 피처 제외)")

    # ── pred × conf → signal_* 결합 ───────────────────────────
    df, signal_cols = _combine_pred_conf(df)

    # ── 파생 피처 생성 ─────────────────────────────────────────
    if 'mtf_trend_1h' not in df.columns:
        df['mtf_trend_1h'] = (df['close'] / df['close'].ewm(span=12).mean()) - 1
    if 'mtf_trend_4h' not in df.columns:
        df['mtf_trend_4h'] = (df['close'] / df['close'].ewm(span=48).mean()) - 1
    if 'regime_break' not in df.columns:
        df['regime_break'] = 0.0
    df = _add_trend_structure_features(df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 피처 후보 목록 구성 ────────────────────────────────────
    trend_feats  = ['ret_12', 'ret_24', 'ret_48', 'hh_count_24', 'hl_count_24', 'trend_accel']
    extra_feats  = signal_cols + RL_SIG_COLS + RL_ALPHA_COLS + ['mtf_trend_1h', 'mtf_trend_4h'] + trend_feats

    combined = [c for c in ULTIMATE_FEATURE_COLS if c in df.columns]
    for c in extra_feats:
        if c in df.columns and c not in combined:
            combined.append(c)
    feature_candidates = list(dict.fromkeys(combined))

    logger.info(f"  ✓ {len(df):,}행, {len(feature_candidates)}개 피처 후보 (signal_*, sig_*, 합성 알파 포함)")
    return df, feature_candidates


# ────────────────────────────────────────────────────────────────
# Triple-Barrier 레이블 생성
# ────────────────────────────────────────────────────────────────
def _label_one(t, closes, atr, highs, lows):
    lbl, _, _ = make_triple_barrier_label(
        closes, atr, t,
        highs    = highs,
        lows     = lows,
        atr_mult = ATR_MULT,
        max_hold = MAX_HOLD_5M,
    )
    return t, lbl


def build_labels(df: pd.DataFrame, n_jobs: int = -1):
    closes = df['close'].values.astype(np.float64)
    highs  = df['high'].values.astype(np.float64) if 'high' in df.columns else closes
    lows   = df['low'].values.astype(np.float64)  if 'low'  in df.columns else closes
    atr    = compute_atr(highs, lows, closes, window=ATR_WINDOW_5M)

    T         = len(df)
    valid_idx = list(range(1, T - MAX_HOLD_5M - 1))
    labels    = np.full(T, -1, dtype=np.int64)

    logger.info(f"Triple-Barrier 레이블 병렬 계산 중... ({len(valid_idx)}개 샘플, n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_label_one)(t, closes, atr, highs, lows) for t in valid_idx
    )
    for t, lbl in results:
        labels[t] = lbl

    cnt   = np.bincount(labels[labels >= 0], minlength=3)
    total = cnt.sum()
    if total == 0:
        raise ValueError("Triple-Barrier 유효 레이블이 0개입니다. ATR/HOLD 파라미터와 입력 데이터를 확인하세요.")
    logger.info(
        f"  DOWN={cnt[0]/total*100:.1f}%  "
        f"FLAT={cnt[1]/total*100:.1f}%  "
        f"UP={cnt[2]/total*100:.1f}%"
    )
    return labels


# ────────────────────────────────────────────────────────────────
# 학습 / Optuna
# ────────────────────────────────────────────────────────────────
def train(data_path: str = DATA_PATH,
          save_path: str = SAVE_PATH,
          n_trials: int  = N_TRIALS,
          max_features: int = MAX_FEATURES):

    df, feature_candidates = load_data(data_path)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]
    config_hash = _build_config_hash(max_features)
    labels = build_labels(df)

    # 유효 행 마스크
    valid_mask = labels >= 0
    X_all = df[feature_candidates].values.astype(np.float32)
    y_all = labels

    # 시간 순서 분할
    valid_indices = np.where(valid_mask)[0]
    n = len(valid_indices)
    if n < 100:
        raise ValueError(f"유효 레이블 샘플이 너무 적습니다: {n} (<100).")
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx   = int(n * (TRAIN_RATIO + VAL_RATIO))
    embargo       = MAX_HOLD_5M + 2

    train_idx = valid_indices[:train_end_idx]
    val_idx   = valid_indices[train_end_idx + embargo : val_end_idx]
    test_idx  = valid_indices[val_end_idx + embargo:]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"분할 결과가 비어 있습니다: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
            f"데이터 길이({n})/비율(TRAIN={TRAIN_RATIO}, VAL={VAL_RATIO})/embargo({embargo})를 조정하세요."
        )

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

    logger.info(f"분할 — Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    # ── 피처 선택 (훈련 셋만 사용) ──
    must = [c for c in MUST_INCLUDE if c in feature_candidates]
    train_df_tmp = df.iloc[train_idx].copy()
    train_df_tmp.index = range(len(train_df_tmp))
    train_df_tmp['_label'] = y_train

    selected_features = auto_select_features(
        train_df_tmp, feature_candidates,
        target_col     = '_label',
        max_features   = max_features,
        corr_threshold = 0.85,
        must_include   = must,
    )
    logger.info(f"선택된 피처: {len(selected_features)}개")

    # 선택된 피처 인덱스로 재구성
    feat_idx = [feature_candidates.index(c) for c in selected_features]
    X_train = X_train[:, feat_idx]
    X_val   = X_val[:,   feat_idx]
    X_test  = X_test[:,  feat_idx]

    # 클래스 가중치 (불균형 보정)
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    class_weight = {i: counts.sum() / (3.0 * max(counts[i], 1)) for i in range(3)}

    # ── Optuna 튜닝 or 저장된 파라미터 재사용 ──
    results_path = os.path.join(os.path.dirname(save_path), 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        prev_data_hash = prev.get('data_hash', '')
        prev_cfg_hash  = prev.get('config_hash', '')
        if prev_data_hash != data_hash:
            logger.warning(f"⚠️ 데이터 해시 불일치 (저장: {prev_data_hash}, 현재: {data_hash}) → Optuna 재실행")
            prev = None
        elif not prev_cfg_hash:
            logger.warning("⚠️ config_hash가 없어 재사용 신뢰 불가 → Optuna 재실행")
            prev = None
        elif prev_cfg_hash != config_hash:
            logger.warning(f"⚠️ 설정 해시 불일치 (저장: {prev_cfg_hash}, 현재: {config_hash}) → Optuna 재실행")
            prev = None
        else:
            logger.info(f"기존 training_results.json 발견 (data/config 해시 일치: {data_hash}/{config_hash}) → Optuna 건너뜀")
    else:
        prev = None

    if prev is not None:
        saved_params = prev['best_params']
        boosted_n = int(saved_params.get('n_estimators', 500) * 1.1)
        best_params = {
            **saved_params,
            'n_estimators': boosted_n,
            'class_weight': class_weight,
            'objective':    'multiclass',
            'num_class':    3,
            'n_jobs':       -1,
            'random_state': 42,
            'verbose':      -1,
        }
        logger.info(f"재사용 파라미터: n_estimators={boosted_n} (×1.1), 나머지 고정")
        best_val_metric = prev.get('best_val_dir_f1', prev.get('best_val_bacc', None))
    else:
        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int('n_estimators', 300, 1200),
                max_depth         = trial.suggest_int('max_depth', 3, 8),
                learning_rate     = trial.suggest_float('learning_rate', 5e-3, 0.15, log=True),
                num_leaves        = trial.suggest_int('num_leaves', 15, 127),
                subsample         = trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                min_child_samples = trial.suggest_int('min_child_samples', 10, 100),
                class_weight      = class_weight,
                objective         = 'multiclass',
                num_class         = 3,
                n_jobs            = -1,
                random_state      = 42,
                verbose           = -1,
            )
            clf = LGBMClassifier(**params)
            clf.fit(
                X_train, y_train,
                eval_set  = [(X_val, y_val)],
                callbacks = [early_stopping(40, verbose=False), log_evaluation(-1)],
            )
            preds = clf.predict(X_val)
            # FLAT을 제외한 UP/DOWN 샘플에 대한 방향성 F1 (macro)
            # FLAT 과다 예측으로 balanced_acc가 오르는 문제를 방지
            dir_mask = y_val != 1
            if dir_mask.sum() == 0:
                return 0.0
            return f1_score(y_val[dir_mask], preds[dir_mask],
                            labels=[0, 2], average='macro', zero_division=0)

        logger.info(f"Optuna 튜닝 시작: {n_trials}회 시행...")
        study = optuna.create_study(
            direction = 'maximize',
            sampler   = optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best Val dir_f1 (UP/DOWN macro): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        boosted_n = int(study.best_params.get('n_estimators', 500) * 1.1)
        best_params = {
            **study.best_params,
            'n_estimators': boosted_n,
            'class_weight': class_weight,
            'objective': 'multiclass',
            'num_class': 3,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1,
        }
        best_val_metric = study.best_value

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])

    logger.info(f"최종 재학습: n_estimators={boosted_n} (×1.1 보정)")
    final_clf = LGBMClassifier(**best_params)
    final_clf.fit(X_trainval, y_trainval)

    # ── 테스트셋 평가 ──
    test_preds = final_clf.predict(X_test)
    test_bacc  = balanced_accuracy_score(y_test, test_preds)
    dir_mask_test = y_test != 1
    test_dir_f1 = f1_score(y_test[dir_mask_test], test_preds[dir_mask_test],
                           labels=[0, 2], average='macro', zero_division=0) if dir_mask_test.sum() > 0 else 0.0
    logger.info(f"\nTest balanced_acc: {test_bacc:.4f}  |  Test dir_f1 (UP/DOWN): {test_dir_f1:.4f}")
    logger.info("\n" + classification_report(y_test, test_preds, target_names=['DOWN','FLAT','UP']))

    # ── 저장 ──
    brain = XGBTrendBrain()
    brain.model        = final_clf
    brain.feature_cols = selected_features
    brain.save(save_path)

    # 결과 메타 기록
    results_path = os.path.join(os.path.dirname(save_path), 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'best_val_dir_f1' : best_val_metric,
            'test_bacc'       : test_bacc,
            'test_dir_f1'     : test_dir_f1,
            'n_features'    : len(selected_features),
            'best_params'   : {k: v for k, v in best_params.items() if k != 'class_weight'},
            'atr_mult'      : ATR_MULT,
            'max_hold_bars' : MAX_HOLD_5M,
            'data_hash'     : data_hash,
            'config_hash'   : config_hash,
        }, f, indent=2)

    logger.info(f"결과 저장: {results_path}")
    return brain


# ────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBTrendBrain 학습')
    parser.add_argument('--data',         type=str, default=DATA_PATH)
    parser.add_argument('--save',         type=str, default=SAVE_PATH)
    parser.add_argument('--n-trials',     type=int, default=N_TRIALS)
    parser.add_argument('--max-features', type=int, default=MAX_FEATURES)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 XGBTrendBrain 학습 (LightGBM 3-class, Triple-Barrier)")
    print(f"   data={args.data}  save={args.save}")
    print(f"   n_trials={args.n_trials}  max_features={args.max_features}")
    print("=" * 70 + "\n")

    brain = train(
        data_path    = args.data,
        save_path    = args.save,
        n_trials     = args.n_trials,
        max_features = args.max_features,
    )

    print("\n" + "=" * 70)
    print(f"🎉 완료! 모델: {args.save}")
    print("=" * 70)
