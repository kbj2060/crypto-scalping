"""
MacroHFT Hyperparameter Optimization with Optuna (v2.6)
================================================================================
현재 아키텍처(v2.6)에 맞게 최적화:
  - DirectionalLoss (이진 분류, 단일 로짓)
  - 윈도우 내 정규화 (regime-invariant)
  - 로컬 변동성 정규화 타겟
  - val direction accuracy 기준 최적화 (minimize → maximize)
  - Walk-Forward 스타일 purge split

탐색 전략:
  - Phase 1 (0-19):  넓은 탐색 — 구조(d_model, n_layers, window) 중심
  - Phase 2 (20-49): 좁은 탐색 — 정규화(dropout, wd, label_smooth) 미세 조정
  - forecast_horizon을 탐색 축에 포함 (3, 6, 12 중 최적 탐색)
"""

import sys
import os
import gc
import json
import logging
import argparse
import optuna
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from macroHFT_model import MacroHFTConfig
from train_macroHFT import (
    load_data, MacroHFTSignalModel, MacroHFTDataset,
    evaluate_model
)
from core.feature_selector import auto_select_features
from core.feature_engineering import MUST_INCLUDE_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 전역 데이터 캐시
# ════════════════════════════════════════════════════════════════
GLOBAL_DATA = {
    'feature_cache': {},
    'data_cache': {},       # horizon별 데이터 캐시
}


def purged_split(df, train_ratio=0.7, val_ratio=0.15, purge_bars=96):
    """Train/Val/Test 사이에 purge gap을 두어 정보 누수 차단."""
    n = len(df)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    train = df.iloc[:t].copy()
    val   = df.iloc[t + purge_bars : v].copy()
    test  = df.iloc[v + purge_bars :].copy()
    return train, val, test


def get_data(horizon, data_path):
    """horizon별로 데이터를 캐시하여 재사용."""
    key = f"h{horizon}"
    if key not in GLOBAL_DATA['data_cache']:
        logger.info(f"📂 데이터 로드 중 (horizon={horizon})...")
        df, feature_cols = load_data(data_path, h=horizon)
        train_df, val_df, test_df = purged_split(df, purge_bars=96)
        GLOBAL_DATA['data_cache'][key] = {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'feature_cols': feature_cols,
        }
        logger.info(f"✅ h={horizon} 로드 완료: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return GLOBAL_DATA['data_cache'][key]


# ════════════════════════════════════════════════════════════════
# Objective
# ════════════════════════════════════════════════════════════════
def objective(trial: optuna.Trial):

    # ── 1. 핵심 구조 파라미터 ────────────────────────────────
    forecast_horizon = 1
    input_window     = trial.suggest_categorical("input_window", [24, 48, 64, 96])
    max_features     = trial.suggest_int("max_features", 15, 40, step=5)

    d_model  = trial.suggest_categorical("d_model", [64, 128, 256])
    n_head   = trial.suggest_categorical("n_head", [4, 8])
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # d_model이 n_head로 나눠지지 않으면 prune
    if d_model % n_head != 0:
        raise optuna.TrialPruned()

    # head_dim이 16 미만이면 RoPE가 무의미 → prune
    if (d_model // n_head) < 16:
        raise optuna.TrialPruned()

    proj_dim       = trial.suggest_categorical("proj_dim", [32, 64, 128, 256])
    decoder_hidden = trial.suggest_categorical("decoder_hidden", [32, 64, 128, 256])

    # ── 2. 정규화 파라미터 ───────────────────────────────────
    dropout        = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.15, step=0.05)
    weight_decay   = trial.suggest_float("weight_decay", 1e-3, 0.05, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)

    # ── 3. 학습 파라미터 ─────────────────────────────────────
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
    recency_bias  = trial.suggest_float("recency_bias", 0.0, 0.1, step=0.02)
    ema_decay     = trial.suggest_float("ema_decay", 0.99, 0.999, step=0.002)
    grad_clip     = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    large_move_wt = trial.suggest_float("large_move_weight", 1.5, 4.0, step=0.5)

    # ── 4. Config 세팅 ───────────────────────────────────────
    cfg = MacroHFTConfig(
        input_window     = input_window,
        forecast_horizon = forecast_horizon,
        target_col       = f'target_ret_1',

        d_model        = d_model,
        n_head         = n_head,
        n_layers       = n_layers,
        proj_dim       = proj_dim,
        decoder_hidden = decoder_hidden,
        dropout        = dropout,
        drop_path_rate = drop_path_rate,
        recency_bias   = recency_bias,

        num_outputs    = 1,
        num_features   = max_features,  # fit()에서 실제 값으로 덮어씀

        learning_rate  = learning_rate,
        batch_size     = batch_size,
        weight_decay   = weight_decay,
        grad_clip      = grad_clip,
        ema_decay      = ema_decay,
        large_move_weight = large_move_wt,

        # Optuna 전용 세팅
        max_epochs     = 150,
        patience       = 25,
        warmup_epochs  = 8,
        lr_scheduler   = 'cosine',
        min_lr         = 1e-6,
        use_ema        = True,
        model_dir      = f'data/macrohft/optuna_trial_{trial.number}',
    )
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ── 5. 데이터 로드 (horizon별 캐시) ──────────────────────
    data = get_data(1, 'data/training_features_5m.csv')
    train_df     = data['train_df']
    val_df       = data['val_df']
    feature_cols = data['feature_cols']

    # ── 6. 피처 캐싱 ────────────────────────────────────────
    cache_key = f"f{max_features}"  # horizon 고정이므로
    if cache_key not in GLOBAL_DATA['feature_cache']:
        GLOBAL_DATA['feature_cache'][cache_key] = auto_select_features(
            train_df, feature_cols,
            target_col     = cfg.target_col,
            max_features   = max_features,
            corr_threshold = 0.85,
            must_include   = MUST_INCLUDE_FEATURES,
        )
    selected = GLOBAL_DATA['feature_cache'][cache_key]
    cfg.num_features = len(selected)

    # ── 7. 학습 ─────────────────────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"🚀 [Trial {trial.number}] h={forecast_horizon} win={input_window} "
        f"feat={cfg.num_features} d={d_model} nh={n_head} L={n_layers}\n"
        f"   proj={proj_dim} dec={decoder_hidden} "
        f"lr={learning_rate:.1e} bs={batch_size} "
        f"drop={dropout:.2f} dpr={drop_path_rate:.2f} "
        f"wd={weight_decay:.1e} ls={label_smoothing:.2f}\n"
        f"   ema={ema_decay:.3f} gc={grad_clip} rb={recency_bias:.2f} "
        f"lmw={large_move_wt:.1f}\n"
        f"{'='*60}"
    )

    model_wrapper = None
    try:
        model_wrapper = MacroHFTSignalModel(cfg)

        # label_smoothing을 fit에 전달하기 위해 cfg에 임시 저장
        cfg._label_smoothing = label_smoothing

        history = model_wrapper.fit(cfg, train_df, val_df, selected)

        # val direction accuracy 기준으로 최적화
        best_val_dir = max(history['val_direction_acc'])
        best_epoch   = history['val_direction_acc'].index(best_val_dir)
        best_val_loss = history['val_loss'][best_epoch]

        # train-val gap 진단
        train_dir_at_best = history['train_direction_acc'][best_epoch]
        gap = train_dir_at_best - best_val_dir

        logger.info(
            f"✅ [Trial {trial.number}] 완료: "
            f"Best Val Dir={best_val_dir:.1%} (epoch {best_epoch+1}) "
            f"Train Dir={train_dir_at_best:.1%} Gap={gap:+.1%} "
            f"Val Loss={best_val_loss:.4f}"
        )

        # 과적합 경고
        if gap > 0.10:
            logger.warning(f"  ⚠️ Train-Val gap {gap:.1%} > 10% — 과적합 의심")

        # Optuna에 중간 결과 보고 (pruning용)
        for i, va in enumerate(history['val_direction_acc']):
            trial.report(va, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_dir

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"❌ [Trial {trial.number}] 에러: {e}", exc_info=True)
        raise optuna.TrialPruned()
    finally:
        if model_wrapper is not None:
            del model_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='MacroHFT v2.6 Optuna Optimization')
    parser.add_argument('--trials', type=int, default=50, help='총 Trial 수')
    parser.add_argument('--jobs',   type=int, default=1,  help='병렬 수 (GPU 1개면 1)')
    parser.add_argument('--data',   type=str, default='data/training_features_5m.csv')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"🤖 MacroHFT v2.6 Optuna Optimization")
    print(f"   목표: Val Direction Accuracy 최대화 ({args.trials} trials)")
    print(f"   탐색: forecast_horizon, 구조, 정규화, 학습률 동시 탐색")
    print("=" * 80)

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,    # 초반 10회는 랜덤 탐색
        multivariate=True,      # 파라미터 간 상관관계 학습
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,     # 8회까지는 pruning 안 함
        n_warmup_steps=15,      # 15 에폭까지는 pruning 안 함
        interval_steps=5,       # 5 에폭마다 pruning 체크
    )

    study = optuna.create_study(
        direction  = "maximize",        # direction accuracy 최대화
        study_name = "MacroHFT_v26",
        sampler    = sampler,
        pruner     = pruner,
    )

    study.optimize(
        objective,
        n_trials   = args.trials,
        n_jobs     = args.jobs,
        gc_after_trial = True,
        show_progress_bar = True,
    )

    # ── 결과 출력 ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🎉 Optuna 최적화 완료!")
    print(f"  최고 Trial:       #{study.best_trial.number}")
    print(f"  최고 Val Dir Acc: {study.best_value:.1%}")
    print("  최적 하이퍼파라미터:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k:25s}: {v}")

    # 상위 5개 Trial 출력
    print(f"\n  📊 상위 5개 Trial:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for t in top_trials:
        if t.value is not None:
            h = t.params.get('forecast_horizon', '?')
            d = t.params.get('d_model', '?')
            L = t.params.get('n_layers', '?')
            print(f"    #{t.number:3d}: {t.value:.1%}  (h={h}, d={d}, L={L})")

    # 저장
    save_path = 'data/macrohft/best_optuna_params_v26.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'top5': [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in top_trials if t.value is not None
        ],
        'timestamp': datetime.now().isoformat(),
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, (np.floating, float)) else str(x))

    print(f"\n💾 결과 저장: {save_path}")

    # MacroHFTConfig 형태로 바로 복붙할 수 있게 출력
    bp = study.best_params
    print(f"\n  📋 Config에 바로 적용할 값:")
    print(f"    input_window     = {bp.get('input_window')}")
    print(f"    forecast_horizon = {bp.get('forecast_horizon')}")
    print(f"    d_model          = {bp.get('d_model')}")
    print(f"    n_head           = {bp.get('n_head')}")
    print(f"    n_layers         = {bp.get('n_layers')}")
    print(f"    proj_dim         = {bp.get('proj_dim')}")
    print(f"    decoder_hidden   = {bp.get('decoder_hidden')}")
    print(f"    dropout          = {bp.get('dropout')}")
    print(f"    drop_path_rate   = {bp.get('drop_path_rate')}")
    print(f"    learning_rate    = {bp.get('learning_rate'):.6f}")
    print(f"    batch_size       = {bp.get('batch_size')}")
    print(f"    weight_decay     = {bp.get('weight_decay'):.6f}")
    print(f"    ema_decay        = {bp.get('ema_decay')}")
    print(f"    grad_clip        = {bp.get('grad_clip')}")
    print(f"    recency_bias     = {bp.get('recency_bias')}")
    print(f"    large_move_weight = {bp.get('large_move_weight')}")
    print(f"    # label_smoothing = {bp.get('label_smoothing')}")
    print("=" * 80)


if __name__ == "__main__":
    main()