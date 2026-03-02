"""
Meta Router Training Cache Generator (Validation Set 전용)
================================================================================
목적: 과거 데이터에 대해 6개 앙상블 모델의 예측값(Pred)과 확신도(Conf)를 미리 추출하여 저장
수정: Data Leakage 방지를 위해 Train 구간을 철저히 배제하고 정확히 Validation 구간만 캐싱합니다.
"""

import os
import sys

# 실행 위치(crypto-scalping/)를 기준으로 필요한 경로 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)          # crypto-scalping/
_ENSEMBLE_DIR = os.path.join(_ROOT_DIR, 'ensemble')
_TFT_DIR = os.path.join(_ENSEMBLE_DIR, 'TFT')
_MACROHFT_DIR = os.path.join(_ENSEMBLE_DIR, 'macroHFT')

for _p in [_ROOT_DIR, _ENSEMBLE_DIR, _TFT_DIR, _MACROHFT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings

# 경고 메시지 숨김
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 내부 라이브러리(TFT 학습 루프 등)의 tqdm 출력 억제
os.environ['TQDM_DISABLE'] = '0'   # 메인 tqdm은 켬
logging.getLogger('chronos').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('uni2ts').setLevel(logging.WARNING)


# 사용자 환경 모듈 임포트
try:
    from ensemble_router import (
        TFTForecaster, MacroHFTForecaster,
        ChronosForecaster, KronosForecaster,
        TimesFMForecaster, MoiraiForecaster
    )
    # TFT 학습 스크립트에서 동일한 전처리 로직(load_data)을 빌려옵니다.
    from train_TFT import load_data
except ImportError as e:
    import traceback
    print(f"❌ 임포트 에러: {e}")
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_cache(data_path: str, output_path: str, target_split: str = 'val', max_lookback: int = 512):
    """
    Args:
        data_path: 원본 OHLCV 피처 데이터 경로
        output_path: 저장될 캐시 CSV 파일 경로
        target_split: 'val' (검증셋) 또는 'test' (테스트셋) 지정
        max_lookback: 파운데이션 모델들에 밀어넣을 과거 데이터 컨텍스트 길이
    """
    logger.info("1. 원본 데이터 로드 중...")
    df_full, feature_cols = load_data(data_path, h=6)
    
    # ── 데이터 분할 로직 (train_TFT.py 와 완벽하게 동일한 비율) ──
    n = len(df_full)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    if target_split == 'val':
        start_idx = train_end
        end_idx = val_end
        logger.info(f"🎯 타겟 구간: Validation Set (인덱스 {start_idx} ~ {end_idx})")
    elif target_split == 'test':
        start_idx = val_end
        end_idx = n
        logger.info(f"🎯 타겟 구간: Test Set (인덱스 {start_idx} ~ {end_idx})")
    else:
        raise ValueError("target_split must be 'val' or 'test'")

    # 첫 번째 타겟 캔들을 예측하기 위해 과거 max_lookback 만큼의 데이터가 추가로 필요함
    slice_start = max(0, start_idx - max_lookback)
    
    # 실제로 캐싱 스크립트가 순회할 잘려진 데이터프레임
    df = df_full.iloc[slice_start:end_idx].reset_index(drop=True)
    logger.info(f"   필요 과거 데이터({max_lookback}개) 포함 총 {len(df)}행 추출 완료.")

    # ── Resume 로직 ──────────────────────────────────────────────────────────
    # 기존 캐시 파일이 있으면 마지막 timestamp 이후부터 이어서 생성합니다.
    # 시작점은 과거 데이터(max_lookback)가 끝난 지점부터입니다.
    resume_start = max_lookback  
    existing_df = None

    abs_output = os.path.abspath(output_path)
    if os.path.exists(abs_output):
        try:
            existing_df = pd.read_csv(abs_output)
            if not existing_df.empty and 'timestamp' in existing_df.columns:
                last_ts = str(existing_df['timestamp'].iloc[-1])
                # 현재 슬라이싱된 df의 timestamp 컬럼과 비교
                ts_series = df['timestamp'].astype(str)
                match = ts_series[ts_series == last_ts]
                if not match.empty:
                    last_idx = match.index[-1]
                    resume_start = last_idx + 1  # 다음 인덱스부터 시작
                    logger.info(
                        f"♻️  기존 캐시 발견: {len(existing_df)}행 로드됨. "
                        f"마지막 timestamp={last_ts} → 이어서 재시작합니다."
                    )
                else:
                    logger.warning("기존 캐시의 마지막 timestamp를 현재 구간에서 찾지 못했습니다. 처음부터 덮어씁니다.")
                    existing_df = None
        except Exception as e:
            logger.warning(f"기존 캐시 읽기 실패 ({e}). 처음부터 덮어씁니다.")
            existing_df = None

    if resume_start >= len(df):
        logger.info("✅ 해당 구간의 모든 데이터가 이미 캐싱되어 있습니다. 종료합니다.")
        return
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("2. 6대 앙상블 모델 메모리 적재 중...")
    models = {
        'tft': TFTForecaster(),
        'macro': MacroHFTForecaster(),
        'chronos': ChronosForecaster(),
        'kronos': KronosForecaster(),
        'timesfm': TimesFMForecaster(),
        'moirai': MoiraiForecaster()
    }
    
    cached_data = []
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    remaining = len(df) - resume_start
    logger.info(f"\n🚀 총 {remaining}개 캔들에 대한 {target_split.upper()} 구간 예측 시작...")
    
    bar = tqdm(
        range(resume_start, len(df)),
        desc=f"{target_split.upper()} 캐시 생성",
        unit="candle",
        ncols=80,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for i in bar:
        df_slice = df.iloc[i - max_lookback : i + 1].copy()
        current_row = df.iloc[i]
        
        # 1) 라우터가 학습할 현재 시장의 피처와 정답(Target) 기록
        row_data = {
            'timestamp': current_row['timestamp'],
            'target_ret_6': current_row['target_ret_6']
        }
        for col in feature_cols:
            row_data[col] = current_row[col]
            
        # 2) 6개 모델에게 '현재 시점'에서의 예측을 요구
        for name, model in models.items():
            pred_val, conf_val = 0.0, 0.0
            if model.available:
                try:
                    out = model.predict(df_slice, horizon=6)
                    if out.median is not None:
                        pred_val = float(out.median[-1].mean())
                        conf_val = float(out.confidence[-1].mean())
                except Exception:
                    pass 
                    
            row_data[f'pred_{name}'] = pred_val
            row_data[f'conf_{name}'] = conf_val
            
        cached_data.append(row_data)
        
        # 3) 5000스텝마다 자동 중간 저장
        if len(cached_data) % 5000 == 0:
            new_df = pd.DataFrame(cached_data)
            save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
            save_df.to_csv(abs_output, index=False)
            logger.info(f"\n🎉 중간 저장 완료! (총 {len(save_df)}행)")
            
    # 최종 저장
    if cached_data:
        new_df = pd.DataFrame(cached_data)
        final_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
        final_df.to_csv(abs_output, index=False)
        logger.info(f"\n🎉 캐시 생성 완료! [{abs_output}] (총 {len(final_df)}행)")


if __name__ == "__main__":
    INPUT_DATA = 'data/training_features_5m.csv'
    
    # 타겟을 'val'로 명시했으므로 파일명도 val임을 알 수 있게 변경
    OUTPUT_CACHE = 'data/ensemble/ensemble_cache_val.csv' 
    
    generate_cache(
        data_path=INPUT_DATA, 
        output_path=OUTPUT_CACHE, 
        target_split='val',   # ★ 핵심 변경점: 'val' 구간만 타겟팅 (15%)
        max_lookback=512
    )