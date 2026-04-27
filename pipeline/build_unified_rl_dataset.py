import os
import sys
import pandas as pd
import numpy as np
import logging

# 프로젝트 루트 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ensemble.seven_model_ensemble import SevenModelEnsemble
from ensemble.ensemble_router import EnsembleRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. 경로 설정
    BASE_FEATURES_2025 = os.path.join(ROOT, "data/splits/year_oos/training_features_2025.csv")
    BASE_RL_2025 = os.path.join(ROOT, "data/splits/year_oos/rl_base_2025.csv")
    OUTPUT_PATH = os.path.join(ROOT, "data/rl_training_2025_unified.csv")

    if not os.path.exists(BASE_FEATURES_2025):
        logger.error(f"원본 데이터 없음: {BASE_FEATURES_2025}")
        return

    # 2. 데이터 로드
    logger.info("2025년 원본 데이터 로드 중...")
    df = pd.read_csv(BASE_FEATURES_2025, parse_dates=['timestamp'])
    df_rl_base = pd.read_csv(BASE_RL_2025, parse_dates=['timestamp'])
    
    # 기본 RL 컬럼 병합
    extra_cols = [c for c in df_rl_base.columns if c not in df.columns and c != 'timestamp']
    if extra_cols:
        df = df.merge(df_rl_base[['timestamp'] + extra_cols], on='timestamp', how='inner')
    logger.info(f"기본 데이터 로드 완료: {len(df):,}행")

    # 3. M7 앙상블 신호 생성 (Clean M7)
    logger.info("M7 전문가 모델 추론 시작...")
    m7 = SevenModelEnsemble()
    
    # 전체 데이터에 대해 배치 추론
    # SevenModelEnsemble의 predict_batch가 내부적으로 최적화되어 있다고 가정
    df_m7 = m7.predict_batch(df)
    
    # M7 주요 지표 병합
    m7_cols = ['m7_confidence', 'm7_gate_block', 'm7_consensus_dir']
    for col in m7_cols:
        if col in df_m7.columns:
            df[col] = df_m7[col].values
            logger.info(f"병합됨: {col}")
    logger.info("M7 신호 병합 완료")

    # 4. 4가지 AI 모델 신호 생성
    logger.info("AI 모델(TiDE, TimesNet, DLinear, PatchTST) 추론 시작...")
    router = EnsembleRouter()
    
    # 정제된 피처 생성 (Z-score, 주기성 인코딩 등 포함)
    refined_ai = router.get_refined_features(df)
    df = pd.concat([df, refined_ai.reset_index(drop=True)], axis=1)
    logger.info(f"AI 정제 피처 병합 완료: {list(refined_ai.columns)}")

    # 5. 최종 저장
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"✅ 최종 2025년 RL 학습셋 생성 완료: {OUTPUT_PATH}")
    logger.info(f"최종 컬럼 수: {len(df.columns)}")

if __name__ == "__main__":
    main()
