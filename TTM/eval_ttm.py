import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 IBM Granite TTM (1M) 제로샷 예측 성능 정밀 타격 검증")
    
    # 1. TTM이 구워낸 데이터 로드
    file_path = "data/training_features_with_ttm.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"❌ {file_path} 파일이 없습니다. 족보 생성을 먼저 확인하세요.")
        return
        
    # 앞쪽 512개(컨텍스트 윈도우) 0값 제거
    df = df.iloc[512:].reset_index(drop=True)
    
    # 2. '진짜' 30분(6봉) 뒤의 수익률 계산 (정답지 만들기)
    df['actual_ret_6'] = df['close'].shift(-6) / df['close'] - 1
    
    # NaN 제거 (마지막 6개 행 등)
    df.dropna(subset=['actual_ret_6', 'tfm_pred_final'], inplace=True)
    
    # 3. 모델 예측값 vs 실제값
    y_pred = df['tfm_pred_final'].values
    y_true = df['actual_ret_6'].values
    
    # 4. 방향성 적중률 (Directional Accuracy)
    pred_dir = np.sign(y_pred)
    true_dir = np.sign(y_true)
    
    valid_idx = true_dir != 0 # 실제 가격 변동이 0인 경우는 제외
    accuracy = (pred_dir[valid_idx] == true_dir[valid_idx]).mean()
    
    # 5. 상관계수 (Pearson Correlation) 및 통계적 유의성
    corr, p_value = pearsonr(y_pred, y_true)
    
    # 6. 평균 절대 오차 (MAE)
    mae = np.abs(y_pred - y_true).mean()
    
    logger.info("\n" + "="*50)
    logger.info("🏆 [IBM Granite TTM 30분 뒤 예측 성능 검증 리포트]")
    logger.info("="*50)
    logger.info(f"▶ 검증 데이터 수: {len(y_pred):,} 캔들")
    logger.info(f"▶ 🎯 방향성 적중률 (Accuracy): {accuracy:.2%}")
    logger.info(f"▶ 🔗 예측-실제 상관계수 (Correlation): {corr:.4f} (p-value: {p_value:.4f})")
    logger.info(f"▶ 📉 평균 절대 오차 (MAE): {mae:.4%}")
    logger.info("="*50)

if __name__ == "__main__":
    main()