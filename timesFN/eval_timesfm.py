import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 구글 TimesFM (200M) 제로샷 예측 성능 검증 시작")
    
    # 1. 데이터 로드 및 초기 0값(512개) 제거
    df = pd.read_csv("data/training_features_with_tfm.csv")
    df = df.iloc[512:].reset_index(drop=True)
    
    # 2. '진짜' 30분(6봉) 뒤의 수익률 계산
    # shift(-6)을 통해 현재 시점에 6봉 뒤의 가격을 끌어와서 실제 수익률을 구합니다.
    df['actual_ret_6'] = df['close'].shift(-6) / df['close'] - 1
    
    # 마지막 6개의 행은 미래를 알 수 없으므로 제거
    df.dropna(subset=['actual_ret_6', 'tfm_pred_final'], inplace=True)
    
    # 3. 모델의 예측값과 실제값 분리
    y_pred = df['tfm_pred_final'].values
    y_true = df['actual_ret_6'].values
    
    # 4. 방향성 적중률 (Directional Accuracy) 계산
    # 예측이 양수(+)면 1, 음수(-)면 -1로 변환하여 방향이 일치하는지 확인
    pred_dir = np.sign(y_pred)
    true_dir = np.sign(y_true)
    
    # 실제 가격이 변동이 없는(0.0) 경우는 제외하고 계산
    valid_idx = true_dir != 0
    accuracy = (pred_dir[valid_idx] == true_dir[valid_idx]).mean()
    
    # 5. 상관계수 (Pearson Correlation) 계산
    corr, p_value = pearsonr(y_pred, y_true)
    
    # 6. 오차율 (MAE: 평균 절대 오차)
    mae = np.abs(y_pred - y_true).mean()
    
    logger.info("\n" + "="*50)
    logger.info("🏆 [구글 TimesFM 30분 뒤 예측 성능 검증 리포트]")
    logger.info("="*50)
    logger.info(f"▶ 검증 데이터 수: {len(y_pred):,} 캔들")
    logger.info(f"▶ 🎯 방향성 적중률 (Accuracy): {accuracy:.2%}")
    logger.info(f"▶ 🔗 예측-실제 상관계수 (Correlation): {corr:.4f} (p-value: {p_value:.4f})")
    logger.info(f"▶ 📉 평균 절대 오차 (MAE): {mae:.4%}")
    logger.info("="*50)

    # 7. 시각화 (산점도) - 퀀트들은 눈으로 봐야 믿습니다.
    plt.figure(figsize=(10, 6))
    
    # 데이터가 너무 많으면 보기 힘드므로 5000개만 샘플링
    sample_size = min(5000, len(y_pred))
    idx = np.random.choice(len(y_pred), sample_size, replace=False)
    
    plt.scatter(y_pred[idx] * 100, y_true[idx] * 100, alpha=0.3, s=10, color='blue')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    # 완벽한 예측선 (y=x)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    plt.title("TimesFM Predicted Return vs Actual Return (30 min horizon)")
    plt.xlabel("TimesFM Predicted Return (%)")
    plt.ylabel("Actual Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("timesfm_accuracy_scatter.png")
    logger.info("📈 산점도 차트가 'timesfm_accuracy_scatter.png'로 저장되었습니다.")

if __name__ == "__main__":
    main()