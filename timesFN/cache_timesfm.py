import pandas as pd
import numpy as np
import timesfm
import logging
from tqdm import tqdm  # 진행률 표시 (pip install tqdm)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("📡 구글 TimesFM 정찰기 가동: 전체 과거 데이터 궤적 스캔 시작")
    
    # 1. 기존 골든 피처 데이터 로드
    file_path = "data/training_features_5m.csv"
    df = pd.read_csv(file_path)
    
    # 2. TimesFM 모델 초기화 (PyTorch 백엔드)
    context_len = 512
    horizon_len = 6
    
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu", # GPU가 있다면 "gpu"로 설정하여 속도 극대화
            per_core_batch_size=32,
            context_len=context_len,
            horizon_len=horizon_len,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
    )
    
    # 3. 새로운 피처를 담을 빈 배열 생성
    tfm_mean = np.zeros(len(df))
    tfm_std = np.zeros(len(df))
    tfm_final = np.zeros(len(df))
    
    prices = df['close'].values
    
    # 4. 롤링 윈도우 방식으로 과거 차트 스캔 (배치 처리로 속도 최적화 가능)
    # 처음 512개 캔들은 룩백(Look-back)이 부족하므로 0으로 둡니다.
    logger.info("⏳ 시계열 궤적 추출 중... (데이터 크기에 따라 시간이 소요됩니다)")
    
    # 고속 처리를 위한 배치 구성 (예: 64개씩 묶어서 TimesFM에 던짐)
    batch_size = 64
    inputs_batch = []
    indices_batch = []
    
    for i in tqdm(range(context_len, len(df))):
        # 과거 512봉 가격 데이터 추출
        window_prices = prices[i - context_len : i]
        inputs_batch.append(window_prices)
        indices_batch.append(i)
        
        # 배치가 꽉 찼거나, 마지막 데이터인 경우 추론 실행
        if len(inputs_batch) == batch_size or i == len(df) - 1:
            # TimesFM 추론
            forecasts, _ = tfm.forecast(inputs=inputs_batch, freq=[0] * len(inputs_batch))
            
            # 결과 저장
            for b_idx, forecast in enumerate(forecasts):
                original_i = indices_batch[b_idx]
                current_price = prices[original_i - 1]
                
                # 미래 6봉의 수익률 궤적
                future_rets = (forecast - current_price) / current_price
                
                tfm_mean[original_i] = np.mean(future_rets)
                tfm_std[original_i] = np.std(future_rets)
                tfm_final[original_i] = future_rets[-1]
            
            # 배치 초기화
            inputs_batch = []
            indices_batch = []

    # 5. 추출된 피처를 원본 데이터프레임에 결합
    df['tfm_pred_mean'] = tfm_mean
    df['tfm_pred_std'] = tfm_std
    df['tfm_pred_final'] = tfm_final
    
    # 6. 결합된 최종 데이터를 새로운 CSV로 저장
    save_path = "data/training_features_with_tfm.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"✅ 완벽하게 결합된 최종 데이터 저장 완료: {save_path}")

if __name__ == "__main__":
    main()