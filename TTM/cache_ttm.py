import pandas as pd
import numpy as np
import torch
import logging
from tqdm import tqdm
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🛸 IBM Granite TTM (TinyTimeMixer) 전술 정찰기 가동")
    
    file_path = "data/training_features_5m.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"❌ {file_path} 파일이 없습니다.")
        return
        
    # TTM-v1 모델은 기본적으로 512개의 과거를 보고 96개를 예측하도록 훈련되어 있습니다.
    context_len = 512
    horizon_len = 6 # 우리는 96개 예측값 중 앞의 6개(30분)만 사용합니다.
    
    if len(df) <= context_len:
        logger.error("❌ 데이터가 너무 적습니다.")
        return

    # 1. IBM TTM 모델 로드
    logger.info("🔄 IBM Granite TTM (1M) 모델 로드 중...")
    model_name = "ibm-granite/granite-timeseries-ttm-v1"
    model = TinyTimeMixerForPrediction.from_pretrained(model_name)
    model.eval() # 추론 모드
    
    # GPU 사용 가능 시 가속
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"⚡ 연산 장치: {device} 기반으로 초고속 추론 시작")

    ttm_mean = np.zeros(len(df))
    ttm_std = np.zeros(len(df))
    ttm_final = np.zeros(len(df))
    
    prices = df['close'].values
    batch_size = 128 # 모델이 매우 가벼우므로 배치 사이즈를 넉넉하게 잡습니다.
    inputs_batch = []
    indices_batch = []
    
    logger.info(f"⏳ 총 {len(df) - context_len}개의 데이터 궤적 추출 시작...")
    
    for i in tqdm(range(context_len, len(df))):
        window_prices = prices[i - context_len : i]
        
        # 안전장치: NaN, Inf 필터링
        if np.isnan(window_prices).any() or np.isinf(window_prices).any():
            continue
            
        inputs_batch.append(window_prices)
        indices_batch.append(i)
        
        # 배치가 꽉 찼거나 마지막 루프일 때 추론 실행
        if len(inputs_batch) == batch_size or i == len(df) - 1:
            if not inputs_batch:
                break
                
            try:
                # TTM 입력 형태: (Batch, Context_len, Channels)
                past_values = torch.tensor(np.array(inputs_batch), dtype=torch.float32).unsqueeze(-1)
                past_values = past_values.to(device)
                
                with torch.no_grad():
                    outputs = model(past_values=past_values)
                    # 예측값 추출 (Batch, Forecast_len, Channels)
                    if hasattr(outputs, 'prediction_outputs'):
                        forecasts = outputs.prediction_outputs.squeeze(-1).cpu().numpy()
                    else:
                        forecasts = outputs.logits.squeeze(-1).cpu().numpy()
                
                # 예측 궤적을 기반으로 3가지 파생 피처 계산
                for b_idx, forecast in enumerate(forecasts):
                    original_i = indices_batch[b_idx]
                    current_price = prices[original_i - 1]
                    
                    if current_price == 0:
                        continue
                        
                    # 96개의 미래 중 우리가 필요한 6봉까지만 잘라냅니다.
                    future_prices = forecast[:horizon_len]
                    future_rets = (future_prices - current_price) / current_price
                    
                    # 기존 모델엔진과 동일한 변수명 유지 (RL 코드 수정을 피하기 위해)
                    ttm_mean[original_i] = np.mean(future_rets)
                    ttm_std[original_i] = np.std(future_rets)
                    ttm_final[original_i] = future_rets[-1]
                    
            except Exception as e:
                logger.warning(f"⚠️ 배치 추론 에러 (인덱스 {indices_batch[0]} 부근): {e}")
            
            # 배치 초기화
            inputs_batch = []
            indices_batch = []

    non_zero_count = np.count_nonzero(ttm_mean)
    logger.info("=" * 50)
    logger.info(f"🔍 추출 완료 검증: 전체 {len(df)} 캔들 중 {non_zero_count} 캔들에 IBM TTM 궤적이 성공적으로 기록되었습니다.")
    logger.info("=" * 50)
    
    # 🚨 기존 RL 스크립트와의 호환성을 위해 컬럼명을 tfm_pred_* 로 맞춥니다.
    df['tfm_pred_mean'] = ttm_mean
    df['tfm_pred_std'] = ttm_std
    df['tfm_pred_final'] = ttm_final
    
    save_path = "data/training_features_with_ttm.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"✅ IBM Granite TTM 통찰력이 결합된 새로운 족보 저장 완료: {save_path}")

if __name__ == "__main__":
    main()