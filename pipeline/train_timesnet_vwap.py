import os

# NVRTC 및 CUDA 라이브러리 충돌 해결을 위한 환경 변수 설정
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "" # GPU 차단하여 CPU 학습 강제

import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_timesnet_vwap():
    data_path = "/home/llewyn/crypto-scalping/data/timesnet_vwap_train.csv"
    save_path = "/home/llewyn/crypto-scalping/data/nf_timesnet"
    
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return

    logger.info("Loading training data...")
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Exogenous columns
    exog_cols = [
        "session_us", "hour_cos", "cvp_poc_dist", "cvp_volume_imbalance",
        "fvg_dist", "breakout_strength", "oi_change_rate", "ofti", "kel",
        "mta_funding", "svps"
    ]
    
    # TimesNet 모델 설정
    # VWAP 이격도의 순환적 특징을 잘 잡도록 하이퍼파라미터 구성
    model = TimesNet(
        h=12,  # 12봉 (1시간) 예측
        input_size=256,
        futr_exog_list=exog_cols,
        # hist_exog_list=exog_cols, # 과거 exog도 사용 가능하나 추론 속도를 위해 미래(당시시점)만 사용
        max_steps=500, # 빠른 학습을 위해 500단계 (필요시 조절)
        learning_rate=1e-3,
        batch_size=32,
        windows_batch_size=128
    )
    
    nf = NeuralForecast(models=[model], freq='5min')
    
    logger.info("Starting training (Target: VWAP Deviation)...")
    nf.fit(df=df)
    
    logger.info(f"Saving model to {save_path}...")
    # 기존 모델 백업 후 저장
    if os.path.exists(save_path):
        import shutil
        backup = save_path + "_backup"
        if os.path.exists(backup): shutil.rmtree(backup)
        shutil.move(save_path, backup)
        logger.info(f"Existing model backed up to {backup}")

    nf.save(path=save_path, overwrite=True)
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    train_timesnet_vwap()
