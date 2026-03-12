"""
train_nf_models.py
NeuralForecast 4대 파운데이션 모델 통합 사전 학습 스크립트
- 단변량(Univariate): PatchTST, iTransformer (가격 패턴 집중)
- 다변량(Multivariate): NHITS, TiDE (7대 위대한 알파 융합)
"""
import os
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, iTransformer, NHITS, TiDE
from neuralforecast.losses.pytorch import HuberLoss

# 10만 표본이 증명한 7대 절대 알파 (외부 변수)
EXOG_COLS = [
    'session_us', 'hour_cos', 'cvp_poc_dist', 
    'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate'
]

def train_neuralforecast():
    csv_path = 'data/training_features_5m.csv'
    model_save_dir = 'data/nf'
    os.makedirs(model_save_dir, exist_ok=True)

    print("📊 1. 데이터 로딩 및 포맷팅...")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna()
    
    df_nf = df[['timestamp', 'close'] + EXOG_COLS].copy()
    df_nf.rename(columns={'timestamp': 'ds', 'close': 'y'}, inplace=True)
    df_nf['ds'] = pd.to_datetime(df_nf['ds'])
    df_nf['unique_id'] = 'ETH'

    print("🧠 2. 4대 파운데이션 모델 정의 (단변량 2종 + 다변량 2종)...")
    models = [
        # ── [A] 단변량 모델 (hist_exog_list 없음) ──
        PatchTST(h=6, input_size=256, max_steps=1000, early_stop_patience_steps=3, loss=HuberLoss()),
        iTransformer(h=6, input_size=256, n_series=1, max_steps=1000, early_stop_patience_steps=3, loss=HuberLoss()),
        
        # ── [B] 다변량 모델 (7대 알파 주입) ──
        NHITS(h=6, input_size=256, hist_exog_list=EXOG_COLS, max_steps=1000, early_stop_patience_steps=3, loss=HuberLoss()),
        TiDE(h=6, input_size=256, hist_exog_list=EXOG_COLS, max_steps=1000, early_stop_patience_steps=3, loss=HuberLoss())
    ]

    nf = NeuralForecast(models=models, freq='5min')

    print("🔥 3. 모델 학습 시작 (val_size=10000 적용으로 조기 종료 활성화)...")
    nf.fit(df=df_nf, val_size=10000)

    print(f"💾 4. 학습 완료. 모델 저장 중... ({model_save_dir})")
    nf.save(path=model_save_dir, overwrite=True)
    print("✅ 4개 모델이 성공적으로 저장되었습니다!")

if __name__ == "__main__":
    train_neuralforecast()