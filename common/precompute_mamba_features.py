# scripts/precompute_mamba_features.py
import pandas as pd
import torch
import numpy as np
from macroHFT.mamba_extractor import MambaFeatureExtractor
from common.feature_engineering import ULTIMATE_FEATURE_COLS

# 1. 체크포인트 로드
extractor = MambaFeatureExtractor(
    checkpoint_path='checkpoints/mamba_predictor.pth',
    enc_in=21,  # base_features 개수 (ULTIMATE_FEATURE_COLS에서 samba 관련 제외)
    device='cuda'
)

# 2. 데이터 로드
df = pd.read_csv('data/training_features.csv', index_col=0, parse_dates=True)

# 3. Mamba 특성을 저장할 새 컬럼 준비
emb_cols = [f'mamba_emb_{i}' for i in range(256)]
for col in ['mamba_pred'] + emb_cols:
    if col not in df.columns:
        df[col] = np.nan

# 4. 각 시점에 대해 특성 추출 (lookback 60 필요)
seq_len = 60
for i in range(seq_len, len(df)):
    # 현재까지의 데이터로 extractor 실행
    df_slice = df.iloc[:i+1]  # extractor 내부에서 마지막 60개 사용
    pred, emb = extractor.extract(df_slice)
    df.loc[df.index[i], 'mamba_pred'] = pred
    for j, val in enumerate(emb):
        df.loc[df.index[i], f'mamba_emb_{j}'] = val

# 5. 저장
df.to_csv('data/training_features_with_mamba.csv')