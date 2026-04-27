import os

# Ollama가 설치한 CUDA 13.0 라이브러리 경로 추가 (NVRTC 에러 해결용)
OLLAMA_CUDA_PATH = "/usr/local/lib/ollama/mlx_cuda_v13"
if os.path.exists(OLLAMA_CUDA_PATH):
    os.environ["LD_LIBRARY_PATH"] = OLLAMA_CUDA_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import pandas as pd
import numpy as np
import logging

ROOT = '/home/llewyn/crypto-scalping'
sys.path.insert(0, ROOT)

from ensemble.ensemble_router import EnsembleRouter

logging.basicConfig(level=logging.INFO)

# Load real data sample
df = pd.read_csv('/home/llewyn/crypto-scalping/data/splits/year_oos/training_features_2025.csv').head(500)

router = EnsembleRouter()
model = router.models['TimesNet']
if model.available:
    prep = model._prepare_data(df)
    print(f'Prep columns: {prep.columns.tolist()}')
    
    # Run prediction for one window
    window = prep.tail(256).copy()
    window.insert(0, 'unique_id', 'test')
    window.insert(1, 'ds', pd.date_range(start='2024-01-01', periods=256, freq='5min'))
    window.rename(columns={'close': 'y'}, inplace=True)
    
    pred_df = model.nf.predict(df=window)
    print('\n### TimesNet Raw Prediction ###')
    print(pred_df)
    print('\nColumns in pred_df:', pred_df.columns.tolist())
else:
    print("TimesNet NOT available")
