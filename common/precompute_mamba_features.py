# scripts/precompute_mamba_features.py
import pandas as pd
import torch
import numpy as np
import os
import tempfile
from tqdm import tqdm
import logging
from common.mamba_extractor import MambaFeatureExtractor
from common.feature_engineering import ULTIMATE_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정
CHECKPOINT_PATH = 'data/checkpoints/mamba_predictor.pth'
INPUT_CSV = 'data/training_features.csv'
OUTPUT_CSV = 'data/training_features_with_mamba.csv'
TEMP_DIR = tempfile.gettempdir()
ENC_IN = 21  # base_features 개수 (SAMBA 관련 컬럼 제외)
SEQ_LEN = 60
SAVE_INTERVAL = 1000  # 중간 저장 간격

# 1. 체크포인트 로드
logger.info("Loading Mamba feature extractor...")
extractor = MambaFeatureExtractor(
    checkpoint_path=CHECKPOINT_PATH,
    enc_in=ENC_IN,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 2. 데이터 로드
logger.info(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)

# 3. Mamba 특성 컬럼 준비 (단편화 방지를 위해 한 번에 추가)
emb_cols = [f'mamba_emb_{i}' for i in range(256)]
new_cols = ['mamba_pred'] + emb_cols

# 이미 존재하는 컬럼이 있으면 덮어쓰기 위해 제거 (선택사항)
existing_new = [c for c in new_cols if c in df.columns]
if existing_new:
    logger.warning(f"Overwriting existing columns: {existing_new[:5]}...")
    df.drop(columns=existing_new, inplace=True)

# 새 컬럼들을 한 번에 추가 (단편화 방지)
new_data = pd.DataFrame(index=df.index, columns=new_cols, dtype=np.float32)
df = pd.concat([df, new_data], axis=1)

# 4. 각 시점에 대해 특성 추출
logger.info("Extracting Mamba features...")
start_idx = SEQ_LEN
end_idx = len(df)

# 임시 파일로 중간 저장 준비
temp_file = os.path.join(TEMP_DIR, 'mamba_features_temp.csv')
if os.path.exists(temp_file):
    logger.info(f"Resuming from temporary file {temp_file}")
    df_temp = pd.read_csv(temp_file, index_col=0, parse_dates=True)
    # 이미 계산된 인덱스는 건너뛰기
    computed_indices = df_temp.index[df_temp['mamba_pred'].notna()]
    start_idx = max(start_idx, len(computed_indices) + 1) if len(computed_indices) > 0 else start_idx
    # 임시 파일의 데이터를 현재 df에 병합 (단, 이미 존재하는 컬럼은 덮어쓰지 않음)
    for col in new_cols:
        df[col] = df_temp[col].combine_first(df[col])

for i in tqdm(range(start_idx, end_idx), desc="Processing"):
    try:
        # 현재까지의 데이터로 extractor 실행 (extractor 내부에서 마지막 SEQ_LEN 사용)
        df_slice = df.iloc[:i+1]  # 전체 데이터 전달 (extractor가 알아서 마지막 60 사용)
        pred, emb = extractor.extract(df_slice)
        
        # 결과 저장
        df.loc[df.index[i], 'mamba_pred'] = pred
        for j, val in enumerate(emb):
            df.loc[df.index[i], f'mamba_emb_{j}'] = val
        
        # 일정 간격마다 임시 저장
        if (i - start_idx + 1) % SAVE_INTERVAL == 0:
            df.to_csv(temp_file)
            logger.info(f"Intermediate save at index {i}")
            
    except Exception as e:
        logger.error(f"Error at index {i}: {e}")
        # 해당 인덱스는 NaN으로 남겨둠
        continue

# 5. 최종 저장 (원자적 쓰기)
logger.info(f"Saving final result to {OUTPUT_CSV}")
final_temp = os.path.join(TEMP_DIR, 'mamba_features_final.csv')
df.to_csv(final_temp)
os.replace(final_temp, OUTPUT_CSV)  # 원자적 이동 (Windows에서는 os.replace 사용)

# 임시 파일 정리
if os.path.exists(temp_file):
    os.remove(temp_file)

logger.info("Done.")