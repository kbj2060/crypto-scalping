# scripts/precompute_mamba_features.py
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from mamba_predictor import MambaForPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정
CHECKPOINT_PATH = 'data/checkpoints/mamba_predictor.pth'
INPUT_CSV = 'data/training_features.csv'
OUTPUT_CSV = 'data/training_features_with_mamba.csv'
SEQ_LEN = 60
ENC_IN = 21                # 원하는 입력 feature 개수 (가능한 최대)
HIDDEN_DIM = 256
BATCH_SIZE = 1024

# 1. 데이터 로드 (날짜 형식 명시)
logger.info("Loading data...")
df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True,
    date_format='%Y-%m-%d %H:%M:%S'  # 데이터에 맞게 조정 (예: '%Y-%m-%d %H:%M:%S')
)

# 2. 숫자형 컬럼만 자동 선택
all_cols = df.columns.tolist()
# SAMBA/Mamba 관련 컬럼 제외
base_candidates = [c for c in all_cols if not c.startswith(('samba_', 'mamba_'))]
# 숫자형 컬럼만 필터링
numeric_cols = df[base_candidates].select_dtypes(include=[np.number]).columns.tolist()
logger.info(f"Found {len(numeric_cols)} numeric columns among base features.")

if len(numeric_cols) < ENC_IN:
    logger.warning(f"Requested ENC_IN={ENC_IN} but only {len(numeric_cols)} numeric columns available. Using all {len(numeric_cols)}.")
    ENC_IN = len(numeric_cols)
else:
    # ENC_IN만큼 앞에서 선택 (또는 원하는 대로 조정)
    numeric_cols = numeric_cols[:ENC_IN]

base_features = numeric_cols
logger.info(f"Using {len(base_features)} features: {base_features[:5]}...")

# 3. 데이터 추출 (float32 변환)
data = df[base_features].values.astype(np.float32)  # (T, ENC_IN)

# 4. 슬라이딩 윈도우 생성 (메모리 효율을 위해 배치 단위로 처리할 수도 있지만, 여기서는 한 번에 생성)
T = len(data)
X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(SEQ_LEN, ENC_IN))
X = X.reshape(-1, SEQ_LEN, ENC_IN)  # (num_windows, SEQ_LEN, ENC_IN)
logger.info(f"Generated {len(X)} windows.")

# 5. 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaForPrediction(enc_in=ENC_IN, d_model=HIDDEN_DIM, n_layers=4)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# 6. 배치 추론
logger.info("Extracting features in batches...")
all_preds = []
all_hiddens = []

with torch.no_grad():
    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Batches"):
        batch = X[i:i+BATCH_SIZE]
        batch_tensor = torch.from_numpy(batch).to(device)
        pred, hidden = model(batch_tensor)  # (batch, pred_len), (batch, hidden_dim)
        all_preds.append(pred.cpu().numpy())
        all_hiddens.append(hidden.cpu().numpy())

preds = np.concatenate(all_preds, axis=0).squeeze()        # (num_windows,)
hiddens = np.concatenate(all_hiddens, axis=0)              # (num_windows, hidden_dim)

# 7. 원본 데이터프레임에 결과 매핑
df_out = df.copy()
emb_cols = [f'mamba_emb_{i}' for i in range(HIDDEN_DIM)]
new_cols = ['mamba_pred'] + emb_cols

# 새 컬럼 초기화
for col in new_cols:
    df_out[col] = np.nan

# 결과 삽입
start_idx = SEQ_LEN
df_out.iloc[start_idx:start_idx+len(preds), df_out.columns.get_loc('mamba_pred')] = preds
for j in range(HIDDEN_DIM):
    df_out.iloc[start_idx:start_idx+len(hiddens), df_out.columns.get_loc(f'mamba_emb_{j}')] = hiddens[:, j]

# 8. 저장 (원자적 쓰기)
logger.info(f"Saving to {OUTPUT_CSV}...")
temp_file = OUTPUT_CSV + '.tmp'
df_out.to_csv(temp_file)
os.replace(temp_file, OUTPUT_CSV)
logger.info("Done.")