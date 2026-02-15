# scripts/precompute_mamba_features.py
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from mamba_predictor import MambaForPrediction  # 모델 직접 import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정
CHECKPOINT_PATH = 'checkpoints/mamba_predictor.pth'
INPUT_CSV = 'data/training_features.csv'
OUTPUT_CSV = 'data/training_features_with_mamba.csv'
SEQ_LEN = 60
ENC_IN = 21          # base_features 개수
HIDDEN_DIM = 256
BATCH_SIZE = 1024    # GPU 메모리에 따라 조절

# 1. 데이터 로드 (numpy로 바로 변환)
logger.info("Loading data...")
df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True,
    date_format='%Y-%m-%d %H:%M:%S'  # 데이터에 맞게 조정 (예: '%Y-%m-%d %H:%M:%S')
)
base_features = [col for col in df.columns if not col.startswith(('samba_', 'mamba_'))][:ENC_IN]
data = df[base_features].values.astype(np.float32)  # (T, enc_in)

# 2. 슬라이딩 윈도우 생성 (벡터화)
T = len(data)
X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(SEQ_LEN, ENC_IN))
X = X.reshape(-1, SEQ_LEN, ENC_IN)  # (T - SEQ_LEN, SEQ_LEN, ENC_IN)

# 3. 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaForPrediction(enc_in=ENC_IN, d_model=HIDDEN_DIM, n_layers=4)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

# 4. 배치 추론
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

# 5. 결과 합치기
preds = np.concatenate(all_preds, axis=0).squeeze()        # (num_windows,)
hiddens = np.concatenate(all_hiddens, axis=0)              # (num_windows, hidden_dim)

# 6. 원본 데이터프레임에 결과 매핑
df_out = df.copy()
# 새 컬럼들 초기화 (이미 있으면 덮어씀)
emb_cols = [f'mamba_emb_{i}' for i in range(HIDDEN_DIM)]
new_cols = ['mamba_pred'] + emb_cols
for col in new_cols:
    df_out[col] = np.nan

# 슬라이딩 윈도우의 마지막 인덱스에 해당하는 위치에 값 채우기
start_idx = SEQ_LEN
df_out.iloc[start_idx:start_idx+len(preds), df_out.columns.get_loc('mamba_pred')] = preds
for j in range(HIDDEN_DIM):
    df_out.iloc[start_idx:start_idx+len(hiddens), df_out.columns.get_loc(f'mamba_emb_{j}')] = hiddens[:, j]

# 7. 저장 (원자적 쓰기)
logger.info(f"Saving to {OUTPUT_CSV}...")
temp_file = OUTPUT_CSV + '.tmp'
df_out.to_csv(temp_file)
os.replace(temp_file, OUTPUT_CSV)
logger.info("Done.")