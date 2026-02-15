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
ENC_IN = 21
HIDDEN_DIM = 256
BATCH_SIZE = 1024

# 1. 데이터 로드
logger.info("Loading data...")
df = pd.read_csv(
    INPUT_CSV,
    index_col=0,
    parse_dates=True,
    date_format='%Y-%m-%d %H:%M:%S'
)

# 2. 숫자형 컬럼 선택 (samba/mamba 제외)
base_candidates = [c for c in df.columns if not c.startswith(('samba_', 'mamba_'))]
numeric_cols = df[base_candidates].select_dtypes(include=[np.number]).columns.tolist()
logger.info(f"Found {len(numeric_cols)} numeric columns.")

if len(numeric_cols) < ENC_IN:
    logger.warning(f"ENC_IN={ENC_IN} > available {len(numeric_cols)}. Using all.")
    ENC_IN = len(numeric_cols)
else:
    numeric_cols = numeric_cols[:ENC_IN]

base_features = numeric_cols
logger.info(f"Using {len(base_features)} features: {base_features[:5]}...")

# 3. 데이터 추출
data = df[base_features].values.astype(np.float32)

# 4. 슬라이딩 윈도우 생성 (올바른 개수 계산)
T = len(data)
num_windows = T - SEQ_LEN + 1  # 🔥 중요: +1 필요
logger.info(f"T={T}, SEQ_LEN={SEQ_LEN}, num_windows={num_windows}")

X_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=(SEQ_LEN, ENC_IN))
X = X_view.reshape(-1, SEQ_LEN, ENC_IN)
assert X.shape[0] == num_windows, f"X shape mismatch: {X.shape[0]} vs {num_windows}"

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
    for i in tqdm(range(0, num_windows, BATCH_SIZE), desc="Batches"):
        batch = X[i:i+BATCH_SIZE].copy()
        batch_tensor = torch.from_numpy(batch).to(device)
        pred, hidden = model(batch_tensor)
        all_preds.append(pred.cpu().numpy())
        all_hiddens.append(hidden.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
hiddens = np.concatenate(all_hiddens, axis=0)

# preds 차원 정리
logger.info(f"Preds shape before squeeze: {preds.shape}")
if preds.ndim == 2 and preds.shape[1] == 1:
    preds = preds.squeeze(axis=1)
elif preds.ndim != 1:
    raise ValueError(f"Unexpected preds shape: {preds.shape}")
logger.info(f"Preds shape after squeeze: {preds.shape}")

assert len(preds) == num_windows, f"Preds length mismatch: {len(preds)} vs {num_windows}"
assert hiddens.shape[0] == num_windows, f"Hiddens length mismatch: {hiddens.shape[0]} vs {num_windows}"

# 7. 새 컬럼 데이터 준비
emb_cols = [f'mamba_emb_{i}' for i in range(HIDDEN_DIM)]
new_cols = ['mamba_pred'] + emb_cols

# 새 데이터프레임 생성 (모두 NaN)
new_data = pd.DataFrame(index=df.index, columns=new_cols, dtype=np.float32)

# 타깃 인덱스 (각 윈도우의 마지막 시점)
start_idx = SEQ_LEN - 1
target_indices = df.index[start_idx:start_idx + num_windows]
logger.info(f"target_indices length: {len(target_indices)}")

# 값 할당 (.loc 사용)
new_data.loc[target_indices, 'mamba_pred'] = preds
for j in range(HIDDEN_DIM):
    new_data.loc[target_indices, f'mamba_emb_{j}'] = hiddens[:, j]

# 8. 기존 df와 병합
df_out = pd.concat([df, new_data], axis=1)

# 9. 저장 (원자적 쓰기)
logger.info(f"Saving to {OUTPUT_CSV}...")
temp_file = OUTPUT_CSV + '.tmp'
df_out.to_csv(temp_file)
os.replace(temp_file, OUTPUT_CSV)
logger.info("Done.")