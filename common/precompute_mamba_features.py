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
    date_format='%Y-%m-%d %H:%M:%S'  # 데이터 형식에 맞게 조정
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
    numeric_cols = numeric_cols[:ENC_IN]   # ENC_IN만큼 앞에서 선택 (또는 원하는 대로)

base_features = numeric_cols
logger.info(f"Using {len(base_features)} features: {base_features[:5]}...")

# 3. 데이터 추출 (float32 변환)
data = df[base_features].values.astype(np.float32)  # (T, ENC_IN)

# 4. 슬라이딩 윈도우 생성 (읽기 전용 뷰 반환 주의)
T = len(data)
# sliding_window_view는 view이므로 나중에 copy 필요
X_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=(SEQ_LEN, ENC_IN))
X = X_view.reshape(-1, SEQ_LEN, ENC_IN)  # (num_windows, SEQ_LEN, ENC_IN)
num_windows = X.shape[0]
logger.info(f"Generated {num_windows} windows.")

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
        batch = X[i:i+BATCH_SIZE].copy()  # 🔥 copy()로 쓰기 가능한 배열 생성
        batch_tensor = torch.from_numpy(batch).to(device)
        pred, hidden = model(batch_tensor)  # (batch, pred_len), (batch, hidden_dim)
        all_preds.append(pred.cpu().numpy())
        all_hiddens.append(hidden.cpu().numpy())

preds = np.concatenate(all_preds, axis=0).squeeze()        # (num_windows,)
hiddens = np.concatenate(all_hiddens, axis=0)              # (num_windows, hidden_dim)

# 길이 확인
assert len(preds) == num_windows, f"Prediction length mismatch: {len(preds)} vs {num_windows}"
assert hiddens.shape[0] == num_windows, f"Hidden length mismatch: {hiddens.shape[0]} vs {num_windows}"

# 7. 원본 데이터프레임에 결과 매핑 (단편화 방지를 위해 pd.concat 사용)
# 새 컬럼 데이터 준비
emb_cols = [f'mamba_emb_{i}' for i in range(HIDDEN_DIM)]
new_cols = ['mamba_pred'] + emb_cols

# 새 컬럼 값을 담을 DataFrame 생성 (인덱스 동일, 모든 값 NaN)
new_data = pd.DataFrame(index=df.index, columns=new_cols, dtype=np.float32)

# 결과 삽입 (iloc로 위치 기반 할당)
start_idx = SEQ_LEN
# mamba_pred
new_data.iloc[start_idx:start_idx+num_windows, new_data.columns.get_loc('mamba_pred')] = preds
# mamba_emb_*
for j in range(HIDDEN_DIM):
    new_data.iloc[start_idx:start_idx+num_windows, new_data.columns.get_loc(f'mamba_emb_{j}')] = hiddens[:, j]

# 기존 df와 새 데이터 병합 (컬럼 순서는 뒤에 붙음)
df_out = pd.concat([df, new_data], axis=1)

# 8. 저장 (원자적 쓰기)
logger.info(f"Saving to {OUTPUT_CSV}...")
temp_file = OUTPUT_CSV + '.tmp'
df_out.to_csv(temp_file)
os.replace(temp_file, OUTPUT_CSV)
logger.info("Done.")