import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append('.')
from common.feature_engineering import ULTIMATE_FEATURE_COLS

# ================== 설정 ==================
DATA_PATH = 'data/training_features.csv'
FEATURE_COLS = ULTIMATE_FEATURE_COLS
SEQ_LEN = 60          # 과거 60스텝 보고 예측
PRED_HORIZON = 1      # 다음 1스텝 return 예측
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
EPOCHS = 20           # 가볍게
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================== 데이터 준비 ==================
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

# return 계산
df['return'] = df['close'].pct_change()
df = df.dropna()

# feature 컬럼 필터링 (존재하는 것만 사용)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"사용 가능한 피처 수: {len(FEATURE_COLS)}")

# feature 정규화 (간단 min-max)
features = df[FEATURE_COLS].values
mean, std = features.mean(0), features.std(0) + 1e-8
features = (features - mean) / std

# ================== LSTM 모델 ==================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # 마지막 타임스텝만

model = SimpleLSTM(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

# ================== 학습 (전체 데이터로 한 번만) ==================
X, y = [], []
for i in range(SEQ_LEN, len(features) - PRED_HORIZON):
    X.append(features[i-SEQ_LEN:i])
    y.append(df['return'].iloc[i + PRED_HORIZON - 1])   # 다음 return

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

print(f"학습 데이터: {X.shape}")

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X))
    total_loss = 0
    for i in range(0, len(X), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        batch_x = X[idx].to(DEVICE)
        batch_y = y[idx].to(DEVICE)
        
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(X):.8f}")

# ================== Rolling Inference ==================
model.eval()
preds = []
with torch.no_grad():
    for i in tqdm(range(SEQ_LEN, len(features))):
        seq = torch.tensor(features[i-SEQ_LEN:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_ret = model(seq).item()
        preds.append(pred_ret)

# 앞부분 padding
pad = [0.0] * SEQ_LEN
df['lstm_pred_return'] = pad + preds

# 저장
os.makedirs('data', exist_ok=True)
df.to_csv('data/training_features_with_lstm.csv')
df[['lstm_pred_return']].to_csv('data/lstm_features.csv')
print("✅ Tiny LSTM feature 생성 완료! → data/lstm_features.csv")