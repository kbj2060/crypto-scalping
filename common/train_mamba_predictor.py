# scripts/train_mamba.py
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from macroHFT.mamba_predictor import MambaForPrediction
from common.feature_engineering import ULTIMATE_FEATURE_COLS

# 데이터 준비 (base features만 사용)
base_features = [col for col in ULTIMATE_FEATURE_COLS if not col.startswith('mamba_')]
df = pd.read_csv('data/training_features.csv', index_col=0)
data = df[base_features].values.astype(np.float32)

# 시퀀스 생성
seq_len = 60
pred_len = 1
X, y = [], []
for i in range(len(data) - seq_len - pred_len + 1):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len:i+seq_len+pred_len, 0])  # log_return 예측
X = np.array(X); y = np.array(y)

# 학습/검증 분할
split = int(0.7 * len(X))
train_data = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
val_data = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))

# 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaForPrediction(enc_in=len(base_features)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(20):
    model.train()
    for xb, yb in DataLoader(train_data, batch_size=32, shuffle=True):
        xb, yb = xb.to(device), yb.to(device)
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} 완료")

torch.save(model.state_dict(), 'checkpoints/mamba_predictor.pth')