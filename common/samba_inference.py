# samba_inference.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import SAMBA  # SAMBA 저장소 내 모델 정의 import

class SAMBAFeatureExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # 모델 파라미터 (사전 학습 시 사용한 값과 동일해야 함)
        model_args = {
            'enc_in': len([col for col in ULTIMATE_FEATURE_COLS if not col.startswith('samba_')]),        # 입력 feature 차원 (ULTIMATE_FEATURE_COLS 길이)
            'd_model': 256,
            'n_layers': 4,
            'seq_len': 60,
            'pred_len': 1,
            'dropout': 0.1,
        }
        
        self.model = SAMBA(**model_args).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 은닉 상태 차원 (d_model)
        self.hidden_dim = model_args['d_model']
    
    @torch.no_grad()
    def extract(self, df, lookback=60):
        """
        df: 최소 lookback 길이 이상의 feature DataFrame
        returns: (pred_return, hidden_state)
        """
        # 입력 feature 선택 (ULTIMATE_FEATURE_COLS와 순서 일치)
        feature_cols = [col for col in ULTIMATE_FEATURE_COLS 
                    if not col.startswith('samba_')]   # 21개
        data = df[feature_cols].values[-lookback:]  # (T, D)
        
        # Tensor 변환 및 배치 차원 추가
        x = torch.FloatTensor(data).unsqueeze(0).to(self.device)  # (1, T, D)
        
        # SAMBA forward (모델 구조에 따라 다를 수 있음)
        # 일반적으로 output = model(x)는 (pred, hidden) 반환
        pred, hidden = self.model(x)  # hidden: (1, T, d_model)
        
        # 마지막 시점의 은닉 상태 사용 (또는 평균)
        last_hidden = hidden[0, -1, :].cpu().numpy()  # (d_model,)
        pred_return = pred[0, 0].item()
        
        return pred_return, last_hidden