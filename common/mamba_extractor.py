# macroHFT/mamba_extractor.py
import torch
import pandas as pd
from common.feature_engineering import ULTIMATE_FEATURE_COLS
from .mamba_predictor import MambaForPrediction

class MambaFeatureExtractor:
    def __init__(self, checkpoint_path, enc_in, device='cuda'):
        self.device = device
        self.model = MambaForPrediction(enc_in=enc_in)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.enc_in = enc_in
        self.hidden_dim = 256  # d_model과 동일

    @torch.no_grad()
    def extract(self, df):
        base_features = [col for col in ULTIMATE_FEATURE_COLS if not col.startswith('samba_')]
        # 필요한 feature 수만큼 선택 (enc_in에 맞춤)
        feature_cols = base_features[:self.enc_in]
        data = df[feature_cols].values[-60:]
        x = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        pred, hidden = self.model(x)
        return pred.item(), hidden[0].cpu().numpy()