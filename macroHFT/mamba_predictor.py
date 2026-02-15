# macroHFT/mamba_predictor.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaForPrediction(nn.Module):
    def __init__(self, enc_in, seq_len=60, pred_len=1, d_model=256, n_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(enc_in, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Linear(d_model, pred_len)
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.norm(x)
        last_hidden = x[:, -1, :]
        pred = self.pred_head(last_hidden)
        return pred, last_hidden