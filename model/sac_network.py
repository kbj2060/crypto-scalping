"""
SAC (Soft Actor-Critic) Network Architecture (Fixed Dimension Mismatch)
- Input Projection Layer 추가 (29 -> 128 차원 변환)
- 모든 xLSTM 레이어 차원 통일 (Dimension Mismatch 해결)
- Multi-Layer, Pre-LN, Residual 구조 유지
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import os

# config 모듈 접근
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = __import__('logging').getLogger(__name__)

# ==========================================
# 1. Core Modules
# ==========================================

class MultiHeadAttention(nn.Module):
    """Weighted Pooling이 적용된 어텐션 레이어"""
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_weight = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        weights = torch.softmax(self.pool_weight(x), dim=1)
        context = (x * weights).sum(dim=1)
        return context


class sLSTMCell(nn.Module):
    """sLSTM Cell (Stabilized Exponential Gating)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, h, c, n):
        combined = torch.cat([x, h], dim=-1)
        gates = self.weight(combined)
        i, f, o, z = gates.chunk(4, dim=-1)
        
        # 학습 초기 안정성을 위해 조금 더 보수적인 범위 사용
        # e^4는 약 54, e^5는 약 148. 4~5 사이 추천
        i = torch.clamp(i, min=-4, max=4)
        f = torch.clamp(f, min=-4, max=4)
        i = torch.exp(i)
        f = torch.exp(f)
        
        c_next = f * c + i * torch.tanh(z)
        n_next = f * n + i
        
        c_next = torch.clamp(c_next, min=-1e6, max=1e6)
        n_next = torch.clamp(n_next, min=1e-6, max=1e6)
        
        h_next = torch.sigmoid(o) * (c_next / n_next)
        h_next = torch.nan_to_num(h_next, nan=0.0)
            
        return h_next, c_next, n_next


# ==========================================
# 2. SAC Actor (Fixed)
# ==========================================

class SACActor(nn.Module):
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128, 
                 num_layers=None, dropout=None, log_std_min=-20, log_std_max=2):
        super().__init__()
        # Config 연동
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NETWORK_NUM_LAYERS
        dropout_p = dropout if dropout is not None else config.NETWORK_DROPOUT
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # [FIX] 0. Input Projection (차원 불일치 해결 핵심)
        # Raw Input(29) -> Hidden Dim(128)으로 먼저 변환
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

        # 1. Multi-Layer xLSTM (이제 입력이 무조건 hidden_dim임)
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(self.hidden_dim, self.hidden_dim)  # All layers same size
            for _ in range(self.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        self.attention = MultiHeadAttention(self.hidden_dim, heads=config.NETWORK_ATTENTION_HEADS)
        
        # 2. Info Encoder
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # 3. Backbone
        combined_dim = self.hidden_dim + 64
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # 4. Heads
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, x, info=None, states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        # [FIX] Projection First
        # (Batch, Seq, 29) -> (Batch, Seq, 128)
        x_emb = self.input_proj(x)
        current_input = self.input_norm(x_emb)
        current_input = self.dropout(current_input)
        
        next_h, next_c, next_n = [], [], []
        
        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            layer_h_list = []
            
            # Loop over sequence
            for t in range(seq_len):
                input_t = current_input[:, t, :]  # (Batch, 128)
                
                # Pre-LN (이제 안전함: 128 -> 128)
                x_norm = self.layer_norms[layer_idx](input_t)
                h_t, c_t, n_t = self.xlstm_layers[layer_idx](x_norm, h_t, c_t, n_t)
                
                # Residual Connection
                out_t = h_t + input_t
                layer_h_list.append(out_t.unsqueeze(1))
            
            # Update input for next layer
            current_input = torch.cat(layer_h_list, dim=1)
            # Optional: lighter dropout between layers
            if layer_idx < self.num_layers - 1:
                current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)
            
        context = self.attention(current_input)
        
        if info is None:
            info = torch.zeros(batch_size, 13).to(device)
        info_emb = self.info_encoder(info)
        
        combined = torch.cat([context, info_emb], dim=-1)
        x_emb = self.backbone(combined)
        
        mu = self.mu_head(x_emb)
        log_std = self.log_std_head(x_emb)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        new_states = (torch.stack(next_h), torch.stack(next_c), torch.stack(next_n))
        return mu, log_std, new_states

    def sample(self, x, info=None, states=None):
        """
        Reparameterization Trick Sampling
        Returns:
            action: Tanh applied action [-1, 1] (Continuous)
            log_prob: Log probability of the action
            mean: Tanh applied mean (for deterministic evaluation)
            next_states: LSTM states
        """
        mu, log_std, next_states = self.forward(x, info, states)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)  # Continuous Action [-1, 1]
        
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu), next_states


# ==========================================
# 3. SAC Critic (Fixed)
# ==========================================

class SACCritic(nn.Module):
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128, 
                 num_layers=None, dropout=None):
        super().__init__()
        # Config 연동
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NETWORK_NUM_LAYERS
        dropout_p = dropout if dropout is not None else config.NETWORK_DROPOUT
        
        # [FIX] Input Projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        self.attention = MultiHeadAttention(self.hidden_dim, heads=config.NETWORK_ATTENTION_HEADS)
        
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        combined_dim = self.hidden_dim + 64 + action_dim
        self.sa_encoder = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        self.q1_head = nn.Linear(128, 1)
        self.q2_head = nn.Linear(128, 1)

    def forward(self, x, action, info=None, states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        # [FIX] Projection
        x_emb = self.input_proj(x)
        current_input = self.input_norm(x_emb)
        current_input = self.dropout(current_input)
        
        next_h, next_c, next_n = [], [], []
        
        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            layer_h_list = []
            
            for t in range(seq_len):
                input_t = current_input[:, t, :]
                
                x_norm = self.layer_norms[layer_idx](input_t)
                h_t, c_t, n_t = self.xlstm_layers[layer_idx](x_norm, h_t, c_t, n_t)
                
                out_t = h_t + input_t  # Residual
                layer_h_list.append(out_t.unsqueeze(1))
            
            current_input = torch.cat(layer_h_list, dim=1)
            # Optional: lighter dropout between layers
            if layer_idx < self.num_layers - 1:
                current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)
            
        context = self.attention(current_input)
        
        if info is None:
            info = torch.zeros(batch_size, 13).to(device)
        info_emb = self.info_encoder(info)

        cat_input = torch.cat([context, info_emb, action], dim=-1)
        features = self.sa_encoder(cat_input)
        
        new_states = (torch.stack(next_h), torch.stack(next_c), torch.stack(next_n))
        return self.q1_head(features), self.q2_head(features), new_states
