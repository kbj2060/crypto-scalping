"""
xLSTM 기반 신경망 구조 (Final Tuned Version)
[Reflected Improvements]
1. xLSTM Dropout 최적화: 시퀀스 중간 끊김 방지를 위해 '마지막 레이어'에만 적용
2. Critic Head 안정화: LayerNorm 추가로 Value Function Variance 감소
3. Attention Heads: Config 유연성 유지 (Stable Default: 4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MultiHeadAttention(nn.Module):
    """Weighted Pooling이 적용된 어텐션 레이어"""
    def __init__(self, hidden_dim, heads=None):
        super().__init__()
        # [개선 3 관련] Head 수가 너무 많으면(dim/head < 16) 노이즈에 취약할 수 있음.
        # hidden=128 기준: 4 heads(dim 32)가 적절, 2 heads(dim 64)는 더 안정적.
        heads = heads if heads is not None else config.NETWORK_ATTENTION_HEADS
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
    """sLSTM Cell (안정화 버전)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, h, c, n, f_prev=None):
        combined = torch.cat([x, h], dim=-1)
        gates = self.weight(combined)
        i, f, o, z = gates.chunk(4, dim=-1)
        
        i = torch.clamp(i, min=-5, max=5)
        f = torch.clamp(f, min=-5, max=5)
        i = torch.exp(i)
        f = torch.exp(f)
        
        c_next = f * c + i * torch.tanh(z)
        n_next = f * n + i
        
        c_next = torch.clamp(c_next, min=-1e6, max=1e6)
        n_next = torch.clamp(n_next, min=1e-6, max=1e6)
        
        h_next = torch.sigmoid(o) * (c_next / n_next)
        
        if torch.isnan(h_next).any() or torch.isinf(h_next).any():
            h_next = torch.nan_to_num(h_next, nan=0.0, posinf=0.0, neginf=0.0)
            
        return h_next, c_next, n_next


class xLSTMActorCritic(nn.Module):
    """
    PPO용 xLSTM Actor-Critic (Final Tuned)
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=None, 
                 num_layers=None, dropout=None, use_checkpointing=None):
        super().__init__()
        
        hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        num_layers = num_layers if num_layers is not None else config.NETWORK_NUM_LAYERS
        dropout = dropout if dropout is not None else config.NETWORK_DROPOUT
        use_checkpointing = use_checkpointing if use_checkpointing is not None else config.NETWORK_USE_CHECKPOINTING
        
        self.hidden_dim = hidden_dim
        self.info_dim = info_dim
        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        
        # 1. Input Processing
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) 
        
        # 2. xLSTM Stack
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 3. Attention
        self.attention = MultiHeadAttention(hidden_dim)
        
        # 4. Info Encoder
        info_encoder_dim = config.NETWORK_INFO_ENCODER_DIM
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(info_encoder_dim, info_encoder_dim)
        )
        
        # 5. Shared Trunk
        combined_dim = hidden_dim + info_encoder_dim
        shared_dim1 = config.NETWORK_SHARED_TRUNK_DIM1
        shared_dim2 = config.NETWORK_SHARED_TRUNK_DIM2
        
        self.shared_trunk = nn.Sequential(
            nn.Linear(combined_dim, shared_dim1),
            nn.LayerNorm(shared_dim1),
            nn.GELU(),
            # Critic 오염 방지를 위해 Shared Trunk Dropout 제거됨
            nn.Linear(shared_dim1, shared_dim2),
            nn.LayerNorm(shared_dim2),
            nn.GELU(),
        )
        
        # 6. Separate Heads
        actor_dim = config.NETWORK_ACTOR_HEAD_DIM
        critic_dim = 32 # Critic 경량화
        
        # [Actor Head]
        self.actor_head = nn.Sequential(
            nn.Linear(shared_dim2, actor_dim),
            nn.GELU(),
            nn.Dropout(dropout), # Policy는 탐험을 위해 Dropout 유지
            nn.Linear(actor_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # [Critic Head] - [개선 2] LayerNorm 추가
        self.critic_head = nn.Sequential(
            nn.Linear(shared_dim2, critic_dim),
            nn.LayerNorm(critic_dim), # [NEW] Value Function 안정화
            nn.GELU(),
            # Critic Dropout 제거됨 (Regression 안정성)
            nn.Linear(critic_dim, 1)
        )

    def _process_layer(self, layer_idx, current_input, h_t, c_t, n_t):
        cell = self.xlstm_layers[layer_idx]
        norm = self.layer_norms[layer_idx]
        
        layer_h_list = []
        seq_len = current_input.size(1)
        
        for t in range(seq_len):
            input_t = current_input[:, t, :]
            x_norm = norm(input_t)
            h_t, c_t, n_t = cell(x_norm, h_t, c_t, n_t, None)
            out_t = input_t + h_t
            layer_h_list.append(out_t.unsqueeze(1))
            
        return torch.cat(layer_h_list, dim=1), h_t, c_t, n_t

    def forward(self, x, info=None, states=None, return_states=False):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        x_emb = self.input_proj(x)
        x_emb = self.input_norm(x_emb)
        # Input level dropout은 데이터 증강 효과가 있어 유지
        current_input = self.dropout(x_emb) 
        
        next_h, next_c, next_n = [], [], []
        
        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            
            if self.use_checkpointing and self.training:
                current_input, h_t, c_t, n_t = checkpoint(
                    self._process_layer, layer_idx, current_input, h_t, c_t, n_t,
                    use_reentrant=False
                )
            else:
                current_input, h_t, c_t, n_t = self._process_layer(
                    layer_idx, current_input, h_t, c_t, n_t
                )
            
            # [개선 1] 마지막 레이어에서만 Dropout 적용 (Temporal Consistency 보호)
            if layer_idx == self.num_layers - 1:
                current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)

        context = self.attention(current_input)
        
        if info is None:
            info = torch.zeros(batch_size, self.info_dim).to(device)
        info_encoded = self.info_encoder(info)
        
        combined = torch.cat([context, info_encoded], dim=-1)
        shared_features = self.shared_trunk(combined)
        
        action_probs = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        if return_states:
            new_states = (
                torch.stack(next_h),
                torch.stack(next_c),
                torch.stack(next_n)
            )
            return action_probs, value, new_states
        else:
            return action_probs, value