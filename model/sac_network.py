"""
Continuous SAC Network Architecture
- Multi-Layer xLSTM + Pre-LN Residuals
- Gaussian Policy (Continuous Action with Tanh Squashing)
- Critic accepts [State, Continuous Action]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .xlstm_network import sLSTMCell, MultiHeadAttention
import sys
import os

# config 모듈 접근
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = __import__('logging').getLogger(__name__)


class SACActor(nn.Module):
    """
    Gaussian Policy Actor for Continuous SAC
    Outputs Mean and LogStd for Normal Distribution
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=None, 
                 num_layers=None, dropout=None, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        # Config 연동
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NETWORK_NUM_LAYERS
        dropout_p = dropout if dropout is not None else config.NETWORK_DROPOUT
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 1. Feature Extractor (Shared xLSTM Structure)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        self.attention = MultiHeadAttention(self.hidden_dim, heads=config.NETWORK_ATTENTION_HEADS)
        
        # 2. Deep Info Encoder
        info_encoder_dim = config.NETWORK_INFO_ENCODER_DIM
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(info_encoder_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU()
        )
        
        # 3. Shared Backbone
        combined_dim = self.hidden_dim + info_encoder_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, config.NETWORK_SHARED_TRUNK_DIM1),
            nn.LayerNorm(config.NETWORK_SHARED_TRUNK_DIM1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(config.NETWORK_SHARED_TRUNK_DIM1, config.NETWORK_SHARED_TRUNK_DIM2),
            nn.LayerNorm(config.NETWORK_SHARED_TRUNK_DIM2),
            nn.GELU()
        )
        
        # 4. Heads (Mean & LogStd)
        self.mu_head = nn.Linear(config.NETWORK_SHARED_TRUNK_DIM2, action_dim)
        self.log_std_head = nn.Linear(config.NETWORK_SHARED_TRUNK_DIM2, action_dim)

    def forward(self, x, info=None, states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # 상태 초기화
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        # Feature Extraction
        current_input = self.dropout(self.input_norm(self.input_proj(x)))
        next_h, next_c, next_n = [], [], []

        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            cell = self.xlstm_layers[layer_idx]
            norm = self.layer_norms[layer_idx]
            
            layer_outputs = []
            for t in range(seq_len):
                input_t = current_input[:, t, :]
                x_norm = norm(input_t)
                h_t, c_t, n_t = cell(x_norm, h_t, c_t, n_t, None)
                out_t = input_t + h_t  # Residual
                layer_outputs.append(out_t.unsqueeze(1))
            
            current_input = torch.cat(layer_outputs, dim=1)
            current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)

        context = self.attention(current_input)
        if info is None: 
            info = torch.zeros(batch_size, 13).to(device)
        info_emb = self.info_encoder(info)
        
        combined = torch.cat([context, info_emb], dim=-1)
        features = self.backbone(combined)
        
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std, (torch.stack(next_h), torch.stack(next_c), torch.stack(next_n))

    def sample(self, x, info=None, states=None):
        """
        Reparameterization Trick Sampling
        Returns:
            action: Tanh applied action [-1, 1]
            log_prob: Log probability of the action
            mean: Tanh applied mean (for deterministic evaluation)
            next_states: LSTM states
        """
        mu, log_std, next_states = self.forward(x, info, states)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # Reparameterization Trick (z ~ N(0, 1))
        z = dist.rsample()
        action = torch.tanh(z)
        
        # Tanh Squash Correction for Log Prob
        # log_prob = log_prob(z) - log(1 - tanh(z)^2)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu), next_states


class SACCritic(nn.Module):
    """
    Continuous Critic
    Accepts State + Continuous Action (Scalar/Vector)
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=None, 
                 num_layers=None, dropout=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.NETWORK_HIDDEN_DIM
        self.num_layers = num_layers if num_layers is not None else config.NETWORK_NUM_LAYERS
        dropout_p = dropout if dropout is not None else config.NETWORK_DROPOUT

        # 1. Feature Extractor (xLSTM)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        self.attention = MultiHeadAttention(self.hidden_dim, heads=config.NETWORK_ATTENTION_HEADS)
        
        # 2. Deep Info Encoder
        info_encoder_dim = config.NETWORK_INFO_ENCODER_DIM
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(info_encoder_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU()
        )
        
        # 3. State-Action Encoder
        # [중요] action_dim(Continuous Scalar/Vector)을 concat
        combined_dim = self.hidden_dim + info_encoder_dim + action_dim
        
        self.sa_encoder = nn.Sequential(
            nn.Linear(combined_dim, config.NETWORK_SHARED_TRUNK_DIM1),
            nn.LayerNorm(config.NETWORK_SHARED_TRUNK_DIM1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(config.NETWORK_SHARED_TRUNK_DIM1, config.NETWORK_SHARED_TRUNK_DIM2),
            nn.LayerNorm(config.NETWORK_SHARED_TRUNK_DIM2),
            nn.GELU()
        )
        
        # 4. Twin Heads
        self.q1_head = nn.Linear(config.NETWORK_SHARED_TRUNK_DIM2, 1)
        self.q2_head = nn.Linear(config.NETWORK_SHARED_TRUNK_DIM2, 1)

    def forward(self, x, action, info=None, states=None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        # Feature Extraction
        current_input = self.dropout(self.input_norm(self.input_proj(x)))
        next_h, next_c, next_n = [], [], []

        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            cell = self.xlstm_layers[layer_idx]
            norm = self.layer_norms[layer_idx]
            
            layer_outputs = []
            for t in range(seq_len):
                input_t = current_input[:, t, :]
                x_norm = norm(input_t)
                h_t, c_t, n_t = cell(x_norm, h_t, c_t, n_t, None)
                out_t = input_t + h_t  # Residual
                layer_outputs.append(out_t.unsqueeze(1))
            
            current_input = torch.cat(layer_outputs, dim=1)
            current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)

        context = self.attention(current_input)
        if info is None: 
            info = torch.zeros(batch_size, 13).to(device)
        info_emb = self.info_encoder(info)

        # State + Continuous Action Concat
        cat_input = torch.cat([context, info_emb, action], dim=-1)
        features = self.sa_encoder(cat_input)
        
        return self.q1_head(features), self.q2_head(features), (torch.stack(next_h), torch.stack(next_c), torch.stack(next_n))
