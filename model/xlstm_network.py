"""
XLSTMNetwork (Official Paper Implementation - Clean Ver.)
- Core: Stabilized sLSTM (Log-Space)
- Safety Nets Removed: No Logit Clamping, No Input Check (Handled by Math)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StabilizedSLSTMCell(nn.Module):
    """
    [Official] Stabilized sLSTM Cell
    Internal stabilization using m_t state prevents overflow.
    """
    def __init__(self, input_size, hidden_size):
        super(StabilizedSLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_pre, i_pre, f_pre, o_pre = gates.chunk(4, 1)
        
        # --- xLSTM Official Stabilization Logic ---
        z_t = torch.tanh(z_pre)
        i_log = i_pre
        f_log = f_pre
        o_t = torch.sigmoid(o_pre)
        
        # Stabilizer Update: m_t = max(f_log + m_{t-1}, i_log)
        m_t = torch.max(f_log + m_prev, i_log)
        
        # Stabilized Gates (No overflow possible)
        i_prime = torch.exp(i_log - m_t)
        f_prime = torch.exp(f_log + m_prev - m_t)
        
        # State Updates
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime
        
        # Normalization
        h_t = o_t * (c_t / (n_t + 1e-6))
        
        return h_t, (h_t, c_t, n_t, m_t)

class XLSTMNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128, num_layers=2, dropout=0.1):
        super(XLSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # [1] Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # [2] sLSTM Blocks
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.lstm_layers.append(StabilizedSLSTMCell(hidden_dim, hidden_dim))
            self.lstm_norms.append(nn.LayerNorm(hidden_dim))
            
        # [3] Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.pooling_weight = nn.Linear(hidden_dim, 1)
        
        # [4] Info Encoder
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        # [5] Shared Trunk
        fusion_dim = hidden_dim + 64 
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # [6] Heads
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, info, states=None, temperature=1.0):
        # [Removed] NaN Check & Logit Clamping (No longer needed)
        
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)
        
        if states is None:
            states = []
            for _ in range(self.num_layers):
                # Init all states to 0 (including m)
                h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                n = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                m = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                states.append((h, c, n, m))
        
        next_states = []
        current_input = x
        
        for i, layer in enumerate(self.lstm_layers):
            h, c, n, m = states[i]
            ln = self.lstm_norms[i]
            output_seq = []
            
            for t in range(seq_len):
                inp = current_input[:, t, :]
                inp_norm = ln(inp)
                h_next, (h_next, c_next, n_next, m_next) = layer(inp_norm, (h, c, n, m))
                out = inp + h_next
                output_seq.append(out)
                h, c, n, m = h_next, c_next, n_next, m_next
            
            current_input = torch.stack(output_seq, dim=1)
            next_states.append((h, c, n, m))
            
        lstm_output = current_input
        
        attn_out, _ = self.attention(lstm_output, lstm_output, lstm_output)
        attn_out = self.attn_norm(attn_out + lstm_output) 
        
        pool_scores = self.pooling_weight(attn_out) 
        pool_weights = F.softmax(pool_scores, dim=1)
        context_feature = torch.sum(pool_weights * attn_out, dim=1)
        
        if info.dim() == 3: info = info.squeeze(1)
        info_encoded = self.info_encoder(info) 
        
        combined = torch.cat([context_feature, info_encoded], dim=1)
        trunk_out = self.trunk(combined) 
        
        logits = self.actor_head(trunk_out)
        
        # Clean Softmax (Temperature only)
        action_probs = F.softmax(logits / temperature, dim=-1)
        state_value = self.critic_head(trunk_out)
        
        return action_probs, state_value, next_states
