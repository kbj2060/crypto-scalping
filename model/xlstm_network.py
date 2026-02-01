import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. Stabilized sLSTM Cell (Core Engine)
# ==============================================================================
class StabilizedSLSTMCell(nn.Module):
    """
    xLSTM의 핵심 셀 (Scalar LSTM with Exponential Gating)
    """
    def __init__(self, input_size, hidden_size):
        super(StabilizedSLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 4배 크기 (z, i, f, o 게이트)
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_pre, i_pre, f_pre, o_pre = gates.chunk(4, 1)
        
        # Activations
        z_t = torch.tanh(z_pre)
        o_t = torch.sigmoid(o_pre)
        
        # Log-Space Stabilization (Exponential Explosion 방지)
        m_t = torch.max(f_pre + m_prev, i_pre)
        i_prime = torch.exp(i_pre - m_t)
        f_prime = torch.exp(f_pre + m_prev - m_t)
        
        # State Updates
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime
        
        # Output Calculation
        h_t = o_t * (c_t / (n_t + 1e-6))
        
        return h_t, (h_t, c_t, n_t, m_t)

# ==============================================================================
# 2. xLSTM Backbone (Feature Extractor)
# ==============================================================================
class SLSTMBackbone(nn.Module):
    """
    시계열 데이터를 받아 압축된 특징 벡터(Last Hidden)를 반환하는 모듈
    Actor와 Critic이 각각 이 클래스의 인스턴스를 하나씩 가짐 (파라미터 분리)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(SLSTMBackbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Stacked sLSTM Cells
        self.lstm_layers = nn.ModuleList([
            StabilizedSLSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, states=None):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)
        
        # 초기 상태 생성 (필요 시)
        if states is None:
            states = []
            for _ in range(self.num_layers):
                zeros = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                states.append((zeros, zeros, zeros, zeros)) # h, c, n, m
        
        next_states = []
        current_input = x
        
        # LSTM Layer Loop
        for i, layer in enumerate(self.lstm_layers):
            h, c, n, m = states[i]
            ln = self.lstm_norms[i]
            output_seq = []
            
            # Time Step Loop
            for t in range(seq_len):
                inp = current_input[:, t, :]
                h_next, (h_next, c_next, n_next, m_next) = layer(inp, (h, c, n, m))
                
                # Residual + Norm
                out = ln(inp + h_next)
                output_seq.append(out)
                
                h, c, n, m = h_next, c_next, n_next, m_next
            
            current_input = torch.stack(output_seq, dim=1)
            next_states.append((h, c, n, m))
            
        # Attention 대신 Last Hidden State 사용 (단순화 & 안정화)
        last_hidden = current_input[:, -1, :]
        
        return last_hidden, next_states

# ==============================================================================
# 3. Decoupled Network (Main Wrapper)
# ==============================================================================
class XLSTMNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, info_dim=15, hidden_dim=128, num_layers=1, dropout=0.1):
        super(XLSTMNetwork, self).__init__()
        
        # -----------------------------------------------------------
        # [Actor Trunk]: 매매 행동 결정 (Policy)
        # -----------------------------------------------------------
        self.actor_backbone = SLSTMBackbone(input_dim, hidden_dim, num_layers, dropout)
        self.actor_info_enc = nn.Linear(info_dim, 64)
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # -----------------------------------------------------------
        # [Critic Trunk]: 가치 평가 (Value) - Actor와 완전히 분리됨!
        # -----------------------------------------------------------
        self.critic_backbone = SLSTMBackbone(input_dim, hidden_dim, num_layers, dropout)
        self.critic_info_enc = nn.Linear(info_dim, 64)
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info, states=None, temperature=1.0):
        # info 차원 정리
        if info.dim() == 3: info = info.squeeze(1)
        
        # -----------------------------------------------------------
        # 1. Actor Forward Path
        # -----------------------------------------------------------
        # Actor용 상태가 있으면 분리해서 사용 (구조상 states는 (actor_states, critic_states) 튜플로 관리 권장)
        # 여기서는 편의상 states=None으로 stateless 학습을 가정하거나, 외부에서 관리
        # PPO는 보통 Rollout 시 stateless로 사용하거나 매 step reset하므로 None 처리
        
        actor_feat, _ = self.actor_backbone(x, states=None)
        actor_info = F.gelu(self.actor_info_enc(info))
        actor_input = torch.cat([actor_feat, actor_info], dim=1)
        
        logits = self.actor_head(actor_input)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # -----------------------------------------------------------
        # 2. Critic Forward Path
        # -----------------------------------------------------------
        critic_feat, _ = self.critic_backbone(x, states=None)
        critic_info = F.gelu(self.critic_info_enc(info))
        critic_input = torch.cat([critic_feat, critic_info], dim=1)
        
        value = self.critic_head(critic_input)
        
        return probs, value, None
