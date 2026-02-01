import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. Feature Extractor: CNN + Stabilized sLSTM
# ==============================================================================
class HybridBackbone(nn.Module):
    """
    [Upgrade] CNN(패턴 인식) + xLSTM(시계열 추론) 하이브리드 백본
    - Conv1D: 캔들의 국소적 특징(패턴) 추출
    - xLSTM: 긴 시계열의 맥락 파악
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(HybridBackbone, self).__init__()
        self.hidden_dim = hidden_dim
        
        # [NEW] CNN Layer for Local Pattern Extraction
        # 입력: (Batch, Seq, Dim) -> (Batch, Dim, Seq)로 변환 필요
        self.cnn_block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # Input Projection (CNN 출력 -> LSTM 입력)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # xLSTM Cells (기존 StabilizedSLSTMCell 활용)
        self.lstm_layers = nn.ModuleList([
            StabilizedSLSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, states=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. CNN Processing
        # (B, L, D) -> (B, D, L) for Conv1d
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_block(x_cnn)
        # (B, D, L) -> (B, L, D) Back to sequence
        x_cnn = x_cnn.permute(0, 2, 1)
        
        x = self.input_proj(x_cnn)
        
        # 2. xLSTM Processing
        if states is None:
            states = []
            for _ in range(len(self.lstm_layers)):
                zeros = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                states.append((zeros, zeros, zeros, zeros))
        
        next_states = []
        current_input = x
        
        for i, layer in enumerate(self.lstm_layers):
            h, c, n, m = states[i]
            ln = self.lstm_norms[i]
            output_seq = []
            
            for t in range(seq_len):
                inp = current_input[:, t, :]
                h_next, (h_next, c_next, n_next, m_next) = layer(inp, (h, c, n, m))
                out = ln(inp + h_next)
                output_seq.append(out)
                h, c, n, m = h_next, c_next, n_next, m_next
            
            current_input = torch.stack(output_seq, dim=1)
            next_states.append((h, c, n, m))
            
        last_hidden = current_input[:, -1, :]
        return last_hidden, next_states

# 기존 StabilizedSLSTMCell은 유지 (코드 생략, 파일 상단에 그대로 두세요)
class StabilizedSLSTMCell(nn.Module):
    # ... (기존 코드와 동일) ...
    def __init__(self, input_size, hidden_size):
        super(StabilizedSLSTMCell, self).__init__()
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, x, state):
        h_prev, c_prev, n_prev, m_prev = state
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        z_pre, i_pre, f_pre, o_pre = gates.chunk(4, 1)
        z_t = torch.tanh(z_pre)
        o_t = torch.sigmoid(o_pre)
        i_log = torch.clamp(i_pre, min=-20.0, max=20.0)
        f_log = torch.clamp(f_pre, min=-20.0, max=20.0)
        m_t = torch.max(f_log + m_prev, i_log)
        i_prime = torch.exp(i_log - m_t)
        f_prime = torch.exp(f_log + m_prev - m_t)
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime
        h_t = o_t * (c_t / (n_t + 1e-6))
        if torch.isnan(h_t).any():
            h_t = torch.nan_to_num(h_t, nan=0.0)
            c_t = torch.nan_to_num(c_t, nan=0.0)
            n_t = torch.nan_to_num(n_t, nan=0.0)
        return h_t, (h_t, c_t, n_t, m_t)

# ==============================================================================
# 3. Decoupled Network (Main Wrapper)
# ==============================================================================
class XLSTMNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, info_dim=15, hidden_dim=128, num_layers=1, dropout=0.1):
        super(XLSTMNetwork, self).__init__()
        
        # [핵심] Actor와 Critic을 완전히 분리 (Decoupling)
        
        # --- Actor Network (Policy) ---
        self.actor_backbone = HybridBackbone(input_dim, hidden_dim, num_layers, dropout)
        self.actor_info = nn.Linear(info_dim, 64)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # --- Critic Network (Value) ---
        self.critic_backbone = HybridBackbone(input_dim, hidden_dim, num_layers, dropout)
        self.critic_info = nn.Linear(info_dim, 64)
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
        if info.dim() == 3: info = info.squeeze(1)
        
        # 1. Actor Forward
        # states는 (actor_states, critic_states) 튜플로 관리되거나 stateless 학습 시 None
        actor_feat, _ = self.actor_backbone(x, states=None) 
        info_emb_a = F.gelu(self.actor_info(info))
        logits = self.actor_head(torch.cat([actor_feat, info_emb_a], dim=1))
        
        # Temperature Scaling (탐험 조절)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # 2. Critic Forward
        critic_feat, _ = self.critic_backbone(x, states=None)
        info_emb_c = F.gelu(self.critic_info(info))
        value = self.critic_head(torch.cat([critic_feat, info_emb_c], dim=1))
        
        return probs, value, None