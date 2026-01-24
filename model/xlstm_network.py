"""
xLSTM 기반 신경망 구조 (Final Upgrade)
[Features]
1. Multi-Layer sLSTM + Pre-LN Residual Connection
2. State Retention (상태 유지 및 반환)
3. Weighted Attention Pooling
4. Info Encoder + Shared Trunk Architecture
5. Dropout & Gradient Checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import sys
import os

# 상위 폴더를 경로에 추가 (config 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MultiHeadAttention(nn.Module):
    """Weighted Pooling이 적용된 어텐션 레이어"""
    def __init__(self, hidden_dim, heads=None):
        super().__init__()
        heads = heads if heads is not None else config.NETWORK_ATTENTION_HEADS
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # [개선 4] 학습 가능한 풀링 가중치 (단순 평균 대체)
        self.pool_weight = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq, hidden)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        
        # Weighted Pooling
        # 시퀀스 내의 각 시점이 얼마나 중요한지 계산 (Softmax)
        weights = torch.softmax(self.pool_weight(x), dim=1)  # (batch, seq, 1)
        context = (x * weights).sum(dim=1)  # (batch, hidden)
        
        return context


class sLSTMCell(nn.Module):
    """sLSTM Cell (안정화 버전): Exponential Gating을 통한 메모리 강화"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gates: i(input), f(forget), o(output), z(cell input)
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, h, c, n, f_prev=None):
        combined = torch.cat([x, h], dim=-1)
        gates = self.weight(combined)
        i, f, o, z = gates.chunk(4, dim=-1)
        
        # 1. 지수 게이트 입력값 제한 (기존 10 -> 5로 하향 조정)
        # exp(5)는 약 148로 수치적 안정성이 훨씬 높습니다.
        i = torch.clamp(i, min=-5, max=5)
        f = torch.clamp(f, min=-5, max=5)
        
        i = torch.exp(i)
        f = torch.exp(f)
        
        # 2. 상태 업데이트
        c_next = f * c + i * torch.tanh(z)
        n_next = f * n + i
        
        # 3. 상태 변수 폭발 방지 (추가)
        # 가중치가 커짐에 따라 c와 n이 무한히 커지는 것을 방지합니다.
        c_next = torch.clamp(c_next, min=-1e6, max=1e6)
        n_next = torch.clamp(n_next, min=1e-6, max=1e6)
        
        h_next = torch.sigmoid(o) * (c_next / n_next)
        
        # 4. 최종 출력 nan 체크 및 방어
        if torch.isnan(h_next).any() or torch.isinf(h_next).any():
            h_next = torch.nan_to_num(h_next, nan=0.0, posinf=0.0, neginf=0.0)
            
        return h_next, c_next, n_next


class xLSTMActorCritic(nn.Module):
    """
    PPO용 xLSTM Actor-Critic (Final Version)
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=None, 
                 num_layers=None, dropout=None, use_checkpointing=None):
        super().__init__()
        # 파라미터 기본값 설정 (config에서 가져오기)
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
        self.dropout = nn.Dropout(dropout)  # [개선 3] Dropout 추가
        
        # 2. xLSTM Stack
        self.xlstm_layers = nn.ModuleList([
            sLSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 3. Attention
        self.attention = MultiHeadAttention(hidden_dim)
        
        # 4. [개선 5] Info Encoder
        info_encoder_dim = config.NETWORK_INFO_ENCODER_DIM
        self.info_encoder = nn.Sequential(
            nn.Linear(info_dim, info_encoder_dim),
            nn.LayerNorm(info_encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(info_encoder_dim, info_encoder_dim)
        )
        
        # 5. [개선 6] Shared Trunk (Late Fusion 이후 공통 처리)
        combined_dim = hidden_dim + info_encoder_dim
        shared_dim1 = config.NETWORK_SHARED_TRUNK_DIM1
        shared_dim2 = config.NETWORK_SHARED_TRUNK_DIM2
        self.shared_trunk = nn.Sequential(
            nn.Linear(combined_dim, shared_dim1),
            nn.LayerNorm(shared_dim1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim1, shared_dim2),
            nn.LayerNorm(shared_dim2),
            nn.GELU(),
        )
        
        # 6. Separate Heads
        actor_dim = config.NETWORK_ACTOR_HEAD_DIM
        critic_dim = config.NETWORK_CRITIC_HEAD_DIM
        self.actor_head = nn.Sequential(
            nn.Linear(shared_dim2, actor_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(actor_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(shared_dim2, critic_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(critic_dim, 1)
        )

    def _process_layer(self, layer_idx, current_input, h_t, c_t, n_t):
        """[개선 7] Checkpointing을 위한 단일 레이어 처리 함수"""
        cell = self.xlstm_layers[layer_idx]
        norm = self.layer_norms[layer_idx]
        
        layer_h_list = []
        seq_len = current_input.size(1)
        
        for t in range(seq_len):
            input_t = current_input[:, t, :]
            
            # [개선 2] Pre-LN Residual Pattern
            # Norm(x) -> Layer -> x + Output
            x_norm = norm(input_t)
            h_t, c_t, n_t = cell(x_norm, h_t, c_t, n_t, None)
            
            # Residual Connection
            out_t = input_t + h_t
            layer_h_list.append(out_t.unsqueeze(1))
            
        return torch.cat(layer_h_list, dim=1), h_t, c_t, n_t

    def forward(self, x, info=None, states=None, return_states=False):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
            info: (batch_size, info_dim) 포지션 정보 텐서 (None이면 0으로 채움)
            states: (h, c, n) 튜플 또는 None
                   - h, c, n: 각각 (num_layers, batch_size, hidden_dim) 형태
            return_states: True면 새 상태도 반환
        Returns:
            action_probs: (batch_size, action_dim) 행동 확률 분포
            value: (batch_size, 1) 상태 가치
            new_states (optional): (h, c, n) 튜플 - 새 상태
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # 상태 초기화
        if states is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            n = torch.ones(self.num_layers, batch_size, self.hidden_dim).to(device)
        else:
            h, c, n = states

        # Input Projection
        x_emb = self.input_proj(x)
        x_emb = self.input_norm(x_emb)
        current_input = self.dropout(x_emb)
        
        next_h, next_c, next_n = [], [], []
        
        # xLSTM Layers Forward
        for layer_idx in range(self.num_layers):
            h_t, c_t, n_t = h[layer_idx], c[layer_idx], n[layer_idx]
            
            if self.use_checkpointing and self.training:
                # Gradient Checkpointing (메모리 절약)
                current_input, h_t, c_t, n_t = checkpoint(
                    self._process_layer, layer_idx, current_input, h_t, c_t, n_t,
                    use_reentrant=False
                )
            else:
                current_input, h_t, c_t, n_t = self._process_layer(
                    layer_idx, current_input, h_t, c_t, n_t
                )
            
            # 레이어 간 Dropout
            current_input = self.dropout(current_input)
            
            next_h.append(h_t)
            next_c.append(c_t)
            next_n.append(n_t)

        # Context Aggregation
        context = self.attention(current_input)  # (batch, hidden)
        
        # Info Encoding
        if info is None:
            info = torch.zeros(batch_size, self.info_dim).to(device)
        info_encoded = self.info_encoder(info)  # (batch, 64)
        
        # Late Fusion & Shared Trunk
        combined = torch.cat([context, info_encoded], dim=-1)
        shared_features = self.shared_trunk(combined)
        
        # Heads
        action_probs = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        # [개선 1] 상태 반환 로직 추가
        if return_states:
            new_states = (
                torch.stack(next_h),
                torch.stack(next_c),
                torch.stack(next_n)
            )
            return action_probs, value, new_states
        else:
            return action_probs, value
