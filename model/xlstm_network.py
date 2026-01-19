"""
xLSTM 기반 신경망 구조
sLSTM(scalar LSTM) 구조를 사용하여 지수 게이팅(Exponential Gating)과 정규화(Normalization) 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMCell(nn.Module):
    """sLSTM Cell: Exponential Gating을 통한 메모리 강화"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gates: i(input), f(forget), o(output), z(cell input)
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, h, c, n, f_prev=None):
        # xLSTM의 핵심: Exponential Gating을 통한 메모리 강화
        combined = torch.cat([x, h], dim=-1)
        gates = self.weight(combined)
        i, f, o, z = gates.chunk(4, dim=-1)
        
        # Exponential gating (지수 게이팅)
        i = torch.exp(i)
        f = torch.exp(f)  # 논문 기반 지수 forget gate
        
        # State updates (정규화 포함)
        c_next = f * c + i * torch.tanh(z)
        n_next = f * n + i  # Normalizer state
        
        # Hidden state output
        h_next = torch.sigmoid(o) * (c_next / (n_next + 1e-8))
        return h_next, c_next, n_next


class xLSTMActorCritic(nn.Module):
    """xLSTM 기반 Actor-Critic 네트워크"""
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.xlstm_cell = sLSTMCell(input_dim, hidden_dim)
        
        # Actor: 정책 분포 (Hold, Long, Short)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: 상태 가치 평가
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, states=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
            states: (h, c, n) 튜플 또는 None
        Returns:
            action_probs: (batch_size, action_dim) 행동 확률 분포
            value: (batch_size, 1) 상태 가치
        """
        batch_size, seq_len, _ = x.size()
        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            n = torch.ones(batch_size, self.hidden_dim).to(x.device)
        else:
            h, c, n = states

        # 시퀀스 처리
        for t in range(seq_len):
            h, c, n = self.xlstm_cell(x[:, t, :], h, c, n, None)
            
        # Actor와 Critic 출력
        action_probs = self.actor(h)
        value = self.critic(h)
        
        return action_probs, value
