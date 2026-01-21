"""
Dueling GRU Network 모델
Double DQN을 위한 신경망 구조
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingGRU(nn.Module):
    """Dueling GRU Network: 상태 가치와 행동 우위를 분리하여 계산"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, action_dim=3):
        """
        Args:
            input_dim: 입력 피처 개수 (XGBoost로 선택된 피처 수)
            hidden_dim: GRU hidden dimension
            num_layers: GRU 레이어 수
            action_dim: 행동 개수 (3: Long/Short/Hold)
        """
        super(DuelingGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim
        
        # 핵심 레이어: 2-Layer GRU
        # LSTM 대신 GRU를 사용하여 연산 속도를 높이고 파라미터를 줄입니다
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Dueling Head: GRU의 마지막 시점 출력을 받아 두 갈래로 나뉨
        # Value Stream (V): 현재 시장 상황 자체가 유리한지 불리한지 판단
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage Stream (A): 각 행동 간의 상대적 우위를 판단
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
        Returns:
            q_values: (batch_size, action_dim) Q-값
        """
        # GRU 처리
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        
        # 마지막 시점의 hidden state 사용
        last_hidden = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Value Stream: V(s)
        value = self.value_stream(last_hidden)  # (batch_size, 1)
        
        # Advantage Stream: A(s, a)
        advantage = self.advantage_stream(last_hidden)  # (batch_size, action_dim)
        
        # 최종 Q-Value 결합 공식
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
