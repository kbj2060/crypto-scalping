"""
Dueling GRU Network + Attention Pooling
시계열의 중요 시점에 가중치를 부여하여 긴 Lookback(60)을 효율적으로 학습
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """시계열의 중요 시점에 집중하는 어텐션 풀링 레이어"""
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim) 입력 시퀀스
        Returns:
            context: (batch, hidden_dim) 가중 평균된 컨텍스트 벡터
        """
        # 어텐션 가중치 계산
        weights = self.attention(x)  # (batch, seq_len, 1)
        # 가중 평균 (Context Vector)
        context = torch.sum(x * weights, dim=1)  # (batch, hidden_dim)
        return context


class DuelingGRU(nn.Module):
    """Dueling GRU Network with Attention Pooling"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, action_dim=3):
        """
        Args:
            input_dim: 입력 피처 개수
            hidden_dim: GRU hidden dimension
            num_layers: GRU 레이어 수
            action_dim: 행동 개수 (3: Long/Short/Hold)
        """
        super(DuelingGRU, self).__init__()
        
        # 1. Feature Extractor (GRU)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 2. [신규] Temporal Attention Layer
        # 60개 시점 중 중요한 순간을 찾아냄
        self.attention = AttentionPooling(hidden_dim)
        
        # 3. Dueling Heads
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
        Returns:
            q_values: (batch_size, action_dim) Q-값
        """
        # GRU 통과
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        # [변경] 단순 마지막 값 사용 -> 어텐션 풀링 사용
        # last_hidden = gru_out[:, -1, :] 
        context_vector = self.attention(gru_out)  # (batch, hidden_dim)
        
        # Dueling Logic
        value = self.value_stream(context_vector)  # (batch, 1)
        advantage = self.advantage_stream(context_vector)  # (batch, action_dim)
        
        # Q-Value 결합 공식
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
