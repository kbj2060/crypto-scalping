"""
Dueling GRU Network + Attention Pooling (Improved)
LayerNorm, GELU, Dropout, Orthogonal Initialization 적용으로 안정성 강화
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
    """Dueling GRU Network with Attention Pooling (Improved)"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, action_dim=3):
        """
        Args:
            input_dim: 입력 피처 개수
            hidden_dim: GRU hidden dimension
            num_layers: GRU 레이어 수
            action_dim: 행동 개수 (3: Long/Short/Hold)
        """
        super(DuelingGRU, self).__init__()
        
        # 1. GRU Layer (Dropout 유지)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0  # 드롭아웃 약간 상향 (0.2 -> 0.3)
        )
        
        # 2. Attention Layer
        self.attention = AttentionPooling(hidden_dim)
        
        # [신규] 3. Layer Normalization (학습 안정화의 핵심)
        # 어텐션을 거친 컨텍스트 벡터를 정규화하여 값의 폭주를 막음
        self.ln = nn.LayerNorm(hidden_dim)
        
        # 4. Dueling Heads (구조 업그레이드)
        # ReLU -> GELU 변경
        # Dropout 추가 (과적합 방지)
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),              # [변경] 더 부드러운 활성화 함수
            nn.Dropout(0.1),        # [신규] 과적합 방지
            nn.Linear(64, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),              # [변경]
            nn.Dropout(0.1),        # [신규]
            nn.Linear(64, action_dim)
        )
        
        # [신규] 5. 가중치 초기화 (학습 속도 향상)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Orthogonal Initialization for GRU, Xavier for Linear"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)  # 순환 신경망엔 직교 초기화가 국룰
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
        Returns:
            q_values: (batch_size, action_dim) Q-값
        """
        # GRU 통과
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        
        # Attention Pooling
        context_vector = self.attention(gru_out)  # (batch, hidden_dim)
        
        # [신규] Layer Normalization 적용
        context_vector = self.ln(context_vector)
        
        # Dueling Logic
        value = self.value_stream(context_vector)  # (batch, 1)
        advantage = self.advantage_stream(context_vector)  # (batch, action_dim)
        
        # Q-Value 결합 공식
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
