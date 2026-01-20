"""
xLSTM 기반 신경망 구조
sLSTM(scalar LSTM) 구조를 사용하여 지수 게이팅(Exponential Gating)과 정규화(Normalization) 구현
Multi-Head Attention과 심층화된 Actor-Critic 적용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """시퀀스 데이터의 중요 지점을 포착하기 위한 어텐션 레이어"""
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, batch_first=True)
        # 어텐션 이후의 정보를 정리할 정규화 층
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq, hidden)
        # 쿼리, 키, 밸류를 동일하게 설정하여 셀프 어텐션 수행
        attn_output, _ = self.attn(x, x, x)
        # 잔차 연결(Residual Connection) 및 정규화
        x = self.norm(x + attn_output)
        # 전체 시퀀스 정보를 요약 (가중 평균)
        return x.mean(dim=1)


class sLSTMCell(nn.Module):
    """sLSTM Cell: Exponential Gating을 통한 메모리 강화"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gates: i(input), f(forget), o(output), z(cell input)
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    # xlstm_network.py의 sLSTMCell 수정
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
    """개선된 xLSTM: Attention과 Deep Actor-Critic 적용 + Late Fusion"""
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.info_dim = info_dim
        self.xlstm_cell = sLSTMCell(input_dim, hidden_dim)
        
        # [로드맵 1] 시퀀스 전체 정보를 읽는 눈 추가
        self.attention = MultiHeadAttention(hidden_dim, heads=4)
        
        # Late Fusion: 시퀀스 정보(hidden_dim) + 포지션 정보(info_dim) 결합
        combined_dim = hidden_dim + info_dim  # 128 + 13 = 141
        
        # [로드맵 3] 심층화된 Actor: LayerNorm으로 수치 안정성 확보
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # [로드맵 3] 심층화된 Critic: 가치 평가의 정밀도 향상
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, info=None, states=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
            info: (batch_size, info_dim) 포지션 정보 텐서 (None이면 0으로 채움)
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

        # [로드맵 1 적용] 모든 타임스텝의 h를 수집
        all_h = []
        for t in range(seq_len):
            h, c, n = self.xlstm_cell(x[:, t, :], h, c, n, None)
            all_h.append(h.unsqueeze(1))
            
        # (batch, seq_len, hidden_dim) 형태로 결합
        seq_h = torch.cat(all_h, dim=1)
        
        # Attention을 통해 20개 캔들의 핵심 정보를 하나의 컨텍스트로 요약
        context = self.attention(seq_h)  # (batch, hidden_dim)
        
        # Late Fusion: 시퀀스 정보와 포지션 정보 결합
        if info is None:
            info = torch.zeros(batch_size, self.info_dim).to(x.device)
        
        # 결합된 정보: (batch, hidden_dim + info_dim)
        combined = torch.cat([context, info], dim=-1)
        
        # 강화된 Actor/Critic 출력
        action_probs = self.actor(combined)
        value = self.critic(combined)
        
        return action_probs, value
