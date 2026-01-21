"""
경량화된 Hybrid xLSTM-Transformer-VSN 신경망 구조
- xLSTM: 지수적 기억을 통한 시계열 특징 추출
- TransformerBlock: FFN이 포함된 경량 Transformer (비선형성 보강)
- VSN: 지표 데이터의 선별적 집중
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. VSN을 위한 하위 모듈 (Gated Residual Network) ---
class GatedLinearUnit(nn.Module):
    """지수 게이팅과 유사한 메커니즘으로 불필요한 신호 억제"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        return x[:, :x.shape[1]//2] * torch.sigmoid(x[:, x.shape[1]//2:])


class GatedResidualNetwork(nn.Module):
    """TFT의 핵심 구성 요소: 비선형 처리와 잔차 연결 결합"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.project = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.project(x)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(self.glu(x))
        return self.norm(x + residual)


# --- 2. 변수 선택 네트워크 (VSN) ---
class VariableSelectionNetwork(nn.Module):
    """13개의 지표 중 현재 상황에 가장 중요한 변수를 선택"""
    def __init__(self, input_dim, num_vars, hidden_dim):
        super().__init__()
        self.num_vars = num_vars
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim) for _ in range(num_vars)
        ])
        self.selector_grn = GatedResidualNetwork(input_dim, hidden_dim, num_vars)

    def forward(self, x):
        # x: (batch, 13)
        # 1. 각 변수별 가중치 계산
        weights = F.softmax(self.selector_grn(x), dim=-1).unsqueeze(1)  # (batch, 1, num_vars)
        
        # 2. 각 변수를 독립적으로 hidden_dim으로 투영
        var_outputs = torch.cat([
            self.var_grns[i](x[:, i:i+1]).unsqueeze(1) for i in range(self.num_vars)
        ], dim=1)  # (batch, 13, hidden_dim)
        
        # 3. 가중치를 적용하여 결합 (Weighted Sum)
        selected_output = torch.matmul(weights, var_outputs).squeeze(1)  # (batch, hidden_dim)
        
        return selected_output


# --- 3. sLSTM Cell ---
class sLSTMCell(nn.Module):
    """sLSTM Cell: Exponential Gating을 통한 메모리 강화"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Gates: i(input), f(forget), o(output), z(cell input)
        self.weight = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, h, c, n):
        combined = torch.cat([x, h], dim=-1)
        gates = self.weight(combined)
        i, f, o, z = gates.chunk(4, dim=-1)
        
        # 지수 게이트 입력값 제한 (수치적 안정성)
        i = torch.exp(torch.clamp(i, -5, 5))
        f = torch.exp(torch.clamp(f, -5, 5))
        
        # 상태 업데이트
        c_next = f * c + i * torch.tanh(z)
        n_next = f * n + i
        
        # 상태 변수 폭발 방지
        c_next = torch.clamp(c_next, min=-1e6, max=1e6)
        n_next = torch.clamp(n_next, min=1e-6, max=1e6)
        
        # 출력 계산 (나눗셈 안정성을 위해 작은 값 추가)
        h_next = torch.sigmoid(o) * (c_next / (n_next + 1e-6))
        
        # NaN/Inf 체크 및 방어
        if torch.isnan(h_next).any() or torch.isinf(h_next).any():
            h_next = torch.nan_to_num(h_next, nan=0.0, posinf=0.0, neginf=0.0)
            
        return h_next, c_next, n_next


# --- 4. 경량 TransformerBlock (FFN 포함) ---
class TransformerBlock(nn.Module):
    """Attention의 단순함을 FFN으로 보강하되 1층만 사용하여 경량화 유지"""
    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        # 비선형 변환을 위한 FFN 추가
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention + Residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + Residual (비선형 패턴 학습)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# --- 5. 최종 경량화된 Actor-Critic 모델 ---
class xLSTMActorCritic(nn.Module):
    """경량화된 Hybrid xLSTM-Attention-VSN 모델"""
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 시퀀스 처리: xLSTM (유지)
        self.xlstm_cell = sLSTMCell(input_dim, hidden_dim)
        
        # 2. 전역 맥락 요약: FFN이 포함된 경량 TransformerBlock
        # 단순 Attention의 비선형성 부족 문제를 해결하기 위해 FFN 추가
        self.transformer = TransformerBlock(hidden_dim)
        
        # 3. 지표 선별: VSN (유지 - 매우 중요)
        self.vsn = VariableSelectionNetwork(info_dim, info_dim, hidden_dim)
        
        combined_dim = hidden_dim + hidden_dim  # 128 + 128 = 256
        
        # 4. 결정부: 층의 깊이를 줄여 빠른 수렴 유도
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, info=None, states=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 입력 시퀀스
            info: (batch_size, info_dim) 포지션 정보 텐서 (None이면 0으로 채움)
            states: (h, c, n) 튜플 또는 None (호환성을 위해 유지)
        Returns:
            action_probs: (batch_size, action_dim) 행동 확률 분포
            value: (batch_size, 1) 상태 가치
        """
        batch_size, seq_len, _ = x.size()
        
        # xLSTM 순방향 전개
        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c, n = h.clone(), h.clone() + 1.0
        else:
            h, c, n = states
        
        all_h = []
        for t in range(seq_len):
            h, c, n = self.xlstm_cell(x[:, t, :], h, c, n)
            all_h.append(h.unsqueeze(1))
        seq_h = torch.cat(all_h, dim=1)  # (batch, 20, 128)
        
        # TransformerBlock으로 전역 맥락 요약 (FFN 포함으로 비선형성 보강)
        seq_h = self.transformer(seq_h)  # (batch, 20, 128)
        chart_context = seq_h.mean(dim=1)  # (batch, 128)
        
        # VSN 처리
        if info is None:
            info = torch.zeros(batch_size, 13).to(x.device)
        info_context = self.vsn(info)  # (batch, 128)
        
        # 최종 결합
        combined = torch.cat([chart_context, info_context], dim=-1)  # (batch, 256)
        
        return self.actor(combined), self.critic(combined)
