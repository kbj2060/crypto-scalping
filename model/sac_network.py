"""
SAC (Soft Actor-Critic) Network Architecture
xLSTM 기반 Feature Extractor를 재사용하여 Gaussian Policy Actor와 Twin Critic 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .xlstm_network import sLSTMCell, MultiHeadAttention

logger = __import__('logging').getLogger(__name__)


class SACActor(nn.Module):
    """
    Gaussian Policy Actor for SAC
    xLSTM으로 시계열 특징 추출 후, Mean(μ)과 LogStd(σ)를 출력
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128, 
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 1. Feature Extractor (xLSTM - 기존 로직 재사용)
        self.xlstm_cell = sLSTMCell(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads=4)
        
        # Late Fusion
        combined_dim = hidden_dim + info_dim
        
        # 2. Shared Backbone
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )
        
        # 3. Heads (Mean & LogStd)
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, x, info=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 시계열 입력
            info: (batch_size, info_dim) 포지션 정보 (None이면 0으로 채움)
        Returns:
            mu: (batch_size, action_dim) 행동의 평균
            log_std: (batch_size, action_dim) 행동의 로그 표준편차
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # xLSTM Processing
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        n = torch.ones(batch_size, self.hidden_dim).to(device)

        all_h = []
        for t in range(seq_len):
            h, c, n = self.xlstm_cell(x[:, t, :], h, c, n, None)
            all_h.append(h.unsqueeze(1))
            
        seq_h = torch.cat(all_h, dim=1)
        context = self.attention(seq_h)  # (batch, hidden)
        
        # Late Fusion
        if info is None:
            info = torch.zeros(batch_size, 13).to(device)  # info_dim=13 가정
            
        combined = torch.cat([context, info], dim=-1)
        x_emb = self.backbone(combined)
        
        # Output
        mu = self.mu_head(x_emb)
        log_std = self.log_std_head(x_emb)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std

    def sample(self, x, info=None):
        """
        Reparameterization Trick을 사용한 행동 샘플링
        
        Returns:
            action: (batch_size, action_dim) Tanh로 압축된 행동 [-1, 1]
            log_prob: (batch_size, 1) 행동의 로그 확률
            mean_action: (batch_size, action_dim) 평균 행동 (평가용)
        """
        mu, log_std = self.forward(x, info)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # Reparameterization Trick (Backprop 가능하게 샘플링)
        z = dist.rsample()
        action = torch.tanh(z)  # -1 ~ 1 사이로 압축
        
        # Log Probability 계산 (Tanh 보정 포함)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)


class SACCritic(nn.Module):
    """
    Twin Q-Network for SAC
    입력: State + Action -> 출력: Q-Value (2개)
    """
    def __init__(self, input_dim, action_dim, info_dim=13, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature Extractor (Actor와 동일한 구조지만 별도 학습)
        self.xlstm_cell = sLSTMCell(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads=4)
        
        combined_dim = hidden_dim + info_dim + action_dim  # Action도 입력으로 들어감
        
        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Q2 Architecture (Twin)
        self.q2 = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, action, info=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) 시계열 입력
            action: (batch_size, action_dim) 행동
            info: (batch_size, info_dim) 포지션 정보
        Returns:
            q1: (batch_size, 1) 첫 번째 Q값
            q2: (batch_size, 1) 두 번째 Q값
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # xLSTM Processing
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        n = torch.ones(batch_size, self.hidden_dim).to(device)

        all_h = []
        for t in range(seq_len):
            h, c, n = self.xlstm_cell(x[:, t, :], h, c, n, None)
            all_h.append(h.unsqueeze(1))
            
        seq_h = torch.cat(all_h, dim=1)
        context = self.attention(seq_h)
        
        if info is None:
            info = torch.zeros(batch_size, 13).to(device)

        # State + Action 결합
        cat_input = torch.cat([context, info, action], dim=-1)
        
        return self.q1(cat_input), self.q2(cat_input)
