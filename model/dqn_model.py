"""
Dueling GRU Network + Attention + NoisyNet + Residual Connection
모든 고급 기법이 적용된 최종 아키텍처
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    NoisyNet: 가중치에 노이즈를 섞어 학습 가능한 탐험을 수행하는 레이어
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 학습 파라미터 (Mu)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # 노이즈 파라미터 (Sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 버퍼 (노이즈 값 저장용)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        context = torch.sum(x * weights, dim=1)
        return context

class DuelingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, action_dim=3, info_dim=3, noisy=True):
        super(DuelingGRU, self).__init__()
        
        self.noisy = noisy
        self.info_dim = info_dim
        
        # [신규] 0. 입력 투영 레이어 (Residual Connection을 위해 차원 맞추기)
        # input_dim(예:15) -> hidden_dim(예:128)으로 늘려줌
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # 1. GRU
        self.gru = nn.GRU(
            input_size=input_dim, # GRU는 원본 입력을 받음
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # GRU 출력 안정화를 위한 LayerNorm
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # 2. Attention Layer
        self.attention = AttentionPooling(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # [추가] Info 통합 레이어 (포지션 정보 처리)
        self.info_proj = nn.Sequential(
            nn.Linear(info_dim, hidden_dim // 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 4)
        )
        self.final_dim = hidden_dim + hidden_dim // 4
        
        # 3. Dueling Heads (NoisyLinear 사용)
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
        self.value_stream = nn.Sequential(
            LinearLayer(self.final_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            LinearLayer(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            LinearLayer(self.final_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            LinearLayer(128, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x, info=None):
        """
        Args:
            x: (batch, seq, input_dim) 시계열 피처
            info: (batch, info_dim) 포지션 정보 [pos_val, pnl_val, hold_val]
        Returns:
            q_values: (batch, action_dim) Q값
        """
        # 1. 입력 투영 (Residual 용)
        # x: (batch, seq, input) -> x_proj: (batch, seq, hidden)
        x_proj = self.input_proj(x)
        
        # 2. GRU 통과
        gru_out, _ = self.gru(x)
        
        # [핵심] 3. 잔차 연결 (Residual Connection)
        # 정보의 고속도로: 원본 정보(투영됨)를 GRU 출력에 더해줌
        gru_out = gru_out + x_proj
        
        # 4. 정규화
        gru_out = self.ln1(gru_out)
        
        # 5. 어텐션 & 헤드
        context_vector = self.attention(gru_out)
        context_vector = self.ln2(context_vector)
        
        # [추가] 6. Info 통합 (포지션 정보)
        if info is not None:
            # info: (batch, info_dim) -> (batch, hidden_dim // 4)
            info_proj = self.info_proj(info)
            context_vector = torch.cat([context_vector, info_proj], dim=-1)
        else:
            # Info가 없으면 0으로 채움 (하위 호환성)
            batch_size = context_vector.size(0)
            info_proj = torch.zeros(batch_size, self.final_dim - hidden_dim, 
                                   device=context_vector.device, dtype=context_vector.dtype)
            context_vector = torch.cat([context_vector, info_proj], dim=-1)
        
        value = self.value_stream(context_vector)
        advantage = self.advantage_stream(context_vector)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        if self.noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
