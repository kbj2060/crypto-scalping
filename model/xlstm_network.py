import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class XLSTMNetwork(nn.Module):
    # [수정] info_dim 인자 추가 (기본값 15)
    def __init__(self, input_dim, action_dim, info_dim=15, hidden_dim=128, num_layers=1, num_heads=4):
        super(XLSTMNetwork, self).__init__()
        
        # 1. 입력 안전장치 (그대로 유지)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 2. Feature Extractor (CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 3. LSTM (Context Memory)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # [복구 1] Multi-Head Attention (과거 패턴 스캔)
        # 차트의 흐름을 놓치지 않도록 "눈"을 달아줍니다.
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.att_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Info Encoder (전략 신호 처리)
        # [핵심 수정] 전달받은 info_dim으로 레이어 생성
        self.info_net = nn.Sequential(
            nn.Linear(info_dim, 64),  # 여기가 동적으로 변해야 합니다!
            nn.LayerNorm(64), # 안정성 강화
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        )
        
        # [복구 2] Shared Trunk (정보 융합소)
        # 차트(128) + 전략(64) = 192 정보를 섞어서 고차원 판단
        combined_dim = hidden_dim + 64
        self.shared_trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU()
        )
        
        # 5. Output Heads (Actor & Critic)
        # Trunk를 통과한 정제된 정보(128)를 사용
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # [필수] 뇌 정지 방지용 초기화 유지
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x, info, states=None):
        # 1. Normalization
        x = self.input_norm(x)
        
        # 2. CNN Feature Extraction
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        
        # 3. LSTM Forward
        self.lstm.flatten_parameters()
        x, new_states = self.lstm(x, states)
        
        # [복구 1 적용] Attention Mechanism
        # LSTM 출력 전체(seq_len)를 봅니다.
        # Query, Key, Value 모두 LSTM 출력 사용 (Self-Attention)
        attn_output, _ = self.attention(x, x, x)
        
        # Residual Connection & Norm (학습 안정성)
        x = self.att_norm(x + attn_output)
        
        # Weighted Pooling (시퀀스 압축: 평균이나 마지막 값보다 똑똑하게 압축)
        # 여기서는 간단히 Global Average Pooling 사용 (전체 맥락 반영)
        context = torch.mean(x, dim=1) 
        
        # 4. Process Info
        # info_dim이 이미 초기화 시점에 맞춰져 있으므로 동적 재생성 불필요
        info_out = self.info_net(info)
        
        # 5. Combine & Shared Trunk
        combined = torch.cat([context, info_out], dim=1)
        
        # [복구 2 적용] 정보 융합
        features = self.shared_trunk(combined)
        
        # 6. Outputs
        probs = self.actor(features)
        value = self.critic(features)
        
        return probs, value, new_states