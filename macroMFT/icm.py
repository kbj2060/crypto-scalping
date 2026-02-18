"""
Intrinsic Curiosity Module for MacroHFT
- 상태 전이 예측 오차를 내재적 보상으로 사용
- 희소한 외부 보상 환경에서 탐색 촉진
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.action_dim = action_dim  # 2 (direction, scale)
        
        # 상태 인코더 (phi)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 역동적 모델: s1, s2 → action 예측 (방향 분류 + scale 회귀)
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.inverse_dir = nn.Linear(hidden_dim, 3)  # 방향 logits
        self.inverse_scale = nn.Linear(hidden_dim, 1)  # scale 평균
        
        # 순방향 모델: s1, a → phi(s2) 예측
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 1, hidden_dim),  # phi(s1) + dir one-hot + scale
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.to(device)
    
    def forward(self, s1, s2, dir_action, scale_action):
        """
        s1, s2: (batch, state_dim) - 연속된 두 상태
        dir_action: (batch,) - 0,1,2
        scale_action: (batch,) - 0~1
        returns:
            forward_loss: 순방향 예측 MSE
            inverse_loss: 역동적 예측 손실 (CE + MSE)
            intrinsic_reward: phi(s2) 예측 오차 (배치별)
        """
        # 상태 인코딩
        phi1 = self.encoder(s1)
        phi2 = self.encoder(s2)
        
        # ----- 역동적 모델: action 예측 -----
        inverse_input = torch.cat([phi1, phi2], dim=-1)
        inverse_feat = self.inverse_net(inverse_input)
        
        dir_pred = self.inverse_dir(inverse_feat)  # (batch, 3)
        scale_pred = self.inverse_scale(inverse_feat).squeeze(-1)  # (batch,)
        
        # 방향 손실 (CrossEntropy)
        dir_loss = F.cross_entropy(dir_pred, dir_action)
        
        # 스케일 손실 (MSE) - scale은 0~1 사이 값
        scale_loss = F.mse_loss(torch.sigmoid(scale_pred), scale_action)
        
        inverse_loss = dir_loss + scale_loss
        
        # ----- 순방향 모델: phi2 예측 -----
        # action one-hot encoding (방향)
        dir_one_hot = F.one_hot(dir_action, num_classes=3).float()  # (batch, 3)
        scale_feat = scale_action.unsqueeze(-1)  # (batch, 1)
        
        forward_input = torch.cat([phi1, dir_one_hot, scale_feat], dim=-1)
        phi2_pred = self.forward_net(forward_input)
        
        forward_loss = F.mse_loss(phi2_pred, phi2.detach())
        
        # 내재적 보상: phi2 예측 오차 (L2 norm)
        intrinsic_reward = torch.norm(phi2_pred - phi2.detach(), dim=-1)
        
        return forward_loss, inverse_loss, intrinsic_reward