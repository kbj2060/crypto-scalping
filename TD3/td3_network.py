"""
TD3 네트워크 - Strategic Mode
- StrategicActor: 연속 행동 출력
- StrategicCritic: Twin Q-Network
"""
import torch
import torch.nn as nn
import numpy as np
from core import config
from common.fusion_transformer import QuantTransformerBackbone, StrategyInteractionLayer, CrossAttentionFusion

# ==============================================================================
# Strategic Networks (TD3 Specific)
# - Mode: 'strategic' (No Mask, Learnable PE)
# - Actor: Continuous Action (Tanh)
# - Critic: State + Action Input
# ==============================================================================

class StrategicActor(nn.Module):
    """TD3용 Actor: Strategic Mode (Learnable PE + No Mask)"""
    def __init__(self, input_dim, action_dim=1, info_dim=12, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        
        # [Fix] Config에서 LOOKBACK 가져오기
        seq_len = getattr(config, 'LOOKBACK', 60)

        # 1. Backbone (Strategic Mode)
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_layers=num_layers, 
            seq_len=seq_len,  # [Fix] 시퀀스 길이 전달
            dropout=dropout, 
            mode='strategic'
        )
        
        # 2. Components
        self.strategy_processor = StrategyInteractionLayer()
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=68) # TD3: Strat(64) + Pos(3) + Vol(1)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # 3. Head (Continuous Action)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh() # Tanh 필수
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, info):
        """
        Args:
            x: [B, T, F] state sequence
            info: [B, 12] - TD3 info (pos+strategies+meta+volatility)
        Returns:
            action: [B, 1] continuous action
            states: None
            risk_gate: scalar
        """
        context, seq_encodings, _ = self.backbone(x)
        if info.dim() == 3: info = info.squeeze(1)
        
        # TD3 Info: [pos_val(1), strategies(8), pos_meta(2), volatility(1)] = 12
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]
        
        pos_info = torch.cat([pos_val, pos_meta, volatility], dim=1) # (B, 4)
        strat_features = self.strategy_processor(strategies)
        query_vec = torch.cat([strat_features, pos_info], dim=1) # (B, 68)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        action = self.actor(fused * gate)
        
        return action, None, gate.mean().item()

class StrategicCritic(nn.Module):
    """TD3용 Twin Critic (Q1, Q2)"""
    def __init__(self, input_dim, action_dim=1, info_dim=12, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        
        # [Fix] Config에서 LOOKBACK 가져오기
        seq_len = getattr(config, 'LOOKBACK', 60)

        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_layers=num_layers,
            seq_len=seq_len,  # [Fix] 시퀀스 길이 전달 
            dropout=dropout, 
            mode='strategic'
        )
        self.strategy_processor = StrategyInteractionLayer()
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=68)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # Twin Q-Networks (State + Action)
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, info, action):
        """
        Args:
            x: [B, T, F] state sequence
            info: [B, 12] TD3 info
            action: [B, 1] action
        Returns:
            (q1, q2): Twin Q-values
        """
        context, seq_encodings, _ = self.backbone(x)
        if info.dim() == 3: info = info.squeeze(1)
        
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]
        
        pos_info = torch.cat([pos_val, pos_meta, volatility], dim=1)
        strat_features = self.strategy_processor(strategies)
        query_vec = torch.cat([strat_features, pos_info], dim=1)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        state_repr = fused * gate
        
        # Concatenate with Action
        q_input = torch.cat([state_repr, action], dim=1)
        return self.q1_net(q_input), self.q2_net(q_input)

# Backward Compatibility Aliases
PositionAwareActor = StrategicActor
TD3Critic = StrategicCritic