"""
TD3 Actor/Critic with QuantTransformerBackbone (Strategic Mode)
- Adapted for Elite 8 Strategies & Ultimate Features (44 dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import config
# xlstm_network 파일에서 필요한 모듈 재사용
from macroHFT.xlstm_network import QuantTransformerBackbone, StrategyInteractionLayer, CrossAttentionFusion


def _get_network_params():
    hidden_dim = getattr(config, 'NETWORK_HIDDEN_DIM', 256)
    num_layers = getattr(config, 'TD_NETWORK_NUM_LAYERS', 3) # TD3는 더 깊게 (3층 추천)
    dropout = getattr(config, 'NETWORK_DROPOUT', 0.1)
    return hidden_dim, num_layers, dropout


class RiskAwareGate(nn.Module):
    """
    [Stricter] 손실 구간에서 페널티 적용
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, pos_context, volatility):
        # pos_context(3) + volatility(1) = 4
        gate_input = torch.cat([pos_context, volatility], dim=1)
        base_gate = self.net(gate_input)
        pnl = pos_context[:, 1:2]
        
        # 큰 손실(-2% 이상) 발생 시 Gate를 닫아버림 (보수적 대응)
        loss_penalty = torch.where(
            pnl < -0.02,
            0.5 * torch.exp(pnl * 10),
            torch.ones_like(pnl)
        )
        return base_gate * loss_penalty


class PositionAwareActor(nn.Module):
    def __init__(self, input_dim, action_dim=1, info_dim=12):
        super(PositionAwareActor, self).__init__()
        hd, nl, do = _get_network_params()

        # 1. Backbone: 'strategic' mode (TD3 최적화)
        # input_dim should be 44 (Ultimate Features)
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hd, 
            n_layers=nl, 
            dropout=do, 
            mode='strategic' 
        )
        
        # Elite 8 Strategies
        self.strategy_processor = StrategyInteractionLayer(strategy_dim=8)
        self.position_gate = RiskAwareGate()

        # 2. Fusion: Cross Attention
        # Query Dim = Strategy(64) + PosContext(3) + Volatility(1) = 68
        self.fusion_attention = CrossAttentionFusion(hidden_dim=hd, query_dim=64+3+1)

        self.head = nn.Sequential(
            nn.Linear(hd, hd),
            nn.GELU(),
            nn.Linear(hd, action_dim),
            nn.Tanh()
        )

    def forward(self, x, info, states=None):
        # seq_encodings 필요
        _, seq_encodings, next_states = self.backbone(x, states)

        if info.dim() == 3:
            info = info.squeeze(1)

        # Info Layout (Elite 8 + Volatility): 
        # [pos_val(1), strategies(8), pos_meta(2), volatility(1)] = Total 12
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]

        pos_context = torch.cat([pos_val, pos_meta], dim=1) # (B, 3)
        
        # Risk Gate
        gate = self.position_gate(pos_context, volatility)

        strat_features = self.strategy_processor(strategies) # (B, 64)
        
        # Construct Query for Cross Attention
        query_vec = torch.cat([strat_features, pos_context, volatility], dim=1) # Size: 68
        
        # Fusing
        fused_repr = self.fusion_attention(seq_encodings, query_vec)
        
        raw_action = self.head(fused_repr)

        # Action Scaling based on Risk Gate
        magnitude = torch.abs(raw_action)
        direction = torch.sign(raw_action)
        scaled_magnitude = magnitude * (0.1 + 0.9 * gate)
        scaled_action = direction * scaled_magnitude

        return scaled_action, next_states, gate.mean()


class TD3Critic(nn.Module):
    def __init__(self, input_dim, action_dim=1, info_dim=12):
        super(TD3Critic, self).__init__()
        hd, nl, do = _get_network_params()

        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hd, 
            n_layers=nl, 
            dropout=do, 
            mode='strategic'
        )
        
        self.strategy_processor = StrategyInteractionLayer(strategy_dim=8)
        
        # Fusion Query Dim = 68
        self.fusion_attention = CrossAttentionFusion(hidden_dim=hd, query_dim=68)

        # Q-Networks
        self.q1_net = nn.Sequential(
            nn.Linear(hd + action_dim, hd),
            nn.LayerNorm(hd),
            nn.GELU(),
            nn.Linear(hd, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(hd + action_dim, hd),
            nn.LayerNorm(hd),
            nn.GELU(),
            nn.Linear(hd, 1)
        )

    def forward(self, x, info, action, states=None):
        _, seq_encodings, _ = self.backbone(x, states)

        if info.dim() == 3:
            info = info.squeeze(1)

        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        volatility = info[:, 11:12]

        strat_features = self.strategy_processor(strategies)
        pos_context = torch.cat([pos_val, pos_meta], dim=1)
        
        query_vec = torch.cat([strat_features, pos_context, volatility], dim=1)
        
        state_repr = self.fusion_attention(seq_encodings, query_vec)

        q1_input = torch.cat([state_repr, action], dim=1)
        q1 = self.q1_net(q1_input)
        
        q2_input = torch.cat([state_repr, action], dim=1)
        q2 = self.q2_net(q2_input)

        q1 = torch.clamp(q1, min=-1.0, max=1.0)
        q2 = torch.clamp(q2, min=-1.0, max=1.0)
        
        return q1, q2