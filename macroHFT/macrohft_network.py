import torch
import torch.nn as nn
import numpy as np
from common.fusion_transformer import QuantTransformerBackbone, StrategyInteractionLayer, CrossAttentionFusion

# ==============================================================================
# MacroHFT Network (PPO Specific)
# - Mode: 'tactical' (Causal Masking, RoPE)
# - Head: Discrete Action (Logits), Value, CVaR, Aux
# ==============================================================================

class MacroHFTNetwork(nn.Module):
    """PPO용 네트워크: Tactical Mode (RoPE + Time Decay Mask)"""
    EXPECTED_INFO_DIM = 11
    
    def __init__(self, input_dim, action_dim, info_dim=11, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. Backbone (Tactical Mode)
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_layers=num_layers, 
            dropout=dropout, 
            mode='tactical'
        )
        
        # 2. Components
        self.strategy_processor = StrategyInteractionLayer()
        self.fusion = CrossAttentionFusion(hidden_dim, query_dim=67) # Strat(64) + Pos(3)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        
        # 3. Heads (Discrete Action)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic_mean = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.critic_cvar = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))
        self.aux = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2, 1))
        
        self.last_gate_mean = 0.5
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, info, states=None, temperature=None):
        context, seq_encodings, next_states = self.backbone(x, states)
        if info.dim() == 3: info = info.squeeze(1)
        
        pos_val = info[:, 0:1]; strategies = info[:, 1:9]; pos_meta = info[:, 9:11]
        pos_info = torch.cat([pos_val, pos_meta], dim=1)
        strat_features = self.strategy_processor(strategies)
        query_vec = torch.cat([strat_features, pos_info], dim=1)
        
        fused = self.fusion(seq_encodings, query_vec)
        gate = self.gate(fused)
        self.last_gate_mean = gate.mean().item()
        final_repr = fused * gate
        
        return self.actor(final_repr), self.critic_mean(final_repr), self.critic_cvar(final_repr), self.aux(final_repr), next_states, self.last_gate_mean

# 기존 코드와의 호환성을 위한 Alias
XLSTMNetwork = MacroHFTNetwork
