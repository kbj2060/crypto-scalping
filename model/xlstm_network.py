import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# Final Genius Architecture (Fixed)
# Features: Temporal Attention, Strategy Interaction, Gated Fusion, Risk-Aware Critic
# ==============================================================================

class ResidualGRU(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        out, h_next = self.gru(x, h)
        out = self.ln(out + x)
        return self.dropout(out), h_next


class SharedBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList([
            ResidualGRU(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Temporal Attention
        self.attention_w = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, x, states=None):
        x = self.input_proj(x)
        if states is None:
            states = [None] * len(self.layers)
        
        next_states = []
        for i, layer in enumerate(self.layers):
            x, h_next = layer(x, states[i])
            next_states.append(h_next)
        
        # Temporal Attention Mechanism
        scores = torch.matmul(x, self.attention_w).squeeze(-1)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        
        return context, next_states


class StrategyInteractionLayer(nn.Module):
    def __init__(self, strategy_dim=12, embedding_dim=32):
        super().__init__()
        self.strategy_dim = strategy_dim
        self.proj = nn.Linear(strategy_dim, strategy_dim * embedding_dim)
        self.embedding_dim = embedding_dim
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Sequential(
            nn.Linear(strategy_dim * embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, strategies):
        B = strategies.size(0)
        x = self.proj(strategies).view(B, self.strategy_dim, self.embedding_dim)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embedding_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        mixed = torch.matmul(attn_weights, V)
        
        out = (x + mixed).view(B, -1)
        return self.out_proj(out)


class XLSTMNetwork(nn.Module):
    EXPECTED_INFO_DIM = 15

    def __init__(self, input_dim, action_dim, info_dim=15, hidden_dim=128, num_layers=2, dropout=0.1):
        super(XLSTMNetwork, self).__init__()
        
        # 1. Backbone
        self.backbone = SharedBackbone(input_dim, hidden_dim, num_layers, dropout)
        
        # 2. Strategy Processor
        self.strategy_processor = StrategyInteractionLayer(strategy_dim=12)
        
        # 3. Gated Fusion Unit (GFU) setup
        self.fusion_dim = hidden_dim + 64 + 3
        
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid()
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic_cvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.aux = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # [Log] Gate Mean 저장용 (forward에서 갱신)
        self.last_gate_mean = 0.5
        
        self.apply(self._init_weights)
        
        # [수정] Hold Bias 0.5 → 0.1: "가만히 있는 게 좋긴 한데..." 수준의 가벼운 힌트만
        with torch.no_grad():
            self.actor[2].bias[0] += 0.1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x, info, states=None, temperature=None):
        context, next_states = self.backbone(x, states)
        
        if info.dim() == 3:
            info = info.squeeze(1)
            
        pos_val = info[:, 0:1]
        strategies = info[:, 1:13]
        pos_meta = info[:, 13:15]
        pos_info = torch.cat([pos_val, pos_meta], dim=1)
        
        strat_features = self.strategy_processor(strategies)
        
        # [Optimization: Gated Fusion]
        combined_input = torch.cat([context, strat_features, pos_info], dim=1)
        
        # Calculate Gate
        gate_values = self.gate(combined_input)
        
        gate_mean = gate_values.mean().item()
        self.last_gate_mean = gate_mean

        gated_input = combined_input * gate_values
        
        fused = self.fusion_proj(gated_input)
        
        logits = self.actor(fused)
        val_mean = self.critic_mean(fused)
        val_cvar = self.critic_cvar(fused)
        aux_val = self.aux(fused)
        
        # Return 6 values including gate_mean
        return logits, val_mean, val_cvar, aux_val, next_states, gate_mean