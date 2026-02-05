import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==============================================================================
# Final Genius Architecture (Updated for Elite 8)
# Features: QuantTransformerBackbone (Strategic/Tactical), CrossAttentionFusion
# ==============================================================================

class QuantTransformerBackbone(nn.Module):
    def __init__(self, state_dim=44, hidden_dim=256, n_layers=2, n_heads=4, seq_len=60, dropout=0.1, mode='strategic'):
        """
        Args:
            state_dim: 입력 피처 차원 (Ultimate Feature Set = 44)
            mode (str): 'strategic' (TD3용) or 'tactical' (MacroHFT용)
        """
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 1. Input Projection
        self.embedding = nn.Linear(state_dim, hidden_dim)
        
        # 2. Positional Encoding Logic
        if mode == 'strategic':
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, hidden_dim))
        else:
            self.register_buffer('pos_embedding', self._create_sinusoidal_pe(seq_len + 1, hidden_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _create_sinusoidal_pe(self, len_seq, d_model):
        pe = torch.zeros(len_seq, d_model)
        position = torch.arange(0, len_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _generate_decay_mask(self, batch_size, seq_len, device):
        """MacroHFT용: 과거 데이터일수록 Attention을 덜 받도록 강제"""
        mask = torch.zeros(seq_len + 1, seq_len + 1, device=device)
        for i in range(seq_len + 1):
            for j in range(seq_len + 1):
                if j > i: # Future masking
                    mask[i, j] = float('-inf')
                else:
                    distance = abs(i - j)
                    mask[i, j] = -0.1 * distance 
        return mask

    def forward(self, x, states=None):
        B, T, _ = x.shape
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embedding[:, :T+1, :]
        x = self.dropout(x)
        
        src_mask = None
        if self.mode == 'tactical':
            src_mask = self._generate_decay_mask(B, T, x.device)

        x = self.transformer(x, mask=src_mask)
        x = self.layer_norm(x)
        
        return x[:, 0, :], x, None


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=256, query_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq_encodings, info_vec):
        query = self.query_proj(info_vec).unsqueeze(1)
        attn_out, _ = self.mha(query, seq_encodings, seq_encodings)
        context = self.norm(query + self.dropout(attn_out)).squeeze(1)
        return context


class StrategyInteractionLayer(nn.Module):
    """Elite 8 전략(8개)을 Self-Attention으로 섞어줌"""
    def __init__(self, strategy_dim=8, embedding_dim=32):
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
    EXPECTED_INFO_DIM = 11  # Elite 8: 1(Val) + 8(Strat) + 2(Meta)

    def __init__(self, input_dim, action_dim, info_dim=11, hidden_dim=256, num_layers=2, dropout=0.1):
        super(XLSTMNetwork, self).__init__()
        
        # 1. Backbone
        self.backbone = QuantTransformerBackbone(
            state_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_layers=num_layers,
            n_heads=4, 
            seq_len=60,
            dropout=dropout,
            mode='tactical'
        )
        
        # 2. Strategy Processor (Elite 8)
        self.strategy_processor = StrategyInteractionLayer(strategy_dim=8)
        
        # 3. Cross Attention Fusion
        # Query = Strategy(64) + PosInfo(3) = 67
        self.fusion_attention = CrossAttentionFusion(hidden_dim=hidden_dim, query_dim=64+3)
        
        # 4. Gated Output
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
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
        
        self.last_gate_mean = 0.5
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, info, states=None, temperature=None):
        # 1. Backbone
        context, seq_encodings, next_states = self.backbone(x, states)
        
        if info.dim() == 3:
            info = info.squeeze(1)
            
        # 2. Info Processing (Elite 8 Info Layout)
        # info structure: [pos_val(1), strategies(8), pos_meta(2)]
        pos_val = info[:, 0:1]
        strategies = info[:, 1:9]
        pos_meta = info[:, 9:11]
        
        pos_info = torch.cat([pos_val, pos_meta], dim=1)  # (B, 3)
        
        strat_features = self.strategy_processor(strategies) # (B, 64)
        
        # Query Vector
        query_vec = torch.cat([strat_features, pos_info], dim=1) # (B, 67)
        
        # 3. Fusion & Gate
        fused_context = self.fusion_attention(seq_encodings, query_vec)
        gate_values = self.gate(fused_context)
        gate_mean = gate_values.mean().item()
        self.last_gate_mean = gate_mean
        
        final_repr = fused_context * gate_values
        
        # 5. Output Heads
        logits = self.actor(final_repr)
        val_mean = self.critic_mean(final_repr)
        val_cvar = self.critic_cvar(final_repr)
        aux_val = self.aux(final_repr)
        
        return logits, val_mean, val_cvar, aux_val, next_states, gate_mean