import torch
import torch.nn as nn
import math

# [2026 SOTA] SwiGLU Activation (LLaMA, PaLM 등 최신 LLM의 표준)
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

# [2026 SOTA] Transformer Block with Pre-Norm & SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network (FFN) -> SwiGLU로 업그레이드
        self.ffn = nn.Sequential(
            SwiGLU(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-Norm Architecture (학습 안정성 ↑)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

# [Core] Trading Transformer
class MacroHFTNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, info_dim, d_model=128, n_head=4, n_layers=2):
        super().__init__()
        
        # 1. Embedding Layer (State -> Vector)
        self.embedding = nn.Linear(state_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model)) # Max Lookback 500
        
        # 2. Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head) for _ in range(n_layers)
        ])
        
        # 3. Context Integration (Info Vector Fusion)
        self.info_fusion = nn.Linear(d_model + info_dim, d_model)
        
        # 4. Heads (Actor & Critic)
        self.actor_head = nn.Sequential(
            SwiGLU(d_model),
            nn.Linear(d_model, action_dim)
        )
        self.critic_head = nn.Sequential(
            SwiGLU(d_model),
            nn.Linear(d_model, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, state_seq, state_info, states=None):
        # state_seq: [Batch, Seq_Len, Dim]
        B, T, D = state_seq.shape
        
        # 1. Embedding + Positional Encoding
        x = self.embedding(state_seq) # [B, T, d_model]
        x = x + self.pos_encoder[:, :T, :]
        
        # 2. Transformer Pass
        for layer in self.layers:
            x = layer(x)
            
        # 3. Last Token Pooling (GPT style: predict next move based on context)
        context_vector = x[:, -1, :] # [B, d_model]
        
        # 4. Fusion with Info (Account status, etc)
        combined = torch.cat([context_vector, state_info], dim=1)
        latent = self.info_fusion(combined)
        
        # 5. Outputs
        logits = self.actor_head(latent)
        value = self.critic_head(latent)
        
        return logits, value, None, None, None, None

# [Expert Definition] MoE를 위한 3가지 특화 트랜스포머
class TrendExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim):
        # Trend: 깊은 레이어 (복잡한 패턴 인식)
        super().__init__(state_dim, action_dim, info_dim, d_model=256, n_head=8, n_layers=4)

class VolatilityExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim):
        # Volatility: 빠른 반응 (얕은 레이어)
        super().__init__(state_dim, action_dim, info_dim, d_model=128, n_head=4, n_layers=2)

class SidewaysExpert(MacroHFTNetwork):
    def __init__(self, state_dim, action_dim, info_dim):
        # Sideways: 통계적 판단 (중간 레이어)
        super().__init__(state_dim, action_dim, info_dim, d_model=192, n_head=6, n_layers=3)