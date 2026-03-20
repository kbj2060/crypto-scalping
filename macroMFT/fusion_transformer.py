import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# Shared Components Library
# (MultiScaleCNN, RoPE, QuantTransformerBackbone, StrategyInteraction, CrossAttn)
# ==============================================================================


class MultiScaleCNN(nn.Module):
    """[Shared] 시계열의 다양한 주기 포착 (Kernel 3, 5, 7)"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        small_dim = hidden_dim // 4
        large_dim = hidden_dim // 2

        self.conv1 = nn.Conv1d(
            input_dim,
            small_dim,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.conv2 = nn.Conv1d(
            input_dim,
            small_dim,
            kernel_size=5,
            padding=2,
            padding_mode="replicate",
        )
        self.conv3 = nn.Conv1d(
            input_dim,
            large_dim,
            kernel_size=7,
            padding=3,
            padding_mode="replicate",
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        out = torch.cat([c1, c2, c3], dim=1)
        out = self.activation(self.bn(out))
        return out.transpose(1, 2)


class RotaryEmbedding(nn.Module):
    """[Shared] RoPE: 상대적 위치 인코딩"""

    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return (x * cos) + (
        torch.cat(
            [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]],
            dim=-1,
        )
        * sin
    )


class QuantTransformerBackbone(nn.Module):
    """[Shared] 핵심 백본: CNN + RoPE/PE + Transformer"""

    def __init__(
        self,
        state_dim=44,
        hidden_dim=256,
        n_layers=2,
        n_heads=4,
        seq_len=60,
        dropout=0.1,
        mode="tactical",
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        # 1. Input Processing
        self.ms_cnn = MultiScaleCNN(state_dim, hidden_dim)

        # 2. Positional Encoding
        if mode == "strategic":  # TD3
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, hidden_dim))
        else:  # PPO
            self.rope = RotaryEmbedding(hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # 최적화 기능 비활성화 (경고 제거)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _generate_decay_mask(self, batch_size, seq_len, device):
        mask = torch.zeros(seq_len + 1, seq_len + 1, device=device)
        for i in range(seq_len + 1):
            for j in range(seq_len + 1):
                if j > i:
                    mask[i, j] = float("-inf")
                else:
                    mask[i, j] = -0.1 * abs(i - j)
        return mask

    def forward(self, x, states=None):
        B, T, _ = x.shape
        x = self.ms_cnn(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.mode == "strategic":
            x = x + self.pos_embedding[:, : T + 1, :]
        else:
            cos, sin = self.rope(x)
            x = apply_rotary_pos_emb(x, cos, sin)

        x = self.dropout(x)

        src_mask = None
        if self.mode == "tactical":
            src_mask = self._generate_decay_mask(B, T, x.device)

        # fusion_transformer.py - QuantTransformerBackbone.forward()
        if self.mode == "ppo":
            # CLS 토큰(0번)은 전체 시퀀스 접근 가능, 시퀀스 토큰은 과거만 접근
            mask = torch.zeros(T + 1, T + 1, device=x.device)
            seq_mask = torch.triu(
                torch.ones(T, T) * float("-inf"),
                diagonal=1,
            ).to(x.device)
            mask[1:, 1:] = seq_mask  # CLS 토큰 제외한 시퀀스에만 캐주얼 마스킹
            src_mask = mask
        else:
            src_mask = None

        x = self.transformer(x, mask=src_mask)
        x = self.layer_norm(x)
        return x[:, 0, :], x, None


class StrategyInteractionLayer(nn.Module):
    """[Shared] Elite 8 전략 융합"""

    def __init__(self, strategy_dim=8, embedding_dim=32):
        super().__init__()
        self.strategy_dim = strategy_dim
        self.proj = nn.Linear(strategy_dim, strategy_dim * embedding_dim)
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(strategy_dim * embedding_dim, 64), nn.GELU(), nn.Dropout(0.1)
        )

    def forward(self, strategies):
        B = strategies.size(0)
        x = self.proj(strategies).view(B, self.strategy_dim, self.embedding_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embedding_dim),
            dim=-1,
        )
        mixed = torch.matmul(attn_weights, V)
        return self.out_proj((x + mixed).view(B, -1))


class CrossAttentionFusion(nn.Module):
    """[Shared] Context Fusion"""

    def __init__(self, hidden_dim=256, query_dim=67):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq_encodings, info_vec):
        query = self.query_proj(info_vec).unsqueeze(1)
        attn_out, _ = self.mha(query, seq_encodings, seq_encodings)
        context = self.norm(query + self.dropout(attn_out)).squeeze(1)
        return context

