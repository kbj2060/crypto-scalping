"""
XLSTM 네트워크 아키텍처 유닛 테스트 (현재 모델 기준).
- TransformerBackbone, XLSTMNetwork
- forward: (logits, value, aux_value, next_states, gate_mean) 등 반환
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from macroHFT.xlstm_network import TransformerBackbone, StrategyInteractionLayer, XLSTMNetwork


class TestTransformerBackbone:
    """TransformerBackbone: (B, L, input_dim) → (B, hidden_dim), next_states=[] 반환."""

    @pytest.fixture
    def backbone(self):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        return TransformerBackbone(input_dim=len(ULTIMATE_FEATURE_COLS), hidden_dim=64, num_layers=1, dropout=0.0, seq_len=60)

    def test_forward_shape(self, backbone):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        B, L, D = 4, 60, len(ULTIMATE_FEATURE_COLS)
        x = torch.randn(B, L, D)
        hidden, next_states = backbone(x, states=None)
        assert hidden.shape == (B, 64)
        assert isinstance(next_states, list)
        assert len(next_states) == 0

    def test_forward_returns_next_states(self, backbone):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        x = torch.randn(2, 60, len(ULTIMATE_FEATURE_COLS))
        context, next_states = backbone(x, None)
        assert context.shape == (2, 64)
        assert next_states is not None
        assert len(next_states) == 0


class TestStrategyInteractionLayer:
    """StrategyInteractionLayer: strategy_scores(8) → (B, 64). Elite 8."""

    @pytest.fixture
    def layer(self):
        return StrategyInteractionLayer(strategy_dim=8, embedding_dim=32)

    def test_forward_shape(self, layer):
        B = 4
        strategy_scores = torch.randn(B, 8)
        out = layer(strategy_scores)
        assert out.shape == (B, 64)

    def test_output_changes_with_strategy_scores(self, layer):
        """전략 점수가 다르면 출력이 달라야 함."""
        B = 2
        scores_zero = torch.zeros(B, 8)
        out_zero = layer(scores_zero)
        scores_high = torch.ones(B, 8) * 100.0
        out_high = layer(scores_high)
        assert not torch.allclose(out_zero, out_high)


class TestXLSTMNetwork:
    """XLSTMNetwork: forward 반환 (logits, val_mean, val_cvar, aux_val, next_states, gate_mean)."""

    @pytest.fixture
    def net(self):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        return XLSTMNetwork(
            input_dim=len(ULTIMATE_FEATURE_COLS),
            action_dim=3,
            info_dim=11,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )

    def test_forward_returns_six_values(self, net):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        B, L, D = 2, 60, len(ULTIMATE_FEATURE_COLS)
        x = torch.randn(B, L, D)
        info = torch.randn(B, 11)
        logits, val_mean, val_cvar, aux_val, next_states, gate_mean = net(x, info, states=None)
        assert logits.shape == (B, 3)
        assert val_mean.shape == (B, 1)
        assert val_cvar.shape == (B, 1)
        assert aux_val.shape == (B, 1)
        assert isinstance(next_states, list)
        assert len(next_states) == 0
        assert isinstance(gate_mean, (int, float))

    def test_logits_to_probs_sum_to_one(self, net):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        x = torch.randn(2, 60, len(ULTIMATE_FEATURE_COLS))
        info = torch.randn(2, 11)
        logits, *_ = net(x, info, states=None)
        probs = F.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))

    def test_info_3d_squeezed(self, net):
        from common.feature_engineering import ULTIMATE_FEATURE_COLS
        x = torch.randn(2, 60, len(ULTIMATE_FEATURE_COLS))
        info = torch.randn(2, 1, 11)
        logits, val_mean, val_cvar, aux_val, *_ = net(x, info, states=None)
        assert logits.shape == (2, 3)
        assert val_mean.shape == (2, 1)
        assert aux_val.shape == (2, 1)

    def test_backbone_and_strategy_processor(self, net):
        assert hasattr(net, "backbone")
        assert isinstance(net.backbone, TransformerBackbone)
        assert hasattr(net, "strategy_processor")
        assert isinstance(net.strategy_processor, StrategyInteractionLayer)
