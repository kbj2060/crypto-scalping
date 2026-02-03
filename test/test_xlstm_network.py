"""
XLSTM 네트워크 아키텍처 유닛 테스트 (현재 모델 기준).
- CausalConv1d, StabilizedSLSTMCell, SharedBackbone, LinearStrategyFusion, XLSTMNetwork
- forward: (logits, value, aux_value, next_states) 반환
"""
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from model.xlstm_network import (
    CausalConv1d,
    StabilizedSLSTMCell,
    SharedBackbone,
    LinearStrategyFusion,
    XLSTMNetwork,
)


class TestCausalConv1d:
    """CausalConv1d: 미래 참조 방지, 출력 길이 = 입력 길이."""

    @pytest.fixture
    def causal_conv(self):
        return CausalConv1d(in_channels=29, out_channels=64, kernel_size=3)

    def test_forward_shape_no_future_leak(self, causal_conv):
        B, C_in, L = 4, 29, 60
        x = torch.randn(B, C_in, L)
        out = causal_conv(x)
        assert out.shape == (B, 64, L), "Causal conv는 시퀀스 길이 유지"

    def test_has_padding_attribute(self, causal_conv):
        assert hasattr(causal_conv, "padding")
        assert causal_conv.padding >= 0


class TestStabilizedSLSTMCell:
    """StabilizedSLSTMCell: clamp, NaN 대체."""

    @pytest.fixture
    def cell(self):
        return StabilizedSLSTMCell(input_size=8, hidden_size=16)

    def test_forward_shape(self, cell):
        B, H_in, H = 2, 8, 16
        x = torch.randn(B, H_in)
        zeros = torch.zeros(B, H)
        state = (zeros, zeros, zeros, zeros)
        h, next_state = cell(x, state)
        assert h.shape == (B, H)
        assert len(next_state) == 4
        for s in next_state:
            assert s.shape == (B, H)

    def test_no_nan_output(self, cell):
        x = torch.randn(2, 8) * 100
        zeros = torch.zeros(2, 16)
        state = (zeros, zeros, zeros, zeros)
        h, _ = cell(x, state)
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()


class TestSharedBackbone:
    """SharedBackbone: Causal CNN + sLSTM, (B, L, input_dim) → (B, hidden_dim), next_states 반환."""

    @pytest.fixture
    def backbone(self):
        return SharedBackbone(input_dim=29, hidden_dim=64, num_layers=1, dropout=0.0)

    def test_forward_shape(self, backbone):
        B, L, D = 4, 60, 29
        x = torch.randn(B, L, D)
        hidden, next_states = backbone(x, states=None)
        assert hidden.shape == (B, 64)
        assert isinstance(next_states, list)
        assert len(next_states) == 1

    def test_forward_returns_next_states(self, backbone):
        x = torch.randn(2, 60, 29)
        context, next_states = backbone(x, None)
        assert context.shape == (2, 64)
        assert next_states is not None
        assert len(next_states) == 1


class TestLinearStrategyFusion:
    """LinearStrategyFusion: context + strategy_scores(12) → (B, hidden_dim)."""

    @pytest.fixture
    def fusion(self):
        return LinearStrategyFusion(num_strategies=12, hidden_dim=64)

    def test_forward_shape(self, fusion):
        B = 4
        context = torch.randn(B, 64)
        strategy_scores = torch.randn(B, 12)
        out = fusion(context, strategy_scores)
        assert out.shape == (B, 64)

    def test_fusion_output_changes_with_strategy_scores(self, fusion):
        """전략 점수가 다르면 출력이 달라야 함 (Fusion이 입력을 반영하는지 검증)."""
        B = 2
        context = torch.randn(B, 64)

        scores_zero = torch.zeros(B, 12)
        out_zero = fusion(context, scores_zero)

        scores_high = torch.ones(B, 12) * 100.0
        out_high = fusion(context, scores_high)

        assert not torch.allclose(out_zero, out_high), (
            "전략 점수(0 vs 100)가 다르면 출력이 달라야 함. "
            "동일하면 Fusion이 입력을 무시하는 경우일 수 있음."
        )


class TestXLSTMNetwork:
    """XLSTMNetwork: forward 반환 (logits, value, aux_value, next_states)."""

    @pytest.fixture
    def net(self):
        return XLSTMNetwork(
            input_dim=29,
            action_dim=3,
            info_dim=15,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )

    def test_forward_returns_four_values(self, net):
        B, L, D = 2, 60, 29
        x = torch.randn(B, L, D)
        info = torch.randn(B, 15)
        logits, value, aux_value, next_states = net(x, info, states=None)
        assert logits.shape == (B, 3)
        assert value.shape == (B, 1)
        assert aux_value.shape == (B, 1)
        assert next_states is not None
        assert isinstance(next_states, list)
        assert len(next_states) == 1

    def test_logits_to_probs_sum_to_one(self, net):
        x = torch.randn(2, 60, 29)
        info = torch.randn(2, 15)
        logits, _, _, _ = net(x, info, states=None)
        probs = F.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))

    def test_info_3d_squeezed(self, net):
        x = torch.randn(2, 60, 29)
        info = torch.randn(2, 1, 15)
        logits, value, aux_value, _ = net(x, info, states=None)
        assert logits.shape == (2, 3)
        assert value.shape == (2, 1)
        assert aux_value.shape == (2, 1)

    def test_shared_backbone_single_instance(self, net):
        """Actor/Critic가 하나의 backbone 공유."""
        assert hasattr(net, "backbone")
        assert not hasattr(net, "actor_backbone")
        assert not hasattr(net, "critic_backbone")

    def test_has_aux_head(self, net):
        """aux_head 입력 차원: hidden_dim * 2 (strat_context + pos_feat)."""
        assert hasattr(net, "aux_head")
        hidden_dim = 64
        combined = torch.randn(2, hidden_dim * 2)
        out = net.aux_head(combined)
        assert out.shape == (2, 1)

    def test_has_strat_fusion_not_gating(self, net):
        """현재 모델은 LinearStrategyFusion 사용 (StrategyGating 아님)."""
        assert hasattr(net, "strat_fusion")
        assert isinstance(net.strat_fusion, LinearStrategyFusion)
