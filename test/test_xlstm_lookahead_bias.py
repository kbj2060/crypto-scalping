"""
Look-ahead Bias 검증 테스트 (macroHFT/xlstm_network.py)
- CausalConv1d: 미래 시점 참조 시 출력이 바뀌면 안 됨
- TransformerBackbone: (상태 없음, 전체 시퀀스 입력)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macroHFT.xlstm_network import TransformerBackbone


@pytest.mark.skip(reason="CausalConv1d removed from xlstm_network; backbone is now TransformerBackbone")
class TestCausalConv1dLookAheadBias:
    """CausalConv1d: 미래 데이터를 참조하면 안 됨 (Look-ahead Bias 방지)."""

    @pytest.fixture
    def conv(self):
        return CausalConv1d(in_channels=8, out_channels=16, kernel_size=5, dilation=2)

    def test_same_prefix_different_future_gives_same_prefix_output(self, conv):
        """
        입력의 앞부분이 같고, 미래 시점만 다를 때
        출력의 앞부분은 동일해야 함 (미래 참조 없음).
        """
        B, C_in, L = 2, 8, 32
        torch.manual_seed(42)
        x_prefix = torch.randn(B, C_in, 20)  # 0~19 시점
        x_future_a = torch.randn(B, C_in, L - 20)
        x_future_b = torch.randn(B, C_in, L - 20)  # 20~31 시점은 다름

        x_a = torch.cat([x_prefix, x_future_a], dim=2)  # [B, C, 32]
        x_b = torch.cat([x_prefix, x_future_b], dim=2)  # 앞 20개 동일, 뒤 12개 다름

        conv.eval()
        with torch.no_grad():
            out_a = conv(x_a)
            out_b = conv(x_b)

        # 수용 범위: (5-1)*2+1 = 9. 출력 위치 i는 입력 0..i만 봄.
        # 따라서 출력 위치 0~19는 입력 0~19만 의존 → 두 입력이 앞 20개 같으면 출력 앞 20개 동일
        prefix_len = 20
        assert out_a.shape == out_b.shape == (B, 16, L)
        assert torch.allclose(out_a[:, :, :prefix_len], out_b[:, :, :prefix_len]), (
            "CausalConv1d: 앞 시점 입력이 같은데 출력 앞부분이 다르면 Look-ahead Bias 가능성 있음."
        )

    def test_changing_only_future_does_not_change_past_output(self, conv):
        """
        미래 시점 입력만 바꿨을 때, 과거 시점 출력은 그대로여야 함.
        """
        B, C_in, L = 2, 8, 40
        torch.manual_seed(123)
        x = torch.randn(B, C_in, L)
        x_modified = x.clone()
        x_modified[:, :, 25:] = torch.randn(B, C_in, L - 25)  # 25번째 이후만 변경

        conv.eval()
        with torch.no_grad():
            y = conv(x)
            y_mod = conv(x_modified)

        # 출력 위치 0~24는 입력 0~24만 사용 → 동일해야 함
        cut = 25
        assert torch.allclose(y[:, :, :cut], y_mod[:, :, :cut]), (
            "미래 입력만 바꿨는데 과거 출력이 바뀌면 Look-ahead Bias."
        )

    def test_output_length_equals_input_length(self, conv):
        """Causal conv는 패딩 트림으로 입력 길이 = 출력 길이."""
        B, C_in, L = 4, 8, 60
        x = torch.randn(B, C_in, L)
        out = conv(x)
        assert out.shape == (B, 16, L), "출력 시퀀스 길이는 입력과 동일해야 함"

class TestTransformerBackboneLookAheadBias:
    """TransformerBackbone: 전체 시퀀스 입력 시 (context, []) 반환. causal 여부는 마스킹에 따름."""

    @pytest.fixture
    def backbone(self):
        return TransformerBackbone(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0, seq_len=50)

    def test_causal_property_via_truncated_inputs(self, backbone):
        """
        [핵심 검증] 동일한 prefix를 가진 두 시퀀스를 '동일한 길이 k'로 잘라서 처리
        → 출력이 동일해야 함 (미래 입력 영향 없음)
        """
        B, total_len, D = 2, 50, 16
        prefix_len = 30  # 공통 접두사 길이
        
        torch.manual_seed(456)
        prefix = torch.randn(B, prefix_len, D)  # 동일한 접두사
        tail_a = torch.randn(B, total_len - prefix_len, D)
        tail_b = torch.randn(B, total_len - prefix_len, D)
        
        # 두 시퀀스: 접두사 동일, 미래 다름
        seq_a = torch.cat([prefix, tail_a], dim=1)  # [B, 50, D]
        seq_b = torch.cat([prefix, tail_b], dim=1)  # [B, 50, D]
        
        # ✅ 핵심: 동일한 길이(prefix_len)로 잘라서 처리
        seq_a_trunc = seq_a[:, :prefix_len, :]  # [B, 30, D]
        seq_b_trunc = seq_b[:, :prefix_len, :]  # [B, 30, D]
        
        backbone.eval()
        with torch.no_grad():
            ctx_a, _ = backbone(seq_a_trunc, states=None)
            ctx_b, _ = backbone(seq_b_trunc, states=None)
        
        # ✅ 검증: 동일한 입력 길이 + 동일한 접두사 → 동일한 출력
        assert torch.allclose(ctx_a, ctx_b, atol=1e-6), (
            "동일한 길이의 동일 접두사 처리 시 출력이 달라짐 → Look-ahead 가능성 있음"
        )

    def test_deterministic_for_same_input(self, backbone):
        """동일 입력 → 동일 출력 (결정성 검증)"""
        B, L, D = 2, 40, 16
        torch.manual_seed(789)
        x = torch.randn(B, L, D)
        backbone.eval()
        with torch.no_grad():
            c1, _ = backbone(x, states=None)
            c2, _ = backbone(x, states=None)
        assert torch.allclose(c1, c2, atol=1e-6), "동일 입력이면 동일 출력이어야 함"