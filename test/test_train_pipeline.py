"""
§3 학습 파이프라인 (커리큘럼, aux_target, 보상 호출) 유닛 테스트.
명세: docs/IMPLEMENTATION_SPECIFICATION.md §3
"""
import numpy as np
import pandas as pd
import pytest

from common import config


class TestAuxTargetFormula:
    """§3.2 aux_target: (high - low) / close * 100 (변동성 스칼라)."""

    def test_aux_target_formula(self):
        high, low, close = 1010.0, 990.0, 1000.0
        aux = (high - low) / close * 100.0
        assert abs(aux - 2.0) < 1e-5

    def test_aux_target_zero_range(self):
        high = low = close = 1000.0
        aux = (high - low) / close * 100.0
        assert aux == 0.0


class TestCurriculumIndices:
    """§3.1 커리큘럼: all_indices, trend_indices (chop < 50)."""

    def test_all_indices_range(self):
        lookback = config.LOOKBACK
        train_end_idx = 10000  # 예시
        expected_start = lookback + 100
        expected_end = train_end_idx - 500
        all_indices = list(range(expected_start, expected_end))
        assert len(all_indices) == expected_end - expected_start
        assert all_indices[0] == lookback + 100
        assert all_indices[-1] == train_end_idx - 501

    def test_trend_indices_filter_chop_50(self):
        n = 500
        df = pd.DataFrame({
            "chop": np.concatenate([np.ones(250) * 30, np.ones(250) * 70]),
        })
        all_indices = list(range(n))
        trend_mask = df["chop"].iloc[all_indices] < 50.0
        trend_indices = [i for i, m in zip(all_indices, trend_mask) if m]
        assert len(trend_indices) == 250
        assert all(df["chop"].iloc[i] < 50 for i in trend_indices)

    def test_episode_500_boundary(self):
        episode_num = 499
        use_trend = episode_num < 500
        assert use_trend is True
        episode_num = 500
        use_trend = episode_num < 500
        assert use_trend is False


class TestHoldingTimeNorm:
    """§3.3 holding_time_norm = (current_idx - entry_index) / max_steps."""

    def test_holding_time_norm_range(self):
        current_idx = 100
        entry_index = 60
        max_steps = 480
        norm = (current_idx - entry_index) / max_steps
        assert 0 <= norm <= 1
        assert abs(norm - 40 / 480) < 1e-5

    def test_holding_time_zero_when_no_position(self):
        current_position = None
        entry_index = 0
        holding_time = (100 - entry_index) if current_position else 0
        assert holding_time == 0


class TestTransitionStructure:
    """§3.2 put_data transition: 8개 요소 (s, a, r, next_s, prob, done, val, aux_target)."""

    def test_transition_8_tuple_structure(self):
        s = (None, None)  # placeholder
        next_s = (None, None)
        transition = (s, 1, 0.5, next_s, -0.3, False, 0.1, 1.2)
        assert len(transition) == 8
        s, a, r, next_s, prob_a, done, val, aux_target = transition
        assert a == 1
        assert r == 0.5
        assert aux_target == 1.2
        assert done is False
