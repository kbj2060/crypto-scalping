"""
보상 함수 (3-Action 상태 변화 기반) 유닛 테스트.
- is_entry / is_exit + trade_done, 클리핑 [-2, 2]
"""
import numpy as np
import pytest

from common.trading_env import TradingEnvironment


@pytest.fixture
def env_for_reward(mock_collector, strategies_8):
    """보상 전용 TradingEnvironment (get_observation 미사용)."""
    return TradingEnvironment(mock_collector, strategies_8, lookback=60)


class TestCalculateRewardSignature:
    """시그니처: holding_time, agent_type 인자 지원 (agent_type 무시)."""

    def test_accepts_holding_time_and_agent_type(self, env_for_reward):
        r = env_for_reward.calculate_reward(
            0.0, 0.0, False, holding_time=0.5, action=0, agent_type="ENTRY"
        )
        assert isinstance(r, (int, float))

    def test_accepts_exit_agent_type(self, env_for_reward):
        r = env_for_reward.calculate_reward(
            0.0, 0.0, False, holding_time=0.0, action=2,
            prev_position='LONG', current_position=None, agent_type="EXIT"
        )
        assert isinstance(r, (int, float))


class TestStateBasedReward:
    """3-Action: 진입(is_entry) -0.05, 청산(is_exit) -0.02."""

    def test_entry_penalty(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r_entry = env_for_reward.calculate_reward(
            0.0, 0.0, False, prev_position=None, current_position='LONG'
        )
        env_for_reward.reset_reward_states()
        r_hold = env_for_reward.calculate_reward(
            0.0, 0.0, False, prev_position=None, current_position=None
        )
        assert r_entry < r_hold
        assert (r_hold - r_entry) >= 0.04  # 진입 페널티 -0.05

    def test_exit_penalty(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r = env_for_reward.calculate_reward(
            0.0, 0.0, False, prev_position='LONG', current_position=None
        )
        assert r <= -0.01  # 청산 페널티 -0.02

    def test_hold_no_entry_exit(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r = env_for_reward.calculate_reward(
            0.0, 0.0, False, prev_position=None, current_position=None
        )
        assert r >= -0.02  # step_pnl 0, 포지션 없음


class TestStepPnlReward:
    """step_pnl: 양수면 보상 증가, 음수면 보상 감소 (log1p 기반)."""

    def test_positive_step_pnl_increases_reward(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r_pos = env_for_reward.calculate_reward(0.01, 0.0, False, action=0)
        env_for_reward.reset_reward_states()
        r_neg = env_for_reward.calculate_reward(-0.01, 0.0, False, action=0)
        assert r_pos > r_neg


class TestResetRewardStates:
    """reset_reward_states(): trade_count 초기화."""

    def test_reset_clears_trade_count(self, env_for_reward):
        env_for_reward.trade_count = 5
        env_for_reward.reset_reward_states()
        assert env_for_reward.trade_count == 0


class TestRewardClipping:
    """보상 클리핑 [-2.0, 2.0]."""

    def test_large_loss_clipped(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r = env_for_reward.calculate_reward(-0.05, 0.0, False, action=0)
        assert r >= -2.0
        assert r <= 0.0

    def test_small_loss_unchanged(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r = env_for_reward.calculate_reward(-0.01, 0.0, False, action=0)
        assert r > -2.0


class TestTradeDoneBonus:
    """청산 시: realized_pnl 기반 보상, trade_count 증가."""

    def test_trade_done_increments_trade_count(self, env_for_reward):
        env_for_reward.reset_reward_states()
        assert env_for_reward.trade_count == 0
        env_for_reward.calculate_reward(
            0.0, 0.0, True, prev_position='LONG', current_position=None
        )
        assert env_for_reward.trade_count == 1

    def test_trade_done_adds_realized_bonus(self, env_for_reward):
        env_for_reward.reset_reward_states()
        r_positive = env_for_reward.calculate_reward(
            0.0, 0.02, True, prev_position='LONG', current_position=None
        )
        env_for_reward.reset_reward_states()
        r_negative = env_for_reward.calculate_reward(
            0.0, -0.02, True, prev_position='LONG', current_position=None
        )
        assert r_positive > r_negative


class TestGetStateDim:
    """get_state_dim() == len(ULTIMATE_FEATURE_COLS)."""

    def test_state_dim_equals_ultimate_features(self, env_for_reward):
        from core.feature_engineering import ULTIMATE_FEATURE_COLS
        assert env_for_reward.get_state_dim() == len(ULTIMATE_FEATURE_COLS)
