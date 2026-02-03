"""
PPO 단일 에이전트 유닛 테스트 (put_data 7/8 호환, train_net aux_loss).
- action_dim=3: [Hold, Buy, Sell]
"""
import numpy as np
import torch
import pytest

from model.ppo_agent import PPOAgent
from model import config


def _make_dummy_state(device="cpu"):
    """(obs_seq, obs_info) 튜플. state_dim=29, lookback=60, info_dim=15."""
    obs_seq = torch.randn(1, 60, 29)
    obs_info = torch.randn(1, 15)
    return (obs_seq, obs_info)


def _make_transition_7():
    s = _make_dummy_state()
    next_s = _make_dummy_state()
    return (s, 0, 0.1, next_s, -0.5, False, 0.2)


def _make_transition_8():
    s = _make_dummy_state()
    next_s = _make_dummy_state()
    return (s, 0, 0.1, next_s, -0.5, False, 0.2, 1.5)  # aux_target=1.5


class TestPutDataCompatibility:
    """put_data: 7개/8개 요소 transition 모두 호환."""

    @pytest.fixture
    def agent(self):
        return PPOAgent(
            state_dim=29,
            action_dim=3,
            info_dim=15,
            hidden_dim=32,
            device="cpu",
        )

    def test_put_data_7_elements(self, agent):
        agent.put_data(_make_transition_7())
        assert len(agent.data) == 1
        assert len(agent.data[0]) == 7

    def test_put_data_8_elements(self, agent):
        agent.put_data(_make_transition_8())
        assert len(agent.data) == 1
        assert len(agent.data[0]) == 8

    def test_train_net_accepts_7_element_buffer(self, agent):
        for _ in range(4):
            agent.put_data(_make_transition_7())
        loss = agent.train_net(episode=1)
        assert isinstance(loss, (int, float))
        assert agent.data == []

    def test_train_net_accepts_8_element_buffer(self, agent):
        for _ in range(4):
            agent.put_data(_make_transition_8())
        loss = agent.train_net(episode=1)
        assert isinstance(loss, (int, float))
        assert agent.data == []


class TestTrainNetAuxLoss:
    """train_net: aux_target 언패킹, aux_loss(MSE) 0.5 가중치."""

    @pytest.fixture
    def agent_with_buffer(self):
        agent = PPOAgent(
            state_dim=29,
            action_dim=3,
            info_dim=15,
            hidden_dim=32,
            device="cpu",
        )
        for _ in range(8):
            agent.put_data(_make_transition_8())
        return agent

    def test_train_net_returns_scalar_loss(self, agent_with_buffer):
        loss = agent_with_buffer.train_net(episode=1)
        assert isinstance(loss, (int, float))
        assert loss >= 0

    def test_train_net_clears_buffer(self, agent_with_buffer):
        agent_with_buffer.train_net(episode=1)
        assert len(agent_with_buffer.data) == 0


class TestSelectActionSingleAgent:
    """select_action: (action, log_prob, value) 반환, action은 0~2 (Hold/Buy/Sell)."""

    @pytest.fixture
    def agent(self):
        return PPOAgent(
            state_dim=29,
            action_dim=3,
            info_dim=15,
            hidden_dim=32,
            device="cpu",
        )

    def test_select_action_shape(self, agent):
        state = _make_dummy_state()
        action, log_prob, value = agent.select_action(state)
        assert action in (0, 1, 2)
        assert isinstance(log_prob, (int, float))
        assert isinstance(value, (int, float))

    def test_select_action_with_mask(self, agent):
        state = _make_dummy_state()
        mask = np.ones(3, dtype=np.float32)
        mask[1] = 0.0
        action, _, _ = agent.select_action(state, action_mask=mask)
        assert action in (0, 2)


class TestPPOAgentSingle:
    """PPOAgent: 단일 에이전트, action_dim=3 (Hold, Buy, Sell)."""

    def test_single_model_created(self):
        agent = PPOAgent(state_dim=29, action_dim=3, info_dim=15, hidden_dim=32, device="cpu")
        assert agent.action_dim == 3
        assert hasattr(agent, "model")
        assert not hasattr(agent, "entry_agent")
        assert not hasattr(agent, "exit_agent")

    def test_reset_episode_states(self):
        agent = PPOAgent(state_dim=29, action_dim=3, info_dim=15, hidden_dim=32, device="cpu")
        agent.reset_episode_states()
        assert agent.current_states is None
