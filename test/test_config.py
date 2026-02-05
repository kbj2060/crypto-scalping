"""
§5 설정 (config.py) 유닛 테스트.
명세: docs/IMPLEMENTATION_SPECIFICATION.md §5
"""
import pytest

from common import config


class TestRewardConfig:
    """보상 관련: REWARD_MULTIPLIER, LOSS_PENALTY_MULTIPLIER, STOP_LOSS_THRESHOLD."""

    def test_reward_multiplier_exists(self):
        assert hasattr(config, "REWARD_MULTIPLIER")
        assert isinstance(config.REWARD_MULTIPLIER, (int, float))

    def test_loss_penalty_multiplier_exists(self):
        assert hasattr(config, "LOSS_PENALTY_MULTIPLIER")
        assert isinstance(config.LOSS_PENALTY_MULTIPLIER, (int, float))

    def test_stop_loss_threshold_negative(self):
        assert hasattr(config, "STOP_LOSS_THRESHOLD")
        assert config.STOP_LOSS_THRESHOLD <= 0


class TestPPOConfig:
    """PPO: PPO_ENTROPY_COEF, PPO_LEARNING_RATE, PPO_EPS_CLIP, PPO_K_EPOCHS."""

    def test_ppo_entropy_coef(self):
        assert hasattr(config, "PPO_ENTROPY_COEF")
        assert 0 <= config.PPO_ENTROPY_COEF <= 1

    def test_ppo_learning_rate_positive(self):
        assert hasattr(config, "PPO_LEARNING_RATE")
        assert config.PPO_LEARNING_RATE > 0

    def test_ppo_eps_clip_positive(self):
        assert hasattr(config, "PPO_EPS_CLIP")
        assert config.PPO_EPS_CLIP > 0

    def test_ppo_k_epochs_positive(self):
        assert hasattr(config, "PPO_K_EPOCHS")
        assert config.PPO_K_EPOCHS >= 1


class TestTrainConfig:
    """학습: TRAIN_BATCH_SIZE, TRAIN_MAX_STEPS_PER_EPISODE."""

    def test_train_batch_size(self):
        assert hasattr(config, "TRAIN_BATCH_SIZE")
        assert config.TRAIN_BATCH_SIZE >= 1

    def test_train_max_steps_per_episode(self):
        assert hasattr(config, "TRAIN_MAX_STEPS_PER_EPISODE")
        assert config.TRAIN_MAX_STEPS_PER_EPISODE >= 1


class TestNetworkConfig:
    """네트워크: NETWORK_HIDDEN_DIM, NETWORK_NUM_LAYERS, NETWORK_DROPOUT."""

    def test_network_hidden_dim(self):
        assert hasattr(config, "NETWORK_HIDDEN_DIM")
        assert config.NETWORK_HIDDEN_DIM >= 1

    def test_network_num_layers(self):
        assert hasattr(config, "NETWORK_NUM_LAYERS")
        assert config.NETWORK_NUM_LAYERS >= 1

    def test_network_dropout_range(self):
        assert hasattr(config, "NETWORK_DROPOUT")
        assert 0 <= config.NETWORK_DROPOUT <= 1


class TestDataSplit:
    """TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0."""

    def test_split_sum_one(self):
        total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
        assert abs(total - 1.0) < 1e-5


class TestLookback:
    """LOOKBACK 양수."""

    def test_lookback_positive(self):
        assert config.LOOKBACK >= 1
