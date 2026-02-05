"""
DataPreprocessor (Rolling Normalization) 유닛 테스트.
전처리 epsilon 처리 및 상수/극단 입력 시 NaN 방지 검증.
"""
import numpy as np
import pytest

from common.preprocess import DataPreprocessor


class TestDataPreprocessorRollingNorm:
    """Rolling (Instance) Normalization: 윈도우 내 mean/std, epsilon 처리."""

    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()

    def test_rolling_norm_handle_constant_input(self, preprocessor):
        """모든 값이 동일한 데이터(변동성 0) → std=0 → epsilon 덕분에 NaN 없음."""
        data = np.ones((60, 29)) * 1000.0
        normalized = preprocessor.transform(data)

        assert not np.isnan(normalized).any(), "상수 입력 시 NaN이 나오면 안 됨 (epsilon 처리)"
        assert np.allclose(normalized, 0.0, atol=1e-5), "상수 입력은 평균이므로 정규화 후 0에 가까워야 함"

    def test_rolling_norm_handle_near_constant_input(self, preprocessor):
        """거의 상수인 데이터(매우 작은 std) → epsilon으로 나눗셈 오류 방지."""
        data = np.ones((60, 29)) * 1000.0
        data[:, 0] += np.linspace(0, 1e-10, 60)  # 한 컬럼만 극소 변동
        normalized = preprocessor.transform(data)

        assert not np.isnan(normalized).any(), "극소 변동 시에도 NaN이 나오면 안 됨"
        assert not np.isinf(normalized).any(), "Inf도 나오면 안 됨"

    def test_rolling_norm_uses_window_stats(self, preprocessor):
        """현재 윈도우만으로 정규화 → 윈도우 평균 0, 표준편차 1에 가깝게."""
        np.random.seed(42)
        data = np.random.randn(60, 29).astype(np.float32)
        normalized = preprocessor.transform(data)

        assert not np.isnan(normalized).any()
        assert normalized.shape == data.shape
        # 윈도우 내에서 평균 0, std 1에 가까움 (axis=0이 seq 차원)
        mean_per_feat = np.mean(normalized, axis=0)
        std_per_feat = np.std(normalized, axis=0)
        assert np.allclose(mean_per_feat, 0.0, atol=1e-5)
        assert np.allclose(std_per_feat, 1.0, atol=1e-5)

    def test_epsilon_attribute_exists(self, preprocessor):
        """epsilon 속성 존재 및 양수."""
        assert hasattr(preprocessor, "epsilon")
        assert preprocessor.epsilon > 0
