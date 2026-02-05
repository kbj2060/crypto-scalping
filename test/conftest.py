"""
공통 픽스처: 구현 명세(IMPLEMENTATION_SPECIFICATION.md) 기반 유닛 테스트용.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_collector():
    """보상/관측 테스트용 최소 DataCollector 모킹 (eth_data, current_index). Elite 8 + Ultimate Feature Set."""
    from common.feature_engineering import ULTIMATE_FEATURE_COLS
    n = 200
    cols = list(ULTIMATE_FEATURE_COLS)
    df = pd.DataFrame(np.zeros((n, len(cols))), columns=cols)
    df["close"] = 1000.0
    df["open"] = 998.0
    df["high"] = 1005.0
    df["low"] = 995.0
    for i in range(8):
        df[f"strategy_{i}"] = 0.0
    mock = type("MockCollector", (), {"eth_data": df, "current_index": 60})()
    return mock


@pytest.fixture
def strategies_8():
    """Elite 8 전략 (obs_info 8 + 3 = 11 차원)."""
    return list(range(8))
