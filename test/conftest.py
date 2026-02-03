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
    """보상/관측 테스트용 최소 DataCollector 모킹 (eth_data, current_index)."""
    n = 200
    cols = [
        "log_return", "roll_return_6", "atr_ratio", "bb_width", "bb_pos",
        "rsi", "macd_hist", "hma_ratio", "cci", "rvol", "taker_ratio",
        "cvd_change", "mfi", "cmf", "vwap_dist", "wick_upper", "wick_lower",
        "range_pos", "swing_break", "chop",
        "btc_return", "btc_rsi", "btc_corr", "btc_vol", "eth_btc_ratio",
        "rsi_15m", "trend_15m", "rsi_1h", "trend_1h",
    ]
    df = pd.DataFrame(np.zeros((n, len(cols))), columns=cols)
    df["close"] = 1000.0
    df["high"] = 1005.0
    df["low"] = 995.0
    for i in range(12):
        df[f"strategy_{i}"] = 0.0
    mock = type("MockCollector", (), {"eth_data": df, "current_index": 60})()
    return mock


@pytest.fixture
def strategies_12():
    """전략 12개 리스트 (obs_info 12 + 3 = 15 차원)."""
    return list(range(12))
