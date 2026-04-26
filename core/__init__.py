"""
핵심 모듈 (Core Modules)
- 바이낸스 API 클라이언트, 데이터 수집, 리스크 관리, 피처 엔지니어링
"""
from .binance_client import BinanceClient
from .data_collector import DataCollector
from .cvp import add_cvp_features

# RiskManager는 공용 모듈(common)에 위치하지만,
# 기존 코드(`from core import RiskManager`)와의 하위 호환을 위해 재노출한다.
from .risk_manager import RiskManager

__all__ = [
    'BinanceClient',
    'DataCollector',
    'FeatureEngineer',
    'add_cvp_features',
    'RiskManager',
]


def __getattr__(name):
    if name == "FeatureEngineer":
        from .feature_engineering import FeatureEngineer

        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
