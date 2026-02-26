"""
Elite 8 - Standard Strategies
================================================================
[DEPRECATED] RL State 통합 모듈(elite_builder.py) 설계에 따라 다음 전략들은 
더 정교한 지표로 대체되거나 제거되었습니다.

🔴 BTCEthCorr       — RL State의 regime_direction이 대체 (후행+중복)
🔴 HMAMomentum      — RL State의 short_term_bias가 대체 (과도한 신호)
🔴 VWAPDeviation    — RL State의 TFT vwap_distance가 대체 (5분봉 신호 희소)
🔴 VolSqueeze       — RL State의 Squeeze 점수 및 regime_trending이 대체

* 하위 호환성을 위해 클래스 골격은 유지하되, 모든 신호는 0.0(Neutral)을 반환합니다.
"""
from .base_strategy import BaseStrategy

class BTCEthCorrelation(BaseStrategy):
    def __init__(self): 
        super().__init__("BTCEthCorr")
        
    def generate_signal(self, row, df=None) -> float:
        return 0.0

class VolatilitySqueeze(BaseStrategy):
    def __init__(self):
        super().__init__("VolSqueeze")
        
    def generate_signal(self, row, df=None) -> float:
        return 0.0

class VWAPDeviation(BaseStrategy):
    def __init__(self): 
        super().__init__("VWAPDeviation")
        
    def generate_signal(self, row, df=None) -> float:
        return 0.0

class HMAMomentum(BaseStrategy):
    def __init__(self): 
        super().__init__("HMAMomentum")
        
    def generate_signal(self, row, df=None) -> float:
        return 0.0