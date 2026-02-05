from .base_strategy import BaseStrategy

class BTCEthCorrelation(BaseStrategy):
    def __init__(self): super().__init__("BTCEthCorr")
    def generate_signal(self, row, df=None):
        if row['btc_corr_60'] > 0.8:
            if row['log_return'] > 0: return 1
            elif row['log_return'] < 0: return -1
        return 0

class VolatilitySqueeze(BaseStrategy):
    """
    6. Volatility Squeeze [Tuned]
    - 문턱 완화: -2.0 -> -1.5 sigma
    - 목표: 하루 1회 -> 하루 3~5회 진입으로 학습 기회 확대
    """
    def __init__(self):
        super().__init__("VolSqueeze")

    def generate_signal(self, row, df=None):
        # [수정] 진입 장벽을 -2.0에서 -1.5로 완화
        # bb_width가 평소보다 1.5표준편차 이상 좁아지면 진입 (상위 6.7% 수준)
        if row.get('bb_width_z', 0) < -1.5:
            if row['close'] > row['open']: return 1
            elif row['close'] < row['open']: return -1
        return 0

class VWAPDeviation(BaseStrategy):
    def __init__(self): super().__init__("VWAPDeviation")
    def generate_signal(self, row, df=None):
        # Rolling Mean(24h) 기준 3% 이격 발생 시 회귀
        if row['vwap_dist'] < -0.03: return 1  # 과매도 -> 매수
        elif row['vwap_dist'] > 0.03: return -1 # 과매수 -> 매도
        return 0

class HMAMomentum(BaseStrategy):
    def __init__(self): super().__init__("HMAMomentum")
    def generate_signal(self, row, df=None):
        if row['hma_slope'] > 0: return 1
        elif row['hma_slope'] < 0: return -1
        return 0