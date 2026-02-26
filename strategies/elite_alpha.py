"""Elite 8 - Alpha: Whale Sentiment [Tuned], Liquidation Squeeze Hunter."""
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class WhaleSentimentDivergence(BaseStrategy):
    """
    Whale Sentiment Divergence [Tuned - Continuous]
    - 고래 비율 1.48 평균 기준 정규화
    - 가격 방향과 고래 방향 다르면 다이버전스로 강한 신호, 같으면 약한 신호
    """

    def __init__(self):
        super().__init__("WhaleSentiment")

    def generate_signal(self, row, df=None) -> float:
        if df is None or len(df) < 2:
            return 0.0
            
        try:
            idx_pos = df.index.get_loc(row.name)
            if idx_pos == 0:
                return 0.0
            prev_row = df.iloc[idx_pos - 1]
        except (KeyError, IndexError, TypeError):
            return 0.0

        try:
            ratio = float(row.get("whale_retail_ratio", 1.0))
            conviction = float(row.get("whale_conviction", 0.0))
            
            cur_close = float(row.get("close", 0.0))
            prev_close = float(prev_row.get("close", 0.0))
            
            # 가격 방향
            price_dir = 1.0 if cur_close > prev_close else -1.0 if cur_close < prev_close else 0.0
            
            # 고래 강도: ratio가 평균(1.48)에서 벗어난 정도
            whale_strength = (ratio - 1.48) * 5.0  # ±0.1 → ±0.5
            whale_dir = whale_strength * (1.0 + abs(conviction))
            
            # 다이버전스: 가격↓ + 고래↑ → 매수(+1), 가격↑ + 고래↓ → 매도(-1)
            if price_dir * whale_dir < 0:
                return float(np.clip(whale_dir, -1.0, 1.0))
            else:
                return float(np.clip(whale_dir * 0.3, -1.0, 1.0))
                
        except (AttributeError, TypeError, ValueError):
            return 0.0


class LiquidationSqueezeHunter(BaseStrategy):
    """Liquidation Squeeze Hunter — OI 급증 + 펀딩비 쏠림 [Continuous]."""

    def __init__(self):
        super().__init__("LiqSqueeze")

    def generate_signal(self, row, df=None) -> float:
        try:
            smf = float(row.get("smart_money_flow", 0.0))
            funding = float(row.get("last_funding_rate", 0.0))
            
            smf_std = 1.0
            if df is not None and "smart_money_flow" in df.columns:
                calc_std = df["smart_money_flow"].std()
                if pd.notna(calc_std) and calc_std > 0:
                    smf_std = calc_std

            # 1. OI 급증 강도
            oi_strength = smf / smf_std
            if oi_strength < 1.0:
                return 0.0  # 급증 아님

            # 2. 펀딩비 쏠림 강도
            funding_strength = abs(funding) / 0.0003
            if funding_strength < 0.3:
                return 0.0  # 쏠림 부족

            # 3. 펀딩비 반대 방향 역배팅
            squeeze_signal = -np.sign(funding) * min(oi_strength, 2.0) * min(funding_strength, 1.5)
            
            return float(np.clip(squeeze_signal / 3.0, -1.0, 1.0))
            
        except (KeyError, TypeError, ValueError):
            return 0.0