"""Elite 8 - Alpha: Whale Sentiment [Tuned], Liquidation Squeeze Hunter."""
import pandas as pd
from .base_strategy import BaseStrategy


class WhaleSentimentDivergence(BaseStrategy):
    """
    Whale Sentiment Divergence [Tuned]
    - 평균 Ratio가 1.48이므로, 기준을 1.55(롱) / 1.40(숏)으로 조정.
    """

    def __init__(self):
        super().__init__("WhaleSentiment")

    def generate_signal(self, row, df=None):
        if df is None or len(df) < 2:
            return 0
        try:
            idx_pos = df.index.get_loc(row.name)
            if idx_pos == 0:
                return 0
            prev_row = df.iloc[idx_pos - 1]
        except (KeyError, IndexError, TypeError):
            return 0
        # Long: 가격 하락 + 고래 비율 1.55 이상(평균 상회) + 확신 증가
        price_dropping = row["close"] < prev_row["close"]
        whale_strong = row.get("whale_retail_ratio", 1.0) > 1.55 and row.get("whale_conviction", 0) > 0.0
        if price_dropping and whale_strong:
            return 1
        # Short: 가격 상승 + 고래 비율 1.40 이하(평균 하회) + 확신 감소
        price_rising = row["close"] > prev_row["close"]
        whale_weak = row.get("whale_retail_ratio", 1.0) < 1.40 and row.get("whale_conviction", 0) < 0.0
        if price_rising and whale_weak:
            return -1
        return 0


class LiquidationSqueezeHunter(BaseStrategy):
    """Liquidation Squeeze Hunter — OI 급증 + 펀딩비 쏠림."""

    def __init__(self):
        super().__init__("LiqSqueeze")

    def generate_signal(self, row, df=None):
        if df is None:
            return 0
        try:
            smf_std = df["smart_money_flow"].std()
            if pd.isna(smf_std) or smf_std == 0:
                smf_std = 1.0
            oi_surge = row.get("smart_money_flow", 0) > smf_std
            high_funding = abs(row.get("last_funding_rate", 0)) > 0.0001
            if oi_surge and high_funding:
                if row.get("last_funding_rate", 0) < 0:
                    return 1
                if row.get("last_funding_rate", 0) > 0:
                    return -1
        except (KeyError, TypeError):
            pass
        return 0
