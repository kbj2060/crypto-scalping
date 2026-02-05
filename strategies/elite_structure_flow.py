"""Elite 8 - Structure & Order Flow: Orderblock FVG, Net Taker Flow."""
from .base_strategy import BaseStrategy


class OrderblockFVGStrategy(BaseStrategy):
    """스마트 머니 흔적(FVG) 근처 반전 매매. RSI + wick_ratio 간략화."""

    def __init__(self):
        super().__init__("OrderblockFVG")

    def generate_signal(self, row, df=None):
        try:
            rsi = row.get("rsi", 50)
            wick = row.get("wick_ratio", 0)
            if rsi < 30 and wick > 0.5:
                return 1
            if rsi > 70 and wick > 0.5:
                return -1
        except (KeyError, TypeError):
            pass
        return 0


class NetTakerFlowStrategy(BaseStrategy):
    """순매수 체결 강도 추종."""

    def __init__(self):
        super().__init__("NetTakerFlow")

    def generate_signal(self, row, df=None):
        try:
            net = row.get("net_taker_ratio", 0)
            acc = row.get("taker_acceleration", 0)
            if net > 0.1 and acc > 0:
                return 1
            if net < -0.1 and acc < 0:
                return -1
        except (KeyError, TypeError):
            pass
        return 0
