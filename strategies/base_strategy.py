"""Elite 8 공통 부모: row + df 기반 시그널 생성."""
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, row, df=None, **kwargs):
        """
        :param row: 현재 캔들 데이터 (Series)
        :param df: 전체 DataFrame (lookback 필요 시)
        :param kwargs: 글로벌 통계값 (예: smf_std) 전달용
        :return: 1 (Long), -1 (Short), 0 (Neutral)
        """
        pass