"""Feature package local base strategy to avoid cross-package import cycles."""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signal(self, row, df=None, **kwargs):
        pass

