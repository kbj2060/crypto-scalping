"""
Kelly Criterion 기반 포지션 사이징
- 장기 복리 성장률을 수학적으로 최대화
- Fractional Kelly로 파산 확률 제거
- 실시간 승률/배당비 추적
"""
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion Position Sizer
    
    f* = (p * b - q) / b
    where:
        f* = optimal fraction of capital to bet
        p = win probability
        q = loss probability (1 - p)
        b = win/loss ratio (avg_win / avg_loss)
    
    실전에서는 Fractional Kelly (0.25x ~ 0.5x)를 적용하여
    추정 오차에 대한 안전 마진을 확보합니다.
    """
    
    def __init__(self, fraction=0.25, max_leverage=20, min_trades=30, window_size=200):
        """
        Args:
            fraction: Kelly fraction (0.25 = Quarter Kelly, 가장 보수적이고 안전)
            max_leverage: 절대 상한 레버리지
            min_trades: 통계적 신뢰를 위한 최소 거래 수
            window_size: Rolling window 크기 (최근 N개 거래)
        """
        self.fraction = fraction
        self.max_leverage = max_leverage
        self.min_trades = min_trades
        
        # Rolling trade history
        self.trade_results = deque(maxlen=window_size)
        
        # Cached statistics
        self._cached_kelly = 0.0
        self._cached_win_rate = 0.0
        self._cached_win_loss_ratio = 0.0
        self._update_counter = 0
        
    def record_trade(self, pnl_pct: float):
        """
        거래 결과 기록
        
        Args:
            pnl_pct: 수익률 (예: 0.02 = +2%, -0.01 = -1%)
        """
        self.trade_results.append(pnl_pct)
        self._update_counter += 1
        
        # 매 10번째 기록마다 통계 업데이트 (성능 최적화)
        if self._update_counter % 10 == 0:
            self._update_statistics()
    
    def _update_statistics(self):
        """승률 및 배당비 재계산"""
        if len(self.trade_results) < self.min_trades:
            return
        
        results = np.array(self.trade_results)
        wins = results[results > 0]
        losses = results[results < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            self._cached_kelly = 0.0
            return
        
        self._cached_win_rate = len(wins) / len(results)
        avg_win = np.mean(wins)
        avg_loss = np.mean(np.abs(losses))
        
        self._cached_win_loss_ratio = avg_win / (avg_loss + 1e-8)
        
        # Kelly Formula
        p = self._cached_win_rate
        q = 1 - p
        b = self._cached_win_loss_ratio
        
        raw_kelly = (p * b - q) / (b + 1e-8)
        
        # Fractional Kelly with floor at 0
        self._cached_kelly = max(0.0, raw_kelly * self.fraction)
        
    def get_optimal_leverage(self, meta_risk_budget: float = 1.0) -> float:
        """
        최적 레버리지 계산
        
        Args:
            meta_risk_budget: MetaController가 할당한 리스크 예산 (0.0 ~ 1.0)
        
        Returns:
            optimal_leverage: 사용할 레버리지 배수
        """
        if len(self.trade_results) < self.min_trades:
            # 데이터 부족 → 보수적 기본값 (3배)
            base_leverage = 3.0
        else:
            # Kelly 기반 최적 레버리지
            # Kelly fraction이 자산의 몇 %를 배팅하라는 것이므로
            # 레버리지로 변환: kelly_f * max_leverage
            base_leverage = self._cached_kelly * self.max_leverage
        
        # MetaController의 리스크 예산 반영
        adjusted_leverage = base_leverage * meta_risk_budget
        
        # 상하한 클리핑
        final_leverage = np.clip(adjusted_leverage, 0.0, self.max_leverage)
        
        return float(final_leverage)
    
    def get_position_size(self, action_intensity: float, meta_risk_budget: float = 1.0) -> float:
        """
        TD3 action을 실제 포지션 크기로 변환
        
        Args:
            action_intensity: TD3 출력 절대값 (0.0 ~ 1.0)
            meta_risk_budget: MetaController 리스크 예산 (0.0 ~ 1.0)
        
        Returns:
            effective_leverage: 실제 적용할 레버리지
        """
        optimal = self.get_optimal_leverage(meta_risk_budget)
        
        # TD3의 action_intensity를 optimal leverage 범위 내에서 스케일링
        # action=1.0이면 optimal_leverage 100% 사용
        # action=0.5이면 optimal_leverage 50% 사용
        effective = action_intensity * optimal
        
        return float(np.clip(effective, 0.0, self.max_leverage))
    
    def get_stats(self) -> dict:
        """현재 통계 반환 (로깅/모니터링용)"""
        return {
            'total_trades': len(self.trade_results),
            'win_rate': self._cached_win_rate,
            'win_loss_ratio': self._cached_win_loss_ratio,
            'raw_kelly': self._cached_kelly / (self.fraction + 1e-8),
            'fractional_kelly': self._cached_kelly,
            'optimal_leverage': self.get_optimal_leverage(),
            'fraction': self.fraction,
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (
            f"KellyCriterion(trades={stats['total_trades']}, "
            f"WR={stats['win_rate']:.1%}, "
            f"W/L={stats['win_loss_ratio']:.2f}, "
            f"Kelly={stats['fractional_kelly']:.3f}, "
            f"OptLev={stats['optimal_leverage']:.1f}x)"
        )
