import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

from . import config
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class TradingEnvironment:
    def __init__(self, data_collector, strategies, lookback=None):
        self.collector = data_collector
        self.strategies = strategies
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False

        # [최적화] 데이터 캐싱 변수 (Pandas iloc 제거용)
        self.cached_features = None   # (T, Feature_Dim)
        self.cached_strategies = None  # (T, Strategy_Count)

        self.initial_balance = getattr(config, 'EVAL_INITIAL_CAPITAL', 10000.0)
        self.reset_reward_states()

    def check_exit_conditions(self, unrealized_pnl, holding_time_steps):
        """강제 청산 조건: Stop Loss / Take Profit / Time Stop."""
        # Stop Loss (config 활용)
        sl_threshold = getattr(config, 'STOP_LOSS_THRESHOLD', -0.02)
        if unrealized_pnl <= sl_threshold:
            return True, "STOP_LOSS"
        # Take Profit
        if unrealized_pnl >= 0.05:
            return True, "TAKE_PROFIT"
        # Time Stop: 보유 스텝이 길고 수익이 미미할 때
        if holding_time_steps > 100 and abs(unrealized_pnl) < 0.005:
            return True, "TIME_STOP"
        return False, None

    def reset_reward_states(self):
        """에피소드 시작 시 리워드 관련 상태 초기화"""
        self.trade_count = 0
        self.step_pnl_ema = 0.0
        self.consecutive_losses = 0
        self.position_changes = []
        self.strategy_confidence = []
        self.volatility_prediction_error = []
        self.current_volatility = 0.0
        self.training_step = 0
        self.episode_step_count = 0
        self.equity_curve = [1.0]
        self.peak_equity = 1.0
        self.return_buffer = []
        self.trade_history = {'count': 0, 'wins': [], 'losses': []}

    def precompute_data(self):
        """
        [Fix] Rolling Normalization 적용
        전체 통계가 아닌 '과거 Lookback 기간'의 통계로 정규화하여 미래 참조 제거.
        """
        if self.collector.eth_data is None:
            return

        logger.info("⚡ 데이터 전처리 중... (Rolling Z-Score 적용)")
        df = self.collector.eth_data.copy()

        target_cols = [
            'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos',
            'rsi', 'macd_hist', 'hma_ratio', 'cci',
            'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
            'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
            'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
            'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = 0.0

        data = df[target_cols]
        rolling = data.rolling(window=self.lookback, min_periods=1)
        roll_mean = rolling.mean()
        roll_std = rolling.std().replace(0, 1e-8)
        normalized_df = (data - roll_mean) / roll_std
        normalized_df = normalized_df.fillna(0).replace([np.inf, -np.inf], 0)

        self.cached_features = torch.FloatTensor(normalized_df.values.astype(np.float32))
        self.scaler_fitted = True

        strat_scores = []
        for i in range(len(self.strategies)):
            col = f'strategy_{i}'
            if col in df.columns:
                strat_scores.append(torch.tensor(df[col].values.astype(np.float32)))
            else:
                strat_scores.append(torch.zeros(len(df), dtype=torch.float32))
        self.cached_strategies = torch.stack(strat_scores, dim=1)

        logger.info(f"✅ 데이터 전처리 완료 (Rolling Norm): {self.cached_features.shape}")

    def update_trading_metrics(self, prev_position, current_position,
                               strategy_scores=None, volatility_pred=None,
                               actual_volatility=None):
        self.position_changes.append(1.0 if prev_position != current_position else 0.0)
        if len(self.position_changes) > 100:
            self.position_changes = self.position_changes[-100:]
        self.training_step += 1

    def get_observation(self, position_info=None, current_index=None):
        """캐시된 Tensor 슬라이싱으로 관측 생성 (iloc 제거)."""
        try:
            if self.cached_features is None:
                self.precompute_data()
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            if curr_idx is None or curr_idx < self.lookback:
                return None
            if curr_idx >= len(self.cached_features):
                return None

            obs_seq = self.cached_features[curr_idx - self.lookback : curr_idx].unsqueeze(0)
            scores = self.cached_strategies[curr_idx]

            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            if not isinstance(position_info, torch.Tensor):
                pos_tensor = torch.tensor(position_info, dtype=torch.float32)
            else:
                pos_tensor = position_info

            obs_info = torch.cat([pos_tensor[0:1], scores, pos_tensor[1:]]).unsqueeze(0)
            return (obs_seq, obs_info)
        except Exception as e:
            logger.error(f"Obs Error: {e}")
            return None

    def calculate_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None):
        """보상: 스텝 보상, 실현 손익, Sharpe 보너스, MDD 페널티."""
        reward = 0.0

        # A. 스텝 보상 (포지션 보유 시)
        if current_position is not None:
            step_reward = step_pnl * 100.0
            step_reward = max(min(step_reward, 2.0), -2.0)
            reward += step_reward

        # B. 매매 실현 손익 (청산 시)
        if trade_done and realized_pnl != 0.0:
            trade_reward = realized_pnl * 100.0
            if realized_pnl > 0:
                trade_reward *= 1.2
            trade_reward = max(min(trade_reward, 5.0), -5.0)
            reward += trade_reward
            if action in [1, 2]:
                reward -= 0.01

        # C. Sharpe 기반 보너스 (안정성)
        self.return_buffer.append(step_pnl)
        if len(self.return_buffer) > 50:
            self.return_buffer.pop(0)
        if len(self.return_buffer) >= 10:
            returns = np.array(self.return_buffer)
            std_returns = np.std(returns)
            if std_returns > 1e-8:
                sharpe = np.mean(returns) / std_returns
                reward += sharpe * 0.1

        # D. MDD 페널티 (순간 급락)
        if step_pnl < -0.05:
            reward -= 1.0

        # E. 최종 클리핑
        reward = max(min(reward, 5.0), -5.0)
        return float(reward)

    def get_state_dim(self):
        """실제 캐시된 피처 차원 반환 (네트워크 에러 방지)."""
        if self.cached_features is not None:
            return self.cached_features.shape[1]
        return 29

    def get_current_equity(self):
        """현재 자산 가치 (1.0 = 100% 시작)."""
        return self.equity_curve[-1] if self.equity_curve else 1.0