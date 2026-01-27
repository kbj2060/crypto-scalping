"""
PPO 전용 트레이딩 환경 (Hybrid Reward Version)
- 학술 연구 기반 최적화 (MDD Penalty + Sortino Ratio + Profit Factor)
- 목표: 거래량 정상화 (950회 -> 50회) 및 수익 곡선 우상향
"""
import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)

class TradingEnvironment:
    def __init__(self, data_collector, strategies, lookback=None):
        self.collector = data_collector
        self.strategies = strategies
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False
        
        # [추가] 하이브리드 리워드용 상태 변수 초기화
        self.reset_reward_states()

    def reset_reward_states(self):
        """에피소드 시작 시 리워드 관련 상태 초기화"""
        self.equity_curve = [10000.0] # 자산 곡선 (시작 10000)
        self.peak_equity = 10000.0    # 최고 자산 (MDD 계산용)
        self.trade_history = {'wins': [], 'losses': [], 'count': 0}
        self.return_buffer = []       # 최근 수익률 (Sortino 계산용)
        self.episode_step_count = 0   # 에피소드 내 스텝 수

    def get_observation(self, position_info=None, current_index=None):
        """
        PPO 상태 관측
        Returns: (obs_seq, obs_info) 튜플 또는 None
        """
        try:
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            if curr_idx is None or curr_idx < self.lookback: return None
            df = self.collector.eth_data
            if df is None or curr_idx >= len(df): return None

            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            if not self.scaler_fitted: return None
            for col in target_cols:
                if col not in df.columns: df[col] = 0.0

            recent_df = df[target_cols].iloc[curr_idx - self.lookback : curr_idx]
            if len(recent_df) < self.lookback: return None
            
            seq = self.preprocessor.transform(recent_df.values.astype(np.float32))
            obs_seq = torch.FloatTensor(seq).unsqueeze(0)

            scores = []
            for i in range(len(self.strategies)):
                col = f'strategy_{i}'
                scores.append(float(df[col].iloc[curr_idx]) if col in df.columns else 0.0)
            
            if position_info is None: position_info = [0.0, 0.0, 0.0]
            obs_info = np.concatenate([scores, position_info], dtype=np.float32)
            obs_info = torch.FloatTensor(obs_info).unsqueeze(0)
            
            return (obs_seq, obs_info)
        except: return None

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0, action=0, prev_position=None, current_position=None):
        """
        [하이브리드 리워드 시스템]
        1. MDD Penalty: 낙폭이 커지면 강력한 처벌 (안정성)
        2. Sortino Ratio: 하방 변동성 대비 수익률 보상 (효율성)
        3. Profit Factor: 승률과 손익비 관리 (일관성)
        4. 거래 빈도 규제: 과잉 거래(Churning) 방지
        """
        reward = 0.0
        self.episode_step_count += 1
        
        # -----------------------------------------------------------
        # 1. 자산 곡선 및 상태 업데이트
        # -----------------------------------------------------------
        current_equity = self.equity_curve[-1]
        
        # 평가금액 업데이트 (포지션 없으면 변동 없음)
        if current_position is not None:
            new_equity = current_equity * (1 + step_pnl)
            self.return_buffer.append(step_pnl)
        else:
            new_equity = current_equity
            self.return_buffer.append(0.0)
            
        self.equity_curve.append(new_equity)
        
        # 버퍼 관리 (메모리 최적화)
        if len(self.return_buffer) > 100: self.return_buffer.pop(0)
        if len(self.equity_curve) > 200: self.equity_curve.pop(0)

        # -----------------------------------------------------------
        # 2. 포지션 보유 중 보상 (Holding Reward)
        # -----------------------------------------------------------
        if current_position is not None:
            # ===== A. Drawdown Penalty (최우선 순위: 생존) =====
            if new_equity > self.peak_equity:
                self.peak_equity = new_equity
                reward += 0.5  # 신고점 갱신 보너스 (작게)
            
            drawdown = (self.peak_equity - new_equity) / self.peak_equity
            
            # [핵심] 낙폭이 1% 넘어가는 순간부터 기하급수적 페널티
            if drawdown > 0.01:
                dd_penalty = -50.0 * (drawdown ** 2) 
                reward += dd_penalty

            # ===== B. Sortino Ratio (효율성) =====
            if len(self.return_buffer) >= 30:
                returns = np.array(self.return_buffer)
                mean_ret = np.mean(returns)
                downside = returns[returns < 0]
                
                if len(downside) > 0:
                    downside_std = np.std(downside) + 1e-8
                else:
                    downside_std = 1e-8
                
                sortino = mean_ret / downside_std
                # Sortino가 양수일 때만 보상 (수익이 나야 의미 있음)
                if sortino > 0:
                    reward += sortino * 2.0

            # ===== C. 추세 추종 가중치 =====
            # 오래 버틸수록 스텝 보상 증가 (Time Factor)
            time_weight = min(1.0 + holding_time * 0.02, 2.0)
            reward += step_pnl * 10.0 * time_weight

        # -----------------------------------------------------------
        # 3. 거래 종료 시 보상 (Terminal Reward)
        # -----------------------------------------------------------
        if trade_done:
            fee = 0.0005 # 0.05% 수수료 가정
            net_pnl = realized_pnl - fee
            
            # 거래 기록 업데이트
            self.trade_history['count'] += 1
            if net_pnl > 0:
                self.trade_history['wins'].append(net_pnl)
            else:
                self.trade_history['losses'].append(abs(net_pnl))
                
            # ===== A. 기본 청산 보상 =====
            reward += net_pnl * 50.0 # 기본 배율
            
            # ===== B. Profit Factor Bonus (일관성) =====
            if len(self.trade_history['wins']) > 0 and len(self.trade_history['losses']) > 0:
                recent_wins = self.trade_history['wins'][-30:]
                recent_losses = self.trade_history['losses'][-30:]
                
                total_win = sum(recent_wins)
                total_loss = sum(recent_losses) + 1e-8
                profit_factor = total_win / total_loss
                
                # PF > 1.5 목표: 달성 시 큰 보너스, 1.0 미만 시 페널티
                if profit_factor > 1.5:
                    reward += (profit_factor - 1.5) * 5.0
                elif profit_factor < 1.0:
                    reward -= (1.0 - profit_factor) * 5.0

            # ===== C. 보유 시간 & 손절 결단력 =====
            if net_pnl > 0:
                # 익절인데 너무 빨리 팔면(5스텝 미만) 페널티 (스캘핑이라도 너무 짧은 건 노이즈)
                if holding_time < 5:
                    reward -= 1.0
                # 적당히 길게 먹으면 보너스
                elif holding_time > 20:
                    reward += 1.0
            else:
                # 손절은 빨리 할수록 칭찬 (-2% 이내 빠른 손절)
                if holding_time < 10 and net_pnl > -0.02:
                    reward += 2.0 

        # -----------------------------------------------------------
        # 4. 거래 빈도 규제 (과잉 거래 방지 - 핵심!)
        # -----------------------------------------------------------
        # 에피소드 진행 100 스텝당 거래 횟수가 너무 많으면 페널티
        # 예: 480 스텝에 20회 이상 거래 -> 과잉
        if self.episode_step_count > 50:
            trade_density = self.trade_history['count'] / self.episode_step_count
            # 10스텝당 1회 이상 거래(0.1)는 너무 잦음 -> 페널티
            if trade_density > 0.1: 
                reward -= 0.5 # 지속적인 페널티로 거래 억제

        # -----------------------------------------------------------
        # 5. 무포지션 페널티 (너무 오래 쉬지 않도록)
        # -----------------------------------------------------------
        if prev_position is None and current_position is None:
            reward -= 0.01

        # 클리핑 (학습 안정성)
        return np.clip(reward, -20, 20)

    def get_state_dim(self):
        return 29
