"""
공통 거래 환경 (TD3 / MacroHFT 공용) - Optimized for Elite 8
- DataCollector + strategies 기반 관측(obs_seq, obs_info) 생성
- [Optimization] Full GPU Caching & Auto-Signal Generation
"""
import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

from common import config
from common.feature_engineering import ULTIMATE_FEATURE_COLS

logger = logging.getLogger(__name__)

# Elite 8 기준: pos_val(1) + strategies(8) + pos_info[1:](2) = 11
INFO_DIM_ELITE8 = 11


class TradingEnvironment:
    def __init__(self, data_collector, strategies, lookback=None):
        self.collector = data_collector
        self.strategies = strategies  # List of initialized strategy objects
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.scaler_fitted = False

        # [최적화 1] 디바이스 설정 (GPU 사용 강제)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [최적화 2] 데이터 캐싱 변수 (GPU Tensor)
        self.cached_features = None   # (T, Feature_Dim) on GPU
        self.cached_strategies = None  # (T, Strategy_Count) on GPU

        self.initial_balance = getattr(config, 'EVAL_INITIAL_CAPITAL', 10000.0)
        self.reset_reward_states()

    def check_exit_conditions(self, unrealized_pnl_roe, holding_time_steps=0):
        """
        [레버리지 시스템] 청산 로직
        
        Args:
            unrealized_pnl_roe: 레버리지가 적용된 ROE (Return On Equity)
                              = (가격 변동률) * (레버리지)
                              예: 가격 +1%, 레버리지 10배 → ROE = +10%
            holding_time_steps: 홀딩 시간 (스텝 수)
        
        Returns:
            (should_exit: bool, reason: str)
        """
        # 1. 강제 청산 (Liquidation) - 바이낸스 마진콜 시뮬레이션
        # 자산의 80%가 날아가면 강제 종료 (AI에게 죽음의 공포를 가르침)
        liquidation_threshold = getattr(config, 'LIQUIDATION_THRESHOLD', -0.80)
        if unrealized_pnl_roe <= liquidation_threshold:
            return True, "LIQUIDATION"  # 게임 오버
        
        # 2. 익절 (Take Profit) - 레버리지 20배면 50% 수익은 금방
        # 가격 2.5% 상승 * 레버리지 20배 = ROE +50%
        take_profit_threshold = getattr(config, 'TAKE_PROFIT_THRESHOLD', 0.50)
        if unrealized_pnl_roe >= take_profit_threshold:
            return True, "TAKE_PROFIT"
        
        # 3. 손절 (Stop Loss) - 자산 기준 -20% 손실
        # 가격 1% 하락 * 레버리지 20배 = ROE -20%
        stop_loss_threshold = getattr(config, 'STOP_LOSS_THRESHOLD', -0.20)
        if unrealized_pnl_roe <= stop_loss_threshold:
            return True, "STOP_LOSS"
        
        # 4. 시간 손절 (Time Stop) - 횡보 시 수수료만 빠져나감
        # 100 스텝 동안 홀딩했는데 수익이 미미하면 정리
        if holding_time_steps > 100 and abs(unrealized_pnl_roe) < 0.01:  # ROE 1% 미만
            return True, "TIME_STOP"
        
        return False, None

    def reset_reward_states(self):
        """에피소드 초기화"""
        self.trade_count = 0
        self.step_pnl_ema = 0.0
        self.consecutive_losses = 0
        self.position_changes = []
        self.training_step = 0
        self.episode_step_count = 0
        self.equity_curve = [1.0]
        self.trade_history = {'count': 0, 'wins': [], 'losses': []}

    def execute_trade(self, action, current_price, current_idx=None):
        """
        [레버리지 시스템] TD3 Action (-1~1)을 레버리지 포지션으로 변환
        
        Args:
            action: TD3 출력값 (-1 ~ 1)
                   - 부호: 방향 (양수=Long, 음수=Short)
                   - 절댓값: 레버리지 강도 (0~1)
            current_price: 현재 가격
            current_idx: 현재 인덱스 (선택)
        
        Returns:
            (entry_price, effective_leverage, trade_executed)
        """
        # 1. 레버리지 계산
        # action의 절댓값이 '비중'이자 '레버리지 강도'
        # 예: action 0.8 * LEVERAGE 20 = 16배 레버리지
        max_leverage = getattr(config, 'LEVERAGE', 20)
        target_leverage = abs(action) * max_leverage
        
        # 2. 최소 레버리지 필터링 (너무 작으면 거래 안 함 → 수수료 방어)
        if target_leverage < 1.0:
            return 0.0, 0.0, False  # 관망
        
        # 3. 진입가 및 수수료 계산
        entry_price = current_price
        
        # 수수료는 레버리지 쓴 전체 금액에 대해 부과됨 (치명적!)
        # fee = (자산 * 레버리지) * 수수료율
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
        transaction_cost = target_leverage * fee_rate
        
        # 4. 즉시 자산 차감 (진입 수수료)
        if len(self.equity_curve) > 0:
            self.equity_curve[-1] *= (1 - transaction_cost)
        
        return entry_price, target_leverage, True

    def precompute_data(self):
        """
        [근본 해결] Full GPU Caching + Strategy Signal Generation
        """
        if self.collector.eth_data is None:
            return

        logger.info(f"⚡ 데이터 전처리 및 캐싱 시작 (Device: {self.device})...")
        df = self.collector.eth_data.copy()

        # 1. Feature Handling
        target_cols = list(ULTIMATE_FEATURE_COLS)

        # Missing columns fill
        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = 0.0

        data = df[target_cols]

        # 2. Rolling Normalization (CPU)
        # FeatureEngineer에서 Z-score가 된 것도 있고 안 된 것도 있어서,
        # 학습 안정성을 위해 전체적으로 한 번 더 정규화합니다.
        rolling = data.rolling(window=self.lookback, min_periods=1)
        roll_mean = rolling.mean()
        roll_std = rolling.std().replace(0, 1e-8)

        normalized_df = (data - roll_mean) / roll_std
        normalized_df = normalized_df.fillna(0).replace([np.inf, -np.inf], 0)

        # To GPU
        self.cached_features = torch.tensor(
            normalized_df.values, dtype=torch.float32, device=self.device
        )
        self.scaler_fitted = True

        # 3. Strategy Signal Generation (Elite 8)
        # 데이터프레임에 strategy_0 등 컬럼이 없으면 직접 계산
        logger.info("   👉 전략 신호 생성 중 (Elite 8 Strategies)...")

        strat_signals = []

        # 전략별로 컬럼이 이미 있는지 확인
        cols_exist = all(f'strategy_{i}' in df.columns for i in range(len(self.strategies)))

        if cols_exist:
            logger.info("   ✅ 기존 전략 신호 사용")
            for i in range(len(self.strategies)):
                strat_signals.append(df[f'strategy_{i}'].values)
        else:
            logger.info("   ⚠️ 전략 신호 신규 계산 (시간이 조금 소요될 수 있음)")
            temp_signals = np.zeros((len(df), len(self.strategies)))

            for i, strategy in enumerate(self.strategies):
                sigs = []
                for idx, row in df.iterrows():
                    sig = strategy.generate_signal(row, df)
                    sigs.append(sig)
                temp_signals[:, i] = sigs

            strat_signals = [temp_signals[:, j] for j in range(len(self.strategies))]

        strat_array = np.stack(strat_signals, axis=1)  # (T, 8)
        self.cached_strategies = torch.tensor(
            strat_array, dtype=torch.float32, device=self.device
        )

        logger.info(f"✅ GPU 캐싱 완료: Features {self.cached_features.shape}, Strat {self.cached_strategies.shape}")

    def update_trading_metrics(self, prev_position, current_position,
                                strategy_scores=None, volatility_pred=None,
                                actual_volatility=None):
        self.position_changes.append(1.0 if prev_position != current_position else 0.0)
        if len(self.position_changes) > 100:
            self.position_changes = self.position_changes[-100:]
        self.training_step += 1

    def get_observation(self, position_info=None, current_index=None):
        """
        [최적화] GPU 텐서 슬라이싱
        """
        if self.cached_features is None:
            self.precompute_data()

        curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)

        if curr_idx is None or curr_idx < self.lookback or curr_idx >= len(self.cached_features):
            return None

        # 1. Features
        obs_seq = self.cached_features[curr_idx - self.lookback : curr_idx].unsqueeze(0)

        # 2. Strategies
        scores = self.cached_strategies[curr_idx]  # (8,)

        # 3. Position Info
        if position_info is None:
            pos_tensor = torch.zeros(3, device=self.device)
        elif isinstance(position_info, torch.Tensor):
            pos_tensor = position_info.to(self.device)
        else:
            pos_tensor = torch.tensor(position_info, dtype=torch.float32, device=self.device)

        # 4. Combine: pos[0](Val) + scores(8) + pos[1:](Meta) = 11
        obs_info = torch.cat([pos_tensor[0:1], scores, pos_tensor[1:]]).unsqueeze(0)

        return (obs_seq, obs_info)

    def calculate_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
        """
        [긴급 처방 2 + 레버리지] TD3 Rescue Reward - 레버리지 인식 보상
        
        Args:
            step_pnl: Step PnL (레버리지 적용된 ROE)
            effective_leverage: 실제 적용된 레버리지
        """
        reward = 0.0
        
        # 1. 평가 손익(Unrealized PnL) 반영 - 긍정 강화
        if step_pnl > 0:
            reward += step_pnl * 100.0  # 수익 = 강력한 칭찬!
        else:
            reward += step_pnl * 50.0   # 손실 = 적당한 실망
        
        # 2. 실현 손익(Realized PnL) 보너스 - 거래 완료 시
        if trade_done:
            if realized_pnl > 0:
                reward += 5.0  # 익절 성공 = 간식 투척! 🍖
            elif realized_pnl < 0:
                reward -= 2.0  # 손절 = 살짝만 아프게
        
        # 3. 포지션 유지 보너스 (Trend Following 유도)
        if current_position is not None and current_position != 0:
            if step_pnl > 0:
                reward += 0.1  # 수익 중에 홀딩 = 인내심 보너스
        
        # 4. [레버리지 시스템] 횡보 페널티
        # 레버리지를 썼는데 횡보하면 수수료만 날아감
        # 고레버리지(5배 이상) + 미미한 수익(0.5% 이하) = 페널티
        if effective_leverage > 5 and abs(step_pnl) < 0.005:
            reward -= 0.5  # "변동성도 없는데 고레버리지 쓰지 마라"
        
        # 5. 큰 손실 경고
        if step_pnl < -0.02:  # ROE -2% 이상
            reward -= 1.0
        
        # 6. 클리핑 제거! 대박이 나면 대박 점수를 그대로
        return float(reward)


    def get_state_dim(self):
        if self.cached_features is not None:
            return self.cached_features.shape[1]
        return len(ULTIMATE_FEATURE_COLS)

    def get_current_equity(self):
        return self.equity_curve[-1] if self.equity_curve else 1.0
