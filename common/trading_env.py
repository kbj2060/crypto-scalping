"""
공통 거래 환경 (TD3 / MacroHFT 공용) - Optimized for Elite 8
- DataCollector + strategies 기반 관측(obs_seq, obs_info) 생성
- [Optimization] Full GPU Caching & Auto-Signal Generation
- execute_trade: 진입/청산 수수료 일관성 개선 (레버리지 비례)
- _load_features: 제거 (PPOTrainer에서 이미 처리, 환경은 데이터만 사용)
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
        self.strategies = strategies
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.scaler_fitted = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cached_features = None
        self.cached_strategies = None

        self.initial_balance = getattr(config, 'EVAL_INITIAL_CAPITAL', 10000.0)
        self.reset_reward_states()

    # ------------------------------------------------------------------
    # 청산 조건 검사 (변경 없음)
    # ------------------------------------------------------------------
    def check_exit_conditions(self, unrealized_pnl_roe, holding_time_steps=0):
        liquidation_threshold = getattr(config, 'LIQUIDATION_THRESHOLD', -0.80)
        if unrealized_pnl_roe <= liquidation_threshold:
            return True, "LIQUIDATION"

        take_profit_threshold = getattr(config, 'TAKE_PROFIT_THRESHOLD', 0.50)
        if unrealized_pnl_roe >= take_profit_threshold:
            return True, "TAKE_PROFIT"

        stop_loss_threshold = getattr(config, 'STOP_LOSS_THRESHOLD', -0.20)
        if unrealized_pnl_roe <= stop_loss_threshold:
            return True, "STOP_LOSS"

        if holding_time_steps > 100 and abs(unrealized_pnl_roe) < 0.01:
            return True, "TIME_STOP"

        return False, None

    # ------------------------------------------------------------------
    # 에피소드 상태 초기화 (변경 없음)
    # ------------------------------------------------------------------
    def reset_reward_states(self):
        self.trade_count = 0
        self.step_pnl_ema = 0.0
        self.consecutive_losses = 0
        self.position_changes = []
        self.training_step = 0
        self.episode_step_count = 0
        self.equity_curve = [1.0]
        self.trade_history = {'count': 0, 'wins': [], 'losses': []}

    def calculate_position_size(self, action_scale, current_price, balance, volatility=None):
        """
        동적 포지션 사이징: Actor의 scale(0~1)을 실제 레버리지/포지션 금액/계약 수로 변환
        
        Args:
            action_scale: Actor 출력 (0~1) – 위험 예산 사용 비율
            current_price: 현재 가격
            balance: 현재 자본금 (USDT)
            volatility: 최근 변동성 (표준편차, 틱 단위)
        
        Returns:
            target_leverage: 실제 적용할 레버리지 배수
            position_value: 포지션 금액 (USDT)
            contracts: 계약 수 (선물)
        """
        max_leverage = getattr(config, 'RISK_MAX_LEVERAGE', 20)
        risk_target_vol = getattr(config, 'RISK_TARGET_VOL', 0.15)
        
        # 1. 기본 레버리지 = Actor 출력 * 최대 레버리지
        base_leverage = action_scale * max_leverage
        base_leverage = max(1.0, base_leverage)  # 최소 1배
        
        # 2. 변동성 기반 레버리지 조절 (목표 변동성 / 현재 변동성)
        if volatility is not None and volatility > 0:
            # 틱 단위 변동성을 연간 변동성으로 환산 (3분봉 기준)
            # 연간 틱 수 = 365일 * 24시간 * 60분 / 3분 = 175,200
            vol_annual = volatility * np.sqrt(365 * 24 * 60 / 3)
            vol_adjustment = risk_target_vol / max(vol_annual, 0.01)
            vol_adjustment = np.clip(
                vol_adjustment,
                getattr(config, 'RISK_VOL_ADJUSTMENT_MIN', 0.5),
                getattr(config, 'RISK_VOL_ADJUSTMENT_MAX', 2.0)
            )
        else:
            vol_adjustment = 1.0
        
        target_leverage = base_leverage * vol_adjustment
        target_leverage = np.clip(target_leverage, 1.0, max_leverage)
        
        # 3. 자본금 기반 포지션 금액
        position_value = balance * target_leverage
        
        # 4. 계약 수 (USDT 무기한 선물 가정)
        contracts = position_value / current_price
        
        return target_leverage, position_value, contracts
        
    # ------------------------------------------------------------------
    # [개선] 거래 실행 – 진입/청산 수수료 일관성 및 방향 처리
    # ------------------------------------------------------------------
    def execute_trade(self, action, current_price, direction=None, balance=None,
                  volatility=None, is_exit=False, leverage=None):
        """
        통합 거래 실행 함수 (동적 사이징 적용)
        
        Args:
            action: 레버리지 비율 (0~1) – 진입 시에만 사용
            current_price: 현재 가격
            direction: 진입 방향 (1=LONG, -1=SHORT) – 진입 시 필수, 청산 시 무시
            balance: 현재 자본금 (USDT) – 진입 시 필수, 청산 시 무시
            volatility: 현재 변동성 (선택, 없으면 1.0)
            is_exit: 청산 여부
            leverage: 청산 시 현재 포지션의 레버리지 (is_exit=True일 때 필수)
        
        Returns:
            entry_price: 진입가 (청산 시 None)
            target_leverage: 적용 레버리지 (청산 시 0.0)
            executed: 실행 여부
            cost: 발생한 수수료 (레버리지 * fee_rate)
            position_value: 포지션 금액 (USDT, 진입 시)
            contracts: 계약 수 (진입 시)
        """
        max_leverage = getattr(config, 'MAX_LEVERAGE', 20)
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)

        # ---------- 청산 ----------
        if is_exit:
            if leverage is None:
                raise ValueError("청산 시 현재 레버리지(leverage)를 반드시 전달해야 합니다.")
            cost = leverage * fee_rate
            return None, 0.0, True, cost, 0.0, 0

        # ---------- 진입 ----------
        if direction is None:
            raise ValueError("진입 시 방향(direction)을 반드시 전달해야 합니다.")
        if balance is None:
            raise ValueError("진입 시 자본금(balance)을 반드시 전달해야 합니다.")
        if action <= 0:
            return 0.0, 0.0, False, 0.0, 0.0, 0

        # 동적 포지션 사이징
        target_leverage, position_value, contracts = self.calculate_position_size(
            action, current_price, balance, volatility
        )

        # 슬리피지
        slippage = np.random.uniform(0.0001, 0.0005)
        if direction == 1:
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)

        # 진입 수수료
        cost = target_leverage * fee_rate

        return entry_price, target_leverage, True, cost, position_value, contracts

    # ------------------------------------------------------------------
    # GPU 캐싱 (cached_strategies.csv 우선 로드)
    # ------------------------------------------------------------------
    def precompute_data(self):
        """GPU 캐싱 + 전략 신호 로드 (CSV 우선, 없으면 계산)"""
        if self.collector.eth_data is None:
            return

        logger.info(f"⚡ 데이터 전처리 및 캐싱 시작 (Device: {self.device})...")
        df = self.collector.eth_data.copy()

        # ---------- Feature 정규화 ----------
        target_cols = list(ULTIMATE_FEATURE_COLS)
        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = 0.0

        data = df[target_cols]
        rolling = data.rolling(window=self.lookback, min_periods=1)
        roll_mean = rolling.mean()
        roll_std = rolling.std().replace(0, 1e-8)
        normalized_df = (data - roll_mean) / roll_std
        normalized_df = normalized_df.fillna(0).replace([np.inf, -np.inf], 0)

        self.cached_features = torch.tensor(
            normalized_df.values, dtype=torch.float32, device=self.device
        )
        self.scaler_fitted = True

        # ---------- 전략 신호 (Elite 8) ----------
        logger.info("   👉 전략 신호 로드 중...")
        strategy_cols_exist = all(f'strategy_{i}' in df.columns for i in range(len(self.strategies)))

        if strategy_cols_exist:
            logger.info("   ✅ 기존 전략 컬럼 사용 (CSV에서 로드됨)")
            strat_array = df[[f'strategy_{i}' for i in range(len(self.strategies))]].values
        else:
            logger.info("   ⚠️ 전략 컬럼 없음 → 신규 계산 (시간 소요)")
            strat_array = np.zeros((len(df), len(self.strategies)), dtype=np.float32)
            for i, strategy in enumerate(self.strategies):
                sigs = []
                for idx, row in df.iterrows():
                    sig = strategy.generate_signal(row, df)
                    sigs.append(sig)
                strat_array[:, i] = sigs

            strat_df = pd.DataFrame(
                strat_array,
                columns=[f'strategy_{i}' for i in range(len(self.strategies))],
                index=df.index
            )
            strat_df.to_csv('data/cached_strategies.csv')
            logger.info(f"   💾 전략 신호 저장 완료: data/cached_strategies.csv")

        self.cached_strategies = torch.tensor(
            strat_array, dtype=torch.float32, device=self.device
        )
        logger.info(f"✅ GPU 캐싱 완료: Features {self.cached_features.shape}, Strat {self.cached_strategies.shape}")

    # ------------------------------------------------------------------
    # 거래 메트릭 업데이트 (변경 없음)
    # ------------------------------------------------------------------
    def update_trading_metrics(self, prev_position, current_position,
                               strategy_scores=None, volatility_pred=None,
                               actual_volatility=None):
        self.position_changes.append(1.0 if prev_position != current_position else 0.0)
        if len(self.position_changes) > 100:
            self.position_changes = self.position_changes[-100:]
        self.training_step += 1

    # ------------------------------------------------------------------
    # 관측 생성 (GPU 텐서)
    # ------------------------------------------------------------------
    def get_observation(self, position_info=None, current_index=None):
        if self.cached_features is None:
            self.precompute_data()

        curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)

        if curr_idx is None or curr_idx < self.lookback or curr_idx >= len(self.cached_features):
            return None

        obs_seq = self.cached_features[curr_idx - self.lookback: curr_idx].unsqueeze(0)
        scores = self.cached_strategies[curr_idx]  # (8,)

        if position_info is None:
            pos_tensor = torch.zeros(3, device=self.device)
        elif isinstance(position_info, torch.Tensor):
            pos_tensor = position_info.to(self.device)
        else:
            pos_tensor = torch.tensor(position_info, dtype=torch.float32, device=self.device)

        obs_info = torch.cat([pos_tensor[0:1], scores, pos_tensor[1:]]).unsqueeze(0)
        return (obs_seq, obs_info)

    # ------------------------------------------------------------------
    # 상태 차원 반환
    # ------------------------------------------------------------------
    def get_state_dim(self):
        if self.cached_features is not None:
            return self.cached_features.shape[1]
        return len(ULTIMATE_FEATURE_COLS)

    # ------------------------------------------------------------------
    # 현재 자본 조회 (더미, train_ppoe에서 balance 직접 관리)
    # ------------------------------------------------------------------
    def get_current_equity(self):
        return self.equity_curve[-1] if self.equity_curve else 1.0