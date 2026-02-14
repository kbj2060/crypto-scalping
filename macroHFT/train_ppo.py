"""
MacroHFT v4.5 Training Script – FINAL FIX (Reward Scale + ICM Coef)
=====================================================================
- 캐싱된 전략 CSV 우선 로드
- 동적 포지션 사이징 + 수수료 완전 반영
- ICM 탐색 보상 (내재적 호기심, 계수 0.01)
- CVaR 리스크 회귀 (ppo_agent에서 처리)
- 샤프 비율 보너스 (스케일 5)
- 리워드 스케일 10 (1% = 1점)
- 청산 시 step_pnl_roe 중복 제거
"""
import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import INFO_DIM_ELITE8, TradingEnvironment
from macroHFT.ppo_agent import PPOAgent
from macroHFT.icm import ICM

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo_v4.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 성능 최적화
if config.USE_CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True
if config.USE_HIGH_MATMUL_PRECISION:
    torch.set_float32_matmul_precision('high')


class PPOTrainer:
    def __init__(self, enable_visualization=False):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        logger.info(f"Elite 8 Strategies Loaded ({len(self.strategies)})")

        # ---------- 데이터 로드 및 캐싱 ----------
        self._load_features()
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._prepare_data_splits()
        self.env.precompute_data()

        # ---------- 에이전트 초기화 ----------
        state_dim = self.env.get_state_dim()
        action_dim = config.ACTION_DIM
        info_dim = INFO_DIM_ELITE8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device} | State Dim: {state_dim} | Action Dim: {action_dim}")

        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=self.device)

        # ---------- NumPy 캐싱 ----------
        df = self.data_collector.eth_data
        self.close_prices = df['close'].values.astype(np.float32)
        self.volatility_data = df.get('volatility_z', np.zeros(len(df))).values.astype(np.float32)
        self.strategy_matrix = df[[f'strategy_{i}' for i in range(8)]].values.astype(np.float32)
        self.last_trade_step = -100

        # ---------- TensorBoard ----------
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S_MacroHFTv4')
        tb_log_dir = os.path.join('logs', 'tensorboard', run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # ---------- 커리큘럼 인덱스 ----------
        self._prepare_curriculum_indices()

        # ---------- Out-of-time 평가 환경 ----------
        self._prepare_oot_test_env()

        # ========== [ICM] 탐색 모듈 초기화 ==========
        self.use_icm = True
        self.icm_coef = 0.001            # 🔥 0.1 → 0.01 → 0.05
        icm_state_dim = state_dim + info_dim
        self.icm = ICM(
            state_dim=icm_state_dim,
            action_dim=2,
            hidden_dim=256,
            device=self.device
        )
        self.icm_optimizer = torch.optim.AdamW(self.icm.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # 데이터 로드 (cached_strategies.csv 우선)
    # ------------------------------------------------------------------
    def _load_features(self):
        path = 'data/training_features.csv'
        if not os.path.exists(path):
            logger.error("Feature file missing. Run feature engineering first.")
            sys.exit(1)

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.ffill().bfill()

        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(cached_strategies_path):
            try:
                cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                if strategy_cols:
                    for col in strategy_cols:
                        df[col] = cached_df[col]
                    logger.info(f"✅ Cached strategies loaded from {cached_strategies_path} ({len(strategy_cols)} cols)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cached strategies: {e}")

        self.data_collector.eth_data = df
        logger.info(f"Features loaded: {df.shape}")

    def _prepare_data_splits(self):
        df = self.data_collector.eth_data
        total = len(df)
        train_end = int(total * config.TRAIN_SPLIT)
        val_end = int(total * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        self.train_end_idx = train_end
        self.val_end_idx = val_end
        self.test_end_idx = total
        logger.info(f"Train: 0:{train_end}, Val: {train_end}:{val_end}, Test: {val_end}:{total}")

    def _prepare_curriculum_indices(self):
        df = self.data_collector.eth_data.iloc[:self.train_end_idx]
        valid = list(range(config.LOOKBACK + 50, self.train_end_idx - 50))
        self.all_indices = valid

        if 'chop_index' in df.columns and 'volatility_z' in df.columns:
            chop = df['chop_index'].values
            vol = df['volatility_z'].values
            self.indices_trend = [i for i in valid if i < len(chop) and chop[i] < 45.0] or valid
            vol_thresh = np.percentile(vol[valid], 75) if len(vol[valid]) > 0 else 1.0
            self.indices_vol = [i for i in valid if i < len(vol) and vol[i] > vol_thresh] or valid
            vol_mean = vol[valid].mean() if len(vol[valid]) > 0 else 0.0
            self.indices_sideways = [i for i in valid if i < len(chop) and chop[i] > 50.0 and vol[i] < vol_mean] or valid
        else:
            self.indices_trend = self.indices_vol = self.indices_sideways = valid
        self.idx_map = [self.indices_trend, self.indices_vol, self.indices_sideways]
        logger.info(f"Curriculum: T={len(self.indices_trend)}, V={len(self.indices_vol)}, S={len(self.indices_sideways)}")

    def _prepare_oot_test_env(self):
        if hasattr(self, 'val_end_idx') and self.val_end_idx < self.test_end_idx:
            logger.info("Building Out-of-time test environment...")
            test_collector = DataCollector(use_saved_data=True)
            test_collector.eth_data = self.data_collector.eth_data.iloc[self.val_end_idx:self.test_end_idx].copy()
            self.test_env = TradingEnvironment(test_collector, self.strategies)
            self.test_env.precompute_data()
            logger.info("OOT test environment ready.")

    # ------------------------------------------------------------------
    # 액션 마스킹
    # ------------------------------------------------------------------
    def get_action_mask(self, current_position):
        """
        액션 마스킹: 9개 행동 각각 허용/금지
        - 인덱스 0~2: HOLD + 레버리지 1,5,10
        - 인덱스 3~5: LONG + 레버리지 1,5,10
        - 인덱스 6~8: SHORT + 레버리지 1,5,10
        """
        mask = np.ones(9, dtype=np.float32)  # 🔥 15 → 9
        if current_position == 'LONG':
            # LONG 관련 행동 금지 (인덱스 3~5)
            mask[3:6] = 0.0
        elif current_position == 'SHORT':
            # SHORT 관련 행동 금지 (인덱스 6~8)
            mask[6:9] = 0.0
        # HOLD는 항상 허용 (0~2)
        return mask

    # ------------------------------------------------------------------
    # 샤프 비율 보너스 (스케일 5)
    # ------------------------------------------------------------------
    def compute_sharpe_reward(self, equity_curve, risk_free_rate=0.0):
        if len(equity_curve) < 10:
            return 0.0
        equity = np.array(equity_curve, dtype=np.float32)
        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        if len(returns) < 2 or np.std(returns) < 1e-8:
            return 0.0
        sharpe = (returns.mean() - risk_free_rate) / (returns.std() + 1e-8)
        sharpe_annual = sharpe * np.sqrt(365 * 24 * 60 / 3)
        reward_bonus = 5.0 * np.tanh(sharpe_annual / 5.0)
        return float(reward_bonus)

    # ------------------------------------------------------------------
    # 리워드 함수 v5.1 (중복 제거, 스케일 10)
    # ------------------------------------------------------------------
    def compute_reward(self, expert_type, step_pnl_roe, realized_pnl_roe, trade_done,
                   holding_time, current_position, effective_leverage):
        if not trade_done:
            return 0.0
        
        # 🔥 로그 수익률: 과도한 양수 리워드 억제, 손실은 더 깊게 반영
        log_return = np.log1p(realized_pnl_roe)  # log(1 + total_trade_return)
        reward = log_return * config.REWARD_PNL_SCALE
        
        # 🔥 레버리지 패널티 (유지)
        if effective_leverage > 0:
            reward += config.REWARD_LEVERAGE_PENALTY * effective_leverage
        
        # 🔥 거래 패널티
        reward += config.REWARD_TRADE_PENALTY
        
        # 🔥 전문가 보너스 (미미한 수준)
        if expert_type == 'trend' and realized_pnl_roe > 0:
            reward += holding_time * config.REWARD_TREND_HOLDING_BONUS
        elif expert_type == 'volatility':
            reward += config.REWARD_VOLATILITY_BONUS
        elif expert_type == 'sideways':
            if realized_pnl_roe > 0:
                reward += config.REWARD_SIDEWAYS_WIN_BONUS
                if realized_pnl_roe < 0.001:
                    reward += config.REWARD_SIDEWAYS_SMALL_BONUS
            else:
                reward += config.REWARD_SIDEWAYS_LOSS_PENALTY
        
        # 🔥 소프트 클리핑
        reward = config.REWARD_CLIP_SCALE * np.tanh(reward / config.REWARD_CLIP_SCALE)
        return float(reward)

    def validate_oot(self, episode):
        """완전히 분리된 테스트 환경에서 에이전트 평가 (3일치 데이터, 여러 윈도우)"""
        if not hasattr(self, 'test_env'):
            logger.warning("OOT test environment not available.")
            return 0.0, 0.0, 0.0

        test_env = self.test_env
        test_env.reset_reward_states()
        
        # 테스트 데이터 전체 길이
        total_steps = len(test_env.cached_features)
        # 시작 인덱스 (LOOKBACK 이후)
        start_idx = config.LOOKBACK + 50
        
        # 3일치 데이터로 제한 (1440 스텝 = 3일 * 24시간 * 60분 / 3분)
        max_test_steps = 1440
        end_idx = min(total_steps, start_idx + max_test_steps)
        if end_idx - start_idx < config.TRAIN_MAX_STEPS_PER_EPISODE:
            logger.warning("Not enough test data for a full window.")
            return 0.0, 0.0, 0.0
        
        # 윈도우 크기 (기존 에피소드 길이와 동일)
        window_size = config.TRAIN_MAX_STEPS_PER_EPISODE  # 480
        # 윈도우 개수 계산 (3일치 데이터 내에서)
        max_windows = (end_idx - start_idx) // window_size
        
        # 각 윈도우의 결과 저장
        window_returns = []
        window_mdds = []
        
        for w in range(max_windows):
            # 윈도우 시작 인덱스
            win_start = start_idx + w * window_size
            test_env.current_index = win_start
            test_env.reset_reward_states()
            
            # 에피소드 초기화
            position = None
            entry_price = 0.0
            effective_leverage = 0.0
            balance = config.EVAL_INITIAL_CAPITAL
            initial_balance = balance
            equity_curve = [balance]
            entry_balance = 0.0
            entry_cost = 0.0
            
            for step in range(window_size):
                idx = test_env.current_index
                if idx >= end_idx - 1:
                    break
                curr_price = test_env.collector.eth_data.iloc[idx]['close']
                
                # 미실현 손익
                if position == 'LONG':
                    unrealized_return = (curr_price - entry_price) / entry_price
                elif position == 'SHORT':
                    unrealized_return = (entry_price - curr_price) / entry_price
                else:
                    unrealized_return = 0.0
                
                pos_val = 1.0 if position == 'LONG' else (-1.0 if position == 'SHORT' else 0.0)
                pos_info = [pos_val, unrealized_return, 0.0]
                state = test_env.get_observation(position_info=pos_info, current_index=idx)
                if state is None:
                    break
                
                action, _, _, _, _, _ = self.agent.select_action(state, deterministic=True)
                direction, scale = action
                
                if direction != 0 and scale >= config.MIN_LEVERAGE / config.MAX_LEVERAGE:
                    if position is None:
                        exec_price, eff_lev, executed, cost, _, _ = test_env.execute_trade(
                            action=scale,
                            current_price=curr_price,
                            direction=1 if direction == 1 else -1,
                            balance=balance,
                            volatility=self.volatility_data[idx] if hasattr(self, 'volatility_data') else None,
                            is_exit=False
                        )
                        if executed:
                            position = 'LONG' if direction == 1 else 'SHORT'
                            entry_price = exec_price
                            effective_leverage = eff_lev
                            entry_cost = cost
                            entry_balance = balance
                            balance *= (1 - entry_cost)
                    
                    elif (direction == 1 and position == 'SHORT') or (direction == 2 and position == 'LONG'):
                        _, _, _, exit_cost, _, _ = test_env.execute_trade(
                            action=scale,
                            current_price=curr_price,
                            is_exit=True,
                            leverage=effective_leverage
                        )
                        realized_return = unrealized_return
                        total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                        total_trade_return = np.clip(total_trade_return, -0.95, 5.0)
                        balance = entry_balance * (1 + total_trade_return)
                        equity_curve.append(balance)
                        position = None
                        effective_leverage = 0.0
                        entry_cost = 0.0
                        entry_balance = 0.0
                
                test_env.current_index += 1
            
            # 에피소드 종료 시 미청산 포지션 처리
            if position is not None:
                final_price = test_env.collector.eth_data.iloc[min(test_env.current_index, end_idx-1)]['close']
                if position == 'LONG':
                    realized_return = (final_price - entry_price) / entry_price
                else:
                    realized_return = (entry_price - final_price) / entry_price
                _, _, _, exit_cost, _, _ = test_env.execute_trade(
                    action=0.0,
                    current_price=final_price,
                    is_exit=True,
                    leverage=effective_leverage
                )
                total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                total_trade_return = np.clip(total_trade_return, -0.95, 5.0)
                balance = entry_balance * (1 + total_trade_return)
                equity_curve.append(balance)
            
            # 윈도우 수익률
            window_return = (balance / initial_balance) - 1.0
            window_returns.append(window_return)
            
            # MDD 계산
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            window_mdd = np.max(drawdown) if len(drawdown) > 0 else 0.0
            window_mdds.append(window_mdd)
        
        # 전체 통계
        avg_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        avg_mdd = np.mean(window_mdds)
        
        # 연간화 Sharpe 비율 (윈도우 개수에 따라 조정)
        # 각 윈도우 길이는 window_size 스텝 = 1일 (480스텝 = 24시간)
        days_per_window = 1.0  # 480스텝 = 1일
        num_years = max_windows * days_per_window / 365.0
        
        if num_years > 0 and std_return > 1e-8:
            sharpe = (avg_return / std_return) * np.sqrt(365 / days_per_window)  # 연간화
        else:
            sharpe = 0.0
        
        # 로깅
        self.writer.add_scalar('OOT/Return_avg', avg_return, episode)
        self.writer.add_scalar('OOT/Sharpe', sharpe, episode)
        self.writer.add_scalar('OOT/MDD_avg', avg_mdd, episode)
        
        logger.info(f"OOT (3일치) | Avg Return: {avg_return:.2%} | Sharpe: {sharpe:.2f} | Avg MDD: {avg_mdd:.2%} (windows: {max_windows})")
        return avg_return, sharpe, avg_mdd
        

    # ------------------------------------------------------------------
    # 에피소드 실행 (수수료 완전 반영 + ICM 통합)
    # ------------------------------------------------------------------
    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE

        # ---------- 모드 결정 ----------
        cycle = (episode_num - 1) % 2
        if cycle == 0:
            mode = 'router'
            current_key = 'router'
            target_indices = self.all_indices
            expert_idx = 0
        else:
            expert_idx = ((episode_num - 1) // 2) % 3
            mode = 'expert'
            current_key = self.agent.expert_names[expert_idx]
            target_indices = self.idx_map[expert_idx]

        start_idx = np.random.choice(target_indices)
        self.data_collector.current_index = start_idx
        self.env.reset_reward_states()

        # ---------- 거래 제한 제거 ----------
        # self.last_trade_step = -100

        # ---------- 자본금 및 포지션 초기화 ----------
        initial_balance = config.EVAL_INITIAL_CAPITAL
        self.balance = initial_balance
        self.equity_curve = [self.balance]
        position = None
        entry_price = 0.0
        effective_leverage = 0.0
        holding_steps = 0
        entry_cost = 0.0
        entry_balance = 0.0
        self.position_value = 0.0
        self.contracts = 0.0

        episode_reward = 0.0
        trade_count = 0
        expert_selection_counts = [0, 0, 0]

        self._prev_unrealized_roe = 0.0
        last_action_scale = 0.0

        # ---------- ICM 상태 저장 ----------
        prev_state_vec = None
        prev_direction = None
        prev_scale = None

        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num} [{current_key.upper()}]", leave=False)

        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx - 1:
                break

            curr_price = self.close_prices[current_idx]

            # ---------- 미실현 손익 ----------
            if position == 'LONG':
                unrealized_return = (curr_price - entry_price) / entry_price
            elif position == 'SHORT':
                unrealized_return = (entry_price - curr_price) / entry_price
            else:
                unrealized_return = 0.0
            unrealized_pnl_roe = unrealized_return * effective_leverage if position else 0.0

            # ---------- 스텝 PnL (미사용) ----------
            if position is None:
                step_pnl_roe = 0.0
                self._prev_unrealized_roe = 0.0
            else:
                if step == 0:
                    step_pnl_roe = 0.0
                else:
                    prev_unrealized = getattr(self, '_prev_unrealized_roe', 0.0)
                    step_pnl_roe = unrealized_pnl_roe - prev_unrealized
                self._prev_unrealized_roe = unrealized_pnl_roe

            # ---------- 포지션 정보 텐서 ----------
            pos_val = 1.0 if position == 'LONG' else (-1.0 if position == 'SHORT' else 0.0)
            pos_info = [pos_val, unrealized_return, holding_steps / max_steps]
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None:
                break

            # ---------- 액션 마스크 ----------
            action_mask = self.get_action_mask(current_position=position)

            # ---------- 행동 선택 ----------
            action, log_prob, value, selected_expert, router_log_prob, router_value = self.agent.select_action(
                state, action_mask=action_mask, mode=mode, expert_idx=expert_idx
            )
            direction, scale = action
            last_action_scale = scale
            expert_selection_counts[selected_expert] += 1

            # ---------- ICM 탐색 보상 ----------
            intrinsic_reward = 0.0
            if self.use_icm and prev_state_vec is not None:
                obs_seq, obs_info = state
                s2_vec = torch.cat([
                    obs_seq[0, -1, :].detach(),
                    obs_info[0, :].detach()
                ], dim=-1).to(self.device)

                forward_loss, inverse_loss, intrinsic = self.icm(
                    prev_state_vec.unsqueeze(0),
                    s2_vec.unsqueeze(0),
                    torch.tensor([prev_direction], device=self.device),
                    torch.tensor([prev_scale], device=self.device)
                )

                icm_loss = forward_loss + inverse_loss
                self.icm_optimizer.zero_grad()
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
                self.icm_optimizer.step()

                intrinsic_reward = intrinsic.item() * self.icm_coef
                self.writer.add_scalar('ICM/ForwardLoss', forward_loss.item(), episode_num * max_steps + step)
                self.writer.add_scalar('ICM/InverseLoss', inverse_loss.item(), episode_num * max_steps + step)
                self.writer.add_scalar('ICM/IntrinsicReward', intrinsic_reward, episode_num * max_steps + step)

            # ---------- 거래 실행 ----------
            trade_done = False
            realized_pnl_roe = 0.0
            realized_return = 0.0

            if direction != 0 and scale >= config.MIN_LEVERAGE / config.MAX_LEVERAGE:
                # ----- 진입 -----
                if position is None:
                    entry_price, eff_lev, executed, cost, position_value, contracts = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        direction=1 if direction == 1 else -1,
                        balance=self.balance,
                        volatility=self.volatility_data[current_idx] if hasattr(self, 'volatility_data') else None,
                        is_exit=False
                    )
                    if executed:
                        entry_balance = self.balance
                        position = 'LONG' if direction == 1 else 'SHORT'
                        effective_leverage = eff_lev
                        entry_cost = cost
                        self.position_value = position_value
                        self.contracts = contracts
                        holding_steps = 0
                        trade_count += 1
                        self.balance *= (1 - entry_cost)

                # ----- 청산 (반대 방향) -----
                elif (direction == 1 and position == 'SHORT') or (direction == 2 and position == 'LONG'):
                    _, _, _, exit_cost, _, _ = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        is_exit=True,
                        leverage=effective_leverage
                    )
                    realized_return = unrealized_return
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1

                    # 🔥 total_trade_return 클리핑 (과도한 변동 억제)
                    total_trade_return = np.clip(total_trade_return, -0.95, 5.0)

                    self.balance = entry_balance * (1 + total_trade_return)
                    self.equity_curve.append(self.balance)
                    realized_pnl_roe = total_trade_return
                    trade_done = True

                    position = None
                    effective_leverage = 0.0
                    entry_cost = 0.0
                    entry_balance = 0.0
                    self.position_value = 0.0
                    self.contracts = 0.0
                    holding_steps = 0
                    self._prev_unrealized_roe = 0.0

            # ---------- 보유 중 ----------
            if position is not None:
                holding_steps += 1

            # ---------- 강제 청산 ----------
            if position is not None:
                should_exit, reason = self.env.check_exit_conditions(unrealized_pnl_roe, holding_steps)
                if should_exit:
                    _, _, _, exit_cost, _, _ = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        is_exit=True,
                        leverage=effective_leverage
                    )
                    realized_return = unrealized_return
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1

                    # 🔥 total_trade_return 클리핑
                    total_trade_return = np.clip(total_trade_return, -0.95, 5.0)

                    self.balance = entry_balance * (1 + total_trade_return)
                    self.equity_curve.append(self.balance)
                    entry_balance = 0.0
                    entry_cost = 0.0
                    realized_pnl_roe = total_trade_return
                    trade_done = True

                    position = None
                    effective_leverage = 0.0
                    self.position_value = 0.0
                    self.contracts = 0.0
                    holding_steps = 0
                    self._prev_unrealized_roe = 0.0
                    pbar.set_postfix({'Exit': reason[:4]})

            # ---------- 외부 보상 계산 ----------
            if mode == 'router':
                expert_type = 'trend'
            else:
                expert_type = self.agent.expert_names[expert_idx]

            extrinsic_reward = self.compute_reward(
                expert_type=expert_type,
                step_pnl_roe=step_pnl_roe,
                realized_pnl_roe=realized_pnl_roe,
                trade_done=trade_done,
                holding_time=holding_steps / max_steps,
                current_position=position,
                effective_leverage=effective_leverage
            )

            total_reward = extrinsic_reward + intrinsic_reward
            self.writer.add_scalar('Reward/Extrinsic', extrinsic_reward, episode_num * max_steps + step)
            self.writer.add_scalar('Reward/Intrinsic', intrinsic_reward, episode_num * max_steps + step)

            # ---------- 다음 상태 ----------
            self.data_collector.current_index += 1
            next_idx = self.data_collector.current_index
            next_pos_val = 1.0 if position == 'LONG' else (-1.0 if position == 'SHORT' else 0.0)
            next_pos_info = [next_pos_val, 0.0, holding_steps / max_steps]
            next_state = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)
            done = (step >= max_steps - 1) or (next_state is None)
            if next_state is None:
                next_state = state

            # ---------- 전이 저장 ----------
            self.agent.put_data((
                state, action, total_reward, next_state, log_prob, done, value,
                0.0, action_mask, selected_expert, router_log_prob, router_value
            ))

            episode_reward += total_reward
            pbar.set_postfix({'R': f'{episode_reward:.2f}', 'Tr': trade_count, 'Bal': f'{self.balance:.0f}'})

            # ---------- ICM 상태 업데이트 ----------
            obs_seq, obs_info = state
            prev_state_vec = torch.cat([
                obs_seq[0, -1, :].detach(),
                obs_info[0, :].detach()
            ], dim=-1).to(self.device)
            prev_direction = direction
            prev_scale = scale

            if done:
                break

        # ---------- 에피소드 종료 강제 청산 ----------
        if position is not None:
            final_price = self.close_prices[min(self.data_collector.current_index, len(self.close_prices)-1)]
            if position == 'LONG':
                realized_return = (final_price - entry_price) / entry_price
            else:
                realized_return = (entry_price - final_price) / entry_price

            _, _, _, exit_cost, _, _ = self.env.execute_trade(
                action=last_action_scale,
                current_price=final_price,
                is_exit=True,
                leverage=effective_leverage
            )
            total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1

            # 🔥 total_trade_return 클리핑
            total_trade_return = np.clip(total_trade_return, -0.95, 5.0)

            self.balance = entry_balance * (1 + total_trade_return)
            self.equity_curve.append(self.balance)

            realized_pnl_roe = total_trade_return

            final_reward = self.compute_reward(
                expert_type=expert_type if mode != 'router' else 'trend',
                step_pnl_roe=0.0,
                realized_pnl_roe=realized_pnl_roe,
                trade_done=True,
                holding_time=0.0,
                current_position=None,
                effective_leverage=effective_leverage
            )
            episode_reward += final_reward
            trade_count += 1

            exit_direction = 2 if position == 'LONG' else 1
            terminal_action = (exit_direction, last_action_scale)

            terminal_state = self.env.get_observation(
                position_info=[0.0, 0.0, 0.0],
                current_index=min(self.data_collector.current_index, len(self.close_prices)-1)
            )
            if terminal_state is None:
                terminal_state = state
            terminal_mask = np.ones(config.ACTION_DIM, dtype=np.float32)
            most_used = np.argmax(expert_selection_counts) if sum(expert_selection_counts) > 0 else 0

            self.agent.put_data((
                state, terminal_action, final_reward, terminal_state,
                1.0, True, 0.0, 0.0, terminal_mask, most_used, None, None
            ))

        pbar.close()

        # ---------- 샤프 비율 보너스 ----------
        sharpe_bonus = self.compute_sharpe_reward(self.equity_curve)
        episode_reward += sharpe_bonus
        self.writer.add_scalar('Reward/SharpeBonus', sharpe_bonus, episode_num)

        # ---------- 에피소드 수익률 ----------
        episode_pnl = (self.balance / initial_balance) - 1.0

        # ---------- 신경망 학습 ----------
        metrics = self.agent.train_net(episode=episode_num, mode=mode, expert_idx=expert_idx)

        # ---------- TensorBoard 로깅 ----------
        self.writer.add_scalar('Reward/Total', episode_reward, episode_num)
        self.writer.add_scalar('Metrics/Return', episode_pnl, episode_num)
        self.writer.add_scalar('Metrics/TradeCount', trade_count, episode_num)
        self.writer.add_scalar('Metrics/PositionValue', self.position_value, episode_num)
        self.writer.add_scalar('Metrics/Contracts', self.contracts, episode_num)
        self.writer.add_scalar('Loss/Expert', metrics.get('Loss', 0.0), episode_num)
        self.writer.add_scalar('Loss/Router', metrics.get('Router_Loss', 0.0), episode_num)

        return episode_reward, trade_count, episode_pnl, current_key

    def _find_file(self, directory, suffix):
        """Dream Team Resume용 파일 찾기 (정확한 이름 또는 와일드카드)"""
        exact = os.path.join(directory, suffix)
        if os.path.exists(exact):
            return exact
        candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getctime)
        return None
    
    def load_dream_team(self, base_dir):
        """이전 학습 폴더에서 best_router, best_trend, best_volatility, best_sideways 로드"""
        logger.info(f"🧬 Assembling Dream Team from: {base_dir}")
        files = {
            'router': 'best_router.pth',
            0: 'best_trend.pth',
            1: 'best_volatility.pth',
            2: 'best_sideways.pth'
        }

        # Router 로드
        router_path = self._find_file(base_dir, files['router'])
        if router_path:
            try:
                ckpt = torch.load(router_path, map_location=self.device)
                if 'router' in ckpt:
                    self.agent.router.load_state_dict(self.agent._strip_prefix(ckpt['router']))
                if 'opt_router' in ckpt:
                    self.agent.opt_router.load_state_dict(ckpt['opt_router'])
                logger.info(f"   ✅ Router System loaded from {os.path.basename(router_path)}")
            except Exception as e:
                logger.warning(f"   ⚠️ Router Load Failed: {e}")

        # 전문가 3종 로드
        expert_names = ['Trend', 'Volatility', 'Sideways']
        for idx in range(3):
            fname = files[idx]
            fpath = self._find_file(base_dir, fname)
            fallback = False
            if not fpath and router_path:
                fpath = router_path  # fallback to router checkpoint
                fallback = True
            if fpath:
                try:
                    ckpt = torch.load(fpath, map_location=self.device)
                    if 'experts' in ckpt and len(ckpt['experts']) > idx:
                        self.agent.experts[idx].load_state_dict(
                            self.agent._strip_prefix(ckpt['experts'][idx]), strict=False
                        )
                    if 'opt_experts' in ckpt and len(ckpt['opt_experts']) > idx:
                        self.agent.opt_experts[idx].load_state_dict(ckpt['opt_experts'][idx])
                    source = "Router Fallback" if fallback else os.path.basename(fpath)
                    logger.info(f"   ✅ {expert_names[idx]} Expert loaded from {source}")
                except Exception as e:
                    logger.warning(f"   ⚠️ {expert_names[idx]} Load Failed: {e}")
                    
    # ------------------------------------------------------------------
    # 메인 학습 루프
    # ------------------------------------------------------------------
    def train(self, num_episodes=3000, resume=True):
        """
        Args:
            num_episodes: 총 학습 에피소드 수
            resume: True면 가장 최근 학습 폴더의 best 모델을 로드하여 이어하기
        """
        logger.info("🚀 MacroHFT v5.0 Training Started (Discrete Leverage + ICM + CVaR + Sharpe + Dream Team)")

        # ---------- Dream Team Resume ----------
        if resume:
            logger.info("♻️ Resuming from Best Individual Models (Dream Team)...")
            root_dir = 'data/macroHFT'
            if os.path.exists(root_dir):
                # 현재 시간으로 생성될 폴더를 제외한 가장 최근 폴더 찾기
                run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                subdirs = sorted(
                    [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
                    reverse=True
                )
                target_dir = None
                for d in subdirs:
                    if d != run_time:
                        target_dir = os.path.join(root_dir, d)
                        break
                if target_dir:
                    self.load_dream_team(target_dir)
                else:
                    logger.warning("⚠️ No previous training directory found. Starting fresh.")
            else:
                logger.warning("⚠️ data/macroHFT directory does not exist. Starting fresh.")
        else:
            logger.info("Starting Fresh Training.")

        # ---------- 저장 경로 설정 ----------
        best_rewards = {name: -float('inf') for name in ['trend', 'volatility', 'sideways', 'router']}
        save_dir = os.path.join('data', 'macroHFT', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)

        self.episode_rewards = []

        for ep in range(1, num_episodes + 1):
            try:
                ep_reward, trades, pnl, key = self.train_episode(ep)
                self.episode_rewards.append(ep_reward)
                avg_r = np.mean(self.episode_rewards[-10:])

                logger.info(
                    f"✅ Ep {ep} [{key.upper()}] | Reward: {ep_reward:.2f} | Avg: {avg_r:.2f} | "
                    f"Trades: {trades} | PnL: {pnl*100:.2f}%"
                )

                if ep_reward > best_rewards[key]:
                    best_rewards[key] = ep_reward
                    save_path = os.path.join(save_dir, f"best_{key}.pth")
                    self.agent.save_model(save_path)
                    logger.info(f"🏆 New best {key} model saved.")

                if ep % 10 == 0:
                    self.validate_oot(ep)

                if ep % config.TRAIN_SAVE_INTERVAL == 0:
                    self.agent.save_model(os.path.join(save_dir, f"checkpoint_ep{ep}.pth"))

            except KeyboardInterrupt:
                logger.info("Training interrupted. Saving final model...")
                self.agent.save_model(os.path.join(save_dir, "interrupted.pth"))
                break
            except Exception as e:
                logger.exception(f"Episode {ep} failed: {e}")
                continue

        self.writer.close()
        logger.info("Training completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from latest Dream Team (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start fresh training')
    args = parser.parse_args()

    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES, resume=args.resume)