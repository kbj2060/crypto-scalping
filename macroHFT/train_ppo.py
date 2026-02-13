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
        self.icm_coef = 0.01            # 🔥 0.1 → 0.01
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
        액션 마스킹: 15개 행동 각각 허용/금지
        - 인덱스 0~4:   HOLD + 레버리지 1,5,10,15,20
        - 인덱스 5~9:   LONG + 레버리지 1,5,10,15,20
        - 인덱스 10~14: SHORT + 레버리지 1,5,10,15,20
        """
        mask = np.ones(15, dtype=np.float32)
        if current_position == 'LONG':
            # LONG 관련 행동 금지 (인덱스 5~9)
            mask[5:10] = 0.0
        elif current_position == 'SHORT':
            # SHORT 관련 행동 금지 (인덱스 10~15)
            mask[10:15] = 0.0
        # HOLD는 항상 허용 (0~4)
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

        # ---------- 거래 제한 제거 (last_trade_step 변수 제거) ----------
        # self.last_trade_step = -100   # 🔥 삭제

        # ---------- 자본금 및 포지션 초기화 ----------
        initial_balance = config.EVAL_INITIAL_CAPITAL
        self.balance = initial_balance
        self.equity_curve = [self.balance]
        position = None
        entry_price = 0.0
        effective_leverage = 0.0
        holding_steps = 0
        entry_cost = 0.0
        entry_balance = 0.0          # 진입 시점 자본 (수수료 차감 전)
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

            # ---------- 미실현 손익 (레버리지 미적용) ----------
            if position == 'LONG':
                unrealized_return = (curr_price - entry_price) / entry_price
            elif position == 'SHORT':
                unrealized_return = (entry_price - curr_price) / entry_price
            else:
                unrealized_return = 0.0
            unrealized_pnl_roe = unrealized_return * effective_leverage if position else 0.0

            # ---------- 스텝 PnL (미사용, 리워드에 미반영) ----------
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

            # ---------- 액션 마스크 (포지션 중복 진입만 방지) ----------
            action_mask = self.get_action_mask(
                current_position=position
            )

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
                # ----- 진입 (포지션 없음) -----
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
                        entry_balance = self.balance  # 🔥 진입 시점 자본 저장 (수수료 차감 전)
                        position = 'LONG' if direction == 1 else 'SHORT'
                        effective_leverage = eff_lev
                        entry_cost = cost
                        self.position_value = position_value
                        self.contracts = contracts
                        holding_steps = 0
                        trade_count += 1
                        self.balance *= (1 - entry_cost)  # 진입 수수료 차감
                        # self.last_trade_step = step  # 🔥 제거

                # ----- 청산 (반대 방향) -----
                elif (direction == 1 and position == 'SHORT') or (direction == 2 and position == 'LONG'):
                    _, _, _, exit_cost, _, _ = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        is_exit=True,
                        leverage=effective_leverage
                    )
                    realized_return = unrealized_return  # 레버리지 미적용 수익률
                    
                    # # ========== 디버깅 코드 ==========
                    # print(f"\n[DEBUG] Trade ID: {trade_count+1}")
                    # print(f"  Entry balance : {entry_balance:.2f}")
                    # print(f"  Entry cost    : {entry_cost:.6f}")
                    # print(f"  Exit cost     : {exit_cost:.6f}")
                    # print(f"  Realized return (price) : {realized_return:.6f}")
                    # print(f"  Leverage      : {effective_leverage:.2f}")
                    # print(f"  Formula       : (1-{entry_cost:.6f}) * (1+{realized_return:.6f}*{effective_leverage:.2f}-{exit_cost:.6f}) - 1")
                    # # =================================
                    
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                    # # ========== 디버깅 코드 ==========
                    # print(f"  total_trade_return = {total_trade_return:.6f}")
                    # print(f"  Balance before: {entry_balance:.2f}")
                    # print(f"  Balance after : {self.balance:.2f}")
                    # print(f"  Expected      : {entry_balance * (1 + total_trade_return):.2f}")
                    # print(f"  Exit reason   : {reason}")
                    # # =================================
                    
                    self.balance = entry_balance * (1 + total_trade_return)
                    self.equity_curve.append(self.balance)
                    realized_pnl_roe = total_trade_return
                    trade_done = True
                    
                    # 포지션 초기화
                    position = None
                    effective_leverage = 0.0
                    entry_cost = 0.0
                    entry_balance = 0.0
                    self.position_value = 0.0
                    self.contracts = 0.0
                    holding_steps = 0
                    self._prev_unrealized_roe = 0.0
                    # self.last_trade_step = step  # 🔥 제거

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
                    
                    # # ========== 디버깅 코드 (강제 청산) ==========
                    # print(f"\n[DEBUG] FORCED LIQUIDATION - Trade ID: {trade_count+1}")
                    # print(f"  Entry balance : {entry_balance:.2f}")
                    # print(f"  Entry cost    : {entry_cost:.6f}")
                    # print(f"  Exit cost     : {exit_cost:.6f}")
                    # print(f"  Realized return (price) : {realized_return:.6f}")
                    # print(f"  Leverage      : {effective_leverage:.2f}")
                    # print(f"  Formula       : (1-{entry_cost:.6f}) * (1+{realized_return:.6f}*{effective_leverage:.2f}-{exit_cost:.6f}) - 1")
                    # # =============================================
                    
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                    
                    # # ========== 디버깅 코드 ==========
                    # print(f"  total_trade_return = {total_trade_return:.6f}")
                    # print(f"  Balance before: {entry_balance:.2f}")
                    # print(f"  Balance after : {self.balance:.2f}")
                    # print(f"  Expected      : {entry_balance * (1 + total_trade_return):.2f}")
                    # print(f"  Exit reason   : {reason}")
                    # # =================================
                    
                    self.balance = entry_balance * (1 + total_trade_return)
                    self.equity_curve.append(self.balance)
                    entry_balance = 0.0
                    entry_cost = 0.0
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
                    # self.last_trade_step = step  # 🔥 제거
                    pbar.set_postfix({'Exit': reason[:4]})

            # ---------- 외부 보상 계산 (청산 시에만) ----------
            if mode == 'router':
                expert_type = 'trend'
            else:
                expert_type = self.agent.expert_names[expert_idx]

            extrinsic_reward = self.compute_reward(
                expert_type=expert_type,
                step_pnl_roe=step_pnl_roe,           # 실제로는 사용 안 함 (trade_done=False면 0)
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

        # ---------- 에피소드 종료: 미청산 포지션 강제 청산 ----------
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
            
            # # ========== 디버깅 코드 (에피소드 종료) ==========
            # print(f"\n[DEBUG] EPISODE END - Forced Liquidation")
            # print(f"  Entry balance : {entry_balance:.2f}")
            # print(f"  Entry cost    : {entry_cost:.6f}")
            # print(f"  Exit cost     : {exit_cost:.6f}")
            # print(f"  Realized return (price) : {realized_return:.6f}")
            # print(f"  Leverage      : {effective_leverage:.2f}")
            # # ================================================
            
            total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
            
            # # ========== 디버깅 코드 ==========
            # print(f"  total_trade_return = {total_trade_return:.6f}")
            # print(f"  Balance before: {entry_balance:.2f}")
            # print(f"  Balance after : {self.balance:.2f}")
            # print(f"  Expected      : {entry_balance * (1 + total_trade_return):.2f}")
            # # =================================
            
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

    # ------------------------------------------------------------------
    # 메인 학습 루프
    # ------------------------------------------------------------------
    def train(self, num_episodes=3000):
        logger.info("🚀 MacroHFT v4.5 Training Started (ICM + CVaR + Sharpe + Fee Fix)")

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

                if ep % 50 == 0:
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
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)