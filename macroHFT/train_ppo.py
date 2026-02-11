"""
PPO Training Script for MacroHFT v8.2 SOTA
===========================================
- Meta-Lambda: 전문가별 손실 회피 계수 자동 튜닝
- Volatility Surprise: 내재적 보상으로 탐험 촉진
- Identity Bonus: PnL 비례 최소화 (고정값 제거)
- Reward Normalization: Running 평균/분산으로 보상 정규화
- Curriculum Learning, Dream Team Resume 지원
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

try:
    from common import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import config
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import INFO_DIM_ELITE8
from common.trading_env import TradingEnvironment
from macroHFT.ppo_agent import PPOAgent
from macroHFT.macrohft_reward import calculate_ppo_reward, reset_reward_tracker

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger('common.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('common.mtf_processor').setLevel(logging.WARNING)

# PyTorch 최적화 (컴파일은 비활성화 권장)
if hasattr(config, 'USE_CUDNN_BENCHMARK') and config.USE_CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True
    logger.info("✅ cuDNN Benchmark Activated")

if hasattr(config, 'USE_HIGH_MATMUL_PRECISION') and config.USE_HIGH_MATMUL_PRECISION:
    torch.set_float32_matmul_precision('high')
    logger.info("✅ TensorCore Precision Optimized (TF32)")

# ----------------------------------------------------------------------
# [권장] PPO 하이퍼파라미터 (config.py에서 재정의 가능)
# ----------------------------------------------------------------------
# PPO_LEARNING_RATE = 3e-5      # 1e-4보다 안정적
# PPO_K_EPOCHS = 5             # 10은 과적합 위험
# PPO_ENTROPY_COEF = 0.05      # 탐험 장려
# PPO_LAMBDA = 0.95            # GAE lambda
# PPO_GAMMA = 0.99            # 할인율
# ----------------------------------------------------------------------

class PPOTrainer:
    def __init__(self, enable_visualization=False):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        logger.info(f"Strategies Init: Elite 8 ({len(self.strategies)})")

        # 1. Load Data
        self._load_features()

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler_dummy()

        # 🔥 핫픽스: TradingEnvironment 속성 강제 추가
        self.env.position = None
        self.env.entry_price = 0.0
        self.env.entry_index = 0
        self.env.reset_reward_states = lambda: None

        # [Monkey Patch] PPO Reward
        import types
        self.env.calculate_reward = types.MethodType(calculate_ppo_reward, self.env)
        logger.info("✅ Tactical Reward Logic Applied")

        state_dim = self.env.get_state_dim()
        action_dim = 3  # [Hold, Buy, Sell]
        info_dim = INFO_DIM_ELITE8

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device} | State Dim: {state_dim} | Info Dim: {info_dim}")

        # 2. Agent 초기화
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        self.episode_rewards = []

        # 3. NumPy 캐싱 (속도 최적화)
        if self.data_collector.eth_data is not None:
            self.close_prices = self.data_collector.eth_data['close'].values.astype(np.float32)
            if 'volatility_z' in self.data_collector.eth_data.columns:
                self.volatility_data = self.data_collector.eth_data['volatility_z'].values.astype(np.float32)
            else:
                self.volatility_data = np.zeros(len(self.close_prices), dtype=np.float32)

            strategy_cols = [f'strategy_{i}' for i in range(len(self.strategies))]
            valid_strat_cols = [c for c in strategy_cols if c in self.data_collector.eth_data.columns]
            if valid_strat_cols:
                self.strategy_matrix = self.data_collector.eth_data[valid_strat_cols].values.astype(np.float32)
            else:
                self.strategy_matrix = np.zeros((len(self.close_prices), len(self.strategies)), dtype=np.float32)

            logger.info(f"✅ Data Caching Complete: {len(self.close_prices):,} rows")
        else:
            raise RuntimeError("Failed to load ETH data.")

        # 4. TensorBoard
        tb_base = os.path.join('logs', 'tensorboard')
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S_PureRL_MoE')
        tb_log_dir = os.path.join(tb_base, run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # 5. 커리큘럼 인덱스
        self._prepare_curriculum_indices()

        # ------------------------------------------------------------------
        # [제안 3] 변동성 예측기 (Volatility Predictor)
        # ------------------------------------------------------------------
        self.volatility_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
        self.vol_optimizer = torch.optim.Adam(
            self.volatility_predictor.parameters(),
            lr=getattr(config, 'VOLATILITY_PREDICTOR_LR', 1e-3)
        )
        self.vol_loss_fn = torch.nn.MSELoss()

        # chop_index 캐싱
        if 'chop_index' in self.data_collector.eth_data.columns:
            self.chop_data = self.data_collector.eth_data['chop_index'].values.astype(np.float32)
        else:
            self.chop_data = np.zeros(len(self.close_prices), dtype=np.float32)

        # ------------------------------------------------------------------
        # [Reward Normalization] Running statistics
        # ------------------------------------------------------------------
        self.reward_rms = {'mean': 0.0, 'std': 1.0, 'count': 0}

    # ------------------------------------------------------------------
    # 액션 마스킹 (기존 유지)
    # ------------------------------------------------------------------
    def get_action_mask(self, current_position, market_volatility, step_count):
        mask = np.ones(3, dtype=np.float32)
        if current_position == 'LONG':
            mask[1] = 0.0
        elif current_position == 'SHORT':
            mask[2] = 0.0
        else:
            if step_count > config.TRAIN_MAX_STEPS_PER_EPISODE - 10:
                mask[1] = 0.5   # 완전 차단 -> 확률적 허용
                mask[2] = 0.5
        return mask

    # ------------------------------------------------------------------
    # 피처 로딩 (Oracle 제거, 순수 RL)
    # ------------------------------------------------------------------
    def _load_features(self):
        path = 'data/training_features.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
            df = df.ffill().bfill()

            cached_strategies_path = 'data/cached_strategies.csv'
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except:
                    pass

            self.data_collector.eth_data = df
            logger.info("✅ Features Loaded (Pure RL Mode)")
        else:
            logger.error("❌ Feature file missing.")
            sys.exit(1)

    def _fit_global_scaler_dummy(self):
        df = self.data_collector.eth_data
        if df is not None:
            train_size = int(len(df) * config.TRAIN_SPLIT)
            self.train_end_idx = train_size
            self.test_start_idx = int(len(df) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
            self.test_end_idx = len(df)
            self.env.scaler_fitted = True

    # ------------------------------------------------------------------
    # 커리큘럼 인덱스 분할
    # ------------------------------------------------------------------
    def _prepare_curriculum_indices(self):
        df = self.data_collector.eth_data.iloc[:self.train_end_idx]
        valid_indices = list(range(config.LOOKBACK + 100, self.train_end_idx - 100))

        if len(valid_indices) < 100:
            self.all_indices = valid_indices
            self.indices_trend = self.indices_vol = self.indices_chop = self.indices_all = valid_indices
            self.idx_map = [valid_indices] * 3
            self.idx_all = valid_indices
            return

        self.all_indices = valid_indices
        self.indices_all = valid_indices

        if 'chop_index' in df.columns and 'volatility_z' in df.columns:
            chop = df['chop_index'].values
            vol = df['volatility_z'].values

            self.indices_trend = [i for i in valid_indices if i < len(chop) and chop[i] < 45.0]
            if len(self.indices_trend) < 100:
                self.indices_trend = valid_indices

            vol_vals = [vol[i] for i in valid_indices if i < len(vol)]
            if vol_vals:
                vol_threshold = np.quantile(vol_vals, 0.75)
                self.indices_vol = [i for i in valid_indices if i < len(vol) and vol[i] > vol_threshold]
            else:
                self.indices_vol = valid_indices
            if len(self.indices_vol) < 100:
                self.indices_vol = valid_indices

            if vol_vals:
                vol_mean = np.mean(vol_vals)
                self.indices_chop = [i for i in valid_indices if i < len(chop) and chop[i] > 50.0 and vol[i] < vol_mean]
            else:
                self.indices_chop = valid_indices
            if len(self.indices_chop) < 100:
                self.indices_chop = valid_indices

            self.idx_map = [self.indices_trend, self.indices_vol, self.indices_chop]
            self.idx_all = self.indices_all
            logger.info("📊 Curriculum Split: T:%d V:%d S:%d",
                        len(self.indices_trend), len(self.indices_vol), len(self.indices_chop))
        else:
            self.idx_map = [self.all_indices] * 3
            self.idx_all = self.all_indices

    # ------------------------------------------------------------------
    # 테스트셋 검증
    # ------------------------------------------------------------------
    def validate_on_test_set(self, max_steps=480):
        if not hasattr(self, 'test_start_idx') or self.test_start_idx >= self.test_end_idx - 1:
            return 0.0, 0.0
        steps = min(max_steps, self.test_end_idx - self.test_start_idx - 1)

        self.agent.reset_episode_states()
        balance = config.EVAL_INITIAL_CAPITAL
        balance_history = [balance]
        current_position = None
        entry_price = 0.0
        entry_index = 0
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)

        for step in range(steps):
            idx = self.test_start_idx + step
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])

            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price

            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max(1, steps)]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                break

            with torch.no_grad():
                action, _, _, _ = self.agent.select_action(state, action_mask=None, deterministic=True)

            realized_pnl = 0.0
            if action == 1:  # Buy
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'SHORT':
                    realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                    current_position = None
            elif action == 2:  # Sell
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'LONG':
                    realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    current_position = None

            if realized_pnl != 0.0:
                balance = balance * (1 + realized_pnl)
            balance_history.append(balance)

        test_reward = (balance - config.EVAL_INITIAL_CAPITAL) / config.EVAL_INITIAL_CAPITAL
        returns = np.diff(balance_history) / (np.array(balance_history[:-1], dtype=float) + 1e-10)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8))
        return test_reward, sharpe

    # ------------------------------------------------------------------
    # [제안 3] 변동성 예측기 학습 (미니배치)
    # ------------------------------------------------------------------
    def _train_volatility_predictor(self):
        if len(self.data_collector.eth_data) < 1000:
            return
        indices = np.random.choice(
            range(config.LOOKBACK, len(self.close_prices) - 1),
            size=32, replace=False
        )
        losses = []
        for idx in indices:
            state = self.env.get_observation(position_info=[0, 0, 0], current_index=idx)
            if state is None:
                continue
            state_tensor = state[0].to(self.agent.device)
            target_vol = torch.as_tensor([self.volatility_data[idx + 1]], dtype=torch.float32, device=self.agent.device)
            pred = self.volatility_predictor(state_tensor[:, -1, :])
            loss = self.vol_loss_fn(pred.squeeze(), target_vol.squeeze())
            losses.append(loss)
        if losses:
            loss = torch.stack(losses).mean()
            self.vol_optimizer.zero_grad()
            loss.backward()
            self.vol_optimizer.step()

    # ------------------------------------------------------------------
    # 에피소드 학습 (핵심)
    # ------------------------------------------------------------------
    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE

        # 커리큘럼: 3에피소드 전문가, 1에피소드 라우터
        cycle = (episode_num - 1) % 4
        if cycle < 3:
            mode = "expert"
            expert_idx = cycle
            current_key = self.agent.expert_names[expert_idx]
            target_indices = self.idx_map[expert_idx]
        else:
            mode = "router"
            expert_idx = 0
            current_key = "router"
            target_indices = self.idx_all

        if not target_indices:
            target_indices = self.all_indices
        start_idx = np.random.choice(target_indices)
        self.data_collector.current_index = start_idx

        # 초기화
        self.env.reset_reward_states()
        reset_reward_tracker()
        self.agent.reset_episode_states()

        current_position = None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        episode_pnl = 0.0
        trade_count = 0
        prev_unrealized_pnl = 0.0
        hold_count, buy_count, sell_count = 0, 0, 0

        expert_selection_counts = [0, 0, 0]  # Trend, Vol, Side

        # 메타 λ 학습을 위한 전문가별 수익 기록
        expert_pnl = {0: 0.0, 1: 0.0, 2: 0.0}
        expert_count = {0: 0, 1: 0, 2: 0}

        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num} [{current_key.upper()}]",
                    leave=False, mininterval=60.0)

        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx - 1:
                break

            curr_price = float(self.close_prices[current_idx])

            # 변동성 라벨 (디버깅용)
            lookback_vol = min(10, current_idx - config.LOOKBACK)
            if lookback_vol > 0:
                past_prices = self.close_prices[current_idx - lookback_vol:current_idx]
                returns = np.diff(past_prices) / (past_prices[:-1] + 1e-10)
                volatility_label = float(np.std(returns) * 100.0)
            else:
                volatility_label = 0.0

            # 미실현 손익
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price

            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0

            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]

            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None:
                break

            prev_pos_str = current_position
            market_vol = float(self.volatility_data[current_idx])
            action_mask = self.get_action_mask(current_position, market_vol, step)

            # ----------------------------------------------------------
            # [제안 3] 변동성 놀라움 (내재적 보상)
            # ----------------------------------------------------------
            with torch.no_grad():
                state_tensor = state[0].to(self.agent.device)
                pred_vol = self.volatility_predictor(state_tensor[:, -1, :]).item()
            actual_vol = float(self.volatility_data[current_idx])
            vol_surprise = abs(actual_vol - pred_vol)
            intrinsic_reward = min(vol_surprise * 0.1, 1.0)

            # ----------------------------------------------------------
            # [제안 4] 정체성 보상에 필요한 chop_index, volatility_z
            # ----------------------------------------------------------
            chop_val = float(self.chop_data[current_idx])
            vol_z = float(self.volatility_data[current_idx])

            # 행동 선택
            action, prob, val, selected_expert = self.agent.select_action(
                state, action_mask=action_mask, mode=mode, expert_idx=expert_idx
            )

            expert_selection_counts[selected_expert] += 1

            # 강제 청산 조건
            should_exit, exit_reason = self.env.check_exit_conditions(unrealized_pnl, holding_time)
            if should_exit and current_position is not None:
                if current_position == 'LONG':
                    action = 2
                elif current_position == 'SHORT':
                    action = 1
                if exit_reason:
                    pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count, 'Exit': exit_reason[:4]})

            if action == 0:
                hold_count += 1
            elif action == 1:
                buy_count += 1
            else:
                sell_count += 1

            # 슬리피지
            slippage = np.random.uniform(0.0001, 0.0005)
            trade_done = False
            realized_pnl = 0.0
            holding_time_norm = holding_time / max_steps

            if action == 1:  # Buy
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price * (1 + slippage)
                    entry_index = current_idx
                    trade_count += 1
                elif current_position == 'SHORT':
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    episode_pnl += realized_pnl
                    current_position = None
            elif action == 2:  # Sell
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price * (1 - slippage)
                    entry_index = current_idx
                    trade_count += 1
                elif current_position == 'LONG':
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    episode_pnl += realized_pnl
                    current_position = None

            # 전략 점수 업데이트
            strategy_scores = self.strategy_matrix[current_idx].tolist()
            self.env.update_trading_metrics(
                prev_position=prev_pos_str,
                current_position=current_position,
                strategy_scores=strategy_scores if strategy_scores else None,
                volatility_pred=0.0,
                actual_volatility=market_vol,
            )

            # ----------------------------------------------------------
            # [제안 1] 메타 λ 가져오기
            # ----------------------------------------------------------
            lambda_val = self.agent.lambda_learner(
                torch.as_tensor([selected_expert], device=self.agent.device)
            ).item()

            # ----------------------------------------------------------
            # 보상 계산 (v8.2 - PnL 중심, 휴리스틱 최소화)
            # ----------------------------------------------------------
            raw_reward = self.env.calculate_reward(
                step_pnl=step_pnl,
                realized_pnl=realized_pnl,
                trade_done=trade_done,
                holding_time=holding_time_norm,
                action=action,
                prev_position=prev_pos_str,
                current_position=current_position,
                expert_idx=selected_expert,
                chop_index=chop_val,
                volatility_z=vol_z,
                lambda_meta=lambda_val
            )

            # 내재적 보상 추가
            raw_reward += intrinsic_reward

            # ----------------------------------------------------------
            # [Reward Normalization] Running 평균/표준편차로 정규화
            # ----------------------------------------------------------
            if self.reward_rms['count'] > 30:  # 10 에피소드 이후부터 정규화
                norm_reward = (raw_reward - self.reward_rms['mean']) / (self.reward_rms['std'] + 1e-8)
                reward = np.clip(norm_reward, -5.0, 5.0)  # -5~5로 클리핑
            else:
                reward = raw_reward  # 초기에는 raw 사용

            # ----------------------------------------------------------
            # Transition 저장
            # ----------------------------------------------------------
            prev_unrealized_pnl = unrealized_pnl if not trade_done else 0.0
            self.data_collector.current_index += 1
            next_idx = self.data_collector.current_index

            next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            next_hold_time = (next_idx - entry_index) if current_position is not None else 0
            next_pos_info = [next_pos_val, 0.0, next_hold_time / max_steps]
            next_state = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)

            done = False if step < max_steps - 1 else True
            if next_state is None:
                done = True
                next_state = state

            self.agent.put_data((
                state, action, reward, next_state, prob, done, val,
                volatility_label, action_mask, selected_expert
            ))

            episode_reward += reward
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count})

            if done:
                break

        # ---------- 에피소드 강제 종료 처리 ----------
        if current_position is not None:
            if current_position == 'LONG':
                realized_pnl = (curr_price - entry_price) / entry_price
                exit_action = 2
            else:
                realized_pnl = (entry_price - curr_price) / entry_price
                exit_action = 1
            episode_pnl += realized_pnl

            final_raw = self.env.calculate_reward(
                step_pnl=0.0, realized_pnl=realized_pnl, trade_done=True,
                holding_time=0.0, action=exit_action,
                prev_position=current_position, current_position=None,
                chop_index=chop_val, volatility_z=vol_z,
                lambda_meta=lambda_val
            )
            # 정규화 적용
            if self.reward_rms['count'] > 5:
                final_reward = np.clip((final_raw - self.reward_rms['mean']) / (self.reward_rms['std'] + 1e-8), -5.0, 5.0)
            else:
                final_reward = final_raw

            episode_reward += final_reward
            trade_count += 1

            safe_idx = min(self.data_collector.current_index, self.train_end_idx - 1)
            last_pos_info = [0.0, 0.0, 0.0]
            last_state = self.env.get_observation(position_info=last_pos_info, current_index=safe_idx)
            if last_state is None:
                last_state = state
            terminal_mask = np.ones(3, dtype=np.float32)

            most_used_expert = (expert_selection_counts.index(max(expert_selection_counts))
                                if sum(expert_selection_counts) > 0 else 0)
            self.agent.put_data((
                state, exit_action, final_reward, last_state, 1.0, True, 0.0, 0.0,
                terminal_mask, most_used_expert
            ))

            # 전문가별 수익 집계
            expert_pnl[selected_expert] += realized_pnl
            expert_count[selected_expert] += 1

        pbar.close()

        # --------------------------------------------------------------
        # 변동성 예측기 학습
        # --------------------------------------------------------------
        self._train_volatility_predictor()

        # --------------------------------------------------------------
        # [제안 1] 메타 λ 업데이트
        # --------------------------------------------------------------
        episode_pnl_list = []
        for idx in range(3):
            if expert_count[idx] > 0:
                avg_pnl = expert_pnl[idx] / expert_count[idx]
                episode_pnl_list.append((idx, avg_pnl))
        self.agent.update_meta_lambdas(episode_pnl_list)

        # --------------------------------------------------------------
        # [Reward Normalization] 에피소드 보상 통계 업데이트 (Running)
        # --------------------------------------------------------------
        count = self.reward_rms['count'] + 1
        delta = episode_reward - self.reward_rms['mean']
        self.reward_rms['mean'] += delta / count
        delta2 = episode_reward - self.reward_rms['mean']
        self.reward_rms['std'] = np.sqrt(((self.reward_rms['std']**2 * self.reward_rms['count']) + delta * delta2) / count)
        self.reward_rms['count'] = count

        # --------------------------------------------------------------
        # 신경망 학습 (PPO + D-PPO + Reward Dist)
        # --------------------------------------------------------------
        metrics = self.agent.train_net(episode=episode_num, mode=mode, expert_idx=expert_idx)

        # TensorBoard 로깅
        if self.writer:
            self.writer.add_scalar('Reward/Total', episode_reward, episode_num)
            self.writer.add_scalar('Metrics/PnL', episode_pnl, episode_num)
            self.writer.add_scalar('Metrics/Trade_Count', trade_count, episode_num)
            self.writer.add_scalar('Reward/Raw_Mean', self.reward_rms['mean'], episode_num)
            self.writer.add_scalar('Reward/Raw_Std', self.reward_rms['std'], episode_num)
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, episode_num)

        return episode_reward, trade_count, episode_pnl, current_key

    # ------------------------------------------------------------------
    # 메인 학습 루프
    # ------------------------------------------------------------------
    def train(self, num_episodes=1000, resume=True):
        logger.info("🚀 PPO Learning Started (Pure RL - No Oracle)")

        best_rewards = {
            'trend': -float('inf'),
            'volatility': -float('inf'),
            'sideways': -float('inf'),
            'router': -float('inf')
        }

        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('data', 'macroHFT', run_time)
        os.makedirs(save_dir, exist_ok=True)
        base_path = os.path.join(save_dir, os.path.splitext(os.path.basename(config.AI_MODEL_PATH))[0])

        # Dream Team Resume
        if resume:
            logger.info("♻️ Resuming from Best Individual Models (Dream Team)...")
            root_dir = 'data/macroHFT'
            if os.path.exists(root_dir):
                subdirs = sorted([d for d in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, d))], reverse=True)
                if subdirs:
                    target_dir = None
                    for d in subdirs:
                        if d != run_time:
                            target_dir = os.path.join(root_dir, d)
                            break
                    if target_dir:
                        self.agent.load_dream_team(target_dir)
                    else:
                        logger.warning("⚠️ No previous training directory found to resume from.")
                else:
                    logger.warning("⚠️ No training history found.")
            else:
                logger.warning("⚠️ data/macroHFT directory does not exist.")
        else:
            logger.info("Starting Fresh Training.")

        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None:
                    continue
                r, c, pnl, current_key = res

                self.episode_rewards.append(r)
                avg_r = np.mean(self.episode_rewards[-10:])
                logger.info(
                    f"✅ Ep {ep} [{current_key.upper()}]: Reward {r:.4f} | Avg {avg_r:.4f} | "
                    f"Trades: {c} | PnL: {pnl * 100:.2f}%"
                )

                if r > best_rewards[current_key]:
                    best_rewards[current_key] = r
                    save_name = f"{base_path}_best_{current_key}.pth"
                    self.agent.save_model(save_name)
                    logger.info(f"🏆 New Best {current_key.upper()} Model! ({r:.4f})")

                if ep % 10 == 0:
                    self.agent.save_model(f"{base_path}_last.pth")

            except KeyboardInterrupt:
                logger.info("Training interrupted.")
                break
            except Exception as e:
                logger.error(f"Ep {ep} Error: {e}")
                continue

        self.writer.close()


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES, resume=False)