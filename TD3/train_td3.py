"""
TD3 Continuous Action Training Script (Updated for Elite 8)
"""
import logging
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config
from common.preprocess import add_volatility_feature
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import TradingEnvironment

try:
    from .td3_agent import TD3Agent
except ImportError:
    from TD3.td3_agent import TD3Agent

os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('common.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('common.mtf_processor').setLevel(logging.WARNING)


class TD3Trainer:
    def __init__(self):
        self.data_collector = DataCollector(use_saved_data=True)
        # Elite 8 Strategies
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        logger.info("전략 초기화: Elite 8 (%d개)", len(self.strategies))
        self._load_features()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data() # GPU Caching & Signal Gen
        
        # [Genius Patch] TD3 전용 리워드 함수 주입 (Monkey Patching)
        from TD3.td3_reward import calculate_td3_reward
        import types
        self.env.calculate_reward = types.MethodType(calculate_td3_reward, self.env)
        logger.info("✅ TD3 전용 리워드 로직(Strategic Reward: Big Trend) 적용 완료")

        state_dim = self.env.get_state_dim() # 44
        action_dim = 1
        # Elite 8 Info: 11 + Volatility(1) = 12
        info_dim = 12 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("TD3 Training on %s | State Dim: %d | Info Dim: %d", device, state_dim, info_dim)

        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_time = run_time
        self.save_dir = os.path.join('data', 'td3', run_time)
        os.makedirs(self.save_dir, exist_ok=True)

        self.agent = TD3Agent(state_dim, action_dim, info_dim, device=device)
        self.writer = SummaryWriter(log_dir=f"logs/tensorboard/td3_cont_{run_time}")
        
        self.pnl_history = deque(maxlen=20)

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if not os.path.exists(path):
            logger.warning("⚠️ 피처 파일이 없습니다. 자동으로 데이터를 생성합니다...")
            try:
                prepare_script = os.path.join('utils', 'prepare_training_data.py')
                logger.info("")
                logger.info("=" * 80)
                result = subprocess.run([sys.executable, prepare_script], timeout=600)
                logger.info("=" * 80)
                logger.info("")
                
                if result.returncode == 0:
                    logger.info("✅ 피처 데이터 생성 완료")
                else:
                    logger.error("❌ 데이터 생성 실패")
                    raise RuntimeError("피처 데이터 생성 실패")
            except subprocess.TimeoutExpired:
                logger.error("❌ 데이터 생성 시간 초과 (10분)")
                raise RuntimeError("피처 데이터 생성 시간 초과")
            except Exception as e:
                logger.error(f"❌ 데이터 생성 중 오류: {e}")
                raise
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    for col in [c for c in cached_df.columns if c.startswith('strategy_')]:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except: pass
            
            # Volatility Feature 추가 (TD3 Info용)
            if 'volatility_20tick' not in df.columns:
                df = add_volatility_feature(df)
            
            self.data_collector.eth_data = df
            logger.info(f"✅ 데이터 로드 완료: {len(df):,}행")
        else:
            raise FileNotFoundError(f"파일 없음: {path}")

    def _fit_global_scaler_dummy(self):
        df = self.data_collector.eth_data
        if df is not None:
            self.train_end_idx = int(len(df) * config.TRAIN_SPLIT)

    def _augment_info(self, info, idx):
        # Info Tensor(11) + Volatility(1) = 12
        try:
            vol = float(self.data_collector.eth_data.iloc[idx].get('volatility_20tick', 0.0))
        except: vol = 0.0

        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2: vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)

    def train(self, resume=True):
        logger.info("TD3 True Continuous Action Training Started...")
        self._fit_global_scaler_dummy()
        
        total_timesteps = 0
        max_episodes = config.TRAIN_NUM_EPISODES
        max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        warmup = config.TD3_WARMUP_STEPS
        best_reward = -float('inf')

        if resume:
            td3_dir = os.path.join('data', 'td3')
            last_model_path = None
            if os.path.isdir(td3_dir):
                subdirs = [d for d in os.listdir(td3_dir) if os.path.isdir(os.path.join(td3_dir, d)) and d != self.run_time]
                for run_name in sorted(subdirs, reverse=True):
                    candidate = os.path.join(td3_dir, run_name, "last_td3_model_actor.pth")
                    if os.path.isfile(candidate):
                        last_model_path = os.path.join(td3_dir, run_name, "last_td3_model")
                        break
            if last_model_path:
                try:
                    self.agent.load(last_model_path)
                    logger.info("모델 로드 완료 (이어하기): %s", last_model_path)
                    # [핵심] 이어하기 = 이미 학습됨 = Warmup 스킵
                    logger.info("⚠️ 이어하기 감지: Warmup을 강제로 스킵합니다.")
                    total_timesteps = warmup + 1  # Warmup 건너뛰기
                except Exception as e:
                    logger.warning("모델 로드 실패 (처음부터 진행): %s", e)
            else:
                logger.info("새로 학습 시작 (이전 모델 미로드)")
        else:
            logger.info("새로 학습 시작 (이전 모델 미로드)")


        TRANSACTION_COST = 0.0005

        for ep in range(1, max_episodes + 1):
            low = config.LOOKBACK + 100
            high = max(low + 1, self.train_end_idx - max_steps - 100)
            start_idx = np.random.randint(low, high)

            self.data_collector.current_index = start_idx
            self.env.reset_reward_states()
            self.agent.position_cooldown = 0
            self.pnl_history.clear()

            rand_start = np.random.rand()
            current_pos_size = 0.0 if rand_start < 0.5 else (0.5 if rand_start < 0.75 else -0.5)

            pos_info = [current_pos_size, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=start_idx)
            if state is None: continue
            
            state = (state[0], self._augment_info(state[1], start_idx))

            episode_reward = 0.0
            episode_trades = 0

            for step in range(max_steps):
                total_timesteps += 1
                curr_idx = self.data_collector.current_index
                is_warmup = total_timesteps < warmup

                if is_warmup:
                    action_val = np.random.uniform(-1, 1)
                    risk_val = 0.5
                else:
                    # [Stabilized] 탐험 노이즈 축소 (0.3 → 0.1)
                    # 리워드 스케일이 줄어들면서 Q-value 크기도 작아짐
                    # 기존 노이즈가 너무 크면 학습된 Q값을 무시하고 랜덤 행동만 반복
                    action_val_arr, _, risk_val = self.agent.select_action(state, noise=0.1)
                    action_val = float(action_val_arr[0])

                target_pos_size = action_val if abs(action_val) > 0.3 else 0.0
                
                # [레버리지 시스템] Action → 레버리지 계산
                # Action의 절댓값 * MAX_LEVERAGE = 실제 레버리지
                max_leverage = getattr(config, 'LEVERAGE', 20)
                effective_leverage = abs(action_val) * max_leverage
                
                # 최소 레버리지 필터 (1배 미만은 거래 안 함)
                if effective_leverage < 1.0:
                    target_pos_size = 0.0
                
                is_opening = (current_pos_size == 0.0) and (target_pos_size != 0.0)
                is_flipping = (current_pos_size * target_pos_size < 0)
                is_strength_change = abs(target_pos_size - current_pos_size) > 0.4
                
                if not (is_opening or is_flipping or is_strength_change):
                    target_pos_size = current_pos_size

                trade_amount = target_pos_size - current_pos_size
                if abs(trade_amount) > 1e-4: episode_trades += 1
                
                # [레버리지 시스템] 수수료 = 레버리지 * 수수료율
                trade_cost = effective_leverage * TRANSACTION_COST if abs(trade_amount) > 1e-4 else 0.0
                current_pos_size = target_pos_size

                curr_price = float(self.data_collector.eth_data.iloc[curr_idx]['close'])
                self.data_collector.current_index += 1
                next_idx = self.data_collector.current_index
                
                if next_idx >= len(self.data_collector.eth_data):
                    done = True
                    next_state = state
                    break
                
                next_price = float(self.data_collector.eth_data.iloc[next_idx]['close'])
                
                # [레버리지 시스템] ROE 계산
                # ROE = (가격 변동률) * (포지션 부호) * (레버리지) - 수수료
                price_return = (next_price - curr_price) / curr_price
                
                # 포지션 부호 반영 (Long = +, Short = -)
                position_direction = np.sign(current_pos_size) if abs(current_pos_size) > 0.01 else 0.0
                raw_return = price_return if position_direction >= 0 else -price_return
                
                # [핵심] 레버리지 적용된 ROE (Return On Equity)
                step_pnl_roe = (raw_return * effective_leverage) - trade_cost if abs(current_pos_size) > 0.01 else 0.0

                self.pnl_history.append(step_pnl_roe)
                
                # [레버리지 시스템] 청산 로직 체크
                should_exit, exit_reason = self.env.check_exit_conditions(
                    unrealized_pnl_roe=step_pnl_roe,
                    holding_time_steps=step
                )
                
                if should_exit:
                    logger.debug(f"   🚨 청산 발동: {exit_reason} | ROE: {step_pnl_roe*100:.2f}%")
                    # [Stabilized] 청산 시 강력한 페널티
                    # 새로운 리워드 스케일(-20~20)에서 -10은 매우 큰 페널티
                    if exit_reason == "LIQUIDATION":
                        step_pnl_roe = -0.80  # 실제 자산 80% 손실
                        # 리워드 함수에서 이를 처리하도록 하되, 별도 페널티 추가 신호
                        # 여기서는 ROE를 그대로 전달하고, 리워드 함수가 처리
                    current_pos_size = 0.0  # 포지션 정리
                
                # [레버리지 시스템] 리워드 계산 (횡보 페널티 포함)
                reward = self.env.calculate_reward(
                    step_pnl=step_pnl_roe,
                    realized_pnl=0.0,
                    trade_done=abs(trade_amount) > 1e-4,
                    holding_time=step,
                    action=action_val,
                    prev_position=0.0,
                    current_position=current_pos_size,
                    effective_leverage=effective_leverage
                )
                
                episode_reward += reward

                next_pos_info = [current_pos_size, step_pnl_roe * 10, 1.0 if abs(trade_amount) < 0.1 else 0.0]
                next_state_raw = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)

                done = (step >= max_steps - 1) or (next_state_raw is None)
                if next_state_raw is None:
                    next_state = state
                else:
                    next_state = (next_state_raw[0], self._augment_info(next_state_raw[1], next_idx))

                self.agent.replay_buffer.add(state, [target_pos_size], reward, next_state, done)
                state = next_state

                if total_timesteps >= warmup:
                    metrics = self.agent.train(batch_size=config.TD3_BATCH_SIZE)
                    if metrics and step % 10 == 0:
                        self.writer.add_scalar('Loss/Critic', metrics.get('critic_loss', 0), total_timesteps)
                        self.writer.add_scalar('Action/Pos_Size', current_pos_size, total_timesteps)

                if done: break

            logger.info("Ep %d | Reward: %.2f | Steps: %d | Trades: %d", ep, episode_reward, total_timesteps, episode_trades)
            self.writer.add_scalar('Episode/Reward', episode_reward, ep)
            self.writer.add_scalar('Episode/Trades', episode_trades, ep)

            self.agent.save(os.path.join(self.save_dir, "last_td3_model"))
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(os.path.join(self.save_dir, "best_td3_model"))
                logger.info("New Best Model! Reward: %.2f", best_reward)

            if ep % 10 == 0:
                self.agent.save(os.path.join(self.save_dir, f"td3_model_{ep}"))


if __name__ == "__main__":
    trainer = TD3Trainer()
    trainer.train(resume=True)