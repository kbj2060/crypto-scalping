"""
PPO 학습 스크립트 (Hierarchical RL Optimized + TensorBoard)
- High-Level(Entry) & Low-Level(Exit) 구조에 맞춘 학습 로직
- 불필요한 Action Masking 및 Switching 로직 제거 (Agent가 알아서 판단)
- 에피소드 종료 시 강제 청산(Force Close) 추가하여 리워드 정합성 확보
- [추가] TensorBoard 실시간 학습 로그 기록
"""
import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# [추가] TensorBoard 라이브러리 임포트
from torch.utils.tensorboard import SummaryWriter

# config import
try:
    from . import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import config
from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)
from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent

# 로깅 설정
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

logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

class PPOTrainer:
    def __init__(self, enable_visualization=False):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        logger.info(f"전략 초기화: {len(self.strategies)}개 전략")
        
        self._load_features()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        state_dim = self.env.get_state_dim()
        # Global Action Dim = 4 (0:Wait/Hold, 1:Long, 2:Short, 3:Exit)
        action_dim = 4  
        # info_dim = 15 (전략 12개 + 포지션 3개)
        info_dim = 15
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"디바이스: {device} | Hierarchical PPO Agent Initialized (Info Dim: {info_dim})")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth" 
        
        try:
            self.agent.load_model(last_model_path)
        except Exception as e:
            logger.warning(f"초기 모델 로드 중 메시지: {e}")
        
        self.episode_rewards = []
        self.avg_rewards = []

        # [추가] TensorBoard Writer 초기화 (실행마다 별도 run으로 표시되도록 타임스탬프 폴더 사용)
        tb_base = os.path.join('logs', 'tensorboard')
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_log_dir = os.path.join(tb_base, run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard Logging Started: {tb_log_dir} (run: {run_name})")

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.ffill().bfill()
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
        else:
            logger.warning("⚠️ 피처 파일이 없습니다.")

    def _fit_global_scaler(self):
        if not self.env.scaler_fitted:
            df = self.data_collector.eth_data
            if df is None:
                return
            
            train_size = int(len(df) * config.TRAIN_SPLIT)
            self.train_end_idx = train_size
            
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
            
            sample = df.iloc[:train_size].sample(n=min(10000, train_size))[target_cols].values.astype(np.float32)
            self.env.preprocessor.fit(sample)
            self.env.scaler_fitted = True
            
            path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            self.env.preprocessor.save(path)

    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        start_min = config.LOOKBACK + 100
        start_max = self.train_end_idx - max_steps - 50
        if start_max <= start_min:
            return None
        
        start_idx = np.random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        
        self.env.reset_reward_states()
        
        current_position = None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        episode_pnl = 0.0  # 에피소드 실현 손익 합계 (로그용)
        trade_count = 0
        prev_unrealized_pnl = 0.0
        
        self.agent.reset_episode_states()
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False)
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx:
                break
            
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
            
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
            
            # Masking 제거: Agent가 포지션 유무에 따라 알아서 판단함
            action, prob, val = self.agent.select_action(state, action_mask=None)
            
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            holding_time_norm = 0.0

            # Action 0: WAIT / HOLD
            if action == 0:
                pass

            # Action 1: LONG (Entry Agent Only)
            elif action == 1:
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = current_idx
                    trade_count += 1

            # Action 2: SHORT (Entry Agent Only)
            elif action == 2:
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = current_idx
                    trade_count += 1

            # Action 3: EXIT (Exit Agent Only)
            elif action == 3:
                if current_position is not None:
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    episode_pnl += realized_pnl
                    holding_time_norm = (current_idx - entry_index) / max_steps
                    current_position = None
                    entry_price = 0.0
                    entry_index = 0
                    trade_count += 1

            reward = self.env.calculate_reward(
                step_pnl=step_pnl,
                realized_pnl=realized_pnl,
                trade_done=trade_done,
                holding_time=holding_time_norm,
                action=action,
                prev_position=prev_pos_str,
                current_position=current_position
            )
            
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
            
            self.agent.put_data((state, action, reward, next_state, prob, done, val))
            episode_reward += reward
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count})
            
            if done:
                break
        
        # 강제 청산 로직 (Episode End Force Close)
        if current_position is not None:
            if current_position == 'LONG':
                realized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                realized_pnl = (entry_price - curr_price) / entry_price
            episode_pnl += realized_pnl
            force_hold_norm = (current_idx - entry_index) / max_steps if current_idx >= entry_index else 0.0
            final_reward = self.env.calculate_reward(
                step_pnl=0.0, realized_pnl=realized_pnl, trade_done=True,
                holding_time=force_hold_norm, action=3,
                prev_position=current_position, current_position=None
            )
            self.agent.put_data((state, 3, final_reward, state, 1.0, True, 0.0))
            episode_reward += final_reward
            trade_count += 1
            
        pbar.close()
        
        # PPO 학습 (Loss 반환)
        loss = self.agent.train_net(episode=episode_num)

        # [추가] TensorBoard에 로그 기록
        if self.writer:
            self.writer.add_scalar('Reward/Total', episode_reward, episode_num)
            self.writer.add_scalar('Metrics/PnL', episode_pnl, episode_num)
            self.writer.add_scalar('Metrics/Trades', trade_count, episode_num)
            self.writer.add_scalar('Loss/Total', loss, episode_num)
        
        return episode_reward, trade_count, episode_pnl

    def train(self, num_episodes=1000):
        logger.info("🚀 Hierarchical PPO 학습 시작 (Entry/Exit Agents)")
        best_reward = -float('inf')
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        best_model = f"{base_path}_best.pth"
        best_scaler = f"{base_path}_best_scaler.pkl"
        last_model = f"{base_path}_last.pth"
        last_scaler = f"{base_path}_last_scaler.pkl"
        
        self.env.preprocessor.save(last_scaler)
        
        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None:
                    continue
                r, c, pnl = res
                self.episode_rewards.append(r)
                avg_r = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_r)
                
                logger.info(f"✅ Ep {ep}: Reward {r:.4f} | Avg {avg_r:.4f} | Trades: {c} | PnL: {pnl:.4f}")
                
                if r > best_reward:
                    best_reward = r
                    logger.info(f"🏆 신기록! ({best_reward:.4f}) -> 저장")
                    self.agent.save_model(best_model)
                    self.env.preprocessor.save(best_scaler)
                
                if ep % 10 == 0:
                    self.agent.save_model(last_model)
                    self.env.preprocessor.save(last_scaler)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ep {ep} Error: {e}")
                continue

        # [추가] TensorBoard Writer 종료
        if self.writer:
            self.writer.close()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)