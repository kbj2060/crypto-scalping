"""
PPO 학습 스크립트 (Action 3 / Info Dim 15 / Force Strategy Recalc)
- 캐시된 전략 파일이 있어도 무조건 재계산하여 0점 문제 해결
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
# 라이브 그래프용 설정
GUI_BACKEND_AVAILABLE = True
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
    GUI_BACKEND_AVAILABLE = False
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from . import config
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import config
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent

from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', handlers=[logging.FileHandler('logs/train_ppo.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

class PPOTrainer:
    def __init__(self, enable_visualization=True):
        self.data_collector = DataCollector(use_saved_data=True)
        # 전략 12개
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]

        logger.info(f"전략 초기화: {len(self.strategies)}개 전략")
        self._load_features()

        # [수정] 무조건 전략 재계산 실행
        self.precalculate_strategies_parallel()

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        state_dim = self.env.get_state_dim()
        action_dim = config.TRAIN_ACTION_DIM

        # info_dim = 15
        info_dim = 15

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"디바이스: {device} | Action Dim: {action_dim} | Info Dim: {info_dim}")

        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)

        # 모델 로드
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth"
        if os.path.exists(last_model_path):
            try:
                self.agent.load_model(last_model_path)
                logger.info(f"✅ 기존 모델 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 모델 로드 실패: {e}")
                logger.warning("🚀 새로운 구조(info_dim=15)로 학습을 처음부터 시작합니다.")

        self.episode_rewards = []
        self.avg_rewards = []
        self.plotting_enabled = True

        # 그래프 초기화
        self._fig = None
        self._ax = None
        self._line1 = None
        self._line2 = None
        if self.plotting_enabled and GUI_BACKEND_AVAILABLE:
            try:
                plt.ion()
                self._fig, self._ax = plt.subplots(figsize=(10, 5))
                self._ax.set_title('PPO Training Progress')
                self._ax.set_xlabel('Episode')
                self._ax.set_ylabel('Reward')
                self._ax.grid(True, alpha=0.3)
                self._line1, = self._ax.plot([], [], label='Reward', alpha=0.3, color='gray')
                self._line2, = self._ax.plot([], [], label='Avg (10)', color='red', linewidth=2)
                self._ax.legend()
            except Exception as e:
                logger.warning(f"Live plot init failed: {e}")
                self._fig = None

    def live_plot(self):
        if not self.plotting_enabled: return
        try:
            x = range(len(self.episode_rewards))
            if self._fig is not None and self._line1 is not None:
                self._line1.set_data(x, self.episode_rewards)
                self._line2.set_data(x, self.avg_rewards)
                self._ax.relim()
                self._ax.autoscale_view()
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                plt.pause(0.01)

            # 파일 저장
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, label='Reward', alpha=0.3, color='gray')
            plt.plot(self.avg_rewards, label='Avg (10)', color='red', linewidth=2)
            plt.title('PPO Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('training_progress.png')
            plt.close()
        except Exception as e:
            logger.warning(f"Plotting error: {e}")

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            # 캐시 파일이 있으면 로드 (0점일 수도 있지만 일단 로드)
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns: df[col] = cached_df[col]
                except: pass

            if df.isnull().values.any(): df = df.fillna(0)
            self.data_collector.eth_data = df

    def precalculate_strategies_parallel(self):
        df = self.data_collector.eth_data
        if df is None: return

        # [수정] Skip 로직 삭제됨 -> 무조건 재계산
        logger.info("⚡ 전략 점수 강제 재계산 중... (시간이 조금 걸립니다)")
        self._precalculate_strategies_sequential(df, config.LOOKBACK+50, len(df))

        # 재계산된 값 저장 (덮어쓰기)
        try:
            cache_path = 'data/cached_strategies.csv'
            strategy_cols = [col for col in df.columns if col.startswith('strategy_')]
            if strategy_cols:
                df[strategy_cols].to_csv(cache_path)
                logger.info(f"💾 전략 점수 캐시 업데이트 완료: {cache_path}")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
        # 초기화
        for i in range(len(self.strategies)):
            if f'strategy_{i}' not in df.columns:
                df[f'strategy_{i}'] = 0.0

        for i in tqdm(range(start_idx, total_len), desc="Calc Strategies"):
            self.data_collector.current_index = i
            for s_idx, strategy in enumerate(self.strategies):
                try:
                    res = strategy.analyze(self.data_collector)
                    score = 0.0
                    if res:
                        conf = float(res.get('confidence', 0.0))
                        sig = res.get('signal', 'NEUTRAL')
                        score = conf if sig == 'LONG' else (-conf if sig == 'SHORT' else 0.0)
                    df.iat[i, df.columns.get_loc(f'strategy_{s_idx}')] = score
                except: continue

    def _fit_global_scaler(self):
        if not self.env.scaler_fitted:
            df = self.data_collector.eth_data
            if df is None: return
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
                if col not in df.columns: df[col] = 0.0
            sample = df.iloc[:train_size].sample(n=min(10000, train_size))[target_cols].values.astype(np.float32)
            self.env.preprocessor.fit(sample)
            self.env.scaler_fitted = True
            path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            self.env.preprocessor.save(path)

    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None: max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        start_min, start_max = config.LOOKBACK + 100, self.train_end_idx - max_steps - 50
        if start_max <= start_min: return None
        start_idx = np.random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        self.env.reset_reward_states()
        current_position, entry_price, entry_index = None, 0.0, 0
        episode_reward, trade_count, prev_unrealized_pnl = 0.0, 0, 0.0
        self.agent.reset_episode_states()

        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False)
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx: break
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])

            unrealized_pnl = (curr_price - entry_price)/entry_price if current_position == 'LONG' else ((entry_price - curr_price)/entry_price if current_position == 'SHORT' else 0.0)
            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0

            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / 1000.0]

            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None: break

            action, prob, val = self.agent.select_action(state, action_mask=[1.0, 1.0, 1.0])
            trade_done, realized_pnl, prev_pos_str = False, 0.0, current_position

            if action == 0:
                if current_position is not None: realized_pnl, trade_done, current_position, trade_count = unrealized_pnl, True, None, trade_count + 1
            elif action == 1:
                if current_position is None: current_position, entry_price, entry_index, trade_count = 'LONG', curr_price, current_idx, trade_count + 1
                elif current_position == 'SHORT': realized_pnl, trade_done, current_position, entry_price, entry_index, trade_count = unrealized_pnl, True, 'LONG', curr_price, current_idx, trade_count + 2
            elif action == 2:
                if current_position is None: current_position, entry_price, entry_index, trade_count = 'SHORT', curr_price, current_idx, trade_count + 1
                elif current_position == 'LONG': realized_pnl, trade_done, current_position, entry_price, entry_index, trade_count = unrealized_pnl, True, 'SHORT', curr_price, current_idx, trade_count + 2

            reward = self.env.calculate_reward(step_pnl, realized_pnl, trade_done, holding_time, action, prev_pos_str, current_position)
            prev_unrealized_pnl = unrealized_pnl if not trade_done else 0.0
            self.data_collector.current_index += 1

            next_idx = self.data_collector.current_index
            if next_idx < len(self.data_collector.eth_data) and next_idx < self.train_end_idx:
                next_price = float(self.data_collector.eth_data.iloc[next_idx]['close'])
                next_pnl = (next_price - entry_price)/entry_price if current_position == 'LONG' else ((entry_price - next_price)/entry_price if current_position == 'SHORT' else 0.0)
                next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
                next_hold = (next_idx - entry_index) if current_position else 0
                next_state = self.env.get_observation(position_info=[next_pos_val, next_pnl * 10, next_hold / 1000.0], current_index=next_idx)
                done = False
            else: done, next_state = True, state

            if next_state is None: done, next_state = True, state

            self.agent.put_data((state, action, reward, next_state, prob, done, val))
            episode_reward += reward
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count})
            if done: break

        pbar.close()
        loss = self.agent.train_net(episode=episode_num)
        return episode_reward, trade_count

    def train(self, num_episodes=1000):
        logger.info("🚀 PPO 학습 시작 (Action 3 Target Position)")
        best_reward = -float('inf')
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        best_model, best_scaler = f"{base_path}_best.pth", f"{base_path}_best_scaler.pkl"
        last_model, last_scaler = f"{base_path}_last.pth", f"{base_path}_last_scaler.pkl"
        self.env.preprocessor.save(last_scaler)

        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None: continue
                r, c = res
                self.episode_rewards.append(r)
                avg_r = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_r)
                logger.info(f"✅ Ep {ep}: Reward {r:.4f} | Avg {avg_r:.4f} | Trades: {c}")

                self.live_plot()

                if r > best_reward:
                    best_reward = r
                    logger.info(f"🏆 신기록! ({best_reward:.4f}) -> 저장")
                    self.agent.save_model(best_model)
                    self.env.preprocessor.save(best_scaler)
                if ep % 10 == 0:
                    self.agent.save_model(last_model)
                    self.env.preprocessor.save(last_scaler)
            except KeyboardInterrupt: break
            except Exception as e: logger.error(f"Ep {ep} Error: {e}"); continue

        if self.plotting_enabled and self._fig is not None:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)
