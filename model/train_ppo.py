"""
PPO 학습 스크립트 (4-Action + No Force Close)
- 에피소드 종료 시 포지션 유지 (강제 청산 안 함)
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
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

try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.info("joblib 미설치: 순차 처리로 진행합니다.")

def calculate_chunk(start_idx, end_idx, strategies, collector_data):
    from core import DataCollector
    temp_collector = DataCollector(use_saved_data=True)
    temp_collector.eth_data = collector_data
    results = {}
    for s_idx in range(len(strategies)):
        results[f'strategy_{s_idx}'] = np.zeros(end_idx - start_idx)

    for i in range(start_idx, end_idx):
        temp_collector.current_index = i
        rel_i = i - start_idx
        for s_idx, strategy in enumerate(strategies):
            try:
                result = strategy.analyze(temp_collector)
                score = 0.0
                if result:
                    conf = float(result.get('confidence', 0.0))
                    signal = result.get('signal', 'NEUTRAL')
                    if signal == 'LONG': score = conf
                    elif signal == 'SHORT': score = -conf
                results[f'strategy_{s_idx}'][rel_i] = score
            except: continue
    return results

class PPOTrainer:
    def __init__(self, enable_visualization=True):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        logger.info(f"전략 초기화: {len(self.strategies)}개 전략")
        
        self._load_features()
        self.precalculate_strategies_parallel()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        state_dim = self.env.get_state_dim()
        action_dim = 4  # 4-Action: HOLD, LONG, SHORT, EXIT
        info_dim = len(self.strategies) + 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"디바이스: {device} | Action Dim: {action_dim} (4-Action: HOLD, LONG, SHORT, EXIT)")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth"
        if os.path.exists(last_model_path):
            try:
                self.agent.load_model(last_model_path)
                logger.info(f"✅ 기존 모델 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 모델 로드 실패: {e}")
                logger.warning("🚀 새로운 구조로 학습을 시작합니다.")
        
        self.episode_rewards = []
        self.avg_rewards = []
        
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_title('PPO Training Progress')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.grid(True, alpha=0.3)
            self.line1, = self.ax.plot([], [], label='Reward', alpha=0.3, color='gray')
            self.line2, = self.ax.plot([], [], label='Avg (10)', color='red', linewidth=2)
            self.ax.legend()
            self.plotting_enabled = True
        except: self.plotting_enabled = False

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # [수정] 데이터 품질 개선 (Forward Fill)
            df = df.ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns: df[col] = cached_df[col]
                except: pass
            self.data_collector.eth_data = df
        else:
            logger.warning("⚠️ 피처 파일이 없습니다. 원본 데이터로 초기화하고 새로 계산합니다.")
            # 파일이 없으면 precalculate 단계에서 생성되도록 둠
            pass

    def precalculate_strategies_parallel(self):
        df = self.data_collector.eth_data
        if df is None: return
        if 'strategy_0' in df.columns: return
        self._precalculate_strategies_sequential(df, config.LOOKBACK+50, len(df))

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
        for i in range(len(self.strategies)): df[f'strategy_{i}'] = 0.0
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
            
            # [수정] 오직 Train Set만 사용하여 스케일러 학습 (누수 방지)
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
        
        start_min = config.LOOKBACK + 100
        start_max = self.train_end_idx - max_steps - 50
        if start_max <= start_min: return None
        
        start_idx = np.random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        
        current_position = None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        trade_count = 0
        prev_unrealized_pnl = 0.0
        
        self.agent.reset_episode_states()
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False)
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx: break
            
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
            if state is None: break

            prev_pos_str = current_position 

            action, prob = self.agent.select_action(state)
            
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            extra_penalty = 0.0
            
            # [삭제] A. 강제 손절 (Safety Net)
            # 이 부분이 있으면 AI는 "버티면 시스템이 알아서 끊어주네?"라고 오해합니다.
            # if current_position is not None and unrealized_pnl < config.STOP_LOSS_THRESHOLD:
            #     realized_pnl = unrealized_pnl
            #     trade_done = True
            #     current_position = None
            #     entry_price = 0.0; entry_index = 0
            #     trade_count += 1
            #     action = 0 
            
            # B. 4-Action Logic
            # Action 0: HOLD (관망)
            if action == 0:
                pass  # 아무것도 하지 않음
            
            # Action 1: LONG (롱 진입/유지)
            elif action == 1:
                if current_position is None:
                    # 신규 롱 진입
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = current_idx
                    trade_count += 1
                # 이미 LONG이면 유지 (Pass)
            
            # Action 2: SHORT (숏 진입/유지)
            elif action == 2:
                if current_position is None:
                    # 신규 숏 진입
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = current_idx
                    trade_count += 1
                # 이미 SHORT면 유지 (Pass)
            
            # Action 3: EXIT (청산)
            elif action == 3:
                if current_position is not None:
                    # 포지션 청산
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    current_position = None
                    entry_price = 0.0
                    entry_index = 0
                    trade_count += 1
                # 포지션이 없으면 아무것도 안 함 (Pass)

            # [수정] 마지막에 강제 청산하는 로직 없음 (사용자 요청)
            # 그냥 루프가 끝나면 포지션 들고 있는 상태로 종료됨

            reward = self.env.calculate_reward(
                step_pnl=step_pnl, 
                realized_pnl=realized_pnl, 
                trade_done=trade_done,  # <-- 여기서 True로 넘어가야 종료 보상(수수료 차감 등)이 계산됨
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
            if next_state is None: done = True; next_state = state
            
            self.agent.put_data((state, action, reward, next_state, prob, done))
            episode_reward += reward
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count})
            
            if done: break
        
        pbar.close()
        loss = self.agent.train_net(episode=episode_num)
        return episode_reward, trade_count

    def live_plot(self):
        if not self.plotting_enabled: return
        try:
            x = range(len(self.episode_rewards))
            self.line1.set_data(x, self.episode_rewards)
            self.line2.set_data(x, self.avg_rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except: pass

    def train(self, num_episodes=1000):
        logger.info("🚀 PPO 학습 시작 (4-Action: HOLD, LONG, SHORT, EXIT + No Force Close)")
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
        
        if self.plotting_enabled: plt.ioff(); plt.show()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)