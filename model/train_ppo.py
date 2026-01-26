"""
PPO 학습 스크립트 (4-Action: HOLD, ENTER_LONG, ENTER_SHORT, EXIT)
- Action 0: HOLD (현 상태 유지)
- Action 1: ENTER_LONG (롱 진입 - 무포지션일 때만)
- Action 2: ENTER_SHORT (숏 진입 - 무포지션일 때만)
- Action 3: EXIT (청산 - 포지션 있을 때만)
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
from model.feature_engineering import FeatureEngineer
from model.mtf_processor import MTFProcessor

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

# 불필요한 로그 생략
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

# 병렬 처리 확인
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.info("joblib 미설치: 순차 처리로 진행합니다.")

def calculate_chunk(start_idx, end_idx, strategies, collector_data):
    """전략 병렬 계산 함수"""
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
            except:
                continue
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
        
        logger.info(f"전략 초기화: {len(self.strategies)}개 전략 사용")
        
        # 1. 데이터 로드 및 전처리
        self._load_features()
        self.precalculate_strategies_parallel()
        
        # 2. 환경 설정
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        # 3. 에이전트 설정 (4-Action)
        state_dim = self.env.get_state_dim()
        # [핵심 수정] Action Dim = 4 (HOLD, LONG, SHORT, EXIT)
        action_dim = 4  
        info_dim = len(self.strategies) + 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"디바이스: {device} | Action Dim: {action_dim} (4-Action Strategy)")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        # 모델 로드 시도
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth"
        if os.path.exists(last_model_path):
            try:
                self.agent.load_model(last_model_path)
                logger.info(f"✅ 기존 모델 로드 완료")
            except Exception as e:
                logger.warning(f"⚠️ 모델 구조 불일치(3->4 변경 등)로 로드 실패: {e}")
                logger.warning("🚀 새로운 구조로 처음부터 학습을 시작합니다.")
        
        self.episode_rewards = []
        self.avg_rewards = []
        
        # 그래프 설정
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_title('PPO Training Progress (4-Action)')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.grid(True, alpha=0.3)
            self.line1, = self.ax.plot([], [], label='Reward', alpha=0.3, color='gray')
            self.line2, = self.ax.plot([], [], label='Avg (10)', color='red', linewidth=2)
            self.ax.legend()
            self.plotting_enabled = True
        except Exception:
            self.plotting_enabled = False

    def _load_features(self):
        """피처 파일 로드"""
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [col for col in cached_df.columns if col.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except Exception:
                    pass
            self.data_collector.eth_data = df
        else:
            logger.error("❌ 피처 파일 없음")
            sys.exit(1)

    def precalculate_strategies_parallel(self):
        """전략 신호 계산"""
        df = self.data_collector.eth_data
        if 'strategy_0' in df.columns: return

        cached_path = 'data/cached_strategies.csv'
        if os.path.exists(cached_path):
            cached_df = pd.read_csv(cached_path, index_col=0, parse_dates=True)
            if len(cached_df) == len(df):
                for col in cached_df.columns:
                    if col.startswith('strategy_'):
                        df[col] = cached_df[col]
                self.data_collector.eth_data = df
                df.to_csv('data/training_features.csv', index=True)
                return

        logger.info("🧠 전략 신호 계산 중...")
        start_idx = config.LOOKBACK + 50
        total_len = len(df)
        
        if JOBLIB_AVAILABLE and total_len > 10000:
            n_jobs = max(1, cpu_count() - 1)
            chunk_size = (total_len - start_idx) // n_jobs
            chunks = [(start_idx + i*chunk_size, start_idx + (i+1)*chunk_size if i < n_jobs-1 else total_len) for i in range(n_jobs)]
            
            results_list = Parallel(n_jobs=n_jobs)(delayed(calculate_chunk)(s, e, self.strategies, df) for s, e in chunks)
            
            for s_idx in range(len(self.strategies)):
                col = f'strategy_{s_idx}'
                df[col] = 0.0
                full_s = np.zeros(total_len)
                for i, (s, e) in enumerate(chunks):
                    full_s[s:e] = results_list[i][col]
                df[col] = full_s
        else:
            self._precalculate_strategies_sequential(df, start_idx, total_len)
            
        df.to_csv('data/training_features.csv', index=True)
        strategy_cols = [c for c in df.columns if c.startswith('strategy_')]
        if strategy_cols:
            df[strategy_cols].to_csv('data/cached_strategies.csv', index=True)
        self.data_collector.eth_data = df

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
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
        """스케일러 학습"""
        if not self.env.scaler_fitted:
            df = self.data_collector.eth_data
            train_size = int(len(df) * config.TRAIN_SPLIT)
            self.train_end_idx = train_size
            
            # Feature extraction logic should ideally be centralized, but here we just fit
            # We assume get_observation handles column checks. 
            # Fitting needs columns to be present.
            # (Simplification: relying on TradingEnvironment's internal column list logic matching this)
            # For robustness, we just use a sample from get_observation in a dry run or similar, 
            # but sticking to previous logic:
            
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            # Ensure columns exist
            for col in target_cols:
                if col not in df.columns: df[col] = 0.0
            
            sample = df.iloc[:train_size].sample(n=min(10000, train_size))[target_cols].values.astype(np.float32)
            self.env.preprocessor.fit(sample)
            self.env.scaler_fitted = True
            
            # Save scaler
            path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            self.env.preprocessor.save(path)

    def train_episode(self, episode_num, max_steps=None):
        """에피소드 학습 (4-Action Logic 적용)"""
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        start_min = config.LOOKBACK + 100
        start_max = self.train_end_idx - max_steps - 50
        if start_max <= start_min: return None
        
        start_idx = np.random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        
        current_position = None  # None, 'LONG', 'SHORT'
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        trade_count = 0
        prev_unrealized_pnl = 0.0
        
        self.agent.reset_episode_states()
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False, unit="step")
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx: break
            
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
            
            # 1. 평가 손익 계산
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0
            
            # 2. 상태 관측
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]
            
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None: break

            prev_pos_str = current_position 

            # -------------------------------------------------------
            # [핵심] Action Mask 생성 (락업 제거, 스위칭 차단 유지)
            # -------------------------------------------------------
            # 기본적으로 모든 액션 허용 [HOLD, LONG, SHORT, EXIT]
            action_mask = [1.0, 1.0, 1.0, 1.0]
            
            # 락업 제거: 5캔들 락업 로직 삭제
            # 스위칭 차단 유지: 포지션이 있을 때 반대 방향 진입 금지
            if current_position == 'LONG':
                # HOLD(0), LONG(1), EXIT(3) 허용 / SHORT(2) 금지 (스위칭 차단)
                action_mask = [1.0, 1.0, 0.0, 1.0]
            elif current_position == 'SHORT':
                # HOLD(0), SHORT(2), EXIT(3) 허용 / LONG(1) 금지 (스위칭 차단)
                action_mask = [1.0, 0.0, 1.0, 1.0]
            # 포지션이 없을 때는 EXIT(3) 금지
            else:
                # HOLD(0), LONG(1), SHORT(2) 허용 / EXIT(3) 금지
                action_mask = [1.0, 1.0, 1.0, 0.0]
            # -------------------------------------------------------

            # 3. 행동 선택 (마스크 전달)
            action, prob = self.agent.select_action(state, action_mask=action_mask)
            
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            extra_penalty = 0.0
            
            # A. 강제 손절 (Safety Net)
            if current_position is not None and unrealized_pnl < config.STOP_LOSS_THRESHOLD:
                realized_pnl = unrealized_pnl
                trade_done = True
                current_position = None
                entry_price = 0.0
                entry_index = 0
                trade_count += 1
                # 손절 직후에는 강제로 HOLD 처리하여 연속 진입 방지
                action = 0 
            
            # B. 4-Action Strict State Machine
            else:
                # ---------------------------------------------------
                # Action 0: HOLD
                # ---------------------------------------------------
                if action == 0:
                    pass # 현 상태 유지

                # ---------------------------------------------------
                # Action 1: ENTER_LONG (진입 전용)
                # ---------------------------------------------------
                elif action == 1:
                    if current_position is None:
                        current_position = 'LONG'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                    else:
                        # 이미 포지션 있으면 무시 (스위칭 불가 -> EXIT 먼저 해야 함)
                        pass

                # ---------------------------------------------------
                # Action 2: ENTER_SHORT (진입 전용)
                # ---------------------------------------------------
                elif action == 2:
                    if current_position is None:
                        current_position = 'SHORT'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                    else:
                        pass # 무시

                # ---------------------------------------------------
                # Action 3: EXIT (청산 전용)
                # ---------------------------------------------------
                elif action == 3:
                    if current_position is not None:
                        realized_pnl = unrealized_pnl
                        trade_done = True
                        current_position = None
                        entry_price = 0.0
                        entry_index = 0
                        trade_count += 1
                    else:
                        pass # 무시

            # 리워드 계산 (4-Action 대응)
            reward = self.env.calculate_reward(
                step_pnl=step_pnl, 
                realized_pnl=realized_pnl, 
                trade_done=trade_done, 
                action=action,              
                prev_position=prev_pos_str,
                current_position=current_position
            )
            
            reward += extra_penalty
            
            # 다음 스텝 준비
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
        logger.info("🚀 PPO 학습 시작 (4-Action: HOLD, ENTER_LONG, ENTER_SHORT, EXIT)")
        
        best_reward = -float('inf')
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        
        # 파일 경로 설정
        best_model = f"{base_path}_best.pth"
        best_scaler = f"{base_path}_best_scaler.pkl"
        last_model = f"{base_path}_last.pth"
        last_scaler = f"{base_path}_last_scaler.pkl"
        
        self.env.preprocessor.save(last_scaler)
        
        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None: continue
                
                reward, count = res
                self.episode_rewards.append(reward)
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_reward)
                
                logger.info(f"✅ Ep {ep}: Reward {reward:.4f} | Avg {avg_reward:.4f} | Trades: {count}")
                self.live_plot()
                
                if reward > best_reward:
                    best_reward = reward
                    logger.info(f"🏆 신기록! ({best_reward:.4f}) -> 저장")
                    self.agent.save_model(best_model)
                    self.env.preprocessor.save(best_scaler)
                
                if ep % 10 == 0:
                    self.agent.save_model(last_model)
                    self.env.preprocessor.save(last_scaler)
                    
            except KeyboardInterrupt:
                logger.info("학습 중단")
                break
            except Exception as e:
                logger.error(f"Ep {ep} Error: {e}")
                continue
        
        if self.plotting_enabled:
            plt.ioff(); plt.show()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)