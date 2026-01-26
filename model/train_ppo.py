"""
PPO 학습 스크립트 (최종 버전: 3-Action Target Position)
- Action 0: NEUTRAL (청산 또는 관망)
- Action 1: LONG (진입 또는 스위칭 또는 홀딩)
- Action 2: SHORT (진입 또는 스위칭 또는 홀딩)
- 개선사항: 1-Step 스위칭 지원, LSTM State Reset 적용, 데이터 누수 방지(Scaler)
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
from strategies import *
from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler('logs/train_ppo.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 병렬 처리 함수
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
                    sig = result.get('signal', 'NEUTRAL')
                    if sig == 'LONG': score = conf
                    elif sig == 'SHORT': score = -conf
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
        
        # 1. 데이터 로드 (Forward Fill 적용)
        self._load_features()
        self.precalculate_strategies_parallel()
        
        # 2. 환경 설정
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        # 3. 에이전트 설정 (3-Action Target)
        state_dim = self.env.get_state_dim()
        action_dim = 3  # 0:Neutral, 1:Long, 2:Short
        info_dim = len(self.strategies) + 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Setting: Device={device} | Action Dim={action_dim} (3-Action Target)")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        # 모델 로드
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth"
        if os.path.exists(last_model_path):
            try:
                self.agent.load_model(last_model_path)
                logger.info("✅ 기존 모델 로드 완료")
            except:
                logger.warning("⚠️ 구조 변경 감지 -> 새 모델 학습 시작")
        
        self.episode_rewards = []
        self.avg_rewards = []
        
        try:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.plotting_enabled = True
        except: self.plotting_enabled = False

    def _load_features(self):
        """
        피처 파일 로드 또는 생성
        """
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        # 1. 파일이 있으면 로드
        if os.path.exists(path):
            logger.info(f"📂 기존 피처 파일 로드: {path}")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Forward Fill로 결측치 처리 (Data Quality)
            df = df.ffill().bfill()
            
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns: df[col] = cached_df[col]
                    logger.info("📂 캐시된 전략 신호 병합 완료")
                except: pass
                
            self.data_collector.eth_data = df
            
        # 2. 파일이 없으면 -> 원본 데이터로 초기화 (새로 계산 준비)
        else:
            logger.warning("⚠️ 피처 파일이 없습니다. 원본 데이터로 초기화하고 새로 계산합니다.")
            # 원본 데이터가 data_collector에 이미 로드되어 있다고 가정
            if self.data_collector.eth_data is None:
                logger.error("❌ 원본 데이터(ETH)도 로드되지 않았습니다. collect_training_data.py를 먼저 실행하세요.")
                sys.exit(1)
            
            # 피처 엔지니어링 수행 (새로 계산)
            # (이 부분은 precalculate_strategies_parallel 에서 수행되거나, 
            #  여기서 명시적으로 feature engineering을 호출해야 할 수 있음)
            #  일단은 빈 상태로 두고 뒤에서 계산하도록 패스
            pass

    def precalculate_strategies_parallel(self):
        """
        [Critical Fix] 전략 신호 계산 시 Look-ahead Bias 차단
        - 전체 데이터가 아닌, 오직 Train Set 구간만 계산합니다.
        - Test Set 구간은 0.0으로 남겨두어 물리적으로 접근 불가능하게 만듭니다.
        """
        df = self.data_collector.eth_data
        if 'strategy_0' in df.columns: return

        logger.info("🧠 전략 신호 계산 중... (Only Train Set)")
        
        # [수정] 전체 길이가 아니라, 학습 데이터 구간까지만 계산
        total_len = len(df)
        train_end_idx = int(total_len * config.TRAIN_SPLIT)
        
        start_idx = config.LOOKBACK + 50
        
        # Joblib 병렬 처리
        if JOBLIB_AVAILABLE and train_end_idx > 10000:
            from multiprocessing import cpu_count
            n_jobs = max(1, cpu_count() - 1)
            # 청크를 나눌 때 train_end_idx를 끝으로 설정
            chunk_size = (train_end_idx - start_idx) // n_jobs
            chunks = [(start_idx + i*chunk_size, start_idx + (i+1)*chunk_size if i < n_jobs-1 else train_end_idx) for i in range(n_jobs)]
            
            results_list = Parallel(n_jobs=n_jobs)(delayed(calculate_chunk)(s, e, self.strategies, df) for s, e in chunks)
            
            for s_idx in range(len(self.strategies)):
                col = f'strategy_{s_idx}'
                df[col] = 0.0  # 초기화 (Test Set은 0으로 유지됨)
                
                # 계산된 Train Set 구간만 채워넣기
                full_s = np.zeros(total_len)
                for i, (s, e) in enumerate(chunks):
                    # 청크 범위만큼만 업데이트
                    chunk_len = e - s
                    # results_list[i][col]의 길이가 chunk_len과 일치하는지 확인
                    if len(results_list[i][col]) == chunk_len:
                        full_s[s:e] = results_list[i][col]
                
                df[col] = full_s
        else:
            # 병렬 처리 불가 시 순차 처리 (범위 제한 적용)
            self._precalculate_strategies_sequential(df, start_idx, train_end_idx)
            
        df.to_csv('data/training_features.csv', index=True)
        # 캐싱도 수행
        strategy_cols = [c for c in df.columns if c.startswith('strategy_')]
        if strategy_cols:
            df[strategy_cols].to_csv('data/cached_strategies.csv', index=True)
            
        self.data_collector.eth_data = df

    def _precalculate_strategies_sequential(self, df, start_idx, end_idx):
        """순차 계산 (Train Set 구간 제한 적용됨)"""
        # 컬럼 초기화
        for i in range(len(self.strategies)): 
            if f'strategy_{i}' not in df.columns:
                df[f'strategy_{i}'] = 0.0
            
        # [핵심] end_idx까지만 루프를 돔 (Train Set 이후는 계산 안 함)
        for i in tqdm(range(start_idx, end_idx), desc="Calc Strategies (Train Only)"):
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
        """[Critical] Look-ahead Bias 방지: 오직 Train Set으로만 Scaler 학습"""
        if not self.env.scaler_fitted:
            df = self.data_collector.eth_data
            
            # config.TRAIN_SPLIT 엄격 적용
            train_size = int(len(df) * config.TRAIN_SPLIT)
            self.train_end_idx = train_size
            
            train_df = df.iloc[:train_size].copy()
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            for col in target_cols:
                if col not in train_df.columns: train_df[col] = 0.0
            
            # 랜덤 샘플링도 Train 데이터 내에서만
            sample = train_df[target_cols].sample(n=min(20000, len(train_df))).values.astype(np.float32)
            self.env.preprocessor.fit(sample)
            self.env.scaler_fitted = True
            
            path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            self.env.preprocessor.save(path)

    def train_episode(self, episode_num, max_steps=None):
        """에피소드 학습 (3-Action Target Position + State Reset)"""
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
        
        # [중요] 에피소드 시작 시 State Reset
        self.agent.reset_episode_states()
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False)
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx: break
            
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
            
            # PnL 계산
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0
            
            # 상태 관측
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]
            
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None: break

            prev_pos_str = current_position 

            # 행동 선택 (3-Action)
            action, prob = self.agent.select_action(state)
            
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            
            # -----------------------------------------------------------
            # 3-Action Logic (Target Position)
            # -----------------------------------------------------------
            
            # Action 0: Neutral (청산 또는 관망)
            if action == 0:
                if current_position is not None:
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    current_position = None
                    entry_price = 0.0; entry_index = 0
                    trade_count += 1
            
            # Action 1: Long (진입 또는 스위칭 또는 홀딩)
            elif action == 1:
                if current_position == 'SHORT': # [즉시 스위칭]
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    # 스위칭 시 즉시 재진입
                    current_position = 'LONG'
                    entry_price = curr_price; entry_index = current_idx
                    trade_count += 1
                elif current_position is None: # 진입
                    current_position = 'LONG'
                    entry_price = curr_price; entry_index = current_idx
                    trade_count += 1
                # 이미 LONG이면 Pass (홀딩)

            # Action 2: Short (진입 또는 스위칭 또는 홀딩)
            elif action == 2:
                if current_position == 'LONG': # [즉시 스위칭]
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    # 스위칭 시 즉시 재진입
                    current_position = 'SHORT'
                    entry_price = curr_price; entry_index = current_idx
                    trade_count += 1
                elif current_position is None: # 진입
                    current_position = 'SHORT'
                    entry_price = curr_price; entry_index = current_idx
                    trade_count += 1
                # 이미 SHORT면 Pass (홀딩)

            # 리워드 계산
            reward = self.env.calculate_reward(
                step_pnl=step_pnl, 
                realized_pnl=realized_pnl, 
                trade_done=trade_done, 
                action=action,              
                prev_position=prev_pos_str,
                current_position=current_position
            )
            
            # [수정] 거래 종료 시 LSTM 상태 초기화 (독립성 보장)
            if trade_done:
                self.agent.reset_episode_states()

            # 다음 스텝
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
            self.ax.relim(); self.ax.autoscale_view()
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
            plt.pause(0.01)
        except: pass

    def train(self, num_episodes=1000):
        logger.info("🚀 PPO 학습 시작 (3-Action Target + State Reset)")
        best_reward = -float('inf')
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        
        self.env.preprocessor.save(f"{base_path}_last_scaler.pkl")
        
        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None: continue
                r, c = res
                self.episode_rewards.append(r)
                avg_r = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_r)
                logger.info(f"✅ Ep {ep}: R {r:.2f} | Avg {avg_r:.2f} | Tr {c}")
                self.live_plot()
                if r > best_reward:
                    best_reward = r
                    self.agent.save_model(f"{base_path}_best.pth")
                    self.env.preprocessor.save(f"{base_path}_best_scaler.pkl")
                if ep % 10 == 0:
                    self.agent.save_model(f"{base_path}_last.pth")
            except KeyboardInterrupt: break
            except Exception as e: logger.error(f"Err: {e}"); continue
        if self.plotting_enabled: plt.ioff(); plt.show()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)