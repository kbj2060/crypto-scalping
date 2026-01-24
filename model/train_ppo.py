"""
PPO 모델 학습 스크립트 (DQN 스타일 데이터 캐싱 적용)
- 피처 데이터를 CSV로 저장하여 재사용 (속도 최적화)
- 학습 시작 시 매번 계산하지 않고 로드만 수행
"""
import logging
import os
import sys
import time
import numpy as np
import pandas as pd

# 시각화 모듈 (선택적)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy,
    CCIReversalStrategy, WilliamsRStrategy  # [추가] 이 2개가 빠져있었습니다!
)

# AI 강화학습 모듈
try:
    import torch
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent
    from model.feature_engineering import FeatureEngineer
    from model.mtf_processor import MTFProcessor
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ AI 모듈 로드 실패: {e}")
    sys.exit(1)

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 피처 엔지니어링 로그 끄기 (WARNING 이상만 출력)
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)


class LiveVisualizer:
    """학습 리워드를 실시간으로 그래프화하는 클래스"""
    def __init__(self, window_size=10):
        if not MATPLOTLIB_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.rewards = []
        self.moving_avg = []
        self.window_size = window_size
        
        self.ax.set_title("Real-time Training Performance")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.line1, = self.ax.plot([], [], label='Episode Reward', alpha=0.3, color='blue')
        self.line2, = self.ax.plot([], [], label=f'Moving Avg ({window_size})', color='red', linewidth=2)
        self.ax.legend()
        self.ax.grid(True)

    def update(self, reward):
        if not self.enabled: return
        self.rewards.append(reward)
        if len(self.rewards) >= self.window_size:
            avg = np.mean(self.rewards[-self.window_size:])
        else:
            avg = np.mean(self.rewards)
        self.moving_avg.append(avg)
        
        x = np.arange(len(self.rewards))
        self.line1.set_data(x, self.rewards)
        self.line2.set_data(x, self.moving_avg)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)


class PPOTrainer:
    """PPO 모델 학습 클래스"""
    def __init__(self, enable_visualization=False):
        # 1. 데이터 수집기 초기화
        self.data_collector = DataCollector(use_saved_data=True)
        
        # 2. 전략 초기화 (12개 전략 완전체)
        self.breakout_strategies = []
        self.range_strategies = []
        
        # 폭발장 전략
        if config.STRATEGIES.get('btc_eth_correlation', False):
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES.get('volatility_squeeze', False):
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES.get('orderblock_fvg', False):
            self.breakout_strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('hma_momentum', False):
            self.breakout_strategies.append(HMAMomentumStrategy())
        if config.STRATEGIES.get('mfi_momentum', False):
            self.breakout_strategies.append(MFIMomentumStrategy())
        
        # [추가] CCI 반전 전략 (폭발/추세용)
        # config에 키가 없다면 기본적으로 추가하거나 config.py 확인 필요
        self.breakout_strategies.append(CCIReversalStrategy())
        
        # 횡보장 전략
        if config.STRATEGIES.get('bollinger_mean_reversion', False):
            self.range_strategies.append(BollingerMeanReversionStrategy())
        if config.STRATEGIES.get('vwap_deviation', False):
            self.range_strategies.append(VWAPDeviationStrategy())
        if config.STRATEGIES.get('range_top_bottom', False):
            self.range_strategies.append(RangeTopBottomStrategy())
        if config.STRATEGIES.get('stoch_rsi_mean_reversion', False):
            self.range_strategies.append(StochRSIMeanReversionStrategy())
        if config.STRATEGIES.get('cmf_divergence', False):
            self.range_strategies.append(CMFDivergenceStrategy())
        
        # [추가] Williams %R 전략 (횡보/반전용)
        self.range_strategies.append(WilliamsRStrategy())
        
        # 전체 합치기 (총 12개)
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"전략 초기화 완료: {len(self.strategies)}개 전략 (목표: 12개)")
        
        # 3. [핵심] 피처 데이터 로드 또는 생성 (DQN 스타일)
        # 파일이 있으면 로드하고, 없으면 생성 후 저장합니다.
        self._load_or_create_features()
        
        # 4. 환경 생성
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        
        # 5. 스케일러 학습 (로드된 데이터 사용)
        self._fit_global_scaler()
        
        # 6. 에이전트 생성
        state_dim = self.env.get_state_dim() # 29
        # info_dim = 전략 점수 개수 + 포지션 정보 개수 (3개)
        info_dim = len(self.strategies) + 3  # 12개 전략 + 3개 포지션 정보 = 15
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"디바이스: {device}")
        logger.info(f"정보 차원: {info_dim} (전략 {len(self.strategies)}개 + 포지션 정보 3개)")
        self.agent = PPOAgent(state_dim, action_dim=3, hidden_dim=128, device=device, info_dim=info_dim)
        
        # 모델 로드
        if os.path.exists(config.AI_MODEL_PATH):
            try:
                self.agent.load_model(config.AI_MODEL_PATH)
                logger.info(f"✅ 기존 모델 로드 완료")
            except:
                logger.info("새 모델로 시작")
        
        # 상태 변수
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        self.prev_pnl = 0.0
        self.episode_rewards = []
        self.total_steps = 0
        
        if enable_visualization:
            self.visualizer = LiveVisualizer()
        else:
            self.visualizer = None

    def _load_or_create_features(self):
        """
        [DQN 스타일] 피처 캐싱 시스템
        1. data/training_features.csv 확인
        2. 있으면 -> 로드 (초고속)
        3. 없으면 -> 계산 후 저장 (최초 1회)
        """
        feature_file_path = 'data/training_features.csv'
        
        # 1. 캐시 파일 확인 및 로드
        if os.path.exists(feature_file_path):
            logger.info(f"📂 캐시된 피처 파일 발견: {feature_file_path}")
            logger.info("⚡ 피처 엔지니어링을 건너뛰고 데이터를 로드합니다...")
            
            try:
                # CSV 로드 (인덱스는 timestamp로 지정)
                # parse_dates=True로 날짜 형식 자동 변환
                df = pd.read_csv(feature_file_path, index_col=0, parse_dates=True)
                
                # 데이터 교체
                self.data_collector.eth_data = df
                logger.info(f"✅ 데이터 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
                return
                
            except Exception as e:
                logger.error(f"파일 로드 중 오류 발생 (재생성합니다): {e}")
        
        # 2. 파일이 없으면 새로 생성
        logger.info("🚀 피처 파일이 없습니다. 새로 생성을 시작합니다 (최초 1회 수행)...")
        
        eth_data = self.data_collector.eth_data
        btc_data = self.data_collector.btc_data
        
        if eth_data is None or len(eth_data) == 0:
            logger.error("원본 데이터가 없습니다.")
            return

        # 인덱스 정리
        if not isinstance(eth_data.index, pd.DatetimeIndex):
            if 'timestamp' in eth_data.columns:
                eth_data.index = pd.to_datetime(eth_data['timestamp'])
            else:
                eth_data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(eth_data), freq='3min')
        
        if btc_data is not None and not isinstance(btc_data.index, pd.DatetimeIndex):
             if 'timestamp' in btc_data.columns:
                btc_data.index = pd.to_datetime(btc_data['timestamp'])

        # Feature Engineering
        fe = FeatureEngineer(eth_data, btc_data)
        df = fe.generate_features()
        
        if df is None: return
            
        # MTF Processing
        mtf = MTFProcessor(df)
        df = mtf.add_mtf_features()
        
        # [중요] CSV로 저장 (나중을 위해)
        os.makedirs('data', exist_ok=True)
        df.to_csv(feature_file_path, index=True)
        
        # 메모리에 적용
        self.data_collector.eth_data = df
        logger.info(f"💾 피처 계산 및 저장 완료: {feature_file_path}")

    def _fit_global_scaler(self):
        """전역 스케일러 학습 (데이터 누수 방지 적용)"""
        try:
            logger.info("🚀 전역 스케일러 학습 시작 (Data Leakage 방지 적용)...")
            df = self.data_collector.eth_data
            
            if df is None or len(df) == 0:
                logger.warning("데이터가 없습니다.")
                return

            # 1. 시계열 데이터 분할 (Time Series Split)
            total_len = len(df)
            train_end = int(total_len * config.TRAIN_SPLIT)
            val_end = int(total_len * config.VAL_SPLIT)
            
            # 나중에 쓰기 위해 저장
            self.train_end_idx = train_end
            self.val_end_idx = val_end
            
            logger.info(f"데이터 분할: Train(~{train_end}), Val(~{val_end}), Test(~{total_len})")

            # 2. 학습 데이터만 추출
            train_df = df.iloc[:train_end].copy()
            
            # 사용할 29개 컬럼 정의 (DQN과 동일)
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            # 없는 컬럼 채우기 & 순서 보장
            missing_cols = [c for c in target_cols if c not in train_df.columns]
            if missing_cols:
                for c in missing_cols:
                    train_df[c] = 0.0
            
            # 샘플링 (Train 데이터 내에서만, 최대 5만개)
            sample_size = min(50000, len(train_df))
            sampled_df = train_df.sample(n=sample_size)[target_cols]
            
            # 스케일러 학습
            data_array = sampled_df.values.astype(np.float32)
            self.env.preprocessor.fit(data_array)
            self.env.scaler_fitted = True
            
            # 스케일러 저장
            scaler_path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            if not scaler_path.endswith('.pkl'):
                scaler_path = config.AI_MODEL_PATH + '_scaler.pkl'
            self.env.preprocessor.save_scaler(scaler_path, feature_names=target_cols)
            
            logger.info("✅ 학습 데이터 기반 스케일러 학습 완료")
            
        except Exception as e:
            logger.error(f"스케일러 학습 실패: {e}", exc_info=True)

    def train_episode(self, episode_num, max_steps=480):
        """
        [수정] 안정적인 에피소드 학습 루프
        - 학습 데이터 구간(0 ~ train_end_idx) 내에서만 랜덤 시작
        - 인덱스 경계 체크 강화
        """
        # 학습 구간 설정
        if not hasattr(self, 'train_end_idx'):
            self.train_end_idx = int(len(self.data_collector.eth_data) * 0.7)
            
        # 시작 가능한 인덱스 범위 (Lookback 확보 ~ 학습구간 끝 - 에피소드 길이)
        start_min = self.env.lookback + 50
        start_max = self.train_end_idx - max_steps - 50
        
        if start_max <= start_min:
            logger.error("학습 데이터 구간이 너무 짧습니다.")
            return None

        # 랜덤 시작점 선택
        import random
        start_idx = random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        
        # 상태 초기화
        self.prev_pnl = 0.0
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        
        # [개선 1] 에피소드 시작 시 LSTM 상태 초기화
        self.agent.reset_episode_states()
        
        episode_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            current_idx = self.data_collector.current_index
            
            # 1. 인덱스 초과 안전장치
            if current_idx >= self.train_end_idx:  # 학습 구간 넘어가면 종료
                break
                
            # 2. 관측 정보 생성
            pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
            holding_time_idx = (current_idx - self.entry_index) if self.entry_index is not None else 0
            hold_val = holding_time_idx / max_steps
            pnl_val = self.prev_pnl * 10
            pos_info = [pos_val, pnl_val, hold_val]
            
            # get_observation 호출 (mask 포함)
            state = self.env.get_observation(
                position_info=pos_info,
                current_index=current_idx,
                entry_index=self.entry_index,
                current_position=self.current_position
            )
            
            if state is None:
                break
            
            # 3. 행동 선택
            action, log_prob = self.agent.select_action(state)
            
            # 4. 가격 데이터 및 보상 계산
            try:
                current_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
                
                # --- 보상 계산 및 포지션 로직 ---
                reward = 0.0
                trade_done = False
                current_pnl = 0.0
                pnl_change = 0.0
                
                # 🛑 [추가] 강제 손절 (Hard Stop Loss) - 2% 손실 시 무조건 청산
                # 포지션이 있을 때만 체크
                if self.current_position is not None:
                    if self.current_position == 'LONG':
                        current_pnl = (current_price - self.entry_price) / self.entry_price
                    elif self.current_position == 'SHORT':
                        current_pnl = (self.entry_price - current_price) / self.entry_price
                    
                    stop_loss_threshold = config.STOP_LOSS_THRESHOLD
                    
                    if current_pnl < stop_loss_threshold:
                        # 강제 청산 로직 실행
                        reward = self.env.calculate_reward(current_pnl, True, 0, 0)
                        # 손절은 뼈아프게 페널티 추가
                        reward -= 1.0
                        
                        trade_done = True
                        logger.info(f"🛑 손절 발동: 수익률 {current_pnl:.2%}, 가격: ${current_price:.2f}")
                        
                        # 포지션 초기화
                        self.current_position = None
                        self.entry_price = None
                        self.entry_index = None
                        self.prev_pnl = 0.0
                        
                        # 트랜지션 저장 및 다음 스텝으로
                        # state는 (obs_seq, obs_info, mask) 튜플이므로 앞 2개만 저장
                        state_to_store = (state[0], state[1])
                        is_terminal = (step == max_steps - 1)
                        self.agent.store_transition(state_to_store, action, log_prob, reward, is_terminal)
                        episode_reward += reward
                        steps += 1
                        self.data_collector.current_index += 1
                        continue  # 이번 스텝 종료
                
                if action == 1: # LONG
                    if self.current_position != 'LONG':
                        if self.current_position == 'SHORT': # 청산
                            pnl = (self.entry_price - current_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl
                            reward = self.env.calculate_reward(pnl, True, 0, pnl_change)
                            trade_done = True
                            logger.info(f"💰 숏 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}, 가격: ${current_price:.2f}")
                            self.prev_pnl = 0.0
                        # 롱 진입
                        self.current_position = 'LONG'
                        self.entry_price = current_price
                        self.entry_index = current_idx
                        self.prev_pnl = 0.0
                        logger.debug(f"📈 롱 진입: ${current_price:.2f} (인덱스: {self.entry_index})")
                elif action == 2: # SHORT
                    if self.current_position != 'SHORT':
                        if self.current_position == 'LONG': # 청산
                            pnl = (current_price - self.entry_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl
                            reward = self.env.calculate_reward(pnl, True, 0, pnl_change)
                            trade_done = True
                            logger.info(f"💰 롱 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}, 가격: ${current_price:.2f}")
                            self.prev_pnl = 0.0
                        # 숏 진입
                        self.current_position = 'SHORT'
                        self.entry_price = current_price
                        self.entry_index = current_idx
                        self.prev_pnl = 0.0
                        logger.debug(f"📉 숏 진입: ${current_price:.2f} (인덱스: {self.entry_index})")
                else: # HOLD
                    if self.current_position:
                        if self.current_position == 'LONG':
                            current_pnl = (current_price - self.entry_price) / self.entry_price
                        else:
                            current_pnl = (self.entry_price - current_price) / self.entry_price
                        pnl_change = current_pnl - self.prev_pnl
                        holding_time = current_idx - self.entry_index
                        reward = self.env.calculate_reward(current_pnl, False, holding_time, pnl_change)
                        self.prev_pnl = current_pnl
                    else:
                        # [수정] 무포지션일 때 관망 보상
                        # 기존: reward = -0.0001 (관망하면 벌점)
                        # 변경: reward = 0.0 (관망은 본전)
                        reward = 0.0
                
                # 5. 트랜지션 저장 (Mask 제외하고 저장)
                # state는 (obs_seq, obs_info, mask) 튜플이므로 앞 2개만 저장
                state_to_store = (state[0], state[1])
                is_terminal = (step == max_steps - 1)
                
                self.agent.store_transition(state_to_store, action, log_prob, reward, is_terminal)
                episode_reward += reward
                steps += 1
                
                # 6. 배치 업데이트
                if len(self.agent.memory) >= config.TRAIN_BATCH_SIZE:
                    # Next State (Bootstrap용)
                    next_idx = current_idx + 1
                    if not is_terminal and next_idx < self.train_end_idx:
                        # 다음 상태 근사
                        next_pos_info = pos_info  # 근사값
                        next_state_full = self.env.get_observation(
                            position_info=next_pos_info,
                            current_index=next_idx,
                            entry_index=self.entry_index,
                            current_position=self.current_position
                        )
                        if next_state_full:
                            next_state = (next_state_full[0], next_state_full[1])  # Mask 제외
                            self.agent.update(next_state=next_state, episode=episode_num)
                    else:
                        self.agent.update(next_state=None, episode=episode_num)
                
                # 인덱스 증가
                self.data_collector.current_index += 1
                
            except Exception as e:
                logger.error(f"Step Error: {e}")
                break
                
        return episode_reward, steps

    def train(self, num_episodes=1000, max_steps_per_episode=None, save_interval=None):
        # 기본값 설정 (config에서 가져오기)
        if max_steps_per_episode is None:
            max_steps_per_episode = config.TRAIN_MAX_STEPS_PER_EPISODE
        if save_interval is None:
            save_interval = config.TRAIN_SAVE_INTERVAL
        logger.info("🚀 학습 시작")
        best_reward = float('-inf')
        scaler_path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
        if not scaler_path.endswith('.pkl'):
            scaler_path = config.AI_MODEL_PATH + '_scaler.pkl'
        
        for episode in range(1, num_episodes + 1):
            try:
                result = self.train_episode(episode, max_steps_per_episode)
                if result is None: continue
                
                reward, steps = result
                self.episode_rewards.append(reward)
                if self.visualizer: self.visualizer.update(reward)
                
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Ep {episode}: Reward {reward:.4f} | Avg {avg_reward:.4f} | Steps {steps}")
                
                # [스케줄러 업데이트]
                self.agent.step_scheduler(avg_reward)
                
                # 모델 저장
                if reward > best_reward:
                    best_reward = reward
                    self.agent.save_model(config.AI_MODEL_PATH)
                    # 스케일러도 저장 (중요)
                    self.env.preprocessor.save_scaler(scaler_path)
                elif episode % save_interval == 0:
                    self.agent.save_model(config.AI_MODEL_PATH)
                    self.env.preprocessor.save_scaler(scaler_path)
                    
            except KeyboardInterrupt:
                logger.info("학습 중단")
                break
            except Exception as e:
                logger.error(f"에피소드 오류: {e}")
                continue

if __name__ == '__main__':
    trainer = PPOTrainer(enable_visualization=True)
    trainer.train()
