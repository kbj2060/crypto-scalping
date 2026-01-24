"""
SAC (Soft Actor-Critic) 모델 학습 스크립트 (Final)
- Best/Last 모델 및 스케일러 분리 저장
- 실시간 리워드 그래프 (Live Plotting)
- 연속형 행동 공간 (Action Dead-zone 적용)
"""
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import DataCollector, BinanceClient
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy,
    CCIReversalStrategy, WilliamsRStrategy
)

from model.trading_env import TradingEnvironment
from model.sac_agent import SACAgent
from model.feature_engineering import FeatureEngineer
from model.mtf_processor import MTFProcessor

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/train_sac.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 피처 엔지니어링 로그 레벨 조정
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)


class SACTrainer:
    """SAC 모델 학습 클래스"""
    
    def __init__(self, enable_visualization=True):
        self.enable_visualization = enable_visualization
        # 1. 데이터 수집기 초기화
        self.data_collector = DataCollector(use_saved_data=True)
        if not self.data_collector.load_saved_data():
            raise ValueError("데이터 로드 실패")
        
        # 2. 전략 초기화
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
        if config.STRATEGIES.get('cci_reversal', False):
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
        if config.STRATEGIES.get('williams_r', False):
            self.range_strategies.append(WilliamsRStrategy())
        
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"✅ 전략 초기화 완료: 총 {len(self.strategies)}개")
        
        # 3. 피처 엔지니어링 (CSV 파일이 있으면 로드, 없으면 계산)
        self._load_or_create_features()
        
        # [핵심] 전략 신호 미리 계산 (Pre-calculation)
        # CSV에 전략 컬럼이 이미 있으면 건너뛰기
        self.precalculate_strategies()
        
        # 4. 환경 생성 (config.LOOKBACK 사용)
        self.env = TradingEnvironment(self.data_collector, self.strategies, lookback=config.LOOKBACK)
        
        # 5. 스케일러 학습
        self._fit_global_scaler()
        
        # 6. Agent 생성
        state_dim = self.env.get_state_dim()  # 29
        action_dim = 1  # 연속형: 매수/매도 강도 (-1 ~ 1)
        info_dim = len(self.strategies) + 3  # 전략 점수 + 포지션 정보
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🔧 디바이스: {device}")
        
        # config에서 네트워크 파라미터 가져오기
        self.agent = SACAgent(
            state_dim, 
            action_dim, 
            info_dim=info_dim, 
            hidden_dim=config.NETWORK_HIDDEN_DIM, 
            device=device
        )
        
        # 모델 로드 (Last 모델 우선 로드)
        base_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model').replace('.pth', '')
        last_model_path = f"{base_path}_last.pth"
        
        if os.path.exists(last_model_path):
            try:
                self.agent.load_model(last_model_path)
                logger.info(f"✅ 기존 모델(Last) 로드: {last_model_path}")
            except Exception as e:
                logger.warning(f"모델 로드 실패 (새 모델로 시작): {e}")
        else:
            logger.info("새 모델로 학습 시작")
        
        # 학습 상태 변수
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        self.prev_pnl = 0.0
        self.episode_rewards = []
        self.avg_rewards = []
        self.total_steps = 0

        # 실시간 그래프 설정
        if self.enable_visualization:
            try:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 5))
                self.ax.set_title('SAC Real-time Training')
                self.ax.set_xlabel('Episode')
                self.ax.set_ylabel('Reward')
                self.ax.grid(True, alpha=0.3)
                self.line1, = self.ax.plot([], [], label='Reward', alpha=0.3, color='gray')
                self.line2, = self.ax.plot([], [], label='Avg (10)', color='red', linewidth=2)
                self.ax.legend()
            except Exception as e:
                logger.warning(f"그래프 초기화 실패 (계속 진행): {e}")
                self.enable_visualization = False
    
    def _load_or_create_features(self):
        """
        피처 파일이 있으면 로드, 없으면 생성
        """
        feature_file_path = 'data/training_features.csv'
        
        if os.path.exists(feature_file_path):
            logger.info("📂 피처 파일 로드 중...")
            try:
                df = pd.read_csv(feature_file_path, index_col=0, parse_dates=True)
                self.data_collector.eth_data = df
                logger.info(f"✅ 피처 파일 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
                return
            except Exception as e:
                logger.warning(f"피처 파일 로드 실패, 재생성합니다: {e}")
        
        # 파일이 없거나 로드 실패 시 재생성
        logger.info("📊 피처 엔지니어링 수행 중...")
        try:
            # ETH 데이터 준비
            eth_data = self.data_collector.eth_data.copy()
            
            # 인덱스가 DatetimeIndex인지 확인 및 변환
            if not isinstance(eth_data.index, pd.DatetimeIndex):
                if 'timestamp' in eth_data.columns:
                    eth_data.index = pd.to_datetime(eth_data['timestamp'], unit='ms')
                else:
                    eth_data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(eth_data), freq='3min')
            
            # BTC 데이터 준비
            btc_data = None
            if hasattr(self.data_collector, 'btc_data') and self.data_collector.btc_data is not None:
                btc_data = self.data_collector.btc_data.copy()
                if not isinstance(btc_data.index, pd.DatetimeIndex):
                    if 'timestamp' in btc_data.columns:
                        btc_data.index = pd.to_datetime(btc_data['timestamp'], unit='ms')
                    else:
                        btc_data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(btc_data), freq='3min')
                
                # 공통 인덱스로 정렬
                common_index = eth_data.index.intersection(btc_data.index)
                if len(common_index) > 0:
                    eth_data = eth_data.loc[common_index]
                    btc_data = btc_data.loc[common_index]
            
            # (1) 기본 기술적 지표 생성
            feature_engineer = FeatureEngineer(eth_data, btc_data)
            df = feature_engineer.generate_features()
            
            if df is None:
                raise ValueError("피처 생성 실패")
            
            # (2) 멀티 타임프레임 지표 추가
            mtf_processor = MTFProcessor(df)
            df = mtf_processor.add_mtf_features()
            
            # 데이터 교체
            self.data_collector.eth_data = df
            if btc_data is not None:
                self.data_collector.btc_data = btc_data
            
            # CSV 저장
            os.makedirs('data', exist_ok=True)
            df.to_csv(feature_file_path)
            logger.info(f"✅ 피처 엔지니어링 완료 및 저장: {len(df)}개 행, {len(df.columns)}개 컬럼")
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}", exc_info=True)
            raise
    
    def precalculate_strategies(self):
        """
        전략 신호 사전 계산 (캐싱 기능 추가)
        - 파일이 있으면 로드 (빠름) ⚡
        - 없으면 계산 후 저장 (느림) 🐢 -> 💾
        """
        # 캐시 파일 경로 설정
        cache_path = 'data/cached_strategies.csv'
        
        # 1. 캐시 파일이 존재하는지 확인
        if os.path.exists(cache_path):
            logger.info(f"⚡ 캐시된 전략 데이터를 발견했습니다! 로드 중... ({cache_path})")
            try:
                # 저장된 파일 불러오기
                cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                self.data_collector.eth_data = cached_df
                logger.info("✅ 전략 데이터 로드 완료 (계산 생략)")
                return
            except Exception as e:
                logger.warning(f"캐시 파일 로드 실패 (새로 계산합니다): {e}")

        # 2. 캐시가 없으면 계산 시작 (기존 로직)
        logger.info("🧠 전략 신호 사전 계산 중 (Pre-calculation)...")
        df = self.data_collector.eth_data
        
        # 전략별 컬럼 초기화
        for i in range(len(self.strategies)):
            df[f'strategy_{i}'] = 0.0
            
        total_len = len(df)
        start_idx = config.LOOKBACK + 50
        
        # 진행률 표시 (tqdm)
        try:
            from tqdm import tqdm
            iterator = tqdm(range(start_idx, total_len), desc="Strategy Calc")
        except ImportError:
            logger.warning("tqdm이 설치되지 않아 진행상황 표시를 건너뜁니다.")
            iterator = range(start_idx, total_len)
        
        for i in iterator:
            self.data_collector.current_index = i
            
            for s_idx, strategy in enumerate(self.strategies):
                try:
                    result = strategy.analyze(self.data_collector)
                    score = 0.0
                    if result:
                        conf = float(result.get('confidence', 0.0))
                        signal = result.get('signal', 'NEUTRAL')
                        
                        if signal == 'LONG': 
                            score = conf
                        elif signal == 'SHORT': 
                            score = -conf
                    
                    df.iat[i, df.columns.get_loc(f'strategy_{s_idx}')] = score
                    
                except Exception:
                    continue
        
        # 3. 계산 끝난 후 파일로 저장 (중요!)
        logger.info(f"💾 계산된 전략 데이터를 저장합니다: {cache_path}")
        os.makedirs('data', exist_ok=True)
        df.to_csv(cache_path)
        logger.info("✅ 전략 신호 계산 및 저장 완료!")
    
    def _fit_global_scaler(self):
        """29개 고급 피처 기반 전역 스케일러 학습 (최적화 버전)"""
        try:
            logger.info("🚀 29개 고급 피처 기반 전역 스케일러 학습 시작...")
            
            if self.data_collector.eth_data is None or len(self.data_collector.eth_data) == 0:
                logger.warning("데이터가 없어 스케일러 학습을 건너뜁니다.")
                return
            
            # 사용할 29개 컬럼 정의
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            # 컬럼 존재 여부 확인
            missing_cols = [c for c in target_cols if c not in self.data_collector.eth_data.columns]
            if missing_cols:
                logger.warning(f"⚠️ 누락된 컬럼이 있어 0으로 채웁니다: {missing_cols}")
                for c in missing_cols:
                    self.data_collector.eth_data[c] = 0.0
            
            # 샘플링
            total_candles = len(self.data_collector.eth_data)
            min_required = self.env.lookback + 100
            sample_size = min(50000, total_candles - min_required)
            
            if total_candles > min_required + sample_size:
                indices = np.linspace(min_required, total_candles - 1, sample_size, dtype=int)
            else:
                indices = np.arange(min_required, total_candles)
            
            logger.info(f"데이터 추출 중... ({len(indices)}개 샘플)")
            
            # 데이터 수집
            all_seq_features = []
            for idx in indices:
                if idx < self.env.lookback:
                    continue
                recent_df = self.data_collector.eth_data[target_cols].iloc[idx-self.env.lookback+1:idx+1]
                if len(recent_df) == self.env.lookback:
                    seq_features = recent_df.values.astype(np.float32)
                    all_seq_features.append(seq_features)
            
            if len(all_seq_features) == 0:
                logger.warning("피처 수집 실패, 스케일러 학습 건너뜀")
                return
            
            all_features_array = np.vstack(all_seq_features)
            
            # NaN 처리
            if np.isnan(all_features_array).any():
                all_features_array = np.nan_to_num(all_features_array)
            
            # 스케일러 학습
            self.env.preprocessor.fit(all_features_array)
            self.env.scaler_fitted = True
            
            logger.info(f"✅ 전역 스케일러 학습 완료: {len(all_features_array)}개 샘플, Feature Dim: {all_features_array.shape[1]}")
            
        except Exception as e:
            logger.error(f"전역 스케일러 학습 실패: {e}", exc_info=True)
            logger.warning("스케일러 학습 실패, 학습 도중 online-fitting으로 대체합니다.")
    
    def live_plot(self):
        """실시간 그래프 업데이트"""
        if not self.enable_visualization:
            return
        try:
            x = range(len(self.episode_rewards))
            self.line1.set_data(x, self.episode_rewards)
            self.line2.set_data(x, self.avg_rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass  # 그래프 오류는 무시하고 학습 계속
    
    def interpret_action(self, action_value):
        """
        [문제 해결 3 & 4] Continuous Action 해석 개선
        -0.3 ~ 0.3 구간: Neutral (Exit/Hold) -> 무한 존버 방지
        
        Args:
            action_value: float, -1 ~ 1 사이의 연속값
        Returns:
            int: 0=NEUTRAL(청산/관망), 1=LONG, 2=SHORT
        """
        threshold = 0.3
        
        if action_value > threshold:
            return 1  # LONG 진입 (강도: action_value)
        elif action_value < -threshold:
            return 2  # SHORT 진입 (강도: abs(action_value))
        else:
            return 0  # NEUTRAL (청산 또는 관망)
    
    def train_episode(self, episode_num, max_steps=None):
        """
        한 에피소드 학습 (Fixed Architecture)
        - Action Dead-zone 적용 (Exit Logic 개선)
        - Next State Indexing 오류 수정
        """
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        # 학습 데이터 범위 설정
        train_size = int(len(self.data_collector.eth_data) * 0.8)
        self.train_end_idx = train_size
        
        # 무작위 시작 인덱스
        start_min = config.LOOKBACK + 100
        start_max = self.train_end_idx - max_steps - 50
        if start_max <= start_min:
            logger.warning("학습 데이터가 부족합니다.")
            return None
        
        start_idx = np.random.randint(start_min, start_max)
        
        # 초기화
        self.data_collector.current_index = start_idx
        self.agent.reset_episode_states()  # [추가] 에피소드 시작 전 뇌 리셋 (중요!)
        current_position = None  # 'LONG', 'SHORT', None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        
        batch_size = getattr(config, 'SAC_BATCH_SIZE', 256)
        
        for step in range(max_steps):
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx:
                break
            
            # Position Info 구성
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position is not None else 0
            
            # Unrealized PnL (관측용으로만 사용, 보상엔 안 씀)
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
                
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]
            
            # 1. 상태 관측 (State) - [문제 해결 1] 인덱스 명시적 전달
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None:
                break

            # 2. 행동 선택 (Action)
            action_continuous = self.agent.select_action(state)  # [-1, 1]
            action_code = self.interpret_action(action_continuous[0])  # 0, 1, 2
            
            # 3. 트레이딩 로직 실행
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            
            # A. 포지션 청산 조건 (신호 반전 or Neutral 신호 or 손절)
            if current_position is not None:
                should_exit = False
                
                # Exit 조건 1: 신호 변경 (Long인데 Short/Neutral 신호 뜸)
                if current_position == 'LONG' and action_code != 1:
                    should_exit = True
                if current_position == 'SHORT' and action_code != 2:
                    should_exit = True
                
                # Exit 조건 2: 손절 (Stop Loss) -2%
                if unrealized_pnl < -0.02:
                    should_exit = True
                
                if should_exit:
                    realized_pnl = unrealized_pnl  # 확정
                    trade_done = True
                    current_position = None  # 포지션 해제
                    entry_price = 0.0
                    entry_index = 0
            
            # B. 신규 진입 조건 (포지션 없을 때만)
            if current_position is None and not trade_done:
                if action_code == 1:  # LONG Entry
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = current_idx
                elif action_code == 2:  # SHORT Entry
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = current_idx
            
            # 4. 보상 계산 (Realized PnL 위주)
            reward = self.env.calculate_reward(realized_pnl, trade_done, holding_time)
            
            # 5. 다음 상태 관측 (Next State) - [문제 해결 1] 인덱스 명시적 전달
            next_idx = current_idx + 1
            self.data_collector.current_index = next_idx  # Loop 진행을 위해 업데이트
            
            # Next Position Info 추정
            next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            next_hold_time = (next_idx - entry_index) if current_position is not None else 0
            
            # 다음 가격 (있다면)
            if next_idx < len(self.data_collector.eth_data):
                next_price = float(self.data_collector.eth_data.iloc[next_idx]['close'])
                next_un_pnl = 0.0
                if current_position == 'LONG':
                    next_un_pnl = (next_price - entry_price) / entry_price
                elif current_position == 'SHORT':
                    next_un_pnl = (entry_price - next_price) / entry_price
            else:
                next_un_pnl = 0.0
                
            next_pos_info = [next_pos_val, next_un_pnl * 10, next_hold_time / max_steps]
            
            next_state = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)
            
            # 종료 여부
            done = False if step < max_steps - 1 else True
            if next_state is None:
                done = True
            
            # Fallback for next_state (끝부분 처리)
            if next_state is None:
                next_state = state

            # 6. 저장 및 학습
            self.agent.memory.push(state, action_continuous, reward, next_state, done)
            episode_reward += reward
            
            if len(self.agent.memory) > batch_size:
                self.agent.update(batch_size=batch_size)
                self.agent.step_schedulers()

        return episode_reward
    
    def train(self, num_episodes=1000, max_steps_per_episode=None, save_interval=None):
        """모델 학습"""
        if max_steps_per_episode is None:
            max_steps_per_episode = config.TRAIN_MAX_STEPS_PER_EPISODE
        if save_interval is None:
            save_interval = config.TRAIN_SAVE_INTERVAL
        
        # 학습 시작 전 스케줄러 설정
        # (총 예상 업데이트 횟수 = 에피소드 * 스텝 수)
        total_steps = num_episodes * max_steps_per_episode
        warmup_ratio = getattr(config, 'SAC_WARMUP_RATIO', 0.05)
        
        logger.info("=" * 60)
        logger.info("🚀 SAC 모델 학습 시작 (Best/Last Save Enabled)")
        logger.info("=" * 60)
        logger.info(f"에피소드 수: {num_episodes}")
        logger.info(f"에피소드당 최대 스텝: {max_steps_per_episode}")
        logger.info(f"모델 저장 간격: {save_interval} 에피소드")
        logger.info(f"스케줄러 설정: 총 {total_steps} 스텝, Warmup {warmup_ratio*100:.1f}%")
        logger.info("=" * 60)
        
        # 스케줄러 설정
        self.agent.setup_schedulers(total_steps, warmup_ratio)
        
        # [NEW] 저장 경로 설정 (Best/Last 분리)
        base_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model').replace('.pth', '')
        
        best_model_path = f"{base_path}_best.pth"
        best_scaler_path = f"{base_path}_best_scaler.pkl"
        
        last_model_path = f"{base_path}_last.pth"
        last_scaler_path = f"{base_path}_last_scaler.pkl"
        
        # 초기 스케일러 저장 (Last에 백업)
        os.makedirs(os.path.dirname(last_scaler_path), exist_ok=True)
        self.env.preprocessor.save(last_scaler_path)
        
        logger.info(f"🚀 SAC 학습 시작 (Best/Last Save Enabled)")
        best_reward = float('-inf')
        
        for episode in range(1, num_episodes + 1):
            try:
                # 에피소드 실행
                episode_reward = self.train_episode(episode_num=episode, max_steps=max_steps_per_episode)
                if episode_reward is None:
                    logger.warning("에피소드 실패, 다음 에피소드로 진행")
                    continue
                
                self.episode_rewards.append(episode_reward)
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_reward)
                
                # 통계 출력
                current_lr = self.agent.actor_scheduler.get_last_lr()[0] if self.agent.actor_scheduler else config.SAC_LEARNING_RATE
                logger.info(f"✅ Ep {episode}: Reward {episode_reward:.4f} | Avg {avg_reward:.4f} | LR {current_lr:.6f}")
                
                # 그래프 갱신
                self.live_plot()
                
                # [NEW] Best 모델 저장 (신기록 갱신 시)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    self.agent.save_model(best_model_path)
                    self.env.preprocessor.save(best_scaler_path)
                    logger.info(f"🏆 신기록 달성! ({best_reward:.4f}) -> Best 모델 저장")
                
                # [NEW] Last 모델 저장 (매번 or 주기적으로)
                if episode % save_interval == 0:
                    os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
                    self.agent.save_model(last_model_path)
                    self.env.preprocessor.save(last_scaler_path)
                    # logger.info(f"💾 정기 저장 완료 (Ep {episode})")
                
            except KeyboardInterrupt:
                logger.info("학습 중단")
                break
            except Exception as e:
                logger.error(f"에피소드 {episode} 실패: {e}", exc_info=True)
                continue
        
        # 최종 모델 저장 (Last)
        os.makedirs(os.path.dirname(last_model_path), exist_ok=True)
        self.agent.save_model(last_model_path)
        self.env.preprocessor.save(last_scaler_path)
        
        # 학습 종료 시 그래프 유지
        if self.enable_visualization:
            plt.ioff()
            plt.show()
        
        logger.info("=" * 60)
        logger.info("✅ 학습 및 스케일러 저장 완료")
        logger.info(f"총 스텝: {self.total_steps}")
        logger.info(f"평균 보상: {sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0:.4f}")
        logger.info(f"Best 모델: {best_model_path}")
        logger.info(f"Last 모델: {last_model_path}")
        logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC 모델 학습')
    parser.add_argument('--episodes', type=int, default=config.TRAIN_NUM_EPISODES, help='학습 에피소드 수')
    parser.add_argument('--steps', type=int, default=config.TRAIN_MAX_STEPS_PER_EPISODE, help='에피소드당 최대 스텝 수')
    parser.add_argument('--save-interval', type=int, default=config.TRAIN_SAVE_INTERVAL, help='모델 저장 간격 (에피소드)')
    parser.add_argument('--no-plot', action='store_true', help='그래프 비활성화')
    
    args = parser.parse_args()
    
    # matplotlib 한글 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        trainer = SACTrainer(enable_visualization=not args.no_plot)
        trainer.train(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_interval=args.save_interval
        )
    except KeyboardInterrupt:
        logger.info("학습 중단")
    except Exception as e:
        logger.error(f"학습 실패: {e}", exc_info=True)
        sys.exit(1)
