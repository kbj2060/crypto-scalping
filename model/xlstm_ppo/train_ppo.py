"""
PPO 모델 학습 스크립트
별도로 실행하여 모델을 학습하고 저장합니다.
"""
import logging
import os
import sys
import time
from datetime import datetime
from collections import deque
import numpy as np

# 시각화 모듈 (선택적)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# 상위 폴더를 경로에 추가 (config, core, strategies 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import DataCollector, BinanceClient
from core.indicators import Indicators
from strategies import (
    BTCEthCorrelationStrategy,
    VolatilitySqueezeStrategy,
    OrderblockFVGStrategy,
    HMAMomentumStrategy,
    MFIMomentumStrategy,
    BollingerMeanReversionStrategy,
    VWAPDeviationStrategy,
    RangeTopBottomStrategy,
    StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy
)

# AI 강화학습 모듈
try:
    import torch
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ AI 모듈 로드 실패: {e}")
    print("torch가 설치되어 있는지 확인하세요: pip install torch")
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

# matplotlib 사용 가능 여부 로깅
if not MATPLOTLIB_AVAILABLE:
    logger.warning("matplotlib가 설치되지 않았습니다. 시각화 기능이 비활성화됩니다.")
    logger.warning("설치 방법: pip install matplotlib")


class LiveVisualizer:
    """학습 리워드를 실시간으로 그래프화하는 클래스"""
    def __init__(self, window_size=10):
        if not MATPLOTLIB_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        plt.ion()  # 대화형 모드 활성화
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
        if not self.enabled:
            return
        
        self.rewards.append(reward)
        
        # 이동 평균 계산
        if len(self.rewards) >= self.window_size:
            avg = np.mean(self.rewards[-self.window_size:])
        else:
            avg = np.mean(self.rewards)
        self.moving_avg.append(avg)
        
        # 데이터 업데이트
        x = np.arange(len(self.rewards))
        self.line1.set_data(x, self.rewards)
        self.line2.set_data(x, self.moving_avg)
        
        # 화면 범위 자동 조절
        self.ax.relim()
        self.ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)  # 짧은 휴식으로 그래프 갱신 보장


class PPOTrainer:
    """PPO 모델 학습 클래스"""
    def __init__(self, enable_visualization=False):
        """
        Args:
            enable_visualization: 시각화 활성화 여부 (기본값: True)
        """
        # 저장된 데이터 사용 (학습용)
        self.data_collector = DataCollector(use_saved_data=True)
        self.client = BinanceClient()
        
        # 전략 초기화
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
        
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"전략 초기화 완료: {len(self.strategies)}개 전략")
        
        # 트레이딩 환경 생성
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        
        # 스케일러 전역 학습 (학습 시작 전 전체 데이터로 한 번만 fit)
        self._fit_global_scaler()
        
        state_dim = self.env.get_state_dim()
        action_dim = 3  # 0: Hold, 1: Long, 2: Short
        
        # PPO 에이전트 생성 (Late Fusion 구조: info_dim=13)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"디바이스: {device}")
        self.agent = PPOAgent(state_dim, action_dim, hidden_dim=128, device=device, info_dim=13)
        
        # 기존 모델 로드 (있는 경우)
        if os.path.exists(config.AI_MODEL_PATH):
            try:
                self.agent.load_model(config.AI_MODEL_PATH)
                logger.info(f"✅ 기존 모델 로드: {config.AI_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"모델 로드 실패 (새 모델로 시작): {e}")
        else:
            logger.info("새 모델로 학습 시작")
        
        # 학습 상태
        self.current_position = None
        self.entry_price = None
        self.entry_index = None  # [수정] entry_time 대신 entry_index 사용 (과거 데이터 학습용)
        self.prev_pnl = 0.0  # 이전 스텝의 수익률 (pnl_change 계산용)
        self.episode_rewards = []
        self.total_steps = 0
        
        # 실시간 시각화 초기화 (옵션)
        if enable_visualization:
            self.visualizer = LiveVisualizer(window_size=10)
        else:
            self.visualizer = None
    
    def _fit_global_scaler(self):
        """8개 핵심 시계열 피처 기반 전역 스케일러 학습 (한 번만 실행)"""
        try:
            logger.info("8개 핵심 피처 기반 전역 스케일러 학습 시작...")
            
            # 전체 데이터 수집
            if self.data_collector.eth_data is None or len(self.data_collector.eth_data) == 0:
                logger.warning("데이터가 없어 스케일러 학습을 건너뜁니다.")
                return
            
            # 샘플링할 데이터 수 (전체 데이터가 너무 크면 샘플링)
            total_candles = len(self.data_collector.eth_data)
            sample_size = min(50000, total_candles)  # 최대 5만개 샘플
            
            # 균등 간격 샘플링 (최소 20개는 필요하므로 20부터 시작)
            if total_candles > sample_size:
                indices = np.linspace(20, total_candles - 1, sample_size, dtype=int)
            else:
                indices = np.arange(20, total_candles)
            
            logger.info(f"스케일러 학습용 데이터: {len(indices)}개 샘플 (전체: {total_candles}개)")
            
            # 8개 핵심 시계열 피처 수집
            window_size = 20
            all_seq_features = []
            
            for idx in indices:
                if idx < window_size:
                    continue
                
                try:
                    # 마지막 20봉 데이터 가져오기
                    window = self.data_collector.eth_data.iloc[idx-window_size+1:idx+1]
                    if len(window) < window_size:
                        continue
                    
                    close = window['close'].values.astype(np.float32)
                    high = window['high'].values.astype(np.float32)
                    low = window['low'].values.astype(np.float32)
                    volume_raw = window['volume'].values.astype(np.float32)
                    trades_raw = window['trades'].values.astype(np.float32)
                    
                    # [추가] VWAP 계산 (Training 환경과 동일한 로직)
                    tp = (high + low + close) / 3
                    vp = tp * volume_raw
                    cumulative_vp = np.cumsum(vp)
                    cumulative_vol = np.cumsum(volume_raw)
                    vwap = cumulative_vp / (cumulative_vol + 1e-8)
                    
                    # 8개 시계열 피처 생성
                    # [최적화] Volume과 Trades에 로그 변환 적용 (거래량 폭발 구간의 극단적 차이 완화)
                    f1 = (window['open'].values - close) / (close + 1e-8)  # Open (close 대비)
                    f2 = (high - close) / (close + 1e-8)   # High (close 대비)
                    f3 = (low - close) / (close + 1e-8)    # Low (close 대비)
                    f4 = np.diff(np.log(close + 1e-8), prepend=np.log(close[0] + 1e-8))  # Log_Return
                    f5 = np.log1p(volume_raw)  # Volume (로그 변환)
                    f6 = np.log1p(trades_raw)  # Trades (로그 변환)
                    f7 = window['taker_buy_base'].values / (volume_raw + 1e-8)  # Taker_Ratio
                    f8 = (close - vwap) / (vwap + 1e-8)  # [NEW] VWAP 이격도
                    
                    # 8개 피처 결합: (20, 8)
                    seq_features = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8])
                    all_seq_features.append(seq_features)
                    
                except Exception as e:
                    logger.debug(f"인덱스 {idx} 처리 실패: {e}")
                    continue
            
            if len(all_seq_features) == 0:
                logger.warning("피처 수집 실패, 스케일러 학습 건너뜀")
                return
            
            # 전체 피처 결합: (N*20, 8)
            all_features_array = np.vstack(all_seq_features)
            
            # 스케일러 학습 (8개 차원)
            self.env.preprocessor.fit(all_features_array)
            self.env.scaler_fitted = True
            
            # [추가] 학습 완료된 스케일러를 파일로 저장
            self.env.preprocessor.save_scaler()
            
            logger.info(f"✅ 8개 피처 스케일러 학습 및 저장 완료: {len(all_features_array)}개 샘플 (VWAP 이격도 포함)")
            
        except Exception as e:
            logger.error(f"전역 스케일러 학습 실패: {e}", exc_info=True)
            logger.warning("스케일러 학습 실패, 첫 관측 시 학습합니다.")
        
    def train_episode(self, episode_num, max_steps=1000, overfitting_test=False, fixed_start_index=1000):
        """한 에피소드 학습
        
        Args:
            episode_num: 현재 에피소드 번호 (엔트로피 스케줄러용)
            max_steps: 최대 스텝 수
            overfitting_test: 과적합 테스트 모드 (True면 고정 인덱스 사용)
            fixed_start_index: 과적합 테스트 시 고정 시작 인덱스
        """
        # [설정] 보유 시간 정규화 기준 (8시간 = 480분/3분)
        # 이 시간이 지나면 1.0으로 고정됨
        MAX_HOLDING_STEPS = 160.0
        
        episode_reward = 0.0
        steps = 0
        
        # 저장된 데이터에서 인덱스 리셋
        if overfitting_test:
            # 과적합 테스트: 고정 시작 인덱스 사용 (같은 데이터 반복 학습)
            self.data_collector.current_index = fixed_start_index
            logger.info(f"🧪 [과적합 테스트] 고정 시작 인덱스: {fixed_start_index}, 최대 스텝: {max_steps}")
        else:
            # [실전 학습] 랜덤 스타트 활성화
            # 매번 다른 구간을 학습하게 하여 다양한 시장 상황을 경험하도록 함
            self.data_collector.reset_index(max_steps=max_steps, random_start=True)
            logger.debug(f"🎲 [실전 학습] 무작위 시작 인덱스: {self.data_collector.current_index}")
        
        # 에피소드 시작 시 이전 수익률 초기화
        self.prev_pnl = 0.0
        self.current_position = None
        self.entry_price = None
        self.entry_index = None  # [수정] entry_time 대신 entry_index 사용
        
        # 초기 데이터 확인
        if self.data_collector.eth_data is None or len(self.data_collector.eth_data) == 0:
            logger.error("저장된 데이터가 없습니다. model/collect_training_data.py를 먼저 실행하세요.")
            return None
        
        # 사용 가능한 최대 스텝 수 계산
        available_steps = len(self.data_collector.eth_data) - self.data_collector.current_index
        actual_steps = min(max_steps, available_steps)
        
        if actual_steps <= 0:
            logger.warning("사용 가능한 데이터가 부족합니다.")
            return None
        
        logger.info(f"에피소드 시작: 총 {len(self.data_collector.eth_data)}개 캔들 중 {actual_steps}개 사용 (인덱스: {self.data_collector.current_index}부터)")
        
        for step in range(actual_steps):
            try:
                # 1. 저장된 데이터에서 다음 캔들로 진행 (인덱스만 증가)
                # get_candles가 현재 인덱스 기준으로 이전 데이터를 반환하므로
                # 인덱스를 먼저 증가시켜야 함
                if self.data_collector.current_index >= len(self.data_collector.eth_data):
                    logger.info("데이터 끝에 도달, 에피소드 종료")
                    break
                
                # 인덱스 증가 (다음 캔들로 이동)
                self.data_collector.current_index += 1
                
                # 2. 포지션 정보 수집 (Late Fusion용)
                pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
                
                # [수정] 절대 시간 기준 정규화 (에피소드 길이 의존성 제거)
                if self.entry_index is not None:
                    elapsed_steps = self.data_collector.current_index - self.entry_index
                    # 480분을 넘으면 1.0으로 고정 (min 사용)
                    hold_val = min(1.0, elapsed_steps / MAX_HOLDING_STEPS)
                else:
                    hold_val = 0.0
                    
                pnl_val = self.prev_pnl * 10  # PnL 스케일 조정
                pos_info = [pos_val, pnl_val, hold_val]
                
                # 3. 상태 관측 (전처리 파이프라인 포함 + 포지션 정보)
                # get_candles가 현재 인덱스 기준으로 이전 lookback개를 반환
                state = self.env.get_observation(position_info=pos_info)
                if state is None:
                    logger.warning("상태 관측 실패, 다음 캔들로 진행")
                    continue
                
                # [디버깅] 첫 번째 에피소드의 첫 스텝에서만 값 확인
                if episode_num == 1 and steps < 5:
                    obs_seq, obs_info = state
                    logger.info(f"\n🔍 [Step {steps}] 입력 데이터 점검:")
                    logger.info(f"   - 시계열 Shape: {obs_seq.shape}")
                    logger.info(f"   - 시계열(Min/Max/Mean): {obs_seq.min().item():.4f} ~ {obs_seq.max().item():.4f} / {obs_seq.mean().item():.4f}")
                    logger.info(f"   - 정보 Shape: {obs_info.shape}")
                    logger.info(f"   - 정보(Min/Max/Mean): {obs_info.min().item():.4f} ~ {obs_info.max().item():.4f} / {obs_info.mean().item():.4f}")
                    
                    # 만약 여기서 10.0을 넘는 숫자가 보이면 정규화가 깨진 것입니다.
                    if abs(obs_seq.max().item()) > 10.0 or abs(obs_seq.min().item()) > 10.0:
                        logger.warning("🚨 경고: 시계열 입력값이 너무 큽니다! 스케일러가 작동하지 않습니다.")
                        logger.warning(f"   값 범위: {obs_seq.min().item():.4f} ~ {obs_seq.max().item():.4f}")
                    
                    if abs(obs_info.max().item()) > 10.0 or abs(obs_info.min().item()) > 10.0:
                        logger.warning("🚨 경고: 정보 입력값이 너무 큽니다!")
                        logger.warning(f"   값 범위: {obs_info.min().item():.4f} ~ {obs_info.max().item():.4f}")
                    
                    # NaN/Inf 체크
                    if torch.isnan(obs_seq).any() or torch.isinf(obs_seq).any():
                        nan_count = torch.isnan(obs_seq).sum().item()
                        inf_count = torch.isinf(obs_seq).sum().item()
                        logger.error(f"🚨 시계열 데이터에 NaN({nan_count}) 또는 Inf({inf_count}) 발생!")
                    
                    if torch.isnan(obs_info).any() or torch.isinf(obs_info).any():
                        nan_count = torch.isnan(obs_info).sum().item()
                        inf_count = torch.isinf(obs_info).sum().item()
                        logger.error(f"🚨 정보 데이터에 NaN({nan_count}) 또는 Inf({inf_count}) 발생!")
                
                # 4. 행동 선택
                action, log_prob = self.agent.select_action(state)
                action_names = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
                action_name = action_names[action]
                
                # 5. 현재 가격 확인 (현재 인덱스의 캔들)
                if self.data_collector.current_index > 0:
                    current_candle = self.data_collector.eth_data.iloc[self.data_collector.current_index - 1]
                    current_price = float(current_candle['close'])
                else:
                    continue
                
                # 6. 보상 계산 및 포지션 업데이트
                reward = 0.0
                trade_done = False
                current_pnl = 0.0
                pnl_change = 0.0
                
                if action == 1:  # LONG
                    if self.current_position != 'LONG':
                        # 기존 포지션 청산
                        if self.current_position == 'SHORT' and self.entry_price:
                            pnl = (self.entry_price - current_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl  # 실현 수익의 변화량
                            reward = self.env.calculate_reward(pnl, True, holding_time=0, pnl_change=pnl_change)
                            trade_done = True
                            logger.info(f"💰 숏 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}")
                            self.prev_pnl = 0.0  # 포지션 청산 후 초기화
                        
                        # 롱 진입
                        self.current_position = 'LONG'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index  # [수정] 인덱스 저장
                        self.prev_pnl = 0.0  # 새 포지션 진입 시 초기화
                        logger.debug(f"📈 롱 진입: ${current_price:.2f} (인덱스: {self.entry_index})")
                
                elif action == 2:  # SHORT
                    if self.current_position != 'SHORT':
                        # 기존 포지션 청산
                        if self.current_position == 'LONG' and self.entry_price:
                            pnl = (current_price - self.entry_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl  # 실현 수익의 변화량
                            reward = self.env.calculate_reward(pnl, True, holding_time=0, pnl_change=pnl_change)
                            trade_done = True
                            logger.info(f"💰 롱 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}")
                            self.prev_pnl = 0.0  # 포지션 청산 후 초기화
                        
                        # 숏 진입
                        self.current_position = 'SHORT'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index  # [수정] 인덱스 저장
                        self.prev_pnl = 0.0  # 새 포지션 진입 시 초기화
                        logger.debug(f"📉 숏 진입: ${current_price:.2f} (인덱스: {self.entry_index})")
                
                else:  # HOLD
                    # 보유 중인 포지션의 수익률 계산
                    if self.current_position and self.entry_price:
                        if self.current_position == 'LONG':
                            current_pnl = (current_price - self.entry_price) / self.entry_price
                        else:  # SHORT
                            current_pnl = (self.entry_price - current_price) / self.entry_price
                        
                        # 이전 스텝 대비 수익률의 변화량 계산
                        pnl_change = current_pnl - self.prev_pnl
                        
                        # [수정] 캔들 인덱스 차이로 보유 시간 계산 (과거 데이터 학습용)
                        holding_time = (self.data_collector.current_index - self.entry_index) if self.entry_index is not None else 0
                        reward = self.env.calculate_reward(current_pnl, False, holding_time, pnl_change)
                        
                        # 다음 스텝을 위해 현재 pnl 저장
                        self.prev_pnl = current_pnl
                
                # 7. 트랜지션 저장
                is_terminal = (step == actual_steps - 1)  # 에피소드 마지막 스텝 여부
                self.agent.store_transition(state, action, log_prob, reward, is_terminal)
                
                episode_reward += reward
                steps += 1
                self.total_steps += 1
                
                # 8. [과적합 테스트] 주기적 업데이트 제거
                # 배치 사이즈를 에피소드 전체로 변경하여 안정성 확보
                # 에피소드 종료 후 한 번에 업데이트하도록 변경됨
                
                # 저장된 데이터는 자동으로 다음 캔들로 진행됨 (대기 불필요)
                
            except KeyboardInterrupt:
                logger.info("학습 중단 요청")
                raise
            except Exception as e:
                logger.error(f"에피소드 실행 중 오류: {e}", exc_info=True)
                time.sleep(5)
                continue
        
        # [과적합 테스트] 에피소드가 끝나면 한 번에 업데이트
        # 이렇게 하면 1000개의 데이터를 통째로 보고 판단하므로 훨씬 안정적입니다.
        if len(self.agent.memory) > 0:
            # next_state는 에피소드 끝났으므로 None
            self.agent.update(next_state=None, episode=episode_num)
            logger.info(f"🚀 에피소드 종료 후 전체 업데이트 (데이터: {len(self.agent.memory)}개)")
        
        return episode_reward, steps
    
    def train(self, num_episodes=100, max_steps_per_episode=100, save_interval=10, overfitting_test=False, fixed_start_index=1000):
        """모델 학습
        
        Args:
            num_episodes: 총 에피소드 수
            max_steps_per_episode: 에피소드당 최대 스텝 수
            save_interval: 모델 저장 간격
            overfitting_test: 과적합 테스트 모드 (True면 고정 인덱스 사용)
            fixed_start_index: 과적합 테스트 시 고정 시작 인덱스
        """
        logger.info("=" * 60)
        if overfitting_test:
            logger.info("🧪 [과적합 테스트 모드] PPO 모델 학습 시작")
            logger.info(f"⚠️  고정 시작 인덱스: {fixed_start_index}, 최대 스텝: {max_steps_per_episode}")
            logger.info("⚠️  같은 데이터를 반복 학습하여 보상이 폭발적으로 상승하는지 확인합니다.")
        else:
            logger.info("🚀 PPO 모델 학습 시작")
        logger.info("=" * 60)
        logger.info(f"에피소드 수: {num_episodes}")
        logger.info(f"에피소드당 최대 스텝: {max_steps_per_episode}")
        logger.info(f"모델 저장 간격: {save_interval} 에피소드")
        logger.info("=" * 60)
        
        best_reward = float('-inf')
        
        for episode in range(1, num_episodes + 1):
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"📚 에피소드 {episode}/{num_episodes}")
                logger.info(f"{'=' * 60}")
                
                # 에피소드 실행 (에피소드 번호 전달)
                result = self.train_episode(
                    episode_num=episode, 
                    max_steps=max_steps_per_episode,
                    overfitting_test=overfitting_test,
                    fixed_start_index=fixed_start_index
                )
                if result is None:
                    logger.warning("에피소드 실패, 다음 에피소드로 진행")
                    continue
                
                episode_reward, steps = result
                self.episode_rewards.append(episode_reward)
                
                # 실시간 그래프 업데이트 (시각화 활성화 시에만)
                if self.visualizer is not None:
                    self.visualizer.update(episode_reward)
                
                # 통계 출력
                avg_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                logger.info(f"✅ 에피소드 {episode} 완료")
                logger.info(f"   총 보상: {episode_reward:.4f}")
                logger.info(f"   스텝 수: {steps}")
                logger.info(f"   최근 10개 평균 보상: {avg_reward:.4f}")
                
                # 최고 성능 모델 저장
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    os.makedirs(os.path.dirname(config.AI_MODEL_PATH), exist_ok=True)
                    self.agent.save_model(config.AI_MODEL_PATH)
                    logger.info(f"🏆 최고 성능 모델 저장: 보상 {best_reward:.4f}")
                
                # 주기적 저장
                elif episode % save_interval == 0:
                    os.makedirs(os.path.dirname(config.AI_MODEL_PATH), exist_ok=True)
                    self.agent.save_model(config.AI_MODEL_PATH)
                    logger.info(f"💾 모델 저장 (에피소드 {episode})")
                
            except KeyboardInterrupt:
                logger.info("학습 중단")
                break
            except Exception as e:
                logger.error(f"에피소드 {episode} 실패: {e}", exc_info=True)
                continue
        
        # 최종 모델 저장
        os.makedirs(os.path.dirname(config.AI_MODEL_PATH), exist_ok=True)
        self.agent.save_model(config.AI_MODEL_PATH)
        logger.info("=" * 60)
        logger.info("✅ 학습 완료")
        logger.info(f"총 스텝: {self.total_steps}")
        logger.info(f"평균 보상: {sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0:.4f}")
        logger.info(f"모델 저장 위치: {config.AI_MODEL_PATH}")
        logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO 모델 학습')
    parser.add_argument('--episodes', type=int, default=100, help='학습 에피소드 수')
    parser.add_argument('--steps', type=int, default=1000, help='에피소드당 최대 스텝 수 (기본값: 1000, 큰 추세 학습용)')
    parser.add_argument('--save-interval', type=int, default=10, help='모델 저장 간격 (에피소드)')
    parser.add_argument('--no-visualize', action='store_true', help='시각화 비활성화')
    parser.add_argument('--overfitting-test', action='store_true', help='과적합 테스트 모드: 고정 인덱스로 같은 데이터 반복 학습')
    parser.add_argument('--fixed-start-index', type=int, default=1000, help='과적합 테스트 시 고정 시작 인덱스 (기본값: 1000)')
    
    args = parser.parse_args()
    
    try:
        trainer = PPOTrainer(enable_visualization=not args.no_visualize)
        trainer.train(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_interval=args.save_interval,
            overfitting_test=args.overfitting_test,
            fixed_start_index=args.fixed_start_index
        )
    except KeyboardInterrupt:
        logger.info("학습 중단")
    except Exception as e:
        logger.error(f"학습 실패: {e}", exc_info=True)
        sys.exit(1)
