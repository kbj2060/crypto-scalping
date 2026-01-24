"""
SAC (Soft Actor-Critic) 모델 학습 스크립트
연속형 행동 공간을 사용하는 Off-policy 알고리즘
"""
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
    
    def __init__(self):
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
        
        self.range_strategies.append(WilliamsRStrategy())
        
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"✅ 전략 초기화 완료: 총 {len(self.strategies)}개")
        
        # 3. 피처 엔지니어링 (전체 데이터에 대해 한 번만 수행)
        self._precalculate_features()
        
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
        
        # 기존 모델 로드 (있는 경우)
        model_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model')
        if os.path.exists(model_path):
            try:
                self.agent.load_model(model_path)
                logger.info(f"✅ 기존 모델 로드: {model_path}")
            except Exception as e:
                logger.warning(f"모델 로드 실패 (새 모델로 시작): {e}")
        else:
            logger.info("새 모델로 학습 시작")
        
        # 학습 상태
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        self.prev_pnl = 0.0
        self.episode_rewards = []
        self.total_steps = 0
    
    def _precalculate_features(self):
        """전체 데이터에 대해 피처 엔지니어링 수행 (한 번만)"""
        try:
            logger.info("📊 전체 데이터 피처 엔지니어링 수행 중...")
            
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
            
            logger.info(f"✅ 피처 엔지니어링 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}", exc_info=True)
            raise
    
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
    
    def interpret_action(self, action_value):
        """
        연속형 액션(-1 ~ 1)을 트레이딩 명령으로 변환
        
        Args:
            action_value: float, -1 ~ 1 사이의 연속값
        Returns:
            int: 0=HOLD, 1=LONG, 2=SHORT
        """
        threshold = 0.3
        
        if action_value > threshold:
            return 1  # LONG
        elif action_value < -threshold:
            return 2  # SHORT
        else:
            return 0  # HOLD
    
    def train_episode(self, episode_num, max_steps=None):
        """한 에피소드 학습"""
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        # 배치 사이즈 Config에서 가져오기 (기본값 256)
        batch_size = config.SAC_BATCH_SIZE
        
        episode_reward = 0.0
        steps = 0
        
        # 저장된 데이터에서 인덱스 리셋 (새 에피소드 시작 - 무작위 시작 인덱스)
        self.data_collector.reset_index(max_steps=max_steps, random_start=True)
        
        # 에피소드 시작 시 이전 수익률 초기화
        self.prev_pnl = 0.0
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        
        # 초기 데이터 확인
        if self.data_collector.eth_data is None or len(self.data_collector.eth_data) == 0:
            logger.error("저장된 데이터가 없습니다.")
            return None
        
        # 사용 가능한 최대 스텝 수 계산
        available_steps = len(self.data_collector.eth_data) - self.data_collector.current_index
        actual_steps = min(max_steps, available_steps)
        
        if actual_steps <= 0:
            logger.warning("사용 가능한 데이터가 부족합니다.")
            return None
        
        logger.info(f"에피소드 시작: 총 {len(self.data_collector.eth_data)}개 캔들 중 {actual_steps}개 사용")
        
        for step in range(actual_steps):
            try:
                # 1. 인덱스 증가 (다음 캔들로 이동)
                if self.data_collector.current_index >= len(self.data_collector.eth_data):
                    break
                
                self.data_collector.current_index += 1
                
                # 2. 포지션 정보 수집
                pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
                hold_val = (self.data_collector.current_index - self.entry_index) / max_steps if self.entry_index is not None else 0.0
                pnl_val = self.prev_pnl * 10
                pos_info = [pos_val, pnl_val, hold_val]
                
                # 3. 상태 관측
                state = self.env.get_observation(position_info=pos_info)
                if state is None:
                    continue
                
                # 4. 행동 선택 (SAC - 연속형)
                action_continuous = self.agent.select_action(state)  # 예: [0.75]
                action_discrete = self.interpret_action(action_continuous[0])
                
                # 5. 현재 가격 확인
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
                
                if action_discrete == 1:  # LONG
                    if self.current_position != 'LONG':
                        # 기존 포지션 청산
                        if self.current_position == 'SHORT' and self.entry_price:
                            pnl = (self.entry_price - current_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl
                            reward = self.env.calculate_reward(pnl, True, holding_time=0, pnl_change=pnl_change)
                            trade_done = True
                            self.prev_pnl = 0.0
                        
                        # 롱 진입
                        self.current_position = 'LONG'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index
                        self.prev_pnl = 0.0
                
                elif action_discrete == 2:  # SHORT
                    if self.current_position != 'SHORT':
                        # 기존 포지션 청산
                        if self.current_position == 'LONG' and self.entry_price:
                            pnl = (current_price - self.entry_price) / self.entry_price
                            pnl_change = pnl - self.prev_pnl
                            reward = self.env.calculate_reward(pnl, True, holding_time=0, pnl_change=pnl_change)
                            trade_done = True
                            self.prev_pnl = 0.0
                        
                        # 숏 진입
                        self.current_position = 'SHORT'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index
                        self.prev_pnl = 0.0
                
                else:  # HOLD
                    # 보유 중인 포지션의 수익률 계산
                    if self.current_position and self.entry_price:
                        if self.current_position == 'LONG':
                            current_pnl = (current_price - self.entry_price) / self.entry_price
                        else:
                            current_pnl = (self.entry_price - current_price) / self.entry_price
                        
                        pnl_change = current_pnl - self.prev_pnl
                        holding_time = (self.data_collector.current_index - self.entry_index) if self.entry_index is not None else 0
                        reward = self.env.calculate_reward(current_pnl, False, holding_time, pnl_change)
                        self.prev_pnl = current_pnl
                
                # 7. 다음 상태 관측
                next_pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
                next_hold_val = (self.data_collector.current_index + 1 - self.entry_index) / max_steps if self.entry_index is not None else 0.0
                next_pnl_val = self.prev_pnl * 10
                next_pos_info = [next_pos_val, next_pnl_val, next_hold_val]
                
                # 임시로 인덱스 증가하여 다음 상태 관측
                temp_index = self.data_collector.current_index
                if temp_index < len(self.data_collector.eth_data):
                    self.data_collector.current_index += 1
                    next_state = self.env.get_observation(position_info=next_pos_info)
                    self.data_collector.current_index = temp_index
                else:
                    next_state = None
                
                # 8. Replay Buffer 저장 (연속형 액션 저장)
                is_terminal = (step == actual_steps - 1)
                self.agent.memory.push(state, action_continuous, reward, next_state, is_terminal)
                
                episode_reward += reward
                steps += 1
                self.total_steps += 1
                
                # 9. 학습 (매 스텝마다 배치를 뽑아서 학습)
                # 메모리가 배치 사이즈보다 클 때만 업데이트
                if len(self.agent.memory) > batch_size:
                    c_loss, a_loss, alpha = self.agent.update(batch_size=batch_size)
                    # [중요] LR 스케줄러 업데이트
                    self.agent.step_schedulers()
                    if step % 100 == 0:
                        current_lr = self.agent.actor_scheduler.get_last_lr()[0] if self.agent.actor_scheduler else config.SAC_LEARNING_RATE
                        logger.debug(f"Step {step}: Critic Loss={c_loss:.4f}, Actor Loss={a_loss:.4f}, Alpha={alpha:.4f}, LR={current_lr:.6f}")
                
            except KeyboardInterrupt:
                logger.info("학습 중단 요청")
                raise
            except Exception as e:
                logger.error(f"에피소드 실행 중 오류: {e}", exc_info=True)
                time.sleep(5)
                continue
        
        return episode_reward, steps
    
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
        logger.info("🚀 SAC 모델 학습 시작")
        logger.info("=" * 60)
        logger.info(f"에피소드 수: {num_episodes}")
        logger.info(f"에피소드당 최대 스텝: {max_steps_per_episode}")
        logger.info(f"모델 저장 간격: {save_interval} 에피소드")
        logger.info(f"스케줄러 설정: 총 {total_steps} 스텝, Warmup {warmup_ratio*100:.1f}%")
        logger.info("=" * 60)
        
        # 스케줄러 설정
        self.agent.setup_schedulers(total_steps, warmup_ratio)
        
        # 스케일러 저장 경로 설정
        scaler_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model').replace('.pth', '_scaler.pkl')
        if not scaler_path.endswith('.pkl'):
            scaler_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model') + '_scaler.pkl'
        
        model_path = config.AI_MODEL_PATH.replace('ppo_model', 'sac_model')
        
        best_reward = float('-inf')
        
        for episode in range(1, num_episodes + 1):
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"📚 에피소드 {episode}/{num_episodes}")
                logger.info(f"{'=' * 60}")
                
                # 에피소드 실행
                result = self.train_episode(episode_num=episode, max_steps=max_steps_per_episode)
                if result is None:
                    logger.warning("에피소드 실패, 다음 에피소드로 진행")
                    continue
                
                episode_reward, steps = result
                self.episode_rewards.append(episode_reward)
                
                # 통계 출력
                avg_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                # 로그 출력 시 현재 LR도 함께 출력
                current_lr = self.agent.actor_scheduler.get_last_lr()[0] if self.agent.actor_scheduler else config.SAC_LEARNING_RATE
                logger.info(f"✅ 에피소드 {episode} 완료")
                logger.info(f"   총 보상: {episode_reward:.4f}")
                logger.info(f"   스텝 수: {steps}")
                logger.info(f"   최근 10개 평균 보상: {avg_reward:.4f}")
                logger.info(f"   현재 학습률: {current_lr:.6f}")
                
                # 최고 성능 모델 저장
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.agent.save_model(model_path)
                    self.env.preprocessor.save(scaler_path)
                    logger.info(f"🏆 최고 성능 모델 & 스케일러 저장 완료: 보상 {best_reward:.4f}")
                
                # 주기적 저장
                elif episode % save_interval == 0:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.agent.save_model(model_path)
                    self.env.preprocessor.save(scaler_path)
                    logger.info(f"💾 정기 저장 완료 (에피소드 {episode})")
                
            except KeyboardInterrupt:
                logger.info("학습 중단")
                break
            except Exception as e:
                logger.error(f"에피소드 {episode} 실패: {e}", exc_info=True)
                continue
        
        # 최종 모델 저장
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.agent.save_model(model_path)
        self.env.preprocessor.save(scaler_path)
        logger.info("=" * 60)
        logger.info("✅ 학습 및 스케일러 저장 완료")
        logger.info(f"총 스텝: {self.total_steps}")
        logger.info(f"평균 보상: {sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0:.4f}")
        logger.info(f"모델 저장 위치: {model_path}")
        logger.info(f"스케일러 저장 위치: {scaler_path}")
        logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC 모델 학습')
    parser.add_argument('--episodes', type=int, default=1000, help='학습 에피소드 수')
    parser.add_argument('--steps', type=int, default=480, help='에피소드당 최대 스텝 수')
    parser.add_argument('--save-interval', type=int, default=50, help='모델 저장 간격 (에피소드)')
    
    args = parser.parse_args()
    
    try:
        trainer = SACTrainer()
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
