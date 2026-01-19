"""
PPO 모델 학습 스크립트
별도로 실행하여 모델을 학습하고 저장합니다.
"""
import logging
import os
import sys
import time
from datetime import datetime

# 상위 폴더를 경로에 추가 (config, core, strategies 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import DataCollector, BinanceClient
from core.indicators import Indicators
from strategies import (
    BTCEthCorrelationStrategy,
    CVDDeltaStrategy,
    VolatilitySqueezeStrategy,
    OrderblockFVGStrategy,
    LiquidationSpikeStrategy,
    BollingerMeanReversionStrategy,
    VWAPDeviationStrategy,
    RangeTopBottomStrategy,
    StochRSIMeanReversionStrategy,
    CVDFakePressureStrategy
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


class PPOTrainer:
    """PPO 모델 학습 클래스"""
    def __init__(self):
        # 저장된 데이터 사용 (학습용)
        self.data_collector = DataCollector(use_saved_data=True)
        self.client = BinanceClient()
        
        # 전략 초기화
        self.breakout_strategies = []
        self.range_strategies = []
        
        # 폭발장 전략
        if config.STRATEGIES.get('btc_eth_correlation', False):
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES.get('cvd_delta', False):
            self.breakout_strategies.append(CVDDeltaStrategy())
        if config.STRATEGIES.get('volatility_squeeze', False):
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES.get('orderblock_fvg', False):
            self.breakout_strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('liquidation_spike', False) and self.client.use_futures:
            self.breakout_strategies.append(LiquidationSpikeStrategy())
        
        # 횡보장 전략
        if config.STRATEGIES.get('bollinger_mean_reversion', False):
            self.range_strategies.append(BollingerMeanReversionStrategy())
        if config.STRATEGIES.get('vwap_deviation', False):
            self.range_strategies.append(VWAPDeviationStrategy())
        if config.STRATEGIES.get('range_top_bottom', False):
            self.range_strategies.append(RangeTopBottomStrategy())
        if config.STRATEGIES.get('stoch_rsi_mean_reversion', False):
            self.range_strategies.append(StochRSIMeanReversionStrategy())
        if config.STRATEGIES.get('cvd_fake_pressure', False):
            self.range_strategies.append(CVDFakePressureStrategy())
        
        self.strategies = self.breakout_strategies + self.range_strategies
        
        if len(self.strategies) == 0:
            raise ValueError("활성화된 전략이 없습니다. config.py에서 전략을 활성화하세요.")
        
        logger.info(f"전략 초기화 완료: {len(self.strategies)}개 전략")
        
        # 트레이딩 환경 생성
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        state_dim = self.env.get_state_dim()
        action_dim = 3  # 0: Hold, 1: Long, 2: Short
        
        # PPO 에이전트 생성
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"디바이스: {device}")
        self.agent = PPOAgent(state_dim, action_dim, hidden_dim=128, device=device)
        
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
        self.entry_time = None
        self.episode_rewards = []
        self.total_steps = 0
        
    def train_episode(self, max_steps=100):
        """한 에피소드 학습"""
        episode_reward = 0.0
        steps = 0
        
        # 저장된 데이터에서 인덱스 리셋 (새 에피소드 시작)
        self.data_collector.reset_index()
        
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
                
                # 2. 상태 관측 (전처리 파이프라인 포함)
                # get_candles가 현재 인덱스 기준으로 이전 lookback개를 반환
                state = self.env.get_observation()
                if state is None:
                    logger.warning("상태 관측 실패, 다음 캔들로 진행")
                    continue
                
                # 2. 행동 선택
                action, log_prob = self.agent.select_action(state)
                action_names = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
                action_name = action_names[action]
                
                # 3. 현재 가격 확인 (현재 인덱스의 캔들)
                if self.data_collector.current_index > 0:
                    current_candle = self.data_collector.eth_data.iloc[self.data_collector.current_index - 1]
                    current_price = float(current_candle['close'])
                else:
                    continue
                
                # 4. 보상 계산 및 포지션 업데이트
                reward = 0.0
                trade_done = False
                
                if action == 1:  # LONG
                    if self.current_position != 'LONG':
                        # 기존 포지션 청산
                        if self.current_position == 'SHORT' and self.entry_price:
                            pnl = (self.entry_price - current_price) / self.entry_price
                            reward = self.env.calculate_reward(pnl, True)
                            trade_done = True
                            logger.info(f"💰 숏 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}")
                        
                        # 롱 진입
                        self.current_position = 'LONG'
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                        logger.debug(f"📈 롱 진입: ${current_price:.2f}")
                
                elif action == 2:  # SHORT
                    if self.current_position != 'SHORT':
                        # 기존 포지션 청산
                        if self.current_position == 'LONG' and self.entry_price:
                            pnl = (current_price - self.entry_price) / self.entry_price
                            reward = self.env.calculate_reward(pnl, True)
                            trade_done = True
                            logger.info(f"💰 롱 청산: 수익률 {pnl:.2%}, 보상: {reward:.4f}")
                        
                        # 숏 진입
                        self.current_position = 'SHORT'
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                        logger.debug(f"📉 숏 진입: ${current_price:.2f}")
                
                else:  # HOLD
                    # 보유 중인 포지션의 수익률 계산
                    if self.current_position and self.entry_price:
                        if self.current_position == 'LONG':
                            pnl = (current_price - self.entry_price) / self.entry_price
                        else:  # SHORT
                            pnl = (self.entry_price - current_price) / self.entry_price
                        
                        holding_time = (datetime.now() - self.entry_time).total_seconds() / 60 if self.entry_time else 0
                        reward = self.env.calculate_reward(pnl, False, holding_time)
                
                # 5. 트랜지션 저장
                is_terminal = False
                self.agent.store_transition(state, action, log_prob, reward, is_terminal)
                
                episode_reward += reward
                steps += 1
                self.total_steps += 1
                
                # 6. 주기적 업데이트 (10개 트랜지션마다)
                if len(self.agent.memory) >= 10:
                    logger.info(f"🔄 모델 업데이트 중... (메모리: {len(self.agent.memory)}개)")
                    self.agent.update()
                    logger.info("✅ 업데이트 완료")
                
                # 저장된 데이터는 자동으로 다음 캔들로 진행됨 (대기 불필요)
                
            except KeyboardInterrupt:
                logger.info("학습 중단 요청")
                raise
            except Exception as e:
                logger.error(f"에피소드 실행 중 오류: {e}", exc_info=True)
                time.sleep(5)
                continue
        
        return episode_reward, steps
    
    def train(self, num_episodes=100, max_steps_per_episode=100, save_interval=10):
        """모델 학습"""
        logger.info("=" * 60)
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
                
                # 에피소드 실행
                result = self.train_episode(max_steps=max_steps_per_episode)
                if result is None:
                    logger.warning("에피소드 실패, 다음 에피소드로 진행")
                    continue
                
                episode_reward, steps = result
                self.episode_rewards.append(episode_reward)
                
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
    parser.add_argument('--steps', type=int, default=100, help='에피소드당 최대 스텝 수')
    parser.add_argument('--save-interval', type=int, default=10, help='모델 저장 간격 (에피소드)')
    
    args = parser.parse_args()
    
    try:
        trainer = PPOTrainer()
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
