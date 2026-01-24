"""
PPO 학습 스크립트 (과잉 거래 방지 & 로그 최적화)
- 최소 보유 시간(Min Holding Time) 3캔들 적용 -> 잦은 매매 방지
- 스텝별 로그 제거 -> 진행바(tqdm)로 깔끔하게 확인
- Action 0의 의미 변경: 유지 -> 청산(Exit)
- AI가 포지션을 유지하려면 계속해서 1(Long)이나 2(Short)를 내뱉어야 함
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt  # 시각화
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

# 로깅 설정 (로그 레벨 조정 - 불필요한 로그 생략)
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

# 피처 엔지니어링 로그 끄기
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

# 병렬 처리 헬퍼 함수 (선택적)
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.info("joblib이 설치되지 않았습니다. 순차 처리로 전략을 계산합니다.")

def calculate_chunk(start_idx, end_idx, strategies, collector_data):
    """전략 계산 청크 (병렬 처리용)"""
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
        
        logger.info(f"전략 초기화 완료: {len(self.strategies)}개 전략")
        
        # 1. 피처 데이터 로드
        self._load_features()
        
        # 2. 전략 사전 계산 (병렬 처리)
        self.precalculate_strategies_parallel()
        
        # 3. 환경 설정
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler()

        # 4. 에이전트 설정
        state_dim = self.env.get_state_dim()
        action_dim = 3  # HOLD, LONG, SHORT
        info_dim = len(self.strategies) + 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"디바이스: {device}")
        logger.info(f"정보 차원: {info_dim} (전략 {len(self.strategies)}개 + 포지션 정보 3개)")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        # 모델 로드
        if os.path.exists(config.AI_MODEL_PATH):
            try:
                self.agent.load_model(config.AI_MODEL_PATH)
                logger.info(f"✅ 기존 모델 로드 완료")
            except Exception as e:
                logger.warning(f"모델 로드 실패: {e}")
        
        self.episode_rewards = []
        self.avg_rewards = []  # 평균 리워드 추적용
        
        # [NEW] 실시간 그래프 설정
        try:
            plt.ion()  # Interactive Mode On
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_title('PPO Real-time Training')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.grid(True, alpha=0.3)
            self.line1, = self.ax.plot([], [], label='Reward', alpha=0.3, color='gray')
            self.line2, = self.ax.plot([], [], label='Avg (10)', color='red', linewidth=2)
            self.ax.legend()
            self.plotting_enabled = True
        except Exception as e:
            logger.warning(f"그래프 초기화 실패 (계속 진행): {e}")
            self.plotting_enabled = False

    def _load_features(self):
        """피처 파일 로드 (전략 캐시 포함)"""
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if os.path.exists(path):
            logger.info("📂 피처 파일 로드")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # 전략 캐시 파일이 있으면 병합
            if os.path.exists(cached_strategies_path):
                logger.info("📂 전략 캐시 파일 발견, 병합 중...")
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    # 전략 컬럼만 병합 (strategy_0, strategy_1, ...)
                    strategy_cols = [col for col in cached_df.columns if col.startswith('strategy_')]
                    if strategy_cols:
                        for col in strategy_cols:
                            if col in cached_df.columns:
                                df[col] = cached_df[col]
                        logger.info(f"✅ 전략 캐시 병합 완료: {len(strategy_cols)}개 전략")
                except Exception as e:
                    logger.warning(f"전략 캐시 병합 실패: {e}")
            
            self.data_collector.eth_data = df
        else:
            logger.error("❌ 피처 파일 없음. 먼저 피처를 생성하세요.")
            sys.exit(1)

    def precalculate_strategies_parallel(self):
        """전략 신호 병렬 계산 (캐시 확인 포함)"""
        df = self.data_collector.eth_data
        
        # 이미 계산되어 있으면 스킵
        if 'strategy_0' in df.columns:
            logger.info("✅ 전략 신호 이미 존재 (계산 생략)")
            return

        # cached_strategies.csv 파일 확인
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(cached_strategies_path):
            logger.info("📂 전략 캐시 파일 발견, 로드 중...")
            try:
                cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                strategy_cols = [col for col in cached_df.columns if col.startswith('strategy_')]
                if strategy_cols and len(strategy_cols) == len(self.strategies):
                    # 인덱스가 일치하는지 확인
                    if len(cached_df) == len(df):
                        for col in strategy_cols:
                            df[col] = cached_df[col]
                        logger.info(f"✅ 전략 캐시 로드 완료: {len(strategy_cols)}개 전략")
                        # training_features.csv에도 저장
                        df.to_csv('data/training_features.csv', index=True)
                        self.data_collector.eth_data = df
                        return
                    else:
                        logger.warning(f"캐시 파일 크기 불일치 (캐시: {len(cached_df)}, 피처: {len(df)}), 재계산합니다.")
                else:
                    logger.warning(f"캐시 파일 전략 개수 불일치 (캐시: {len(strategy_cols)}, 필요: {len(self.strategies)}), 재계산합니다.")
            except Exception as e:
                logger.warning(f"전략 캐시 로드 실패: {e}, 재계산합니다.")

        logger.info("🧠 전략 신호 계산 시작...")
        total_len = len(df)
        start_idx = config.LOOKBACK + 50
        
        # 병렬 처리 시도
        if JOBLIB_AVAILABLE and total_len > 10000:
            try:
                n_jobs = max(1, cpu_count() - 1)
                chunk_size = (total_len - start_idx) // n_jobs
                chunks = [(start_idx + i*chunk_size, 
                          start_idx + (i+1)*chunk_size if i < n_jobs-1 else total_len) 
                         for i in range(n_jobs)]
                
                logger.info(f"병렬 처리: {n_jobs}개 작업으로 분할")
                results_list = Parallel(n_jobs=n_jobs)(
                    delayed(calculate_chunk)(s, e, self.strategies, df) 
                    for s, e in chunks
                )
                
                # 결과 병합
                for s_idx in range(len(self.strategies)):
                    col_name = f'strategy_{s_idx}'
                    df[col_name] = 0.0
                    full_series = np.zeros(total_len)
                    for chunk_idx, (s, e) in enumerate(chunks):
                        full_series[s:e] = results_list[chunk_idx][col_name]
                    df[col_name] = full_series
                
                logger.info("✅ 병렬 계산 완료")
            except Exception as e:
                logger.warning(f"병렬 처리 실패, 순차 처리로 전환: {e}")
                self._precalculate_strategies_sequential(df, start_idx, total_len)
        else:
            # 순차 처리
            self._precalculate_strategies_sequential(df, start_idx, total_len)
        
        # 저장 (training_features.csv와 cached_strategies.csv 모두 저장)
        df.to_csv('data/training_features.csv', index=True)
        # 전략 컬럼만 별도로 저장 (캐시용)
        strategy_cols = [col for col in df.columns if col.startswith('strategy_')]
        if strategy_cols:
            cached_df = df[strategy_cols].copy()
            cached_df.to_csv('data/cached_strategies.csv', index=True)
            logger.info(f"💾 전략 캐시 저장 완료: {len(strategy_cols)}개 전략")
        self.data_collector.eth_data = df

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
        """순차 처리 전략 계산"""
        for i in range(len(self.strategies)):
            df[f'strategy_{i}'] = 0.0
        
        original_index = getattr(self.data_collector, 'current_index', 0)
        
        try:
            iterator = tqdm(range(start_idx, total_len), desc="Strategy Calc")
        except NameError:
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
                        if signal == 'LONG': score = conf
                        elif signal == 'SHORT': score = -conf
                    df.iat[i, df.columns.get_loc(f'strategy_{s_idx}')] = score
                except:
                    continue
        
        self.data_collector.current_index = original_index

    def _fit_global_scaler(self):
        """스케일러 학습"""
        try:
            logger.info("전역 스케일러 학습 시작 (Train Set 80%만 사용)...")
            df = self.data_collector.eth_data
            
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
                if col not in train_df.columns:
                    train_df[col] = 0.0
            
            sample_size = min(config.TRAIN_SAMPLE_SIZE, len(train_df))
            sampled_df = train_df.sample(n=sample_size)[target_cols]
            
            data_array = sampled_df.values.astype(np.float32)
            self.env.preprocessor.fit(data_array)
            self.env.scaler_fitted = True
            
            scaler_path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            if not scaler_path.endswith('.pkl'):
                scaler_path = config.AI_MODEL_PATH + '_scaler.pkl'
            self.env.preprocessor.save_scaler(scaler_path)
            
            logger.info(f"✅ 스케일러 학습 완료")
        except Exception as e:
            logger.error(f"스케일러 학습 실패: {e}", exc_info=True)

    def train_episode(self, episode_num, max_steps=None):
        """에피소드 학습"""
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        # 시작 인덱스 설정
        start_min = config.LOOKBACK + 100
        start_max = self.train_end_idx - max_steps - 50
        
        if start_max <= start_min:
            logger.error("학습 데이터 구간이 너무 짧습니다.")
            return None
        
        start_idx = np.random.randint(start_min, start_max)
        self.data_collector.current_index = start_idx
        
        # [중요] 변수 초기화 (NameError 해결)
        current_position = None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        trade_count = 0  # 거래 횟수 추적
        
        # [과잉 거래 방지] 최소 보유 시간 설정 (3~5 캔들)
        min_holding_steps = config.MIN_HOLDING_TIME if hasattr(config, 'MIN_HOLDING_TIME') else 3
        
        # LSTM 상태 초기화
        self.agent.reset_episode_states()
        
        # 진행바 설정
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num}", leave=False, unit="step")
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx:
                break
            
            # Position Info
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position is not None else 0
            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])
            
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]
            
            # 1. 관측
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None:
                break
            
            # 2. 행동
            action, prob = self.agent.select_action(state)
            
            # 3. 트레이딩 로직
            reward = 0.0
            trade_done = False
            realized_pnl = 0.0
            
            # [잠금 로직] 최소 보유 시간 미달 시 강제로 포지션 유지 (행동 무시)
            is_locked = (current_position is not None) and (holding_time < min_holding_steps)
            
            # A. 강제 청산 (Stop Loss - 잠금 무시하고 즉시 손절)
            if current_position is not None and unrealized_pnl < config.STOP_LOSS_THRESHOLD:
                realized_pnl = unrealized_pnl
                trade_done = True
                current_position = None
                entry_price = 0.0
                entry_index = 0
                trade_count += 1
            
            # B. 모델 행동 실행 (잠겨있지 않을 때만)
            elif not is_locked and not trade_done:
                if action == 1:  # LONG 신호
                    if current_position == 'SHORT':  # 스위칭 (Short -> Long)
                        realized_pnl = unrealized_pnl
                        trade_done = True
                        current_position = 'LONG'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                    elif current_position is None:  # 신규 진입 (Open Long)
                        current_position = 'LONG'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                        reward = 0  # 진입 시점엔 보상 0
                    # 이미 LONG이면 유지 (Maintain) - 아무것도 하지 않음
                        
                elif action == 2:  # SHORT 신호
                    if current_position == 'LONG':  # 스위칭 (Long -> Short)
                        realized_pnl = unrealized_pnl
                        trade_done = True
                        current_position = 'SHORT'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                    elif current_position is None:  # 신규 진입 (Open Short)
                        current_position = 'SHORT'
                        entry_price = curr_price
                        entry_index = current_idx
                        trade_count += 1
                        reward = 0
                    # 이미 SHORT면 유지 (Maintain) - 아무것도 하지 않음
                
                elif action == 0:  # EXIT / NEUTRAL 신호
                    if current_position is not None:
                        # [핵심 변경] Action 0이 나오면 포지션 청산!
                        realized_pnl = unrealized_pnl
                        trade_done = True
                        current_position = None
                        entry_price = 0.0
                        entry_index = 0
                        trade_count += 1
                    else:
                        # 포지션 없으면 계속 관망 (Stay)
                        reward = 0  # 관망에 대한 보상 (필요시 작은 양수 부여 가능)
            
            # 보상 계산 (거래 완료 시)
            if trade_done:
                reward = self.env.calculate_reward(realized_pnl, True, holding_time)
            else:
                # 포지션 유지 중에도 시간 페널티 적용
                if current_position is not None:
                    reward = self.env.calculate_reward(0.0, False, holding_time)
            
            # 4. 다음 상태 (안전하게 생성)
            next_idx = current_idx + 1
            self.data_collector.current_index = next_idx
            
            # Next Info 계산
            next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            next_hold_time = (next_idx - entry_index) if current_position is not None else 0
            
            next_un_pnl = 0.0
            if next_idx < len(self.data_collector.eth_data) and current_position is not None:
                try:
                    next_price = float(self.data_collector.eth_data.iloc[next_idx]['close'])
                    if current_position == 'LONG':
                        next_un_pnl = (next_price - entry_price) / entry_price
                    elif current_position == 'SHORT':
                        next_un_pnl = (entry_price - next_price) / entry_price
                except:
                    pass
            
            next_pos_info = [next_pos_val, next_un_pnl * 10, next_hold_time / max_steps]
            next_state = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)
            
            done = False if step < max_steps - 1 else True
            if next_state is None:
                done = True
                next_state = state  # Fallback
            
            # 5. 데이터 저장
            self.agent.put_data((state, action, reward, next_state, prob, done))
            episode_reward += reward
            
            # 진행바 업데이트 (현재 수익, 거래횟수 표시)
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count, 'P': f'{pos_val:.1f}'})
            
            if done:
                break
        
        pbar.close()
        # 에피소드 종료 후 학습
        loss = self.agent.train_net(episode=episode_num)
        return episode_reward

    def live_plot(self):
        """[NEW] 윈도우에 실시간으로 그래프 그리기"""
        if not self.plotting_enabled:
            return
        
        try:
            x = range(len(self.episode_rewards))
            
            # 데이터 업데이트
            self.line1.set_data(x, self.episode_rewards)
            self.line2.set_data(x, self.avg_rewards)
            
            # 축 범위 자동 조정
            self.ax.relim()
            self.ax.autoscale_view()
            
            # 화면 갱신
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # 약간의 딜레이 (GUI 반응용)
            plt.pause(0.01)
            
        except Exception:
            pass

    def train(self, num_episodes=1000):
        """학습 메인 루프"""
        logger.info("🚀 PPO 학습 시작 (Best Model Separation + Real-time Plotting)")
        
        best_reward = -float('inf')
        
        # 경로 설정 (확장자 분리)
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        best_model_path = f"{base_path}_best.pth"
        best_scaler_path = f"{base_path}_best_scaler.pkl"
        last_model_path = f"{base_path}_last.pth"
        last_scaler_path = f"{base_path}_last_scaler.pkl"
        
        # 초기 스케일러 저장 (Last에 저장)
        self.env.preprocessor.save(last_scaler_path)
        
        for ep in range(1, num_episodes + 1):
            try:
                reward = self.train_episode(ep)
                if reward is None:
                    continue
                
                self.episode_rewards.append(reward)
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.avg_rewards.append(avg_reward)
                
                # 로그는 매번 출력 (진행 상황 확인용)
                logger.info(f"✅ Ep {ep}: Reward {reward:.4f} | Avg {avg_reward:.4f}")
                
                # [NEW] 실시간 그래프 업데이트
                self.live_plot()
                
                # [핵심] 최고 기록 갱신 시 -> '_best' 파일에 저장
                if reward > best_reward:
                    best_reward = reward
                    logger.info(f"🏆 신기록! ({best_reward:.4f}) -> Best 모델 저장")
                    
                    self.agent.save_model(best_model_path)
                    self.env.preprocessor.save(best_scaler_path)
                
                # [핵심] 정기 저장 -> '_last' 파일에 저장 (혹은 매번 저장)
                # 에러 등으로 멈췄을 때 이어서 하기 위함
                if ep % 10 == 0:
                    self.agent.save_model(last_model_path)
                    self.env.preprocessor.save(last_scaler_path)
                    
            except Exception as e:
                logger.error(f"Ep {ep} Fail: {e}")
                continue
        
        # 학습 종료 시 그래프 창 유지
        if self.plotting_enabled:
            try:
                plt.ioff()  # Interactive Mode Off
                logger.info("그래프 창을 닫으려면 창을 직접 닫아주세요.")
            except:
                pass

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)
