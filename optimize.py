"""
Optuna Hyperparameter Optimization Script
PPO 모델의 핵심 하이퍼파라미터를 자동으로 튜닝하여 최적의 조합을 찾습니다.
"""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
import os
import sys
import numpy as np
import torch
import json
from tqdm import tqdm

# 기존 모듈 임포트
from model import config
from core import DataCollector
from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)

# 로깅 설정 (Optuna 로그만 보이게 조정)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptunaOptimizer")
logging.getLogger("model").setLevel(logging.WARNING)
logging.getLogger("core").setLevel(logging.WARNING)

class OptimizationObjective:
    def __init__(self):
        # 데이터는 한 번만 로드하여 메모리에 유지 (속도 최적화)
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        # 피처 로드 및 스케일러 미리 준비
        self._prepare_data()

    def _prepare_data(self):
        # 1. 피처 데이터 로드
        import pandas as pd
        path = 'data/training_features.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            self.data_collector.eth_data = df
        
        # 2. 환경 초기화 및 스케일러 학습
        self.base_env = TradingEnvironment(self.data_collector, self.strategies)
        
        # 스케일러 학습 (train_ppo.py 로직 차용)
        if not self.base_env.scaler_fitted:
            df = self.data_collector.eth_data
            train_size = int(len(df) * 0.7)
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
            self.base_env.preprocessor.fit(sample)
            self.base_env.scaler_fitted = True

    def __call__(self, trial):
        # ============================================================
        # 1. 하이퍼파라미터 제안 (Search Space)
        # ============================================================
        
        # 학습 관련
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        
        # 모델 구조 관련
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128]) # 256은 너무 무거움
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        
        # Config 덮어쓰기 (전역 설정 변경)
        config.PPO_LEARNING_RATE = lr
        config.PPO_GAMMA = gamma
        config.PPO_EPS_CLIP = clip_eps
        config.PPO_ENTROPY_COEF = entropy_coef
        config.TRAIN_BATCH_SIZE = batch_size
        config.NETWORK_HIDDEN_DIM = hidden_dim
        config.NETWORK_DROPOUT = dropout

        # ============================================================
        # 2. 에이전트 및 환경 초기화
        # ============================================================
        state_dim = self.base_env.get_state_dim()
        action_dim = 4
        info_dim = 15
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 매 Trial마다 새로운 에이전트 생성
        agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        # 짧은 학습 수행 (예: 50 에피소드만 돌려서 가능성 타진)
        N_EPISODES = 30  # 최적화 속도를 위해 짧게 설정
        MAX_STEPS = 480
        
        total_rewards = []
        
        # ============================================================
        # 3. 학습 루프 (축소판)
        # ============================================================
        for ep in range(1, N_EPISODES + 1):
            # 에피소드 시작점 랜덤 선택
            start_min = config.LOOKBACK + 100
            start_max = self.train_end_idx - MAX_STEPS - 50
            start_idx = np.random.randint(start_min, start_max)
            
            self.data_collector.current_index = start_idx
            self.base_env.reset_reward_states()
            agent.reset_episode_states()
            
            current_position = None
            entry_price = 0.0
            entry_index = 0
            ep_reward = 0.0
            
            # Step Loop
            for step in range(MAX_STEPS):
                curr_idx = self.data_collector.current_index
                if curr_idx >= self.train_end_idx: break
                
                curr_price = float(self.data_collector.eth_data.iloc[curr_idx]['close'])
                
                # PnL 계산
                unrealized_pnl = 0.0
                if current_position == 'LONG':
                    unrealized_pnl = (curr_price - entry_price) / entry_price
                elif current_position == 'SHORT':
                    unrealized_pnl = (entry_price - curr_price) / entry_price
                
                # State 생성
                pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
                holding_time = (curr_idx - entry_index) if current_position else 0
                pos_info = [pos_val, unrealized_pnl * 10, holding_time / MAX_STEPS]
                state = self.base_env.get_observation(position_info=pos_info, current_index=curr_idx)
                
                if state is None: break
                
                # Action 선택
                action, prob, val = agent.select_action(state, action_mask=None)
                
                # Logic 수행 (간소화)
                trade_done = False
                realized_pnl = 0.0
                step_pnl = unrealized_pnl # Simplified step pnl logic for speed
                
                if action == 1 and current_position is None: # LONG
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = curr_idx
                elif action == 2 and current_position is None: # SHORT
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = curr_idx
                elif action == 3 and current_position is not None: # EXIT
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    current_position = None
                    entry_price = 0.0
                
                # 보상 계산
                reward = self.base_env.calculate_reward(
                    step_pnl, realized_pnl, trade_done, 
                    holding_time/MAX_STEPS, action, None, current_position
                )
                
                # Next State 준비
                self.data_collector.current_index += 1
                next_idx = self.data_collector.current_index
                
                next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
                next_hold = (next_idx - entry_index) if current_position else 0
                next_state = self.base_env.get_observation(
                    position_info=[next_pos_val, 0.0, next_hold/MAX_STEPS], 
                    current_index=next_idx
                )
                done = False if step < MAX_STEPS - 1 else True
                if next_state is None: done = True
                
                # 데이터 저장
                if next_state is not None:
                     agent.put_data((state, action, reward, next_state, prob, done, val))
                
                ep_reward += reward
                if done: break
            
            # 에피소드 종료 후 학습
            agent.train_net(episode=ep)
            total_rewards.append(ep_reward)
            
            # Optuna Pruning (가망 없으면 조기 종료)
            # 최근 5개 에피소드 평균 보상으로 판단
            avg_reward = np.mean(total_rewards[-5:])
            trial.report(avg_reward, ep)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 최종 평가 Metric: 마지막 10 에피소드의 평균 보상
        final_score = np.mean(total_rewards[-10:])
        return final_score

if __name__ == "__main__":
    # 데이터베이스 생성 (중단 후 이어하기 가능)
    study_name = "ppo_optimization"
    storage_name = "sqlite:///optuna_ppo.db"
    
    # Objective 초기화 (데이터 로딩)
    objective = OptimizationObjective()
    
    # Study 생성
    sampler = TPESampler(seed=42) # 베이지안 최적화 샘플러
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize", 
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True
    )
    
    print("🚀 하이퍼파라미터 최적화 시작 (예상 소요 시간: 수 시간)")
    print(f"   로그는 '{storage_name}'에 저장됩니다.")
    
    # 최적화 실행 (n_trials: 시도 횟수)
    try:
        study.optimize(objective, n_trials=50, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")

    # 결과 출력
    print("\n" + "="*50)
    print("🏆 Best Params Found:")
    print(json.dumps(study.best_params, indent=4))
    print(f"🏆 Best Reward: {study.best_value:.4f}")
    print("="*50)
    
    # 결과를 JSON 파일로 저장
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ 'best_params.json' 파일 저장 완료.")
    print("👉 model/config.py 파일을 열어 위 값들로 수정하세요.")