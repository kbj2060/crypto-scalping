"""
PPO 평가 스크립트
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from common import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import config

from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import INFO_DIM_ELITE8
from common.trading_env import TradingEnvironment
from macroHFT.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)

class PPOEvaluator:
    def __init__(self, mode='test', model_type='best'):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        self._load_data()
        self._ensure_strategies_calculated()
        
        total_len = len(self.data_collector.eth_data)
        if mode == 'test':
            self.start_idx = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
            self.end_idx = total_len
        else:
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = PPOAgent(self.env.get_state_dim(), 3, info_dim=INFO_DIM_ELITE8, device=device)
        
        # 모델 로드 (경로 생략 - 실제 경로에 맞춰주세요)
        try:
            self.agent.load_model('data/macroHFT/ppo_model_last.pth')
        except:
            pass

    def get_action_mask(self, current_position, market_volatility=0.0):
        mask = np.ones(3, dtype=np.float32)
        if current_position == 'LONG': mask[1] = 0.0
        elif current_position == 'SHORT': mask[2] = 0.0
        return mask

    def _load_data(self):
        path = 'data/training_features.csv'
        if os.path.exists(path):
            self.data_collector.eth_data = pd.read_csv(path, index_col=0, parse_dates=True).fillna(0)
    
    def _ensure_strategies_calculated(self):
        # 전략 컬럼 확인 및 계산 로직 (기존과 동일)
        pass

    def evaluate(self):
        current_position = None
        entry_price = 0.0
        balance_history = [10000.0]
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1)):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # Volatility Z
            vol_z = 0.0
            if 'volatility_z' in self.data_collector.eth_data.columns:
                vol_z = float(self.data_collector.eth_data.iloc[idx]['volatility_z'])
            
            # Observation
            state = self.env.get_observation(position_info=[0,0,0], current_index=idx)
            if state is None: continue
            
            mask = self.get_action_mask(current_position, vol_z)
            with torch.no_grad():
                action, _, _ = self.agent.select_action(state, action_mask=mask)
            
            # Trade Execution (Simplified)
            pnl = 0.0
            if action == 1: # Buy
                if current_position is None: current_position = 'LONG'; entry_price = curr_price
                elif current_position == 'SHORT': pnl = (entry_price - curr_price)/entry_price; current_position = None
            elif action == 2: # Sell
                if current_position is None: current_position = 'SHORT'; entry_price = curr_price
                elif current_position == 'LONG': pnl = (curr_price - entry_price)/entry_price; current_position = None
            
            balance_history.append(balance_history[-1] * (1 + pnl))
            
        print(f"Final Balance: {balance_history[-1]:.2f}")

if __name__ == "__main__":
    evaluator = PPOEvaluator()
    evaluator.evaluate()