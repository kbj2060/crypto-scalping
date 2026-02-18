"""
TD3 Evaluator (Updated for Elite 8)
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

try:
    from core import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import config

from core.feature_engineering import ULTIMATE_FEATURE_COLS
from common.preprocess import add_volatility_feature
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import TradingEnvironment

try:
    from .td3_agent import TD3Agent
except ImportError:
    from TD3.td3_agent import TD3Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

TRANSACTION_COST = getattr(config, 'TRANSACTION_COST', 0.0005)

class TD3Evaluator:
    def __init__(self, mode='test', model_type='best', run_dir=None):
        self.mode = mode
        self.model_type = model_type
        self.data_collector = DataCollector(use_saved_data=True)
        # Elite 8 Strategies
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        self._load_data()
        
        total_len = len(self.data_collector.eth_data)
        if mode == 'test':
            self.start_idx = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
            self.end_idx = total_len
        else:
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()

        state_dim = self.env.get_state_dim() # 44
        action_dim = 1
        info_dim = 12 # 11 + 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.agent = TD3Agent(state_dim, action_dim, info_dim, device=device)
        self._load_model(run_dir)

    def _load_data(self):
        path = 'data/training_features.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            if 'volatility_20tick' not in df.columns:
                df = add_volatility_feature(df)
            self.data_collector.eth_data = df
            logger.info(f"✅ 데이터 로드 완료: {len(df):,}행")
        else:
            logger.error("Feature file not found.")
            sys.exit(1)

    def _load_model(self, run_dir):
        """모델 로드 (가장 최근 run_dir에서)"""
        base_name = f"{self.model_type}_td3_model"
        
        if run_dir:
            # 특정 run_dir이 지정된 경우
            model_path = os.path.join('data', 'td3', run_dir, base_name)
        else:
            # 가장 최근 모델 탐색
            td3_dir = os.path.join('data', 'td3')
            if not os.path.isdir(td3_dir):
                logger.error("TD3 모델 디렉토리가 없습니다: %s", td3_dir)
                return
            
            subdirs = [d for d in os.listdir(td3_dir) if os.path.isdir(os.path.join(td3_dir, d))]
            if not subdirs:
                logger.error("TD3 모델 없음")
                return
            
            # 최신 순 정렬
            for run_name in sorted(subdirs, reverse=True):
                candidate = os.path.join(td3_dir, run_name, f"{base_name}_actor.pth")
                if os.path.isfile(candidate):
                    model_path = os.path.join(td3_dir, run_name, base_name)
                    break
            else:
                logger.error("모델 파일을 찾을 수 없습니다.")
                return
        
        try:
            self.agent.load(model_path)
            logger.info("✅ 모델 로드 완료: %s", model_path)
        except Exception as e:
            logger.error("❌ 모델 로드 실패: %s", e)

    def _augment_info(self, info, idx):
        try:
            vol = float(self.data_collector.eth_data.iloc[idx].get('volatility_20tick', 0.0))
        except: vol = 0.0
        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2: vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)

    def evaluate(self):
        logger.info("[START] TD3 Evaluation...")
        logger.info(f"평가 기간: {self.start_idx} ~ {self.end_idx} ({self.end_idx - self.start_idx}개 스텝)")
        
        current_pos_size = 0.0
        balance = 10000.0
        balance_history = [balance]
        trade_count = 0
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            pos_info = [current_pos_size, 0.0, 0.0]
            
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None: continue
            state = (state[0], self._augment_info(state[1], idx))

            action_arr, _, _ = self.agent.select_action(state, noise=0.0)
            target_pos = float(action_arr[0])
            if abs(target_pos) < 0.3: target_pos = 0.0 # Deadzone

            # Execution
            trade_amt = target_pos - current_pos_size
            if abs(trade_amt) > 1e-4:
                trade_count += 1
            cost = abs(trade_amt) * TRANSACTION_COST
            current_pos_size = target_pos
            
            next_price = float(self.data_collector.eth_data.iloc[idx+1]['close'])
            pnl = current_pos_size * (next_price - curr_price) / curr_price - cost
            balance *= (1 + pnl)
            balance_history.append(balance)

        # 결과 출력
        final_return = (balance - 10000) / 10000 * 100
        logger.info("=" * 60)
        logger.info(f"최종 잔고: ${balance:.2f}")
        logger.info(f"수익률: {final_return:.2f}%")
        logger.info(f"총 거래 횟수: {trade_count}")
        
        # Sharpe Ratio 계산
        returns = np.diff(balance_history) / (np.array(balance_history[:-1]) + 1e-10)
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 20)  # 연환산
            logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        
        logger.info("=" * 60)

if __name__ == "__main__":
    evaluator = TD3Evaluator(mode='test', model_type='best')
    evaluator.evaluate()
