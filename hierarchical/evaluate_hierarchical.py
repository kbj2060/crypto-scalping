"""
계층적 RL 평가 스크립트
- MetaController + TacticalAgent 통합 평가
- Kelly Criterion 적용된 실전 시뮬레이션
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from common.preprocess import add_volatility_feature
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import TradingEnvironment

from hierarchical.meta_controller import MetaController
from hierarchical.tactical_agent import GoalConditionedTD3Agent
from hierarchical.kelly_criterion import KellyCriterion

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

TRANSACTION_COST = getattr(config, 'TRANSACTION_COST', 0.0005)
DECISION_INTERVAL = 5


class HierarchicalEvaluator:
    def __init__(self, mode='test', run_dir=None):
        self.mode = mode
        self.data_collector = DataCollector(use_saved_data=True)
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
        
        state_dim = self.env.get_state_dim()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.meta = MetaController(state_dim, info_dim=11, hidden_dim=256, 
                                   device=device, decision_interval=DECISION_INTERVAL)
        self.tactical = GoalConditionedTD3Agent(state_dim, 1, 12, device=device)
        self.kelly = KellyCriterion(fraction=0.25, max_leverage=config.LEVERAGE)
        
        self.close_prices = self.data_collector.eth_data['close'].values.astype(np.float32)
        vol_col = 'volatility_20tick'
        if vol_col in self.data_collector.eth_data.columns:
            self.vol_data = self.data_collector.eth_data[vol_col].values.astype(np.float32)
        else:
            self.vol_data = np.zeros(len(self.close_prices), dtype=np.float32)
        
        self._load_model(run_dir)
    
    def _load_data(self):
        path = 'data/training_features.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            if 'volatility_20tick' not in df.columns:
                df = add_volatility_feature(df)
            self.data_collector.eth_data = df
    
    def _load_model(self, run_dir):
        hier_dir = os.path.join('data', 'hierarchical')
        if run_dir:
            base = os.path.join(hier_dir, run_dir)
        else:
            if not os.path.isdir(hier_dir):
                logger.error("No hierarchical models found")
                return
            subdirs = sorted(os.listdir(hier_dir), reverse=True)
            base = None
            for d in subdirs:
                if os.path.exists(os.path.join(hier_dir, d, 'meta_best.pth')):
                    base = os.path.join(hier_dir, d)
                    break
            if base is None:
                logger.error("No valid model directory found")
                return
        
        self.meta.load(os.path.join(base, 'meta_best.pth'))
        self.tactical.load(os.path.join(base, 'tactical_best'))
    
    def _augment_info(self, info, idx):
        try: vol = float(self.vol_data[idx])
        except: vol = 0.0
        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2: vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)
    
    def evaluate(self):
        logger.info(f"[START] Hierarchical Evaluation ({self.mode})")
        logger.info(f"Range: {self.start_idx} ~ {self.end_idx} ({self.end_idx - self.start_idx} steps)")
        
        self.meta.reset()
        current_pos = 0.0
        balance = 10000.0
        balance_history = [balance]
        trade_count = 0
        meta_decisions = 0
        direction_log = []
        
        current_goal = None
        step_counter = 0
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating"):
            pos_info = [current_pos, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                continue
            
            # Meta Decision (every K steps)
            self.meta.step()
            if current_goal is None or self.meta.should_decide():
                meta_state = (state[0], state[1][..., :11] if isinstance(state[1], torch.Tensor) 
                             else np.asarray(state[1]).flatten()[:11])
                current_goal = self.meta.select_action(meta_state, deterministic=True)
                meta_decisions += 1
                direction_log.append(current_goal['direction'])
            
            # Tactical Execution
            tactical_state = (state[0], self._augment_info(state[1], idx))
            action_arr, _ = self.tactical.select_action(tactical_state, current_goal, noise=0.0)
            target_pos = float(action_arr[0])
            
            # Kelly Position Sizing
            risk_budget = current_goal.get('risk_budget', 0.3)
            effective_lev = self.kelly.get_position_size(abs(target_pos), risk_budget)
            
            if effective_lev < 1.0 or abs(target_pos) < 0.15:
                target_pos = 0.0
            
            trade_amt = target_pos - current_pos
            if abs(trade_amt) > 1e-4:
                trade_count += 1
            cost = effective_lev * TRANSACTION_COST if abs(trade_amt) > 1e-4 else 0.0
            current_pos = target_pos
            
            # PnL
            curr_price = float(self.close_prices[idx])
            next_price = float(self.close_prices[idx + 1])
            price_return = (next_price - curr_price) / curr_price
            pos_dir = np.sign(current_pos) if abs(current_pos) > 0.01 else 0.0
            pnl = (price_return * pos_dir * effective_lev) - cost if abs(current_pos) > 0.01 else 0.0
            
            self.kelly.record_trade(pnl)
            balance *= (1 + pnl)
            balance_history.append(balance)
            step_counter += 1
        
        # Results
        final_return = (balance - 10000) / 10000 * 100
        returns = np.diff(balance_history) / (np.array(balance_history[:-1]) + 1e-10)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 20)
        
        max_balance = np.maximum.accumulate(balance_history)
        drawdowns = (np.array(balance_history) - max_balance) / (max_balance + 1e-10)
        max_dd = np.min(drawdowns) * 100
        
        # Direction distribution
        dir_counts = {0: 0, 1: 0, 2: 0}
        for d in direction_log:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        
        logger.info("=" * 60)
        logger.info(f"💰 최종 잔고: ${balance:.2f}")
        logger.info(f"📈 수익률: {final_return:.2f}%")
        logger.info(f"📊 Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"📉 Max Drawdown: {max_dd:.2f}%")
        logger.info(f"🔄 총 거래: {trade_count}")
        logger.info(f"🧠 Meta 결정: {meta_decisions} (Flat:{dir_counts[0]}, Long:{dir_counts[1]}, Short:{dir_counts[2]})")
        logger.info(f"📐 Kelly: {self.kelly}")
        logger.info("=" * 60)
        
        return {
            'balance': balance,
            'return_pct': final_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trade_count,
        }


if __name__ == "__main__":
    evaluator = HierarchicalEvaluator(mode='test')
    evaluator.evaluate()
