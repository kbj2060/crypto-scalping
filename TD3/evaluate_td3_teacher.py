"""
TD3 Teacher-Guided Evaluator (Single Run Mode)
- Config의 설정을 그대로 가져와 1회 정밀 평가 수행
- [Logic] 학습 코드와 100% 동일한 로직 (Stateful, No Precompute)
- [Target] Deadzone=0.6, MinTrade=0.6 (Config.py 기준)
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

from common.preprocess import add_volatility_feature
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import TradingEnvironment

try:
    from TD3.td3_agent import TD3Agent
except ImportError:
    from td3_agent import TD3Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# Config에서 설정값 가져오기
TRANSACTION_COST = getattr(config, 'TRANSACTION_COST', 0.0005)
TARGET_DEADZONE = getattr(config, 'TD3_DEADZONE', 0.6)
TARGET_MIN_TRADE = getattr(config, 'TD3_MIN_TRADE_SIZE', 0.6)


class TD3TeacherEvaluator:
    def __init__(self, mode='test', model_type='best', run_dir=None):
        self.mode = mode
        self.model_type = model_type
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
        elif mode == 'val':
            self.start_idx = int(total_len * config.TRAIN_SPLIT)
            self.end_idx = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        else: # full
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()

        state_dim = self.env.get_state_dim()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = TD3Agent(state_dim, 1, 12, device=device)
        self._load_model(run_dir)

    def _load_data(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    if strategy_cols:
                        df[strategy_cols] = cached_df[strategy_cols]
                        logger.info(f"✅ 캐시된 전략 데이터 로드 완료: {len(strategy_cols)}개 전략")
                except Exception as e:
                    logger.warning(f"⚠️ 캐시 데이터 로드 중 오류 (재계산 진행): {e}")

            if 'volatility_20tick' not in df.columns:
                df = add_volatility_feature(df)
            self.data_collector.eth_data = df
            logger.info(f"✅ 데이터 로드 완료: {len(df):,}행")
        else:
            logger.error("Feature file not found: %s", path)
            sys.exit(1)

    def _load_model(self, run_dir):
        base_name = f"{self.model_type}_td3_teacher_model"
        if run_dir:
            model_path = os.path.join('data', 'td3_teacher', run_dir, base_name)
        else:
            td3_dir = os.path.join('data', 'td3_teacher')
            if not os.path.isdir(td3_dir):
                logger.error("TD3 Teacher 모델 디렉토리가 없습니다.")
                return
            subdirs = [d for d in os.listdir(td3_dir) if os.path.isdir(os.path.join(td3_dir, d))]
            if not subdirs:
                logger.error("TD3 Teacher 모델 없음")
                return
            
            model_path = None
            for run_name in sorted(subdirs, reverse=True):
                candidate = os.path.join(td3_dir, run_name, f"{base_name}_actor.pth")
                if os.path.isfile(candidate):
                    model_path = os.path.join(td3_dir, run_name, base_name)
                    break
            
            if model_path is None:
                base_name_last = "last_td3_teacher_model"
                for run_name in sorted(subdirs, reverse=True):
                    candidate = os.path.join(td3_dir, run_name, f"{base_name_last}_actor.pth")
                    if os.path.isfile(candidate):
                        model_path = os.path.join(td3_dir, run_name, base_name_last)
                        break
        
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

    def evaluate_single_run(self):
        """Config 설정값으로 1회 평가 실행"""
        deadzone = TARGET_DEADZONE
        min_strength = TARGET_MIN_TRADE
        
        logger.info("=" * 80)
        logger.info(f"🚀 Single Evaluation Started")
        logger.info(f"   - Deadzone: {deadzone}")
        logger.info(f"   - Min Trade Size: {min_strength}")
        logger.info("=" * 80)
        
        current_pos_size = 0.0
        prev_pnl = 0.0
        prev_trade_flag = 0.0
        
        balance = 10000.0  # 초기 자본금 $10,000
        initial_balance = balance
        max_balance = balance
        max_drawdown = 0.0
        
        trade_count = 0
        total_cost = 0.0
        trade_pnls = []
        current_trade_pnl = 0.0
        
        pos_counts = {'long': 0, 'short': 0, 'flat': 0}
        max_leverage = 1.0  # Phase 1: No Leverage
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # 1. Observation
            pos_info = [current_pos_size, prev_pnl, prev_trade_flag]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None: continue
            state = (state[0], self._augment_info(state[1], idx))
            
            # 2. Action
            action_arr, _, _ = self.agent.select_action(state, noise=0.0)
            action_val = float(action_arr[0])
            
            # 3. Position Logic
            target_pos_size = action_val if abs(action_val) > deadzone else 0.0
            trade_amount = target_pos_size - current_pos_size
            
            # Min Trade Filter
            if abs(trade_amount) < min_strength:
                target_pos_size = current_pos_size
                trade_amount = 0.0
            
            # 4. Trade Execution
            if abs(trade_amount) > 1e-4:
                trade_count += 1
                if current_pos_size != 0.0 and (target_pos_size == 0.0 or np.sign(target_pos_size) != np.sign(current_pos_size)):
                    trade_pnls.append(current_trade_pnl)
                    current_trade_pnl = 0.0
            
            trade_cost = abs(trade_amount) * max_leverage * TRANSACTION_COST
            total_cost += trade_cost
            current_pos_size = target_pos_size
            
            if current_pos_size > 0.01: pos_counts['long'] += 1
            elif current_pos_size < -0.01: pos_counts['short'] += 1
            else: pos_counts['flat'] += 1
            
            # 5. PnL Update
            next_price = float(self.data_collector.eth_data.iloc[idx+1]['close'])
            price_return = (next_price - curr_price) / curr_price
            step_pnl = (current_pos_size * max_leverage * price_return) - trade_cost
            
            current_trade_pnl += step_pnl
            balance *= (1 + step_pnl)
            
            if balance > max_balance: max_balance = balance
            drawdown = (max_balance - balance) / max_balance
            if drawdown > max_drawdown: max_drawdown = drawdown
            
            # State Update
            prev_pnl = step_pnl * 100.0
            prev_trade_flag = 1.0 if abs(trade_amount) < 0.1 else 0.0
            
            if balance < 100: # 파산 보호
                logger.warning("⚠️ 파산 감지 (Balance < $100)")
                break
            
        if current_pos_size != 0.0:
            trade_pnls.append(current_trade_pnl)
            
        # 결과 계산
        wins = len([p for p in trade_pnls if p > 0])
        losses = len([p for p in trade_pnls if p < 0])
        win_rate = wins / max(1, wins + losses) * 100
        total_pnl = sum(trade_pnls) if trade_pnls else 0.0
        profit_factor = abs(sum(p for p in trade_pnls if p > 0) / (sum(p for p in trade_pnls if p < 0) + 1e-10))
        final_return = (balance - initial_balance) / initial_balance * 100

        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 Evaluation Results (Config Settings)")
        logger.info("=" * 60)
        logger.info(f"  💰 Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"  💰 Final Balance:   ${balance:,.2f}")
        logger.info(f"  📈 Return:          {final_return:+.2f}%")
        logger.info(f"  📉 Max Drawdown:    {max_drawdown*100:.2f}%")
        logger.info(f"  💸 Total Fees:      ${total_cost*initial_balance:.2f}")
        logger.info("-" * 60)
        logger.info(f"  🔄 Trades:          {trade_count}")
        logger.info(f"  ✅ Win Rate:        {win_rate:.1f}% ({wins}/{wins+losses})")
        logger.info(f"  ⚖️ Profit Factor:   {profit_factor:.2f}")
        logger.info("-" * 60)
        logger.info(f"  🟢 Long:            {pos_counts['long'] / max(1, sum(pos_counts.values())) * 100:.1f}%")
        logger.info(f"  🔴 Short:           {pos_counts['short'] / max(1, sum(pos_counts.values())) * 100:.1f}%")
        logger.info(f"  ⚪ Flat:            {pos_counts['flat'] / max(1, sum(pos_counts.values())) * 100:.1f}%")
        logger.info("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model', type=str, default='best')
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()
    
    evaluator = TD3TeacherEvaluator(mode=args.mode, model_type=args.model, run_dir=args.run_dir)
    evaluator.evaluate_single_run()