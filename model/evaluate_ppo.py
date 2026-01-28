"""
PPO 평가 스크립트 (Info Dim = 15 Fixed)
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

try:
    from . import config
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import config
    from model.trading_env import TradingEnvironment
    from model.ppo_agent import PPOAgent

from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class PPOEvaluator:
    def __init__(self, mode='test', model_type='best'):
        self.mode = mode
        self.data_collector = DataCollector(use_saved_data=True)
        # 전략 12개 (학습과 동일)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]

        self._load_data()

        total_len = len(self.data_collector.eth_data)
        train_end = int(total_len * config.TRAIN_SPLIT)
        val_end = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        if mode == 'val': self.start_idx, self.end_idx = train_end, val_end
        elif mode == 'test': self.start_idx, self.end_idx = val_end, total_len
        else: self.start_idx, self.end_idx = config.LOOKBACK + 100, total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)

        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        scaler_path = f"{base_path}_{model_type}_scaler.pkl"
        if os.path.exists(scaler_path):
            self.env.preprocessor.load(scaler_path)
            self.env.scaler_fitted = True
            logger.info(f"[OK] Scaler Loaded: {scaler_path}")
        else:
            logger.error("[ERROR] Scaler not found.")
            sys.exit(1)

        state_dim = self.env.get_state_dim()
        action_dim = 3  # 3-Action Target Position

        # [수정] info_dim = 15
        real_info_dim = 15

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = PPOAgent(state_dim, action_dim, info_dim=real_info_dim, device=device)

        model_path = f"{base_path}_{model_type}.pth"
        entry_path = f"{base_path}_{model_type}_entry.pth"

        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            logger.info(f"[OK] Model Loaded: {model_path}")
        elif os.path.exists(entry_path):
            self.agent.load_model(entry_path)
            logger.info(f"[OK] Entry model loaded: {entry_path}")
        else:
            logger.error(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

    def _load_data(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns: df[col] = cached_df[col]
                except: pass
            if df.isnull().values.any(): df = df.fillna(0)
            self.data_collector.eth_data = df
        else:
            logger.error("[ERROR] Feature file not found.")
            sys.exit(1)

    def evaluate(self):
        logger.info("[START] Backtest (3-Action Target Position Logic)...")
        current_position, entry_price, entry_index = None, 0.0, 0
        trades, balance_history = [], [10000.0]
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
        self.agent.reset_episode_states()

        pbar = tqdm(range(self.start_idx, self.end_idx - 1), desc="Backtest")
        for idx in pbar:
            self.data_collector.current_index = idx
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            unrealized_pnl = (curr_price - entry_price)/entry_price if current_position == 'LONG' else ((entry_price - curr_price)/entry_price if current_position == 'SHORT' else 0.0)

            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / 1000.0]

            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None: continue

            with torch.no_grad():
                obs_seq, obs_info = state
                if not isinstance(obs_seq, torch.Tensor): obs_seq = torch.FloatTensor(obs_seq).to(self.agent.device)
                else: obs_seq = obs_seq.to(self.agent.device)
                if not isinstance(obs_info, torch.Tensor): obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.agent.device)
                else: obs_info = obs_info.to(self.agent.device)

                probs, _, self.agent.current_states = self.agent.model(obs_seq, obs_info, self.agent.current_states)
                action = torch.argmax(probs).item()

            realized_pnl, trade_occurred, trade_type = 0.0, False, ""

            if action == 0:
                if current_position == 'LONG': realized_pnl, trade_occurred, trade_type, current_position = (curr_price - entry_price)/entry_price - fee_rate, True, "EXIT_L", None
                elif current_position == 'SHORT': realized_pnl, trade_occurred, trade_type, current_position = (entry_price - curr_price)/entry_price - fee_rate, True, "EXIT_S", None
            elif action == 1:
                if current_position is None: current_position, entry_price, entry_index = 'LONG', curr_price, idx
                elif current_position == 'SHORT': realized_pnl, trade_occurred, trade_type, current_position, entry_price, entry_index = (entry_price - curr_price)/entry_price - fee_rate, True, "SWITCH_L", 'LONG', curr_price, idx
            elif action == 2:
                if current_position is None: current_position, entry_price, entry_index = 'SHORT', curr_price, idx
                elif current_position == 'LONG': realized_pnl, trade_occurred, trade_type, current_position, entry_price, entry_index = (curr_price - entry_price)/entry_price - fee_rate, True, "SWITCH_S", 'SHORT', curr_price, idx

            if trade_occurred:
                balance_history.append(balance_history[-1] * (1 + realized_pnl))
                trades.append({'net_pnl': realized_pnl, 'type': trade_type})
            pbar.set_postfix({'Bal': f"${balance_history[-1]:.0f}"})

        self._print_report(trades, balance_history)

    def _print_report(self, trades, balance_history):
        if not trades:
            print("\n[INFO] No trades executed.")
            return

        df = pd.DataFrame(trades)
        final_balance = balance_history[-1]
        roi = (final_balance - 10000.0) / 10000.0 * 100

        num_trades = len(df)
        win_trades = df[df['net_pnl'] > 0]
        loss_trades = df[df['net_pnl'] <= 0]
        win_rate = (len(win_trades) / num_trades * 100) if num_trades > 0 else 0.0

        print("\n" + "="*60)
        print(f" BACKTEST REPORT (Action 3 Optimized, Info Dim=15)")
        print("="*60)
        print(f" Final Balance:   ${final_balance:,.2f}")
        print(f" Net ROI:         {roi:+.2f}%")
        print(f" Total Trades:    {num_trades}")
        print(f" Win Rate:        {win_rate:.2f}% ({len(win_trades)}W / {len(loss_trades)}L)")

        total_loss = abs(loss_trades['net_pnl'].sum())
        total_win = win_trades['net_pnl'].sum()

        if total_loss > 0:
            pf = total_win / total_loss
            print(f" Profit Factor:   {pf:.2f}")
        else:
            print(f" Profit Factor:   Inf (No Loss)")
        print("="*60)

        try:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(balance_history, label='Equity', color='blue')
            plt.title('Backtest Equity Curve')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.hist(df['net_pnl'] * 100, bins=50, color='skyblue', edgecolor='black')
            plt.title('PnL Distribution (%)')
            plt.xlabel('PnL %')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"backtest_result_{timestamp}.png"
            plt.savefig(save_path)
            print(f"[OK] Graph saved to: {save_path}")
            plt.close()
        except: pass

if __name__ == "__main__":
    evaluator = PPOEvaluator(mode='test', model_type='last')
    evaluator.evaluate()
