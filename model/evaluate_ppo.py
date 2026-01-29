"""
PPO 평가 스크립트 (Fixed: Skip Calculation if Cache Exists)
- cached_strategies.csv가 로드되면 재계산 없이 바로 사용
- Action 3 구조 및 Stochastic Policy 적용 유지
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

class PPOEvaluatorAction3:
    def __init__(self, mode='test', model_type='best'):
        self.mode = mode
        self.data_collector = DataCollector(use_saved_data=True)
        # 전략 12개
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]

        # 1. 데이터 로드 (수정된 로직 적용)
        self._load_data()

        # 2. 전략 점수 확인 (캐시 있으면 계산 스킵)
        self._ensure_strategies_calculated()

        total_len = len(self.data_collector.eth_data)
        
        # 평가 범위 설정
        if hasattr(self, 'is_test_data') and self.is_test_data:
            self.start_idx = config.LOOKBACK + 50
            self.end_idx = total_len - 1
            logger.info(f"[INFO] Using Full Test Data ({self.start_idx} ~ {self.end_idx})")
        else:
            train_end = int(total_len * config.TRAIN_SPLIT)
            val_end = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
            if mode == 'val':
                self.start_idx = train_end
                self.end_idx = val_end
                logger.info(f"[INFO] Mode: VALIDATION ({self.start_idx} ~ {self.end_idx})")
            elif mode == 'test':
                self.start_idx = val_end
                self.end_idx = total_len
                logger.info(f"[INFO] Mode: TEST ({self.start_idx} ~ {self.end_idx})")
            else: 
                self.start_idx = config.LOOKBACK + 100
                self.end_idx = total_len
                logger.info(f"[INFO] Mode: FULL DATA ({self.start_idx} ~ {self.end_idx})")

        self.env = TradingEnvironment(self.data_collector, self.strategies)

        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        scaler_path = f"{base_path}_{model_type}_scaler.pkl"
        if os.path.exists(scaler_path):
            self.env.preprocessor.load(scaler_path)
            self.env.scaler_fitted = True
            logger.info(f"[OK] Scaler Loaded: {scaler_path}")
        else:
            logger.error("[ERROR] Scaler not found. Evaluation might be wrong.")
            sys.exit(1)

        state_dim = self.env.get_state_dim()
        action_dim = 3  # Action 3
        info_dim = 15   # Info 15

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)

        model_path = f"{base_path}_{model_type}.pth"
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            logger.info(f"[OK] Model Loaded: {model_path}")
        else:
            logger.error(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

    def _load_data(self):
        path = 'data/training_features.csv'
        if not os.path.exists(path):
            logger.error("[ERROR] Feature file not found.")
            sys.exit(1)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.ffill().bfill()
        cache_path = 'data/cached_strategies.csv'
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached strategies from {cache_path}...")
                cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                for col in strategy_cols:
                    df[col] = cached_df[col].reindex(df.index).ffill().bfill().fillna(0)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        if df.isnull().values.any():
            df = df.fillna(0)
        self.data_collector.eth_data = df

    def _ensure_strategies_calculated(self):
        df = self.data_collector.eth_data
        if df is None: return
        
        # [수정] 전략 컬럼이 존재하면 무조건 계산 생략 (0점 체크 제거)
        all_strategies_exist = True
        for i in range(len(self.strategies)):
            if f'strategy_{i}' not in df.columns:
                all_strategies_exist = False
                break
        
        if all_strategies_exist:
            logger.info("✅ Strategies loaded from cache. Skipping calculation.")
            return
        
        # 전략 컬럼이 없는 경우에만 계산 수행
        logger.info("⚡ Calculating strategies for NEW data... (This may take a while)")
        self._precalculate_strategies_sequential(df, config.LOOKBACK+50, len(df))

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
        for i in range(len(self.strategies)): df[f'strategy_{i}'] = 0.0
        for i in tqdm(range(start_idx, total_len), desc="Calc Strategies"):
            self.data_collector.current_index = i
            for s_idx, strategy in enumerate(self.strategies):
                try:
                    res = strategy.analyze(self.data_collector)
                    score = 0.0
                    if res:
                        conf = float(res.get('confidence', 0.0))
                        sig = res.get('signal', 'NEUTRAL')
                        score = conf if sig == 'LONG' else (-conf if sig == 'SHORT' else 0.0)
                    df.iat[i, df.columns.get_loc(f'strategy_{s_idx}')] = score
                except: continue

    def evaluate(self):
        logger.info("[START] Backtest (Action 3)...")
        current_position, entry_price, entry_index = None, 0.0, 0
        trades, balance_history = [], [10000.0]
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
        self.agent.reset_episode_states()
        
        pbar = tqdm(range(self.start_idx, self.end_idx - 1), desc="Backtest")
        
        prob_sum = np.zeros(3)
        step_count = 0

        for idx in pbar:
            self.data_collector.current_index = idx
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            unrealized_pnl = 0.0
            if current_position == 'LONG': unrealized_pnl = (curr_price - entry_price)/entry_price
            elif current_position == 'SHORT': unrealized_pnl = (entry_price - curr_price)/entry_price
            
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
                
                prob_np = probs.cpu().numpy().flatten()
                prob_sum += prob_np
                step_count += 1
                
            action = torch.argmax(probs).item()

            realized_pnl, trade_occurred, trade_type = 0.0, False, ""
            
            # Action 3 Logic
            if action == 0: # Neutral
                if current_position == 'LONG': realized_pnl, trade_occurred, trade_type, current_position = (curr_price - entry_price)/entry_price - fee_rate, True, "EXIT_L", None
                elif current_position == 'SHORT': realized_pnl, trade_occurred, trade_type, current_position = (entry_price - curr_price)/entry_price - fee_rate, True, "EXIT_S", None
            elif action == 1: # Long
                if current_position is None: current_position, entry_price, entry_index = 'LONG', curr_price, idx
                elif current_position == 'SHORT': realized_pnl, trade_occurred, trade_type, current_position, entry_price, entry_index = (entry_price - curr_price)/entry_price - fee_rate, True, "SWITCH_L", 'LONG', curr_price, idx
            elif action == 2: # Short
                if current_position is None: current_position, entry_price, entry_index = 'SHORT', curr_price, idx
                elif current_position == 'LONG': realized_pnl, trade_occurred, trade_type, current_position, entry_price, entry_index = (curr_price - entry_price)/entry_price - fee_rate, True, "SWITCH_S", 'SHORT', curr_price, idx

            if trade_occurred:
                balance_history.append(balance_history[-1] * (1 + realized_pnl))
                trades.append({'net_pnl': realized_pnl, 'type': trade_type})
            
            pbar.set_postfix({'Bal': f"${balance_history[-1]:.0f}"})

        if step_count > 0:
            avg_probs = prob_sum / step_count
            print("\n" + "="*40)
            print(f"🧐 Model Confidence Diagnosis:")
            print(f" Neutral (0): {avg_probs[0]*100:.1f}%")
            print(f" Long    (1): {avg_probs[1]*100:.1f}%")
            print(f" Short   (2): {avg_probs[2]*100:.1f}%")
            print("="*40)

        self._print_report(trades, balance_history)

    def _print_report(self, trades, balance_history):
        if not trades: print("\n[INFO] No trades executed."); return
        df = pd.DataFrame(trades)
        final_balance = balance_history[-1]
        roi = (final_balance - 10000.0) / 10000.0 * 100
        
        num_trades = len(df)
        win_trades = df[df['net_pnl'] > 0]
        loss_trades = df[df['net_pnl'] <= 0]
        win_rate = (len(win_trades) / num_trades * 100) if num_trades > 0 else 0.0
        
        print("\n" + "="*60 + "\n BACKTEST REPORT\n" + "="*60)
        print(f" Final Balance:   ${final_balance:,.2f}")
        print(f" Net ROI:         {roi:+.2f}%")
        print(f" Total Trades:    {num_trades}")
        print(f" Win Rate:        {win_rate:.2f}% ({len(win_trades)}W / {len(loss_trades)}L)")
        print("="*60)
        
        try:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1); plt.plot(balance_history, label='Equity', color='blue'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.subplot(2, 1, 2); plt.hist(df['net_pnl'] * 100, bins=50, color='skyblue', edgecolor='black'); plt.grid(True, alpha=0.3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"backtest_result_{timestamp}.png"); plt.close()
        except: pass

if __name__ == "__main__":
    evaluator = PPOEvaluatorAction3(mode='test', model_type='best')
    evaluator.evaluate()