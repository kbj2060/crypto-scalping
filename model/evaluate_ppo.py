"""
PPO 평가 스크립트 (Fixed: Strategy Calculation Added)
- 전략 점수가 0으로 나오는 문제 해결
- cached_strategies.csv 로드 및 필요 시 재계산 로직 추가
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

# config import
try:
    from . import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import config

from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy,
    CCIReversalStrategy, WilliamsRStrategy  # [수정] 12개 전략 모두 포함
)
from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class PPOEvaluator:
    def __init__(self, mode='test', model_type='best'):
        self.mode = mode
        self.data_collector = DataCollector(use_saved_data=True)
        # [수정] 전략 12개로 통일 (Info Dim 15를 맞추기 위함)
        # 전략 12개 (학습과 동일하게 설정)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        # 1. 데이터 로드 (전략 캐시 포함)
        self._load_data()
        
        # 2. 전략 점수 확인 및 재계산 (0점 방지)
        self._ensure_strategies_calculated()
        
        # 평가 범위 설정
        total_len = len(self.data_collector.eth_data)
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
        
        # Scaler 로드
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        scaler_path = f"{base_path}_{model_type}_scaler.pkl"
        
        if os.path.exists(scaler_path):
            self.env.preprocessor.load(scaler_path)
            self.env.scaler_fitted = True
            logger.info(f"[OK] Scaler Loaded: {scaler_path}")
        else:
            logger.error(f"[ERROR] Scaler not found: {scaler_path}")
            sys.exit(1)

        # 에이전트 설정
        state_dim = self.env.get_state_dim()
        action_dim = 4
        # [수정] info_dim = 15 설정
        real_info_dim = 15
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 모델 초기화
        self.agent = PPOAgent(state_dim, action_dim, info_dim=real_info_dim, device=device)

        # [수정 2] 모델 파일 체크 로직 개선 (계층형 모델 지원)
        model_base_path = f"{base_path}_{model_type}.pth"
        entry_model_path = f"{base_path}_{model_type}_entry.pth"
        
        if os.path.exists(entry_model_path):
            self.agent.load_model(model_base_path)
            logger.info(f"[OK] Hierarchical Model Loaded: {entry_model_path}")
        elif os.path.exists(model_base_path):
            try:
                self.agent.load_model(model_base_path)
                logger.info(f"[OK] Standard Model Loaded: {model_base_path}")
            except Exception:
                logger.error("[ERROR] Failed to load model structure.")
                sys.exit(1)
        else:
            logger.error(f"[ERROR] Model file not found. Checked: {entry_model_path}")
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
        """전략 컬럼이 없거나 전부 0이면 재계산"""
        df = self.data_collector.eth_data
        if df is None:
            return
        needs_calc = False
        for i in range(len(self.strategies)):
            col = f'strategy_{i}'
            if col not in df.columns:
                needs_calc = True
                break
            if df[col].iloc[config.LOOKBACK:].abs().sum() == 0:
                logger.warning(f"⚠️ {col} seems to be all zeros. Recalculating...")
                needs_calc = True
                break
        if needs_calc:
            logger.info("⚡ Calculating strategies for evaluation...")
            self._precalculate_strategies_sequential(df, config.LOOKBACK + 50, len(df))
        else:
            logger.info("✅ Strategies are ready.")

    def _precalculate_strategies_sequential(self, df, start_idx, total_len):
        for i in range(len(self.strategies)):
            if f'strategy_{i}' not in df.columns:
                df[f'strategy_{i}'] = 0.0
        for i in tqdm(range(start_idx, total_len), desc="Calc Strategies (Eval)"):
            self.data_collector.current_index = i
            for s_idx, strategy in enumerate(self.strategies):
                try:
                    res = strategy.analyze(self.data_collector)
                    score = 0.0
                    if res:
                        conf = float(res.get('confidence', 0.0))
                        sig = res.get('signal', 'NEUTRAL')
                        score = conf if sig == 'LONG' else (-conf if sig == 'SHORT' else 0.0)
                    col_name = f'strategy_{s_idx}'
                    df.loc[df.index[i], col_name] = score
                except Exception:
                    continue

    def evaluate(self):
        logger.info("[START] Backtest (Hierarchical Logic)...")
        
        current_position = None
        entry_price = 0.0
        entry_index = 0
        
        trades = []
        balance_history = [10000.0]
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
        
        self.agent.reset_episode_states()
        
        pbar = tqdm(range(self.start_idx, self.end_idx - 1), desc="Backtest")
        
        for idx in pbar:
            self.data_collector.current_index = idx
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / 1000]
            
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None: continue
            
            with torch.no_grad():
                action, _, _ = self.agent.select_action(state, action_mask=None)
            
            realized_pnl = 0.0
            trade_occurred = False
            trade_type = ""

            # Action 0: WAIT/HOLD
            if action == 0:
                pass
            
            # Action 1: LONG (Entry)
            elif action == 1:
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx

            # Action 2: SHORT (Entry)
            elif action == 2:
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx

            # Action 3: EXIT (Exit)
            elif action == 3:
                if current_position is not None:
                    if current_position == 'LONG':
                        realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    elif current_position == 'SHORT':
                        realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                    
                    trade_occurred = True
                    trade_type = "EXIT"
                    current_position = None

            # 거래 기록
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
        print(f" BACKTEST REPORT")
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