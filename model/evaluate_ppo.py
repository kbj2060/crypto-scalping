"""
PPO 평가 스크립트 (4-Action + Action Masking + Switching Logic)
- 학습 환경(train_ppo.py)과 동일한 로직으로 평가 (스위칭 지원)
- Action Masking 적용으로 불가능한 행동 차단
- Argmax(탐욕적) 선택으로 성능 평가
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
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        self._load_data()
        
        # 데이터 구간 설정
        total_len = len(self.data_collector.eth_data)
        train_end = int(total_len * config.TRAIN_SPLIT)
        val_end = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        
        if mode == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
            logger.info(f"[INFO] Evaluation Mode: VALIDATION Set ({self.start_idx} ~ {self.end_idx})")
        elif mode == 'test':
            self.start_idx = val_end
            self.end_idx = total_len
            logger.info(f"[INFO] Evaluation Mode: TEST Set ({self.start_idx} ~ {self.end_idx})")
        else: 
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len
            logger.info(f"[INFO] Evaluation Mode: FULL DATA ({self.start_idx} ~ {self.end_idx})")

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        
        # Scaler 로드
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        scaler_path = f"{base_path}_{model_type}_scaler.pkl"
        
        if os.path.exists(scaler_path):
            self.env.preprocessor.load(scaler_path)
            self.env.scaler_fitted = True
            logger.info(f"[OK] Scaler Loaded: {scaler_path}")
        else:
            logger.error("[ERROR] Scaler file not found. Train first!")
            sys.exit(1)

        # 에이전트 설정
        state_dim = self.env.get_state_dim()
        action_dim = 4  # 4-Action: 0=HOLD, 1=LONG, 2=SHORT, 3=EXIT
        real_info_dim = len(self.strategies) + 3 
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.agent = PPOAgent(state_dim, action_dim, info_dim=real_info_dim, device=device)
        except TypeError:
            self.agent = PPOAgent(state_dim, action_dim, device=device)

        model_path = f"{base_path}_{model_type}.pth"
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            logger.info(f"[OK] Model Loaded: {model_path}")
        else:
            logger.error("[ERROR] Model file not found.")
            sys.exit(1)

        # 네트워크 패치 (필요 시)
        try:
            if isinstance(self.agent.model.info_net, torch.nn.Sequential):
                current_in_features = self.agent.model.info_net[0].in_features
            else:
                current_in_features = self.agent.model.info_net.in_features
            
            if current_in_features != real_info_dim:
                logger.warning(f"[WARN] Network expects {current_in_features}, data has {real_info_dim}. Patching...")
                new_layer = torch.nn.Linear(real_info_dim, 64).to(device)
                torch.nn.init.orthogonal_(new_layer.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(new_layer.bias, 0.0)
                
                if isinstance(self.agent.model.info_net, torch.nn.Sequential):
                    self.agent.model.info_net[0] = new_layer
                else:
                    self.agent.model.info_net = new_layer
                logger.info("[OK] Network patched successfully.")
        except Exception as e:
            logger.warning(f"[WARN] Patching failed: {e}")

    def _load_data(self):
        path = 'data/training_features.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.ffill().bfill() 
            self.data_collector.eth_data = df
        else:
            logger.error("[ERROR] Feature file not found.")
            sys.exit(1)

    def _precalculate_strategies_for_eval(self):
        df = self.data_collector.eth_data
        
        # [수정 1] 전략 컬럼 존재 여부 확인 및 초기화
        # 파일에 strategy 컬럼이 없으면 먼저 0.0으로 생성해줘야 합니다.
        strategies_initialized = True
        for i in range(len(self.strategies)):
            col_name = f'strategy_{i}'
            if col_name not in df.columns:
                df[col_name] = 0.0
                strategies_initialized = False
        
        # [수정 2] 이미 계산되어 있는지 값으로 확인 (컬럼이 원래 있었을 때만)
        if strategies_initialized:
            check_idx = min(self.start_idx + 10, len(df) - 1)
            # strategy_0 값이 0이 아니면 이미 계산된 것으로 간주하고 스킵
            if df.iloc[check_idx]['strategy_0'] != 0.0:
                logger.info("[INFO] Strategies seem to be pre-calculated.")
                return

        logger.info("[INFO] Calculating strategies...")
        
        # [수정 3] 전략 계산 루프
        for i in tqdm(range(self.start_idx, self.end_idx), desc="Strategies"):
            self.data_collector.current_index = i
            for s_idx, strategy in enumerate(self.strategies):
                try:
                    res = strategy.analyze(self.data_collector)
                    score = 0.0
                    if res:
                        conf = float(res.get('confidence', 0.0))
                        sig = res.get('signal', 'NEUTRAL')
                        score = conf if sig == 'LONG' else (-conf if sig == 'SHORT' else 0.0)
                    
                    # 위에서 컬럼을 생성했으므로 get_loc에서 에러가 나지 않음
                    df.iat[i, df.columns.get_loc(f'strategy_{s_idx}')] = score
                except: continue
        
        self.data_collector.eth_data = df

    def evaluate(self):
        logger.info("[START] Running Backtest (with Action Masking & Switching)...")
        
        current_position = None
        entry_price = 0.0
        entry_index = 0
        
        trades = []
        balance_history = [10000.0]
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.001)
        
        self._precalculate_strategies_for_eval()
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
            
            # ---------------------------------------------------------
            # [Action Masking] 학습 코드와 동일한 마스킹 적용
            # ---------------------------------------------------------
            # Action: 0=HOLD, 1=LONG, 2=SHORT, 3=EXIT
            action_mask = [1.0, 1.0, 1.0, 1.0]
            
            if current_position is None:
                # 무포지션: EXIT(3) 불가
                action_mask = [1.0, 1.0, 1.0, 0.0]
            elif current_position == 'LONG':
                # LONG 보유: LONG(1) 불가 (SHORT는 스위칭 허용), EXIT 가능
                action_mask = [1.0, 0.0, 1.0, 1.0]
            elif current_position == 'SHORT':
                # SHORT 보유: SHORT(2) 불가 (LONG은 스위칭 허용), EXIT 가능
                action_mask = [1.0, 1.0, 0.0, 1.0]

            # ---------------------------------------------------------
            # [Model Inference] Argmax with Masking
            # ---------------------------------------------------------
            with torch.no_grad():
                obs_seq, obs_info = state
                if not isinstance(obs_seq, torch.Tensor):
                    obs_seq = torch.FloatTensor(obs_seq).to(self.agent.device)
                else: obs_seq = obs_seq.to(self.agent.device)
                    
                if not isinstance(obs_info, torch.Tensor):
                    obs_info = torch.FloatTensor(obs_info).unsqueeze(0).to(self.agent.device)
                else: obs_info = obs_info.to(self.agent.device)
                
                probs, _, self.agent.current_states = self.agent.model(obs_seq, obs_info, self.agent.current_states)
                
                # 마스킹 적용
                mask_tensor = torch.FloatTensor(action_mask).to(self.agent.device)
                probs = probs * mask_tensor
                
                # 결정적 선택 (Argmax)
                action = torch.argmax(probs).item()
            
            trade_occurred = False
            realized_pnl = 0.0

            # ---------------------------------------------------------
            # [Action Logic] Switching 포함
            # ---------------------------------------------------------
            
            # Action 0: HOLD
            if action == 0:
                pass 
            
            # Action 1: LONG
            elif action == 1:
                if current_position is None:
                    # 신규 롱 진입
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'SHORT':
                    # [Switching] SHORT 청산 -> LONG 진입
                    # 1. 청산
                    realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                    balance_history.append(balance_history[-1] * (1 + realized_pnl))
                    trades.append({'net_pnl': realized_pnl, 'type': 'SWITCH_L'})
                    trade_occurred = True
                    
                    # 2. 진입
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
            
            # Action 2: SHORT
            elif action == 2:
                if current_position is None:
                    # 신규 숏 진입
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'LONG':
                    # [Switching] LONG 청산 -> SHORT 진입
                    # 1. 청산
                    realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    balance_history.append(balance_history[-1] * (1 + realized_pnl))
                    trades.append({'net_pnl': realized_pnl, 'type': 'SWITCH_S'})
                    trade_occurred = True
                    
                    # 2. 진입
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx

            # Action 3: EXIT
            elif action == 3:
                if current_position is not None:
                    if current_position == 'LONG':
                        realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    elif current_position == 'SHORT':
                        realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                        
                    balance_history.append(balance_history[-1] * (1 + realized_pnl))
                    trades.append({'net_pnl': realized_pnl, 'type': 'EXIT'})
                    trade_occurred = True
                    current_position = None
            
            pbar.set_postfix({'Bal': f"${balance_history[-1]:.0f}"})

        try:
            self._print_report(trades, balance_history)
        except Exception as e:
            logger.error(f"[ERROR] Report generation failed: {e}")

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
            
        except Exception as plot_err:
            logger.warning(f"[WARN] Graph plotting failed: {plot_err}")

if __name__ == "__main__":
    evaluator = PPOEvaluator(mode='test', model_type='best')
    evaluator.evaluate()