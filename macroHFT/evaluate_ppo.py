"""
MacroHFT Evaluator - Dream Team Ensemble Loading (최신 로직 반영)
================================================================
각 전문가의 Best Checkpoint에서 해당 전문가의 가중치만 추출하여 로드합니다.
- Router: best_router.pth 에서 로드
- Trend Expert: best_trend.pth 에서 로드
- Volatility Expert: best_volatility.pth 에서 로드
- Sideways Expert: best_sideways.pth 에서 로드

[변경사항]
- select_action 6개 반환값 처리
- pos_info: unrealized_pnl * 10 제거
- action_mask: 마지막 스텝 규칙 제거
- _strip_prefix 추가 (컴파일 모델 호환)
- holding_time 정규화 추가
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import glob

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
from common.trading_env import TradingEnvironment, INFO_DIM_ELITE8
from macroHFT.ppo_agent import PPOAgent

class PPOEvaluator:
    def __init__(self, model_dir=None):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        
        self._load_data()
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.scaler_fitted = True 

        # 평가 구간 (테스트셋: TRAIN_SPLIT + VAL_SPLIT 이후)
        total_len = len(self.data_collector.eth_data)
        self.start_idx = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        self.end_idx = total_len
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = PPOAgent(self.env.get_state_dim(), 3, INFO_DIM_ELITE8, device=self.device)
        
        # [핵심] 앙상블 로드 실행 (컴파일 호환)
        self._load_ensemble_model(model_dir)

    def _load_data(self):
        path = 'data/training_features.csv'
        cached_strat = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            if os.path.exists(cached_strat):
                try:
                    c_df = pd.read_csv(cached_strat, index_col=0, parse_dates=True)
                    for c in c_df.columns: 
                        if c.startswith('strategy_'): df[c] = c_df[c]
                except: pass
            self.data_collector.eth_data = df
            print(f"✅ Data Loaded: {len(df)} rows")
        else:
            raise FileNotFoundError("Data missing")

    def _strip_prefix(self, state_dict):
        """torch.compile 접두사(_orig_mod.) 제거 (ppo_agent와 동일)"""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def _load_ensemble_model(self, base_dir):
        """
        [Dream Team Loader]
        각 체크포인트 파일에서 '해당 역할'을 맡은 네트워크 가중치만 부분 로드
        """
        if base_dir is None:
            # 최신 폴더 자동 찾기
            root = 'data/macroHFT'
            if os.path.exists(root):
                subs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], reverse=True)
                if subs: base_dir = os.path.join(root, subs[0])
            else:
                base_dir = '.'
        
        print(f"📂 Loading Ensemble from: {base_dir}")
        
        # 1. Router Load (from best_router.pth)
        router_path = self._find_file(base_dir, 'best_router.pth')
        if router_path:
            ckpt = torch.load(router_path, map_location=self.device)
            if 'router' in ckpt:
                # _strip_prefix 적용
                self.agent.router.load_state_dict(self._strip_prefix(ckpt['router']))
                print(f"   ✅ Router loaded from {os.path.basename(router_path)}")
        else:
            print("   ⚠️ Router checkpoint not found! Using random weights.")

        # 2. Experts Load (각각의 Best 파일에서 추출)
        expert_files = {
            0: 'best_trend.pth',
            1: 'best_volatility.pth',
            2: 'best_sideways.pth'
        }
        expert_names = ['Trend', 'Volatility', 'Sideways']
        
        for idx, fname in expert_files.items():
            fpath = self._find_file(base_dir, fname)
            if fpath:
                ckpt = torch.load(fpath, map_location=self.device)
                if 'experts' in ckpt and len(ckpt['experts']) > idx:
                    # _strip_prefix 적용
                    self.agent.experts[idx].load_state_dict(
                        self._strip_prefix(ckpt['experts'][idx])
                    )
                    print(f"   ✅ {expert_names[idx]} Expert loaded from {os.path.basename(fpath)}")
            else:
                # 파일이 없으면 router 파일에 있는 expert라도 씀 (fallback)
                if router_path:
                    ckpt = torch.load(router_path, map_location=self.device)
                    if 'experts' in ckpt and len(ckpt['experts']) > idx:
                        self.agent.experts[idx].load_state_dict(
                            self._strip_prefix(ckpt['experts'][idx])
                        )
                        print(f"   ⚠️ {expert_names[idx]} Expert loaded from Router checkpoint (Fallback)")

    def _find_file(self, directory, suffix):
        """정확한 이름 매칭 or 접미사 매칭"""
        if os.path.exists(os.path.join(directory, suffix)):
            return os.path.join(directory, suffix)
        candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getctime)  # 가장 최신
        return None

    def get_action_mask(self, current_position):
        """액션 마스킹 (마지막 스텝 규칙 제거, train_ppo와 통일)"""
        mask = np.ones(3, dtype=np.float32)
        if current_position == 'LONG':
            mask[1] = 0.0  # Buy 금지
        elif current_position == 'SHORT':
            mask[2] = 0.0  # Sell 금지
        return mask

    def evaluate(self):
        self.agent.reset_episode_states()
        
        balance = config.EVAL_INITIAL_CAPITAL
        initial_balance = balance
        current_position = None
        entry_price = 0.0
        entry_index = self.start_idx
        trade_count = 0
        expert_counts = {0:0, 1:0, 2:0}
        
        print("🚀 Starting Ensemble Evaluation...")
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1)):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # --- 포지션 정보 구성 (train_ppo와 동일) ---
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            holding_time = (idx - entry_index) if current_position else 0
            # holding_time 정규화 (에피소드 최대 스텝은 config에서, 여기서는 120 가정)
            max_steps = getattr(config, 'TRAIN_MAX_STEPS_PER_EPISODE', 120)
            holding_time_norm = holding_time / max_steps
            
            pos_info = [pos_val, unrealized_pnl, holding_time_norm]  # ✨ unrealized_pnl * 10 제거
            
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                break
            
            mask = self.get_action_mask(current_position)
            
            # --- Action Selection (6개 반환값 수신) ---
            with torch.no_grad():
                action, _, _, selected_expert, _, _ = self.agent.select_action(
                    state, action_mask=mask, mode='router', deterministic=True
                )
            
            expert_counts[selected_expert] += 1
            
            # --- Trade Execution (슬리피지/수수료) ---
            fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
            slippage = 0.0003  # 평가 시에는 고정 슬리피지 사용 (선택)
            
            if action == 1:  # Buy
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price * (1 + slippage)
                    entry_index = idx
                    trade_count += 1
                elif current_position == 'SHORT':
                    realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                    balance *= (1 + realized_pnl)
                    current_position = None
            elif action == 2:  # Sell
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price * (1 - slippage)
                    entry_index = idx
                    trade_count += 1
                elif current_position == 'LONG':
                    realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    balance *= (1 + realized_pnl)
                    current_position = None
        
        # --- 에피소드 종료 시 미청산 포지션 강제 청산 ---
        if current_position is not None:
            last_price = float(self.data_collector.eth_data.iloc[self.end_idx - 1]['close'])
            fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)
            if current_position == 'LONG':
                realized_pnl = (last_price - entry_price) / entry_price - fee_rate
            else:
                realized_pnl = (entry_price - last_price) / entry_price - fee_rate
            balance *= (1 + realized_pnl)
            trade_count += 1
        
        final_return = (balance / initial_balance - 1) * 100
        print("\n" + "="*50)
        print(f"📊 Dream Team Ensemble Evaluation Result")
        print(f"   Test Period: {self.start_idx} ~ {self.end_idx}")
        print(f"   Return: {final_return:.2f}%")
        print(f"   Final Balance: ${balance:.2f}")
        print(f"   Trades: {trade_count}")
        print("-" * 50)
        print("🧠 Expert Usage:")
        total_steps = sum(expert_counts.values())
        if total_steps > 0:
            print(f"   Trend:      {expert_counts[0]/total_steps*100:5.1f}% ({expert_counts[0]} steps)")
            print(f"   Volatility: {expert_counts[1]/total_steps*100:5.1f}% ({expert_counts[1]} steps)")
            print(f"   Sideways:   {expert_counts[2]/total_steps*100:5.1f}% ({expert_counts[2]} steps)")
        print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, 
                        help='Directory containing best_*.pth files')
    args = parser.parse_args()
    
    evaluator = PPOEvaluator(model_dir=args.dir)
    evaluator.evaluate()