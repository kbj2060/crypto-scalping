"""
TD3 Threshold Sweep Evaluator (Fixed)
- Deadzone / MinTrade 조합을 스윕하여 최적 필터링 찾기
- [Fix] PnL 계산 오류 수정 (Effective Leverage -> Real Position)
- [Fix] Logic Sync: train_td3_teacher.py와 동일한 포지션 필터링 적용
- [Note] 빠른 속도를 위해 Stateless Approximation(항상 포지션 0 가정) 사용
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

TRANSACTION_COST = getattr(config, 'TRANSACTION_COST', 0.0005)


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
        else:
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()

        state_dim = self.env.get_state_dim()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # info_dim=12 확인 (PosInfo 3 + Strategy 8 + Vol 1)
        self.agent = TD3Agent(state_dim, 1, 12, device=device)
        self._load_model(run_dir)
        
        self.cached_actions = None
        self.cached_prices = None

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
                logger.error("TD3 Teacher 모델 디렉토리가 없습니다: %s", td3_dir)
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
            
            if model_path is None:
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
        except:
            vol = 0.0
        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2: vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)

    def precompute_actions(self):
        """모든 스텝의 action + price를 미리 계산 (Stateless Approximation)"""
        logger.info("⏳ Action 사전 계산 중...")
        logger.warning("⚠️ Note: 빠른 스윕을 위해 포지션 상태를 0.0(Neutral)으로 가정하고 액션을 생성합니다.")
        
        actions = []
        prices = []
        valid_indices = []
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Precomputing"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            next_price = float(self.data_collector.eth_data.iloc[idx + 1]['close'])
            
            # [Stateless] 포지션 0 가정
            pos_info = [0.0, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                continue
            state = (state[0], self._augment_info(state[1], idx))
            
            action_arr, _, _ = self.agent.select_action(state, noise=0.0)
            action_val = float(action_arr[0])
            
            actions.append(action_val)
            prices.append((curr_price, next_price))
            valid_indices.append(idx)
        
        self.cached_actions = np.array(actions)
        self.cached_prices = np.array(prices)
        self.valid_indices = valid_indices
        
        logger.info(f"✅ {len(actions):,}개 action 캐싱 완료")
        
        abs_actions = np.abs(self.cached_actions)
        logger.info(f"📊 Action 분포: mean={abs_actions.mean():.3f}, std={abs_actions.std():.3f}")

    def simulate(self, deadzone, min_strength_change):
        """특정 Deadzone/MinChange 설정으로 시뮬레이션"""
        current_pos_size = 0.0
        equity = 1.0
        trade_count = 0
        total_cost = 0.0
        trade_pnls = []
        current_trade_roe = 0.0
        max_equity = 1.0
        max_drawdown = 0.0
        pos_counts = {'long': 0, 'short': 0, 'flat': 0}
        
        # [Fix] Phase 1: No Leverage
        max_leverage = 1.0
        
        for i in range(len(self.cached_actions)):
            action_val = self.cached_actions[i]
            curr_price, next_price = self.cached_prices[i]
            
            # 1. Deadzone
            target_pos_size = action_val if abs(action_val) > deadzone else 0.0
            
            trade_amount = target_pos_size - current_pos_size

            # 2. Min Trade Size Filter (Sync with train_td3_teacher.py)
            # 학습 코드와 동일한 로직 적용 (단순 차이 비교)
            if abs(trade_amount) < min_strength_change:
                target_pos_size = current_pos_size
                trade_amount = 0.0
            
            # 3. Trade Check
            if abs(trade_amount) > 1e-4:
                trade_count += 1
                if current_pos_size != 0.0 and (
                    target_pos_size == 0.0 or np.sign(target_pos_size) != np.sign(current_pos_size)
                ):
                    trade_pnls.append(current_trade_roe)
                    current_trade_roe = 0.0
            
            # 수수료: 거래량 비례
            trade_cost = abs(trade_amount) * max_leverage * TRANSACTION_COST
            total_cost += trade_cost
            current_pos_size = target_pos_size
            
            # Position counting
            if current_pos_size > 0.01: pos_counts['long'] += 1
            elif current_pos_size < -0.01: pos_counts['short'] += 1
            else: pos_counts['flat'] += 1
            
            # 4. PnL (Fix: Use held position size)
            # 보유한 포지션만큼만 수익/손실 발생
            price_return = (next_price - curr_price) / curr_price
            step_pnl = (current_pos_size * max_leverage * price_return) - trade_cost
            
            current_trade_roe += step_pnl
            equity *= (1 + step_pnl)
            
            if equity > max_equity: max_equity = equity
            dd = (max_equity - equity) / max_equity
            if dd > max_drawdown: max_drawdown = dd
            
            if equity <= 0.001:
                break
        
        if current_pos_size != 0.0 and current_trade_roe != 0.0:
            trade_pnls.append(current_trade_roe)
        
        wins = len([p for p in trade_pnls if p > 0])
        losses = len([p for p in trade_pnls if p < 0])
        win_rate = wins / max(1, wins + losses) * 100
        pf = abs(sum(p for p in trade_pnls if p > 0) / (sum(p for p in trade_pnls if p < 0) + 1e-10))
        
        return {
            'deadzone': deadzone,
            'min_strength': min_strength_change,
            'return_pct': (equity - 1.0) * 100,
            'max_dd': max_drawdown * 100,
            'trades': trade_count,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': pf,
            'total_cost_pct': total_cost * 100,
            'flat_pct': pos_counts['flat'] / max(1, sum(pos_counts.values())) * 100,
        }

    def sweep(self):
        """다양한 설정을 스윕"""
        if self.cached_actions is None:
            self.precompute_actions()
        
        # 스윕 그리드 (필요에 따라 범위 조정 가능)
        deadzones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        min_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = []
        
        logger.info("")
        logger.info("=" * 100)
        logger.info("🔍 Threshold Sweep 시작")
        logger.info("=" * 100)
        
        for dz in deadzones:
            for ms in min_strengths:
                r = self.simulate(dz, ms)
                results.append(r)
        
        results.sort(key=lambda x: x['return_pct'], reverse=True)
        
        logger.info("")
        logger.info(f"{'DZ':>4} | {'MinStr':>6} | {'Return%':>9} | {'MaxDD%':>7} | {'Trades':>6} | {'WinR%':>6} | {'PF':>5} | {'Cost%':>8} | {'Flat%':>6}")
        logger.info("-" * 100)
        
        for r in results:
            ret_str = f"{r['return_pct']:+.1f}%"
            logger.info(
                f"{r['deadzone']:>4.1f} | {r['min_strength']:>6.1f} | {ret_str:>9} | "
                f"{r['max_dd']:>6.1f}% | {r['trades']:>6} | {r['win_rate']:>5.1f}% | "
                f"{r['profit_factor']:>5.2f} | {r['total_cost_pct']:>7.1f}% | {r['flat_pct']:>5.1f}%"
            )
        
        logger.info("")
        logger.info("🏆 Top 5 설정:")
        for i, r in enumerate(results[:5]):
            logger.info(
                f"  #{i+1}: DZ={r['deadzone']}, MinStr={r['min_strength']} → "
                f"Return {r['return_pct']:+.1f}% | Trades {r['trades']} | WR {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f}"
            )
        
        profitable = [r for r in results if r['return_pct'] > 0]
        logger.info(f"\n✅ 수익 설정: {len(profitable)}/{len(results)}개")
        
        if not profitable:
            logger.info("")
            logger.info("⚠️ 모든 설정에서 손실. 모델이 아직 충분히 학습되지 않았거나(Q1 낮음), 시장이 매우 어렵습니다.")
        
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TD3 Threshold Sweep Evaluator")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'val', 'full'])
    parser.add_argument('--model', type=str, default='best', choices=['best', 'last'])
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()
    
    evaluator = TD3TeacherEvaluator(mode=args.mode, model_type=args.model, run_dir=args.run_dir)
    evaluator.sweep()