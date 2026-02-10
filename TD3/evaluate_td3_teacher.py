"""
TD3 Threshold Sweep Evaluator
- Deadzone / MinTrade 조합을 스윕하여 최적 필터링 찾기
- 학습 중단 없이 평가만으로 성능 개선 탐색
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
        self.agent = TD3Agent(state_dim, 1, 12, device=device)
        self._load_model(run_dir)
        
        # [핵심] 모든 스텝의 action을 미리 계산해서 캐싱
        # → 스윕 시 모델 추론 반복 없이 필터만 바꿔가며 시뮬레이션
        self.cached_actions = None
        self.cached_prices = None

    def _load_data(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv' # [추가] 캐시 파일 경로 지정
        
        if os.path.exists(path):
            # 1. 기본 피처 데이터 로드
            df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()
            
            # 2. [핵심] 캐시된 전략 데이터가 있으면 병합 (재계산 방지)
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    # 'strategy_'로 시작하는 컬럼만 골라서 병합
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    if strategy_cols:
                        # 인덱스 기준으로 병합
                        df[strategy_cols] = cached_df[strategy_cols]
                        logger.info(f"✅ 캐시된 전략 데이터 로드 완료: {len(strategy_cols)}개 전략")
                except Exception as e:
                    logger.warning(f"⚠️ 캐시 데이터 로드 중 오류 (재계산 진행): {e}")

            # 3. 변동성 피처 추가 (없을 경우)
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
        """모든 스텝의 action + price를 미리 계산 (1회만 실행)"""
        logger.info("⏳ Action 사전 계산 중...")
        
        actions = []
        prices = []
        valid_indices = []
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Precomputing"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            next_price = float(self.data_collector.eth_data.iloc[idx + 1]['close'])
            
            pos_info = [0.0, 0.0, 0.0]  # 포지션 없는 상태에서 action 관측
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
        
        # Action 분포 통계
        abs_actions = np.abs(self.cached_actions)
        logger.info(f"📊 Action 분포: mean={abs_actions.mean():.3f}, std={abs_actions.std():.3f}")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pct = (abs_actions > threshold).mean() * 100
            logger.info(f"   |action| > {threshold}: {pct:.1f}%")

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
        
        max_leverage = getattr(config, 'LEVERAGE', 20)
        
        for i in range(len(self.cached_actions)):
            action_val = self.cached_actions[i]
            curr_price, next_price = self.cached_prices[i]
            
            # 1. Deadzone
            target_pos_size = action_val if abs(action_val) > deadzone else 0.0
            
            # 2. Leverage
            effective_leverage = abs(action_val) * max_leverage
            if effective_leverage < 1.0:
                target_pos_size = 0.0
            
            # 3. 포지션 변경 필터 (strength_change 파라미터화)
            is_opening = (current_pos_size == 0.0) and (target_pos_size != 0.0)
            is_flipping = (current_pos_size * target_pos_size < 0)
            is_strength_change = abs(target_pos_size - current_pos_size) > min_strength_change
            
            if not (is_opening or is_flipping or is_strength_change):
                target_pos_size = current_pos_size
            
            # 4. Trade
            trade_amount = target_pos_size - current_pos_size
            if abs(trade_amount) > 1e-4:
                trade_count += 1
                if current_pos_size != 0.0 and (
                    target_pos_size == 0.0 or np.sign(target_pos_size) != np.sign(current_pos_size)
                ):
                    trade_pnls.append(current_trade_roe)
                    current_trade_roe = 0.0
            
            trade_cost = effective_leverage * TRANSACTION_COST if abs(trade_amount) > 1e-4 else 0.0
            total_cost += trade_cost
            current_pos_size = target_pos_size
            
            # Position counting
            if current_pos_size > 0.1: pos_counts['long'] += 1
            elif current_pos_size < -0.1: pos_counts['short'] += 1
            else: pos_counts['flat'] += 1
            
            # 5. PnL (train과 동일)
            price_return = (next_price - curr_price) / curr_price
            position_direction = np.sign(current_pos_size) if abs(current_pos_size) > 0.01 else 0.0
            raw_return = price_return if position_direction >= 0 else -price_return
            step_pnl = (raw_return * effective_leverage) - trade_cost if abs(current_pos_size) > 0.01 else 0.0
            
            current_trade_roe += step_pnl
            equity *= (1 + step_pnl)
            
            if equity > max_equity: max_equity = equity
            dd = (max_equity - equity) / max_equity
            if dd > max_drawdown: max_drawdown = dd
            
            # 파산 체크
            if equity <= 0.001:
                break
        
        # 마지막 포지션 정리
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
        
        # 스윕 그리드
        deadzones = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        min_strengths = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        results = []
        
        logger.info("")
        logger.info("=" * 100)
        logger.info("🔍 Threshold Sweep 시작")
        logger.info("=" * 100)
        
        for dz in deadzones:
            for ms in min_strengths:
                r = self.simulate(dz, ms)
                results.append(r)
        
        # 결과 정렬 (수익률 기준)
        results.sort(key=lambda x: x['return_pct'], reverse=True)
        
        # 테이블 출력
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
        
        # Top 5 강조
        logger.info("")
        logger.info("🏆 Top 5 설정:")
        for i, r in enumerate(results[:5]):
            logger.info(
                f"  #{i+1}: DZ={r['deadzone']}, MinStr={r['min_strength']} → "
                f"Return {r['return_pct']:+.1f}% | Trades {r['trades']} | WR {r['win_rate']:.1f}% | PF {r['profit_factor']:.2f}"
            )
        
        # 수익 나는 설정 수
        profitable = [r for r in results if r['return_pct'] > 0]
        logger.info(f"\n✅ 수익 설정: {len(profitable)}/{len(results)}개")
        
        if not profitable:
            logger.info("")
            logger.info("⚠️ 모든 설정에서 손실. 모델 자체의 방향 예측력이 부족할 수 있음.")
            logger.info("   → 학습을 더 진행하거나, 수수료 공식 수정 필요")
        
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