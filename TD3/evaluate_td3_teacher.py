"""
TD3 Teacher-Guided Evaluator
- train_td3_teacher.py와 동일한 로직으로 테스트셋 PnL 검증
- No Leverage, Simplified PnL, Deadzone/MinTrade 일치
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
DEADZONE = getattr(config, 'TD3_DEADZONE', 0.3)
MIN_TRADE_SIZE = getattr(config, 'TD3_MIN_TRADE_SIZE', 0.3)


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
        else:  # full
            self.start_idx = config.LOOKBACK + 100
            self.end_idx = total_len

        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()

        state_dim = self.env.get_state_dim()
        action_dim = 1
        info_dim = 12
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.agent = TD3Agent(state_dim, action_dim, info_dim, device=device)
        self._load_model(run_dir)

    def _load_data(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'  # [추가] 캐시 파일 경로 지정
        
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
        """모델 로드 - td3_teacher 폴더에서 탐색"""
        base_name = f"{self.model_type}_td3_teacher_model"
        
        if run_dir:
            model_path = os.path.join('data', 'td3_teacher', run_dir, base_name)
        else:
            # td3_teacher 폴더에서 최신 모델 탐색
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
                # fallback: last 모델도 시도
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
        """Info Tensor(11) + Volatility(1) = 12 — train_td3_teacher와 동일"""
        try:
            vol = float(self.data_collector.eth_data.iloc[idx].get('volatility_20tick', 0.0))
        except:
            vol = 0.0
        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2:
                vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)

    def evaluate(self):
        logger.info("=" * 70)
        logger.info("[START] TD3 Teacher-Guided Evaluation")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Period: idx {self.start_idx} ~ {self.end_idx} ({self.end_idx - self.start_idx:,} steps)")
        logger.info(f"  Deadzone: {DEADZONE} | Min Trade Size: {MIN_TRADE_SIZE}")
        logger.info(f"  Transaction Cost: {TRANSACTION_COST}")
        logger.info("=" * 70)
        
        # === State Variables (train_td3_teacher.py와 동일) ===
        current_pos_size = 0.0
        balance = 10000.0
        initial_balance = balance
        balance_history = [balance]
        trade_count = 0
        cumulative_pnl = 0.0
        
        # 상세 통계
        wins = 0
        losses = 0
        total_cost = 0.0
        position_counts = {'long': 0, 'short': 0, 'flat': 0}
        trade_pnls = []  # 개별 거래 PnL 추적
        current_trade_pnl = 0.0  # 현재 진행 중인 거래의 누적 PnL
        max_balance = balance
        max_drawdown = 0.0
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # === Observation (train_td3_teacher와 동일) ===
            pos_info = [current_pos_size, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                continue
            state = (state[0], self._augment_info(state[1], idx))

            # === Action (Deterministic, No Noise) ===
            action_arr, _, _ = self.agent.select_action(state, noise=0.0)
            action_val = float(action_arr[0])

            # === Position Logic (train_td3_teacher와 완전 동일) ===
            target_pos_size = action_val if abs(action_val) > DEADZONE else 0.0
            
            trade_amount = target_pos_size - current_pos_size
            
            # 최소 변경폭 필터
            if abs(trade_amount) < MIN_TRADE_SIZE:
                target_pos_size = current_pos_size
                trade_amount = 0.0
            
            if abs(trade_amount) > 1e-4:
                trade_count += 1
                # 이전 포지션 청산 시 거래 PnL 기록
                if current_pos_size != 0.0 and (target_pos_size == 0.0 or np.sign(target_pos_size) != np.sign(current_pos_size)):
                    trade_pnls.append(current_trade_pnl)
                    if current_trade_pnl > 0:
                        wins += 1
                    elif current_trade_pnl < 0:
                        losses += 1
                    current_trade_pnl = 0.0
            
            trade_cost = abs(trade_amount) * TRANSACTION_COST
            total_cost += trade_cost
            current_pos_size = target_pos_size
            
            # Position counting
            if current_pos_size > 0.01:
                position_counts['long'] += 1
            elif current_pos_size < -0.01:
                position_counts['short'] += 1
            else:
                position_counts['flat'] += 1

            # === PnL Calculation (train_td3_teacher와 동일: No Leverage) ===
            next_price = float(self.data_collector.eth_data.iloc[idx + 1]['close'])
            price_return = (next_price - curr_price) / curr_price
            step_pnl = (current_pos_size * price_return) - trade_cost
            
            current_trade_pnl += step_pnl
            cumulative_pnl += step_pnl
            balance *= (1 + step_pnl)
            balance_history.append(balance)
            
            # Max Drawdown 추적
            if balance > max_balance:
                max_balance = balance
            drawdown = (max_balance - balance) / max_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 마지막 열린 포지션 정리
        if current_pos_size != 0.0:
            trade_pnls.append(current_trade_pnl)
            if current_trade_pnl > 0:
                wins += 1
            elif current_trade_pnl < 0:
                losses += 1

        # === Results ===
        final_return = (balance - initial_balance) / initial_balance * 100
        total_steps = position_counts['long'] + position_counts['short'] + position_counts['flat']
        
        # Sharpe Ratio (3분봉 기준 연환산)
        returns = np.diff(balance_history) / (np.array(balance_history[:-1]) + 1e-10)
        sharpe = 0.0
        sortino = 0.0
        if len(returns) > 1:
            # 3분봉 → 1년 = 365 * 24 * 20 = 175,200 bars
            annualize = np.sqrt(175200)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * annualize
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            sortino = np.mean(returns) / (downside_std + 1e-8) * annualize

        win_rate = wins / max(1, wins + losses) * 100
        avg_win = np.mean([p for p in trade_pnls if p > 0]) * 100 if any(p > 0 for p in trade_pnls) else 0
        avg_loss = np.mean([p for p in trade_pnls if p < 0]) * 100 if any(p < 0 for p in trade_pnls) else 0
        profit_factor = abs(sum(p for p in trade_pnls if p > 0) / (sum(p for p in trade_pnls if p < 0) + 1e-10))

        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 TD3 Teacher-Guided Evaluation Results")
        logger.info("=" * 70)
        logger.info(f"  💰 Initial Balance:  ${initial_balance:,.2f}")
        logger.info(f"  💰 Final Balance:    ${balance:,.2f}")
        logger.info(f"  📈 Return:           {final_return:+.2f}%")
        logger.info(f"  📉 Max Drawdown:     {max_drawdown*100:.2f}%")
        logger.info(f"  💸 Total Cost:       ${total_cost*initial_balance:.2f}")
        logger.info("-" * 70)
        logger.info(f"  🔄 Total Trades:     {trade_count}")
        logger.info(f"  ✅ Wins:             {wins} ({win_rate:.1f}%)")
        logger.info(f"  ❌ Losses:           {losses}")
        logger.info(f"  📊 Avg Win:          {avg_win:+.4f}%")
        logger.info(f"  📊 Avg Loss:         {avg_loss:+.4f}%")
        logger.info(f"  📊 Profit Factor:    {profit_factor:.2f}")
        logger.info("-" * 70)
        logger.info(f"  📏 Sharpe Ratio:     {sharpe:.4f}")
        logger.info(f"  📏 Sortino Ratio:    {sortino:.4f}")
        logger.info("-" * 70)
        long_pct = position_counts['long'] / max(1, total_steps) * 100
        short_pct = position_counts['short'] / max(1, total_steps) * 100
        flat_pct = position_counts['flat'] / max(1, total_steps) * 100
        logger.info(f"  🟢 Long:             {long_pct:.1f}%")
        logger.info(f"  🔴 Short:            {short_pct:.1f}%")
        logger.info(f"  ⚪ Flat:             {flat_pct:.1f}%")
        logger.info("=" * 70)
        
        # 날짜 범위 출력
        try:
            start_date = self.data_collector.eth_data.index[self.start_idx]
            end_date = self.data_collector.eth_data.index[self.end_idx - 1]
            logger.info(f"  📅 Period: {start_date} ~ {end_date}")
        except:
            pass
        logger.info("=" * 70)
        
        return {
            'balance': balance,
            'return_pct': final_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'trades': trade_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TD3 Teacher-Guided Evaluator")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'val', 'full'],
                        help='Evaluation mode: test(15%%), val(15%%), full(all)')
    parser.add_argument('--model', type=str, default='best', choices=['best', 'last'],
                        help='Model to load: best or last')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Specific run directory name (e.g., 20260210_091234)')
    args = parser.parse_args()
    
    evaluator = TD3TeacherEvaluator(mode=args.mode, model_type=args.model, run_dir=args.run_dir)
    evaluator.evaluate()