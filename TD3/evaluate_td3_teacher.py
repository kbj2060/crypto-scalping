"""
TD3 Teacher-Guided Evaluator (v2 - Train Logic 100% Aligned)
- train_td3_teacher.py와 PnL 공식 완전 일치
- Leverage, 수수료, 포지션 변경 필터 모두 동일
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
            if info.dim() == 2:
                vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)

    def evaluate(self):
        logger.info("=" * 70)
        logger.info("[START] TD3 Teacher-Guided Evaluation (v2 - Train-Aligned)")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Period: idx {self.start_idx} ~ {self.end_idx} ({self.end_idx - self.start_idx:,} steps)")
        logger.info(f"  Max Leverage: {config.LEVERAGE}")
        logger.info(f"  Transaction Cost: {TRANSACTION_COST}")
        logger.info("=" * 70)
        
        # === State Variables ===
        current_pos_size = 0.0
        equity = 1.0  # 정규화된 자산 (1.0 = 100%)
        equity_history = [equity]
        trade_count = 0
        
        # 상세 통계
        position_counts = {'long': 0, 'short': 0, 'flat': 0}
        trade_pnls = []
        current_trade_roe = 0.0
        max_equity = equity
        max_drawdown = 0.0
        total_cost_accumulated = 0.0
        
        for idx in tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating"):
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # === Observation ===
            pos_info = [current_pos_size, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                continue
            state = (state[0], self._augment_info(state[1], idx))

            # === Action (Deterministic) ===
            action_arr, _, _ = self.agent.select_action(state, noise=0.0)
            action_val = float(action_arr[0])

            # ============================================================
            # [핵심] train_td3_teacher.py 라인 244~263 100% 복사
            # ============================================================
            
            # 1. Deadzone 적용
            target_pos_size = action_val if abs(action_val) > 0.3 else 0.0
            
            # 2. Leverage 계산 (train과 동일)
            max_leverage = getattr(config, 'LEVERAGE', 20)
            effective_leverage = abs(action_val) * max_leverage
            
            # 3. 최소 레버리지 필터
            if effective_leverage < 1.0:
                target_pos_size = 0.0
            
            # 4. 포지션 변경 3중 필터 (train과 동일)
            is_opening = (current_pos_size == 0.0) and (target_pos_size != 0.0)
            is_flipping = (current_pos_size * target_pos_size < 0)
            is_strength_change = abs(target_pos_size - current_pos_size) > 0.4
            
            if not (is_opening or is_flipping or is_strength_change):
                target_pos_size = current_pos_size
            
            # 5. Trade amount 계산
            trade_amount = target_pos_size - current_pos_size
            if abs(trade_amount) > 1e-4:
                trade_count += 1
                
                # 포지션 청산 시 거래 PnL 기록
                if current_pos_size != 0.0 and (
                    target_pos_size == 0.0 or 
                    np.sign(target_pos_size) != np.sign(current_pos_size)
                ):
                    trade_pnls.append(current_trade_roe)
                    current_trade_roe = 0.0
            
            # 6. 수수료 계산 (train과 동일: leverage * TRANSACTION_COST)
            trade_cost = effective_leverage * TRANSACTION_COST if abs(trade_amount) > 1e-4 else 0.0
            total_cost_accumulated += trade_cost
            
            current_pos_size = target_pos_size
            
            # Position counting
            if current_pos_size > 0.1:
                position_counts['long'] += 1
            elif current_pos_size < -0.1:
                position_counts['short'] += 1
            else:
                position_counts['flat'] += 1

            # ============================================================
            # [핵심] PnL 계산 (train_td3_teacher.py 라인 284~287 100% 복사)
            # ============================================================
            next_price = float(self.data_collector.eth_data.iloc[idx + 1]['close'])
            price_return = (next_price - curr_price) / curr_price
            
            position_direction = np.sign(current_pos_size) if abs(current_pos_size) > 0.01 else 0.0
            raw_return = price_return if position_direction >= 0 else -price_return
            
            # [핵심] 레버리지 적용 ROE (학습과 동일!)
            step_pnl_roe = (raw_return * effective_leverage) - trade_cost if abs(current_pos_size) > 0.01 else 0.0
            
            current_trade_roe += step_pnl_roe
            
            # 자산 업데이트 (ROE 기반)
            equity *= (1 + step_pnl_roe)
            equity_history.append(equity)
            
            # Max Drawdown 추적
            if equity > max_equity:
                max_equity = equity
            drawdown = (max_equity - equity) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            # 청산 체크 (train과 동일)
            should_exit, exit_reason = self.env.check_exit_conditions(
                unrealized_pnl_roe=step_pnl_roe,
                holding_time_steps=0
            )
            if should_exit:
                current_pos_size = 0.0
                if current_trade_roe != 0.0:
                    trade_pnls.append(current_trade_roe)
                    current_trade_roe = 0.0

        # 마지막 열린 포지션 정리
        if current_pos_size != 0.0 and current_trade_roe != 0.0:
            trade_pnls.append(current_trade_roe)

        # === Results ===
        initial_balance = 10000.0
        final_balance = initial_balance * equity
        final_return = (equity - 1.0) * 100
        total_steps = position_counts['long'] + position_counts['short'] + position_counts['flat']
        
        # Win/Loss
        wins = len([p for p in trade_pnls if p > 0])
        losses = len([p for p in trade_pnls if p < 0])
        win_rate = wins / max(1, wins + losses) * 100
        avg_win = np.mean([p for p in trade_pnls if p > 0]) * 100 if wins > 0 else 0
        avg_loss = np.mean([p for p in trade_pnls if p < 0]) * 100 if losses > 0 else 0
        profit_factor = abs(sum(p for p in trade_pnls if p > 0) / (sum(p for p in trade_pnls if p < 0) + 1e-10))
        
        # Sharpe / Sortino
        returns = np.diff(equity_history) / (np.array(equity_history[:-1]) + 1e-10)
        sharpe = sortino = 0.0
        if len(returns) > 1:
            annualize = np.sqrt(175200)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * annualize
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-8
            sortino = np.mean(returns) / (downside_std + 1e-8) * annualize

        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 TD3 Teacher Evaluation Results (Train-Aligned v2)")
        logger.info("=" * 70)
        logger.info(f"  💰 Initial Balance:  ${initial_balance:,.2f}")
        logger.info(f"  💰 Final Balance:    ${final_balance:,.2f}")
        logger.info(f"  📈 Return (ROE):     {final_return:+.2f}%")
        logger.info(f"  📉 Max Drawdown:     {max_drawdown*100:.2f}%")
        logger.info(f"  💸 Total Cost (ROE): {total_cost_accumulated*100:.2f}%")
        logger.info("-" * 70)
        logger.info(f"  🔄 Total Trades:     {trade_count}")
        logger.info(f"  ✅ Wins:             {wins} ({win_rate:.1f}%)")
        logger.info(f"  ❌ Losses:           {losses}")
        logger.info(f"  📊 Avg Win ROE:      {avg_win:+.4f}%")
        logger.info(f"  📊 Avg Loss ROE:     {avg_loss:+.4f}%")
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
            'equity': equity,
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
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'val', 'full'])
    parser.add_argument('--model', type=str, default='best', choices=['best', 'last'])
    parser.add_argument('--run-dir', type=str, default=None)
    args = parser.parse_args()
    
    evaluator = TD3TeacherEvaluator(mode=args.mode, model_type=args.model, run_dir=args.run_dir)
    evaluator.evaluate()