"""
MacroHFT v5.0 Evaluator – Discrete Leverage + Dream Team Ensemble
====================================================================
- 이산 레버리지(9개 행동) 완벽 대응
- train_ppo.py의 train_episode 로직과 동일한 거래 실행 (execute_trade 사용)
- Dream Team 로딩: 각 전문가의 best checkpoint에서 해당 네트워크만 로드
- Out-of-time 테스트셋 자동 평가
- 전문가 사용 비율, 수익률, 샤프 비율, MDD 출력
- 🔥 평가 시 eval() 모드 적용 (Dropout 비활성화)
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import glob

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

        # ---------- 데이터 로드 및 캐싱 ----------
        self._load_features()
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()
        self.env.scaler_fitted = True

        # ---------- 테스트셋 구간 (Out-of-time) ----------
        total_len = len(self.data_collector.eth_data)
        self.start_idx = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        self.end_idx = total_len

        # ---------- 에이전트 초기화 ----------
        state_dim = self.env.get_state_dim()
        action_dim = config.ACTION_DIM          # 9
        info_dim = INFO_DIM_ELITE8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=self.device)

        # ---------- Dream Team 앙상블 로드 ----------
        self._load_ensemble_model(model_dir)

        # ---------- Close prices 캐싱 (속도 향상) ----------
        self.close_prices = self.data_collector.eth_data['close'].values.astype(np.float32)
        self.volatility_data = self.data_collector.eth_data.get('volatility_z', np.zeros(total_len)).values.astype(np.float32)

    # ------------------------------------------------------------------
    # 데이터 로드 (cached_strategies.csv 우선)
    # ------------------------------------------------------------------
    def _load_features(self):
        path = 'data/training_features.csv'
        if not os.path.exists(path):
            raise FileNotFoundError("training_features.csv not found")

        df = pd.read_csv(path, index_col=0, parse_dates=True).ffill().bfill()

        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(cached_strategies_path):
            try:
                cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                for col in strategy_cols:
                    df[col] = cached_df[col]
            except Exception as e:
                print(f"⚠️ Failed to load cached strategies: {e}")

        self.data_collector.eth_data = df
        print(f"✅ Data loaded: {df.shape}")

    # ------------------------------------------------------------------
    # Dream Team Ensemble Loader (train_ppo의 load_dream_team과 동일 로직)
    # ------------------------------------------------------------------
    def _find_file(self, directory, suffix):
        exact = os.path.join(directory, suffix)
        if os.path.exists(exact):
            return exact
        candidates = glob.glob(os.path.join(directory, f"*{suffix}"))
        if candidates:
            return max(candidates, key=os.path.getctime)
        return None

    def _strip_prefix(self, state_dict):
        """torch.compile 접두사(_orig_mod.) 제거 (ppo_agent와 동일)"""
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def _load_ensemble_model(self, base_dir):
        """Dream Team: 각 전문가의 best checkpoint에서 개별 로드"""
        if base_dir is None:
            root = 'data/macroHFT'
            if os.path.exists(root):
                subs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], reverse=True)
                if subs:
                    base_dir = os.path.join(root, subs[0])
                    print(f"📂 Auto-detected latest directory: {base_dir}")
                else:
                    base_dir = '.'
            else:
                base_dir = '.'

        print(f"📂 Loading Dream Team Ensemble from: {base_dir}")

        # 1. Router 로드 (best_router.pth)
        router_path = self._find_file(base_dir, 'best_router.pth')
        if router_path:
            ckpt = torch.load(router_path, map_location=self.device)
            if 'router' in ckpt:
                self.agent.router.load_state_dict(self._strip_prefix(ckpt['router']))
                print(f"   ✅ Router loaded from {os.path.basename(router_path)}")
        else:
            print("   ⚠️ Router checkpoint not found! Using random weights.")

        # 2. Experts 로드 (각각 best 파일에서)
        expert_files = {0: 'best_trend.pth', 1: 'best_volatility.pth', 2: 'best_sideways.pth'}
        expert_names = ['Trend', 'Volatility', 'Sideways']

        for idx, fname in expert_files.items():
            fpath = self._find_file(base_dir, fname)
            if fpath:
                ckpt = torch.load(fpath, map_location=self.device)
                if 'experts' in ckpt and len(ckpt['experts']) > idx:
                    self.agent.experts[idx].load_state_dict(
                        self._strip_prefix(ckpt['experts'][idx]), strict=False
                    )
                    print(f"   ✅ {expert_names[idx]} Expert loaded from {os.path.basename(fpath)}")
            else:
                # Fallback: router checkpoint에 포함된 experts 사용
                if router_path:
                    ckpt = torch.load(router_path, map_location=self.device)
                    if 'experts' in ckpt and len(ckpt['experts']) > idx:
                        self.agent.experts[idx].load_state_dict(
                            self._strip_prefix(ckpt['experts'][idx]), strict=False
                        )
                        print(f"   ⚠️ {expert_names[idx]} Expert loaded from Router checkpoint (Fallback)")

    # ------------------------------------------------------------------
    # 액션 마스킹 (9차원, train_ppo와 완전 동일)
    # ------------------------------------------------------------------
    def get_action_mask(self, current_position):
        """
        액션 마스킹: 9개 행동 각각 허용/금지
        - 인덱스 0~2: HOLD + 레버리지 1,5,10
        - 인덱스 3~5: LONG + 레버리지 1,5,10
        - 인덱스 6~8: SHORT + 레버리지 1,5,10
        """
        mask = np.ones(config.ACTION_DIM, dtype=np.float32)
        if current_position == 'LONG':
            mask[3:6] = 0.0
        elif current_position == 'SHORT':
            mask[6:9] = 0.0
        return mask

    # ------------------------------------------------------------------
    # 평가 실행 (train_episode의 거래 로직과 동일, 리워드/학습 없음)
    # ------------------------------------------------------------------
    def evaluate(self):
        # 🔥 평가 모드 전환 (Dropout 비활성화) - agent.eval() 제거
        for expert in self.agent.experts:
            expert.eval()
        self.agent.router.eval()

        balance = config.EVAL_INITIAL_CAPITAL
        initial_balance = balance
        position = None
        entry_price = 0.0
        effective_leverage = 0.0
        holding_steps = 0
        entry_cost = 0.0
        entry_balance = 0.0

        trade_count = 0
        expert_counts = {0: 0, 1: 0, 2: 0}

        equity_curve = [balance]

        max_steps = self.end_idx - self.start_idx - 1
        pbar = tqdm(total=max_steps, desc="Evaluating Dream Team")

        idx = self.start_idx
        while idx < self.end_idx - 1:
            curr_price = self.close_prices[idx]

            # ---------- 미실현 손익 ----------
            if position == 'LONG':
                unrealized_return = (curr_price - entry_price) / entry_price
            elif position == 'SHORT':
                unrealized_return = (entry_price - curr_price) / entry_price
            else:
                unrealized_return = 0.0

            # ---------- 포지션 정보 텐서 ----------
            pos_val = 1.0 if position == 'LONG' else (-1.0 if position == 'SHORT' else 0.0)
            pos_info = [pos_val, unrealized_return, holding_steps / config.TRAIN_MAX_STEPS_PER_EPISODE]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                break

            # ---------- 액션 마스크 ----------
            action_mask = self.get_action_mask(position)

            # ---------- 행동 선택 (deterministic) ----------
            with torch.no_grad():
                action, _, _, selected_expert, _, _, _ = self.agent.select_action(
                    state, action_mask=action_mask, mode='router', deterministic=True
                )

            expert_counts[selected_expert] += 1
            direction, scale = action

            # ---------- 거래 실행 (execute_trade 사용) ----------
            trade_done = False
            realized_pnl_roe = 0.0

            if direction != 0 and scale >= config.MIN_LEVERAGE / config.MAX_LEVERAGE:
                # ----- 진입 -----
                if position is None:
                    entry_price, eff_lev, executed, cost, position_value, contracts = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        direction=1 if direction == 1 else -1,
                        balance=balance,
                        volatility=self.volatility_data[idx] if hasattr(self, 'volatility_data') else None,
                        is_exit=False
                    )
                    if executed:
                        entry_balance = balance
                        position = 'LONG' if direction == 1 else 'SHORT'
                        effective_leverage = eff_lev
                        entry_cost = cost
                        holding_steps = 0
                        trade_count += 1
                        balance *= (1 - entry_cost)
                        equity_curve.append(balance)

                # ----- 청산 (반대 방향) -----
                elif (direction == 1 and position == 'SHORT') or (direction == 2 and position == 'LONG'):
                    _, _, _, exit_cost, _, _ = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        is_exit=True,
                        leverage=effective_leverage
                    )
                    realized_return = unrealized_return
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                    balance = entry_balance * (1 + total_trade_return)
                    equity_curve.append(balance)
                    trade_done = True

                    # 포지션 초기화
                    position = None
                    effective_leverage = 0.0
                    entry_cost = 0.0
                    entry_balance = 0.0
                    holding_steps = 0

            # ---------- 보유 중 ----------
            if position is not None:
                holding_steps += 1

            # ---------- 강제 청산 (stop loss / take profit) ----------
            if position is not None:
                unrealized_pnl_roe = unrealized_return * effective_leverage
                should_exit, reason = self.env.check_exit_conditions(unrealized_pnl_roe, holding_steps)
                if should_exit:
                    _, _, _, exit_cost, _, _ = self.env.execute_trade(
                        action=scale,
                        current_price=curr_price,
                        is_exit=True,
                        leverage=effective_leverage
                    )
                    realized_return = unrealized_return
                    total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
                    balance = entry_balance * (1 + total_trade_return)
                    equity_curve.append(balance)
                    trade_done = True

                    position = None
                    effective_leverage = 0.0
                    entry_cost = 0.0
                    entry_balance = 0.0
                    holding_steps = 0
                    pbar.set_postfix({'Exit': reason[:4]})

            idx += 1
            pbar.update(1)

        # ---------- 에피소드 종료: 미청산 포지션 강제 청산 ----------
        if position is not None:
            final_price = self.close_prices[min(idx, len(self.close_prices)-1)]
            if position == 'LONG':
                realized_return = (final_price - entry_price) / entry_price
            else:
                realized_return = (entry_price - final_price) / entry_price

            _, _, _, exit_cost, _, _ = self.env.execute_trade(
                action=scale,
                current_price=final_price,
                is_exit=True,
                leverage=effective_leverage
            )
            total_trade_return = (1 - entry_cost) * (1 + realized_return * effective_leverage - exit_cost) - 1
            balance = entry_balance * (1 + total_trade_return)
            equity_curve.append(balance)
            trade_count += 1

        pbar.close()

        # ---------- 성과 지표 계산 ----------
        total_return = (balance / initial_balance) - 1.0
        returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-10)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365 * 24 * 60 / 3)
        mdd = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve).max()

        # ---------- 결과 출력 ----------
        print("\n" + "="*60)
        print("📊 Dream Team Ensemble Evaluation Result (Discrete Leverage v5.0)")
        print("="*60)
        print(f"   Test Period      : {self.start_idx} ~ {self.end_idx} ({self.end_idx - self.start_idx} steps)")
        print(f"   Initial Balance  : ${initial_balance:.2f}")
        print(f"   Final Balance    : ${balance:.2f}")
        print(f"   Total Return     : {total_return*100:.2f}%")
        print(f"   Sharpe Ratio (annual): {sharpe:.3f}")
        print(f"   Max Drawdown     : {mdd*100:.2f}%")
        print(f"   Total Trades     : {trade_count}")
        print("-" * 60)
        print("🧠 Expert Usage:")
        total_steps = sum(expert_counts.values())
        if total_steps > 0:
            print(f"   Trend Expert      : {expert_counts[0]/total_steps*100:5.1f}% ({expert_counts[0]} steps)")
            print(f"   Volatility Expert : {expert_counts[1]/total_steps*100:5.1f}% ({expert_counts[1]} steps)")
            print(f"   Sideways Expert   : {expert_counts[2]/total_steps*100:5.1f}% ({expert_counts[2]} steps)")
        print("="*60)

        return total_return, sharpe, mdd, trade_count, expert_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing best_*.pth files (default: latest)')
    args = parser.parse_args()

    evaluator = PPOEvaluator(model_dir=args.dir)
    evaluator.evaluate()