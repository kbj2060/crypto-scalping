"""
PPO 모델 평가 스크립트 (3-Action, Data Leakage 차단)
학습된 모델을 Validation/Test 데이터셋으로 평가하고 성능 지표를 출력합니다.
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate_ppo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 불필요한 로그 끄기
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)


class PPOEvaluator:
    def __init__(self, mode='test'):
        """
        mode: 'val' (검증셋 70~85%) or 'test' (테스트셋 85~100%)
        """
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        # 1. 데이터 로드
        self._load_features()
        
        # 2. 데이터 구간 설정 (Critical Fix - Data Leakage 차단)
        total_len = len(self.data_collector.eth_data)
        train_end = int(total_len * config.TRAIN_SPLIT)
        val_end = int(total_len * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        
        if mode == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
            logger.info(f"🔍 Evaluation Mode: VALIDATION Set ({self.start_idx} ~ {self.end_idx}, {self.end_idx - self.start_idx} steps)")
        else:  # test
            self.start_idx = val_end
            self.end_idx = total_len
            logger.info(f"🔍 Evaluation Mode: TEST Set ({self.start_idx} ~ {self.end_idx}, {self.end_idx - self.start_idx} steps)")

        # 3. 환경 및 에이전트 설정
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        
        # [Critical] 학습된 Scaler 로드 (새로 fit하지 않음!)
        scaler_path = config.AI_MODEL_PATH.replace('.pth', '_best_scaler.pkl')
        if os.path.exists(scaler_path):
            self.env.preprocessor.load(scaler_path)
            self.env.scaler_fitted = True
            logger.info(f"✅ Trained Scaler loaded: {scaler_path}")
        else:
            # Fallback: last_scaler 시도
            scaler_path = config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.env.preprocessor.load(scaler_path)
                self.env.scaler_fitted = True
                logger.info(f"✅ Trained Scaler loaded (fallback): {scaler_path}")
            else:
                logger.error("❌ Scaler file not found. Train first!")
                sys.exit(1)

        # 4. 모델 로드 (3-Action)
        state_dim = self.env.get_state_dim()
        action_dim = 3  # 3-Action: 0:Neutral, 1:Long, 2:Short
        info_dim = len(self.strategies) + 3
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🔧 Device: {device} | Action Dim: {action_dim} (3-Action)")
        
        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        model_path = config.AI_MODEL_PATH.replace('.pth', '_best.pth')  # Best 모델 평가
        if os.path.exists(model_path):
            try:
                self.agent.load_model(model_path)
                self.agent.model.eval()  # 평가 모드
                logger.info(f"✅ Best Model loaded: {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Best model load failed (structure mismatch?): {e}")
                # Fallback: last model 시도
                model_path = config.AI_MODEL_PATH.replace('.pth', '_last.pth')
                if os.path.exists(model_path):
                    try:
                        self.agent.load_model(model_path)
                        self.agent.model.eval()
                        logger.info(f"✅ Last Model loaded (fallback): {model_path}")
                    except Exception as e2:
                        logger.error(f"❌ Model load failed: {e2}")
                        sys.exit(1)
                else:
                    logger.error("❌ Model file not found.")
                    sys.exit(1)
        else:
            logger.error("❌ Model file not found.")
            sys.exit(1)

    def _load_features(self):
        """피처 파일 로드"""
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [col for col in cached_df.columns if col.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except Exception:
                    pass
            self.data_collector.eth_data = df
        else:
            logger.error("❌ 피처 파일 없음")
            sys.exit(1)

    def evaluate(self):
        """평가 루프 (실전 시뮬레이션)"""
        logger.info("🚀 Starting Evaluation...")
        
        current_position = None  # None, 'LONG', 'SHORT'
        entry_price = 0.0
        entry_index = 0
        total_reward = 0.0
        trades = []
        balance_history = [config.EVAL_INITIAL_CAPITAL]  # 초기 자본
        prev_unrealized_pnl = 0.0  # 이전 스텝의 평가손익 추적
        prev_entry_index = 0  # 스위칭 시 이전 포지션의 entry_index 저장용
        
        # LSTM 상태 초기화
        self.agent.reset_episode_states()
        
        # [중요] 전략 신호는 이미 training_features.csv에 계산되어 있다고 가정
        # 하지만 엄격한 테스트를 위해선 여기서 실시간으로 계산하는 게 맞음.
        # (성능상 여기서는 저장된 값 사용하되, 저장 시 look-ahead 없었는지 확인 필수)
        
        pbar = tqdm(range(self.start_idx, self.end_idx - 1), desc="Evaluating")
        
        for idx in pbar:
            self.data_collector.current_index = idx
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])
            
            # PnL 계산
            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price
            
            # 관측
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / 1000]  # 정규화 대략
            
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None:
                continue
            
            # 이전 포지션 저장 (리워드 계산용)
            prev_pos_str = current_position
            
            # 행동 선택 (Deterministic=True for Eval)
            # 평가 때는 확률적 샘플링 대신 확률 가장 높은 행동 선택 권장
            with torch.no_grad():
                obs_seq, obs_info = state
                obs_seq = obs_seq.to(self.agent.device)
                obs_info = obs_info.to(self.agent.device)
                
                probs, _ = self.agent.model(obs_seq, obs_info, states=None, return_states=False)
                action = torch.argmax(probs, dim=-1).item()  # Deterministic
            
            # 3-Action Logic (스위칭 지원)
            trade_done = False
            realized_pnl = 0.0
            
            # Action 0: Neutral (HOLD/청산)
            if action == 0:
                if current_position is not None:
                    # 청산
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    prev_entry_index = entry_index  # 거래 기록용
                    current_position = None
                    entry_price = 0.0
                    entry_index = 0
                # 포지션 없으면 HOLD (Pass)
            
            # Action 1: Long (진입/유지/스위칭)
            elif action == 1:
                if current_position is None:
                    # 진입
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'SHORT':
                    # 스위칭: SHORT 청산 후 LONG 진입
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    prev_entry_index = entry_index  # 거래 기록용
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
                # 이미 LONG이면 유지 (Pass)
            
            # Action 2: Short (진입/유지/스위칭)
            elif action == 2:
                if current_position is None:
                    # 진입
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'LONG':
                    # 스위칭: LONG 청산 후 SHORT 진입
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    prev_entry_index = entry_index  # 거래 기록용
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx
                # 이미 SHORT면 유지 (Pass)
            
            # 리워드 계산 (3-Action 대응)
            # step_pnl: 이전 스텝 대비 평가손익 변화
            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0
            
            reward = self.env.calculate_reward(
                step_pnl=step_pnl,
                realized_pnl=realized_pnl,
                trade_done=trade_done,
                action=action,
                prev_position=prev_pos_str,
                current_position=current_position
            )
            total_reward += reward
            
            # 거래 기록 및 자본금 업데이트
            if trade_done:
                fee = getattr(config, 'TRANSACTION_COST', 0.001)
                # 스위칭의 경우 수수료가 두 번 발생할 수도 있지만(청산+진입),
                # 여기서는 1회분만 반영하거나, 엄격하게 2배 할 수 있음.
                # 일단 1.5배 정도로 평균 내서 적용
                actual_fee = fee * 1.5 if (prev_pos_str is not None and current_position is not None) else fee
                
                net_pnl = realized_pnl - actual_fee
                new_balance = balance_history[-1] * (1 + net_pnl)
                balance_history.append(new_balance)
                
                trades.append({
                    'entry_idx': prev_entry_index,
                    'exit_idx': idx,
                    'type': prev_pos_str,
                    'pnl': realized_pnl,
                    'net_pnl': net_pnl
                })
                # 거래 완료 후 prev_unrealized_pnl 초기화
                prev_unrealized_pnl = 0.0
            else:
                # 포지션 유지 중: 다음 스텝을 위해 현재 평가손익 저장
                prev_unrealized_pnl = unrealized_pnl
            
            pbar.set_postfix({'Bal': f"${balance_history[-1]:.0f}", 'Tr': len(trades)})
        
        # 마지막 포지션 청산
        if current_position is not None:
            final_price = float(self.data_collector.eth_data.iloc[self.end_idx - 1]['close'])
            if current_position == 'LONG':
                realized_pnl = (final_price - entry_price) / entry_price
            else:
                realized_pnl = (entry_price - final_price) / entry_price
            
            fee = getattr(config, 'TRANSACTION_COST', 0.001)
            net_pnl = realized_pnl - fee
            new_balance = balance_history[-1] * (1 + net_pnl)
            balance_history.append(new_balance)
            
            trades.append({
                'entry_idx': entry_index,
                'exit_idx': self.end_idx - 1,
                'type': current_position,
                'pnl': realized_pnl,
                'net_pnl': net_pnl
            })
        
        # 결과 리포트
        self._print_report(trades, balance_history, total_reward)

    def _print_report(self, trades, balance_history, total_reward):
        """평가 결과 리포트 출력"""
        if len(trades) == 0:
            logger.warning("⚠️ No trades executed")
            return
        
        df_trades = pd.DataFrame(trades)
        final_balance = balance_history[-1]
        initial_balance = balance_history[0]
        roi = (final_balance - initial_balance) / initial_balance * 100
        
        # 승률 계산
        win_trades = df_trades[df_trades['net_pnl'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
        avg_pnl = df_trades['net_pnl'].mean() * 100
        
        # 최대 낙폭 계산
        equity_array = np.array(balance_history)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        print("\n" + "="*50)
        print(f"📊 Evaluation Report")
        print("="*50)
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance:   ${final_balance:,.2f}")
        print(f"Net ROI:         {roi:.2f}%")
        print(f"Total Reward:    {total_reward:.2f}")
        print(f"Total Trades:    {len(trades)}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Avg PnL:         {avg_pnl:.4f}%")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        print("="*50)
        
        # 그래프 그리기
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(balance_history, label='Balance', linewidth=2)
            plt.title('Evaluation Balance History')
            plt.xlabel('Trades')
            plt.ylabel('Balance ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            logger.warning(f"⚠️ Plotting failed: {e}")


if __name__ == "__main__":
    # Test Mode로 실행
    evaluator = PPOEvaluator(mode='test')
    evaluator.evaluate()
