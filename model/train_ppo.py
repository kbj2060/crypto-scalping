"""
PPO 학습 스크립트 (Curriculum Learning + TensorBoard)
- 수정: Action Mask 저장 기능 추가 (KL Divergence 폭발 방지)
"""
import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)
from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)

class PPOTrainer:
    def __init__(self, enable_visualization=False):
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            BTCEthCorrelationStrategy(), VolatilitySqueezeStrategy(), OrderblockFVGStrategy(),
            HMAMomentumStrategy(), MFIMomentumStrategy(), BollingerMeanReversionStrategy(),
            VWAPDeviationStrategy(), RangeTopBottomStrategy(), StochRSIMeanReversionStrategy(),
            CMFDivergenceStrategy(), CCIReversalStrategy(), WilliamsRStrategy()
        ]
        
        logger.info(f"전략 초기화: {len(self.strategies)}개 전략")
        self._load_features()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self._fit_global_scaler_dummy()

        state_dim = self.env.get_state_dim()
        action_dim = 3  # [Hold, Buy, Sell]
        info_dim = 15   # [pos_val(1) + strategies(12) + pos_info[1:](2)]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device} | State Dim: {state_dim} | Info Dim: {info_dim}")

        self.agent = PPOAgent(state_dim, action_dim, info_dim=info_dim, device=device)
        
        # 모델 로드
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        last_model_path = f"{base_path}_last.pth" 
        try:
            self.agent.load_model(last_model_path)
        except Exception as e:
            logger.warning(f"초기 모델 로드 실패 (무시 가능): {e}")
        
        self.episode_rewards = []
        
        # TensorBoard
        tb_base = os.path.join('logs', 'tensorboard')
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_log_dir = os.path.join(tb_base, run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        
        self._prepare_curriculum_indices()

    def get_action_mask(self, current_position, market_volatility, step_count):
        """3-Action 마스킹: [Hold, Buy, Sell]"""
        mask = np.ones(3, dtype=np.float32)
        if current_position == 'LONG':
            mask[1] = 0.0  # Buy(추가매수) 불가
        elif current_position == 'SHORT':
            mask[2] = 0.0  # Sell(추가매도) 불가
        else:
            # [점검] 변동성 제한이 너무 엄격하면 거래를 막음. 0.0001로 완화하거나 아래 주석 처리 권장
            if market_volatility < 0.0001:
                mask[1] = 0.0
                mask[2] = 0.0
        return mask

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True)
                    strategy_cols = [c for c in cached_df.columns if c.startswith('strategy_')]
                    for col in strategy_cols:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except:
                    pass
            self.data_collector.eth_data = df
        else:
            logger.warning("⚠️ 피처 파일이 없습니다.")

    def _fit_global_scaler_dummy(self):
        df = self.data_collector.eth_data
        if df is not None:
            train_size = int(len(df) * config.TRAIN_SPLIT)
            self.train_end_idx = train_size
            self.test_start_idx = int(len(df) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
            self.test_end_idx = len(df)
            self.env.scaler_fitted = True

    def _prepare_curriculum_indices(self):
        """
        [MacroHFT] 데이터를 3가지 시장 국면(Trend, Volatility, Sideways)으로 분류하여 인덱싱
        """
        df = self.data_collector.eth_data.iloc[:self.train_end_idx]
        valid_indices = list(range(config.LOOKBACK + 100, self.train_end_idx - 500))

        if len(valid_indices) < 100:
            logger.warning("데이터가 너무 적어 커리큘럼 인덱스를 생성할 수 없습니다.")
            self.all_indices = valid_indices
            self.trend_indices = self.indices_trend = self.indices_vol = self.indices_chop = self.indices_all = valid_indices
            self.idx_trend = self.idx_vol = self.idx_side = valid_indices
            self.idx_map = [valid_indices, valid_indices, valid_indices]
            self.idx_all = valid_indices
            return

        self.all_indices = valid_indices
        self.indices_all = valid_indices

        if 'chop' in df.columns and 'atr_ratio' in df.columns:
            chop = df['chop'].values
            atr = df['atr_ratio'].values
            self.indices_trend = [i for i in valid_indices if i < len(chop) and chop[i] < 45.0]
            atr_vals = [atr[i] for i in valid_indices if i < len(atr)]
            atr_threshold = np.quantile(atr_vals, 0.75) if atr_vals else 0.0
            self.indices_vol = [i for i in valid_indices if i < len(atr) and atr[i] > atr_threshold]
            atr_mean = np.mean(atr_vals) if atr_vals else 0.0
            self.indices_chop = [
                i for i in valid_indices
                if i < len(chop) and i < len(atr) and chop[i] > 50.0 and atr[i] < atr_mean
            ]
            self.trend_indices = self.indices_trend
            self.idx_trend = self.indices_trend
            self.idx_vol = self.indices_vol
            self.idx_side = self.indices_chop
            self.idx_map = [self.idx_trend, self.idx_vol, self.idx_side]
            self.idx_all = self.indices_all
            logger.info("📊 [MacroHFT Data Split]")
            logger.info(f"  - Trend: {len(self.indices_trend)} | Volatility: {len(self.indices_vol)} | Sideways: {len(self.indices_chop)}")
        else:
            self.indices_trend = self.all_indices
            self.indices_vol = self.all_indices
            self.indices_chop = self.all_indices
            self.trend_indices = self.all_indices
            self.idx_trend = self.idx_vol = self.idx_side = self.all_indices
            self.idx_map = [self.all_indices, self.all_indices, self.all_indices]
            self.idx_all = self.all_indices
            logger.info(f"📚 커리큘럼: chop/atr 없음 → 전체 {len(self.all_indices)}개 사용")

    def validate_on_test_set(self, max_steps=480):
        if not hasattr(self, 'test_start_idx') or self.test_start_idx >= self.test_end_idx - 1:
            return 0.0, 0.0
        steps = min(max_steps, self.test_end_idx - self.test_start_idx - 1)
        
        self.agent.reset_episode_states()
        balance = config.EVAL_INITIAL_CAPITAL
        balance_history = [balance]
        current_position = None
        entry_price = 0.0
        entry_index = 0
        fee_rate = getattr(config, 'TRANSACTION_COST', 0.0005)

        for step in range(steps):
            idx = self.test_start_idx + step
            curr_price = float(self.data_collector.eth_data.iloc[idx]['close'])

            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price

            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max(1, steps)]
            state = self.env.get_observation(position_info=pos_info, current_index=idx)
            if state is None: break

            with torch.no_grad():
                # 검증 시에는 마스크 없이 순수 모델 판단을 보거나, 동일하게 마스크 적용 가능
                # 여기서는 검증이므로 마스크 없이 진행하거나 간단히 적용
                action, _, _ = self.agent.select_action(state, action_mask=None)

            realized_pnl = 0.0
            if action == 1:
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'SHORT':
                    realized_pnl = (entry_price - curr_price) / entry_price - fee_rate
                    current_position = None
                    entry_price = 0.0
            elif action == 2:
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price
                    entry_index = idx
                elif current_position == 'LONG':
                    realized_pnl = (curr_price - entry_price) / entry_price - fee_rate
                    current_position = None
                    entry_price = 0.0

            if realized_pnl != 0.0:
                balance = balance * (1 + realized_pnl)
            balance_history.append(balance)

        test_reward = (balance - config.EVAL_INITIAL_CAPITAL) / config.EVAL_INITIAL_CAPITAL
        returns = np.diff(balance_history) / (np.array(balance_history[:-1], dtype=float) + 1e-10)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8))
        return test_reward, sharpe

    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None:
            max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE

        phase_1_epochs = 1000
        if episode_num <= phase_1_epochs:
            mode = "expert"
            expert_idx = episode_num % 3
            target_indices = getattr(self, "idx_map", [self.all_indices] * 3)[expert_idx]
            if len(target_indices) < 50:
                target_indices = getattr(self, "idx_all", self.all_indices)
            mode_name = f"Pretrain: {self.agent.expert_names[expert_idx].upper()}"
        else:
            mode = "router"
            expert_idx = 0
            target_indices = getattr(self, "idx_all", self.all_indices)
            mode_name = "Training: ROUTER"

        if not target_indices:
            target_indices = self.all_indices
        start_idx = np.random.choice(target_indices)
        self.data_collector.current_index = start_idx
        self.env.reset_reward_states()
        self.agent.reset_episode_states()

        current_position = None
        entry_price = 0.0
        entry_index = 0
        episode_reward = 0.0
        episode_pnl = 0.0
        trade_count = 0
        prev_unrealized_pnl = 0.0
        hold_count, buy_count, sell_count = 0, 0, 0
        
        pbar = tqdm(range(max_steps), desc=f"Ep {episode_num} [{mode_name}]", leave=False)
        
        for step in pbar:
            current_idx = self.data_collector.current_index
            if current_idx >= self.train_end_idx - 1:
                break

            curr_price = float(self.data_collector.eth_data.iloc[current_idx]['close'])

            # Aux Target (Volatility)
            lookback_vol = min(10, current_idx - config.LOOKBACK)
            if lookback_vol > 0:
                past_prices = self.data_collector.eth_data['close'].iloc[current_idx-lookback_vol:current_idx].values
                returns = np.diff(past_prices) / (past_prices[:-1] + 1e-10)
                volatility_label = float(np.std(returns) * 100.0)
            else:
                volatility_label = 0.0

            unrealized_pnl = 0.0
            if current_position == 'LONG':
                unrealized_pnl = (curr_price - entry_price) / entry_price
            elif current_position == 'SHORT':
                unrealized_pnl = (entry_price - curr_price) / entry_price

            step_pnl = unrealized_pnl - prev_unrealized_pnl if current_position else 0.0
            
            pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            holding_time = (current_idx - entry_index) if current_position else 0
            pos_info = [pos_val, unrealized_pnl * 10, holding_time / max_steps]
            
            state = self.env.get_observation(position_info=pos_info, current_index=current_idx)
            if state is None: break

            prev_pos_str = current_position

            # Action Masking
            market_vol = 0.0
            if 'atr_ratio' in self.data_collector.eth_data.columns:
                market_vol = float(self.data_collector.eth_data.iloc[current_idx]['atr_ratio'])
            
            action_mask = self.get_action_mask(current_position, market_vol, step)
            action, prob, val = self.agent.select_action(
                state, action_mask=action_mask, mode=mode, expert_idx=expert_idx
            )

            # Stop Loss / Take Profit / Time Stop 강제 청산
            should_exit, exit_reason = self.env.check_exit_conditions(unrealized_pnl, holding_time)
            if should_exit and current_position is not None:
                if current_position == 'LONG':
                    action = 2
                elif current_position == 'SHORT':
                    action = 1
                if exit_reason:
                    pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count, 'Exit': exit_reason[:4]})

            if action == 0: hold_count += 1
            elif action == 1: buy_count += 1
            else: sell_count += 1

            slippage = np.random.uniform(0.0001, 0.0005)
            trade_done = False
            realized_pnl = 0.0
            holding_time_norm = holding_time / max_steps

            if action == 1:  # Buy
                if current_position is None:
                    current_position = 'LONG'
                    entry_price = curr_price * (1 + slippage)
                    entry_index = current_idx
                    trade_count += 1
                elif current_position == 'SHORT':
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    episode_pnl += realized_pnl
                    current_position = None
                    entry_price = 0.0
                    trade_count += 1
            elif action == 2:  # Sell
                if current_position is None:
                    current_position = 'SHORT'
                    entry_price = curr_price * (1 - slippage)
                    entry_index = current_idx
                    trade_count += 1
                elif current_position == 'LONG':
                    realized_pnl = unrealized_pnl
                    trade_done = True
                    episode_pnl += realized_pnl
                    current_position = None
                    entry_price = 0.0
                    trade_count += 1

            # Metric Update
            strategy_scores = []
            for i in range(len(self.strategies)):
                col = f'strategy_{i}'
                if col in self.data_collector.eth_data.columns:
                    strategy_scores.append(float(self.data_collector.eth_data[col].iloc[current_idx]))
            
            self.env.update_trading_metrics(
                prev_position=prev_pos_str,
                current_position=current_position,
                strategy_scores=strategy_scores if strategy_scores else None,
                volatility_pred=0.0,
                actual_volatility=market_vol,
            )

            reward = self.env.calculate_reward(
                step_pnl=step_pnl, realized_pnl=realized_pnl, trade_done=trade_done,
                holding_time=holding_time_norm, action=action,
                prev_position=prev_pos_str, current_position=current_position,
            )

            prev_unrealized_pnl = unrealized_pnl if not trade_done else 0.0
            self.data_collector.current_index += 1
            next_idx = self.data_collector.current_index
            
            next_pos_val = 1.0 if current_position == 'LONG' else (-1.0 if current_position == 'SHORT' else 0.0)
            next_hold_time = (next_idx - entry_index) if current_position is not None else 0
            next_pos_info = [next_pos_val, 0.0, next_hold_time / max_steps]
            next_state = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)
            
            done = False if step < max_steps - 1 else True
            if next_state is None:
                done = True
                next_state = state
            
            # [✅ 중요] action_mask를 9번째 인자로 추가 저장
            self.agent.put_data((state, action, reward, next_state, prob, done, val, volatility_label, action_mask))
            episode_reward += reward
            pbar.set_postfix({'R': f'{episode_reward:.1f}', 'Tr': trade_count})
            
            if done: break
        
        # 강제 청산 (에피소드 종료 시): last_state 안전하게 조회
        if current_position is not None:
            if current_position == 'LONG':
                realized_pnl = (curr_price - entry_price) / entry_price
                exit_action = 2
            else:
                realized_pnl = (entry_price - curr_price) / entry_price
                exit_action = 1
            episode_pnl += realized_pnl
            final_reward = self.env.calculate_reward(
                step_pnl=0.0, realized_pnl=realized_pnl, trade_done=True,
                holding_time=0.0, action=exit_action,
                prev_position=current_position, current_position=None,
            )
            safe_idx = min(self.data_collector.current_index, self.train_end_idx - 1)
            last_pos_info = [0.0, 0.0, 0.0]
            last_state = self.env.get_observation(position_info=last_pos_info, current_index=safe_idx)
            if last_state is None:
                last_state = state
            terminal_mask = np.ones(3, dtype=np.float32)
            self.agent.put_data((state, exit_action, final_reward, last_state, 1.0, True, 0.0, 0.0, terminal_mask))
            episode_reward += final_reward
            trade_count += 1
            
        pbar.close()

        metrics = self.agent.train_net(episode=episode_num, mode=mode, expert_idx=expert_idx)
        total_steps = hold_count + buy_count + sell_count
        if episode_num % 10 == 0:
            logger.info(f"Ep {episode_num} [{mode_name}] Reward: {episode_reward:.1f} | Trades: {trade_count}")

        if self.writer:
            self.writer.add_scalar('Reward/Total', episode_reward, episode_num)
            self.writer.add_scalar('Metrics/PnL', episode_pnl, episode_num)
            self.writer.add_scalar('Metrics/Trade_Count', trade_count, episode_num)
            self.writer.add_scalar('Actions/Hold_Rate', hold_count / max(1, total_steps), episode_num)
            self.writer.add_scalar('Actions/Buy_Rate', buy_count / max(1, total_steps), episode_num)
            self.writer.add_scalar('Actions/Sell_Rate', sell_count / max(1, total_steps), episode_num)
            
            # [✅ 로그 기록] 딕셔너리 풀어서 기록
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, episode_num)

        return episode_reward, trade_count, episode_pnl

    def train(self, num_episodes=1000):
        logger.info("🚀 PPO 학습 시작 (Action Mask Fix)")
        best_reward = -float('inf')
        base_path = config.AI_MODEL_PATH.replace('.pth', '')
        
        for ep in range(1, num_episodes + 1):
            try:
                res = self.train_episode(ep)
                if res is None: continue
                r, c, pnl = res
                self.episode_rewards.append(r)
                avg_r = np.mean(self.episode_rewards[-10:])
                logger.info(f"✅ Ep {ep}: Reward {r:.4f} | Avg {avg_r:.4f} | Trades: {c} | PnL: {pnl*100:.2f}%")
                
                if r > best_reward:
                    best_reward = r
                    self.agent.save_model(f"{base_path}_best.pth")
                if ep % 50 == 0:
                    self.agent.save_model(f"{base_path}_last.pth")
                    
            except KeyboardInterrupt:
                logger.info("학습 중단 요청됨.")
                break
            except Exception as e:
                logger.error(f"Ep {ep} Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.writer.close()

if __name__ == "__main__":
    # 실행부
    trainer = PPOTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES)