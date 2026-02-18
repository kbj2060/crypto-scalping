"""
계층적 RL 통합 학습 스크립트
- MetaController (Level 2): PPO, K=5 스텝 주기
- TacticalAgent (Level 1): Goal-Conditioned TD3, 매 스텝
- Kelly Criterion: 포지션 사이징 최적화

학습 순서:
  Phase 1 (Ep 1~500): MetaController만 학습 (TacticalAgent는 랜덤)
  Phase 2 (Ep 501~1500): TacticalAgent만 학습 (MetaController frozen)
  Phase 3 (Ep 1501~): 둘 다 학습 (End-to-End)
"""
import logging
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from common.preprocess import add_volatility_feature
from core.feature_engineering import ULTIMATE_FEATURE_COLS
from core import DataCollector
from strategies import (
    WhaleSentimentDivergence, LiquidationSqueezeHunter,
    OrderblockFVGStrategy, NetTakerFlowStrategy,
    BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum,
)
from common.trading_env import TradingEnvironment

from hierarchical.meta_controller import MetaController
from hierarchical.tactical_agent import GoalConditionedTD3Agent
from hierarchical.kelly_criterion import KellyCriterion
from hierarchical.hierarchical_reward import HierarchicalRewardCalculator

os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('common.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('common.mtf_processor').setLevel(logging.WARNING)

# PyTorch 최적화
if getattr(config, 'USE_CUDNN_BENCHMARK', False):
    torch.backends.cudnn.benchmark = True
if getattr(config, 'USE_HIGH_MATMUL_PRECISION', False):
    torch.set_float32_matmul_precision('high')

# ==============================================================================
# Training Phases
# ==============================================================================
PHASE1_END = 500      # MetaController 단독 학습
PHASE2_END = 1500     # TacticalAgent 단독 학습
# PHASE3: 1501~ 공동 학습

DECISION_INTERVAL = 5  # MetaController 결정 주기 (5 × 3분 = 15분)
TD3_UPDATE_FREQ = 10   # TD3 학습 주기 (10 스텝마다 한 번 학습, Off-policy 이점 활용)
TRANSACTION_COST = getattr(config, 'TRANSACTION_COST', 0.0005)


class HierarchicalTrainer:
    def __init__(self):
        # ============================================================
        # 1. Data & Environment Setup
        # ============================================================
        self.data_collector = DataCollector(use_saved_data=True)
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        logger.info("전략 초기화: Elite 8 (%d개)", len(self.strategies))
        self._load_features()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data()
        
        # NumPy 캐싱
        self.close_prices = self.data_collector.eth_data['close'].values.astype(np.float32)
        if 'volatility_20tick' not in self.data_collector.eth_data.columns:
            self.data_collector.eth_data = add_volatility_feature(self.data_collector.eth_data)
        self.volatility_data = self.data_collector.eth_data.get('volatility_20tick', 
                               pd.Series(np.zeros(len(self.close_prices)))).values.astype(np.float32)
        
        # ============================================================
        # 2. Agents Setup
        # ============================================================
        state_dim = self.env.get_state_dim()  # 44
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Level 2: MetaController (PPO)
        self.meta = MetaController(
            state_dim=state_dim,
            info_dim=11,  # Elite 8 기본 info
            hidden_dim=256,
            device=device,
            decision_interval=DECISION_INTERVAL
        )
        
        # Level 1: TacticalAgent (Goal-Conditioned TD3)
        self.tactical = GoalConditionedTD3Agent(
            state_dim=state_dim,
            action_dim=1,
            info_dim=12,  # 11 + volatility
            device=device
        )
        
        # Kelly Criterion
        self.kelly = KellyCriterion(
            fraction=0.25,        # Quarter Kelly (안전)
            max_leverage=config.LEVERAGE,
            min_trades=30,
            window_size=200
        )
        
        # Reward Calculator
        self.reward_calc = HierarchicalRewardCalculator(decision_interval=DECISION_INTERVAL)
        
        logger.info(f"🏗️ Hierarchical RL | Device: {device}")
        logger.info(f"   MetaController: PPO (hidden=256, interval={DECISION_INTERVAL})")
        logger.info(f"   TacticalAgent: Goal-Conditioned TD3 (goal_dim={self.tactical.goal_dim})")
        logger.info(f"   Kelly: fraction={self.kelly.fraction}, max_lev={self.kelly.max_leverage}")
        logger.info(f"   State Dim: {state_dim} | Info Dim: 12 | Goal Dim: {self.tactical.goal_dim}")
        
        # ============================================================
        # 3. Training Infrastructure
        # ============================================================
        self.train_end_idx = int(len(self.data_collector.eth_data) * config.TRAIN_SPLIT)
        
        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('data', 'hierarchical', run_time)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=f"logs/tensorboard/hierarchical_{run_time}")
        self.device = device
        
        # Warmup tracking
        self.total_tactical_steps = 0
        self.tactical_warmup = config.TD3_WARMUP_STEPS
    
    # ==================================================================
    # Data Loading
    # ==================================================================
    
    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if not os.path.exists(path):
            logger.warning("⚠️ 피처 파일 없음. 자동 생성...")
            try:
                result = subprocess.run(
                    [sys.executable, os.path.join('utils', 'prepare_training_data.py')],
                    timeout=600
                )
                if result.returncode != 0:
                    raise RuntimeError("피처 생성 실패")
            except Exception as e:
                raise RuntimeError(f"피처 생성 실패: {e}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S').ffill().bfill()
        
        if os.path.exists(cached_strategies_path):
            try:
                cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
                for col in [c for c in cached_df.columns if c.startswith('strategy_')]:
                    df[col] = cached_df[col]
            except:
                pass
        
        if 'volatility_20tick' not in df.columns:
            df = add_volatility_feature(df)
        
        self.data_collector.eth_data = df
        logger.info(f"✅ 데이터 로드: {len(df):,}행")
    
    # ==================================================================
    # Info Augmentation
    # ==================================================================
    
    def _augment_info(self, info, idx):
        """Info(11) + Volatility(1) = 12"""
        try:
            vol = float(self.volatility_data[idx])
        except:
            vol = 0.0
        
        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2:
                vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)
    
    def _get_meta_info(self, info):
        """Meta용 info: 첫 11차원만 (volatility 제외)"""
        if isinstance(info, torch.Tensor):
            return info[..., :11]
        return np.asarray(info).flatten()[:11]
    
    # ==================================================================
    # Main Training Loop
    # ==================================================================
    
    def train(self, num_episodes=3000, resume=True):
        logger.info("=" * 70)
        logger.info("🚀 Hierarchical RL Training Started")
        logger.info(f"   Phase 1 (Ep 1~{PHASE1_END}): MetaController 단독")
        logger.info(f"   Phase 2 (Ep {PHASE1_END+1}~{PHASE2_END}): TacticalAgent 단독")
        logger.info(f"   Phase 3 (Ep {PHASE2_END+1}~): 공동 학습")
        logger.info("=" * 70)
        
        if resume:
            self._try_load_models()
        
        best_reward = -float('inf')
        max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        
        for ep in range(1, num_episodes + 1):
            # Phase 결정
            if ep <= PHASE1_END:
                phase = 1
                train_meta = True
                train_tactical = False
            elif ep <= PHASE2_END:
                phase = 2
                train_meta = False
                train_tactical = True
            else:
                phase = 3
                train_meta = True
                train_tactical = True
            
            # 에피소드 실행
            result = self._run_episode(ep, max_steps, train_meta, train_tactical)
            if result is None:
                continue
            
            ep_reward, ep_trades, ep_pnl, meta_decisions = result
            
            # 학습
            meta_metrics = {}
            tactical_metrics = {}
            
            if train_meta:
                meta_metrics = self.meta.train_net()
            
            # TacticalAgent는 매 스텝에서 이미 학습됨 (TD3)
            
            # Logging
            phase_str = f"P{phase}"
            logger.info(
                f"Ep {ep} [{phase_str}] | Reward: {ep_reward:.2f} | "
                f"Trades: {ep_trades} | PnL: {ep_pnl*100:.2f}% | "
                f"Meta Decisions: {meta_decisions} | Kelly: {self.kelly}"
            )
            
            self._log_tensorboard(ep, ep_reward, ep_trades, ep_pnl, 
                                  meta_decisions, meta_metrics, phase)
            
            # Model Saving
            if ep % 10 == 0:
                self._save_models('last')
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                self._save_models('best')
                logger.info(f"🏆 New Best! Reward: {best_reward:.2f}")
        
        self.writer.close()
        logger.info("✅ Training Complete!")
    
    # ==================================================================
    # Episode Execution
    # ==================================================================
    
    def _run_episode(self, episode_num, max_steps, train_meta, train_tactical):
        """단일 에피소드 실행"""
        
        # 시작 인덱스 설정
        low = config.LOOKBACK + 100
        high = max(low + 1, self.train_end_idx - max_steps - 100)
        start_idx = np.random.randint(low, high)
        
        self.data_collector.current_index = start_idx
        self.env.reset_reward_states()
        self.meta.reset()
        self.reward_calc.reset()
        
        # 초기 상태
        current_pos_size = 0.0
        pos_info = [current_pos_size, 0.0, 0.0]
        
        state = self.env.get_observation(position_info=pos_info, current_index=start_idx)
        if state is None:
            return None
        
        # Augmented info for tactical
        tactical_info = self._augment_info(state[1], start_idx)
        tactical_state = (state[0], tactical_info)
        
        # Meta의 첫 결정
        meta_info = self._get_meta_info(state[1])
        meta_state = (state[0], meta_info)
        current_goal = self.meta.select_action(meta_state, deterministic=(not train_meta))
        goal_tensor = self.tactical._goal_to_tensor(current_goal)
        self.reward_calc.on_meta_decision(current_goal['direction'])
        
        # Episode tracking
        episode_reward = 0.0
        episode_trades = 0
        episode_pnl = 0.0
        meta_decision_count = 1
        
        # Meta의 현재 결정에 대한 상태 저장
        meta_decision_state = meta_state
        meta_decision_log_prob = current_goal['log_prob']
        meta_decision_value = current_goal['value']
        
        for step in range(max_steps):
            curr_idx = self.data_collector.current_index
            self.total_tactical_steps += 1
            
            # ==============================================================
            # MetaController: K스텝마다 결정
            # ==============================================================
            self.meta.step()
            
            if self.meta.should_decide() and step > 0:
                # 이전 K스텝의 Meta 보상 계산
                meta_reward = self.reward_calc.calculate_meta_reward()
                
                # 새 Meta 상태
                new_meta_state = (state[0], self._get_meta_info(state[1]))
                
                # Meta 경험 저장
                if train_meta:
                    done_meta = False
                    self.meta.put_data((
                        meta_decision_state,
                        current_goal['direction'],
                        meta_reward,
                        meta_decision_log_prob,
                        done_meta,
                        meta_decision_value,
                    ))
                
                # 새 결정
                current_goal = self.meta.select_action(new_meta_state, deterministic=(not train_meta))
                goal_tensor = self.tactical._goal_to_tensor(current_goal)
                self.reward_calc.on_meta_decision(current_goal['direction'])
                
                meta_decision_state = new_meta_state
                meta_decision_log_prob = current_goal['log_prob']
                meta_decision_value = current_goal['value']
                meta_decision_count += 1
            
            # ==============================================================
            # TacticalAgent: 매 스텝 실행
            # ==============================================================
            
            is_warmup = self.total_tactical_steps < self.tactical_warmup
            
            if is_warmup or not train_tactical:
                # Warmup: 랜덤 액션 (단, Meta 방향은 참조)
                if current_goal['direction'] == 1:
                    action_val = np.random.uniform(0, 1)
                elif current_goal['direction'] == 2:
                    action_val = np.random.uniform(-1, 0)
                else:
                    action_val = np.random.uniform(-0.3, 0.3)
                gate_mean = 0.5
            else:
                action_arr, gate_mean = self.tactical.select_action(
                    tactical_state, current_goal, noise=0.15
                )
                action_val = float(action_arr[0])
            
            # ==============================================================
            # Position & Leverage via Kelly
            # ==============================================================
            
            risk_budget = current_goal.get('risk_budget', 0.3)
            effective_leverage = self.kelly.get_position_size(
                abs(action_val), risk_budget
            )
            
            # [Kelly Decoupling] 학습 초기(Phase 1, 2)에는 최소 레버리지 보장
            # Kelly가 0배를 출력해도 강제로 탐험하여 데이터를 수집함
            if not train_meta or not train_tactical:  # Phase 1 or 2
                effective_leverage = max(effective_leverage, 1.0)  # 최소 1배 보장
            
            # 포지션 크기 결정
            target_pos_size = np.sign(action_val) * min(abs(action_val), 1.0) if effective_leverage >= 1.0 else 0.0
            
            # Deadzone
            if abs(target_pos_size) < 0.15:
                target_pos_size = 0.0
            
            # 포지션 변경 필터링
            is_opening = (current_pos_size == 0.0) and (target_pos_size != 0.0)
            is_flipping = (current_pos_size * target_pos_size < 0)
            is_strength_change = abs(target_pos_size - current_pos_size) > 0.3
            
            if not (is_opening or is_flipping or is_strength_change):
                target_pos_size = current_pos_size
            
            trade_amount = target_pos_size - current_pos_size
            if abs(trade_amount) > 1e-4:
                episode_trades += 1
            
            trade_cost = effective_leverage * TRANSACTION_COST if abs(trade_amount) > 1e-4 else 0.0
            current_pos_size = target_pos_size
            
            # ==============================================================
            # Price Movement & PnL
            # ==============================================================
            
            curr_price = float(self.close_prices[curr_idx])
            self.data_collector.current_index += 1
            next_idx = self.data_collector.current_index
            
            if next_idx >= len(self.close_prices):
                break
            
            next_price = float(self.close_prices[next_idx])
            price_return = (next_price - curr_price) / curr_price
            
            position_direction = np.sign(current_pos_size) if abs(current_pos_size) > 0.01 else 0.0
            raw_return = price_return * position_direction
            
            step_pnl_roe = (raw_return * effective_leverage) - trade_cost if abs(current_pos_size) > 0.01 else 0.0
            
            # Kelly에 거래 결과 기록
            if abs(trade_amount) > 1e-4 and step_pnl_roe != 0:
                self.kelly.record_trade(step_pnl_roe)
            
            episode_pnl += step_pnl_roe
            
            # ==============================================================
            # 청산 체크
            # ==============================================================
            should_exit, exit_reason = self.env.check_exit_conditions(
                step_pnl_roe, holding_time_steps=step
            )
            if should_exit:
                current_pos_size = 0.0
                if exit_reason == "LIQUIDATION":
                    step_pnl_roe = -0.80
            
            # ==============================================================
            # Rewards
            # ==============================================================
            
            # Meta 누적
            self.reward_calc.accumulate_for_meta(step_pnl_roe)
            
            # Tactical 보상
            tactical_reward = self.reward_calc.calculate_tactical_reward(
                step_pnl=step_pnl_roe,
                trade_done=abs(trade_amount) > 1e-4,
                realized_pnl=step_pnl_roe if abs(trade_amount) > 1e-4 else 0.0,
                action=action_val,
                meta_goal=current_goal,
                effective_leverage=effective_leverage
            )
            
            episode_reward += tactical_reward
            
            # ==============================================================
            # Next State
            # ==============================================================
            
            next_pos_info = [current_pos_size, step_pnl_roe * 10, 
                            1.0 if abs(trade_amount) < 0.1 else 0.0]
            next_state_raw = self.env.get_observation(
                position_info=next_pos_info, current_index=next_idx
            )
            
            done = (step >= max_steps - 1) or (next_state_raw is None)
            
            if next_state_raw is None:
                next_tactical_state = tactical_state
                next_goal_tensor = goal_tensor
            else:
                next_tactical_info = self._augment_info(next_state_raw[1], next_idx)
                next_tactical_state = (next_state_raw[0], next_tactical_info)
                next_goal_tensor = goal_tensor  # Goal은 K스텝 동안 유지
                state = next_state_raw
            
            # Tactical 경험 저장 & 학습
            if train_tactical:
                self.tactical.replay_buffer.add(
                    tactical_state, goal_tensor,
                    [target_pos_size], tactical_reward,
                    next_tactical_state, next_goal_tensor,
                    done
                )
                
                # [Delayed Update] 매 스텝이 아닌 TD3_UPDATE_FREQ마다 학습 (3-5배 속도 향상)
                if self.total_tactical_steps >= self.tactical_warmup:
                    if self.total_tactical_steps % TD3_UPDATE_FREQ == 0:
                        t_metrics = self.tactical.train(batch_size=config.TD3_BATCH_SIZE)
            
            tactical_state = next_tactical_state
            
            if done:
                break
        
        # 에피소드 종료: 마지막 Meta 보상
        if train_meta and self.reward_calc.meta_step_count > 0:
            meta_reward = self.reward_calc.calculate_meta_reward()
            self.meta.put_data((
                meta_decision_state,
                current_goal['direction'],
                meta_reward,
                meta_decision_log_prob,
                True,  # done
                meta_decision_value,
            ))
        
        return episode_reward, episode_trades, episode_pnl, meta_decision_count
    
    # ==================================================================
    # Utilities
    # ==================================================================
    
    def _log_tensorboard(self, ep, reward, trades, pnl, meta_decisions, meta_metrics, phase):
        self.writer.add_scalar('Episode/Reward', reward, ep)
        self.writer.add_scalar('Episode/Trades', trades, ep)
        self.writer.add_scalar('Episode/PnL', pnl, ep)
        self.writer.add_scalar('Episode/Phase', phase, ep)
        self.writer.add_scalar('Meta/Decisions', meta_decisions, ep)
        
        kelly_stats = self.kelly.get_stats()
        self.writer.add_scalar('Kelly/WinRate', kelly_stats['win_rate'], ep)
        self.writer.add_scalar('Kelly/OptimalLeverage', kelly_stats['optimal_leverage'], ep)
        self.writer.add_scalar('Kelly/FractionalKelly', kelly_stats['fractional_kelly'], ep)
        
        if meta_metrics:
            for k, v in meta_metrics.items():
                self.writer.add_scalar(f'Meta/{k}', v, ep)
    
    def _save_models(self, suffix):
        self.meta.save(os.path.join(self.save_dir, f'meta_{suffix}.pth'))
        self.tactical.save(os.path.join(self.save_dir, f'tactical_{suffix}'))
        
        # Kelly 상태도 저장
        import json
        kelly_path = os.path.join(self.save_dir, f'kelly_{suffix}.json')
        with open(kelly_path, 'w') as f:
            json.dump(self.kelly.get_stats(), f, indent=2)
    
    def _try_load_models(self):
        """이전 모델 탐색 및 로드"""
        hier_dir = os.path.join('data', 'hierarchical')
        if not os.path.isdir(hier_dir):
            logger.info("이전 계층적 모델 없음, 새로 학습 시작")
            return
        
        subdirs = sorted(
            [d for d in os.listdir(hier_dir) if os.path.isdir(os.path.join(hier_dir, d))],
            reverse=True
        )
        
        for run_name in subdirs:
            meta_path = os.path.join(hier_dir, run_name, 'meta_last.pth')
            tactical_path = os.path.join(hier_dir, run_name, 'tactical_last')
            
            if os.path.exists(meta_path):
                self.meta.load(meta_path)
                self.tactical.load(tactical_path)
                logger.info(f"✅ 이전 모델 로드: {run_name}")
                
                # 이어하기면 warmup 스킵
                self.total_tactical_steps = self.tactical_warmup + 1
                logger.info("⚠️ 이어하기: Tactical Warmup 스킵")
                return
        
        logger.info("이전 모델 없음, 새로 학습 시작")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    trainer = HierarchicalTrainer()
    trainer.train(num_episodes=config.TRAIN_NUM_EPISODES, resume=False)
