"""
Phase 1: Teacher-Guided TD3 Training (Final Optimized)
- [System] Linux/WSL2 자동 감지 및 가속 모드(TF32, Benchmark) 활성화
- [Fix] Resume 시 에피소드/람다 초기화 문제 해결 (State 저장)
- [Fix] Oracle Guidance + No Leverage (안정적 학습)
"""
import logging
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import platform  # OS 감지용
import json
import re
import glob
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
    from .td3_agent import TD3Agent
except ImportError:
    from TD3.td3_agent import TD3Agent

os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('common.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('common.mtf_processor').setLevel(logging.WARNING)


class TeacherGuidedTD3Trainer:
    def __init__(self):
        self.data_collector = DataCollector(use_saved_data=True)
        # Elite 8 Strategies
        self.strategies = [
            WhaleSentimentDivergence(), LiquidationSqueezeHunter(),
            OrderblockFVGStrategy(), NetTakerFlowStrategy(),
            BTCEthCorrelation(), VolatilitySqueeze(), VWAPDeviation(), HMAMomentum(),
        ]
        logger.info("전략 초기화: Elite 8 (%d개)", len(self.strategies))
        self._load_features()
        
        self.env = TradingEnvironment(self.data_collector, self.strategies)
        self.env.precompute_data() # GPU Caching & Signal Gen
        
        # TD3 전용 리워드 함수 주입
        from TD3.td3_reward import calculate_td3_reward
        import types
        self.env.calculate_reward = types.MethodType(calculate_td3_reward, self.env)
        logger.info("✅ TD3 전용 리워드 로직 (Simplified) 적용 완료")

        state_dim = self.env.get_state_dim() # 44
        action_dim = 1
        info_dim = 12

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 🚀 [Linux/WSL2 전용 가속 모드] 🚀
        # 스크립트가 리눅스 환경을 감지하면 자동으로 최적화를 수행합니다.
        if platform.system() == 'Linux':
            logger.info("=====================================================")
            logger.info("🐧 Linux/WSL2 Environment Detected: Engaging Turbo Mode")
            logger.info("=====================================================")
            
            # 1. TensorFloat-32 (TF32) 활성화 (RTX 3000번대 이상 필수)
            # FP32와 거의 같은 정확도로 FP16에 준하는 속도를 냄
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
                logger.info("   ✅ TF32 Precision Enabled (High Speed Matmul)")
            except AttributeError:
                logger.warning("   ⚠️ PyTorch version too old for set_float32_matmul_precision")

            # 2. CuDNN Benchmark 활성화
            # 입력 크기가 고정된 경우(RL은 대부분 해당) 최적의 알고리즘을 찾아 속도 향상
            torch.backends.cudnn.benchmark = True
            logger.info("   ✅ CuDNN Benchmark Enabled")
            
            # 3. (옵션) Torch Compile
            # PyTorch 2.0 이상에서 모델 컴파일 (호환성 문제 시 주석 처리)
            # config.USE_TORCH_COMPILE = True 
        else:
            logger.info(f"🖥️ OS Detected: {platform.system()} (Standard Mode)")

        logger.info("Teacher-Guided TD3 Training on %s | State Dim: %d | Info Dim: %d", device, state_dim, info_dim)

        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_time = run_time
        self.save_dir = os.path.join('data', 'td3_teacher', run_time)
        os.makedirs(self.save_dir, exist_ok=True)

        self.agent = TD3Agent(state_dim, action_dim, info_dim, device=device)
        self.writer = SummaryWriter(log_dir=f"logs/tensorboard/td3_teacher_{run_time}")
        
        self.pnl_history = deque(maxlen=20)
        
        self.oracle_df = None
        oracle_path = 'data/training_features_with_oracle.csv'
        if not os.path.exists(oracle_path):
            logger.warning("⚠️ Oracle 라벨 파일이 없습니다. 생성을 시작합니다...")
            try:
                oracle_script = os.path.join('utils', 'generate_oracle_labels.py')
                if not os.path.exists(oracle_script):
                     oracle_script = 'generate_oracle_labels.py'
                
                if os.path.exists(oracle_script):
                    result = subprocess.run([sys.executable, oracle_script], timeout=600)
                    if result.returncode != 0:
                        raise RuntimeError("Oracle 라벨 생성 실패")
                    logger.info("✅ Oracle 라벨 생성 완료")
                else:
                    logger.error("❌ Oracle generator script not found.")
            except Exception as e:
                logger.error(f"❌ Oracle 라벨 생성 실패: {e}")
                raise
        
        if os.path.exists(oracle_path):
            self.oracle_df = pd.read_csv(oracle_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
            logger.info(f"✅ Oracle 라벨 로드 완료: {len(self.oracle_df):,}행")
        else:
             logger.error("❌ Oracle data not loaded.")

    def _load_features(self):
        path = 'data/training_features.csv'
        cached_strategies_path = 'data/cached_strategies.csv'
        
        if not os.path.exists(path):
            logger.warning("⚠️ 피처 파일이 없습니다. 자동으로 데이터를 생성합니다...")
            try:
                prepare_script = os.path.join('utils', 'prepare_training_data.py')
                if not os.path.exists(prepare_script):
                    prepare_script = 'prepare_training_data.py'
                
                if os.path.exists(prepare_script):
                    result = subprocess.run([sys.executable, prepare_script], timeout=600)
                    if result.returncode == 0:
                        logger.info("✅ 피처 데이터 생성 완료")
                    else:
                        raise RuntimeError("피처 데이터 생성 실패")
                else:
                     logger.error("❌ prepare_training_data.py not found.")
            except Exception as e:
                logger.error(f"❌ 데이터 생성 중 오류: {e}")
                raise
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S').ffill().bfill()
            if os.path.exists(cached_strategies_path):
                try:
                    cached_df = pd.read_csv(cached_strategies_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
                    for col in [c for c in cached_df.columns if c.startswith('strategy_')]:
                        if col in cached_df.columns:
                            df[col] = cached_df[col]
                except: pass
            
            if 'volatility_20tick' not in df.columns:
                df = add_volatility_feature(df)
            
            self.data_collector.eth_data = df
            logger.info(f"✅ 데이터 로드 완료: {len(df):,}행")
        else:
            raise FileNotFoundError(f"파일 없음: {path}")

    def _fit_global_scaler_dummy(self):
        df = self.data_collector.eth_data
        if df is not None:
            self.train_end_idx = int(len(df) * config.TRAIN_SPLIT)

    def _augment_info(self, info, idx):
        try:
            vol = float(self.data_collector.eth_data.iloc[idx].get('volatility_20tick', 0.0))
        except: vol = 0.0

        if isinstance(info, torch.Tensor):
            vol_t = torch.tensor([vol], device=info.device, dtype=info.dtype)
            if info.dim() == 2: vol_t = vol_t.unsqueeze(0)
            return torch.cat([info, vol_t], dim=-1)
        return np.append(np.asarray(info).flatten(), vol).astype(np.float32)
    
    def _get_oracle_action(self, idx):
        try:
            oracle_label = int(self.oracle_df.iloc[idx]['oracle_action'])
            return float(oracle_label)
        except:
            return 0.0

    # [New] 학습 상태 저장 (Smart Save)
    def save_state(self, path, episode, total_timesteps, teacher_lambda):
        state = {
            'episode': episode,
            'total_timesteps': total_timesteps,
            'teacher_lambda': teacher_lambda,
            'timestamp': datetime.now().isoformat()
        }
        with open(path + "_state.json", 'w') as f:
            json.dump(state, f, indent=4)
            
    # [New] 학습 상태 로드 (Smart Resume)
    def load_state(self, path):
        json_path = path + "_state.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    state = json.load(f)
                logger.info(f"📥 학습 상태 복원 성공: Ep {state['episode']}, Lambda {state.get('teacher_lambda', '?')}")
                return state['episode'], state['total_timesteps']
            except Exception as e:
                logger.warning(f"⚠️ 상태 파일 손상, 체크포인트 검색으로 대체: {e}")
        
        try:
            model_dir = os.path.dirname(path)
            checkpoints = glob.glob(os.path.join(model_dir, "td3_teacher_model_*_actor.pth"))
            if checkpoints:
                episodes = [int(re.findall(r'model_(\d+)_actor', f)[0]) for f in checkpoints if re.findall(r'model_(\d+)_actor', f)]
                if episodes:
                    last_ep = max(episodes)
                    est_timesteps = last_ep * config.TRAIN_MAX_STEPS_PER_EPISODE
                    logger.info(f"🕵️ 체크포인트에서 에피소드 추론: Ep {last_ep} (추정)")
                    return last_ep + 1, est_timesteps
        except Exception as e:
            logger.warning(f"⚠️ 체크포인트 추론 실패: {e}")

        return 1, 0

    def train(self, resume=True):
        logger.info("🎓 Teacher-Guided TD3 Training Started (Optimized)...")
        
        # [튜닝] 람다 감소 기간: 2000 에피소드로 단축 (빠른 독립 유도)
        LAMBDA_ANNEAL_EPISODES = config.TD3_LAMBDA_ANNEAL_EPISODES
        logger.info(f"📚 Lambda Annealing: 1.0 → 0.0 ({LAMBDA_ANNEAL_EPISODES} episodes)")
        
        self._fit_global_scaler_dummy()
        
        total_timesteps = 0
        max_episodes = config.TRAIN_NUM_EPISODES
        max_steps = config.TRAIN_MAX_STEPS_PER_EPISODE
        warmup = config.TD3_WARMUP_STEPS
        best_reward = -float('inf')
        start_episode = 1

        if resume:
            td3_dir = os.path.join('data', 'td3_teacher')
            last_model_path = None
            if os.path.isdir(td3_dir):
                subdirs = [d for d in os.listdir(td3_dir) if os.path.isdir(os.path.join(td3_dir, d)) and d != self.run_time]
                for run_name in sorted(subdirs, reverse=True):
                    candidate = os.path.join(td3_dir, run_name, "last_td3_teacher_model_actor.pth")
                    if os.path.isfile(candidate):
                        last_model_path = os.path.join(td3_dir, run_name, "last_td3_teacher_model")
                        break
            if last_model_path:
                try:
                    self.agent.load(last_model_path)
                    logger.info("모델 로드 완료 (이어하기): %s", last_model_path)
                    
                    # [Smart Resume] 상태 복원
                    start_episode, saved_timesteps = self.load_state(last_model_path)
                    if start_episode > 1:
                        total_timesteps = saved_timesteps
                        logger.info(f"🔄 학습 재개: Episode {start_episode} / Steps {total_timesteps}")
                    else:
                        logger.info("⚠️ 이어하기 감지: Warmup을 강제로 스킵합니다.")
                        total_timesteps = warmup + 1
                        
                except Exception as e:
                    logger.warning("모델 로드 실패 (처음부터 진행): %s", e)
            else:
                logger.info("새로 학습 시작 (이전 모델 미로드)")
        else:
            logger.info("새로 학습 시작 (이전 모델 미로드)")

        TRANSACTION_COST = 0.0005

        for ep in range(start_episode, max_episodes + 1):
            teacher_lambda = max(0.0, 1.0 - (ep / LAMBDA_ANNEAL_EPISODES))
            
            low = config.LOOKBACK + 100
            high = max(low + 1, self.train_end_idx - max_steps - 100)
            start_idx = np.random.randint(low, high)

            self.data_collector.current_index = start_idx
            self.env.reset_reward_states()
            self.agent.position_cooldown = 0
            self.pnl_history.clear()

            rand_start = np.random.rand()
            current_pos_size = 0.0 if rand_start < 0.5 else (0.5 if rand_start < 0.75 else -0.5)

            pos_info = [current_pos_size, 0.0, 0.0]
            state = self.env.get_observation(position_info=pos_info, current_index=start_idx)
            if state is None: continue
            
            state = (state[0], self._augment_info(state[1], start_idx))

            episode_reward = 0.0
            episode_trades = 0

            for step in range(max_steps):
                total_timesteps += 1
                curr_idx = self.data_collector.current_index
                is_warmup = total_timesteps < warmup

                if is_warmup:
                    action_val = np.random.uniform(-1, 1)
                    risk_val = 0.5
                else:
                    action_val_arr, _, risk_val = self.agent.select_action(state, noise=0.1)
                    action_val = float(action_val_arr[0])

                # [Phase 1 Logic] No Leverage, Simplified
                target_pos_size = action_val if abs(action_val) > 0.1 else 0.0
                
                trade_amount = target_pos_size - current_pos_size
                if abs(trade_amount) > 1e-4:
                    episode_trades += 1
                
                trade_cost = abs(trade_amount) * TRANSACTION_COST
                current_pos_size = target_pos_size

                curr_price = float(self.data_collector.eth_data.iloc[curr_idx]['close'])
                self.data_collector.current_index += 1
                next_idx = self.data_collector.current_index
                
                if next_idx >= len(self.data_collector.eth_data):
                    done = True
                    next_state = state
                    break
                
                next_price = float(self.data_collector.eth_data.iloc[next_idx]['close'])
                price_return = (next_price - curr_price) / curr_price
                
                step_pnl = (current_pos_size * price_return) - trade_cost
                self.pnl_history.append(step_pnl)
                
                reward = self.env.calculate_reward(
                    step_pnl=step_pnl,
                    realized_pnl=0.0,
                    trade_done=abs(trade_amount) > 1e-4,
                    holding_time=step,
                    action=action_val,
                    prev_position=0.0,
                    current_position=current_pos_size,
                    effective_leverage=1.0 
                )
                
                episode_reward += reward

                next_pos_info = [current_pos_size, step_pnl * 100, 1.0 if abs(trade_amount) < 0.1 else 0.0]
                next_state_raw = self.env.get_observation(position_info=next_pos_info, current_index=next_idx)

                done = (step >= max_steps - 1) or (next_state_raw is None)
                if next_state_raw is None:
                    next_state = state
                else:
                    next_state = (next_state_raw[0], self._augment_info(next_state_raw[1], next_idx))

                oracle_act = self._get_oracle_action(curr_idx)
                self.agent.replay_buffer.add(state, [target_pos_size], reward, next_state, done, oracle_action=oracle_act)
                state = next_state

                if total_timesteps >= warmup:
                    metrics = self.agent.train(batch_size=config.TD3_BATCH_SIZE, teacher_lambda=teacher_lambda)
                    if metrics and step % 10 == 0:
                        self.writer.add_scalar('Loss/Critic', metrics.get('critic_loss', 0), total_timesteps)
                        self.writer.add_scalar('Loss/Teacher', metrics.get('l_teacher', 0), total_timesteps)
                        self.writer.add_scalar('Action/Pos_Size', current_pos_size, total_timesteps)

                if done: break

            # [Position Stats] Long/Short/Flat 비율 계산
            total_positions = sum(position_counts.values())
            long_ratio = (position_counts['long'] / total_positions * 100) if total_positions > 0 else 0
            short_ratio = (position_counts['short'] / total_positions * 100) if total_positions > 0 else 0
            flat_ratio = (position_counts['flat'] / total_positions * 100) if total_positions > 0 else 0
            
            logger.info("Ep %d | Lambda: %.3f | Reward: %.2f | Steps: %d | Trades: %d | L/S/F: %.1f%%/%.1f%%/%.1f%%", 
                        ep, teacher_lambda, episode_reward, total_timesteps, episode_trades,
                        long_ratio, short_ratio, flat_ratio)
            
            self.writer.add_scalar('Episode/Reward', episode_reward, ep)
            self.writer.add_scalar('Episode/Trades', episode_trades, ep)
            self.writer.add_scalar('Hyperparameter/TeacherLambda', teacher_lambda, ep)
            
            # [Position Distribution]
            self.writer.add_scalar('Position/Long_Ratio', long_ratio, ep)
            self.writer.add_scalar('Position/Short_Ratio', short_ratio, ep)
            self.writer.add_scalar('Position/Flat_Ratio', flat_ratio, ep)
            
            self.agent.save(os.path.join(self.save_dir, "last_td3_teacher_model"))
            # [Smart Save] 상태 저장
            self.save_state(os.path.join(self.save_dir, "last_td3_teacher_model"), ep + 1, total_timesteps, teacher_lambda)

            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(os.path.join(self.save_dir, "best_td3_teacher_model"))
                logger.info("New Best Model! Reward: %.2f", best_reward)

            if ep % 100 == 0:
                self.agent.save(os.path.join(self.save_dir, f"td3_teacher_model_{ep}"))
                self.save_state(os.path.join(self.save_dir, f"td3_teacher_model_{ep}"), ep + 1, total_timesteps, teacher_lambda)


if __name__ == "__main__":
    trainer = TeacherGuidedTD3Trainer()
    # 윈도우에서 옮겨온 경우 True로 하면 이어서 학습, False로 하면 새로 시작
    trainer.train(resume=True)