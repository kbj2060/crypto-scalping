"""
Trading Router — 2-Agent Independent Learning (Long vs Short)
================================================================================
1. 복잡한 MoE를 폐기하고 직관적인 Long / Short 독립 에이전트로 분리
2. Action Space 통일: 0(Hold), 1(Enter), 2(Close)
3. RobustIQN (MLP) 기반 경량화 신경망 + SimpleBuffer (Regime 필터 제거)
4. SimpleRouter: 두 에이전트의 Q-Value(진입 우위)를 직관적으로 비교하여 의사결정
"""
import os, sys, logging, random, argparse, gc
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_lightning as pl
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, 'ensemble'), os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path: sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# ═══════════════════════════════════════════════════════════════════════════
# [상수 및 차원 정의]
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PRED = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst', 'pred_itransformer', 'pred_nhits', 'pred_tide']
MODEL_CONF = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst', 'conf_itransformer', 'conf_nhits', 'conf_tide']

ELITE_COLS = [
    'sig_whale', 'sig_liq_squeeze', 'sig_net_taker', 'sig_orderblock',
    'sig_hurst_ofi', 'sig_funding_cascade', 'sig_multifractal', 'sig_cluster_fib',
    'sig_oi_divergence', 'sig_top_trader_squeeze', 'sig_btc_corr_breakout',
    'sig_ai_squeeze', 'sig_vp_gravity'  
]

ALPHA_7_COLS = [
    'session_us', 'hour_cos', 'cvp_poc_dist', 
    'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate'
]

REGIME_COLS = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']

TARGET_COL = 'log_return'
RL_REQUIRED_COLS = ['timestamp', 'close'] + MODEL_PRED + MODEL_CONF + ELITE_COLS + ALPHA_7_COLS + REGIME_COLS + [TARGET_COL]

FEATURE_DIM = len(MODEL_PRED) + len(MODEL_CONF) + 3 + len(ELITE_COLS) + len(ALPHA_7_COLS) + len(REGIME_COLS)
STATE_DIM = FEATURE_DIM + 5  

def row_to_market_row(row: pd.Series) -> dict:
    return {k: v for k, v in row.items()}

# ═══════════════════════════════════════════════════════════════════════════
# 1. 하이브리드 배치 마이닝 엔진 (기존과 동일)
# ═══════════════════════════════════════════════════════════════════════════
def generate_training_csv(input_csv: str, output_csv: str):
    # (데이터 마이닝 로직은 원본 그대로 사용하시면 됩니다. 생략 없이 기존 함수를 복사해 넣으세요.)
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 2. 거래 환경 (TradingEnv) - 단일화된 Action Space (0:Hold, 1:Enter, 2:Close)
# ═══════════════════════════════════════════════════════════════════════════
class TradingEnv:
    STATE_DIM = STATE_DIM

    def __init__(self, df, initial_balance=10000.0, fee=0.0006, slip=0.0003, phase='train', agent_role='long_agent'):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.agent_role = agent_role  # 'long_agent', 'short_agent', 'neutral'
        
        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1
        self.MAX_LEVERAGE = 1.0
        self.MAX_HOLD = {'train': 72, 'val': 144, 'test': 288}

        self.reset()

    def reset(self, start_idx=None):
        if self.phase == 'train':
            self.start_step = start_idx if start_idx is not None else random.randint(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = self.start_step + self.MAX_EPISODE_STEPS

        self.balance = self.initial_balance
        self.pos = None
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0 
        
        self.total_trades = 0
        self.win_trades = 0
        self.active_steps = 0
        
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0 
        self.hold_count = 0
        
        return self._build_state(self.current_step)

    def step(self, action, leverage_rate=1.0):
        current_price = self.df.loc[self.current_step, 'close']
        
        # SL/TP 및 강제 청산 (훈련 시 강제로 2(Close) 발동)
        if self.pos is not None:
            if self.unrealized_pnl <= -0.015 or self.unrealized_pnl >= 0.030 or self.hold_count > self.MAX_HOLD[self.phase]:
                if self.phase == 'train': action = 2 
                else: action = 0 # Val 환경에서는 0이 청산명령

        reward = 0.0
        is_closed = False
        realized_pnl = 0.0

        is_entering_long = False
        is_entering_short = False
        is_closing = False

        # 💡 [NEW] 0:Hold, 1:Enter, 2:Close 에 맞춘 명확한 분기
        if self.phase == 'train':
            if action == 1 and self.pos is None:
                if self.agent_role == 'long_agent': is_entering_long = True
                elif self.agent_role == 'short_agent': is_entering_short = True
            elif action == 2 and self.pos is not None:
                is_closing = True
        else: # phase == 'val' (라우터는 1:Long, 2:Short, 0:Hold/Close 로 통신)
            if action == 1 and self.pos is None: is_entering_long = True
            elif action == 2 and self.pos is None: is_entering_short = True
            elif action == 0 and self.pos is not None: is_closing = True

        # 실행 및 보상 (순수 보상)
        if is_entering_long:
            self.pos = 'LONG'
            self.entry_price = current_price * (1 + self.slip)
            self.entry_idx = self.current_step
            self.peak_pnl = 0.0
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate 
            reward -= self.fee * self.current_leverage 
            self.active_steps += 1
            
        elif is_entering_short:
            self.pos = 'SHORT'
            self.entry_price = current_price * (1 - self.slip)
            self.entry_idx = self.current_step
            self.peak_pnl = 0.0
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            reward -= self.fee * self.current_leverage 
            self.active_steps += 1
            
        elif is_closing:
            if self.pos == 'LONG': realized_pnl = (current_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else: realized_pnl = (self.entry_price - current_price * (1 + self.slip)) / self.entry_price
            
            realized_pnl *= self.current_leverage 
            realized_pnl -= self.fee * self.current_leverage 
            self.balance *= (1 + realized_pnl)
            
            self.total_trades += 1
            if realized_pnl > 0: self.win_trades += 1
            
            reward += realized_pnl # 순수 realized_pnl
            
            self.pos = None
            self.current_leverage = 0.0
            is_closed = True

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self.df.loc[self.current_step, 'close'] if not done else current_price

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == 'LONG': self.unrealized_pnl = (next_price - self.entry_price) / self.entry_price * self.current_leverage
            else: self.unrealized_pnl = (self.entry_price - next_price) / self.entry_price * self.current_leverage
            
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
            self.active_steps += 1 

        info = {'pnl_pct': (self.balance / self.initial_balance - 1) * 100, 'wr': self.win_trades / max(1, self.total_trades)}
        return self._build_state(self.current_step), reward, done, info
        
    @property
    def win_rate(self): return self.win_trades / max(1, self.total_trades)

    def _build_state(self, idx):
        if idx not in self.df.index: return np.zeros(self.STATE_DIM, dtype=np.float32)
        preds = self.df.loc[idx, MODEL_PRED].values.astype(np.float32)
        confs = self.df.loc[idx, MODEL_CONF].values.astype(np.float32)
        stats = np.array([np.mean(preds), np.std(preds), np.mean(confs)], dtype=np.float32)
        elite = self.df.loc[idx, ELITE_COLS].values.astype(np.float32)
        alpha7 = self.df.loc[idx, ALPHA_7_COLS].values.astype(np.float32)
        regimes = self.df.loc[idx, REGIME_COLS].values.astype(np.float32)

        pos_features = np.array([
            1.0 if self.pos == 'LONG' else (-1.0 if self.pos == 'SHORT' else 0.0),
            self.entry_price / self.df.loc[idx, 'close'] - 1 if self.pos is not None else 0.0,
            self.unrealized_pnl, 
            self.max_drawdown, 
            self.hold_count / self.MAX_HOLD[self.phase]
        ], dtype=np.float32)

        state = np.concatenate([preds, confs, stats, elite, alpha7, regimes, pos_features])
        return np.nan_to_num(state, 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# 3. 모델 및 버퍼 (MLP + SimpleBuffer)
# ═══════════════════════════════════════════════════════════════════════════
class SimpleBuffer:
    def __init__(self, capacity=300000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self): return len(self.buffer)

class RobustIQN(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=128):
        super(RobustIQN, self).__init__()
        self.action_dim = action_dim
        
        self.feat_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        
        self.phi = nn.Linear(64, 64)
        self.fc_q = nn.Sequential(nn.SiLU(), nn.Linear(64, action_dim))
        self.pos_encoder = torch.zeros(1, state_dim, 1)

    def forward(self, state, num_quantiles=32):
        batch_size = state.size(0)
        x = self.feat_extractor(state) 
        
        tau = torch.rand(batch_size, num_quantiles, 1).to(state.device)
        pi_mtx = torch.arange(1, 65).float().to(state.device) * torch.pi
        cos_tau = torch.cos(tau * pi_mtx)
        phi_x = self.phi(cos_tau)
        
        x_tile = x.unsqueeze(1).expand(-1, num_quantiles, -1)
        q_quantiles = self.fc_q(x_tile * phi_x) 
        
        return q_quantiles, tau

class IQNAgent:
    def __init__(self, model, lr=1e-4, gamma=0.99, tau=0.005, device='cuda'):
        self.model = model
        self.state_dim = model.feat_extractor[0].in_features  # Linear layer에서 직접 추출
        self.target_model = type(model)(self.state_dim, model.action_dim).to(device)
        self.target_model.load_state_dict(model.state_dict(), strict=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.memory = None 
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def act(self, state, eps=0.0):
        if random.random() < eps: return random.randint(0, self.model.action_dim - 1)
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(state_ts)[0].mean(dim=1).squeeze(0)
        return torch.argmax(q).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        s, a, r, ns, d = self.memory.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q, tau_online = self.model(s) 
        q_a = q.gather(2, a.unsqueeze(1).expand(-1, 32, -1)).squeeze(2)

        with torch.no_grad():
            next_actions = self.model(ns)[0].mean(dim=1).argmax(dim=1, keepdim=True)
            q_target, _ = self.target_model(ns) 
            q_target_a = q_target.gather(2, next_actions.unsqueeze(1).expand(-1, 32, -1)).squeeze(2)
            target = r + self.gamma * (1 - d) * q_target_a

        td_error = target.unsqueeze(1) - q_a.unsqueeze(2)
        huber = F.huber_loss(td_error, torch.zeros_like(td_error), reduction='none', delta=1.0)
        
        tau_expanded = tau_online.transpose(1, 2)
        indicator = (td_error.detach() < 0).float()
        
        quantile_loss = torch.abs(tau_expanded - indicator) * huber
        loss = quantile_loss.mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ═══════════════════════════════════════════════════════════════════════════
# 4. 실전 메타 라우터 (SimpleRouter)
# ═══════════════════════════════════════════════════════════════════════════
class SimpleRouter:
    def __init__(self, model_long, model_short, device='cuda'):
        self.model_long = model_long.eval()
        self.model_short = model_short.eval()
        self.device = device
    
    def decide(self, features, pos):
        preds = np.array([features.get(c, 0.) for c in MODEL_PRED], dtype=np.float32)
        confs = np.array([features.get(c, 0.) for c in MODEL_CONF], dtype=np.float32)
        stats = np.array([np.mean(preds), np.std(preds), np.mean(confs)], dtype=np.float32)
        elite = np.array([features.get(c, 0.) for c in ELITE_COLS], dtype=np.float32)
        alpha7 = np.array([features.get(c, 0.) for c in ALPHA_7_COLS], dtype=np.float32)
        regimes = np.array([features.get(c, 0.) for c in REGIME_COLS], dtype=np.float32)
        
        current_price = features.get('close', 1.0) 
        current_pos = pos.get('type')
        entry_price_diff = pos.get('entry_price', current_price) / current_price - 1 if current_pos is not None else 0.0

        pos_arr = np.array([
            1.0 if current_pos == 'LONG' else (-1.0 if current_pos == 'SHORT' else 0.0),
            entry_price_diff, 
            pos.get('unrealized', 0.), 
            pos.get('mdd', 0.), 
            pos.get('hold_norm', 0.)
        ], dtype=np.float32)

        state = torch.tensor(np.concatenate([preds, confs, stats, elite, alpha7, regimes, pos_arr]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_long_dist, _ = self.model_long(state.unsqueeze(0))
            q_short_dist, _ = self.model_short(state.unsqueeze(0))

        q_long_mean = q_long_dist.squeeze(0).mean(dim=0).cpu().numpy()
        q_short_mean = q_short_dist.squeeze(0).mean(dim=0).cpu().numpy()

        q_l_hold, q_l_enter, q_l_close = q_long_mean[0], q_long_mean[1], q_long_mean[2]
        q_s_hold, q_s_enter, q_s_close = q_short_mean[0], q_short_mean[1], q_short_mean[2]

        final_action = 0 # 기본: Hold/Close
        leverage_rate = 1.0 

        if current_pos == 'LONG':
            # Long 포지션 유지 중: 청산(0) 또는 홀딩(1=유지신호)
            # val_env에서 action=0 → 청산, action=1 → pos 있을 때 아무것도 안함(홀딩)
            if q_l_close > q_l_hold:
                final_action = 0  # 청산
            else:
                final_action = 1  # 홀딩 유지 (val_env: pos있고 action=1 → 무시=홀딩)
                
        elif current_pos == 'SHORT':
            # Short 포지션 유지 중: 청산(0) 또는 홀딩(2=유지신호)
            # val_env에서 action=0 → 청산, action=2 → pos 있을 때 아무것도 안함(홀딩)
            if q_s_close > q_s_hold:
                final_action = 0  # 청산
            else:
                final_action = 2  # 홀딩 유지 (val_env: pos있고 action=2 → 무시=홀딩)
                
        else:
            # 관망 중: adv > 0이면 진입 (threshold 제거 — 초기 Q값이 작아서 0.05 넘기 불가)
            adv_long = q_l_enter - q_l_hold
            adv_short = q_s_enter - q_s_hold

            if adv_long > 0 and adv_long >= adv_short:
                final_action = 1  # Long 진입
            elif adv_short > 0 and adv_short > adv_long:
                final_action = 2  # Short 진입
            else:
                final_action = 0  # Hold

        return final_action, leverage_rate, {}

# ═══════════════════════════════════════════════════════════════════════════
# 5. 메인 훈련 루프
# ═══════════════════════════════════════════════════════════════════════════
def train():
    CSV_PATH = 'data/ensemble/rl_training_data_full.csv'
    if not os.path.exists(CSV_PATH):
        return logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")
        
    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env_long = TradingEnv(df_train, phase='train', agent_role='long_agent')
    env_short = TradingEnv(df_train, phase='train', agent_role='short_agent')

    # Action_dim = 3 (Hold, Enter, Close)
    model_long = RobustIQN(env_long.STATE_DIM, 3).to(device)
    model_short = RobustIQN(env_short.STATE_DIM, 3).to(device) 

    agent_long = IQNAgent(model_long, device=device)
    agent_short = IQNAgent(model_short, device=device)

    agent_long.memory = SimpleBuffer(capacity=300000)
    agent_short.memory = SimpleBuffer(capacity=300000)

    NEP = 1000
    BATCH = 256
    UPDATE_FREQ = 4 
    MIN_BUFFER = 2048
    global_step = 0
    # [Fix 1] Epsilon: 에폭 기준 → 글로벌 스텝 기준으로 변경
    # 에폭 기준이면 ep=13에서 eps≈0.97 (사실상 랜덤)
    # 스텝 기준이면 20000스텝(약 5에폭) 후 eps=0.1 도달
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY_STEPS = 600000
    
    os.makedirs('data/ensemble', exist_ok=True)
    best_val_pnl = -float('inf')

    logger.info("🚀 [훈련 시작] 2-Agent 독립 학습 (Long/Short) + SimpleRouter")
    for ep in range(1, NEP + 1):
        start_idx = random.randint(0, len(df_train) - env_long.MAX_EPISODE_STEPS - 1)
        s_long = env_long.reset(start_idx)
        s_short = env_short.reset(start_idx)
        
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))
        done = False

        while not done:
            global_step += 1
            
            # [Long Agent Step]
            a_long = agent_long.act(s_long, eps)
            ns_long, r_long, d_long, _ = env_long.step(a_long, leverage_rate=1.0)
            agent_long.memory.push(s_long, a_long, r_long, ns_long, d_long)
            s_long = ns_long
            
            # [Short Agent Step]
            a_short = agent_short.act(s_short, eps)
            ns_short, r_short, d_short, _ = env_short.step(a_short, leverage_rate=1.0)
            agent_short.memory.push(s_short, a_short, r_short, ns_short, d_short)
            s_short = ns_short

            done = d_long 

            if global_step % UPDATE_FREQ == 0:
                if len(agent_long.memory) >= MIN_BUFFER: agent_long.update(BATCH)
                if len(agent_short.memory) >= MIN_BUFFER: agent_short.update(BATCH)

        ep_pnl = {
            'long': (env_long.balance / 10000 - 1) * 100,
            'short': (env_short.balance / 10000 - 1) * 100
        }

        logger.info(f"Ep {ep:04d} | [Long] PnL:{ep_pnl['long']:5.1f}% | [Short] PnL:{ep_pnl['short']:5.1f}% | eps:{eps:.3f} | buf:{len(agent_long.memory)}")

        if ep % 5 == 0:
            router = SimpleRouter(model_long, model_short, device)
            val_env = TradingEnv(df_val, phase='val', agent_role='neutral')
            obs = val_env.reset()
            d = False
            
            while not d:
                feat = df_val.iloc[val_env.current_step].to_dict()
                pos_info = {
                    'type': val_env.pos, 
                    'entry_price': val_env.entry_price,
                    'unrealized': val_env.unrealized_pnl, 
                    'mdd': val_env.max_drawdown, 
                    'hold_norm': val_env.hold_count/val_env.MAX_HOLD['val']
                }
                
                action, leverage_rate, _ = router.decide(feat, pos_info)
                obs, _, d, _ = val_env.step(action, leverage_rate=leverage_rate)
                
            val_pnl_pct = (val_env.balance / 10000 - 1) * 100
            logger.info(f"    [VAL Router] PnL: {val_pnl_pct:.2f}% | Tr: {val_env.total_trades} | WR: {val_env.win_rate*100:.0f}%")

            if val_pnl_pct > best_val_pnl:
                best_val_pnl = val_pnl_pct
                save_path = 'data/ensemble/best_2agents.pth'
                
                torch.save({
                    'model_long': model_long.state_dict(),
                    'model_short': model_short.state_dict(),
                    'best_pnl': best_val_pnl,
                    'epoch': ep
                }, save_path)
                
                logger.info(f"    🎉 [NEW BEST] 2-Agent 모델 저장 완료 (PnL: {best_val_pnl:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['generate_csv', 'train'])
    args = parser.parse_args()

    INPUT_CSV = 'data/training_features_5m.csv'
    OUTPUT_CSV = 'data/ensemble/rl_training_data_full.csv'

    if args.mode == 'generate_csv':
        generate_training_csv(INPUT_CSV, OUTPUT_CSV)
    elif args.mode == 'train':
        train()