"""
Trading Router — 4-Agent MoE Transformer IQN + Kelly Router (Ultimate Fixed Version)
================================================================================
1. 7대 파운데이션 AI 모델 인퍼런스 & 5대 마켓 레짐 계산
2. 4-Agent 체제 복구 (Bull, Bear, Sup, Res) + TransformerIQN
3. Action Space 완벽 통일: 에이전트는 0(Hold/Close), 1(Enter)만 출력
4. 글로벌 스텝 기반 EPS Decay (600,000 스텝) + 순수 실현수익(Realized PnL) 보상
5. Kelly Betting 라우터: 초기 학습을 고려한 허들 완화 (0.2)
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

# 청산 에이전트 state: 시장 피처 + 포지션 정보 (진입 에이전트와 동일 구조 사용)
EXIT_STATE_DIM = STATE_DIM  # 동일한 state 사용 (pos_features에 방향/수익률 포함되어 있음)

def row_to_market_row(row: pd.Series) -> dict:
    return {k: v for k, v in row.items()}

# ═══════════════════════════════════════════════════════════════════════════
# 1. 하이브리드 배치 마이닝 엔진
# ═══════════════════════════════════════════════════════════════════════════
def generate_training_csv(input_csv: str, output_csv: str):
    print("[generate_csv] 시작")
    from strategies.elite_strategies import BaseStrategy # type: ignore
    from strategies.elite_builder import EliteSignals, row_to_market_row # type: ignore
    
    logger.info("🚀 [단계 1] 원본 피처 데이터 로드 및 전처리...")
    df = pd.read_csv(input_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    if 'close' in df.columns:
        if 'mtf_trend_1h' not in df.columns:
            df['mtf_trend_1h'] = df['close'].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)

    if 'smart_money_flow' in df.columns:
        df['smf_std'] = df['smart_money_flow'].expanding(min_periods=288).std().bfill().fillna(1.0)
    else: df['smf_std'] = 1.0

    logger.info("🚀 [단계 1.5] 적응형(Adaptive) 날씨 요정 - 레짐 스캔 중...")
    diff_abs_sum = df['close'].diff().abs().rolling(24).sum()
    net_change = df['close'] - df['close'].shift(24)
    er = (net_change.abs() / diff_abs_sum).fillna(0)
    
    raw_vol = df['close'].pct_change().rolling(24).std().fillna(0)
    vol_mean_24h = raw_vol.rolling(288).mean().bfill()
    vol_std_24h = raw_vol.rolling(288).std().bfill() + 1e-8
    vol_z = (raw_vol - vol_mean_24h) / vol_std_24h
    
    mtf_1h_trend = df['mtf_trend_1h'].fillna(0.0) 

    for col in REGIME_COLS: df[col] = 0.0

    # 1. ER 허들 대폭 낮춤: 0.35 -> 0.20 (추세가 살짝만 보여도 Bull/Bear 투입)
    bull_idx = (er >= 0.20) & (net_change > 0) & (mtf_1h_trend > 0)
    bear_idx = (er >= 0.20) & (net_change < 0) & (mtf_1h_trend < 0)

    # 2. 변동성 Z-score 허들 낮춤: ±1.0 -> ±0.5 (일반적인 횡보/휩소도 Chop/Whipsaw가 담당)
    # 단, Bull/Bear에 속하지 않은(추세가 없는) 구간 중에서만 찾음
    chop_idx = ~(bull_idx | bear_idx) & (vol_z < -0.5)
    whipsaw_idx = ~(bull_idx | bear_idx) & (vol_z > 0.5)

    # 3. 위 4개에 모두 속하지 않는 아주 애매한 구간만 Normal로 처리
    df.loc[bull_idx, 'regime_bull'] = 1.0
    df.loc[bear_idx, 'regime_bear'] = 1.0
    df.loc[chop_idx, 'regime_chop'] = 1.0
    df.loc[whipsaw_idx, 'regime_whipsaw'] = 1.0
    df.loc[~(chop_idx | whipsaw_idx | bull_idx | bear_idx), 'regime_normal'] = 1.0

    L = len(df)
    max_lookback = 256
    resume_start = max_lookback  
    abs_output = os.path.abspath(output_csv)
    
    if os.path.exists(abs_output):
        try:
            existing_df = pd.read_csv(abs_output, usecols=['timestamp'])
            if not existing_df.empty:
                last_ts = str(existing_df['timestamp'].iloc[-1])
                df['timestamp_str'] = df['timestamp'].astype(str)
                match = df[df['timestamp_str'] == last_ts]
                if not match.empty:
                    resume_start = match.index[-1] + 1
                    logger.info(f"♻️ 이어하기: {last_ts} 이후부터 재개 (인덱스: {resume_start})")
                df.drop(columns=['timestamp_str'], inplace=True)
        except Exception: pass

    if resume_start >= L: return logger.info("✅ 마이닝이 이미 완료되었습니다.")

    logger.info("🧠 [단계 2] 앙상블 모델 메모 적재...")
    from ensemble.ensemble_router import (
        TimesFMForecaster, ChronosForecaster, TTMForecaster, 
        PatchTSTForecaster, ITransformerForecaster, NHITSForecaster, TiDEForecaster
    )
    
    elite_extractor = EliteSignals()
    CHUNK_SIZE = 1024 
    
    nf_models_list = [m for m in ['patchtst', 'itransformer', 'nhits', 'tide'] if f'pred_{m}' in MODEL_PRED]
    nf_forecaster = None
    if nf_models_list:
        from neuralforecast import NeuralForecast
        nf_forecaster = NeuralForecast.load('data/nf')
        
    ttm_model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'pred_ttm' in MODEL_PRED:
        try:
            from tsfm_public import TinyTimeMixerForPrediction
            ttm_model = TinyTimeMixerForPrediction.frompretrained("ibm-granite/granite-timeseries-ttm-r1").to(device).eval()
        except Exception as e: logger.warning(f"TTM 로드 실패: {e}")

    fallback_models = {}
    if 'pred_timesfm' in MODEL_PRED: fallback_models['timesfm'] = TimesFMForecaster()
    if 'pred_chronos' in MODEL_PRED: fallback_models['chronos'] = ChronosForecaster()

    def get_direction(traj):
        if len(traj) < 2: return float(np.mean(traj))
        slope = float(np.polyfit(np.arange(len(traj)), traj, 1)[0])
        delta = traj[-1] - traj[0]
        if slope > 0 and delta > 0: return 1.0
        if slope < 0 and delta < 0: return -1.0
        return float(np.mean(traj))

    logger.info(f"🚀 [단계 3] 하이브리드 배치 마이닝 시작 (CHUNK: {CHUNK_SIZE})")
    df_records = df.to_dict('records')
    np_closes = df['close'].values
    np_timestamps = df['timestamp'].values
    alpha_matrix = df[ALPHA_7_COLS].values

    for chunk_start in range(resume_start, L, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, L)
        chunk_len = chunk_end - chunk_start
        logger.info(f"⏳ 청크 처리 중: [{chunk_start} ~ {chunk_end-1}] ({chunk_len}개)")
        chunk_data = []

        for i in range(chunk_start, chunk_end):
            current_row = df_records[i]
            prev_row = df_records[i - 1] if i > 0 else current_row
            row_res = {
                'timestamp': current_row['timestamp'], 'close': current_row['close'], 
                'log_return': current_row['log_return']
            }
            for col in ALPHA_7_COLS: row_res[col] = float(current_row.get(col, 0.0))
            for col in REGIME_COLS: row_res[col] = float(current_row.get(col, 0.0))
                
            all_sigs = elite_extractor.compute_all(current=row_to_market_row(current_row), prev=row_to_market_row(prev_row), smf_std=float(current_row.get('smf_std', 1.0)))

            row_res.update({k: float(v) for k, v in all_sigs.items() if k in ELITE_COLS})
            
            for m in MODEL_PRED: row_res[m] = 0.0
            for c in MODEL_CONF: row_res[c] = 0.5
            chunk_data.append(row_res)

        if ttm_model is not None:
            ttm_ctx = ttm_model.config.context_length
            needed_start = chunk_start - ttm_ctx + 1
            if needed_start < 0:
                closes_slice = np.pad(np_closes[0:chunk_end], (abs(needed_start), 0), 'edge')
            else: closes_slice = np_closes[needed_start:chunk_end]
                
            windows = np.lib.stride_tricks.sliding_window_view(closes_slice, ttm_ctx)
            means = windows.mean(axis=1, keepdims=True)
            stds = windows.std(axis=1, keepdims=True) + 1e-8
            scaled = (windows - means) / stds
            
            inp = torch.tensor(scaled, dtype=torch.float32).unsqueeze(-1).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
                out = ttm_model(past_values=inp).prediction_outputs.squeeze(-1).cpu().numpy()
                
            out_unscaled = out * stds + means
            for i, traj in enumerate(out_unscaled):
                chunk_data[i]['pred_ttm'] = get_direction(traj[:6])
                chunk_data[i]['conf_ttm'] = 0.6

        if nf_forecaster is not None:
            nf_rows = []
            dummy_dates = pd.date_range(end=pd.Timestamp.now(), periods=256, freq='5min')
            for i, end_idx in enumerate(range(chunk_start, chunk_end)):
                start_idx = end_idx - 255
                uid = f'w_{i}'
                y_vals = np_closes[start_idx : end_idx + 1]
                a_vals = alpha_matrix[start_idx : end_idx + 1]
                for step_idx in range(256):
                    row_dict = {'unique_id': uid, 'ds': dummy_dates[step_idx], 'y': y_vals[step_idx]}
                    for a_idx, a_name in enumerate(ALPHA_7_COLS):
                        row_dict[a_name] = a_vals[step_idx][a_idx]
                    nf_rows.append(row_dict)
                    
            batch_df = pd.DataFrame(nf_rows)
            batch_df.fillna(0.0, inplace=True)
            with SuppressOutput():
                out_df = nf_forecaster.predict(df=batch_df)
                
            for i in range(chunk_len):
                uid = f'w_{i}'
                uid_out = out_df.loc[uid] if uid in out_df.index else out_df[out_df['unique_id'] == uid]
                for m_alias in nf_models_list:
                    m_real = {'patchtst':'PatchTST', 'itransformer':'iTransformer', 'nhits':'NHITS', 'tide':'TiDE'}[m_alias]
                    if m_real in uid_out:
                        chunk_data[i][f'pred_{m_alias}'] = get_direction(uid_out[m_real].values[:6])

        if fallback_models:
            for i, end_idx in enumerate(range(chunk_start, chunk_end)):
                df_slice = df.iloc[end_idx - 255 : end_idx + 1] 
                for name, model in fallback_models.items():
                    if getattr(model, 'available', False):
                        try:
                            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                                out = model.predict(df_slice, horizon=6)
                                if out and out.median is not None:
                                    chunk_data[i][f'pred_{name}'] = get_direction(out.median[-1])
                                    chunk_data[i][f'conf_{name}'] = float(out.confidence[-1].mean())
                        except: pass

        new_df = pd.DataFrame(chunk_data, columns=RL_REQUIRED_COLS)
        is_new = not os.path.exists(abs_output)
        new_df.to_csv(abs_output, mode='a', header=is_new, index=False)
        
        del chunk_data, new_df, nf_rows
        if nf_forecaster is not None: del batch_df, out_df
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    logger.info(f"🎉 하이브리드 마이닝 완료! 파일: {output_csv}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 거래 환경 (TradingEnv) - Action Space 통일 (에이전트 0/1, 라우터 0/1/2)
# ═══════════════════════════════════════════════════════════════════════════
class TradingEnv:
    STATE_DIM = STATE_DIM

    def __init__(self, df, initial_balance=10000.0, fee=0.0005, slip=0.0002, phase='train', agent_role='bull_sniper'):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.agent_role = agent_role
        
        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1
        self.MAX_LEVERAGE = 1.0
        self.MAX_HOLD = {'train': 72, 'val': 144, 'test': 288}

        # ✅ 속도 최적화: pandas loc 제거 → numpy 배열 pre-compute
        feat_cols = MODEL_PRED + MODEL_CONF + ELITE_COLS + ALPHA_7_COLS + REGIME_COLS
        self._feat_np  = self.df[feat_cols].values.astype(np.float32)
        self._close_np = self.df['close'].values.astype(np.float32)
        self._n_pred, self._n_conf = len(MODEL_PRED), len(MODEL_CONF)
        self._n_elite, self._n_alpha = len(ELITE_COLS), len(ALPHA_7_COLS)

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
        current_price = self._close_np[self.current_step]

        # 안전망: SL(-2%)만 강제 청산 유지 (파산 방지)
        # TP/MAX_HOLD 제거 → 청산 에이전트가 직접 결정
        if self.pos is not None:
            if self.unrealized_pnl <= -0.02:
                action = 0  # SL 안전망

        reward = 0.0
        is_closed = False
        realized_pnl = 0.0

        is_entering_long = False
        is_entering_short = False
        is_closing = False

        # 💡 [핵심] 에이전트는 무조건 0(청산/유지)과 1(진입)만 씁니다.
        if self.phase == 'train':
            if action == 1 and self.pos is None:
                if self.agent_role in ['bull_sniper', 'support_buyer', 'normal_long']: is_entering_long = True
                elif self.agent_role in ['bear_sniper', 'resistance_seller', 'normal_short']: is_entering_short = True
            elif action == 0 and self.pos is not None:
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
            
            reward += realized_pnl # 순수 realized_pnl 반영
            
            self.pos = None
            self.current_leverage = 0.0
            is_closed = True

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[self.current_step] if not done else current_price

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == 'LONG': self.unrealized_pnl = (next_price - self.entry_price) / self.entry_price * self.current_leverage
            else: self.unrealized_pnl = (self.entry_price - next_price) / self.entry_price * self.current_leverage
            
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
            self.active_steps += 1
            # ✅ 청산 에이전트 shaped reward: 홀딩 중 방향 신호 (진입 에이전트는 이 reward 안 받음)
            # → 손실 중엔 강하게 청산 유도, 수익 중엔 약하게 홀딩 장려 (비대칭)
            if self.phase == 'train':
                if self.unrealized_pnl > 0:
                    reward += self.unrealized_pnl * 0.15  # 수익 중: 홀딩 약하게 장려
                else:
                    reward += self.unrealized_pnl * 0.30  # 손실 중: 빠른 청산 강하게 학습
                # 좀비 포지션 방지: 100 스텝 초과 홀딩 시 패널티
                if self.hold_count > 100:
                    reward -= 0.001 * (self.hold_count - 100)

        info = {'pnl_pct': (self.balance / self.initial_balance - 1) * 100, 'wr': self.win_trades / max(1, self.total_trades)}
        return self._build_state(self.current_step), reward, done, info

    @property
    def win_rate(self): return self.win_trades / max(1, self.total_trades)

    def _build_state(self, idx):
        if idx < 0 or idx >= len(self._feat_np):
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        # ✅ numpy 직접 접근 (pandas loc 대비 ~10x 빠름)
        row = self._feat_np[idx]
        o = 0
        preds  = row[o:o+self._n_pred];  o += self._n_pred
        confs  = row[o:o+self._n_conf];  o += self._n_conf
        stats  = np.array([preds.mean(), preds.std(), confs.mean()], dtype=np.float32)
        elite  = row[o:o+self._n_elite]; o += self._n_elite
        alpha7 = row[o:o+self._n_alpha]; o += self._n_alpha
        regimes= row[o:]

        close = self._close_np[idx]
        pos_features = np.array([
            1.0 if self.pos == 'LONG' else (-1.0 if self.pos == 'SHORT' else 0.0),
            self.entry_price / close - 1 if self.pos is not None else 0.0,
            self.unrealized_pnl,
            self.max_drawdown,
            self.hold_count / self.MAX_HOLD[self.phase]
        ], dtype=np.float32)

        return np.nan_to_num(np.concatenate([preds, confs, stats, elite, alpha7, regimes, pos_features]), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2-2. 청산 전용 환경 (ExitEnv)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeReplayBuffer:
    def __init__(self, capacity=300000, target_regimes=None, warmup_steps=8000):
        self.buffer = deque(maxlen=capacity)
        self.target_regimes = target_regimes or []
        self.warmup_steps = warmup_steps  # warmup 동안은 regime 필터 없이 전부 저장
        self._push_count = 0

    def push(self, state, action, reward, next_state, done, current_regimes_dict):
        self._push_count += 1
        if self._push_count < self.warmup_steps:
            # warmup 구간: regime 필터 없이 전부 저장 → buffer 빠르게 채움
            self.buffer.append((state, action, reward, next_state, done))
            return
        # warmup 이후: 주특기 레짐 100%, 기타 10%
        is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
        if is_target or random.random() < 0.1:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self): return len(self.buffer)


class PrioritizedRegimeReplayBuffer(RegimeReplayBuffer):
    """TD-error 기반 우선순위 샘플링으로 어려운 샘플에 집중"""
    def __init__(self, capacity=300000, target_regimes=None, warmup_steps=8000,
                 alpha=0.6, beta=0.4):
        super().__init__(capacity, target_regimes, warmup_steps)
        self.priorities   = deque(maxlen=capacity)
        self.alpha        = alpha
        self.beta         = beta
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, current_regimes_dict):
        prev_len = len(self.buffer)
        super().push(state, action, reward, next_state, done, current_regimes_dict)
        if len(self.buffer) > prev_len:  # 실제로 추가된 경우에만 우선순위 추가
            self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        pri   = np.array(list(self.priorities), dtype=np.float32) ** self.alpha
        probs = pri / (pri.sum() + 1e-8)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (1.0 / (len(self.buffer) * probs[indices] + 1e-8)) ** self.beta
        weights = (weights / weights.max()).astype(np.float32)
        batch   = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d), indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = float(abs(err) + 1e-6) ** self.alpha
            self.priorities[idx] = p
            self.max_priority    = max(self.max_priority, p)

class RobustIQN(nn.Module):
    """Dueling IQN — V(s) + A(s,a) 분리로 학습 안정성 향상"""
    def __init__(self, state_dim, action_dim=2, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.feat_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 64),        nn.LayerNorm(64),         nn.SiLU()
        )
        self.phi        = nn.Linear(64, 64)
        # Dueling: Value stream (상태 가치)
        self.value_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        # Dueling: Advantage stream (행동 이점)
        self.adv_head   = nn.Sequential(nn.SiLU(), nn.Linear(64, action_dim))

    def forward(self, state, num_quantiles=8):
        batch_size = state.size(0)
        feat    = self.feat_extractor(state)
        tau     = torch.rand(batch_size, num_quantiles, 1, device=state.device)
        cos_tau = torch.cos(tau * torch.arange(1, 65, device=state.device).float() * torch.pi)
        phi_x   = self.phi(cos_tau)
        shared  = feat.unsqueeze(1).expand(-1, num_quantiles, -1) * phi_x  # (B, NQ, 64)
        # Dueling combination: Q = V + (A - mean(A))
        v = self.value_head(shared)                          # (B, NQ, 1)
        a = self.adv_head(shared)                            # (B, NQ, action_dim)
        q = v + (a - a.mean(dim=2, keepdim=True))            # (B, NQ, action_dim)
        return q, tau

class TransformerIQN(nn.Module):
    def __init__(self, state_dim, action_dim=2, d_model=64, nhead=4, num_layers=1):
        super(TransformerIQN, self).__init__()
        self.action_dim = action_dim
        
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, state_dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.phi = nn.Linear(64, d_model)
        self.fc_q = nn.Sequential(nn.SiLU(), nn.Linear(d_model, action_dim))

    def forward(self, state, num_quantiles=32):
        batch_size = state.size(0)
        
        x = state.unsqueeze(-1)
        x = self.feature_embed(x) 
        x = x + self.pos_encoder 
        x = self.transformer(x)
        x = x.mean(dim=1) 
        
        tau = torch.rand(batch_size, num_quantiles, 1).to(state.device)
        pi_mtx = torch.arange(1, 65).float().to(state.device) * torch.pi
        cos_tau = torch.cos(tau * pi_mtx)
        phi_x = self.phi(cos_tau)
        
        x_tile = x.unsqueeze(1).expand(-1, num_quantiles, -1)
        q_quantiles = self.fc_q(x_tile * phi_x) 
        
        return q_quantiles, tau

class IQNAgent:
    NUM_QUANTILES = 8  # ✅ 32 → 8: ~3x 속도 향상, 학습 품질 유지

    def __init__(self, model, lr=5e-5, gamma=0.99, tau=0.005, device='cuda'):
        self.model = model
        # RobustIQN / TransformerIQN 모두 호환
        if hasattr(model, 'feat_extractor'):
            self.state_dim = model.feat_extractor[0].in_features
        else:
            self.state_dim = model.pos_encoder.shape[1]
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
            q = self.model(state_ts, num_quantiles=self.NUM_QUANTILES)[0].mean(dim=1).squeeze(0)
        return torch.argmax(q).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        is_per = isinstance(self.memory, PrioritizedRegimeReplayBuffer)
        if is_per:
            s, a, r, ns, d, per_indices, per_weights = self.memory.sample(batch_size)
            per_w = torch.FloatTensor(per_weights).to(self.device)
        else:
            s, a, r, ns, d = self.memory.sample(batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        NQ = self.NUM_QUANTILES
        q, tau_online = self.model(s, num_quantiles=NQ)
        q_a = q.gather(2, a.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)

        with torch.no_grad():
            next_actions = self.model(ns, num_quantiles=NQ)[0].mean(dim=1).argmax(dim=1, keepdim=True)
            q_target, _  = self.target_model(ns, num_quantiles=NQ)
            q_target_a   = q_target.gather(2, next_actions.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)
            target = r + self.gamma * (1 - d) * q_target_a

        td_error  = target.unsqueeze(1) - q_a.unsqueeze(2)
        huber     = F.huber_loss(td_error, torch.zeros_like(td_error), reduction='none', delta=1.0)
        tau_exp   = tau_online.transpose(1, 2)
        indicator = (td_error.detach() < 0).float()
        loss_per_sample = (torch.abs(tau_exp - indicator) * huber).mean(dim=1).mean(dim=1)  # (B,)

        if is_per:
            loss = (loss_per_sample * per_w).mean()
            # TD-error로 우선순위 업데이트
            td_err_np = td_error.detach().abs().mean(dim=(1, 2)).cpu().numpy()
            self.memory.update_priorities(per_indices, td_err_np)
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ═══════════════════════════════════════════════════════════════════════════
# 4. 실전 메타 라우터 (MoEIQNTrader) — 8-Agent 페어 구조
#    진입: Bull / Bear / Sup / Res
#    청산: Bull_Exit / Bear_Exit / Sup_Exit / Res_Exit
# ═══════════════════════════════════════════════════════════════════════════
class MoEIQNTrader:
    def __init__(self,
                 model_bull, model_bear, model_sup, model_res,
                 model_bull_exit, model_bear_exit, model_sup_exit, model_res_exit,
                 model_normal_long, model_normal_short,
                 model_normal_long_exit, model_normal_short_exit,
                 df, device='cuda'):
        self.model_bull              = model_bull.eval()
        self.model_bear              = model_bear.eval()
        self.model_sup               = model_sup.eval()
        self.model_res               = model_res.eval()
        self.model_bull_exit         = model_bull_exit.eval()
        self.model_bear_exit         = model_bear_exit.eval()
        self.model_sup_exit          = model_sup_exit.eval()
        self.model_res_exit          = model_res_exit.eval()
        self.model_normal_long       = model_normal_long.eval()
        self.model_normal_short      = model_normal_short.eval()
        self.model_normal_long_exit  = model_normal_long_exit.eval()
        self.model_normal_short_exit = model_normal_short_exit.eval()
        self.df                      = df
        self.device                  = device
        self._active_pair            = None  # 진입시킨 에이전트 이름 기억

    def _state_tensor(self, features, pos):
        preds   = np.array([features.get(c, 0.) for c in MODEL_PRED],   dtype=np.float32)
        confs   = np.array([features.get(c, 0.) for c in MODEL_CONF],   dtype=np.float32)
        stats   = np.array([preds.mean(), preds.std(), confs.mean()],    dtype=np.float32)
        elite   = np.array([features.get(c, 0.) for c in ELITE_COLS],   dtype=np.float32)
        alpha7  = np.array([features.get(c, 0.) for c in ALPHA_7_COLS], dtype=np.float32)
        regimes = np.array([features.get(c, 0.) for c in REGIME_COLS],  dtype=np.float32)
        cur_p   = features.get('close', 1.0)
        pt      = pos.get('type')
        pos_arr = np.array([
            1.0 if pt == 'LONG' else (-1.0 if pt == 'SHORT' else 0.0),
            pos.get('entry_price', cur_p) / cur_p - 1 if pt else 0.0,
            pos.get('unrealized', 0.),
            pos.get('mdd', 0.),
            pos.get('hold_norm', 0.)
        ], dtype=np.float32)
        vec = np.concatenate([preds, confs, stats, elite, alpha7, regimes, pos_arr])
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, current_idx, features, pos):
        cur_pos = pos.get('type')
        state   = self._state_tensor(features, pos)

        with torch.no_grad():
            # 항상 12개 모델 모두 추론 (진입+청산 판단에 재활용)
            q_bull         = self.model_bull(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_bear         = self.model_bear(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_sup          = self.model_sup(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_res          = self.model_res(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_normal_long  = self.model_normal_long(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            q_normal_short = self.model_normal_short(state)[0].mean(dim=1).squeeze(0).cpu().numpy()

            if cur_pos is not None and self._active_pair is not None:
                exit_model = {
                    'bull':         self.model_bull_exit,
                    'bear':         self.model_bear_exit,
                    'sup':          self.model_sup_exit,
                    'res':          self.model_res_exit,
                    'normal_long':  self.model_normal_long_exit,
                    'normal_short': self.model_normal_short_exit,
                }[self._active_pair]
                q_exit = exit_model(state)[0].mean(dim=1).squeeze(0).cpu().numpy()
            else:
                q_exit = None

        adv_bull         = q_bull[1]         - q_bull[0]
        adv_bear         = q_bear[1]         - q_bear[0]
        adv_sup          = q_sup[1]          - q_sup[0]
        adv_res          = q_res[1]          - q_res[0]
        adv_normal_long  = q_normal_long[1]  - q_normal_long[0]
        adv_normal_short = q_normal_short[1] - q_normal_short[0]

        kelly_bull         = max(0., adv_bull         / (q_bull.std()         + 0.05))
        kelly_bear         = max(0., adv_bear         / (q_bear.std()         + 0.05))
        kelly_sup          = max(0., adv_sup          / (q_sup.std()          + 0.05))
        kelly_res          = max(0., adv_res          / (q_res.std()          + 0.05))
        kelly_normal_long  = max(0., adv_normal_long  / (q_normal_long.std()  + 0.05))
        kelly_normal_short = max(0., adv_normal_short / (q_normal_short.std() + 0.05))

        CLOSE_KELLY_THRESHOLD = 0.3  # 반대 에이전트 강도 임계값

        # ── 포지션 있음: 청산 판단 ────────────────────────────────────
        if cur_pos is not None and self._active_pair is not None:

            # 신호 1: Exit 에이전트가 청산 권고
            exit_signal = (q_exit is not None) and (q_exit[1] > q_exit[0])

            # 신호 2: 반대 진입 에이전트가 강하게 반대 방향 신호
            if cur_pos == 'LONG':
                opp_signal = (adv_bear > 0 and kelly_bear > CLOSE_KELLY_THRESHOLD) or \
                             (adv_res  > 0 and kelly_res  > CLOSE_KELLY_THRESHOLD) or \
                             (self._active_pair == 'normal_long' and
                              adv_normal_short > 0 and kelly_normal_short > CLOSE_KELLY_THRESHOLD)
            else:  # SHORT
                opp_signal = (adv_bull > 0 and kelly_bull > CLOSE_KELLY_THRESHOLD) or \
                             (adv_sup  > 0 and kelly_sup  > CLOSE_KELLY_THRESHOLD) or \
                             (self._active_pair == 'normal_short' and
                              adv_normal_long > 0 and kelly_normal_long > CLOSE_KELLY_THRESHOLD)

            if exit_signal or opp_signal:
                active = self._active_pair
                self._active_pair = None
                return 0, 0.0, {'agent': f'{active}_exit+opp' if opp_signal else f'{active}_exit'}
            else:
                hold_action = 1 if cur_pos == 'LONG' else 2
                return hold_action, 0.0, {'agent': 'HOLD'}

        # ── 포지션 없음: 레짐별 진입 에이전트 선택 ──────────────────
        is_chop = features.get('regime_chop', 0.) == 1. or features.get('regime_whipsaw', 0.) == 1.
        is_bull = features.get('regime_bull', 0.) == 1.
        is_bear = features.get('regime_bear', 0.) == 1.

        final_action, active_agent, selected_kelly = 0, "NONE", 0.0

        if is_chop:
            if adv_sup > 0 and adv_sup >= adv_res:
                final_action, active_agent, selected_kelly, self._active_pair = 1, "SUP_BUY",      kelly_sup,          'sup'
            elif adv_res > 0:
                final_action, active_agent, selected_kelly, self._active_pair = 2, "RES_SELL",     kelly_res,          'res'
        elif is_bull:
            if adv_bull > 0:
                final_action, active_agent, selected_kelly, self._active_pair = 1, "BULL_SNIPE",   kelly_bull,         'bull'
        elif is_bear:
            if adv_bear > 0:
                final_action, active_agent, selected_kelly, self._active_pair = 2, "BEAR_SNIPE",   kelly_bear,         'bear'
        else:  # regime_normal: 전담 에이전트 사용
            if adv_normal_long > 0 and adv_normal_long >= adv_normal_short:
                final_action, active_agent, selected_kelly, self._active_pair = 1, "NORMAL_LONG",  kelly_normal_long,  'normal_long'
            elif adv_normal_short > 0:
                final_action, active_agent, selected_kelly, self._active_pair = 2, "NORMAL_SHORT", kelly_normal_short, 'normal_short'

        if final_action == 0:
            self._active_pair = None

        leverage_rate = np.clip(selected_kelly * 0.5, 0.1, 1.0) if final_action != 0 else 0.0
        return final_action, leverage_rate, {'agent': active_agent, 'kelly': selected_kelly}

# ═══════════════════════════════════════════════════════════════════════════
# 5. 메인 훈련 루프
# ═══════════════════════════════════════════════════════════════════════════
def train():
    CSV_PATH = 'data/ensemble/rl_training_data_full.csv'
    if not os.path.exists(CSV_PATH):
        return logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")

    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx].reset_index(drop=True)
    df_val    = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_train_reg = df_train[REGIME_COLS].values.astype(np.float32)

    # ── 레짐별 시작 인덱스 풀 ────────────────────────────────────────────
    MAX_EP    = 4096
    _safe_end = len(df_train) - MAX_EP - 1
    regime_starts = {
        'bull':   [i for i in range(_safe_end) if df_train_reg[i, REGIME_COLS.index('regime_bull')]     == 1.0],
        'bear':   [i for i in range(_safe_end) if df_train_reg[i, REGIME_COLS.index('regime_bear')]     == 1.0],
        'chop':   [i for i in range(_safe_end) if df_train_reg[i, REGIME_COLS.index('regime_chop')]     == 1.0
                                               or df_train_reg[i, REGIME_COLS.index('regime_whipsaw')] == 1.0],
        'normal': [i for i in range(_safe_end) if df_train_reg[i, REGIME_COLS.index('regime_normal')]   == 1.0],
        'any':    list(range(_safe_end)),
    }
    for k in ['bull', 'bear', 'chop', 'normal']:
        if len(regime_starts[k]) < 100:
            logger.warning(f"⚠️ {k} 레짐 시작점 {len(regime_starts[k])}개 부족 → 전체 풀 사용")
            regime_starts[k] = regime_starts['any']
    logger.info(f"레짐 시작점 풀: bull={len(regime_starts['bull'])} bear={len(regime_starts['bear'])} chop={len(regime_starts['chop'])} normal={len(regime_starts['normal'])}")

    # ── 환경: 페어당 TradingEnv 하나 ────────────────────────────────────
    # 진입+청산이 같은 환경에서 순서대로 일어남 → PnL이 통합
    env_bull         = TradingEnv(df_train, phase='train', agent_role='bull_sniper')
    env_bear         = TradingEnv(df_train, phase='train', agent_role='bear_sniper')
    env_sup          = TradingEnv(df_train, phase='train', agent_role='support_buyer')
    env_res          = TradingEnv(df_train, phase='train', agent_role='resistance_seller')
    env_normal_long  = TradingEnv(df_train, phase='train', agent_role='normal_long')
    env_normal_short = TradingEnv(df_train, phase='train', agent_role='normal_short')

    # ── 모델: 진입×6 + 청산×6 ────────────────────────────────────────────
    model_bull              = RobustIQN(STATE_DIM, 2).to(device)  # 진입: 0=홀드, 1=진입
    model_bear              = RobustIQN(STATE_DIM, 2).to(device)
    model_sup               = RobustIQN(STATE_DIM, 2).to(device)
    model_res               = RobustIQN(STATE_DIM, 2).to(device)
    model_normal_long       = RobustIQN(STATE_DIM, 2).to(device)
    model_normal_short      = RobustIQN(STATE_DIM, 2).to(device)
    model_bull_exit         = RobustIQN(STATE_DIM, 2).to(device)  # 청산: 0=홀드, 1=청산
    model_bear_exit         = RobustIQN(STATE_DIM, 2).to(device)
    model_sup_exit          = RobustIQN(STATE_DIM, 2).to(device)
    model_res_exit          = RobustIQN(STATE_DIM, 2).to(device)
    model_normal_long_exit  = RobustIQN(STATE_DIM, 2).to(device)
    model_normal_short_exit = RobustIQN(STATE_DIM, 2).to(device)

    # ── 에이전트 ─────────────────────────────────────────────────────────
    agent_bull              = IQNAgent(model_bull,              device=device)
    agent_bear              = IQNAgent(model_bear,              device=device)
    agent_sup               = IQNAgent(model_sup,               device=device)
    agent_res               = IQNAgent(model_res,               device=device)
    agent_normal_long       = IQNAgent(model_normal_long,       device=device)
    agent_normal_short      = IQNAgent(model_normal_short,      device=device)
    agent_bull_exit         = IQNAgent(model_bull_exit,         device=device)
    agent_bear_exit         = IQNAgent(model_bear_exit,         device=device)
    agent_sup_exit          = IQNAgent(model_sup_exit,          device=device)
    agent_res_exit          = IQNAgent(model_res_exit,          device=device)
    agent_normal_long_exit  = IQNAgent(model_normal_long_exit,  device=device)
    agent_normal_short_exit = IQNAgent(model_normal_short_exit, device=device)

    # 진입 에이전트: 우선순위 레짐 필터 버퍼 (PER)
    agent_bull.memory              = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_bull'])
    agent_bear.memory              = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_bear'])
    agent_sup.memory               = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_chop', 'regime_whipsaw'])
    agent_res.memory               = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_chop', 'regime_whipsaw'])
    agent_normal_long.memory       = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_normal'])
    agent_normal_short.memory      = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_normal'])
    # 청산 에이전트: 우선순위 레짐 필터 버퍼 (PER)
    agent_bull_exit.memory         = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_bull'])
    agent_bear_exit.memory         = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_bear'])
    agent_sup_exit.memory          = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_chop', 'regime_whipsaw'])
    agent_res_exit.memory          = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_chop', 'regime_whipsaw'])
    agent_normal_long_exit.memory  = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_normal'])
    agent_normal_short_exit.memory = PrioritizedRegimeReplayBuffer(300000, target_regimes=['regime_normal'])

    # 페어 묶음: (env, 진입agent, 청산agent, 시작풀)
    pairs = [
        (env_bull,         agent_bull,         agent_bull_exit,         'bull'),
        (env_bear,         agent_bear,         agent_bear_exit,         'bear'),
        (env_sup,          agent_sup,          agent_sup_exit,          'chop'),
        (env_res,          agent_res,          agent_res_exit,          'chop'),
        (env_normal_long,  agent_normal_long,  agent_normal_long_exit,  'normal'),
        (env_normal_short, agent_normal_short, agent_normal_short_exit, 'normal'),
    ]

    NEP             = 1000
    BATCH           = 512
    UPDATE_FREQ     = 16
    MIN_BUFFER      = 2048
    global_step     = 0
    EPS_START       = 1.0
    EPS_END         = 0.10
    EPS_DECAY_STEPS = 1500000  # 충분한 탐색 (20000은 너무 짧음)

    os.makedirs('data/ensemble', exist_ok=True)
    best_val_pnl   = -float('inf')
    best_val_score = -float('inf')
    val_pnl_history: list = []  # 최근 val PnL 기록 (Sharpe 추정용)
    start_ep       = 1
    CHECKPOINT_PATH = 'data/ensemble/train_checkpoint.pth'

    # ── 체크포인트 저장 헬퍼 ───────────────────────────────────────────────
    def _save_checkpoint(epoch):
        torch.save({
            'model_bull':                  model_bull.state_dict(),
            'model_bear':                  model_bear.state_dict(),
            'model_sup':                   model_sup.state_dict(),
            'model_res':                   model_res.state_dict(),
            'model_normal_long':           model_normal_long.state_dict(),
            'model_normal_short':          model_normal_short.state_dict(),
            'model_bull_exit':             model_bull_exit.state_dict(),
            'model_bear_exit':             model_bear_exit.state_dict(),
            'model_sup_exit':              model_sup_exit.state_dict(),
            'model_res_exit':              model_res_exit.state_dict(),
            'model_normal_long_exit':      model_normal_long_exit.state_dict(),
            'model_normal_short_exit':     model_normal_short_exit.state_dict(),
            'opt_bull':                    agent_bull.optimizer.state_dict(),
            'opt_bear':                    agent_bear.optimizer.state_dict(),
            'opt_sup':                     agent_sup.optimizer.state_dict(),
            'opt_res':                     agent_res.optimizer.state_dict(),
            'opt_normal_long':             agent_normal_long.optimizer.state_dict(),
            'opt_normal_short':            agent_normal_short.optimizer.state_dict(),
            'opt_bull_exit':               agent_bull_exit.optimizer.state_dict(),
            'opt_bear_exit':               agent_bear_exit.optimizer.state_dict(),
            'opt_sup_exit':                agent_sup_exit.optimizer.state_dict(),
            'opt_res_exit':                agent_res_exit.optimizer.state_dict(),
            'opt_normal_long_exit':        agent_normal_long_exit.optimizer.state_dict(),
            'opt_normal_short_exit':       agent_normal_short_exit.optimizer.state_dict(),
            'global_step':                 global_step,
            'best_val_pnl':               best_val_pnl,
            'best_val_score':             best_val_score,
            'val_pnl_history':            val_pnl_history,
            'epoch':                       epoch,
        }, CHECKPOINT_PATH)

    # ── 이전 체크포인트 자동 복원 ─────────────────────────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model_bull.load_state_dict(ckpt['model_bull'])
        model_bear.load_state_dict(ckpt['model_bear'])
        model_sup.load_state_dict(ckpt['model_sup'])
        model_res.load_state_dict(ckpt['model_res'])
        model_normal_long.load_state_dict(ckpt['model_normal_long'])
        model_normal_short.load_state_dict(ckpt['model_normal_short'])
        model_bull_exit.load_state_dict(ckpt['model_bull_exit'])
        model_bear_exit.load_state_dict(ckpt['model_bear_exit'])
        model_sup_exit.load_state_dict(ckpt['model_sup_exit'])
        model_res_exit.load_state_dict(ckpt['model_res_exit'])
        model_normal_long_exit.load_state_dict(ckpt['model_normal_long_exit'])
        model_normal_short_exit.load_state_dict(ckpt['model_normal_short_exit'])
        # target network 동기화
        for _ag in [agent_bull, agent_bear, agent_sup, agent_res,
                    agent_normal_long, agent_normal_short,
                    agent_bull_exit, agent_bear_exit, agent_sup_exit, agent_res_exit,
                    agent_normal_long_exit, agent_normal_short_exit]:
            _ag.target_model.load_state_dict(_ag.model.state_dict())
        # optimizer 복원
        agent_bull.optimizer.load_state_dict(ckpt['opt_bull'])
        agent_bear.optimizer.load_state_dict(ckpt['opt_bear'])
        agent_sup.optimizer.load_state_dict(ckpt['opt_sup'])
        agent_res.optimizer.load_state_dict(ckpt['opt_res'])
        agent_normal_long.optimizer.load_state_dict(ckpt['opt_normal_long'])
        agent_normal_short.optimizer.load_state_dict(ckpt['opt_normal_short'])
        agent_bull_exit.optimizer.load_state_dict(ckpt['opt_bull_exit'])
        agent_bear_exit.optimizer.load_state_dict(ckpt['opt_bear_exit'])
        agent_sup_exit.optimizer.load_state_dict(ckpt['opt_sup_exit'])
        agent_res_exit.optimizer.load_state_dict(ckpt['opt_res_exit'])
        agent_normal_long_exit.optimizer.load_state_dict(ckpt['opt_normal_long_exit'])
        agent_normal_short_exit.optimizer.load_state_dict(ckpt['opt_normal_short_exit'])
        # 훈련 상태 복원
        global_step     = ckpt['global_step']
        best_val_pnl    = ckpt['best_val_pnl']
        best_val_score  = ckpt['best_val_score']
        val_pnl_history = ckpt.get('val_pnl_history', [])
        start_ep        = ckpt['epoch'] + 1
        logger.info(f"♻️  [체크포인트 복원] ep={ckpt['epoch']} → {start_ep}부터 재시작 | global_step={global_step} | best_pnl={best_val_pnl:.2f}%")
    else:
        logger.info("🚀 [훈련 시작] 새 학습 — 체크포인트 없음")

    try:
        for ep in range(start_ep, NEP + 1):

            def pick_start(pool):
                if random.random() < 0.7: return random.choice(pool)
                return random.choice(regime_starts['any'])

            # 각 페어 리셋 (하나의 env)
            pair_states = []
            # Delayed reward 추적: 진입 시점 (state, action, regimes) 저장
            # 청산 시 realized_pnl을 진입 에이전트에게 나중에 전달
            pair_entry_cache = [None] * len(pairs)  # (entry_s, entry_a, entry_regimes)
            for env, agent_e, agent_x, pool_key in pairs:
                s = env.reset(pick_start(regime_starts[pool_key]))
                pair_states.append(s)

            eps  = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))
            done = False

            while not done:
                global_step += 1
                eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

                idx = pairs[0][0].current_step
                if idx >= pairs[0][0].end_step or idx >= len(df_train) - 1:
                    break

                for i, (env, agent_e, agent_x, _) in enumerate(pairs):
                    s = pair_states[i]
                    # [BUG-2 FIX] 각 env의 실제 current_step으로 레짐 계산
                    env_idx = env.current_step
                    current_regimes = {r: float(df_train_reg[env_idx, ri]) for ri, r in enumerate(REGIME_COLS)}

                    if env.pos is None:
                        # ── 포지션 없음: 진입 에이전트 act ───────────────
                        a = agent_e.act(s, eps)
                        ns, r, d, _ = env.step(a)

                        if a == 1 and env.pos is not None:
                            # 진입 성공 → 결과를 모르니 아직 메모리 저장 안 함
                            # entry_cache에 (state, action, regimes, entry_r) 보관
                            # [BUG-1 FIX] 진입 수수료 r도 함께 저장
                            pair_entry_cache[i] = (s, a, current_regimes, r)
                            # 진입 에이전트는 이 스텝 reward 없음 (청산 시 delayed로 받음)
                        else:
                            # 홀드 → 즉시 저장 (reward=0)
                            agent_e.memory.push(s, a, r, ns, d, current_regimes)

                    else:
                        # ── 포지션 있음: 청산 에이전트 act ──────────────
                        # agent_x: 0=홀드(shaped reward), 1=청산(realized_pnl)
                        # env.step: 0=청산, 1=진입불가(홀딩처리)
                        a = agent_x.act(s, eps)
                        # [BUG-2 FIX] SL(-2%) 강제 청산 시 메모리 오염 방지
                        # exit agent가 a=0(홀드) 선택했더라도 SL이 발동하면
                        # step() 내부에서 action=0(청산)으로 덮어씀 → 메모리에는 close로 기록해야 함
                        if env.pos is not None and env.unrealized_pnl <= -0.02:
                            a_for_mem = 1   # SL 강제 청산 → close(1)로 기록
                            env_action = 0
                        else:
                            a_for_mem = a
                            env_action = 0 if a == 1 else 1
                        ns, r, d, _ = env.step(env_action)
                        agent_x.memory.push(s, a_for_mem, r, ns, d, current_regimes)

                        # 청산됐으면 진입 에이전트에게 delayed reward 전달
                        if env.pos is None and pair_entry_cache[i] is not None:
                            entry_s, entry_a, entry_reg, entry_r = pair_entry_cache[i]
                            # [BUG-1 FIX] total_r = 진입 수수료 + 청산 reward (왕복 수수료 완전 반영)
                            # → 진입 에이전트: "그 타이밍에 진입한 결과가 이거였다"
                            agent_e.memory.push(entry_s, entry_a, entry_r + r, ns, d, entry_reg)
                            pair_entry_cache[i] = None

                    pair_states[i] = ns
                    done = done or d

                if global_step % UPDATE_FREQ == 0:
                    for _, agent_e, agent_x, _ in pairs:
                        if len(agent_e.memory) >= MIN_BUFFER: agent_e.update(BATCH)
                        if len(agent_x.memory) >= MIN_BUFFER: agent_x.update(BATCH)

            # ── 에피소드 종료 시 미청산 포지션 처리 ─────────────────────
            # 에피소드가 끝났는데 아직 entry_cache에 남아있으면
            # unrealized_pnl을 delayed reward로 줘서 학습 신호 제공
            for i, (env, agent_e, agent_x, _) in enumerate(pairs):
                if pair_entry_cache[i] is not None:
                    entry_s, entry_a, entry_reg, entry_r = pair_entry_cache[i]
                    final_reward = entry_r + env.unrealized_pnl  # 진입 수수료 + 미실현 수익
                    final_ns     = pair_states[i]
                    agent_e.memory.push(entry_s, entry_a, final_reward, final_ns, True, entry_reg)
                    pair_entry_cache[i] = None

            # ── 에폭 로그 ─────────────────────────────────────────────────
            names = ['Bull', 'Bear', 'Sup ', 'Res ', 'NorL', 'NorS']
            for name, (env, agent_e, agent_x, _) in zip(names, pairs):
                pnl = (env.balance / 10000 - 1) * 100
                logger.info(
                    f"Ep {ep:04d} [{name}] "
                    f"PnL:{pnl:6.1f}% Tr:{env.total_trades:4d} WR:{env.win_rate*100:4.0f}% | "
                    f"buf_e:{len(agent_e.memory):6d} buf_x:{len(agent_x.memory):6d} | eps:{eps:.3f}"
                )

            # ── Val 평가 (10에폭마다) ─────────────────────────────────────
            if ep % 10 == 0:
                router = MoEIQNTrader(
                    model_bull, model_bear, model_sup, model_res,
                    model_bull_exit, model_bear_exit, model_sup_exit, model_res_exit,
                    model_normal_long, model_normal_short,
                    model_normal_long_exit, model_normal_short_exit,
                    df_val, device
                )
                val_env = TradingEnv(df_val, phase='val', agent_role='neutral')
                obs = val_env.reset()
                d   = False

                while not d:
                    feat = df_val.iloc[val_env.current_step].to_dict()
                    pos_info = {
                        'type':        val_env.pos,
                        'entry_price': val_env.entry_price,
                        'unrealized':  val_env.unrealized_pnl,
                        'mdd':         val_env.max_drawdown,
                        'hold_norm':   val_env.hold_count / val_env.MAX_HOLD['val']
                    }
                    action, leverage_rate, info = router.decide(val_env.current_step, feat, pos_info)
                    obs, _, d, _ = val_env.step(action, leverage_rate=leverage_rate)

                val_pnl_pct = (val_env.balance / 10000 - 1) * 100
                val_pnl_history.append(val_pnl_pct)
                # Sharpe 추정: 최근 10개 val PnL 기록 기반
                if len(val_pnl_history) >= 3:
                    _arr   = np.array(val_pnl_history[-10:])
                    sharpe_est = float(np.mean(_arr) / (np.std(_arr) + 1e-6))
                else:
                    sharpe_est = 0.0
                # 복합 스코어: PnL 40% + 승률 30% + Sharpe 30%
                val_score = val_pnl_pct * 0.4 + val_env.win_rate * 30 + sharpe_est * 10
                logger.info(
                    f"    [VAL] PnL:{val_pnl_pct:.2f}% | Tr:{val_env.total_trades} | "
                    f"WR:{val_env.win_rate*100:.0f}% | Sharpe:{sharpe_est:.2f} | Score:{val_score:.2f} | eps:{eps:.3f}"
                )

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_pnl   = val_pnl_pct
                    torch.save({
                        'model_bull':              model_bull.state_dict(),
                        'model_bear':              model_bear.state_dict(),
                        'model_sup':               model_sup.state_dict(),
                        'model_res':               model_res.state_dict(),
                        'model_normal_long':       model_normal_long.state_dict(),
                        'model_normal_short':      model_normal_short.state_dict(),
                        'model_bull_exit':         model_bull_exit.state_dict(),
                        'model_bear_exit':         model_bear_exit.state_dict(),
                        'model_sup_exit':          model_sup_exit.state_dict(),
                        'model_res_exit':          model_res_exit.state_dict(),
                        'model_normal_long_exit':  model_normal_long_exit.state_dict(),
                        'model_normal_short_exit': model_normal_short_exit.state_dict(),
                        'best_pnl':                best_val_pnl,
                        'epoch':                   ep
                    }, 'data/ensemble/best_moe_agents.pth')
                    logger.info(f"    🎉 [NEW BEST] 저장 (PnL:{best_val_pnl:.2f}% Score:{best_val_score:.2f})")

            # 10 에폭마다 체크포인트 저장 (중단 후 재시작 지원)
            if ep % 10 == 0:
                _save_checkpoint(ep)
                logger.info(f"    💾 [체크포인트] ep={ep} 저장 완료")

    except KeyboardInterrupt:
        logger.info("⚠️  학습 중단 감지 — 체크포인트 저장 중...")
        _save_checkpoint(ep)
        logger.info(f"✅ 체크포인트 저장 완료 (ep={ep}). 재시작 시 자동 복원됩니다.")

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