import os
import sys
import asyncio
import time
import logging
import gc
import json
import re
import numpy as np
import pandas as pd
import torch
import ccxt.async_support as ccxt
import warnings
from datetime import datetime, timedelta
from collections import deque
import os
from dotenv import load_dotenv

load_dotenv()

noisy_loggers = [
    "pytorch_lightning",
    "pytorch_lightning.utilities.rank_zero",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning_fabric",
    "lightning_fabric.utilities.rank_zero",
    "neuralforecast",
    "nixtla"
]

for name in noisy_loggers:
    l = logging.getLogger(name)
    l.setLevel(logging.ERROR) # ERROR 이상만 출력되도록 격하
    l.propagate = False       # 핵심 ⭐: 루트 로거로 메시지가 전파되는 것을 물리적으로 절단

# Gemini SDK / HTTP 클라이언트 INFO 로그 정리
for name in ["httpx", "google", "google.genai", "google_genai"]:
    l = logging.getLogger(name)
    l.setLevel(logging.WARNING)
    l.propagate = False

# 2. Warning 메시지도 정규식 수준에서 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch")
warnings.filterwarnings("ignore", ".*", module="lightning_fabric")


# 💡 [1. 경로 설정]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATHS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "models"),
    os.path.join(_THIS_DIR, "timesfm"),
    os.path.join(_THIS_DIR, "uni2ts", "src"),
    os.path.join(_THIS_DIR, "strategies"),
    os.path.join(_THIS_DIR, "ensemble"),
]
for p in TARGET_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from core.feature_engineering import FeatureEngineer
from ensemble.supervised.live_supervised_hub import SupervisedTrendHub
from ensemble.unsupervised.live_unsupervised_hub import UnsupervisedRegimeHub
from ensemble.arbiter_gatekeeper import BrainAOutput, BrainBOutput, DualBrainArbiter
from ensemble.ensemble_router import (
    TFTForecaster, MacroHFTForecaster, ChronosForecaster,
    KronosForecaster, TimesFMForecaster, MoiraiForecaster,
    TTMForecaster, NHITSForecaster, TiDEForecaster, PatchTSTForecaster,
)

# RL (4-Agent + GatingNet7) 상수 및 라우터
from ensemble.train_rl_agent import (
    GatingRouter7, GatingNet7, RobustIQN,
    STATE_DIM         as RL_STATE_DIM,
    STACKED_STATE_DIM as RL_STACKED_STATE_DIM,
    STATE_PRED        as RL_STATE_PRED,
    STATE_CONF        as RL_STATE_CONF,
    STATE_ELITE       as RL_STATE_ELITE,
    STATE_ALPHA       as RL_STATE_ALPHA,
    STATE_SYNTH       as RL_STATE_SYNTH,
    REGIME_COLS,
)
from strategies.elite_builder import EliteSignals, row_to_market_row
from strategies.elite_strategies import NewEliteSignalEngine

# SAC 라우터 (선택적 로드 — 체크포인트 없으면 비활성화)
try:
    from ensemble.train_rl_sac_agent import GaussianActor, SACRouter as _SACRouter
    _SAC_AVAILABLE = True
except ImportError as _sac_err:
    _SAC_AVAILABLE = False


class Colors:
    GREEN, RED, YELLOW, CYAN, RESET, BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LiveBot")


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


COMPACT_MODE = _env_flag('COMPACT_MODE', True)
LLM_ENABLED = _env_flag('LLM_ENABLED', False)
LLM_DISABLED_SENTINEL = '[LLM_DISABLED]'
LLM_SIGNAL_MIN_ABS = float(os.getenv("LLM_SIGNAL_MIN_ABS", "0.10"))
LLM_SIGNAL_TOP_K = int(os.getenv("LLM_SIGNAL_TOP_K", "8"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

SIG_INTERPRETATIONS = {
    'whale': ('고래 매수 다이버전스 포착', '고래 매도 다이버전스 포착', '고래 움직임 관망'),
    'liq_squeeze': ('숏 청산 스퀴즈 위협 (롱 반등)', '롱 청산 스퀴즈 위협 (숏 반등)', '청산 스퀴즈 위험 낮음'),
    'net_taker': ('시장가 매수 쏠림 강함', '시장가 매도 쏠림 강함', '시장가 매수/매도 균형'),
    'orderblock': ('지지 매물대(FVG) 강한 반등', '저항 매물대(FVG) 강한 거절', '주요 오더블록 미형성'),
    'hurst_ofi': ('프랙탈 상승 오더플로우 지속', '프랙탈 하락 오더플로우 지속', '랜덤워크/혼조세 (방향성 없음)'),
    'funding_cascade': ('펀딩비 극단적 음수 (숏 청산대기)', '펀딩비 극단적 양수 (롱 청산대기)', '펀딩비/미결제약정 안정적'),
    'multifractal': ('장기 노이즈 캔슬링: 상승', '장기 노이즈 캔슬링: 하락', '추세 노이즈 상쇄됨'),
    'cluster_fib': ('피보나치+CVP 클러스터 지지', '피보나치+CVP 클러스터 저항', '주요 방어선 부재'),
    'oi_divergence': ('하락 중 OI 증가 (숏 스퀴즈 잠재)', '상승 중 OI 증가 (롱 스퀴즈 잠재)', 'OI와 가격 동조화'),
    'top_trader_squeeze': ('탑트레이더 숏 쏠림 (상승폭발 대기)', '탑트레이더 롱 쏠림 (하락폭발 대기)', '탑트레이더 포지션 균형'),
    'btc_corr_breakout': ('BTC 커플링 이탈 (독자 상승)', 'BTC 커플링 이탈 (독자 하락)', 'BTC 방향성 강력 동조'),
    'ai_squeeze': ('변동성 응축 후 상방 폭발', '변동성 응축 후 하방 폭발', '변동성 평이함 (응축 없음)'),
    'vp_gravity': ('POC(매물대) 하단 이탈 후 탄성 반등', 'POC(매물대) 상단 이탈 후 탄성 하락', '최다 매물대(POC) 부근 체류'),
    'volume_confirm': ('거래량+방향성 확인 (추세 진입)', '거래량+방향성 확인 (추세 하락)', '거래량 방향 불명확'),
    'liquidity_trap': ('EQL 스탑헌팅 후 상승 반전', 'EQH 스탑헌팅 후 하락 반전', '스탑헌팅 패턴 없음'),
    'trend_health': ('추세 건강도 양호 (방향성 지속)', '추세 건강도 악화 (반전 주의)', '추세 건강도 중립'),
}
_RL_ELITE_KEYS = {c.replace('sig_', '') for c in RL_STATE_ELITE}


def _to_signal_view(sig_key: str, value: float) -> dict:
    base_key = sig_key.replace('sig_', '')
    long_msg, short_msg, neutral_msg = SIG_INTERPRETATIONS.get(base_key, ('LONG', 'SHORT', 'NEUTRAL'))
    if value > 0:
        direction, meaning = 'LONG', long_msg
    elif value < 0:
        direction, meaning = 'SHORT', short_msg
    else:
        direction, meaning = 'NEUTRAL', neutral_msg
    return {
        'name': base_key,
        'raw': float(value),
        'abs': float(abs(value)),
        'direction': direction,
        'meaning': meaning,
        'in_rl_state': base_key in _RL_ELITE_KEYS,
    }


def _active_signal_views(elite_sigs: dict, min_abs: float = LLM_SIGNAL_MIN_ABS, top_k: int = LLM_SIGNAL_TOP_K) -> list[dict]:
    rows = [_to_signal_view(k, float(v)) for k, v in elite_sigs.items()]
    rows.sort(key=lambda x: x['abs'], reverse=True)
    active = [r for r in rows if r['direction'] != 'NEUTRAL' and r['abs'] >= float(min_abs)]
    return active[:max(1, int(top_k))]


# ════════════════════════════════════════════════════════════════
# 0. 공통 헬퍼
# ════════════════════════════════════════════════════════════════
def _traj_direction(traj: np.ndarray) -> float:
    """slope+delta 합의 → {-1.0, 0.0, 1.0}  (get_direction 동일 로직)"""
    if len(traj) < 2:
        return float(np.sign(np.mean(traj)))
    slope = float(np.polyfit(np.arange(len(traj)), traj, 1)[0])
    delta = float(traj[-1] - traj[0])
    if slope > 0 and delta > 0:
        return 1.0
    if slope < 0 and delta < 0:
        return -1.0
    return 0.0


def _traj_conf(traj: np.ndarray) -> float:
    """tanh(|기울기|/표준편차) — get_conf 동일 로직"""
    if len(traj) < 2:
        return 0.5
    slope = float(np.polyfit(np.arange(len(traj), dtype=float), traj, 1)[0])
    std = float(np.std(traj)) + 1e-6
    return float(np.tanh(abs(slope) / std))


def _compute_mdjd(last_row: pd.Series, df: pd.DataFrame):
    """processed_df 마지막 행에서 pred_mdjd, conf_mdjd 계산"""
    def _g(col, default=0.0):
        return float(last_row.get(col, default))

    sq_vals = df['squeeze_power'].values if 'squeeze_power' in df.columns else np.zeros(len(df))
    sq_window = sq_vals[-288:] if len(sq_vals) >= 288 else sq_vals
    sq_mean = float(np.mean(sq_window))
    sq_std  = float(np.std(sq_window)) + 1e-8
    sq_z    = (sq_vals[-1] - sq_mean) / sq_std if len(sq_vals) > 0 else 0.0

    trend_4h = _g('mtf_trend_4h')
    trend_std = float(df['mtf_trend_4h'].std() + 1e-8) if 'mtf_trend_4h' in df.columns else 1e-8
    trend_z   = trend_4h / trend_std

    D = (0.005 * _g('smart_money_flow') * (1.0 + np.tanh(_g('whale_conviction')))
         + 0.002 * trend_4h)
    I = (0.003 * _g('net_taker_ratio')
         * np.exp(np.tanh(_g('taker_acceleration')))
         * (max(0.0, _g('amihud_illiquidity_z')) + 1.0))
    J = (0.01 * np.tanh(float(sq_z))
         * np.tanh(_g('funding_pressure'))
         * (1.0 if _g('breakout_strength') > 0.4 else 0.0))
    G = (-0.005 * _g('cvp_poc_dist')
         * np.exp(-float(np.clip(_g('cvp_volume_imbalance'), -5.0, 5.0)))
         * (1.0 - np.tanh(abs(trend_z))))

    R_hat = D + I + J + G
    return float(np.sign(R_hat)), float(np.tanh(abs(R_hat) * 100))


# ════════════════════════════════════════════════════════════════
# 1. 데이터 수집기
# ════════════════════════════════════════════════════════════════
class BinanceLiveFetcher:
    def __init__(self, symbol='ETHUSDT', timeframe='5m', limit=2500):
        self.symbol = symbol.replace('/', '')
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})

    def load_local_data(self):
        try:
            eth_df = pd.read_csv('data/test/eth_test_data.csv')
            btc_df = pd.read_csv('data/test/btc_test_data.csv')
            for df in [eth_df, btc_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cols = df.columns.drop('timestamp')
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            logger.info(f"{Colors.GREEN}📂 로컬 데이터 로드 성공{Colors.RESET}")
            return eth_df, btc_df
        except Exception as e:
            logger.error(f"로컬 로드 실패: {e}")
            return None, None

    async def fetch_klines_raw(self, symbol, target_limit):
        all_klines = []
        last_end_time = None
        while len(all_klines) < target_limit:
            params = {'symbol': symbol, 'interval': self.timeframe, 'limit': 1000}
            if last_end_time: params['endTime'] = last_end_time - 1
            klines = await self.exchange.fapiPublicGetKlines(params)
            if not klines: break
            all_klines = klines + all_klines
            last_end_time = klines[0][0]
            if len(klines) < 1000: break
        return all_klines[-target_limit:]

    async def fetch_ancillary_data(self, limit=500):
        tasks = [
            self.exchange.fapiDataGetOpenInterestHist({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortPositionRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetGlobalLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTakerlongshortRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiPublicGetFundingRate({'symbol': self.symbol, 'limit': limit})
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _process_to_df(self, eth_klines, btc_klines, ancillary_results):
        eth_df = pd.DataFrame(eth_klines).iloc[:, :11]
        eth_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        eth_df[eth_df.columns.drop('timestamp')] = eth_df[eth_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='coerce')

        btc_df = pd.DataFrame(btc_klines).iloc[:, [0, 4, 5, 7]]
        btc_df.columns = ['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df[btc_df.columns.drop('timestamp')] = btc_df[btc_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='coerce')

        if ancillary_results:
            mappings = [
                (0, 'sumOpenInterestValue', 'sum_open_interest_value'),
                (1, 'longShortRatio', 'sum_toptrader_long_short_ratio'),
                (2, 'longShortRatio', 'count_toptrader_long_short_ratio'),
                (3, 'longShortRatio', 'count_long_short_ratio'),
                (5, 'fundingRate', 'last_funding_rate'),
            ]
            for idx, key, new_name in mappings:
                res = ancillary_results[idx]
                if isinstance(res, list) and len(res) > 0:
                    try:
                        temp_df = pd.DataFrame(res)
                        t_col = next((c for c in ['timestamp', 'fundingTime', 'time'] if c in temp_df.columns), None)
                        if t_col and key in temp_df.columns:
                            subset = temp_df[[t_col, key]].rename(columns={t_col: 'timestamp', key: new_name})
                            subset['timestamp'] = pd.to_datetime(subset['timestamp'], unit='ms')
                            subset[new_name] = pd.to_numeric(subset[new_name], errors='coerce')
                            eth_df = pd.merge_asof(eth_df.sort_values('timestamp'), subset.sort_values('timestamp'), on='timestamp', direction='nearest')
                    except Exception: pass
        return eth_df.ffill().bfill(), btc_df

    async def fetch_initial_data(self):
        eth_klines = await self.fetch_klines_raw(self.symbol, self.limit)
        btc_klines = await self.fetch_klines_raw('BTCUSDT', self.limit)
        ancillary = await self.fetch_ancillary_data(500)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

    async def fetch_latest_patch(self):
        eth_klines = await self.exchange.fapiPublicGetKlines({'symbol': self.symbol, 'interval': self.timeframe, 'limit': 5})
        btc_klines = await self.exchange.fapiPublicGetKlines({'symbol': 'BTCUSDT', 'interval': self.timeframe, 'limit': 5})
        ancillary = await self.fetch_ancillary_data(5)
        return self._process_to_df(eth_klines, btc_klines, ancillary)


# ════════════════════════════════════════════════════════════════
# 2-A. 대시보드용 6대 파운데이션 앙상블 (표시 전용)
# ════════════════════════════════════════════════════════════════
class EnsemblePredictor:
    MODEL_ORDER = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']

    def __init__(self):
        self.models = {
            'TFT':      TFTForecaster(),
            'MacroHFT': MacroHFTForecaster(),
            'Chronos':  ChronosForecaster(),
            'Kronos':   KronosForecaster(),
            'TimesFM':  TimesFMForecaster(),
            'Moirai':   MoiraiForecaster(),
        }

    async def predict_all_async(self, df: pd.DataFrame):
        preds, confs = [], []
        loop = asyncio.get_event_loop()

        def _run_inference(m):
            if not getattr(m, 'available', False): return None
            try: return m.predict(df, horizon=6)
            except Exception: return None

        tasks = [loop.run_in_executor(None, _run_inference, self.models[name]) for name in self.MODEL_ORDER]
        results = await asyncio.gather(*tasks)

        for res in results:
            p_val, c_val = 0.0, 0.5
            if res is not None and getattr(res, 'median', None) is not None:
                traj = np.array(res.median[-1], dtype=np.float32)
                p_val = _traj_direction(traj)
                c_val = _traj_conf(traj)
            preds.append(p_val)
            confs.append(c_val)

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.array(preds), np.array(confs)


# ════════════════════════════════════════════════════════════════
# 2-B. Ridge 선형 퀀트 시그널 (라이브 트레이딩용)
# ════════════════════════════════════════════════════════════════
class RidgeSignalComputer:
    """data/ridge_model.joblib 로드 후 매 틱마다 pred_ridge / conf_ridge 반환."""

    _RIDGE_FEATURES = [
        'log_return', 'rsi', 'macd_hist', 'bb_width', 'volatility_z',
        'garman_klass_vol', 'last_funding_rate', 'oi_change_rate',
        'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
        'big_trade_ratio', 'hurst_12', 'hurst_48', 'hurst_288',
        'realized_vol_ratio', 'amihud_illiquidity_z',
    ]

    def __init__(self, model_path: str = 'data/ridge_model.joblib'):
        self._ridge   = None
        self._scaler  = None
        self._feats   = self._RIDGE_FEATURES
        self._abs_buf = deque(maxlen=500)   # rolling MAD 추적용

        if os.path.exists(model_path):
            try:
                from joblib import load as joblib_load
                pkg = joblib_load(model_path)
                self._ridge  = pkg['ridge']
                self._scaler = pkg['scaler']
                self._feats  = pkg.get('features', self._RIDGE_FEATURES)
                logger.info(f"✅ Ridge 퀀트 모델 로드: {len(self._feats)}개 피처")
            except Exception as e:
                logger.warning(f"⚠️ Ridge 모델 로드 실패: {e} — pred_ridge=0 사용")
        else:
            logger.warning(f"⚠️ Ridge 모델 없음 ({model_path}) — generate_csv 후 재기동 필요")

    def predict(self, df: pd.DataFrame) -> dict:
        if self._ridge is None:
            return {'pred_ridge': 0.0, 'conf_ridge': 0.5}
        row = df.iloc[-1]
        x   = np.array([float(row.get(f, 0.0)) for f in self._feats], dtype=np.float32)
        x   = np.nan_to_num(x, nan=0.0).reshape(1, -1)
        raw = float(self._ridge.predict(self._scaler.transform(x))[0])
        self._abs_buf.append(abs(raw))
        mad = float(np.median(list(self._abs_buf))) if self._abs_buf else 1e-8
        conf = float(np.tanh(abs(raw) / max(mad, 1e-8)))
        return {'pred_ridge': raw, 'conf_ridge': conf}


# ════════════════════════════════════════════════════════════════
# 2-C. 신경망 state 구성용 NF/TTM 예측
# ════════════════════════════════════════════════════════════════
class NFStatePredictor:
    """RL_STATE_PRED 전체 커버 (싱글 로드, 공유 사용)"""

    def __init__(self):
        # NF 4종은 UnifiedNFForecaster 싱글톤이므로 중복 로드 없음
        self.models = {
            'ttm':      TTMForecaster(),
            'tide':     TiDEForecaster(),
            'patchtst': PatchTSTForecaster(),
            'timesfm':  TimesFMForecaster(),
            'chronos':  ChronosForecaster(),
        }
        self.ridge = RidgeSignalComputer()

    def predict(self, df: pd.DataFrame) -> dict:
        """pred_{name}, conf_{name} 키로 구성된 dict 반환 (mdjd 제외)"""
        result = {}
        for name, model in self.models.items():
            try:
                out = model.predict(df, horizon=6) if getattr(model, 'available', False) else None
                if out is not None and getattr(out, 'median', None) is not None:
                    traj = np.array(out.median[-1], dtype=np.float32)
                    p, c = _traj_direction(traj), _traj_conf(traj)
                else:
                    p, c = 0.0, 0.5
            except Exception:
                p, c = 0.0, 0.5
            result[f'pred_{name}'] = p
            result[f'conf_{name}'] = c
        result.update(self.ridge.predict(df))
        return result


# ════════════════════════════════════════════════════════════════
# 2-D. 텔레그램 알림 (포지션 변화 시 fire-and-forget)
# ════════════════════════════════════════════════════════════════
class TelegramNotifier:
    """포지션 ENTER / EXIT / FLIP 시 텔레그램 메시지 전송."""

    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self._ok     = bool(self.token and self.chat_id)
        if not self._ok:
            logger.warning("⚠️ 텔레그램 미설정 — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 환경변수 필요")

    def _do_send(self, text: str) -> None:
        import urllib.request as _ur
        import urllib.error as _ue
        import json as _json
        url  = self._API.format(token=self.token)
        body = _json.dumps({'chat_id': self.chat_id, 'text': text,
                            'parse_mode': 'HTML'}).encode()
        req  = _ur.Request(url, data=body,
                           headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with _ur.urlopen(req, timeout=8) as r:
                raw = r.read().decode('utf-8', errors='ignore')
            try:
                payload = _json.loads(raw) if raw else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict) and payload.get('ok') is False:
                logger.warning(f"⚠️ 텔레그램 전송 실패: {payload.get('description', 'unknown')}")
                return
            logger.info("📨 텔레그램 전송 완료")
        except _ue.HTTPError as e:
            body_txt = ""
            try:
                body_txt = e.read().decode('utf-8', errors='ignore')
            except Exception:
                body_txt = repr(e)
            logger.warning(f"⚠️ 텔레그램 HTTP 오류 {e.code}: {body_txt[:300]}")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 전송 예외: {e}")

    async def notify(self, text: str) -> None:
        if not self._ok:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_send, text)


def _tg_trade_msg(ex_code: str, current_price: float,
                  timestamp_kst, regime_name: str, meta_result: dict) -> str:
    """텔레그램 전송용 포지션 변화 메시지 포맷."""
    fa    = int(meta_result.get('final_action', 0))
    kelly = float(meta_result.get('unified_kelly', 0.0))
    arb   = str(meta_result.get('arbiter_mode', 'N/A'))
    gate  = '✓' if meta_result.get('gate_passed', True) else '✗'
    ts_   = meta_result.get('trend_signal') or {}
    t_dir = {0: '▼ DOWN', 1: '─ FLAT', 2: '▲ UP'}.get(int(ts_.get('trend_dir', 1)), '?')
    icon  = {
        'ENTER_LONG':           '🟩',
        'ENTER_SHORT':          '🟥',
        'EXIT_LONG':            '⬜',
        'EXIT_SHORT':           '⬜',
        'FLIP_LONG_TO_SHORT':   '🔄',
        'FLIP_SHORT_TO_LONG':   '🔄',
    }.get(ex_code, '🟨')
    action_word = {1: 'LONG', 2: 'SHORT', 0: 'HOLD'}.get(fa, '?')
    return (
        f"{icon} <b>{ex_code}</b>  ({action_word})\n"
        f"💰 ETH ${current_price:,.2f}   🕐 {timestamp_kst.strftime('%m-%d %H:%M')} KST\n"
        f"🌍 {regime_name}   Kelly: {kelly:.3f}   Gate: {gate}\n"
        f"📈 Trend: {t_dir}   Arbiter: {arb}"
    )


# ════════════════════════════════════════════════════════════════
# 2-E. LLM 시장 분석기 (DeepSeek API)
# ════════════════════════════════════════════════════════════════
class LLMAnalyzer:
    """오더북 + 체결 흐름 + 선물 지표를 수집해 DeepSeek 모델에 방향성 분석 요청."""

    MODEL   = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
    FALLBACK_MODELS = tuple(
        m.strip() for m in os.getenv(
            "DEEPSEEK_LLM_FALLBACKS",
            "deepseek-chat",
        ).split(",") if m.strip()
    )
    API_KEY = (
        os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("LLM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    API_TIMEOUT_SEC = float(os.getenv("DEEPSEEK_TIMEOUT_SEC", "25"))

    def __init__(self, exchange, symbol: str = 'ETHUSDT'):
        self.exchange    = exchange
        self.symbol      = symbol
        self._fail_count = 0
        self._resolved_model: str | None = None
        self._pending_task: asyncio.Task | None = None   # fire-and-forget 태스크
        self._pending_meta: dict = {}                    # 태스크 생성 당시 regime_name 등
        if not self.API_KEY:
            logger.warning("⚠️ DeepSeek API 키가 설정되지 않았습니다. (DEEPSEEK_API_KEY/LLM_API_KEY)")

    @staticmethod
    def _extract_deepseek_text(resp_json: dict) -> str:
        if not isinstance(resp_json, dict):
            return ""
        choices = resp_json.get('choices', [])
        if not isinstance(choices, list) or not choices:
            return ""
        msg = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
        content = msg.get('content', '')
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(p.get('text', '')) for p in content if isinstance(p, dict)]
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _is_model_not_found(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ('404' in msg) or ('model_not_found' in msg) or ('not found' in msg) or ('unknown model' in msg)

    def _model_candidates(self) -> list[str]:
        candidates: list[str] = []
        if self._resolved_model:
            candidates.append(self._resolved_model)
        candidates.append(self.MODEL)
        candidates.extend(self.FALLBACK_MODELS)
        seen = set()
        uniq: list[str] = []
        for m in candidates:
            mm = (m or '').strip()
            if not mm:
                continue
            if mm in seen:
                continue
            seen.add(mm)
            uniq.append(mm)
        return uniq

    # ── 오더북 ────────────────────────────────────────────────────
    async def _fetch_orderbook(self) -> dict:
        try:
            res   = await self.exchange.fapiPublicGetDepth({'symbol': self.symbol, 'limit': 20})
            bids  = [[float(p), float(q)] for p, q in res['bids'][:5]]
            asks  = [[float(p), float(q)] for p, q in res['asks'][:5]]
            bid_v = sum(q for _, q in bids)
            ask_v = sum(q for _, q in asks)
            imbal = (bid_v - ask_v) / (bid_v + ask_v + 1e-8) * 100
            spread = asks[0][0] - bids[0][0] if bids and asks else 0.0
            return {'bid_ask_imbalance_pct': round(imbal, 2),
                    'spread_usdt': round(spread, 3),
                    'top5_bids': bids, 'top5_asks': asks}
        except Exception as e:
            return {'error': str(e)}

    # ── 최근 100건 체결 ───────────────────────────────────────────
    async def _fetch_agg_trades(self) -> dict:
        try:
            trades   = await self.exchange.fapiPublicGetAggTrades({'symbol': self.symbol, 'limit': 100})
            buy_vol  = sum(float(t['q']) for t in trades if not t['m'])   # m=False: buyer is taker
            sell_vol = sum(float(t['q']) for t in trades if     t['m'])
            total    = buy_vol + sell_vol + 1e-8
            buy_pct  = round(buy_vol / total * 100, 1)
            dominant = 'BUY' if buy_pct > 55 else 'SELL' if buy_pct < 45 else 'NEUTRAL'
            return {'buyer_aggressor_pct': buy_pct,
                    'seller_aggressor_pct': round(100 - buy_pct, 1),
                    'dominant_side': dominant,
                    'n_trades': len(trades)}
        except Exception as e:
            return {'error': str(e)}

    # ── 이미 계산된 피처 추출 (최근 N개 캔들) ─────────────────────
    @staticmethod
    def _extract_features(last_row) -> dict:
        def g(col, d=0.0):
            return float(last_row.get(col, d))
        return {
            'funding_rate':              g('last_funding_rate'),
            'oi_change_rate':            g('oi_change_rate'),
            'top_trader_ls_ratio':       g('sum_toptrader_long_short_ratio'),
            'global_ls_ratio':           g('count_long_short_ratio'),
            'net_taker_ratio':           g('net_taker_ratio'),
            'whale_retail_ratio':        g('whale_retail_ratio'),
            'smart_money_flow':          g('smart_money_flow'),
            'funding_pressure':          g('funding_pressure'),
            'squeeze_power':             g('squeeze_power'),
            'rsi':                       round(g('rsi', 50), 1),
            'macd_hist':                 round(g('macd_hist'), 6),
            'volatility_z':              round(g('volatility_z'), 3),
            'garch_vol_z':               round(g('garch_vol_z'), 3),
            'jump_flag':                 int(g('jump_flag')),
            'evt_tail_flag':             int(g('evt_tail_flag')),
            'ou_halflife_bars':          round(g('ou_halflife'), 1),
            'mtf_trend_1h':              g('mtf_trend_1h'),
            'mtf_trend_4h':              g('mtf_trend_4h'),
        }

    @staticmethod
    def _extract_candle_series(df: pd.DataFrame, n: int = 12) -> list[dict]:
        """최근 n개 캔들의 핵심 지표를 리스트로 반환 (프롬프트 시계열용)."""
        rows = df.tail(n)
        result = []
        for _, row in rows.iterrows():
            def g(col, d=0.0): return float(row.get(col, d))
            regime = ('BULL' if g('regime_bull') == 1.0 else
                      'BEAR' if g('regime_bear') == 1.0 else
                      'CHOP' if g('regime_chop') == 1.0 else
                      'WHIP' if g('regime_whipsaw') == 1.0 else 'NORM')
            ts = row.get('timestamp', '')
            ts_str = str(ts)[:16] if ts != '' else '?'
            result.append({
                'ts':              ts_str,
                'close':           round(g('close'), 2),
                'rsi':             round(g('rsi', 50), 1),
                'macd_h':          round(g('macd_hist'), 5),
                'vol_z':           round(g('volatility_z'), 2),
                'net_taker':       round(g('net_taker_ratio'), 3),
                'fund_pres':       round(g('funding_pressure'), 3),
                'smart_mf':        round(g('smart_money_flow'), 3),
                'squeeze':         round(g('squeeze_power'), 3),
                'regime':          regime,
            })
        return result

    # ── 프롬프트 빌드 ────────────────────────────────────────────
    def _build_prompt(self, price: float, regime: str,
                      orderbook: dict, trades: dict,
                      feats: dict, elite_sigs: dict,
                      candle_series: list[dict]) -> str:
        elite_active = _active_signal_views(elite_sigs, min_abs=LLM_SIGNAL_MIN_ABS, top_k=6)
        elite_active_min = [
            {
                'name': s['name'],
                'dir': s['direction'],
                'meaning': s['meaning'],
            }
            for s in elite_active
        ]
        last_c = candle_series[-1] if candle_series else {}
        ctx = {
            'symbol': f'{self.symbol} Perp',
            'price': round(float(price), 2),
            'regime': regime,
            'orderflow': {
                'imbalance_pct': float(orderbook.get('bid_ask_imbalance_pct')),
                'buy_pct': float(trades.get('buyer_aggressor_pct')),
                'dominant': str(trades.get('dominant_side')),
            },
            'market': {
                'funding': float(feats.get('funding_rate')),
                'oi_chg': float(feats.get('oi_change_rate')),
                'trend_1h': float(feats.get('mtf_trend_1h')),
                'trend_4h': float(feats.get('mtf_trend_4h')),
            },
            'tech': {
                'rsi': float(last_c.get('rsi', feats.get('rsi'))),
                'macd_h': float(last_c.get('macd_h', feats.get('macd_hist'))),
                'vol_z': float(last_c.get('vol_z', feats.get('volatility_z'))),
                'squeeze': float(last_c.get('squeeze', feats.get('squeeze_power'))),
            },
            'active_signals': elite_active_min,
        }
        return (
            "ETH/USDT 30분 방향만 판단.\n"
            "active_signals.meaning(시그널 해석)을 가장 우선 반영.\n"
            "출력은 반드시 한 줄, 아래 둘 중 하나 형식만:\n"
            "UP LOW|MEDIUM|HIGH\n"
            "DOWN LOW|MEDIUM|HIGH\n"
            f"{json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))}"
        )

    # ── DeepSeek 동기 호출 (executor에서 실행) ──────────────────────
    def _call_deepseek_sync(self, prompt: str) -> str:
        if not self.API_KEY:
            return '[DeepSeek API 키 없음 — DEEPSEEK_API_KEY/LLM_API_KEY 설정 필요]'
        try:
            import urllib.request
            import urllib.error

            last_err: Exception | None = None
            tried: list[str] = []
            for model_name in self._model_candidates():
                tried.append(model_name)
                payload = {
                    'model': model_name,
                    'messages': [
                        {'role': 'system', 'content': '당신은 트레이딩 방향 분류기다. 출력 포맷을 반드시 지켜라.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    'temperature': self.LLM_TEMPERATURE,
                    'max_tokens': 512,
                }
                req = urllib.request.Request(
                    self.API_URL,
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers={
                        'Authorization': f'Bearer {self.API_KEY}',
                        'Content-Type': 'application/json',
                    },
                    method='POST',
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.API_TIMEOUT_SEC) as resp:
                        resp_json = json.loads(resp.read().decode('utf-8'))
                    self._resolved_model = model_name
                    self._fail_count = 0
                    raw = self._extract_deepseek_text(resp_json)
                    return raw if raw else '[LLM 응답 없음]'
                except urllib.error.HTTPError as he:
                    body = he.read().decode('utf-8', errors='ignore')
                    e_model = RuntimeError(f'HTTP {he.code}: {body[:300]}')
                    last_err = e_model
                    if he.code == 404 or self._is_model_not_found(e_model):
                        continue
                    raise e_model
                except Exception as e_model:
                    last_err = e_model
                    if self._is_model_not_found(e_model):
                        continue
                    raise

            self._fail_count += 1
            return (
                f"[DeepSeek 모델 호출 실패 ({self._fail_count}회): "
                f"tried={','.join(tried)} err={last_err}]"
            )
        except Exception as e:
            self._fail_count += 1
            return f'[DeepSeek 오류 ({self._fail_count}회): {e}]'

    # ── 메인 진입점 (fire-and-forget) ────────────────────────────
    # 이번 사이클에 백그라운드 태스크 시작, 결과는 다음 사이클에서 수거
    # → 봇 사이클이 LLM 응답 대기로 블로킹되지 않음
    async def _run_analysis(self, processed_df: pd.DataFrame,
                            price: float, elite_sigs: dict, regime_name: str) -> str:
        last_row  = processed_df.iloc[-1]
        orderbook, trades = await asyncio.gather(
            self._fetch_orderbook(),
            self._fetch_agg_trades(),
        )
        feats         = self._extract_features(last_row)
        candle_series = self._extract_candle_series(processed_df, n=12)
        prompt = self._build_prompt(price, regime_name, orderbook, trades,
                                    feats, elite_sigs, candle_series)
        loop   = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_deepseek_sync, prompt)

    def fire(self, processed_df: pd.DataFrame,
             price: float, elite_sigs: dict, regime_name: str):
        """이번 사이클에 분석 태스크를 백그라운드로 시작."""
        if self._pending_task is not None and not self._pending_task.done():
            return  # 이전 태스크가 아직 실행 중 — 중복 시작 방지
        self._pending_task = asyncio.create_task(
            self._run_analysis(processed_df, price, elite_sigs, regime_name)
        )
        self._pending_meta = {'regime_name': regime_name}

    def collect(self) -> str | None:
        """이전 사이클의 LLM 결과를 수거. 아직 완료 안 됐으면 None 반환."""
        if self._pending_task is None:
            return None
        if not self._pending_task.done():
            return None
        try:
            result = self._pending_task.result()
        except Exception as e:
            result = f'[LLM 태스크 예외: {e}]'
        self._pending_task = None
        return result

    def collect_meta(self) -> dict:
        return self._pending_meta


def _print_llm_section(answer: str, regime_name: str, current_time_kst):
    print(f" 레짐: {regime_name} | 시간 : {current_time_kst.strftime('%Y-%m-%d %H:%M')} KST ")
    print("─" * 70)
    for line in answer.strip().splitlines():
        print(f" {line}")


def _compute_regime(df, window=24):
    close = df['close']
    net_change = close - close.shift(window)
    diff_abs   = close.diff().abs().rolling(window).sum()
    er         = net_change.abs() / (diff_abs + 1e-8)
    raw_vol    = close.pct_change().rolling(window).std()
    vol_z      = (raw_vol - raw_vol.rolling(window * 4).mean()) / (raw_vol.rolling(window * 4).std() + 1e-8)
    ema12      = close.ewm(span=12).mean()
    ema26      = close.ewm(span=26).mean()
    mtf        = (ema12 - ema26) / (ema26 + 1e-8) * 100

    er_v   = float(er.iloc[-1])         if er.notna().iloc[-1]       else 0.0
    volz_v = float(vol_z.iloc[-1])      if vol_z.notna().iloc[-1]    else 0.0
    nc_v   = float(net_change.iloc[-1]) if net_change.notna().iloc[-1] else 0.0
    mtf_v  = float(mtf.iloc[-1])        if mtf.notna().iloc[-1]      else 0.0

    bull = er_v >= 0.20 and nc_v > 0 and mtf_v > 0
    bear = er_v >= 0.20 and nc_v < 0 and mtf_v < 0
    chop = (not bull) and (not bear) and volz_v < -0.5
    whip = (not bull) and (not bear) and volz_v >  0.5
    norm = not (bull or bear or chop or whip)
    return {
        'regime_bull': 1.0 if bull else 0.0, 'regime_bear': 1.0 if bear else 0.0,
        'regime_chop': 1.0 if chop else 0.0, 'regime_whipsaw': 1.0 if whip else 0.0,
        'regime_normal': 1.0 if norm else 0.0,
    }


# ════════════════════════════════════════════════════════════════
# 3-A. PolymarketFetcher — ETH/BTC 5분봉 Up/Down 확률 수집
# ════════════════════════════════════════════════════════════════
class PolymarketFetcher:
    """
    Polymarket Gamma API에서 BTC·ETH 5분봉 Up/Down 시장 확률을 비동기로 조회.

    슬러그 생성 원리:
      window_ts = int(time.time()) - (int(time.time()) % 300)
      slug = f"{asset}-updown-5m-{window_ts}"
      → 예: btc-updown-5m-1773897300

    API 엔드포인트:
      GET https://gamma-api.polymarket.com/events?slug={slug}
      응답: [{markets: [{outcomes, outcomePrices, ...}]}]

    outcomePrices: JSON 문자열 "[up_price, down_price]"  (각 0~1, 합 ≈ 1)
    """

    GAMMA_URL  = "https://gamma-api.polymarket.com/events"
    CLOB_URL   = "https://clob.polymarket.com/price"   # mid-price fallback

    def __init__(self):
        self._session = None   # ccxt 내부 세션 재사용 (lazy init)

    async def _get(self, url: str, params: dict) -> list | dict | None:
        """aiohttp 없이 asyncio + ccxt exchange의 세션을 빌리거나
        asyncio executor 로 requests 동기 호출."""
        import urllib.request
        import urllib.parse
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if params else url
        loop = asyncio.get_event_loop()

        def _sync_get():
            try:
                req = urllib.request.Request(
                    full_url,
                    headers={
                        'User-Agent': 'Mozilla/5.0',
                        'Accept':     'application/json',
                    }
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    return json.loads(resp.read().decode())
            except Exception as e:
                logger.debug(f"Polymarket GET 오류 ({full_url}): {e}")
                return None

        return await loop.run_in_executor(None, _sync_get)

    def _current_window_ts(self) -> int:
        """현재 5분봉 윈도우 시작 타임스탬프 (UTC)."""
        now = int(time.time())
        return now - (now % 300)

    async def _fetch_market(self, asset: str, window_ts: int) -> dict | None:
        """
        단일 asset(btc|eth)의 현재 5분봉 시장 데이터를 반환.
        현재 윈도우에 시장이 없으면 직전 윈도우(-300s)로 fallback.
        반환: {'up': float, 'down': float, 'slug': str, 'title': str}
              또는 None (조회 실패 시)
        """
        for ts_offset in [0, -300, -600]:
            slug = f"{asset}-updown-5m-{window_ts + ts_offset}"
            data = await self._get(self.GAMMA_URL, {'slug': slug})
            if not data:
                continue

            event = data[0] if isinstance(data, list) and data else None
            if not event:
                continue

            markets = event.get('markets', [])
            if not markets:
                continue

            # outcomePrices: "[\"0.54\", \"0.46\"]"  (Up=index0, Down=index1)
            for m in markets:
                raw_prices = m.get('outcomePrices')
                raw_outcomes = m.get('outcomes')
                if not raw_prices:
                    continue
                try:
                    prices   = json.loads(raw_prices)   # ["0.54", "0.46"]
                    outcomes = json.loads(raw_outcomes) if raw_outcomes else ['Up', 'Down']
                    # outcomes 순서가 Up/Down인지 확인
                    if len(prices) < 2:
                        continue
                    # Polymarket 5분봉 시장: 첫 번째 outcome=Up, 두 번째=Down
                    up_idx   = next((i for i, o in enumerate(outcomes) if 'up'   in str(o).lower()), 0)
                    down_idx = next((i for i, o in enumerate(outcomes) if 'down' in str(o).lower()), 1)
                    up_prob   = float(prices[up_idx])
                    down_prob = float(prices[down_idx])
                    return {
                        'up':    round(up_prob,   4),
                        'down':  round(down_prob, 4),
                        'slug':  slug,
                        'title': event.get('title', slug),
                    }
                except (json.JSONDecodeError, ValueError, IndexError):
                    continue

        return None   # 모든 윈도우 조회 실패

    async def fetch(self) -> dict:
        """
        BTC·ETH 5분봉 Up/Down 확률을 동시에 조회.
        반환:
            {
                'btc': {'up': 0.54, 'down': 0.46, 'slug': '...', 'title': '...'},
                'eth': {'up': 0.48, 'down': 0.52, 'slug': '...', 'title': '...'},
                'ok':  True | False,   # 하나라도 성공하면 True
            }
        """
        window_ts = self._current_window_ts()
        btc_task  = asyncio.create_task(self._fetch_market('btc', window_ts))
        eth_task  = asyncio.create_task(self._fetch_market('eth', window_ts))
        btc_res, eth_res = await asyncio.gather(btc_task, eth_task, return_exceptions=True)

        btc = btc_res if isinstance(btc_res, dict) else None
        eth = eth_res if isinstance(eth_res, dict) else None

        return {
            'btc': btc,
            'eth': eth,
            'ok':  btc is not None or eth is not None,
        }


def _print_polymarket_section(poly_data: dict, current_price: float):
    """MetaRouter 대시보드 바로 아래에 출력할 폴리마켓 확률 패널."""
    C = Colors
    btc = poly_data.get('btc')
    eth = poly_data.get('eth')

    def _prob_bar(prob: float, width: int = 20) -> str:
        """확률을 █ 바로 시각화."""
        filled = round(prob * width)
        return '█' * filled + '░' * (width - filled)

    def _color_prob(prob: float, is_up: bool) -> str:
        """확률에 따라 색상 부여."""
        if is_up:
            return C.GREEN if prob >= 0.55 else (C.RED if prob <= 0.45 else C.YELLOW)
        else:
            return C.RED if prob >= 0.55 else (C.GREEN if prob <= 0.45 else C.YELLOW)

    print(C.BOLD + "╔══════════ [ 🎯 Polymarket 5분봉 크라우드 확률 ] ══════════╗" + C.RESET)

    for label, asset_data in [('BTC', btc), ('ETH', eth)]:
        if asset_data is None:
            print(f"  {label}  : 데이터 없음 (API 조회 실패 또는 시장 미개설)")
            continue
        up   = asset_data['up']
        down = asset_data['down']
        up_c   = _color_prob(up,   is_up=True)
        down_c = _color_prob(down, is_up=False)
        bar_up   = _prob_bar(up,   16)
        bar_down = _prob_bar(down, 16)
        print(f"  {C.BOLD}{label}{C.RESET} 5분봉")
        print(f"    UP  {up_c}{up*100:5.1f}%{C.RESET}  [{bar_up}]")
        print(f"    DN  {down_c}{down*100:5.1f}%{C.RESET}  [{bar_down}]")

    print(C.BOLD + "╚═══════════════════════════════════════════════════════════╝" + C.RESET)


def _parse_llm_direction(answer: str | None) -> tuple[str, str]:
    if not answer:
        return 'PENDING', '-'
    text = str(answer).strip()
    if text == LLM_DISABLED_SENTINEL or 'LLM_DISABLED' in text.upper():
        return 'DISABLED', '-'
    up = text.upper()
    m = re.search(r'\b(UP|DOWN|LONG|SHORT|NEUTRAL)\b', up)
    token = m.group(1) if m else 'UNKNOWN'
    if token in ('UP', 'LONG'):
        direction = 'UP'
    elif token in ('DOWN', 'SHORT'):
        direction = 'DOWN'
    elif token == 'NEUTRAL':
        direction = 'UNKNOWN'
    else:
        direction = 'UNKNOWN'

    if re.search(r'(높음|HIGH)', text, flags=re.IGNORECASE):
        conf = 'HIGH'
    elif re.search(r'(보통|MEDIUM)', text, flags=re.IGNORECASE):
        conf = 'MEDIUM'
    elif re.search(r'(낮음|LOW)', text, flags=re.IGNORECASE):
        conf = 'LOW'
    else:
        m_pct = re.search(r'(\d{1,3}(?:\.\d+)?)\s*%', text)
        if m_pct:
            v = float(m_pct.group(1))
            if v >= 70:
                conf = 'HIGH'
            elif v >= 50:
                conf = 'MEDIUM'
            else:
                conf = 'LOW'
        else:
            conf = '-'
    return direction, conf


def _pos_transition_label(prev_pos: str | None, cur_pos: str | None) -> str:
    if prev_pos == cur_pos:
        if cur_pos is None:
            return 'STAY FLAT'
        return f'HOLD {cur_pos}'
    if prev_pos is None and cur_pos is not None:
        return f'ENTER {cur_pos}'
    if prev_pos is not None and cur_pos is None:
        return f'EXIT {prev_pos}'
    return f'FLIP {prev_pos}->{cur_pos}'


def _print_final_trade_summary(timestamp_kst, current_price: float,
                               regime_name: str, rl_action: int, rl_info: dict,
                               meta_result: dict, prev_llm_answer: str | None,
                               prev_pos: str | None, cur_pos: str | None,
                               poly_data: dict | None = None,
                               sac_info: dict | None = None):
    C = Colors
    fa = int(meta_result.get('final_action', 0))

    def _action_word(a: int) -> str:
        return {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(int(a), 'UNKNOWN')

    def _action_color(a: int) -> str:
        return {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(int(a), C.RESET)

    def _bar(v: float, w: int = 8) -> str:
        x = float(np.clip(v, 0.0, 1.0))
        n = int(round(x * w))
        return '█' * n + '░' * (w - n)

    def _trend_word(tdir: int) -> str:
        return {0: 'DOWN', 1: 'FLAT', 2: 'UP'}.get(int(tdir), 'UNKNOWN')

    def _trend_color(tdir: int) -> str:
        return {0: C.RED, 1: C.YELLOW, 2: C.GREEN}.get(int(tdir), C.RESET)

    def _llm_conf_word(conf: str) -> str:
        return {'HIGH': 'HIGH', 'MEDIUM': 'MEDIUM', 'LOW': 'LOW'}.get(conf, 'UNKNOWN')

    def _llm_conf_color(conf: str) -> str:
        return {'HIGH': C.GREEN, 'MEDIUM': C.YELLOW, 'LOW': C.RED}.get(conf, C.CYAN)

    def _exec_code(pp: str | None, cp: str | None) -> tuple[str, str]:
        if pp == cp:
            if cp is None:
                return '·', 'STAY_FLAT'
            return '↔', 'HOLD_LONG' if cp == 'LONG' else 'HOLD_SHORT'
        if pp is None and cp == 'LONG':
            return '↗', 'ENTER_LONG'
        if pp is None and cp == 'SHORT':
            return '↘', 'ENTER_SHORT'
        if pp == 'LONG' and cp is None:
            return '✕', 'EXIT_LONG'
        if pp == 'SHORT' and cp is None:
            return '✕', 'EXIT_SHORT'
        if pp == 'LONG' and cp == 'SHORT':
            return '⇄', 'FLIP_LONG_TO_SHORT'
        if pp == 'SHORT' and cp == 'LONG':
            return '⇄', 'FLIP_SHORT_TO_LONG'
        return '·', _pos_transition_label(pp, cp)

    def _poly_text(asset_data: dict | None, label: str) -> str:
        if not isinstance(asset_data, dict):
            return f"POLYMARKET_{label}: NO_DATA"
        up = float(asset_data.get('up', 0.5))
        down = float(asset_data.get('down', 1.0 - up))
        return f"POLYMARKET_{label}: UP {up*100:5.1f}% | DOWN {down*100:5.1f}%"

    def _poly_color(asset_data: dict | None) -> str:
        if not isinstance(asset_data, dict):
            return C.YELLOW
        up = float(asset_data.get('up', 0.5))
        if up >= 0.55:
            return C.GREEN
        if up <= 0.45:
            return C.RED
        return C.YELLOW

    long_edge = float(rl_info.get('long_edge', 0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    rl_kelly = float(rl_info.get('kelly', 0.0))
    meta_kelly = float(meta_result.get('unified_kelly', 0.0))
    source = str(meta_result.get('source', 'N/A'))
    arb_mode = str(meta_result.get('arbiter_mode', 'N/A'))
    gate_passed = bool(meta_result.get('gate_passed', True))

    ts = meta_result.get('trend_signal') or {}
    t_dir = 1
    t_strength = 0.0
    t_rev = 0.0
    p_dn = p_fl = p_up = 0.0
    if isinstance(ts, dict) and ts:
        t_dir = int(ts.get('trend_dir', 1))
        t_strength = float(ts.get('strength', 0.0))
        t_rev = float(ts.get('rev_prob', 0.0))
        probs = ts.get('probs', [])
        if isinstance(probs, (list, tuple)) and len(probs) >= 3:
            p_dn = float(probs[0])
            p_fl = float(probs[1])
            p_up = float(probs[2])
        p_dn = float(ts.get('prob_dn', ts.get('p_down', p_dn)))
        p_fl = float(ts.get('prob_flat', ts.get('p_flat', p_fl)))
        p_up = float(ts.get('prob_up', ts.get('p_up', p_up)))

    llm_dir, llm_conf = _parse_llm_direction(prev_llm_answer)
    ex_icon, ex_code = _exec_code(prev_pos, cur_pos)
    sac_info = sac_info or {}
    sac_available = bool(sac_info.get('available', False))
    sac_action = int(sac_info.get('action', 0))
    sac_raw = float(sac_info.get('raw_action', 0.0))
    sac_size = float(sac_info.get('leverage', sac_info.get('kelly', 0.0)))
    sac_score = float(sac_info.get('score', abs(sac_raw)))

    edge_gap = abs(long_edge - short_edge)
    if long_edge > short_edge:
        edge_side_word = 'LONG_BIAS'
        edge_side_color = C.GREEN
    elif short_edge > long_edge:
        edge_side_word = 'SHORT_BIAS'
        edge_side_color = C.RED
    else:
        edge_side_word = 'NEUTRAL_BIAS'
        edge_side_color = C.YELLOW

    rl_word = _action_word(rl_action)
    rl_color = _action_color(rl_action)
    final_word = _action_word(fa)
    final_color = _action_color(fa)
    trend_word = _trend_word(t_dir)
    trend_color = _trend_color(t_dir)
    llm_dir_color = {'UP': C.GREEN, 'DOWN': C.RED, 'PENDING': C.CYAN, 'DISABLED': C.YELLOW}.get(llm_dir, C.RESET)
    llm_conf_color = _llm_conf_color(llm_conf)
    gate_word = 'PASS' if gate_passed else 'BLOCK'
    gate_color = C.GREEN if gate_passed else C.RED
    poly_data = poly_data or {}
    poly_btc = poly_data.get('btc') if isinstance(poly_data, dict) else None
    poly_eth = poly_data.get('eth') if isinstance(poly_data, dict) else None

    W = 62
    _SEP  = "─" * W
    _SEP2 = "═" * W

    def _action_arrow(a: int) -> str:
        return {0: '─', 1: '▲', 2: '▼'}.get(int(a), '?')

    def _trend_arrow(tdir: int) -> str:
        return {0: '▼', 1: '─', 2: '▲'}.get(int(tdir), '?')

    def _llm_arrow(d: str) -> str:
        return {'UP': '▲', 'DOWN': '▼', 'PENDING': '?', 'DISABLED': '·'}.get(d, '─')

    fa_arrow = _action_arrow(fa)
    rl_arrow = _action_arrow(rl_action)
    trend_arrow = _trend_arrow(t_dir)
    llm_arrow = _llm_arrow(llm_dir)
    gate_icon = '✓' if gate_passed else '✗'
    gate_log = meta_result.get('gate_log') or {}

    # ── 헤더: 최종 결과 화살표 ──────────────────────────────────────
    print(_SEP2)
    ts_str = timestamp_kst.strftime('%Y-%m-%d %H:%M')
    header_left = f"{final_color}{C.BOLD}{fa_arrow}{fa_arrow}  {final_word}  →  {ex_code}{C.RESET}"
    header_right = f"{C.CYAN}{ts_str}{C.RESET}"
    print(f" {header_left}  {header_right}")
    print(f"     {C.CYAN}${current_price:,.2f}  |  레짐: {regime_name}{C.RESET}")
    print(_SEP)

    # ── 서브 시그널 각 1행 ──────────────────────────────────────────
    # RL
    print(f"  {rl_color}{rl_arrow} RL{C.RESET}      {rl_color}{rl_word:<6}{C.RESET}"
          f"  엣지 {edge_side_color}{edge_side_word} {edge_gap:+.3f}{C.RESET}"
          f"  Kelly: {_bar(meta_kelly, 8)} {meta_kelly:.3f}")
    # SAC (참고용)
    if sac_available:
        sac_word = _action_word(sac_action)
        sac_color = _action_color(sac_action)
        sac_arrow = _action_arrow(sac_action)
        print(f"  {sac_color}{sac_arrow} SAC{C.RESET}     {sac_color}{sac_word:<6}{C.RESET}"
              f"  raw={sac_raw:+.3f}  size={_bar(sac_size, 8)} {sac_size:.3f}"
              f"  score={sac_score:.2f}")
    else:
        print(f"  {C.YELLOW}· SAC{C.RESET}     {C.YELLOW}DISABLED{C.RESET}"
              f"  ckpt/data unavailable")
    # TREND
    dn_c = C.RED if p_dn > 0.4 else C.RESET
    up_c = C.GREEN if p_up > 0.4 else C.RESET
    trend_model = str(ts.get("trend_model", "N/A")) if isinstance(ts, dict) else "N/A"
    print(f"  {trend_color}{trend_arrow} TREND{C.RESET}   {trend_color}{trend_word:<6}{C.RESET}"
          f"  str={t_strength:.2f}  rev={t_rev:.2f}"
          f"  {dn_c}DN={p_dn:.0%}{C.RESET} FL={C.YELLOW}{p_fl:.0%}{C.RESET} {up_c}UP={p_up:.0%}{C.RESET}"
          f"  model={trend_model}")
    # LLM
    llm_word = llm_dir if llm_dir in ('UP', 'DOWN', 'PENDING', 'DISABLED') else 'UNKNOWN'
    print(f"  {llm_dir_color}{llm_arrow} LLM{C.RESET}     {llm_dir_color}{llm_word:<6}{C.RESET}"
          f"  신뢰도: {_llm_conf_color(llm_conf)}{_llm_conf_word(llm_conf)}{C.RESET}")
    # GATE
    if not gate_passed and isinstance(gate_log, dict):
        blocked_gate = gate_log.get('blocked_gate', 'N/A')
        blocked_val  = gate_log.get(blocked_gate, '')
        print(f"  {gate_color}{gate_icon} GATE{C.RESET}    {gate_color}{gate_word:<6}{C.RESET}"
              f"  mode: {arb_mode}  차단: {C.RED}{blocked_gate}={blocked_val}{C.RESET}")
    else:
        print(f"  {gate_color}{gate_icon} GATE{C.RESET}    {gate_color}{gate_word:<6}{C.RESET}"
              f"  mode: {arb_mode}  source: {source}")
    # POLY
    if poly_btc is not None or poly_eth is not None:
        def _poly_pick(asset_data):
            if not isinstance(asset_data, dict):
                return C.YELLOW, '─', None
            up = float(asset_data.get('up', 0.5))
            dn = float(asset_data.get('down', 1.0 - up))
            if up > dn:
                return C.GREEN, '▲', up
            if dn > up:
                return C.RED, '▼', dn
            return C.YELLOW, '─', up

        btc_c, btc_arrow, btc_prob = _poly_pick(poly_btc)
        eth_c, eth_arrow, eth_prob = _poly_pick(poly_eth)
        btc_txt = f"{btc_c}{btc_arrow} {btc_prob*100:.1f}%{C.RESET}" if btc_prob is not None else f"{C.YELLOW}N/A{C.RESET}"
        eth_txt = f"{eth_c}{eth_arrow} {eth_prob*100:.1f}%{C.RESET}" if eth_prob is not None else f"{C.YELLOW}N/A{C.RESET}"
        print(f"  {C.CYAN}• POLY{C.RESET}    BTC {btc_txt}  |  ETH {eth_txt}")

    # ── 의사결정 체인 ────────────────────────────────────────────────
    print(_SEP)
    llm_conf_word = _llm_conf_word(llm_conf)
    llm_conf_col = _llm_conf_color(llm_conf)
    gate_chain_col = C.GREEN if gate_passed else C.RED
    if sac_available:
        sac_word = _action_word(sac_action)
        sac_color = _action_color(sac_action)
        sac_chain = f"SAC={sac_color}{sac_word}{C.RESET}({C.CYAN}{sac_raw:+.2f}{C.RESET}) → "
    else:
        sac_chain = ""
    decision_chain = (
        f"RL={rl_color}{rl_word}{C.RESET} → "
        f"{sac_chain}"
        f"TREND={trend_color}{trend_word}{C.RESET} → "
        f"LLM={llm_dir_color}{llm_word}{C.RESET}({llm_conf_col}{llm_conf_word}{C.RESET}) → "
        f"FINAL={final_color}{final_word}{C.RESET}({gate_chain_col}{gate_word}{C.RESET}) → "
        f"EXEC={ex_icon} {ex_code}"
    )
    if not gate_passed and isinstance(gate_log, dict):
        blocked_gate = gate_log.get('blocked_gate', 'N/A')
        decision_chain += f"  [{C.RED}차단={blocked_gate}{C.RESET}]"
    print(f"  {decision_chain}")
    print(_SEP2)


# ════════════════════════════════════════════════════════════════
# 3-A-2. SACLiveRouter — SAC Actor 라이브 추론 (선택적)
# ════════════════════════════════════════════════════════════════
class SACLiveRouter:
    """학습된 SAC Actor로 참고용 연속 action 추론.

    체크포인트가 없거나 임포트 실패 시 조용히 비활성화.
    IQN MoELiveRouter와 동일한 features/pos_dict 인터페이스 사용.
    """

    _SAC_PATH = 'data/ensemble/ckpt/best_sac_agents.pth'
    _NULL_INFO = {
        'agent': 'SAC', 'raw_action': 0.0, 'kelly': 0.0,
        'long_edge': 0.0, 'short_edge': 0.0, 'score': 0.0,
    }

    def __init__(self):
        self.available = False
        self._router = None
        if not _SAC_AVAILABLE:
            logger.warning("⚠️ SAC 모듈 없음 (train_rl_sac_agent 임포트 실패) — 참고 패널 비활성화")
            return
        if not os.path.exists(self._SAC_PATH):
            logger.warning(f"⚠️ SAC 체크포인트 없음: {self._SAC_PATH} — 참고 패널 비활성화")
            return
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt = torch.load(self._SAC_PATH, map_location=device, weights_only=False)
            actor = GaussianActor(state_dim=RL_STACKED_STATE_DIM).to(device)
            actor.load_state_dict(ckpt['actor'])
            actor.eval()
            self._router = _SACRouter(actor, device=device)
            self.available = True
            logger.info(f"✅ SACLiveRouter 로드 완료: {self._SAC_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ SAC 로드 실패: {e}")

    def decide(self, features: dict, pos_dict: dict) -> tuple[int, float, dict]:
        if not self.available or self._router is None:
            return 0, 0.0, dict(self._NULL_INFO)
        try:
            return self._router.decide(features, pos_dict)
        except Exception as e:
            logger.debug(f"SAC 추론 실패: {e}")
            return 0, 0.0, dict(self._NULL_INFO)


# ════════════════════════════════════════════════════════════════
# 3-B. MetaRouter — RL + XGB 듀얼 브레인 (Arbiter + Gatekeeper) 통합 레이어
# ════════════════════════════════════════════════════════════════
# [설계 원칙]
# - MoELiveRouter 코드 변경 없음
# - _run_cycle() 에서 meta_router.fuse() 한 줄로 RL/XGB/게이트 통합
# - 포지션은 MetaRouter가 단일 소스로 관리

# ── 튜닝 상수 ────────────────────────────────────────────────────
_META_SOLO_RL_THRESH   = 0.02   # RL 진입 score 최솟값 (0.05 -> 0.02 완화)
_META_HISTORY_N        = 30     # 동적 가중치 추적 윈도우

# 레짐별 RL 우선순위 배율
_REGIME_RL_WEIGHT = {
    'bull': 1.20, 'bear': 1.20, 'normal': 1.00,
    'chop': 0.80, 'whipsaw': 0.75,
}


def _meta_rl_score(rl_action: int, rl_info: dict, regime: dict) -> float:
    """GatingRouter7 info 구조로 RL 신호 강도 계산 (0~1)."""
    if rl_action == 0:
        return 0.0
    kelly      = float(max(0.0, rl_info.get('kelly', 0.0)))
    long_edge  = float(rl_info.get('long_edge',  0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    agent_score = float(max(0.0, rl_info.get('score', 0.0)))

    # 실제 RL 액션 방향과 정렬된 edge를 우선 반영
    action_edge = long_edge if rl_action == 1 else short_edge if rl_action == 2 else 0.0
    edge_gap    = abs(long_edge - short_edge)
    edge_strength = max(abs(action_edge), edge_gap * 0.5)

    active     = next((k.replace('regime_', '') for k, v in regime.items() if v == 1.0), 'normal')
    regime_w   = _REGIME_RL_WEIGHT.get(active, 1.0)
    # kelly만 곱하면 진입이 과도하게 억제되므로 완만한 바닥 가중치를 둔다.
    signal_strength = float(np.tanh(max(edge_strength, agent_score) * 3.0))
    kelly_term      = 0.25 + 0.75 * kelly
    raw             = signal_strength * kelly_term * regime_w
    return float(np.clip(raw, 0.0, 1.0))


class MetaRouter:
    """RL MoE 4-Agent 신호를 받아 단일 최종 액션을 결정."""

    def __init__(self):
        # 동적 가중치 추적 (승률 기반)
        self._rl_score_acc  = 1.0
        self._history       = []        # {'rl':int,'final':int,'src':str,'outcome':None}
        self._arbiter       = DualBrainArbiter(boost_factor=1.2, veto_threshold=0.35, hmm_conf_threshold=0.5)

        # MetaRouter 자체 포지션 상태
        self.pos:          str | None = None
        self.entry_price:  float      = 0.0
        self.hold_count:   int        = 0
        self.peak_equity:  float      = 1.0
        self.cur_equity:   float      = 1.0

    def record_outcome(self, realized_pnl_pct: float):
        """포지션 청산 후 실현 PnL을 피드백해 동적 가중치를 보정."""
        if not self._history:
            return
        rec  = self._history[-1]
        correct = realized_pnl_pct > 0.0
        if rec['src'] in ('RL_SOLO', 'RL_ACTIVE') or str(rec.get('src', '')).startswith('ARB_'):
            self._rl_score_acc += 1.0 if correct else -0.5
            self._rl_score_acc  = max(0.1, self._rl_score_acc)

    # ── 메인 진입점 ───────────────────────────────────────────────
    def fuse(self, rl_action: int, rl_info: dict,
             regime: dict, current_price: float = 0.0,
             trend_signal=None, garch_vol_z: float = 0.0) -> dict:
        """
        Returns dict:
          final_action  : 0/1/2
          unified_kelly : 0~1
          source        : 'ARB_*'|'RL_ACTIVE'|'FLAT'
          rl_score, meta_score
          trend_signal  : TrendSignal.to_arbiter_dict() 또는 None
          trend_veto    : 적용된 trend filter 사유 문자열 또는 None
        """
        rl_score = _meta_rl_score(rl_action, rl_info, regime)
        trend_veto = None
        arbiter_mode = 'N/A'
        gate_passed = (rl_action != 0)
        gate_log = {}

        if trend_signal is not None:
            try:
                brain_a = BrainAOutput.from_router_output(
                    action=rl_action,
                    leverage=float(np.clip(float(rl_info.get('kelly', 0.0)), 0.0, 1.0)),
                    info=rl_info,
                )
                if isinstance(trend_signal, dict):
                    brain_b = BrainBOutput.from_dict(trend_signal)
                    trend_signal_dict = dict(trend_signal)
                else:
                    brain_b = BrainBOutput.from_signal(trend_signal)
                    trend_signal_dict = trend_signal.to_arbiter_dict() if hasattr(trend_signal, 'to_arbiter_dict') else None

                portfolio_mdd = 0.0
                if self.pos is not None and self.peak_equity > 0:
                    portfolio_mdd = min((self.cur_equity / self.peak_equity) - 1.0, 0.0)

                hmm_state = str(rl_info.get('hmm_state', 'lv-range'))
                hmm_probs = rl_info.get('hmm_probs', [0.25, 0.25, 0.25, 0.25])
                if not isinstance(hmm_probs, (list, tuple)) or len(hmm_probs) == 0:
                    hmm_probs = [0.25, 0.25, 0.25, 0.25]

                decision = self._arbiter.decide(
                    brain_a=brain_a,
                    brain_b=brain_b,
                    garch_vol_z=float(garch_vol_z),
                    portfolio_mdd=float(portfolio_mdd),
                    hmm_state=hmm_state,
                    hmm_probs=list(hmm_probs),
                )
                final_action = int(decision.final_action)
                unified_kelly = float(np.clip(decision.final_lev, 0.0, 1.0))
                arbiter_mode = decision.arbiter_mode
                gate_passed = bool(decision.gate_passed)
                gate_log = decision.gate_log or {}

                if final_action == 0:
                    if arbiter_mode == 'VETO':
                        source = 'ARB_VETO'
                        trend_veto = 'ARB_VETO'
                    elif not gate_passed:
                        source = 'ARB_GATE_BLOCK'
                        trend_veto = 'ARB_GATE_BLOCK'
                    else:
                        source = f'ARB_{arbiter_mode}'
                else:
                    source = f'ARB_{arbiter_mode}'
            except Exception as e:
                logger.warning(f"Arbiter 적용 실패 → legacy fallback: {e}")
                trend_signal_dict = trend_signal.to_arbiter_dict() if hasattr(trend_signal, 'to_arbiter_dict') else (trend_signal if isinstance(trend_signal, dict) else None)
                if rl_action == 0:
                    final_action, source = 0, 'FLAT'
                elif rl_score >= _META_SOLO_RL_THRESH:
                    final_action, source = rl_action, 'RL_ACTIVE'
                else:
                    final_action, source = 0, 'FLAT'
                unified_kelly = self._calc_kelly(final_action, rl_info, rl_score)
                if final_action != 0 and trend_signal is not None and not isinstance(trend_signal, dict):
                    final_action, unified_kelly, trend_veto = self._apply_trend_filter(
                        final_action, unified_kelly, trend_signal
                    )
        else:
            trend_signal_dict = None
            if rl_action == 0:
                final_action, source = 0, 'FLAT'
            elif rl_score >= _META_SOLO_RL_THRESH:
                final_action, source = rl_action, 'RL_ACTIVE'
            else:
                final_action, source = 0, 'FLAT'
            unified_kelly = self._calc_kelly(final_action, rl_info, rl_score)

        meta_score = rl_score if final_action != 0 else 0.0

        # trend veto 로 action=0 됐으면 source 갱신
        if final_action == 0 and trend_veto is not None:
            source = trend_veto

        self._update_pos(final_action, current_price)
        self._history.append({'rl': rl_action, 'final': final_action, 'src': source, 'outcome': None, 'arb_mode': arbiter_mode})
        if len(self._history) > _META_HISTORY_N * 2:
            self._history = self._history[-_META_HISTORY_N:]

        return {
            'final_action':  final_action,
            'unified_kelly': unified_kelly,
            'source':        source,
            'rl_score':      rl_score,
            'meta_score':    meta_score,
            'rl_action':     rl_action,
            'trend_signal':  trend_signal_dict,
            'trend_veto':    trend_veto,
            'arbiter_mode':  arbiter_mode,
            'gate_passed':   gate_passed,
            'gate_log':      gate_log,
        }

    # ── TrendContextBrain 거버넌스 ────────────────────────────────
    def _apply_trend_filter(self, action: int, kelly: float, trend_signal) -> tuple:
        """4h 추세 기반 진입 거부 / Kelly 조정.

        VETO_STRENGTH  : 이 이상 강한 역방향 추세 → 진입 거부 (action=0)
        BOOST_STRENGTH : 이 이상 동방향 추세 → Kelly 부스트 (+25%)
        CHOP_STRENGTH  : FLAT 추세 강도 → Kelly 축소 (×0.8)
        REV_VETO_PROB  : 반전 확률 이 이상 → Kelly 축소 (×0.6)
        """
        VETO_STRENGTH  = 0.70   # 완화: 매우 강한 역방향만 거부
        BOOST_STRENGTH = 0.35
        CHOP_STRENGTH  = 0.30
        REV_VETO_PROB  = 0.70   # 완화: 반전 확률 매우 높을 때만 축소

        veto = None

        # ① 강한 역방향 추세 → 진입 거부
        if trend_signal.strength >= VETO_STRENGTH:
            if trend_signal.trend_dir == 0 and action == 1:   # 4h 하락 + 롱 진입
                return 0, 0.0, 'TREND_DOWN_VETO'
            if trend_signal.trend_dir == 2 and action == 2:   # 4h 상승 + 숏 진입
                return 0, 0.0, 'TREND_UP_VETO'

        # ② 동방향 추세 → Kelly 부스트
        if trend_signal.strength >= BOOST_STRENGTH:
            aligned = (trend_signal.trend_dir == 2 and action == 1) or \
                      (trend_signal.trend_dir == 0 and action == 2)
            if aligned:
                kelly = float(np.clip(kelly * (1.0 + trend_signal.strength * 0.25), 0.0, 1.0))
                veto  = 'TREND_BOOST'

        # ③ FLAT(횡보) 추세 → Kelly 축소
        if trend_signal.trend_dir == 1 and trend_signal.strength >= CHOP_STRENGTH:
            kelly *= 0.80
            veto   = veto or 'TREND_CHOP_REDUCE'

        # ④ 반전 확률 높음 → Kelly 추가 축소
        if trend_signal.rev_prob >= REV_VETO_PROB:
            kelly = float(np.clip(kelly * 0.60, 0.0, 1.0))
            veto  = veto or 'TREND_REV_REDUCE'

        return action, kelly, veto

    def _calc_kelly(self, final_action, rl_info, rl_score):
        if final_action == 0:
            return 0.0
        rl_kelly = float(rl_info.get('kelly', 0.0))
        return float(np.clip(rl_kelly, 0.0, 1.0))

    def _update_pos(self, final_action, current_price):
        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'LONG', current_price, 0
            self.peak_equity = self.cur_equity = 1.0
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'SHORT', current_price, 0
            self.peak_equity = self.cur_equity = 1.0
        elif final_action == 0 and self.pos is not None:
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        elif self.pos is not None:
            self.hold_count += 1
            if current_price > 0 and self.entry_price > 0:
                if self.pos == 'LONG':
                    self.cur_equity = 1.0 + (current_price - self.entry_price) / self.entry_price
                else:
                    self.cur_equity = 1.0 + (self.entry_price - current_price) / self.entry_price
                self.peak_equity = max(self.peak_equity, self.cur_equity)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.pos is None or self.entry_price == 0.0:
            return 0.0
        if self.pos == 'LONG':
            return (current_price - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - current_price) / self.entry_price * 100.0

    def print_meta_dashboard(self, result: dict, current_price: float = 0.0):
        C = Colors
        src    = result['source']
        fa     = result['final_action']
        arb_mode = result.get('arbiter_mode', 'N/A')
        gate_passed = bool(result.get('gate_passed', True))
        gate_log = result.get('gate_log', {}) or {}
        action_str = {0: f'{C.YELLOW}🟨 관망{C.RESET}',
                      1: f'{C.GREEN}🟩 LONG{C.RESET}',
                      2: f'{C.RED}🟥 SHORT{C.RESET}'}.get(fa, '?')
        rl_a_str = {0:'관망', 1:'LONG', 2:'SHORT'}.get(result['rl_action'], '?')

        unr = self.unrealized_pnl(current_price)
        unr_color = C.GREEN if unr > 0 else (C.RED if unr < 0 else C.YELLOW)

        fa_arrow = {0: '─', 1: '▲', 2: '▼'}.get(fa, '?')
        fa_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(fa, C.RESET)
        fa_word  = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(fa, '?')
        rl_arrow = {0: '─', 1: '▲', 2: '▼'}.get(result['rl_action'], '?')
        rl_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(result['rl_action'], C.RESET)
        gate_icon = '✓' if gate_passed else '✗'
        gate_color = C.GREEN if gate_passed else C.RED

        # 헤더: 최종 결정 화살표
        print(f" {fa_color}{C.BOLD}{fa_arrow}{fa_arrow}  {fa_word}{C.RESET}"
              f"  score={result['meta_score']:.3f}  Kelly={result['unified_kelly']:.3f}"
              f"  source: {C.CYAN}{src}{C.RESET}")
        # RL 서브시그널
        print(f"  {rl_color}{rl_arrow} RL{C.RESET}      {rl_color}{rl_a_str:<6}{C.RESET}"
              f"  rl_score={result['rl_score']:.3f}")
        # Gate + Arbiter
        gate_line = f"  {gate_color}{gate_icon} Gate{C.RESET}    {gate_color}{'PASS' if gate_passed else 'BLOCK'}{C.RESET}  mode={C.CYAN}{arb_mode}{C.RESET}"
        if not gate_passed:
            blk_gate = next((g for g in ['gate1', 'gate2', 'gate3', 'gate4', 'gate5', 'gate6']
                             if str(gate_log.get(g, '')).startswith('BLOCK')), None)
            if blk_gate is not None:
                gate_line += f"  {C.RED}차단={blk_gate}{C.RESET}"
        print(gate_line)
        # 포지션
        if self.pos is not None:
            pos_color = C.GREEN if self.pos == 'LONG' else C.RED
            print(f"  {pos_color}● 포지션{C.RESET}  {pos_color}{self.pos}{C.RESET}"
                  f"  진입가={self.entry_price:.2f}"
                  f"  미실현={unr_color}{unr:+.2f}%{C.RESET}  보유={self.hold_count}봉")
        # XGBTrend
        ts = result.get('trend_signal')
        tv = result.get('trend_veto')
        if ts is not None:
            t_dir = ts['trend_dir']
            t_arrow = {0: '▼', 1: '─', 2: '▲'}.get(t_dir, '?')
            t_color = {0: C.RED, 1: C.YELLOW, 2: C.GREEN}.get(t_dir, C.RESET)
            t_word  = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}.get(t_dir, '?')
            t_model = str(ts.get('trend_model', 'N/A')) if isinstance(ts, dict) else 'N/A'
            veto_str = f"  {C.RED}[{tv}]{C.RESET}" if tv else ''
            probs = ts.get('probs', [])
            if len(probs) == 3:
                dn_c = C.RED if probs[0] > 0.4 else C.RESET
                up_c = C.GREEN if probs[2] > 0.4 else C.RESET
                probs_str = (f"  {dn_c}DN={probs[0]:.0%}{C.RESET}"
                             f" FL={C.YELLOW}{probs[1]:.0%}{C.RESET}"
                             f" {up_c}UP={probs[2]:.0%}{C.RESET}")
            else:
                probs_str = ''
            print(f"  {t_color}{t_arrow} Trend{C.RESET}   {t_color}{t_word:<6}{C.RESET}"
                  f"  str={ts['strength']:.2f}  rev={ts['rev_prob']:.2f}"
                  f"{probs_str}  model={t_model}{veto_str}")


# ════════════════════════════════════════════════════════════════
# 3-C. MoE 6-Agent 라우터 — RL_STATE_DIM=35, STACKED=70
# ════════════════════════════════════════════════════════════════
class MoELiveRouter:
    # 1. 클래스 변수 정의 (MODEL_ORDER 에러 해결 핵심)
    MODEL_ORDER = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend([
            'data/ensemble/ckpt/best_rl_agents.pth',
            'data/ensemble/ckpt/rl_checkpoint.pth',
        ])

        tried = []
        seen = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            tried.append(p)
            if os.path.exists(p):
                return p

        tried_msg = ', '.join(tried)
        raise FileNotFoundError(
            f"RL 체크포인트 파일이 없습니다. 확인 경로: {tried_msg}. "
            "먼저 `venv/bin/python ensemble/train_rl_agent.py`로 학습하여 "
            "`best_rl_agents.pth` 또는 `rl_checkpoint.pth`를 생성하세요."
        )

    def __init__(self, model_path: str | None = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 퀀트 시그널 추출기 초기화
        from strategies.elite_builder import EliteSignals
        self.elite_extractor = EliteSignals()
        self.new_elite_engine = NewEliteSignalEngine()

        # v7 상수 및 모델 로드
        from ensemble.train_rl_agent import (
            RobustIQN as RL_RobustIQN,
            GatingRouter7, GatingNet7,
            STATE_DIM as RL_STATE_DIM,
            STACKED_STATE_DIM as RL_STACKED_STATE_DIM
        )

        resolved_model_path = self._resolve_model_path(model_path)
        logger.info(f"✅ RL 체크포인트 로드 경로: {resolved_model_path}")

        ckpt = torch.load(resolved_model_path, map_location=self.device, weights_only=False)
        agent_names = ['bull', 'bear', 'chop_long', 'chop_short', 'normal_long', 'normal_short']
        required_keys = [f'model_{n}' for n in agent_names] + ['gating_net']
        missing = [k for k in required_keys if k not in ckpt]
        if missing:
            raise KeyError(
                f"체크포인트 키 누락: {missing}. "
                f"파일이 RL 학습 체크포인트 형식이 아닙니다: {resolved_model_path}"
            )

        def _load(key):
            # 훈련 체크포인트 규격(2-Action)에 맞춤
            m = RL_RobustIQN(RL_STACKED_STATE_DIM, 2, raw_state_dim=RL_STATE_DIM).to(self.device)
            m.load_state_dict(ckpt[key])
            m.eval()
            return m

        # 6-Agent + 1-Flat(GatingNet7) 매핑
        models_dict = {name: _load(f'model_{name}') for name in agent_names}
        
        gating_net = GatingNet7(RL_STACKED_STATE_DIM).to(self.device)
        gating_net.load_state_dict(ckpt['gating_net'], strict=False)
        gating_net.eval()

        self.trader = GatingRouter7(models_dict, gating_net, self.device)
        
        # 상태 변수
        self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        self.peak_equity, self.current_equity = 1.0, 1.0
        logger.info(f"✅ MoE 사령관 대시보드 인터페이스 연결 완료.")

    def get_signal(self, processed_df: pd.DataFrame, preds, confs, nf_preds: dict):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        smf_std  = processed_df['smart_money_flow'].std() if 'smart_money_flow' in processed_df.columns else 1.0

        cur_market  = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs  = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)

        # NewEliteSignalEngine: sig_volume_confirm, sig_liquidity_trap, sig_trend_health
        try:
            _tail = processed_df.tail(100).copy()
            self.new_elite_engine.compute(_tail)
            _last = _tail.iloc[-1]
            for col in ['sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health']:
                elite_sigs[col] = float(_last.get(col, 0.0))
        except Exception as _e:
            logger.warning(f"NewEliteSignalEngine 계산 실패: {_e}")
            for col in ['sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health']:
                elite_sigs.setdefault(col, 0.0)

        features = {}

        # RL_STATE_PRED: ['pred_tide','pred_ridge','pred_patchtst','pred_timesfm','pred_chronos','pred_ttm','pred_mdjd']
        pred_mdjd, conf_mdjd = _compute_mdjd(last_row, processed_df)
        for col in RL_STATE_PRED:
            if col == 'pred_mdjd':
                features[col] = pred_mdjd
            else:
                features[col] = float(nf_preds.get(col, 0.0))
        # RL_STATE_CONF: ['conf_timesfm','conf_chronos','conf_ttm','conf_patchtst','conf_tide','conf_mdjd','conf_ridge']
        for col in RL_STATE_CONF:
            if col == 'conf_mdjd':
                features[col] = conf_mdjd
            else:
                features[col] = float(nf_preds.get(col, 0.5))

        # RL_STATE_ELITE: ['sig_ai_squeeze','sig_whale','sig_oi_divergence','sig_volume_confirm','sig_liquidity_trap','sig_trend_health']
        for col in RL_STATE_ELITE:
            features[col] = float(elite_sigs.get(col, 0.0))

        # RL_STATE_ALPHA: ['hour_cos','garch_vol_z','breakout_strength','fvg_dist','oi_change_rate','cvp_volume_imbalance']
        for col in RL_STATE_ALPHA:
            features[col] = float(last_row.get(col, 0.0))

        # Regime
        regime = _compute_regime(processed_df)
        features.update(regime)

        # RL_STATE_SYNTH: ['ofti', 'kel']
        for col in RL_STATE_SYNTH:
            features[col] = float(last_row.get(col, 0.0))

        features['close'] = float(last_row['close'])

        unr = 0.0
        if self.pos is not None:
            cp = float(last_row['close'])
            unr = (cp - self.entry_price) / self.entry_price if self.pos == 'LONG' else (self.entry_price - cp) / self.entry_price
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity: self.peak_equity = self.current_equity
        else:
            self.current_equity = self.peak_equity = 1.0

        pos_dict = {
            'type': self.pos, 'entry_price': self.entry_price,
            # GatingRouter7._state_tensor과 동일한 정규화 — _build_state 기준
            'unrealized': float(np.tanh(unr / 0.02)),
            'mdd': float(np.clip(min((self.current_equity / self.peak_equity) - 1.0, 0.0) / 0.05, -1.0, 1.0)),
            'hold_norm': min(self.hold_count / 144.0, 1.0)  # TradingEnv._build_state 기준 /144
        }

        final_action, leverage_rate, info = self.trader.decide(features, pos_dict)

        return final_action, info, elite_sigs, regime, features, pos_dict

    def print_dashboard(self, current_price, preds, confs, final_action, info, regime, elite_sigs, timestamp,
                        llm_answer: str | None = None, llm_regime_name: str | None = None, llm_time_kst=None):
        pnl_pct = (self.current_equity - 1.0) * 100
        regime_name  = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')
        active_agent = info.get('agent', 'NONE')
        kelly        = info.get('kelly', 0.0)
        long_edge    = info.get('long_edge', 0.0)
        short_edge   = info.get('short_edge', 0.0)

        def format_edge(edge):
            if edge > 0.01:  return f"{Colors.GREEN}{edge:+.3f} (진입희망){Colors.RESET}"
            elif edge < -0.01: return f"{Colors.RED}{edge:+.3f} (진입거부){Colors.RESET}"
            else:              return f"{Colors.YELLOW}{edge:+.3f} (완전중립){Colors.RESET}"

        _is_holding = (self.pos is not None)
        action_str = {
            0: f'{Colors.YELLOW}🟨 관망 / 청산 (HOLD / CLOSE){Colors.RESET}',
            1: f'{Colors.GREEN}🟩 {"롱 유지 (HOLDING)" if _is_holding else "롱 진입 (LONG)"}{Colors.RESET}',
            2: f'{Colors.RED}🟥 {"숏 유지 (HOLDING)" if _is_holding else "숏 진입 (SHORT)"}{Colors.RESET}',
        }.get(final_action, '?')

        pos_str = {'LONG': f'{Colors.GREEN}🟩 LONG 보유{Colors.RESET}',
                   'SHORT': f'{Colors.RED}🟥 SHORT 보유{Colors.RESET}',
                   None: f'{Colors.YELLOW}🟨 무포지션{Colors.RESET}'}.get(self.pos, '?')

        print("\n" + Colors.BOLD + "╔════════════════════ [ 6-Agent MoE 사령관 대시보드 ] ═════════════════════╗" + Colors.RESET)
        print(f" ⏱️ 타임: {timestamp.strftime('%Y-%m-%d %H:%M')} KST | 💰 ETH: ${current_price:,.2f}")

        print("-" * 68)
        print(f" {Colors.CYAN}[ 🧠 6대 파운데이션 AI 앙상블 예측 ]{Colors.RESET}")
        for i, model_name in enumerate(self.MODEL_ORDER):
            pred_dir = f"{Colors.GREEN}▲{Colors.RESET}" if preds[i] > 0 else f"{Colors.RED}▼{Colors.RESET}" if preds[i] < 0 else f"{Colors.YELLOW}-{Colors.RESET}"
            print(f"  • {model_name:<10} : {pred_dir:<5} {confs[i]:.1%}")

        print("-" * 68)
        print(f" {Colors.YELLOW}[ ⚔️ 13대 엘리트 퀀트 시그널 분석 ]{Colors.RESET}")

        def _print_sig_row(k, v, in_rl: bool):
            base_key = k.replace('sig_', '')
            star = f"{Colors.CYAN}★{Colors.RESET}" if in_rl else " "
            interp_tuple = SIG_INTERPRETATIONS.get(base_key, ('LONG', 'SHORT', 'NONE'))
            if v > 0:
                icon = f"{Colors.GREEN}▲{Colors.RESET}"; mc = Colors.GREEN; msg = interp_tuple[0]
            elif v < 0:
                icon = f"{Colors.RED}▼{Colors.RESET}";  mc = Colors.RED;   msg = interp_tuple[1]
            else:
                icon = f"{Colors.YELLOW}─{Colors.RESET}"; mc = Colors.YELLOW; msg = interp_tuple[2]
            print(f" {star}{icon}  {base_key:<18}  (강도: {v:+.2f})")
            print(f"      {mc}{msg}{Colors.RESET}")

        rl_sigs    = sorted([(k, v) for k, v in elite_sigs.items() if k.replace('sig_', '') in _RL_ELITE_KEYS],
                            key=lambda x: abs(x[1]), reverse=True)
        other_sigs = sorted([(k, v) for k, v in elite_sigs.items() if k.replace('sig_', '') not in _RL_ELITE_KEYS],
                            key=lambda x: abs(x[1]), reverse=True)
        for k, v in rl_sigs:
            _print_sig_row(k, v, in_rl=True)
        if other_sigs:
            print(f"  {'─'*64}")
        for k, v in other_sigs:
            _print_sig_row(k, v, in_rl=False)

        print("-" * 68)
        print(f" {Colors.CYAN}[ 🤖 MoE 6-Agent + GatingNet7 독립 판단 ]{Colors.RESET}")
        if self.pos is not None:
            pnl_color = Colors.GREEN if pnl_pct > 0 else Colors.RED
            print(f" ⏳ 보유 캔들: {self.hold_count:<4} | 📈 미실현 수익: {pnl_color}{pnl_pct:+.2f}%{Colors.RESET}")
        print(f" 🌍 시장 레짐: {regime_name:<10} | 📊 현재 포지션: {pos_str}")
        print(f" ⚖️ [엣지 비교] {Colors.GREEN}롱(L): {long_edge:+.3f}{Colors.RESET} vs {Colors.RED}숏(S): {short_edge:+.3f}{Colors.RESET}  Kelly={kelly:.3f}")
        hmm_state = info.get('hmm_state')
        if hmm_state:
            hmm_probs = info.get('hmm_probs', [])
            hmm_color = {
                'bull-trend': Colors.GREEN, 'bear-trend': Colors.RED,
                'hv-chop': Colors.YELLOW, 'lv-range': Colors.CYAN,
            }.get(hmm_state, Colors.RESET)
            probs_str = ' '.join(f'{p:.2f}' for p in hmm_probs) if hmm_probs else ''
            print(f" 🔮 HMM 레짐: {hmm_color}{hmm_state}{Colors.RESET}  [{probs_str}]")
        print(f" 🤖 담당 에이전트: {Colors.CYAN}{active_agent}{Colors.RESET}")
        print(f" 🎯 최종 결단: {action_str}")

        # LLM 분석 출력
        if llm_answer:
            print("-" * 68)
            _rn = llm_regime_name or regime_name
            _t  = llm_time_kst or timestamp
            _print_llm_section(llm_answer, _rn, _t)

        print(Colors.BOLD + "╚════════════════════════════════════════════════════════════════════════╝" + Colors.RESET)


# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher      = BinanceLiveFetcher(limit=2500)
    fe_engine    = FeatureEngineer()
    ensemble     = EnsemblePredictor()
    nf_predictor = NFStatePredictor()
    llm_analyzer = LLMAnalyzer(fetcher.exchange, symbol='ETHUSDT') if LLM_ENABLED else None
    if LLM_ENABLED:
        logger.info("🤖 LLM 모드: ON")
    else:
        logger.info("🤖 LLM 모드: OFF (LLM_ENABLED=0 또는 DISABLE_LLM=1)")

    try:
        bot = MoELiveRouter()
    except Exception as e:
        logger.error(f"❌ MoE 라우터 초기화 실패: {e}")
        return
    sac_router = SACLiveRouter()
    logger.info(f"🤖 SAC 참고 신호: {'ON' if sac_router.available else 'OFF'}")

    # ── MetaRouter 초기화 ──────────────────────────────────────
    meta_router = MetaRouter()
    # 직전 포지션 청산 시점 추적용 (PnL 피드백을 위해)
    _prev_meta_pos: str | None = None

    def _sync_bot_with_meta():
        """RL 입력용 포지션 상태를 MetaRouter 단일 소스로 동기화."""
        bot.pos         = meta_router.pos
        bot.entry_price = meta_router.entry_price
        bot.hold_count  = meta_router.hold_count
        bot.current_equity = meta_router.cur_equity
        bot.peak_equity    = meta_router.peak_equity

    # ── 지도/비지도 허브 초기화 (trading_bot 본문 길이 축소) ──────────
    trend_hub = SupervisedTrendHub(
        xgb_meta_path="data/trend_xgb/trend_xgb.json",
        multitarget_meta_path="data/ensemble/supervised/multi_target_lgbm.json",
        blend_weights=(0.5, 0.5),
    )
    unsup_hub = UnsupervisedRegimeHub()
    logger.info("🤖 SupervisedTrendHub: %s", trend_hub.status())
    logger.info("🧩 %s", unsup_hub.summary_line())

    # ── Polymarket 크라우드 확률 수집기 ────────────────────────
    poly_fetcher = PolymarketFetcher()
    # ── 텔레그램 알림 ──────────────────────────────────────────
    tg_notifier = TelegramNotifier()
    logger.info("📨 텔레그램 알림 조건: 포지션 변화(ENTER/EXIT/FLIP) 발생 시에만 전송")
    logger.info(
        f"📺 출력 모드: {'COMPACT (요약패널 전용)' if COMPACT_MODE else 'STANDARD (대시보드 + 요약패널)'}"
        f" | LLM: {'ON' if LLM_ENABLED else 'OFF'}"
    )

    async def _run_cycle(processed_df, eth_buffer):
        """한 사이클: 에이전트 판단 + MetaRouter 통합 + Polymarket + LLM fire-and-forget."""
        nonlocal _prev_meta_pos

        preds, confs = await ensemble.predict_all_async(processed_df)
        nf_preds     = nf_predictor.predict(processed_df)

        # RL 라우터 입력 포지션은 MetaRouter 상태를 단일 소스로 사용
        _sync_bot_with_meta()
        signal_out = bot.get_signal(processed_df, preds, confs, nf_preds)
        if isinstance(signal_out, (tuple, list)) and len(signal_out) >= 6:
            rl_action, info, elite_sigs, regime, rl_features, rl_pos_dict = signal_out[:6]
        else:
            rl_action, info, elite_sigs, regime = signal_out  # backward-compat
            rl_features, rl_pos_dict = None, None

        sac_info = {
            'available': bool(getattr(sac_router, 'available', False)),
            'action': 0,
            'raw_action': 0.0,
            'leverage': 0.0,
            'score': 0.0,
        }
        if rl_features is not None and rl_pos_dict is not None:
            sac_action, sac_lev, sac_pred_info = sac_router.decide(rl_features, rl_pos_dict)
            if isinstance(sac_pred_info, dict):
                sac_info.update(sac_pred_info)
            sac_info['action'] = int(sac_action)
            sac_info['leverage'] = float(sac_lev)
            sac_info['available'] = bool(getattr(sac_router, 'available', False))

        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price    = float(eth_buffer['close'].iloc[-1])
        regime_name      = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')

        # ── XGBTrendBrain 피처 보강: NF pred/conf 주입 ───────────
        # SyntheticAlpha/VolatilityModel/NewElite는 FeatureEngineer에서 이미 계산됨
        # 외부 모델 예측값(pred_*/conf_*)만 여기서 추가 주입
        _last_idx = processed_df.index[-1]
        for _col, _val in nf_preds.items():   # pred_timesfm, conf_chronos 등 12개
            processed_df.loc[_last_idx, _col] = float(_val)
        _pm, _cm = _compute_mdjd(processed_df.iloc[-1], processed_df)
        processed_df.loc[_last_idx, 'pred_mdjd'] = _pm
        processed_df.loc[_last_idx, 'conf_mdjd'] = _cm

        # ── TrendContextBrain 4h 추세 추론 ───────────────────────
        trend_signal = None
        if trend_hub.available:
            try:
                # 학습 피처가 포함된 processed_df를 넣어 추세 필터 신뢰도 확보
                trend_signal = trend_hub.predict_from_df(processed_df)
            except Exception as e:
                logger.warning(f"SupervisedTrendHub 추론 실패: {e}")

        # ── MetaRouter 신호 융합 ──────────────────────────────
        prev_meta_pos = _prev_meta_pos
        meta_result = meta_router.fuse(
            rl_action     = rl_action,
            rl_info       = info,
            regime        = regime,
            current_price = current_price,
            trend_signal  = trend_signal,
            garch_vol_z   = float(processed_df.iloc[-1].get('garch_vol_z', 0.0)),
        )

        # 직전 사이클에 포지션이 있었다가 이번에 청산됐으면 PnL 피드백
        if _prev_meta_pos is not None and meta_router.pos is None:
            meta_router.record_outcome(meta_router.cur_equity - 1.0)

        # ── 텔레그램 알림: 포지션이 바뀐 경우만 (ENTER / EXIT / FLIP) ──
        _new_pos = meta_router.pos
        if prev_meta_pos != _new_pos:
            if prev_meta_pos is None and _new_pos:
                _tg_code = f"ENTER_{_new_pos}"
            elif prev_meta_pos and _new_pos is None:
                _tg_code = f"EXIT_{prev_meta_pos}"
            elif prev_meta_pos and _new_pos:
                _tg_code = f"FLIP_{prev_meta_pos}_TO_{_new_pos}"
            else:
                _tg_code = None
            if _tg_code:
                asyncio.create_task(tg_notifier.notify(
                    _tg_trade_msg(_tg_code, current_price, current_time_kst,
                                  regime_name, meta_result)
                ))

        _prev_meta_pos = _new_pos

        # 출력 직전에도 동기화해 대시보드 포지션 표기를 일치시킴
        _sync_bot_with_meta()

        # ── Polymarket 크라우드 확률 조회 (비동기, 5초 타임아웃) ─
        try:
            poly_data = await asyncio.wait_for(poly_fetcher.fetch(), timeout=5.0)
        except asyncio.TimeoutError:
            poly_data = {'btc': None, 'eth': None, 'ok': False}
        except Exception as e:
            logger.debug(f"Polymarket 조회 예외: {e}")
            poly_data = {'btc': None, 'eth': None, 'ok': False}

        # 이전 사이클 LLM 결과 수거 (요약 패널 전용)
        prev_answer = llm_analyzer.collect() if llm_analyzer is not None else LLM_DISABLED_SENTINEL

        # 요약 결과 먼저 출력 (핵심 의사결정 선노출)
        _print_final_trade_summary(
            timestamp_kst=current_time_kst,
            current_price=current_price,
            regime_name=regime_name,
            rl_action=rl_action,
            rl_info=info,
            meta_result=meta_result,
            prev_llm_answer=prev_answer,
            prev_pos=prev_meta_pos,
            cur_pos=meta_router.pos,
            poly_data=poly_data,
            sac_info=sac_info,
        )

        # 출력: COMPACT_MODE면 요약 패널만, 아니면 설명(상세) 로그를 아래에 출력
        if not COMPACT_MODE:
            bot.print_dashboard(
                current_price, preds, confs, rl_action, info, regime, elite_sigs, current_time_kst,
                llm_answer=None,
            )
            meta_router.print_meta_dashboard(meta_result, current_price)

        # 이번 사이클 LLM 분석 시작 (백그라운드, 블로킹 없음)
        if llm_analyzer is not None:
            llm_analyzer.fire(processed_df, current_price, elite_sigs, regime_name)

    try:
        if use_local:
            eth_buffer, btc_buffer = fetcher.load_local_data()
        else:
            logger.info("초기 캔들 데이터 수집 중...")
            eth_buffer, btc_buffer = await fetcher.fetch_initial_data()

        if eth_buffer is None: return

        processed_df = fe_engine.process(eth_buffer, btc_buffer)
        await _run_cycle(processed_df, eth_buffer)

        first_run = True
        while not use_local:
            if not first_run:
                now = time.time()
                wait_sec = int(max(0, (now - (now % 300) + 300 + 2) - now))
                for r in range(wait_sec, 0, -1):
                    sys.stdout.write(f"\r{Colors.CYAN}⏳ 다음 5분봉까지 대기 중... ({r}초 남음)   {Colors.RESET}")
                    sys.stdout.flush()
                    await asyncio.sleep(1)

                print()
                logger.info("🔄 최신 캔들 데이터를 갱신합니다.")
                new_eth, new_btc = await fetcher.fetch_latest_patch()
                eth_buffer = pd.concat([eth_buffer, new_eth]).drop_duplicates('timestamp').tail(2500)
                btc_buffer = pd.concat([btc_buffer, new_btc]).drop_duplicates('timestamp').tail(2500)
            else:
                logger.info(f"{Colors.GREEN}🚀 봇 실시간 롤링 가동 시작!{Colors.RESET}")
                first_run = False

            processed_df = fe_engine.process(eth_buffer, btc_buffer)
            await _run_cycle(processed_df, eth_buffer)

    finally:
        await fetcher.exchange.close()


if __name__ == "__main__":
    asyncio.run(main(use_local=False))
