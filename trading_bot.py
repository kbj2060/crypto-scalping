import os
import sys
import asyncio
import time
import logging
import gc
import json
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
from ensemble.train_trend import TrendContextBrain  # TrendSignal 타입이 이 모듈에 정의됨
from ensemble.trend_xgb.trend_xgb_model import XGBTrendBrain
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


class Colors:
    GREEN, RED, YELLOW, CYAN, RESET, BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'

if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LiveBot")


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
            mappings = [(0, 'sumOpenInterestValue', 'sum_open_interest_value'), (1, 'longShortRatio', 'sum_toptrader_long_short_ratio'), (3, 'longShortRatio', 'count_long_short_ratio'), (5, 'fundingRate', 'last_funding_rate')]
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
    """data/ridge_model.pkl 로드 후 매 틱마다 pred_ridge / conf_ridge 반환."""

    _RIDGE_FEATURES = [
        'log_return', 'rsi', 'macd_hist', 'bb_width', 'volatility_z',
        'garman_klass_vol', 'last_funding_rate', 'oi_change_rate',
        'net_taker_ratio', 'taker_acceleration', 'trade_intensity',
        'big_trade_ratio', 'hurst_12', 'hurst_48', 'hurst_288',
        'realized_vol_ratio', 'amihud_illiquidity_z',
    ]

    def __init__(self, model_path: str = 'data/ridge_model.pkl'):
        self._ridge   = None
        self._scaler  = None
        self._feats   = self._RIDGE_FEATURES
        self._abs_buf = deque(maxlen=500)   # rolling MAD 추적용

        if os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    pkg = pickle.load(f)
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
# 2-C. 신경망 state 구성용 NF/TTM 예측 (RL/LS 공용)
# ════════════════════════════════════════════════════════════════
class NFStatePredictor:
    """RL_STATE_PRED ∪ LS_STATE_PRED 전체 커버 (싱글 로드, 공유 사용)"""

    def __init__(self):
        # NF 4종은 UnifiedNFForecaster 싱글톤이므로 중복 로드 없음
        self.models = {
            'ttm':      TTMForecaster(),
            'nhits':    NHITSForecaster(),
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
# 2-D. LLM 시장 분석기 (Llama-4 Scout via Cerebras)
# ════════════════════════════════════════════════════════════════
class LLMAnalyzer:
    """오더북 + 체결 흐름 + 선물 지표를 수집해 Llama-4 Scout(Cerebras)에 방향성 분석 요청."""

    # cerebras-cloud-sdk 공식 클라이언트 사용 (pip install cerebras-cloud-sdk)
    # urllib 직접 호출 시 Cloudflare Browser Integrity Check(1010)에 차단됨
    MODEL   = "qwen-3-235b-a22b-instruct-2507"
    API_KEY = os.getenv("CEREBRAS_API_KEY", "")

    def __init__(self, exchange, symbol: str = 'ETHUSDT'):
        self.exchange    = exchange
        self.symbol      = symbol
        self._fail_count = 0
        self._pending_task: asyncio.Task | None = None   # fire-and-forget 태스크
        self._pending_meta: dict = {}                    # 태스크 생성 당시 regime_name 등
        if not self.API_KEY:
            logger.warning("⚠️ CEREBRAS_API_KEY 환경변수가 설정되지 않았습니다. LLM 분석이 비활성화됩니다.")

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
        ctx = {
            'symbol':             f'{self.symbol} Perp',
            'price':              price,
            'regime':             regime,
            # 최근 12캔들 시계열 (5분봉 = 1시간)
            'recent_12_candles':  candle_series,
            'orderbook': {
                'imbalance_pct': orderbook.get('bid_ask_imbalance_pct', 0),
                'spread':        orderbook.get('spread_usdt', 0),
                'bids5':         orderbook.get('top5_bids', []),
                'asks5':         orderbook.get('top5_asks', []),
            },
            'trades_100': {
                'buy_pct':    trades.get('buyer_aggressor_pct', 50),
                'dominant':   trades.get('dominant_side', 'NEUTRAL'),
            },
            'futures': {
                'funding':    feats['funding_rate'],
                'oi_chg':     feats['oi_change_rate'],
                'ls_top':     feats['top_trader_ls_ratio'],
                'ls_global':  feats['global_ls_ratio'],
                'jump':       feats['jump_flag'],
                'tail':       feats['evt_tail_flag'],
                'ou_halflife':feats['ou_halflife_bars'],
                'trend_1h':   feats['mtf_trend_1h'],
                'trend_4h':   feats['mtf_trend_4h'],
            }
        }

        return (
            "ETH/USDT 선물 퀀트 분석가. 아래 JSON으로 향후 30분 방향성을 판단하세요.\n"
            "candle 필드: ts=시각, rsi, macd_h, vol_z=변동성Z, net_taker(>0.5=매수우위), "
            "fund_pres=펀딩압력, smart_mf=스마트머니플로우, squeeze=청산스퀴즈, regime=레짐\n"
            "imbalance_pct>0=매수우위 / buy_pct>55%=매수압력 / funding극단양수=롱과열 / "
            f"```json\n{json.dumps(ctx, ensure_ascii=False, separators=(',', ':'))}\n```\n\n"
            "## 방향성: [LONG/SHORT/NEUTRAL] (신뢰도: 낮음/보통/높음)\n"
            "근거 없이 결과만 얘기해줘 .\n"
        )

    # ── Cerebras 동기 호출 (executor에서 실행) ───────────────────
    # cerebras-cloud-sdk 사용: urllib 대신 공식 SDK → Cloudflare 1010 차단 우회
    # 설치: pip install cerebras-cloud-sdk
    def _call_cerebras_sync(self, prompt: str) -> str:
        if not self.API_KEY:
            return '[Cerebras API 키 없음 — 환경변수 CEREBRAS_API_KEY 를 설정하세요]'
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            return '[cerebras-cloud-sdk 미설치 — pip install cerebras-cloud-sdk 실행 후 재시작]'
        try:
            client = Cerebras(api_key=self.API_KEY)
            resp = client.chat.completions.create(
                model=self.MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            self._fail_count = 0
            raw = resp.choices[0].message.content.strip()
            return raw if raw else '[LLM 응답 없음]'
        except Exception as e:
            self._fail_count += 1
            return f'[Cerebras 오류 ({self._fail_count}회): {e}]'

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
        return await loop.run_in_executor(None, self._call_cerebras_sync, prompt)

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
    print(Colors.BOLD + "╔══════════════════ [ 🤖 Llama-4 Scout LLM 시장 분석 ] ══════════════════╗" + Colors.RESET)
    print(f" 레짐: {regime_name} | 시간 : {current_time_kst.strftime('%Y-%m-%d %H:%M')} KST ")
    print("─" * 70)
    for line in answer.strip().splitlines():
        print(f" {line}")
    print(Colors.BOLD + "╚═══════════════════════════════════════════════════════════════════════╝" + Colors.RESET)


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


# ════════════════════════════════════════════════════════════════
# 3-B. MetaRouter — RL 4-Agent + LS 2-Agent 신호 통합 융합 레이어
# ════════════════════════════════════════════════════════════════
# [설계 원칙]
# - MoELiveRouter / LSLiveRouter 코드 변경 없음
# - _run_cycle() 에서 meta_router.fuse() 한 줄로 두 신호를 통합
# - 포지션은 MetaRouter가 단일 소스로 관리
#
# [충돌 처리 행렬]
#   RL\LS  |  관망(0)           |  Long(1)          |  Short(2)
#   -------+--------------------+-------------------+--------------------
#   관망(0)|  관망              |  LS score 임계값  |  LS score 임계값
#   Long(1)|  RL score 임계값  |  Long (합의)      |  score 비교 or 관망
#  Short(2)|  RL score 임계값  |  score 비교 or 관망| Short (합의)

# ── 튜닝 상수 ────────────────────────────────────────────────────
_META_CONSENSUS_BOOST  = 1.15   # 합의 시 Kelly 부스트
_META_SOLO_RL_THRESH   = 0.25   # RL 단독 진입 score 최솟값
_META_SOLO_LS_THRESH   = 0.20   # LS 단독 진입 score 최솟값
_META_CONFLICT_MARGIN  = 0.10   # 충돌 시 score 차이가 이 미만이면 관망 강제
_META_HISTORY_N        = 30     # 동적 가중치 추적 윈도우

# 레짐별 RL 우선순위 배율 (bull/bear → RL 특화, chop/whip → LS 민감)
_REGIME_RL_WEIGHT = {
    'bull': 1.20, 'bear': 1.20, 'normal': 1.00,
    'chop': 0.80, 'whipsaw': 0.75,
}


def _meta_rl_score(rl_action: int, rl_info: dict, regime: dict) -> float:
    """GatingRouter7 info 구조로 RL 신호 강도 계산 (0~1)."""
    if rl_action == 0:
        return 0.0
    kelly      = float(rl_info.get('kelly', 0.0))
    long_edge  = float(rl_info.get('long_edge',  0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    edge_gap   = abs(long_edge - short_edge)
    active     = next((k.replace('regime_', '') for k, v in regime.items() if v == 1.0), 'normal')
    regime_w   = _REGIME_RL_WEIGHT.get(active, 1.0)
    raw        = float(np.tanh(edge_gap * 5.0)) * kelly * regime_w
    return float(np.clip(raw, 0.0, 1.0))


def _meta_ls_score(ls_action: int, ls_info: dict) -> float:
    """HCRouter info 구조로 LS 신호 강도 계산 (0~1).
    HCRouter.decide()가 반환하는 info 키:
      진입 시: adv(방향 advantage), consensus(가중 합의)
      관망 시: adv_L, adv_S, ml_cons
    """
    if ls_action == 0:
        return 0.0
    adv_l     = float(ls_info.get('adv_L', ls_info.get('adv', 0.0)))
    adv_s     = float(ls_info.get('adv_S', 0.0))
    consensus = abs(float(ls_info.get('consensus', ls_info.get('ml_cons', 0.0))))
    if ls_action == 1:
        adv_gap = adv_l - adv_s if adv_s != 0.0 else abs(adv_l)
    else:
        adv_gap = adv_s - adv_l if adv_l != 0.0 else abs(adv_s)
    raw = float(np.tanh(abs(adv_gap) * 3.0)) * (0.5 + 0.5 * consensus)
    return float(np.clip(raw, 0.0, 1.0))


class MetaRouter:
    """RL MoE 4-Agent + LS 2-Agent 신호를 받아 단일 최종 액션을 결정."""

    def __init__(self):
        # 동적 가중치 추적 (승률 기반)
        self._rl_score_acc  = 1.0
        self._ls_score_acc  = 1.0
        self._history       = []        # {'rl':int,'ls':int,'final':int,'src':str,'outcome':None}

        # MetaRouter 자체 포지션 상태
        self.pos:          str | None = None
        self.entry_price:  float      = 0.0
        self.hold_count:   int        = 0
        self.peak_equity:  float      = 1.0
        self.cur_equity:   float      = 1.0

    # ── 동적 가중치 ───────────────────────────────────────────────
    def _get_weights(self):
        total  = self._rl_score_acc + self._ls_score_acc + 1e-8
        rl_w   = float(np.clip(2.0 * self._rl_score_acc / total, 0.40, 1.60))
        ls_w   = float(np.clip(2.0 * self._ls_score_acc / total, 0.40, 1.60))
        return rl_w, ls_w

    def record_outcome(self, realized_pnl_pct: float):
        """포지션 청산 후 실현 PnL을 피드백해 동적 가중치를 보정."""
        if not self._history:
            return
        rec  = self._history[-1]
        correct = realized_pnl_pct > 0.0
        if rec['src'] in ('CONSENSUS', 'RL_WIN', 'RL_SOLO'):
            self._rl_score_acc += 1.0 if correct else -0.5
            self._rl_score_acc  = max(0.1, self._rl_score_acc)
        if rec['src'] in ('CONSENSUS', 'LS_WIN', 'LS_SOLO'):
            self._ls_score_acc += 1.0 if correct else -0.5
            self._ls_score_acc  = max(0.1, self._ls_score_acc)

    # ── 메인 진입점 ───────────────────────────────────────────────
    def fuse(self, rl_action: int, rl_info: dict,
             ls_action: int, ls_info: dict,
             regime: dict, current_price: float = 0.0,
             trend_signal=None) -> dict:
        """
        Returns dict:
          final_action  : 0/1/2
          unified_kelly : 0~1
          source        : 'CONSENSUS'|'RL_SOLO'|'LS_SOLO'|'RL_WIN'|'LS_WIN'|'FLAT'|'HOLD'
          conflict_type : 'AGREE'|'CONFLICT'|'RL_SOLO'|'LS_SOLO'|'BOTH_FLAT'
          rl_score, ls_score, meta_score, rl_weight, ls_weight
          trend_signal  : TrendSignal.to_arbiter_dict() 또는 None
          trend_veto    : 적용된 trend filter 사유 문자열 또는 None
        """
        rl_w, ls_w = self._get_weights()
        rl_score   = _meta_rl_score(rl_action, rl_info, regime) * rl_w
        ls_score   = _meta_ls_score(ls_action, ls_info) * ls_w

        final_action, source, conflict_type = self._decide(
            rl_action, ls_action, rl_score, ls_score
        )
        unified_kelly = self._calc_kelly(
            final_action, source, rl_info, rl_score, ls_score
        )

        # ── TrendContextBrain 거버넌스 레이어 ─────────────────────
        # 4h 추세가 5분봉 진입 신호를 거부(VETO)하거나 Kelly를 조정
        trend_veto = None
        if trend_signal is not None and final_action != 0:
            final_action, unified_kelly, trend_veto = self._apply_trend_filter(
                final_action, unified_kelly, trend_signal
            )

        if source == 'CONSENSUS':
            meta_score = min((rl_score + ls_score) / 2.0 * _META_CONSENSUS_BOOST, 1.0)
        elif source in ('RL_WIN', 'RL_SOLO'):
            meta_score = rl_score
        elif source in ('LS_WIN', 'LS_SOLO'):
            meta_score = ls_score
        else:
            meta_score = 0.0

        # trend veto 로 action=0 됐으면 source 갱신
        if final_action == 0 and trend_veto is not None:
            source = trend_veto

        self._update_pos(final_action, current_price)
        self._history.append({'rl': rl_action, 'ls': ls_action,
                              'final': final_action, 'src': source, 'outcome': None})
        if len(self._history) > _META_HISTORY_N * 2:
            self._history = self._history[-_META_HISTORY_N:]

        return {
            'final_action':  final_action,
            'unified_kelly': unified_kelly,
            'source':        source,
            'conflict_type': conflict_type,
            'rl_score':      rl_score,
            'ls_score':      ls_score,
            'meta_score':    meta_score,
            'rl_action':     rl_action,
            'ls_action':     ls_action,
            'rl_weight':     rl_w,
            'ls_weight':     ls_w,
            'trend_signal':  trend_signal.to_arbiter_dict() if trend_signal is not None else None,
            'trend_veto':    trend_veto,
        }

    # ── TrendContextBrain 거버넌스 ────────────────────────────────
    def _apply_trend_filter(self, action: int, kelly: float, trend_signal) -> tuple:
        """4h 추세 기반 진입 거부 / Kelly 조정.

        VETO_STRENGTH  : 이 이상 강한 역방향 추세 → 진입 거부 (action=0)
        BOOST_STRENGTH : 이 이상 동방향 추세 → Kelly 부스트 (+25%)
        CHOP_STRENGTH  : FLAT 추세 강도 → Kelly 축소 (×0.8)
        REV_VETO_PROB  : 반전 확률 이 이상 → Kelly 축소 (×0.6)
        """
        VETO_STRENGTH  = 0.55
        BOOST_STRENGTH = 0.35
        CHOP_STRENGTH  = 0.30
        REV_VETO_PROB  = 0.60

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

    # ── 4단계 의사결정 ────────────────────────────────────────────
    def _decide(self, rl_action, ls_action, rl_score, ls_score):
        # ① 양쪽 모두 관망
        if rl_action == 0 and ls_action == 0:
            return 0, 'FLAT', 'BOTH_FLAT'

        # ② 합의 (같은 방향 진입)
        if rl_action != 0 and rl_action == ls_action:
            return rl_action, 'CONSENSUS', 'AGREE'

        # ③ 보유 중 → 청산·역방향 신호가 하나라도 있으면 청산 우선
        if self.pos is not None:
            pos_is_long  = (self.pos == 'LONG')
            pos_is_short = (self.pos == 'SHORT')
            rl_close  = (rl_action == 0)
            ls_close  = (ls_action == 0)
            rl_rev    = (pos_is_long and rl_action == 2) or (pos_is_short and rl_action == 1)
            ls_rev    = (pos_is_long and ls_action == 2) or (pos_is_short and ls_action == 1)
            if rl_close or ls_close or rl_rev or ls_rev:
                return 0, 'HOLD', 'AGREE'

        # ④ 단독 신호 (한쪽만 진입 의사)
        if rl_action != 0 and ls_action == 0:
            if rl_score >= _META_SOLO_RL_THRESH:
                return rl_action, 'RL_SOLO', 'RL_SOLO'
            return 0, 'FLAT', 'RL_SOLO'

        if ls_action != 0 and rl_action == 0:
            if ls_score >= _META_SOLO_LS_THRESH:
                return ls_action, 'LS_SOLO', 'LS_SOLO'
            return 0, 'FLAT', 'LS_SOLO'

        # ⑤ 충돌 (방향 반대)
        diff = rl_score - ls_score
        if abs(diff) < _META_CONFLICT_MARGIN:
            return 0, 'FLAT', 'CONFLICT'
        if diff > 0:
            return rl_action, 'RL_WIN', 'CONFLICT'
        return ls_action, 'LS_WIN', 'CONFLICT'

    def _calc_kelly(self, final_action, source, rl_info, rl_score, ls_score):
        if final_action == 0:
            return 0.0
        rl_kelly = float(rl_info.get('kelly', 0.0))
        ls_kelly = min(ls_score * 0.8, 1.0)
        if source == 'CONSENSUS':
            return float(np.clip((rl_kelly + ls_kelly) / 2.0 * _META_CONSENSUS_BOOST, 0.0, 1.0))
        if source in ('RL_WIN', 'RL_SOLO'):
            return float(np.clip(rl_kelly, 0.0, 1.0))
        return float(np.clip(ls_kelly, 0.0, 1.0))

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
        ct     = result['conflict_type']
        fa     = result['final_action']
        action_str = {0: f'{C.YELLOW}🟨 관망{C.RESET}',
                      1: f'{C.GREEN}🟩 LONG{C.RESET}',
                      2: f'{C.RED}🟥 SHORT{C.RESET}'}.get(fa, '?')
        rl_a_str = {0:'관망', 1:'LONG', 2:'SHORT'}.get(result['rl_action'], '?')
        ls_a_str = {0:'관망', 1:'LONG', 2:'SHORT'}.get(result['ls_action'], '?')

        # 충돌유형 색상
        ct_color = C.GREEN if ct == 'AGREE' else (C.RED if ct == 'CONFLICT' else C.YELLOW)
        unr = self.unrealized_pnl(current_price)
        unr_color = C.GREEN if unr > 0 else (C.RED if unr < 0 else C.YELLOW)

        print(C.BOLD + "╔══════════ [ ⚡ MetaRouter 통합 판단 ] ══════════╗" + C.RESET)
        print(f"  RL 신호: {rl_a_str:<5}  score={result['rl_score']:.3f}  weight={result['rl_weight']:.2f}")
        print(f"  LS 신호: {ls_a_str:<5}  score={result['ls_score']:.3f}  weight={result['ls_weight']:.2f}")
        print(f"  충돌유형: {ct_color}{ct}{C.RESET}   source: {C.CYAN}{src}{C.RESET}")
        print(f"  ▶ 최종결정: {action_str}   Kelly={result['unified_kelly']:.3f}   meta_score={result['meta_score']:.3f}")
        if self.pos is not None:
            print(f"  포지션: {self.pos}  진입가={self.entry_price:.2f}  "
                  f"미실현={unr_color}{unr:+.2f}%{C.RESET}  보유={self.hold_count}봉")
        ts = result.get('trend_signal')
        tv = result.get('trend_veto')
        if ts is not None:
            dir_map = {0: f'{C.RED}↓DOWN{C.RESET}', 1: f'{C.YELLOW}→FLAT{C.RESET}', 2: f'{C.GREEN}↑UP{C.RESET}'}
            dir_str = dir_map.get(ts['trend_dir'], '?')
            veto_str = f'  [{C.RED}{tv}{C.RESET}]' if tv else ''
            print(f"  4h추세: {dir_str}  str={ts['strength']:.2f}  rev={ts['rev_prob']:.2f}{veto_str}")
        print(C.BOLD + "╚══════════════════════════════════════════════════╝" + C.RESET)


# ════════════════════════════════════════════════════════════════
# 3-B. LS 2-Agent 라우터 (롱돌이/숏돌이) — HCRouter 기반
# ════════════════════════════════════════════════════════════════
class LSLiveRouter:
    def __init__(self, model_path='data/ensemble/ckpt/best_ls_agents.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.elite_extractor = EliteSignals()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LS 모델을 찾을 수 없습니다: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # train_ls_agent.py 기준:
        #   RobustIQN(STACKED_STATE_DIM, action_dim=2, raw_state_dim=STATE_DIM)
        #   STATE_DIM = FEATURE_DIM + 4  (pos_features 4개 — entry_price 없음)
        #   FEATURE_DIM = 7+7+3+4+7+5+9 = 42  →  STATE_DIM = 46
        #   STACKED_STATE_DIM = STATE_DIM * STACK_N = 46 * 4 = 184
        from ensemble.train_ls_agent import (
            RobustIQN as LSRobustIQN,
            HCRouter,
            STACKED_STATE_DIM as _LS_STACKED,
            STATE_DIM         as _LS_STATE,
        )

        def _load(key):
            m = LSRobustIQN(_LS_STACKED, 2, raw_state_dim=_LS_STATE).to(self.device)
            m.load_state_dict(ckpt[key])
            m.eval()
            return m

        model_long  = _load('model_long')
        model_short = _load('model_short')

        # HCRouter — GatingNet 없음 (Hysteresis-Consensus 방식)
        self.trader = HCRouter(model_long, model_short, self.device)
        # 라이브 추론 시 eval() 명시 (학습 루프와 달리 재전환 불필요)
        model_long.eval()
        model_short.eval()

        self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        self.peak_equity, self.current_equity = 1.0, 1.0
        logger.info(f"✅ {Colors.GREEN}LS 2-Agent HCRouter 탑재 완료{Colors.RESET} "
                    f"(state_dim={_LS_STATE}, stacked={_LS_STACKED})")

    def get_signal(self, processed_df: pd.DataFrame, preds, confs, nf_preds: dict):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        smf_std  = processed_df['smart_money_flow'].std() if 'smart_money_flow' in processed_df.columns else 1.0

        cur_market  = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs  = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)

        features = {}

        # LS STATE_PRED/CONF: train_ls_agent.py 와 동일 7+7 컬럼
        # ['pred_timesfm','pred_chronos','pred_ttm','pred_patchtst','pred_tide','pred_mdjd','pred_ridge']
        pred_mdjd, conf_mdjd = _compute_mdjd(last_row, processed_df)
        for col in LS_STATE_PRED:
            if col == 'pred_mdjd':
                features[col] = pred_mdjd
            else:
                features[col] = float(nf_preds.get(col, 0.0))
        for col in LS_STATE_CONF:
            if col == 'conf_mdjd':
                features[col] = conf_mdjd
            else:
                features[col] = float(nf_preds.get(col, 0.5))

        # LS STATE_ELITE: ['sig_orderblock','sig_ai_squeeze','sig_whale','sig_oi_divergence']
        for col in LS_STATE_ELITE:
            features[col] = float(elite_sigs.get(col, 0.0))

        # LS STATE_ALPHA: ['hour_cos','breakout_strength','fvg_dist','cvp_poc_dist','session_us','oi_change_rate','cvp_volume_imbalance']
        for col in LS_STATE_ALPHA:
            features[col] = float(last_row.get(col, 0.0))

        # Regime
        features.update(_compute_regime(processed_df))

        # LS STATE_SYNTH: ['fcsz','mta_funding','ofti','vebr','cada','wpad','svps','kel','mtmb']
        for col in LS_STATE_SYNTH:
            features[col] = float(last_row.get(col, 0.0))

        features['close'] = float(last_row['close'])

        # 미실현 손익 계산
        unr = 0.0
        if self.pos is not None:
            cp  = float(last_row['close'])
            unr = (cp - self.entry_price) / self.entry_price if self.pos == 'LONG' else (self.entry_price - cp) / self.entry_price
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity: self.peak_equity = self.current_equity
        else:
            self.current_equity = self.peak_equity = 1.0

        # HCRouter._state_tensor / TradingEnv._build_state 와 동일한 정규화
        # pos_features 4개: [pos_flag, tanh(unr/0.02), clip(mdd/0.05), hold/144]
        # entry_price 항목 없음 — LS TradingEnv._build_state 참조
        pos_dict = {
            'type': self.pos,
            'unrealized': float(np.tanh(unr / 0.02)),
            'mdd': float(np.clip(min((self.current_equity / self.peak_equity) - 1.0, 0.0) / 0.05, -1.0, 1.0)),
            'hold_norm': min(self.hold_count / 144.0, 1.0),
        }

        final_action, _, info = self.trader.decide(features, pos_dict)

        # 포지션 상태 갱신
        # HCRouter: action=1 → LONG 진입/유지, action=2 → SHORT 진입/유지, action=0 → 청산/관망
        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'LONG', float(last_row['close']), 0
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'SHORT', float(last_row['close']), 0
        elif final_action == 0 and self.pos is not None:
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        elif self.pos is not None:
            self.hold_count += 1

        pnl_pct = (self.current_equity - 1.0) * 100
        return final_action, info, pnl_pct


# ════════════════════════════════════════════════════════════════
# 3-C. MoE 4-Agent 라우터 — RL_STATE_DIM=45
# ════════════════════════════════════════════════════════════════
class MoELiveRouter:
    # 1. 클래스 변수 정의 (MODEL_ORDER 에러 해결 핵심)
    MODEL_ORDER = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']

    def __init__(self, model_path='data/ensemble/ckpt/best_rl_agents.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 퀀트 시그널 추출기 초기화
        from strategies.elite_builder import EliteSignals
        self.elite_extractor = EliteSignals()

        # v7 상수 및 모델 로드
        from ensemble.train_rl_agent import (
            RobustIQN as RL_RobustIQN,
            GatingRouter7, GatingNet7,
            STATE_DIM as RL_STATE_DIM,
            STACKED_STATE_DIM as RL_STACKED_STATE_DIM
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 경로를 확인하십시오: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        def _load(key):
            # 훈련 체크포인트 규격(2-Action)에 맞춤
            m = RL_RobustIQN(RL_STACKED_STATE_DIM, 2, raw_state_dim=RL_STATE_DIM).to(self.device)
            if key in ckpt:
                m.load_state_dict(ckpt[key])
            m.eval()
            return m

        # 6-Agent + 1-Flat(GatingNet7) 매핑
        agent_names = ['bull', 'bear', 'chop_long', 'chop_short', 'normal_long', 'normal_short']
        models_dict = {name: _load(f'model_{name}') for name in agent_names}
        
        gating_net = GatingNet7(RL_STACKED_STATE_DIM).to(self.device)
        if 'gating_net' in ckpt:
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

        features = {}

        # RL_STATE_PRED: ['pred_timesfm','pred_chronos','pred_ttm','pred_patchtst','pred_tide','pred_mdjd','pred_ridge']
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

        # RL_STATE_ELITE: ['sig_orderblock','sig_ai_squeeze','sig_whale','sig_oi_divergence']
        for col in RL_STATE_ELITE:
            features[col] = float(elite_sigs.get(col, 0.0))

        # RL_STATE_ALPHA: ['hour_cos','breakout_strength','fvg_dist','cvp_poc_dist','session_us','oi_change_rate','cvp_volume_imbalance']
        for col in RL_STATE_ALPHA:
            features[col] = float(last_row.get(col, 0.0))

        # Regime
        regime = _compute_regime(processed_df)
        features.update(regime)

        # RL_STATE_SYNTH: ['fcsz','mta_funding','ofti','vebr','cada','wpad','svps','kel','mtmb']
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

        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'LONG', float(last_row['close']), 0
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = 'SHORT', float(last_row['close']), 0
        elif final_action == 0 and self.pos is not None:
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
        elif self.pos is not None:
            self.hold_count += 1

        return final_action, info, elite_sigs, regime

    def print_dashboard(self, current_price, preds, confs, final_action, info, regime, elite_sigs, timestamp,
                        ls_action=None, ls_info=None, ls_pnl=None, ls_pos=None,
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

        print("\n" + Colors.BOLD + "╔════════════════════ [ 12-Agent MoE 사령관 대시보드 ] ════════════════════╗" + Colors.RESET)
        print(f" ⏱️ 타임: {timestamp.strftime('%Y-%m-%d %H:%M')} KST | 💰 ETH: ${current_price:,.2f}")

        print("-" * 68)
        print(f" {Colors.CYAN}[ 🧠 6대 파운데이션 AI 앙상블 예측 ]{Colors.RESET}")
        for i, model_name in enumerate(self.MODEL_ORDER):
            pred_dir = f"{Colors.GREEN}상승(L){Colors.RESET}" if preds[i] > 0 else f"{Colors.RED}하락(S){Colors.RESET}" if preds[i] < 0 else f"{Colors.YELLOW}중립(-){Colors.RESET}"
            print(f"  • {model_name:<10} : {pred_dir:<15} (신뢰도: {confs[i]:.1%})")

        print("-" * 68)
        print(f" {Colors.YELLOW}[ ⚔️ 13대 엘리트 퀀트 시그널 분석 ]{Colors.RESET}")

        sig_interpretations = {
            'whale':            ('고래 매수 다이버전스 포착', '고래 매도 다이버전스 포착', '고래 움직임 관망'),
            'liq_squeeze':      ('숏 청산 스퀴즈 위협 (롱 반등)', '롱 청산 스퀴즈 위협 (숏 반등)', '청산 스퀴즈 위험 낮음'),
            'net_taker':        ('시장가 매수 쏠림 강함', '시장가 매도 쏠림 강함', '시장가 매수/매도 균형'),
            'orderblock':       ('지지 매물대(FVG) 강한 반등', '저항 매물대(FVG) 강한 거절', '주요 오더블록 미형성'),
            'hurst_ofi':        ('프랙탈 상승 오더플로우 지속', '프랙탈 하락 오더플로우 지속', '랜덤워크/혼조세 (방향성 없음)'),
            'funding_cascade':  ('펀딩비 극단적 음수 (숏 청산대기)', '펀딩비 극단적 양수 (롱 청산대기)', '펀딩비/미결제약정 안정적'),
            'multifractal':     ('장기 노이즈 캔슬링: 상승', '장기 노이즈 캔슬링: 하락', '추세 노이즈 상쇄됨'),
            'cluster_fib':      ('피보나치+CVP 클러스터 지지', '피보나치+CVP 클러스터 저항', '주요 방어선 부재'),
            'oi_divergence':    ('하락 중 OI 증가 (숏 스퀴즈 잠재)', '상승 중 OI 증가 (롱 스퀴즈 잠재)', 'OI와 가격 동조화'),
            'top_trader_squeeze': ('탑트레이더 숏 쏠림 (상승폭발 대기)', '탑트레이더 롱 쏠림 (하락폭발 대기)', '탑트레이더 포지션 균형'),
            'btc_corr_breakout': ('BTC 커플링 이탈 (독자 상승)', 'BTC 커플링 이탈 (독자 하락)', 'BTC 방향성 강력 동조'),
            'ai_squeeze':       ('변동성 응축 후 상방 폭발', '변동성 응축 후 하방 폭발', '변동성 평이함 (응축 없음)'),
            'vp_gravity':       ('POC(매물대) 하단 이탈 후 탄성 반등', 'POC(매물대) 상단 이탈 후 탄성 하락', '최다 매물대(POC) 부근 체류'),
        }

        sorted_elite_sigs = sorted(elite_sigs.items(), key=lambda item: abs(item[1]), reverse=True)
        for k, v in sorted_elite_sigs:
            base_key = k.replace('sig_', '')
            if base_key in sig_interpretations:
                msg_long, msg_short, msg_none = sig_interpretations[base_key]
                if v > 0:
                    interp = f"{Colors.GREEN}{msg_long:<25}{Colors.RESET}"; icon = f"{Colors.GREEN}▲{Colors.RESET}"
                elif v < 0:
                    interp = f"{Colors.RED}{msg_short:<25}{Colors.RESET}";  icon = f"{Colors.RED}▼{Colors.RESET}"
                else:
                    interp = f"{Colors.YELLOW}{msg_none:<25}{Colors.RESET}"; icon = f"{Colors.YELLOW}-{Colors.RESET}"
            else:
                interp = f"{Colors.GREEN}LONG{Colors.RESET}" if v > 0 else f"{Colors.RED}SHORT{Colors.RESET}" if v < 0 else f"{Colors.YELLOW}NONE{Colors.RESET}"
                icon   = f"{Colors.GREEN}▲{Colors.RESET}" if v > 0 else f"{Colors.RED}▼{Colors.RESET}" if v < 0 else f"{Colors.YELLOW}-{Colors.RESET}"
            print(f"  {icon} {base_key:<20} : {interp} (강도: {v:+.2f})")

        print("-" * 68)
        print(f" {Colors.CYAN}[ 🤖 LR MoE 4-Agent 독립 판단 ]{Colors.RESET}")
        if self.pos is not None:
            pnl_color = Colors.GREEN if pnl_pct > 0 else Colors.RED
            print(f" ⏳ 보유 캔들: {self.hold_count:<4} | 📈 미실현 수익: {pnl_color}{pnl_pct:+.2f}%{Colors.RESET}")
        print(f" 🌍 시장 레짐: {regime_name:<10} | 📊 현재 포지션: {pos_str}")
        print(f" ⚖️ [현재 국면 엣지 비교] {Colors.GREEN}롱돌이(L): {long_edge:.3f}{Colors.RESET} vs {Colors.RED}숏돌이(S): {short_edge:.3f}{Colors.RESET}")
        print(f" 🤖 담당 특수부대: {active_agent:<15} | 🎯 선택된 Kelly: {kelly:.3f}")
        print(f" 🎯 최종 결단: {action_str}")

        if ls_action is not None:
            print("-" * 68)
            print(f" {Colors.CYAN}[ 🤖 LS 2-Agent HCRouter 롱돌이/숏돌이 독립 판단 ]{Colors.RESET}")
            ls_action_str = {
                0: f'{Colors.YELLOW}🟨 관망 / 청산{Colors.RESET}',
                1: f'{Colors.GREEN}🟩 롱 진입 (Long){Colors.RESET}',
                2: f'{Colors.RED}🟥 숏 진입 (Short){Colors.RESET}'
            }.get(ls_action, '?')
            ls_agent_name = (ls_info or {}).get('agent', 'N/A')
            # HCRouter info 키: adv(진입 시), adv_L/adv_S(관망 시), consensus/ml_cons
            ls_adv_l   = (ls_info or {}).get('adv_L', (ls_info or {}).get('adv', None))
            ls_adv_s   = (ls_info or {}).get('adv_S', None)
            ls_cons    = (ls_info or {}).get('consensus', (ls_info or {}).get('ml_cons', None))
            ls_pos_str = {'LONG': f'{Colors.GREEN}🟩 LONG{Colors.RESET}', 'SHORT': f'{Colors.RED}🟥 SHORT{Colors.RESET}',
                          None: f'{Colors.YELLOW}무포지션{Colors.RESET}'}.get(ls_pos, f'{Colors.YELLOW}무포지션{Colors.RESET}')
            adv_str = ""
            if ls_adv_l is not None and ls_adv_s is not None:
                adv_str = f" | L-adv: {ls_adv_l:+.4f} / S-adv: {ls_adv_s:+.4f}"
            elif ls_adv_l is not None:
                adv_str = f" | adv: {ls_adv_l:+.4f}"
            cons_str = f" | consensus: {ls_cons:+.4f}" if ls_cons is not None else ""
            pnl_str = ""
            if ls_pnl is not None:
                pnl_color = Colors.GREEN if ls_pnl > 0 else (Colors.RED if ls_pnl < 0 else Colors.YELLOW)
                pnl_str = f" | 미실현: {pnl_color}{ls_pnl:+.2f}%{Colors.RESET}"
            print(f"  포지션: {ls_pos_str}{pnl_str}")
            print(f"  에이전트: {ls_agent_name}{adv_str}{cons_str}")
            print(f"  결단: {ls_action_str}")

        # LLM 분석 출력은 LS 블록 아래로 배치 (요청사항)
        if llm_answer:
            print("-" * 68)
            _rn = llm_regime_name or regime_name
            _t  = llm_time_kst or timestamp
            _print_llm_section(llm_answer, _rn, _t)

        print(Colors.BOLD + "╚════════════════════════════════════════════════════════════════════════╝\n" + Colors.RESET)


# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher      = BinanceLiveFetcher(limit=2500)
    fe_engine    = FeatureEngineer()
    ensemble     = EnsemblePredictor()
    nf_predictor = NFStatePredictor()
    llm_analyzer = LLMAnalyzer(fetcher.exchange, symbol='ETHUSDT')

    try:
        bot = MoELiveRouter('data/ensemble/ckpt/best_rl_agents.pth')
    except Exception as e:
        logger.error(f"❌ MoE 라우터 초기화 실패: {e}")
        return

    ls_bot = None
    try:
        ls_bot = LSLiveRouter('data/ensemble/ckpt/best_ls_agents.pth')
    except Exception as e:
        logger.warning(f"⚠️ LS 라우터 초기화 실패 (미학습 상태일 수 있음): {e}")

    # ── MetaRouter 초기화 ──────────────────────────────────────
    meta_router = MetaRouter()
    # 직전 포지션 청산 시점 추적용 (PnL 피드백을 위해)
    _prev_meta_pos: str | None = None

    # ── XGBTrendBrain 초기화 (LightGBM 3-class, Brain B) ───────────
    # 학습: python ensemble/trend_xgb/train_trend_xgb.py
    # 저장: data/trend_xgb/trend_xgb.pkl
    trend_brain = None
    try:
        trend_brain = XGBTrendBrain.load('data/trend_xgb/trend_xgb.pkl')
        logger.info("✅ XGBTrendBrain (LightGBM Triple-Barrier) 로드 완료")
    except Exception as e:
        logger.warning(f"⚠️ XGBTrendBrain 미로드 (학습 전이거나 파일 없음): {e}")

    # ── Polymarket 크라우드 확률 수집기 ────────────────────────
    poly_fetcher = PolymarketFetcher()

    async def _run_cycle(processed_df, eth_buffer):
        """한 사이클: 에이전트 판단 + MetaRouter 통합 + Polymarket + LLM fire-and-forget."""
        nonlocal _prev_meta_pos

        preds, confs = await ensemble.predict_all_async(processed_df)
        nf_preds     = nf_predictor.predict(processed_df)

        final_action, info, elite_sigs, regime = bot.get_signal(processed_df, preds, confs, nf_preds)
        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price    = float(eth_buffer['close'].iloc[-1])
        regime_name      = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')

        ls_action, ls_info, ls_pnl = (None, None, None)
        if ls_bot is not None:
            ls_action, ls_info, ls_pnl = ls_bot.get_signal(processed_df, preds, confs, nf_preds)

        # ── TrendContextBrain 4h 추세 추론 ───────────────────────
        trend_signal = None
        if trend_brain is not None:
            try:
                trend_signal = trend_brain.predict_from_df(eth_buffer)
            except Exception as e:
                logger.debug(f"TrendBrain 추론 실패: {e}")

        # ── MetaRouter 신호 융합 ──────────────────────────────
        meta_result = meta_router.fuse(
            rl_action     = final_action,
            rl_info       = info,
            ls_action     = ls_action if ls_action is not None else 0,
            ls_info       = ls_info   if ls_info   is not None else {},
            regime        = regime,
            current_price = current_price,
            trend_signal  = trend_signal,
        )

        # 직전 사이클에 포지션이 있었다가 이번에 청산됐으면 PnL 피드백
        if _prev_meta_pos is not None and meta_router.pos is None:
            realized_approx = float(ls_pnl) if ls_pnl is not None else 0.0
            meta_router.record_outcome(realized_approx)
        _prev_meta_pos = meta_router.pos

        # ── Polymarket 크라우드 확률 조회 (비동기, 5초 타임아웃) ─
        try:
            poly_data = await asyncio.wait_for(poly_fetcher.fetch(), timeout=5.0)
        except asyncio.TimeoutError:
            poly_data = {'btc': None, 'eth': None, 'ok': False}
        except Exception as e:
            logger.debug(f"Polymarket 조회 예외: {e}")
            poly_data = {'btc': None, 'eth': None, 'ok': False}

        # 이전 사이클 LLM 결과 수거 (완료됐으면 대시보드 하단(LS 아래)로 출력)
        prev_answer = llm_analyzer.collect()
        prev_meta   = llm_analyzer.collect_meta() if prev_answer is not None else {}

        # 기존 RL 대시보드 출력 (LLM 블록은 LS 아래로 이동)
        bot.print_dashboard(
            current_price, preds, confs, final_action, info, regime, elite_sigs, current_time_kst,
            ls_action=ls_action, ls_info=ls_info, ls_pnl=ls_pnl, ls_pos=ls_bot.pos if ls_bot else None,
            llm_answer=prev_answer,
            llm_regime_name=prev_meta.get('regime_name', None),
            llm_time_kst=current_time_kst,
        )

        # MetaRouter 통합 판단 출력
        meta_router.print_meta_dashboard(meta_result, current_price)

        # Polymarket 크라우드 확률 출력
        _print_polymarket_section(poly_data, current_price)

        # 이번 사이클 LLM 분석 시작 (백그라운드, 블로킹 없음)
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