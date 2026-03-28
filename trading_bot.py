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
from ensemble.seven_model_ensemble import SevenModelEnsemble
from ensemble.unsupervised.live_unsupervised_hub import UnsupervisedRegimeHub
from ensemble.ensemble_router import (
    TFTForecaster, MacroHFTForecaster, ChronosForecaster,
    KronosForecaster, TimesFMForecaster, MoiraiForecaster,
)
from ensemble.train_rl_agent import (
    STATE_PRED as DSAC_STATE_PRED,
    STATE_CONF as DSAC_STATE_CONF,
    STATE_ELITE as DSAC_STATE_ELITE,
    STATE_ALPHA as DSAC_STATE_ALPHA,
    STATE_SYNTH as DSAC_STATE_SYNTH,
    OnlineHMMDetector,
)
from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM,
    GaussianActor as DSACGaussianActor,
    SACRouter as DSACRouter,
)
from ensemble.train_rl_dsac_v2 import (
    GaussianActorV2 as DSACGaussianActorV2,
)
from strategies.elite_builder import EliteSignals, row_to_market_row
from strategies.elite_strategies import NewEliteSignalEngine


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
DSAC_ONLY_MODE = True
ENSEMBLE_PREDICTOR_ENABLED = _env_flag('ENSEMBLE_PREDICTOR_ENABLED', False)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
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


def _trend_signal_from_m7(m7_last: dict | None) -> dict | None:
    """SevenModelEnsemble 출력(dict)을 DSACTrendRouter 입력 포맷으로 변환."""
    if not isinstance(m7_last, dict) or not m7_last:
        return None

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(m7_last.get(key, default))
        except Exception:
            return float(default)

    agg_dn = float(np.clip(_f("m7_prob_dn", 0.0), 0.0, 1.0))
    agg_fl = float(np.clip(_f("m7_prob_fl", 0.0), 0.0, 1.0))
    agg_up = float(np.clip(_f("m7_prob_up", 0.0), 0.0, 1.0))
    s = agg_dn + agg_fl + agg_up
    if s <= 1e-12:
        agg_dn = agg_fl = agg_up = 1.0 / 3.0
    else:
        agg_dn, agg_fl, agg_up = agg_dn / s, agg_fl / s, agg_up / s

    p_dn = float(np.clip(_f("m7_trend_xgb_dn", agg_dn), 0.0, 1.0))
    p_fl = float(np.clip(_f("m7_trend_xgb_fl", agg_fl), 0.0, 1.0))
    p_up = float(np.clip(_f("m7_trend_xgb_up", agg_up), 0.0, 1.0))
    s_xgb = p_dn + p_fl + p_up
    if s_xgb <= 1e-12:
        p_dn = p_fl = p_up = 1.0 / 3.0
    else:
        p_dn, p_fl, p_up = p_dn / s_xgb, p_fl / s_xgb, p_up / s_xgb

    t_dir = int(np.argmax([p_dn, p_fl, p_up]))
    m7_action = int(np.clip(round(_f("m7_action", 0.0)), -1, 1))

    m7_conf = float(np.clip(_f("m7_confidence", 0.0), 0.0, 1.0))
    m7_gate_block = 1 if _f("m7_gate_block", 0.0) >= 0.5 else 0
    xgb_top = max(p_dn, p_fl, p_up)
    xgb_second = sorted([p_dn, p_fl, p_up])[1]
    strength = float(np.clip((xgb_top - 1.0 / 3.0) * 1.5 + (xgb_top - xgb_second) * 0.6, 0.0, 1.0))
    rev_prob = float(np.clip((1.0 - strength) * 0.70 + (0.30 if m7_gate_block else 0.0), 0.0, 1.0))

    return {
        "trend_dir": t_dir,
        "strength": strength,
        "rev_prob": rev_prob,
        "prob_dn": p_dn,
        "prob_flat": p_fl,
        "prob_up": p_up,
        "probs": [p_dn, p_fl, p_up],
        "trend_model": "TREND_XGB",
        "m7_confidence": m7_conf,
        "m7_action": m7_action,
        "m7_prob_dn": agg_dn,
        "m7_prob_fl": agg_fl,
        "m7_prob_up": agg_up,
        "m7_size": float(np.clip(_f("m7_size", 0.0), 0.0, 1.0)),
        "m7_gate_block": m7_gate_block,
        "m7_quality_pred": _f("m7_quality_pred", 0.0),
        "m7_hold_pred": _f("m7_hold_pred", 0.0),
        "m7_target_hold": float(max(0.0, _f("m7_target_hold", 0.0))),
        "m7_q10": _f("m7_q10", 0.0),
        "m7_q50": _f("m7_q50", 0.0),
        "m7_q90": _f("m7_q90", 0.0),
        "m7_qwidth": float(max(0.0, _f("m7_qwidth", 0.0))),
        "m7_trend_xgb_dn": float(np.clip(_f("m7_trend_xgb_dn", 0.0), 0.0, 1.0)),
        "m7_trend_xgb_fl": float(np.clip(_f("m7_trend_xgb_fl", 0.0), 0.0, 1.0)),
        "m7_trend_xgb_up": float(np.clip(_f("m7_trend_xgb_up", 0.0), 0.0, 1.0)),
        "m7_gmm_cluster": _f("m7_gmm_cluster", -1.0),
        "m7_gmm_conf": float(np.clip(_f("m7_gmm_conf", 0.0), 0.0, 1.0)),
        "m7_gmm_vol_rank": float(np.clip(_f("m7_gmm_vol_rank", 0.5), 0.0, 1.0)),
        "m7_hdb_label": _f("m7_hdb_label", -1.0),
        "m7_hdb_prob": float(np.clip(_f("m7_hdb_prob", 0.0), 0.0, 1.0)),
        "m7_iso_pred": _f("m7_iso_pred", 1.0),
        "m7_iso_score": _f("m7_iso_score", 0.0),
        "m7_iso_anom": 1.0 if _f("m7_iso_anom", 0.0) >= 0.5 else 0.0,
        "m7_vae_error": _f("m7_vae_error", 0.0),
        "m7_vae_threshold": _f("m7_vae_threshold", 0.0),
        "m7_vae_anom": 1.0 if _f("m7_vae_anom", 0.0) >= 0.5 else 0.0,
        "m7_expected_ret": _f("m7_expected_ret", 0.0),
        "m7_tail_risk": _f("m7_tail_risk", 0.0),
        "m7_composite_score": _f("m7_composite_score", 0.0),
    }


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
        loop = asyncio.get_running_loop()

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
    pnl_line = ""
    trade_pnl = meta_result.get("trade_pnl_pct", None) if isinstance(meta_result, dict) else None
    if trade_pnl is not None:
        try:
            p = float(trade_pnl)
            p_icon = "🟢" if p > 0 else ("🔴" if p < 0 else "🟨")
            pnl_line = f"\n{p_icon} Event PnL: {p:+.2f}%"
        except Exception:
            pnl_line = ""
    elif ex_code.startswith("ENTER_"):
        pnl_line = "\n🟨 Event PnL: +0.00% (entry)"
    return (
        f"{icon} <b>{ex_code}</b>  ({action_word})\n"
        f"💰 ETH ${current_price:,.2f}   🕐 {timestamp_kst.strftime('%m-%d %H:%M')} KST\n"
        f"🌍 {regime_name}   Kelly: {kelly:.3f}   Gate: {gate}{pnl_line}\n"
        f"📈 Trend: {t_dir}   Arbiter: {arb}"
    )


def _compute_regime(df, window=24):
    regime_cols = ['regime_bull', 'regime_bear', 'regime_chop', 'regime_whipsaw', 'regime_normal']
    if all(col in df.columns for col in regime_cols):
        last = df.iloc[-1]
        vals = {col: float(last.get(col, 0.0)) for col in regime_cols}
        if any(np.isfinite(v) and abs(v) > 1e-8 for v in vals.values()):
            best_col = max(regime_cols, key=lambda c: vals[c])
            return {col: (1.0 if col == best_col else 0.0) for col in regime_cols}

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


def _session_flags_from_timestamp(ts) -> dict[str, float]:
    ts_kst = pd.Timestamp(ts)
    if ts_kst.tzinfo is None:
        ts_kst = ts_kst.tz_localize("Asia/Seoul")
    else:
        ts_kst = ts_kst.tz_convert("Asia/Seoul")
    ts_utc = ts_kst.tz_convert("UTC")
    try:
        import pandas_market_calendars as mcal

        day = ts_utc.date()
        flags = {}
        for name, cal_name in (
            ("session_asia", "JPX"),
            ("session_europe", "LSE"),
            ("session_us", "NYSE"),
        ):
            cal = mcal.get_calendar(cal_name)
            sched = cal.schedule(start_date=day, end_date=day)
            active = False
            if not sched.empty:
                minutes = mcal.date_range(sched, frequency="1min")
                active = bool(ts_utc.floor("min") in minutes)
            flags[name] = 1.0 if active else 0.0
        return flags
    except Exception:
        hour = ts_utc.hour + (ts_utc.minute / 60.0)
        return {
            "session_asia": 1.0 if 0.0 <= hour < 8.0 else 0.0,
            "session_europe": 1.0 if 8.0 <= hour < 16.0 else 0.0,
            "session_us": 1.0 if 14.5 <= hour < 21.0 else 0.0,
        }


def _print_final_trade_summary(timestamp_kst, current_price: float,
                               regime_name: str, rl_action: int, rl_info: dict,
                               meta_result: dict,
                               prev_pos: str | None, cur_pos: str | None):
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

    def _kelly_text(v: float) -> str:
        if v >= 0.70:
            return "강함"
        if v >= 0.40:
            return "보통"
        if v >= 0.15:
            return "약함"
        return "매우약함"

    def _strength_text(v: float) -> str:
        if v >= 0.70:
            return "강함"
        if v >= 0.40:
            return "보통"
        return "약함"

    def _reversal_text(v: float) -> str:
        if v >= 0.70:
            return "반전위험 큼"
        if v >= 0.40:
            return "반전주의"
        return "추세유지 우세"

    def _quality_text(v: float) -> str:
        if v >= 0.015:
            return "진입품질 좋음"
        if v >= 0.0:
            return "진입품질 보통"
        if v >= -0.015:
            return "진입품질 약함"
        return "진입품질 나쁨"

    def _vol_rank_text(v: float) -> str:
        if v >= 0.75:
            return "고변동"
        if v <= 0.25:
            return "저변동"
        return "중간변동"

    def _exit_score_text(v: float) -> str:
        av = abs(v)
        if av >= 0.45:
            return "반대추세 강함"
        if av >= 0.28:
            return "반대추세 확인중"
        if av >= 0.12:
            return "추세 흔들림"
        return "추세 양호"

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

    long_edge = float(rl_info.get('long_edge', 0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    rl_kelly = float(rl_info.get('kelly', 0.0))
    meta_kelly = float(meta_result.get('unified_kelly', 0.0))
    source = str(meta_result.get('source', 'N/A'))
    arb_mode = str(meta_result.get('arbiter_mode', 'N/A'))
    gate_passed = bool(meta_result.get('gate_passed', True))
    fdiag = meta_result.get("fusion_diag", {}) if isinstance(meta_result, dict) else {}

    ts = meta_result.get('trend_signal') or {}
    t_dir = 1
    t_strength = 0.0
    t_rev = 0.0
    p_dn = p_fl = p_up = 0.0
    m7_size = 0.0
    m7_quality = 0.0
    m7_target_hold = 0
    m7_vol_rank = 0.5
    m7_qwidth = 0.0
    m7_iso_anom = 0
    m7_vae_anom = 0
    cb_active = int(float(fdiag.get("cb_active", 0.0))) if isinstance(fdiag, dict) else 0
    is_lowvol_range = int(float(fdiag.get("is_lowvol_range", 0.0))) if isinstance(fdiag, dict) else 0
    is_highvol_trend = int(float(fdiag.get("is_highvol_trend", 0.0))) if isinstance(fdiag, dict) else 0
    uncertainty_scale = float(fdiag.get("uncertainty_scale", 1.0)) if isinstance(fdiag, dict) else 1.0
    trend_exit_score = float(meta_result.get("trend_exit_score", 0.0)) if isinstance(meta_result, dict) else 0.0
    trend_mismatch_streak = int(meta_result.get("trend_mismatch_streak", 0) or 0) if isinstance(meta_result, dict) else 0
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
        m7_size = float(np.clip(ts.get("m7_size", 0.0), 0.0, 1.0))
        m7_quality = float(ts.get("m7_quality_pred", 0.0))
        m7_target_hold = int(max(0, round(float(ts.get("m7_target_hold", 0.0)))))
        m7_vol_rank = float(np.clip(ts.get("m7_gmm_vol_rank", 0.5), 0.0, 1.0))
        m7_qwidth = float(max(0.0, ts.get("m7_qwidth", fdiag.get("m7_qwidth", 0.0))))
        m7_iso_anom = 1 if float(ts.get("m7_iso_anom", 0.0)) >= 0.5 else 0
        m7_vae_anom = 1 if float(ts.get("m7_vae_anom", 0.0)) >= 0.5 else 0

    ex_icon, ex_code = _exec_code(prev_pos, cur_pos)

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
    gate_word = 'PASS' if gate_passed else 'BLOCK'
    gate_color = C.GREEN if gate_passed else C.RED

    W = 62
    _SEP  = "─" * W
    _SEP2 = "═" * W

    def _action_arrow(a: int) -> str:
        return {0: '─', 1: '▲', 2: '▼'}.get(int(a), '?')

    def _trend_arrow(tdir: int) -> str:
        return {0: '▼', 1: '─', 2: '▲'}.get(int(tdir), '?')

    fa_arrow = _action_arrow(fa)
    rl_arrow = _action_arrow(rl_action)
    trend_arrow = _trend_arrow(t_dir)
    gate_icon = '✓' if gate_passed else '✗'
    gate_log = meta_result.get('gate_log') or {}

    # ── 헤더: 최종 결과 화살표 ──────────────────────────────────────
    print(_SEP2)
    ts_str = timestamp_kst.strftime('%Y-%m-%d %H:%M')
    header_left = f"{final_color}{C.BOLD}{fa_arrow}{fa_arrow}  {final_word}  →  {ex_code}{C.RESET}"
    header_right = f"{C.CYAN}{ts_str}{C.RESET}"
    print(f" {header_left}  {header_right}")
    print(f"     {C.CYAN}${current_price:,.2f}  |  시장상태: {regime_name}{C.RESET}")
    session_flags = _session_flags_from_timestamp(timestamp_kst)
    session_parts = []
    for label, key in (("ASIA", "session_asia"), ("EUROPE", "session_europe"), ("US", "session_us")):
        active = float(session_flags.get(key, 0.0)) >= 0.5
        scol = C.GREEN if active else C.YELLOW
        sword = "ON" if active else "OFF"
        session_parts.append(f"{label}={scol}{sword}{C.RESET}")
    print(f"     {C.CYAN}세션:{C.RESET} " + "  ".join(session_parts))
    print(_SEP)

    # ── 서브 시그널 각 1행 ──────────────────────────────────────────
    print(f"  {rl_color}{rl_arrow} 신호{C.RESET}  {rl_color}{rl_word:<6}{C.RESET}"
          f"  엣지 {edge_side_color}{edge_side_word} {edge_gap:+.3f}{C.RESET}"
          f"  Kelly: {_bar(meta_kelly, 8)} {meta_kelly:.3f} ({_kelly_text(meta_kelly)})")
    # TREND
    dn_c = C.RED if p_dn > 0.4 else C.RESET
    up_c = C.GREEN if p_up > 0.4 else C.RESET
    trend_model = str(ts.get("trend_model", "N/A")) if isinstance(ts, dict) else "N/A"
    print(f"  {trend_color}{trend_arrow} 중기추세{C.RESET} {trend_color}{trend_word:<6}{C.RESET}"
          f"  str={t_strength:.2f}  rev={t_rev:.2f}"
          f"  {dn_c}DN={p_dn:.0%}{C.RESET} FL={C.YELLOW}{p_fl:.0%}{C.RESET} {up_c}UP={p_up:.0%}{C.RESET}"
          f"  model={trend_model}")
    print(f"  {C.CYAN}• 해석{C.RESET}    추세강도={_strength_text(t_strength)}  반전판정={_reversal_text(t_rev)}")
    print(f"  {C.CYAN}• M7{C.RESET}      포지션크기={m7_size:.2f}  품질={m7_quality:+.3f} ({_quality_text(m7_quality)})"
          f"  목표보유={m7_target_hold:>2d}봉")
    print(f"  {C.CYAN}• 장기추세{C.RESET} 점수={trend_exit_score:+.2f} ({_exit_score_text(trend_exit_score)})"
          f"  불일치누적={trend_mismatch_streak}봉")
    print(f"  {C.CYAN}• 변동성{C.RESET}  변동성상태={_vol_rank_text(m7_vol_rank)} ({m7_vol_rank:.2f})"
          f"  불확실성폭={m7_qwidth:.4f}  스케일={uncertainty_scale:.2f}")
    print(f"  {C.CYAN}• 이상감지{C.RESET} 회로차단={cb_active}  저변동횡보={is_lowvol_range}  고변동추세={is_highvol_trend}"
          f"  이상치=iso:{m7_iso_anom} vae:{m7_vae_anom}")
    if prev_pos != cur_pos:
        trade_pnl = meta_result.get("trade_pnl_pct", None)
        if trade_pnl is None and prev_pos is None and cur_pos is not None:
            trade_pnl = 0.0
        if trade_pnl is not None:
            try:
                p = float(trade_pnl)
                p_col = C.GREEN if p > 0 else (C.RED if p < 0 else C.YELLOW)
                print(f"  {C.CYAN}• TRADE{C.RESET}   event_pnl={p_col}{p:+.2f}%{C.RESET}")
            except Exception:
                pass
    # GATE
    if not gate_passed and isinstance(gate_log, dict):
        blocked_gate = gate_log.get('blocked_gate', 'N/A')
        blocked_val  = gate_log.get(blocked_gate, '')
        print(f"  {gate_color}{gate_icon} GATE{C.RESET}    {gate_color}{gate_word:<6}{C.RESET}"
              f"  mode: {arb_mode}  차단: {C.RED}{blocked_gate}={blocked_val}{C.RESET}")
    else:
        print(f"  {gate_color}{gate_icon} GATE{C.RESET}    {gate_color}{gate_word:<6}{C.RESET}"
              f"  mode: {arb_mode}  source: {source}")
    # ── 의사결정 체인 ────────────────────────────────────────────────
    print(_SEP)
    gate_chain_col = C.GREEN if gate_passed else C.RED
    decision_chain = (
        f"SIGNAL={rl_color}{rl_word}{C.RESET} → "
        f"중기추세={trend_color}{trend_word}{C.RESET} → "
        f"FINAL={final_color}{final_word}{C.RESET}({gate_chain_col}{gate_word}{C.RESET}) → "
        f"EXEC={ex_icon} {ex_code}"
    )
    if not gate_passed and isinstance(gate_log, dict):
        blocked_gate = gate_log.get('blocked_gate', 'N/A')
        decision_chain += f"  [{C.RED}차단={blocked_gate}{C.RESET}]"
    print(f"  {decision_chain}")
    print(f"  {C.CYAN}• 최종해석{C.RESET} 신호={rl_word} / 중기추세={trend_word} / 시장상태={regime_name} / 실행={ex_code}")
    print(_SEP2)


# ════════════════════════════════════════════════════════════════
# 3-A. DSACSignalRouter — DSAC Actor 추론 입력 생성 + 추론
# ════════════════════════════════════════════════════════════════
class DSACSignalRouter:
    DEFAULT_BEST_MODEL_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_v2_agents.pth"

    @staticmethod
    def _build_actor_from_ckpt(ckpt: dict, device: str):
        actor_state = ckpt.get("actor")
        if not isinstance(actor_state, dict):
            raise KeyError("DSAC 체크포인트 actor 키 없음")

        state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
        is_v2 = any(k.startswith("feat.input_proj.") for k in actor_state.keys())
        actor_cls = DSACGaussianActorV2 if is_v2 else DSACGaussianActor
        actor = actor_cls(state_dim=state_dim).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor, ("V2" if is_v2 else "V1")

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> str:
        resolved = model_path or DSACSignalRouter.DEFAULT_BEST_MODEL_PATH
        if resolved and os.path.exists(resolved):
            return resolved
        raise FileNotFoundError(f"DSAC best 체크포인트 파일이 없습니다: {resolved}")

    def __init__(self, model_path: str | None = None, hmm_detector: OnlineHMMDetector | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pos: str | None = None
        self.entry_price: float = 0.0
        self.hold_count: int = 0
        self.current_leverage: float = 0.0
        self.hmm = hmm_detector
        self.trade_fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
        self.trade_slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
        self.peak_equity: float = 1.0
        self.current_equity: float = 1.0
        self.elite_extractor = EliteSignals()
        self.new_elite_engine = NewEliteSignalEngine()

        ckpt_path = self._resolve_model_path(model_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        actor, actor_ver = self._build_actor_from_ckpt(ckpt, self.device)
        self.router = DSACRouter(actor, device=self.device)
        logger.info("✅ DSACSignalRouter 로드 완료 (%s): %s", actor_ver, ckpt_path)

    def decide(self, processed_df: pd.DataFrame, nf_preds: dict, m7_signal: dict | None = None):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        smf_std = processed_df["smart_money_flow"].std() if "smart_money_flow" in processed_df.columns else 1.0

        cur_market = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)
        try:
            tail_df = processed_df.tail(100).copy()
            self.new_elite_engine.compute(tail_df)
            tail_last = tail_df.iloc[-1]
            for col in ["sig_volume_confirm", "sig_liquidity_trap", "sig_trend_health"]:
                elite_sigs[col] = float(tail_last.get(col, 0.0))
        except Exception:
            for col in ["sig_volume_confirm", "sig_liquidity_trap", "sig_trend_health"]:
                elite_sigs.setdefault(col, 0.0)

        features: dict[str, float] = {}
        for col in DSAC_STATE_PRED:
            features[col] = float(nf_preds.get(col, 0.0))
        for col in DSAC_STATE_CONF:
            features[col] = float(nf_preds.get(col, 0.5))
        for col in DSAC_STATE_ELITE:
            features[col] = float(elite_sigs.get(col, 0.0))
        for col in DSAC_STATE_ALPHA:
            features[col] = float(last_row.get(col, 0.0))

        regime = None
        if self.hmm is not None:
            try:
                hmm_row = {
                    "log_return": float(last_row.get("log_return", 0.0)),
                    "garch_vol_z": float(last_row.get("garch_vol_z", 0.0)),
                    "oi_change_rate": float(last_row.get("oi_change_rate", 0.0)),
                }
                hmm_feat = self.hmm.get_features(hmm_row)
                hmm_probs = np.asarray(hmm_feat[:4], dtype=np.float32)
                hmm_idx = int(np.argmax(hmm_probs))
                regime = {
                    "regime_bull": 1.0 if hmm_idx == 0 else 0.0,
                    "regime_bear": 1.0 if hmm_idx == 1 else 0.0,
                    "regime_chop": 0.0,
                    "regime_whipsaw": 1.0 if hmm_idx == 2 else 0.0,
                    "regime_normal": 1.0 if hmm_idx == 3 else 0.0,
                }
            except Exception:
                regime = None
        if regime is None:
            regime = _compute_regime(processed_df)
        features.update(regime)
        for col in DSAC_STATE_SYNTH:
            features[col] = float(last_row.get(col, 0.0))
        features["close"] = float(last_row.get("close", 0.0))

        # 학습 시 사용한 m7_*를 라이브 추론 입력에도 주입해 train/infer 스키마 불일치 최소화
        if isinstance(m7_signal, dict):
            features["m7_prob_dn"] = float(m7_signal.get("m7_prob_dn", m7_signal.get("prob_dn", 0.0)))
            features["m7_prob_fl"] = float(m7_signal.get("m7_prob_fl", m7_signal.get("prob_flat", 0.0)))
            features["m7_prob_up"] = float(m7_signal.get("m7_prob_up", m7_signal.get("prob_up", 0.0)))
            for k in [
                "m7_quality_pred",
                "m7_hold_pred",
                "m7_q10",
                "m7_q50",
                "m7_q90",
                "m7_qwidth",
                "m7_gmm_cluster",
                "m7_gmm_conf",
                "m7_gmm_vol_rank",
                "m7_iso_score",
                "m7_iso_anom",
                "m7_vae_error",
                "m7_vae_threshold",
                "m7_vae_anom",
                "m7_hdb_label",
                "m7_hdb_prob",
            ]:
                if k in m7_signal:
                    features[k] = float(m7_signal.get(k, 0.0))

        unr = 0.0
        if self.pos is not None and self.entry_price > 0:
            cp = float(last_row["close"])
            lev = float(np.clip(self.current_leverage, 0.0, 1.0))
            if self.pos == "LONG":
                entry_exec = self.entry_price * (1.0 + self.trade_slip)
                exit_exec = cp * (1.0 - self.trade_slip)
                gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
            else:
                entry_exec = self.entry_price * (1.0 - self.trade_slip)
                exit_exec = cp * (1.0 + self.trade_slip)
                gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
            total_fee = 2.0 * self.trade_fee * lev
            unr = float(gross * lev - total_fee)
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
        else:
            self.current_equity = 1.0
            self.peak_equity = 1.0

        raw_drawdown = float(min((self.current_equity / max(self.peak_equity, 1e-8)) - 1.0, 0.0))
        effective_hold_count = int(self.hold_count + 1) if self.pos is not None else 0
        pos_dict = {
            "type": self.pos,
            "entry_price": self.entry_price,
            "unrealized": float(unr),
            "mdd": raw_drawdown,
            "hold_count": float(effective_hold_count),
            "hold_norm": min(effective_hold_count / 96.0, 1.0),
            "margin_usage": float(np.clip(self.current_leverage if self.pos is not None else 0.0, 0.0, 1.0)),
        }

        action, leverage, info = self.router.decide(features, pos_dict)
        return int(action), float(leverage), dict(info or {}), elite_sigs, regime


# ════════════════════════════════════════════════════════════════
# 3-B. DSACTrendRouter — DSAC + SevenModel(M7) 다요소 융합
# ════════════════════════════════════════════════════════════════
class DSACTrendRouter:
    def __init__(self):
        self.pos: str | None = None
        self.entry_price: float = 0.0
        self.hold_count: int = 0
        self.current_leverage: float = 0.0
        self.peak_equity: float = 1.0
        self.cur_equity: float = 1.0
        self.last_realized_pnl: float | None = None
        self.last_closed_hold_count: int = 0
        self._open_trade_diag: dict | None = None
        self.trade_history: deque[dict] = deque(maxlen=2000)
        self.recent_realized: deque[float] = deque(maxlen=20)
        self.loss_streak: int = 0
        self.cooldown_bars_left: int = 0
        self.trend_mismatch_streak: int = 0
        self.last_summary_ts: datetime | None = None
        self._last_state_save_ts: datetime | None = None

        # Legacy trend knobs (backward-compatible)
        self.veto_strength = float(os.getenv("TREND_VETO_STRENGTH", "0.70"))
        self.boost_strength = float(os.getenv("TREND_BOOST_STRENGTH", "0.35"))
        self.chop_strength = float(os.getenv("TREND_CHOP_STRENGTH", "0.30"))
        self.rev_reduce_prob = float(os.getenv("TREND_REV_REDUCE_PROB", "0.70"))

        # Multi-model fusion knobs
        self.dsac_weight = float(os.getenv("FUSE_DSAC_WEIGHT", "0.55"))
        self.m7_weight = float(os.getenv("FUSE_M7_WEIGHT", "0.45"))
        self.min_enter_score = float(os.getenv("FUSE_MIN_ENTER_SCORE", "0.12"))
        self.flip_min_margin = float(os.getenv("FUSE_FLIP_MIN_MARGIN", "0.08"))
        self.min_live_kelly = float(os.getenv("FUSE_MIN_LIVE_KELLY", "0.04"))
        self.hold_exit_margin = float(os.getenv("FUSE_HOLD_EXIT_MARGIN", "0.02"))
        self.exit_score_buffer = float(os.getenv("FUSE_EXIT_SCORE_BUFFER", "0.07"))
        self.exit_kelly_ratio = float(os.getenv("FUSE_EXIT_KELLY_RATIO", "0.55"))
        self.reverse_exit_margin = float(os.getenv("FUSE_REVERSE_EXIT_MARGIN", "0.10"))
        self.dsac_soft_exit = _env_flag("FUSE_DSAC_SOFT_EXIT", True)
        self.force_hold_exit = _env_flag("FUSE_FORCE_HOLD_EXIT", False)
        self.anomaly_force_exit = _env_flag("FUSE_ANOMALY_FORCE_EXIT", False)
        self.dsac_only_hard_stop = float(os.getenv("DSAC_ONLY_HARD_STOP", "0.025"))
        self.dsac_only_max_hold = int(os.getenv("DSAC_ONLY_MAX_HOLD", "96"))
        self.dsac_only_reverse_min = float(os.getenv("DSAC_ONLY_REVERSE_MIN", "0.45"))
        self.dsac_only_trail_arm = float(os.getenv("DSAC_ONLY_TRAIL_ARM", "0.012"))
        self.dsac_only_trail_gap = float(os.getenv("DSAC_ONLY_TRAIL_GAP", "0.008"))
        self.dsac_only_vol_scale_enable = _env_flag("DSAC_ONLY_VOL_SCALE_ENABLE", True)
        self.dsac_only_cooldown_loss = float(os.getenv("DSAC_ONLY_COOLDOWN_LOSS", "0.05"))
        self.dsac_only_cooldown_streak = int(os.getenv("DSAC_ONLY_COOLDOWN_STREAK", "4"))
        self.dsac_only_cooldown_bars = int(os.getenv("DSAC_ONLY_COOLDOWN_BARS", "10"))
        self.dsac_only_trend_exit_enable = _env_flag("DSAC_ONLY_TREND_EXIT_ENABLE", True)
        self.dsac_only_trend_exit_hold_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_HOLD_BARS", "48"))
        self.dsac_only_trend_exit_confirm_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_CONFIRM_BARS", "3"))
        self.dsac_only_trend_exit_score = float(os.getenv("DSAC_ONLY_TREND_EXIT_SCORE", "0.30"))
        self.dsac_only_trend_exit_quality = float(os.getenv("DSAC_ONLY_TREND_EXIT_QUALITY", "-0.010"))
        self.dsac_only_vae_block_ratio = float(os.getenv("DSAC_ONLY_VAE_BLOCK_RATIO", "1.35"))
        self.trade_fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
        self.trade_slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
        self.live_state_path = os.getenv("DSAC_LIVE_STATE_PATH", "data/ensemble/dsac_live_state.json")

        # Layer-1 alpha gate (direction + quality)
        self.dir_prob_th = float(os.getenv("FUSE_DIR_PROB_TH", "0.60"))
        self.quality_fee_floor = float(os.getenv("FUSE_QUALITY_FEE_FLOOR", "0.0015"))
        self.quality_lowvol_bonus = float(os.getenv("FUSE_QUALITY_LOWVOL_BONUS", "0.0007"))

        # Layer-2 uncertainty sizing + dynamic quantile stop
        self.qwidth_full_th = float(os.getenv("FUSE_QWIDTH_FULL_TH", "0.008"))
        self.qwidth_half_th = float(os.getenv("FUSE_QWIDTH_HALF_TH", "0.018"))
        self.stop_wide_mult = float(os.getenv("FUSE_STOP_WIDE_MULT", "1.35"))
        self.stop_tight_mult = float(os.getenv("FUSE_STOP_TIGHT_MULT", "0.90"))

        # Layer-3 regime switch
        self.range_cluster_id = int(os.getenv("FUSE_RANGE_CLUSTER_ID", "0"))
        self.range_vol_rank_max = float(os.getenv("FUSE_RANGE_VOL_RANK_MAX", "0.35"))
        self.range_score_boost = float(os.getenv("FUSE_RANGE_SCORE_BOOST", "0.04"))
        self.range_prob_boost = float(os.getenv("FUSE_RANGE_PROB_BOOST", "0.03"))
        self.trend_vol_rank_min = float(os.getenv("FUSE_TREND_VOL_RANK_MIN", "0.70"))
        self.trend_strength_min = float(os.getenv("FUSE_TREND_STRENGTH_MIN", "0.55"))
        self.trend_prob_relax = float(os.getenv("FUSE_TREND_PROB_RELAX", "0.04"))

        # Layer-3 circuit breaker
        self.cb_enable = _env_flag("FUSE_CIRCUIT_BREAKER", True)
        self.cb_iso_score_th = float(os.getenv("FUSE_CB_ISO_SCORE_TH", "0.060"))
        self.cb_vae_ratio_th = float(os.getenv("FUSE_CB_VAE_RATIO_TH", "1.15"))
        self.cb_hdb_noise = _env_flag("FUSE_CB_HDB_NOISE", True)
        self.cb_hdb_prob_max = float(os.getenv("FUSE_CB_HDB_PROB_MAX", "0.10"))
        self.cb_hdb_min_samples = int(os.getenv("FUSE_CB_HDB_MIN_SAMPLES", "200"))
        self.cb_hdb_disable_noise_ratio = float(os.getenv("FUSE_CB_HDB_DISABLE_NOISE_RATIO", "0.85"))
        self.cb_hdb_disable_zero_prob_ratio = float(os.getenv("FUSE_CB_HDB_DISABLE_ZERO_PROB_RATIO", "0.98"))
        self._hdb_seen = 0
        self._hdb_noise_hits = 0
        self._hdb_zero_prob_hits = 0

        self.use_tuned_json = _env_flag("FUSE_USE_TUNED_JSON", True)
        self.tuned_json_path = os.getenv("FUSE_TUNED_JSON_PATH", "data/ensemble/fuse_walkforward_best.json")
        self._apply_tuned_defaults()

        # ── 수익 보호 스텝 스탑 (브레이크이븐 포함) ─────────────────────────
        # peak_equity 기준 (레버리지 적용 수익률 단위)
        # (최대수익 달성 임계값, 최소 허용 수익률) 쌍의 리스트 (내림차순)
        self.step_stop_enable = _env_flag("DSAC_STEP_STOP_ENABLE", True)
        self.step_stop_levels: list[tuple[float, float]] = [
            (0.020, 0.012),   # 2.0% 달성 후 → 1.2% 수익 잠금
            (0.015, 0.007),   # 1.5% 달성 후 → 0.7% 수익 잠금
            (0.010, 0.003),   # 1.0% 달성 후 → 0.3% 수익 잠금
            (0.006, 0.000),   # 0.6% 달성 후 → 브레이크이븐
        ]

        # ── 자금 조달 비율(Funding Rate) 게이트 ──────────────────────────────
        # 크라우딩 방향 포지션 진입 시 kelly 축소
        self.funding_gate_enable  = _env_flag("FUSE_FUNDING_GATE",        True)
        self.funding_long_th      = float(os.getenv("FUSE_FUNDING_LONG_TH",    "0.0010"))  # >0.1%/8h: 롱 크라우딩
        self.funding_short_th     = float(os.getenv("FUSE_FUNDING_SHORT_TH",   "-0.0010")) # <-0.1%/8h: 숏 크라우딩
        self.funding_reduce       = float(os.getenv("FUSE_FUNDING_REDUCE",     "0.75"))    # 크라우딩 시 kelly 배수
        self.funding_extreme_th   = float(os.getenv("FUSE_FUNDING_EXTREME_TH", "0.0030"))  # 극단 크라우딩 임계값

        # ── BTC-ETH 상관 필터 ─────────────────────────────────────────────────
        # BTC 3봉 방향이 DSAC 방향과 불일치 시 kelly 축소, 일치 시 소폭 부스트
        self.btc_corr_enable      = _env_flag("FUSE_BTC_CORR",            True)
        self.btc_corr_misalign    = float(os.getenv("FUSE_BTC_CORR_MISALIGN",    "0.85"))  # 반대방향 kelly 배수
        self.btc_corr_align_boost = float(os.getenv("FUSE_BTC_CORR_ALIGN_BOOST", "1.08"))  # 같은방향 kelly 부스트
        self.btc_corr_move_th     = float(os.getenv("FUSE_BTC_CORR_MOVE_TH",     "0.004")) # BTC 3봉 유의미 변화 임계값

        # ── 안티 찹(횡보) 필터 ────────────────────────────────────────────────
        # 최근 N봉에서 방향 전환이 잦으면 지그재그 장세로 판단해 kelly 축소
        self.chop_filter_enable   = _env_flag("FUSE_CHOP_FILTER",         True)
        self.chop_window          = int(os.getenv("FUSE_CHOP_WINDOW",         "12"))   # 관찰 봉수
        self.chop_turns_max       = int(os.getenv("FUSE_CHOP_TURNS_MAX",      "7"))   # 이 횟수 이상 전환 시 찹 판정
        self.chop_kelly_scale     = float(os.getenv("FUSE_CHOP_KELLY_SCALE",  "0.65"))

        # ── 거래량 확인 필터 ──────────────────────────────────────────────────
        # 저거래량 구간 진입 시 kelly 축소 (스캘핑 노이즈 방지)
        self.volume_confirm_enable = _env_flag("FUSE_VOLUME_CONFIRM",      True)
        self.volume_min_ratio      = float(os.getenv("FUSE_VOLUME_MIN_RATIO",  "0.50"))  # 20봉 평균 대비 최소 비율
        self.volume_low_kelly      = float(os.getenv("FUSE_VOLUME_LOW_KELLY",  "0.80"))  # 저거래량 시 kelly 배수

        # Online adaptation knobs
        self.online_adapt = False
        self.adapt_alpha = float(os.getenv("FUSE_ADAPT_ALPHA", "0.08"))
        self.adapt_lr = float(os.getenv("FUSE_ADAPT_LR", "0.06"))
        self.adapt_weight_step = float(os.getenv("FUSE_ADAPT_WEIGHT_STEP", "0.01"))
        self.adapt_state_path = os.getenv("FUSE_ADAPT_STATE_PATH", "data/ensemble/fuse_online_state.json")

        self.trade_count: int = 0
        self.win_rate_ema: float = 0.5
        self.pnl_ema: float = 0.0
        self.dsac_perf_ema: float = 0.0
        self.m7_perf_ema: float = 0.0

        self._normalize_weights()
        self._load_live_state()

    def _apply_tuned_defaults(self) -> None:
        if not self.use_tuned_json:
            return
        path = self.tuned_json_path
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning("⚠️ FUSE 튜닝 JSON 로드 실패: %s", e)
            return

        env_export = payload.get("env_export", {}) if isinstance(payload, dict) else {}
        best_params = payload.get("best_params", {}) if isinstance(payload, dict) else {}
        if not isinstance(env_export, dict):
            env_export = {}
        if not isinstance(best_params, dict):
            best_params = {}

        def _choose(env_key: str, best_key: str):
            if os.getenv(env_key) is not None:
                return None
            if env_key in env_export:
                return env_export.get(env_key)
            return best_params.get(best_key)

        def _to_float(v, cur):
            try:
                return float(v)
            except Exception:
                return cur

        def _to_bool(v, cur):
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
            try:
                return float(v) >= 0.5
            except Exception:
                return cur

        self.dsac_weight = _to_float(_choose("FUSE_DSAC_WEIGHT", "dsac_weight"), self.dsac_weight)
        self.m7_weight = _to_float(_choose("FUSE_M7_WEIGHT", "m7_weight"), self.m7_weight)
        self.min_enter_score = _to_float(_choose("FUSE_MIN_ENTER_SCORE", "min_enter_score"), self.min_enter_score)
        self.flip_min_margin = _to_float(_choose("FUSE_FLIP_MIN_MARGIN", "flip_min_margin"), self.flip_min_margin)
        self.min_live_kelly = _to_float(_choose("FUSE_MIN_LIVE_KELLY", "min_live_kelly"), self.min_live_kelly)
        self.hold_exit_margin = _to_float(_choose("FUSE_HOLD_EXIT_MARGIN", "hold_exit_margin"), self.hold_exit_margin)
        self.veto_strength = _to_float(_choose("TREND_VETO_STRENGTH", "veto_strength"), self.veto_strength)
        self.chop_strength = _to_float(_choose("TREND_CHOP_STRENGTH", "chop_strength"), self.chop_strength)
        self.rev_reduce_prob = _to_float(_choose("TREND_REV_REDUCE_PROB", "rev_reduce_prob"), self.rev_reduce_prob)
        anom_val = _choose("FUSE_ANOMALY_FORCE_EXIT", "anomaly_force_exit")
        if anom_val is not None:
            self.anomaly_force_exit = _to_bool(anom_val, self.anomaly_force_exit)

        logger.info(
            "🧪 FUSE 튜닝값 반영: w_dsac=%.3f w_m7=%.3f enter=%.3f flip=%.3f min_k=%.3f",
            self.dsac_weight, self.m7_weight, self.min_enter_score, self.flip_min_margin, self.min_live_kelly,
        )

    def record_outcome(self, realized_pnl_pct: float):
        pnl = float(realized_pnl_pct)
        self.last_realized_pnl = None
        self.trade_count += 1
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
        if (
            len(self.recent_realized) >= 5
            and sum(list(self.recent_realized)[-5:]) <= -abs(self.dsac_only_cooldown_loss)
        ) or self.loss_streak >= max(1, self.dsac_only_cooldown_streak):
            self.cooldown_bars_left = max(self.cooldown_bars_left, max(1, self.dsac_only_cooldown_bars))
        a = float(np.clip(self.adapt_alpha, 0.001, 0.5))
        reward = float(np.tanh(pnl / 0.01))
        self.pnl_ema = (1.0 - a) * self.pnl_ema + a * pnl
        self.win_rate_ema = (1.0 - a) * self.win_rate_ema + a * (1.0 if pnl > 0 else 0.0)

        self.dsac_perf_ema = (1.0 - a) * self.dsac_perf_ema + a * reward
        self.m7_perf_ema = (1.0 - a) * self.m7_perf_ema + a * reward
        self._save_live_state()
        self._open_trade_diag = None

    def _update_pos(self, final_action: int, current_price: float, leverage: float | None = None):
        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = "LONG", current_price, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.last_realized_pnl = None
            self.trend_mismatch_streak = 0
            self._save_live_state()
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = "SHORT", current_price, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.last_realized_pnl = None
            self.trend_mismatch_streak = 0
            self._save_live_state()
        elif final_action == 0 and self.pos is not None:
            if self.entry_price > 0 and current_price > 0:
                self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
            self.current_leverage = 0.0
            self.peak_equity = 1.0
            self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self._save_live_state()
        elif self.pos is not None and self.entry_price > 0 and current_price > 0:
            self.hold_count += 1
            if leverage is not None:
                self.current_leverage = float(np.clip(leverage, 0.0, 1.0))
            self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.peak_equity = max(self.peak_equity, self.cur_equity)
            self.last_realized_pnl = None
            self._save_live_state()

    def _normalize_weights(self) -> None:
        self.dsac_weight = float(max(0.0, self.dsac_weight))
        self.m7_weight = float(max(0.0, self.m7_weight))
        s = self.dsac_weight + self.m7_weight
        if s <= 1e-12:
            self.dsac_weight, self.m7_weight = 0.5, 0.5
            return
        self.dsac_weight /= s
        self.m7_weight /= s

    def _load_adapt_state(self) -> None:
        path = self.adapt_state_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dsac_weight = float(data.get("dsac_weight", self.dsac_weight))
            self.m7_weight = float(data.get("m7_weight", self.m7_weight))
            loaded_enter = float(data.get("min_enter_score", self.min_enter_score))
            loaded_kelly = float(data.get("min_live_kelly", self.min_live_kelly))
            # 과거 적응상태가 임계값을 더 높여 저장된 경우, 현재 완화값보다 높아지지 않게 캡
            self.min_enter_score = min(loaded_enter, self.min_enter_score)
            self.min_live_kelly = min(loaded_kelly, self.min_live_kelly)
            self.trade_count = int(data.get("trade_count", self.trade_count))
            self.win_rate_ema = float(data.get("win_rate_ema", self.win_rate_ema))
            self.pnl_ema = float(data.get("pnl_ema", self.pnl_ema))
            self.dsac_perf_ema = float(data.get("dsac_perf_ema", self.dsac_perf_ema))
            self.m7_perf_ema = float(data.get("m7_perf_ema", self.m7_perf_ema))
            self._normalize_weights()
            logger.info(
                "♻️ FUSE 적응상태 로드: w_dsac=%.3f w_m7=%.3f enter=%.3f min_k=%.3f trades=%d",
                self.dsac_weight, self.m7_weight, self.min_enter_score, self.min_live_kelly, self.trade_count,
            )
        except Exception as e:
            logger.warning("⚠️ FUSE 적응상태 로드 실패: %s", e)

    def _save_adapt_state(self) -> None:
        path = self.adapt_state_path
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                "dsac_weight": self.dsac_weight,
                "m7_weight": self.m7_weight,
                "min_enter_score": self.min_enter_score,
                "min_live_kelly": self.min_live_kelly,
                "trade_count": self.trade_count,
                "win_rate_ema": self.win_rate_ema,
                "pnl_ema": self.pnl_ema,
                "dsac_perf_ema": self.dsac_perf_ema,
                "m7_perf_ema": self.m7_perf_ema,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
        except Exception as e:
            logger.warning("⚠️ FUSE 적응상태 저장 실패: %s", e)

    def _load_live_state(self) -> None:
        path = self.live_state_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.pos = data.get("pos")
            self.entry_price = float(data.get("entry_price", 0.0))
            self.hold_count = int(data.get("hold_count", 0))
            self.current_leverage = float(np.clip(data.get("current_leverage", 0.0), 0.0, 1.0))
            self.peak_equity = float(max(data.get("peak_equity", 1.0), 1e-8))
            self.cur_equity = float(max(data.get("cur_equity", 1.0), 1e-8))
            self.last_realized_pnl = data.get("last_realized_pnl", None)
            self.last_closed_hold_count = int(data.get("last_closed_hold_count", 0))
            self.loss_streak = int(data.get("loss_streak", 0))
            self.cooldown_bars_left = int(data.get("cooldown_bars_left", 0))
            self.trend_mismatch_streak = int(data.get("trend_mismatch_streak", 0))
            self.recent_realized = deque(
                [float(x) for x in data.get("recent_realized", [])],
                maxlen=20,
            )
            self.trade_history = deque(data.get("trade_history", []), maxlen=2000)
            saved_at = data.get("saved_at")
            if saved_at:
                try:
                    saved_ts = pd.Timestamp(saved_at).tz_localize(None) if pd.Timestamp(saved_at).tzinfo is not None else pd.Timestamp(saved_at)
                    elapsed_bars = max(0, int((pd.Timestamp.utcnow().tz_localize(None) - saved_ts) / pd.Timedelta(minutes=5)))
                    self.cooldown_bars_left = max(0, self.cooldown_bars_left - elapsed_bars)
                except Exception:
                    pass
            logger.info(
                "♻️ DSAC 라이브 상태 로드: pos=%s entry=%.2f hold=%d lev=%.3f cooldown=%d",
                self.pos, self.entry_price, self.hold_count, self.current_leverage, self.cooldown_bars_left,
            )
        except Exception as e:
            logger.warning("⚠️ DSAC 라이브 상태 로드 실패: %s", e)

    def _save_live_state(self) -> None:
        path = self.live_state_path
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                "pos": self.pos,
                "entry_price": self.entry_price,
                "hold_count": self.hold_count,
                "current_leverage": self.current_leverage,
                "peak_equity": self.peak_equity,
                "cur_equity": self.cur_equity,
                "last_realized_pnl": self.last_realized_pnl,
                "last_closed_hold_count": self.last_closed_hold_count,
                "loss_streak": self.loss_streak,
                "cooldown_bars_left": self.cooldown_bars_left,
                "trend_mismatch_streak": self.trend_mismatch_streak,
                "recent_realized": list(self.recent_realized),
                "trade_history": list(self.trade_history),
                "saved_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=True)
        except Exception as e:
            logger.warning("⚠️ DSAC 라이브 상태 저장 실패: %s", e)

    def _net_pnl_frac(self, current_price: float) -> float:
        if self.pos is None or self.entry_price <= 0.0 or current_price <= 0.0:
            return 0.0
        lev = float(np.clip(self.current_leverage, 0.0, 1.0))
        if self.pos == "LONG":
            entry_exec = self.entry_price * (1.0 + self.trade_slip)
            exit_exec = current_price * (1.0 - self.trade_slip)
            gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
        else:
            entry_exec = self.entry_price * (1.0 - self.trade_slip)
            exit_exec = current_price * (1.0 + self.trade_slip)
            gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
        total_fee = 2.0 * self.trade_fee * lev
        return float(gross * lev - total_fee)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.pos is None or self.entry_price == 0.0:
            return 0.0
        return self._net_pnl_frac(current_price) * 100.0

    def decrement_cooldown(self) -> None:
        if self.cooldown_bars_left > 0:
            self.cooldown_bars_left -= 1

    def vol_scale(self, garch_vol_z: float = 0.0, micro_vol: float = 0.0) -> float:
        if not self.dsac_only_vol_scale_enable:
            return 1.0
        gv = abs(float(garch_vol_z))
        if gv >= 2.0:
            return 0.4
        if gv >= 1.2:
            return 0.7
        if gv <= 0.3:
            return 1.0
        return 0.85

    def should_trailing_stop(self) -> bool:
        peak_gain = float(self.peak_equity - 1.0)
        draw_from_peak = float(self.peak_equity - self.cur_equity)
        return peak_gain >= self.dsac_only_trail_arm and draw_from_peak >= self.dsac_only_trail_gap

    def long_trend_score(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> float:
        last_row = processed_df.iloc[-1]

        def _sf(v, d: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return float(d)

        ts = trend_signal if isinstance(trend_signal, dict) else {}
        p_dn = float(np.clip(_sf(ts.get("prob_dn", ts.get("m7_prob_dn", 1.0 / 3.0))), 0.0, 1.0))
        p_fl = float(np.clip(_sf(ts.get("prob_flat", ts.get("m7_prob_fl", 1.0 / 3.0))), 0.0, 1.0))
        p_up = float(np.clip(_sf(ts.get("prob_up", ts.get("m7_prob_up", 1.0 / 3.0))), 0.0, 1.0))
        ps = p_dn + p_fl + p_up
        if ps <= 1e-12:
            p_dn = p_fl = p_up = 1.0 / 3.0
        else:
            p_dn, p_fl, p_up = p_dn / ps, p_fl / ps, p_up / ps

        m7_q50 = _sf(ts.get("m7_q50", 0.0))
        m7_quality = _sf(ts.get("m7_quality_pred", 0.0))
        trend_1h = _sf(last_row.get("mtf_trend_1h", 0.0))
        trend_4h = _sf(last_row.get("mtf_trend_4h", 0.0))
        closes = processed_df["close"].tail(12).astype(float).values if "close" in processed_df.columns else np.array([], dtype=float)
        ret_12 = ((closes[-1] / closes[0]) - 1.0) if len(closes) >= 2 and abs(closes[0]) > 1e-8 else 0.0

        model_edge = (
            0.55 * (p_up - p_dn)
            + 0.20 * float(np.tanh(m7_q50 * 220.0))
            + 0.10 * float(np.tanh(m7_quality * 12.0))
        )
        mtf_edge = float(np.tanh((trend_1h + trend_4h + ret_12 * 80.0) / 2.4))
        return float(np.clip(0.75 * model_edge + 0.25 * mtf_edge, -1.0, 1.0))

    def update_trend_mismatch(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> tuple[bool, float, str]:
        if not self.dsac_only_trend_exit_enable or self.pos is None:
            self.trend_mismatch_streak = 0
            return False, 0.0, ""

        score = self.long_trend_score(processed_df, trend_signal)
        quality = 0.0
        if isinstance(trend_signal, dict):
            try:
                quality = float(trend_signal.get("m7_quality_pred", 0.0))
            except Exception:
                quality = 0.0

        mismatch = False
        reason = ""
        if self.hold_count >= max(1, self.dsac_only_trend_exit_hold_bars):
            if self.pos == "LONG" and score <= -abs(self.dsac_only_trend_exit_score) and quality <= self.dsac_only_trend_exit_quality:
                mismatch = True
                reason = "DSAC_ONLY_M7_LONG_MISMATCH"
            elif self.pos == "SHORT" and score >= abs(self.dsac_only_trend_exit_score) and quality >= -self.dsac_only_trend_exit_quality:
                mismatch = True
                reason = "DSAC_ONLY_M7_SHORT_MISMATCH"

        self.trend_mismatch_streak = (self.trend_mismatch_streak + 1) if mismatch else 0
        should_exit = self.trend_mismatch_streak >= max(1, self.dsac_only_trend_exit_confirm_bars)
        return should_exit, score, reason

    def step_stop_floor(self) -> float:
        """peak_equity 기준으로 단계적 수익 보호 스탑 하한선을 반환한다 (레버리지 적용 수익률 단위).

        - peak_equity에서 설정된 최대수익 임계값을 초과하면 해당 수익률 이하로 떨어질 시 청산.
        - 미달 시 기본 하드스탑(-dsac_only_hard_stop)으로 폴백.
        """
        if not self.step_stop_enable:
            return -abs(self.dsac_only_hard_stop)
        peak_gain = float(self.peak_equity - 1.0)
        for gain_th, stop_fl in self.step_stop_levels:
            if peak_gain >= gain_th:
                return float(stop_fl)
        return -abs(self.dsac_only_hard_stop)

    def funding_kelly_factor(self, funding_rate: float, intended_side: int) -> float:
        """펀딩 비율에 따른 Kelly 조정 계수를 반환한다.

        크라우딩 방향(펀딩 비율이 높은 방향)으로 진입할 때 kelly를 축소하여
        청산 리스크(funding squeeze) 노출을 줄인다.
        """
        if not self.funding_gate_enable or intended_side == 0:
            return 1.0
        fr = float(funding_rate)
        if intended_side > 0:   # LONG 방향
            if fr >= self.funding_extreme_th:
                return float(self.funding_reduce * 0.80)   # 극단 롱 크라우딩 → 강한 축소
            if fr >= self.funding_long_th:
                return float(self.funding_reduce)
        else:                   # SHORT 방향
            if fr <= -self.funding_extreme_th:
                return float(self.funding_reduce * 0.80)   # 극단 숏 크라우딩 → 강한 축소
            if fr <= self.funding_short_th:
                return float(self.funding_reduce)
        return 1.0

    def reconcile_external_position(self, pos_type: str | None, entry_price: float, leverage: float = 0.0) -> None:
        ext_pos = pos_type if pos_type in {"LONG", "SHORT"} else None
        ext_entry = float(entry_price) if entry_price and entry_price > 0 else 0.0
        ext_lev = self.current_leverage if self.current_leverage > 0.0 else 1.0
        if ext_pos is None:
            if self.pos is not None:
                logger.warning("⚠️ 외부 포지션 없음으로 복원 상태 초기화")
                self.pos, self.entry_price, self.hold_count = None, 0.0, 0
                self.current_leverage = 0.0
                self.peak_equity = 1.0
                self.cur_equity = 1.0
                self._save_live_state()
            return
        if self.pos != ext_pos or abs(self.entry_price - ext_entry) > 1e-6:
            logger.info("♻️ 외부 포지션 기준 상태 보정: %s %.2f lev=%.3f", ext_pos, ext_entry, ext_lev)
            self.pos = ext_pos
            self.entry_price = ext_entry
            self.current_leverage = ext_lev
            self.hold_count = 0
            self.peak_equity = 1.0
            self.cur_equity = 1.0
            self._save_live_state()

    def append_trade_history(self, timestamp_kst, pnl_frac: float) -> None:
        ts_str = timestamp_kst.isoformat() if hasattr(timestamp_kst, "isoformat") else str(timestamp_kst)
        self.trade_history.append({
            "ts": ts_str,
            "pnl_frac": float(pnl_frac),
            "hold_bars": int(self.last_closed_hold_count),
        })
        self._save_live_state()

    def performance_summary(self, now_kst) -> str:
        if not self.trade_history:
            return "perf 24h pnl:+0.00% wr:0% | 7d pnl:+0.00% wr:0% | all pnl:+0.00%"
        now_ts = pd.Timestamp(now_kst)
        pnl_all = float(sum(float(x.get("pnl_frac", 0.0)) for x in self.trade_history)) * 100.0
        def _window(hours: int):
            rows = []
            for row in self.trade_history:
                try:
                    ts = pd.Timestamp(row.get("ts"))
                except Exception:
                    continue
                if ts >= now_ts - pd.Timedelta(hours=hours):
                    rows.append(row)
            if not rows:
                return 0.0, 0.0
            pnl = float(sum(float(x.get("pnl_frac", 0.0)) for x in rows)) * 100.0
            wr = 100.0 * sum(1 for x in rows if float(x.get("pnl_frac", 0.0)) > 0) / len(rows)
            return pnl, wr
        pnl_24h, wr_24h = _window(24)
        pnl_7d, wr_7d = _window(24 * 7)
        return (
            f"perf 24h pnl:{pnl_24h:+.2f}% wr:{wr_24h:.0f}%"
            f" | 7d pnl:{pnl_7d:+.2f}% wr:{wr_7d:.0f}%"
            f" | all pnl:{pnl_all:+.2f}% cd:{self.cooldown_bars_left}"
        )

    @staticmethod
    def _side_from_action(action_int: int) -> int:
        a = int(action_int)
        if a == 1:
            return 1
        if a == 2:
            return -1
        return 0

    @staticmethod
    def _action_from_side(side: int) -> int:
        if side > 0:
            return 1
        if side < 0:
            return 2
        return 0

    def fuse(self, dsac_action: int, dsac_info: dict, regime: dict, current_price: float = 0.0,
             trend_signal=None, garch_vol_z: float = 0.0,
             funding_rate: float = 0.0, btc_3bar_ret: float = 0.0,
             aux_chop_factor: float = 1.0, aux_volume_factor: float = 1.0) -> dict:
        ts = dict(trend_signal) if isinstance(trend_signal, dict) else (
            trend_signal.to_arbiter_dict() if trend_signal is not None and hasattr(trend_signal, "to_arbiter_dict") else None
        )

        dsac_action = int(dsac_action)
        dsac_side = self._side_from_action(dsac_action)
        cur_side = 1 if self.pos == "LONG" else (-1 if self.pos == "SHORT" else 0)
        raw_action = float(dsac_info.get("raw_action", 0.0))
        base_kelly = float(dsac_info.get("kelly", dsac_info.get("score", 0.0)))
        base_kelly = float(np.clip(base_kelly, 0.0, 1.0))
        dsac_score = float(np.clip(max(abs(raw_action), abs(float(dsac_info.get("score", 0.0))), base_kelly), 0.0, 1.0))

        # M7 defaults (missing model 시 중립값)
        t_dir = int(ts.get("trend_dir", 1)) if isinstance(ts, dict) else 1
        t_str = float(np.clip(ts.get("strength", 0.0), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        t_rev = float(np.clip(ts.get("rev_prob", 0.0), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        p_dn = float(np.clip(ts.get("prob_dn", ts.get("m7_prob_dn", 0.0)), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        p_fl = float(np.clip(ts.get("prob_flat", ts.get("m7_prob_fl", 0.0)), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        p_up = float(np.clip(ts.get("prob_up", ts.get("m7_prob_up", 0.0)), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        _p_sum = p_dn + p_fl + p_up
        if _p_sum > 1e-12:
            p_dn, p_fl, p_up = p_dn / _p_sum, p_fl / _p_sum, p_up / _p_sum
        else:
            p_dn = p_fl = p_up = 1.0 / 3.0

        m7_conf = float(np.clip(ts.get("m7_confidence", t_str), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        m7_size = float(np.clip(ts.get("m7_size", 0.0), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        m7_quality = float(ts.get("m7_quality_pred", 0.0)) if isinstance(ts, dict) else 0.0
        m7_target_hold = int(max(0, round(float(ts.get("m7_target_hold", 0.0))))) if isinstance(ts, dict) else 0
        m7_vol_rank = float(np.clip(ts.get("m7_gmm_vol_rank", 0.5), 0.0, 1.0)) if isinstance(ts, dict) else 0.5
        m7_gmm_cluster = int(round(float(ts.get("m7_gmm_cluster", -1.0)))) if isinstance(ts, dict) else -1
        m7_hdb_label = int(round(float(ts.get("m7_hdb_label", -1.0)))) if isinstance(ts, dict) else -1
        m7_hdb_prob = float(np.clip(ts.get("m7_hdb_prob", 0.0), 0.0, 1.0)) if isinstance(ts, dict) else 0.0
        m7_iso_anom = bool(float(ts.get("m7_iso_anom", 0.0)) >= 0.5) if isinstance(ts, dict) else False
        m7_vae_anom = bool(float(ts.get("m7_vae_anom", 0.0)) >= 0.5) if isinstance(ts, dict) else False
        m7_gate_block = bool(float(ts.get("m7_gate_block", 0.0)) >= 0.5) if isinstance(ts, dict) else False
        m7_iso_score = float(ts.get("m7_iso_score", 0.0)) if isinstance(ts, dict) else 0.0
        m7_vae_error = float(ts.get("m7_vae_error", 0.0)) if isinstance(ts, dict) else 0.0
        m7_vae_threshold = float(ts.get("m7_vae_threshold", 0.0)) if isinstance(ts, dict) else 0.0
        m7_q10 = float(ts.get("m7_q10", 0.0)) if isinstance(ts, dict) else 0.0
        m7_q50 = float(ts.get("m7_q50", 0.0)) if isinstance(ts, dict) else 0.0
        m7_q90 = float(ts.get("m7_q90", 0.0)) if isinstance(ts, dict) else 0.0
        m7_qwidth = float(max(0.0, ts.get("m7_qwidth", 0.0))) if isinstance(ts, dict) else 0.0
        xgb_dn = float(np.clip(ts.get("m7_trend_xgb_dn", p_dn), 0.0, 1.0)) if isinstance(ts, dict) else p_dn
        xgb_up = float(np.clip(ts.get("m7_trend_xgb_up", p_up), 0.0, 1.0)) if isinstance(ts, dict) else p_up
        m7_expected_ret = float(ts.get("m7_expected_ret", 0.0)) if isinstance(ts, dict) else 0.0
        m7_tail_risk = abs(float(ts.get("m7_tail_risk", 0.0))) if isinstance(ts, dict) else 0.0
        m7_composite = float(np.clip(ts.get("m7_composite_score", 0.0), -1.0, 1.0)) if isinstance(ts, dict) else 0.0

        # HDBSCAN이 상시 noise만 내는 경우 회로 차단기에서 제외 (모델 붕괴 방지)
        self._hdb_seen += 1
        if m7_hdb_label == -1:
            self._hdb_noise_hits += 1
        if m7_hdb_prob <= 1e-8:
            self._hdb_zero_prob_hits += 1
        hdb_noise_ratio = float(self._hdb_noise_hits / max(1, self._hdb_seen))
        hdb_zero_prob_ratio = float(self._hdb_zero_prob_hits / max(1, self._hdb_seen))
        hdb_reliable = not (
            self._hdb_seen >= max(20, self.cb_hdb_min_samples)
            and hdb_noise_ratio >= self.cb_hdb_disable_noise_ratio
            and hdb_zero_prob_ratio >= self.cb_hdb_disable_zero_prob_ratio
        )

        vae_ratio = float(m7_vae_error / max(m7_vae_threshold, 1e-8)) if m7_vae_threshold > 1e-8 else (1.0 if m7_vae_anom else 0.0)
        cb_iso = bool(m7_iso_anom and m7_iso_score >= self.cb_iso_score_th)
        cb_vae = bool(m7_vae_anom and vae_ratio >= self.cb_vae_ratio_th)
        cb_hdb = bool(
            self.cb_hdb_noise
            and hdb_reliable
            and (m7_hdb_label == -1)
            and (m7_hdb_prob > 1e-8)
            and (m7_hdb_prob <= self.cb_hdb_prob_max)
        )
        circuit_breaker = bool(self.cb_enable and (cb_iso or cb_vae or cb_hdb))
        cb_tags = []
        if cb_iso:
            cb_tags.append("CB_ISO")
        if cb_vae:
            cb_tags.append("CB_VAE")
        if cb_hdb:
            cb_tags.append("CB_HDB")

        # 강한 이상 징후면 신규 진입 차단, 옵션으로 보유 포지션 강제 청산
        # (완화형) iso/vae 동시 탐지라도 점수 강도가 약하면 soft_anomaly로 분류
        hard_anomaly = bool(
            m7_gate_block
            or (
                m7_iso_anom
                and m7_vae_anom
                and (
                    m7_iso_score >= (self.cb_iso_score_th * 1.4)
                    or vae_ratio >= (self.cb_vae_ratio_th * 1.15)
                )
            )
        )
        soft_anomaly = (m7_iso_anom ^ m7_vae_anom) and not hard_anomaly

        # M7 리스크 신뢰도(사이즈/신뢰 결합) — 방향 결정에는 사용하지 않음
        m7_score = float(np.clip(m7_conf * (0.60 + 0.40 * m7_size), 0.0, 1.0))
        if t_dir == 1:
            m7_score *= 0.65

        # 방향은 DSAC 단독
        dsac_signed = float(dsac_side * dsac_score)
        fused_signed = dsac_signed
        fused_abs = abs(fused_signed)
        fused_side = 1 if fused_signed > self.flip_min_margin else (-1 if fused_signed < -self.flip_min_margin else 0)

        # Layer-3 Regime switch: 저변동 횡보 vs 고변동 추세
        is_lowvol_range = bool(m7_gmm_cluster == self.range_cluster_id and m7_vol_rank <= self.range_vol_rank_max)
        is_highvol_trend = bool(m7_vol_rank >= self.trend_vol_rank_min and t_dir != 1 and t_str >= self.trend_strength_min)
        enter_score_thr = float(np.clip(
            self.min_enter_score + (self.range_score_boost if is_lowvol_range else 0.0) - (0.02 if is_highvol_trend else 0.0),
            0.08,
            0.45,
        ))
        dir_prob_thr = float(np.clip(
            self.dir_prob_th + (self.range_prob_boost if is_lowvol_range else 0.0) - (self.trend_prob_relax if is_highvol_trend else 0.0),
            0.45,
            0.80,
        ))
        quality_thr = float(self.quality_fee_floor + (self.quality_lowvol_bonus if is_lowvol_range else 0.0))

        # 환경/품질 기반 Kelly 조정계수
        quality_factor = float(np.clip(1.0 + np.tanh(m7_quality * 12.0) * 0.20, 0.70, 1.25))
        ret_factor = float(np.clip(0.90 + np.tanh(abs(m7_expected_ret) * 250.0) * 0.20, 0.85, 1.15))
        tail_factor = float(np.clip(1.0 - min(m7_tail_risk * 180.0, 0.45), 0.55, 1.00))
        vol_factor = 1.0
        if m7_vol_rank >= 0.85:
            vol_factor *= 0.70  # 0.55 → 0.70: 극고변동 페널티 완화
        elif m7_vol_rank >= 0.70:
            vol_factor *= 0.85  # 0.75 → 0.85: 고변동 페널티 완화
        elif m7_vol_rank <= 0.20:
            vol_factor *= 1.08
        # hdb_label==-1 페널티 제거: circuit breaker에서 이미 처리됨
        if m7_hdb_prob < 0.20:
            vol_factor *= 0.93  # 0.90 → 0.93: 완화
        if float(garch_vol_z) >= 2.0:
            vol_factor *= 0.80  # 0.75 → 0.80: 완화
        elif float(garch_vol_z) >= 1.2:
            vol_factor *= 0.92  # 0.88 → 0.92: 완화

        regime_chop = float(regime.get("regime_chop", 0.0)) if isinstance(regime, dict) else 0.0
        regime_whipsaw = float(regime.get("regime_whipsaw", 0.0)) if isinstance(regime, dict) else 0.0
        regime_bull = float(regime.get("regime_bull", 0.0)) if isinstance(regime, dict) else 0.0
        regime_bear = float(regime.get("regime_bear", 0.0)) if isinstance(regime, dict) else 0.0
        dir_ref = dsac_side if dsac_side != 0 else cur_side
        regime_factor = float(np.clip(
            1.0 - 0.25 * regime_chop - 0.20 * regime_whipsaw
            + (0.10 if dir_ref > 0 else 0.0) * regime_bull
            + (0.10 if dir_ref < 0 else 0.0) * regime_bear,
            0.60,
            1.15,
        ))

        rev_factor = 0.60 if t_rev >= self.rev_reduce_prob else 1.0
        chop_factor = 0.80 if (t_dir == 1 and t_str >= self.chop_strength) else 1.0
        anom_factor = 0.0 if hard_anomaly else (0.50 if soft_anomaly else 1.0)

        agree_factor = 1.0

        # ── 보조 필터 계수 ─────────────────────────────────────────────────
        # 방향 기준 (진입 또는 현재 보유 방향)
        _intended_side_fuse = dsac_side if dsac_side != 0 else cur_side

        # 1) 자금 조달 비율: 크라우딩 방향 진입 kelly 축소
        fund_factor = self.funding_kelly_factor(float(funding_rate), _intended_side_fuse)

        # 2) BTC 상관 필터: ETH-BTC 3봉 방향 불일치 시 축소, 일치 시 소폭 부스트
        btc_factor = 1.0
        if self.btc_corr_enable and abs(float(btc_3bar_ret)) >= self.btc_corr_move_th and _intended_side_fuse != 0:
            _btc_up = float(btc_3bar_ret) > 0
            _aligned = (_btc_up and _intended_side_fuse > 0) or (not _btc_up and _intended_side_fuse < 0)
            btc_factor = self.btc_corr_align_boost if _aligned else self.btc_corr_misalign

        # 3) 안티 찹, 4) 저거래량 — _run_cycle에서 계산된 값을 그대로 사용
        chop_fac = float(np.clip(aux_chop_factor,   0.10, 1.20))
        vol_fac  = float(np.clip(aux_volume_factor, 0.10, 1.00))

        # 기본 Kelly (DSAC 비중 확대: DSAC 75% + M7 size 25%)
        kelly_pre = float(np.clip(0.75 * base_kelly + 0.25 * m7_size, 0.0, 1.0))
        q_full = float(max(1e-6, self.qwidth_full_th))
        q_half = float(max(q_full + 1e-6, self.qwidth_half_th))
        if m7_qwidth <= q_full:
            uncertainty_scale = 1.0
        elif m7_qwidth >= q_half:
            uncertainty_scale = 0.5
        else:
            rr = (m7_qwidth - q_full) / (q_half - q_full)
            uncertainty_scale = float(np.clip(1.0 - 0.5 * rr, 0.5, 1.0))

        # 캐스케이드 인수를 별도 계산해 하한선 적용 (hard anomaly 제외)
        _size_scale = (
            quality_factor * ret_factor * tail_factor
            * vol_factor * regime_factor * rev_factor * chop_factor
            * uncertainty_scale
        )
        if anom_factor > 0.0:
            _size_scale = max(_size_scale, 0.30)  # 과도한 축소 방지
        unified_kelly = float(np.clip(
            kelly_pre * _size_scale * anom_factor * agree_factor
            * fund_factor * btc_factor * chop_fac * vol_fac,
            0.0,
            1.0,
        ))

        # Hysteresis: 진입 기준보다 완화된 "유지 기준"으로 1봉 왕복을 줄임
        exit_score_thr = float(np.clip(self.min_enter_score - max(0.0, self.exit_score_buffer), 0.02, 0.40))
        exit_kelly_thr = float(np.clip(self.min_live_kelly * max(0.0, self.exit_kelly_ratio), 0.0, 1.0))
        live_pnl_frac = 0.0
        quant_stop_frac = 0.0

        final_side = dsac_side
        source = "DSAC_BASE"
        gate_reasons: list[str] = []
        cb_soft_hold = False

        # 0) Circuit breaker (생존 최우선)
        if circuit_breaker:
            # (완화형) 무포지션 + hard anomaly 아님 + DSAC 고신뢰면 제한적 진입 허용
            can_soft_bypass = (
                cur_side == 0
                and (not hard_anomaly)
                and dsac_side != 0
                and dsac_score >= (enter_score_thr + 0.08)
                and unified_kelly >= max(self.min_live_kelly, 0.03)
            )
            # (완화형) 보유중 + hard anomaly 아님 + DSAC 중립/동일방향이면 즉시청산 대신 유지
            can_soft_hold = (
                cur_side != 0
                and (not hard_anomaly)
                and (dsac_side == 0 or dsac_side == cur_side)
                and dsac_score >= max(exit_score_thr, 0.12)
                and unified_kelly >= max(exit_kelly_thr, 0.02)
            )
            if can_soft_bypass:
                final_side = dsac_side
                source = "CB_SOFT_BYPASS_DSAC_ENTER"
                gate_reasons.append("CB_SOFT_BYPASS")
            elif can_soft_hold:
                final_side = cur_side
                source = "CB_SOFT_HOLD_IN_POS"
                gate_reasons.append("CB_SOFT_HOLD")
                cb_soft_hold = True
            else:
                final_side = 0
                source = "CIRCUIT_BREAKER_EXIT" if cur_side != 0 else "CIRCUIT_BREAKER_BLOCK"
                gate_reasons.extend(cb_tags or ["CIRCUIT_BREAKER"])

        # 1) 무포지션: DSAC 단독 방향 + M7 리스크 게이트
        elif cur_side == 0:
            candidate_side = 0
            candidate_source = "DSAC_HOLD_LOW_SCORE"

            if hard_anomaly:
                gate_reasons.append("M7_HARD_ANOMALY_BLOCK")
                candidate_source = "RISK_BLOCK_ANOMALY"
            elif dsac_side != 0 and dsac_score >= enter_score_thr:
                candidate_side = dsac_side
                candidate_source = "DSAC_ENTER"
            else:
                gate_reasons.append("LOW_DSAC_SCORE")

            final_side = candidate_side
            source = candidate_source

            if final_side != 0:
                if unified_kelly < self.min_live_kelly:
                    final_side = 0
                    source = "RISK_HOLD_LOW_KELLY"
                    gate_reasons.append("LOW_KELLY")

        # 2) 보유중: DSAC + Safety + Hold + Quantile stop
        else:
            final_side = cur_side
            source = "IN_POS_HOLD"

            # DSAC 자체 청산 신호는 우선(단, 리스크 양호하면 soft-hold)
            if dsac_side == 0:
                keep_by_fuse = (not hard_anomaly) and (unified_kelly >= exit_kelly_thr)
                if self.dsac_soft_exit and keep_by_fuse:
                    final_side = cur_side
                    source = "RISK_HYST_HOLD"
                    gate_reasons.append("DSAC_NEUTRAL_HOLD")
                else:
                    final_side = 0
                    source = "DSAC_EXIT"
            elif dsac_side != cur_side:
                reverse_strength = abs(raw_action)
                reverse_exit_thr = float(np.clip(enter_score_thr + self.reverse_exit_margin, 0.20, 0.70))
                weak_reverse_hold = (
                    reverse_strength < reverse_exit_thr
                    and (not hard_anomaly)
                    and unified_kelly >= max(exit_kelly_thr * 0.7, 0.015)
                )
                if weak_reverse_hold:
                    final_side = cur_side
                    source = "DSAC_REVERSE_WEAK_HOLD"
                    gate_reasons.append("DSAC_REVERSE_WEAK_HOLD")
                else:
                    final_side = 0
                    source = "DSAC_REVERSE_EXIT"
                    gate_reasons.append("DSAC_REVERSE")

            # 이상치 게이트: 옵션 또는 강한 리스크면 청산
            if final_side != 0 and hard_anomaly and (self.anomaly_force_exit or m7_conf >= 0.75):
                final_side = 0
                source = "M7_ANOMALY_EXIT"
                gate_reasons.append("M7_HARD_ANOMALY_EXIT")

            # Hold 모델 기반 시간 청산
            if final_side != 0 and m7_target_hold > 0 and self.hold_count >= m7_target_hold:
                if self.force_hold_exit:
                    final_side = 0
                    source = "MTL_HOLD_TIMEOUT"
                    gate_reasons.append("HOLD_TIMEOUT")
                elif m7_composite <= self.hold_exit_margin:
                    final_side = 0
                    source = "M7_TARGET_HOLD_EXIT"
                    gate_reasons.append("TARGET_HOLD_EXCEEDED")

            # Quantile 기반 Dynamic SL (q10/q90)
            if final_side != 0 and self.entry_price > 0 and current_price > 0:
                if cur_side > 0:
                    live_pnl_frac = float((current_price - self.entry_price) / self.entry_price)
                    stop_base = float(min(m7_q10, -1e-6))
                else:
                    live_pnl_frac = float((self.entry_price - current_price) / self.entry_price)
                    stop_base = float(-max(m7_q90, 1e-6))

                stop_mult = float(max(0.10, self.stop_wide_mult if is_highvol_trend else self.stop_tight_mult))
                quant_stop_frac = float(stop_base * stop_mult)
                if live_pnl_frac <= quant_stop_frac:
                    final_side = 0
                    source = "QTL_DYNAMIC_STOP"
                    gate_reasons.append("QTL_STOP")

            # 단계적 수익 보호 스탑 (브레이크이븐 포함) — quantile stop과 독립적으로 작동
            if final_side != 0 and self.entry_price > 0 and current_price > 0:
                _step_fl = self.step_stop_floor()
                _cur_lev_gain = live_pnl_frac * max(self.current_leverage, 0.01)
                if _cur_lev_gain <= _step_fl:
                    final_side = 0
                    source = "FUSE_STEP_STOP" if self.peak_equity >= 1.006 else "FUSE_HARD_STOP"
                    gate_reasons.append("STEP_STOP")

            # 반전 확률 매우 높으면 리스크 축소
            if final_side != 0 and t_rev >= max(self.rev_reduce_prob, 0.85) and m7_conf >= 0.65:
                unified_kelly = float(np.clip(unified_kelly * 0.55, 0.0, 1.0))
                source = "M7_REV_REDUCE"

        final_action = self._action_from_side(final_side)
        if final_action == 0:
            unified_kelly = 0.0

        # DSAC가 진입/유지하려 했는데 차단되었는지 표시
        gate_passed = not (dsac_side != 0 and final_action == 0)
        trend_veto = gate_reasons[0] if gate_reasons else None
        gate_log = {}
        if not gate_passed:
            gate_log = {
                "blocked_gate": "risk_guard",
                "risk_guard": ",".join(gate_reasons[:3]) if gate_reasons else "BLOCK",
            }

        if cur_side == 0 and final_side != 0:
            self._open_trade_diag = {
                "source": source,
                "dsac_score": dsac_score,
                "m7_score": m7_score,
                "entry_side": float(final_side),
                "entry_kelly": unified_kelly,
                "m7_quality": m7_quality,
                "m7_qwidth": m7_qwidth,
            }
        elif final_side == 0:
            # 청산 확정 사이클에서는 outcome 기록 시 재사용하기 위해 유지, 다음 진입 시 갱신됨
            pass

        self._update_pos(final_action, current_price, unified_kelly)
        return {
            "final_action": final_action,
            "unified_kelly": unified_kelly,
            "source": source,
            "rl_score": float(dsac_info.get("score", 0.0)),
            "meta_score": unified_kelly if final_action != 0 else 0.0,
            "rl_action": int(dsac_action),
            "trend_signal": ts,
            "trend_veto": trend_veto,
            "arbiter_mode": "DSAC_RISK_GUARD",
            "gate_passed": gate_passed,
            "gate_log": gate_log,
            "fusion_diag": {
                "direction_source": 1.0,  # 1.0 = DSAC_ONLY
                "dsac_score": dsac_score,
                "m7_score": m7_score,
                "fused_signed": fused_signed,
                "fused_abs": fused_abs,
                "m7_size": m7_size,
                "m7_quality": m7_quality,
                "m7_q10": m7_q10,
                "m7_q50": m7_q50,
                "m7_q90": m7_q90,
                "m7_qwidth": m7_qwidth,
                "m7_vol_rank": m7_vol_rank,
                "m7_gmm_cluster": float(m7_gmm_cluster),
                "m7_hdb_label": float(m7_hdb_label),
                "m7_hdb_prob": m7_hdb_prob,
                "hard_anomaly": 1.0 if hard_anomaly else 0.0,
                "soft_anomaly": 1.0 if soft_anomaly else 0.0,
                "cb_iso": 1.0 if cb_iso else 0.0,
                "cb_vae": 1.0 if cb_vae else 0.0,
                "cb_hdb": 1.0 if cb_hdb else 0.0,
                "cb_active": 1.0 if circuit_breaker else 0.0,
                "cb_soft_hold": 1.0 if cb_soft_hold else 0.0,
                "hdb_reliable": 1.0 if hdb_reliable else 0.0,
                "is_lowvol_range": 1.0 if is_lowvol_range else 0.0,
                "is_highvol_trend": 1.0 if is_highvol_trend else 0.0,
                "uncertainty_scale": float(uncertainty_scale),
                "target_hold": float(m7_target_hold),
                "hold_count": float(self.hold_count),
                "enter_score_thr": float(enter_score_thr),
                "dir_prob_thr": float(dir_prob_thr),
                "quality_thr": float(quality_thr),
                "exit_score_thr": float(exit_score_thr),
                "exit_kelly_thr": float(exit_kelly_thr),
                "reverse_exit_thr": float(np.clip(enter_score_thr + self.reverse_exit_margin, 0.20, 0.70)),
                "reverse_strength": float(abs(raw_action)),
                "quant_stop_frac": float(quant_stop_frac),
                "live_pnl_frac": float(live_pnl_frac),
                "step_stop_floor": float(self.step_stop_floor()),
                "p_up": float(p_up),
                "p_dn": float(p_dn),
                "p_fl": float(p_fl),
                "xgb_up": float(xgb_up),
                "xgb_dn": float(xgb_dn),
                "fund_factor": float(fund_factor),
                "btc_factor": float(btc_factor),
                "chop_fac": float(chop_fac),
                "vol_fac": float(vol_fac),
                "btc_3bar_ret": float(btc_3bar_ret),
                "funding_rate": float(funding_rate),
            },
        }

    def print_meta_dashboard(self, result: dict, current_price: float = 0.0):
        C = Colors
        fa = int(result.get("final_action", 0))
        src = str(result.get("source", "N/A"))
        gate_passed = bool(result.get("gate_passed", True))
        gate_icon = "✓" if gate_passed else "✗"
        gate_color = C.GREEN if gate_passed else C.RED
        fa_arrow = {0: "─", 1: "▲", 2: "▼"}.get(fa, "?")
        fa_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(fa, C.RESET)
        fa_word = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(fa, "?")

        print(f" {fa_color}{C.BOLD}{fa_arrow}{fa_arrow}  {fa_word}{C.RESET}"
              f"  score={float(result.get('meta_score', 0.0)):.3f}"
              f"  Kelly={float(result.get('unified_kelly', 0.0)):.3f}"
              f"  source: {C.CYAN}{src}{C.RESET}")
        print(f"  {gate_color}{gate_icon} Gate{C.RESET}    {gate_color}{'PASS' if gate_passed else 'BLOCK'}{C.RESET}"
              f"  mode={C.CYAN}{result.get('arbiter_mode', 'DSAC_RISK_GUARD')}{C.RESET}")
        print(
            f"  {C.CYAN}• RISK{C.RESET}    step_stop={'ON' if self.step_stop_enable else 'OFF'}"
            f"  trail={self.dsac_only_trail_arm:.3f}/{self.dsac_only_trail_gap:.3f}"
            f"  max_hold={self.dsac_only_max_hold}"
            f"  vol_scale={'ON' if self.dsac_only_vol_scale_enable else 'OFF'}"
            f"  cooldown={self.cooldown_bars_left}"
        )

        if self.pos is not None:
            unr = self.unrealized_pnl(current_price)
            pos_color = C.GREEN if self.pos == "LONG" else C.RED
            unr_color = C.GREEN if unr > 0 else (C.RED if unr < 0 else C.YELLOW)
            print(f"  {pos_color}● 포지션{C.RESET}  {pos_color}{self.pos}{C.RESET}"
                  f"  진입가={self.entry_price:.2f}  미실현={unr_color}{unr:+.2f}%{C.RESET}  보유={self.hold_count}봉")


# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher      = BinanceLiveFetcher(limit=2500)
    fe_engine    = FeatureEngineer()
    ensemble     = EnsemblePredictor() if ENSEMBLE_PREDICTOR_ENABLED else None
    live_hmm: OnlineHMMDetector | None = None
    live_hmm_steps = 0
    logger.info("🧱 부가 기능: ensemble=%s", "ON" if ENSEMBLE_PREDICTOR_ENABLED else "OFF")

    # ── DSAC + SevenModel(M7) 융합 라우터 초기화 ─────────────────────
    meta_router = DSACTrendRouter()
    meta_router.online_adapt = False
    logger.info("🧭 실행 모드: DSAC_ONLY (최소 리스크 레이어만 유지)")
    _prev_meta_pos: str | None = None

    def _sync_dsac_with_meta():
        dsac_router.pos = meta_router.pos
        dsac_router.entry_price = meta_router.entry_price
        dsac_router.hold_count = meta_router.hold_count
        dsac_router.current_leverage = meta_router.current_leverage
        dsac_router.current_equity = meta_router.cur_equity
        dsac_router.peak_equity = meta_router.peak_equity

    async def _fetch_exchange_position():
        try:
            if hasattr(fetcher.exchange, "fetch_positions"):
                positions = await fetcher.exchange.fetch_positions([fetcher.symbol])
                for p in positions or []:
                    contracts = float(p.get("contracts", p.get("positionAmt", 0.0)) or 0.0)
                    if abs(contracts) <= 1e-12:
                        continue
                    side = str(p.get("side", "")).upper()
                    if side not in {"LONG", "SHORT"}:
                        side = "LONG" if contracts > 0 else "SHORT"
                    entry = float(p.get("entryPrice", p.get("entry_price", 0.0)) or 0.0)
                    lev = float(p.get("leverage", 0.0) or 0.0)
                    return {"type": side, "entry_price": entry, "leverage": lev}
        except Exception as e:
            logger.debug("exchange.fetch_positions 복원 실패: %s", e)
        try:
            raw = await fetcher.exchange.fapiPrivateV2GetPositionRisk()
            for row in raw or []:
                if str(row.get("symbol", "")).upper() != fetcher.symbol.upper():
                    continue
                amt = float(row.get("positionAmt", 0.0) or 0.0)
                if abs(amt) <= 1e-12:
                    continue
                side = "LONG" if amt > 0 else "SHORT"
                entry = float(row.get("entryPrice", 0.0) or 0.0)
                lev = float(row.get("leverage", 0.0) or 0.0)
                return {"type": side, "entry_price": entry, "leverage": lev}
        except Exception as e:
            logger.debug("positionRisk 복원 실패: %s", e)
        return None

    def _bars_stale(eth_df: pd.DataFrame) -> bool:
        if eth_df is None or len(eth_df) == 0:
            return True
        last_ts = pd.Timestamp(eth_df['timestamp'].iloc[-1])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_localize(None)
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        age = now_utc - last_ts
        if age > pd.Timedelta(minutes=15):
            logger.warning("⚠️ 최신 봉 지연 감지: last=%s age=%s", last_ts, age)
            return True
        return False

    # ── 7-모델 통합 추세 추론기 초기화 ───────────────────────────────
    trend_hub = SevenModelEnsemble()
    unsup_hub = UnsupervisedRegimeHub()
    _m7_avail = sum([
        int(trend_hub.trend_xgb.available),
        int(trend_hub.multi_target.available),
        int(trend_hub.quantile.available),
        int(trend_hub.gmm.available),
        int(trend_hub.hdbscan.available),
        int(trend_hub.isolation.available),
        int(trend_hub.vae.available),
    ])
    logger.info("🤖 SevenModelEnsemble 로드: %d/7 모델 사용 가능", _m7_avail)
    logger.info("🧩 %s", unsup_hub.summary_line())

    # ── 텔레그램 알림 ──────────────────────────────────────────
    tg_notifier = TelegramNotifier()
    logger.info("📨 텔레그램 알림 조건: 포지션 변화(ENTER/EXIT/FLIP) 발생 시에만 전송")
    logger.info(
        f"📺 출력 모드: {'COMPACT (요약패널 전용)' if COMPACT_MODE else 'STANDARD (대시보드 + 요약패널)'}"
        f" | EXEC: DSAC_ONLY"
    )

    async def _run_cycle(processed_df, eth_buffer):
        """한 사이클: DSAC_ONLY 판단 + 집행."""
        nonlocal _prev_meta_pos
        nonlocal live_hmm_steps

        meta_router.decrement_cooldown()
        if ENSEMBLE_PREDICTOR_ENABLED and ensemble is not None:
            await ensemble.predict_all_async(processed_df)
        nf_preds = {}

        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price    = float(eth_buffer['close'].iloc[-1])
        regime_name      = 'UNKNOWN'

        # ── SevenModelEnsemble(M7) 메타 신호 추론 ────────────────────
        m7_last = None
        trend_signal = None
        try:
            m7_last = trend_hub.predict_last(processed_df)
            trend_signal = _trend_signal_from_m7(m7_last)
        except Exception as e:
            logger.warning(f"SevenModelEnsemble 추론 실패: {e}")

        # DSAC 입력 생성/추론 (m7_* 주입 버전)
        _sync_dsac_with_meta()
        dsac_action, dsac_lev, info, elite_sigs, regime = dsac_router.decide(
            processed_df,
            nf_preds,
            m7_signal=trend_signal,
        )
        if live_hmm is not None:
            live_hmm_steps += 1
            if live_hmm_steps % 24 == 0:
                try:
                    live_hmm.update_online(n_iter=3)
                    logger.info("🧠 Live HMM 온라인 업데이트 완료")
                except Exception as e:
                    logger.debug("Live HMM 온라인 업데이트 실패: %s", e)
        info.setdefault("agent", "DSAC")
        info.setdefault("kelly", float(dsac_lev))
        info.setdefault("score", float(abs(info.get("raw_action", 0.0))))
        regime_name = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')

        # ── 보조 필터 계수 계산 ─────────────────────────────────────────

        # BTC 3봉 수익률 (ETH-BTC 상관 필터용)
        _btc_3bar_ret = 0.0
        if 'close_btc' in processed_df.columns and len(processed_df) >= 4:
            _btc_arr = processed_df['close_btc'].values
            _btc_base = float(_btc_arr[-4]) if abs(float(_btc_arr[-4])) > 1e-8 else 1.0
            _btc_3bar_ret = float((_btc_arr[-1] - _btc_base) / _btc_base)

        # 안티 찹: 최근 N봉 방향 전환 횟수
        _aux_chop_factor = 1.0
        if meta_router.chop_filter_enable and len(processed_df) >= meta_router.chop_window + 2:
            _cls = processed_df['close'].values[-(meta_router.chop_window + 2):]
            _dirs = np.sign(np.diff(_cls.astype(np.float64)))
            _nonzero = _dirs[_dirs != 0]
            if len(_nonzero) >= 2:
                _turns = int(np.sum(np.diff(_nonzero) != 0))
                if _turns >= meta_router.chop_turns_max:
                    _aux_chop_factor = float(meta_router.chop_kelly_scale)

        # 거래량 확인
        _aux_volume_factor = 1.0
        if meta_router.volume_confirm_enable and 'volume' in processed_df.columns:
            _vols = processed_df['volume'].values
            if len(_vols) >= 21:
                _vol_mean = float(np.mean(_vols[-21:-1]))
                _vol_cur  = float(_vols[-1])
                if _vol_mean > 1e-8 and (_vol_cur / _vol_mean) < meta_router.volume_min_ratio:
                    _aux_volume_factor = float(meta_router.volume_low_kelly)
        _iso_anom = bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5)
        _vae_anom = bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5)
        _vae_err = float((trend_signal or {}).get("m7_vae_error", 0.0) or 0.0)
        _vae_th = float((trend_signal or {}).get("m7_vae_threshold", 0.0) or 0.0)
        _vae_ratio = (_vae_err / max(_vae_th, 1e-8)) if _vae_th > 1e-8 else (1.0 if _vae_anom else 0.0)

        # ── DSAC + M7 다요소 융합 (방향/사이즈/레짐/이상치/보유시간) ──
        prev_meta_pos = _prev_meta_pos
        if DSAC_ONLY_MODE:
            _garch_vol_z = float(processed_df.iloc[-1].get('garch_vol_z', 0.0))
            _kelly = float(np.clip(dsac_lev * meta_router.vol_scale(_garch_vol_z, 0.0), 0.0, 1.0))
            _fa = int(dsac_action)
            _dsac_only_source = "DSAC_ONLY"
            _raw_action = float(info.get("raw_action", 0.0))
            _trend_exit_score = 0.0
            _live_unr = 0.0
            if meta_router.pos is not None and meta_router.entry_price > 0:
                _live_unr = float(meta_router._net_pnl_frac(current_price))
                _reverse_action = 2 if (_raw_action <= -meta_router.dsac_only_reverse_min) else (1 if (_raw_action >= meta_router.dsac_only_reverse_min) else 0)
                _opp_reverse = (
                    (meta_router.pos == "LONG" and _reverse_action == 2)
                    or (meta_router.pos == "SHORT" and _reverse_action == 1)
                )
                _trend_exit, _trend_exit_score, _trend_exit_reason = meta_router.update_trend_mismatch(processed_df, trend_signal)
                # 단계별 수익 보호 스탑 (브레이크이븐 포함) — 기존 하드스탑 대체
                _step_floor   = meta_router.step_stop_floor()
                _cur_lev_gain = _live_unr
                if _cur_lev_gain <= _step_floor:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = ("DSAC_STEP_STOP" if meta_router.peak_equity >= 1.006
                                         else "DSAC_ONLY_HARD_STOP")
                elif meta_router.should_trailing_stop():
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_TRAILING_STOP"
                elif meta_router.hold_count >= max(1, meta_router.dsac_only_max_hold):
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_MAX_HOLD"
                elif _opp_reverse:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_REVERSE_EXIT"
                elif _trend_exit:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = _trend_exit_reason
            else:
                meta_router.trend_mismatch_streak = 0
            if meta_router.cooldown_bars_left > 0 and meta_router.pos is None and _fa != 0:
                _fa = 0
                _kelly = 0.0
                _dsac_only_source = "DSAC_ONLY_COOLDOWN"
            # 신규 진입 시에는 SIGNAL과 중기추세 정렬을 요구
            if _fa != 0 and meta_router.pos is None:
                _signal_side = 1 if _fa == 1 else -1
                _trend_dir = int((trend_signal or {}).get("trend_dir", 1))
                _trend_side = 1 if _trend_dir == 2 else (-1 if _trend_dir == 0 else 0)
                if _trend_side == 0:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_TREND_FLAT_BLOCK"
                elif _signal_side != _trend_side:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_TREND_MISMATCH_BLOCK"
            # 신규 진입 시 보조 필터(VAE/BTC/안티찹/거래량)만 적용
            if _fa != 0 and meta_router.pos is None:
                if _iso_anom and _vae_anom and _vae_ratio >= meta_router.dsac_only_vae_block_ratio:
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = "DSAC_ONLY_ISO_VAE_BLOCK"
                _btc_fac  = 1.0
                if meta_router.btc_corr_enable and abs(_btc_3bar_ret) >= meta_router.btc_corr_move_th:
                    _intended_side_dsac = 1 if _fa == 1 else -1
                    _btc_up  = _btc_3bar_ret > 0
                    _aligned = (_btc_up and _intended_side_dsac > 0) or (not _btc_up and _intended_side_dsac < 0)
                    _btc_fac = meta_router.btc_corr_align_boost if _aligned else meta_router.btc_corr_misalign
                if _fa != 0:
                    _kelly = float(np.clip(
                        _kelly * _btc_fac * _aux_chop_factor * _aux_volume_factor,
                        0.0, 1.0,
                    ))
            if _fa == 0:
                _kelly = 0.0
            meta_router._update_pos(_fa, current_price, _kelly)
            _score = float(np.clip(max(abs(float(info.get("raw_action", 0.0))), abs(float(info.get("score", 0.0))), _kelly), 0.0, 1.0))
            _side = 1 if _fa == 1 else (-1 if _fa == 2 else 0)
            meta_result = {
                "final_action": _fa,
                "unified_kelly": _kelly,
                "source": _dsac_only_source,
                "rl_score": float(info.get("score", 0.0)),
                "meta_score": _kelly if _fa != 0 else 0.0,
                "rl_action": _fa,
                "trend_signal": trend_signal,
                "trend_exit_score": float(_trend_exit_score),
                "trend_mismatch_streak": int(meta_router.trend_mismatch_streak),
                "trend_veto": None,
                "arbiter_mode": "DSAC_ONLY",
                "gate_passed": True,
                "gate_log": {},
                "fusion_diag": {
                    "direction_source": 1.0,
                    "dsac_score": _score,
                    "m7_score": 0.0,
                    "fused_signed": float(_side * _score),
                    "fused_abs": _score,
                    "cb_active": 0.0,
                    "is_lowvol_range": 0.0,
                    "is_highvol_trend": 0.0,
                    "uncertainty_scale": 1.0,
                },
            }
        rl_action = int(dsac_action)
        trade_pnl_pct: float | None = None

        # 직전 사이클에 포지션이 있었다가 이번에 청산됐으면 PnL 피드백
        if _prev_meta_pos is not None and meta_router.pos is None:
            realized = meta_router.last_realized_pnl
            if realized is None:
                realized = float(meta_router.cur_equity - 1.0)
            trade_pnl_pct = float(realized) * 100.0
            meta_router.record_outcome(float(realized))
            meta_router.append_trade_history(current_time_kst, float(realized))

        # ── 텔레그램 알림: 포지션이 바뀐 경우만 (ENTER / EXIT / FLIP) ──
        _new_pos = meta_router.pos
        if _prev_meta_pos is None and _new_pos is not None and trade_pnl_pct is None:
            trade_pnl_pct = 0.0
        if trade_pnl_pct is not None:
            meta_result["trade_pnl_pct"] = float(trade_pnl_pct)
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

        # 요약 결과 먼저 출력 (핵심 의사결정 선노출)
        _print_final_trade_summary(
            timestamp_kst=current_time_kst,
            current_price=current_price,
            regime_name=regime_name,
            rl_action=rl_action,
            rl_info=info,
            meta_result=meta_result,
            prev_pos=prev_meta_pos,
            cur_pos=meta_router.pos,
        )

        # 출력: COMPACT_MODE면 요약 패널만, 아니면 설명(상세) 로그를 아래에 출력
        if not COMPACT_MODE:
            meta_router.print_meta_dashboard(meta_result, current_price)
        logger.info("📊 %s", meta_router.performance_summary(current_time_kst))

    try:
        if use_local:
            eth_buffer, btc_buffer = fetcher.load_local_data()
        else:
            logger.info("초기 캔들 데이터 수집 중...")
            eth_buffer, btc_buffer = await fetcher.fetch_initial_data()

        if eth_buffer is None: return
        try:
            processed_boot = fe_engine.process(eth_buffer, btc_buffer)
            live_hmm = OnlineHMMDetector()
            live_hmm.fit(processed_boot, n_iter=15)
            logger.info("🧠 Live HMM 초기화 완료")
        except Exception as e:
            live_hmm = None
            logger.warning("⚠️ Live HMM 초기화 실패, fallback regime 사용: %s", e)
        try:
            dsac_router = DSACSignalRouter(hmm_detector=live_hmm)
        except Exception as e:
            logger.error(f"❌ DSAC 라우터 초기화 실패: {e}")
            return
        if not use_local:
            restored = await _fetch_exchange_position()
            if restored:
                meta_router.reconcile_external_position(restored.get("type"), float(restored.get("entry_price", 0.0)), float(restored.get("leverage", 0.0)))
        if _bars_stale(eth_buffer):
            logger.warning("⚠️ stale candle 상태로 첫 사이클 스킵")
            return

        await _run_cycle(processed_boot, eth_buffer)

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
                if _bars_stale(eth_buffer):
                    logger.warning("⚠️ 데이터 지연으로 이번 사이클 판단 스킵")
                    continue
            else:
                logger.info(f"{Colors.GREEN}🚀 봇 실시간 롤링 가동 시작!{Colors.RESET}")
                first_run = False

            processed_df = fe_engine.process(eth_buffer, btc_buffer)
            await _run_cycle(processed_df, eth_buffer)

    finally:
        await fetcher.exchange.close()


if __name__ == "__main__":
    asyncio.run(main(use_local=False))
