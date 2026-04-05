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

from features.engineering import FeatureEngineer
from features.elite import NewEliteSignalEngine
from features.m7 import trend_signal_from_m7
from features.schema import (
    STATE_PRED as DSAC_STATE_PRED,
    STATE_CONF as DSAC_STATE_CONF,
    STATE_ELITE as DSAC_STATE_ELITE,
    STATE_ALPHA as DSAC_STATE_ALPHA,
    STATE_SYNTH as DSAC_STATE_SYNTH,
    ELITE_BUILDER_REQUIRED_COLS,
    NF_RUNTIME_REQUIRED_COLS,
    build_active_feature_keep,
)
from features.registry import M7_LIVE_STRICT_COLS
from ensemble.seven_model_ensemble import SevenModelEnsemble
from ensemble.llm_advisor import LLMAdvisor, LLMDecision
from ensemble.unsupervised.live_unsupervised_hub import UnsupervisedRegimeHub
from ensemble.ensemble_router import (
    ChronosForecaster, PatchTSTForecaster, TiDEForecaster,
)
from enhanced_trading_engine import EnhancedTradingEngine
from ensemble.train_rl_agent import OnlineHMMDetector
from ensemble.train_rl_dsac_long_agent import (
    STATE_DIM as LONG_STATE_DIM,
    SigmoidActor as LongSigmoidActor,
    DSACLongRouter,
)
from ensemble.train_rl_dsac_short_agent import (
    STATE_DIM as SHORT_STATE_DIM,
    SigmoidActor as ShortSigmoidActor,
    DSACShortRouter,
)
from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM as BASE_DSAC_STATE_DIM,
    GaussianActor as BaseDSACGaussianActor,
    DSACRouter as BaseDSACRouter,
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


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


COMPACT_MODE = _env_flag('COMPACT_MODE', True)
DSAC_ONLY_MODE = True
ENSEMBLE_PREDICTOR_ENABLED = _env_flag('ENSEMBLE_PREDICTOR_ENABLED', False)
M7_ENTRY_PRICE_ENABLE = _env_flag('M7_ENTRY_PRICE_ENABLE', False)
# LIVE_STRICT_FEATURE_GUARD removed — strict validation is always enforced
DSAC_PURE_RL_MODE = _env_flag("DSAC_PURE_RL_MODE", True)
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
    return trend_signal_from_m7(m7_last)


def _confidence_from_std(std: float) -> float:
    s = max(float(std), 1e-6)
    return float(1.0 / (1.0 + s))


def _norm_tanh(x: float, scale: float) -> float:
    s = max(float(scale), 1e-8)
    return float(np.tanh(float(x) / s))


def _regime_signed(regime: dict[str, float] | None) -> float:
    if not isinstance(regime, dict):
        return 0.0
    if float(regime.get("regime_bull", 0.0)) >= 0.5:
        return 1.0
    if float(regime.get("regime_bear", 0.0)) >= 0.5:
        return -1.0
    return 0.0


def _trend_from_row(row: pd.Series | dict) -> tuple[float, float]:
    get = row.get if hasattr(row, "get") else lambda k, d=0.0: d
    mtf_1h = float(get("mtf_trend_1h", 0.0) or 0.0)
    mtf_4h = float(get("mtf_trend_4h", 0.0) or 0.0)
    trend_strength = float(np.clip(0.5 * (abs(mtf_1h) + abs(mtf_4h)), 0.0, 1.0))
    signed = float(np.sign(mtf_1h + 0.75 * mtf_4h))
    return signed, trend_strength


# ════════════════════════════════════════════════════════════════
# 1. 데이터 수집기
# ════════════════════════════════════════════════════════════════
class BinanceLiveFetcher:
    def __init__(self, symbol='ETHUSDT', timeframe='5m', limit=2500):
        self.symbol = symbol.replace('/', '')
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        self.api_retries = int(os.getenv("BINANCE_API_RETRIES", "4"))
        self.api_retry_delay_sec = float(os.getenv("BINANCE_API_RETRY_DELAY_SEC", "1.5"))

    async def _call_with_retry(self, label: str, fn):
        last_error = None
        for attempt in range(1, self.api_retries + 1):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                if attempt >= self.api_retries:
                    break
                sleep_sec = self.api_retry_delay_sec * attempt
                logger.warning(
                    "⚠️ %s 실패(%d/%d): %s | %.1fs 후 재시도",
                    label,
                    attempt,
                    self.api_retries,
                    e,
                    sleep_sec,
                )
                await asyncio.sleep(sleep_sec)
        raise RuntimeError(f"{label} failed after {self.api_retries} attempts") from last_error

    def load_local_data(self):
        try:
            eth_df = pd.read_csv('data/test/eth_test_data.csv')
            btc_df = pd.read_csv('data/test/btc_test_data.csv')
            for df in [eth_df, btc_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cols = df.columns.drop('timestamp')
                df[cols] = df[cols].apply(pd.to_numeric, errors='raise')
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
            klines = await self._call_with_retry(
                f"fetch_klines_raw[{symbol}]",
                lambda: self.exchange.fapiPublicGetKlines(params),
            )
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
        eth_df[eth_df.columns.drop('timestamp')] = eth_df[eth_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='raise')

        btc_df = pd.DataFrame(btc_klines).iloc[:, [0, 4, 5, 7]]
        btc_df.columns = ['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df[btc_df.columns.drop('timestamp')] = btc_df[btc_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='raise')

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
                if isinstance(res, Exception):
                    raise RuntimeError(f"ancillary[{idx}] fetch failed for {new_name}: {res}") from res
                if isinstance(res, list) and len(res) > 0:
                    try:
                        temp_df = pd.DataFrame(res)
                        t_col = next((c for c in ['timestamp', 'fundingTime', 'time'] if c in temp_df.columns), None)
                        if t_col and key in temp_df.columns:
                            subset = temp_df[[t_col, key]].rename(columns={t_col: 'timestamp', key: new_name})
                            subset['timestamp'] = pd.to_datetime(subset['timestamp'], unit='ms')
                            subset[new_name] = pd.to_numeric(subset[new_name], errors='raise')
                            # Prevent look-ahead in live merge: use only latest known ancillary value.
                            eth_df = pd.merge_asof(
                                eth_df.sort_values('timestamp'),
                                subset.sort_values('timestamp'),
                                on='timestamp',
                                direction='backward',
                            )
                    except Exception: raise
        eth_df = eth_df.ffill().bfill()
        nan_cols = [c for c in eth_df.columns if eth_df[c].isna().any()]
        if nan_cols:
            raise RuntimeError(f"NaN values remain after ffill+bfill in columns: {', '.join(nan_cols)}")
        required = [
            'sum_open_interest_value',
            'sum_toptrader_long_short_ratio',
            'count_long_short_ratio',
            'last_funding_rate',
        ]
        missing = [c for c in required if c not in eth_df.columns]
        if missing:
            raise RuntimeError(f"ancillary columns missing after merge: {','.join(missing)}")
        return eth_df, btc_df

    async def fetch_initial_data(self):
        eth_klines = await self.fetch_klines_raw(self.symbol, self.limit)
        btc_klines = await self.fetch_klines_raw('BTCUSDT', self.limit)
        ancillary = await self.fetch_ancillary_data(500)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

    async def fetch_latest_patch(self):
        eth_klines = await self._call_with_retry(
            f"fetch_latest_patch[{self.symbol}]",
            lambda: self.exchange.fapiPublicGetKlines({'symbol': self.symbol, 'interval': self.timeframe, 'limit': 5}),
        )
        btc_klines = await self._call_with_retry(
            "fetch_latest_patch[BTCUSDT]",
            lambda: self.exchange.fapiPublicGetKlines({'symbol': 'BTCUSDT', 'interval': self.timeframe, 'limit': 5}),
        )
        ancillary = await self.fetch_ancillary_data(5)
        return self._process_to_df(eth_klines, btc_klines, ancillary)


# ════════════════════════════════════════════════════════════════
# 2-A. 대시보드용 6대 파운데이션 앙상블 (표시 전용)
# ════════════════════════════════════════════════════════════════
class EnsemblePredictor:
    MODEL_ORDER = ['PatchTST', 'Chronos', 'TiDE']

    def __init__(self):
        self.models = {
            'PatchTST': PatchTSTForecaster(),
            'Chronos': ChronosForecaster(),
            'TiDE': TiDEForecaster(),
        }
        self.last_trace: list[dict[str, object]] = []

    async def predict_all_async(self, df: pd.DataFrame):
        preds, confs = [], []
        # NOTE:
        # PatchTST/TiDE는 UnifiedNFForecaster의 동일 NF 인스턴스를 공유하므로
        # 병렬 추론 시 간헐적 race condition으로 한 모델 출력이 누락될 수 있다.
        # DSAC strict guard 안정성을 위해 순차 추론으로 고정한다.
        results = []
        for name in self.MODEL_ORDER:
            m = self.models[name]
            if not getattr(m, 'available', False):
                results.append(None)
                continue
            try:
                results.append(m.predict(df, horizon=6))
            except Exception as e:
                logger.warning("⚠️ %s 추론 실패: %s", name, e)
                results.append(None)

        def _extract_last_conf(res) -> float:
            try:
                c = getattr(res, "confidence", None)
                if c is None:
                    return float("nan")
                arr = np.asarray(c, dtype=np.float32)
                if arr.ndim == 0:
                    v = float(arr)
                elif arr.ndim == 1:
                    v = float(arr[-1])
                else:
                    v = float(arr[-1][-1])
                return v if np.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        traces: list[dict[str, object]] = []
        for name, res in zip(self.MODEL_ORDER, results):
            p_val, c_val = float("nan"), float("nan")
            conf_src = "none"
            traj_last = float("nan")
            traj_std = float("nan")
            traj_zero_like = False
            if res is not None and getattr(res, 'median', None) is not None:
                traj = np.array(res.median[-1], dtype=np.float32)
                if np.all(np.isfinite(traj)):
                    traj_last = float(traj[-1]) if traj.size > 0 else float("nan")
                    traj_std = float(np.std(traj)) if traj.size > 0 else float("nan")
                    traj_zero_like = bool(np.allclose(traj, 0.0, atol=1e-9))
                    p_val = _traj_direction(traj)
                    # 1) 모델 confidence 우선 사용
                    c_val = _extract_last_conf(res)
                    conf_src = "model"
                    # 2) confidence 비정상일 때만 궤적 기반 보정
                    if not np.isfinite(c_val):
                        c_val = _traj_conf(traj)
                        conf_src = "traj_fallback"
                    c_val = float(np.clip(c_val, 0.0, 1.0))
            traces.append({
                "model": name,
                "pred": float(p_val) if np.isfinite(p_val) else float("nan"),
                "conf": float(c_val) if np.isfinite(c_val) else float("nan"),
                "traj_last": traj_last,
                "traj_std": traj_std,
                "traj_zero_like": traj_zero_like,
                "conf_src": conf_src,
                "ok": bool(np.isfinite(p_val) and np.isfinite(c_val)),
                "is_zero": bool(np.isfinite(p_val) and np.isfinite(c_val) and abs(float(p_val)) < 1e-12 and abs(float(c_val)) < 1e-12),
            })
            preds.append(p_val)
            confs.append(c_val)
        self.last_trace = traces

        # 모델별 pred/conf 추적 로그 (NaN/0 주입 여부 추적용)
        try:
            _parts = []
            for t in traces:
                _p = t["pred"]
                _c = t["conf"]
                _p_s = "nan" if not np.isfinite(_p) else f"{float(_p):+.4f}"
                _c_s = "nan" if not np.isfinite(_c) else f"{float(_c):.4f}"
                _ts = t.get("traj_std", float("nan"))
                _ts_s = "nan" if not np.isfinite(_ts) else f"{float(_ts):.6f}"
                _z = "Z0" if bool(t.get("traj_zero_like", False)) else "Z-"
                _flag = "OK" if t["ok"] else "MISS"
                _parts.append(f"{t['model']}:{_flag}(pred={_p_s},conf={_c_s},src={t['conf_src']},std={_ts_s},{_z})")
            logger.info("🔎 DSAC pred/conf 추적: %s", " | ".join(_parts))
        except Exception:
            pass

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
        f"🌍 {regime_name}   Kelly: {kelly:.3f}{pnl_line}\n"
        f"📈 Trend: {t_dir}   Source: {meta_result.get('source', 'DSAC_ONLY')}"
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
                row = sched.iloc[0]
                ts_min = ts_utc.floor("min")
                market_open = pd.Timestamp(row.get("market_open"))
                market_close = pd.Timestamp(row.get("market_close"))
                break_start = row.get("break_start", pd.NaT)
                break_end = row.get("break_end", pd.NaT)
                in_main = bool(market_open <= ts_min <= market_close)
                in_break = False
                if pd.notna(break_start) and pd.notna(break_end):
                    break_start = pd.Timestamp(break_start)
                    break_end = pd.Timestamp(break_end)
                    in_break = bool(break_start <= ts_min < break_end)
                active = bool(in_main and not in_break)
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

    def _ambiguity_text(v: float) -> str:
        if v >= 2.0:
            return "양방향 충돌 큼"
        if v >= 1.0:
            return "양방향 경합"
        if v >= 0.0:
            return "약한 경합"
        return "방향 분리 양호"

    def _confidence_text(v: float) -> str:
        if v >= 0.65:
            return "안정적"
        if v >= 0.50:
            return "보통"
        if v >= 0.35:
            return "불안정"
        return "매우 불안정"

    def _conviction_text(v: float) -> str:
        if v >= 1.0:
            return "진입 강함"
        if v >= 0.60:
            return "진입 가능"
        if v >= 0.30:
            return "진입 약함"
        return "진입 부족"

    def _agreement_text(v: float) -> str:
        if v >= 1.5:
            return "방향 우위 뚜렷"
        if v >= 0.8:
            return "방향 우위 있음"
        if v >= 0.4:
            return "방향 우위 약함"
        return "방향 혼재"

    def _hibernation_text(v: float) -> str:
        if v >= 0.85:
            return "시장 과열/이상"
        if v >= 0.60:
            return "이상치 주의"
        if v >= 0.30:
            return "약한 이상 신호"
        return "정상 범위"

    def _amihud_text(v: float) -> str:
        if v >= 1.5:
            return "유동성 매우 나쁨"
        if v >= 0.8:
            return "유동성 나쁨"
        if v >= 0.2:
            return "유동성 보통"
        return "유동성 양호"

    def _gate(ok: bool, label: str, detail: str = "") -> str:
        """VALUE/TH[✓] 형식 컬러 게이트 토큰."""
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        text = label + (f"/{detail}" if detail else "")
        return f"{col}{text}[{icon}]{C.RESET}"

    def _status_badge(ok: bool, ok_label: str = "PASS", fail_label: str = "FAIL") -> str:
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        label = ok_label if ok else fail_label
        return f"{col}[{label} {icon}]{C.RESET}"

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
    primary_action = int(rl_info.get("primary_action", 0))
    primary_raw = float(rl_info.get("primary_raw", 0.0))
    primary_kelly = float(rl_info.get("primary_kelly", 0.0))
    target_action = int(rl_info.get("target_action", 0))
    net_score = float(rl_info.get("net_score", 0.0))
    agreement_count = int(rl_info.get("agreement_count", 0))
    rl_kelly = float(rl_info.get('kelly', 0.0))
    long_raw = float(rl_info.get('_long_raw', long_edge))
    short_raw = float(rl_info.get('_short_raw', short_edge))
    long_action = int(rl_info.get('_long_action', 1 if long_raw > 0.0 else 0))
    short_action = int(rl_info.get('_short_action', 2 if short_raw > 0.0 else 0))
    long_kelly = float(rl_info.get('_long_kelly', long_raw))
    short_kelly = float(rl_info.get('_short_kelly', short_raw))
    conviction = float(rl_info.get('conviction', abs(long_edge - short_edge)))
    agreement = float(rl_info.get('agreement', abs(long_edge - short_edge)))
    ambiguity = float(rl_info.get('ambiguity', min(long_edge, short_edge)))
    confidence = float(rl_info.get('confidence', 0.0))
    selected_side = str(rl_info.get('_selected_side', 'HOLD'))
    final_kelly = float(meta_result.get('unified_kelly', 0.0))
    source = str(meta_result.get('source', 'N/A'))
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
    entry_price_reco = 0.0
    tp_price_reco = 0.0
    sl_price_reco = 0.0
    entry_offset_reco = 0.0
    tp_offset_reco = 0.0
    sl_offset_reco = 0.0
    cb_active = int(meta_result.get("cb_active", 0) or 0) if isinstance(meta_result, dict) else 0
    is_lowvol_range = int(meta_result.get("is_lowvol_range", 0) or 0) if isinstance(meta_result, dict) else 0
    is_highvol_trend = int(meta_result.get("is_highvol_trend", 0) or 0) if isinstance(meta_result, dict) else 0
    uncertainty_scale = float(meta_result.get("uncertainty_scale", 1.0)) if isinstance(meta_result, dict) else 1.0
    trend_exit_score = float(meta_result.get("trend_exit_score", 0.0)) if isinstance(meta_result, dict) else 0.0
    trend_mismatch_streak = int(meta_result.get("trend_mismatch_streak", 0) or 0) if isinstance(meta_result, dict) else 0
    hibernation_score = float(meta_result.get("hibernation_score", 0.0)) if isinstance(meta_result, dict) else 0.0
    integral_gap_ma = float(meta_result.get("integral_gap_ma", 0.0)) if isinstance(meta_result, dict) else 0.0
    illiq_amihud = float(meta_result.get("illiq_amihud", 0.0)) if isinstance(meta_result, dict) else 0.0
    illiq_rsvol = float(meta_result.get("illiq_rsvol", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_bayes_mult = float(meta_result.get("sizing_bayes_mult", 1.0)) if isinstance(meta_result, dict) else 1.0
    sizing_qwidth_mult = float(meta_result.get("sizing_qwidth_mult", 1.0)) if isinstance(meta_result, dict) else 1.0
    sizing_mtf_mult = float(meta_result.get("sizing_mtf_mult", 1.0)) if isinstance(meta_result, dict) else 1.0
    sizing_smart_mult = float(meta_result.get("sizing_smart_mult", 1.0)) if isinstance(meta_result, dict) else 1.0
    sizing_mdd_mult = float(meta_result.get("sizing_mdd_mult", 1.0)) if isinstance(meta_result, dict) else 1.0
    sizing_bayes_z = float(meta_result.get("sizing_bayes_z", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_qwidth = float(meta_result.get("sizing_qwidth", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_mtf_align = int(meta_result.get("sizing_mtf_align", 0) or 0) if isinstance(meta_result, dict) else 0
    sizing_smart_flow = float(meta_result.get("sizing_smart_flow", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_taker_accel = float(meta_result.get("sizing_taker_accel", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_recent_pnl_sum = float(meta_result.get("sizing_recent_pnl_sum", 0.0)) if isinstance(meta_result, dict) else 0.0
    sizing_loss_streak = int(meta_result.get("sizing_loss_streak", 0) or 0) if isinstance(meta_result, dict) else 0
    position_signal = str(meta_result.get("position_signal", "")) if isinstance(meta_result, dict) else ""
    position_reason = str(meta_result.get("position_reason", "")) if isinstance(meta_result, dict) else ""
    position_own_support = float(meta_result.get("position_own_support", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_opp_pressure = float(meta_result.get("position_opp_pressure", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_net_edge = float(meta_result.get("position_net_edge", 0.0)) if isinstance(meta_result, dict) else 0.0
    hold_reason = str(meta_result.get("hold_reason", "")) if isinstance(meta_result, dict) else ""
    block_reason = str(meta_result.get("block_reason", "")) if isinstance(meta_result, dict) else ""
    router_enter_threshold = float(meta_result.get("router_enter_threshold", 0.0)) if isinstance(meta_result, dict) else 0.0
    router_min_agreement_threshold = float(meta_result.get("router_min_agreement_threshold", 0.0)) if isinstance(meta_result, dict) else 0.0
    router_max_confidence_std = float(meta_result.get("router_max_confidence_std", 1.50)) if isinstance(meta_result, dict) else 1.50
    adaptive_enter_offset = float(meta_result.get("adaptive_enter_offset", 0.0)) if isinstance(meta_result, dict) else 0.0
    adaptive_agreement_offset = float(meta_result.get("adaptive_agreement_offset", 0.0)) if isinstance(meta_result, dict) else 0.0
    router_std_gate_ok = bool(meta_result.get("router_std_gate_ok", True)) if isinstance(meta_result, dict) else True
    router_dual_high_hold = bool(meta_result.get("router_dual_high_hold", False)) if isinstance(meta_result, dict) else False
    long_logit = float(rl_info.get("long_logit", 0.0))
    short_logit = float(rl_info.get("short_logit", 0.0))
    long_std = float(rl_info.get("long_std", 1.0))
    short_std = float(rl_info.get("short_std", 1.0))
    selected_std = float(rl_info.get("selected_std", long_std if long_raw >= short_raw else short_std))
    router_max_confidence_std = float(rl_info.get("max_confidence_std", 1.50))
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
        m7_qwidth = float(max(0.0, ts.get("m7_qwidth", meta_result.get("m7_qwidth", 0.0) if isinstance(meta_result, dict) else 0.0)))
        m7_iso_anom = 1 if float(ts.get("m7_iso_anom", 0.0)) >= 0.5 else 0
        m7_vae_anom = 1 if float(ts.get("m7_vae_anom", 0.0)) >= 0.5 else 0
        if t_dir == 2:
            entry_price_reco = float(ts.get("m7_entry_long_price", 0.0))
            entry_offset_reco = float(ts.get("m7_entry_long_offset", 0.0))
        elif t_dir == 0:
            entry_price_reco = float(ts.get("m7_entry_short_price", 0.0))
            entry_offset_reco = float(ts.get("m7_entry_short_offset", 0.0))
        tp_price_reco = float(ts.get("m7_tp_price", 0.0))
        sl_price_reco = float(ts.get("m7_sl_price", 0.0))
        tp_offset_reco = float(ts.get("m7_tp_offset", 0.0))
        sl_offset_reco = float(ts.get("m7_sl_offset", 0.0))

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
    long_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(long_action), '?')
    short_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(short_action), '?')

    rl_word = _action_word(rl_action)
    rl_color = _action_color(rl_action)
    final_word = _action_word(fa)
    final_color = _action_color(fa)
    trend_word = _trend_word(t_dir)
    trend_color = _trend_color(t_dir)
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
    # ── 헤더 ────────────────────────────────────────────────────────
    print(_SEP2)
    ts_str = timestamp_kst.strftime('%Y-%m-%d %H:%M')
    session_flags = _session_flags_from_timestamp(timestamp_kst)
    session_parts = []
    for label, key in (("ASIA", "session_asia"), ("EUROPE", "session_europe"), ("US", "session_us")):
        active = float(session_flags.get(key, 0.0)) >= 0.5
        scol = C.GREEN if active else C.YELLOW
        sword = "ON" if active else "OFF"
        session_parts.append(f"{label}={scol}{sword}{C.RESET}")
    header_left = f"{final_color}{C.BOLD}{fa_arrow}{fa_arrow}  {final_word}  →  {ex_code}{C.RESET}"
    print(f" {header_left}  {C.CYAN}{ts_str}  ${current_price:,.2f}{C.RESET}")
    print(f"     {C.CYAN}{regime_name}{C.RESET}  {'  '.join(session_parts)}")
    print(_SEP)

    # ── 신호 / DSAC 엔진 ─────────────────────────────────────────────
    print(f"  {rl_color}{rl_arrow} 신호{C.RESET}  {rl_color}{rl_word:<6}{C.RESET}"
          f" {edge_side_color}{edge_side_word} {edge_gap:+.3f}{C.RESET}"
          f"  Kelly: {_bar(final_kelly, 8)} {final_kelly:.3f} ({_kelly_text(final_kelly)})")
    print(
        f"  {C.CYAN}• DSAC{C.RESET}  "
        f"L:{long_agent_arrow}{_action_word(long_action):<5} r={C.GREEN}{long_raw:.3f}{C.RESET} k={long_kelly:.3f}"
        f"  S:{short_agent_arrow}{_action_word(short_action):<5} r={C.RED}{short_raw:.3f}{C.RESET} k={short_kelly:.3f}"
    )
    print(
        f"  {C.CYAN}• Primary{C.RESET} "
        f"{_action_arrow(primary_action)}{_action_word(primary_action):<5}"
        f" raw={primary_raw:+.3f} k={primary_kelly:.3f}"
        f"  → target={_action_word(target_action):<5}"
        f" net={net_score:+.3f} votes={agreement_count}"
    )
    print(
        f"          → 결정 = {selected_side:<6}"
        f"  conv={conviction:.3f} ({_conviction_text(conviction)})"
        f"  agr={agreement:.3f} ({_agreement_text(agreement)})"
    )
    print(
        f"  {C.CYAN}• 점수{C.RESET}  "
        f"L={C.GREEN}{long_logit:+.2f}{C.RESET}(±{long_std:.2f})"
        f"  S={C.RED}{short_logit:+.2f}{C.RESET}(±{short_std:.2f})"
        f"  amb={ambiguity:+.2f} ({_ambiguity_text(ambiguity)})"
        f"  conf={confidence:.3f}"
    )

    # ── 추세 / M7 ────────────────────────────────────────────────────
    dn_c = C.RED if p_dn > 0.4 else C.RESET
    up_c = C.GREEN if p_up > 0.4 else C.RESET
    trend_model = str(ts.get("trend_model", "N/A")) if isinstance(ts, dict) else "N/A"
    print(f"  {trend_color}{trend_arrow} 추세{C.RESET}    {trend_color}{trend_word:<6}{C.RESET}"
          f"  str={t_strength:.2f}  rev={t_rev:.2f}"
          f"  {dn_c}DN={p_dn:.0%}{C.RESET} FL={C.YELLOW}{p_fl:.0%}{C.RESET} {up_c}UP={p_up:.0%}{C.RESET}"
          f"  [{trend_model}]")
    if entry_price_reco > 0.0 or tp_price_reco > 0.0 or sl_price_reco > 0.0:
        print(
            f"  {C.CYAN}• 가격{C.RESET}    진입={entry_price_reco:,.2f}({entry_offset_reco:+.3%})"
            f"  TP={tp_price_reco:,.2f}({tp_offset_reco:+.3%})"
            f"  SL={sl_price_reco:,.2f}({sl_offset_reco:+.3%})"
        )

    # ── 보호 / HOLD ──────────────────────────────────────────────────
    print(
        f"  {C.CYAN}• 보호{C.RESET}    hib={hibernation_score:.2f} ({_hibernation_text(hibernation_score)})"
        f"  cb={cb_active}  amihud={illiq_amihud:.2f} ({_amihud_text(illiq_amihud)})"
    )
    if hold_reason or block_reason:
        print(
            f"  {C.CYAN}• HOLD{C.RESET}    {C.YELLOW}{hold_reason or '-'}{C.RESET}"
            f"  block={C.RED}{block_reason or '-'}{C.RESET}"
        )

    # ── 진입/청산 장벽 ───────────────────────────────────────────────
    _br = block_reason or ""
    _conv_ok = conviction >= router_enter_threshold
    _agr_ok  = agreement  >= router_min_agreement_threshold
    _std_ok  = router_std_gate_ok
    _dual_ok = not router_dual_high_hold
    hibernation_score_th = float(meta_result.get("hibernation_score_th", 0.85)) if isinstance(meta_result, dict) else 0.85
    _hib_ok  = hibernation_score < hibernation_score_th
    _cb_ok   = cb_active == 0
    _trend_ok = "trend" not in _br
    _intg_ok  = "integral" not in _br
    _cool_ok  = "cooldown" not in _br

    if cur_pos is None:
        # 라우터 게이트 (행 1)
        g_conv = _gate(_conv_ok, f"CONV={conviction:.3f}", f"{router_enter_threshold:.3f}")
        g_agr  = _gate(_agr_ok,  f"AGR={agreement:.3f}",  f"{router_min_agreement_threshold:.3f}")
        g_std  = _gate(_std_ok,  f"STD={selected_std:.2f}", f"{router_max_confidence_std:.2f}")
        g_dual = _gate(_dual_ok, f"DUAL={ambiguity:.2f}")
        entry_result = _status_badge(final_word != "HOLD", "PASS", "FAIL")
        print(f"  {C.CYAN}• 진입장벽{C.RESET}  {entry_result}  {g_conv}  {g_agr}  {g_std}  {g_dual}")
        # 보호 게이트 (행 2)
        g_hib  = _gate(_hib_ok,  f"HIB={hibernation_score:.2f}", f"{hibernation_score_th:.2f}")
        g_cb   = _gate(_cb_ok,   "CB")
        g_trend = _gate(_trend_ok, "TREND")
        row2 = [g_hib, g_cb, g_trend]
        if not _intg_ok:
            row2.append(_gate(False, "INTG"))
        if not _cool_ok:
            row2.append(_gate(False, "COOL"))
        if adaptive_enter_offset != 0.0 or adaptive_agreement_offset != 0.0:
            row2.append(f"{C.CYAN}적응={adaptive_enter_offset:+.3f}/{adaptive_agreement_offset:+.3f}{C.RESET}")
        print(f"             {'  '.join(row2)}")
    else:
        _own_ok = position_own_support >= 1.10
        _opp_ok = position_opp_pressure < 0.90
        _net_ok = position_net_edge > -0.10
        g_own = _gate(_own_ok, f"OWN={position_own_support:.2f}", "1.10")
        g_opp = _gate(_opp_ok, f"OPP={position_opp_pressure:.2f}", "0.90")
        g_net = _gate(_net_ok, f"NET={position_net_edge:+.2f}", "−0.10")
        if position_signal == "EXIT":
            manage_result = _status_badge(False, "유지", "청산")
            g_action = _gate(True, f"EXIT:{position_reason or '-'}")
        elif position_signal == "REDUCE":
            manage_result = f"{C.YELLOW}[축소!]{C.RESET}"
            g_action = _gate(True, f"REDUCE:{position_reason or '-'}")
        else:
            manage_result = _status_badge(True, "유지", "청산")
            g_action = _gate(True, f"HOLD:{position_reason or 'ok'}")
        print(f"  {C.CYAN}• 청산장벽{C.RESET}  {manage_result}  {g_own}  {g_opp}  {g_net}  {g_action}")

    # ── 체결 이벤트 ──────────────────────────────────────────────────
    if prev_pos != cur_pos:
        trade_pnl = meta_result.get("trade_pnl_pct", None)
        if trade_pnl is None and prev_pos is None and cur_pos is not None:
            trade_pnl = 0.0
        if trade_pnl is not None:
            try:
                p = float(trade_pnl)
                p_col = C.GREEN if p > 0 else (C.RED if p < 0 else C.YELLOW)
                print(f"  {C.CYAN}• TRADE{C.RESET}   pnl={p_col}{p:+.2f}%{C.RESET}")
            except Exception:
                pass

    # ── 의사결정 체인 ────────────────────────────────────────────────
    print(f"  {C.CYAN}• 소스{C.RESET}    {source}")
    print(_SEP)
    decision_chain = (
        f"SIGNAL={rl_color}{rl_word}{C.RESET} → "
        f"추세={trend_color}{trend_word}{C.RESET} → "
        f"FINAL={final_color}{final_word}{C.RESET} → "
        f"EXEC={ex_icon} {ex_code}"
    )
    print(f"  {decision_chain}")
    print(_SEP2)


# ════════════════════════════════════════════════════════════════
# 3-A. DSACSignalRouter — DSAC Actor 추론 입력 생성 + 추론
# ════════════════════════════════════════════════════════════════
class DSACSignalRouter:
    DEFAULT_SINGLE_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth"
    DEFAULT_LONG_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_long_agents.pth"
    DEFAULT_SHORT_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_short_agents.pth"
    LEGACY_SINGLE_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_agent.pth"
    LEGACY_LONG_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_long.pth"
    LEGACY_SHORT_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_short.pth"

    @staticmethod
    def _build_primary_actor_from_ckpt(ckpt: dict, device: str):
        actor_state = ckpt.get("actor")
        if not isinstance(actor_state, dict):
            raise KeyError("DSAC primary 체크포인트 actor 키 없음")
        state_dim = int(ckpt.get("state_dim", BASE_DSAC_STATE_DIM) or BASE_DSAC_STATE_DIM)
        actor = BaseDSACGaussianActor(state_dim=state_dim).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor, "DSAC_PRIMARY"

    @staticmethod
    def _build_actor_from_ckpt(ckpt: dict, device: str, side: str):
        actor_state = ckpt.get("actor")
        if not isinstance(actor_state, dict):
            raise KeyError("DSAC 체크포인트 actor 키 없음")

        if side == "long":
            state_dim = int(ckpt.get("state_dim", LONG_STATE_DIM) or LONG_STATE_DIM)
            actor = LongSigmoidActor(state_dim=state_dim).to(device)
        elif side == "short":
            state_dim = int(ckpt.get("state_dim", SHORT_STATE_DIM) or SHORT_STATE_DIM)
            actor = ShortSigmoidActor(state_dim=state_dim).to(device)
        else:
            raise ValueError(f"지원하지 않는 DSAC specialist side: {side}")
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor, f"DSAC_{side.upper()}"

    @staticmethod
    def _resolve_model_path(primary: str | None, *fallbacks: str) -> str:
        for candidate in (primary, *fallbacks):
            if candidate and os.path.exists(candidate):
                return candidate
        searched = [c for c in (primary, *fallbacks) if c]
        raise FileNotFoundError(f"DSAC specialist 체크포인트 파일이 없습니다: {searched}")

    @staticmethod
    def _regime_name(regime: dict[str, float] | None) -> str:
        if not isinstance(regime, dict):
            return "normal"
        return next((k.replace("regime_", "") for k, v in regime.items() if float(v) == 1.0), "normal")

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "cuda" in msg or "acceleratorerror" in exc.__class__.__name__.lower()

    def _entry_threshold(self, regime: dict[str, float] | None, trend_signal: dict | None, ambiguity: float) -> float:
        regime_name = self._regime_name(regime)
        trend_dir = int((trend_signal or {}).get("trend_dir", 1))
        enter_th = self.base_enter_th
        if regime_name == "bull" and trend_dir == 2:
            enter_th = self.trend_align_enter_th
        elif regime_name == "bear" and trend_dir == 0:
            enter_th = self.trend_align_enter_th
        elif regime_name == "chop":
            enter_th = self.chop_enter_th
        elif regime_name == "whipsaw":
            enter_th = self.whipsaw_enter_th
        if float(ambiguity) > self.ambiguity_penalty_start:
            enter_th += self.ambiguity_penalty_add
        enter_th += float(getattr(self, "adaptive_enter_offset", 0.0))
        return float(np.clip(enter_th, 0.05, 0.90))

    def _effective_min_agreement(self) -> float:
        base = float(self.min_agreement_th)
        offset = float(getattr(self, "adaptive_agreement_offset", 0.0))
        return float(np.clip(base + offset, 0.05, 0.95))

    @staticmethod
    def _support(logit: float, std: float) -> float:
        return float(logit * _confidence_from_std(std))

    def _base_signal_info(
        self,
        primary_action: int,
        primary_lev: float,
        primary_raw: float,
        long_action: int,
        long_lev: float,
        long_raw: float,
        long_logit: float,
        long_std: float,
        short_action: int,
        short_lev: float,
        short_raw: float,
        short_logit: float,
        short_std: float,
    ) -> dict[str, float | str | int]:
        direction_score = float(long_logit - short_logit)
        agreement = float(abs(direction_score))
        ambiguity = float(min(long_logit, short_logit))
        selected_std = float(long_std if direction_score >= 0.0 else short_std)
        confidence = float(1.0 / (1.0 + selected_std))
        conviction = float(agreement * confidence)
        return {
            "agent": "DSAC_DUAL",
            "long_edge": float(long_raw),
            "short_edge": float(short_raw),
            "raw_action": direction_score,
            "direction_score": direction_score,
            "agreement": agreement,
            "ambiguity": ambiguity,
            "conviction": conviction,
            "confidence": confidence,
            "selected_std": selected_std,
            "primary_action": int(primary_action),
            "primary_kelly": float(primary_lev),
            "primary_raw": float(primary_raw),
            "long_logit": float(long_logit),
            "short_logit": float(short_logit),
            "long_std": float(long_std),
            "short_std": float(short_std),
            "_long_raw": float(long_raw),
            "_short_raw": float(short_raw),
            "_long_action": int(long_action),
            "_short_action": int(short_action),
            "_long_kelly": float(long_lev),
            "_short_kelly": float(short_lev),
            "_selected_side": "HOLD",
        }

    def _entry_rule(
        self,
        regime: dict[str, float] | None,
        trend_signal: dict | None,
        long_action: int,
        long_lev: float,
        long_info: dict,
        short_action: int,
        short_lev: float,
        short_info: dict,
        base_info: dict[str, float | str | int],
    ) -> tuple[int, float, dict[str, float | str | int]]:
        primary_action = int(base_info.get("primary_action", 0))
        primary_lev = float(base_info.get("primary_kelly", 0.0))
        primary_raw = float(base_info.get("primary_raw", 0.0))
        regime_name = self._regime_name(regime)
        ambiguity = float(base_info["ambiguity"])
        long_raw = float(base_info["long_edge"])
        short_raw = float(base_info["short_edge"])
        enter_th = self._entry_threshold(regime, trend_signal, ambiguity)
        effective_agreement_th = self._effective_min_agreement()

        if regime_name == "bull":
            w_p, w_l, w_s = 0.35, 0.50, 0.15
        elif regime_name == "bear":
            w_p, w_l, w_s = 0.35, 0.15, 0.50
        elif regime_name == "normal":
            w_p, w_l, w_s = 0.50, 0.25, 0.25
        else:  # chop / whipsaw
            w_p, w_l, w_s = 0.30, 0.35, 0.35

        long_logit = float(base_info["long_logit"])
        short_logit = float(base_info["short_logit"])
        long_std = float(base_info["long_std"])
        short_std = float(base_info["short_std"])
        long_support = float(np.clip(self._support(long_logit, long_std), 0.0, 1.5))
        short_support = float(np.clip(self._support(short_logit, short_std), 0.0, 1.5))

        p_long = max(primary_raw, 0.0)
        p_short = max(-primary_raw, 0.0)
        p_mag = max(abs(primary_raw), primary_lev)
        if primary_action == 1 and p_long < 1e-8:
            p_long = p_mag
        elif primary_action == 2 and p_short < 1e-8:
            p_short = p_mag

        long_score = float(w_p * p_long + w_l * long_support)
        short_score = float(w_p * p_short + w_s * short_support)
        net_score = float(long_score - short_score)
        agreement = float(abs(net_score))
        required_score = float(max(enter_th, effective_agreement_th))
        target_action = 1 if net_score > 0 else (2 if net_score < 0 else 0)

        if target_action == 1:
            selected_side = "LONG"
            selected_action = int(long_action)
            selected_lev = float(long_lev)
            selected_info = dict(long_info or {})
            selected_logit = long_logit
            selected_std = long_std
            opp_logit = short_logit
            opp_std = short_std
        elif target_action == 2:
            selected_side = "SHORT"
            selected_action = int(short_action)
            selected_lev = float(short_lev)
            selected_info = dict(short_info or {})
            selected_logit = short_logit
            selected_std = short_std
            opp_logit = long_logit
            opp_std = long_std
        else:
            selected_side = "HOLD"
            selected_action = 0
            selected_lev = 0.0
            selected_info = {}
            selected_logit = 0.0
            selected_std = 1.0
            opp_logit = 0.0
            opp_std = 1.0

        confidence = _confidence_from_std(selected_std)
        conviction = float(agreement * confidence)
        std_gate_ok = bool(selected_std <= self.max_confidence_std)
        opp_confidence = float(_confidence_from_std(opp_std))
        opp_support = float(self._support(opp_logit, opp_std))
        opp_veto = bool(
            target_action in (1, 2)
            and opp_support >= self.opp_veto_support_th
            and opp_std <= self.opp_veto_std_max
        )

        primary_vote = 0
        if p_long >= self.ens_vote_min_strength:
            primary_vote = 1
        elif p_short >= self.ens_vote_min_strength:
            primary_vote = -1
        long_vote = 1 if int(long_action) == 1 else 0
        short_vote = -1 if int(short_action) == 2 else 0
        target_sign = 1 if target_action == 1 else (-1 if target_action == 2 else 0)
        agreement_count = int(sum(1 for v in (primary_vote, long_vote, short_vote) if v == target_sign)) if target_sign != 0 else 0
        if agreement_count >= 3:
            agreement_mult = float(self.ens_agree_mult_3)
        elif agreement_count == 2:
            agreement_mult = float(self.ens_agree_mult_2)
        elif agreement_count == 1:
            agreement_mult = float(self.ens_agree_mult_1)
        else:
            agreement_mult = 0.0

        # agreement_count별 차등 진입 임계값:
        # 전원 합의(3)면 낮은 문턱, 2개 합의면 높은 문턱 요구
        if agreement_count >= 3:
            required_score = max(0.12, effective_agreement_th)
        elif agreement_count == 2:
            required_score = max(0.08, min(required_score, 0.12))
        elif agreement_count == 1:
            required_score = max(0.06, min(required_score, 0.10))
        relax_mult = 1.0
        if target_action in (1, 2) and self.no_entry_streak >= self.streak_relax_start:
            relax_steps = self.no_entry_streak - self.streak_relax_start + 1
            relax_mult = max(self.streak_relax_floor, 1.0 - self.streak_relax_step * float(relax_steps))
            required_score = max(0.06, required_score * relax_mult)

        hold_reasons: list[str] = []
        direction_conf = float(abs(base_info.get("direction_score", 0.0)))
        direction_override = bool(
            target_action in (1, 2)
            and direction_conf >= max(0.12, required_score * 1.4)
            and std_gate_ok
            and (not opp_veto)
            and selected_action in (1, 2)
        )
        can_enter = bool(
            target_action in (1, 2)
            and std_gate_ok
            and (not opp_veto)
            and (agreement >= required_score or direction_override)
            and agreement_count >= self.ens_min_votes
        )
        starvation_override = False
        if (
            not can_enter
            and target_action in (1, 2)
            and std_gate_ok
            and (not opp_veto)
            and self.no_entry_streak >= (self.streak_relax_start + 6)
            and agreement_count >= 1
            and max(long_support, short_support) >= 0.05
            and abs(net_score) >= 0.003
        ):
            starvation_override = True
            can_enter = True
        if regime_name in ("chop", "whipsaw"):
            if long_std > 1.2 and short_std > 1.2:
                can_enter = False
                hold_reasons.append("chop_both_uncertain")
        # Primary가 HOLD일 때: specialist 2개 합의만으로 부족, 추가 조건 요구
        if can_enter and primary_action == 0:
            if agreement_count < 2 and selected_std > 1.25:
                can_enter = False
                hold_reasons.append("primary_hold_guard")
        conviction_override = False
        if (
            not can_enter
            and target_action in (1, 2)
            and std_gate_ok
            and (not opp_veto)
            and selected_action in (1, 2)
            and conviction >= max(0.16, enter_th * 1.15)
            and abs(float(base_info.get("direction_score", 0.0))) >= 0.05
        ):
            conviction_override = True
            can_enter = True
        if can_enter:
            base_kelly = float(np.clip(
                self.ens_primary_mix * primary_lev + (1.0 - self.ens_primary_mix) * selected_lev,
                0.0, 1.0
            ))
            adj_kelly = float(np.clip(base_kelly * agreement_mult, 0.0, 1.0))
            if starvation_override:
                adj_kelly = float(np.clip(adj_kelly, 0.04, 0.12))
            if conviction_override:
                adj_kelly = float(np.clip(adj_kelly, 0.08, 0.20))
            if direction_override:
                adj_kelly = float(np.clip(adj_kelly, 0.10, 0.25))
            if regime_name in ("chop", "whipsaw"):
                adj_kelly = float(np.clip(adj_kelly * 0.50, 0.0, 1.0))
            out = dict(base_info)
            out.update(selected_info)
            out.update({
                "agent": "DSAC_PRIMARY_SPECIALIST",
                "long_edge": long_raw,
                "short_edge": short_raw,
                "_selected_side": selected_side,
                "agreement": agreement,
                "confidence": confidence,
                "selected_std": selected_std,
                "conviction": conviction,
                "target_action": int(target_action),
                "required_score": float(required_score),
                "long_score": float(long_score),
                "short_score": float(short_score),
                "net_score": float(net_score),
                "agreement_count": int(agreement_count),
                "agreement_mult": float(agreement_mult),
                "entry_relax_mult": float(relax_mult),
                "no_entry_streak": int(self.no_entry_streak),
                "chop_size_mult": float(0.50 if regime_name in ("chop", "whipsaw") else 1.0),
                "opp_confidence": opp_confidence,
                "opp_support": opp_support,
                "opp_veto": int(opp_veto),
                "starvation_override": int(starvation_override),
                "conviction_override": int(conviction_override),
                "direction_override": int(direction_override),
                "kelly": adj_kelly,
                "score": conviction,
                "primary_action": int(primary_action),
            })
            return int(target_action), adj_kelly, out

        dual_high_hold = False
        if target_action == 0:
            hold_reasons.append("target_hold")
        if primary_action == 0:
            hold_reasons.append("primary_hold")
        if not std_gate_ok:
            hold_reasons.append("std_gate")
        if opp_veto:
            hold_reasons.append("opp_conf_veto")
        if agreement < required_score:
            hold_reasons.append("enter_th")
        if agreement_count < self.ens_min_votes:
            hold_reasons.append("low_votes")

        out = dict(base_info)
        out.update({
            "agent": "DSAC_PRIMARY_HOLD",
            "score": conviction,
            "agreement": agreement,
            "confidence": confidence,
            "selected_std": selected_std,
            "target_action": int(target_action),
            "required_score": float(required_score),
            "long_score": float(long_score),
            "short_score": float(short_score),
            "net_score": float(net_score),
            "agreement_count": int(agreement_count),
            "agreement_mult": float(agreement_mult),
            "opp_confidence": opp_confidence,
            "opp_support": opp_support,
            "opp_veto": int(opp_veto),
            "enter_threshold": float(enter_th),
            "min_agreement_threshold": float(effective_agreement_th),
            "base_min_agreement_threshold": float(self.min_agreement_th),
            "adaptive_enter_offset": float(self.adaptive_enter_offset),
            "adaptive_agreement_offset": float(self.adaptive_agreement_offset),
            "entry_relax_mult": float(relax_mult),
            "no_entry_streak": int(self.no_entry_streak),
            "std_gate_ok": std_gate_ok,
            "dual_high_hold": dual_high_hold,
            "max_confidence_std": float(self.max_confidence_std),
            "_selected_side": selected_side,
            "hold_reason": ",".join(hold_reasons) if hold_reasons else "router_hold",
        })
        return 0, 0.0, out

    def _position_rule_from_logits(
        self,
        side: str,
        own_lev: float,
        own_logit: float,
        own_std: float,
        opp_logit: float,
        opp_std: float,
        primary_raw: float,
        base_info: dict[str, float | str | int],
        side_info: dict,
        unrealized_pnl: float = 0.0,
    ) -> tuple[int, float, dict[str, float | str | int]]:
        opp_logit_eff = float(opp_logit)
        if side == "LONG" and primary_raw <= -self.pos_primary_reverse_th:
            opp_logit_eff += float(abs(primary_raw) * self.pos_primary_reverse_weight)
        elif side == "SHORT" and primary_raw >= self.pos_primary_reverse_th:
            opp_logit_eff += float(abs(primary_raw) * self.pos_primary_reverse_weight)
        pos_action, pos_kelly, pos_diag = self._position_rule(
            side=side,
            own_lev=own_lev,
            own_logit=own_logit,
            own_std=own_std,
            opp_logit=opp_logit_eff,
            opp_std=opp_std,
            unrealized_pnl=unrealized_pnl,
        )
        # When primary and specialist strongly agree on held direction, allow controlled size boost.
        if int(pos_action) in (1, 2):
            _same_dir = (side == "LONG" and primary_raw > 0.0) or (side == "SHORT" and primary_raw < 0.0)
            if _same_dir and abs(float(primary_raw)) >= self.pos_primary_same_dir_boost_th:
                pos_kelly = float(np.clip(
                    float(pos_kelly) * self.pos_primary_same_dir_boost_mult,
                    0.0,
                    1.0,
                ))
                pos_diag = dict(pos_diag)
                pos_diag["primary_same_dir_boost"] = float(self.pos_primary_same_dir_boost_mult)
        score = float(max(self._support(own_logit, own_std), float(base_info["conviction"])))
        out = dict(base_info)
        out.update(dict(side_info or {}))
        out.update({
            "agent": f"DSAC_DUAL_{side}",
            "_selected_side": side,
            "primary_raw": float(primary_raw),
            "opp_logit_eff": float(opp_logit_eff),
            "kelly": float(pos_kelly),
            "score": score,
        })
        out.update(pos_diag)
        return int(pos_action), float(pos_kelly), out

    def _position_rule(
        self,
        side: str,
        own_lev: float,
        own_logit: float,
        own_std: float,
        opp_logit: float,
        opp_std: float,
        unrealized_pnl: float = 0.0,
    ) -> tuple[int, float, dict[str, float | str]]:
        own_support_raw = self._support(own_logit, own_std)
        opp_pressure_raw = self._support(opp_logit, opp_std)
        alpha = float(np.clip(self.pos_support_ema_alpha, 0.05, 0.95))
        prev_own = self._pos_support_ema.get(side)
        prev_opp = self._pos_opp_support_ema.get(side)
        own_support = float(own_support_raw if prev_own is None else (alpha * own_support_raw + (1.0 - alpha) * prev_own))
        opp_pressure = float(opp_pressure_raw if prev_opp is None else (alpha * opp_pressure_raw + (1.0 - alpha) * prev_opp))
        self._pos_support_ema[side] = own_support
        self._pos_opp_support_ema[side] = opp_pressure
        net_edge = float(own_support - opp_pressure)
        ambiguity = float(min(own_logit, opp_logit))
        reduce_flag = bool(
            ambiguity >= self.pos_ambiguity_high
            and abs(own_logit - opp_logit) < self.pos_ambiguity_gap
        )
        if net_edge <= self.pos_exit_net_edge or opp_pressure >= self.pos_opp_pressure_exit:
            return 0, 0.0, {
                "position_signal": "EXIT",
                "position_reason": "OPP_PRESSURE",
                "own_support": own_support,
                "opp_pressure": opp_pressure,
                "net_edge": net_edge,
                "own_support_raw": own_support_raw,
                "opp_pressure_raw": opp_pressure_raw,
            }
        hold_kelly = float(np.clip(
            float(own_lev) * np.clip(0.55 + 0.45 * np.tanh(net_edge / max(self.pos_kelly_scale, 1e-6)), 0.20, 1.00),
            0.0, 1.0,
        ))
        if own_support <= self.pos_reduce_support or reduce_flag:
            return (1 if side == "LONG" else 2), float(np.clip(hold_kelly * self.pos_reduce_mult, 0.0, 1.0)), {
                "position_signal": "REDUCE",
                "position_reason": "AMBIGUITY" if reduce_flag else "WEAK_SUPPORT",
                "own_support": own_support,
                "opp_pressure": opp_pressure,
                "net_edge": net_edge,
                "own_support_raw": own_support_raw,
                "opp_pressure_raw": opp_pressure_raw,
            }
        if own_support >= self.pos_hold_support and net_edge >= self.pos_hold_net_edge:
            hold_boost = 1.0
            if float(unrealized_pnl) >= 0.0:
                if net_edge >= self.pos_hold_boost_edge_hi:
                    hold_boost = self.pos_hold_boost_mult_hi
                elif net_edge >= self.pos_hold_boost_edge_lo:
                    hold_boost = self.pos_hold_boost_mult_lo
            boosted_kelly = float(np.clip(hold_kelly * hold_boost, 0.0, 1.0))
            return (1 if side == "LONG" else 2), boosted_kelly, {
                "position_signal": "HOLD",
                "position_reason": "SUPPORTED",
                "own_support": own_support,
                "opp_pressure": opp_pressure,
                "net_edge": net_edge,
                "hold_boost": float(hold_boost),
                "own_support_raw": own_support_raw,
                "opp_pressure_raw": opp_pressure_raw,
            }
        return (1 if side == "LONG" else 2), float(np.clip(hold_kelly * 0.85, 0.0, 1.0)), {
            "position_signal": "HOLD",
            "position_reason": "NEUTRAL",
            "own_support": own_support,
            "opp_pressure": opp_pressure,
            "net_edge": net_edge,
            "own_support_raw": own_support_raw,
            "opp_pressure_raw": opp_pressure_raw,
        }

    def __init__(
        self,
        model_path: str | None = None,
        long_path: str | None = None,
        short_path: str | None = None,
        single_path: str | None = None,
        hmm_detector: OnlineHMMDetector | None = None,
    ):
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
        self.base_enter_th = float(os.getenv("DSAC_DUAL_BASE_ENTER_TH", "0.08"))
        self.trend_align_enter_th = float(os.getenv("DSAC_DUAL_TREND_ALIGN_ENTER_TH", "0.07"))
        self.chop_enter_th = float(os.getenv("DSAC_DUAL_CHOP_ENTER_TH", "0.15"))
        self.whipsaw_enter_th = float(os.getenv("DSAC_DUAL_WHIPSAW_ENTER_TH", "0.15"))
        self.min_agreement_th = float(os.getenv("DSAC_DUAL_MIN_AGREEMENT_TH", "0.08"))
        self.ambiguity_penalty_start = float(os.getenv("DSAC_DUAL_AMBIG_PENALTY_START", "2.50"))
        self.ambiguity_penalty_add = float(os.getenv("DSAC_DUAL_AMBIG_PENALTY_ADD", "0.08"))
        self.max_confidence_std = float(os.getenv("DSAC_DUAL_MAX_CONFIDENCE_STD", "2.00"))
        self.dual_high_logit_hold = float(os.getenv("DSAC_DUAL_HIGH_LOGIT_HOLD", "2.80"))
        self.dual_high_logit_gap = float(os.getenv("DSAC_DUAL_HIGH_LOGIT_GAP", "0.85"))
        self.opp_veto_support_th = float(os.getenv("DSAC_DUAL_OPP_VETO_SUPPORT_TH", "0.30"))
        self.opp_veto_std_max = float(os.getenv("DSAC_DUAL_OPP_VETO_STD_MAX", "1.20"))
        self.ens_vote_min_strength = float(os.getenv("DSAC_DUAL_ENS_VOTE_MIN_STRENGTH", "0.03"))
        self.ens_min_votes = int(os.getenv("DSAC_DUAL_ENS_MIN_VOTES", "1"))
        self.ens_agree_mult_3 = float(os.getenv("DSAC_DUAL_ENS_AGREE_MULT_3", "1.00"))
        self.ens_agree_mult_2 = float(os.getenv("DSAC_DUAL_ENS_AGREE_MULT_2", "0.80"))
        self.ens_agree_mult_1 = float(os.getenv("DSAC_DUAL_ENS_AGREE_MULT_1", "0.45"))
        self.ens_primary_mix = float(os.getenv("DSAC_DUAL_ENS_PRIMARY_MIX", "0.40"))
        self.pos_primary_reverse_th = float(os.getenv("DSAC_DUAL_POS_PRIMARY_REVERSE_TH", "0.10"))
        self.pos_primary_reverse_weight = float(os.getenv("DSAC_DUAL_POS_PRIMARY_REVERSE_WEIGHT", "0.30"))
        self.adaptive_enter_offset = 0.0
        self.adaptive_agreement_offset = 0.0
        self.no_entry_streak = 0
        self.streak_relax_start = int(os.getenv("DSAC_DUAL_RELAX_START_BARS", "6"))
        self.streak_relax_step = float(os.getenv("DSAC_DUAL_RELAX_STEP", "0.08"))
        self.streak_relax_floor = float(os.getenv("DSAC_DUAL_RELAX_FLOOR", "0.45"))
        self.pos_hold_support = float(os.getenv("DSAC_DUAL_POS_HOLD_SUPPORT", "1.10"))
        self.pos_hold_net_edge = float(os.getenv("DSAC_DUAL_POS_HOLD_NET_EDGE", "0.05"))
        self.pos_reduce_support = float(os.getenv("DSAC_DUAL_POS_REDUCE_SUPPORT", "0.55"))
        self.pos_reduce_mult = float(os.getenv("DSAC_DUAL_POS_REDUCE_MULT", "0.35"))
        self.pos_exit_net_edge = float(os.getenv("DSAC_DUAL_POS_EXIT_NET_EDGE", "-0.25"))
        self.pos_opp_pressure_exit = float(os.getenv("DSAC_DUAL_POS_OPP_PRESSURE_EXIT", "1.15"))
        self.pos_ambiguity_high = float(os.getenv("DSAC_DUAL_POS_AMBIG_HIGH", "1.90"))
        self.pos_ambiguity_gap = float(os.getenv("DSAC_DUAL_POS_AMBIG_GAP", "1.00"))
        self.pos_kelly_scale = float(os.getenv("DSAC_DUAL_POS_KELLY_SCALE", "0.80"))
        self.pos_hold_boost_edge_lo = float(os.getenv("DSAC_DUAL_POS_HOLD_BOOST_EDGE_LO", "0.15"))
        self.pos_hold_boost_edge_hi = float(os.getenv("DSAC_DUAL_POS_HOLD_BOOST_EDGE_HI", "0.30"))
        self.pos_hold_boost_mult_lo = float(os.getenv("DSAC_DUAL_POS_HOLD_BOOST_MULT_LO", "1.10"))
        self.pos_hold_boost_mult_hi = float(os.getenv("DSAC_DUAL_POS_HOLD_BOOST_MULT_HI", "1.20"))
        self.pos_primary_same_dir_boost_th = float(os.getenv("DSAC_DUAL_POS_PRIMARY_SAME_DIR_BOOST_TH", "0.30"))
        self.pos_primary_same_dir_boost_mult = float(os.getenv("DSAC_DUAL_POS_PRIMARY_SAME_DIR_BOOST_MULT", "1.15"))
        self.pos_support_ema_alpha = float(os.getenv("DSAC_DUAL_POS_SUPPORT_EMA_ALPHA", "0.65"))
        self._pos_support_ema: dict[str, float] = {}
        self._pos_opp_support_ema: dict[str, float] = {}
        self.elite_extractor = EliteSignals()
        self.new_elite_engine = NewEliteSignalEngine()

        if model_path and not long_path and not short_path:
            long_path = model_path
            short_path = model_path
        self.single_ckpt_path = self._resolve_model_path(single_path, self.DEFAULT_SINGLE_PATH, self.LEGACY_SINGLE_PATH)
        self.long_ckpt_path = self._resolve_model_path(long_path, self.DEFAULT_LONG_PATH, self.LEGACY_LONG_PATH)
        self.short_ckpt_path = self._resolve_model_path(short_path, self.DEFAULT_SHORT_PATH, self.LEGACY_SHORT_PATH)
        self._load_specialists(self.device)

    def _load_specialists(self, device: str) -> None:
        single_ckpt = torch.load(self.single_ckpt_path, map_location=device, weights_only=False)
        long_ckpt = torch.load(self.long_ckpt_path, map_location=device, weights_only=False)
        short_ckpt = torch.load(self.short_ckpt_path, map_location=device, weights_only=False)
        single_actor, single_ver = self._build_primary_actor_from_ckpt(single_ckpt, device)
        long_actor, long_ver = self._build_actor_from_ckpt(long_ckpt, device, "long")
        short_actor, short_ver = self._build_actor_from_ckpt(short_ckpt, device, "short")
        self.device = device
        self.primary_router = BaseDSACRouter(single_actor, device=device, hmm_detector=self.hmm)
        self.long_router = DSACLongRouter(long_actor, device=device)
        self.short_router = DSACShortRouter(short_actor, device=device)
        logger.info("✅ DSAC primary 로드 완료 (%s, %s): %s", single_ver, device, self.single_ckpt_path)
        logger.info("✅ DSACSignalRouter 로드 완료 (%s, %s): %s", long_ver, device, self.long_ckpt_path)
        logger.info("✅ DSACSignalRouter 로드 완료 (%s, %s): %s", short_ver, device, self.short_ckpt_path)

    @staticmethod
    def _require_finite(mapping, key: str, context: str) -> float:
        if key not in mapping:
            raise ValueError(f"[FEATURE_MISSING] {context}.{key} missing")
        try:
            val = float(mapping[key])
        except Exception as e:
            raise ValueError(f"[FEATURE_INVALID] {context}.{key} cast failed: {e}") from e
        if not np.isfinite(val):
            raise ValueError(f"[FEATURE_INVALID] {context}.{key} is not finite: {mapping[key]}")
        return val

    def decide(self, processed_df: pd.DataFrame, nf_preds: dict, m7_signal: dict | None = None):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        if "smart_money_flow" not in processed_df.columns:
            raise KeyError("processed_df missing required column: smart_money_flow")
        smf_std = processed_df["smart_money_flow"].std()

        cur_market = row_to_market_row(last_row)
        prev_market = row_to_market_row(prev_row)
        elite_sigs = self.elite_extractor.compute_all(current=cur_market, prev=prev_market, smf_std=smf_std)
        tail_df = processed_df.tail(100).copy()
        self.new_elite_engine.compute(tail_df)
        tail_last = tail_df.iloc[-1]
        for col in ["sig_volume_confirm", "sig_liquidity_trap", "sig_trend_health"]:
            elite_sigs[col] = self._require_finite(tail_last, col, "tail_last")

        features: dict[str, float] = {}
        for col in DSAC_STATE_PRED:
            features[col] = self._require_finite(nf_preds, col, "nf_preds")
        for col in DSAC_STATE_CONF:
            features[col] = self._require_finite(nf_preds, col, "nf_preds")
        for col in DSAC_STATE_ELITE:
            features[col] = self._require_finite(elite_sigs, col, "elite_sigs")
        for col in DSAC_STATE_ALPHA:
            features[col] = self._require_finite(last_row, col, "last_row")

        regime = None
        if self.hmm is not None:
            hmm_row = {
                "log_return": self._require_finite(last_row, "log_return", "last_row"),
                "garch_vol_z": self._require_finite(last_row, "garch_vol_z", "last_row"),
                "oi_change_rate": self._require_finite(last_row, "oi_change_rate", "last_row"),
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
        if regime is None:
            regime = _compute_regime(processed_df)
        features.update(regime)
        for col in DSAC_STATE_SYNTH:
            features[col] = self._require_finite(last_row, col, "last_row")
        features["close"] = self._require_finite(last_row, "close", "last_row")
        # 변동성 모델 피처 — specialist _build_compact_state에서 직접 조회
        for col in ("jump_z", "evt_excess_z", "garch_vol_z", "jump_flag", "evt_tail_flag", "log_return"):
            features[col] = self._require_finite(last_row, col, "last_row")
        # HL spread proxy — 실제 OHLC 기반 bid-ask spread 근사값 (spread 컬럼 미존재 시)
        _h = self._require_finite(last_row, "high", "last_row")
        _l = self._require_finite(last_row, "low", "last_row")
        _c = features["close"]
        features["current_spread"] = float(np.clip((_h - _l) / max(_c, 1e-8), 0.0, 0.05))

        # 학습 시 사용한 m7_*를 라이브 추론 입력에도 주입해 train/infer 스키마 불일치 최소화
        if not isinstance(m7_signal, dict):
            raise ValueError("[FEATURE_MISSING] m7_signal unavailable")
        if "m7_prob_dn" in m7_signal:
            features["m7_prob_dn"] = self._require_finite(m7_signal, "m7_prob_dn", "m7_signal")
            features["m7_prob_fl"] = self._require_finite(m7_signal, "m7_prob_fl", "m7_signal")
            features["m7_prob_up"] = self._require_finite(m7_signal, "m7_prob_up", "m7_signal")
        else:
            features["m7_prob_dn"] = self._require_finite(m7_signal, "prob_dn", "m7_signal")
            features["m7_prob_fl"] = self._require_finite(m7_signal, "prob_flat", "m7_signal")
            features["m7_prob_up"] = self._require_finite(m7_signal, "prob_up", "m7_signal")
        for k in M7_LIVE_STRICT_COLS:
            features[k] = self._require_finite(m7_signal, k, "m7_signal")
        # specialist _build_compact_state에서 직접 조회하는 m7 오프셋 컬럼
        for k in ("m7_tp_offset", "m7_sl_offset"):
            features[k] = self._require_finite(m7_signal, k, "m7_signal")

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

        try:
            primary_action, primary_lev, primary_info = self.primary_router.decide(features, pos_dict)
            long_action, long_lev, long_info = self.long_router.decide(features, pos_dict)
            short_action, short_lev, short_info = self.short_router.decide(features, pos_dict)
        except Exception as e:
            if self.device == "cuda" and self._is_cuda_runtime_error(e):
                logger.warning("⚠️ DSAC dual CUDA 추론 실패, CPU로 폴백합니다: %s", e)
                self._load_specialists("cpu")
                primary_action, primary_lev, primary_info = self.primary_router.decide(features, pos_dict)
                long_action, long_lev, long_info = self.long_router.decide(features, pos_dict)
                short_action, short_lev, short_info = self.short_router.decide(features, pos_dict)
            else:
                raise

        primary_raw = float((primary_info or {}).get("raw_action", 0.0))
        long_raw = float((long_info or {}).get("raw_action", 0.0))
        short_raw = float((short_info or {}).get("raw_action", 0.0))
        long_logit = float((long_info or {}).get("logit", 0.0))
        short_logit = float((short_info or {}).get("logit", 0.0))
        long_std = float(max((long_info or {}).get("std", 1.0), 1e-6))
        short_std = float(max((short_info or {}).get("std", 1.0), 1e-6))
        info = self._base_signal_info(
            primary_action=int(primary_action),
            primary_lev=float(primary_lev),
            primary_raw=primary_raw,
            long_action=int(long_action),
            long_lev=float(long_lev),
            long_raw=long_raw,
            long_logit=long_logit,
            long_std=long_std,
            short_action=int(short_action),
            short_lev=float(short_lev),
            short_raw=short_raw,
            short_logit=short_logit,
            short_std=short_std,
        )
        direction_score = float(info["direction_score"])
        conviction = float(info["conviction"])
        if self.pos not in ("LONG", "SHORT"):
            self._pos_support_ema.clear()
            self._pos_opp_support_ema.clear()

        if self.pos == "LONG":
            return (
                *self._position_rule_from_logits(
                side="LONG",
                own_lev=float(long_lev),
                own_logit=long_logit,
                own_std=long_std,
                opp_logit=short_logit,
                opp_std=short_std,
                unrealized_pnl=float(unr),
                primary_raw=primary_raw,
                base_info=info,
                side_info=long_info,
                ),
                elite_sigs,
                regime,
            )

        if self.pos == "SHORT":
            return (
                *self._position_rule_from_logits(
                side="SHORT",
                own_lev=float(short_lev),
                own_logit=short_logit,
                own_std=short_std,
                opp_logit=long_logit,
                opp_std=long_std,
                unrealized_pnl=float(unr),
                primary_raw=primary_raw,
                base_info=info,
                side_info=short_info,
                ),
                elite_sigs,
                regime,
            )

        entry_action, entry_kelly, entry_info = self._entry_rule(
            regime=regime,
            trend_signal=m7_signal,
            long_action=int(long_action),
            long_lev=float(long_lev),
            long_info=dict(long_info or {}),
            short_action=int(short_action),
            short_lev=float(short_lev),
            short_info=dict(short_info or {}),
            base_info=info,
        )
        if entry_action in (1, 2):
            self.no_entry_streak = 0
        else:
            self.no_entry_streak = min(self.no_entry_streak + 1, 10000)
            entry_info = dict(entry_info)
            entry_info["no_entry_streak"] = int(self.no_entry_streak)
        return int(entry_action), float(entry_kelly), entry_info, elite_sigs, regime


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
        self.position_exit_streak: int = 0
        self.last_summary_ts: datetime | None = None
        self._last_state_save_ts: datetime | None = None
        self.adaptive_enter_offset: float = 0.0
        self.adaptive_agreement_offset: float = 0.0

        # 포지션/청산 관리
        self.min_live_kelly = float(os.getenv("FUSE_MIN_LIVE_KELLY", "0.04"))
        self.dsac_only_hard_stop = float(os.getenv("DSAC_ONLY_HARD_STOP", "0.025"))
        self.dsac_only_max_hold = int(os.getenv("DSAC_ONLY_MAX_HOLD", "36"))
        self.dsac_only_reverse_min = float(os.getenv("DSAC_ONLY_REVERSE_MIN", "0.45"))
        self.dsac_only_trail_arm = float(os.getenv("DSAC_ONLY_TRAIL_ARM", "0.012"))
        self.dsac_only_trail_gap = float(os.getenv("DSAC_ONLY_TRAIL_GAP", "0.008"))
        self.dsac_only_vol_scale_enable = _env_flag("DSAC_ONLY_VOL_SCALE_ENABLE", True)
        self.dsac_only_cooldown_enable = _env_flag("DSAC_ONLY_COOLDOWN_ENABLE", False)
        self.dsac_only_cooldown_loss = float(os.getenv("DSAC_ONLY_COOLDOWN_LOSS", "0.05"))
        self.dsac_only_cooldown_streak = int(os.getenv("DSAC_ONLY_COOLDOWN_STREAK", "4"))
        self.dsac_only_cooldown_bars = int(os.getenv("DSAC_ONLY_COOLDOWN_BARS", "0"))
        self.dsac_only_chop_entry_kelly_mult = float(os.getenv("DSAC_ONLY_CHOP_ENTRY_KELLY_MULT", "0.50"))
        self.dsac_only_whipsaw_entry_kelly_mult = float(os.getenv("DSAC_ONLY_WHIPSAW_ENTRY_KELLY_MULT", "0.55"))
        self.dsac_only_trend_exit_enable = _env_flag("DSAC_ONLY_TREND_EXIT_ENABLE", True)
        self.dsac_only_trend_exit_hold_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_HOLD_BARS", "24"))
        self.dsac_only_trend_exit_confirm_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_CONFIRM_BARS", "2"))
        self.position_exit_confirm_bars = int(os.getenv("DSAC_POSITION_EXIT_CONFIRM_BARS", "2"))
        self.dsac_only_trend_exit_score = float(os.getenv("DSAC_ONLY_TREND_EXIT_SCORE", "0.20"))
        self.dsac_only_trend_exit_quality = float(os.getenv("DSAC_ONLY_TREND_EXIT_QUALITY", "0.000"))
        self.dsac_only_vae_block_ratio = float(os.getenv("DSAC_ONLY_VAE_BLOCK_RATIO", "1.35"))

        # 신규 진입 차단
        self.integral_enable = _env_flag("DSAC_SIGNAL_INTEGRAL_ENABLE", False)
        self.integral_window = int(os.getenv("DSAC_SIGNAL_INTEGRAL_WINDOW", "2"))
        self.integral_min_gap = float(os.getenv("DSAC_SIGNAL_INTEGRAL_MIN_GAP", "0.65"))
        self.integral_require_sign = _env_flag("DSAC_SIGNAL_INTEGRAL_REQUIRE_SIGN", True)
        self.hibernation_enable = _env_flag("DSAC_HIBERNATION_ENABLE", True)
        self.hibernation_score_th = float(os.getenv("DSAC_HIBERNATION_SCORE_TH", "0.85"))
        self.hibernation_bars = int(os.getenv("DSAC_HIBERNATION_BARS", "6"))
        self.hibernation_kelly_mult = float(os.getenv("DSAC_HIBERNATION_KELLY_MULT", "0.35"))
        self.illiquidity_veto_enable = _env_flag("DSAC_ILLIQUIDITY_VETO_ENABLE", False)
        self.illiquidity_amihud_th = float(os.getenv("DSAC_ILLIQUIDITY_AMIHUD_TH", "1.40"))
        self.illiquidity_rsvol_th = float(os.getenv("DSAC_ILLIQUIDITY_RSVOL_TH", "0.015"))
        self.illiquidity_vol_rank_th = float(os.getenv("DSAC_ILLIQUIDITY_VOLRANK_TH", "0.78"))
        self.illiquidity_scale_mult = float(os.getenv("DSAC_ILLIQUIDITY_SCALE_MULT", "0.50"))
        self.trend_align_kelly_mult = float(os.getenv("DSAC_TREND_ALIGN_KELLY_MULT", "1.20"))
        self.trend_flat_kelly_mult = float(os.getenv("DSAC_TREND_FLAT_KELLY_MULT", "0.70"))
        self.trend_mismatch_kelly_mult = float(os.getenv("DSAC_TREND_MISMATCH_KELLY_MULT", "0.45"))
        self.min_entry_kelly = float(os.getenv("DSAC_MIN_ENTRY_KELLY", "0.06"))
        self.churn_cooldown_bars = int(os.getenv("DSAC_CHURN_COOLDOWN_BARS", "1"))
        self.churn_hold_bars = int(os.getenv("DSAC_CHURN_HOLD_BARS", "2"))
        self.churn_pnl_abs = float(os.getenv("DSAC_CHURN_PNL_ABS", "0.0015"))

        # 신규 진입 사이징
        self.sizing_bayes_enable = _env_flag("DSAC_SIZING_BAYES_ENABLE", True)
        self.sizing_bayes_z_scale = float(os.getenv("DSAC_SIZING_BAYES_Z_SCALE", "1.5"))
        self.sizing_bayes_min_mult = float(os.getenv("DSAC_SIZING_BAYES_MIN_MULT", "0.35"))
        self.sizing_bayes_max_mult = float(os.getenv("DSAC_SIZING_BAYES_MAX_MULT", "1.80"))
        self.sizing_qwidth_enable = _env_flag("DSAC_SIZING_QWIDTH_ENABLE", True)
        self.sizing_qwidth_ref = float(os.getenv("DSAC_SIZING_QWIDTH_REF", "0.008"))
        self.sizing_qwidth_min_mult = float(os.getenv("DSAC_SIZING_QWIDTH_MIN_MULT", "0.50"))
        self.sizing_qwidth_max_mult = float(os.getenv("DSAC_SIZING_QWIDTH_MAX_MULT", "1.20"))
        self.sizing_mtf_enable = _env_flag("DSAC_SIZING_MTF_ENABLE", True)
        self.sizing_mtf_partial_mult = float(os.getenv("DSAC_SIZING_MTF_PARTIAL_MULT", "1.15"))
        self.sizing_mtf_full_mult = float(os.getenv("DSAC_SIZING_MTF_FULL_MULT", "1.35"))
        self.sizing_smart_enable = _env_flag("DSAC_SIZING_SMART_ENABLE", True)
        self.sizing_smart_same_sign_mult = float(os.getenv("DSAC_SIZING_SMART_SAME_SIGN_MULT", "1.20"))
        self.sizing_smart_opp_sign_mult = float(os.getenv("DSAC_SIZING_SMART_OPP_SIGN_MULT", "0.60"))
        self.sizing_taker_boost_th = float(os.getenv("DSAC_SIZING_TAKER_BOOST_TH", "0.20"))
        self.sizing_taker_boost_mult = float(os.getenv("DSAC_SIZING_TAKER_BOOST_MULT", "1.10"))
        self.sizing_mdd_enable = _env_flag("DSAC_SIZING_MDD_ENABLE", True)
        self.sizing_mdd_loss_streak_th = int(os.getenv("DSAC_SIZING_MDD_LOSS_STREAK_TH", "3"))
        self.sizing_mdd_reduce_mult = float(os.getenv("DSAC_SIZING_MDD_REDUCE_MULT", "0.35"))
        self.sizing_recent_pnl_window = int(os.getenv("DSAC_SIZING_RECENT_PNL_WINDOW", "10"))
        self.sizing_recent_pnl_cut = float(os.getenv("DSAC_SIZING_RECENT_PNL_CUT", "-0.01"))

        # 진입가 추천
        self.entry_reco_enable = _env_flag("DSAC_ENTRY_RECO_ENABLE", True)
        self.entry_reco_min_strength = float(os.getenv("DSAC_ENTRY_RECO_MIN_STRENGTH", "0.55"))
        self.entry_reco_min_quality = float(os.getenv("DSAC_ENTRY_RECO_MIN_QUALITY", "-0.002"))
        self.entry_reco_max_offset = float(os.getenv("DSAC_ENTRY_RECO_MAX_OFFSET", "0.0045"))
        self.entry_reco_price_buffer = float(os.getenv("DSAC_ENTRY_RECO_PRICE_BUFFER", "0.0002"))
        self.trade_fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
        self.trade_slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
        self.live_state_path = os.getenv("DSAC_LIVE_STATE_PATH", "data/ensemble/dsac_live_state.json")
        self.adaptive_gate_enable = _env_flag("DSAC_ADAPTIVE_GATE_ENABLE", True)
        self.adaptive_gate_pnl_window = int(os.getenv("DSAC_ADAPTIVE_GATE_PNL_WINDOW", "8"))
        self.adaptive_gate_enter_step = float(os.getenv("DSAC_ADAPTIVE_GATE_ENTER_STEP", "0.01"))
        self.adaptive_gate_agreement_step = float(os.getenv("DSAC_ADAPTIVE_GATE_AGREEMENT_STEP", "0.01"))
        self.adaptive_gate_loosen_step = float(os.getenv("DSAC_ADAPTIVE_GATE_LOOSEN_STEP", "0.02"))
        self.adaptive_gate_enter_min = float(os.getenv("DSAC_ADAPTIVE_GATE_ENTER_MIN", "-0.18"))
        self.adaptive_gate_enter_max = float(os.getenv("DSAC_ADAPTIVE_GATE_ENTER_MAX", "0.08"))
        self.adaptive_gate_agreement_min = float(os.getenv("DSAC_ADAPTIVE_GATE_AGREEMENT_MIN", "-0.14"))
        self.adaptive_gate_agreement_max = float(os.getenv("DSAC_ADAPTIVE_GATE_AGREEMENT_MAX", "0.08"))
        self.adaptive_gate_flat_bars = int(os.getenv("DSAC_ADAPTIVE_GATE_FLAT_BARS", "10"))
        self.adaptive_gate_loss_streak_th = int(os.getenv("DSAC_ADAPTIVE_GATE_LOSS_STREAK_TH", "4"))
        self.adaptive_gate_bad_pnl_cut = float(os.getenv("DSAC_ADAPTIVE_GATE_BAD_PNL_CUT", "-0.015"))
        self.adaptive_gate_good_pnl_cut = float(os.getenv("DSAC_ADAPTIVE_GATE_GOOD_PNL_CUT", "0.006"))
        self.adaptive_flat_cycles: int = 0

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
        self.btc_corr_misalign    = float(os.getenv("FUSE_BTC_CORR_MISALIGN",    "0.80"))  # 반대방향 kelly 배수
        self.btc_corr_align_boost = float(os.getenv("FUSE_BTC_CORR_ALIGN_BOOST", "1.12"))  # 같은방향 kelly 부스트
        self.btc_corr_move_th     = float(os.getenv("FUSE_BTC_CORR_MOVE_TH",     "0.004")) # BTC 3봉 유의미 변화 임계값

        # ── 안티 찹(횡보) 필터 ────────────────────────────────────────────────
        # 최근 N봉에서 방향 전환이 잦으면 지그재그 장세로 판단해 kelly 축소
        self.chop_filter_enable   = _env_flag("FUSE_CHOP_FILTER",         True)
        self.chop_window          = int(os.getenv("FUSE_CHOP_WINDOW",         "12"))   # 관찰 봉수
        self.chop_turns_max       = int(os.getenv("FUSE_CHOP_TURNS_MAX",      "7"))   # 이 횟수 이상 전환 시 찹 판정
        self.chop_kelly_scale     = float(os.getenv("FUSE_CHOP_KELLY_SCALE",  "0.50"))

        # ── 거래량 확인 필터 ──────────────────────────────────────────────────
        # 저거래량 구간 진입 시 kelly 축소 (스캘핑 노이즈 방지)
        self.volume_confirm_enable = _env_flag("FUSE_VOLUME_CONFIRM",      True)
        self.volume_min_ratio      = float(os.getenv("FUSE_VOLUME_MIN_RATIO",  "0.50"))  # 20봉 평균 대비 최소 비율
        self.volume_low_kelly      = float(os.getenv("FUSE_VOLUME_LOW_KELLY",  "0.75"))  # 저거래량 시 kelly 배수

        self._load_live_state()

    def record_outcome(self, realized_pnl_pct: float):
        pnl = float(realized_pnl_pct)
        self.last_realized_pnl = None
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
        if self.dsac_only_cooldown_enable:
            if (
                len(self.recent_realized) >= 5
                and sum(list(self.recent_realized)[-5:]) <= -abs(self.dsac_only_cooldown_loss)
            ) or self.loss_streak >= max(1, self.dsac_only_cooldown_streak):
                self.cooldown_bars_left = max(self.cooldown_bars_left, max(1, self.dsac_only_cooldown_bars))
        else:
            self.cooldown_bars_left = 0
        self._save_live_state()
        self._open_trade_diag = None

    def update_adaptive_gate(self, final_action: int, in_position: bool) -> tuple[float, float]:
        if not self.adaptive_gate_enable:
            self.adaptive_enter_offset = 0.0
            self.adaptive_agreement_offset = 0.0
            return 0.0, 0.0

        if in_position:
            self.adaptive_flat_cycles = 0
        elif int(final_action) == 0:
            self.adaptive_flat_cycles += 1
        else:
            self.adaptive_flat_cycles = 0

        window = max(1, int(self.adaptive_gate_pnl_window))
        recent_vals = list(self.recent_realized)[-window:]
        recent_pnl_sum = float(sum(recent_vals)) if recent_vals else 0.0

        enter_offset = 0.0
        agreement_offset = 0.0
        if self.loss_streak >= max(1, self.adaptive_gate_loss_streak_th) or recent_pnl_sum <= self.adaptive_gate_bad_pnl_cut:
            enter_offset += float(self.adaptive_gate_enter_step)
            agreement_offset += float(self.adaptive_gate_agreement_step)
        elif self.cooldown_bars_left == 0 and self.loss_streak == 0 and recent_pnl_sum >= self.adaptive_gate_good_pnl_cut:
            enter_offset -= float(self.adaptive_gate_loosen_step)
            agreement_offset -= float(self.adaptive_gate_loosen_step)

        if self.pos is None and self.adaptive_flat_cycles >= max(1, self.adaptive_gate_flat_bars):
            enter_offset -= float(self.adaptive_gate_loosen_step)
            agreement_offset -= float(self.adaptive_gate_loosen_step * 0.5)

        self.adaptive_enter_offset = float(np.clip(
            enter_offset,
            self.adaptive_gate_enter_min,
            self.adaptive_gate_enter_max,
        ))
        self.adaptive_agreement_offset = float(np.clip(
            agreement_offset,
            self.adaptive_gate_agreement_min,
            self.adaptive_gate_agreement_max,
        ))
        return self.adaptive_enter_offset, self.adaptive_agreement_offset

    def _choose_entry_price(self, final_action: int, current_price: float, trend_signal: dict | None = None) -> float:
        px = max(float(current_price), 0.0)
        if not self.entry_reco_enable or px <= 0.0 or not isinstance(trend_signal, dict):
            return px
        strength = float(trend_signal.get("strength", 0.0) or 0.0)
        quality = float(trend_signal.get("m7_quality_pred", 0.0) or 0.0)
        if strength < self.entry_reco_min_strength or quality < self.entry_reco_min_quality:
            return px
        if final_action == 1:
            reco_px = float(trend_signal.get("m7_entry_long_price", 0.0) or 0.0)
            reco_off = abs(float(trend_signal.get("m7_entry_long_offset", 0.0) or 0.0))
            if reco_px > 0.0 and reco_px <= px * (1.0 + self.entry_reco_price_buffer) and reco_off <= self.entry_reco_max_offset:
                return reco_px
        elif final_action == 2:
            reco_px = float(trend_signal.get("m7_entry_short_price", 0.0) or 0.0)
            reco_off = abs(float(trend_signal.get("m7_entry_short_offset", 0.0) or 0.0))
            if reco_px > 0.0 and reco_px >= px * (1.0 - self.entry_reco_price_buffer) and reco_off <= self.entry_reco_max_offset:
                return reco_px
        return px

    def _update_pos(
        self,
        final_action: int,
        current_price: float,
        leverage: float | None = None,
        trend_signal: dict | None = None,
    ):
        entry_px = self._choose_entry_price(final_action, current_price, trend_signal)
        # Flip support: close opposite position first, then open new side.
        if final_action == 1 and self.pos == "SHORT":
            if self.entry_price > 0 and current_price > 0:
                self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            self.pos, self.entry_price, self.hold_count = "LONG", entry_px, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
            return
        if final_action == 2 and self.pos == "LONG":
            if self.entry_price > 0 and current_price > 0:
                self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            self.pos, self.entry_price, self.hold_count = "SHORT", entry_px, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
            return
        if final_action == 1 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = "LONG", entry_px, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.last_realized_pnl = None
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
        elif final_action == 2 and self.pos is None:
            self.pos, self.entry_price, self.hold_count = "SHORT", entry_px, 0
            self.current_leverage = float(np.clip(leverage if leverage is not None else self.current_leverage, 0.0, 1.0))
            self.peak_equity = self.cur_equity = 1.0
            self.last_realized_pnl = None
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
        elif final_action == 0 and self.pos is not None:
            if self.entry_price > 0 and current_price > 0:
                self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            if (
                self.last_closed_hold_count <= self.churn_hold_bars
                and abs(self.last_realized_pnl) <= self.churn_pnl_abs
            ):
                self.cooldown_bars_left = max(self.cooldown_bars_left, self.churn_cooldown_bars)
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
            self.current_leverage = 0.0
            self.peak_equity = 1.0
            self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
        elif self.pos is not None and self.entry_price > 0 and current_price > 0:
            self.hold_count += 1
            if leverage is not None:
                self.current_leverage = float(np.clip(leverage, 0.0, 1.0))
            self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.peak_equity = max(self.peak_equity, self.cur_equity)
            self.last_realized_pnl = None
            self._save_live_state()

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
            self.position_exit_streak = int(data.get("position_exit_streak", 0))
            self.adaptive_flat_cycles = int(data.get("adaptive_flat_cycles", 0))
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
                "position_exit_streak": self.position_exit_streak,
                "adaptive_flat_cycles": self.adaptive_flat_cycles,
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

    def print_meta_dashboard(self, result: dict, current_price: float = 0.0):
        C = Colors
        fa = int(result.get("final_action", 0))
        src = str(result.get("source", "N/A"))
        fa_arrow = {0: "─", 1: "▲", 2: "▼"}.get(fa, "?")
        fa_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(fa, C.RESET)
        fa_word = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(fa, "?")

        print(f" {fa_color}{C.BOLD}{fa_arrow}{fa_arrow}  {fa_word}{C.RESET}"
              f"  score={float(result.get('rl_score', 0.0)):.3f}"
              f"  Kelly={float(result.get('unified_kelly', 0.0)):.3f}"
              f"  source: {C.CYAN}{src}{C.RESET}")
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
    llm_advisor  = LLMAdvisor()
    ensemble     = EnsemblePredictor() if ENSEMBLE_PREDICTOR_ENABLED else None
    dsac_nf_predictor = EnsemblePredictor()
    live_hmm: OnlineHMMDetector | None = None
    live_hmm_steps = 0
    logger.info("🧱 부가 기능: ensemble=%s", "ON" if ENSEMBLE_PREDICTOR_ENABLED else "OFF")
    try:
        _nf_status = {
            "PatchTST": bool(getattr(dsac_nf_predictor.models.get("PatchTST"), "available", False)),
            "Chronos": bool(getattr(dsac_nf_predictor.models.get("Chronos"), "available", False)),
            "TiDE": bool(getattr(dsac_nf_predictor.models.get("TiDE"), "available", False)),
        }
        logger.info("🧠 DSAC pred/conf 공급모델 상태: %s", _nf_status)
    except Exception:
        pass

    # ── DSAC + SevenModel(M7) 융합 라우터 초기화 ─────────────────────
    meta_router = DSACTrendRouter()
    enhanced_engine = EnhancedTradingEngine()
    logger.info("🧭 실행 모드: DSAC_ONLY (최소 리스크 레이어만 유지)")
    _prev_meta_pos: str | None = None

    def _sync_dsac_with_meta():
        dsac_router.pos = meta_router.pos
        dsac_router.entry_price = meta_router.entry_price
        dsac_router.hold_count = meta_router.hold_count
        dsac_router.current_leverage = meta_router.current_leverage
        dsac_router.current_equity = meta_router.cur_equity
        dsac_router.peak_equity = meta_router.peak_equity
        dsac_router.adaptive_enter_offset = meta_router.adaptive_enter_offset
        dsac_router.adaptive_agreement_offset = meta_router.adaptive_agreement_offset

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
    trend_hub = SevenModelEnsemble(strict=bool(M7_ENTRY_PRICE_ENABLE))
    if not M7_ENTRY_PRICE_ENABLE and getattr(trend_hub, "entry_price", None) is not None:
        trend_hub.entry_price.available = False
        trend_hub.entry_price.model = None
        trend_hub.entry_price.feature_cols = []
        logger.info("🧩 M7 EntryPrice 모델 비활성화: 현재 라이브 피처 스키마 기준 (M7_ENTRY_PRICE_ENABLE=OFF)")
        required_states = [
            ("trend_xgb", trend_hub.trend_xgb),
            ("multi_target_lgbm", trend_hub.multi_target),
            ("quantile_forest", trend_hub.quantile),
            ("gmm_volatility", trend_hub.gmm),
            ("hdbscan_regime", trend_hub.hdbscan),
            ("isolation_forest", trend_hub.isolation),
            ("vae_anomaly", trend_hub.vae),
        ]
        missing = [name for name, state in required_states if not state.available]
        if missing:
            raise RuntimeError("M7 startup failed (entry model disabled): missing required model(s): " + ", ".join(missing))
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
    signal_gap_hist: deque[float] = deque(maxlen=3)

    runtime_feature_keep: set[str] = build_active_feature_keep(
        include_entry_price=bool(M7_ENTRY_PRICE_ENABLE),
        include_m7_artifacts=True,
    )

    def _prune_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        keep_cols = [c for c in df.columns if c in runtime_feature_keep]
        if not keep_cols:
            return df
        dropped = len(df.columns) - len(keep_cols)
        if dropped > 0:
            logger.info("🧹 런타임 피처 프루닝: %d -> %d 컬럼 (drop=%d)", len(df.columns), len(keep_cols), dropped)
        return df[keep_cols].copy()

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
        _required_last_cols = set(ELITE_BUILDER_REQUIRED_COLS) | set(NF_RUNTIME_REQUIRED_COLS) | {"close"}
        _missing_last = [c for c in sorted(_required_last_cols) if c not in processed_df.columns]
        if _missing_last:
            raise RuntimeError(f"[LIVE_REQUIRED_FEATURE_MISSING] {','.join(_missing_last)}")
        _last_row = processed_df.iloc[-1]
        _invalid_last = []
        for _col in sorted(_required_last_cols):
            try:
                _v = float(_last_row[_col])
            except Exception:
                _invalid_last.append(_col)
                continue
            if not np.isfinite(_v):
                _invalid_last.append(_col)
        if _invalid_last:
            raise RuntimeError(f"[LIVE_REQUIRED_FEATURE_INVALID] {','.join(_invalid_last)}")

        ens_preds = None
        ens_confs = None
        try:
            ens_preds, ens_confs = await dsac_nf_predictor.predict_all_async(processed_df)
        except Exception as e:
            logger.warning("⚠️ DSAC pred/conf 공급용 앙상블 예측 실패: %s", e)
        nf_preds = {}

        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price    = float(eth_buffer['close'].iloc[-1])
        regime_name      = 'UNKNOWN'
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        meta_router.update_adaptive_gate(final_action=0, in_position=(meta_router.pos is not None))

        # ── M7 선행 요구(pred/conf -> signal_*)를 위해 마지막 봉 pred/conf 주입 ──
        if ens_preds is not None and ens_confs is not None and dsac_nf_predictor is not None and len(processed_df) > 0:
            _last_idx = processed_df.index[-1]
            _name_to_idx = {n: i for i, n in enumerate(getattr(dsac_nf_predictor, "MODEL_ORDER", []))}
            _inject_map = {
                "PatchTST": ("pred_patchtst", "conf_patchtst"),
                "Chronos": ("pred_chronos", "conf_chronos"),
                "TiDE": ("pred_tide", "conf_tide"),
            }
            for _mname, (_pcol, _ccol) in _inject_map.items():
                _idx = _name_to_idx.get(_mname)
                _model = getattr(dsac_nf_predictor, "models", {}).get(_mname) if hasattr(dsac_nf_predictor, "models") else None
                if _idx is None or _model is None or not getattr(_model, "available", False):
                    continue
                try:
                    _pv = float(ens_preds[_idx])
                    _cv = float(ens_confs[_idx])
                except Exception:
                    continue
                if np.isfinite(_pv):
                    processed_df.at[_last_idx, _pcol] = _pv
                if np.isfinite(_cv):
                    processed_df.at[_last_idx, _ccol] = _cv

        # ── M7 추론 전: elite signals를 processed_df에 사전 주입 ───────
        # (HDBSCAN 등 모델이 sig_* 피처를 사용하므로, M7 추론 전에 계산해야 함)
        try:
            _pre_last = processed_df.iloc[-1]
            _pre_prev = processed_df.iloc[-2] if len(processed_df) >= 2 else _pre_last
            _pre_smf_std = processed_df["smart_money_flow"].std() if "smart_money_flow" in processed_df.columns else 1.0
            _pre_cur = row_to_market_row(_pre_last)
            _pre_prev_mkt = row_to_market_row(_pre_prev)
            _pre_elite = dsac_router.elite_extractor.compute_all(
                current=_pre_cur, prev=_pre_prev_mkt, smf_std=_pre_smf_std
            )
            for _sig_col, _sig_val in _pre_elite.items():
                if isinstance(_sig_col, str) and _sig_col.startswith("sig_"):
                    processed_df.at[_last_idx, _sig_col] = float(_sig_val)
        except Exception as _pre_e:
            logger.debug("M7용 elite signals 사전 계산 실패: %s", _pre_e)

        # ── SevenModelEnsemble(M7) 메타 신호 추론 ────────────────────
        m7_last = None
        trend_signal = None
        try:
            m7_last = trend_hub.predict_last(processed_df)
            trend_signal = _trend_signal_from_m7(m7_last)
        except Exception as e:
            logger.warning(f"SevenModelEnsemble 추론 실패: {e}")
            logger.warning("M7 피처 생성 실패로 이번 사이클 스킵")
            return

        # DSAC 입력용 NF pred/conf 채우기:
        # 1) processed_df 마지막 행 우선 (pred/conf 쌍이 모두 유효할 때만 채움)
        # 2) 부족분은 3-앙상블(사용 가능 모델만)으로 보완
        _last = processed_df.iloc[-1]
        _need_cols = list(DSAC_STATE_PRED) + list(DSAC_STATE_CONF)
        for _pcol, _ccol in zip(DSAC_STATE_PRED, DSAC_STATE_CONF):
            if _pcol not in _last.index or _ccol not in _last.index:
                continue
            try:
                _pv = float(_last[_pcol])
                _cv = float(_last[_ccol])
            except Exception:
                continue
            if np.isfinite(_pv) and np.isfinite(_cv):
                nf_preds[_pcol] = _pv
                nf_preds[_ccol] = _cv

        # 보완 매핑: DSAC pred/conf 3개 모델 직접 공급
        _ens_map = {
            "pred_patchtst": "PatchTST",
            "conf_patchtst": "PatchTST",
            "pred_chronos": "Chronos",
            "conf_chronos": "Chronos",
            "pred_tide": "TiDE",
            "conf_tide": "TiDE",
        }
        if ens_preds is not None and ens_confs is not None and dsac_nf_predictor is not None:
            _name_to_idx = {n: i for i, n in enumerate(getattr(dsac_nf_predictor, "MODEL_ORDER", []))}
            for _key, _mname in _ens_map.items():
                if _key in nf_preds:
                    continue
                _idx = _name_to_idx.get(_mname)
                _model = getattr(dsac_nf_predictor, "models", {}).get(_mname) if hasattr(dsac_nf_predictor, "models") else None
                if _idx is None or _model is None or not getattr(_model, "available", False):
                    continue
                try:
                    _val = float(ens_preds[_idx]) if _key.startswith("pred_") else float(ens_confs[_idx])
                except Exception:
                    continue
                if np.isfinite(_val):
                    nf_preds[_key] = _val

        _missing = []
        _invalid = []
        for _col in _need_cols:
            if _col not in nf_preds:
                _missing.append(_col)
                continue
            try:
                _v = float(nf_preds[_col])
            except Exception:
                _invalid.append(_col)
                continue
            if not np.isfinite(_v):
                _invalid.append(_col)
                continue
        if _missing or _invalid:
            msg_parts = []
            if _missing:
                msg_parts.append(f"missing={','.join(_missing)}")
            if _invalid:
                msg_parts.append(f"invalid={','.join(_invalid)}")
            raise RuntimeError(f"[DSAC_PREDCONF_REQUIRED] {' | '.join(msg_parts)}")

        # DSAC 입력 생성/추론 (m7_* 주입 버전)
        _sync_dsac_with_meta()
        try:
            dsac_action, dsac_lev, info, elite_sigs, regime = dsac_router.decide(
                processed_df,
                nf_preds,
                m7_signal=trend_signal,
            )
        except Exception as e:
            logger.warning("DSAC 입력 피처 검증 실패로 사이클 스킵: %s", e)
            return
        if live_hmm is not None:
            live_hmm_steps += 1
            if live_hmm_steps % 24 == 0:
                try:
                    live_hmm.update_online(n_iter=3)
                    logger.info("🧠 Live HMM 온라인 업데이트 완료")
                except Exception as e:
                    logger.debug("Live HMM 온라인 업데이트 실패: %s", e)
        info.setdefault("agent", "DSAC_DUAL")
        info.setdefault("kelly", float(dsac_lev))
        info.setdefault("long_edge", float(info.get("_long_raw", 0.0)))
        info.setdefault("short_edge", float(info.get("_short_raw", 0.0)))
        info.setdefault("conviction", float(abs(info.get("raw_action", 0.0))))
        info.setdefault("agreement", float(abs(info.get("raw_action", 0.0))))
        info.setdefault("ambiguity", 0.0)
        info.setdefault("score", float(max(abs(info.get("raw_action", 0.0)), float(info.get("conviction", 0.0)))))
        regime_name = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')
        signal_gap_hist.append(float(info.get("raw_action", 0.0)))

        # ── 이상치 감지 (hibernation EXIT 전용) ─────────────────────────
        _iso_anom = bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5)
        _vae_anom = bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5)
        _vae_err = float((trend_signal or {}).get("m7_vae_error", 0.0) or 0.0)
        _vae_th = float((trend_signal or {}).get("m7_vae_threshold", 0.0) or 0.0)
        _vae_ratio = (_vae_err / max(_vae_th, 1e-8)) if _vae_th > 1e-8 else (1.0 if _vae_anom else 0.0)
        _jump_z = float(processed_df.iloc[-1].get("jump_z", 0.0) or 0.0)
        _evt_z = float(processed_df.iloc[-1].get("evt_excess_z", 0.0) or 0.0)
        _iso_score = float((trend_signal or {}).get("m7_iso_score", 0.0) or 0.0)
        _hib_score = float(np.clip(max(
            1.0 if _iso_anom else 0.0,
            min(_vae_ratio / 1.35, 1.5),
            min(abs(_jump_z) / 3.0, 1.5),
            min(abs(_evt_z) / 3.0, 1.5),
        ) / 1.5, 0.0, 1.0))

        # ── DSAC + M7 다요소 융합 (방향/사이즈/레짐/이상치/보유시간) ──
        prev_meta_pos = _prev_meta_pos
        if DSAC_ONLY_MODE:
            _dsac_only_source = "DSAC_PURE_RL" if DSAC_PURE_RL_MODE else "DSAC_ONLY"
            _hold_reason = str(info.get("hold_reason", ""))
            _block_reason = ""
            _trend_exit_score = 0.0

            # 순수 RL 모드: 학습 추론 결과(action/kelly)를 그대로 집행한다.
            # (게이트/스케일링/보호 레이어를 거치지 않음)
            if DSAC_PURE_RL_MODE:
                # Match training env execution: use primary continuous action and the same thresholds.
                _a = float(info.get("primary_raw", info.get("raw_action", 0.0)))
                _abs = abs(_a)
                _pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.06"))
                _close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.06"))
                _max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
                _force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}
                _fa = 0
                _kelly = 0.0
                if meta_router.pos is None:
                    if _a > _pos_th:
                        _fa, _kelly = 1, min(_abs, _max_kelly)
                    elif _a < -_pos_th:
                        _fa, _kelly = 2, min(_abs, _max_kelly)
                elif meta_router.pos == "LONG":
                    _live_unr = float(meta_router._net_pnl_frac(current_price))
                    if _force_close and _live_unr <= -0.025:
                        _fa, _kelly = 0, 0.0
                        _dsac_only_source = "DSAC_PURE_RL_FORCE_CLOSE"
                    elif _abs < _close_th:
                        _fa, _kelly = 0, 0.0
                    elif _a < -_pos_th:
                        _fa, _kelly = 2, min(_abs, _max_kelly)
                    else:
                        _fa, _kelly = 1, min(_abs, _max_kelly)
                else:  # SHORT
                    _live_unr = float(meta_router._net_pnl_frac(current_price))
                    if _force_close and _live_unr <= -0.025:
                        _fa, _kelly = 0, 0.0
                        _dsac_only_source = "DSAC_PURE_RL_FORCE_CLOSE"
                    elif _abs < _close_th:
                        _fa, _kelly = 0, 0.0
                    elif _a > _pos_th:
                        _fa, _kelly = 1, min(_abs, _max_kelly)
                    else:
                        _fa, _kelly = 2, min(_abs, _max_kelly)
                _kelly = float(np.clip(_kelly, 0.0, 1.0))
            else:
                _kelly = float(np.clip(dsac_lev, 0.0, 1.0))
                _fa = int(dsac_action)
                _position_signal = str(info.get("position_signal", "HOLD"))
                _position_reason = str(info.get("position_reason", ""))
                if _position_signal == "EXIT":
                    _fa = 0
                    _kelly = 0.0
                    _dsac_only_source = f"DSAC_LOGIT_EXIT:{_position_reason or 'RULE'}"
                elif _position_signal == "REDUCE":
                    _fa = 1 if meta_router.pos == "LONG" else 2
                    _kelly = float(np.clip(_kelly, 0.0, 1.0))
                    _dsac_only_source = f"DSAC_LOGIT_REDUCE:{_position_reason or 'RULE'}"

            meta_router._update_pos(_fa, current_price, _kelly, trend_signal)

            _sizing_diag = {
                "bayes_mult": 1.0,
                "qwidth_mult": 1.0,
                "mtf_mult": 1.0,
                "smart_mult": 1.0,
                "mdd_mult": 1.0,
                "illiquidity_mult": 1.0,
                "bayes_z": 0.0,
                "qwidth": 0.0,
                "mtf_align": 0.0,
                "smart_flow": 0.0,
                "taker_accel": 0.0,
                "recent_pnl_sum": 0.0,
                "loss_streak": int(meta_router.loss_streak),
            }
            meta_result = {
                "final_action": _fa,
                "unified_kelly": _kelly,
                "source": _dsac_only_source,
                "enhanced_source": _dsac_only_source,
                "rl_score": float(info.get("score", 0.0)),
                "rl_action": _fa,
                "trend_signal": trend_signal,
                "trend_exit_score": float(_trend_exit_score),
                "trend_mismatch_streak": int(meta_router.trend_mismatch_streak),
                "hibernation_score": float(_hib_score),
                "hibernation_score_th": float(meta_router.hibernation_score_th),
                "integral_gap_ma": float(np.mean(np.abs(np.asarray(signal_gap_hist, dtype=np.float64)))) if signal_gap_hist else 0.0,
                "illiq_amihud": float(processed_df.iloc[-1].get("amihud_illiquidity_z", 0.0) or 0.0),
                "illiq_rsvol": float(processed_df.iloc[-1].get("rogers_satchell_vol", 0.0) or 0.0),
                "cb_active": 0,
                "is_lowvol_range": 0,
                "is_highvol_trend": 0,
                "uncertainty_scale": 1.0,
                "m7_qwidth": float((trend_signal or {}).get("m7_qwidth", 0.0) or 0.0),
                "position_signal": str(info.get("position_signal", "")),
                "position_reason": str(info.get("position_reason", "")),
                "position_own_support": float(info.get("own_support", 0.0)),
                "position_opp_pressure": float(info.get("opp_pressure", 0.0)),
                "position_net_edge": float(info.get("net_edge", 0.0)),
                "hold_reason": str(_hold_reason),
                "block_reason": str(_block_reason),
                "router_enter_threshold": float(info.get("enter_threshold", 0.0)),
                "router_min_agreement_threshold": float(info.get("min_agreement_threshold", 0.0)),
                "router_max_confidence_std": float(info.get("max_confidence_std", dsac_router.max_confidence_std)),
                "router_base_min_agreement_threshold": float(info.get("base_min_agreement_threshold", 0.0)),
                "adaptive_enter_offset": float(info.get("adaptive_enter_offset", meta_router.adaptive_enter_offset)),
                "adaptive_agreement_offset": float(info.get("adaptive_agreement_offset", meta_router.adaptive_agreement_offset)),
                "router_std_gate_ok": bool(info.get("std_gate_ok", True)),
                "router_dual_high_hold": bool(info.get("dual_high_hold", False)),
                "sizing_bayes_mult": float(_sizing_diag.get("bayes_mult", 1.0)),
                "sizing_qwidth_mult": float(_sizing_diag.get("qwidth_mult", 1.0)),
                "sizing_mtf_mult": float(_sizing_diag.get("mtf_mult", 1.0)),
                "sizing_smart_mult": float(_sizing_diag.get("smart_mult", 1.0)),
                "sizing_mdd_mult": float(_sizing_diag.get("mdd_mult", 1.0)),
                "sizing_bayes_z": float(_sizing_diag.get("bayes_z", 0.0)),
                "sizing_qwidth": float(_sizing_diag.get("qwidth", 0.0)),
                "sizing_mtf_align": float(_sizing_diag.get("mtf_align", 0.0)),
                "sizing_smart_flow": float(_sizing_diag.get("smart_flow", 0.0)),
                "sizing_taker_accel": float(_sizing_diag.get("taker_accel", 0.0)),
                "sizing_recent_pnl_sum": float(_sizing_diag.get("recent_pnl_sum", 0.0)),
                "sizing_loss_streak": int(_sizing_diag.get("loss_streak", 0)),
            }
        rl_action = int(dsac_action)
        trade_pnl_pct: float | None = None

        # 직전 사이클에 포지션이 있었다가 이번에 청산됐으면 PnL 피드백
        if _prev_meta_pos is not None and meta_router.pos is None:
            realized = meta_router.last_realized_pnl
            if realized is None:
                realized = float(meta_router.cur_equity - 1.0)
            trade_pnl_pct = float(realized) * 100.0
            enhanced_engine.on_trade_close(float(realized))
            meta_router.record_outcome(float(realized))
            meta_router.append_trade_history(current_time_kst, float(realized))

        # ── 텔레그램 알림: 포지션이 바뀐 경우만 (ENTER / EXIT / FLIP) ──
        _new_pos = meta_router.pos
        if _prev_meta_pos is None and _new_pos is not None:
            enhanced_engine.on_position_open()
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
            if "enhanced_diag" in info:
                enhanced_engine.print_enhanced_dashboard({
                    "action": _fa,
                    "kelly": _kelly,
                    "source": _dsac_only_source,
                    "diagnostics": info.get("enhanced_diag", {}),
                })
        logger.info("📊 %s", meta_router.performance_summary(current_time_kst))

        # ── LLM 어드바이저 ─────────────────────────────────────────────
        _llm_ctx = {
            # 시장
            "close":              float(processed_df.iloc[-1].get("close", 0)),
            "log_return":         float(processed_df.iloc[-1].get("log_return", 0)),
            "regime":             regime_name,
            "garch_vol_z":        float(processed_df.iloc[-1].get("garch_vol_z", 0)),
            "jump_z":             float(processed_df.iloc[-1].get("jump_z", 0)),
            "evt_excess_z":       float(processed_df.iloc[-1].get("evt_excess_z", 0)),
            "last_funding_rate":  float(processed_df.iloc[-1].get("last_funding_rate", 0)),
            "funding_pressure":   float(processed_df.iloc[-1].get("funding_pressure", 0)),
            # 에이전트
            "primary_action":     int(info.get("primary_action", 0)),
            "primary_lev":        float(info.get("primary_kelly", 0)),
            "primary_std":        float(info.get("primary_std", 0)),
            "long_action":        int(info.get("_long_action", 0)),
            "long_lev":           float(info.get("_long_kelly", 0)),
            "long_logit":         float(info.get("long_logit", 0)),
            "long_std":           float(info.get("long_std", 0)),
            "short_action":       int(info.get("_short_action", 0)),
            "short_lev":          float(info.get("_short_kelly", 0)),
            "short_logit":        float(info.get("short_logit", 0)),
            "short_std":          float(info.get("short_std", 0)),
            # M7
            "m7_prob_dn":         float((trend_signal or {}).get("m7_prob_dn", 0)),
            "m7_prob_fl":         float((trend_signal or {}).get("m7_prob_fl", 0)),
            "m7_prob_up":         float((trend_signal or {}).get("m7_prob_up", 0)),
            "m7_confidence":      float((trend_signal or {}).get("m7_confidence", 0)),
            "m7_gate_block":      int((trend_signal or {}).get("m7_gate_block", 0)),
            "m7_q10":             float((trend_signal or {}).get("m7_q10", 0)),
            "m7_q50":             float((trend_signal or {}).get("m7_q50", 0)),
            "m7_q90":             float((trend_signal or {}).get("m7_q90", 0)),
            "m7_tp_offset":       float((trend_signal or {}).get("m7_tp_offset", 0)),
            "m7_sl_offset":       float((trend_signal or {}).get("m7_sl_offset", 0)),
            # 엘리트 시그널
            "sig_whale":          float((elite_sigs or {}).get("sig_whale", 0)),
            "sig_oi_divergence":  float((elite_sigs or {}).get("sig_oi_divergence", 0)),
            "sig_volume_confirm": float((elite_sigs or {}).get("sig_volume_confirm", 0)),
            "sig_trend_health":   float((elite_sigs or {}).get("sig_trend_health", 0)),
            # 포지션
            "position_type":      meta_router.pos,
            "entry_price":        float(meta_router.entry_price),
            "unrealized_pnl":     float(meta_router.cur_equity - 1.0),
            "hold_count":         int(meta_router.hold_count),
            # 컨센서스
            "agreement_count":    int(info.get("agreement_count", 0)),
            "net_score":          float(info.get("net_score", 0)),
            "kelly":              float(_kelly),
        }
        llm_result: LLMDecision | None = await llm_advisor.advise(_llm_ctx)
        if llm_result is not None:
            logger.info("🤖 %s", llm_result)

    try:
        if use_local:
            eth_buffer, btc_buffer = fetcher.load_local_data()
        else:
            logger.info("초기 캔들 데이터 수집 중...")
            try:
                eth_buffer, btc_buffer = await fetcher.fetch_initial_data()
            except Exception as e:
                logger.error("❌ 초기 캔들 수집 실패: %s", e)
                return

        if eth_buffer is None: return
        try:
            processed_boot = fe_engine.process(eth_buffer, btc_buffer)
        except Exception as e:
            logger.error("❌ 초기 피처 처리 실패: %s", e)
            return
        try:
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
                try:
                    new_eth, new_btc = await fetcher.fetch_latest_patch()
                except Exception as e:
                    logger.warning("⚠️ 최신 캔들 갱신 실패(이번 사이클 스킵): %s", e)
                    continue
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
