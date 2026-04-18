import os
import sys
import asyncio
import time
import logging
import gc
import json
import math
import importlib.util
import re
import numpy as np
import pandas as pd
import torch
import ccxt.async_support as ccxt
import warnings
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

# ── HFT 마이크로스트럭처 및 꼬리 위험 요격기 ──
from microstructure_scanner import MicrostructureScanner
from tail_risk_interceptor import TailRiskInterceptor
from playbook_router import PlaybookRouter

# Long/Short specialist 임포트 제거 — Primary DSAC 단독 운용
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
DSAC_PURE_RL_MODE = _env_flag("DSAC_PURE_RL_MODE", True)
ENH_RUNTIME_ENABLE = _env_flag("ENH_RUNTIME_ENABLE", False)
DASHBOARD_STATE_PATH = os.getenv("DASHBOARD_STATE_PATH", "data/live/dashboard_state.json")
DASHBOARD_EVENTS_PATH = os.getenv("DASHBOARD_EVENTS_PATH", "data/live/dashboard_events.jsonl")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
ENSEMBLE_BALANCED_METRICS_PATH = os.getenv(
    "ENSEMBLE_BALANCED_METRICS_PATH",
    "data/ensemble/metrics/param_ensemble_result.json",
)
ENSEMBLE_LOWFREQ_METRICS_PATH = os.getenv(
    "ENSEMBLE_LOWFREQ_METRICS_PATH",
    "data/ensemble/metrics/param_ensemble_lowfreq_grid.json",
)
ENSEMBLE_BALANCED_PARAMS_PATH = os.getenv(
    "ENSEMBLE_BALANCED_PARAMS_PATH",
    "data/ensemble/metrics/param_ensemble_result.json",
)
ENSEMBLE_LOWFREQ_PARAMS_PATH = os.getenv(
    "ENSEMBLE_LOWFREQ_PARAMS_PATH",
    "data/ensemble/metrics/param_ensemble_lowfreq_highpnl.json",
)
ENSEMBLE_TRACKER_STATE_PATH = os.getenv(
    "ENSEMBLE_TRACKER_STATE_PATH",
    "data/live/ensemble_tracker_state.json",
)
ENSEMBLE_TRACKER_RECORDS_PATH = os.getenv(
    "ENSEMBLE_TRACKER_RECORDS_PATH",
    "data/live/ensemble_trade_records.jsonl",
)
AGENT_TRACKER_STATE_PATH = os.getenv(
    "AGENT_TRACKER_STATE_PATH",
    "data/live/agent_tracker_state.json",
)
AGENT_TRACKER_RECORDS_PATH = os.getenv(
    "AGENT_TRACKER_RECORDS_PATH",
    "data/live/agent_trade_records.jsonl",
)
ENSEMBLE_TRACKER_FEE_RATE = float(os.getenv("ENSEMBLE_TRACKER_FEE_RATE", "0.0005"))
ENSEMBLE_TRACKER_SLIP_RATE = float(os.getenv("ENSEMBLE_TRACKER_SLIP_RATE", "0.0002"))
ENSEMBLE_TRACKER_EXIT_ON_HOLD = _env_flag("ENSEMBLE_TRACKER_EXIT_ON_HOLD", False)
ENSEMBLE_OVERHEAT_Z_WIN = int(float(os.getenv("ENSEMBLE_OVERHEAT_Z_WIN", "120")))
ENSEMBLE_OVERHEAT_Z_MIN = int(float(os.getenv("ENSEMBLE_OVERHEAT_Z_MIN", "20")))
QUANT_MICRO_DB_PATH = os.getenv("QUANT_MICRO_DB_PATH", "data/live/microstructure.duckdb")
QUANT_TAIL_DB_PATH = os.getenv("QUANT_TAIL_DB_PATH", "data/live/tail_risk.duckdb")
QUANT_BAR_MINUTES = int(float(os.getenv("QUANT_BAR_MINUTES", "1")))
QUANT_LOOKBACK_MINUTES = int(float(os.getenv("QUANT_LOOKBACK_MINUTES", "15")))
QUANT_HORIZON_MINUTES = 30
QUANT_TOP_K_FEATURES = int(float(os.getenv("QUANT_TOP_K_FEATURES", "25")))
QUANT_MAX_HISTORY_ROWS = int(float(os.getenv("QUANT_MAX_HISTORY_ROWS", "3000")))
QUANT_LOGIC_PATH = os.getenv("QUANT_LOGIC_PATH", "quant/live_30m_direction_quant.py")
POLYMARKET_ENABLE = _env_flag("POLYMARKET_ENABLE", True)
POLYMARKET_EVENT_SLUG = os.getenv("POLYMARKET_EVENT_SLUG", "ethereum-price-on-april-19").strip()
POLYMARKET_SLUG_AUTO = _env_flag("POLYMARKET_SLUG_AUTO", True)
POLYMARKET_SLUG_TZ = os.getenv("POLYMARKET_SLUG_TZ", "Asia/Seoul").strip() or "Asia/Seoul"
POLYMARKET_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com/events")
POLYMARKET_CLOB_PRICE_URL = os.getenv("POLYMARKET_CLOB_PRICE_URL", "https://clob.polymarket.com/price")
POLYMARKET_CLOB_BOOK_URL = os.getenv("POLYMARKET_CLOB_BOOK_URL", "https://clob.polymarket.com/book")
POLYMARKET_TIMEOUT_SEC = float(os.getenv("POLYMARKET_TIMEOUT_SEC", "2.5"))
POLYMARKET_MAX_MARKETS = int(float(os.getenv("POLYMARKET_MAX_MARKETS", "20")))
POLYMARKET_EXIT_ENABLE = _env_flag("POLYMARKET_EXIT_ENABLE", True)
POLYMARKET_SHOCK_1M_TH = float(os.getenv("POLYMARKET_SHOCK_1M_TH", "0.04"))  # 4.0%p
POLYMARKET_SHOCK_Z_TH = float(os.getenv("POLYMARKET_SHOCK_Z_TH", "4.0"))
POLYMARKET_SHOCK_CUM3_TH = float(os.getenv("POLYMARKET_SHOCK_CUM3_TH", "0.025"))  # 2.5%p
POLYMARKET_SHOCK_Z_WIN = int(float(os.getenv("POLYMARKET_SHOCK_Z_WIN", "120")))
POLYMARKET_SHOCK_DYN_ENABLE = _env_flag("POLYMARKET_SHOCK_DYN_ENABLE", False)
POLYMARKET_SHOCK_DYN_Q = float(os.getenv("POLYMARKET_SHOCK_DYN_Q", "0.997"))
POLYMARKET_SHOCK_DYN_MIN_OBS = int(float(os.getenv("POLYMARKET_SHOCK_DYN_MIN_OBS", "120")))
POLYMARKET_SHOCK_COOLDOWN_SEC = float(os.getenv("POLYMARKET_SHOCK_COOLDOWN_SEC", "1200"))
POLYMARKET_SHOCK_PEAK_ONLY = _env_flag("POLYMARKET_SHOCK_PEAK_ONLY", True)
POLYMARKET_SHOCK_PEAK_WINDOW_SEC = float(os.getenv("POLYMARKET_SHOCK_PEAK_WINDOW_SEC", "600"))

_ENSEMBLE_OI_DELTA_WIN: deque[float] = deque(maxlen=max(30, ENSEMBLE_OVERHEAT_Z_WIN))
_ENSEMBLE_FUNDING_WIN: deque[float] = deque(maxlen=max(30, ENSEMBLE_OVERHEAT_Z_WIN))
_ENSEMBLE_LAST_OVERHEAT_OBS: tuple[float, float] | None = None
_QUANT_CARD_CACHE: dict[str, object] = {"minute_key": "", "payload": {}}
_QUANT_SNAPSHOT_MTIME: float | None = None
_POLYMARKET_CACHE: dict[str, object] = {"updated_at": 0.0, "payload": {}}
_POLYMARKET_HISTORY: deque[dict] = deque(maxlen=720)
_POLYMARKET_LAST_EMERGENCY_TS: float = 0.0


def _resolve_quant_logic_path() -> str:
    candidates = [
        os.path.join(_THIS_DIR, QUANT_LOGIC_PATH),
        os.path.join(_THIS_DIR, "quant", "live_30m_direction_quant.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _load_quant_snapshot_fn():
    global _QUANT_SNAPSHOT_MTIME
    script_path = _resolve_quant_logic_path()
    if not os.path.exists(script_path):
        _QUANT_SNAPSHOT_MTIME = None
        return None
    try:
        spec = importlib.util.spec_from_file_location("build_geometric_objective_dataset", script_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # dataclass/typing introspection requires module to exist in sys.modules
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _QUANT_SNAPSHOT_MTIME = float(os.path.getmtime(script_path))
        return getattr(mod, "compute_live_quant_snapshot", None)
    except Exception as e:
        logger.warning("퀀트 카드 스크립트 로딩 실패: %s", e)
        return None


_QUANT_SNAPSHOT_FN = _load_quant_snapshot_fn()


def _poly_http_json(url: str, timeout: float = 2.5):
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            "Connection": "close",
        },
    )
    with urlopen(req, timeout=max(0.2, float(timeout))) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _poly_clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _poly_extract_token_ids(market: dict) -> list[str]:
    cand = market.get("clobTokenIds", [])
    out: list[str] = []
    if isinstance(cand, list):
        for x in cand:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(cand, str):
        txt = cand.strip()
        if not txt:
            return out
        if txt.startswith("["):
            try:
                arr = json.loads(txt)
                if isinstance(arr, list):
                    for x in arr:
                        s = str(x).strip()
                        if s:
                            out.append(s)
            except Exception:
                pass
        else:
            for x in txt.split(","):
                s = str(x).strip().strip("\"").strip("'")
                if s:
                    out.append(s)
    return out


def _poly_parse_strike_from_text(text: str) -> float | None:
    if not text:
        return None
    t = str(text)
    m = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", t)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    m2 = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([kK])\b", t)
    if m2:
        try:
            return float(m2.group(1)) * 1000.0
        except Exception:
            return None
    return None


def _poly_market_label(market: dict) -> str:
    for k in ("question", "title", "name", "description"):
        v = market.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown_market"


def _poly_price_from_resp(payload) -> float | None:
    if isinstance(payload, (int, float)):
        x = float(payload)
        return _poly_clamp01(x) if np.isfinite(x) else None
    if isinstance(payload, str):
        try:
            return _poly_clamp01(float(payload))
        except Exception:
            return None
    if isinstance(payload, dict):
        for k in ("price", "mid", "value"):
            if k in payload:
                try:
                    return _poly_clamp01(float(payload.get(k)))
                except Exception:
                    pass
    return None


def _poly_book_imbalance(token_id: str) -> dict:
    try:
        q = urlencode({"token_id": token_id})
        url = f"{POLYMARKET_CLOB_BOOK_URL}?{q}"
        data = _poly_http_json(url, timeout=POLYMARKET_TIMEOUT_SEC)
        bids = list((data or {}).get("bids", []) or [])
        asks = list((data or {}).get("asks", []) or [])
        b_notional = 0.0
        a_notional = 0.0
        for row in bids[:20]:
            p = float(row.get("price", 0.0) or 0.0)
            s = float(row.get("size", 0.0) or 0.0)
            if p > 0.0 and s > 0.0:
                b_notional += (p * s)
        for row in asks[:20]:
            p = float(row.get("price", 0.0) or 0.0)
            s = float(row.get("size", 0.0) or 0.0)
            if p > 0.0 and s > 0.0:
                a_notional += (p * s)
        den = b_notional + a_notional
        imb = ((b_notional - a_notional) / den) if den > 1e-12 else 0.0
        return {
            "bid_notional": float(b_notional),
            "ask_notional": float(a_notional),
            "imbalance": float(np.clip(imb, -1.0, 1.0)),
        }
    except Exception:
        return {"bid_notional": 0.0, "ask_notional": 0.0, "imbalance": 0.0}


def _resolve_polymarket_slug() -> tuple[str, str]:
    base = str(POLYMARKET_EVENT_SLUG or "").strip()
    if not POLYMARKET_SLUG_AUTO:
        return base, "manual"
    m = re.match(r"^(.*?-price)-on-[a-z]+-\d{1,2}$", base)
    if m is None:
        return base, "manual_pattern_mismatch"
    prefix = str(m.group(1)).strip()
    if not prefix:
        return base, "manual_prefix_empty"
    try:
        now_local = pd.Timestamp.now(tz=POLYMARKET_SLUG_TZ)
        target = now_local
        month = target.strftime("%B").lower()
        day = str(int(target.day))
        return f"{prefix}-on-{month}-{day}", "auto_today"
    except Exception:
        return base, "manual_time_fallback"


def _fetch_polymarket_rows_by_slug(slug: str) -> list[dict]:
    query = urlencode({"slug": slug})
    url = f"{POLYMARKET_GAMMA_URL}?{query}"
    rows: list[dict] = []
    ev_raw = _poly_http_json(url, timeout=POLYMARKET_TIMEOUT_SEC)
    if isinstance(ev_raw, list):
        ev = dict(ev_raw[0] or {}) if ev_raw else {}
    elif isinstance(ev_raw, dict):
        items = ev_raw.get("events", ev_raw.get("data", []))
        if isinstance(items, list):
            ev = dict(items[0] or {}) if items else dict(ev_raw)
        else:
            ev = dict(ev_raw)
    else:
        ev = {}
    mkts = list((ev or {}).get("markets", []) or [])
    for m in mkts[:max(1, POLYMARKET_MAX_MARKETS)]:
        md = dict(m or {})
        token_ids = _poly_extract_token_ids(md)
        if not token_ids:
            continue
        token_id = token_ids[0]
        purl = f"{POLYMARKET_CLOB_PRICE_URL}?{urlencode({'token_id': token_id, 'side': 'BUY'})}"
        try:
            price_payload = _poly_http_json(purl, timeout=POLYMARKET_TIMEOUT_SEC)
            prob = _poly_price_from_resp(price_payload)
        except Exception:
            prob = None
        if prob is None:
            continue
        label = _poly_market_label(md)
        strike = _poly_parse_strike_from_text(label)
        rows.append({
            "label": label,
            "token_id": token_id,
            "prob": float(prob),
            "strike": float(strike) if strike is not None else np.nan,
        })
    return rows


def _poly_hist_delta(hist: list[dict], sec: float) -> float:
    if not hist:
        return 0.0
    try:
        now_ts = float(hist[-1].get("ts", 0.0) or 0.0)
        now_mode = float(hist[-1].get("mode_prob", 0.0) or 0.0)
    except Exception:
        return 0.0
    target = now_ts - float(max(1.0, sec))
    prev_mode = None
    for row in reversed(hist):
        rts = float(row.get("ts", 0.0) or 0.0)
        if rts <= target:
            try:
                prev_mode = float(row.get("mode_prob", 0.0) or 0.0)
            except Exception:
                prev_mode = None
            break
    if prev_mode is None:
        return 0.0
    return float(now_mode - prev_mode)


def _poly_hist_delta_series(hist: list[dict], sec: float, cap: int = 240) -> list[float]:
    n = len(hist)
    if n < 2:
        return []
    use_n = min(max(2, int(cap)), n)
    sub = hist[-use_n:]
    ts = np.array([float(x.get("ts", 0.0) or 0.0) for x in sub], dtype=float)
    mode = np.array([float(x.get("mode_prob", 0.0) or 0.0) for x in sub], dtype=float)
    out: list[float] = []
    j = 0
    for i in range(use_n):
        target = ts[i] - float(max(1.0, sec))
        while j + 1 < i and ts[j + 1] <= target:
            j += 1
        if ts[j] <= target:
            out.append(float(mode[i] - mode[j]))
    return out


def _poly_hist_delta_pairs(hist: list[dict], sec: float, cap: int = 240) -> list[tuple[float, float]]:
    n = len(hist)
    if n < 2:
        return []
    use_n = min(max(2, int(cap)), n)
    sub = hist[-use_n:]
    ts = np.array([float(x.get("ts", 0.0) or 0.0) for x in sub], dtype=float)
    mode = np.array([float(x.get("mode_prob", 0.0) or 0.0) for x in sub], dtype=float)
    out: list[tuple[float, float]] = []
    j = 0
    for i in range(use_n):
        target = ts[i] - float(max(1.0, sec))
        while j + 1 < i and ts[j + 1] <= target:
            j += 1
        if ts[j] <= target:
            out.append((float(ts[i]), float(mode[i] - mode[j])))
    return out


def _build_polymarket_snapshot(current_price: float) -> dict:
    now_iso = pd.Timestamp.utcnow().isoformat()
    resolved_slug, slug_mode = _resolve_polymarket_slug()
    used_slug = resolved_slug or POLYMARKET_EVENT_SLUG
    if not POLYMARKET_ENABLE:
        return {
            "updated_at": now_iso,
            "status": "DISABLED",
            "slug": used_slug,
            "slug_mode": slug_mode,
            "error": "disabled",
            "markets_count": 0,
            "priced_count": 0,
            "mode_label": "-",
            "mode_prob": 0.0,
            "weighted_target": float(current_price or 0.0),
            "tail_up_prob": 0.0,
            "tail_down_prob": 0.0,
            "prob_momentum_1m": 0.0,
            "prob_price_corr": 0.0,
            "book_imbalance": 0.0,
            "book_bid_notional": 0.0,
            "book_ask_notional": 0.0,
            "event_volatility": 0.0,
            "signal": "HOLD",
            "risk_state": "NORMAL",
            "shock_delta_1m": 0.0,
            "shock_delta_3m": 0.0,
            "shock_z_1m": 0.0,
            "shock_dyn_th_1m": 0.0,
            "shock_trigger": False,
            "shock_trigger_reason": "",
            "snapshot_ts_epoch": float(time.time()),
        }

    rows: list[dict] = []
    try:
        rows = _fetch_polymarket_rows_by_slug(used_slug)
        if not rows and used_slug != POLYMARKET_EVENT_SLUG:
            rows = _fetch_polymarket_rows_by_slug(POLYMARKET_EVENT_SLUG)
            if rows:
                used_slug = POLYMARKET_EVENT_SLUG
                slug_mode = "fallback_manual_slug"
    except Exception as e:
        return {
            "updated_at": now_iso,
            "status": "ERROR",
            "slug": used_slug,
            "slug_mode": slug_mode,
            "error": str(e),
            "markets_count": 0,
            "priced_count": 0,
            "mode_label": "-",
            "mode_prob": 0.0,
            "weighted_target": float(current_price or 0.0),
            "tail_up_prob": 0.0,
            "tail_down_prob": 0.0,
            "prob_momentum_1m": 0.0,
            "prob_price_corr": 0.0,
            "book_imbalance": 0.0,
            "book_bid_notional": 0.0,
            "book_ask_notional": 0.0,
            "event_volatility": 0.0,
            "signal": "HOLD",
            "risk_state": "NORMAL",
            "shock_delta_1m": 0.0,
            "shock_delta_3m": 0.0,
            "shock_z_1m": 0.0,
            "shock_dyn_th_1m": 0.0,
            "shock_trigger": False,
            "shock_trigger_reason": "",
            "snapshot_ts_epoch": float(time.time()),
        }

    if not rows:
        return {
            "updated_at": now_iso,
            "status": "EMPTY",
            "slug": used_slug,
            "slug_mode": slug_mode,
            "error": "no_priced_markets",
            "markets_count": 0,
            "priced_count": 0,
            "mode_label": "-",
            "mode_prob": 0.0,
            "weighted_target": float(current_price or 0.0),
            "tail_up_prob": 0.0,
            "tail_down_prob": 0.0,
            "prob_momentum_1m": 0.0,
            "prob_price_corr": 0.0,
            "book_imbalance": 0.0,
            "book_bid_notional": 0.0,
            "book_ask_notional": 0.0,
            "event_volatility": 0.0,
            "signal": "HOLD",
            "risk_state": "NORMAL",
            "shock_delta_1m": 0.0,
            "shock_delta_3m": 0.0,
            "shock_z_1m": 0.0,
            "shock_dyn_th_1m": 0.0,
            "shock_trigger": False,
            "shock_trigger_reason": "",
            "snapshot_ts_epoch": float(time.time()),
        }

    rows_sorted = sorted(rows, key=lambda x: float(x.get("prob", 0.0)), reverse=True)
    mode = rows_sorted[0]
    probs = np.array([float(x.get("prob", 0.0)) for x in rows_sorted], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    prob_sum = float(np.sum(probs))
    w = (probs / prob_sum) if prob_sum > 1e-12 else np.full_like(probs, 1.0 / max(1, len(probs)))
    strikes = np.array([float(x.get("strike", np.nan)) for x in rows_sorted], dtype=float)
    strike_ok = np.isfinite(strikes)
    if np.any(strike_ok):
        weighted_target = float(np.sum(strikes[strike_ok] * w[strike_ok]) / max(np.sum(w[strike_ok]), 1e-12))
    else:
        weighted_target = float(current_price or 0.0)

    cur_px = float(current_price or 0.0)
    if cur_px > 0.0 and np.any(strike_ok):
        up_mask = strike_ok & (strikes >= cur_px * 1.03)
        dn_mask = strike_ok & (strikes <= cur_px * 0.97)
        tail_up_prob = float(np.sum(w[up_mask])) if np.any(up_mask) else 0.0
        tail_down_prob = float(np.sum(w[dn_mask])) if np.any(dn_mask) else 0.0
    else:
        tail_up_prob = 0.0
        tail_down_prob = 0.0

    event_volatility = float(np.std(probs))
    _POLYMARKET_HISTORY.append({
        "ts": float(time.time()),
        "mode_prob": float(mode.get("prob", 0.0)),
        "weighted_target": float(weighted_target),
        "current_price": float(cur_px),
    })
    hist = list(_POLYMARKET_HISTORY)
    prob_momentum_1m = _poly_hist_delta(hist, sec=60.0)
    shock_delta_3m = _poly_hist_delta(hist, sec=180.0)
    d1m_series = _poly_hist_delta_series(hist, sec=60.0, cap=max(180, POLYMARKET_SHOCK_Z_WIN * 3))
    d1m_pairs = _poly_hist_delta_pairs(hist, sec=60.0, cap=max(180, POLYMARKET_SHOCK_Z_WIN * 3))
    shock_z_1m = 0.0
    shock_dyn_th_1m = 0.0
    shock_peak_abs_1m = 0.0
    shock_is_peak = False
    if d1m_series:
        abs_arr = np.abs(np.array(d1m_series, dtype=float))
        if len(abs_arr) >= int(max(20, POLYMARKET_SHOCK_DYN_MIN_OBS)):
            q = float(np.clip(POLYMARKET_SHOCK_DYN_Q, 0.50, 0.9999))
            shock_dyn_th_1m = float(np.quantile(abs_arr, q))
        win = np.array(d1m_series[-max(20, POLYMARKET_SHOCK_Z_WIN):], dtype=float)
        if len(win) >= 10:
            mu = float(np.mean(win))
            sd = float(np.std(win))
            if sd > 1e-12 and np.isfinite(sd):
                shock_z_1m = float((d1m_series[-1] - mu) / sd)
    if d1m_pairs:
        now_ts = float(hist[-1].get("ts", time.time()) if hist else time.time())
        lo_ts = now_ts - float(max(30.0, POLYMARKET_SHOCK_PEAK_WINDOW_SEC))
        recent_abs = [abs(float(v)) for (t, v) in d1m_pairs if float(t) >= lo_ts]
        if recent_abs:
            shock_peak_abs_1m = float(max(recent_abs))
            shock_is_peak = bool(abs(prob_momentum_1m) >= (shock_peak_abs_1m - 1e-12))
    eff_th = float(max(0.0, POLYMARKET_SHOCK_1M_TH))
    if POLYMARKET_SHOCK_DYN_ENABLE and shock_dyn_th_1m > 0.0:
        eff_th = max(eff_th, float(shock_dyn_th_1m))
    cond_abs = abs(prob_momentum_1m) >= eff_th
    cond_z = abs(shock_z_1m) >= float(max(0.0, POLYMARKET_SHOCK_Z_TH))
    cond_c3 = abs(shock_delta_3m) >= float(max(0.0, POLYMARKET_SHOCK_CUM3_TH))
    cond_peak = (not POLYMARKET_SHOCK_PEAK_ONLY) or bool(shock_is_peak)
    shock_trigger = bool(cond_abs and cond_z and cond_c3 and cond_peak)
    if shock_trigger:
        shock_trigger_reason = (
            f"|d1m|={abs(prob_momentum_1m)*100:.2f}%p>=th{eff_th*100:.2f}%p "
            f"& |z|={abs(shock_z_1m):.2f}>={POLYMARKET_SHOCK_Z_TH:.2f} "
            f"& |d3m|={abs(shock_delta_3m)*100:.2f}%p>={POLYMARKET_SHOCK_CUM3_TH*100:.2f}%p"
            f"& peak={int(bool(shock_is_peak))}"
        )
    else:
        shock_trigger_reason = ""
    corr = 0.0
    if len(hist) >= 20:
        px = np.array([float(x.get("current_price", 0.0)) for x in hist], dtype=float)
        pp = np.array([float(x.get("mode_prob", 0.0)) for x in hist], dtype=float)
        px_ret = np.diff(px) / np.maximum(np.abs(px[:-1]), 1e-8)
        pp_ret = np.diff(pp)
        if len(px_ret) >= 5 and np.std(px_ret) > 1e-12 and np.std(pp_ret) > 1e-12:
            corr = float(np.corrcoef(px_ret, pp_ret)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0

    book = _poly_book_imbalance(str(mode.get("token_id", "")))
    imb = float(book.get("imbalance", 0.0))
    lead_edge = prob_momentum_1m + (0.2 * imb)
    if lead_edge > 0.03:
        signal = "LONG"
    elif lead_edge < -0.03:
        signal = "SHORT"
    else:
        signal = "HOLD"
    risk_state = "HIGH_VOL" if event_volatility >= 0.12 else ("WATCH" if event_volatility >= 0.07 else "NORMAL")
    leverage_mult = 0.7 if risk_state == "HIGH_VOL" else (0.85 if risk_state == "WATCH" else 1.0)

    return {
        "updated_at": now_iso,
        "status": "LIVE",
        "slug": used_slug,
        "slug_mode": slug_mode,
        "error": "",
        "markets_count": int(len(rows)),
        "priced_count": int(len(rows)),
        "mode_label": str(mode.get("label", "-")),
        "mode_prob": float(mode.get("prob", 0.0)),
        "weighted_target": float(weighted_target),
        "tail_up_prob": float(tail_up_prob),
        "tail_down_prob": float(tail_down_prob),
        "prob_momentum_1m": float(prob_momentum_1m),
        "shock_delta_1m": float(prob_momentum_1m),
        "shock_delta_3m": float(shock_delta_3m),
        "shock_z_1m": float(shock_z_1m),
        "shock_dyn_th_1m": float(shock_dyn_th_1m),
        "shock_peak_abs_1m": float(shock_peak_abs_1m),
        "shock_is_peak": bool(shock_is_peak),
        "shock_trigger": bool(shock_trigger),
        "shock_trigger_reason": str(shock_trigger_reason),
        "snapshot_ts_epoch": float(hist[-1].get("ts", time.time()) if hist else time.time()),
        "prob_price_corr": float(corr),
        "book_imbalance": float(imb),
        "book_bid_notional": float(book.get("bid_notional", 0.0)),
        "book_ask_notional": float(book.get("ask_notional", 0.0)),
        "event_volatility": float(event_volatility),
        "signal": str(signal),
        "risk_state": str(risk_state),
        "recommended_kelly_mult": float(leverage_mult),
    }


def _get_polymarket_snapshot_cached(current_price: float) -> dict:
    now = time.time()
    cached = dict(_POLYMARKET_CACHE.get("payload", {}) or {})
    updated_at = float(_POLYMARKET_CACHE.get("updated_at", 0.0) or 0.0)
    if cached and (now - updated_at) < 8.0:
        return cached
    fresh = _build_polymarket_snapshot(current_price=float(current_price or 0.0))
    _POLYMARKET_CACHE["payload"] = dict(fresh)
    _POLYMARKET_CACHE["updated_at"] = float(now)
    return dict(fresh)


def _polymarket_exit_guard(pos: str | None, entry_price: float, poly: dict) -> tuple[bool, str]:
    """
    Return (force_exit, reason).
    Rule:
      - only when 1m polymarket shock magnitude >= threshold
      - HOLD if prediction is favorable from entry price
      - otherwise EXIT
    """
    global _POLYMARKET_LAST_EMERGENCY_TS
    if not POLYMARKET_EXIT_ENABLE:
        return False, ""
    side = str(pos or "").upper()
    if side not in ("LONG", "SHORT"):
        return False, ""
    if entry_price <= 0.0:
        return False, ""
    status = str((poly or {}).get("status", "")).upper()
    if status != "LIVE":
        return False, ""
    mom = float((poly or {}).get("shock_delta_1m", (poly or {}).get("prob_momentum_1m", 0.0)) or 0.0)
    d3 = float((poly or {}).get("shock_delta_3m", 0.0) or 0.0)
    z1 = float((poly or {}).get("shock_z_1m", 0.0) or 0.0)
    is_peak = bool((poly or {}).get("shock_is_peak", False))
    dyn = float((poly or {}).get("shock_dyn_th_1m", 0.0) or 0.0)
    eff_th = float(max(0.0, POLYMARKET_SHOCK_1M_TH))
    if POLYMARKET_SHOCK_DYN_ENABLE and dyn > 0.0:
        eff_th = max(eff_th, dyn)
    cond_abs = abs(mom) >= eff_th
    cond_z = abs(z1) >= float(max(0.0, POLYMARKET_SHOCK_Z_TH))
    cond_c3 = abs(d3) >= float(max(0.0, POLYMARKET_SHOCK_CUM3_TH))
    cond_peak = (not POLYMARKET_SHOCK_PEAK_ONLY) or is_peak
    trigger = bool((poly or {}).get("shock_trigger", False))
    if not (trigger or (cond_abs and cond_z and cond_c3 and cond_peak)):
        return False, ""
    now_ts = float((poly or {}).get("snapshot_ts_epoch", time.time()) or time.time())
    cooldown = float(max(0.0, POLYMARKET_SHOCK_COOLDOWN_SEC))
    if cooldown > 0.0 and _POLYMARKET_LAST_EMERGENCY_TS > 0.0:
        if (now_ts - _POLYMARKET_LAST_EMERGENCY_TS) < cooldown:
            remain = int(max(0.0, cooldown - (now_ts - _POLYMARKET_LAST_EMERGENCY_TS)))
            return False, f"POLYMARKET_SHOCK_COOLDOWN({remain}s)"
    _POLYMARKET_LAST_EMERGENCY_TS = now_ts
    tgt = float((poly or {}).get("weighted_target", 0.0) or 0.0)
    if tgt <= 0.0:
        return False, ""
    favorable = (tgt > entry_price) if side == "LONG" else (tgt < entry_price)
    if favorable:
        return False, (
            "POLYMARKET_EMERGENCY_HOLD("
            f"|d1m|={abs(mom)*100:.2f}%p>=th{eff_th*100:.2f}%p,"
            f"|z|={abs(z1):.2f},|d3m|={abs(d3)*100:.2f}%p,peak={int(is_peak)},target={tgt:.2f},entry={entry_price:.2f})"
        )
    return True, (
        "POLYMARKET_EMERGENCY_EXIT("
        f"|d1m|={abs(mom)*100:.2f}%p>=th{eff_th*100:.2f}%p,"
        f"|z|={abs(z1):.2f},|d3m|={abs(d3)*100:.2f}%p,peak={int(is_peak)},target={tgt:.2f},entry={entry_price:.2f})"
    )


def _build_quant_formula_card(eth_df: pd.DataFrame, current_price: float, current_time_kst) -> dict:
    global _QUANT_SNAPSHOT_FN, _QUANT_SNAPSHOT_MTIME
    minute_key = pd.Timestamp(current_time_kst).strftime("%Y-%m-%d %H:%M")
    cached = dict(_QUANT_CARD_CACHE.get("payload", {}) or {})
    if _QUANT_CARD_CACHE.get("minute_key") == minute_key and cached:
        return cached
    script_path = _resolve_quant_logic_path()
    try:
        now_mtime = float(os.path.getmtime(script_path)) if os.path.exists(script_path) else None
    except Exception:
        now_mtime = None
    if now_mtime is not None and _QUANT_SNAPSHOT_MTIME is not None and now_mtime != _QUANT_SNAPSHOT_MTIME:
        _QUANT_SNAPSHOT_FN = _load_quant_snapshot_fn()
    if _QUANT_SNAPSHOT_FN is None:
        _QUANT_SNAPSHOT_FN = _load_quant_snapshot_fn()
    if _QUANT_SNAPSHOT_FN is None:
        payload = {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "signal": "HOLD",
            "direction": "NEUTRAL",
            "prob_up": 0.5,
            "prob_down": 0.5,
            "pred_price_30m": float(current_price or 0.0),
            "current_price": float(current_price or 0.0),
            "expected_return_pct": 0.0,
            "confidence": 0.0,
            "win_rate_model": 0.0,
            "win_rate_baseline": 0.0,
            "rmse_model": 0.0,
            "rmse_naive": 0.0,
            "r2_model": 0.0,
            "r2_naive": 0.0,
            "alpha": 0.0,
            "l2": 0.0,
            "error": "quant_fn_not_loaded",
        }
        return payload
    try:
        close_df = pd.DataFrame({
            "ts": pd.to_datetime(eth_df["timestamp"], utc=True, errors="coerce"),
            "close": pd.to_numeric(eth_df["close"], errors="coerce"),
        }).dropna(subset=["ts", "close"])
        payload = _QUANT_SNAPSHOT_FN(
            micro_db_path=QUANT_MICRO_DB_PATH,
            tail_db_path=QUANT_TAIL_DB_PATH,
            close_df=close_df,
            current_price=float(current_price or 0.0),
            lookback_minutes=int(max(1, QUANT_LOOKBACK_MINUTES)),
            horizon_minutes=int(max(1, QUANT_HORIZON_MINUTES)),
            bar_minutes=int(max(1, QUANT_BAR_MINUTES)),
            top_k_features=int(max(5, QUANT_TOP_K_FEATURES)),
            max_history_rows=int(max(500, QUANT_MAX_HISTORY_ROWS)),
        )
        _QUANT_CARD_CACHE["minute_key"] = minute_key
        _QUANT_CARD_CACHE["payload"] = payload
        return dict(payload)
    except Exception as e:
        payload = {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "signal": "HOLD",
            "direction": "NEUTRAL",
            "prob_up": 0.5,
            "prob_down": 0.5,
            "pred_price_30m": float(current_price or 0.0),
            "current_price": float(current_price or 0.0),
            "expected_return_pct": 0.0,
            "confidence": 0.0,
            "win_rate_model": 0.0,
            "win_rate_baseline": 0.0,
            "rmse_model": 0.0,
            "rmse_naive": 0.0,
            "r2_model": 0.0,
            "r2_naive": 0.0,
            "alpha": 0.0,
            "l2": 0.0,
            "error": str(e),
        }
        # 에러 시에는 분 캐시를 고정하지 않아 다음 사이클 즉시 재시도
        return payload

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


def _atomic_write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now_kst_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Seoul").isoformat()


def _read_json_safe(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_ensemble_cards() -> dict:
    now_iso = _now_kst_iso()
    bal_raw = _read_json_safe(ENSEMBLE_BALANCED_METRICS_PATH)
    low_raw = _read_json_safe(ENSEMBLE_LOWFREQ_METRICS_PATH)
    try:
        bal_updated = pd.Timestamp.fromtimestamp(os.path.getmtime(ENSEMBLE_BALANCED_METRICS_PATH), tz="UTC").isoformat()
    except Exception:
        bal_updated = now_iso
    try:
        low_updated = pd.Timestamp.fromtimestamp(os.path.getmtime(ENSEMBLE_LOWFREQ_METRICS_PATH), tz="UTC").isoformat()
    except Exception:
        low_updated = now_iso

    bal_res = dict(bal_raw.get("ensemble_result", {}) or {})
    bal_search = dict(bal_raw.get("search", {}) or {})
    bal = {
        "name": "균형 앙상블",
        "spec": f"k={int(bal_search.get('top_k', 10) or 10)} / votes={int(bal_search.get('min_votes', 6) or 6)}",
        "param_updated_at": bal_updated,
        "update_cycle": "주 1회 업데이트",
        "pnl_pct": float(bal_res.get("pnl_pct", 0.0) or 0.0),
        "mdd_pct": float(bal_res.get("mdd_pct", 0.0) or 0.0),
        "trades": int(bal_res.get("trades", 0) or 0),
        "win_rate": float(bal_res.get("win_rate", 0.0) or 0.0),
        "sharpe": float(bal_res.get("sharpe", 0.0) or 0.0),
    }

    low_best = dict(low_raw.get("best", {}) or {})
    low = {
        "name": "저빈도 고수익 앙상블",
        "spec": f"k={int(low_best.get('k', 10) or 10)} / votes={int(low_best.get('votes', 7) or 7)}",
        "param_updated_at": low_updated,
        "update_cycle": "주 1회 업데이트",
        "pnl_pct": float(low_best.get("pnl_pct", 0.0) or 0.0),
        "mdd_pct": float(low_best.get("mdd_pct", 0.0) or 0.0),
        "trades": int(low_best.get("trades", 0) or 0),
        "win_rate": float(low_best.get("win_rate", 0.0) or 0.0),
        "sharpe": float(low_best.get("sharpe", 0.0) or 0.0),
    }

    return {
        "updated_at": now_iso,
        "balanced": bal,
        "lowfreq": low,
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pb_score(ev: dict | None) -> float:
    if not isinstance(ev, dict):
        return 0.0
    meta = ev.get("meta", {}) or {}
    raw = meta.get("unified_score", None)
    try:
        if raw is not None and np.isfinite(float(raw)):
            return _clamp01(float(raw))
    except Exception:
        pass
    return 1.0 if bool(ev.get("matched", False)) else 0.0


def _pb_action(ev: dict | None, default: int = 0) -> int:
    if not isinstance(ev, dict):
        return int(default)
    try:
        a = int(ev.get("action", default))
        return a if a in (0, 1, 2) else int(default)
    except Exception:
        return int(default)


def _action_text(a: int) -> str:
    return "LONG" if int(a) == 1 else ("SHORT" if int(a) == 2 else "HOLD")


def _safe_float(v, d: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(d)
    except Exception:
        return float(d)


def _zscore_last(win: deque[float], x: float) -> float:
    if len(win) < max(2, ENSEMBLE_OVERHEAT_Z_MIN):
        return 0.0
    arr = np.array(win, dtype=np.float64)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float((x - mu) / sd)


def _update_overheat_score(ms: dict) -> float:
    global _ENSEMBLE_LAST_OVERHEAT_OBS
    oi_delta = _safe_float(ms.get("oi_delta_pct", 0.0), 0.0)
    funding = _safe_float(ms.get("funding_rate", 0.0), 0.0)
    obs = (round(oi_delta, 12), round(funding, 12))
    if _ENSEMBLE_LAST_OVERHEAT_OBS != obs:
        _ENSEMBLE_OI_DELTA_WIN.append(float(oi_delta))
        _ENSEMBLE_FUNDING_WIN.append(float(funding))
        _ENSEMBLE_LAST_OVERHEAT_OBS = obs
    oi_z = _zscore_last(_ENSEMBLE_OI_DELTA_WIN, oi_delta)
    funding_z = _zscore_last(_ENSEMBLE_FUNDING_WIN, funding)
    if len(_ENSEMBLE_OI_DELTA_WIN) < ENSEMBLE_OVERHEAT_Z_MIN or len(_ENSEMBLE_FUNDING_WIN) < ENSEMBLE_OVERHEAT_Z_MIN:
        # 워밍업 구간 fallback: 기존 스케일 유지
        overheat = (oi_delta * 100.0) + (funding * 1500.0)
    else:
        overheat = oi_z + funding_z
    ms["overheat_score"] = float(overheat)
    ms["overheat_oi_z"] = float(oi_z)
    ms["overheat_funding_z"] = float(funding_z)
    ms["overheat_samples"] = int(min(len(_ENSEMBLE_OI_DELTA_WIN), len(_ENSEMBLE_FUNDING_WIN)))
    return float(overheat)


def _load_param_pool(path: str, top_key: str) -> list[dict]:
    raw = _read_json_safe(path)
    arr = list(raw.get(top_key, []) or [])
    out: list[dict] = []
    for item in arr:
        p = dict((item or {}).get("params", {}) or {})
        if p:
            out.append(p)
    return out


def _param_vote_from_snapshot(p: dict, ms: dict, tr: dict, current_pos: str = "NONE") -> tuple[int, float, float]:
    obi = _safe_float(ms.get("obi", 0.0), 0.0)
    taker_buy_ratio = _safe_float(ms.get("taker_buy_ratio", 0.5), 0.5)
    flow = 2.0 * max(0.0, min(1.0, taker_buy_ratio)) - 1.0
    nif = _safe_float(ms.get("nif_whale", 0.0), 0.0)
    absb = _safe_float(ms.get("shadow_absorption_score", 0.0), 0.0)
    tox = _safe_float(ms.get("shadow_toxicity_score", 0.0), 0.0)
    qcol = _safe_float(ms.get("shadow_queue_collapse", 0.0), 0.0)
    eai = _safe_float(ms.get("eai", 0.0), 0.0)
    oi_delta = _safe_float(ms.get("oi_delta_pct", 0.0), 0.0)
    funding = _safe_float(ms.get("funding_rate", 0.0), 0.0)
    pv30 = abs(_safe_float(ms.get("price_volatility_30m", 0.0), 0.0))
    warmup_price_samples = int(_safe_float(ms.get("warmup_price_samples", 0.0), 0.0))
    data_stale = bool(ms.get("data_stale", False))
    long_usd = _safe_float(tr.get("long_usd_1m", 0.0), 0.0)
    short_usd = _safe_float(tr.get("short_usd_1m", 0.0), 0.0)
    aft = _safe_float(tr.get("aftershock_prob", tr.get("shadow_aftershock_prob", 0.0)), 0.0)

    liq = (short_usd - long_usd) / (abs(short_usd) + abs(long_usd) + 1e-8)
    overheat = _safe_float(ms.get("overheat_score", (oi_delta * 100.0) + (funding * 1500.0)), 0.0)

    raw = (
        _safe_float(p.get("w_nif", 0.0)) * nif
        + _safe_float(p.get("w_flow", 0.0)) * flow
        + _safe_float(p.get("w_obi", 0.0)) * (-obi)
        + _safe_float(p.get("w_abs", 0.0)) * absb
        + _safe_float(p.get("w_liq", 0.0)) * liq
        + _safe_float(p.get("w_eai", 0.0)) * np.tanh(eai / 2.0)
        - _safe_float(p.get("w_tox", 0.0)) * tox
        - _safe_float(p.get("w_aft", 0.0)) * aft
    )

    temp = max(1e-4, _safe_float(p.get("temp", 0.25), 0.25))
    bias = _safe_float(p.get("bias", 0.0), 0.0)
    long_gate = 1.0 if overheat < _safe_float(p.get("overheat_long_max", 1.0), 1.0) else 0.0
    short_boost = _safe_float(p.get("short_boost", 1.0), 1.0) if overheat > _safe_float(p.get("overheat_short_min", 1.0), 1.0) else 1.0
    base_long = float(1.0 / (1.0 + np.exp(-np.clip((raw - bias) / temp, -40.0, 40.0))))
    base_short = float(1.0 / (1.0 + np.exp(-np.clip((-raw - bias) / temp, -40.0, 40.0))))
    tail_pen = float(np.clip(1.0 - (_safe_float(p.get("tail_tox", 0.0)) * tox + _safe_float(p.get("tail_qc", 0.0)) * qcol + _safe_float(p.get("tail_aft", 0.0)) * aft), 0.0, 1.0))
    ls = float(base_long * long_gate * tail_pen)
    ss = float(base_short * short_boost * tail_pen)

    atr_min = _safe_float(p.get("atr_min", 0.0008), 0.0008)
    # 30분 변동성(pv30)이 준비되지 않은 재시작 초기 구간에서는
    # WS가 정상(LIVE)이고 최소 가격 샘플(5분) 확보 시 soft-score 투표를 허용한다.
    tradable = bool(pv30 >= atr_min * 0.5) or bool((pv30 <= 0.0) and (not data_stale) and (warmup_price_samples >= 5))
    entry = _safe_float(p.get("entry", 0.7), 0.7)
    fire = float(np.clip(entry * 0.85, 0.0, 1.0))
    pos_u = str(current_pos or "NONE").upper()
    long_th = fire if pos_u == "LONG" else entry
    short_th = fire if pos_u == "SHORT" else entry
    if not tradable:
        return 0, ls, ss
    needed = long_th if ls >= ss else short_th
    if max(ls, ss) < needed:
        return 0, ls, ss
    if abs(ls - ss) < 0.05:
        return 0, ls, ss
    if ls > ss and ls >= long_th:
        return 1, ls, ss
    if ss > ls and ss >= short_th:
        return 2, ls, ss
    return 0, ls, ss


def _ensemble_vote_runtime(params: list[dict], min_votes: int, ms: dict, tr: dict, base_kelly: float, veto_on: bool, tag: str, current_pos: str = "NONE") -> dict:
    if not params:
        return {
            "decision": "HOLD",
            "action": 0,
            "confidence_score": 0,
            "kelly_weight": 0.0,
            "reason": f"{tag}_NO_PARAMS",
            "votes_long": 0,
            "votes_short": 0,
            "votes_hold": 0,
            "pool_k": 0,
            "required_votes": int(min_votes),
        }
    votes_l = 0
    votes_s = 0
    votes_h = 0
    ls_sum = 0.0
    ss_sum = 0.0
    blocked_by_tradable = 0
    blocked_by_fire = 0
    for p in params:
        a, ls, ss = _param_vote_from_snapshot(p, ms, tr, current_pos=current_pos)
        ls_sum += float(ls)
        ss_sum += float(ss)
        atr_min = _safe_float(p.get("atr_min", 0.0008), 0.0008)
        entry = _safe_float(p.get("entry", 0.7), 0.7)
        fire = float(np.clip(entry * 0.85, 0.0, 1.0))
        pos_u = str(current_pos or "NONE").upper()
        long_th = fire if pos_u == "LONG" else entry
        short_th = fire if pos_u == "SHORT" else entry
        pv30 = abs(_safe_float(ms.get("price_volatility_30m", 0.0), 0.0))
        warmup_price_samples = int(_safe_float(ms.get("warmup_price_samples", 0.0), 0.0))
        data_stale = bool(ms.get("data_stale", False))
        tradable = bool(pv30 >= atr_min * 0.5) or bool((pv30 <= 0.0) and (not data_stale) and (warmup_price_samples >= 5))
        if not tradable:
            blocked_by_tradable += 1
        else:
            needed = long_th if ls >= ss else short_th
            if max(ls, ss) < needed or abs(ls - ss) < 0.05:
                blocked_by_fire += 1
        if a == 1:
            votes_l += 1
        elif a == 2:
            votes_s += 1
        else:
            votes_h += 1

    k = len(params)
    if veto_on:
        action = 0
        reason = f"{tag}_VETO_SHIELD_ACTIVE"
    elif votes_l >= int(min_votes) and votes_l > votes_s:
        action = 1
        reason = f"{tag}_VOTE_LONG"
    elif votes_s >= int(min_votes) and votes_s > votes_l:
        action = 2
        reason = f"{tag}_VOTE_SHORT"
    else:
        action = 0
        reason = f"{tag}_VOTE_INSUFFICIENT"

    win_votes = votes_l if action == 1 else (votes_s if action == 2 else max(votes_l, votes_s))
    conf = int(max(0.0, min(100.0, 100.0 * (float(win_votes) / max(k, 1)))))
    kelly = 0.0 if action == 0 else float(min(1.0, max(0.0, float(base_kelly) * (0.40 + 0.60 * (float(win_votes) / max(k, 1))))))

    return {
        "decision": _action_text(action),
        "action": int(action),
        "confidence_score": int(conf),
        "kelly_weight": float(kelly),
        "reason": str(reason),
        "votes_long": int(votes_l),
        "votes_short": int(votes_s),
        "votes_hold": int(votes_h),
        "pool_k": int(k),
        "required_votes": int(min_votes),
        "long_score_avg": float(ls_sum / max(k, 1)),
        "short_score_avg": float(ss_sum / max(k, 1)),
        "blocked_by_tradable": int(blocked_by_tradable),
        "blocked_by_fire": int(blocked_by_fire),
    }


def _build_ensemble_runtime(pb_list: list[dict], base_action: int, base_kelly: float, ms: dict | None = None, tr: dict | None = None) -> dict:
    now_iso = _now_kst_iso()
    ms = dict(ms or {})
    _update_overheat_score(ms)
    tr = dict(tr or {})
    _trk_state = _load_ensemble_tracker_state()
    bal_pos = str(((_trk_state.get("balanced", {}) or {}).get("pos", "NONE"))).upper()
    low_pos = str(((_trk_state.get("lowfreq", {}) or {}).get("pos", "NONE"))).upper()
    by = {}
    for x in (pb_list or []):
        n = str((x or {}).get("name", ""))
        if n:
            by[n] = dict(x or {})

    veto = by.get("PB_VETO_SHIELD")
    # 소프트 점수만으로 강제 HOLD하지 않고, 실제 매칭 시에만 VETO를 건다.
    veto_on = bool((veto or {}).get("matched", False))
    static_cards = _load_ensemble_cards()
    bal_params = _load_param_pool(ENSEMBLE_BALANCED_PARAMS_PATH, "top_params")
    low_params = _load_param_pool(ENSEMBLE_LOWFREQ_PARAMS_PATH, "top10_singles")
    bal_min_votes = int((_read_json_safe(ENSEMBLE_BALANCED_PARAMS_PATH).get("search", {}) or {}).get("min_votes", 6) or 6)
    low_min_votes = int((_read_json_safe(ENSEMBLE_LOWFREQ_METRICS_PATH).get("best", {}) or {}).get("votes", 7) or 7)
    balanced_live = _ensemble_vote_runtime(
        params=bal_params[:10],
        min_votes=bal_min_votes,
        ms=ms,
        tr=tr,
        base_kelly=base_kelly,
        veto_on=veto_on,
        tag="BALANCED",
        current_pos=bal_pos,
    )
    lowfreq_live = _ensemble_vote_runtime(
        params=(low_params[:10] if low_params else bal_params[:10]),
        min_votes=low_min_votes,
        ms=ms,
        tr=tr,
        base_kelly=base_kelly,
        veto_on=veto_on,
        tag="LOWFREQ",
        current_pos=low_pos,
    )
    balanced_live["updated_at"] = str(now_iso)
    lowfreq_live["updated_at"] = str(now_iso)
    return {
        "updated_at": now_iso,
        "balanced": {
            **dict(static_cards.get("balanced", {}) or {}),
            "live": balanced_live,
        },
        "lowfreq": {
            **dict(static_cards.get("lowfreq", {}) or {}),
            "live": lowfreq_live,
        },
    }


def _default_tracker_state() -> dict:
    now = _now_kst_iso()
    return {
        "balanced": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
        "lowfreq": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
    }


def _load_ensemble_tracker_state() -> dict:
    # records를 진실 소스로 사용: records가 비면 누적도 즉시 초기화
    st = _default_tracker_state()
    try:
        if not os.path.exists(ENSEMBLE_TRACKER_RECORDS_PATH):
            return st
        if os.path.getsize(ENSEMBLE_TRACKER_RECORDS_PATH) <= 0:
            return st
        with open(ENSEMBLE_TRACKER_RECORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                k = str(row.get("ensemble", ""))
                if k not in ("balanced", "lowfreq"):
                    continue
                n = st[k]
                ev = str(row.get("event", "")).upper()
                ts = str(row.get("ts", n.get("updated_at", _now_kst_iso())))
                if ev == "OPEN":
                    n["pos"] = str(row.get("side", "NONE")).upper()
                    n["entry_price"] = _safe_float(row.get("entry_price", 0.0), 0.0)
                    n["entry_kelly"] = _safe_float(row.get("entry_kelly", n.get("entry_kelly", 0.0)), 0.0)
                    n["unrealized_pnl_pct"] = 0.0
                    n["opened_at"] = ts
                    n["updated_at"] = ts
                elif ev == "CLOSE":
                    pnl_pct = _safe_float(row.get("pnl_pct", 0.0), 0.0)
                    eq_row = row.get("equity", None)
                    if eq_row is not None:
                        n["equity"] = _safe_float(eq_row, n.get("equity", 1.0))
                    else:
                        n["equity"] = float(n.get("equity", 1.0)) * (1.0 + pnl_pct / 100.0)
                    peak_eq = float(n.get("peak_equity", 1.0) or 1.0)
                    cur_eq = float(n.get("equity", 1.0) or 1.0)
                    peak_eq = max(peak_eq, cur_eq, 1e-12)
                    dd_pct = float((1.0 - (cur_eq / peak_eq)) * 100.0)
                    n["peak_equity"] = float(peak_eq)
                    n["mdd_pct"] = float(max(_safe_float(n.get("mdd_pct", 0.0), 0.0), dd_pct))
                    n["trades"] = int(n.get("trades", 0) or 0) + 1
                    if pnl_pct > 0.0:
                        n["wins"] = int(n.get("wins", 0) or 0) + 1
                    n["last_pnl_pct"] = float(pnl_pct)
                    n["pos"] = "NONE"
                    n["entry_price"] = 0.0
                    n["entry_kelly"] = 0.0
                    n["opened_at"] = ""
                    n["unrealized_pnl_pct"] = 0.0
                    n["updated_at"] = ts
    except Exception:
        # records 파싱 실패 시 기존 state 파일 fallback
        raw = _read_json_safe(ENSEMBLE_TRACKER_STATE_PATH)
        if raw:
            for k in ("balanced", "lowfreq"):
                node = dict(raw.get(k, {}) or {})
                if node:
                    st[k].update(node)
    return st


def _save_ensemble_tracker_state(state: dict) -> None:
    _atomic_write_json(ENSEMBLE_TRACKER_STATE_PATH, state)


def _close_tracker_trade(node: dict, name: str, now_iso: str, price: float) -> None:
    pos = str(node.get("pos", "NONE"))
    if pos not in ("LONG", "SHORT"):
        return
    entry = float(node.get("entry_price", 0.0) or 0.0)
    if entry <= 0.0 or price <= 0.0:
        node["pos"] = "NONE"
        node["entry_price"] = 0.0
        return

    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    fee = float(ENSEMBLE_TRACKER_FEE_RATE)
    exit_px = price * (1.0 - slip if pos == "LONG" else 1.0 + slip)
    rr = (exit_px - entry) / max(entry, 1e-12)
    if pos == "SHORT":
        rr = -rr
    pnl_frac = float(rr - (2.0 * fee))

    eq = float(node.get("equity", 1.0) or 1.0)
    eq *= (1.0 + pnl_frac)
    node["equity"] = float(eq)
    peak_eq = float(node.get("peak_equity", 1.0) or 1.0)
    peak_eq = max(peak_eq, eq, 1e-12)
    dd_pct = float((1.0 - (eq / peak_eq)) * 100.0)
    node["peak_equity"] = float(peak_eq)
    node["mdd_pct"] = float(max(_safe_float(node.get("mdd_pct", 0.0), 0.0), dd_pct))
    node["trades"] = int(node.get("trades", 0) or 0) + 1
    if pnl_frac > 0:
        node["wins"] = int(node.get("wins", 0) or 0) + 1
    node["last_pnl_pct"] = float(pnl_frac * 100.0)
    node["updated_at"] = now_iso

    _append_jsonl(ENSEMBLE_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "ensemble": name,
        "event": "CLOSE",
        "side": pos,
        "entry_price": entry,
        "exit_price": float(exit_px),
        "pnl_pct": float(pnl_frac * 100.0),
        "equity": float(eq),
    })
    node["pos"] = "NONE"
    node["entry_price"] = 0.0
    node["entry_kelly"] = 0.0
    node["opened_at"] = ""
    node["unrealized_pnl_pct"] = 0.0


def _open_tracker_trade(node: dict, name: str, now_iso: str, price: float, action: int, kelly: float = 0.0) -> None:
    if price <= 0.0 or int(action) not in (1, 2):
        return
    side = "LONG" if int(action) == 1 else "SHORT"
    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    entry = price * (1.0 + slip if side == "LONG" else 1.0 - slip)
    node["pos"] = side
    node["entry_price"] = float(entry)
    node["entry_kelly"] = float(max(0.0, kelly))
    node["unrealized_pnl_pct"] = 0.0
    node["opened_at"] = str(now_iso)
    node["updated_at"] = now_iso
    _append_jsonl(ENSEMBLE_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "ensemble": name,
        "event": "OPEN",
        "side": side,
        "entry_price": float(entry),
        "entry_kelly": float(max(0.0, kelly)),
    })


def _update_ensemble_tracker(ensembles: dict, current_price: float, now_iso: str) -> dict:
    st = _load_ensemble_tracker_state()
    price = float(current_price or 0.0)

    for key in ("balanced", "lowfreq"):
        node = dict(st.get(key, {}) or {})
        live = dict((ensembles.get(key, {}) or {}).get("live", {}) or {})
        action = int(live.get("action", 0) or 0)
        live_kelly = _safe_float(live.get("kelly_weight", 0.0), 0.0)
        cur_pos = str(node.get("pos", "NONE"))

        # 과거 포맷(OPEN에 entry_kelly 없음)로 열린 포지션 복구:
        # 포지션이 살아있고 entry_kelly가 비어있으면 현재 live_kelly를 1회 백필
        if cur_pos in ("LONG", "SHORT") and _safe_float(node.get("entry_kelly", 0.0), 0.0) <= 0.0 and live_kelly > 0.0:
            node["entry_kelly"] = float(live_kelly)
            node["updated_at"] = now_iso

        if cur_pos in ("LONG", "SHORT"):
            entry = float(node.get("entry_price", 0.0) or 0.0)
            if price > 0.0 and entry > 0.0:
                mark_px = price * (1.0 - ENSEMBLE_TRACKER_SLIP_RATE if cur_pos == "LONG" else 1.0 + ENSEMBLE_TRACKER_SLIP_RATE)
                rr = (mark_px - entry) / max(entry, 1e-12)
                if cur_pos == "SHORT":
                    rr = -rr
                node["unrealized_pnl_pct"] = float((rr - (2.0 * ENSEMBLE_TRACKER_FEE_RATE)) * 100.0)
            else:
                node["unrealized_pnl_pct"] = 0.0

            pos_action = 1 if cur_pos == "LONG" else 2
            should_close = False
            if action in (1, 2) and action != pos_action:
                # 명시적 반대 시그널일 때만 청산/전환
                should_close = True
            elif action == 0 and ENSEMBLE_TRACKER_EXIT_ON_HOLD:
                # 옵션: HOLD 시 즉시 청산(기본 OFF)
                should_close = True
            if should_close:
                _close_tracker_trade(node=node, name=key, now_iso=now_iso, price=price)
                if action in (1, 2):
                    _open_tracker_trade(node=node, name=key, now_iso=now_iso, price=price, action=action, kelly=live_kelly)
        else:
            node["unrealized_pnl_pct"] = 0.0
            if action in (1, 2):
                _open_tracker_trade(node=node, name=key, now_iso=now_iso, price=price, action=action, kelly=live_kelly)

        st[key] = node

    _save_ensemble_tracker_state(st)
    return st


def _ensemble_tracker_summary(tracker_state: dict) -> dict:
    out = {}
    for key in ("balanced", "lowfreq"):
        n = dict((tracker_state or {}).get(key, {}) or {})
        eq = float(n.get("equity", 1.0) or 1.0)
        tr = int(n.get("trades", 0) or 0)
        wins = int(n.get("wins", 0) or 0)
        wr = (100.0 * wins / tr) if tr > 0 else 0.0
        out[key] = {
            "total_return_pct": float((eq - 1.0) * 100.0),
            "trades": tr,
            "win_rate": float(wr),
            "pos": str(n.get("pos", "NONE")),
            "entry_price": float(n.get("entry_price", 0.0) or 0.0),
            "entry_kelly": float(n.get("entry_kelly", 0.0) or 0.0),
            "opened_at": str(n.get("opened_at", "")),
            "unrealized_pnl_pct": float(n.get("unrealized_pnl_pct", 0.0) or 0.0),
            "last_pnl_pct": float(n.get("last_pnl_pct", 0.0) or 0.0),
            "mdd_pct": float(n.get("mdd_pct", 0.0) or 0.0),
            "updated_at": str(n.get("updated_at", _now_kst_iso())),
        }
    return out


def _default_agent_tracker_state() -> dict:
    now = _now_kst_iso()
    return {
        "long": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
        "short": {
            "pos": "NONE",
            "entry_price": 0.0,
            "entry_kelly": 0.0,
            "opened_at": "",
            "unrealized_pnl_pct": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "mdd_pct": 0.0,
            "trades": 0,
            "wins": 0,
            "last_pnl_pct": 0.0,
            "updated_at": now,
        },
    }


def _load_agent_tracker_state() -> dict:
    st = _default_agent_tracker_state()
    try:
        if not os.path.exists(AGENT_TRACKER_RECORDS_PATH):
            return st
        if os.path.getsize(AGENT_TRACKER_RECORDS_PATH) <= 0:
            return st
        with open(AGENT_TRACKER_RECORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                k = str(row.get("agent", "")).lower()
                if k not in ("long", "short"):
                    continue
                n = st[k]
                ev = str(row.get("event", "")).upper()
                ts = str(row.get("ts", n.get("updated_at", _now_kst_iso())))
                if ev == "OPEN":
                    n["pos"] = str(row.get("side", "NONE")).upper()
                    n["entry_price"] = _safe_float(row.get("entry_price", 0.0), 0.0)
                    n["entry_kelly"] = _safe_float(row.get("entry_kelly", n.get("entry_kelly", 0.0)), 0.0)
                    n["unrealized_pnl_pct"] = 0.0
                    n["opened_at"] = ts
                    n["updated_at"] = ts
                elif ev == "CLOSE":
                    pnl_pct = _safe_float(row.get("pnl_pct", 0.0), 0.0)
                    eq_row = row.get("equity", None)
                    if eq_row is not None:
                        n["equity"] = _safe_float(eq_row, n.get("equity", 1.0))
                    else:
                        n["equity"] = float(n.get("equity", 1.0)) * (1.0 + pnl_pct / 100.0)
                    peak_eq = float(n.get("peak_equity", 1.0) or 1.0)
                    cur_eq = float(n.get("equity", 1.0) or 1.0)
                    peak_eq = max(peak_eq, cur_eq, 1e-12)
                    dd_pct = float((1.0 - (cur_eq / peak_eq)) * 100.0)
                    n["peak_equity"] = float(peak_eq)
                    n["mdd_pct"] = float(max(_safe_float(n.get("mdd_pct", 0.0), 0.0), dd_pct))
                    n["trades"] = int(n.get("trades", 0) or 0) + 1
                    if pnl_pct > 0.0:
                        n["wins"] = int(n.get("wins", 0) or 0) + 1
                    n["last_pnl_pct"] = float(pnl_pct)
                    n["pos"] = "NONE"
                    n["entry_price"] = 0.0
                    n["entry_kelly"] = 0.0
                    n["opened_at"] = ""
                    n["unrealized_pnl_pct"] = 0.0
                    n["updated_at"] = ts
    except Exception:
        raw = _read_json_safe(AGENT_TRACKER_STATE_PATH)
        if raw:
            for k in ("long", "short"):
                node = dict(raw.get(k, {}) or {})
                if node:
                    st[k].update(node)
    return st


def _save_agent_tracker_state(state: dict) -> None:
    _atomic_write_json(AGENT_TRACKER_STATE_PATH, state)


def _close_agent_trade(node: dict, name: str, now_iso: str, price: float) -> None:
    pos = str(node.get("pos", "NONE"))
    if pos not in ("LONG", "SHORT"):
        return
    entry = float(node.get("entry_price", 0.0) or 0.0)
    if entry <= 0.0 or price <= 0.0:
        node["pos"] = "NONE"
        node["entry_price"] = 0.0
        return

    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    fee = float(ENSEMBLE_TRACKER_FEE_RATE)
    exit_px = price * (1.0 - slip if pos == "LONG" else 1.0 + slip)
    rr = (exit_px - entry) / max(entry, 1e-12)
    if pos == "SHORT":
        rr = -rr
    pnl_frac = float(rr - (2.0 * fee))

    eq = float(node.get("equity", 1.0) or 1.0)
    eq *= (1.0 + pnl_frac)
    node["equity"] = float(eq)
    peak_eq = float(node.get("peak_equity", 1.0) or 1.0)
    peak_eq = max(peak_eq, eq, 1e-12)
    dd_pct = float((1.0 - (eq / peak_eq)) * 100.0)
    node["peak_equity"] = float(peak_eq)
    node["mdd_pct"] = float(max(_safe_float(node.get("mdd_pct", 0.0), 0.0), dd_pct))
    node["trades"] = int(node.get("trades", 0) or 0) + 1
    if pnl_frac > 0:
        node["wins"] = int(node.get("wins", 0) or 0) + 1
    node["last_pnl_pct"] = float(pnl_frac * 100.0)
    node["updated_at"] = now_iso

    _append_jsonl(AGENT_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "agent": name,
        "event": "CLOSE",
        "side": pos,
        "entry_price": entry,
        "exit_price": float(exit_px),
        "pnl_pct": float(pnl_frac * 100.0),
        "equity": float(eq),
    })
    node["pos"] = "NONE"
    node["entry_price"] = 0.0
    node["entry_kelly"] = 0.0
    node["opened_at"] = ""
    node["unrealized_pnl_pct"] = 0.0


def _open_agent_trade(node: dict, name: str, now_iso: str, price: float, action: int, kelly: float = 0.0) -> None:
    if price <= 0.0 or int(action) not in (1, 2):
        return
    side = "LONG" if int(action) == 1 else "SHORT"
    slip = float(ENSEMBLE_TRACKER_SLIP_RATE)
    entry = price * (1.0 + slip if side == "LONG" else 1.0 - slip)
    node["pos"] = side
    node["entry_price"] = float(entry)
    node["entry_kelly"] = float(max(0.0, kelly))
    node["unrealized_pnl_pct"] = 0.0
    node["opened_at"] = str(now_iso)
    node["updated_at"] = now_iso
    _append_jsonl(AGENT_TRACKER_RECORDS_PATH, {
        "ts": now_iso,
        "agent": name,
        "event": "OPEN",
        "side": side,
        "entry_price": float(entry),
        "entry_kelly": float(max(0.0, kelly)),
    })


def _update_agent_tracker(agent_actions: dict, current_price: float, now_iso: str) -> dict:
    st = _load_agent_tracker_state()
    price = float(current_price or 0.0)
    specs = (
        ("long", 1, "LONG"),
        ("short", 2, "SHORT"),
    )

    for key, expect_action, expect_side in specs:
        node = dict(st.get(key, {}) or {})
        live = dict(agent_actions.get(key, {}) or {})
        action = int(live.get("action", 0) or 0)
        live_kelly = _safe_float(live.get("kelly_weight", 0.0), 0.0)
        cur_pos = str(node.get("pos", "NONE"))

        if cur_pos in ("LONG", "SHORT") and _safe_float(node.get("entry_kelly", 0.0), 0.0) <= 0.0 and live_kelly > 0.0:
            node["entry_kelly"] = float(live_kelly)
            node["updated_at"] = now_iso

        if cur_pos == expect_side:
            entry = float(node.get("entry_price", 0.0) or 0.0)
            if price > 0.0 and entry > 0.0:
                mark_px = price * (1.0 - ENSEMBLE_TRACKER_SLIP_RATE if cur_pos == "LONG" else 1.0 + ENSEMBLE_TRACKER_SLIP_RATE)
                rr = (mark_px - entry) / max(entry, 1e-12)
                if cur_pos == "SHORT":
                    rr = -rr
                node["unrealized_pnl_pct"] = float((rr - (2.0 * ENSEMBLE_TRACKER_FEE_RATE)) * 100.0)
            else:
                node["unrealized_pnl_pct"] = 0.0
            if action != expect_action:
                _close_agent_trade(node=node, name=key, now_iso=now_iso, price=price)
        elif cur_pos in ("LONG", "SHORT"):
            node["unrealized_pnl_pct"] = 0.0
            _close_agent_trade(node=node, name=key, now_iso=now_iso, price=price)
            if action == expect_action:
                _open_agent_trade(node=node, name=key, now_iso=now_iso, price=price, action=expect_action, kelly=live_kelly)
        elif action == expect_action:
            node["unrealized_pnl_pct"] = 0.0
            _open_agent_trade(node=node, name=key, now_iso=now_iso, price=price, action=expect_action, kelly=live_kelly)
        else:
            node["unrealized_pnl_pct"] = 0.0

        st[key] = node

    _save_agent_tracker_state(st)
    return st


def _agent_tracker_summary(tracker_state: dict) -> dict:
    out = {}
    for key in ("long", "short"):
        n = dict((tracker_state or {}).get(key, {}) or {})
        eq = float(n.get("equity", 1.0) or 1.0)
        tr = int(n.get("trades", 0) or 0)
        wins = int(n.get("wins", 0) or 0)
        wr = (100.0 * wins / tr) if tr > 0 else 0.0
        out[key] = {
            "total_return_pct": float((eq - 1.0) * 100.0),
            "trades": tr,
            "win_rate": float(wr),
            "pos": str(n.get("pos", "NONE")),
            "entry_price": float(n.get("entry_price", 0.0) or 0.0),
            "entry_kelly": float(n.get("entry_kelly", 0.0) or 0.0),
            "opened_at": str(n.get("opened_at", "")),
            "unrealized_pnl_pct": float(n.get("unrealized_pnl_pct", 0.0) or 0.0),
            "last_pnl_pct": float(n.get("last_pnl_pct", 0.0) or 0.0),
            "mdd_pct": float(n.get("mdd_pct", 0.0) or 0.0),
            "updated_at": str(n.get("updated_at", _now_kst_iso())),
        }
    return out


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
        self.ancillary_period = os.getenv("BINANCE_ANCILLARY_PERIOD", "5m")
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
            self.exchange.fapiDataGetOpenInterestHist({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortAccountRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortPositionRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetGlobalLongShortAccountRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
            self.exchange.fapiDataGetTakerlongshortRatio({'symbol': self.symbol, 'period': self.ancillary_period, 'limit': limit}),
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
                (4, 'buySellRatio', 'taker_long_short_ratio'),
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
                    c_val = _extract_last_conf(res)
                    conf_src = "model"
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
# 2-D. 텔레그램 알림
# ════════════════════════════════════════════════════════════════
class TelegramNotifier:
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
            logger.info("📨 텔레그램 전송 완료")
        except Exception as e:
            logger.warning(f"⚠️ 텔레그램 전송 예외: {e}")

    async def notify(self, text: str) -> None:
        if not self._ok:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_send, text)


def _tg_trade_msg(ex_code: str, current_price: float,
                  timestamp_kst, regime_name: str, meta_result: dict) -> str:
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
            pass
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
        if cur_pos is None: return 'STAY FLAT'
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
        for name, cal_name in (("session_asia", "JPX"), ("session_europe", "LSE"), ("session_us", "NYSE")):
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

    def _action_word(a: int) -> str: return {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(int(a), 'UNKNOWN')
    def _action_color(a: int) -> str: return {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(int(a), C.RESET)
    def _bar(v: float, w: int = 8) -> str:
        x = float(np.clip(v, 0.0, 1.0))
        n = int(round(x * w))
        return '█' * n + '░' * (w - n)
    def _trend_word(tdir: int) -> str: return {0: 'DOWN', 1: 'FLAT', 2: 'UP'}.get(int(tdir), 'UNKNOWN')
    def _trend_color(tdir: int) -> str: return {0: C.RED, 1: C.YELLOW, 2: C.GREEN}.get(int(tdir), C.RESET)
    def _kelly_text(v: float) -> str:
        if v >= 0.70: return "강함"
        if v >= 0.40: return "보통"
        if v >= 0.15: return "약함"
        return "매우약함"
    def _conviction_text(v: float) -> str:
        if v >= 1.0: return "진입 강함"
        if v >= 0.60: return "진입 가능"
        if v >= 0.30: return "진입 약함"
        return "진입 부족"
    def _agreement_text(v: float) -> str:
        if v >= 1.5: return "방향 우위 뚜렷"
        if v >= 0.8: return "방향 우위 있음"
        if v >= 0.4: return "방향 우위 약함"
        return "방향 혼재"
    def _ambiguity_text(v: float) -> str:
        if v >= 2.0: return "양방향 충돌 큼"
        if v >= 1.0: return "양방향 경합"
        if v >= 0.0: return "약한 경합"
        return "방향 분리 양호"
    def _hibernation_text(v: float) -> str:
        if v >= 0.85: return "시장 과열/이상"
        if v >= 0.60: return "이상치 주의"
        if v >= 0.30: return "약한 이상 신호"
        return "정상 범위"
    def _amihud_text(v: float) -> str:
        if v >= 1.5: return "유동성 매우 나쁨"
        if v >= 0.8: return "유동성 나쁨"
        if v >= 0.2: return "유동성 보통"
        return "유동성 양호"
    def _gate(ok: bool, label: str, detail: str = "") -> str:
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        text = label + (f"/{detail}" if detail else "")
        return f"{col}{text}[{icon}]{C.RESET}"
    def _status_badge(ok: bool, ok_label: str = "PASS", fail_label: str = "FAIL") -> str:
        icon = "✓" if ok else "✗"
        col = C.GREEN if ok else C.RED
        label = ok_label if ok else fail_label
        return f"{col}[{label} {icon}]{C.RESET}"

    def _exec_code(pp: str | None, cp: str | None) -> tuple[str, str]:
        if pp == cp:
            if cp is None: return '·', 'STAY_FLAT'
            return '↔', 'HOLD_LONG' if cp == 'LONG' else 'HOLD_SHORT'
        if pp is None and cp == 'LONG': return '↗', 'ENTER_LONG'
        if pp is None and cp == 'SHORT': return '↘', 'ENTER_SHORT'
        if pp == 'LONG' and cp is None: return '✕', 'EXIT_LONG'
        if pp == 'SHORT' and cp is None: return '✕', 'EXIT_SHORT'
        if pp == 'LONG' and cp == 'SHORT': return '⇄', 'FLIP_LONG_TO_SHORT'
        if pp == 'SHORT' and cp == 'LONG': return '⇄', 'FLIP_SHORT_TO_LONG'
        return '·', _pos_transition_label(pp, cp)

    long_edge = float(rl_info.get('long_edge', 0.0))
    short_edge = float(rl_info.get('short_edge', 0.0))
    primary_action = int(rl_info.get("primary_action", 0))
    primary_raw = float(rl_info.get("primary_raw", 0.0))
    primary_kelly = float(rl_info.get("primary_kelly", 0.0))
    primary_disp_action = int(rl_info.get("primary_model_action", primary_action))
    primary_disp_raw = float(rl_info.get("primary_model_raw", primary_raw))
    primary_disp_kelly = float(rl_info.get("primary_model_kelly", primary_kelly))
    primary_disp_std = float(rl_info.get("primary_model_std", rl_info.get("primary_std", 0.0)))
    target_action = int(rl_info.get("target_action", 0))
    net_score = float(rl_info.get("net_score", 0.0))
    agreement_count = int(rl_info.get("agreement_count", 0))
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
    entry_price_reco = tp_price_reco = sl_price_reco = 0.0
    entry_offset_reco = tp_offset_reco = sl_offset_reco = 0.0
    cb_active = int(meta_result.get("cb_active", 0) or 0) if isinstance(meta_result, dict) else 0
    hibernation_score = float(meta_result.get("hibernation_score", 0.0)) if isinstance(meta_result, dict) else 0.0
    illiq_amihud = float(meta_result.get("illiq_amihud", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_signal = str(meta_result.get("position_signal", "")) if isinstance(meta_result, dict) else ""
    position_reason = str(meta_result.get("position_reason", "")) if isinstance(meta_result, dict) else ""
    position_own_support = float(meta_result.get("position_own_support", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_opp_pressure = float(meta_result.get("position_opp_pressure", 0.0)) if isinstance(meta_result, dict) else 0.0
    position_net_edge = float(meta_result.get("position_net_edge", 0.0)) if isinstance(meta_result, dict) else 0.0
    hold_reason = str(meta_result.get("hold_reason", "")) if isinstance(meta_result, dict) else ""
    block_reason = str(meta_result.get("block_reason", "")) if isinstance(meta_result, dict) else ""
    router_enter_threshold = float(meta_result.get("router_enter_threshold", 0.15)) if isinstance(meta_result, dict) else 0.15
    router_min_agreement_threshold = float(meta_result.get("router_min_agreement_threshold", 0.0)) if isinstance(meta_result, dict) else 0.0
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
            p_dn, p_fl, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        p_dn = float(ts.get('prob_dn', ts.get('p_down', p_dn)))
        p_fl = float(ts.get('prob_flat', ts.get('p_flat', p_fl)))
        p_up = float(ts.get('prob_up', ts.get('p_up', p_up)))
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
        edge_side_word, edge_side_color = 'LONG_BIAS', C.GREEN
    elif short_edge > long_edge:
        edge_side_word, edge_side_color = 'SHORT_BIAS', C.RED
    else:
        edge_side_word, edge_side_color = 'NEUTRAL_BIAS', C.YELLOW

    long_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(long_action), '?')
    short_agent_arrow = {0: '─', 1: '▲', 2: '▼'}.get(int(short_action), '?')

    rl_word, rl_color = _action_word(rl_action), _action_color(rl_action)
    final_word, final_color = _action_word(fa), _action_color(fa)
    trend_word, trend_color = _trend_word(t_dir), _trend_color(t_dir)
    W = 62
    _SEP  = "─" * W
    _SEP2 = "═" * W

    def _action_arrow(a: int) -> str: return {0: '─', 1: '▲', 2: '▼'}.get(int(a), '?')
    def _trend_arrow(tdir: int) -> str: return {0: '▼', 1: '─', 2: '▲'}.get(int(tdir), '?')

    fa_arrow = _action_arrow(fa)
    rl_arrow = _action_arrow(rl_action)
    trend_arrow = _trend_arrow(t_dir)

    print(_SEP2)
    ts_str = timestamp_kst.strftime('%Y-%m-%d %H:%M')
    session_flags = _session_flags_from_timestamp(timestamp_kst)
    session_parts = []
    for label, key in (("ASIA", "session_asia"), ("EUROPE", "session_europe"), ("US", "session_us")):
        active = float(session_flags.get(key, 0.0)) >= 0.5
        scol, sword = (C.GREEN, "ON") if active else (C.YELLOW, "OFF")
        session_parts.append(f"{label}={scol}{sword}{C.RESET}")
    header_left = f"{final_color}{C.BOLD}{fa_arrow}{fa_arrow}  {final_word}  →  {ex_code}{C.RESET}"
    print(f" {header_left}  {C.CYAN}{ts_str}  ${current_price:,.2f}{C.RESET}")
    print(f"     {C.CYAN}{regime_name}{C.RESET}  {'  '.join(session_parts)}")
    print(_SEP)

    print(f"  {rl_color}{rl_arrow} 신호{C.RESET}  {rl_color}{rl_word:<6}{C.RESET}"
          f" {edge_side_color}{edge_side_word} {edge_gap:+.3f}{C.RESET}"
          f"  Kelly: {_bar(final_kelly, 8)} {final_kelly:.3f} ({_kelly_text(final_kelly)})")
    print(f"  {C.CYAN}• DSAC{C.RESET}  "
          f"L:{long_agent_arrow}{_action_word(long_action):<5} r={C.GREEN}{long_raw:.3f}{C.RESET} k={long_kelly:.3f}"
          f"  S:{short_agent_arrow}{_action_word(short_action):<5} r={C.RED}{short_raw:.3f}{C.RESET} k={short_kelly:.3f}")
    print(f"  {C.CYAN}• Primary(DIRECT){C.RESET} "
          f"{_action_arrow(primary_disp_action)}{_action_word(primary_disp_action):<5}"
          f" raw={primary_disp_raw:+.3f} k={primary_disp_kelly:.3f} std={primary_disp_std:.3f}"
          f"  → target={_action_word(target_action):<5}"
          f" net={net_score:+.3f} votes={agreement_count}")
    print(f"          → 결정 = {selected_side:<6}"
          f"  conv={conviction:.3f} ({_conviction_text(conviction)})"
          f"  agr={agreement:.3f} ({_agreement_text(agreement)})")
    print(f"  {C.CYAN}• 점수{C.RESET}  "
          f"L={C.GREEN}{long_logit:+.2f}{C.RESET}(±{long_std:.2f})"
          f"  S={C.RED}{short_logit:+.2f}{C.RESET}(±{short_std:.2f})"
          f"  amb={ambiguity:+.2f} ({_ambiguity_text(ambiguity)})"
          f"  conf={confidence:.3f}")

    dn_c, up_c = (C.RED if p_dn > 0.4 else C.RESET), (C.GREEN if p_up > 0.4 else C.RESET)
    trend_model = str(ts.get("trend_model", "N/A")) if isinstance(ts, dict) else "N/A"
    print(f"  {trend_color}{trend_arrow} 추세{C.RESET}    {trend_color}{trend_word:<6}{C.RESET}"
          f"  str={t_strength:.2f}  rev={t_rev:.2f}"
          f"  {dn_c}DN={p_dn:.0%}{C.RESET} FL={C.YELLOW}{p_fl:.0%}{C.RESET} {up_c}UP={p_up:.0%}{C.RESET}"
          f"  [{trend_model}]")
    if entry_price_reco > 0.0 or tp_price_reco > 0.0 or sl_price_reco > 0.0:
        print(f"  {C.CYAN}• 가격{C.RESET}    진입={entry_price_reco:,.2f}({entry_offset_reco:+.3%})"
              f"  TP={tp_price_reco:,.2f}({tp_offset_reco:+.3%})"
              f"  SL={sl_price_reco:,.2f}({sl_offset_reco:+.3%})")

    print(f"  {C.CYAN}• 보호{C.RESET}    hib={hibernation_score:.2f} ({_hibernation_text(hibernation_score)})"
          f"  cb={cb_active}  amihud={illiq_amihud:.2f} ({_amihud_text(illiq_amihud)})")
    if hold_reason or block_reason:
        print(f"  {C.CYAN}• HOLD{C.RESET}    {C.YELLOW}{hold_reason or '-'}{C.RESET}"
              f"  block={C.RED}{block_reason or '-'}{C.RESET}")

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
        g_conv = _gate(_conv_ok, f"CONV={conviction:.3f}", f"{router_enter_threshold:.3f}")
        g_agr  = _gate(_agr_ok,  f"AGR={agreement:.3f}",  f"{router_min_agreement_threshold:.3f}")
        g_std  = _gate(_std_ok,  f"STD={selected_std:.2f}", f"{router_max_confidence_std:.2f}")
        g_dual = _gate(_dual_ok, f"DUAL={ambiguity:.2f}")
        entry_result = _status_badge(final_word != "HOLD", "PASS", "FAIL")
        print(f"  {C.CYAN}• 진입장벽{C.RESET}  {entry_result}  {g_conv}  {g_agr}  {g_std}  {g_dual}")
        g_hib  = _gate(_hib_ok,  f"HIB={hibernation_score:.2f}", f"{hibernation_score_th:.2f}")
        g_cb   = _gate(_cb_ok,   "CB")
        g_trend = _gate(_trend_ok, "TREND")
        row2 = [g_hib, g_cb, g_trend]
        if not _intg_ok: row2.append(_gate(False, "INTG"))
        if not _cool_ok: row2.append(_gate(False, "COOL"))
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

    print(f"  {C.CYAN}• 소스{C.RESET}    {source}")
    print(_SEP)
    decision_chain = (f"SIGNAL={rl_color}{rl_word}{C.RESET} → "
                      f"추세={trend_color}{trend_word}{C.RESET} → "
                      f"FINAL={final_color}{final_word}{C.RESET} → "
                      f"EXEC={ex_icon} {ex_code}")
    print(f"  {decision_chain}")
    print(_SEP2)


# ════════════════════════════════════════════════════════════════
# 3-A. DSACSignalRouter
# ════════════════════════════════════════════════════════════════
class DSACSignalRouter:
    DEFAULT_SINGLE_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth"
    LEGACY_SINGLE_PATH = "/home/llewyn/crypto-scalping/data/ensemble/ckpt/best_dsac_agent.pth"

    @staticmethod
    def _build_primary_actor_from_ckpt(ckpt: dict, device: str):
        actor_state = ckpt.get("actor")
        if not isinstance(actor_state, dict): raise KeyError("DSAC primary 체크포인트 actor 키 없음")
        state_dim = int(ckpt.get("state_dim", BASE_DSAC_STATE_DIM) or BASE_DSAC_STATE_DIM)
        actor = BaseDSACGaussianActor(state_dim=state_dim).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor, "DSAC_PRIMARY"

    @staticmethod
    def _resolve_model_path(primary: str | None, *fallbacks: str) -> str:
        for candidate in (primary, *fallbacks):
            if candidate and os.path.exists(candidate): return candidate
        searched = [c for c in (primary, *fallbacks) if c]
        raise FileNotFoundError(f"DSAC specialist 체크포인트 파일이 없습니다: {searched}")

    @staticmethod
    def _regime_name(regime: dict[str, float] | None) -> str:
        if not isinstance(regime, dict): return "normal"
        return next((k.replace("regime_", "") for k, v in regime.items() if float(v) == 1.0), "normal")

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "cuda" in msg or "acceleratorerror" in exc.__class__.__name__.lower()

    def __init__(self, model_path: str | None = None, single_path: str | None = None, hmm_detector: OnlineHMMDetector | None = None):
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

        self.enter_th   = float(os.getenv("DSAC_ENTER_TH",   "0.12"))
        self.exit_th    = float(os.getenv("DSAC_EXIT_TH",    "0.12"))
        self.close_th   = float(os.getenv("DSAC_CLOSE_TH",   "0.03"))
        self.max_kelly  = float(os.getenv("DSAC_MAX_KELLY",   "0.35"))

        self.elite_extractor = EliteSignals()
        self.new_elite_engine = NewEliteSignalEngine()

        self.single_ckpt_path = self._resolve_model_path(single_path or model_path, self.DEFAULT_SINGLE_PATH, self.LEGACY_SINGLE_PATH)
        self._load_primary(self.device)

    def _load_primary(self, device: str) -> None:
        ckpt = torch.load(self.single_ckpt_path, map_location=device, weights_only=False)
        actor, ver = self._build_primary_actor_from_ckpt(ckpt, device)
        self.device = device
        self.primary_router = BaseDSACRouter(actor, device=device, hmm_detector=self.hmm)
        logger.info("✅ DSAC Primary 로드 완료 (%s, %s): %s", ver, device, self.single_ckpt_path)

    @staticmethod
    def _require_finite(mapping, key: str, context: str) -> float:
        if key not in mapping: raise ValueError(f"[FEATURE_MISSING] {context}.{key} missing")
        try: val = float(mapping[key])
        except Exception as e: raise ValueError(f"[FEATURE_INVALID] {context}.{key} cast failed: {e}") from e
        if not np.isfinite(val): raise ValueError(f"[FEATURE_INVALID] {context}.{key} is not finite: {mapping[key]}")
        return val

    def decide(self, processed_df: pd.DataFrame, nf_preds: dict, m7_signal: dict | None = None):
        last_row = processed_df.iloc[-1]
        prev_row = processed_df.iloc[-2]
        if "smart_money_flow" not in processed_df.columns: raise KeyError("processed_df missing required column: smart_money_flow")
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
        for col in DSAC_STATE_PRED: features[col] = self._require_finite(nf_preds, col, "nf_preds")
        for col in DSAC_STATE_CONF: features[col] = self._require_finite(nf_preds, col, "nf_preds")
        for col in DSAC_STATE_ELITE: features[col] = self._require_finite(elite_sigs, col, "elite_sigs")
        for col in DSAC_STATE_ALPHA: features[col] = self._require_finite(last_row, col, "last_row")

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
                "regime_chop": 1.0 if hmm_idx == 2 else 0.0,
                "regime_whipsaw": 0.0,
                "regime_normal": 1.0 if hmm_idx == 3 else 0.0,
            }
        if regime is None: regime = _compute_regime(processed_df)
        features.update(regime)
        for col in DSAC_STATE_SYNTH: features[col] = self._require_finite(last_row, col, "last_row")
        features["close"] = self._require_finite(last_row, "close", "last_row")
        for col in ("jump_z", "evt_excess_z", "garch_vol_z", "jump_flag", "evt_tail_flag", "log_return"):
            features[col] = self._require_finite(last_row, col, "last_row")
        _h = self._require_finite(last_row, "high", "last_row")
        _l = self._require_finite(last_row, "low", "last_row")
        _c = features["close"]
        features["current_spread"] = float(np.clip((_h - _l) / max(_c, 1e-8), 0.0, 0.05))

        if not isinstance(m7_signal, dict): raise ValueError("[FEATURE_MISSING] m7_signal unavailable")
        if "m7_prob_dn" in m7_signal:
            features["m7_prob_dn"] = self._require_finite(m7_signal, "m7_prob_dn", "m7_signal")
            features["m7_prob_fl"] = self._require_finite(m7_signal, "m7_prob_fl", "m7_signal")
            features["m7_prob_up"] = self._require_finite(m7_signal, "m7_prob_up", "m7_signal")
        else:
            features["m7_prob_dn"] = self._require_finite(m7_signal, "prob_dn", "m7_signal")
            features["m7_prob_fl"] = self._require_finite(m7_signal, "prob_flat", "m7_signal")
            features["m7_prob_up"] = self._require_finite(m7_signal, "prob_up", "m7_signal")
        for k in M7_LIVE_STRICT_COLS: features[k] = self._require_finite(m7_signal, k, "m7_signal")
        for k in ("m7_tp_offset", "m7_sl_offset"): features[k] = self._require_finite(m7_signal, k, "m7_signal")

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
        except Exception as e:
            if self.device == "cuda" and self._is_cuda_runtime_error(e):
                logger.warning("⚠️ DSAC Primary CUDA 추론 실패, CPU로 폴백합니다: %s", e)
                self._load_primary("cpu")
                primary_action, primary_lev, primary_info = self.primary_router.decide(features, pos_dict)
            else:
                raise

        primary_raw = float((primary_info or {}).get("raw_action", 0.0))
        kelly = float(np.clip(float(primary_lev), 0.0, self.max_kelly))

        if self.pos == "LONG":
            if abs(primary_raw) < self.close_th or primary_raw < -self.exit_th:
                action, final_kelly, pos_signal = 0, 0.0, "EXIT"
            else:
                action, final_kelly, pos_signal = 1, kelly, "HOLD"
        elif self.pos == "SHORT":
            if abs(primary_raw) < self.close_th or primary_raw > self.exit_th:
                action, final_kelly, pos_signal = 0, 0.0, "EXIT"
            else:
                action, final_kelly, pos_signal = 2, kelly, "HOLD"
        else:
            if primary_raw > self.enter_th:
                action, final_kelly, pos_signal = 1, kelly, "LONG_ENTRY"
            elif primary_raw < -self.enter_th:
                action, final_kelly, pos_signal = 2, kelly, "SHORT_ENTRY"
            else:
                action, final_kelly, pos_signal = 0, 0.0, "HOLD"

        info = {
            "agent": "DSAC_PRIMARY",
            "raw_action": primary_raw,
            "primary_raw": primary_raw,
            "primary_action": int(primary_action),
            "primary_kelly": float(primary_lev),
            "kelly": final_kelly,
            "score": float(abs(primary_raw)),
            "conviction": float(abs(primary_raw)),
            "agreement": float(abs(primary_raw)),
            "ambiguity": 0.0,
            "long_edge": float(max(primary_raw, 0.0)),
            "short_edge": float(max(-primary_raw, 0.0)),
            "_long_raw": float(max(primary_raw, 0.0)),
            "_short_raw": float(max(-primary_raw, 0.0)),
            "_long_action": int(1 if primary_raw > 0.0 else 0),
            "_short_action": int(2 if primary_raw < 0.0 else 0),
            "_long_kelly": float(max(primary_lev, 0.0) if primary_raw > 0.0 else 0.0),
            "_short_kelly": float(max(primary_lev, 0.0) if primary_raw < 0.0 else 0.0),
            "_selected_side": "LONG" if action == 1 else ("SHORT" if action == 2 else "HOLD"),
            "position_signal": pos_signal,
            "enter_th": self.enter_th,
            "exit_th": self.exit_th,
            "close_th": self.close_th,
        }
        if action == 0 and self.pos is None: info["hold_reason"] = f"below_enter_th({self.enter_th:.3f})"
        return int(action), float(final_kelly), info, elite_sigs, regime


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
        self.adaptive_enter_offset: float = 0.0
        self.adaptive_agreement_offset: float = 0.0

        self.min_live_kelly = float(os.getenv("FUSE_MIN_LIVE_KELLY", "0.04"))
        self.dsac_only_hard_stop = float(os.getenv("DSAC_ONLY_HARD_STOP", "0.025"))
        self.dsac_only_max_hold = int(os.getenv("DSAC_ONLY_MAX_HOLD", "36"))
        self.dsac_only_trail_arm = float(os.getenv("DSAC_ONLY_TRAIL_ARM", "0.012"))
        self.dsac_only_trail_gap = float(os.getenv("DSAC_ONLY_TRAIL_GAP", "0.008"))
        self.dsac_only_vol_scale_enable = _env_flag("DSAC_ONLY_VOL_SCALE_ENABLE", True)
        self.dsac_only_cooldown_enable = _env_flag("DSAC_ONLY_COOLDOWN_ENABLE", False)
        self.dsac_only_trend_exit_enable = _env_flag("DSAC_ONLY_TREND_EXIT_ENABLE", False)
        self.dsac_only_trend_exit_hold_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_HOLD_BARS", "24"))
        self.dsac_only_trend_exit_confirm_bars = int(os.getenv("DSAC_ONLY_TREND_EXIT_CONFIRM_BARS", "2"))
        self.dsac_only_trend_exit_score = float(os.getenv("DSAC_ONLY_TREND_EXIT_SCORE", "0.20"))
        self.dsac_only_trend_exit_quality = float(os.getenv("DSAC_ONLY_TREND_EXIT_QUALITY", "0.000"))

        self.hibernation_enable = _env_flag("DSAC_HIBERNATION_ENABLE", True)
        self.hibernation_score_th = float(os.getenv("DSAC_HIBERNATION_SCORE_TH", "0.85"))
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

        self.step_stop_enable = _env_flag("DSAC_STEP_STOP_ENABLE", True)
        self.step_stop_levels: list[tuple[float, float]] = [
            (0.020, 0.012), (0.015, 0.007), (0.010, 0.003), (0.006, 0.000),
        ]

        self._load_live_state()

    def record_outcome(self, realized_pnl_pct: float):
        pnl = float(realized_pnl_pct)
        self.last_realized_pnl = None
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
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

        self.adaptive_enter_offset = float(np.clip(enter_offset, self.adaptive_gate_enter_min, self.adaptive_gate_enter_max))
        self.adaptive_agreement_offset = float(np.clip(agreement_offset, self.adaptive_gate_agreement_min, self.adaptive_gate_agreement_max))
        return self.adaptive_enter_offset, self.adaptive_agreement_offset

    def _choose_entry_price(self, final_action: int, current_price: float, trend_signal: dict | None = None) -> float:
        px = max(float(current_price), 0.0)
        if not self.entry_reco_enable or px <= 0.0 or not isinstance(trend_signal, dict): return px
        strength = float(trend_signal.get("strength", 0.0) or 0.0)
        quality = float(trend_signal.get("m7_quality_pred", 0.0) or 0.0)
        if strength < self.entry_reco_min_strength or quality < self.entry_reco_min_quality: return px
        
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

    def _update_pos(self, final_action: int, current_price: float, leverage: float | None = None, trend_signal: dict | None = None):
        entry_px = self._choose_entry_price(final_action, current_price, trend_signal)
        if final_action == 1 and self.pos == "SHORT":
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
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
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
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
            if self.entry_price > 0 and current_price > 0: self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.last_realized_pnl = float(self.cur_equity - 1.0)
            self.last_closed_hold_count = int(self.hold_count)
            self.pos, self.entry_price, self.hold_count = None, 0.0, 0
            self.current_leverage = 0.0
            self.peak_equity = 1.0
            self.cur_equity = 1.0
            self.trend_mismatch_streak = 0
            self.position_exit_streak = 0
            self._save_live_state()
        elif self.pos is not None and self.entry_price > 0 and current_price > 0:
            self.hold_count += 1
            if leverage is not None: self.current_leverage = float(np.clip(leverage, 0.0, 1.0))
            self.cur_equity = 1.0 + self._net_pnl_frac(current_price)
            self.peak_equity = max(self.peak_equity, self.cur_equity)
            self.last_realized_pnl = None
            self._save_live_state()

    def _load_live_state(self) -> None:
        path = self.live_state_path
        if not path or not os.path.exists(path): return
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
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
            self.recent_realized = deque([float(x) for x in data.get("recent_realized", [])], maxlen=20)
            self.trade_history = deque(data.get("trade_history", []), maxlen=2000)
        except Exception as e:
            logger.warning("⚠️ DSAC 라이브 상태 로드 실패: %s", e)

    def _save_live_state(self) -> None:
        path = self.live_state_path
        if not path: return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "pos": self.pos, "entry_price": self.entry_price, "hold_count": self.hold_count,
                "current_leverage": self.current_leverage, "peak_equity": self.peak_equity,
                "cur_equity": self.cur_equity, "last_realized_pnl": self.last_realized_pnl,
                "last_closed_hold_count": self.last_closed_hold_count, "loss_streak": self.loss_streak,
                "cooldown_bars_left": self.cooldown_bars_left, "trend_mismatch_streak": self.trend_mismatch_streak,
                "position_exit_streak": self.position_exit_streak, "adaptive_flat_cycles": self.adaptive_flat_cycles,
                "recent_realized": list(self.recent_realized), "trade_history": list(self.trade_history),
                "saved_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=True)
        except Exception as e:
            logger.warning("⚠️ DSAC 라이브 상태 저장 실패: %s", e)

    def _force_close_record(self, price: float, reason: str = "") -> None:
        if self.pos is None: return
        pnl = self._net_pnl_frac(price) if price > 0 else 0.0
        self.trade_history.append({
            "ts": datetime.utcnow().isoformat(timespec="seconds"), "side": self.pos,
            "entry": self.entry_price, "exit": price, "hold": self.hold_count,
            "pnl": round(pnl, 6), "reason": reason or "FORCE_CLOSE", "liq_forced": True,
        })
        self.recent_realized.append(pnl)
        self.loss_streak = 0 if pnl > 0 else (self.loss_streak + 1)
        logger.warning("🚨 FORCE_CLOSE 기록 | pos=%s entry=%.4f exit=%.4f pnl=%.4f%% 사유=%s",
                       self.pos, self.entry_price, price, pnl * 100, reason)
        self.pos, self.entry_price, self.hold_count, self.current_leverage, self._open_trade_diag = None, 0.0, 0, 0.0, None

    def _net_pnl_frac(self, current_price: float) -> float:
        if self.pos is None or self.entry_price <= 0.0 or current_price <= 0.0: return 0.0
        lev = float(np.clip(self.current_leverage, 0.0, 1.0))
        if self.pos == "LONG":
            entry_exec = self.entry_price * (1.0 + self.trade_slip)
            exit_exec = current_price * (1.0 - self.trade_slip)
            gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
        else:
            entry_exec = self.entry_price * (1.0 - self.trade_slip)
            exit_exec = current_price * (1.0 + self.trade_slip)
            gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
        return float(gross * lev - (2.0 * self.trade_fee * lev))

    def unrealized_pnl(self, current_price: float) -> float:
        return self._net_pnl_frac(current_price) * 100.0

    def decrement_cooldown(self) -> None:
        if self.cooldown_bars_left > 0: self.cooldown_bars_left -= 1

    def long_trend_score(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> float:
        last_row = processed_df.iloc[-1]
        def _sf(v, d: float = 0.0) -> float:
            try: return float(v)
            except Exception: return float(d)
        ts = trend_signal if isinstance(trend_signal, dict) else {}
        p_dn = float(np.clip(_sf(ts.get("prob_dn", ts.get("m7_prob_dn", 1.0 / 3.0))), 0.0, 1.0))
        p_fl = float(np.clip(_sf(ts.get("prob_flat", ts.get("m7_prob_fl", 1.0 / 3.0))), 0.0, 1.0))
        p_up = float(np.clip(_sf(ts.get("prob_up", ts.get("m7_prob_up", 1.0 / 3.0))), 0.0, 1.0))
        ps = p_dn + p_fl + p_up
        if ps <= 1e-12: p_dn, p_fl, p_up = 1.0/3.0, 1.0/3.0, 1.0/3.0
        else: p_dn, p_fl, p_up = p_dn / ps, p_fl / ps, p_up / ps

        m7_q50 = _sf(ts.get("m7_q50", 0.0))
        m7_quality = _sf(ts.get("m7_quality_pred", 0.0))
        trend_1h = _sf(last_row.get("mtf_trend_1h", 0.0))
        trend_4h = _sf(last_row.get("mtf_trend_4h", 0.0))
        closes = processed_df["close"].tail(12).astype(float).values if "close" in processed_df.columns else np.array([], dtype=float)
        ret_12 = ((closes[-1] / closes[0]) - 1.0) if len(closes) >= 2 and abs(closes[0]) > 1e-8 else 0.0

        model_edge = 0.55 * (p_up - p_dn) + 0.20 * float(np.tanh(m7_q50 * 220.0)) + 0.10 * float(np.tanh(m7_quality * 12.0))
        mtf_edge = float(np.tanh((trend_1h + trend_4h + ret_12 * 80.0) / 2.4))
        return float(np.clip(0.75 * model_edge + 0.25 * mtf_edge, -1.0, 1.0))

    def update_trend_mismatch(self, processed_df: pd.DataFrame, trend_signal: dict | None) -> tuple[bool, float, str]:
        if not self.dsac_only_trend_exit_enable or self.pos is None:
            self.trend_mismatch_streak = 0
            return False, 0.0, ""

        score = self.long_trend_score(processed_df, trend_signal)
        quality = float(trend_signal.get("m7_quality_pred", 0.0)) if isinstance(trend_signal, dict) else 0.0

        mismatch, reason = False, ""
        if self.hold_count >= max(1, self.dsac_only_trend_exit_hold_bars):
            if self.pos == "LONG" and score <= -abs(self.dsac_only_trend_exit_score) and quality <= self.dsac_only_trend_exit_quality:
                mismatch, reason = True, "DSAC_ONLY_M7_LONG_MISMATCH"
            elif self.pos == "SHORT" and score >= abs(self.dsac_only_trend_exit_score) and quality >= -self.dsac_only_trend_exit_quality:
                mismatch, reason = True, "DSAC_ONLY_M7_SHORT_MISMATCH"

        self.trend_mismatch_streak = (self.trend_mismatch_streak + 1) if mismatch else 0
        return (self.trend_mismatch_streak >= max(1, self.dsac_only_trend_exit_confirm_bars)), score, reason

    def reconcile_external_position(self, pos_type: str | None, entry_price: float, leverage: float = 0.0) -> None:
        ext_pos = pos_type if pos_type in {"LONG", "SHORT"} else None
        ext_entry = float(entry_price) if entry_price and entry_price > 0 else 0.0
        ext_lev = self.current_leverage if self.current_leverage > 0.0 else 1.0
        if ext_pos is None:
            if self.pos is not None:
                self.pos, self.entry_price, self.hold_count, self.current_leverage, self.peak_equity, self.cur_equity = None, 0.0, 0, 0.0, 1.0, 1.0
                self._save_live_state()
            return
        if self.pos != ext_pos or abs(self.entry_price - ext_entry) > 1e-6:
            self.pos, self.entry_price, self.current_leverage, self.hold_count, self.peak_equity, self.cur_equity = ext_pos, ext_entry, ext_lev, 0, 1.0, 1.0
            self._save_live_state()

    def append_trade_history(self, timestamp_kst, pnl_frac: float) -> None:
        ts_str = timestamp_kst.isoformat() if hasattr(timestamp_kst, "isoformat") else str(timestamp_kst)
        self.trade_history.append({"ts": ts_str, "pnl_frac": float(pnl_frac), "hold_bars": int(self.last_closed_hold_count)})
        self._save_live_state()

    def performance_metrics(self, now_kst) -> dict:
        if not self.trade_history:
            return {"pnl_24h": 0.0, "wr_24h": 0.0, "trades_24h": 0, "pnl_7d": 0.0, "wr_7d": 0.0, "trades_7d": 0, "pnl_all": 0.0, "wr_all": 0.0, "trades_all": 0, "cooldown_bars_left": int(self.cooldown_bars_left)}
        now_ts = pd.Timestamp(now_kst)
        def _window(hours: int):
            rows = [r for r in self.trade_history if (pd.Timestamp(r.get("ts", "2000-01-01")) >= now_ts - pd.Timedelta(hours=hours))]
            if not rows: return 0.0, 0.0, 0
            return float(sum(float(x.get("pnl_frac", 0.0)) for x in rows)) * 100.0, 100.0 * sum(1 for x in rows if float(x.get("pnl_frac", 0.0)) > 0) / len(rows), len(rows)
        p24, w24, t24 = _window(24)
        p7, w7, t7 = _window(24 * 7)
        pall = float(sum(float(x.get("pnl_frac", 0.0)) for x in self.trade_history)) * 100.0
        wall = 100.0 * sum(1 for x in self.trade_history if float(x.get("pnl_frac", 0.0)) > 0) / len(self.trade_history)
        return {"pnl_24h": p24, "wr_24h": w24, "trades_24h": t24, "pnl_7d": p7, "wr_7d": w7, "trades_7d": t7, "pnl_all": pall, "wr_all": wall, "trades_all": len(self.trade_history), "cooldown_bars_left": int(self.cooldown_bars_left)}

    def performance_summary(self, now_kst) -> str:
        m = self.performance_metrics(now_kst)
        return f"perf 24h pnl:{m['pnl_24h']:+.2f}% wr:{m['wr_24h']:.0f}% | 7d pnl:{m['pnl_7d']:+.2f}% wr:{m['wr_7d']:.0f}% | all pnl:{m['pnl_all']:+.2f}% cd:{m['cooldown_bars_left']}"

    def print_meta_dashboard(self, result: dict, current_price: float = 0.0):
        C = Colors
        fa = int(result.get("final_action", 0))
        src = str(result.get("source", "N/A"))
        fa_arrow = {0: "─", 1: "▲", 2: "▼"}.get(fa, "?")
        fa_color = {0: C.YELLOW, 1: C.GREEN, 2: C.RED}.get(fa, C.RESET)
        fa_word = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(fa, "?")

        print(f" {fa_color}{C.BOLD}{fa_arrow}{fa_arrow}  {fa_word}{C.RESET}  score={float(result.get('rl_score', 0.0)):.3f}  Kelly={float(result.get('unified_kelly', 0.0)):.3f}  source: {C.CYAN}{src}{C.RESET}")
        print(f"  {C.CYAN}• RISK{C.RESET}    step_stop={'ON' if self.step_stop_enable else 'OFF'}  trail={self.dsac_only_trail_arm:.3f}/{self.dsac_only_trail_gap:.3f}  max_hold={self.dsac_only_max_hold}  vol_scale={'ON' if self.dsac_only_vol_scale_enable else 'OFF'}  cooldown={self.cooldown_bars_left}")

        if self.pos is not None:
            unr = self.unrealized_pnl(current_price)
            pos_color = C.GREEN if self.pos == "LONG" else C.RED
            unr_color = C.GREEN if unr > 0 else (C.RED if unr < 0 else C.YELLOW)
            print(f"  {pos_color}● 포지션{C.RESET}  {pos_color}{self.pos}{C.RESET}  진입가={self.entry_price:.2f}  미실현={unr_color}{unr:+.2f}%{C.RESET}  보유={self.hold_count}봉")


# ════════════════════════════════════════════════════════════════
# 4. 비동기 메인 루프
# ════════════════════════════════════════════════════════════════
async def main(use_local=False):
    fetcher      = BinanceLiveFetcher(limit=2500)
    fe_engine    = FeatureEngineer()
    llm_advisor  = LLMAdvisor()
    try:
        llm_advisor.enabled = False
    except Exception:
        pass
    ensemble     = EnsemblePredictor() if ENSEMBLE_PREDICTOR_ENABLED else None
    dsac_nf_predictor = EnsemblePredictor()
    live_hmm: OnlineHMMDetector | None = None
    live_hmm_steps = 0
    logger.info("🧱 부가 기능: ensemble=%s", "ON" if ENSEMBLE_PREDICTOR_ENABLED else "OFF")
    
    # ── DSAC + SevenModel(M7) 융합 라우터 초기화 ─────────────────────
    meta_router = DSACTrendRouter()
    enhanced_engine = EnhancedTradingEngine()
    logger.info("🧭 실행 모드: DSAC_ONLY (최소 리스크 레이어만 유지)")
    _prev_meta_pos: str | None = None

    # ── 선행 레이더 & 사후 요격기 시작 ──────────────────────────────────
    _symbol = fetcher.symbol.lower()  # e.g. "ethusdt"
    
    ms_scanner = MicrostructureScanner(symbol=_symbol)
    ms_scanner.start()
    
    tr_interceptor = TailRiskInterceptor(symbol=_symbol)
    tr_interceptor.start()
    playbook_router = PlaybookRouter()
    _dashboard_shadow_task: asyncio.Task | None = None
    _shadow_prev_price: float | None = None
    _shadow_quant_minute_key: str = ""

    async def _fetch_quant_close_1m(limit: int = 1000) -> pd.DataFrame:
        klines = await fetcher._call_with_retry(
            f"fetch_quant_close_1m[{fetcher.symbol}]",
            lambda: fetcher.exchange.fapiPublicGetKlines(
                {"symbol": fetcher.symbol, "interval": "1m", "limit": int(max(100, min(limit, 1500)))}
            ),
        )
        if not klines:
            return pd.DataFrame(columns=["timestamp", "close"])
        qdf = pd.DataFrame(klines).iloc[:, [0, 4]]
        qdf.columns = ["timestamp", "close"]
        qdf["timestamp"] = pd.to_datetime(qdf["timestamp"], unit="ms", utc=True, errors="coerce")
        qdf["close"] = pd.to_numeric(qdf["close"], errors="coerce")
        qdf = qdf.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        return qdf

    async def _dashboard_shadow_loop():
        """10초마다 micro/tail/playbook 필드만 dashboard_state.json 갱신."""
        nonlocal _shadow_prev_price, _shadow_quant_minute_key
        while True:
            try:
                now = time.time()
                await asyncio.sleep((10.0 - (now % 10.0)) + 0.05)

                ms = dict(ms_scanner.get_signal() or {})
                tr_shadow = dict(getattr(tr_interceptor, "_shadow_state", {}) or {})
                tr_bucket = str(tr_shadow.get("shadow_risk_bucket", "normal"))
                tr_reco = "HOLD" if tr_bucket == "high" else ("REDUCE" if tr_bucket == "watch" else "FOLLOW")

                state = {}
                try:
                    if os.path.exists(DASHBOARD_STATE_PATH):
                        with open(DASHBOARD_STATE_PATH, "r", encoding="utf-8") as f:
                            state = json.load(f)
                except Exception:
                    state = {}

                _state_price = float(state.get("price", 0.0) or 0.0)
                _mark_price = float(ms.get("mark_price", 0.0) or 0.0)
                _cur_price = _mark_price if _mark_price > 0.0 else _state_price
                _prev_price = float(_shadow_prev_price if _shadow_prev_price is not None else _cur_price)
                _price_change_pct = (_cur_price - _prev_price) / max(abs(_prev_price), 1e-8) if _prev_price > 0 else 0.0
                _shadow_prev_price = _cur_price if _cur_price > 0 else _shadow_prev_price
                tr_pb = dict(
                    tr_interceptor.get_playbook_signal(
                        price_change_pct=_price_change_pct,
                        current_price=_cur_price,
                    ) or {}
                )
                _base_action = int((state.get("signal") or {}).get("final_action", 0) or 0)
                _base_kelly = float((state.get("signal") or {}).get("unified_kelly", 0.0) or 0.0)
                _base_pos = str((state.get("position") or {}).get("current", "NONE") or "NONE")
                _pb_eval = playbook_router.evaluate_all(
                    action=_base_action,
                    pos=None if _base_pos == "NONE" else _base_pos,
                    kelly=_base_kelly,
                    ms=ms,
                    tr=tr_pb,
                )
                _pb = dict(_pb_eval.get("winner_mft", {}) or {})
                _pb_hft = dict(_pb_eval.get("winner_hft", {}) or {})
                _pb_mft = dict(_pb_eval.get("winner_mft", {}) or {})
                _pb_list = list(_pb_eval.get("evaluations", []) or [])
                _hits = ",".join([str(x.get("name", "")) for x in _pb_list if bool(x.get("matched", False))]) or "-"
                logger.info(
                    "📘 PLAYBOOK(10s) HFT=%s MFT=%s mft_action=%s mft_kelly=%.3f hits=%s",
                    _pb_hft.get("name", "NONE"),
                    _pb_mft.get("name", "NONE"),
                    _pb_mft.get("action", _base_action),
                    float(_pb_mft.get("kelly", _base_kelly)),
                    _hits,
                )
                _sess_flags = _session_flags_from_timestamp(
                    state.get("cycle_timestamp_kst", pd.Timestamp.now(tz="Asia/Seoul"))
                )

                # Keep `updated_at` as 5m main-cycle timestamp.
                # Shadow loop (10s) uses separate marker so DSAC/agent cards
                # are not perceived as refreshed every 10 seconds.
                state["shadow_updated_at"] = pd.Timestamp.utcnow().isoformat()
                if _cur_price > 0.0:
                    state["price"] = float(_cur_price)
                _pos = dict(state.get("position", {}) or {})
                _pos_side = str(_pos.get("current", "NONE") or "NONE")
                if _pos_side in {"LONG", "SHORT"} and _cur_price > 0.0:
                    _pos["unrealized_pnl_pct"] = float(meta_router.unrealized_pnl(_cur_price))
                else:
                    _pos["unrealized_pnl_pct"] = 0.0
                state["position"] = _pos
                state["session"] = {
                    "session_asia": float(_sess_flags.get("session_asia", 0.0)),
                    "session_europe": float(_sess_flags.get("session_europe", 0.0)),
                    "session_us": float(_sess_flags.get("session_us", 0.0)),
                }
                state["microstructure"] = {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "mark_price": float(ms.get("mark_price", 0.0)),
                    "obi": float(ms.get("obi", 0.0)),
                    "taker_buy_ratio": float(ms.get("taker_buy_ratio", 0.5)),
                    "spoofing_score": float(ms.get("spoofing_score", 0.0)),
                    "spoofing_bias": int(ms.get("spoofing_bias", 0)),
                    "nif_whale": float(ms.get("nif_whale", 0.0)),
                    "nif_retail": float(ms.get("nif_retail", 0.0)),
                    "nif_bias": int(ms.get("nif_bias", 0)),
                    "eai": float(ms.get("eai", 0.0)),
                    "eai_bias": int(ms.get("eai_bias", 0)),
                    "oi_delta_pct": float(ms.get("oi_delta_pct", 0.0)),
                    "oi_delta_cum_5m": float(ms.get("oi_delta_cum_5m", 0.0)),
                    "oi_delta_cum_5m_bucket_start_ts": int(ms.get("oi_delta_cum_5m_bucket_start_ts", 0)),
                    "whale_flow_10s_ratio": float(ms.get("whale_flow_10s_ratio", 0.0)),
                    "whale_buy_10s_usd": float(ms.get("whale_buy_10s_usd", 0.0)),
                    "whale_sell_10s_usd": float(ms.get("whale_sell_10s_usd", 0.0)),
                    "whale_flow_cum_5m_ratio": float(ms.get("whale_flow_cum_5m_ratio", 0.0)),
                    "whale_buy_cum_5m_usd": float(ms.get("whale_buy_cum_5m_usd", 0.0)),
                    "whale_sell_cum_5m_usd": float(ms.get("whale_sell_cum_5m_usd", 0.0)),
                    "whale_flow_cum_5m_bucket_start_ts": int(ms.get("whale_flow_cum_5m_bucket_start_ts", 0)),
                    "funding_rate": float(ms.get("funding_rate", 0.0)),
                    "signal_bias": int(ms.get("signal_bias", 0)),
                    "kelly_mult": float(ms.get("kelly_mult", 1.0)),
                    "toxicity_score": float(ms.get("shadow_toxicity_score", 0.0)),
                    "toxicity_regime": str(ms.get("shadow_toxicity_regime", "normal")),
                    "queue_collapse": float(ms.get("shadow_queue_collapse", 0.0)),
                    "absorption_score": float(ms.get("shadow_absorption_score", 0.0)),
                    "queue_bias": int(ms.get("shadow_queue_bias", 0)),
                    "regime_tag": str(ms.get("shadow_regime_tag", "normal")),
                    "regime_conf": float(ms.get("shadow_regime_conf", 0.0)),
                    "price_change_30m": float(ms.get("price_change_30m", 0.0)),
                    "price_volatility_30m": float(ms.get("price_volatility_30m", 0.0)),
                    "vwap_gap_15m": float(ms.get("vwap_gap_15m", 0.0)),
                    "price_breakout_60m": bool(ms.get("price_breakout_60m", False)),
                    "price_breakdown_60m": bool(ms.get("price_breakdown_60m", False)),
                    "nif_whale_sum_30m": float(ms.get("nif_whale_sum_30m", 0.0)),
                    "nif_whale_avg_30m": float(ms.get("nif_whale_avg_30m", 0.0)),
                    "nif_whale_std_30m": float(ms.get("nif_whale_std_30m", 0.0)),
                    "whale_short_build_ratio_30m": float(ms.get("whale_short_build_ratio_30m", 0.0)),
                    "whale_long_close_ratio_30m": float(ms.get("whale_long_close_ratio_30m", 0.0)),
                    "whale_sell_presence_ratio_30m": float(ms.get("whale_sell_presence_ratio_30m", 0.0)),
                    "whale_sell_effective_ratio_30m": float(ms.get("whale_sell_effective_ratio_30m", 0.0)),
                    "whale_long_build_ratio_30m": float(ms.get("whale_long_build_ratio_30m", 0.0)),
                    "whale_short_cover_ratio_30m": float(ms.get("whale_short_cover_ratio_30m", 0.0)),
                    "whale_buy_presence_ratio_30m": float(ms.get("whale_buy_presence_ratio_30m", 0.0)),
                    "whale_buy_effective_ratio_30m": float(ms.get("whale_buy_effective_ratio_30m", 0.0)),
                    "whale_position_bias_30m": str(ms.get("whale_position_bias_30m", "중립")),
                    "whale_position_window_min": int(ms.get("whale_position_window_min", 5)),
                    "whale_position_estimate": str(ms.get("whale_position_estimate", "NEUTRAL")),
                    "whale_position_confidence": int(ms.get("whale_position_confidence", 0)),
                    "whale_position_score": float(ms.get("whale_position_score", 0.0)),
                    "absorption_avg_30m": float(ms.get("absorption_avg_30m", 0.0)),
                    "bias_avg_30m": float(ms.get("bias_avg_30m", 0.0)),
                    "toxicity_avg_30m": float(ms.get("toxicity_avg_30m", 0.0)),
                    "eai_delta_15m": float(ms.get("eai_delta_15m", 0.0)),
                    "data_stale": bool(ms.get("data_stale", False)),
                    "depth_connected": bool(ms.get("depth_connected", False)),
                    "trade_connected": bool(ms.get("trade_connected", False)),
                    "poll_connected": bool(ms.get("poll_connected", False)),
                    "status_line": str(ms_scanner.status_line()),
                }
                state["tail_risk"] = {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "aftershock_prob": float(tr_shadow.get("shadow_aftershock_prob", 0.0)),
                    "half_life_min": float(tr_shadow.get("shadow_decay_half_life", 0.0)),
                    "risk_bucket": tr_bucket,
                    "z_long": float(tr_pb.get("z_long", 0.0)),
                    "z_short": float(tr_pb.get("z_short", 0.0)),
                    "lai": float(tr_pb.get("lai", 0.0)),
                    "long_usd_1m": float(tr_pb.get("long_usd_1m", 0.0)),
                    "short_usd_1m": float(tr_pb.get("short_usd_1m", 0.0)),
                    "hawkes_active": bool(tr_pb.get("hawkes_active", False)),
                    "hawkes_decay_level": float(tr_pb.get("hawkes_decay_level", 0.0)),
                    "crisis_type": str(tr_pb.get("crisis_type", "")),
                    "liq_cluster_direction": int(tr_pb.get("liq_cluster_direction", 0)),
                    "liq_cluster_strength": float(tr_pb.get("liq_cluster_strength", 0.0)),
                    "distance_to_cluster_pct": float(tr_pb.get("distance_to_cluster_pct", 1.0)),
                    "liq_cluster_price": float(tr_pb.get("liq_cluster_price", 0.0)),
                    "z_bias": int(-1 if float(tr_pb.get("z_long", 0.0)) > float(tr_pb.get("z_short", 0.0)) else (1 if float(tr_pb.get("z_short", 0.0)) > float(tr_pb.get("z_long", 0.0)) else 0)),
                    "recommendation": tr_reco,
                    "status_line": str(tr_interceptor.status_line()),
                }
                state["playbook"] = {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "matched": bool(_pb.get("matched", False)),
                    "name": str(_pb.get("name", "NONE")),
                    "priority": int(_pb.get("priority", 0)),
                    "action": int(_pb.get("action", _base_action)),
                    "kelly": float(_pb.get("kelly", _base_kelly)),
                    "reason": str(_pb.get("reason", "")),
                    "emergency_exit": bool(_pb.get("emergency_exit", False)),
                    "widen_trailing_stop": bool(_pb.get("widen_trailing_stop", False)),
                    "meta": dict(_pb.get("meta", {}) or {}),
                    "hft": _pb_hft,
                    "mft": _pb_mft,
                    "evaluations": _pb_list,
                }
                _ens = _build_ensemble_runtime(
                    pb_list=_pb_list,
                    base_action=_base_action,
                    base_kelly=_base_kelly,
                    ms=ms,
                    tr=tr_pb,
                )
                _loop = asyncio.get_running_loop()
                _poly = await _loop.run_in_executor(
                    None,
                    lambda: _get_polymarket_snapshot_cached(_cur_price),
                )
                _trk = await _loop.run_in_executor(
                    None,
                    lambda: _update_ensemble_tracker(
                        ensembles=_ens,
                        current_price=_cur_price,
                        now_iso=_now_kst_iso(),
                    ),
                )
                _ens["tracker"] = _ensemble_tracker_summary(_trk)
                state["ensembles"] = _ens
                state["polymarket"] = dict(_poly or {})

                now_kst = pd.Timestamp.now(tz="Asia/Seoul")
                quant_minute_key = now_kst.strftime("%Y-%m-%d %H:%M")
                if _shadow_quant_minute_key != quant_minute_key:
                    try:
                        qdf = await _fetch_quant_close_1m(limit=1000)
                        if len(qdf) > 0:
                            q_cur = float(qdf["close"].iloc[-1])
                            state["quant_formula"] = _build_quant_formula_card(
                                eth_df=qdf,
                                current_price=q_cur,
                                current_time_kst=now_kst,
                            )
                            _shadow_quant_minute_key = quant_minute_key
                    except Exception as _qerr:
                        logger.debug("quant shadow update skip: %s", _qerr)

                await _loop.run_in_executor(None, _atomic_write_json, DASHBOARD_STATE_PATH, state)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("dashboard shadow loop skip: %s", e)

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
                    if abs(contracts) <= 1e-12: continue
                    side = str(p.get("side", "")).upper()
                    if side not in {"LONG", "SHORT"}: side = "LONG" if contracts > 0 else "SHORT"
                    entry = float(p.get("entryPrice", p.get("entry_price", 0.0)) or 0.0)
                    lev = float(p.get("leverage", 0.0) or 0.0)
                    return {"type": side, "entry_price": entry, "leverage": lev}
        except Exception as e:
            logger.debug("exchange.fetch_positions 복원 실패: %s", e)
        return None

    def _bars_stale(eth_df: pd.DataFrame) -> bool:
        if eth_df is None or len(eth_df) == 0: return True
        last_ts = pd.Timestamp(eth_df['timestamp'].iloc[-1])
        if last_ts.tzinfo is not None: last_ts = last_ts.tz_localize(None)
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        if (now_utc - last_ts) > pd.Timedelta(minutes=15):
            logger.warning("⚠️ 최신 봉 지연 감지: last=%s age=%s", last_ts, (now_utc - last_ts))
            return True
        return False

    trend_hub = SevenModelEnsemble(strict=bool(M7_ENTRY_PRICE_ENABLE))
    unsup_hub = UnsupervisedRegimeHub()
    
    runtime_feature_keep: set[str] = build_active_feature_keep(
        include_entry_price=bool(M7_ENTRY_PRICE_ENABLE),
        include_m7_artifacts=True,
    )

    tg_notifier = TelegramNotifier()

    async def _run_cycle(processed_df, eth_buffer):
        """한 사이클: DSAC_ONLY 판단 + 집행."""
        nonlocal _prev_meta_pos
        nonlocal live_hmm_steps

        meta_router.decrement_cooldown()
        
        _last_row = processed_df.iloc[-1]
        
        ens_preds, ens_confs = None, None
        try:
            ens_preds, ens_confs = await dsac_nf_predictor.predict_all_async(processed_df)
        except Exception as e:
            logger.warning("⚠️ DSAC pred/conf 공급용 앙상블 예측 실패: %s", e)
        nf_preds = {}

        current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
        current_price    = float(eth_buffer['close'].iloc[-1])
        regime_name      = 'UNKNOWN'
        _last_idx = processed_df.index[-1]

        if ens_preds is not None and ens_confs is not None and dsac_nf_predictor is not None and len(processed_df) > 0:
            _name_to_idx = {n: i for i, n in enumerate(getattr(dsac_nf_predictor, "MODEL_ORDER", []))}
            _inject_map = {
                "PatchTST": ("pred_patchtst", "conf_patchtst"),
                "Chronos": ("pred_chronos", "conf_chronos"),
                "TiDE": ("pred_tide", "conf_tide"),
            }
            for _mname, (_pcol, _ccol) in _inject_map.items():
                _idx = _name_to_idx.get(_mname)
                _model = getattr(dsac_nf_predictor, "models", {}).get(_mname) if hasattr(dsac_nf_predictor, "models") else None
                if _idx is None or _model is None or not getattr(_model, "available", False): continue
                try:
                    _pv, _cv = float(ens_preds[_idx]), float(ens_confs[_idx])
                    if np.isfinite(_pv): processed_df.at[_last_idx, _pcol] = _pv
                    if np.isfinite(_cv): processed_df.at[_last_idx, _ccol] = _cv
                except Exception: continue

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

        m7_last = None
        trend_signal = None
        try:
            m7_last = trend_hub.predict_last(processed_df)
            trend_signal = _trend_signal_from_m7(m7_last)
        except Exception as e:
            logger.warning("M7 피처 생성 실패로 이번 사이클 스킵")
            return

        _last = processed_df.iloc[-1]
        for _pcol, _ccol in zip(DSAC_STATE_PRED, DSAC_STATE_CONF):
            if _pcol not in _last.index or _ccol not in _last.index: continue
            try:
                _pv, _cv = float(_last[_pcol]), float(_last[_ccol])
                if np.isfinite(_pv) and np.isfinite(_cv):
                    nf_preds[_pcol], nf_preds[_ccol] = _pv, _cv
            except Exception: continue

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

        info.setdefault("agent", "DSAC_DUAL")
        info.setdefault("kelly", float(dsac_lev))
        info.setdefault("long_edge", float(info.get("_long_raw", 0.0)))
        info.setdefault("short_edge", float(info.get("_short_raw", 0.0)))
        info.setdefault("conviction", float(abs(info.get("raw_action", 0.0))))
        info.setdefault("agreement", float(abs(info.get("raw_action", 0.0))))
        info.setdefault("ambiguity", 0.0)
        info.setdefault("score", float(max(abs(info.get("raw_action", 0.0)), float(info.get("conviction", 0.0)))))
        regime_name = next((k.replace('regime_', '').upper() for k, v in regime.items() if v == 1.0), 'UNKNOWN')
        
        _iso_anom = bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5)
        _vae_anom = bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5)
        _vae_err = float((trend_signal or {}).get("m7_vae_error", 0.0) or 0.0)
        _vae_th = float((trend_signal or {}).get("m7_vae_threshold", 0.0) or 0.0)
        _vae_ratio = (_vae_err / max(_vae_th, 1e-8)) if _vae_th > 1e-8 else (1.0 if _vae_anom else 0.0)
        _jump_z = float(processed_df.iloc[-1].get("jump_z", 0.0) or 0.0)
        _evt_z = float(processed_df.iloc[-1].get("evt_excess_z", 0.0) or 0.0)
        _hib_score = float(np.clip(max(
            1.0 if _iso_anom else 0.0,
            min(_vae_ratio / 1.35, 1.5),
            min(abs(_jump_z) / 3.0, 1.5),
            min(abs(_evt_z) / 3.0, 1.5),
        ) / 1.5, 0.0, 1.0))

        prev_meta_pos = _prev_meta_pos
        _dsac_only_source = "DSAC_PURE_RL" if DSAC_PURE_RL_MODE else "DSAC_ONLY"
        _hold_reason = str(info.get("hold_reason", ""))
        _block_reason = ""
        _trend_exit_score = 0.0

        if DSAC_PURE_RL_MODE:
            _a = float(info.get("primary_raw", info.get("raw_action", 0.0)))
            _abs = abs(_a)
            _pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.15"))
            _close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.00"))
            _max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
            _force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}
            _fa, _kelly = 0, 0.0
            
            if meta_router.pos is None:
                if _a > _pos_th: _fa, _kelly = 1, min(_abs, _max_kelly)
                elif _a < -_pos_th: _fa, _kelly = 2, min(_abs, _max_kelly)
            elif meta_router.pos == "LONG":
                _live_unr = float(meta_router._net_pnl_frac(current_price))
                if _force_close and _live_unr <= -0.025: _fa, _kelly, _dsac_only_source = 0, 0.0, "DSAC_PURE_RL_FORCE_CLOSE"
                elif _abs < _close_th: _fa, _kelly = 0, 0.0
                elif _a < -_pos_th: _fa, _kelly = 2, min(_abs, _max_kelly)
                else: _fa, _kelly = 1, min(_abs, _max_kelly)
            else:  # SHORT
                _live_unr = float(meta_router._net_pnl_frac(current_price))
                if _force_close and _live_unr <= -0.025: _fa, _kelly, _dsac_only_source = 0, 0.0, "DSAC_PURE_RL_FORCE_CLOSE"
                elif _abs < _close_th: _fa, _kelly = 0, 0.0
                elif _a > _pos_th: _fa, _kelly = 1, min(_abs, _max_kelly)
                else: _fa, _kelly = 2, min(_abs, _max_kelly)
            _kelly = float(np.clip(_kelly, 0.0, 1.0))
        else:
            _kelly = float(np.clip(dsac_lev, 0.0, 1.0))
            _fa = int(dsac_action)
            _position_signal = str(info.get("position_signal", "HOLD"))
            _position_reason = str(info.get("position_reason", ""))
            if _position_signal == "EXIT": _fa, _kelly, _dsac_only_source = 0, 0.0, f"DSAC_LOGIT_EXIT:{_position_reason or 'RULE'}"
            elif _position_signal == "REDUCE": _fa, _dsac_only_source = (1 if meta_router.pos == "LONG" else 2), f"DSAC_LOGIT_REDUCE:{_position_reason or 'RULE'}"

        _trend_exit, _trend_exit_score, _trend_exit_reason = meta_router.update_trend_mismatch(
            processed_df, trend_signal
        )
        if _trend_exit and meta_router.pos is not None:
            _fa, _kelly, _dsac_only_source = 0, 0.0, _trend_exit_reason or "DSAC_ONLY_TREND_EXIT"

        if ENH_RUNTIME_ENABLE:
            _session_flags = _session_flags_from_timestamp(current_time_kst)
            _enhanced = enhanced_engine.process(
                dsac_action=int(_fa), dsac_kelly=float(_kelly), dsac_info=info,
                processed_df=processed_df, eth_buffer=eth_buffer, btc_buffer=btc_buffer,
                meta_router=meta_router, regime=regime, trend_signal=trend_signal,
                session_flags=_session_flags,
            )
            _fa = int(_enhanced.get("action", _fa))
            _kelly = float(np.clip(_enhanced.get("kelly", _kelly), 0.0, 1.0))
            _dsac_only_source = str(_enhanced.get("source", _dsac_only_source))

        # ── 📡 선행 레이더 (MicrostructureScanner) 개입 ────────────────
        ms_signal = ms_scanner.get_signal()
        ms_kelly_mult = ms_scanner.get_kelly_multiplier()
        
        # 스캐너 배수가 1.0이 아닐 경우 레버리지(Kelly) 비중 조정
        if ms_kelly_mult != 1.0:
            old_kelly = _kelly
            _kelly = float(np.clip(_kelly * ms_kelly_mult, 0.0, 1.0))
            _dsac_only_source = f"{_dsac_only_source} | MS_BOOST(x{ms_kelly_mult:.2f})"
            logger.info(f"📡 [레이더 개입] 미시구조 탐지! Kelly 비중 조정: {old_kelly:.3f} -> {_kelly:.3f}")
            
        # NIF 스마트머니 매매 편향에 따른 진입 제한
        ms_bias = ms_signal.get("signal_bias", 0)
        if _fa == 1 and ms_bias == -1:
            _kelly *= 0.5  # 롱 보는데 고래는 파는 중
            _dsac_only_source += " [NIF_WARN:WHALE_SELL]"
        elif _fa == 2 and ms_bias == 1:
            _kelly *= 0.5  # 숏 보는데 고래는 사는 중
            _dsac_only_source += " [NIF_WARN:WHALE_BUY]"


        # ── 🛡️ 사후 요격기 (TailRiskInterceptor) 개입 ────────────────
        # LAI(청산 흡수 지수) 계산을 위해 이전 1분봉 종가 획득
        prev_price = float(eth_buffer['close'].iloc[-2]) if len(eth_buffer) >= 2 else current_price

        _fa, _kelly, _tr_reason = tr_interceptor.intercept(
            action=_fa, 
            pos=meta_router.pos, 
            kelly=_kelly, 
            current_price=current_price, 
            prev_price=prev_price
        )
        
        if _tr_reason:
            logger.warning("🛡️ TAIL RISK INTERCEPT: %s", _tr_reason)
            _dsac_only_source = f"TR_INTERCEPT({_tr_reason})"

        # ── 🎯 Polymarket 1분 급변 시 강제 청산 가드 ────────────────
        try:
            _poly_live = _get_polymarket_snapshot_cached(float(current_price))
        except Exception:
            _poly_live = {}
        _poly_force_exit, _poly_reason = _polymarket_exit_guard(
            pos=meta_router.pos,
            entry_price=float(meta_router.entry_price or 0.0),
            poly=dict(_poly_live or {}),
        )
        if _poly_force_exit:
            _fa, _kelly = 0, 0.0
            _dsac_only_source = str(_poly_reason)
            logger.warning("🎯 POLYMARKET EXIT: %s", _poly_reason)
        elif _poly_reason:
            logger.info("🎯 POLYMARKET HOLD: %s", _poly_reason)

        # ── Playbook Router: 분석/대시보드 전용 (실제 매매결정에는 미개입) ──
        _ms_exec = dict(ms_signal or {})
        _price_change_pct_exec = (current_price - prev_price) / max(abs(prev_price), 1e-8) if prev_price > 0 else 0.0
        _tr_pb_exec = dict(
            tr_interceptor.get_playbook_signal(
                price_change_pct=_price_change_pct_exec,
                current_price=current_price,
            ) or {}
        )
        _pb_exec_eval = playbook_router.evaluate_all(
            action=int(_fa),
            pos=meta_router.pos,
            kelly=float(_kelly),
            ms=_ms_exec,
            tr=_tr_pb_exec,
        )
        _pb_exec = dict(_pb_exec_eval.get("winner_mft", {}) or {})
        _pb_exec_hft = dict(_pb_exec_eval.get("winner_hft", {}) or {})
        _pb_exec_mft = dict(_pb_exec_eval.get("winner_mft", {}) or {})
        _pb_exec_list = list(_pb_exec_eval.get("evaluations", []) or [])

        # ── 최종 포지션 업데이트 및 대시보드 저장 ──
        meta_router._update_pos(_fa, current_price, _kelly, trend_signal)
        meta_router.update_adaptive_gate(final_action=int(_fa), in_position=(meta_router.pos is not None))

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
            "illiq_amihud": float(processed_df.iloc[-1].get("amihud_illiquidity_z", 0.0) or 0.0),
            "cb_active": 0,
            "m7_qwidth": float((trend_signal or {}).get("m7_qwidth", 0.0) or 0.0),
            "position_signal": str(info.get("position_signal", "")),
            "position_reason": str(info.get("position_reason", "")),
            "position_own_support": float(info.get("own_support", 0.0)),
            "position_opp_pressure": float(info.get("opp_pressure", 0.0)),
            "position_net_edge": float(info.get("net_edge", 0.0)),
            "hold_reason": str(_hold_reason),
            "block_reason": str(_block_reason),
        }
        rl_action = int(dsac_action)
        trade_pnl_pct: float | None = None

        _new_pos = meta_router.pos
        _position_closed = (_prev_meta_pos is not None and _new_pos != _prev_meta_pos)
        if _position_closed:
            realized = meta_router.last_realized_pnl
            if realized is None: realized = float(meta_router.cur_equity - 1.0)
            trade_pnl_pct = float(realized) * 100.0
            enhanced_engine.on_trade_close(float(realized))
            meta_router.record_outcome(float(realized))
            meta_router.append_trade_history(current_time_kst, float(realized))

        if _new_pos is not None and _new_pos != _prev_meta_pos:
            enhanced_engine.on_position_open()
        if _prev_meta_pos is None and _new_pos is not None and trade_pnl_pct is None:
            trade_pnl_pct = 0.0
        if trade_pnl_pct is not None: meta_result["trade_pnl_pct"] = float(trade_pnl_pct)

        _agent_actions = {
            "long": {
                "action": int(info.get("_long_action", 0)),
                "kelly_weight": float(max(0.0, _safe_float(info.get("_long_kelly", info.get("_long_raw", 0.0)), 0.0))),
            },
            "short": {
                "action": int(info.get("_short_action", 0)),
                "kelly_weight": float(max(0.0, _safe_float(info.get("_short_kelly", info.get("_short_raw", 0.0)), 0.0))),
            },
        }
        _agent_tracker_state = _update_agent_tracker(
            agent_actions=_agent_actions,
            current_price=current_price,
            now_iso=str(current_time_kst),
        )
        _agent_tracker = _agent_tracker_summary(_agent_tracker_state)
        
        if prev_meta_pos != _new_pos:
            if prev_meta_pos is None and _new_pos: _tg_code = f"ENTER_{_new_pos}"
            elif prev_meta_pos and _new_pos is None: _tg_code = f"EXIT_{prev_meta_pos}"
            elif prev_meta_pos and _new_pos: _tg_code = f"FLIP_{prev_meta_pos}_TO_{_new_pos}"
            else: _tg_code = None
            if _tg_code:
                asyncio.create_task(tg_notifier.notify(
                    _tg_trade_msg(_tg_code, current_price, current_time_kst, regime_name, meta_result)
                ))

        _prev_meta_pos = _new_pos

        _print_final_trade_summary(
            timestamp_kst=current_time_kst, current_price=current_price,
            regime_name=regime_name, rl_action=rl_action, rl_info=info,
            meta_result=meta_result, prev_pos=prev_meta_pos, cur_pos=meta_router.pos,
        )

        if not COMPACT_MODE:
            meta_router.print_meta_dashboard(meta_result, current_price)
            if "enhanced_diag" in info:
                enhanced_engine.print_enhanced_dashboard({
                    "action": _fa, "kelly": _kelly, "source": _dsac_only_source,
                    "diagnostics": info.get("enhanced_diag", {}),
                })
        _perf_metrics = meta_router.performance_metrics(current_time_kst)
        try:
            _ms = _ms_exec
            _tr_shadow = dict(getattr(tr_interceptor, "_shadow_state", {}) or {})
            _tr_bucket = str(_tr_shadow.get("shadow_risk_bucket", "normal"))
            if _tr_bucket == "high":
                _tr_reco = "HOLD"
            elif _tr_bucket == "watch":
                _tr_reco = "REDUCE"
            else:
                _tr_reco = "FOLLOW"
            _tr_pb = _tr_pb_exec
            _pb = _pb_exec
            _pb_hft = _pb_exec_hft
            _pb_mft = _pb_exec_mft
            _pb_overall = dict(_pb_exec_eval.get("winner", {}) or {})
            _pb_list = _pb_exec_list
            _ms_for_llm = dict(ms_signal or {})
            _matched = [x for x in _pb_list if bool(x.get("matched", False))]
            if _matched:
                _top_consensus = _matched
            else:
                _top_consensus = [{
                    "name": "NO_ACTIVE_PLAYBOOK",
                    "matched": False,
                    "reason": "현재 시장은 횡보 중이거나 HFT/MFT 특이 조건이 발동되지 않은 평온한 상태입니다.",
                }]
            _llm_payload = {
                "portfolio_state": {
                    "current_position": str(meta_router.pos or "NONE"),
                    "unrealized_pnl_pct": float(meta_router.unrealized_pnl(current_price) if meta_router.pos else 0.0),
                },
                "market_environment": {
                    "funding_rate": float(_ms_for_llm.get("funding_rate", 0.0)),
                    "eai_energy": float(_ms_for_llm.get("eai", 0.0)),
                    "whale_cvd_30m": float(_ms_for_llm.get("nif_whale_sum_30m", 0.0)),
                    "toxicity": float(_ms_for_llm.get("shadow_toxicity_score", 0.0)),
                    "price_volatility_30m": float(_ms_for_llm.get("price_volatility_30m", 0.0)),
                },
                "playbook_consensus": {
                    "winner": {
                        "name": str(_pb_overall.get("name", "NONE")),
                        "matched": bool(_pb_overall.get("matched", False)),
                        "action": int(_pb_overall.get("action", 0)),
                        "kelly": float(_pb_overall.get("kelly", 0.0)),
                        "reason": str(_pb_overall.get("reason", "")),
                    },
                    "top_playbooks": _top_consensus,
                },
            }
            _llm_advice = {
                "enabled": False,
                "updated_at": pd.Timestamp.utcnow().isoformat(),
                "decision": "HOLD",
                "confidence_score": 0,
                "kelly_weight": 0.0,
                "reasoning": "LLM 미실행",
            }
            if False and bool(getattr(llm_advisor, "enabled", False)):
                _llm_ctx = {
                    "close": float(current_price),
                    "llm_router_payload": _llm_payload,
                }
                _llm_decision = await llm_advisor.advise(_llm_ctx)
                if _llm_decision is not None:
                    _llm_advice = {
                        "enabled": True,
                        "updated_at": pd.Timestamp.utcnow().isoformat(),
                        "decision": str(_llm_decision.decision),
                        "confidence_score": int(_llm_decision.conviction),
                        "kelly_weight": float(_llm_decision.size) / 100.0,
                        "reasoning": str(getattr(_llm_decision, "reasoning", "") or ""),
                        "tp": _llm_decision.tp,
                        "sl": _llm_decision.sl,
                    }
                    logger.info(
                        "🤖 LLM(5m) decision=%s conf=%d kelly=%.2f",
                        _llm_advice["decision"],
                        _llm_advice["confidence_score"],
                        _llm_advice["kelly_weight"],
                    )
                else:
                    _llm_advice["reasoning"] = "LLM 응답 없음/파싱 실패"

            _trades_tail = []
            _eq = 1.0
            for _row in list(meta_router.trade_history)[-120:]:
                _p = float(_row.get("pnl_frac", _row.get("pnl", 0.0)) or 0.0)
                _eq *= (1.0 + _p)
                _trades_tail.append({
                    "ts": str(_row.get("ts", "")),
                    "pnl_pct": _p * 100.0,
                    "equity": _eq,
                })

            _sess_flags_live = _session_flags_from_timestamp(current_time_kst)
            _poly_snapshot = dict(_POLYMARKET_CACHE.get("payload", {}) or {})
            if not _poly_snapshot:
                _poly_snapshot = _get_polymarket_snapshot_cached(float(current_price))
            _quant_formula = dict(_QUANT_CARD_CACHE.get("payload", {}) or {})
            if not _quant_formula:
                _quant_formula = _build_quant_formula_card(
                    eth_df=eth_buffer,
                    current_price=float(current_price),
                    current_time_kst=current_time_kst,
                )
            _dashboard_state = {
                "schema_version": "live.dashboard.v2",
                "updated_at": pd.Timestamp.utcnow().isoformat(),
                "cycle_timestamp_kst": str(current_time_kst),
                "session": {
                    "session_asia": float(_sess_flags_live.get("session_asia", 0.0)),
                    "session_europe": float(_sess_flags_live.get("session_europe", 0.0)),
                    "session_us": float(_sess_flags_live.get("session_us", 0.0)),
                },
                "price": float(current_price),
                "regime": str(regime_name),
                "position": {
                    "current": meta_router.pos or "NONE",
                    "entry_price": float(meta_router.entry_price or 0.0),
                    "hold_bars": int(meta_router.hold_count or 0),
                    "unrealized_pnl_pct": float(meta_router.unrealized_pnl(current_price) if meta_router.pos else 0.0),
                    "trade_pnl_pct": float(trade_pnl_pct) if trade_pnl_pct is not None else None,
                },
                "signal": {
                    "rl_action": int(rl_action),
                    "final_action": int(_fa),
                    "source": str(_dsac_only_source),
                    "unified_kelly": float(_kelly),
                    "hold_reason": str(_hold_reason),
                    "block_reason": str(_block_reason),
                },
                "agents": {
                    "primary": {
                        "action": int(info.get("primary_model_action", info.get("primary_action", 0))),
                        "raw": float(info.get("primary_model_raw", info.get("primary_raw", 0.0))),
                        "std": float(info.get("primary_model_std", info.get("primary_std", 0.0))),
                    },
                    "long": {
                        "action": int(info.get("_long_action", 0)),
                        "logit": float(info.get("_long_raw", 0.0)),
                        "kelly_weight": float(_agent_actions.get("long", {}).get("kelly_weight", 0.0)),
                        "std": float(info.get("long_std", 0.0)),
                        "decision_at": str(current_time_kst),
                    },
                    "short": {
                        "action": int(info.get("_short_action", 0)),
                        "logit": float(info.get("_short_raw", 0.0)),
                        "kelly_weight": float(_agent_actions.get("short", {}).get("kelly_weight", 0.0)),
                        "std": float(info.get("short_std", 0.0)),
                        "decision_at": str(current_time_kst),
                    },
                    "agreement_count": int(info.get("agreement_count", 0)),
                    "net_score": float(info.get("net_score", 0.0)),
                    "conviction": float(info.get("conviction", 0.0)),
                    "tracker": _agent_tracker,
                },
                "trend": {
                    "prob_up": float((trend_signal or {}).get("prob_up", (trend_signal or {}).get("m7_prob_up", 0.0) or 0.0)),
                    "prob_dn": float((trend_signal or {}).get("prob_dn", (trend_signal or {}).get("m7_prob_dn", 0.0) or 0.0)),
                    "prob_fl": float((trend_signal or {}).get("prob_flat", (trend_signal or {}).get("m7_prob_fl", 0.0) or 0.0)),
                    "strength": float((trend_signal or {}).get("strength", 0.0) or 0.0),
                    "reversal_risk": float((trend_signal or {}).get("rev_prob", 0.0) or 0.0),
                },
                "risk": {
                    "hibernation_score": float(_hib_score),
                    "hibernation_th": float(meta_router.hibernation_score_th),
                    "amihud": float(processed_df.iloc[-1].get("amihud_illiquidity_z", 0.0) or 0.0),
                    "cooldown_bars_left": int(meta_router.cooldown_bars_left),
                },
                "microstructure": {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "obi": float(_ms.get("obi", 0.0)),
                    "taker_buy_ratio": float(_ms.get("taker_buy_ratio", 0.5)),
                    "spoofing_score": float(_ms.get("spoofing_score", 0.0)),
                    "spoofing_bias": int(_ms.get("spoofing_bias", 0)),
                    "nif_whale": float(_ms.get("nif_whale", 0.0)),
                    "nif_retail": float(_ms.get("nif_retail", 0.0)),
                    "nif_bias": int(_ms.get("nif_bias", 0)),
                    "eai": float(_ms.get("eai", 0.0)),
                    "eai_bias": int(_ms.get("eai_bias", 0)),
                    "oi_delta_pct": float(_ms.get("oi_delta_pct", 0.0)),
                    "oi_delta_cum_5m": float(_ms.get("oi_delta_cum_5m", 0.0)),
                    "oi_delta_cum_5m_bucket_start_ts": int(_ms.get("oi_delta_cum_5m_bucket_start_ts", 0)),
                    "whale_flow_10s_ratio": float(_ms.get("whale_flow_10s_ratio", 0.0)),
                    "whale_buy_10s_usd": float(_ms.get("whale_buy_10s_usd", 0.0)),
                    "whale_sell_10s_usd": float(_ms.get("whale_sell_10s_usd", 0.0)),
                    "whale_flow_cum_5m_ratio": float(_ms.get("whale_flow_cum_5m_ratio", 0.0)),
                    "whale_buy_cum_5m_usd": float(_ms.get("whale_buy_cum_5m_usd", 0.0)),
                    "whale_sell_cum_5m_usd": float(_ms.get("whale_sell_cum_5m_usd", 0.0)),
                    "whale_flow_cum_5m_bucket_start_ts": int(_ms.get("whale_flow_cum_5m_bucket_start_ts", 0)),
                    "funding_rate": float(_ms.get("funding_rate", 0.0)),
                    "signal_bias": int(_ms.get("signal_bias", 0)),
                    "kelly_mult": float(_ms.get("kelly_mult", 1.0)),
                    "toxicity_score": float(_ms.get("shadow_toxicity_score", 0.0)),
                    "toxicity_regime": str(_ms.get("shadow_toxicity_regime", "normal")),
                    "queue_collapse": float(_ms.get("shadow_queue_collapse", 0.0)),
                    "absorption_score": float(_ms.get("shadow_absorption_score", 0.0)),
                    "queue_bias": int(_ms.get("shadow_queue_bias", 0)),
                    "regime_tag": str(_ms.get("shadow_regime_tag", "normal")),
                    "regime_conf": float(_ms.get("shadow_regime_conf", 0.0)),
                    "price_change_30m": float(_ms.get("price_change_30m", 0.0)),
                    "price_volatility_30m": float(_ms.get("price_volatility_30m", 0.0)),
                    "vwap_gap_15m": float(_ms.get("vwap_gap_15m", 0.0)),
                    "price_breakout_60m": bool(_ms.get("price_breakout_60m", False)),
                    "price_breakdown_60m": bool(_ms.get("price_breakdown_60m", False)),
                    "nif_whale_sum_30m": float(_ms.get("nif_whale_sum_30m", 0.0)),
                    "nif_whale_avg_30m": float(_ms.get("nif_whale_avg_30m", 0.0)),
                    "nif_whale_std_30m": float(_ms.get("nif_whale_std_30m", 0.0)),
                    "whale_short_build_ratio_30m": float(_ms.get("whale_short_build_ratio_30m", 0.0)),
                    "whale_long_close_ratio_30m": float(_ms.get("whale_long_close_ratio_30m", 0.0)),
                    "whale_sell_presence_ratio_30m": float(_ms.get("whale_sell_presence_ratio_30m", 0.0)),
                    "whale_sell_effective_ratio_30m": float(_ms.get("whale_sell_effective_ratio_30m", 0.0)),
                    "whale_long_build_ratio_30m": float(_ms.get("whale_long_build_ratio_30m", 0.0)),
                    "whale_short_cover_ratio_30m": float(_ms.get("whale_short_cover_ratio_30m", 0.0)),
                    "whale_buy_presence_ratio_30m": float(_ms.get("whale_buy_presence_ratio_30m", 0.0)),
                    "whale_buy_effective_ratio_30m": float(_ms.get("whale_buy_effective_ratio_30m", 0.0)),
                    "whale_position_bias_30m": str(_ms.get("whale_position_bias_30m", "중립")),
                    "whale_position_window_min": int(_ms.get("whale_position_window_min", 5)),
                    "whale_position_estimate": str(_ms.get("whale_position_estimate", "NEUTRAL")),
                    "whale_position_confidence": int(_ms.get("whale_position_confidence", 0)),
                    "whale_position_score": float(_ms.get("whale_position_score", 0.0)),
                    "absorption_avg_30m": float(_ms.get("absorption_avg_30m", 0.0)),
                    "bias_avg_30m": float(_ms.get("bias_avg_30m", 0.0)),
                    "toxicity_avg_30m": float(_ms.get("toxicity_avg_30m", 0.0)),
                    "eai_delta_15m": float(_ms.get("eai_delta_15m", 0.0)),
                    "data_stale": bool(_ms.get("data_stale", False)),
                    "depth_connected": bool(_ms.get("depth_connected", False)),
                    "trade_connected": bool(_ms.get("trade_connected", False)),
                    "poll_connected": bool(_ms.get("poll_connected", False)),
                    "status_line": str(ms_scanner.status_line()),
                },
                "tail_risk": {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "aftershock_prob": float(_tr_shadow.get("shadow_aftershock_prob", 0.0)),
                    "half_life_min": float(_tr_shadow.get("shadow_decay_half_life", 0.0)),
                    "risk_bucket": str(_tr_bucket),
                    "z_long": float(_tr_pb.get("z_long", 0.0)),
                    "z_short": float(_tr_pb.get("z_short", 0.0)),
                    "lai": float(_tr_pb.get("lai", 0.0)),
                    "long_usd_1m": float(_tr_pb.get("long_usd_1m", 0.0)),
                    "short_usd_1m": float(_tr_pb.get("short_usd_1m", 0.0)),
                    "hawkes_active": bool(_tr_pb.get("hawkes_active", False)),
                    "hawkes_decay_level": float(_tr_pb.get("hawkes_decay_level", 0.0)),
                    "crisis_type": str(_tr_pb.get("crisis_type", "")),
                    "liq_cluster_direction": int(_tr_pb.get("liq_cluster_direction", 0)),
                    "liq_cluster_strength": float(_tr_pb.get("liq_cluster_strength", 0.0)),
                    "distance_to_cluster_pct": float(_tr_pb.get("distance_to_cluster_pct", 1.0)),
                    "liq_cluster_price": float(_tr_pb.get("liq_cluster_price", 0.0)),
                    "z_bias": int(-1 if float(_tr_pb.get("z_long", 0.0)) > float(_tr_pb.get("z_short", 0.0)) else (1 if float(_tr_pb.get("z_short", 0.0)) > float(_tr_pb.get("z_long", 0.0)) else 0)),
                    "recommendation": _tr_reco,
                    "status_line": str(tr_interceptor.status_line()),
                },
                "playbook": {
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "matched": bool(_pb.get("matched", False)),
                    "name": str(_pb.get("name", "NONE")),
                    "priority": int(_pb.get("priority", 0)),
                    "action": int(_pb.get("action", _fa)),
                    "kelly": float(_pb.get("kelly", _kelly)),
                    "reason": str(_pb.get("reason", "")),
                    "emergency_exit": bool(_pb.get("emergency_exit", False)),
                    "widen_trailing_stop": bool(_pb.get("widen_trailing_stop", False)),
                    "meta": dict(_pb.get("meta", {}) or {}),
                    "hft": _pb_hft,
                    "mft": _pb_mft,
                    "evaluations": _pb_list,
                },
                "performance": {
                    "pnl_24h": float(_perf_metrics.get("pnl_24h", 0.0)),
                    "wr_24h": float(_perf_metrics.get("wr_24h", 0.0)),
                    "pnl_7d": float(_perf_metrics.get("pnl_7d", 0.0)),
                    "wr_7d": float(_perf_metrics.get("wr_7d", 0.0)),
                    "pnl_all": float(_perf_metrics.get("pnl_all", 0.0)),
                    "wr_all": float(_perf_metrics.get("wr_all", 0.0)),
                },
                "llm": dict(_llm_advice or {}),
                "trades_tail": _trades_tail,
                "polymarket": dict(_poly_snapshot or {}),
                "quant_formula": dict(_quant_formula or {}),
                "ensembles": (lambda _e: {**_e, "tracker": _ensemble_tracker_summary(_load_ensemble_tracker_state())})(
                    _build_ensemble_runtime(
                        pb_list=_pb_list,
                        base_action=_fa,
                        base_kelly=_kelly,
                        ms=_ms,
                        tr=_tr_pb,
                    )
                ),
            }
            _loop = asyncio.get_running_loop()
            await _loop.run_in_executor(None, _atomic_write_json, DASHBOARD_STATE_PATH, _dashboard_state)
            if _position_closed or (_prev_meta_pos is None and _new_pos is not None):
                await _loop.run_in_executor(None, _append_jsonl, DASHBOARD_EVENTS_PATH, {
                    "ts": str(current_time_kst),
                    "event": _pos_transition_label(prev_meta_pos, _new_pos),
                    "from": prev_meta_pos,
                    "to": _new_pos,
                    "price": float(current_price),
                    "kelly": float(_kelly),
                    "pnl_pct": (float(trade_pnl_pct) if trade_pnl_pct is not None else 0.0),
                    "regime": str(regime_name),
                })
        except Exception as _dash_e:
            logger.debug("dashboard state write skip: %s", _dash_e)
        logger.info("📊 %s", meta_router.performance_summary(current_time_kst))

    try:
        _dashboard_shadow_task = asyncio.create_task(_dashboard_shadow_loop())
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
        try: processed_boot = fe_engine.process(eth_buffer, btc_buffer)
        except Exception as e:
            logger.error("❌ 초기 피처 처리 실패: %s", e)
            return
            
        try:
            dsac_router = DSACSignalRouter(hmm_detector=live_hmm)
        except Exception as e:
            logger.error(f"❌ DSAC 라우터 초기화 실패: {e}")
            return
            
        if not use_local:
            restored = await _fetch_exchange_position()
            if restored: meta_router.reconcile_external_position(restored.get("type"), float(restored.get("entry_price", 0.0)), float(restored.get("leverage", 0.0)))
        # 재시작 직후 기존 포지션을 "현재 기준점"으로 고정한다.
        # (None으로 두면 첫 사이클에서 기존 포지션도 신규 진입처럼 기록될 수 있음)
        _prev_meta_pos = meta_router.pos
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
                    # sys.stdout.write(f"\r{Colors.CYAN}⏳ 다음 5분봉까지 대기 중... ({r}초 남음)   {Colors.RESET}")
                    # sys.stdout.flush()
                    await asyncio.sleep(1)

                print()
                logger.info("🔄 최신 캔들 데이터를 갱신합니다.")
                try: new_eth, new_btc = await fetcher.fetch_latest_patch()
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
        # 종료 시 두 모니터 모두 정상 종료 처리
        if _dashboard_shadow_task and not _dashboard_shadow_task.done():
            _dashboard_shadow_task.cancel()
        ms_scanner.stop()
        tr_interceptor.stop()
        await fetcher.exchange.close()


if __name__ == "__main__":
    asyncio.run(main(use_local=False))
