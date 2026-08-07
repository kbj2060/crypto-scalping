import json
import logging
import os
import re
import time
from collections import deque
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import numpy as np
import pandas as pd


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


POLYMARKET_ENABLE = _env_flag("POLYMARKET_ENABLE", True)
POLYMARKET_EVENT_SLUG = os.getenv("POLYMARKET_EVENT_SLUG", "").strip()
POLYMARKET_EVENT_PREFIX = os.getenv("POLYMARKET_EVENT_PREFIX", "ethereum-price").strip() or "ethereum-price"
POLYMARKET_SLUG_TZ = os.getenv("POLYMARKET_SLUG_TZ", "Asia/Seoul").strip() or "Asia/Seoul"
POLYMARKET_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com/events")
POLYMARKET_CLOB_PRICE_URL = os.getenv("POLYMARKET_CLOB_PRICE_URL", "https://clob.polymarket.com/price")
POLYMARKET_CLOB_BOOK_URL = os.getenv("POLYMARKET_CLOB_BOOK_URL", "https://clob.polymarket.com/book")
POLYMARKET_TIMEOUT_SEC = float(os.getenv("POLYMARKET_TIMEOUT_SEC", "2.5"))
POLYMARKET_MAX_MARKETS = int(float(os.getenv("POLYMARKET_MAX_MARKETS", "20")))
POLYMARKET_SLUG_LOOKAHEAD_DAYS = int(float(os.getenv("POLYMARKET_SLUG_LOOKAHEAD_DAYS", "5")))
POLYMARKET_EXIT_ENABLE = _env_flag("POLYMARKET_EXIT_ENABLE", False)
POLYMARKET_SHOCK_1M_TH = float(os.getenv("POLYMARKET_SHOCK_1M_TH", "0.02"))
POLYMARKET_SHOCK_Z_TH = float(os.getenv("POLYMARKET_SHOCK_Z_TH", "1.5"))
POLYMARKET_SHOCK_CUM3_TH = float(os.getenv("POLYMARKET_SHOCK_CUM3_TH", "0.005"))
POLYMARKET_SHOCK_Z_WIN = int(float(os.getenv("POLYMARKET_SHOCK_Z_WIN", "120")))
POLYMARKET_SHOCK_DYN_ENABLE = _env_flag("POLYMARKET_SHOCK_DYN_ENABLE", False)
POLYMARKET_SHOCK_DYN_Q = float(os.getenv("POLYMARKET_SHOCK_DYN_Q", "0.997"))
POLYMARKET_SHOCK_DYN_MIN_OBS = int(float(os.getenv("POLYMARKET_SHOCK_DYN_MIN_OBS", "120")))
POLYMARKET_SHOCK_COOLDOWN_SEC = float(os.getenv("POLYMARKET_SHOCK_COOLDOWN_SEC", "5400"))
POLYMARKET_SHOCK_PEAK_ONLY = _env_flag("POLYMARKET_SHOCK_PEAK_ONLY", False)
POLYMARKET_SHOCK_PEAK_WINDOW_SEC = float(os.getenv("POLYMARKET_SHOCK_PEAK_WINDOW_SEC", "600"))

_POLYMARKET_CACHE: dict[str, object] = {"updated_at": 0.0, "payload": {}}
_POLYMARKET_HISTORY: deque[dict] = deque(maxlen=720)
_POLYMARKET_LAST_EMERGENCY_TS: float = 0.0


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


def _poly_compact_label(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    def _num(token: str) -> float | None:
        try:
            return float(str(token).replace(",", ""))
        except Exception:
            return None

    m_between = re.search(
        r"between\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*and\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_between:
        a = _num(m_between.group(1))
        b = _num(m_between.group(2))
        if a is not None and b is not None:
            lo, hi = sorted((a, b))
            return f"{int(lo) if lo.is_integer() else lo:g}-{int(hi) if hi.is_integer() else hi:g}"

    m_less = re.search(
        r"(less than|below|under|at most)\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_less:
        x = _num(m_less.group(2))
        if x is not None:
            return f"<{int(x) if x.is_integer() else x:g}"

    m_greater = re.search(
        r"(greater than|above|over|at least)\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if m_greater:
        x = _num(m_greater.group(2))
        if x is not None:
            return f">{int(x) if x.is_integer() else x:g}"

    strike = _poly_parse_strike_from_text(s)
    if strike is not None:
        if re.search(r"(less than|below|under|at most)", s, flags=re.IGNORECASE):
            return f"<{int(strike) if float(strike).is_integer() else strike:g}"
        if re.search(r"(greater than|above|over|at least)", s, flags=re.IGNORECASE):
            return f">{int(strike) if float(strike).is_integer() else strike:g}"
        return f"{int(strike) if float(strike).is_integer() else strike:g}"
    return s


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
            "raw_book": data if isinstance(data, dict) else {},
        }
    except Exception:
        return {"bid_notional": 0.0, "ask_notional": 0.0, "imbalance": 0.0, "raw_book": {}}


def _slug_for_date(prefix: str, ts: pd.Timestamp) -> str:
    month = ts.strftime("%B").lower()
    day = str(int(ts.day))
    return f"{prefix}-on-{month}-{day}"


def _event_has_open_markets(ev: dict) -> bool:
    if not isinstance(ev, dict) or bool(ev.get("closed", False)):
        return False
    mkts = list((ev or {}).get("markets", []) or [])
    for row in mkts:
        m = dict(row or {})
        if bool(m.get("closed", False)):
            continue
        if m.get("acceptingOrders") is False:
            continue
        if _poly_extract_token_ids(m):
            return True
    return False


def _fetch_polymarket_event_by_slug(slug: str) -> dict:
    query = urlencode({"slug": slug})
    url = f"{POLYMARKET_GAMMA_URL}?{query}"
    ev_raw = _poly_http_json(url, timeout=POLYMARKET_TIMEOUT_SEC)
    if isinstance(ev_raw, list):
        return dict(ev_raw[0] or {}) if ev_raw else {}
    if isinstance(ev_raw, dict):
        items = ev_raw.get("events", ev_raw.get("data", []))
        if isinstance(items, list):
            return dict(items[0] or {}) if items else dict(ev_raw)
        return dict(ev_raw)
    return {}


def _resolve_polymarket_slug() -> tuple[str, str]:
    prefix = str(POLYMARKET_EVENT_PREFIX or "").strip() or "ethereum-price"
    if POLYMARKET_EVENT_SLUG:
        return POLYMARKET_EVENT_SLUG, "env_override"
    try:
        now_local = pd.Timestamp.now(tz=POLYMARKET_SLUG_TZ)
        fallback = _slug_for_date(prefix, now_local)
        for offset in range(0, max(0, POLYMARKET_SLUG_LOOKAHEAD_DAYS) + 1):
            candidate_ts = now_local + pd.Timedelta(days=offset)
            slug = _slug_for_date(prefix, candidate_ts)
            try:
                ev = _fetch_polymarket_event_by_slug(slug)
                if _event_has_open_markets(ev):
                    mode = "auto_active_today" if offset == 0 else f"auto_active_plus_{offset}d"
                    return slug, mode
            except Exception:
                continue
        return fallback, "auto_today_fallback_no_open_event"
    except Exception:
        now_utc = pd.Timestamp.utcnow()
        return _slug_for_date(prefix, now_utc), "auto_utc_fallback"


def _fetch_polymarket_rows_by_slug(slug: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    ev = _fetch_polymarket_event_by_slug(slug)
    mkts = list((ev or {}).get("markets", []) or [])
    for m in mkts[:max(1, POLYMARKET_MAX_MARKETS)]:
        md = dict(m or {})
        token_ids = _poly_extract_token_ids(md)
        if not token_ids:
            continue
        token_id = token_ids[0]
        purl = f"{POLYMARKET_CLOB_PRICE_URL}?{urlencode({'token_id': token_id, 'side': 'BUY'})}"
        price_payload = {}
        try:
            price_payload = _poly_http_json(purl, timeout=POLYMARKET_TIMEOUT_SEC)
            prob = _poly_price_from_resp(price_payload)
        except Exception:
            prob = None
        if prob is None:
            continue
        label = _poly_market_label(md)
        strike = _poly_parse_strike_from_text(label)
        rows.append(
            {
                "label": label,
                "token_id": token_id,
                "prob": float(prob),
                "strike": float(strike) if strike is not None else np.nan,
                "raw_market": md,
                "raw_price": price_payload if isinstance(price_payload, (dict, list, str, int, float, bool)) else {},
            }
        )
    return rows, ev


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


def _poly_hist_label_delta(hist: list[dict], label: str, sec: float) -> float:
    if not hist:
        return 0.0
    key = str(label or "")
    if not key:
        return 0.0
    try:
        now_ts = float(hist[-1].get("ts", 0.0) or 0.0)
        now_map = dict(hist[-1].get("probs_map", {}) or {})
        now_prob = float(now_map.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0
    target = now_ts - float(max(1.0, sec))
    prev_prob = None
    for row in reversed(hist):
        rts = float(row.get("ts", 0.0) or 0.0)
        if rts <= target:
            try:
                rmap = dict(row.get("probs_map", {}) or {})
                prev_prob = float(rmap.get(key, 0.0) or 0.0)
            except Exception:
                prev_prob = None
            break
    if prev_prob is None:
        return 0.0
    return float(now_prob - prev_prob)


def _poly_hist_label_delta_series(hist: list[dict], label: str, sec: float, cap: int = 240) -> list[float]:
    n = len(hist)
    if n < 2:
        return []
    key = str(label or "")
    if not key:
        return []
    use_n = min(max(2, int(cap)), n)
    sub = hist[-use_n:]
    ts = np.array([float(x.get("ts", 0.0) or 0.0) for x in sub], dtype=float)
    prob = np.array([float(dict(x.get("probs_map", {}) or {}).get(key, 0.0) or 0.0) for x in sub], dtype=float)
    out: list[float] = []
    j = 0
    for i in range(use_n):
        target = ts[i] - float(max(1.0, sec))
        while j + 1 < i and ts[j + 1] <= target:
            j += 1
        if ts[j] <= target:
            out.append(float(prob[i] - prob[j]))
    return out


def _poly_hist_label_delta_pairs(hist: list[dict], label: str, sec: float, cap: int = 240) -> list[tuple[float, float]]:
    n = len(hist)
    if n < 2:
        return []
    key = str(label or "")
    if not key:
        return []
    use_n = min(max(2, int(cap)), n)
    sub = hist[-use_n:]
    ts = np.array([float(x.get("ts", 0.0) or 0.0) for x in sub], dtype=float)
    prob = np.array([float(dict(x.get("probs_map", {}) or {}).get(key, 0.0) or 0.0) for x in sub], dtype=float)
    out: list[tuple[float, float]] = []
    j = 0
    for i in range(use_n):
        target = ts[i] - float(max(1.0, sec))
        while j + 1 < i and ts[j + 1] <= target:
            j += 1
        if ts[j] <= target:
            out.append((float(ts[i]), float(prob[i] - prob[j])))
    return out


def _empty_snapshot(now_iso: str, used_slug: str, slug_mode: str, status: str, error: str, raw_event: dict | None = None) -> dict:
    return {
        "updated_at": now_iso,
        "status": status,
        "slug": used_slug,
        "slug_mode": slug_mode,
        "error": error,
        "markets_count": 0,
        "priced_count": 0,
        "mode_label": "-",
        "mode_prob": 0.0,
        "weighted_target": 0.0,
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
        "shock_basis_label": "",
        "shock_trigger": False,
        "shock_trigger_reason": "",
        "snapshot_ts_epoch": float(time.time()),
        "raw_payload": {
            "slug": used_slug,
            "slug_mode": slug_mode,
            "gamma_event": raw_event if isinstance(raw_event, dict) else {},
            "markets": [],
            "mode_book": {},
        },
    }


def _build_polymarket_snapshot(current_price: float) -> dict:
    now_iso = pd.Timestamp.utcnow().isoformat()
    used_slug, slug_mode = _resolve_polymarket_slug()
    if not POLYMARKET_ENABLE:
        return _empty_snapshot(now_iso, used_slug, slug_mode, "DISABLED", "disabled")

    try:
        rows, raw_event = _fetch_polymarket_rows_by_slug(used_slug)
    except Exception as e:
        return _empty_snapshot(now_iso, used_slug, slug_mode, "ERROR", str(e))

    if not rows:
        return _empty_snapshot(now_iso, used_slug, slug_mode, "EMPTY", "no_priced_markets", raw_event=raw_event)

    rows_sorted = sorted(rows, key=lambda x: float(x.get("prob", 0.0)), reverse=True)
    mode = rows_sorted[0]
    top3 = rows_sorted[:3]
    top3_labels = [str(x.get("label", "") or "") for x in top3 if str(x.get("label", "") or "")]
    probs = np.array([float(x.get("prob", 0.0)) for x in rows_sorted], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    prob_sum = float(np.sum(probs))
    w = (probs / prob_sum) if prob_sum > 1e-12 else np.full_like(probs, 1.0 / max(1, len(probs)))
    strikes = np.array([float(x.get("strike", np.nan)) for x in rows_sorted], dtype=float)
    strike_ok = np.isfinite(strikes)
    weighted_target = (
        float(np.sum(strikes[strike_ok] * w[strike_ok]) / max(np.sum(w[strike_ok]), 1e-12))
        if np.any(strike_ok)
        else float(current_price or 0.0)
    )

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
    _POLYMARKET_HISTORY.append(
        {
            "ts": float(time.time()),
            "mode_prob": float(mode.get("prob", 0.0)),
            "weighted_target": float(weighted_target),
            "current_price": float(cur_px),
            "probs_map": {str(x.get("label", "") or ""): float(x.get("prob", 0.0) or 0.0) for x in rows_sorted},
        }
    )
    hist = list(_POLYMARKET_HISTORY)
    prob_momentum_1m = _poly_hist_delta(hist, sec=60.0)
    shock_delta_3m = _poly_hist_delta(hist, sec=180.0)
    shock_candidates: list[dict] = []
    now_ts = float(hist[-1].get("ts", time.time()) if hist else time.time())
    lo_ts = now_ts - float(max(30.0, POLYMARKET_SHOCK_PEAK_WINDOW_SEC))
    for lb in top3_labels:
        lb_d1 = _poly_hist_label_delta(hist, label=lb, sec=60.0)
        lb_d3 = _poly_hist_label_delta(hist, label=lb, sec=180.0)
        lb_series = _poly_hist_label_delta_series(hist, label=lb, sec=60.0, cap=max(180, POLYMARKET_SHOCK_Z_WIN * 3))
        lb_pairs = _poly_hist_label_delta_pairs(hist, label=lb, sec=60.0, cap=max(180, POLYMARKET_SHOCK_Z_WIN * 3))
        lb_z = 0.0
        lb_dyn = 0.0
        lb_peak_abs = 0.0
        lb_is_peak = False
        if lb_series:
            abs_arr = np.abs(np.array(lb_series, dtype=float))
            if len(abs_arr) >= int(max(20, POLYMARKET_SHOCK_DYN_MIN_OBS)):
                q = float(np.clip(POLYMARKET_SHOCK_DYN_Q, 0.50, 0.9999))
                lb_dyn = float(np.quantile(abs_arr, q))
            win = np.array(lb_series[-max(20, POLYMARKET_SHOCK_Z_WIN):], dtype=float)
            if len(win) >= 10:
                mu = float(np.mean(win))
                sd = float(np.std(win))
                if sd > 1e-12 and np.isfinite(sd):
                    lb_z = float((lb_series[-1] - mu) / sd)
        if lb_pairs:
            recent_abs = [abs(float(v)) for (t, v) in lb_pairs if float(t) >= lo_ts]
            if recent_abs:
                lb_peak_abs = float(max(recent_abs))
                lb_is_peak = bool(abs(lb_d1) >= (lb_peak_abs - 1e-12))
        eff_th = float(max(0.0, POLYMARKET_SHOCK_1M_TH))
        if POLYMARKET_SHOCK_DYN_ENABLE and lb_dyn > 0.0:
            eff_th = max(eff_th, float(lb_dyn))
        cond_abs = abs(lb_d1) >= eff_th
        cond_z = abs(lb_z) >= float(max(0.0, POLYMARKET_SHOCK_Z_TH))
        cond_c3 = abs(lb_d3) >= float(max(0.0, POLYMARKET_SHOCK_CUM3_TH))
        cond_peak = (not POLYMARKET_SHOCK_PEAK_ONLY) or bool(lb_is_peak)
        lb_trigger = bool(cond_abs and cond_z and cond_c3 and cond_peak)
        shock_candidates.append(
            {
                "label": lb,
                "d1": float(lb_d1),
                "d3": float(lb_d3),
                "z": float(lb_z),
                "dyn_th": float(lb_dyn),
                "eff_th": float(eff_th),
                "peak_abs": float(lb_peak_abs),
                "is_peak": bool(lb_is_peak),
                "trigger": bool(lb_trigger),
            }
        )
    if shock_candidates:
        triggered = [x for x in shock_candidates if bool(x.get("trigger", False))]
        selected = max(triggered, key=lambda x: abs(float(x.get("d1", 0.0)))) if triggered else max(
            shock_candidates, key=lambda x: abs(float(x.get("d1", 0.0)))
        )
        shock_trigger = bool(len(triggered) > 0)
        shock_basis_label = str(selected.get("label", ""))
        prob_momentum_1m = float(selected.get("d1", prob_momentum_1m))
        shock_delta_3m = float(selected.get("d3", shock_delta_3m))
        shock_z_1m = float(selected.get("z", 0.0))
        shock_dyn_th_1m = float(selected.get("dyn_th", 0.0))
        shock_peak_abs_1m = float(selected.get("peak_abs", 0.0))
        shock_is_peak = bool(selected.get("is_peak", False))
        eff_th = float(selected.get("eff_th", float(max(0.0, POLYMARKET_SHOCK_1M_TH))))
        if shock_trigger:
            shock_trigger_reason = (
                f"top3_any label={shock_basis_label} "
                f"|d1m|={abs(prob_momentum_1m)*100:.2f}%p>=th{eff_th*100:.2f}%p "
                f"& |z|={abs(shock_z_1m):.2f}>={POLYMARKET_SHOCK_Z_TH:.2f} "
                f"& |d3m|={abs(shock_delta_3m)*100:.2f}%p>={POLYMARKET_SHOCK_CUM3_TH*100:.2f}%p"
                f"& peak={int(bool(shock_is_peak))}"
            )
        else:
            shock_trigger_reason = ""
    else:
        shock_basis_label = ""
        shock_z_1m = 0.0
        shock_dyn_th_1m = 0.0
        shock_peak_abs_1m = 0.0
        shock_is_peak = False
        shock_trigger = False
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
    signal = "LONG" if lead_edge > 0.03 else ("SHORT" if lead_edge < -0.03 else "HOLD")
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
        "shock_basis_label": str(shock_basis_label),
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
        "raw_payload": {
            "slug": used_slug,
            "slug_mode": slug_mode,
            "gamma_event": raw_event if isinstance(raw_event, dict) else {},
            "markets": [
                {
                    "label": str(r.get("label", "") or ""),
                    "token_id": str(r.get("token_id", "") or ""),
                    "prob": float(r.get("prob", 0.0) or 0.0),
                    "strike": (float(r.get("strike")) if pd.notna(r.get("strike")) else None),
                    "market": dict(r.get("raw_market", {}) or {}),
                    "price_buy": r.get("raw_price", {}),
                }
                for r in rows
            ],
            "mode_book": dict(book.get("raw_book", {}) or {}),
        },
    }


def get_polymarket_snapshot_cached(current_price: float) -> dict:
    now = time.time()
    cached = dict(_POLYMARKET_CACHE.get("payload", {}) or {})
    updated_at = float(_POLYMARKET_CACHE.get("updated_at", 0.0) or 0.0)
    if cached and (now - updated_at) < 8.0:
        return cached
    fresh = _build_polymarket_snapshot(current_price=float(current_price or 0.0))
    _POLYMARKET_CACHE["payload"] = dict(fresh)
    _POLYMARKET_CACHE["updated_at"] = float(now)
    return dict(fresh)


def polymarket_public_view(snapshot: dict) -> dict:
    out = dict(snapshot or {})
    raw_payload = dict(out.get("raw_payload", {}) or {})
    markets = list(raw_payload.get("markets", []) or [])
    if markets:
        compact = []
        for market in markets:
            item = dict(market or {})
            try:
                prob = float(item.get("prob", 0.0) or 0.0)
            except Exception:
                prob = 0.0
            compact.append(
                {
                    "label": _poly_compact_label(str(item.get("label", "") or "")),
                    "prob": prob,
                }
            )
        out["markets"] = sorted(compact, key=lambda x: float(x.get("prob", 0.0)), reverse=True)[:5]
    out.pop("raw_payload", None)
    return out


def append_polymarket_snapshot_to_duckdb(
    db_path: str,
    snapshot: dict,
    current_price: float,
    raw_payload: dict | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Append one 10s raw snapshot.

    Derived AI features are intentionally not persisted in the live store.
    They should be rebuilt from snapshot_json/markets_json during replay,
    training, or model-specific preprocessing.
    """
    try:
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ts = pd.Timestamp(snapshot.get("updated_at", pd.Timestamp.utcnow().isoformat()))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        raw_obj = dict(raw_payload or {})
        con = duckdb.connect(db_path)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_markets_10s_json (
                ts TIMESTAMP WITH TIME ZONE,
                markets_json VARCHAR,
                snapshot_json VARCHAR,
                current_price DOUBLE,
                schema_version INTEGER
            )
            """
        )
        existing_cols = {str(r[1]) for r in con.execute("PRAGMA table_info('polymarket_markets_10s_json')").fetchall()}
        if "snapshot_json" not in existing_cols:
            con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN snapshot_json VARCHAR")
        if "current_price" not in existing_cols:
            con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN current_price DOUBLE")
        if "schema_version" not in existing_cols:
            con.execute("ALTER TABLE polymarket_markets_10s_json ADD COLUMN schema_version INTEGER")
        markets = list(raw_obj.get("markets", []) or [])
        compact = []
        for m in markets:
            md = dict(m or {})
            try:
                prob = float(md.get("prob", 0.0) or 0.0)
            except Exception:
                prob = 0.0
            compact.append(
                {
                    "label": _poly_compact_label(str(md.get("label", "") or "")),
                    "prob": prob,
                }
            )
        compact = sorted(compact, key=lambda x: float(x.get("prob", 0.0)), reverse=True)[:5]
        payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        snapshot_payload = json.dumps(
            {
                **dict(snapshot or {}),
                "raw_payload": raw_obj,
                "storage_format": "raw_snapshot_v2",
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        con.execute(
            """
            INSERT INTO polymarket_markets_10s_json (
                ts, markets_json, snapshot_json, current_price, schema_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [ts.to_pydatetime(), payload, snapshot_payload, float(current_price or 0.0), 2],
        )
        con.close()
    except Exception as e:
        if logger is not None:
            logger.debug("polymarket duckdb append skip: %s", e)


def polymarket_exit_guard(pos: str | None, entry_price: float, poly: dict) -> tuple[bool, str]:
    return False, ""
