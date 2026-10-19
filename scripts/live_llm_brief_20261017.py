#!/usr/bin/env python3
"""**LLM 해석 입력 조립기**(ETH) — 대시보드 API + 바이낸스 공개 REST → 스냅샷 + 윈도우 + 규칙 사건 + 청산맵 1d/7d/30d + 풋프린트 1h/2h/4h/1d (2026-10-17).

왜: 사용자 «청산맵 1d/7d/30d 와 풋프린트 1h/2h/4h/1d 도 입력으로» → «그렇게 해줘». 형식 = research_llm_haiku_window_read_20261017 의 C(가장 정확).
읽기만 한다(GET): 서버 /api/market-context · /api/box-hold · /api/liquidation-map(+/tiers 7·30) · /api/footprint?bars=288 ·
  /api/footprint-daily(+/cells) · /api/liquidation-5m-history · /api/macro-calendar · /api/news · /api/trend,
  바이낸스 fapi klines·premiumIndexKlines·openInterestHist(5m·1h) · 현물 klines. 🔴주문·쓰기 없음. 방향 모델 신호(dir4h)는 일부러 뺀다(해석 전용).
  python scripts/live_llm_brief_20261017.py [--base http://127.0.0.1:8787] [--out brief.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard import llm_brief as lb  # noqa: E402

FAPI, SPOT = "https://fapi.binance.com", "https://api.binance.com"
r = lb.r


def get(url: str, params: dict | None = None, timeout: float = 30):
    q = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    with urllib.request.urlopen(urllib.request.Request(q, headers={"User-Agent": "llm-brief"}), timeout=timeout) as resp:
        return json.load(resp)


def closed(rows: list, now_ms: int) -> list:
    return [x for x in rows if int(x[6]) < now_ms]


def windows(base: str, now_ms: int, fp_bars: list, lq_bars: list) -> tuple[dict, dict]:
    """w5 = 마지막 닫힌 5분봉 24개 · w1h = 닫힌 1시간봉 24개. 시각 = 봉 마감 UTC HH:MM."""
    out = {}
    for iv, n in (("5m", 24), ("1h", 24)):
        sec = 300 if iv == "5m" else 3600
        k = closed(get(f"{FAPI}/fapi/v1/klines", {"symbol": "ETHUSDT", "interval": iv, "limit": n + 2}), now_ms)[-n:]
        t0 = [int(x[0]) for x in k]
        px_now = float(k[-1][4])
        prem = {int(x[0]): float(x[4]) * 1e4 for x in get(f"{FAPI}/fapi/v1/premiumIndexKlines", {"symbol": "ETHUSDT", "interval": iv, "limit": n + 2})}
        oi = {int(x["timestamp"]): float(x["sumOpenInterest"]) for x in get(f"{FAPI}/futures/data/openInterestHist", {"symbol": "ETHUSDT", "period": iv, "limit": n + 3})}
        sp = {int(x[0]): (2 * float(x[9]) - float(x[5])) for x in get(f"{SPOT}/api/v3/klines", {"symbol": "ETHUSDT", "interval": iv, "limit": n + 2})}
        # OI 스냅샷 시각 = 봉 마감(= 다음 봉 시작) -- 그 봉 끝 값
        oi_end = [oi.get(t + sec * 1000) for t in t0]
        o0 = next((x for x in oi_end if x), None)
        # 고래·리테일 = 풋프린트 봉(5분, 시작 시각 초) → 같은 창으로 묶음
        wb = {}
        for b in fp_bars:
            if b.get("agg") or not b.get("levels"):
                continue
            L = np.asarray(b["levels"], float)
            key = int(b["time"]) // sec * sec * 1000
            a = wb.setdefault(key, np.zeros(4))
            a += [L[:, 3].sum(), L[:, 4].sum(), L[:, 5].sum(), L[:, 6].sum()]
        lq = {}
        for b in lq_bars:
            key = int(datetime.fromisoformat(b["ts"]).timestamp()) // sec * sec * 1000
            a = lq.setdefault(key, [0.0, 0.0])
            a[0] += float(b.get("long_usd") or 0); a[1] += float(b.get("short_usd") or 0)
        imb = lambda a, i: r((a[i] - a[i + 1]) / (a[i] + a[i + 1])) if a is not None and a[i] + a[i + 1] > 0 else None  # noqa: E731
        out[iv] = {"time": [lb.hhmm(t / 1000 + sec) for t in t0], "px_pct": [r((float(x[4]) / px_now - 1) * 100) for x in k],
                   "vol": [r(float(x[5])) for x in k], "perp_net": [r(2 * float(x[9]) - float(x[5])) for x in k],
                   "spot_net": [r(sp.get(t)) for t in t0], "oi_chg_pct": [r((x / o0 - 1) * 100) if x and o0 else None for x in oi_end],
                   "basis_bp": [r(prem.get(t)) for t in t0], "whale_imb": [imb(wb.get(t), 0) for t in t0], "retail_imb": [imb(wb.get(t), 2) for t in t0],
                   "liq_long_usd": [r(lq[t][0]) if t in lq else None for t in t0], "liq_short_usd": [r(lq[t][1]) if t in lq else None for t in t0]}
        out[iv + "_close"] = {t // 1000: float(x[4]) for t, x in zip(t0, k)}
    return out["5m"], out["1h"], out["5m_close"]


def snapshot(mc: dict, bh: dict, cal: dict, news: dict, trend: dict, now: datetime) -> dict:
    tag = lambda q: None if q is None else "low (bottom 10% of 7d)" if q <= .1 else "high (top 10% of 7d)" if q >= .9 else "normal"  # noqa: E731
    oi_all = (mc.get("oi") or {}).get("all") or {}
    z = oi_all.get("z1h")
    flow = mc.get("flow") or {}
    s = {"time_utc": now.strftime("%H:%M %a"), "mid": r(mc.get("mid"), 6),
         "basis_bp": r((mc.get("basis") or {}).get("bp")), "basis_chg_30m_bp": r((mc.get("basis") or {}).get("d30_bp")),
         "basis_pct_rank_7d": r((mc.get("basis") or {}).get("pct7d"), 2), "basis_tag": tag((mc.get("basis") or {}).get("pct7d")),
         "funding_8h_pct": {"binance": r(((mc.get("funding") or {}).get("bn") or 0) * 100), "binance_pct_rank_180d": r((mc.get("funding") or {}).get("bn_pct180"), 2),
                            "okx": r(((mc.get("funding") or {}).get("okx") or 0) * 100), "hyperliquid": r(((mc.get("funding") or {}).get("hl_8h") or 0) * 100)},
         "oi_all_venues": {"chg_1h_pct": r(oi_all.get("d1h_pct")), "chg_24h_pct": r(oi_all.get("d24h_pct")), "z_1h": r(z),
                           "tag": None if z is None else "surging" if z >= 1.5 else "dropping fast" if z <= -1.5 else "normal",
                           "by_venue_1h_pct": {k: r(v) for k, v in (oi_all.get("by_venue") or {}).items()}},
         "long_short_ratio": {k: r(v) for k, v in (mc.get("ls") or {}).items() if k != "age_s"},
         "taker_flow_60m_eth_all_venues": {k: r(v) for k, v in (flow.get("net60_all") or flow.get("net60") or {}).items()},
         "taker_flow_60m_z_vs_same_hour": {k: r(v) for k, v in (flow.get("z60h_all") or flow.get("z60h") or {}).items()},
         "btc_move_bp_recent": r((mc.get("btc") or {}).get("move_bp")),
         "orderbook_depth_pct_rank": {"bid_25bp": r((mc.get("book") or {}).get("bid25_pct"), 2), "ask_25bp": r((mc.get("book") or {}).get("ask25_pct"), 2)},
         "hyperliquid_whale_liq_levels": mc.get("hl_liq"),
         "volume_profile": {k: ({kk: r(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else r(v, 5))
                            for k, v in (mc.get("profile") or {}).items() if k in ("prev_day", "vwap_week", "avwap_hi", "avwap_lo")},
         "options_dvol": {"dvol": r((mc.get("iv_rank") or {}).get("dvol")), "rank_365d": r((mc.get("iv_rank") or {}).get("rank365"), 2)},
         "daily_trend_votes": {"signal_-1to1": r(trend.get("signal")), "ups": trend.get("ups"), "n": trend.get("n_votes")}}
    if bh.get("ok"):
        s["box_hold"] = {k: {"p_hold": round(v["p"] * 100), "typical": round(v["base"] * 100), "half_width_pct": v["half_pct"],
                             "tag": "likely to break (bottom 20%)" if v["p"] < v["q20"] else "likely to hold (top 20%)" if v["p"] > v["q80"] else "normal"}
                         for k, v in bh["scales"].items() if v.get("ok")}
    nxt = []
    for e in cal.get("events", []):
        try:
            dt = datetime.fromisoformat(e["time_utc"])
        except Exception:  # noqa: BLE001
            continue
        if now - timedelta(hours=1) <= dt <= now + timedelta(hours=24) and e.get("importance", "high") == "high":
            nxt.append({"in_min": round((dt - now).total_seconds() / 60), "what": e.get("title_ko") or e.get("category")})
    s["scheduled_events_24h"] = nxt[:6]
    hl = []
    for it in sorted(news.get("items", []), key=lambda x: -(x.get("ts_ms") or 0)):
        if (it.get("relevant") or 0) >= .5 and (it.get("impact") or 0) >= 1.5 and it.get("asset") in ("ETH", "BTC", "macro"):
            hl.append({"min_ago": round((now.timestamp() * 1000 - it["ts_ms"]) / 60000), "title": (it.get("title") or "")[:140],
                       "sentiment_judge": it.get("sentiment"), "impact_0to3": r(it.get("impact"), 2)})
    s["recent_news_6h"] = [h for h in hl if h["min_ago"] <= 360][:4]
    return s


def build(base: str) -> dict:
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    g = lambda p, **q: get(base + p, q or None)  # noqa: E731
    mc, bh, lm = g("/api/market-context", asset="eth"), g("/api/box-hold"), g("/api/liquidation-map", asset="eth")
    t7, t30 = g("/api/liquidation-map/tiers", asset="eth", days=7), g("/api/liquidation-map/tiers", asset="eth", days=30)
    fp = g("/api/footprint", asset="eth", bars=288)
    d3 = (now - timedelta(days=3)).strftime("%Y-%m-%d"); d0 = now.strftime("%Y-%m-%d")
    fpd = g("/api/footprint-daily", asset="eth", since=d3)
    row = 2
    fpc = g("/api/footprint-daily/cells", asset="eth", **{"from": d3, "to": d0, "row": row})
    lq = g("/api/liquidation-5m-history", asset="eth", bars=288)
    cal, news, trend = g("/api/macro-calendar"), g("/api/news"), g("/api/trend", asset="eth")
    w5, w1h, close5 = windows(base, now_ms, fp.get("bars", []), lq.get("bars", []))
    px = float(mc.get("mid") or close5[max(close5)])
    return {"asof_utc": now.strftime("%Y-%m-%d %H:%M"),
            "snapshot": snapshot(mc, bh, cal, news, trend, now),
            "w5": w5, "w1h": w1h, "events": lb.rule_events(w5),
            "liquidation_map": {"note": "estimated liquidation levels from volume-weighted virtual entries (model, not actual positions); above price = shorts, below = longs",
                                "1d": lb.liqmap_summary(lm.get("tier_profile"), px), "7d": lb.liqmap_summary(t7.get("tier_profile"), px),
                                "30d": lb.liqmap_summary(t30.get("tier_profile"), px)},
            "footprint": {"note": "taker volume by price (Binance perp + OKX swap); delta = buy - sell in ETH; whale >= $100k orders, retail < $10k",
                          "1h": lb.footprint_summary(fp.get("bars", []), 12, px, close5),
                          "2h": lb.footprint_summary(fp.get("bars", []), 24, px, close5, group=3),
                          "4h": lb.footprint_summary(fp.get("bars", []), 48, px, close5, group=3),
                          "1d": lb.footprint_daily_summary(fpd.get("cols", {}), fpc.get("cells"), float(fpc.get("row", row)), px)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--out")
    a = ap.parse_args()
    t = time.time()
    brief = build(a.base)
    txt = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
    if a.out:
        Path(a.out).write_text(txt)
    print(f"brief {len(txt):,} bytes in {time.time() - t:.1f}s · sections " + ", ".join(f"{k}={len(json.dumps(v, ensure_ascii=False)):,}" for k, v in brief.items()))
