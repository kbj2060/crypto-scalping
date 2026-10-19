#!/usr/bin/env python3
"""**Haiku 가 청산맵 1d/7d/30d · 풋프린트 1h/2h/4h/1d 요약을 맞게 읽나** — 과거 16 시점, 주장 단위 기계 채점 (2026-10-17).

왜: 사용자 «청산맵 1d/7d/30d 와 풋프린트 1h/2h/4h/1d 도 입력으로» → «그렇게 해줘»(조립기 + 읽기 정확도 시험).
입력 = 라이브 조립기(scripts/live_llm_brief_20261017.py)와 같은 요약 함수(dashboard/llm_brief.py) · 과거 시점 재구성:
  청산맵 = 바이낸스 1h·5m 봉(REST endTime) + OI(openInterestHist) → compute_tier_profile(서버와 같은 호출: 1d=latest_window_5m, 7·30d=1h 창) ·
  풋프린트 1·2·4h = 서버 체결 테이프(binance+okx trade_tape_1s, 읽기 전용, /tmp 에서 실행) → 5분×가격 0.5$ 봉 ·
  풋프린트 1d = 서버 /api/footprint-daily(+cells), T 이전 완결 3일 · 윈도우 = REST(봉·프리미엄·OI·현물) + 테이프 고래/리테일(청산은 없음).
기간: 2026-10-02 ~ 10-15(테이프 09-30~), 16 시점(≥8h, 8 무작위 + 8 직전 2h 움직임 상위 25%).
출력: 한국어 해석 ≤6문장 + facts[...] (청산맵·풋프린트에 대한 기계 채점 사실) — 채점 = 요약 값과 대조.
  python scripts/research_llm_haiku_liqfp_read_20261017.py build | eval | --selfcheck
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard import llm_brief as lb  # noqa: E402
from scripts import live_llm_brief_20261017 as live  # noqa: E402
from scripts.live_liquidation_map_20260824 import compute_tier_profile, latest_window_5m  # noqa: E402

OUT = ROOT / "tmp/haiku_liqfp_20261017"
BDIR = ROOT / "tmp/haiku_liqfp_20261017_inputs"
SSH = ["ssh", "-p", "2222", "llewyn@192.168.1.89"]
N, PER = 16, 8
START, END = "2026-10-02 06:00", "2026-10-15 20:00"

DUMP = r'''
import sqlite3, json, sys
T = json.loads(sys.argv[1]); out = {}
for t in T:
    bars = {}
    for f, sym in (("data/hot/binance_tape.sqlite", "ethusdt"), ("data/hot/okx_tape.sqlite", "ETH-USDT-SWAP")):
        c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        q = """select ts_sec/300*300, price_bin, sum(buy_qty), sum(sell_qty), sum(whale_buy_qty), sum(whale_sell_qty), sum(retail_buy_qty), sum(retail_sell_qty)
               from trade_tape_1s where symbol=? and ts_sec>=? and ts_sec<? group by 1, 2"""
        for row in c.execute(q, (sym, t - 48 * 300, t)):
            bars.setdefault(row[0], {}).setdefault(row[1], [0.0] * 6)
            a = bars[row[0]][row[1]]
            for i in range(6):
                a[i] += row[2 + i] or 0.0
    out[str(t)] = {str(b): [[k] + v for k, v in lv.items()] for b, lv in bars.items()}
print(json.dumps(out))
'''


def kl(interval: str, end_ms: int, n: int, spot: bool = False) -> pd.DataFrame:
    url = f"{live.SPOT}/api/v3/klines" if spot else f"{live.FAPI}/fapi/v1/klines"
    rows = []
    while len(rows) < n:
        lim = min(1500 if not spot else 1000, n - len(rows))
        part = live.get(url, {"symbol": "ETHUSDT", "interval": interval, "limit": lim, "endTime": (rows[0][0] - 1) if rows else end_ms - 1})
        if not part:
            break
        rows = part + rows
    df = pd.DataFrame(rows[-n:], columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades", "taker_buy_base", "tq", "ig"])
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


def tick_from(levels_by_bar: dict, px: float) -> float:
    pb = np.median([int(k) for lv in levels_by_bar.values() for k, *_ in lv])
    return 10 ** round(np.log10(px / pb))          # 가격칸 단위(ETH 0.1$ 등)


def fp_bars(dump: dict, px: float) -> list[dict]:
    """테이프 덤프 {bar: [[price_bin, buy, sell, wb, ws, rb, rs], …]} → 라이브 풋프린트 봉 모양(0.5$ 칸)."""
    tick = tick_from(dump, px)
    out = []
    for b in sorted(dump, key=int):
        agg = {}
        for pbin, *v in dump[b]:
            p = round(pbin * tick * 2) / 2
            a = agg.setdefault(p, [0.0] * 6)
            for i in range(6):
                a[i] += v[i]
        # 라이브 열 순서 [price, buy, sell, whale_buy, whale_sell, retail_buy, retail_sell]
        out.append({"time": int(b), "levels": [[p, a[0], a[1], a[2], a[3], a[4], a[5]] for p, a in sorted(agg.items())]})
    return out


def windows_hist(T_ms: int, fpb: list[dict]) -> tuple[dict, dict, dict]:
    out = {}
    for iv, sec in (("5m", 300), ("1h", 3600)):
        k = kl(iv, T_ms, 24)
        t0 = [int(x) for x in k.open_time]
        px_now = float(k.close.iloc[-1])
        prem = {int(x[0]): float(x[4]) * 1e4 for x in live.get(f"{live.FAPI}/fapi/v1/premiumIndexKlines", {"symbol": "ETHUSDT", "interval": iv, "limit": 30, "endTime": T_ms - 1})}
        oi = {int(x["timestamp"]): float(x["sumOpenInterest"]) for x in live.get(f"{live.FAPI}/futures/data/openInterestHist",
                                                                                  {"symbol": "ETHUSDT", "period": iv, "limit": 30, "endTime": T_ms})}
        sp = kl(iv, T_ms, 24, spot=True)
        spn = {int(a): 2 * b - c for a, b, c in zip(sp.open_time, sp.taker_buy_base, sp.volume)}
        oi_end = [oi.get(t + sec * 1000) for t in t0]
        o0 = next((x for x in oi_end if x), None)
        wb = {}
        for b in fpb:
            L = np.asarray(b["levels"], float)
            a = wb.setdefault(int(b["time"]) // sec * sec * 1000, np.zeros(4))
            a += [L[:, 3].sum(), L[:, 4].sum(), L[:, 5].sum(), L[:, 6].sum()]
        imb = lambda a, i: lb.r((a[i] - a[i + 1]) / (a[i] + a[i + 1])) if a is not None and a[i] + a[i + 1] > 0 else None  # noqa: E731
        out[iv] = {"time": [lb.hhmm(t / 1000 + sec) for t in t0], "px_pct": [lb.r((c / px_now - 1) * 100) for c in k.close],
                   "vol": [lb.r(v) for v in k.volume], "perp_net": [lb.r(2 * a - b) for a, b in zip(k.taker_buy_base, k.volume)],
                   "spot_net": [lb.r(spn.get(t)) for t in t0], "oi_chg_pct": [lb.r((x / o0 - 1) * 100) if x and o0 else None for x in oi_end],
                   "basis_bp": [lb.r(prem.get(t)) for t in t0], "whale_imb": [imb(wb.get(t), 0) for t in t0], "retail_imb": [imb(wb.get(t), 2) for t in t0]}
        if iv == "5m":
            out["close"] = {t // 1000: float(c) for t, c in zip(t0, k.close)}
    return out["5m"], out["1h"], out["close"]


def liqmaps(T_ms: int) -> tuple[dict, float]:
    T = pd.Timestamp(T_ms, unit="ms", tz="UTC")
    h1 = kl("1h", T_ms, 30 * 24 + 2)
    h1c = h1[h1.close_time < T_ms].reset_index(drop=True)
    m5 = kl("5m", T_ms, 460)
    m5 = m5[m5.close_time < T_ms].reset_index(drop=True)
    px = float(m5.close.iloc[-1])
    oih = live.get(f"{live.FAPI}/futures/data/openInterestHist", {"symbol": "ETHUSDT", "period": "1h", "limit": 2, "endTime": T_ms})
    oi_usd = float(oih[-1]["sumOpenInterestValue"]) if oih else None
    win = latest_window_5m(h1c.tail(48).reset_index(drop=True), m5, 24)
    out = {"1d": compute_tier_profile(win[0], px, oi_usd, m5) if win else None}
    for days in (7, 30):
        df = h1[h1.open_time < T_ms].tail(days * 24).reset_index(drop=True)      # 서버 7·30d = 형성 봉 포함
        out[f"{days}d"] = compute_tier_profile(df, float(df.close.iloc[-1]), oi_usd)
    return out, px


FACT_SPEC = """facts: list EVERY claim your interpretation makes about the liquidation map or the footprint, each as one of:
  {"topic":"liqmap","horizon":"1d|7d|30d","what":"nearest_band_dist_pct","side":"above|below","value":number}
  {"topic":"liqmap","horizon":"1d|7d|30d","what":"heavier_side_within_5pct","value":"above|below"}
  {"topic":"liqmap","horizon":"1d|7d|30d","what":"largest_band_peak","side":"above|below","value":price}
  {"topic":"footprint","window":"1h|2h|4h","what":"delta_sign"|"whale_delta_sign"|"retail_delta_sign","value":"buy|sell"}
  {"topic":"footprint","window":"1h|2h|4h","what":"price_vs_value_area","value":"above|inside|below"}
  {"topic":"footprint","window":"1h|2h|4h","what":"pattern","value":<pattern kind as given>,"at":"HH:MM"}
  {"topic":"footprint","window":"1d","date":"MM-DD","what":"close_vs_value_area","value":"above|inside|below"}
  {"topic":"footprint","window":"1d","date":"MM-DD","what":"delta_sign","value":"buy|sell"}
  {"topic":"footprint","window":"1d","date":"MM-DD","what":"bigger_liquidation_side","value":"long|short"}"""
PROMPT = """You will interpret the ETH perpetual futures market (Binance ETHUSDT) at several independent snapshots (different days, shuffled).
Each snapshot has: w5 (last 2h, 5-minute bars), w1h (last 24h, 1-hour bars), rule-detected events, a liquidation map summary
(1d/7d/30d: estimated liquidation bands above price = shorts, below = longs; model estimates, not actual positions) and a footprint summary
(1h/2h/4h windows of 5-minute taker volume by price; 1d = last complete days). Times are UTC bar closes.
For EACH snapshot write a Korean interpretation (at most 6 sentences) covering the recent flow AND what the liquidation map and footprint show.
Rules: cite numbers from the data; never invent data; do NOT predict future direction; the liquidation map and footprint are maps, not signals.
""" + FACT_SPEC + """
Output ONLY one JSON array: [{"id": "...", "interpretation": "...", "facts": [...]}, ...]

Snapshots:
"""


def pick_times() -> list[int]:
    k = kl("5m", int(pd.Timestamp(END, tz="UTC").value // 10**6), 4500)
    k = k[k.timestamp >= pd.Timestamp(START, tz="UTC")].reset_index(drop=True)
    k["mv2h"] = np.log(k.close).diff(24).abs()
    k = k.dropna()
    rng = np.random.default_rng(20261017)
    chosen: list[int] = []
    def take(pool, upto):
        for i in rng.permutation(len(pool)):
            t = int(pool.close_time.iloc[i]) + 1                    # 결정 = 봉 마감 직후
            if all(abs(t - u) >= 8 * 3600_000 for u in chosen):
                chosen.append(t)
                if len(chosen) >= upto:
                    return
    take(k[k.mv2h >= k.mv2h.quantile(.75)], PER); take(k, N)
    return sorted(chosen)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True); BDIR.mkdir(parents=True, exist_ok=True)
    T = pick_times()
    (OUT / "dump.py").write_text(DUMP)
    subprocess.run(["scp", "-q", "-P", "2222", str(OUT / "dump.py"), "llewyn@192.168.1.89:/tmp/llmfp_dump.py"], check=True)
    res = subprocess.run(SSH + [f"cd ~/crypto-scalping && nice -n 10 ~/miniconda3/envs/quant_ai/bin/python /tmp/llmfp_dump.py '{json.dumps([t // 1000 for t in T])}'"],
                         capture_output=True, text=True, check=True, timeout=900)
    dumps = json.loads(res.stdout)
    base = "http://127.0.0.1:18787"
    items, truth = {}, {}
    for j, t in enumerate(T):
        sid = f"f{j:02d}"
        lm, px = liqmaps(t)
        fpb = fp_bars(dumps[str(t // 1000)], px)
        w5, w1h, close5 = windows_hist(t, fpb)
        day = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        d_from, d_to = (day - timedelta(days=3)).strftime("%Y-%m-%d"), (day - timedelta(days=1)).strftime("%Y-%m-%d")
        fpd = live.get(f"{base}/api/footprint-daily", {"asset": "eth", "since": d_from})
        cols = {k: v[:3] for k, v in fpd["cols"].items()} if fpd["cols"]["d"][:3][-1] == d_to else {k: [x for x, d in zip(v, fpd["cols"]["d"]) if d <= d_to][-3:] for k, v in fpd["cols"].items()}
        cols["part"] = [0] * len(cols["d"])
        cells = live.get(f"{base}/api/footprint-daily/cells", {"asset": "eth", "from": d_from, "to": d_to, "row": 2})
        brief = {"time_utc": day.strftime("%H:%M %a"), "w5": w5, "w1h": w1h, "events": lb.rule_events(w5),
                 "liquidation_map": {k: lb.liqmap_summary(v, px) for k, v in lm.items()},
                 "footprint": {"1h": lb.footprint_summary(fpb, 12, px, close5), "2h": lb.footprint_summary(fpb, 24, px, close5, group=3),
                               "4h": lb.footprint_summary(fpb, 48, px, close5, group=3),
                               "1d": lb.footprint_daily_summary(cols, cells.get("cells"), 2.0, px)}}
        truth[sid] = {"T": t, **{k: brief[k] for k in ("liquidation_map", "footprint")}}
        items[sid] = {"id": sid, **brief}
        print(sid, day, f"{len(json.dumps(brief)):,}B", flush=True)
    (OUT / "truth.json").write_text(json.dumps(truth))
    order = list(items); np.random.default_rng(5).shuffle(order)
    for b in range(0, N, PER):
        (BDIR / f"batch_{b // PER}.txt").write_text(PROMPT + "\n".join(json.dumps(items[s], separators=(",", ":"), ensure_ascii=False) for s in order[b:b + PER]))
    print("→", sorted(p.name for p in BDIR.glob("*.txt")))


def check(f: dict, tr: dict) -> bool | None:
    """사실 하나를 요약 값과 대조. None = 형식 밖."""
    try:
        if f["topic"] == "liqmap":
            s = tr["liquidation_map"][f["horizon"]]
            if f["what"] == "heavier_side_within_5pct":
                w = s["within_5pct"]
                return ("above" if w["above"] > w["below"] else "below") == f["value"]
            rows = s["above_short_liq" if f["side"] == "above" else "below_long_liq"]
            if not rows:
                return False
            if f["what"] == "nearest_band_dist_pct":
                return abs(float(f["value"]) - rows[0]["dist_pct"]) <= max(0.05, 0.1 * abs(rows[0]["dist_pct"]))
            if f["what"] == "largest_band_peak":
                big = max(rows, key=lambda x: x["size"])
                return abs(float(f["value"]) / big["peak"] - 1) <= 0.003
            return None
        fp = tr["footprint"]
        if f["window"] == "1d":
            day = next((d for d in fp["1d"] if d["date"] == f.get("date")), None)
            if day is None:
                return False
            if f["what"] == "close_vs_value_area":
                return day.get("close_vs_value_area") == f["value"]
            if f["what"] == "delta_sign":
                return ("buy" if day["delta_eth"] > 0 else "sell") == f["value"]
            if f["what"] == "bigger_liquidation_side":
                return ("long" if day["liq_long_usd_m"] > day["liq_short_usd_m"] else "short") == f["value"]
            return None
        w = fp[f["window"]]
        if f["what"] in ("delta_sign", "whale_delta_sign", "retail_delta_sign"):
            v = w[f["what"].replace("_sign", "")]
            return ("buy" if v > 0 else "sell") == f["value"]
        if f["what"] == "price_vs_value_area":
            return (w["price_vs_value_area"] or "").split("_")[0] == str(f["value"]).split("_")[0]   # «inside» · «inside_value_area» 둘 다(요약 원문 표기)
        if f["what"] == "pattern":
            return any(m["kind"] == f["value"] and m["at"] == f.get("at") for m in w["pattern_marks"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def evaluate() -> None:
    truth = json.loads((OUT / "truth.json").read_text())
    ans = {}
    for f in sorted((OUT / "out").glob("batch_*.json")):
        for a in json.loads(f.read_text()):
            ans[a["id"]] = a
    rows = []
    for sid, tr in truth.items():
        for f in ans.get(sid, {}).get("facts", []):
            key = f.get("topic", "?") + ":" + (f.get("horizon") or f.get("window") or "?") + ":" + f.get("what", "?")
            rows.append({"id": sid, "key": key, "topic": f.get("topic"), "ok": check(f, tr)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "scored.csv", index=False)
    print(f"시점 {R.id.nunique()} · 사실 {len(R)} · 형식 밖 {R.ok.isna().sum()}")
    g = R.dropna().groupby("topic").ok.agg(["mean", "size"]); print(g.round(3))
    print(R.dropna().groupby("key").ok.agg(["mean", "size"]).round(3).to_string())
    bad = R[R.ok == False]  # noqa: E712
    print("틀린 사실 표본:", bad.key.value_counts().head(8).to_dict())


def _selfcheck() -> None:
    tr = {"liquidation_map": {"1d": {"above_short_liq": [{"dist_pct": 1.5, "size": 4, "peak": 2448}, {"dist_pct": 3.5, "size": 9, "peak": 2496}],
                                     "below_long_liq": [{"dist_pct": -0.7, "size": 3, "peak": 2392}], "within_5pct": {"above": 9, "below": 3}}},
          "footprint": {"1h": {"delta": -5, "whale_delta": 3, "retail_delta": -1, "price_vs_value_area": "inside_value_area",
                               "pattern_marks": [{"at": "17:20", "kind": "sell_exhaustion_at_new_low"}]},
                        "1d": [{"date": "10-15", "delta_eth": -1, "liq_long_usd_m": 117, "liq_short_usd_m": 17, "close_vs_value_area": "above"}]}}
    T, F = True, False
    cases = [({"topic": "liqmap", "horizon": "1d", "what": "nearest_band_dist_pct", "side": "above", "value": 1.5}, T),
             ({"topic": "liqmap", "horizon": "1d", "what": "nearest_band_dist_pct", "side": "below", "value": -1.7}, F),
             ({"topic": "liqmap", "horizon": "1d", "what": "heavier_side_within_5pct", "value": "above"}, T),
             ({"topic": "liqmap", "horizon": "1d", "what": "largest_band_peak", "side": "above", "value": 2496}, T),
             ({"topic": "footprint", "window": "1h", "what": "delta_sign", "value": "sell"}, T),
             ({"topic": "footprint", "window": "1h", "what": "whale_delta_sign", "value": "sell"}, F),
             ({"topic": "footprint", "window": "1h", "what": "price_vs_value_area", "value": "inside"}, T),
             ({"topic": "footprint", "window": "1h", "what": "pattern", "value": "sell_exhaustion_at_new_low", "at": "17:20"}, T),
             ({"topic": "footprint", "window": "1d", "date": "10-15", "what": "bigger_liquidation_side", "value": "long"}, T),
             ({"topic": "footprint", "window": "1d", "date": "10-15", "what": "close_vs_value_area", "value": "inside"}, F),
             ({"topic": "liqmap", "horizon": "7d", "what": "heavier_side_within_5pct", "value": "above"}, None)]
    for f, want in cases:
        assert check(f, tr) is want, (f, check(f, tr), want)
    d = {"1700000000": [[24141, 1, 2, 0, 0, 1, 0], [24146, 3, 0, 1, 0, 0, 0]]}
    b = fp_bars(d, 2414.0)
    assert b[0]["levels"] == [[2414.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0], [2414.5, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0]], b
    print("selfcheck OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        _selfcheck()
    elif a[:1] == ["build"]:
        build()
    elif a[:1] == ["eval"]:
        evaluate()
