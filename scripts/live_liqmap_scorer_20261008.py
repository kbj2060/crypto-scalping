"""청산지도 전진 점수 기록기 (2026-10-08, 사용자 «점수 기록기 만들고 대시보드 알림에 넣어줘»).

매시 07분(cron): 4시간 전 정시 o 마다 «그때 만들 수 있던 청산지도»와 [o, o+4h) 실측 청산을 한 줄로 data/live/liqmap_score.jsonl 에 적는다.
  지도 변형 = A 현행(1일·6단·거래량) · B 14단 · D 테이커 쪽 나눔 · F 반감기 48h  (research_eth_liqmap_calibrate_realized_20261008 후보 중
  TRAIN 상위·대조용. 10-08 사전등록 불통과 → 표본을 쌓아 다시 판정: 판정 = dashboard/verdict_calendar.json «liqmap_fwd»).
  실측 = 바이낸스·OKX(ETH-USDT-SWAP)·Bybit 청산(hot), 위치 = 청산 시각 바이낸스 마크(1초) -- 파산가 필드는 안 쓴다.
  지도·실측 모두 기준가 cp(직전 봉 종가) 대비 0.2% 상대 칸. 지도는 쪽마다 합 1, ±SPAN 칸만.
  hot 보존(8일) 안이면 빠진 시각을 채운다(멱등: 이미 적힌 origin 은 건너뜀). 바이낸스 REST 는 시간당 1회(서버에서만).
판정: python scripts/research_eth_liqmap_calibrate_realized_20261008.py --forward data/live/liqmap_score.jsonl
실행: python scripts/live_liqmap_scorer_20261008.py [--selftest]
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_liqmap_venue_leverage_validation_20261008 as V  # noqa: E402

OUT = ROOT / "data/live/liqmap_score.jsonl"
STATUS = ROOT / "data/live/liqmap_score_status.json"     # 알림 센터가 읽는 요약(판정 대상 = FWD_SINCE 이후 줄 수)
FWD_SINCE = "2026-10-08T00:00:00+00:00"
HOT = ROOT / "data/hot"
H, W, SPAN, BACK_H = 4, 24, 60, 168
VENUES = ("binance", "okx", "bybit")


def variants(k: pd.DataFrame) -> dict:
    vol, qv, tbq = k.volume.to_numpy(), k.qv.to_numpy(), k.tbq.to_numpy()
    return {"A": dict(tiers=V.T6, vol=vol), "B": dict(tiers=V.T14, vol=vol),
            "D": dict(tiers=V.T6, vol=(tbq, qv - tbq)), "F": dict(tiers=V.T6, vol=vol, halflife=48.0)}


def klines(limit: int) -> pd.DataFrame:
    u = f"https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1h&limit={limit}"
    raw = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "liqmap-scorer"}), timeout=20))
    k = pd.DataFrame({"t": [int(r[0]) for r in raw], "high": [float(r[2]) for r in raw], "low": [float(r[3]) for r in raw],
                      "close": [float(r[4]) for r in raw], "volume": [float(r[5]) for r in raw], "qv": [float(r[7]) for r in raw],
                      "tbq": [float(r[10]) for r in raw], "ct": [int(r[6]) for r in raw]})
    k = k[k.ct < time.time() * 1000].drop(columns="ct").reset_index(drop=True)       # 진행 중인 봉은 버린다
    k["timestamp"] = pd.to_datetime(k.t, unit="ms", utc=True)
    return k


def _q(db: str, sql: str, args) -> pd.DataFrame:
    c = sqlite3.connect(f"file:{HOT / db}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, c, params=args)
    finally:
        c.close()


def hot_liqs(t0: int, t1: int) -> pd.DataFrame:
    a = (t0, t1)
    d = pd.concat([
        _q("binance_ctx.sqlite", "SELECT ts_ms, side, qty FROM liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", a).assign(venue="binance"),
        _q("okx_ctx.sqlite", "SELECT ts_ms, pos_side AS side, sz_base AS qty FROM okx_liquidations WHERE inst_id='ETH-USDT-SWAP' AND ts_ms>=? AND ts_ms<?", a).assign(venue="okx"),
        _q("bybit_liq.sqlite", "SELECT ts_ms, side, qty FROM bybit_liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", a).assign(venue="bybit"),
    ], ignore_index=True)
    d["side"] = d.side.map({"long": "long", "short": "short", "Buy": "long", "Sell": "short"})   # Bybit Buy = 롱 청산
    m = _q("binance_ctx.sqlite", "SELECT ts_ms, mark FROM mark_price_1s WHERE symbol='ethusdt' AND ts_ms>=? AND ts_ms<?", (t0 - 5000, t1))
    d = pd.merge_asof(d.dropna(subset=["side"]).sort_values("ts_ms"), m.sort_values("ts_ms"), on="ts_ms", direction="backward", tolerance=5000)
    d = d.dropna(subset=["mark"])
    d["usd"] = d.qty * d.mark
    return d


def _norm(mp: dict) -> dict:
    out = {}
    for s in ("long", "short"):
        tot = sum(v for (a, o), v in mp.items() if a == s and abs(o) <= SPAN) or 1.0
        out[s] = {str(o): round(v / tot, 6) for (a, o), v in sorted(mp.items()) if a == s and abs(o) <= SPAN and v / tot >= 1e-6}
    return out


def row_for(k: pd.DataFrame, i: int, liq: pd.DataFrame) -> dict:
    """기준 시각 k.t[i] 의 한 줄: 지도(봉 < i) + 실측([i, i+H))."""
    maps, cp = {}, None
    for name, sp in variants(k).items():
        cp, mp = V.liq_map(k, i, W, sp["tiers"], "splice", sp["vol"], halflife=sp.get("halflife", V.LM.RECENCY_HALFLIFE_HOURS))
        maps[name] = _norm(mp)
    rng, r = V.realized(k, liq, i, H, cp, VENUES)
    return {"origin": k.timestamp.iloc[i].isoformat(), "cp": cp, "H": H,
            "long_lo": rng["long"].start, "short_hi": rng["short"].stop - 1,
            "real": {s: {str(o): round(u, 2) for (a, o), u in sorted(r.items()) if a == s} for s in ("long", "short")},
            "n_liq": int(((liq.ts_ms >= k.t.iloc[i]) & (liq.ts_ms < k.t.iloc[i] + H * 3600_000)).sum()), "maps": maps}


def done_origins() -> set[str]:
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as fh:
        return {json.loads(line)["origin"] for line in fh if line.strip()}


def write_status() -> None:
    o = sorted(done_origins())
    fwd = [x for x in o if x >= FWD_SINCE]
    STATUS.write_text(json.dumps({"n_fwd": len(fwd), "days_fwd": len({x[:10] for x in fwd}), "last_origin": o[-1] if o else None,
                                  "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), encoding="utf-8")


def main() -> None:
    k = klines(W + H + BACK_H + 2)
    done = done_origins()
    todo = [i for i in range(W, len(k) - H + 1) if k.timestamp.iloc[i].isoformat() not in done]
    if not todo:
        write_status(); print(time.strftime("%F %T"), "새 기준 시각 없음"); return
    liq = hot_liqs(int(k.t.iloc[todo[0]]), int(k.t.iloc[todo[-1]]) + H * 3600_000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as fh:
        for i in todo:
            fh.write(json.dumps(row_for(k, i, liq), ensure_ascii=False) + "\n")
    write_status()
    print(time.strftime("%F %T"), f"적음 {len(todo)}줄 ({k.timestamp.iloc[todo[0]]} ~ {k.timestamp.iloc[todo[-1]]}) · 청산 {len(liq)}건")


def selftest() -> None:
    ts = pd.date_range("2026-10-01", periods=W + H, freq="1h", tz="UTC")
    px = 100 + np.sin(np.arange(len(ts)) / 3.0)
    k = pd.DataFrame({"timestamp": ts, "t": ts.astype("int64") // 10**6, "high": px + 0.5, "low": px - 0.5, "close": px,
                      "volume": 10.0, "qv": 1000.0, "tbq": 600.0})
    cp = float(px[W - 1]); t0 = int(k.t.iloc[W])
    liq = pd.DataFrame({"ts_ms": [t0 + 1000, t0 + 2000, t0 - 1000, t0 + H * 3600_000], "side": ["long", "short", "long", "long"],
                        "usd": [50.0, 70.0, 999.0, 999.0], "mark": [cp * 0.995, cp * 1.003, cp * 0.99, cp * 0.99], "venue": "binance"})
    r = row_for(k, W, liq)
    assert r["real"]["long"] == {"-3": 50.0} and r["real"]["short"] == {"1": 70.0}   # 창 밖 두 건은 안 센다
    assert r["n_liq"] == 2 and set(r["maps"]) == {"A", "B", "D", "F"}
    assert all(abs(sum(r["maps"]["A"][s].values()) - 1) < 1e-3 for s in ("long", "short") if r["maps"]["A"][s])
    assert r["long_lo"] <= -1 and r["short_hi"] >= 0
    print("selftest OK -- 창 경계([o, o+4h))·쪽·상대 칸 · 지도 쪽마다 합 1 · 변형 4")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
