"""추정 청산맵 변형을 실측 청산으로 검증 (2026-10-08, 사용자 «바이낸스+OKX 로 청산맵? 레버리지를 다양하게 하면 꽉 차지 않나? 실제 청산 데이터로 검증»).

질문: 청산맵이 «가격이 실제로 지나간 구간 안에서» 청산이 많이 터진 가격을 미리 가리키는가 -- 그리고 아래 변형이 그걸 낫게 하는가.
  창(1·7·30일) · 레버리지(6단 현행 / 14단 / 2~125배 로그 40단) · 진입가(현행 = 롱 중간값·숏 종가 / 봉 저가~고가에 고르게) ·
  거래량 가중(바이낸스 / 바이낸스+OKX / OKX).
설계(인과):
  - 기준 시각 i(매 정시) 지도 = i 이전에 닫힌 1시간봉만(생존 필터도 그 안). 기준가 cp = 직전 봉 종가.
  - 실측 = [i, i+H) 의 바이낸스·OKX·Bybit 청산 $. 🔴Bybit·OKX 가격 필드는 파산가(마크 ±수십 bp)라 위치는 «청산 시각의 바이낸스 마크가(1초)»로 잡는다.
  - 칸 = cp 대비 0.2% 상대 칸. 롱 쪽 = 그 창 최저가~cp 사이 지나간 칸, 숏 쪽 = cp~최고가.
  - 지표 = 지나간 칸들 안에서 (지도 밀도, 실측 청산$) 스피어만 IC, 기준 시각·쪽마다 → 평균. 상위 20% 칸이 잡은 실측 $ 몫.
  - 🔴거리 교란(현재가 가까운 칸이 오래 머물러 더 많이 터진다 · 지도도 거리 모양이 있다) → 대조군 = 다른 시각(±48h 밖)의 지도를
    상대 칸 그대로 옮겨 붙인 «시간 섞은 지도» 30회. 초과 IC = 진짜 − 섞음 = «그 시각 고유 정보». 일 블록 부트스트랩 CI.
실행: python scripts/research_eth_liqmap_venue_leverage_validation_20261008.py [--selftest]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import live_liquidation_map_20260824 as LM  # noqa: E402

MAIN = Path("/home/kbj20/crypto-scalping")
WORK = MAIN / "tmp/liqmap_validation_20261008"
OKX_BACKUP = Path.home() / "backups/crypto-scalping-server/data/lake/okx/liquidations/coin=ETH"
EVAL_BIN = 0.002
T6 = (10, 20, 25, 50, 75, 100)
T14 = (2, 3, 5, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100, 125)
TCONT = tuple(np.round(np.geomspace(2, 125, 40), 2))


# ---------------------------------------------------------------- 데이터
def k1h_binance() -> pd.DataFrame:
    from build_eth_liq_heat_daily_20261008 import load_1h
    k = load_1h()
    return k[k.timestamp >= "2026-08-01"].reset_index(drop=True)


def k1h_okx() -> pd.DataFrame:
    f = WORK / "okx_1h.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows, after = [], ""
    while True:   # OKX REST(바이낸스 아님) history-candles, 100개씩 과거로
        u = "https://www.okx.com/api/v5/market/history-candles?instId=ETH-USDT-SWAP&bar=1H&limit=100" + (f"&after={after}" if after else "")
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=20))["data"]
        if not d:
            break
        rows += d; after = d[-1][0]
        if int(after) < pd.Timestamp("2026-08-01", tz="UTC").value // 10**6:
            break
    k = pd.DataFrame({"t": [int(r[0]) for r in rows], "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                      "close": [float(r[4]) for r in rows], "volume": [float(r[6]) for r in rows], "confirm": [r[8] for r in rows]})
    k = k[k.confirm == "1"].drop(columns="confirm").drop_duplicates("t").sort_values("t").reset_index(drop=True)
    k["timestamp"] = pd.to_datetime(k.t, unit="ms", utc=True)
    k.to_parquet(f); return k


def liquidations() -> pd.DataFrame:
    a = pd.read_parquet(WORK / "eth_liq_all_20261008.parquet")
    b = pd.concat([pd.read_parquet(p) for p in sorted(OKX_BACKUP.glob("*/*.parquet"))])
    b = b[b.inst_id == "ETH-USDT-SWAP"].drop(columns="side").rename(columns={"pos_side": "side", "sz_base": "qty", "bk_px": "price"})[["ts_ms", "side", "qty", "price"]].assign(venue="okx")
    d = pd.concat([a, b], ignore_index=True).drop_duplicates(["venue", "ts_ms", "side", "qty", "price"])
    d["side"] = d.side.map({"long": "long", "short": "short", "Buy": "long", "Sell": "short"})   # Bybit Buy = 롱 청산(메모 bybit_liquidation_collector)
    m = pd.read_parquet(WORK / "eth_mark_1s_20261008.parquet").sort_values("ts_ms")
    d = pd.merge_asof(d.sort_values("ts_ms"), m, on="ts_ms", direction="backward", tolerance=5000).dropna(subset=["mark"])
    d["usd"] = d.qty * d.mark
    return d


# ---------------------------------------------------------------- 지도
def liq_map(k: pd.DataFrame, i: int, W: int, tiers, entry: str, vol, tier_w=None, halflife: float = LM.RECENCY_HALFLIFE_HOURS,
            by_tier: bool = False) -> tuple[float, dict]:
    """봉 [i-W, i) 로 만든 지도 → (cp, {('long'|'short', 상대칸): 가중}).
    vol = 배열(양쪽 같은 가중) 또는 (롱 배열, 숏 배열) · tier_w = 레버리지별 가중(기본 균등) · by_tier = 키 앞에 레버리지 번호."""
    vl, vs = vol if isinstance(vol, tuple) else (vol, vol)
    tw = np.full(len(tiers), 1 / len(tiers)) if tier_w is None else np.asarray(tier_w, float)
    d = k.iloc[i - W:i]
    hi, lo, cl = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
    n, cp = len(d), float(cl[-1])
    age = (n - 1 - np.arange(n)).astype(float)
    rec = np.exp(-age / halflife)
    wl, ws = vl[i - W:i] * rec, vs[i - W:i] * rec
    fmin = np.full(n, np.inf); fmax = np.full(n, -np.inf)
    fmin[:-1] = np.minimum.accumulate(lo[::-1])[::-1][1:]; fmax[:-1] = np.maximum.accumulate(hi[::-1])[::-1][1:]
    if entry == "range":
        q = np.linspace(0, 1, 5)
        el = es = lo[:, None] + (hi - lo)[:, None] * q[None, :]; ew = 1 / len(q)
    else:
        el, es, ew = ((hi + lo) / 2)[:, None], cl[:, None], 1.0
    out: dict = {}
    for ti, L in enumerate(tiers):
        for side, ent, w in (("long", el, wl * tw[ti]), ("short", es, ws * tw[ti])):
            e = ent * ((1 - 1 / L + LM.MAINTENANCE_MARGIN_RATE) if side == "long" else (1 + 1 / L - LM.MAINTENANCE_MARGIN_RATE))
            alive = (e < fmin[:, None]) if side == "long" else (e > fmax[:, None])
            off = np.floor((e / cp - 1) / EVAL_BIN).astype(int)
            ok = alive & ((off < 0) if side == "long" else (off >= 0))
            ww = np.broadcast_to(w[:, None] * ew * ent, e.shape)[ok]
            for o, v in zip(off[ok], ww):
                key = (ti, side, int(o)) if by_tier else (side, int(o))
                out[key] = out.get(key, 0.0) + float(v)
    return cp, out


# ---------------------------------------------------------------- 평가
def realized(k: pd.DataFrame, liq: pd.DataFrame, i: int, H: int, cp: float, venues) -> tuple[dict, dict]:
    t0 = int(k.t.iloc[i]); t1 = t0 + H * 3600_000
    p = k.iloc[i:i + H]
    rng = {"long": range(int(np.floor((p.low.min() / cp - 1) / EVAL_BIN)), 0),
           "short": range(0, int(np.floor((p.high.max() / cp - 1) / EVAL_BIN)) + 1)}
    x = liq[(liq.ts_ms >= t0) & (liq.ts_ms < t1) & liq.venue.isin(venues)]
    off = np.floor((x.mark / cp - 1) / EVAL_BIN).astype(int)
    r: dict = {}
    for s, o, u in zip(x.side, off, x.usd):
        if (s == "long" and o < 0) or (s == "short" and o >= 0):
            r[(s, int(o))] = r.get((s, int(o)), 0.0) + u
    return rng, r


def score(mp: dict, rng: dict, r: dict) -> list[tuple[str, float, float]]:
    res = []
    for s in ("long", "short"):
        bins = list(rng[s])
        y = np.array([r.get((s, b), 0.0) for b in bins]); x = np.array([mp.get((s, b), 0.0) for b in bins])
        if len(bins) < 4 or y.sum() <= 0 or np.ptp(x) == 0:
            continue
        ic = spearmanr(x, y).statistic
        top = x >= np.quantile(x, 0.8)
        res.append((s, float(ic), float(y[top].sum() / y.sum())))
    return res


def run(k, liq, origins, H, venues, tiers, entry, W, vol, nshuf=30, seed=0, maps=None, **kw):
    maps = maps or {i: liq_map(k, i, W, tiers, entry, vol, **kw) for i in origins}
    rows, rs = [], np.random.default_rng(seed)
    for i in origins:
        cp, mp = maps[i]
        rng, r = realized(k, liq, i, H, cp, venues)
        sc = score(mp, rng, r)
        if not sc:
            continue
        far = [j for j in origins if abs(j - i) >= 48]
        sh = [score(maps[j][1], rng, r) for j in rs.choice(far, size=nshuf, replace=False)]
        for s, ic, cap in sc:
            null = [x for z in sh for x in z if x[0] == s]
            rows.append({"i": i, "day": str(k.timestamp.iloc[i].date()), "side": s, "ic": ic, "cap": cap,
                         "ic_null": np.mean([x[1] for x in null]), "cap_null": np.mean([x[2] for x in null])})
    return pd.DataFrame(rows), maps


def boot(df: pd.DataFrame, col: str, n=2000, seed=1) -> tuple[float, float, float]:
    g = df.groupby("day")[col].agg(["sum", "count"]); rs = np.random.default_rng(seed)
    idx = rs.integers(0, len(g), size=(n, len(g)))
    est = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return float(df[col].mean()), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def fill_share(maps: dict) -> float:
    """현재가 ±8% 안 0.2% 칸 중 값이 있는 몫(«꽉 찼나»)."""
    lim = int(0.08 / EVAL_BIN)
    return float(np.mean([sum(1 for (s, o) in mp if -lim <= o < lim) / (2 * lim) for _, mp in maps.values()]))


def selftest() -> None:
    """현행 설정(6단·중간/종가·바이낸스)이 라이브 compute_tier_profile 과 같은 롱/숏 몫을 내는지 + IC 부호."""
    ts = pd.date_range("2026-10-01", periods=60, freq="1h", tz="UTC")
    px = 100 + 3 * np.sin(np.arange(60) / 4.0)
    k = pd.DataFrame({"timestamp": ts, "t": ts.astype("int64") // 10**6, "high": px + 0.4, "low": px - 0.4, "close": px,
                      "volume": np.full(60, 10.0) + np.arange(60)})
    cp, mp = liq_map(k, 60, 24, T6, "splice", k.volume.to_numpy())
    tp = LM.compute_tier_profile(k.iloc[36:60].reset_index(drop=True), cp)
    vals = np.sum([t["values"] for t in tp["tiers"]], axis=0)
    prices = (tp["lo"] + np.arange(len(vals))) * tp["bin_width"]
    long_live = vals[prices < cp].sum() / vals.sum()
    long_mine = sum(v for (s, _), v in mp.items() if s == "long") / sum(mp.values())
    assert abs(long_live - long_mine) < 0.02, (long_live, long_mine)
    rng = {"long": range(-5, 0), "short": range(0, 5)}
    r = {("long", -3): 10.0, ("long", -1): 1.0}
    m2 = {("long", -3): 5.0, ("long", -1): 1.0, ("long", -2): 0.1}
    assert score(m2, rng, r)[0][1] > 0.5   # 맞는 칸을 가리키면 IC 양수
    print(f"selftest OK -- 롱 몫 라이브 {long_live:.3f} vs 이 지도 {long_mine:.3f} · IC 부호")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    kb, ko, liq = k1h_binance(), k1h_okx(), liquidations()
    k = kb.merge(ko[["t", "volume"]].rename(columns={"volume": "vol_okx"}), on="t", how="left")
    print("1h 봉", len(k), k.timestamp.min(), "~", k.timestamp.max(), "· OKX 결측", int(k.vol_okx.isna().sum()))
    k["vol_okx"] = k.vol_okx.fillna(k.volume * k.vol_okx.sum() / k.volume[k.vol_okx.notna()].sum())
    print("거래량 상관(1h, 바이낸스 vs OKX)", round(float(np.corrcoef(k.volume, k.vol_okx)[0, 1]), 3),
          "· OKX/바이낸스", round(float(k.vol_okx.sum() / k.volume.sum()), 3))
    print("청산", liq.groupby("venue").agg(n=("usd", "size"), usd=("usd", "sum"), t0=("ts_ms", "min")).assign(t0=lambda x: pd.to_datetime(x.t0, unit="ms")))
    m = pd.read_parquet(WORK / "eth_mark_1s_20261008.parquet"); m["min"] = m.ts_ms // 60000
    r1 = m.groupby("min").mark.agg(["first", "last"]); r1 = (r1["last"] / r1["first"] - 1)
    liq["ret1m"] = r1.reindex(liq.ts_ms // 60000).to_numpy()
    print("부호 점검 -- 청산 분의 1분 수익률(bp), 롱<0·숏>0 이어야:", (liq.groupby(["venue", "side"]).ret1m.mean() * 1e4).round(1).to_dict())

    t_lo = int(max(liq.ts_ms.min(), pd.Timestamp("2026-09-21", tz="UTC").value // 10**6))
    origins = [i for i in range(720, len(k) - 24) if int(k.t.iloc[i]) >= t_lo]
    print("기준 시각", len(origins), k.timestamp.iloc[origins[0]], "~", k.timestamp.iloc[origins[-1]])
    vb, vbo, vo = k.volume.to_numpy(), (k.volume + k.vol_okx).to_numpy(), k.vol_okx.to_numpy()
    variants = [
        ("현행 1일·6단·바이낸스", 24, T6, "splice", vb),
        ("7일", 168, T6, "splice", vb),
        ("30일", 720, T6, "splice", vb),
        ("1일·14단", 24, T14, "splice", vb),
        ("1일·연속40단", 24, TCONT, "splice", vb),
        ("1일·봉범위 진입", 24, T6, "range", vb),
        ("1일·바이낸스+OKX", 24, T6, "splice", vbo),
        ("1일·OKX만", 24, T6, "splice", vo),
        ("30일·연속40단·봉범위", 720, TCONT, "range", vb),
        ("30일·연속40단·봉범위·바+OKX", 720, TCONT, "range", vbo),
    ]
    out = []
    for H in (4, 24):
        for tgt, venues in (("전체", ("binance", "okx", "bybit")), ("바이낸스", ("binance",)), ("OKX", ("okx",))):
            if H == 24 and tgt != "전체":
                continue
            for name, W, tiers, entry, vol in variants:
                if tgt != "전체" and name not in ("현행 1일·6단·바이낸스", "1일·바이낸스+OKX", "1일·OKX만"):
                    continue
                df, maps = run(k, liq, origins, H, venues, tiers, entry, W, vol)
                df["ex"] = df.ic - df.ic_null; df["cex"] = df.cap - df.cap_null
                ic = boot(df, "ic"); ex = boot(df, "ex"); cex = boot(df, "cex")
                row = {"H": H, "target": tgt, "variant": name, "n": len(df), "fill": round(fill_share(maps), 3),
                       "IC": round(ic[0], 3), "IC_null": round(df.ic_null.mean(), 3),
                       "excess": round(ex[0], 3), "ex_lo": round(ex[1], 3), "ex_hi": round(ex[2], 3),
                       "top20_cap": round(df.cap.mean(), 3), "cap_ex": round(cex[0], 3), "cap_lo": round(cex[1], 3), "cap_hi": round(cex[2], 3),
                       "ex_long": round(df[df.side == "long"].ex.mean(), 3), "ex_short": round(df[df.side == "short"].ex.mean(), 3)}
                out.append(row); print(row, flush=True)
    pd.DataFrame(out).to_csv(WORK / "results.csv", index=False)
    print("저장", WORK / "results.csv")


def paired() -> None:
    """변형 − 현행, 같은 기준 시각·쪽끼리 짝지은 초과 IC 차이(일 블록 CI)."""
    kb, ko, liq = k1h_binance(), k1h_okx(), liquidations()
    k = kb.merge(ko[["t", "volume"]].rename(columns={"volume": "vol_okx"}), on="t", how="left")
    t_lo = int(max(liq.ts_ms.min(), pd.Timestamp("2026-09-21", tz="UTC").value // 10**6))
    origins = [i for i in range(720, len(k) - 24) if int(k.t.iloc[i]) >= t_lo]
    vb, vbo = k.volume.to_numpy(), (k.volume + k.vol_okx.fillna(0)).to_numpy()
    allv = ("binance", "okx", "bybit")
    for H in (4, 24):
        base, _ = run(k, liq, origins, H, allv, T6, "splice", 24, vb)
        base["ex"] = base.ic - base.ic_null
        for name, W, tiers, entry, vol in (("7일", 168, T6, "splice", vb), ("30일", 720, T6, "splice", vb), ("14단", 24, T14, "splice", vb),
                                            ("연속40단", 24, TCONT, "splice", vb), ("바이낸스+OKX", 24, T6, "splice", vbo)):
            v, _ = run(k, liq, origins, H, allv, tiers, entry, W, vol)
            v["ex"] = v.ic - v.ic_null
            m = base.merge(v, on=["i", "day", "side"], suffixes=("_b", "_v"))
            m["d"] = m.ex_v - m.ex_b; m["dc"] = m.cap_v - m.cap_b
            print(H, name, "n", len(m), "초과IC 차", [round(x, 3) for x in boot(m, "d")], "상위20% 몫 차", [round(x, 3) for x in boot(m, "dc")], flush=True)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else paired() if "--paired" in sys.argv else main()
