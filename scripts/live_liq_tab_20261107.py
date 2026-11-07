#!/usr/bin/env python3
"""풋프린트 카드 «청산» 탭의 무거운 몫을 미리 계산한다 (2026-11-07, 사용자 «I 로 구현», 입력 기간 탭 30·60·90·180일).

매시 cron: data/live/liq_tab_<coin>.json 을 원자적으로 쓴다. 대시보드는 그 파일을 내주기만 한다(GIL·이벤트 루프 보호).
- 남은 매물 히트맵: 칸 = 그날 마감에 아직 안 닿은 청산가. 입력 = 지난 W일 1시간봉 · 감쇠 없음. 그날 뒤 봉은 안 본다(인과).
  청산가 식은 라이브 청산맵과 같다(티어·유지증거금 공유, 롱 = (고+저)/2 진입 · 숏 = 종가 진입).
- 오늘 누적의 비교 띠(지난 30일 같은 시각 · 바이낸스+OKX) · 7일 실측 위치(바이낸스 체결가) · 실측 탄력 · 자기 흥분 지수.
전부 서술용 추정이다. 실측 포지션이 아니고 판정·신호가 아니다.

입력: 1시간봉 = fapi klines(서버에서만 · 코인당 시간 1회, 첫 실행만 몇 쪽) · 청산 = lake(봉인분)+hot(오늘).
실행: python scripts/live_liq_tab_20261107.py [--coin eth] [--selftest]   (LIQ_TAB_NO_REST=1 이면 REST 없이 저장된 봉만)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.coin_config import COIN_CONFIG  # noqa: E402
from scripts.live_liquidation_map_20260824 import LEVERAGE_TIERS, MAINTENANCE_MARGIN_RATE  # noqa: E402

LAKE, LIVE, HOT = ROOT / "data" / "lake", ROOT / "data" / "live", ROOT / "data" / "hot"
WINDOWS = ((30, 90), (60, 90), (90, 90), (180, 180))   # (입력 일수, 표시 일수) -- 90일까지는 표시 창을 고정해 탭을 눌러도 축이 안 움직인다
KEEP_DAYS = 370                                        # 180일 표시의 왼쪽 칸도 입력 180일을 다 채우려면 360일이 필요하다
ROW_PCT, REAL_PCT = 0.004, 0.0025                      # 히트맵 칸 0.4% · 실측 위치 칸 0.25%
HEAT_CUT = 0.1                                         # t < .1 은 화면 투명도(.85·t^2.5)로 안 보인다 -- 파일 크기를 줄인다


def fetch_k1h(symbol: str, cache: Path, now_ms: int) -> pd.DataFrame:
    """닫힌 1시간봉(t·high·low·close·qv). 저장된 봉 뒤만 받는다. REST 실패 = 저장분으로 계속(다음 시간에 메운다)."""
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["t", "high", "low", "close", "qv"])
    if os.environ.get("LIQ_TAB_NO_REST"):
        return old
    start = int(old.t.max()) + 3_600_000 if len(old) else now_ms - KEEP_DAYS * 86_400_000
    new: list[list] = []
    try:
        while start < now_ms - 3_600_000:
            u = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&startTime={start}&limit=1500"
            raw = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "liq-tab"}), timeout=20))
            if not raw:
                break
            new += [[int(r[0]), float(r[2]), float(r[3]), float(r[4]), float(r[7])] for r in raw if int(r[6]) < now_ms]
            if len(raw) < 1500:
                break
            start = int(raw[-1][0]) + 3_600_000
            time.sleep(0.5)
    except OSError as exc:
        print(f"{symbol}: klines {exc!r} -- 저장분으로 계속", flush=True)
    if not new:
        return old
    add = pd.DataFrame(new, columns=["t", "high", "low", "close", "qv"])
    k = (pd.concat([old, add]) if len(old) else add).drop_duplicates("t").sort_values("t")
    k = k[k.t >= now_ms - KEEP_DAYS * 86_400_000].reset_index(drop=True)
    tmp = cache.with_suffix(".tmp")
    k.to_parquet(tmp)
    tmp.replace(cache)
    return k


def first_touch(level: np.ndarray, side: str, hi: np.ndarray, lo: np.ndarray, chunk: int = 400) -> np.ndarray:
    """진입 봉 i 의 청산가가 i **뒤** 어느 봉에서 처음 닿았나(없으면 -1)."""
    out, ix = np.full(len(level), -1, dtype=np.int64), np.arange(len(hi))
    for a in range(0, len(level), chunk):
        lv, idx = level[a:a + chunk, None], np.arange(a, min(a + chunk, len(level)))[:, None]
        hit = ((lo[None, :] <= lv) if side == "long" else (hi[None, :] >= lv)) & (ix[None, :] > idx)
        out[a:a + chunk] = np.where(hit.any(1), hit.argmax(1), -1)
    return out


def positions(k: pd.DataFrame) -> pd.DataFrame:
    """가상 포지션 표: 봉 i · 쪽 · 청산가 · 무게(거래대금) · 처음 닿은 봉(sw) · 진입 일(e) · 닿은 일(s, 안 닿으면 큰 수)."""
    hi, lo, cl, qv = (k[c].to_numpy(float) for c in ("high", "low", "close", "qv"))
    mid, w = (hi + lo) / 2, qv / len(LEVERAGE_TIERS)
    day = (k.t.to_numpy(np.int64) // 86_400_000).astype(np.int64)
    rows = []
    for lev in LEVERAGE_TIERS:
        for side, liq in (("long", mid * (1 - 1 / lev + MAINTENANCE_MARGIN_RATE)), ("short", cl * (1 + 1 / lev - MAINTENANCE_MARGIN_RATE))):
            sw = first_touch(liq, side, hi, lo)
            rows.append(pd.DataFrame({"i": np.arange(len(k)), "side": side, "liq": liq, "w": w, "sw": sw}))
    r = pd.concat(rows, ignore_index=True)
    r["e"] = day[r.i.to_numpy()]
    r["s"] = np.where(r.sw >= 0, day[np.clip(r.sw.to_numpy(), 0, len(k) - 1)], np.int64(10) ** 9)
    return r


def heat_matrix(r: pd.DataFrame, last_day: int, view: int, win: int, row: float) -> tuple[np.ndarray, float]:
    """(칸 × 표시일) 행렬. 열 d = 진입일 ∈ [d−win, d] 이고 d 마감까지 안 닿은 무게 합. 같은 날 닿은 건 그날 칸에 없다."""
    d0 = last_day - view + 1
    st, en = np.maximum(r.e.to_numpy(), d0), np.minimum(np.minimum(r.e.to_numpy() + win, r.s.to_numpy() - 1), last_day)
    b = np.floor(r.liq.to_numpy() / row).astype(np.int64)
    m = st <= en
    if not m.any():
        return np.zeros((0, view)), 0.0
    lo_b, hi_b = int(b[m].min()), int(b[m].max()) + 1
    mat = np.zeros((hi_b - lo_b, view + 1))
    np.add.at(mat, (b[m] - lo_b, st[m] - d0), r.w.to_numpy()[m])
    np.subtract.at(mat, (b[m] - lo_b, en[m] - d0 + 1), r.w.to_numpy()[m])
    out = np.cumsum(mat, axis=1)[:, :view]
    out[np.abs(out) < out.max() * 1e-9] = 0      # 더하고 뺀 부동소수 찌꺼기
    return out, lo_b * row


def encode_heat(mat: np.ndarray, lo: float, row: float) -> dict:
    """칸값 → t(98분위 = 1000 인 정수). 열마다 [첫 칸, 값…] 으로 앞뒤 0 을 뗀다."""
    pos = mat[mat > 0]
    if not len(pos):
        return {"lo": lo, "row": row, "cols": []}
    t = np.minimum(1.0, mat / float(np.quantile(pos, 0.98)))
    t[t < HEAT_CUT] = 0
    cols = []
    for j in range(mat.shape[1]):
        nz = np.nonzero(t[:, j])[0]
        cols.append([int(nz[0])] + np.round(t[nz[0]:nz[-1] + 1, j] * 1000).astype(int).tolist() if len(nz) else [])
    return {"lo": lo, "row": row, "cols": cols}


def remain_kpi(r: pd.DataFrame, last_day: int, win: int, price: float, row: float) -> dict:
    """오늘 기준 입력 win 일의 남은 양. near3 = 현재가 ±3% 안 몫 · old7 = 7일보다 오래된 몫 · prof = 1% 칸 쪽별 몫(합 1)."""
    z = r[r.e >= last_day - win]
    a = z[z.s > last_day]
    if not len(a):
        return {}
    lo_, up = a[a.side == "long"], a[a.side == "short"]
    big = lambda x: float((x.groupby(np.round(x.liq / row))["w"].sum().idxmax()) * row) if len(x) else None  # noqa: E731
    tot, step = float(a.w.sum()), price * 0.01
    prof = a.groupby([np.round(a.liq / step), "side"])["w"].sum().reset_index()
    prof.columns = ["b", "side", "w"]
    return {"ll": float(lo_.w.sum()), "ul": float(up.w.sum()), "lm": float(z[z.side == "long"].w.sum()), "um": float(z[z.side == "short"].w.sum()),
            "lb": big(lo_), "ub": big(up), "near3": float(a[(a.liq / price - 1).abs() <= 0.03].w.sum() / tot), "old7": float(a[a.e < last_day - 7].w.sum() / tot),
            "prof": [[round(float(b * step), 8), 0 if s == "long" else 1, round(float(w / tot), 5)] for b, s, w in prof.itertuples(index=False)]}


def lake_read(sub: str, cols: str, where: str = "") -> pd.DataFrame:
    import duckdb
    if not any((LAKE / sub).glob("**/*.parquet")):
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        return con.execute(f"select {cols} from read_parquet('{LAKE / sub}/**/*.parquet', hive_partitioning=true, union_by_name=true) {where}").df()
    finally:
        con.close()


def events(coin: str, symbol: str) -> pd.DataFrame:
    """청산 이벤트(t·side·usd·px·venue). 바이낸스 = lake + hot · OKX = lake(ETH-USDT-SWAP 꼴, 파산가라 위치에는 안 쓴다)."""
    bn = lake_read(f"binance/liquidations/coin={coin}", "ts_ms t, side, usd, price px").assign(venue="bn")
    ok = lake_read("okx/liquidations/coin=ALL", "ts_ms t, pos_side side, bk_px*sz_base usd, bk_px px", f"where inst_id = '{coin}-USDT-SWAP'").assign(venue="ok")
    frames = [bn, ok]
    try:
        c = sqlite3.connect(f"file:{HOT / 'binance_ctx.sqlite'}?mode=ro", uri=True, timeout=5)
        try:
            since = int(bn.t.max()) if len(bn) else 0
            frames.append(pd.read_sql("select ts_ms t, side, usd, price px from liquidations where symbol = ? and ts_ms > ?", c, params=[symbol, since]).assign(venue="bn"))
        finally:
            c.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"{coin}: hot 청산 {exc!r} -- lake 만", flush=True)
    e = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame(columns=["t", "side", "usd", "px", "venue"])
    e = e.dropna(subset=["usd"])
    e["t"] = e.t.astype(np.int64)
    return e.sort_values("t").reset_index(drop=True)


def band(e: pd.DataFrame, today: int) -> dict | None:
    """지난 30일(양 거래소 수집된 날만) UTC 하루 5분 누적의 분위 -- 오늘 누적과 같은 축."""
    x = e[(e.t // 86_400_000) < today].copy()
    if not len(x):
        return None
    x["d"], x["b"] = x.t // 86_400_000, (x.t % 86_400_000) // 300_000
    ok_days = set(x[x.venue == "ok"].d) or set(x.d)
    days = sorted(set(x.d) & ok_days)[-30:]
    if len(days) < 7:
        return None
    m = np.zeros((len(days), 288))
    idx = {d: i for i, d in enumerate(days)}
    y = x[x.d.isin(days)]
    np.add.at(m, (y.d.map(idx).to_numpy(), y.b.to_numpy().astype(int)), y.usd.to_numpy())
    c = m.cumsum(1)
    return {"n": len(days), **{k: np.round(np.percentile(c, q, axis=0)).astype(np.int64).tolist() for k, q in (("p25", 25), ("p50", 50), ("p75", 75))}}


def branching(t_ms: np.ndarray, p: int = 30, bin_s: int = 10) -> float:
    """Kirchner(2017) INAR(p) 추정 -- 10초 건수의 AR 계수 합. 하루 안의 기저 변화 탓에 위로 치우친다(그래서 셔플 대조를 같이 낸다)."""
    x = np.bincount((t_ms // (bin_s * 1000)).astype(int), minlength=86_400 // bin_s).astype(float)
    y, xx = x[p:], np.column_stack([np.ones(len(x) - p)] + [x[p - k:-k] for k in range(1, p + 1)])
    return float(np.linalg.lstsq(xx, y, rcond=None)[0][1:].sum())


def branch_series(e: pd.DataFrame, today: int, days: int = 46) -> list:
    x = e[((e.t // 86_400_000) >= today - days) & ((e.t // 86_400_000) < today)]
    out = []
    for d, z in x.groupby(x.t // 86_400_000):
        if len(z) < 60:
            continue
        t = z.t.to_numpy() - int(d) * 86_400_000
        ctl = np.random.default_rng(int(d)).integers(0, 86_400_000, len(t))
        out.append([pd.Timestamp(int(d), unit="D").strftime("%Y-%m-%d"), round(branching(t), 3), round(branching(ctl), 3), int(len(z))])
    return out


def real_rows(e: pd.DataFrame, price: float, now_ms: int) -> list:
    """7일 바이낸스 체결 위치 [칸 가격, 롱$, 숏$]."""
    x = e[(e.venue == "bn") & (e.t >= now_ms - 7 * 86_400_000) & (e.px > 0)]
    if not len(x):
        return []
    step = price * REAL_PCT
    g = x.groupby([np.round(x.px / step), "side"])["usd"].sum().unstack(fill_value=0.0)
    return [[round(float(b * step), 8), round(float(r.get("long", 0.0))), round(float(r.get("short", 0.0)))] for b, r in g.iterrows()]


def elasticity(coin: str, k: pd.DataFrame, now_ms: int) -> dict | None:
    """1시간 |이동| ≥ .5% 인 시간의 «1% 이동당 실제 청산» 중앙값(바이낸스 · 30일). 초당 1건 제한이 있어 하한이다."""
    tr = lake_read(f"binance/tail_risk_1m/coin={coin}", "epoch_ms(ts) t, long_usd_1m l, short_usd_1m s")
    if not len(tr):
        return None
    tr = tr[tr.t >= now_ms - 30 * 86_400_000]
    h = tr.assign(h=tr.t // 3_600_000).groupby("h")[["l", "s"]].sum()
    r = k.assign(h=k.t // 3_600_000, r=k.close.pct_change() * 100).set_index("h").r
    j = h.join(r, how="inner").dropna()
    up, dn = j[j.r >= 0.5], j[j.r <= -0.5]
    if len(up) < 10 or len(dn) < 10:
        return None
    return {"up": float(((up.l + up.s) / up.r).median()), "dn": float(((dn.l + dn.s) / -dn.r).median()), "n_up": int(len(up)), "n_dn": int(len(dn))}


def build(coin: str, now_ms: int) -> dict | None:
    symbol = COIN_CONFIG[coin]["binance_symbol"]
    k = fetch_k1h(symbol, LIVE / f"liq_tab_k1h_{coin}.parquet", now_ms)
    if len(k) < 24 * 8:
        print(f"{coin}: 1시간봉 {len(k)}개 -- 건너뜀", flush=True)
        return None
    price = float(k.close.iloc[-1])
    row = float(f"{price * ROW_PCT:.2g}")
    r = positions(k)
    day = k.t // 86_400_000
    last_day = int(day.iloc[-1])
    daily = k.groupby(day).agg(o=("close", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    daily = daily[daily.index > last_day - 180]
    made = r.groupby(["e", "side"])["w"].sum().unstack(fill_value=0.0)
    left = r[r.s > last_day].groupby(["e", "side"])["w"].sum().unstack(fill_value=0.0)
    days = []
    for d, x in daily.iterrows():
        pct = lambda side: round(float(left.loc[d, side] / made.loc[d, side]), 3) if d in left.index and side in left.columns and made.loc[d, side] > 0 else 0.0  # noqa: E731,B023
        days.append([pd.Timestamp(int(d), unit="D").strftime("%Y-%m-%d"), x.o, x.h, x.l, x.c, pct("long"), pct("short")])
    remain = {}
    for win, view in WINDOWS:
        mat, lo = heat_matrix(r, last_day, view, win, row)
        remain[str(win)] = {"view": view, **encode_heat(mat, lo, row), **remain_kpi(r, last_day, win, price, row)}
    e = events(coin.upper(), symbol)
    today = now_ms // 86_400_000
    return {"ok": True, "v": 1, "coin": coin, "generated_ms": now_ms, "price": price, "kline_end_ms": int(k.t.iloc[-1]) + 3_600_000, "days": days, "remain": remain,
            "band": band(e, today), "branch": branch_series(e, today), "real": real_rows(e, price, now_ms), "beta": elasticity(coin.upper(), k, now_ms)}


def selftest() -> None:
    # 합성: 100 → 110 으로 올랐다가 9일째 90 까지 빠진다.
    n = 24 * 12
    t = (np.arange(n) * 3_600_000 + 19_000 * 86_400_000).astype(np.int64)
    c = np.r_[np.linspace(100, 110, 24 * 8), np.linspace(110, 90, 24), np.full(n - 24 * 9, 90.0)]
    k = pd.DataFrame({"t": t, "high": c + 0.05, "low": c - 0.05, "close": c, "qv": np.full(n, 1000.0)})
    hi, lo = k.high.to_numpy(), k.low.to_numpy()
    ft = first_touch(np.full(n, 95.0), "long", hi, lo)
    assert ft[0] > 24 * 8 and lo[ft[0]] <= 95 and all(lo[1:ft[0]] > 95), "첫 닿음"
    assert first_touch(np.full(n, 1.0), "long", hi, lo)[0] == -1, "안 닿으면 -1"
    r = positions(k)
    last = int(t[-1] // 86_400_000)
    row = 0.5
    m, lo_px = heat_matrix(r, last, 12, 30, row)
    assert m.shape[1] == 12 and lo_px > 0
    # ① 인과: 뒤 봉을 잘라도 과거 열은 그대로
    cut = 24 * 7
    r2 = positions(k.iloc[:cut].reset_index(drop=True))
    last2 = int(t[cut - 1] // 86_400_000)
    m2, lo2 = heat_matrix(r2, last2, 7, 30, row)
    a = {round(lo_px + i * row, 4): m[i, 0:7] for i in range(m.shape[0])}
    for i in range(m2.shape[0]):
        p = round(lo2 + i * row, 4)
        got = a.get(p, np.zeros(7))
        assert np.allclose(got[:7], m2[i], atol=1e-6), f"미래 봉이 과거 칸을 바꿨다 {p}"
    # ② 하락으로 쓸린 롱 청산가는 쓸린 뒤 칸에 없다
    ml, lo_l = heat_matrix(r[r.side == "long"], last, 12, 30, row)
    px = lo_l + np.arange(ml.shape[0]) * row
    before, after = ml[(px >= 91) & (px < 104), 7].sum(), ml[(px >= 91) & (px < 104), 10].sum()
    assert before > 0 and after == 0, (before, after)
    # ③ 입력 기간: win=1 이면 만든 지 2일 넘은 매물이 빠져 양이 줄거나 같다
    m3, _ = heat_matrix(r, last, 12, 1, row)
    assert m3[:, 11].sum() <= m[:, 11].sum() + 1e-6 and m3[:, 11].sum() < m[:, 7].sum()
    # ④ 부호화 왕복: t 1000 이 있고 빈 열은 []
    enc = encode_heat(m, lo_px, row)
    assert len(enc["cols"]) == 12 and max(max(c[1:]) for c in enc["cols"] if c) == 1000
    assert encode_heat(np.zeros((3, 2)), 1.0, row)["cols"] == []
    # ⑤ KPI: 쪽별 몫의 합 = 1 · 분기비: 무작위 < 군집
    kpi = remain_kpi(r, last, 30, 90.0, row)
    assert abs(sum(x[2] for x in kpi["prof"]) - 1) < 1e-3 and 0 <= kpi["near3"] <= 1
    rng = np.random.default_rng(1)
    base = rng.integers(0, 86_400_000, 800)
    burst = np.concatenate([base + rng.integers(0, 15_000, 800), base + rng.integers(0, 30_000, 800), base]) % 86_400_000
    assert branching(burst) > branching(rng.integers(0, 86_400_000, 2400)) + 0.2, "군집이 더 커야 한다"
    print("selftest OK -- 첫 닿음 · 인과(뒤 봉 무관) · 쓸린 칸 사라짐 · 입력 기간 · 부호화 · KPI · 분기비")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    coins = [a.coin] if a.coin else [c for c in COIN_CONFIG if (LAKE / f"binance/liquidations/coin={c.upper()}").exists()]
    now_ms = int(time.time() * 1000)
    for coin in coins:
        t0 = time.time()
        try:
            out = build(coin, now_ms)
        except Exception as exc:  # noqa: BLE001 -- 한 코인 실패가 다음 코인을 막지 않는다
            print(f"{coin}: 실패 {type(exc).__name__}: {exc}", flush=True)
            continue
        if out is None:
            continue
        p = LIVE / f"liq_tab_{coin}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(p)
        print(f"{coin}: {p.stat().st_size // 1024}KB · {time.time() - t0:.1f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
