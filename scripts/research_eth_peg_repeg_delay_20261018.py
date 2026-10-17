#!/usr/bin/env python3
"""peg 메이커 «앞질려도 H초 기다렸다 다시 걸기» 시뮬레이션 (2026-10-18 사용자 «3초 말고 5초?» → «2번(먼저 재 보기)»).

지금 배포(live_manual_peg_execute_20260912.fill_maker): 3초마다 확인 · 앞질렸으면(매수: 최우선 매수호가 > 내 가격) 곧바로
취소하고 새 최우선 호가에 다시 건다(H=0). H초 = «앞질린 상태가 H초 이어질 때만» 다시 건다(확인은 3초 그대로). H=inf = 안 쫓음(static).

원천(ETH 만):
  호가   : ETHUSDT bookTicker 원시(~/backups/crypto-scalping-server/data/live/orderflow/bookticker/ETHUSDT, 09-14~10-16)
  체결   : data.binance.vision ETHUSDT·ETHUSDC aggTrades(tmp/peg_repeg_delay_20261018/agg)
  🔴주문은 ETHUSDC 인데 USDC 호가 기록이 없다 -> 주 판정 = USDT 호가·체결로 같은 정책을 재생(상대 비교).
    USDC 근사(부): USDC 호가 = USDT 호가 - 직전 60초 두 종목 1초 마지막 체결가 차의 중앙값(인과) · 체결 = USDC aggTrades.

체결 규칙(10-04 research_eth_limit_fill_adverse 의 주 규칙 «through»): 내 가격보다 나쁜 가격의 테이커 체결이 나오면 내 가격에 체결.
  (같은 가격 줄 앞사람 소진은 안 센다 -- 보수. 부 규칙 «queue» = through ∪ 내 가격 테이커 누적 ≥ 도착 때 앞 잔량 + 내 수량)

    python scripts/research_eth_peg_repeg_delay_20261018.py --selftest
    python scripts/research_eth_peg_repeg_delay_20261018.py --run          # -> tmp/peg_repeg_delay_20261018/orders.parquet · report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import data_store as ds  # noqa: E402

OUT = ROOT / "tmp/peg_repeg_delay_20261018"
AGG = OUT / "agg"
BT_DIR = Path.home() / "backups/crypto-scalping-server/data/live/orderflow/bookticker/ETHUSDT"
TK = 100                                     # ETH 틱 0.01 -> 정수 틱

# ── 사전등록 (결과 보기 전에 고정) ──────────────────────────────────────────────
CRITERIA = dict(
    grid_s=60,                 # 60초마다 매수·매도 각 1건(최우선 지정가 peg)
    lat_ms=450,                # 결정 -> 거래소 도착(10-04 집행 표와 같은 값). 취소~재접수 공백도 이 값
    poll_s=3.0, deadline_s=120.0, repeg_max=40,   # 배포 상수 그대로(POLL_SEC · FALLBACK_SEC · REPEG_MAX)
    H=(0, 3, 6, 9, 15, 30, np.inf),               # 0 = 지금 배포 · inf = 안 쫓음
    unit=1.0,                  # 내 수량(ETH, queue 규칙용)
    fees={"usdc": (0.0, 4.0), "usdt": (2.0, 5.0)},  # (메이커, 테이커) bp -- USDC 계정 실요율 0/4 · USDT 실측 2/5
    dev_frac=0.6,              # 앞 60% 날 = DEV(H 고르기) · 뒤 40% = HOLDOUT(판정)
    # 판정: DEV 에서 평균 비용(USDC 수수료, through) 최소 H* -> HOLDOUT 에서 H* − H0 날 블록 부트스트랩 95% CI 가 0 미만
    #       이고 99분위 비용이 H0 보다 나쁘지 않을 때만 «바꿀 근거 있음». USDC 근사에서 부호가 같아야 함(부).
    boot=2000, seed=20261018,
)


# ── 원천 ───────────────────────────────────────────────────────────────────────
def load_bt(day: str) -> dict:
    d0 = pd.Timestamp(day)
    hs = [(d0 + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H") for h in range(-1, 25)]
    fs = sorted(p for p in BT_DIR.iterdir() if p.name[:13] in hs)
    a = np.concatenate([ds.read_bt(p) for p in fs]) if fs else np.zeros(0, ds.BT_DTYPE)
    a = a[(a["bid_px"] > 0) & (a["ask_px"] > a["bid_px"])]
    a = a[np.argsort(a["ts_ms"], kind="stable")]
    return dict(ts=a["ts_ms"].astype(np.int64), bid=np.round(a["bid_px"] * TK).astype(np.int64),
                ask=np.round(a["ask_px"] * TK).astype(np.int64), bq=a["bid_qty"].astype("f8"), aq=a["ask_qty"].astype("f8"))


def load_agg(sym: str, day: str) -> dict:
    z = AGG / f"{sym}-aggTrades-{day}.zip"
    t = pd.read_csv(z, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    t = t.sort_values("transact_time", kind="stable")
    return dict(ts=t.transact_time.to_numpy(np.int64), tick=np.round(t.price.to_numpy() * TK).astype(np.int64),
                qty=t.quantity.to_numpy("f8"), sell=t.is_buyer_maker.to_numpy(bool))   # buyer_maker = 매도 테이커


def usdc_quotes(bt: dict, tu: dict, tc: dict) -> dict:
    """USDC 근사 호가 = USDT 호가 − 기초차(직전 60초 1초 마지막 체결가 차의 중앙값, 그 초는 안 씀 = 인과)."""
    def last_per_sec(tr):
        return pd.Series(tr["tick"], index=tr["ts"] // 1000).groupby(level=0).last()
    d = (last_per_sec(tu) - last_per_sec(tc)).dropna()
    full = d.reindex(np.arange(d.index.min(), d.index.max() + 1)).ffill()
    med = full.rolling(60, min_periods=10).median().shift(1)
    b = np.round(med.reindex(bt["ts"] // 1000).to_numpy())
    ok = np.isfinite(b)
    bi = np.where(ok, b, 0).astype(np.int64)
    return dict(ts=bt["ts"][ok], bid=(bt["bid"] - bi)[ok], ask=(bt["ask"] - bi)[ok], bq=bt["bq"][ok] / 8, aq=bt["aq"][ok] / 8)   # 잔량 1/8(10-04 실측)


def q_at(bt: dict, t: int):
    i = int(np.searchsorted(bt["ts"], t, "right")) - 1
    if i < 0 or t - bt["ts"][i] > 2000:
        return None
    return int(bt["bid"][i]), int(bt["ask"][i]), float(bt["bq"][i]), float(bt["aq"][i])


# ── 한 주문 재생 ───────────────────────────────────────────────────────────────
def first_hit(tr: dict, side: int, p: int, a: int, b: int, rule: str, ahead: float, used: float):
    """(a, b] 안 첫 체결 시각(ms)과 갱신된 내 가격 테이커 누적. 없으면 (None, used)."""
    i, j = np.searchsorted(tr["ts"], a, "right"), np.searchsorted(tr["ts"], b, "right")
    if i >= j:
        return None, used
    ts, tk, q, sl = tr["ts"][i:j], tr["tick"][i:j], tr["qty"][i:j], tr["sell"][i:j]
    hit = sl if side > 0 else ~sl                                   # 내 지정가를 칠 수 있는 테이커 쪽
    k = np.flatnonzero(hit & ((tk < p) if side > 0 else (tk > p)))
    t = int(ts[k[0]]) if len(k) else None
    if rule == "queue":
        cum = used + np.cumsum(np.where(hit & (tk == p), q, 0.0))
        k2 = np.flatnonzero(cum >= ahead + CRITERIA["unit"])
        if len(k2) and (t is None or ts[k2[0]] < t):
            t = int(ts[k2[0]])
        used = float(cum[-1])
    return t, used


def simulate(side: int, td: int, H: float, bt: dict, tr: dict, rule: str = "through") -> dict | None:
    """td = 첫 결정 시각(ms). 반환: 체결가(틱)·체결 여부·시간·리페그 수·놓친 뚫림(재접수 공백 중 옛 가격이 뚫림)."""
    C = CRITERIA
    lat, poll = C["lat_ms"], int(C["poll_s"] * 1000)
    q0 = q_at(bt, td)
    if q0 is None:
        return None
    m0 = (q0[0] + q0[1]) / 2
    t_end = td + lat + int(C["deadline_s"] * 1000)
    repegs, missed, t_dec, gap_from, p_old = 0, 0, td, None, None
    while repegs <= C["repeg_max"] and t_dec + lat < t_end:
        q = q_at(bt, t_dec)
        if q is None:
            return None
        p = q[0] if side > 0 else q[1]
        ta = t_dec + lat
        if gap_from is not None:                                    # 취소~재접수 공백에 옛 가격이 뚫렸나(사용자 «지나갔는데 안 됨»)
            t_m, _ = first_hit(tr, side, p_old, gap_from, ta, "through", 0.0, 0.0)
            missed += t_m is not None
        qa = q_at(bt, ta)
        if qa is None:
            return None
        if (side > 0 and qa[1] <= p) or (side < 0 and qa[0] >= p):  # post-only 거부(-5022) -> 새 호가로 즉시 다시
            repegs += 1
            t_dec, gap_from, p_old = ta, ta, p
            continue
        best = qa[0] if side > 0 else qa[1]
        ahead = (qa[2] if side > 0 else qa[3]) if best == p else (0.0 if (best - p) * side < 0 else np.inf)
        used, t0, drift0, repeg = 0.0, ta, None, False
        while t0 < t_end:
            t1 = min(t0 + poll, t_end)
            tf, used = first_hit(tr, side, p, t0, t1, rule, ahead, used)
            if tf is not None:
                return dict(px=p, filled=True, t_fill=(tf - td) / 1e3, repegs=repegs, missed=missed, m0=m0)
            t0 = t1
            if t1 >= t_end:
                break
            qp = q_at(bt, t1)
            if qp is None:
                continue                                            # 호가 모름 = 앞지름 모름(배포 코드와 같은 취급)
            if not (qp[0] > p if side > 0 else qp[1] < p):
                drift0 = None
                continue
            drift0 = t1 if drift0 is None else drift0
            if t1 - drift0 >= H * 1000:
                repegs += 1
                t_dec, gap_from, p_old, repeg = t1, t1, p, True
                break
        if not repeg:
            break
    qe = q_at(bt, t_end)
    if qe is None:
        return None
    return dict(px=qe[1] if side > 0 else qe[0], filled=False, t_fill=np.nan, repegs=repegs, missed=missed, m0=m0)


def day_job(args) -> pd.DataFrame:
    day, venue = args
    bt = load_bt(day)
    tu = load_agg("ETHUSDT", day)
    if venue == "usdc":
        tc = load_agg("ETHUSDC", day)
        bt, tr = usdc_quotes(bt, tu, tc), tc
    else:
        tr = tu
    if not len(bt["ts"]) or not len(tr["ts"]):
        return pd.DataFrame()
    d0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    rows = []
    for td in range(d0 + 60_000, d0 + 86_400_000 - 130_000, CRITERIA["grid_s"] * 1000):
        for side in (1, -1):
            for H in CRITERIA["H"]:
                for rule in ("through", "queue"):
                    r = simulate(side, td, H, bt, tr, rule)
                    if r:
                        rows.append(dict(day=day, venue=venue, td=td, side=side, H=H, rule=rule, **r))
    df = pd.DataFrame(rows)
    print(f"{venue} {day}: {len(df)}", flush=True)
    return df


# ── 비용·판정 ───────────────────────────────────────────────────────────────────
def costs(df: pd.DataFrame, fee: str) -> pd.Series:
    fm, ft = CRITERIA["fees"][fee]
    return df.side * (df.px / df.m0 - 1) * 1e4 + np.where(df.filled, fm, ft)


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, fee: str) -> tuple[float, float, float]:
    """같은 주문 짝(day·td·side) 비용 차 b − a 의 날 블록 부트스트랩."""
    k = ["day", "td", "side"]
    m = a.assign(c=costs(a, fee))[k + ["c"]].merge(b.assign(c=costs(b, fee))[k + ["c"]], on=k, suffixes=("a", "b"))
    g = (m.cb - m.ca).groupby(m.day).agg(["sum", "count"])
    rng = np.random.default_rng(CRITERIA["seed"])
    days = g.index.to_numpy()
    bs = [g.loc[s, "sum"].sum() / g.loc[s, "count"].sum() for s in (rng.choice(days, len(days)) for _ in range(CRITERIA["boot"]))]
    return float((m.cb - m.ca).mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(df: pd.DataFrame, fee: str) -> pd.DataFrame:
    df = df.assign(c=costs(df, fee))
    return df.groupby("H").agg(n=("c", "size"), cost=("c", "mean"), p90=("c", lambda x: x.quantile(.9)),
                               p99=("c", lambda x: x.quantile(.99)), fill=("filled", "mean"),
                               t_fill_med=("t_fill", "median"), repegs=("repegs", "mean"),
                               missed=("missed", lambda x: (x > 0).mean())).round(4)


def run():
    days = [d.date().isoformat() for d in pd.date_range("2026-09-14", "2026-10-16")]
    days = [d for d in days if (AGG / f"ETHUSDT-aggTrades-{d}.zip").exists() and (AGG / f"ETHUSDC-aggTrades-{d}.zip").exists()]
    with ProcessPoolExecutor(3) as ex:                              # 하루 호가 ~수백 MB -- 메모리 15GB
        parts = list(ex.map(day_job, [(d, v) for v in ("usdt", "usdc") for d in days]))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / "orders.parquet")
    n_dev = int(len(days) * CRITERIA["dev_frac"])
    dev, ho = set(days[:n_dev]), set(days[n_dev:])
    rep = {"days": [days[0], days[-1], len(days)], "dev_days": n_dev, "H": [str(h) for h in CRITERIA["H"]],
           "criteria": {k: v for k, v in CRITERIA.items() if k != "H"}}
    for venue in ("usdt", "usdc"):
        for rule in ("through", "queue"):
            for fee in ("usdc", "usdt"):
                x = df[(df.venue == venue) & (df.rule == rule)]
                rep[f"{venue}|{rule}|fee_{fee}"] = {
                    part: summarize(x[x.day.isin(s)], fee).reset_index().astype({"H": str}).to_dict("records")
                    for part, s in (("dev", dev), ("holdout", ho))}
    # 판정(사전등록): usdt 원천 · through · USDC 수수료
    x = df[(df.venue == "usdt") & (df.rule == "through")]
    sd = summarize(x[x.day.isin(dev)], "usdc")
    h_star = sd.drop(index=0.0, errors="ignore").cost.idxmin()
    h0, hs = x[(x.H == 0) & x.day.isin(ho)], x[(x.H == h_star) & x.day.isin(ho)]
    d, lo, hi = boot_diff(h0, hs, "usdc")
    p99_0, p99_s = costs(h0, "usdc").quantile(.99), costs(hs, "usdc").quantile(.99)
    xc = df[(df.venue == "usdc") & (df.rule == "through")]
    dc, loc, hic = boot_diff(xc[(xc.H == 0) & xc.day.isin(ho)], xc[(xc.H == h_star) & xc.day.isin(ho)], "usdc")
    rep["verdict"] = {"H_star_dev": str(h_star), "holdout_diff_bp": [d, lo, hi], "p99_H0": p99_0, "p99_Hstar": p99_s,
                      "usdc_approx_diff_bp": [dc, loc, hic], "pass": bool(hi < 0 and p99_s <= p99_0 and dc < 0)}
    rep["holdout_diff_all"] = {str(h): boot_diff(h0, x[(x.H == h) & x.day.isin(ho)], "usdc") for h in CRITERIA["H"] if h != 0}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(rep["verdict"], ensure_ascii=False, default=float))


# ── 자체점검 ───────────────────────────────────────────────────────────────────
def selftest():
    def T(ts_, tk_, sell_):
        return dict(ts=np.array(ts_, np.int64), tick=np.array(tk_, np.int64), qty=np.ones(len(ts_)), sell=np.array(sell_, bool))
    # 합성 호가: 매수호가 100.00 → 1.5초부터 101.00(앞질림). 매도 테이커 9초 99.99 · 30초 100.99
    ts = np.arange(0, 200_000, 100, dtype=np.int64)
    bid = np.where(ts < 1500, 10000, 10100).astype(np.int64)
    bt = dict(ts=ts, bid=bid, ask=bid + 1, bq=np.full(len(ts), 5.0), aq=np.full(len(ts), 5.0))
    tr = T([9_000, 30_000], [9999, 10099], [True, True])
    # H=0: 0.45초 도착(100) · 3.45초 확인에서 앞질림 → 다시(101, 3.9초 도착) · 9초 99.99 매도가 101 도 뚫음 → 101 에 체결(1 비싸게).
    #       공백(3.45~3.9초)엔 체결 없어 «놓친 뚫림» 0
    r = simulate(1, 0, 0, bt, tr)
    assert r["filled"] and r["px"] == 10100 and r["repegs"] == 1 and abs(r["t_fill"] - 9.0) < 1e-9 and r["missed"] == 0, r
    # H=inf: 100 에 그대로 → 9초 99.99 가 뚫음 → 100 에 체결(더 싸게)
    r = simulate(1, 0, np.inf, bt, tr)
    assert r["filled"] and r["px"] == 10000 and r["repegs"] == 0 and abs(r["t_fill"] - 9.0) < 1e-9, r
    # H=6: 3.45 앞질림 시작 · 6.45 아직 3초 · 9초 체결이 먼저 → 100 에 체결(리페그 0)
    r = simulate(1, 0, 6, bt, tr)
    assert r["filled"] and r["px"] == 10000 and r["repegs"] == 0, r
    # 공백에 옛 가격이 뚫리면 missed: 3.6초 99.99 매도(공백 3.45~3.9초 안) -- H=0
    r = simulate(1, 0, 0, bt, T([3_600, 30_000], [9999, 10099], [True, True]))
    assert r["missed"] == 1 and r["px"] == 10100, r
    # 체결 없음 → 120.45초에 시장가(매도호가 101.01)
    r = simulate(1, 0, 0, bt, T([], [], []))
    assert not r["filled"] and r["px"] == 10101, r
    # 매도 쪽 거울상: 매도호가 100.01 → 1.5초부터 99.01(앞질림) · H=inf · 9초 매수 테이커 100.02 → 100.01 에 체결
    bts = dict(ts=ts, bid=np.where(ts < 1500, 10000, 9900), ask=np.where(ts < 1500, 10001, 9901), bq=np.full(len(ts), 5.0), aq=np.full(len(ts), 5.0))
    r = simulate(-1, 0, np.inf, bts, T([9_000], [10002], [False]))
    assert r["filled"] and r["px"] == 10001, r
    # post-only 거부: 0.4초부터 매도호가 100.00 = 내 매수 100.00 과 교차 → 리페그 · 새 매수호가 99.99 에 걸려 5초 99.98 매도로 체결
    bt2 = dict(ts=ts, bid=np.where(ts < 400, 10000, 9999), ask=np.where(ts < 400, 10001, 10000), bq=np.full(len(ts), 5.0), aq=np.full(len(ts), 5.0))
    r = simulate(1, 0, 0, bt2, T([5_000], [9998], [True]))
    assert r["repegs"] == 1 and r["filled"] and r["px"] == 9999, r
    # queue 규칙: 앞 잔량 5 + 내 1 = 6 ETH 가 내 가격에서 팔려야 체결(through 없으면)
    r = simulate(1, 0, np.inf, bt, dict(ts=np.arange(1000, 7000, 1000, dtype=np.int64), tick=np.full(6, 10000), qty=np.ones(6), sell=np.ones(6, bool)), "queue")
    assert r["filled"] and abs(r["t_fill"] - 6.0) < 1e-9, r
    # 비용: 매수 체결가 = mid → 0 + 메이커 수수료 · 미체결은 테이커 수수료
    df = pd.DataFrame(dict(side=[1, -1], px=[10000.5, 10000.5], m0=[10000.5, 10000.5], filled=[True, False]))
    assert np.allclose(costs(df, "usdc"), [0.0, 4.0])
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.run:
        run()
