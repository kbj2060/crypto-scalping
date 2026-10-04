"""깊은 호가벽 5분 — ETH HO3(3차 미접촉) + OI 조건 첫 판정 + BTC·SOL·XRP·HYPE 재현 (2026-10-04).

사전등록: docs/experiments/deep_wall_ho3_multicoin_prereg_20261004.md (정의·창·판정 기준 전부 거기 고정).
--trade: docs/experiments/deep_wall_maker_trading_prereg_20261004.md (Q1 30·60분 통제 · 메이커 매매 P1·P2).
패널 리플레이는 research_rt5_1s_panel_build_20260920 의 bt_file/dd_file 을 그대로 쓰고 틱 배율만 코인별로 바꾼다.
체결·OI 는 서버 lake parquet(10-01 저장 재설계 이후 원천). 서버에서 실행: nice -n 19.
"""
from __future__ import annotations

import glob
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rt5_1s_panel_build_20260920 as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAKE = ROOT / "data/lake/binance"
TICKS = {"ETH": 100, "BTC": 10, "SOL": 100, "XRP": 10_000, "HYPE": 1_000}
T = lambda s: int(pd.Timestamp(s).timestamp())  # noqa: E731
END = T("2026-10-03T00:00Z")
WIN = {"ETH": (T("2026-09-26T19:40Z"), END)} | {c: (T("2026-09-26T10:00Z"), END) for c in ("BTC", "SOL", "XRP", "HYPE")}
WARM = 86400


def hour_files(d: Path, lo: int, hi: int) -> list[str]:
    return [f for f in sorted(glob.glob(str(d / "*"))) if lo - 3600 <= T(Path(f).name[:13] + ":00Z") < hi]


def lake(kind: str, coin: str, lo: int, hi: int) -> pd.DataFrame:
    days = pd.date_range(pd.to_datetime(lo, unit="s").normalize(), pd.to_datetime(hi, unit="s"), freq="D")
    fs = [p for d in days for p in glob.glob(str(LAKE / kind / f"coin={coin}" / f"date={d:%Y-%m-%d}" / "*.parquet"))]
    return pd.concat([pd.read_parquet(f) for f in fs]) if fs else pd.DataFrame()


def build(coin: str, trade_mode: bool = False) -> pd.DataFrame:
    lo, hi = WIN[coin]; lo0 = lo - WARM
    B.TICK = TICKS[coin]                     # 풀 워커는 fork 라 이 값을 물려받는다
    sym = f"{coin}USDT"
    with ProcessPoolExecutor(max_workers=6) as ex:
        bt = pd.concat(list(ex.map(B.bt_file, hour_files(B.ROOT / "data/live/orderflow/bookticker" / sym, lo0, hi))))
        dd = pd.concat([d for d in ex.map(B.dd_file, hour_files(B.ROOT / "data/live/orderflow/depthdiff" / sym, lo0, hi)) if len(d)])
    bt = bt[~bt.index.duplicated(keep="last")]; dd = dd[~dd.index.duplicated(keep="last")]
    tp = lake("tape", coin, lo0, hi)
    tr = tp.groupby("ts_sec")[["buy_qty", "sell_qty"]].sum()
    P = pd.DataFrame(index=pd.RangeIndex(max(lo0, int(bt.index.min())), hi, name="ts_sec"))
    P = P.join(bt[["bt_mid", "bt_spread_bp"]]).join(dd[["dd_bid50", "dd_ask50"]]).join(tr)
    P = P.join(tp[tp.sell_qty > 0].groupby("ts_sec").price_bin.min().rename("smin"))
    P = P.join(tp[tp.buy_qty > 0].groupby("ts_sec").price_bin.max().rename("bmax"))
    P[["buy_qty", "sell_qty"]] = P[["buy_qty", "sell_qty"]].fillna(0.0)
    oi = lake("oi_1s", coin, lo0, hi)
    if len(oi):
        oi = oi.sort_values("ts_ms"); oi["d"] = oi.open_interest.diff(); oi["sec"] = oi.ts_ms // 1000
        P = P.join(oi.groupby("sec").agg(oi_last=("open_interest", "last"), oi_d=("d", "sum")))
        P["oi_last"] = P.oi_last.ffill(limit=30)
        P["oi_d300"] = P.oi_d.fillna(0.0).where(P.oi_last.notna()).rolling(300, min_periods=150).sum()
    else:
        P["oi_d300"] = np.nan
    mid = P.bt_mid.ffill(limit=5)
    P["fwd300"] = (np.log(mid.shift(-300)) - np.log(mid)) * 1e4
    P["dmid300"] = (np.log(mid) - np.log(mid.shift(300))) * 1e4
    P["dmid60"] = (np.log(mid) - np.log(mid.shift(60))) * 1e4
    P["dmid1800"] = (np.log(mid) - np.log(mid.shift(1800))) * 1e4
    P["dmid3600"] = (np.log(mid) - np.log(mid.shift(3600))) * 1e4
    hi24 = mid.rolling(86400, min_periods=3600).max(); lo24 = mid.rolling(86400, min_periods=3600).min()
    P["range_pos"] = (mid - lo24) / (hi24 - lo24)
    rs = lambda s, w: s.rolling(w, min_periods=w // 2).sum()  # noqa: E731
    P["tr_imb60"] = rs(P.buy_qty - P.sell_qty, 60) / rs(P.buy_qty + P.sell_qty, 60).replace(0, np.nan)
    P["dd_imb50"] = (P.dd_bid50 - P.dd_ask50) / (P.dd_bid50 + P.dd_ask50)
    P["mid"] = mid
    if not trade_mode:
        return P[(P.index >= lo) & (P.index < hi)]
    P = P[P.index < hi].copy()
    P.attrs["bin_med"] = float(tp.price_bin.median())          # join 이 attrs 를 버리므로 마지막에 단다
    return P                    # 매매는 문턱 예열(직전 24h)부터 돈다


CTRL = (("dmid300", 20), ("range_pos", 10), ("dmid60", 10), ("tr_imb60", 10))
CTRL_Q1 = CTRL + (("dmid1800", 10), ("dmid3600", 10))


def prep(P: pd.DataFrame, lag: int = 0, ctrl=CTRL) -> pd.DataFrame:
    Q = P.copy()
    r = Q.dd_imb50.shift(lag).rank(pct=True)          # lag 초 늦게 본 벽(통제·수익은 지금 시점 그대로)
    Q["buy"] = r >= .8; Q["sell"] = r <= .2
    y = Q.fwd300.copy()
    for c, nq in ctrl:
        d = pd.qcut(Q[c].rank(method="first"), nq, labels=False)
        y = y - y.groupby(d).transform("mean")
    Q["res"] = y
    Q["blk"] = Q.index // 3600
    return Q


def msn(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)) if len(v) > 2 else (np.nan, np.nan, len(v))


def spread(Q, a, b, col="res", min_n=30):
    v = []
    for _, g in Q.groupby("blk"):
        x = g.loc[a.reindex(g.index, fill_value=False), col].dropna()
        z = g.loc[b.reindex(g.index, fill_value=False), col].dropna()
        if len(x) >= min_n and len(z) >= min_n:
            v.append(x.mean() - z.mean())
    return msn(v)


def oi_diff(Q, min_n=30):
    """블록마다 (OI↓ 안 벽 스프레드 − OI↑ 안 벽 스프레드). 네 부분 각 ≥ min_n 초."""
    dn = Q.oi_d300 < 0; up = Q.oi_d300 >= 0
    v = []
    for _, g in Q.groupby("blk"):
        parts = [g.loc[side & m.loc[g.index], "res"].dropna()
                 for m in (dn, up) for side in (g.buy, g.sell)]
        if min(len(p) for p in parts) >= min_n:
            v.append((parts[0].mean() - parts[1].mean()) - (parts[2].mean() - parts[3].mean()))
    return msn(v)


def one(Q, m, col="res", min_n=30):
    return msn([g.loc[m.loc[g.index], col].dropna().mean() for _, g in Q.groupby("blk") if m.loc[g.index].sum() >= min_n])


fmt = lambda m, se, n: f"{m:+7.2f} ± {se:5.2f} (t {m/se:+5.2f}, {n}블록)" if np.isfinite(se) and se > 0 else f"n/a ({n}블록)"  # noqa: E731


def selftest():
    rng = np.random.default_rng(0); n = 6 * 3600
    imb = rng.normal(size=n)
    Q = pd.DataFrame({"dd_imb50": imb, "fwd300": 5 * np.sign(imb) * (np.abs(imb) > .84) + rng.normal(size=n),
                      "dmid300": rng.normal(size=n), "range_pos": rng.random(n), "dmid60": rng.normal(size=n),
                      "tr_imb60": rng.normal(size=n), "oi_d300": rng.normal(size=n)}, index=np.arange(n))
    Q = prep(Q)
    m, se, k = spread(Q, Q.buy, Q.sell)
    assert k == 6 and 9 < m < 11, (m, k)                 # 매수벽 +5, 매도벽 −5 → 스프레드 ≈ 10
    d, _, k2 = oi_diff(Q)
    assert k2 == 6 and abs(d) < 1.0, d                   # OI 와 무관하게 만든 데이터 → 차 ≈ 0
    Q.loc[Q.oi_d300 < 0, "res"] *= 2                     # OI↓ 에서만 효과 2배 → 차 ≈ +10
    d, _, _ = oi_diff(Q)
    assert 8 < d < 12, d
    # 매매 시뮬: 신호가 늘 +이면 첫 봉 뒤 롱, 가격이 오르면 자산 증가 / 체결은 «뚫고 지나감»일 때만
    n = 3 * 86400 // 2; idx = np.arange(n) + 86400 * 10
    m = 100 + np.arange(n) * 1e-4
    P = pd.DataFrame({"mid": m, "bt_spread_bp": 1.0, "smin": np.floor(m * 0.9999 / 0.01) - 1, "bmax": np.nan}, index=idx)
    x = np.where(np.arange(n) % 2 == 0, 1.0, 0.5)                 # 봉마감 초(짝수)는 1.0 > 80분위 → 롱
    r = sim(P, x, 0.01, idx[0])
    assert r["n_maker"] == 1 and r["n_taker"] == 0 and r["daily"].sum() > 0, r
    P["smin"] = np.nan                                           # 아무것도 안 뚫리면 체결 0
    r = sim(P, x, 0.01, idx[0])
    assert r["n"] == 0 and abs(r["daily"].sum()) < 1e-9, r
    print("selftest ok")


BINS = (1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0)
MAKER_BP, TAKER_BP = 0.0, 4.0


def bin_size(P: pd.DataFrame) -> float:
    r = float(np.nanmedian(P.mid)) / P.attrs["bin_med"]
    return min(BINS, key=lambda b: abs(np.log(b / r)))


def sim(P: pd.DataFrame, x: np.ndarray, bs: float, t0: int) -> dict:
    """사전등록 매매 규칙. x = 신호(봉마감 초에서만 읽음). t0 이전 초는 문턱 예열만(거래 없음)."""
    idx = P.index.to_numpy(); mid = P.mid.ffill().to_numpy()
    hs = P.bt_spread_bp.to_numpy() / 2e4
    bid = P.mid.to_numpy() * (1 - hs); ask = P.mid.to_numpy() * (1 + hs)
    smin = P.smin.to_numpy(); bmax = P.bmax.to_numpy()
    hist: deque = deque(maxlen=288)
    pos, entry, t_fill, real, target, dec = 0, np.nan, -10**9, 0.0, 0, -10**9
    order = None; eq = np.zeros(len(idx)); fills = []
    for i, s in enumerate(idx):
        if s % 300 == 0 and np.isfinite(x[i]):
            if len(hist) >= 144 and s >= t0:
                thr = np.quantile(np.abs(hist), .8)
                target = 1 if x[i] >= thr else -1 if x[i] <= -thr else 0; dec = i
            hist.append(x[i])
        if order and order["kind"] == "enter" and order["side"] != target:
            order = None                                         # 진입 대기 중 목표가 바뀌면 취소
        if order is None and target != pos and i >= dec + 2 and (pos == 0 or s - t_fill >= 900):
            order = dict(kind="exit", side=-pos, dl=i + 60) if pos else dict(kind="enter", side=target, dl=i + 300)
        if order and i >= 1:
            sd = order["side"]; p = bid[i - 1] if sd > 0 else ask[i - 1]
            hit = np.isfinite(p) and (((smin[i] + 1) * bs <= p) if sd > 0 else (bmax[i] * bs > p))
            if hit:
                fills.append((i, sd, p, True))
                if order["kind"] == "exit":
                    real += pos * (p / entry - 1) * 1e4 - MAKER_BP; pos = 0
                else:
                    pos, entry, t_fill = sd, p, s; real -= MAKER_BP
                order = None
            elif i >= order["dl"]:
                if order["kind"] == "exit" and np.isfinite(bid[i]):
                    px = bid[i] if pos > 0 else ask[i]
                    fills.append((i, -pos, px, False))
                    real += pos * (px / entry - 1) * 1e4 - TAKER_BP; pos = 0
                    order = None
                elif order["kind"] == "enter":
                    order = None
        eq[i] = real + (pos * (mid[i] / entry - 1) * 1e4 if pos else 0.0)
    day = idx // 86400
    last = pd.Series(eq, index=day).groupby(level=0).last()
    daily = last.diff().fillna(last)                             # 첫날은 0 에서 시작
    mk = [f for f in fills if f[3]]
    adv = [f[1] * (mid[min(f[0] + 10, len(mid) - 1)] / f[2] - 1) * 1e4 for f in mk]
    return dict(daily=daily, n=len(fills), n_maker=len(mk), n_taker=len(fills) - len(mk),
                adv10=float(np.mean(adv)) if adv else np.nan)


def trade():
    T0 = T("2026-10-01T00:00Z")
    rows, zs = [], []
    for coin in ("ETH", "BTC", "SOL", "XRP", "HYPE"):
        P = build(coin, trade_mode=True)
        lo, hi = WIN[coin]
        Q = prep(P[(P.index >= lo) & (P.index < hi)], ctrl=CTRL_Q1)
        m, se, n = spread(Q, Q.buy, Q.sell)
        print(f"\n== {coin}  Q1 스프레드(+30·60분 통제) {fmt(m, se, n)}", flush=True)
        if coin != "ETH":
            zs.append(m / se)
        else:
            print(f"  ⇒ Q1 ETH: {'통과' if m > 0 and m / se >= 2 else '불통과'}")
        bs = bin_size(P)
        t_start = lo + 86400 if coin != "ETH" else lo        # 사전등록: 비-ETH 는 24h 예열 뒤(09-27T10Z~) 거래
        for rule, x in (("벽", P.dd_imb50.to_numpy()), ("대조(−30분 수익)", -P.dmid1800.to_numpy())):
            r = sim(P, x, bs, t_start)
            d = r["daily"][r["daily"].index >= t_start // 86400]
            print(f"  [{rule}] 가격칸 {bs:g} · 체결 {r['n']}건(메이커 {r['n_maker']} · 테이커 청산 {r['n_taker']}) · "
                  f"메이커 체결 10초 역선택 {r['adv10']:+.2f}bp · 일 손익 " + " ".join(f"{pd.to_datetime(k*86400, unit='s'):%m-%d}:{v:+.0f}" for k, v in d.items()), flush=True)
            for k, v in d.items():
                rows.append((coin, rule, int(k), float(v)))
    df = pd.DataFrame(rows, columns=["coin", "rule", "day", "pnl"])
    pos = sum(z > 0 for z in zs); Z = sum(zs) / np.sqrt(len(zs))
    print(f"\nQ1 다코인 t = {[round(z, 2) for z in zs]} · 양수 {pos}/4 · Z {Z:+.2f} ⇒ {'통과' if pos >= 3 and Z >= 2 else '불통과'}")
    # 판정용: ETH 는 10-01 부터, 다른 코인은 거래 시작 다음 날 일부터(09-27 은 10시부터 = 부분일 포함)
    ok = (df.coin != "ETH") | (df.day >= T0 // 86400)
    j = df[ok & (df.day >= T("2026-09-27T00:00Z") // 86400)]
    W = j[j.rule == "벽"].groupby("day").pnl.mean(); C = j[j.rule != "벽"].groupby("day").pnl.mean()
    w = msn(W); dlt = msn((W - C).dropna())
    tot = j[(j.rule == "벽") & (j.coin != "ETH")].groupby("coin").pnl.sum()
    print(f"\n일 클러스터 {len(W)}일 · 벽 {fmt(*w)} · 일 표준편차 {W.std():.0f}bp · 대조 {fmt(*msn(C))}")
    print(f"비-ETH 창 합계: " + " ".join(f"{k} {v:+.0f}" for k, v in tot.items()))
    need = (2 * W.std() / w[0]) ** 2 if w[0] > 0 else float("nan")
    print(f"⇒ P1: {'통과' if w[0] > 0 and w[0] / w[1] >= 2 and (tot > 0).sum() >= 3 else '불통과'} (t≥2 에 필요한 일수 ≈ {need:.0f})")
    print(f"벽 − 대조 {fmt(*dlt)} ⇒ P2: {'통과' if dlt[0] > 0 and dlt[0] / dlt[1] >= 2 else '불통과'}")
    ref = df[(df.coin == "ETH") & (df.day < T0 // 86400)]
    print("참고(판정 밖) ETH 09-26~09-30 일 손익: " + " ".join(f"{r.rule}:{r.pnl:+.0f}" for r in ref.itertuples()))


def lags():
    """탐색(사전등록 밖): 벽 신호를 L초 늦게 써도 남는가 — 매매로 옮길 수 있는지의 첫 질문."""
    for coin in ("ETH", "BTC", "SOL", "XRP", "HYPE"):
        P = build(coin)
        print(f"\n== {coin}", flush=True)
        for L in (0, 1, 5, 30, 60, 120):
            Q = prep(P, L)
            print(f"  지연 {L:3d}초  스프레드 {fmt(*spread(Q, Q.buy, Q.sell))} | 매수벽 {fmt(*one(Q, Q.buy))} | 매도벽 {fmt(*one(Q, Q.sell))}", flush=True)


def main():
    if "--selftest" in sys.argv:
        return selftest()
    if "--lags" in sys.argv:
        return lags()
    if "--trade" in sys.argv:
        return trade()
    zs = []
    for coin in ("ETH", "BTC", "SOL", "XRP", "HYPE"):
        Q = prep(build(coin))
        d = (np.log(Q.mid.dropna().iloc[-1]) - np.log(Q.mid.dropna().iloc[0])) * 1e4
        raw = spread(Q, Q.buy, Q.sell, "fwd300"); res = spread(Q, Q.buy, Q.sell)
        print(f"\n== {coin}  {len(Q)/86400:.2f}일 · 구간수익 {d:+.0f}bp · depth 유효 {100*Q.dd_imb50.notna().mean():.0f}% "
              f"· 테이커 유효 {100*Q.tr_imb60.notna().mean():.0f}% · OI 유효 {100*Q.oi_d300.notna().mean():.0f}%", flush=True)
        print(f"  원시 스프레드 {fmt(*raw)}\n  통제 스프레드 {fmt(*res)}")
        print(f"  매수벽 단독 {fmt(*one(Q, Q.buy))} · 매도벽 단독 {fmt(*one(Q, Q.sell))}")
        t = res[0] / res[1]
        if coin == "ETH":
            print(f"  ⇒ H1 판정: {'통과' if res[0] > 0 and t >= 2 else '불통과'}")
            dn = Q.oi_d300 < 0; up = Q.oi_d300 >= 0
            print(f"  OI↓ 안 스프레드 {fmt(*spread(Q, Q.buy & dn, Q.sell & dn))} · OI↑ 안 {fmt(*spread(Q, Q.buy & up, Q.sell & up))}")
            od = oi_diff(Q)
            print(f"  OI↓ − OI↑ {fmt(*od)}")
            print(f"  ⇒ H2 판정: {'통과' if od[0] > 0 and od[0] / od[1] >= 2 else '불통과'}", flush=True)
        else:
            zs.append(t)
    pos = sum(z > 0 for z in zs); Z = sum(zs) / np.sqrt(len(zs))
    print(f"\n다코인 t = {[round(z, 2) for z in zs]} · 양수 {pos}/4 · Stouffer Z = {Z:+.2f}")
    print(f"⇒ H3 판정: {'통과' if pos >= 3 and Z >= 2 else '불통과'}")


if __name__ == "__main__":
    main()
