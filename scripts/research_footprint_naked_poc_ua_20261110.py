"""맨 POC·미완 경매 5분 레벨 재방문 vs 위약 -- 사전등록 docs/prereg/footprint_naked_poc_unfinished_auction_20261110.md (770acc21).

실행: python scripts/research_footprint_naked_poc_ua_20261110.py <tape lake coin=ETH 폴더> [--selfcheck]
원천 = 바이낸스 ETHUSDT 체결 tape(1초 × $0.1 칸). 행 $0.5 · 결정 τ = 봉 b+1 마감 · 결과 창 [τ, τ+H) 초.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

ROW, GAP, BAR, DAY = 5, 120, 300, 86400   # 행 = $0.1 칸 5개($0.5) · 결측 연속 120초 넘으면 제외
SHIFTS = (1, 2, 3, -1, -2, -3)


def load(folder: str) -> pd.DataFrame:
    fs = sorted(glob.glob(f"{folder}/date=*/part.parquet"))
    t = pd.concat([pd.read_parquet(f, columns=["symbol", "ts_sec", "price_bin", "buy_qty", "sell_qty"]) for f in fs], ignore_index=True)
    assert set(t.symbol.str.upper().unique()) == {"ETHUSDT"}, t.symbol.unique()   # 수집기는 소문자 심볼로 쓴다
    return t.drop(columns="symbol")


def seconds(t: pd.DataFrame):
    """전체 초 격자: 최저·최고 체결가·VWAP(그 초) · 결측 표시 · 앞쪽 창 최고/최저 준비."""
    px = t.price_bin * 0.1
    q = t.buy_qty + t.sell_qty
    g = pd.DataFrame({"s": t.ts_sec, "px": px, "pq": px * q, "q": q}).groupby("s").agg(lo=("px", "min"), hi=("px", "max"), pq=("pq", "sum"), q=("q", "sum"))
    s0, s1 = int(g.index.min()), int(g.index.max())
    n = s1 - s0 + 1
    lo, hi, vw = (np.full(n, np.nan) for _ in range(3))
    i = g.index.to_numpy() - s0
    lo[i], hi[i], vw[i] = g.lo, g.hi, g.pq / g.q
    empty = np.isnan(vw)
    # 연속 결측 120초 넘는 칸을 «나쁨»으로
    bad = np.zeros(n, bool)
    run_start = None
    for k in np.flatnonzero(np.diff(np.r_[0, empty.astype(int), 0]) != 0):
        if run_start is None:
            run_start = k
        else:
            if k - run_start > GAP:
                bad[run_start:k] = True
            run_start = None
    badc = np.r_[0, np.cumsum(bad)]
    # 마지막 체결 VWAP(앞으로 채움) · 다음 체결 VWAP(뒤로 채움)
    last = pd.Series(vw).ffill().to_numpy()
    nxt = pd.Series(vw).bfill().to_numpy()
    return s0, lo, hi, last, nxt, badc


def fwd_ext(arr: np.ndarray, H: int, fn: str) -> np.ndarray:
    """out[t] = arr[t .. t+H-1] 의 max/min (결측 무시)."""
    r = pd.Series(arr[::-1]).rolling(H, min_periods=1)
    return (r.max() if fn == "max" else r.min()).to_numpy()[::-1]


def bars(t: pd.DataFrame) -> pd.DataFrame:
    t = t.assign(bar=t.ts_sec // BAR, row=np.rint(t.price_bin / ROW).astype(int), tot=t.buy_qty + t.sell_qty)
    cells = t.groupby(["bar", "row"], sort=True).agg(buy=("buy_qty", "sum"), sell=("sell_qty", "sum"), tot=("tot", "sum")).reset_index()
    out = []
    for b, c in cells.groupby("bar", sort=True):
        hi_r, lo_r = c.row.max(), c.row.min()
        top, bot = c[c.row == hi_r].iloc[0], c[c.row == lo_r].iloc[0]
        out.append((b, hi_r, lo_r, c, top.buy > 0 and top.sell > 0, (top.buy > 0) != (top.sell > 0),
                    bot.buy > 0 and bot.sell > 0, (bot.buy > 0) != (bot.sell > 0)))
    return out


def poc_row(c: pd.DataFrame, close_px: float) -> int:
    m = c[c.tot == c.tot.max()]
    return int(m.row.iloc[np.argmin(np.abs(m.row.to_numpy() * 0.5 - close_px))])


class Ctx:
    def __init__(self, t: pd.DataFrame):
        self.s0, self.lo, self.hi, self.last, self.nxt, self.badc = seconds(t)
        self.n = len(self.lo)
        self.ext = {}

    def ok(self, a: int, b: int) -> bool:   # 초 [a, b) 안에 나쁜 칸 없고 데이터 범위 안
        ia, ib = a - self.s0, b - self.s0
        return 0 <= ia and ib <= self.n and self.badc[ib] - self.badc[ia] == 0

    def P(self, tau: int) -> float:   # τ 직전 마지막 체결 초의 VWAP
        return float(self.last[tau - 1 - self.s0])

    def touch(self, tau: int, H: int, L: float, P: float) -> bool:
        if (H, "max") not in self.ext:
            self.ext[(H, "max")], self.ext[(H, "min")] = fwd_ext(self.hi, H, "max"), fwd_ext(self.lo, H, "min")
        i = tau - self.s0
        return bool(self.ext[(H, "max")][i] >= L - 0.25) if L > P else bool(self.ext[(H, "min")][i] <= L + 0.25)

    def placebo(self, tau: int, H: int, d: float, r: int, shifts=SHIFTS) -> float | None:
        hits = []
        for k in shifts:
            tp = tau + k * DAY
            if not self.ok(tp - 600, tp + H):
                continue
            Pp, op = self.P(tp), float(self.nxt[tp - 600 - self.s0])
            mp = int(np.sign(Pp - op))
            if mp == 0:
                continue
            Lp = Pp * (1 + r * mp * d)
            hits.append(self.touch(tp, H, Lp, Pp))
        return float(np.mean(hits)) if hits else None


def events(t: pd.DataFrame, cx: Ctx) -> list[dict]:
    bl = bars(t)
    ev = []
    for (b, hi_r, lo_r, c, ua_hi, fa_hi, ua_lo, fa_lo), (b1, hi1, lo1, *_ ) in zip(bl, bl[1:]):
        if b1 != b + 1:
            continue
        tau = (b + 2) * BAR
        if not cx.ok(b * BAR, tau + 4 * 3600):
            continue
        P = cx.P(tau)
        op = float(cx.nxt[b * BAR - cx.s0])
        m = int(np.sign(P - op))
        if m == 0:
            continue
        close_b = cx.P((b + 1) * BAR)
        cand = []
        pr = poc_row(c, close_b)
        if not (lo1 <= pr <= hi1):
            cand.append(("npoc", pr * 0.5))
        if hi1 < hi_r:
            if ua_hi: cand.append(("ua", hi_r * 0.5))
            if fa_hi: cand.append(("fa", hi_r * 0.5))
        if lo1 > lo_r:
            if ua_lo: cand.append(("ua", lo_r * 0.5))
            if fa_lo: cand.append(("fa", lo_r * 0.5))
        for kind, L in cand:
            s = int(np.sign(L - P))
            if s == 0:
                continue
            ev.append({"kind": kind, "tau": tau, "day": tau // DAY, "L": L, "P": P, "s": s, "m": m, "d": abs(L - P) / P})
    return ev


def score(ev: list[dict], cx: Ctx, H: int, level=None) -> pd.DataFrame:
    rows = []
    for e in ev:
        L = e["L"] if level is None else level(e)
        if L is None:
            continue
        s = int(np.sign(L - e["P"]))
        if s == 0:
            continue
        d = abs(L - e["P"]) / e["P"]
        pl = cx.placebo(e["tau"], H, d, s * e["m"])
        if pl is None:
            continue
        rows.append({"day": e["day"], "hit": cx.touch(e["tau"], H, L, e["P"]), "pl": pl,
                     "mir": cx.touch(e["tau"], H, e["P"] * (1 - s * d), e["P"]), "dbp": d * 1e4,
                     "pl_p1": cx.placebo(e["tau"], H, d, s * e["m"], (1,)), "pl_m1": cx.placebo(e["tau"], H, d, s * e["m"], (-1,))})
    return pd.DataFrame(rows)


def boot(df: pd.DataFrame, a: str, b: str, n: int = 5000, seed: int = 7) -> dict:
    df = df.dropna(subset=[a, b])
    diff = df[a].astype(float) - df[b].astype(float)
    g = pd.DataFrame({"day": df.day, "x": diff}).groupby("day").x.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    est = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return {"delta": float(diff.mean()), "lo90": float(np.quantile(est, .10)), "ci90": [float(np.quantile(est, .05)), float(np.quantile(est, .95))],
            "p_le0": float((est <= 0).mean()), "n": int(len(df)), "days": int(len(g)),
            "hit": f"{int(df[a].astype(float).sum())}/{len(df)}", "rate_a": float(df[a].astype(float).mean()), "rate_b": float(df[b].astype(float).mean())}


def holm(ps: dict, alpha: float = .10) -> dict:
    out, k = {}, len(ps)
    for j, (name, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        out[name] = p <= alpha / (k - j) and all(out.values())
    return out


def main(folder: str) -> None:
    t = load(folder)
    cx = Ctx(t)
    ev = events(t, cx)
    res = {"events": {k: sum(e["kind"] == k for e in ev) for k in ("npoc", "ua", "fa")}, "days_n": int(t.ts_sec.floordiv(DAY).nunique())}
    sc = {}
    for H in (3600, 14400):
        for k in ("npoc", "ua", "fa"):
            sc[(k, H)] = score([e for e in ev if e["kind"] == k], cx, H)
    # 장치 대조(맨 POC 사건 기준, H=1h)
    pos = score([e for e in ev if e["kind"] == "npoc"], cx, 3600, level=lambda e: float(cx.nxt[e["tau"] + 1800 - cx.s0]))
    res["ctrl_positive"] = boot(pos, "hit", "pl")
    res["ctrl_negative"] = boot(sc[("npoc", 3600)].dropna(subset=["pl_p1", "pl_m1"]), "pl_p1", "pl_m1")
    for (k, H), df in sc.items():
        res[f"{k}_H{H // 3600}h_vs_shift"] = boot(df, "hit", "pl")
        res[f"{k}_H{H // 3600}h_vs_mirror"] = boot(df, "hit", "mir")
        df = df.assign(bin=pd.cut(df.dbp, [0, 10, 30, 1e9], labels=["<=10bp", "10-30bp", ">30bp"]))
        res[f"{k}_H{H // 3600}h_dist"] = {str(b): {"n": int(len(g)), "hit": float(g.hit.mean()), "pl": float(g.pl.mean()), "mir": float(g.mir.mean())} for b, g in df.groupby("bin", observed=True)}
        res[f"{k}_H{H // 3600}h_dbp_median"] = float(sc[(k, H)].dbp.median())
    ps = {"A_npoc": res["npoc_H1h_vs_shift"]["p_le0"], "B_ua": res["ua_H1h_vs_shift"]["p_le0"]}
    pm = {"A_npoc": res["npoc_H1h_vs_mirror"]["p_le0"], "B_ua": res["ua_H1h_vs_mirror"]["p_le0"]}
    res["holm_main"], res["holm_mirror"] = holm(ps), holm(pm)
    # 보조: 미완 vs 끝난 경매 닿음률(H=1h) -- 서로 다른 사건이라 짝 차가 아니라 일 블록 평균 차
    ua, fa = sc[("ua", 3600)], sc[("fa", 3600)]
    res["aux_ua_minus_fa_1h"] = {"ua": f"{int(ua.hit.sum())}/{len(ua)}", "fa": f"{int(fa.hit.sum())}/{len(fa)}", "diff": float(ua.hit.mean() - fa.hit.mean())}
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


def selfcheck() -> None:
    assert holm({"a": .04, "b": .09}) == {"a": True, "b": True}
    assert holm({"a": .06, "b": .09}) == {"a": False, "b": False}
    assert holm({"a": .04, "b": .2}) == {"a": True, "b": False}
    a = np.array([1., np.nan, 3., 2., np.nan])
    assert list(fwd_ext(a, 2, "max")) == [1., 3., 3., 2., np.nan] or np.allclose(fwd_ext(a, 2, "max")[:4], [1, 3, 3, 2])
    c = pd.DataFrame({"row": [10, 11, 12], "tot": [5., 9., 9.]})
    assert poc_row(c, 6.05) == 12 and poc_row(c, 5.4) == 11
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main(sys.argv[1])
