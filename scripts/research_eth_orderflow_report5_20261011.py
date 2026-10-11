"""오더플로우 레포트 5전략을 «레포트 문구 그대로» 기계화해 검정 (2026-10-11, 사용자 «위 레포트의 설명에 따라 다시 검증»).

판정 원칙(10-10 사용자): 주 지표 = 그 규칙이 말하는 «목적»의 정확도. 수익(테이커 왕복 8bp 뺀 값)은 보조.
느슨한 검정(사용자 «너무 타이트하지 않게»): 주 기간 2025-01~ 초과의 일 블록 부트스트랩 95% CI 가 0 배제 + 2025·2026 같은 부호면 «통과».

비교 원칙: 초과 = 사건 − 대조. 대조는 «레포트 규칙에서 검정하려는 요소 하나만 뺀 것».
  S1·S3   같은 위치·같은 가격 확인·같은 손절인데 오더플로 조건(stack·큰 델타)이 없는 봉 → 오더플로 요소의 몫.
  S4·S5b  같은 규칙을 위약 레벨(7일 전 레벨 × 오늘 시가/그날 시가)에 적용 → 레벨 요소의 몫.
  S2a     같은 |30분 수익| 10분위 안 전체 → 델타 요소의 몫.  S5a 수용 아님이지만 지금 바깥(같은 거리 10분위).
층화(q)는 각 시험의 거리/위험/크기 10분위(사건 비중으로 칸 차이를 가중).

데이터: ETHUSDT 1분봉(klines1m 월 + klines1m_daily) → 5분봉, 델타 = 2·taker_buy − 볼륨.
        실제 가격줄 풋프린트 칸(5분 × $0.5, build_eth_footprint_cells_5m_20261005.py 캐시, 2023-01~2026-10-04).
        S2c 만 실측 청산(바이낸스, 2026-09-20~10-08)·OI 5분(metrics, 스탬프 규약은 상관으로 확인).
경계: 판정은 확인 봉 j 종가에서, 라벨은 j 이후(j+H 종가, 경주는 j+1 봉부터). 레벨은 전일(UTC) 프로파일.

S1 풋프린트  지지(전일 저가·VAL·24h 저가) 0.5ATR 안 봉에 매수 stacked imbalance(대각 3:1, 3줄) →
             6봉 안 직전 12봉 고점 돌파 + 그동안 구간 아래끝 유지 → 롱(숏 거울). 목적: 이후 오른다.
             대조: 같은 위치 양봉(stack 없음)에 같은 확인. 손절 = 사건 봉 저가.
S2 델타      (a) 30분 델타·가격 일치(A) vs 불일치(B/C): 레포트 해석은 둘 다 가격 방향 → 그 방향으로 이어지나.
             (b) CVD 다이버전스: 4h 신고가인데 CVD 는 4h 최고 아님 → 목적: 이후 2h 연장(+0.5ATR) 실패가 잦다.
             (c) 델타×OI 표: 상승·델타+·OI↓ 봉은 OI↑ 봉보다 숏 청산이 크다(목적 = 원인 귀속).
S3 흡수/공격 흡수 숏: 저항 0.5ATR 안, 15분 델타 합 상위 20%, 15분 진척 ≤ 0.25ATR → 6봉 안 15분 저점 이탈 → 숏,
             손절 = 15분 고점(무효화). 롱 거울. 대조: 같은 조건인데 |델타 합| ≤ 중앙. 목적: 1R 먼저 닿기.
             공격: 큰 델타 + stack + 12봉 고점 돌파 + 다음 봉 저가가 돌파선 위 → 롱(거울). 대조: 같은 돌파·유지, 델타 ≤ 중앙·stack 없음.
S4 LVN/HVN   LVN: 봉이 LVN 을 넘어 마감 + 그 방향 큰 델타 → 목적: 24봉 안 다음 HVN 도달. 위약 레벨 대비.
             HVN: 아래에서 HVN 에 닿은 봉에 큰 매수 델타 → 3봉 뒤 HVN 위면 롱 / 0.2ATR 아래면 숏. 위약 레벨 대비.
S5 AMT       전일 VAH 상향 첫 돌파(VAL 거울) → 30분 관찰. 수용 = 6종가 중 ≥4 바깥 + 관찰 거래량 ≥50% 바깥.
             (a) 수용이면 2h 뒤에도 바깥.  (b) 거부(관찰 끝 종가가 안으로 복귀) → 24봉 안 POC 도달.

실행: python scripts/research_eth_orderflow_report5_20261011.py [--selftest]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET, RATIO, STACK_MIN, ATR_N = 0.5, 3.0, 3, 12
COST = 8.0
HS = (6, 12, 24)
PERIODS = {"주 2025~": ("2025-01-01", "2100-01-01"), "2025": ("2025-01-01", "2026-01-01"),
           "2026": ("2026-01-01", "2100-01-01"), "보조 2023~24": ("2023-01-01", "2025-01-01")}
CELLS = "../execution-footprint-imbalance-044af8/tmp/footprint_cells/ETHUSDT"


def main_repo() -> Path:
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    return Path(common).resolve().parent


# ---------------------------------------------------------------- 데이터
def load_bars(t0: int, t1: int) -> pd.DataFrame:
    root = main_repo() / "data/binance_vision"
    fs = sorted(glob.glob(str(root / "klines1m/ETHUSDT-1m-*.parquet"))) + \
        sorted(glob.glob(str(root / "klines1m_daily/ETHUSDT-1m-*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["t", "h", "l", "c", "v", "tb"]) for f in fs]).drop_duplicates("t")
    d = d[(d.t >= t0 * 1000) & (d.t < t1 * 1000)].sort_values("t")
    d["b"] = d.t // 300_000 * 300
    g = d.groupby("b", sort=True)
    o = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "v": g.v.sum(), "tb": g.tb.sum()})
    o = o.reindex(np.arange(o.index.min(), o.index.max() + 300, 300))
    o["open"] = o.close.shift(1)
    o["delta"] = 2 * o.tb - o.v
    pc = o.close.shift(1)
    tr = np.maximum(o.high - o.low, np.maximum((o.high - pc).abs(), (o.low - pc).abs()))
    o["atr"] = tr.rolling(ATR_N, min_periods=ATR_N).mean()
    o["rs"] = np.maximum(BUCKET, np.round(0.2 * o.atr / BUCKET) * BUCKET)
    o["dn"] = o.delta / o.delta.abs().rolling(288, min_periods=200).mean().shift(1)   # 직전 24h 평균 |델타| 단위
    return o


def bar_flags(cells: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """봉별 대각 stacked imbalance(대시보드 행 = 0.2ATR). 10-05 교과서 검정의 DIAG 정의 그대로."""
    rs = bars.rs.reindex(cells.b.to_numpy()).to_numpy()
    c = cells.assign(r=np.floor(cells.k.to_numpy() * BUCKET / rs + 1e-9), rs=rs).dropna(subset=["r"])
    c = c.groupby(["b", "r"], sort=True, as_index=False).agg(buy=("buy", "sum"), sell=("sell", "sum"), rs=("rs", "first"))
    b, r, bu, se = c.b.to_numpy(), c.r.to_numpy(), c.buy.to_numpy(), c.sell.to_numpy()
    same_prev = np.r_[False, (b[1:] == b[:-1]) & (r[1:] == r[:-1] + 1)]
    same_next = np.r_[same_prev[1:], False]
    sell_below = np.where(same_prev, np.r_[0.0, se[:-1]], 0.0)
    buy_above = np.where(same_next, np.r_[bu[1:], 0.0], 0.0)
    nz = pd.Series(np.r_[bu, se], index=np.r_[b, b])
    med = nz[nz > 0].groupby(level=0).median().reindex(b).to_numpy()
    flags = {"b": (bu > 0) & (bu > RATIO * sell_below) & (bu >= med),
             "s": (se > 0) & (se > RATIO * buy_above) & (se >= med)}
    out = pd.DataFrame(index=bars.index)
    for name, f in flags.items():
        run_id = np.cumsum(~(f & np.r_[False, f[:-1]] & same_prev))
        R = pd.DataFrame({"b": b, "r": r, "run": run_id, "rs": c.rs.to_numpy()})[f]
        g = R.groupby("run").agg(b=("b", "first"), lo=("r", "min"), hi=("r", "max"), n=("r", "size"), rs=("rs", "first"))
        g = g[g.n >= STACK_MIN]
        g = g.sort_values(["b", "n", "lo"], ascending=[True, False, name == "b"]).drop_duplicates("b").set_index("b")
        out[f"stk_{name}"] = out.index.isin(g.index)
        out[f"zlo_{name}"] = (g.lo * g.rs).reindex(out.index)
        out[f"zhi_{name}"] = ((g.hi + 1) * g.rs).reindex(out.index)
    return out


def day_profile(price: np.ndarray, vol: np.ndarray) -> dict:
    """한 날 가격줄 거래량 → POC·VAH·VAL(70%)·HVN·LVN. 칸 = 중앙가 5bp($0.5 배수), 노드는 5칸 평활 국소 극값."""
    bw = max(BUCKET, round(np.median(price) * 5e-4 / BUCKET) * BUCKET)
    i = np.floor((price - price.min()) / bw).astype(int)
    h = np.bincount(i, weights=vol)
    poc = int(np.argmax(h)); lo = hi = poc; tot = h[poc]
    while tot < 0.7 * h.sum() and (lo > 0 or hi < len(h) - 1):
        up = h[hi + 1] if hi < len(h) - 1 else -1.0
        dn = h[lo - 1] if lo > 0 else -1.0
        if up >= dn:
            hi += 1; tot += up
        else:
            lo -= 1; tot += dn
    s = np.convolve(h, np.ones(5) / 5, mode="same")
    W = 5
    mx = pd.Series(s).rolling(2 * W + 1, center=True, min_periods=1).max().to_numpy()
    mn = pd.Series(s).rolling(2 * W + 1, center=True, min_periods=1).min().to_numpy()
    hv = np.flatnonzero((s == mx) & (s >= 1.2 * s.mean()))
    lv = np.flatnonzero((s == mn) & (s <= 0.6 * s.mean()))
    if len(hv):
        lv = lv[(lv > hv.min()) & (lv < hv.max())]
    else:
        lv = lv[:0]
    p = lambda j: price.min() + (np.asarray(j) + 0.5) * bw
    return {"poc": float(p(poc)), "vah": float(p(hi) + bw / 2), "val": float(p(lo) - bw / 2),
            "pdh": float(price.max()), "pdl": float(price.min()), "hvn": p(hv), "lvn": p(lv)}


# ---------------------------------------------------------------- 라벨·통계
def fwd(close: np.ndarray, H: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    out[:-H] = (close[H:] / close[:-H] - 1) * 1e4
    return out


def race(high, low, j, sgn, entry, stop, n=24, rr=1.0):
    """j 종가 진입, j+1 부터 n 봉: +rr·R 먼저면 1, 손절 먼저(같은 봉 둘 다 = 손절)면 0, 둘 다 없으면 nan."""
    R = abs(entry - stop)
    if not np.isfinite(R) or R <= 0:
        return np.nan
    tp = entry + sgn * rr * R
    for k in range(j + 1, min(j + 1 + n, len(high))):
        if (sgn > 0 and low[k] <= stop) or (sgn < 0 and high[k] >= stop):
            return 0.0
        if (sgn > 0 and high[k] >= tp) or (sgn < 0 and low[k] <= tp):
            return 1.0
    return np.nan


def strat_ci(val, ev, q, day, B=1000, seed=7):
    """층화 초과(사건−대조, 사건 비중 가중)와 일 블록 포아송 부트스트랩 95%."""
    days, di = np.unique(day, return_inverse=True)
    Q, D = int(q.max()) + 1, len(days)
    idx = di * Q + q
    S = [np.bincount(idx[ev == g], weights=val[ev == g], minlength=D * Q).reshape(D, Q) for g in (1, 0)]
    N = [np.bincount(idx[ev == g], minlength=D * Q).reshape(D, Q).astype(float) for g in (1, 0)]
    ok = (N[0].sum(0) > 0) & (N[1].sum(0) > 0)
    pi = np.where(ok, N[0].sum(0), 0.0); pi = pi / pi.sum()

    def est(w):
        with np.errstate(invalid="ignore", divide="ignore"):
            dlt = (w @ S[0]) / (w @ N[0]) - (w @ S[1]) / (w @ N[1])
        return np.nansum(dlt * pi, -1)
    point = float(est(np.ones(D)))
    w = np.random.default_rng(seed).poisson(1.0, (B, D)).astype(float)
    lo, hi = np.nanpercentile(est(w), [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(val, ev, q, day, sel):
    m1 = sel & (ev == 1) & np.isfinite(val); m0 = sel & (ev == 0) & np.isfinite(val)
    if m1.sum() < 20 or m0.sum() < 20:
        return {"n": int(m1.sum())}
    k = m1 | m0
    ex, lo, hi = strat_ci(val[k], ev[k].astype(int), q[k], day[k])
    return {"n": int(m1.sum()), "n_ctl": int(m0.sum()), "ev": round(float(val[m1].mean()), 3),
            "ctl": round(float(val[m0].mean()), 3), "ex": round(ex, 3), "ci": [round(lo, 3), round(hi, 3)]}


def dedup(idx: np.ndarray, gap: int = 12) -> np.ndarray:
    keep, last = [], -10**9
    for i in np.sort(idx):
        if i - last >= gap:
            keep.append(i); last = i
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------- 시험 공통 묶음
class Ctx:
    def __init__(self, bars, flags, levels, levels_pl, cells):
        self.o = bars; self.f = flags
        self.H, self.L, self.C, self.Op = (bars[k].to_numpy() for k in ("high", "low", "close", "open"))
        self.atr = bars.atr.to_numpy(); self.dn = bars.dn.to_numpy()
        self.ts = bars.index.to_numpy(); self.day = self.ts // 86400
        self.lv, self.lvp = levels, levels_pl
        self.cells = cells; self.cb = cells.b.to_numpy()
        self.ck = cells.k.to_numpy() * BUCKET; self.cv = (cells.buy + cells.sell).to_numpy()
        self.rhi = pd.Series(self.H).shift(12).rolling(276, min_periods=200).max().to_numpy()
        self.rlo = pd.Series(self.L).shift(12).rolling(276, min_periods=200).min().to_numpy()
        self.hi12 = pd.Series(self.H).rolling(12).max().to_numpy()      # t 포함 직전 12봉
        self.lo12 = pd.Series(self.L).rolling(12).min().to_numpy()
        self.fw = {H: fwd(self.C, H) for H in HS}
        self.n = len(self.C)
        self.near_ = {s: self._near_all(s) for s in (+1, -1)}

    def _near_all(self, side):
        """side +1 지지(저가 기준) / −1 저항(고가 기준): 전일 극값·VA 끝·24h 극값 중 하나에 0.5ATR 안."""
        key = ("pdl", "val") if side > 0 else ("pdh", "vah")
        x = self.L if side > 0 else self.H
        ok = np.zeros(self.n, bool)
        lv = [np.array([self.lv.get(int(d), {}).get(k, np.nan) for d in self.day]) for k in key]
        for L in lv + [self.rlo if side > 0 else self.rhi]:
            ok |= np.abs(x - L) <= 0.5 * self.atr
        return ok

    def vol_beyond_share(self, i0, i1, level):
        a, b = np.searchsorted(self.cb, self.ts[i0]), np.searchsorted(self.cb, self.ts[i1], side="right")
        v, p = self.cv[a:b], self.ck[a:b]
        return float(v[p > level].sum() / v.sum()) if v.sum() > 0 else np.nan


def rows(ctx, j, sgn, stop, ev, q, tag):
    r = {"tag": tag, "j": j, "sgn": sgn, "ev": ev, "q": q, "day": int(ctx.day[j]),
         "race1": race(ctx.H, ctx.L, j, sgn, ctx.C[j], stop) if np.isfinite(stop) else np.nan}
    for H in HS:
        r[f"r{H}"] = sgn * ctx.fw[H][j]
    return r


# ---------------------------------------------------------------- S1 풋프린트
def s1(ctx, out):
    for sgn, s in ((+1, "b"), (-1, "s")):
        stk, zlo, zhi = ctx.f[f"stk_{s}"].to_numpy(), ctx.f[f"zlo_{s}"].to_numpy(), ctx.f[f"zhi_{s}"].to_numpy()
        other = ctx.f["stk_" + ("s" if s == "b" else "b")].to_numpy()
        body = np.sign(ctx.C - ctx.Op)
        for i in np.flatnonzero(ctx.near_[sgn]):
            if i < 300 or i > ctx.n - 40 or other[i]:
                continue
            ev = 1 if stk[i] else (0 if body[i] == sgn else -1)
            if ev < 0:
                continue
            ref = ctx.hi12[i] if sgn > 0 else ctx.lo12[i]
            edge = (zlo[i] if sgn > 0 else zhi[i]) if ev else (ctx.L[i] if sgn > 0 else ctx.H[i])
            for j in range(i + 1, i + 7):
                if (sgn > 0 and ctx.L[j] < edge) or (sgn < 0 and ctx.H[j] > edge):
                    break
                if (sgn > 0 and ctx.C[j] > ref) or (sgn < 0 and ctx.C[j] < ref):
                    stop = ctx.L[i] if sgn > 0 else ctx.H[i]
                    out.append(rows(ctx, j, sgn, stop, ev, abs(ctx.C[j] - stop) / ctx.atr[j], "S1 풋프린트 stack→돌파"))
                    break


# ---------------------------------------------------------------- S2 델타
def s2(ctx, out, th):
    d6 = pd.Series(ctx.dn).rolling(6).sum().to_numpy()
    r6 = np.full(ctx.n, np.nan); r6[6:] = (ctx.C[6:] / ctx.C[:-6] - 1) * 1e4
    for i in range(300, ctx.n - 30, 6):               # 겹치지 않는 30분 창
        if not (np.isfinite(d6[i]) and np.isfinite(r6[i]) and r6[i] != 0 and np.isfinite(ctx.atr[i])):
            continue
        sgn = int(np.sign(r6[i]))
        big = d6[i] >= th["d6_hi"] or d6[i] <= th["d6_lo"]
        agree = np.sign(d6[i]) == sgn
        q = abs(r6[i]) / ctx.atr[i]
        if not big or agree:
            out.append(rows(ctx, i, sgn, np.nan, int(big), q, "S2a 델타·가격 일치(A)→지속"))
        if not big or not agree:
            out.append(rows(ctx, i, sgn, np.nan, int(big), q, "S2a 델타·가격 불일치(B/C)→가격 방향"))
    # (b) CVD 다이버전스
    cvd = np.cumsum(np.nan_to_num(ctx.o.delta.to_numpy()))
    hh = pd.Series(ctx.H).shift(1).rolling(48).max().to_numpy(); ll = pd.Series(ctx.L).shift(1).rolling(48).min().to_numpy()
    ch = pd.Series(cvd).shift(1).rolling(48).max().to_numpy(); cl = pd.Series(cvd).shift(1).rolling(48).min().to_numpy()
    for sgn in (+1, -1):
        cand = np.flatnonzero((ctx.H > hh) if sgn > 0 else (ctx.L < ll))
        for i in dedup(cand[(cand > 300) & (cand < ctx.n - 30)]):
            if not np.isfinite(ctx.atr[i]):
                continue
            ev = int(cvd[i] < ch[i]) if sgn > 0 else int(cvd[i] > cl[i])
            ext = (ctx.H[i + 1:i + 25].max() >= ctx.H[i] + 0.5 * ctx.atr[i]) if sgn > 0 else \
                  (ctx.L[i + 1:i + 25].min() <= ctx.L[i] - 0.5 * ctx.atr[i])
            q = ((ctx.H[i] - hh[i]) if sgn > 0 else (ll[i] - ctx.L[i])) / ctx.atr[i]
            r = rows(ctx, i, -sgn, np.nan, ev, q, "S2b CVD 다이버전스→연장 실패")
            r["no_ext"] = float(not ext)
            out.append(r)


def s2c(cells) -> dict:
    """델타×OI 표의 «숏/롱 청산 기여» 귀속을 실측 청산으로(바이낸스 2026-09-20~10-04).
    1분봉이 09-15~10-04 에 없어 봉은 풋프린트 칸 델타 + 1초 마크가격 5분 수익으로 만든다."""
    root = main_repo() / "tmp/liqmap_validation_20261008"
    mk = pd.read_parquet(root / "eth_mark_1s_20261008.parquet")
    mk["b"] = mk.ts_ms // 300_000 * 300
    px = mk.groupby("b").mark.agg(["first", "last"])
    cg = cells[cells.b >= px.index.min()].groupby("b")
    bars = pd.DataFrame({"open": px["first"], "close": px["last"]}).join(
        (cg.buy.sum() - cg.sell.sum()).rename("delta"), how="inner")
    liq = pd.read_parquet(root / "eth_liq_all_20261008.parquet")
    liq = liq[liq.venue == "binance"].copy()
    liq["b"] = liq.ts_ms // 300_000 * 300
    liq["usd"] = liq.qty * liq.price
    lq = liq.pivot_table(index="b", columns="side", values="usd", aggfunc="sum").fillna(0)
    m = pd.read_parquet(root / "metrics_5m.parquet")
    oi = pd.Series(m.oi_usd.to_numpy(), index=m.ts.astype("int64").to_numpy() // 10**9).sort_index()
    o = bars.loc[(bars.index >= lq.index.min()) & (bars.index <= lq.index.max())].copy()
    o["ret"] = np.log(o.close / o.open)
    o = o.join(lq, how="left").fillna({"short": 0.0, "long": 0.0})
    # oi_usd 는 계약×가격이라 가격이 ΔOI 를 지배한다 → 계약 수(ETH)로 바꾼다.
    oi = oi.reindex(o.index)
    # S: 스탬프 = 스냅샷(봉 T 시가 시점) → 봉 T 변화 = oi[T+5]/종가 − oi[T]/시가
    dS = oi.shift(-1) / o.close - oi / o.open
    # P: 스탬프 + 5분(2024-03-04 이후 규약) → 봉 T 종가 시점 = oi[T] → 봉 T 변화 = oi[T]/종가[T] − oi[T−5]/종가[T−5]
    cp = oi / o.close; dP = cp - cp.shift(1)
    liqsum = o.short + o.long
    rep = {"n봉": len(o), "corr(수익, short 청산$)": round(float(o.ret.corr(o.short)), 3),
           "corr(수익, long 청산$)": round(float(o.ret.corr(o.long)), 3),
           "corr(청산합, −ΔOI계약) S/P": [round(float(liqsum.corr(-dS)), 3), round(float(liqsum.corr(-dP)), 3)]}
    o["doi"] = dP if liqsum.corr(-dP) >= liqsum.corr(-dS) else dS
    o["absq"] = pd.qcut(o.ret.abs(), 5, labels=False, duplicates="drop")
    res = {}
    for name, cond, col in (("상승·델타+ → 숏청산", (o.ret > 0) & (o.delta > 0), "short"),
                            ("하락·델타− → 롱청산", (o.ret < 0) & (o.delta < 0), "long")):
        g = o[cond & o.doi.notna()]
        a = []
        for _, x in g.groupby("absq"):
            dn_, up_ = x[x.doi < 0], x[x.doi > 0]
            if len(dn_) > 10 and len(up_) > 10:
                a.append((dn_[col].mean(), up_[col].mean(), (dn_[col] > 0).mean(), (up_[col] > 0).mean()))
        a = np.array(a)
        res[name] = {"n OI↓": int((g.doi < 0).sum()), "n OI↑": int((g.doi > 0).sum()),
                     "청산$ OI↓": round(float(a[:, 0].mean())), "청산$ OI↑": round(float(a[:, 1].mean())),
                     "청산 있는 봉 비율 OI↓": round(float(a[:, 2].mean()), 3), "OI↑": round(float(a[:, 3].mean()), 3)}
    rep["표(|수익| 5분위 안 평균)"] = res
    return rep


# ---------------------------------------------------------------- S3 흡수·공격
def s3(ctx, out, th):
    d3 = pd.Series(ctx.dn).rolling(3).sum().to_numpy()
    near3 = {s: ctx.near_[s] | np.r_[False, ctx.near_[s][:-1]] | np.r_[False, False, ctx.near_[s][:-2]] for s in (+1, -1)}
    for sgn in (+1, -1):   # sgn = 진입 방향. 숏(−1) = 저항에서 매수 흡수, 롱(+1) = 지지에서 매도 흡수
        th_big = th["d3_hi"] if sgn < 0 else -th["d3_lo"]
        # near_[+1] = 지지(저가 기준), near_[−1] = 저항(고가 기준) -- 숏(sgn=−1) 은 저항, 롱(+1) 은 지지.
        # 🔴10-12 수정: 처음엔 near3[-sgn] 이라 반대 위치(숏을 지지에서)를 검정했다.
        for i in np.flatnonzero(near3[sgn]):
            if i < 300 or i > ctx.n - 40 or not np.isfinite(ctx.atr[i]):
                continue
            prog = -sgn * (ctx.C[i] - ctx.C[i - 3]) / ctx.atr[i]       # 공격 방향 진척(ATR)
            if not np.isfinite(prog) or prog > 0.25:
                continue
            agg = -sgn * d3[i]                                           # 공격 쪽 델타 합(+ = 큼)
            ev = 1 if agg >= th_big else (0 if abs(d3[i]) <= th["d3_med"] else -1)
            if ev < 0:
                continue
            wh, wl = ctx.H[i - 2:i + 1].max(), ctx.L[i - 2:i + 1].min()
            for j in range(i + 1, i + 7):
                if (sgn < 0 and ctx.H[j] > wh) or (sgn > 0 and ctx.L[j] < wl):
                    break                                                # 무효화가 먼저
                if (sgn < 0 and ctx.C[j] < wl) or (sgn > 0 and ctx.C[j] > wh):
                    stop = wh if sgn < 0 else wl
                    out.append(rows(ctx, j, sgn, stop, ev, abs(ctx.C[j] - stop) / ctx.atr[j], "S3 흡수→단기 극값 이탈 진입"))
                    break
        # 공격: 큰 델타 + stack + 고점 돌파 + 다음 봉 유지
        s = "b" if sgn > 0 else "s"
        stk = ctx.f[f"stk_{s}"].to_numpy()
        prev = (pd.Series(ctx.H).shift(1).rolling(12).max() if sgn > 0 else pd.Series(ctx.L).shift(1).rolling(12).min()).to_numpy()
        brk = (ctx.C > prev) if sgn > 0 else (ctx.C < prev)
        hold = np.r_[(ctx.L[1:] >= prev[:-1]) if sgn > 0 else (ctx.H[1:] <= prev[:-1]), False]
        big = sgn * ctx.dn >= (th["dn_hi"] if sgn > 0 else -th["dn_lo"])
        small = np.abs(ctx.dn) <= th["dn_med_abs"]
        for i in dedup(np.flatnonzero(brk & hold)):
            if i < 300 or i > ctx.n - 40 or not np.isfinite(ctx.atr[i + 1]):
                continue
            ev = 1 if (big[i] and stk[i]) else (0 if (small[i] and not stk[i]) else -1)
            if ev < 0:
                continue
            stop = ctx.L[i] if sgn > 0 else ctx.H[i]
            out.append(rows(ctx, i + 1, sgn, stop, ev, abs(ctx.C[i + 1] - stop) / ctx.atr[i + 1], "S3 공격 돌파·유지→지속"))


# ---------------------------------------------------------------- S4 LVN/HVN · S5 AMT (레벨 vs 위약 레벨)
def s4(ctx, out, th):
    for pl in (False, True):
        ev = 0 if pl else 1
        src = ctx.lvp if pl else ctx.lv
        for i in range(300, ctx.n - 40):
            d = src.get(int(ctx.day[i]))
            if d is None or not np.isfinite(ctx.atr[i]) or not np.isfinite(ctx.dn[i]):
                continue
            if abs(ctx.dn[i]) >= th["dn_hi_abs"]:
                sgn = int(np.sign(ctx.dn[i]))
                for L in d["lvn"]:
                    crossed = (ctx.Op[i] < L < ctx.C[i]) if sgn > 0 else (ctx.Op[i] > L > ctx.C[i])
                    if not crossed:
                        continue
                    tg = d["hvn"][d["hvn"] > ctx.C[i]] if sgn > 0 else d["hvn"][d["hvn"] < ctx.C[i]]
                    if not len(tg):
                        continue
                    T = tg.min() if sgn > 0 else tg.max()
                    hit = ctx.H[i + 1:i + 25].max() >= T if sgn > 0 else ctx.L[i + 1:i + 25].min() <= T
                    r = rows(ctx, i, sgn, np.nan, ev, abs(T - ctx.C[i]) / ctx.atr[i], "S4 LVN 통과→다음 HVN 도달")
                    r["hit"] = float(hit); out.append(r)
                    break
            if ctx.dn[i] < th["dn_hi_abs"]:
                continue
            for Hn in d["hvn"]:
                if not (ctx.H[i] >= Hn - 0.3 * ctx.atr[i] and ctx.C[i - 1] < Hn):
                    continue
                j = i + 3
                if ctx.C[j] > Hn:
                    sgn = 1
                elif ctx.C[j] < Hn - 0.2 * ctx.atr[j]:
                    sgn = -1
                else:
                    break
                stop = ctx.L[i:j + 1].min() if sgn > 0 else ctx.H[i:j + 1].max()
                tag = "S4 HVN 위 유지→롱" if sgn > 0 else "S4 HVN 거부→숏"
                out.append(rows(ctx, j, sgn, stop, ev, abs(ctx.C[j] - stop) / ctx.atr[j], tag))
                break


def s5(ctx, out):
    for pl in (False, True):
        src = ctx.lvp if pl else ctx.lv
        for sgn, key in ((+1, "vah"), (-1, "val")):
            seen = set()
            for i in range(300, ctx.n - 40):
                dd = int(ctx.day[i])
                d = src.get(dd)
                if d is None or (dd, sgn) in seen or not np.isfinite(ctx.atr[i]):
                    continue
                L = d[key]
                if not ((ctx.C[i] > L >= ctx.C[i - 1]) if sgn > 0 else (ctx.C[i] < L <= ctx.C[i - 1])):
                    continue
                seen.add((dd, sgn))
                s = i + 6
                beyond = sgn * (ctx.C[i + 1:s + 1] - L) > 0
                va = ctx.vol_beyond_share(i + 1, s, L)
                share = va if sgn > 0 else 1 - va
                accept = beyond.sum() >= 4 and share >= 0.5
                still = sgn * (ctx.C[s] - L) > 0
                if not pl and still:
                    r = rows(ctx, s, sgn, np.nan, int(accept), sgn * (ctx.C[s] - L) / ctx.atr[s], "S5a 수용→2h 뒤에도 바깥")
                    r["above2h"] = float(sgn * (ctx.C[s + 24] - L) > 0); out.append(r)
                if not still:
                    P = d["poc"]
                    hit = (ctx.L[s + 1:s + 25].min() <= P) if sgn > 0 else (ctx.H[s + 1:s + 25].max() >= P)
                    r = rows(ctx, s, -sgn, np.nan, 0 if pl else 1, abs(ctx.C[s] - P) / ctx.atr[s], "S5b 거부→POC 도달")
                    r["hit"] = float(hit); out.append(r)


# ---------------------------------------------------------------- 실행
def build_levels(cells: pd.DataFrame, bars: pd.DataFrame):
    g = cells.assign(day=cells.b // 86400, v=cells.buy + cells.sell).groupby(["day", "k"], sort=True).v.sum().reset_index()
    prof = {int(d): day_profile(x.k.to_numpy() * BUCKET, x.v.to_numpy()) for d, x in g.groupby("day")}
    op = bars.open.groupby(bars.index // 86400).first()
    lv, lvp = {}, {}
    for d in prof:
        if d - 1 in prof:
            lv[d] = prof[d - 1]
        if d - 8 in prof and d in op.index and d - 7 in op.index and np.isfinite(op[d]) and np.isfinite(op[d - 7]):
            k = op[d] / op[d - 7]
            lvp[d] = {key: val * k for key, val in prof[d - 8].items()}
    return lv, lvp


def report(df: pd.DataFrame) -> dict:
    to_day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
    res = {}
    for tag, g in df.groupby("tag", sort=False):
        res[tag] = {}
        metrics = [m for m in ("hit", "no_ext", "above2h", "race1", "r6", "r12", "r24") if g[m].notna().any()]
        qv = g.q.to_numpy().astype(float)
        edges = np.nanpercentile(qv, np.arange(10, 100, 10))
        q = np.searchsorted(edges, np.nan_to_num(qv))
        for pname, (a, b) in PERIODS.items():
            sel = ((g.day >= to_day(a)) & (g.day < to_day(b))).to_numpy()
            res[tag][pname] = {m: summarize(g[m].to_numpy().astype(float), g.ev.to_numpy(), q, g.day.to_numpy(), sel)
                               for m in metrics}
            m1 = sel & (g.ev == 1).to_numpy()
            res[tag][pname]["사건 순 r12(−8bp)"] = round(float(np.nanmean(g.r12.to_numpy()[m1]) - COST), 2) if m1.sum() else None
    return res


def run(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, t1 = int(pd.Timestamp("2023-01-01").timestamp()), int(pd.Timestamp("2026-10-07").timestamp())
    bars = load_bars(t0, t1)
    fs = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / CELLS / "*.parquet")))
    cells = pd.concat([pd.read_parquet(f) for f in fs]).sort_values(["b", "k"]).reset_index(drop=True)
    print("bars", bars.shape, "cells", cells.shape, flush=True)
    s2c_rep = s2c(cells)
    print(json.dumps(s2c_rep, ensure_ascii=False, indent=1), flush=True)
    bars = bars[bars.index <= cells.b.max()]
    flags = bar_flags(cells, bars)
    lv, lvp = build_levels(cells, bars)
    ctx = Ctx(bars, flags, lv, lvp, cells)
    aux = bars.index < pd.Timestamp("2025-01-01").timestamp()
    dn = bars.dn.to_numpy()[aux]; dn = dn[np.isfinite(dn)]
    d3 = bars.dn.rolling(3).sum().to_numpy()[aux]; d3 = d3[np.isfinite(d3)]
    d6 = bars.dn.rolling(6).sum().to_numpy()[aux]; d6 = d6[np.isfinite(d6)]
    th = {"dn_hi": np.percentile(dn, 80), "dn_lo": np.percentile(dn, 20), "dn_med_abs": np.percentile(np.abs(dn), 50),
          "dn_hi_abs": np.percentile(np.abs(dn), 80), "d3_hi": np.percentile(d3, 80), "d3_lo": np.percentile(d3, 20),
          "d3_med": np.percentile(np.abs(d3), 50), "d6_hi": np.percentile(d6, 80), "d6_lo": np.percentile(d6, 20)}
    print("문턱(보조기간에서 고정)", {k: round(float(v), 3) for k, v in th.items()}, flush=True)
    out = []
    for name, fn in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5)):
        fn(ctx, out) if name in ("S1", "S5") else fn(ctx, out, th)
        print(name, "done", len(out), flush=True)
    df = pd.DataFrame(out)
    for c in ("hit", "no_ext", "above2h"):
        if c not in df:
            df[c] = np.nan
    df.to_parquet(out_dir / "events.parquet")
    res = report(df)
    res["S2c 델타×OI 귀속(실측 청산)"] = s2c_rep
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", out_dir / "results.json")


def exp_c(out_dir: Path):
    """실험 C(10-12): 4h 신고/저가 사건에서 가격만 쓴 기준 모델 대비 CVD·델타 증분.
    목적 = 2h 안 +0.5ATR 연장 실패(no_ext), 보조 = 2h 되돌림 부호(rev). 걷기 전진: ≤2024 학습→2025, ≤2025 학습→2026."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    o = load_bars(int(pd.Timestamp("2023-01-01").timestamp()), int(pd.Timestamp("2026-10-07").timestamp()))
    H, L, C, A, V = (o[k].to_numpy() for k in ("high", "low", "close", "atr", "v"))
    dn = o.dn.to_numpy(); delta = np.nan_to_num(o.delta.to_numpy())
    cvd = np.cumsum(delta); md = o.delta.abs().rolling(288, min_periods=200).mean().shift(1).to_numpy()
    S = lambda x: pd.Series(x)
    hh, ll = S(H).shift(1).rolling(48).max().to_numpy(), S(L).shift(1).rolling(48).min().to_numpy()
    ch, cl = S(cvd).shift(1).rolling(48).max().to_numpy(), S(cvd).shift(1).rolling(48).min().to_numpy()
    rhi, rlo = S(H).shift(12).rolling(276).max().to_numpy(), S(L).shift(12).rolling(276).min().to_numpy()
    vr = V / S(V).rolling(288).mean().shift(1).to_numpy()
    atr_r = A / S(A).rolling(288).mean().shift(1).to_numpy()
    d6 = S(dn).rolling(6).sum().to_numpy(); d48 = S(dn).rolling(48).sum().to_numpy()
    rows_ = []
    for sgn in (+1, -1):
        cand = np.flatnonzero((H > hh) if sgn > 0 else (L < ll))
        for i in dedup(cand[(cand > 600) & (cand < len(C) - 30)]):
            if not (np.isfinite(A[i]) and np.isfinite(md[i]) and np.all(np.isfinite(C[i:i + 25]))):
                continue
            ext = (H[i + 1:i + 25].max() >= H[i] + 0.5 * A[i]) if sgn > 0 else (L[i + 1:i + 25].min() <= L[i] - 0.5 * A[i])
            rows_.append({
                "day": int(o.index[i] // 86400), "side": sgn,
                "brk": ((H[i] - hh[i]) if sgn > 0 else (ll[i] - L[i])) / A[i],
                "r1h": sgn * np.log(C[i] / C[i - 12]) / (A[i] / C[i]), "r4h": sgn * np.log(C[i] / C[i - 48]) / (A[i] / C[i]),
                "r24h": sgn * np.log(C[i] / C[i - 288]) / (A[i] / C[i]),
                "d24": ((H[i] - rhi[i]) if sgn > 0 else (rlo[i] - L[i])) / A[i],
                "atrp": A[i] / C[i], "atr_r": atr_r[i], "vr": vr[i], "hour": (o.index[i] // 3600) % 24,
                "clpos": sgn * (C[i] - (H[i] + L[i]) / 2) / max(H[i] - L[i], 1e-9),
                # 오더플로(방향은 사건 쪽 = +)
                "div": float(cvd[i] < ch[i]) if sgn > 0 else float(cvd[i] > cl[i]),
                "cvd_gap": sgn * ((cvd[i] - ch[i]) if sgn > 0 else (cvd[i] - cl[i])) / md[i],
                "dn0": sgn * dn[i], "d6": sgn * d6[i], "d48": sgn * d48[i],
                "no_ext": float(not ext), "rev": float(sgn * (C[i + 24] - C[i]) < 0)})
    df = pd.DataFrame(rows_).replace([np.inf, -np.inf], np.nan).dropna()
    base = ["side", "brk", "r1h", "r4h", "r24h", "d24", "atrp", "atr_r", "vr", "hour", "clpos"]
    sets = {"+CVD 다이버": base + ["div", "cvd_gap"], "+델타": base + ["dn0", "d6", "d48"],
            "+CVD+델타": base + ["div", "cvd_gap", "dn0", "d6", "d48"]}
    folds = {"2025": ("2025-01-01", "2026-01-01"), "2026": ("2026-01-01", "2100-01-01")}
    tod = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
    res = {"n": len(df), "기저율": {y: round(float(df[y].mean()), 3) for y in ("no_ext", "rev")}}
    rng = np.random.default_rng(7)
    models = {"HGB": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                            min_samples_leaf=100, random_state=0),
              "로지스틱": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    for y in ("no_ext", "rev"):
        for mname, mk in models.items():
            for fname, (a, b) in folds.items():
                tr, te = df[df.day < tod(a)], df[(df.day >= tod(a)) & (df.day < tod(b))]
                p0 = mk().fit(tr[base], tr[y]).predict_proba(te[base])[:, 1]
                days, di = np.unique(te.day, return_inverse=True)
                W = rng.poisson(1.0, (300, len(days)))
                yt = te[y].to_numpy()
                key = f"{y}·{mname}·{fname}"
                res[key] = {"n_test": len(te), "기준 AUC": round(roc_auc_score(yt, p0), 4)}
                for sname, cols in sets.items():
                    p1 = mk().fit(tr[cols], tr[y]).predict_proba(te[cols])[:, 1]
                    bs = [roc_auc_score(yt, p1, sample_weight=w[di]) - roc_auc_score(yt, p0, sample_weight=w[di])
                          for w in W if w.sum() > 0]
                    d = roc_auc_score(yt, p1) - roc_auc_score(yt, p0)
                    res[key][sname] = [round(d, 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]]
    (out_dir / "exp_c.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def selftest():
    assert race(np.array([100, 101.2]), np.array([100, 99.5]), 0, +1, 100.0, 99.0, n=1) == 1.0
    assert race(np.array([100, 101.2]), np.array([100, 98.9]), 0, +1, 100.0, 99.0, n=1) == 0.0   # 같은 봉 둘 다 = 손절
    assert race(np.array([100, 100.5]), np.array([100, 98.0]), 0, -1, 100.0, 101.0, n=1) == 1.0
    rng = np.random.default_rng(0)
    p = np.r_[rng.normal(2000, 1.5, 500), rng.normal(2010, 1.0, 20), rng.normal(2020, 1.5, 800)]
    pr = day_profile(p, np.ones_like(p))
    assert 2017 < pr["poc"] < 2023 and pr["val"] < pr["poc"] < pr["vah"], pr
    assert len(pr["hvn"]) >= 2 and any(2003 < x < 2017 for x in pr["lvn"]), pr
    assert dedup(np.array([0, 5, 12, 13, 30])).tolist() == [0, 12, 30]
    v = np.r_[np.ones(50), np.zeros(50)]; e = np.r_[np.ones(50), np.zeros(50)].astype(int)
    ex, lo, hi = strat_ci(v, e, np.zeros(100, int), np.arange(100) % 10)
    assert abs(ex - 1) < 1e-9
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tmp/orderflow_report5_20261011"))
    ap.add_argument("--exp-c", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.exp_c:
        a.out.mkdir(parents=True, exist_ok=True); exp_c(a.out)
    else:
        run(a.out)
