"""다이버전스·흡수 표식 이후 움직임 -- 사전등록 docs/prereg/divergence_absorption_after_20261102.md (정의·판정 고정, 결과 보고 안 바꿈)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path("/home/kbj20/crypto-scalping/data")
a = pd.read_csv(D / "eth_5m_1year.csv", usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
a["t"] = pd.to_datetime(a.timestamp)
g = pd.read_csv(D / "eth_5m_2026_gap.csv", usecols=["ts", "open", "high", "low", "close", "volume", "taker_buy_base"])
g["t"] = pd.to_datetime(g.ts.astype("int64"), unit="ms")
df = pd.concat([a.drop(columns="timestamp"), g.drop(columns="ts")]).drop_duplicates("t").sort_values("t").set_index("t")
assert df.index.is_monotonic_increasing
gap = df.index.to_series().diff().dropna()
print("bars", len(df), df.index[0], "~", df.index[-1], "| gaps>5m:", int((gap > pd.Timedelta("5min")).sum()), "max gap", gap.max())
df["delta"] = 2 * df.taker_buy_base - df.volume
W = 4032   # 14일
o, h, l, c, d = df.open.values, df.high.values, df.low.values, df.close.values, df.delta
def roll(n):
    return (d.rolling(n).sum().values, c - pd.Series(o, index=df.index).shift(n - 1).values,
            (pd.Series(h, index=df.index).rolling(n).max() - pd.Series(l, index=df.index).rolling(n).min()).values)
D6, dp6, R6 = roll(6); D3, dp3, _ = roll(3)
sig6 = pd.Series(D6, index=df.index).rolling(W).std().shift(1).values
sig3 = pd.Series(D3, index=df.index).rolling(W).std().shift(1).values
xy = pd.Series(D3 * dp3, index=df.index).rolling(W).sum().shift(1); xx = pd.Series(D3 ** 2, index=df.index).rolling(W).sum().shift(1)
beta = (xy / xx).values
with np.errstate(invalid="ignore"):
    dv = (np.sign(D6) * np.sign(dp6) < 0) & (np.abs(D6) >= sig6) & (np.abs(dp6) >= 0.25 * R6)
    big3 = np.abs(D3) >= sig3
    ab = big3 & (dp3 * np.sign(D3) < 0.25 * beta * np.abs(D3))
    agree6 = (np.sign(D6) * np.sign(dp6) > 0) & ~np.isnan(sig6)
    ab_pool = big3 & ~ab
def dedupe(mask, gap_bars=6):
    out, last = [], -10**9
    for i in np.flatnonzero(mask):
        if i - last >= gap_bars: out.append(i); last = i
    return np.array(out, dtype=int)
n = len(df); idx = df.index
HS = {1: "5분", 3: "15분", 6: "30분", 12: "60분"}
def fwd(i, hh):   # P0 = 다음 봉 시가, P_h = i+hh 번째 봉 종가
    ok = i + hh < n
    r = np.full(len(i), np.nan); j = i[ok]
    r[ok] = (c[j + hh] / o[j + 1] - 1) * 1e4
    return r
def splits(i):
    t = idx[i]
    return {"학습": (t < "2026-01-01"), "검증": (t >= "2026-01-01") & (t < "2026-06-01"), "시험": (t >= "2026-06-01")}
def cells(x, edges): return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
rng = np.random.default_rng(20261102)
def boot(diff, day):   # 일 블록 부트스트랩 단측
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=diff); k = np.bincount(inv)
    m = np.array([s[b].sum() / k[b].sum() for b in (rng.integers(0, len(u), len(u)) for _ in range(2000))])
    return np.percentile(m, 10), float((m <= 0).mean())
res = {}
def kind(name, ev, sgn, pool_mask, mats):
    pool = np.flatnonzero(pool_mask & ~np.isnan(sig6) & ~np.isnan(beta))
    ps = splits(pool); es = splits(ev)
    for hh in HS:
        pv = sgn[pool] * fwd(pool, hh); evv = sgn[ev] * fwd(ev, hh)
        for sp in ("학습", "검증", "시험"):
            pm, em = ps[sp] & ~np.isnan(pv), es[sp] & ~np.isnan(evv)
            if em.sum() == 0: res[(name, hh, sp)] = dict(n=0); continue
            # 셀 = 풀 분위(구간 안), 풀 평균
            mats_p = [m[pool][pm] for m in mats]; mats_e = [m[ev][em] for m in mats]
            cp = np.zeros(pm.sum(), int); ce = np.zeros(em.sum(), int)
            for (x_p, x_e, nb) in [(mats_p[0], mats_e[0], 10)] + ([(mats_p[1], mats_e[1], 3)] if len(mats) > 1 else []):
                edges = np.quantile(x_p, np.linspace(0, 1, nb + 1)); cp = cp * nb + cells(x_p, edges); ce = ce * nb + cells(x_e, edges)
            cm = pd.Series(pv[pm]).groupby(cp).mean()
            base = np.array([cm.get(k, np.nan) for k in ce]); ok = ~np.isnan(base)
            diff = evv[em][ok] - base[ok]; day = np.asarray(idx[ev][em][ok].normalize().astype("int64"))
            lo10, p = boot(diff, day) if ok.sum() >= 2 else (np.nan, np.nan)
            res[(name, hh, sp)] = dict(n=int(ok.sum()), pool=int(pm.sum()), raw=float(evv[em].mean()), ctrl=float(np.mean(base[ok])), diff=float(diff.mean()),
                                       lo10=float(lo10), p=float(p), days=int(len(np.unique(day))), up=float((evv[em] > 0).mean()))
dv_i = dedupe(dv); ab_i = dedupe(ab)
sg_dv = np.sign(dp6); sg_ab = -np.sign(D3)
kind("DV", dv_i, sg_dv, agree6, [np.abs(dp6), np.abs(D6)])
kind("AB", ab_i, sg_ab, ab_pool, [np.abs(D3)])
print("\n사건 수(중복 제거) DV", len(dv_i), "AB", len(ab_i), "| 분할:", {k: int(v.sum()) for k, v in splits(dv_i).items()}, {k: int(v.sum()) for k, v in splits(ab_i).items()})
rows = []
for (nm, hh, sp), r in res.items():
    if r.get("n", 0) == 0: continue
    rows.append(dict(kind=nm, h=HS[hh], split=sp, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
T = pd.DataFrame(rows); pd.set_option("display.width", 200); print(T.to_string(index=False))
# Holm: 주 판정 4개(시험, 15·60분)
prim = [(k, hh) for k in ("DV", "AB") for hh in (3, 12)]
ps = sorted([(res[(k, hh, "시험")].get("p", 1.0), k, hh) for k, hh in prim])
holm = {}; m = len(ps); run = 0
for rank, (p, k, hh) in enumerate(ps):
    run = max(run, min(1.0, p * (m - rank))); holm[(k, hh)] = run
print("\n주 판정(Holm 단측 시험):")
for k, hh in prim:
    v, t = res[(k, hh, "검증")], res[(k, hh, "시험")]
    ok = v.get("n", 0) >= 30 and t.get("n", 0) >= 30
    passed = ok and v["diff"] > 0 and t["diff"] > 0 and t["lo10"] > 0
    print(f"  {k} {HS[hh]}: 검증 n={v.get('n')} 차이 {v.get('diff', float('nan')):+.2f}bp · 시험 n={t.get('n')} 차이 {t.get('diff', float('nan')):+.2f}bp 하한10% {t.get('lo10', float('nan')):+.2f} p={t.get('p', float('nan')):.3f} Holm {holm[(k, hh)]:.3f} -> {'표본 부족' if not ok else ('통과' if passed else '불통과')}")
T.to_json(Path(__file__).with_name("divergence_absorption_after_20261102.json"), orient="records", force_ascii=False)
