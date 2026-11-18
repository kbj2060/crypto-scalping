#!/usr/bin/env python3
"""ETH 지지·저항(1h·4h·24h 고저의 접근·돌파) × 체결 강도 — 머신 러닝 재시도 + 강화학습 가능성 (2026-11-18).

  python scripts/research_sr_ml_rl_20261118.py selftest
  python scripts/research_sr_ml_rl_20261118.py build train      # 2025-12-31 이전 사건만(탐색·적합)
  python scripts/research_sr_ml_rl_20261118.py build test       # 2026 사건(사전등록 커밋 뒤에만)
  python scripts/research_sr_ml_rl_20261118.py explore          # 학습 구간 정화 6토막 교차검증: 부스팅·트리 앙상블·로지스틱
  python scripts/research_sr_ml_rl_20261118.py rl               # 강화학습 가능성: 진입 밴딧(1단계) · 청산 시점 학습(적합 Q 반복 = 뒤로 풀기)

시각 규약: 사건 초 a = 그 분에서 문턱을 처음 넘은 초(레벨·σ 는 그 분 이전 분봉만). 결정 d = a + 5초. 피쳐 = 초 ≤ d · 분 < 사건 분.
기준가 P = d 까지 마지막 체결가. 라벨·보상 = d+1 초부터. m_ = 부호 없는 크기 재료 · s_ = 사건 쪽 부호 재료.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_sr_intensity_20261118 as R  # noqa: E402

OUT = R.OUT
W, PRE, FWD, PAD = 5, 3600, 3600, 10
WINS, LOOKS, D_AP, GAP = (1, 3, 10, 30, 60, 300, 900, 3600), (60, 240, 1440), 0.1, 60
STEP, NSTEP = 30, 30                                                    # 청산 학습: 30초 간격 30걸음(15분)
EPS = 1e-9


def candidates(m: pd.DataFrame, t0: str, t1: str) -> pd.DataFrame:
    """분 단위 후보. kind brk = 고가(저가)가 직전 look 분 고점(저점) L 을 넘음 · apr = L 의 0.1σ 안으로 들어옴. 각 (look, kind, side)에서 직전 60분에 같은 사건 없음."""
    lc = np.log(m.c); s1h = (lc.diff().rolling(1440, min_periods=1200).std() * np.sqrt(60)).shift(1)
    first = lambda b: b & ~(b.astype(float).rolling(GAP).max().shift(1).fillna(0).astype(bool)) & s1h.notna()
    a0, a1 = pd.Timestamp(t0).value // 10**9 // 60, pd.Timestamp(t1).value // 10**9 // 60; out = []
    for look in LOOKS:
        hh, ll = m.h.rolling(look, min_periods=int(look * .98)).max().shift(1), m.l.rolling(look, min_periods=int(look * .98)).min().shift(1)
        up_ap, dn_ap = hh * np.exp(-D_AP * s1h), ll * np.exp(D_AP * s1h)
        for kind, side, b, th, L in (("brk", 1, m.h > hh, hh, hh), ("brk", -1, m.l < ll, ll, ll), ("apr", 1, m.h >= up_ap, up_ap, hh), ("apr", -1, m.l <= dn_ap, dn_ap, ll)):
            f = first(b); ix = m.index[f & (m.index >= a0) & (m.index < a1)]
            out.append(pd.DataFrame({"mi": ix, "side": side, "kind": kind, "look": look, "th": th[ix].to_numpy(), "L": L[ix].to_numpy(), "s1h": s1h[ix].to_numpy()}))
    return pd.concat(out).sort_values(["mi", "look", "kind", "side"]).reset_index(drop=True)


def one(T, M_: dict, mi: int, side: int, kind: str, look: int, th: float, L: float, s1h: float):
    X = T.window(mi * 60 - PRE - PAD, mi * 60 + 60 + W + FWD + 2)
    if X is None or mi - M_["base"] - 1441 < 0:                           # 분봉 맥락이 1,441분 앞까지 본다(음수 색인은 끝에서부터 = 미래)
        return None
    o = PRE + PAD; hi_, lo_ = X["ph"][o:o + 60], X["pl"][o:o + 60]
    seg = (hi_ > th if kind == "brk" else hi_ >= th) if side > 0 else (lo_ < th if kind == "brk" else lo_ <= th)
    if not seg.any():
        return None
    a = o + int(seg.argmax()); d = a + W; assert d - PRE >= 0
    pc, ph, pl = X["pc"], X["ph"], X["pl"]; lp = np.log(pc); P = pc[d]; vol = X["bq"] + X["sq"]; sv = side * (X["bq"] - X["sq"]); s1s = s1h / 60
    cs = lambda x: np.concatenate([[0.0], np.cumsum(x[d - PRE + 1:d + 1])]); tail = lambda c, w: c[-1] - c[-1 - w]
    d1 = np.diff(lp[d - PRE:d + 1]); cv, csv, cn, cq, cr2, cab = cs(vol), cs(sv), cs(X["nt"]), cs(X["q2"]), np.concatenate([[0.0], np.cumsum(d1 ** 2)]), np.concatenate([[0.0], np.cumsum(np.abs(d1))])
    bv, bn = tail(cv, PRE) / PRE + EPS, tail(cn, PRE) / PRE + EPS
    f = dict(asec=int(mi * 60 - PRE - PAD + a), side=side, kind=kind, look=look, L=L, s1h=s1h)
    for w in WINS:
        v, n, net = tail(cv, w), tail(cn, w), lp[d] - lp[d - w]
        if w < PRE:
            f[f"m_vol{w}"], f[f"m_n{w}"], f[f"m_sz{w}"] = np.log((v / w + EPS) / bv), np.log((n / w + EPS) / bn), np.log((v + EPS) / (n + EPS)) - np.log(bv / bn)
        f[f"s_imb{w}"], f[f"s_ret{w}"] = tail(csv, w) / (v + EPS), side * net / (s1s * np.sqrt(w))
        f[f"m_rv{w}"], f[f"m_conc{w}"] = np.log(np.sqrt(tail(cr2, w)) / (s1s * np.sqrt(w)) + 1e-3), np.log(tail(cq, w) / (v * v + EPS) + 1e-6)
        if w >= 10:
            hi, lo = ph[d - w + 1:d + 1].max(), pl[d - w + 1:d + 1].min()
            f[f"s_rpos{w}"] = ((P - lo) if side > 0 else (hi - P)) / (hi - lo + EPS); f[f"s_eff{w}"] = side * net / (tail(cab, w) + EPS)
    ob = slice(a, d + 1)
    f.update(s_imbo=sv[ob].sum() / (vol[ob].sum() + EPS), m_volo=np.log((vol[ob].sum() / (W + 1) + EPS) / bv), s_pos=side * np.log(P / L) / s1h,
             s_ext=side * np.log((ph[ob].max() if side > 0 else pl[ob].min()) / L) / s1h, m_asec=(a - o) / 60.0)
    H, Lo, C, V, TB = M_["H"], M_["L"], M_["C"], M_["V"], M_["TB"]; i = mi - M_["base"]; wd = slice(i - look, i)
    f.update(m_lage=np.log(look - int(np.nanargmax(H[wd] if side > 0 else -Lo[wd]))) / np.log(look), m_near=float((np.abs(np.log((H[wd] if side > 0 else Lo[wd]) / L)) < 0.1 * s1h).sum()) / look,
             m_rng=np.log(np.nanmax(H[wd]) / np.nanmin(Lo[wd])) / (s1h * np.sqrt(look / 60)), s_side=float(side), m_kind=float(kind == "brk"), m_look=np.log(look), m_ls=np.log(s1h))
    for h in (15, 60, 240, 1440):
        f[f"s_mom{h}"] = side * np.log(C[i - 1] / C[i - 1 - h]) / (s1h * np.sqrt(h / 60))
    for h in (60, 240):
        f[f"s_tb{h}"] = side * (2 * np.nansum(TB[i - h:i]) / max(np.nansum(V[i - h:i]), 1e-9) - 1)
    phf, plf = ph[d + 1:d + 1 + FWD], pl[d + 1:d + 1 + FWD]
    for nm, T_, D in (("y_fp10_300", 300, .10), ("y_fp15_900", 900, .15), ("y_fp25_3600", 3600, .25)):   # 대칭 배리어 먼저 · 못 닿으면 시한 수익의 부호 → 전체 사건에서 정해진다
        fp = R.first_pass(phf[:T_], plf[:T_], P, D * s1h); z = (fp if fp in (1.0, -1.0) else np.sign(np.log(pc[d + T_] / P))) * side
        f[nm] = 1.0 if z > 0 else 0.0 if z < 0 else np.nan
    for h in (60, 300, 900, 3600):
        f[f"r_{h}"] = side * np.log(pc[d + h] / P) * 1e4                 # 사건 쪽 수익(bp)
    for k in range(1, NSTEP + 1):                                         # 청산 학습용 경로(결정 뒤 — 피쳐로 쓰지 않는다)
        t = d + k * STEP; sl = slice(t - STEP + 1, t + 1)
        f[f"q_ret{k}"], f[f"q_imb{k}"], f[f"q_vol{k}"] = side * np.log(pc[t] / P) * 1e4, sv[sl].sum() / (vol[sl].sum() + EPS), np.log((vol[sl].sum() / STEP + EPS) / bv)
    return f


def build(m: pd.DataFrame, t0: str, t1: str, tape=None) -> pd.DataFrame:
    c = candidates(m, t0, t1); T = tape or R.Tape(); M_ = R.marr(m); rows = []
    for r in c.itertuples():
        f = one(T, M_, int(r.mi), int(r.side), r.kind, int(r.look), float(r.th), float(r.L), float(r.s1h))
        if f is not None:
            rows.append(f)
    d = pd.DataFrame(rows).sort_values("asec").reset_index(drop=True); d["ts"] = pd.to_datetime(d.asec, unit="s")
    hod = (d.asec % 86400) / 86400; d["m_hsin"], d["m_hcos"], d["m_wkend"] = np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod), (d.ts.dt.dayofweek >= 5).astype(float)
    a = d.asec.to_numpy(); d["m_prev60"] = (np.arange(len(a)) - np.searchsorted(a, a - 3600)).astype(float)       # 직전 60분의 다른 사건 수(자기보다 앞선 것만)
    return d


def cols(d: pd.DataFrame):
    return [c for c in d if c.startswith("m_")], [c for c in d if c.startswith("s_")]


def folds(ts: pd.Series, n: int = 6):
    e = np.linspace(0, len(ts), n + 1).astype(int); t = ts.to_numpy()
    for j in range(n):
        k = np.arange(e[j], e[j + 1]); yield np.flatnonzero((t < t[k[0]] - np.timedelta64(5, "h")) | (t > t[k[-1]] + np.timedelta64(5, "h"))), k


def hgb(reg: bool = False, seed: int = 0, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    p = dict(max_depth=4, max_iter=300, learning_rate=.03, min_samples_leaf=200, l2_regularization=10.0, max_bins=64, random_state=seed); p.update(kw)
    return (HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier)(**p)


def et():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=100, max_features=.3, n_jobs=8, random_state=0))


def oof(d, X, y, mk, reg=False):
    ok = np.isfinite(y); p = np.full(len(y), np.nan)
    for t, k in folds(d.ts):
        t = t[ok[t]]; m_ = mk().fit(d[X].iloc[t], y[t]); p[k] = m_.predict(d[X].iloc[k]) if reg else m_.predict_proba(d[X].iloc[k])[:, 1]
    return p


def fauc(d, y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(y) & np.isfinite(s); a = [roc_auc_score(y[k[ok[k]]], s[k[ok[k]]]) for _, k in folds(d.ts)]
    return float(np.mean(a)), float(np.min(a))


def explore() -> None:
    import warnings; warnings.filterwarnings("ignore")
    d = pd.read_parquet(OUT / "ml_train.parquet"); MAG, SGN = cols(d); ALL = MAG + SGN; res = {}
    print(f"학습 사건 {len(d)} · {d.ts.min().date()}~{d.ts.max().date()} · 종류×거리 {d.groupby(['kind', 'look']).size().to_dict()} · 재료 크기 {len(MAG)} + 부호 {len(SGN)}")
    for yc in ("y_fp10_300", "y_fp15_900", "y_fp25_3600"):
        y = d[yc].to_numpy(); P = {}
        for nm, X, mk in (("로지스틱 크기", MAG, R._lr), ("로지스틱 전체", ALL, R._lr), ("부스팅 크기", MAG, hgb), ("부스팅 부호", SGN, hgb), ("부스팅 전체", ALL, hgb),
                          ("부스팅 깊게", ALL, lambda: hgb(max_depth=6, max_iter=500, learning_rate=.02)), ("트리 앙상블 전체", ALL, et)):
            P[nm] = oof(d, X, y, mk)
        P["평균(부스팅 전체+트리+로지스틱)"] = np.mean([pd.Series(P[k]).rank(pct=True).to_numpy() for k in ("부스팅 전체", "트리 앙상블 전체", "로지스틱 전체")], axis=0)
        print(f"\n== {yc} · 사건 쪽 먼저 비율 {np.nanmean(y):.4f} (n {int(np.isfinite(y).sum())})"); res[yc] = P
        for nm, p in P.items():
            a, mn = fauc(d, y, p); print(f"  {nm:28s} 교차 AUC {a:.4f} (토막 최소 {mn:.4f})")
        p = P["부스팅 전체"]; q = pd.qcut(p, 10, labels=False); rc = "r_" + yc.split("_")[-1]
        print("  부스팅 전체 10분위(낮음→높음) 사건 쪽 먼저 비율: " + " ".join(f"{np.nanmean(y[q == i]):.3f}" for i in range(10)))
        print(f"  같은 10분위 평균 사건 쪽 수익({rc}, bp): " + " ".join(f"{d[rc].to_numpy()[q == i].mean():+.1f}" for i in range(10)))
        for kd in ("brk", "apr"):
            for lk in LOOKS:
                g = ((d.kind == kd) & (d.look == lk)).to_numpy(); gd = d[g].reset_index(drop=True)
                print(f"    {kd} {lk:4d}분 n {g.sum():5d} 기저 {np.nanmean(y[g]):.3f} 부스팅 전체 AUC {fauc(gd, y[g], p[g])[0]:.3f} · 크기만 {fauc(gd, y[g], P['부스팅 크기'][g])[0]:.3f}")
    pd.DataFrame({f"{yc}|{k}": v for yc, P in res.items() for k, v in P.items()}).to_parquet(OUT / "ml_train_oof.parquet")


def _boot(v, day, rng, nb=1000):
    days = np.unique(day); s = pd.Series(v).groupby(day).agg(["sum", "count"]); S_, C_ = s["sum"].to_numpy(), s["count"].to_numpy(); ix = rng.integers(0, len(days), (nb, len(days)))
    b = S_[ix].sum(1) / np.maximum(C_[ix].sum(1), 1); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def rl() -> None:
    """강화학습 가능성(학습 구간 교차검증 · 딥러닝 아님): ① 진입 = 1단계 Q(문맥 밴딧) ② 청산 = 유한 지평 적합 Q 반복(마지막 걸음부터 뒤로)."""
    import warnings; warnings.filterwarnings("ignore")
    d = pd.read_parquet(OUT / "ml_train.parquet"); MAG, SGN = cols(d); ALL = MAG + SGN; rng = np.random.default_rng(R.SEED); day = d.ts.dt.floor("D").to_numpy()
    print(f"사건 {len(d)} · 비용: 시장가 왕복 10bp(엄격) · 0bp(통과 후보 등급)")
    print("\n① 진입 밴딧 — Q(s, 사건 쪽) = 사건 쪽 수익 기대(부스팅 회귀, 교차 예측). 정책 = Q 부호 쪽으로 진입, |Q| 상위만")
    for h in (300, 900, 3600):
        r = d[f"r_{h}"].to_numpy(); q = oof(d, ALL, np.clip(r, *np.percentile(r, [1, 99])), lambda: hgb(reg=True), reg=True); pnl = np.sign(q) * r
        ic = float(pd.Series(q).corr(pd.Series(r), method="spearman")); rnd = float(np.mean([np.mean(rng.choice([-1, 1], len(r)) * r) for _ in range(200)]))
        print(f"  지평 {h:4d}초 | 예측↔실제 순위상관 {ic:+.4f} | 늘 사건 쪽 {r.mean():+.2f}bp · 늘 반대 {-r.mean():+.2f} · 무작위 {rnd:+.2f} | 정책(전부) {pnl.mean():+.2f}bp [95% {_boot(pnl, day, rng)[0]:+.2f}, {_boot(pnl, day, rng)[1]:+.2f}] | 한 건 수익 표준편차 {r.std():.0f}bp")
        for top in (.2, .05):
            g = np.abs(q) >= np.quantile(np.abs(q), 1 - top); lo, hi = _boot(pnl[g], day[g], rng)
            print(f"      |Q| 상위 {int(top * 100):2d}% n {g.sum():5d} 비용 전 {pnl[g].mean():+.2f}bp [{lo:+.2f}, {hi:+.2f}] · 비용 10bp 뒤 {pnl[g].mean() - 10:+.2f} · 예측한 |Q| 중앙 {np.median(np.abs(q[g])):.2f}bp · 승률 {np.mean(pnl[g] > 0):.3f}")
    print("\n② 청산 시점 — 상태 = (지금 수익·최고·최저·직전 30초 매수 비중·거래량·걸음), 행동 = 보유/청산, 30초 × 30걸음. 뒤에서부터 걸음마다 «계속 보유 가치»를 회귀(적합 Q 반복)")
    Q = np.stack([d[f"q_ret{k}"].to_numpy() for k in range(1, NSTEP + 1)], 1); I_ = np.stack([d[f"q_imb{k}"].to_numpy() for k in range(1, NSTEP + 1)], 1); V_ = np.stack([d[f"q_vol{k}"].to_numpy() for k in range(1, NSTEP + 1)], 1)
    ctx = d[["m_ls", "m_kind", "m_look", "s_pos", "m_vol60"]].to_numpy()
    for nm, sg in (("사건 쪽 진입", 1.0), ("반대쪽 진입", -1.0)):
        G = sg * Q; mx, mn = np.maximum.accumulate(G, 1), np.minimum.accumulate(G, 1); val = np.full(len(d), np.nan); stop_k = np.full(len(d), NSTEP)
        st = lambda k, ix: np.column_stack([G[ix, k], mx[ix, k], mn[ix, k], sg * I_[ix, k], V_[ix, k], ctx[ix]])
        for t, te in folds(d.ts):
            Vn = G[t, NSTEP - 1].copy(); models = {}
            for k in range(NSTEP - 2, -1, -1):                                 # 걸음 k(0 = 첫 30초)에서: 계속 보유 가치 ≈ E[다음 걸음부터의 정책 가치 | 상태]
                m_ = hgb(reg=True, max_iter=60, learning_rate=.1, min_samples_leaf=300).fit(st(k, t), Vn); models[k] = m_
                Vn = np.where(G[t, k] >= m_.predict(st(k, t)), G[t, k], Vn)
            alive = np.ones(len(te), bool); v = G[te, NSTEP - 1].copy(); sk = np.full(len(te), NSTEP)
            for k in range(NSTEP - 1):
                ex = alive & (G[te, k] >= models[k].predict(st(k, te))); v[ex] = G[te, k][ex]; sk[ex] = k + 1; alive &= ~ex
            val[te], stop_k[te] = v, sk
        hold = G[:, -1]; rs = G[np.arange(len(d)), rng.integers(0, NSTEP, len(d))]; diff = val - hold; lo, hi = _boot(diff, day, rng)
        print(f"  {nm}: 학습한 청산 {val.mean():+.2f}bp · 끝까지 보유 {hold.mean():+.2f} · 무작위 시점 청산 {rs.mean():+.2f} | 학습 − 보유 {diff.mean():+.2f}bp [95% {lo:+.2f}, {hi:+.2f}] · 평균 청산 걸음 {stop_k.mean():.1f}/30 · 미리 아는 최선(상한) {G.max(1).mean():+.1f}bp")


# ── 사전등록 채점(docs/prereg/sr_ml_rl_20261118.md) ────────────────────────────────────────────────
YS, YMAIN, NB, QH = ("y_fp15_900", "y_fp10_300", "y_fp25_3600"), "y_fp15_900", 2000, 300
MKS = (hgb, et, R._lr)                                                  # 고정 앙상블: 세 모델 예측 확률의 평균(조율 없음)


def auc_(y, s):
    from scipy.stats import rankdata
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]; n1 = y.sum(); n0 = len(y) - n1
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def oof_ens(d, X, y):
    return np.mean([oof(d, X, y, mk) for mk in MKS], 0)


def fit_ens(d, X, y):
    ok = np.isfinite(y); return [mk().fit(d[X][ok], y[ok]) for mk in MKS]


def pred_ens(ms, d, X):
    return np.mean([m_.predict_proba(d[X])[:, 1] for m_ in ms], 0)


def sets(d):
    MAG, SGN = cols(d); return {"전체": MAG + SGN, "크기만": MAG, "크기+체결 매수 비중": MAG + [c for c in SGN if c.startswith(("s_imb", "s_tb"))]}


def cvens() -> None:
    """학습 구간 교차 예측(고정 앙상블) — 사전등록에 적는 설계 수치 + 채점 문턱(10·90 분위)의 출처."""
    import warnings; warnings.filterwarnings("ignore")
    d = pd.read_parquet(OUT / "ml_train.parquet"); o = {}
    for yc in YS:
        y = d[yc].to_numpy()
        for nm, X in sets(d).items():
            o[f"{yc}|{nm}"] = p = oof_ens(d, X, y); a, mn = fauc(d, y, p); print(f"{yc} {nm:18s} 교차 AUC {a:.4f} (토막 최소 {mn:.4f})", flush=True)
    r = d[f"r_{QH}"].to_numpy(); o["q"] = oof(d, sets(d)["전체"], np.clip(r, *np.percentile(r, [1, 99])), lambda: hgb(reg=True), reg=True)
    print(f"밴딧 |Q| 95분위 {np.quantile(np.abs(o['q']), .95):.3f}bp"); pd.DataFrame(o).to_parquet(OUT / "ml_train_oof_ens.parquet")


def evaluate(te: pd.DataFrame, y, P: dict, thr: tuple, rng) -> dict:
    """한 구간 채점. P = 재료 묶음별 점수. thr = 학습 교차 예측 «전체» 점수의 10·90 분위."""
    p, pm = P["전체"], P["크기만"]; ok = np.isfinite(y); day = te.ts.dt.floor("D").to_numpy(); mon = te.ts.dt.to_period("M").astype(str).to_numpy()
    top, bot = p >= thr[1], p <= thr[0]; days = np.unique(day); idx = {d_: np.flatnonzero(day == d_) for d_ in days}; b = {k: [] for k in ("a", "g", "dm", "df")}
    for _ in range(NB):                                                  # 달력일 블록 부트스트랩
        pk = np.concatenate([idx[d_] for d_ in rng.choice(days, len(days))]); yy = y[pk]; a = auc_(yy, p[pk])
        b["a"].append(a); b["g"].append(np.nanmean(yy[top[pk]]) - np.nanmean(yy[bot[pk]])); b["dm"].append(a - auc_(yy, pm[pk])); b["df"].append(auc_(yy, P["크기+체결 매수 비중"][pk]) - auc_(yy, pm[pk]))
    def placebo(strata, n=1000):
        out = []; grp = [g for g in pd.Series(np.arange(len(y))).groupby(strata).indices.values()]
        for _ in range(n):
            y2 = y.copy()
            for g in grp:
                y2[g] = rng.permutation(y[g])
            out.append(auc_(y2, p))
        return out
    pl_m = placebo(mon); pl_s = placebo(pd.Series(mon) + "_" + pd.qcut(pm, 5, labels=False, duplicates="drop").astype(str))
    a, am, af = auc_(y, p), auc_(y, pm), auc_(y, P["크기+체결 매수 비중"]); cnt = lambda g: [int(np.nansum(y[g])), int(np.isfinite(y[g]).sum())]
    q = pd.qcut(p, 10, labels=False, duplicates="drop"); rc = "r_" + [c for c in YS if te[c].to_numpy() is y or np.array_equal(te[c].to_numpy(), y, equal_nan=True)][0].split("_")[-1]
    o = {"사건": int(ok.sum()), "날": int(len(days)), "사건 쪽 먼저 비율": float(np.nanmean(y)), "방향_AUC": a, "방향_AUC_하한90": float(np.nanpercentile(b["a"], 10)), "월 안 위약_95분위": float(np.percentile(pl_m, 95)),
         "상위10%[사건 쪽 먼저, n]": cnt(top), "하위10%[사건 쪽 먼저, n]": cnt(bot), "집단차": float(np.nanmean(y[top]) - np.nanmean(y[bot])), "집단차_하한90": float(np.nanpercentile(b["g"], 10)),
         "AUC_크기만": am, "전체−크기만": a - am, "전체−크기만_하한90": float(np.nanpercentile(b["dm"], 10)), "층 안 위약_평균": float(np.mean(pl_s)), "층 안 위약_95분위": float(np.percentile(pl_s, 95)),
         "AUC_크기+체결 매수 비중": af, "체결 매수 비중 몫": af - am, "체결 매수 비중 몫_하한90": float(np.nanpercentile(b["df"], 10)),
         "10분위 사건 쪽 먼저 비율": [round(float(np.nanmean(y[q == i])), 3) for i in range(int(np.nanmax(q)) + 1)], f"10분위 평균 수익bp({rc})": [round(float(te[rc].to_numpy()[q == i].mean()), 2) for i in range(int(np.nanmax(q)) + 1)],
         "종류×거리 AUC[n]": {f"{k}{l}": [round(auc_(y[g], p[g]), 3), int(g.sum())] for k in ("brk", "apr") for l in LOOKS for g in [((te.kind == k) & (te.look == l)).to_numpy()]},
         "월별 AUC": {m_: round(auc_(y[mon == m_], p[mon == m_]), 3) for m_ in sorted(set(mon))}, "위·아래 AUC": [round(auc_(y[(te.side == s_).to_numpy()], p[(te.side == s_).to_numpy()]), 3) for s_ in (1, -1)]}
    o["1_방향"] = bool(o["방향_AUC_하한90"] > .5 and a > o["월 안 위약_95분위"]); o["2_집단차"] = bool(o["집단차_하한90"] > 0); o["통과"] = bool(o["1_방향"] and o["2_집단차"])
    o["후보_점추정"] = bool(a > .5 and o["집단차"] > 0); o["부호 재료 고유 몫(3·4)"] = bool(o["전체−크기만_하한90"] > 0 and a > o["층 안 위약_95분위"])
    return o


def score() -> None:
    import json
    import warnings; warnings.filterwarnings("ignore")
    tr, al = pd.read_parquet(OUT / "ml_train.parquet"), pd.read_parquet(OUT / "ml_test.parquet"); oo = pd.read_parquet(OUT / "ml_train_oof_ens.parquet"); rng = np.random.default_rng(R.SEED)
    seg = {"검증": al[al.ts < R.TEST0 - R.PG].reset_index(drop=True), "시험": al[al.ts >= R.TEST0].reset_index(drop=True)}; S_ = sets(tr); out = {"학습 사건": len(tr)}
    for yc in YS:
        ytr = tr[yc].to_numpy(); ms = {nm: fit_ens(tr, X, ytr) for nm, X in S_.items()}; thr = tuple(float(v) for v in np.quantile(oo[f"{yc}|전체"], [.1, .9])); out[yc] = {"문턱(학습 교차 10·90분위)": thr}
        for nm, te in seg.items():
            out[yc][nm] = evaluate(te, te[yc].to_numpy(), {k: pred_ens(ms[k], te, S_[k]) for k in S_}, thr, rng)
    # 매매 등급(양측 95%): 진입 밴딧 — 학습에서 맞춘 Q 로 방향을 정하고 |Q| 가 학습 교차 예측 95분위 이상인 사건만 진입
    r = tr[f"r_{QH}"].to_numpy(); mq = hgb(reg=True).fit(tr[S_["전체"]], np.clip(r, *np.percentile(r, [1, 99]))); qthr = float(np.quantile(np.abs(oo["q"]), .95)); out["밴딧"] = {"|Q| 문턱bp": qthr, "지평초": QH, "비용bp": 10}
    for nm, te in seg.items():
        q = mq.predict(te[S_["전체"]]); rr = te[f"r_{QH}"].to_numpy(); g = np.abs(q) >= qthr; pnl = (np.sign(q) * rr)[g]; day = te.ts.dt.floor("D").to_numpy()[g]; lo, hi = _boot(pnl, day, rng, 2000) if g.sum() > 30 else (np.nan, np.nan)
        rd = [float(np.mean(rng.choice([-1, 1], g.sum()) * rr[g])) for _ in range(1000)]; al_ = np.sign(q) * rr; lo2, hi2 = _boot(al_, te.ts.dt.floor("D").to_numpy(), rng, 2000)
        out["밴딧"][nm] = {"진입 수": int(g.sum()), "날": int(len(np.unique(day))), "비용 전 평균bp": float(pnl.mean()), "비용 전 95%": [lo, hi], "비용 뒤 평균bp": float(pnl.mean() - 10), "비용 뒤 95%": [lo - 10, hi - 10], "승률": float(np.mean(pnl > 0)),
                          "이긴 건 평균bp": float(pnl[pnl > 0].mean()), "진 건 평균bp": float(pnl[pnl <= 0].mean()), "늘 사건 쪽bp": float(rr[g].mean()), "늘 반대bp": float(-rr[g].mean()), "무작위 방향 97.5분위bp": float(np.percentile(rd, 97.5)),
                          "전 사건 정책 비용 전bp": float(al_.mean()), "전 사건 95%": [lo2, hi2], "예측↔실제 순위상관": float(pd.Series(q).corr(pd.Series(rr), method="spearman")),
                          "엄격 통과(비용 뒤 양측95% 하한>0)": bool(lo - 10 > 0), "통과 후보(비용 0 하한>0)": bool(lo > 0)}
    ytr = tr[YMAIN].to_numpy(); days = np.unique(tr.ts.dt.floor("D")); dd = tr.ts.dt.floor("D").to_numpy(); rs = []                 # 강건성: 학습 날짜를 다시 뽑아 부스팅만 10번
    for i in range(10):
        # 라벨 없는 1건 제외(첫 실행이 여기서 멈춤 — 결과 출력 전, 주 판정 계산과 무관)
        pk = np.flatnonzero(np.isin(dd, rng.choice(days, len(days))) & np.isfinite(ytr)); m_ = hgb(seed=i).fit(tr[S_["전체"]].iloc[pk], ytr[pk]); rs.append(round(auc_(seg["시험"][YMAIN].to_numpy(), m_.predict_proba(seg["시험"][S_["전체"]])[:, 1]), 4))
    out["학습 날짜 재추출 부스팅 시험 AUC"] = rs
    out["판정"] = {"통과(시험 주 라벨 1·2)": out[YMAIN]["시험"]["통과"], "통과 후보(검증·시험 점추정)": bool(out[YMAIN]["검증"]["후보_점추정"] and out[YMAIN]["시험"]["후보_점추정"]),
                  "부호 재료 고유 몫(시험 3·4)": out[YMAIN]["시험"]["부호 재료 고유 몫(3·4)"], "밴딧 엄격": out["밴딧"]["시험"]["엄격 통과(비용 뒤 양측95% 하한>0)"], "밴딧 통과 후보": out["밴딧"]["시험"]["통과 후보(비용 0 하한>0)"]}
    Path(R.ROOT / "docs/experiments/sr_ml_rl_result_20261118.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=1))


YCONF = "y_fp10_300"                                                    # 확인 검정(docs/prereg/sr_ml_confirm_2023_20261118.md): 2026 채점의 보조 라벨을 손대지 않은 2023 으로 1회


def confirm() -> None:
    import json
    import warnings; warnings.filterwarnings("ignore")
    tr, te = pd.read_parquet(OUT / "ml_train.parquet"), pd.read_parquet(OUT / "ml_confirm.parquet"); oo = pd.read_parquet(OUT / "ml_train_oof_ens.parquet"); rng = np.random.default_rng(R.SEED + 1)
    S_ = sets(tr); ytr = tr[YCONF].to_numpy(); ms = {nm: fit_ens(tr, X, ytr) for nm, X in S_.items()}; thr = tuple(float(v) for v in np.quantile(oo[f"{YCONF}|전체"], [.1, .9]))
    P = {k: pred_ens(ms[k], te, S_[k]) for k in S_}; y = te[YCONF].to_numpy(); out = {"라벨": YCONF, "구간": [str(te.ts.min()), str(te.ts.max())], "문턱": thr, "확인": evaluate(te, y, P, thr, rng)}
    _, first = np.unique(te.asec.to_numpy(), return_index=True); u = te.iloc[np.sort(first)].reset_index(drop=True); ix = np.sort(first)      # 같은 초의 중복 사건을 하나로(보고용)
    out["중복 제거(같은 초 1건)"] = {k: v for k, v in evaluate(u, y[ix], {k: v[ix] for k, v in P.items()}, thr, rng).items() if k in ("사건", "방향_AUC", "방향_AUC_하한90", "월 안 위약_95분위", "집단차", "집단차_하한90", "전체−크기만", "전체−크기만_하한90", "통과")}
    out["판정"] = {"통과(1·2)": out["확인"]["통과"], "부호 재료 고유 몫(3·4)": out["확인"]["부호 재료 고유 몫(3·4)"]}
    Path(R.ROOT / "docs/experiments/sr_ml_confirm_2023_20261118.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=1))


FWD0, FWD_JUDGE = "2026-11-15 18:00", "2026-12-16"                       # 전진 판정(docs/prereg/sr_ml_forward_20261118.md): 판정일 전의 실행은 기록만


def forward() -> None:
    """얼린 모델(학습 2024~25 재적합 — 결정적)을 2026-11-16 이후 새 Vision 일 파일에 적용해 장부에 한 줄 덧붙인다."""
    import json
    import warnings; warnings.filterwarnings("ignore")
    import build_eth_aggtrades_1s_20261118 as B
    day, last, today = pd.Timestamp("2026-11-16"), None, pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    while day < today:                                                    # 정적 파일만(REST 아님). 아직 안 올라온 날에서 멈춘다
        if any(k in B.one(day.strftime("%Y-%m-%d")) for k in ("없음", "실패")):
            break
        last, day = day, day + pd.Timedelta(days=1)
    if last is None:
        print("전진 자료 없음"); return
    cache = OUT / "min_from_tape_fwd.parquet"; cache.unlink(missing_ok=True); te = build(R.minute_bars(cache=cache), FWD0, str(last + pd.Timedelta(days=1)))
    tr, oo = pd.read_parquet(OUT / "ml_train.parquet"), pd.read_parquet(OUT / "ml_train_oof_ens.parquet"); S_ = sets(tr); ytr = tr[YCONF].to_numpy(); rng = np.random.default_rng(R.SEED + 2)
    thr = tuple(float(v) for v in np.quantile(oo[f"{YCONF}|전체"], [.1, .9])); P = {k: pred_ens(fit_ens(tr, X, ytr), te, X) for k, X in S_.items()}; e = evaluate(te, te[YCONF].to_numpy(), P, thr, rng)
    final = bool(last >= pd.Timestamp(FWD_JUDGE) - pd.Timedelta(days=1))
    row = {"실행": str(pd.Timestamp.now(tz="UTC"))[:16], "구간": [str(te.ts.min()), str(te.ts.max())], "판정 실행인가": final, **{k: e[k] for k in ("사건", "날", "사건 쪽 먼저 비율", "방향_AUC", "방향_AUC_하한90", "월 안 위약_95분위",
           "상위10%[사건 쪽 먼저, n]", "하위10%[사건 쪽 먼저, n]", "집단차", "집단차_하한90", "전체−크기만", "통과")}}
    p = R.ROOT / "docs/experiments/sr_ml_forward_ledger_20261118.json"; led = json.loads(p.read_text()) if p.exists() else []; led.append(row); p.write_text(json.dumps(led, ensure_ascii=False, indent=1)); print(json.dumps(row, ensure_ascii=False, indent=1))


def selftest() -> None:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); n = 4 * 86400; s0 = 1_700_000_000 // 86400 * 86400
        p = 2000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n)))
        def write(pp, bq, sq):
            for j in range(4):
                sl = slice(j * 86400, (j + 1) * 86400); day = pd.Timestamp(s0 + j * 86400, unit="s").strftime("%Y-%m-%d")
                pd.DataFrame({"s": np.arange(s0 + j * 86400, s0 + (j + 1) * 86400), "po": pp[sl], "ph": pp[sl] * (1 + 1e-6), "pl": pp[sl] * (1 - 1e-6), "pc": pp[sl], "bq": bq[sl], "sq": sq[sl],
                              "nb": 1, "ns": 1, "nt": 3.0, "mq": 1.0, "q2": bq[sl] ** 2 + sq[sl] ** 2}).to_parquet(root / f"ETHUSDT-{day}.parquet")
        bq, sq = rng.exponential(1, n), rng.exponential(1, n); write(p, bq, sq)
        m = R.minute_bars(root, cache=root / "m.parquet"); t0, t1 = pd.Timestamp(s0 + 2 * 86400, unit="s"), pd.Timestamp(s0 + 3 * 86400 - 2 * 3600, unit="s")
        d = build(m, str(t0), str(t1), tape=R.Tape(root)); assert len(d) > 20, len(d); MAG, SGN = cols(d)
        assert not any(c.startswith(("y_", "r_", "q_")) for c in MAG + SGN) and len(MAG) > 40 and len(SGN) > 30
        e = d.iloc[len(d) // 2]; dsec = int(e.asec) + W; i = dsec - s0                                    # 손 계산 대조
        assert abs(e.r_900 - e.side * np.log(p[i + 900] / p[i]) * 1e4) < 1e-6 and abs(e.s_ret60 - e.side * np.log(p[i] / p[i - 60]) / (e.s1h / 60 * np.sqrt(60))) < 1e-6
        assert abs(e.s_imb10 - e.side * (bq[i - 9:i + 1].sum() - sq[i - 9:i + 1].sum()) / (bq[i - 9:i + 1].sum() + sq[i - 9:i + 1].sum())) < 1e-6
        up = np.flatnonzero((p[i + 1:i + 901] * (1 + 1e-6) >= p[i] * np.exp(.15 * e.s1h))); dn = np.flatnonzero((p[i + 1:i + 901] * (1 - 1e-6) <= p[i] * np.exp(-.15 * e.s1h)))
        fu, fd = (up[0] if len(up) else 900), (dn[0] if len(dn) else 900); z = (1 if fu < fd else -1 if fd < fu else np.sign(np.log(p[i + 900] / p[i]))) * e.side; assert e.y_fp15_900 == float(z > 0)
        # 미래 불변: 결정 초 뒤의 체결을 전부 바꿔도 그 사건(과 그 앞 사건)의 피쳐·사건 선택이 같다
        p2, b2, s2 = p.copy(), bq.copy(), sq.copy(); p2[i + 1:] *= 1.05; b2[i + 1:] = 7.0; s2[i + 1:] = 0.1
        root2 = root / "b"; root2.mkdir(); root_, root = root, root2; write(p2, b2, s2); root = root_
        m2 = R.minute_bars(root2, cache=root2 / "m.parquet"); d2 = build(m2, str(t0), str(t1), tape=R.Tape(root2))
        a_, b_ = d[d.asec <= e.asec].reset_index(drop=True), d2[d2.asec <= e.asec].reset_index(drop=True)
        assert len(a_) == len(b_) and (a_[["asec", "side", "kind", "look"]].to_numpy() == b_[["asec", "side", "kind", "look"]].to_numpy()).all(), "사건 선택이 미래에 의존"
        assert np.allclose(a_[MAG + SGN].to_numpy(float), b_[MAG + SGN].to_numpy(float), equal_nan=True, atol=1e-9), "피쳐가 결정 뒤 자료에 의존"
        assert not np.allclose(a_.r_900.to_numpy()[-1], b_.r_900.to_numpy()[-1]), "대조: 라벨은 바뀌어야 한다"
        # 채점기: 정보 없는 점수는 불통과, 라벨을 아는 점수는 통과, 손 AUC 대조
        from sklearn.metrics import roc_auc_score
        yy = rng.integers(0, 2, 3000).astype(float); ss = rng.normal(size=3000); assert abs(auc_(yy, ss) - roc_auc_score(yy, ss)) < 1e-12
        te = pd.DataFrame({"ts": pd.Timestamp("2026-06-01") + pd.to_timedelta(np.sort(rng.integers(0, 90 * 86400, 3000)), unit="s"), "kind": rng.choice(["brk", "apr"], 3000), "look": rng.choice(LOOKS, 3000), "side": rng.choice([1, -1], 3000),
                           "y_fp15_900": yy, "y_fp10_300": 1 - yy, "y_fp25_3600": np.r_[yy[1:], 0.0], "r_900": rng.normal(size=3000)})
        global NB; nb0, NB = NB, 200; nz = lambda: rng.normal(size=3000)
        e0 = evaluate(te, yy, {"전체": nz(), "크기만": nz(), "크기+체결 매수 비중": nz()}, (-1.28, 1.28), rng); e1 = evaluate(te, yy, {"전체": yy + nz(), "크기만": nz(), "크기+체결 매수 비중": nz()}, (-.8, 1.8), rng); NB = nb0
        assert not e0["통과"] and not e0["부호 재료 고유 몫(3·4)"] and e1["통과"] and e1["부호 재료 고유 몫(3·4)"] and e1["방향_AUC"] > .7, (e0["방향_AUC"], e1["방향_AUC"])
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "build":
        rng_ = {"train": ("2024-01-01", str(R.VAL0 - R.PG)), "test": (str(R.VAL0), "2026-11-15 18:00"), "confirm": ("2023-01-01", "2023-12-31 18:00")}[sys.argv[2]]
        m_ = R.minute_bars(cache=OUT / "min_from_tape_2023.parquet") if sys.argv[2] == "confirm" else R.minute_bars()      # 2023 은 받은 뒤 분봉을 따로 만든다(학습·시험 분봉 캐시는 그대로)
        d_ = build(m_, *rng_); d_.to_parquet(OUT / f"ml_{sys.argv[2]}.parquet"); print(sys.argv[2], len(d_), d_.ts.min(), d_.ts.max())
    elif cmd == "confirm":
        confirm()
    elif cmd == "forward":
        forward()
    elif cmd == "explore":
        explore()
    elif cmd == "rl":
        rl()
    elif cmd == "cvens":
        cvens()
    elif cmd == "score":
        score()
    else:
        sys.exit(__doc__)
