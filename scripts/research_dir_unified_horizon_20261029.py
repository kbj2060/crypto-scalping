#!/usr/bin/env python3
"""15분·1시간·4h 방향 예측 — 구간 합 통합 모델(S) · 리지 결합층(C) vs 현재 3모델(B) (2026-10-29).

사전등록: docs/prereg/dir_unified_horizon_20261029.md (075a441a, 계산 전 커밋).
B 15분·1시간 = 배포 live_eth_dir_short_20261028.train/predict 그대로 · B 4h = live_eth_dir4h_20261015.train 의 적합 네 줄(365일·[::2]·HGB·시드 0).
적합 시점 A = 2026-01-01(검증 예측) · B = 2026-06-01(시험 예측). 결정 시각 t = 봉 마감, 입력 = 봉 t−5분 행.

    python scripts/research_dir_unified_horizon_20261029.py --selftest
    python scripts/research_dir_unified_horizon_20261029.py      # (서버 저장소 루트) → tmp/dir_unified_20261029/{preds.npz, report.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir_short_20261028 as ds  # noqa: E402

d4 = ds.d4
OUT = ROOT / "tmp/dir_unified_20261029"
FIT = {"A": pd.Timestamp("2026-01-01"), "B": pd.Timestamp("2026-06-01")}
VAL, TEST = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01")), (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-28"))
HS = {"15m": 3, "1h": 12, "4h": 48}
SEG = [(0, 3), (3, 12), (12, 48)]                     # 구간 [a, b) 봉
SEEDS_S = [int(x) for x in np.random.default_rng(2026102904).integers(1, 2**31 - 1, 5)]


def seg_targets(ret: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """→ (3, N) 구간 수익 합 / (σ24·√길이). 끝을 넘으면 NaN."""
    cs = np.r_[0.0, np.cumsum(ret)]; n = len(ret); out = np.full((3, n), np.nan); k = np.arange(n - 48 + 1)
    for j, (a, b) in enumerate(SEG):
        out[j, k] = (cs[k + b] - cs[k + a]) / (np.where(sig[k] > 0, sig[k], np.nan) * np.sqrt(b - a))
    return out


def seg_to_horizons(s: np.ndarray) -> dict:
    """구간 예측 (3, N) → 지평 예측(σ24 단위): 1시간은 15분을, 4h 는 1시간을 포함."""
    p15 = s[0] * np.sqrt(3); p1h = p15 + s[1] * 3.0
    return {"15m": p15, "1h": p1h, "4h": p1h + s[2] * 6.0}


def fit_S(X, Y, tr):
    """쌓은 행(구간 3 × 학습 행) + seg 열 → LightGBM × 목적 2 × 시드 5. 단위 보존(표준편차로 안 나눔)."""
    import lightgbm as lgb
    Xs = np.vstack([np.column_stack([X[tr], np.full(len(tr), j, np.float32)]) for j in range(3)])
    ys = np.concatenate([Y[j, tr] for j in range(3)])
    ms = []
    for sd in SEEDS_S:
        for k in ("clip2", "huber"):
            ms.append(lgb.LGBMRegressor(**ds.BASE, **({"objective": "huber", "alpha": 1.0} if k == "huber" else {}), n_jobs=ds.NJ, verbose=-1,
                                        random_state=sd).fit(Xs, ds.target(ys, k)))
    return ms


def predict_S(ms, X):
    return np.stack([np.mean([m.predict(np.column_stack([X, np.full(len(X), j, np.float32)])) for m in ms], 0) for j in range(3)])


def fit_all():
    from sklearn.ensemble import HistGradientBoostingRegressor
    frames = {c: pd.read_parquet(ds.PANEL / f"{c}USDT.parquet") for c in ["ETH"] + d4.COINS}
    book = pd.read_parquet(ds.BOOK)
    t, X, ret, sig = ds.dataset(frames, book); t_ns = t.asi8
    F4 = d4.build_features(frames); ts4 = F4.ts.to_numpy(); y4 = F4.y_4h.to_numpy(); X4 = F4[d4.FEATS].to_numpy(np.float32)
    i4 = pd.Series(np.arange(len(F4)), index=pd.DatetimeIndex(ts4) + pd.Timedelta("5min")).reindex(t).to_numpy()   # 결정 시각 t ↔ 4h 행
    assert np.isfinite(i4).all(), "4h 피쳐 행이 결정 시각과 안 맞음"
    i4 = i4.astype(int)
    a, b = X4[i4], X[:, :65]; same = np.isclose(a, b, equal_nan=True, rtol=1e-5).mean(0)   # 한 봉 어긋나면 수익 열이 크게 갈린다
    print("4h 피쳐 ↔ X 앞 65열 같은 비율: 최소", round(float(same.min()), 4), "· 1 미만 열", [d4.FEATS[i] for i in np.flatnonzero(same < .999)], flush=True)
    assert same[d4.FEATS.index("P_r5m")] > .999, "4h 행 맞춤이 한 봉 어긋남"
    Y = seg_targets(ret, sig); valid = np.isfinite(ret) & (sig > 0)
    P = {}
    for nm, month in FIT.items():
        m0 = int(np.searchsorted(t_ns, month.value)); assert t_ns[m0] == month.value
        art = ds.train(frames, book, month)
        pb = ds.predict(art, X)
        cut = np.datetime64(month) - np.timedelta64(5 * d4.H, "m")
        tr4 = (ts4 < cut) & (ts4 >= cut - np.timedelta64(365, "D")) & np.isfinite(y4)
        h4 = HistGradientBoostingRegressor(random_state=0, **d4.HGB).fit(X4[tr4][::2], y4[tr4][::2])
        trS = ds.train_rows(t_ns, valid & np.isfinite(Y).all(0), m0, 48)
        assert trS.max() + 48 < m0
        S = predict_S(fit_S(X, Y, trS), X)
        P[nm] = dict(B15=pb[3], B1h=pb[12], B4h=h4.predict(X4[i4]), S0=S[0], S1=S[1], S2=S[2])
        print(f"적합 {nm} {month.date()} · S 학습 {len(trS):,}행 · 4h 학습 {int(tr4.sum()):,}행", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "preds.npz", t=t.asi8, ret=ret, sig=sig, **{f"{k}_{nm}": v for nm, d in P.items() for k, v in d.items()})


# ── 채점 ─────────────────────────────────────────────────────────────────
def day_sums(x, y, day):
    """날짜별 [n, Σx, Σy, Σx², Σy², Σxy] — 부트스트랩에서 피어슨을 합으로 다시 만든다."""
    z = np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])
    return pd.DataFrame(z).groupby(day).sum().to_numpy()


def corr_from(S):
    n, sx, sy, sxx, syy, sxy = S.T if S.ndim == 2 else S
    return (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))


def boot_idx(nd, rng, B=2000, blk=5):
    st = rng.integers(0, nd - blk + 1, (B, -(-nd // blk)))
    return ((st[:, :, None] + np.arange(blk)).reshape(B, -1))[:, :nd]


def rank(x):
    return pd.Series(x).rank().to_numpy()


def tails(p, ref):
    lo, hi = np.quantile(ref, [.2, .8]); return np.where(p >= hi, 1, np.where(p <= lo, -1, 0))


def evaluate():
    from sklearn.linear_model import Ridge
    z = np.load(OUT / "preds.npz"); t = pd.DatetimeIndex(z["t"]); ret, sig = z["ret"], z["sig"]
    R = {h: ds.fwd(ret, np.ones_like(ret), n) for h, n in HS.items()}                      # 평가 표적: 단순수익 합
    Yz = {h: np.clip(ds.fwd(ret, sig, n), -5, 5) for h, n in HS.items()}                   # 결합층 표적
    end = lambda w: (t >= w[0]) & (t + pd.Timedelta(minutes=5 * 48) <= w[1]) & np.isfinite(R["4h"])  # noqa: E731
    mv, mt = end(VAL), end(TEST)
    arms = {}
    for nm in ("A", "B"):
        Sh = seg_to_horizons(np.stack([z[f"S{j}_{nm}"] for j in range(3)]))
        arms[nm] = {"B": {"15m": z[f"B15_{nm}"], "1h": z[f"B1h_{nm}"], "4h": z[f"B4h_{nm}"]}, "S": Sh}
    base_v = np.column_stack([arms["A"]["B"][h][mv] for h in HS]); mu, sd = base_v.mean(0), base_v.std(0)
    coef = {}
    for nm in ("A", "B"):
        Z = (np.column_stack([arms[nm]["B"][h] for h in HS]) - mu) / sd
        arms[nm]["C"] = {}
        for h in HS:
            if h not in coef:
                r = Ridge(alpha=1.0).fit(((base_v - mu) / sd), Yz[h][mv]); coef[h] = r
            arms[nm]["C"][h] = coef[h].predict(Z)
    rep = {"prereg": "docs/prereg/dir_unified_horizon_20261029.md@075a441a", "rows": {"val": int(mv.sum()), "test": int(mt.sum())},
           "seeds_S": SEEDS_S, "ridge": {h: dict(coef=[round(float(c), 4) for c in r.coef_], icpt=round(float(r.intercept_), 4)) for h, r in coef.items()}}
    rng = np.random.default_rng(20261029)
    for win, nm, m in (("val", "A", mv), ("test", "B", mt)):
        day = t[m].normalize(); udays = np.unique(day); nd = len(udays); di = np.searchsorted(udays, day)
        idx = boot_idx(nd, rng)
        ic, dsum = {}, {}
        for a in ("B", "S", "C"):
            for h in HS:
                x, y = rank(arms[nm][a][h][m]), rank(R[h][m])
                ic[(a, h)] = float(np.corrcoef(x, y)[0, 1])
                dsum[(a, h)] = day_sums(x, y, di)
        out = {"ic": {f"{a}/{h}": round(v, 4) for (a, h), v in ic.items()}}
        for a in ("S", "C"):
            dd = np.stack([corr_from(dsum[(a, h)][idx].sum(1)) - corr_from(dsum[("B", h)][idx].sum(1)) for h in HS])   # (3, B)
            mean = dd.mean(0)
            out[a] = dict(d={h: round(ic[(a, h)] - ic[("B", h)], 4) for h in HS},
                          d_q10={h: round(float(np.quantile(dd[i], .1)), 4) for i, h in enumerate(HS)},
                          d_q90={h: round(float(np.quantile(dd[i], .9)), 4) for i, h in enumerate(HS)},
                          mean=round(float(np.mean([ic[(a, h)] - ic[("B", h)] for h in HS])), 4), mean_q10=round(float(np.quantile(mean, .1)), 4),
                          p_not_better=round(float((mean <= 0).mean()), 4))
        # 일관성: 문턱 = 적합 A 의 검증 예측 20/80 분위
        coh = {}
        for a in ("B", "S", "C"):
            sg = {h: tails(arms[nm][a][h][m], arms["A"][a][h][mv]) for h in HS}
            c = {}
            for p, q in (("15m", "1h"), ("15m", "4h"), ("1h", "4h")):
                j = (sg[p] != 0) & (sg[q] != 0)
                same, opp = j & (sg[p] == sg[q]), j & (sg[p] != sg[q])
                hit = lambda k, h: round(float((np.sign(R[h][m][k]) == sg[h][k]).mean()), 4) if k.any() else None  # noqa: E731
                c[f"{p}~{q}"] = dict(both=int(j.sum()), opposite=round(float(opp.sum() / max(j.sum(), 1)), 4),
                                     hit_same={p: hit(same, p), q: hit(same, q)}, hit_opp={p: hit(opp, p), q: hit(opp, q)})
            coh[a] = c
        out["coherence"] = coh
        mon = t[m].to_period("M")
        out["monthly_ic"] = {f"{a}/{h}": {str(p): round(float(pd.Series(arms[nm][a][h][m][mon == p]).corr(pd.Series(R[h][m][mon == p]), method="spearman")), 4)
                                          for p in mon.unique()} for a in ("B", "S", "C") for h in HS}
        rep[win] = out
    v, s = rep["val"], rep["test"]
    verdict = {}
    for a in ("S", "C"):
        no_worse = all(s[a]["d_q90"][h] >= 0 for h in HS)
        val_ok = v[a]["mean"] > 0 if a == "S" else True
        adopt = s[a]["mean_q10"] > 0 and no_worse and val_ok
        noninf = (not adopt) and s[a]["mean_q10"] > -0.005 and no_worse and val_ok and \
            s["coherence"][a]["15m~4h"]["opposite"] < s["coherence"]["B"]["15m~4h"]["opposite"]
        verdict[a] = dict(adopt=adopt, noninferior=noninf, no_worse=no_worse, val_ok=val_ok,
                          candidate=bool(s[a]["mean"] > 0 and (v[a]["mean"] > 0 if a == "S" else True)))
    rep["verdict"] = verdict
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ("rows", "verdict")}, ensure_ascii=False))
    for w in ("val", "test"):
        print(w, json.dumps(rep[w]["ic"]), json.dumps({a: rep[w][a] for a in ("S", "C")}, ensure_ascii=False))
        print(" coherence", json.dumps({a: {k: c["opposite"] for k, c in rep[w]["coherence"][a].items()} for a in ("B", "S", "C")}))


def selftest():
    rng = np.random.default_rng(0); n = 500
    ret = rng.normal(0, 1e-3, n); sig = np.full(n, 1e-3)
    Y = seg_targets(ret, sig); P = seg_to_horizons(Y)
    for h, k in HS.items():   # 참 구간 값을 넣으면 지평 예측 × σ = 실제 지평 수익
        f = ds.fwd(ret, np.ones(n), k); ok = np.isfinite(P[h])
        assert np.allclose(P[h][ok] * 1e-3, f[ok]), h
    assert np.isnan(Y[:, -47:]).all() and np.isfinite(Y[:, :-47]).all()   # 마지막 완전 창 = 봉 n−48
    x, y = rng.normal(size=300), rng.normal(size=300); day = np.repeat(np.arange(30), 10)
    S = day_sums(x, y, day)
    assert np.isclose(corr_from(S.sum(0)), np.corrcoef(x, y)[0, 1])
    idx = boot_idx(30, rng, B=3)
    assert idx.shape == (3, 30) and idx.max() < 30 and idx.min() >= 0
    p = np.arange(10.0); assert tails(p, p).tolist() == [-1, -1, 0, 0, 0, 0, 0, 0, 1, 1]
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        if not (OUT / "preds.npz").exists() or "--refit" in sys.argv:
            fit_all()
        evaluate()
