#!/usr/bin/env python3
"""XRP 15분·1시간 방향 예측 — 배포 ETH 레시피(65열판) 이식 검정 (사전등록 docs/prereg/xrp_dir_short_20261031.md).

팔 X = XRP 로 학습 · 팔 E = ETH 로 학습해 XRP 피쳐에 적용. 레시피·표적·시드·학습 행 = scripts/live_eth_dir_short_20261028.py 그대로(호가 깊이 열만 뺌).
대상 코인 바꾸기 = frames["ETH"] 자리에 XRP, frames["XRP"] 자리에 ETH(교차 코인 묶음 유지) — build_features 수정 없음.

    python scripts/research_xrp_dir_short_20261031.py --selftest
    python scripts/research_xrp_dir_short_20261031.py            # 학습(매달 10회 × 2팔 × 2지평 × 20모형) → tmp/xrp_dir_short_20261031/preds.npz
    python scripts/research_xrp_dir_short_20261031.py --score    # 채점 → report.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as ds  # noqa: E402

PANEL = ROOT / "data/live/dirshort_panel"
OUT = ROOT / "tmp/xrp_dir_short_20261031"
MONTHS = pd.date_range("2026-01-01", "2026-10-01", freq="MS")
VAL = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01"))
TEST = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-28"))
NB, NP = 2000, 200


def frames_for(target: str) -> dict:
    f = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    if target != "ETH":   # 대상 코인을 ETH 자리로, ETH 는 대상 코인의 교차 자리로(BTC 면 X_btc_*·U_btc_* = ETH)
        f["ETH"], f[target] = f[target], f["ETH"]
    return f


def dataset65(frames: dict):
    """ds.dataset 에서 호가 깊이 열만 뺀 것(같은 시각 규약: 피쳐 = 봉 t−5분 행)."""
    e = frames["ETH"].drop_duplicates("timestamp").set_index("timestamp").close.astype(float).sort_index()
    g = pd.date_range(e.index[0], e.index[-1], freq="5min")
    P = pd.Series(e.reindex(g).ffill().to_numpy(), index=g + pd.Timedelta("5min"))
    t = P.index[P.index >= ds.START]
    ret = P.reindex(t + pd.Timedelta("5min")).to_numpy() / P.reindex(t).to_numpy() - 1
    sig = P.pct_change().rolling(288).std().reindex(t).to_numpy()
    dc = frames["ETH"].set_index("timestamp").close.ffill().groupby(lambda x: x.normalize()).last()
    F = d4.build_features(frames, dc).set_index("ts").reindex(t - pd.Timedelta("5min"))[d4.FEATS]
    return t, F.to_numpy(np.float32), ret, sig


D: dict = {}   # fork 로 일꾼에 넘김


def fit_month(job):
    """(학습 코인, h, 달) → 그 달 XRP 행 예측(시드 5 평균)."""
    import lightgbm as lgb
    src, h, month = job
    t, X, ret, sig = D[src]; t_ns = t.asi8
    m0 = int(np.searchsorted(t_ns, month.value)); m1 = int(np.searchsorted(t_ns, (month + pd.offsets.MonthBegin(1)).value))
    y = ds.fwd(ret, sig, h); ok = np.isfinite(ret) & (sig > 0) & np.isfinite(y)
    tr = ds.train_rows(t_ns, ok, m0, h); hl = ds.HZ[h]
    w = None if hl is None else 0.5 ** ((t_ns[m0] - t_ns[tr]) / ds.DAY / hl)
    Xp = D["XRP"][1][m0:m1]; out = []
    for seed in ds.SEEDS:
        acc = 0.0
        for k in ds.ENS:
            m = lgb.LGBMRegressor(**ds.BASE, **({"objective": "huber", "alpha": 1.0} if k == "huber" else {}), n_jobs=ds.NJ, verbose=-1,
                                  random_state=seed).fit(X[tr], ds.target(y[tr], k), sample_weight=w)
            acc = acc + m.predict(Xp) / float(m.predict(X[tr[::50]]).std() + 1e-12)
        out.append(acc)
    print(f"  {src} h{h} {month.date()} 학습 {len(tr):,}행 → 예측 {m1 - m0:,}", flush=True)
    return job, m0, np.array(out)


def run():
    for s in ("XRP", "ETH"):
        D[s] = dataset65(frames_for(s))
    assert (D["XRP"][0] == D["ETH"][0]).all(), "두 코인 결정 시각 격자가 다르다"
    t = D["XRP"][0]; n = len(t)
    jobs = [(s, h, m) for s in ("XRP", "ETH") for h in ds.HZ for m in MONTHS]
    P = {f"{s}_{h}": np.full((len(ds.SEEDS), n), np.nan) for s in ("XRP", "ETH") for h in ds.HZ}
    with get_context("fork").Pool(3) as pool:
        for (s, h, m), m0, pr in pool.imap_unordered(fit_month, jobs):
            P[f"{s}_{h}"][:, m0:m0 + pr.shape[1]] = pr
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "preds.npz", t=t.asi8, ret=D["XRP"][2], sig=D["XRP"][3], **P)
    print("saved", OUT / "preds.npz")


# ── 채점 ──────────────────────────────────────────────────────────────────
def spear(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot_ic(p, y, day, rng, nb=NB, blk=5):
    """5일 블록 날짜 재표집 → 풀링 스피어만 분포."""
    ud = np.unique(day); starts = np.arange(0, len(ud), blk)
    rows = [np.flatnonzero(np.isin(day, ud[s:s + blk])) for s in starts]
    out = np.empty(nb)
    for b in range(nb):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        out[b] = spear(p[idx], y[idx])
    return out


def score(coin="XRP", out=OUT):
    z = np.load(out / "preds.npz"); t = pd.DatetimeIndex(z["t"]); ret, sig = z["ret"], z["sig"]
    rng = np.random.default_rng(20261031); rep = {}
    for h, nm in ds.NAMES.items():
        y = ds.fwd(ret, sig, h)
        mval = (t >= VAL[0]) & (t < VAL[1]) & np.isfinite(y)
        mtest = (t >= TEST[0]) & (t < TEST[1]) & np.isfinite(y)
        arms = {a: z[f"{a}_{h}"].mean(0) for a in (coin, "ETH")}
        val_ic = {a: spear(p[mval], y[mval]) for a, p in arms.items()}
        pick = max(val_ic, key=val_ic.get)
        r = {"val_ic": val_ic, "pick": pick, "n_val": int(mval.sum()), "n_test": int(mtest.sum()),
             "test_days": int(len(np.unique(t[mtest].normalize())))}
        day = t[mtest].normalize().asi8; yt = y[mtest]
        for a, p in arms.items():
            assert np.isfinite(p[mtest]).all() and np.isfinite(p[mval]).all(), f"{a} 예측 구멍"
            bs = boot_ic(p[mtest], yt, day, rng)
            r[a] = {"test_ic": spear(p[mtest], yt), "p10": float(np.quantile(bs, .10)), "p05": float(np.quantile(bs, .05)),
                    "p95": float(np.quantile(bs, .95)),
                    "seed_ic": [spear(z[f"{a}_{h}"][i][mtest], yt) for i in range(len(ds.SEEDS))],
                    "month_ic": {str(m.date()): spear(p[mm], y[mm]) for m in MONTHS
                                 for mm in [(t >= m) & (t < m + pd.offsets.MonthBegin(1)) & np.isfinite(y)] if mm.sum() > 1000}}
        # ΔIC X−E 같은 블록
        bd = boot_ic(arms[coin][mtest], yt, day, np.random.default_rng(7)) - boot_ic(arms["ETH"][mtest], yt, day, np.random.default_rng(7))
        r["d_ic_X_minus_E"] = {"est": r[coin]["test_ic"] - r["ETH"]["test_ic"], "p05": float(np.quantile(bd, .05)), "p95": float(np.quantile(bd, .95))}
        # 위약(고른 팔)
        p_full = arms[pick]; idx = np.flatnonzero((t >= TEST[0]) & (t < TEST[1]))
        nday = len(idx) // 288; ks = rng.choice(np.arange(1, nday), size=min(NP, nday - 1), replace=False)
        pt = p_full[idx]; ok = np.isfinite(y[idx])
        circ = np.array([spear(np.roll(pt, int(k) * 288)[ok], y[idx][ok]) for k in ks])
        rnd = np.array([spear(rng.standard_normal(ok.sum()), y[idx][ok]) for _ in range(NP)])
        ic = r[pick]["test_ic"]
        r["placebo"] = {"circ_q90": float(np.quantile(circ, .90)), "circ_q95": float(np.quantile(circ, .95)), "circ_pct": float((circ < ic).mean()),
                        "rand_q95": float(np.quantile(rnd, .95)), "n_circ": int(len(ks))}
        strict = r[pick]["p10"] > 0 and ic > r["placebo"]["circ_q95"] and ic > r["placebo"]["rand_q95"]
        cand = (not strict) and ic > 0 and ic > r["placebo"]["circ_q90"]
        r["verdict"] = "엄격 통과" if strict else ("통과 후보" if cand else "불통과")
        # 표시 규칙 적중(검증 20/80 문턱)
        lo, hi = np.quantile(arms[pick][mval], [.2, .8]); pp = arms[pick][mtest]; up = yt > 0
        tail = (pp <= lo) | (pp >= hi); hit = np.where(pp >= hi, up, ~up)
        r["display"] = {"thr": [float(lo), float(hi)], "tail_hits": f"{int(hit[tail].sum())}/{int(tail.sum())}",
                        "tail_rate": float(hit[tail].mean()), "mid_rate": float((np.sign(pp[~tail] - np.median(arms[pick][mval])) == np.sign(yt[~tail])).mean()),
                        "base_up": float(up.mean()), "long_tail_rate": float(up[pp >= hi].mean()), "short_tail_rate": float((~up)[pp <= lo].mean())}
        rep[nm] = r
        print(nm, json.dumps({k: r[k] for k in ("val_ic", "pick", "verdict")}, ensure_ascii=False), "test", round(ic, 4), "p10", round(r[pick]["p10"], 4))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


def selftest(coin="XRP"):
    f = frames_for(coin)
    raw = pd.read_parquet(PANEL / f"{coin}USDT.parquet").set_index("timestamp").close
    assert f["ETH"].close.iloc[-1] == raw.iloc[-1] and coin in f and f[coin].close.iloc[-1] != raw.iloc[-1]
    sub = {k: v[v.timestamp >= "2025-10-01"] for k, v in f.items()}
    old, ds.START = ds.START, pd.Timestamp("2025-11-01")
    t, X, ret, sig = dataset65(sub); ds.START = old
    j = d4.FEATS.index("P_r5m"); i = 5000
    lc = np.log(raw)   # 결정 시각 t 의 P_r5m = 봉 t−5분(마감 t) 의 로그수익 — 대상 코인 자신
    want = (lc[t[i] - pd.Timedelta("5min")] - lc[t[i] - pd.Timedelta("10min")]) * 1e4
    assert abs(X[i, j] - want) < 1e-3, (X[i, j], want)
    assert abs(ret[i] - (raw[t[i]] / raw[t[i] - pd.Timedelta("5min")] - 1)) < 1e-12   # 표적 첫 봉 = 결정 뒤 봉
    rng = np.random.default_rng(0); y = rng.standard_normal(3000); day = np.repeat(np.arange(30), 100)
    assert abs(np.median(boot_ic(y, y, day, rng, nb=50)) - 1) < 1e-9
    assert abs(np.median(boot_ic(rng.standard_normal(3000), y, day, rng, nb=200))) < .03
    print("selftest ok", coin)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--score" in sys.argv:
        score()
    else:
        run()
