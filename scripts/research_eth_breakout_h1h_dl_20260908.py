#!/usr/bin/env python3
"""H=1시간(12봉) · 배리어 ±0.30% -- **TabPFN / TabICL** + B=30 셔플 귀무 (2026-09-08).

사용자: *"최고 구성에 대해 B=30 정도로 셔플해주고 H=1h·P=0.30% 딥러닝 실행해줘"*

## 왜 B=30 인가
순열검정의 최소 p 는 1/(B+1) 이다. B=5 → p≤0.167, B=10 → 0.091, **B=30 → 0.032** 로
비로소 0.05 아래로 내려간다. TabICL B=5 결과(세 창 모두 셔플 max 초과)는 방향은 맞았지만
증거로는 약했고, VAL 은 셔플 max 대비 **0.0009 차이**였다.

## ⚠️셔플 평균이 0.50 이 아니다 (실측)
TabICL B=5 셔플 평균 VAL **0.5163** / OOS 0.5003 / HOLDOUT 0.5132.
일군집 구조 때문에 귀무가 0.5 보다 위다 -- **기저(다수결)가 아니라 셔플 분포가 진짜 기준선**이다.
P=0.30% 라 기저 자체도 창별 0.539/0.569/0.518 로 높다(부록 AP).

## 피쳐셋 (HGB 절제 결과 상위)
`BTC동조만(5)` · `기준(69)` · `기준+f154(223)` -- 154 엔지니어링 세트가 HOLDOUT AUC 를
0.5953→0.6066 으로 올린 유일한 축이다. TabICL 은 컬럼 상한 100 이라 TRAIN 전용 상위로 절단.

사용: --model {tabpfn,tabicl} [--null-only --rep K]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, P = 12, 0.0030
EMB = pd.Timedelta(hours=4)
SEEDS = [11, 23, 47]
SUB = 18000
CHUNK, BOOT = 4000, 3000
TABICL_MAX_COL = 100


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def day_ci(v, day, rng, Bt=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (Bt, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def build():
    d = pd.read_parquet(MY / "dataset_v5_f154.parquet").sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    return d, y, okm


def sets_of(d, model):
    from scipy.stats import spearmanr
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    btc = ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign", "dir_up", "T_atr"]
    f154 = [c for c in d.columns if c.startswith("i_")]
    S = {"BTC동조만(5)": btc, "기준(69)": base}
    if model == "tabpfn":
        S[f"기준+f154({len(base)+len(f154)})"] = base + f154
    else:                                    # ⚠️TabICL 컬럼 상한 100 -- TRAIN 전용 선별
        return S, base, f154
    return S, base, f154


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("tabpfn", "tabicl"), required=True)
    ap.add_argument("--null-only", action="store_true")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--nrep", type=int, default=0)
    ap.add_argument("--set", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)
    from sklearn.metrics import roc_auc_score
    d, y, okm = build()
    ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    S, base, f154 = sets_of(d, a.model)
    if a.model == "tabicl":
        from scipy.stats import spearmanr
        tr0 = (sp == "TRAIN") & okm
        sc = []
        for c in f154:
            v = d[c].to_numpy(float)[tr0]; m = np.isfinite(v)
            if m.sum() < 500 or len(np.unique(v[m])) < 3: continue
            r = spearmanr(v[m], y[tr0][m]).statistic
            if np.isfinite(r): sc.append((abs(r), c))
        sc.sort(reverse=True)
        top = [c for _, c in sc[:TABICL_MAX_COL - len(base)]]
        S[f"기준+f154top{len(base)+len(top)}"] = base + top

    def fit_pred(Xtr, ytr, Xte, seed):
        if a.model == "tabpfn":
            from tabpfn import TabPFNClassifier
            c = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=True)
        else:
            from tabicl import TabICLClassifier
            c = TabICLClassifier(device="cpu", n_estimators=2, random_state=seed, verbose=False)
        c.fit(Xtr, ytr.astype(int)); return c.predict_proba(Xte)[:, 1]

    def wf(X, yy_in, shuffle=False, srng=None, seeds=None):
        seeds = seeds or (SEEDS if a.model == "tabpfn" else [20260908])
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = yy_in.copy()
            if shuffle: yy[itr] = srng.permutation(yy[itr])
            ps = []
            for s in seeds:
                sel = itr if len(itr) <= SUB else np.random.default_rng(s).choice(itr, SUB, replace=False)
                ps.append(fit_pred(X[sel], yy[sel], X[te], s))
            pred[te] = np.mean(ps, axis=0)
        return pred

    print(f"모델 {a.model} · 사건 {okm.sum():,} · 돌파율 {y[okm].mean():.4f}")
    print("창별 기저: " + " ".join(
        f"{w} {max(y[okm&(sp==w)].mean(),1-y[okm&(sp==w)].mean()):.4f}" for w in WINS), flush=True)

    if a.null_only:
        cols = S[a.set]
        X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out = []
        for b in range(a.rep, a.rep + a.nrep):
            t0 = time.time()
            p = wf(X, y, True, np.random.default_rng(9000 + b))
            r = {w: float(((p[np.isfinite(p) & (sp == w) & okm] > 0.5).astype(int)
                           == y[np.isfinite(p) & (sp == w) & okm]).mean()) for w in WINS}
            out.append(r); print(f"   셔플{b} {time.time()-t0:.0f}초 " +
                                 " ".join(f"{w[:4]} {r[w]:.4f}" for w in WINS), flush=True)
        json.dump(out, open(MY / f"null_{a.model}_{a.rep}.json", "w"))
        return 0

    accs = {}
    for nm, cols in S.items():
        X = np.nan_to_num(d[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        t0 = time.time(); pred = wf(X, y)
        line = f"{a.model} {nm:>20}({len(cols):>3}) | "
        accs[nm] = {}
        for w in WINS:
            m = np.isfinite(pred) & (sp == w) & okm
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b_ = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                     f"기저{b_:.3f} | ")
            accs[nm][w] = float(acc.mean())
        print(line + f"{time.time()-t0:.0f}초", flush=True)
    best = max(accs, key=lambda k: min(accs[k].values()))
    json.dump({"acc": accs, "best": best}, open(MY / f"h1h_{a.model}.json", "w"), ensure_ascii=False)
    print(f"⭐최고: {best}")
    print(json.dumps({"best": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
