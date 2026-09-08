#!/usr/bin/env python3
"""H=1h·±0.30% **커버리지별 B=32 셔플 귀무** (HGB, 2026-09-08).

전건(커버 100%)은 이미 통과했다: 세 창 p=0.030, 관측−셔플평균 +3.5~10.3σ.
남은 질문은 **커버리지를 줄인 지점도 통과하는가**다. 지금까지는 "부분집합 기저"와 비교했는데
정확한 기준선은 **같은 절차를 라벨 셔플에 적용한 분포**다.

## 절차 (관측과 완전히 동일하게 셔플에도 적용)
1. TRAIN 앞 70% 로 적합 → TRAIN 뒤 30% 예측의 |p−0.5| 분위로 **임계값** 산정
2. 워크포워드 예측에 그 임계값을 적용해 상위 c% 만 채점
3. 셔플 복제마다 **1~2를 통째로 다시** 한다(임계값도 셔플 모델에서 재산정)
   -- 임계값을 관측 것으로 고정하면 셔플 모델의 확신 분포가 달라 커버리지가 어긋난다.
실현 커버리지를 반드시 병기한다.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "3"
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, P, NB = 12, 0.0030, 32
COV = (1.0, 0.5, 0.3, 0.2)
EMB = pd.Timedelta(hours=4)
CHUNK, SEED = 4000, 20260908


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


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
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
    ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = d[base].to_numpy(np.float32)
    trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
    tr_fit = trm & (ts <= cut).to_numpy(); tr_hold = trm & (ts > cut).to_numpy()

    def mk(shuffle, rs):
        """(워크포워드 예측, TRAIN 뒤30% 확신분포) -- 셔플이면 라벨을 섞어 둘 다 재생성."""
        yy_all = y.copy()
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = y.copy()
            if shuffle: yy[itr] = rs.permutation(yy[itr])
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=SEED)
            c.fit(X[itr], yy[itr]); pred[te] = c.predict_proba(X[te])[:, 1]
        yf = y.copy()
        if shuffle: yf[np.flatnonzero(tr_fit)] = rs.permutation(yf[np.flatnonzero(tr_fit)])
        c0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.15, random_state=SEED)
        c0.fit(X[tr_fit], yf[tr_fit])
        return pred, np.abs(c0.predict_proba(X[tr_hold])[:, 1] - 0.5)

    def score(pred, conf_tr):
        out = {}
        for cv in COV:
            thr = 0.0 if cv >= 1.0 else float(np.quantile(conf_tr, 1 - cv))
            for w in WINS:
                m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred - 0.5) >= thr)
                mw = np.isfinite(pred) & (sp == w) & okm
                out[(cv, w)] = (float(((pred[m] > 0.5).astype(int) == y[m]).mean()) if m.sum() >= 30
                                else np.nan, float(m.sum() / max(mw.sum(), 1)), int(m.sum()))
        return out

    t0 = time.time(); p0, c0 = mk(False, None); O = score(p0, c0)
    print(f"관측 {time.time()-t0:.0f}초", flush=True)
    NUL = {k: [] for k in O}
    for b in range(NB):
        pb, cb = mk(True, np.random.default_rng(6000 + b))
        S = score(pb, cb)
        for k in O: NUL[k].append(S[k][0])
        if (b + 1) % 8 == 0: print(f"   셔플 {b+1}/{NB}", flush=True)
    print("\n" + "=" * 104)
    print(f"{'커버':>6}{'창':>15} {'관측':>8}{'실현커버':>9}{'n':>6} | {'셔플평균':>9}{'sd':>7}{'max':>8}"
          f"{'p':>7}{'σ':>7}")
    print("=" * 104)
    res = []
    for cv in COV:
        for w in WINS:
            o, rc, n = O[(cv, w)]
            a = np.array([x for x in NUL[(cv, w)] if np.isfinite(x)])
            if not np.isfinite(o) or len(a) < 10:
                print(f"{cv:>5.0%}{w:>15} {'n부족':>8}"); continue
            ge = (a >= o).sum(); p = (ge + 1) / (len(a) + 1)
            print(f"{cv:>5.0%}{w:>15} {o:>8.4f}{rc:>9.0%}{n:>6} | {a.mean():>9.4f}{a.std():>7.4f}"
                  f"{a.max():>8.4f}{p:>7.3f}{(o-a.mean())/max(a.std(),1e-9):>+7.1f} "
                  f"{'✅' if p < 0.05 else '❌'}")
            res.append(dict(cov=cv, win=w, obs=o, real_cov=rc, n=n, null_mean=float(a.mean()),
                            null_sd=float(a.std()), null_max=float(a.max()), p=float(p)))
    R = pd.DataFrame(res); R.to_csv(MY / "h1h_null32_coverage.csv", index=False)
    ok = R.groupby("cov")["p"].apply(lambda s: (s < 0.05).all())
    print(f"\n⭐세 창 모두 p<0.05 인 커버리지: {[f'{c:.0%}' for c in ok[ok].index.tolist()]}")
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
