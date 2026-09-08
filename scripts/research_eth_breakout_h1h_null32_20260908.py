#!/usr/bin/env python3
"""H=1h·±0.30% **B=32 셔플 귀무** (HGB, 2026-09-08).

TabPFN `기준(69)` 이 이 라벨에서 세 창 모두 기저를 넘었다(VAL .5654/OOS .6010/HOLD .5554,
기저 .539/.569/.518). 그러나 **기저는 진짜 기준선이 아니다** -- H=48 실측에서 셔플 평균이
0.5163/0.5003/0.5132 로 0.50 이 아니었다(일군집 구조).
이 라벨은 기저가 더 높아(.539/.569/.518) **셔플 분포도 그만큼 위에 있을 수 있다.**
HGB 는 폴드당 수초라 B=32 를 빨리 낼 수 있다 -- TabPFN 귀무(GPU) 전에 **셔플 분포의 위치**를
먼저 확정한다. 같은 라벨·같은 피쳐·같은 워크포워드다.
⚠️스레드 제한: 앞서 5개 병렬 TabICL 이 load 32/12코어를 만들어 전부 5배 느려졌다.
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
    from sklearn.metrics import roc_auc_score
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

    def run(shuffle, rs):
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
        return pred

    t0 = time.time(); p0 = run(False, None)
    obs = {}
    for w in WINS:
        m = np.isfinite(p0) & (sp == w) & okm
        obs[w] = dict(acc=float(((p0[m] > 0.5).astype(int) == y[m]).mean()),
                      auc=float(roc_auc_score(y[m], p0[m])),
                      base=float(max(y[m].mean(), 1 - y[m].mean())), n=int(m.sum()))
    print(f"관측(HGB 기준69) {time.time()-t0:.0f}초 · " + " | ".join(
        f"{w[:4]} {obs[w]['acc']:.4f} A{obs[w]['auc']:.3f} 기저{obs[w]['base']:.3f}" for w in WINS),
        flush=True)
    nl = {w: [] for w in WINS}; na = {w: [] for w in WINS}
    for b in range(NB):
        t1 = time.time(); pb = run(True, np.random.default_rng(4000 + b))
        for w in WINS:
            m = np.isfinite(pb) & (sp == w) & okm
            nl[w].append(float(((pb[m] > 0.5).astype(int) == y[m]).mean()))
            na[w].append(float(roc_auc_score(y[m], pb[m])))
        print(f"   셔플{b+1}/{NB} {time.time()-t1:.0f}초 " +
              " ".join(f"{w[:4]} {nl[w][-1]:.4f}" for w in WINS), flush=True)
        json.dump({"obs": obs, "null_acc": nl, "null_auc": na},
                  open(MY / "h1h_null32_hgb.json", "w"), ensure_ascii=False)
    print("\n" + "=" * 96)
    for w in WINS:
        a = np.array(nl[w]); ge = (a >= obs[w]["acc"]).sum()
        au = np.array(na[w])
        print(f"{w:>14} 정확도 관측 {obs[w]['acc']:.4f} · 셔플평균 {a.mean():.4f} sd {a.std():.4f} "
              f"max {a.max():.4f} · p={(ge+1)/(NB+1):.3f} {'✅' if (ge+1)/(NB+1)<0.05 else '❌'}")
        print(f"{'':>14} AUC   관측 {obs[w]['auc']:.4f} · 셔플평균 {au.mean():.4f} "
              f"max {au.max():.4f} · p={((au>=obs[w]['auc']).sum()+1)/(NB+1):.3f}")
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
