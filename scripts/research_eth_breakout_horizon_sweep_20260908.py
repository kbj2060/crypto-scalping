#!/usr/bin/env python3
"""돌파/되돌림 **호라이즌 스윕** -- HGB (2026-09-08).

사용자: *"hgb로 호라이즌 스윕도 테스트해줘"*

지금까지는 H=48봉(4시간) 고정이었다. 피쳐는 전부 트리거 분 `s1-1` 기준이라 **H 와 무관**하므로
라벨만 다시 만들면 된다. 결정 시점·기준가·피쳐는 그대로 두고 **청산 지평만** 바꾼다.

격자: H ∈ {6,12,24,48,96,144}봉 = 30분/1h/2h/4h/8h/12h  ×  P ∈ {0.25%, 0.50%}
라벨: 트리거 레벨에서 ±P 첫터치(1분봉), 미터치는 H봉 뒤 종가 부호 → **커버리지 100%**
⚠️엠바고를 **H 이상**으로 스케일한다(라벨 창이 테스트 구간을 침범하지 않도록). 기존 4h 고정은
   H=48 에 딱 맞춘 값이었고 H 를 늘리면 부족해진다.
같이 낸다: 해소율 · 기저(다수결) · 라벨 셔플 대조군 · 모델 없는 BTC 규칙.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (6, 12, 24, 48, 96, 144)
P_GRID = (0.0025, 0.0050)
CHUNK = 4000
SEED = 20260908
BOOT = 3000


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


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, emb, shuffle=False, rng=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - emb).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        yy = y.copy()
        if shuffle: yy[np.flatnonzero(tr)] = rng.permutation(yy[np.flatnonzero(tr)])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(MY / "dataset_v4.parquet")
    d = d.sort_values("timestamp").reset_index(drop=True)
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
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = d[base].to_numpy(np.float32)
    ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    ss = d["v2_mv_same_sign"].to_numpy(float)
    print(f"사건 {len(d):,} · 피쳐 {len(base)}\n" + "=" * 124, flush=True)
    print(f"{'H':>6}{'배리어':>7}{'해소율':>7} | " + " | ".join(f"{w[:8]:>30}" for w in WINS)
          + " | BTC규칙(되돌림율,커버)")
    print("=" * 124)
    big = 1 << 30; rows = []
    for H in H_GRID:
        emb = max(pd.Timedelta(hours=4), pd.Timedelta(minutes=int(H * 5)))
        okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5))
        for Pv in P_GRID:
            tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0),
                                 entry * (1 + Pv), entry * (1 - Pv), H * 5)
            uo = tu >= 0; do_ = td >= 0
            au = np.where(uo, tu, big); ad = np.where(do_, td, big)
            upf = uo & (au < ad); dnf = do_ & (ad < au)
            cont = np.where(sgn > 0, upf, dnf); rev = np.where(sgn > 0, dnf, upf)
            x5 = np.minimum(bt + H, len(C5) - 1)
            clo = (C5[x5] - entry) / entry * 1e4 * sgn
            y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
            res = (cont | rev)
            pred = wf(X, y, ts, months, uniq, emb)
            line = f"{H:>4}봉{'±'+str(Pv*100)+'%':>7}{res[okm].mean():>7.3f} | "
            rec = dict(H=H, P=Pv, resolve=float(res[okm].mean()))
            for w in WINS:
                m = np.isfinite(pred) & (sp == w) & okm
                if m.sum() < 100: line += f"{'--':>30} | "; continue
                acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
                lo, hi = day_ci(acc, day[m], rng)
                b_ = max(y[m].mean(), 1 - y[m].mean())
                mark = "✅" if lo > b_ else " "
                line += (f"{acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                         f"기저{b_:.3f}{mark} | ")
                rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo; rec[f"{w}_base"] = float(b_)
            # 모델 없는 BTC 규칙 (표본외 3창 합산)
            mb = okm & np.isin(sp, WINS) & (ss == 0)
            if mb.sum() > 100:
                line += f"{1-y[mb].mean():.3f}, {mb.sum()/max((okm&np.isin(sp,WINS)).sum(),1):.0%}"
                rec["btc_rev"] = float(1 - y[mb].mean())
            print(line, flush=True)
            rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(MY / "horizon_sweep.csv", index=False)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS) for i in R.index]
    print("\n" + "=" * 124)
    print(f"⭐세 창 모두 CI 하한 > 기저: {int(R.pass3.sum())}/{len(R)}")
    if R.pass3.any():
        print(R[R.pass3][["H", "P", "resolve"] + [f"{w}_acc" for w in WINS]].round(4).to_string(index=False))
    R["m"] = R[[f"{w}_acc" for w in WINS]].min(1) - R[[f"{w}_base" for w in WINS]].max(1)
    print("\n=== 세 창 최소 초과정확도 상위 5 ===")
    print(R.sort_values("m", ascending=False).head(5)
          [["H", "P", "resolve"] + [f"{w}_acc" for w in WINS] + ["btc_rev", "m"]].round(4).to_string(index=False))
    print(json.dumps({"cells": len(R), "pass3": int(R.pass3.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
