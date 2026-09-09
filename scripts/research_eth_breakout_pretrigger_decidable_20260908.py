#!/usr/bin/env python3
"""트리거 **전에** 돌파/되돌림을 결정할 수 있는가 -- 메이커 구조의 성립 조건 (2026-09-08).

## 왜 이게 관건인가
메이커 진입은 **트리거 레벨에 지정가를 미리 걸어두는 것**이다(실측: 99.3% 관통 → 체결 가능).
그런데 레벨에 걸린 주문은 발현의 **반대편**에 서게 된다 = 되돌림 포지션.
  · 모델이 되돌림이면 그대로 보유 → 메이커 진입 ✅
  · 모델이 돌파면 반대 포지션 → 청산+역전(테이커 2회) ❌
그런데 현행 모델은 **트리거가 나야** 점수를 낸다(경로 피쳐 v2_mv_* 가 트리거 분을 필요로 함).
즉 주문을 걸 시점엔 판정이 없다. **앵커 봉 정보만으로 돌파/되돌림을 맞힐 수 있는가**가
메이커 구조 성립의 필요조건이다.

## 팔 3개
  anchor45   앵커 봉 `ba` 시점 피쳐만 (방향·경로 없음) -- 실제로 주문 걸 때 쓸 수 있는 정보
  bt1_nodir  트리거 직전 봉 `bt-1` 피쳐이되 방향·경로 제외 -- 정보 **시점**의 효과를 분리
  full69     현행 모델 (참조 상한)
⚠️ anchor45 는 발현 방향을 모른다. 라벨은 발현 방향 기준이므로 이 팔은
   "방향과 무관하게 계속 갈까 되돌아올까"를 묻는 셈이다 -- 대칭 문제라 학습 가능성은 있다.
   셔플 귀무로 판정한다.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B     # noqa: E402
import live_eth_breakout_features_20260908 as LF        # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, TM, P, NB = 12, 0.75, 0.0025, 32
EMB = pd.Timedelta(hours=4)
SEED, CHUNK = 20260908, 4000
COV = (1.0, 0.5, 0.3, 0.2)


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(axis=1); ad = hd.any(axis=1)
        tu[a:b] = np.where(au, hu.argmax(axis=1), -1); td[a:b] = np.where(ad, hd.argmax(axis=1), -1)
    return tu, td


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float); TB = eth["taker_buy_base"].to_numpy(float)
    btc = B._load_kl(B.BTC_KL)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)

    F, LV, atr5 = LF.bar_features(C5, H5, L5, V5, TB, bt5)
    mz = np.load(ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz", allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(mz["ts"]))
    ei = list(np.load(ROOT / "tmp/xsec_perp_screen_20260908/panel.npz",
                      allow_pickle=True)["syms"]).index("ETHUSDT")
    met = {"retail": mz["count_long_short_ratio"][:, ei], "ttc": mz["count_toptrader_long_short_ratio"][:, ei],
           "ttp": mz["sum_toptrader_long_short_ratio"][:, ei], "tkv": mz["sum_taker_long_short_vol_ratio"][:, ei]}
    XS = LF.metric_features(met, xts, ts5)

    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    ba = d["bar_idx"].to_numpy()                       # 앵커 봉
    T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(ba + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1) & (ba >= 900)
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))

    # ---- 팔별 피쳐 행렬 ----
    fk = sorted(F.keys()); xk = sorted(XS.keys())
    sigc = [f"sig_{s}" for s in ("sweep", "smt", "taker", "kal", "strz", "orth", "fib", "dem")]
    stat = d[sigc + ["n_signals", "side_bottom", "atr_at_anchor"]].to_numpy(np.float32)

    def at(bar_idx):
        cols = [F[k][bar_idx] for k in fk] + [XS[k][bar_idx] for k in xk]
        hh = np.stack(cols, axis=1).astype(np.float32)
        hr = pd.to_datetime(ts5[bar_idx])
        extra = np.stack([hr.hour.to_numpy(), hr.dayofweek.to_numpy(), T], axis=1).astype(np.float32)
        return np.hstack([hh, extra, stat])

    X_anchor = at(np.clip(ba, 0, len(C5) - 1))
    X_bt1 = at(np.clip(bt - 1, 0, len(C5) - 1))
    full = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X_full = d[full].to_numpy(np.float32)
    ARMS = {f"anchor{X_anchor.shape[1]}(주문 시점)": X_anchor,
            f"bt1_nodir{X_bt1.shape[1]}(시점만 늦춤)": X_bt1,
            f"full{X_full.shape[1]}(현행)": X_full}

    ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
    tr_fit = np.flatnonzero(trm & (ts <= cut).to_numpy()); tr_hold = trm & (ts > cut).to_numpy()
    print(f"사건 {int(okm.sum()):,} · 돌파율 {y[okm].mean():.4f}\n", flush=True)

    def run(X, shuf=False, rs=None):
        Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = y.copy()
            if shuf: yy[itr] = rs.permutation(yy[itr])
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=SEED)
            pred[te] = c.fit(Xc[itr], yy[itr]).predict_proba(Xc[te])[:, 1]
        yf = y.copy()
        if shuf: yf[tr_fit] = rs.permutation(yf[tr_fit])
        c0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.15, random_state=SEED)
        conf = np.abs(c0.fit(Xc[tr_fit], yf[tr_fit]).predict_proba(Xc[tr_hold])[:, 1] - 0.5)
        return pred, conf

    def score(pred, conf):
        o = {}
        for cv in COV:
            thr = 0.0 if cv >= 1.0 else float(np.quantile(conf, 1 - cv))
            for w in WINS:
                m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred - 0.5) >= thr)
                o[(cv, w)] = float(((pred[m] > 0.5).astype(int) == y[m]).mean()) if m.sum() >= 30 else np.nan
        return o

    print("=" * 108)
    for nm, X in ARMS.items():
        O = score(*run(X))
        NUL = {k: [] for k in O}
        for b in range(NB):
            S = score(*run(X, True, np.random.default_rng(7700 + b)))
            for k in O: NUL[k].append(S[k])
        print(f"### {nm}")
        for cv in COV:
            cells, ps = [], []
            for w in WINS:
                o = O[(cv, w)]; a = np.array([x for x in NUL[(cv, w)] if np.isfinite(x)])
                p_ = ((a >= o).sum() + 1) / (len(a) + 1)
                ps.append(p_)
                cells.append(f"{w[:4]} {o:.4f}({(o-a.mean())*100:+5.2f})p{p_:.3f}")
            ok3 = all(x < 0.05 for x in ps)
            print(f"   커버{int(cv*100):>4}% | " + " ".join(cells) + ("  ✅" if ok3 else "  ❌"))
        print(flush=True)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
