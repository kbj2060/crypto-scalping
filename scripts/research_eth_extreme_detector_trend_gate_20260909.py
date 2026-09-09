#!/usr/bin/env python3
"""극점 탐지기의 **역추세 약점** 진단과 추세 게이트 (2026-09-09, 사용자 지적).

사용자: *"되돌림엔 강한데 돌파엔 너무 약하다. 9/3, 9/6처럼 돌파할 때 계속 반대 포지션
        신호를 주니까 손실이 커진다."*

검정
  A. 추세 버킷 × 측면별 등급 정밀도 — 역추세 콜이 실제로 나쁜가, 얼마나 나쁜가
  B. 모델이 추세를 이미 아는가 — 역추세 구간에서 **발동 빈도**가 줄어드는가
  C. 추세 게이트 — 강한 추세 구간의 역추세 콜을 죽이면 정밀도/커버리지가 어떻게 되나
  D. 대안: 게이트 대신 **추세 상호작용 피쳐 추가** 재학습이 더 나은가

추세 정의(전부 인과적, 봉 i 까지): ret144 = (close/close[-144] − 1)/atr, 7일 롤링 분위.
  상위 20% = 강한 상승 · 하위 20% = 강한 하락 · 나머지 = 중립.
역추세 = 강한 상승에서 천장 콜 · 강한 하락에서 바닥 콜.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import live_evidence_signal_dashboard_20260823 as EV
import build_eth_anchor_label_dataset_20260907 as B

OUT = ROOT / "tmp/eth_signal_map_20260909"; CACHE = OUT / "klcache"
W, H_EVAL, COST = 12, 48, 10.0
SEEDS = [20260909, 771233, 305610, 517758, 961476]
VAL0 = pd.Timestamp("2026-04-01")


def newest(sym):
    best, bn = None, 0
    for f in glob.glob(str(CACHE / f"{sym}_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--q", type=float, default=0.20)
    a = ap.parse_args()
    kl, btc = newest("ETHUSDT"), newest("BTCUSDT")
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception:
        fund = None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    ts = pd.to_datetime(sig["timestamp"]); n = len(sig)
    op = sig.open.to_numpy(float); hi_ = sig.high.to_numpy(float); lo_ = sig.low.to_numpy(float)
    cl = sig.close.to_numpy(float); atr = sig.atr_pct.to_numpy(float)
    btc_cl = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts)).ffill().to_numpy(float)
    lo, hi = 900, n - H_EVAL - W - 3

    S = pd.DataFrame(index=range(n))
    for c in ("p_fast", "p_slow", "delta_z", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
              "ret3_z", "atr_pct", "dem", "kalman_dev_z"):
        S[c] = sig[c].to_numpy(float)
    c_s = pd.Series(cl)
    S["atr_pctile"] = pd.Series(atr).rolling(2016, min_periods=500).rank(pct=True).to_numpy()
    for w in (12, 48, 144):
        S[f"ret{w}"] = (c_s / c_s.shift(w) - 1).to_numpy() / np.maximum(atr, 1e-9)
        rmin = pd.Series(lo_).rolling(w).min().to_numpy(); rmax = pd.Series(hi_).rolling(w).max().to_numpy()
        S[f"pos_in_range{w}"] = (cl - rmin) / np.maximum(rmax - rmin, 1e-9)
        S[f"dist_lo{w}_atr"] = (cl - rmin) / np.maximum(cl * atr, 1e-9)
        S[f"dist_hi{w}_atr"] = (rmax - cl) / np.maximum(cl * atr, 1e-9)
    b_s = pd.Series(btc_cl)
    S["btc_ret12"] = (b_s / b_s.shift(12) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["btc_ret48"] = (b_s / b_s.shift(48) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["eth_btc_div"] = S["ret12"] - S["btc_ret12"]
    S["hour"] = ts.dt.hour.to_numpy(); S["weekday"] = ts.dt.weekday.to_numpy()
    for s in B.SIGNALS: S[f"f_{s}"] = 0.0
    FEATS = list(S.columns) + ["n_signals", "is_bottom"]
    tq = pd.Series(S["ret144"].to_numpy()).rolling(2016, min_periods=500).rank(pct=True).to_numpy()

    rows = []
    for sd, long in (("bottom", True), ("top", False)):
        fires = {s: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS}
        anyf = np.zeros(n, bool); cnt = np.zeros(n, int)
        for s in B.SIGNALS: anyf |= fires[s]; cnt += fires[s].astype(int)
        idx = np.flatnonzero(anyf); idx = idx[(idx >= lo) & (idx <= hi)]
        X = S.iloc[idx].copy()
        for s in B.SIGNALS: X[f"f_{s}"] = fires[s][idx].astype(float)
        X["n_signals"] = cnt[idx].astype(float); X["is_bottom"] = 1.0 if long else 0.0
        X["_i"] = idx; X["_ts"] = ts.iloc[idx].to_numpy(); X["_long"] = long; X["_tq"] = tq[idx]
        fe = np.array([lo_[i + 1:i + 1 + W].min() if long else hi_[i + 1:i + 1 + W].max() for i in idx])
        X["_y"] = ((fe >= lo_[idx]) if long else (fe <= hi_[idx])).astype(int)
        rows.append(X)
    A = pd.concat(rows, ignore_index=True).sort_values("_ts").reset_index(drop=True)
    A[FEATS] = A[FEATS].replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=FEATS + ["_tq"]).reset_index(drop=True)
    # 추세 상호작용 피쳐(팔 D 용)
    A["trend_q"] = A._tq
    A["against"] = np.where(A._long, 1 - A._tq, A._tq)          # 1에 가까울수록 역추세
    A["trend_x_side"] = A._tq * (A._long.astype(float) * 2 - 1)
    XFE = FEATS + ["trend_q", "against", "trend_x_side"]

    tr = A[A._ts < VAL0]; oo = A[A._ts >= VAL0].reset_index(drop=True)
    days = (oo._ts.max() - oo._ts.min()).total_seconds() / 86400
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def fit(feats):
        P = np.zeros(len(oo))
        for sd_ in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                               l2_regularization=1.0, random_state=sd_,
                                               early_stopping=True, validation_fraction=0.15)
            m.fit(tr[feats], tr._y); P += m.predict_proba(oo[feats])[:, 1] / len(SEEDS)
        return P
    oo["p"] = fit(FEATS)
    cuts = {g: float(oo.p.quantile(1 - q)) for g, q in (("강", 0.05), ("중", 0.10))}
    oo["grade"] = np.where(oo.p >= cuts["강"], "강", np.where(oo.p >= cuts["중"], "중", "-"))
    oo["bucket"] = np.where(oo._tq >= 1 - a.q, "강한상승",
                    np.where(oo._tq <= a.q, "강한하락", "중립"))
    oo["ctr"] = ((oo._long & (oo.bucket == "강한하락")) | ((~oo._long) & (oo.bucket == "강한상승")))

    def netbp(d):
        i = d._i.to_numpy(); e = op[i + 1]; x = cl[i + H_EVAL]; r = (x - e) / e * 1e4
        return np.where(d._long.to_numpy(), r, -r) - COST
    print(f"OOS {len(oo):,}건 ({days:.0f}일) · AUC {roc_auc_score(oo._y, oo.p):.4f}\n" + "=" * 104)
    print("A. 추세 버킷 × 측면 — 등급 강+중 (하루 4.6건)의 정밀도·손익")
    print(f"{'추세':<10}{'측면':>6}{'성격':>8}{'건수':>7}{'건/일':>7}{'정밀도':>9}{'순bp':>9}")
    top = oo[oo.grade != "-"]
    for bkt in ("강한상승", "중립", "강한하락"):
        for long, lab in ((True, "바닥"), (False, "천장")):
            q = top[(top.bucket == bkt) & (top._long == long)]
            if len(q) < 20: continue
            kind = "역추세" if ((long and bkt == "강한하락") or ((not long) and bkt == "강한상승")) \
                else ("순추세" if bkt != "중립" else "중립")
            print(f"{bkt:<10}{lab:>6}{kind:>8}{len(q):>7}{len(q)/days:>7.2f}"
                  f"{q._y.mean()*100:>8.1f}%{netbp(q).mean():>9.2f}")
    print(f"\n  역추세 콜  {int(top.ctr.sum())}건 정밀도 {top[top.ctr]._y.mean()*100:.1f}% "
          f"· 순 {netbp(top[top.ctr]).mean():+.2f}bp")
    print(f"  그 외      {int((~top.ctr).sum())}건 정밀도 {top[~top.ctr]._y.mean()*100:.1f}% "
          f"· 순 {netbp(top[~top.ctr]).mean():+.2f}bp")

    print("\nB. 모델이 추세를 이미 아는가 — 버킷별 **발동 비중**(모집단 대비)")
    for bkt in ("강한상승", "중립", "강한하락"):
        pop = oo[oo.bucket == bkt]
        t = top[top.bucket == bkt]
        ctr_pop = pop[((pop._long) & (bkt == "강한하락")) | ((~pop._long) & (bkt == "강한상승"))]
        ctr_top = t[((t._long) & (bkt == "강한하락")) | ((~t._long) & (bkt == "강한상승"))]
        if len(pop) < 50: continue
        print(f"  {bkt:<8} 모집단 {len(pop):>6} 중 역추세 {len(ctr_pop):>5}({len(ctr_pop)/len(pop)*100:>4.1f}%)"
              f" → 등급부여 {len(t):>4} 중 역추세 {len(ctr_top):>4}({len(ctr_top)/max(len(t),1)*100:>4.1f}%)")

    print("\nC. 추세 게이트 — 강한 추세 구간의 역추세 콜을 제거")
    for lab, q in (("게이트 없음(현행)", top), ("역추세 제거", top[~top.ctr])):
        print(f"  {lab:<18}{len(q):>6}건 {len(q)/days:>5.2f}건/일 · 정밀도 {q._y.mean()*100:>5.1f}%"
              f" · 순 {netbp(q).mean():+6.2f}bp")
    print("\nD. 추세 상호작용 피쳐를 넣고 재학습")
    oo["p2"] = fit(XFE)
    c2 = {g: float(oo.p2.quantile(1 - q)) for g, q in (("강", 0.05), ("중", 0.10))}
    t2 = oo[oo.p2 >= c2["중"]]
    ctr2 = ((t2._long & (t2.bucket == "강한하락")) | ((~t2._long) & (t2.bucket == "강한상승")))
    print(f"  AUC {roc_auc_score(oo._y, oo.p2):.4f} (기존 {roc_auc_score(oo._y, oo.p):.4f})"
          f" · 등급 {len(t2)}건 정밀도 {t2._y.mean()*100:.1f}% · 순 {netbp(t2).mean():+.2f}bp"
          f" · 역추세 비중 {ctr2.mean()*100:.1f}% (기존 {top.ctr.mean()*100:.1f}%)")
    oo.to_csv(OUT / "trend_gate_oos.csv", index=False)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
