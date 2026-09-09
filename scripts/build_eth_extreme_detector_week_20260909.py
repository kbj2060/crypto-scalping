#!/usr/bin/env python3
"""**극점 탐지기** — 증거신호를 피쳐로 쓴 극점 모델을 최근 일주일에 적용 (2026-09-09, 사용자 지시).

용도가 자동매매가 아니라 **사람이 보는 탐지기**다. 그 용도로는 이 세션 결과가 근거가 된다:
  · 발동 봉이 실제 ±12봉 극점일 확률이 무작위의 3.6~12.4배
  · 극점 타깃 메타라벨 OOS AUC 0.7249 · 상위10% 정밀도 57.7%(기저 24.1%)
⚠️수익성은 별개다 — 이 모델의 상위10%로 매매하면 +2.4~3.9bp 로 비용 여유가 없다.
  **"여기가 국소 극단일 확률"을 주는 것이지 "사거나 팔라"가 아니다.**

모집단  증거신호 8종 중 하나라도 발동한 봉(측면별). 학습 TRAIN ~2026-03-31, 이후는 표본외.
라벨    i+1..i+12 에서 봉 i 의 저점/고점을 깨지 않는가 (피쳐는 봉 i 까지 — 경계 계약 준수)
등급    표본외 점수 분포의 상위 5% / 10% / 25% → 강 / 중 / 약
출력    tmp/eth_signal_map_20260909/extreme_detector_week.csv
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
W, SEEDS = 12, [20260909, 771233, 305610, 517758, 961476]
VAL0 = pd.Timestamp("2026-04-01")


def newest(sym):
    best, bn = None, 0
    for f in glob.glob(str(CACHE / f"{sym}_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()
    kl, btc = newest("ETHUSDT"), newest("BTCUSDT")
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception:
        fund = None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    ts = pd.to_datetime(sig["timestamp"]); n = len(sig)
    hi_ = sig.high.to_numpy(float); lo_ = sig.low.to_numpy(float)
    cl = sig.close.to_numpy(float); atr = sig.atr_pct.to_numpy(float)
    btc_cl = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts)).ffill().to_numpy(float)

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
    # 추세 분위(7일 롤링) — 게이트용. 인과적(봉 i 까지).
    tq = pd.Series(S["ret144"].to_numpy()).rolling(2016, min_periods=500).rank(pct=True).to_numpy()
    for s in B.SIGNALS: S[f"f_{s}"] = 0.0
    FEATS = list(S.columns) + ["n_signals", "is_bottom"]

    rows = []
    for sd, long in (("bottom", True), ("top", False)):
        fires = {s: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS}
        anyf = np.zeros(n, bool); cnt = np.zeros(n, int)
        for s in B.SIGNALS: anyf |= fires[s]; cnt += fires[s].astype(int)
        idx = np.flatnonzero(anyf); idx = idx[idx >= 900]
        X = S.iloc[idx].copy()
        for s in B.SIGNALS: X[f"f_{s}"] = fires[s][idx].astype(float)
        X["n_signals"] = cnt[idx].astype(float); X["is_bottom"] = 1.0 if long else 0.0
        X["_i"] = idx; X["_ts"] = ts.iloc[idx].to_numpy(); X["_long"] = long
        X["_tq"] = tq[idx]
        X["_names"] = [",".join(s for s in B.SIGNALS if fires[s][i]) for i in idx]
        # 라벨: 12봉 뒤까지 있어야 확정. 없으면 미해소(-1)
        y = np.full(len(idx), -1, dtype=int)
        okm = idx + W < n
        if long:
            y[okm] = (np.array([lo_[i + 1:i + 1 + W].min() for i in idx[okm]]) >= lo_[idx[okm]]).astype(int)
        else:
            y[okm] = (np.array([hi_[i + 1:i + 1 + W].max() for i in idx[okm]]) <= hi_[idx[okm]]).astype(int)
        X["_y"] = y
        rows.append(X)
    A = pd.concat(rows, ignore_index=True).sort_values("_ts").reset_index(drop=True)
    A[FEATS] = A[FEATS].replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=FEATS).reset_index(drop=True)

    tr = A[(A._ts < VAL0) & (A._y >= 0)]
    from sklearn.ensemble import HistGradientBoostingClassifier
    P = np.zeros(len(A))
    for sd_ in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                           l2_regularization=1.0, random_state=sd_,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(tr[FEATS], tr._y)
        P += m.predict_proba(A[FEATS])[:, 1] / len(SEEDS)
    A["p"] = P
    last = A._ts.max().normalize()
    win0 = last - pd.Timedelta(days=a.days - 1) - pd.Timedelta(hours=9)   # KST 기준 창 여유
    # 등급 컷: **최근 창을 뺀** 표본외 구간 분포에서 잡는다(순환 방지)
    refm = (A._ts >= VAL0) & (A._ts < win0) & (A._y >= 0)
    cuts = {g: float(A.loc[refm, "p"].quantile(1 - q))
            for g, q in (("강", 0.05), ("중", 0.10), ("약", 0.25))}
    # 🔴추세 게이트(2026-09-09 사용자 지적): 강한 추세 구간(상하 20%)에서는 콜을 억제한다.
    #   근거: 강한상승 천장 콜은 정밀도 51.6% 인데 **적중 +10.8 / 빗나감 −60.7bp** 로 완전 비대칭.
    #   표본외 161일: 게이트 없음 −3.36bp → 중립 구간만 +2.27bp (4.64→2.93건/일).
    A["bucket"] = np.where(A._tq >= 0.80, "강한상승", np.where(A._tq <= 0.20, "강한하락", "중립"))
    A["gated"] = A.bucket != "중립"
    A["grade"] = np.where(A.p >= cuts["강"], "강",
                  np.where(A.p >= cuts["중"], "중",
                   np.where(A.p >= cuts["약"], "약", "-")))
    ref = A[refm]                     # ⚠️grade 부여 **뒤에** 잘라야 컬럼이 따라온다
    print(f"모집단 {len(A):,} · 학습 {len(tr):,}(~{VAL0:%Y-%m-%d}) · 등급컷 기준 {len(ref):,}건 "
          f"({ref._ts.min():%m-%d}~{ref._ts.max():%m-%d})")
    print(f"등급 임계  강 {cuts['강']:.4f} · 중 {cuts['중']:.4f} · 약 {cuts['약']:.4f}\n")
    print("■ 등급별 실측 정밀도 (표본외, 최근 창 제외 — 이 숫자가 마크의 뜻이다)")
    print(f"{'등급':>6}{'건수':>8}{'건/일':>8}{'정밀도':>9}{'기저대비':>10}")
    dref = (ref._ts.max() - ref._ts.min()).total_seconds() / 86400
    base = ref._y.mean()
    prec = {}
    for g in ("강", "중", "약", "-"):
        q = ref[(ref.grade == g) & (~ref.gated)]
        if not len(q): continue
        prec[g] = float(q._y.mean())
        print(f"{g:>6}{len(q):>8}{len(q)/dref:>8.2f}{q._y.mean()*100:>8.1f}%{(q._y.mean()-base)*100:>+9.1f}pp")
    print(f"{'전건':>6}{len(ref):>8}{len(ref)/dref:>8.2f}{base*100:>8.1f}%")
    gq = ref[ref.gated & (ref.grade != "-")]
    print(f"  (게이트로 억제된 등급 콜 {len(gq)}건 · 정밀도 {gq._y.mean()*100:.1f}%"
          f" — 정밀도로는 안 보이지만 손익이 −3.36 → +2.27bp 로 갈린다)")
    W_ = A[A._ts >= win0].copy()
    W_.to_csv(OUT / "extreme_detector_week.csv", index=False)
    json.dump({"cuts": cuts, "precision": prec, "base": float(base)},
              open(OUT / "extreme_detector_meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n최근 {a.days}일 후보 {len(W_)}건 · 강 {int((W_.grade=='강').sum())} "
          f"· 중 {int((W_.grade=='중').sum())} · 약 {int((W_.grade=='약').sum())}")
    print(json.dumps({"done": True, "n_week": len(W_)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
