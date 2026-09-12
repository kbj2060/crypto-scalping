"""MFE/MAE 회귀 — 저변동 배제 + 예측 R 게이팅. 진입·TP·SL 이 한 예측에서 나온다.

숏의 MFE 는 롱의 |MAE| 이므로 R_숏 = 1/R_롱 이다. 모델이 (MFE, |MAE|) 둘만 예측하면
방향(둘 중 큰 쪽)·TP(큰 값)·SL(작은 값)이 동시에 정해진다 — 진입과 청산의 공동 학습.

판정 순서: ① 예측 R ↔ 실제 R 스피어만 ② 예측 R 십분위별 실제 R 단조성
③ 그 다음에 순손익. ①②가 없으면 ③의 양수는 표본 잡음이다.
비용은 실측 비대칭(익절=peg resting 5.52bp / 손절=taker 7.79bp / 시간청산 7.79bp).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from core.event_label_engine import (atr_volatility, combine_sample_weights,  # noqa: E402
                                     return_attribution_weights, sample_uniqueness_weights)

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
H, ATR_WIN = 12, 96                       # 1시간 지평 · ATR 창
C_TP, C_SL = 0.000552, 0.000779           # 실측 메이커: 익절 resting / 손절 taker
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
ATR_KEEP = 0.70                           # 저변동 배제: ATR 상위 30% (상위 10%는 시간에 뭉쳐 3개월이 표본부족)
GATES = [0.10, 0.25, 0.50, 1.00]          # 예측 R 상위 비율


def _fwd(close, opn, slip):
    """앞 H봉 롱 기준 MFE/MAE + 시간청산 수익."""
    n = len(close)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for i in range(n - 2):
        E = opn[i + 1] * (1 + slip)
        seg = close[i + 1:min(i + 1 + H, n)]
        if len(seg) < 2:
            continue
        r = (seg * (1 - slip) - E) / E
        mfe[i], mae[i] = float(r.max()), float(r.min())
    return mfe, np.abs(mae)


def _pnl(close, opn, slip, idx, side, tp, sl):
    """예측 배리어로 실제 경로를 태워 건별 순손익(비용 차감)."""
    out = np.empty(len(idx))
    n = len(close)
    for j, i in enumerate(idx):
        s = side[j]
        E = opn[i + 1] * (1 + slip if s > 0 else 1 - slip)
        seg = close[i + 1:min(i + 1 + H, n)]
        r = ((seg * (1 - slip) - E) / E) if s > 0 else ((E - seg * (1 + slip)) / E)
        w = np.flatnonzero(r >= tp[j])
        l = np.flatnonzero(r <= -sl[j])
        iw = int(w[0]) if len(w) else 1 << 30
        il = int(l[0]) if len(l) else 1 << 30
        if iw < il:
            out[j] = tp[j] - C_TP
        elif il < iw:
            out[j] = -sl[j] - C_SL
        else:
            out[j] = float(r[-1]) - C_SL
    return out


def main() -> int:
    _fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    n = len(frame)
    t = pd.to_datetime(frame["timestamp"])
    mon = t.dt.to_period("M").astype(str).to_numpy()
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)

    mfe, mae = _fwd(close, opn, slip)
    atr = atr_volatility(frame["high"], frame["low"], frame["close"], window=ATR_WIN).to_numpy()
    ok = np.isfinite(mfe) & np.isfinite(mae) & (mae > 1e-6) & (mfe > 0) & np.isfinite(atr)
    R = np.where(ok, np.maximum(mfe, mae) / np.minimum(mfe, mae), np.nan)
    print(f"[모집단] ATR({ATR_WIN}) 상위 {(1-ATR_KEEP)*100:.0f}% · 임계값은 **학습구간에서만** 산출(인과) "
          f"· 전체 유효 {int(ok.sum()):,}봉 · 실제 R 중앙 {np.nanmedian(R):.3f}", flush=True)
    idx_all = np.arange(n)
    t1 = np.minimum(idx_all + H, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx_all, t1, n),
                                 return_attribution_weights(close, idx_all, t1))

    rows = []
    for m in TEST_MONTHS:
        te_i = int(np.argmax(mon == m))
        pre = ok.copy()
        pre[max(te_i - H, 0):] = False
        if pre.sum() < 500:
            print(f"  {m} 학습구간 부족")
            continue
        thr = float(np.nanquantile(atr[pre], ATR_KEEP))    # 인과: 과거만으로 컷 결정
        tr = pre & (atr >= thr)
        te = ok & (mon == m) & (atr >= thr)
        if tr.sum() < 1500 or te.sum() < 150:
            print(f"  {m} 표본 부족 (컷 {thr*100:.3f}% · 학습 {int(tr.sum())} / 검증 {int(te.sum())})")
            continue
        ti, ei = np.flatnonzero(tr), np.flatnonzero(te)
        for seed in SEEDS:
            ps = {}
            for nm, y in (("mfe", mfe), ("mae", mae)):
                g = HistGradientBoostingRegressor(max_iter=250, random_state=seed % 2**31,
                                                  early_stopping=True, validation_fraction=0.15)
                g.fit(X[ti], y[ti], sample_weight=wgt[ti])
                ps[nm] = np.clip(g.predict(X[ei]), 1e-5, None)
            mh, ah = ps["mfe"], ps["mae"]
            Rh = np.maximum(mh, ah) / np.minimum(mh, ah)
            side = np.where(mh >= ah, 1, -1)
            tp = np.maximum(mh, ah)
            sl = np.minimum(mh, ah)
            rho = float(spearmanr(Rh, R[ei], nan_policy="omit").statistic)
            for gate in GATES:
                k = Rh >= np.quantile(Rh, 1 - gate)
                p = _pnl(close, opn, slip, ei[k], side[k], tp[k], sl[k])
                rows.append({"월": m, "seed": seed, "gate": gate, "rho": rho, "n": int(k.sum()),
                             "exp_bp": float(p.mean() * 1e4), "wr": float((p > 0).mean() * 100),
                             "realR": float(np.nanmedian(R[ei][k]))})
        d = pd.DataFrame(rows)
        d = d[d.월 == m]
        print(f"  {m} 컷 {thr*100:.3f}% 학습{int(tr.sum()):5,d}/검증{int(te.sum()):4,d}  "
              f"rho {d.rho.median():+.4f}  │ " + " │ ".join(
            f"게이트{int(g*100):3d}% 기대 {d[d.gate==g].exp_bp.median():+6.2f}bp "
            f"실제R {d[d.gate==g].realR.median():.2f}" for g in GATES), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mfe_mae_regression_gate.csv", index=False)
    print(f"\n{'게이트':>7s} {'rho중앙':>8s} {'기대손익':>10s} {'양수월':>7s} {'실제R중앙':>9s} {'평균건수':>8s}")
    for g in GATES:
        q = df[df.gate == g]
        mo = q.groupby("월")["exp_bp"].median()
        print(f"{int(g*100):6d}% {q.rho.median():+8.4f} {q.exp_bp.median():+9.2f}bp "
              f"{int((mo>0).sum()):3d}/{len(mo)} {q.realR.median():9.2f} {q.n.mean():8.0f}")
    print(f"\n무작위 진입 기대(게이트100% = 전 모집단)와 비교한다. rho 가 0 이면 게이팅은 무작위 부분추출이다.")
    print("단일 연도 · 5시드 · 비용 실측 비대칭(익절 5.52bp / 손절·시간청산 7.79bp).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
