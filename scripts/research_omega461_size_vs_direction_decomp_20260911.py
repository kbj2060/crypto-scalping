"""크기 vs 방향 분해 — MFE/MAE 예측의 어느 성분이 살아 있는가.

앞 실험에서 예측 R ↔ 실제 R 상관이 +0.002 였다. 가설: 크기(MFE+MAE)는 예측되고
방향(MFE-MAE)이 안 된다. 맞다면 방향 축은 닫히고 크기 축은 사이징·리스크로 쓸 수 있다.

⭐ATR 단독 대조군 필수 — 저장소 기록이 "크기는 atr_pct 단독 AUC 0.8216, 57피쳐 기여 +0.001"
이므로, 모델이 ATR 을 넘는지가 진짜 질문이다.
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
import research_omega461_mfe_mae_regression_gate_20260911 as mg  # noqa: E402
from core.event_label_engine import (atr_volatility, combine_sample_weights,  # noqa: E402
                                     return_attribution_weights, sample_uniqueness_weights)

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def _rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 50 else np.nan


def main() -> int:
    _fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    n = len(frame)
    mon = pd.to_datetime(frame["timestamp"]).dt.to_period("M").astype(str).to_numpy()
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)
    mfe, mae = mg._fwd(close, opn, slip)
    atr = atr_volatility(frame["high"], frame["low"], frame["close"], window=96).to_numpy()
    ok = np.isfinite(mfe) & np.isfinite(mae) & (mae > 1e-6) & (mfe > 0) & np.isfinite(atr)
    size, direc = mfe + mae, mfe - mae
    R = np.maximum(mfe, mae) / np.minimum(mfe, mae)
    idx = np.arange(n)
    t1 = np.minimum(idx + mg.H, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx, t1, n),
                                 return_attribution_weights(close, idx, t1))
    print(f"[설정] 1시간 지평 · 전 유효 {int(ok.sum()):,}봉 · 시드 {len(SEEDS)}")
    print(f"{'월':9s} {'검증':>6s} │ {'MFE':>7s} {'MAE':>7s} {'크기':>7s} {'방향':>7s} {'R':>7s} │ "
          f"{'ATR단독 크기':>11s} {'모델-ATR':>9s}")
    rows = []
    for m in TEST_MONTHS:
        te_i = int(np.argmax(mon == m))
        tr = ok.copy()
        tr[max(te_i - mg.H, 0):] = False
        te = ok & (mon == m)
        if tr.sum() < 3000 or te.sum() < 300:
            continue
        ti, ei = np.flatnonzero(tr), np.flatnonzero(te)
        acc = {k: [] for k in ("mfe", "mae", "size", "dir", "R")}
        for seed in SEEDS:
            ph = {}
            for nm, y in (("mfe", mfe), ("mae", mae)):
                g = HistGradientBoostingRegressor(max_iter=250, random_state=seed % 2**31,
                                                  early_stopping=True, validation_fraction=0.15)
                g.fit(X[ti], y[ti], sample_weight=wgt[ti])
                ph[nm] = np.clip(g.predict(X[ei]), 1e-6, None)
            acc["mfe"].append(_rho(ph["mfe"], mfe[ei]))
            acc["mae"].append(_rho(ph["mae"], mae[ei]))
            acc["size"].append(_rho(ph["mfe"] + ph["mae"], size[ei]))
            acc["dir"].append(_rho(ph["mfe"] - ph["mae"], direc[ei]))
            acc["R"].append(_rho(np.maximum(ph["mfe"], ph["mae"]) / np.minimum(ph["mfe"], ph["mae"]), R[ei]))
        base_size = _rho(atr[ei], size[ei])
        r = {k: float(np.median(v)) for k, v in acc.items()}
        r.update({"월": m, "n": int(te.sum()), "atr_size": base_size,
                  "delta": r["size"] - base_size})
        rows.append(r)
        print(f"{m:9s} {r['n']:6,d} │ {r['mfe']:+7.3f} {r['mae']:+7.3f} {r['size']:+7.3f} "
              f"{r['dir']:+7.3f} {r['R']:+7.3f} │ {base_size:+11.3f} {r['delta']:+9.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "size_vs_direction_decomp.csv", index=False)
    print(f"\n{'성분':10s} {'rho 중앙':>9s} {'양수월':>7s}   해석")
    for k, nm, itp in (("mfe", "MFE", "먹을 폭"), ("mae", "MAE", "각오할 폭"),
                       ("size", "크기 MFE+MAE", "변동성"), ("dir", "방향 MFE-MAE", "어느 쪽"),
                       ("R", "손익비 R", "타점 품질")):
        print(f"{nm:12s} {df[k].median():+8.3f} {int((df[k]>0).sum()):3d}/{len(df)}   {itp}")
    print(f"\nATR 단독 크기 예측 rho 중앙 {df.atr_size.median():+.3f} · "
          f"모델이 ATR 대비 {df.delta.median():+.3f} ({int((df.delta>0).sum())}/{len(df)}개월 우세)")
    print("단일 연도 · 5시드 · 겹치는 앞창이라 유효표본은 봉 수보다 훨씬 작다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
