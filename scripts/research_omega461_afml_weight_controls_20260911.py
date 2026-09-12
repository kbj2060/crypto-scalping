"""C팔의 +132% 가 가중 로직 때문인가 잡음 증폭인가 — 가중치 대조군 4종.

표본가중이 24건짜리 결과를 두 배 이상 움직였다면 의심해야 한다. 셔플 가중이 같은 이득을
내면 로직이 아니라 분산이다(AFML Ch.4 의 주장은 '겹치는 표본의 중복을 깎는다'이므로,
같은 분포를 무작위로 배정하면 그 효과가 사라져야 한다).
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
import research_omega461_label_oracle_ceiling_20260910 as oc  # noqa: E402
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
from core.event_label_engine import (combine_sample_weights, return_attribution_weights,  # noqa: E402
                                     return_dispersion_volatility, sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, FIX_TP, FIX_SL = 288, 0.10, 0.075, 0.040
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    n = len(frame)
    t = pd.to_datetime(frame["timestamp"])
    mon = t.dt.to_period("M").astype(str).to_numpy()
    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)

    vol = return_dispersion_volatility(frame["close"], window=12, lookback=288)
    med = float(vol.median())
    tp_a = (FIX_TP / med * vol).to_numpy()
    sl_a = (FIX_SL / med * vol).to_numpy()
    out = lift._outcomes(close, opn, tp_a, sl_a, slip)
    a = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)
    lab = np.where(np.isnan(a), 0, np.where(a > TAU, 1, np.where(a < -TAU, -1, 0))).astype(np.int64)

    idx = np.arange(n)
    t1 = np.minimum(idx + W, n - 1)
    uq = sample_uniqueness_weights(idx, t1, n)
    ra = return_attribution_weights(close, idx, t1)
    both = combine_sample_weights(uq, ra)
    rng = np.random.default_rng(20260911)
    shuf = both.copy()
    rng.shuffle(shuf)
    print(f"[가중 분포] 유일성 중앙 {np.median(uq):.4f}  수익기여 중앙 {np.median(ra):.4f}  "
          f"결합 중앙 {np.median(both):.4f}  결합 q95/q05 {np.quantile(both,.95)/max(np.quantile(both,.05),1e-9):.1f}x")

    ARMS = [("무가중 (B)", None), ("유일성만", uq), ("수익기여만", ra),
            ("결합 (C)", both), ("결합 셔플 ← 트립와이어", shuf)]
    rows = []
    for nm, wgt in ARMS:
        for m in TEST_MONTHS:
            te = mon == m
            te_i = int(np.argmax(te))
            tr = np.zeros(n, dtype=bool)
            tr[:max(te_i - W, 0)] = True
            if tr.sum() < 1000:
                continue
            res = []
            for seed in SEEDS:
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                     early_stopping=True, validation_fraction=0.15)
                clf.fit(X[tr], lab[tr], sample_weight=None if wgt is None else wgt[tr])
                pred = np.zeros(n, dtype=np.int64)
                pred[te] = clf.predict(X[te])
                comp = dict(base)
                d = comp["dec"].copy()
                d["take_profit"], d["stop_loss"] = tp_a, sl_a
                comp["dec"] = d
                r = oc._arm(comp, pred, tp_a, sl_a, frame, fee, slip, device)
                res.append({"pnl": r["pnl"], "n": r["n"]})
            dd = pd.DataFrame(res)
            rows.append({"팔": nm, "월": m, "pnl": dd.pnl.median(),
                         "양수": int((dd.pnl > 0).sum()), "건수": int(dd.n.median())})
        g = pd.DataFrame(rows)
        g = g[g.팔 == nm]
        print(f"  {nm:22s} 월중앙 {g.pnl.median():+8.2f}%  양수월 {int((g.pnl>0).sum())}/{len(g)}  "
              f"복리 {(np.prod(1+g.pnl/100)-1)*100:+9.2f}%  평균 {g.건수.mean():4.1f}건", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "afml_weight_controls.csv", index=False)
    print("\n셔플이 결합과 비슷하면 이득은 가중 '로직'이 아니라 분산이다.")
    print(f"\n{df.pivot(index='월', columns='팔', values='pnl').round(2).to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
