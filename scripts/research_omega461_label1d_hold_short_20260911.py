"""라벨 지평 ≠ 보유 지평 — 신호가 사는 1일에서 라벨을 만들고 짧게 보유해 건수를 늘린다.

지금까지 라벨 지평과 보유 지평을 항상 같이 움직였다(1시간/1시간 → lift 0, 6시간/6시간 →
-2pp, 1일/1일 → +3.64pp). 신호는 1일에만 살고 건수는 짧아야 나온다 — 둘을 분리해 본다.

건별 샤프 0.21 · 41건이면 t=1.35(잡음) · 500건이면 t=4.7. 지금 구조가 신호봉 27,000 중
22건만 잡는다(포착률 0.08%). 건수가 병목이라는 가설의 직접 시험이다.

라벨·모델·가중은 검증된 것 그대로 고정하고 **실행 배리어만** 바꾼다.
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
from core.event_label_engine import (atr_volatility, combine_sample_weights,  # noqa: E402
                                     return_attribution_weights, return_dispersion_volatility,
                                     sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, RR = 288, 0.10, 1.88          # 라벨: 검증된 지연제거 평활(1일) 그대로
LBL_TP, LBL_SL = 0.075, 0.040         # 라벨을 만드는 배리어(1일 스케일) — 고정
MULTS = [0.5, 1.0, 2.0, 4.0, 8.0, 14.75]   # 실행 배리어 배수(14.75 = 1일 스케일 대조군)
ATR_KEEP = 0.30                        # 저변동 하위 30% 배제(인과 컷)
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
MK_FEE = 0.000276                      # 메이커 peg 레그 2.76bp(수수료+추격), 슬리피지 0


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
    atr = atr_volatility(frame["high"], frame["low"], frame["close"], window=96).to_numpy()
    lab_tp, lab_sl = (LBL_TP / med * vol).to_numpy(), (LBL_SL / med * vol).to_numpy()
    out = lift._outcomes(close, opn, lab_tp, lab_sl, slip)
    a = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)
    lab = np.where(np.isnan(a), 0, np.where(a > TAU, 1, np.where(a < -TAU, -1, 0))).astype(np.int64)
    warm = ~np.isfinite(lab_tp) | ~np.isfinite(atr)
    lab[warm] = 0
    idx = np.arange(n)
    t1 = np.minimum(idx + W, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx, t1, n),
                                 return_attribution_weights(close, idx, t1))
    print(f"[라벨] 지연제거 평활 w={W} τ={TAU} · 1일 배리어에서 생성 (고정)")
    print(f"[실행] ret_disp×배수·RR {RR} · 저변동 하위 {ATR_KEEP*100:.0f}% 배제(인과 컷)", flush=True)

    rows = []
    for m in TEST_MONTHS:
        te = mon == m
        te_i = int(np.argmax(te))
        tr = np.zeros(n, dtype=bool)
        tr[:max(te_i - W, 0)] = True
        if tr.sum() < 5000:
            continue
        cut = float(np.nanquantile(atr[tr], ATR_KEEP))
        preds = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X[tr], lab[tr], sample_weight=wgt[tr])
            p = np.zeros(n, dtype=np.int64)
            p[te] = clf.predict(X[te])
            p[atr < cut] = 0                       # 저변동 배제
            preds.append(p)
        for mult in MULTS:
            tp_a, sl_a = (mult * RR * vol).to_numpy(), (mult * vol).to_numpy()
            need_t = 2 * (fee + slip) / (np.nanmean(tp_a) + np.nanmean(sl_a)) * 100
            need_m = 2 * MK_FEE / (np.nanmean(tp_a) + np.nanmean(sl_a)) * 100
            res = []
            for p in preds:
                comp = dict(base)
                d = comp["dec"].copy()
                d["take_profit"], d["stop_loss"] = tp_a, sl_a
                comp["dec"] = d
                rt = oc._arm(comp, p, tp_a, sl_a, frame, fee, slip, device)
                rm = oc._arm(comp, p, tp_a, sl_a, frame, MK_FEE, 0.0, device)
                res.append({"pnl_t": rt["pnl"], "pnl_m": rm["pnl"], "n": rm["n"], "wr": rm["wr"]})
            dd = pd.DataFrame(res)
            rows.append({"월": m, "mult": mult, "tp": float(np.nanmean(tp_a)) * 100,
                         "sl": float(np.nanmean(sl_a)) * 100, "need_t": need_t, "need_m": need_m,
                         "n": dd.n.median(), "wr": dd.wr.median(),
                         "pnl_t": dd.pnl_t.median(), "pnl_m": dd.pnl_m.median(),
                         "pos_m": int((dd.pnl_m > 0).sum())})
        g = pd.DataFrame(rows)
        g = g[g.월 == m]
        print(f"  {m} 컷 {cut*100:.3f}%  " + " │ ".join(
            f"×{r.mult:<5g} {int(r.n):3d}건 {r.pnl_m:+7.2f}%" for r in g.itertuples()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label1d_hold_short.csv", index=False)
    print(f"\n{'배수':>6s} {'TP%':>6s} {'SL%':>6s} {'필요(테)':>8s} {'필요(메)':>8s} {'총건수':>7s} "
          f"{'WR':>6s} {'복리(테)':>10s} {'복리(메)':>10s} {'양수월(메)':>9s}")
    for mult in MULTS:
        q = df[df.mult == mult]
        if not len(q):
            continue
        ct = (np.prod(1 + q.pnl_t.to_numpy() / 100) - 1) * 100
        cm = (np.prod(1 + q.pnl_m.to_numpy() / 100) - 1) * 100
        print(f"{mult:6g} {q.tp.iloc[0]:5.2f}% {q.sl.iloc[0]:5.2f}% {q.need_t.iloc[0]:7.2f}pp "
              f"{q.need_m.iloc[0]:7.2f}pp {int(q.n.sum()):7d} {q.wr.median():5.1f}% "
              f"{ct:+9.2f}% {cm:+9.2f}% {int((q.pnl_m>0).sum()):4d}/{len(q)}")
    print("\n가설: 배수를 줄이면 건수가 늘고 엣지가 표현된다. 필요실력이 같이 오르므로 "
          "그 교차점이 있는지가 판정이다. 대조군은 ×14.75(=1일 스케일).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
