"""AFML 처방 3요소를 하나씩 켜서 분리 측정 — 변동성 적응 배리어 · 표본가중 · CUSUM 이벤트.

오늘 ATR 실험은 `atr_pct × mult` 를 썼는데, `core/event_label_engine.py` 주석이 그 방식이
이 저장소에서 이미 두 번 오진단으로 밝혀졌다고 적어놨다. 처방은 `return_dispersion_volatility`
(창 누적 로그수익률 std)다. 여기서는 그것으로 바꾸되 **평균 배리어 크기를 배포와 맞춘다** --
안 그러면 배리어 크기 스윕을 또 하는 것뿐이다(오늘 ATR 짝비교에서 실제로 그랬다).

겹치는 라벨(봉 i 가 [i+1,i+288] 을 봄)은 유효표본을 43,200 → ~150 으로 떨어뜨린다.
AFML Ch.4 유일성·수익기여 가중이 그 보정이고, 엔진에 이미 있다.
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
from core.event_label_engine import (TripleBarrierConfig, apply_triple_barrier,  # noqa: E402
                                     combine_sample_weights, cusum_filter,
                                     return_attribution_weights, return_dispersion_volatility,
                                     sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU = 288, 0.10
FIX_TP, FIX_SL = 0.075, 0.040
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def _smoothed(out):
    a = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)
    return np.where(np.isnan(a), 0, np.where(a > TAU, 1, np.where(a < -TAU, -1, 0))).astype(np.int64)


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
    pt_m, sl_m = FIX_TP / med, FIX_SL / med          # 평균 배리어 크기를 배포와 맞춤
    tp_a = (pt_m * vol).to_numpy()
    sl_a = (sl_m * vol).to_numpy()
    print(f"[변동성] return_dispersion 중앙 {med*100:.4f}%  → pt_mult {pt_m:.2f} sl_mult {sl_m:.2f}")
    print(f"[적응 배리어] TP 평균 {np.nanmean(tp_a)*100:.2f}% (q05 {np.nanquantile(tp_a,.05)*100:.2f} "
          f"q95 {np.nanquantile(tp_a,.95)*100:.2f})  vs 고정 {FIX_TP*100:.2f}%", flush=True)

    out_fix = lift._outcomes(close, opn, FIX_TP, FIX_SL, slip)
    out_ada = lift._outcomes(close, opn, tp_a, sl_a, slip)
    lab_fix, lab_ada = _smoothed(out_fix), _smoothed(out_ada)

    ev = cusum_filter(close, (1.0 * vol).to_numpy())
    tb = apply_triple_barrier(frame, ev, vol, TripleBarrierConfig(pt_mult=pt_m, sl_mult=sl_m, max_hold=4032))
    print(f"[CUSUM] 이벤트 {len(ev):,}봉 · 배리어 라벨 {dict(tb['label'].value_counts().sort_index())} "
          f"· 접촉 {dict(tb['touch_type'].value_counts())}", flush=True)

    idx_all = np.arange(n)
    t1_all = np.minimum(idx_all + W, n - 1)
    w_all = combine_sample_weights(sample_uniqueness_weights(idx_all, t1_all, n),
                                   return_attribution_weights(close, idx_all, t1_all))
    ev_i = tb["event_idx"].to_numpy(np.int64)
    t1_e = tb["t1_idx"].to_numpy(np.int64)
    w_ev = combine_sample_weights(sample_uniqueness_weights(ev_i, t1_e, n),
                                  return_attribution_weights(close, ev_i, t1_e))
    lab_ev = np.zeros(n, dtype=np.int64)
    lab_ev[ev_i] = tb["label"].to_numpy(np.int64)
    is_ev = np.zeros(n, dtype=bool)
    is_ev[ev_i] = True
    wev_full = np.zeros(n)
    wev_full[ev_i] = w_ev
    print(f"[가중] 전봉 유일성×수익기여 중앙 {np.median(w_all):.4f} · 유효표본비 "
          f"{w_all.sum()**2/np.sum(w_all**2)/n:.4f}  │ CUSUM {np.median(w_ev):.4f}", flush=True)

    ARMS = [("A 고정·전봉·무가중", lab_fix, tp_a * 0 + FIX_TP, sl_a * 0 + FIX_SL, None, None),
            ("B 적응·전봉·무가중", lab_ada, tp_a, sl_a, None, None),
            ("C 적응·전봉·AFML가중", lab_ada, tp_a, sl_a, w_all, None),
            ("D 적응·CUSUM·엔진라벨", lab_ev, tp_a, sl_a, wev_full, is_ev)]
    rows = []
    for nm, lab, tpv, slv, wgt, evm in ARMS:
        for m in TEST_MONTHS:
            te = mon == m
            te_i = int(np.argmax(te))
            tr = np.zeros(n, dtype=bool)
            tr[:max(te_i - W, 0)] = True
            if evm is not None:
                tr &= evm
            if tr.sum() < 1000:
                continue
            res = []
            for seed in SEEDS:
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                     early_stopping=True, validation_fraction=0.15)
                clf.fit(X[tr], lab[tr], sample_weight=None if wgt is None else wgt[tr])
                pm = te & evm if evm is not None else te
                pred = np.zeros(n, dtype=np.int64)
                pred[pm] = clf.predict(X[pm])
                comp = dict(base)
                d = comp["dec"].copy()
                d["take_profit"], d["stop_loss"] = tpv, slv
                comp["dec"] = d
                r = oc._arm(comp, pred, tpv, slv, frame, fee, slip, device)
                res.append({"pnl": r["pnl"], "n": r["n"], "wr": r["wr"],
                            "long": int((pred[pm] > 0).sum()), "sig": int((pred[pm] != 0).sum())})
            dd = pd.DataFrame(res)
            rows.append({"팔": nm, "월": m, "pnl": dd.pnl.median(), "양수": int((dd.pnl > 0).sum()),
                         "건수": int(dd.n.median()), "신호": int(dd.sig.median()),
                         "롱%": dd["long"].median() / max(dd.sig.median(), 1) * 100})
            r0 = rows[-1]
            print(f"  {nm:22s} {m}  PnL {r0['pnl']:+8.2f}%  {r0['양수']}/{len(SEEDS)}  "
                  f"{r0['건수']:3d}건  신호 {r0['신호']:6,d}  롱 {r0['롱%']:3.0f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "afml_barrier_arms.csv", index=False)
    print(f"\n{'팔':24s} {'월PnL중앙':>10s} {'양수월':>7s} {'복리':>10s} {'평균건수':>8s}")
    for nm, *_ in ARMS:
        g = df[df.팔 == nm]
        if not len(g):
            continue
        print(f"{nm:24s} {g.pnl.median():+9.2f}% {int((g.pnl>0).sum()):3d}/{len(g)} "
              f"{(np.prod(1+g.pnl/100)-1)*100:+9.2f}% {g.건수.mean():8.1f}")
    print("\n대조: 오늘 분류 walk-forward 복리 +9.77% · 회귀 게이팅 8/8 셀 음수")
    print("단일 연도·전환 1회 — 방향 판단용이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
