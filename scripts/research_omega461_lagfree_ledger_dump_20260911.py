"""현행 라벨 로직(지연제거 평활 + 적응배리어 + AFML가중)의 walk-forward 원장 덤프 — 차트용."""
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
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
from core.event_label_engine import (combine_sample_weights, return_attribution_weights,  # noqa: E402
                                     return_dispersion_volatility, sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, FIX_TP, FIX_SL = 288, 0.10, 0.075, 0.040
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEED = 615372041


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
    tp_a, sl_a = (FIX_TP / med * vol).to_numpy(), (FIX_SL / med * vol).to_numpy()
    out = lift._outcomes(close, opn, tp_a, sl_a, slip)
    a = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)
    lab = np.where(np.isnan(a), 0, np.where(a > TAU, 1, np.where(a < -TAU, -1, 0))).astype(np.int64)
    idx = np.arange(n)
    t1 = np.minimum(idx + W, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx, t1, n),
                                 return_attribution_weights(close, idx, t1))

    pred = np.zeros(n, dtype=np.int64)
    for m in TEST_MONTHS:
        te = mon == m
        te_i = int(np.argmax(te))
        tr = np.zeros(n, dtype=bool)
        tr[:max(te_i - W, 0)] = True
        if tr.sum() < 5000:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED % 2**31,
                                             early_stopping=True, validation_fraction=0.15)
        clf.fit(X[tr], lab[tr], sample_weight=wgt[tr])
        pred[te] = clf.predict(X[te])
        print(f"  {m} 학습 {int(tr.sum()):,}봉 → 신호 {int((pred[te]!=0).sum()):,}봉 "
              f"(롱 {int((pred[te]>0).sum()):,} / 숏 {int((pred[te]<0).sum()):,})", flush=True)

    dec = base["dec"].copy()
    dec["side"] = pred
    dec["action"] = np.where(pred > 0, omega.ACTION_LONG,
                             np.where(pred < 0, omega.ACTION_SHORT, omega.ACTION_CASH))
    dec["notional_exposure"] = np.where(pred != 0, 0.45, 0.0)
    dec["take_profit"], dec["stop_loss"] = tp_a, sl_a
    comp = dict(base)
    comp["dec"] = dec
    comp["margin"] = np.full(n, 0.225)
    comp["leverage"] = np.full(n, 2.0)
    import train_eval_omega4_2_risk_sidecar_20260622 as sc
    sc._predict_exit_prob_one = lambda *x, **k: 0.0
    _s, lg = greedy_replay(frame, {"zig075": comp}, fee=fee, slip=slip,
                           cost_mult=retest.COST_MULT, device=device)
    lg["tp_level"] = tp_a[lg["entry_i"].to_numpy()]
    lg["sl_level"] = sl_a[lg["entry_i"].to_numpy()]
    lg.to_csv(OUT / "lagfree_wf_ledger.csv", index=False)
    pd.DataFrame({"timestamp": frame["timestamp"], "close": close, "tp": tp_a, "sl": sl_a,
                  "signal": pred, "label": lab, "adv": a,
                  "lw": lf._fwd_wr(out[:, 0], W), "sw": lf._fwd_wr(out[:, 1], W),
                  "w_afml": wgt}).to_csv(OUT / "lagfree_wf_series.csv", index=False)

    # 정답지대로 매매했을 때의 원장 (오라클 실행)
    deco = base["dec"].copy()
    deco["side"] = lab
    deco["action"] = np.where(lab > 0, omega.ACTION_LONG,
                              np.where(lab < 0, omega.ACTION_SHORT, omega.ACTION_CASH))
    deco["notional_exposure"] = np.where(lab != 0, 0.45, 0.0)
    deco["take_profit"], deco["stop_loss"] = tp_a, sl_a
    co = dict(base)
    co["dec"] = deco
    co["margin"], co["leverage"] = np.full(n, 0.225), np.full(n, 2.0)
    _s2, lo = greedy_replay(frame, {"zig075": co}, fee=fee, slip=slip,
                            cost_mult=retest.COST_MULT, device=device)
    lo.to_csv(OUT / "lagfree_oracle_ledger.csv", index=False)
    ro = lo["trade_return"].to_numpy(float)
    print(f"[정답지 오라클] {len(lo)}건  복리 {(np.prod(1+ro)-1)*100:+.2f}%  WR {(ro>0).mean()*100:.1f}%  "
          f"{dict(lo['reason'].value_counts())}  롱 {(lo['side']>0).mean()*100:.0f}%")
    r = lg["trade_return"].to_numpy(float)
    print(f"\n원장 {len(lg)}건  복리 {(np.prod(1+r)-1)*100:+.2f}%  WR {(r>0).mean()*100:.1f}%  "
          f"{dict(lg['reason'].value_counts())}  롱 {(lg['side']>0).mean()*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
