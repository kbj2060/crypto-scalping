"""원시 트리플배리어 라벨 덤프 — 평활 없이 봉마다 자기 앞길로 답을 매긴다.

평활 라벨(w=288)은 앞 1일 승률의 롱-숏 차이라, 그 창 안의 모든 봉이 같은 답을 받는다
(정답지 차트 3번 패널: 승률이 0% 아니면 100%). 그래서 '하루 중 어느 봉이 좋은 타점인가'가
라벨에서 지워진다. 원시 라벨은 그 평활을 빼고 봉 i 의 실제 배리어 결과를 그대로 답으로 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sc  # noqa: E402
from core.event_label_engine import return_dispersion_volatility  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, FIX_TP, FIX_SL = 288, 0.10, 0.075, 0.040
sc._predict_exit_prob_one = lambda *a, **k: 0.0


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    n = len(frame)
    vol = return_dispersion_volatility(frame["close"], window=12, lookback=288)
    med = float(vol.median())
    tp_a, sl_a = (FIX_TP / med * vol).to_numpy(), (FIX_SL / med * vol).to_numpy()
    out = lift._outcomes(close, opn, tp_a, sl_a, slip)

    raw = np.where(out[:, 0] == 1, 1, np.where(out[:, 1] == 1, -1, 0)).astype(np.int64)
    a = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)
    sm = np.where(np.isnan(a), 0, np.where(a > TAU, 1, np.where(a < -TAU, -1, 0))).astype(np.int64)
    warm = ~np.isfinite(tp_a) | ~np.isfinite(sl_a)     # 변동성 워밍업: 배리어가 NaN 이면 청산이 영영 안 걸린다
    sm[warm] = 0
    raw[warm] = 0
    print(f"워밍업 마스크 {int(warm.sum()):,}봉 현금 처리")
    print(f"원시 라벨 {dict(pd.Series(raw).value_counts().sort_index())}")
    print(f"평활 라벨 {dict(pd.Series(sm).value_counts().sort_index())}")
    both = (raw != 0) & (sm != 0)
    print(f"둘 다 비현금인 봉 {int(both.sum()):,} 중 부호 일치 {float((raw[both]==sm[both]).mean())*100:.2f}%")

    df = pd.DataFrame({"timestamp": frame["timestamp"], "close": close, "tp": tp_a, "sl": sl_a,
                       "raw": raw, "smooth": sm, "adv": a,
                       "long_win": out[:, 0], "short_win": out[:, 1]})
    df["day"] = df["timestamp"].dt.date
    g = df[df.raw != 0].groupby("day")["raw"]
    share = g.apply(lambda v: (v == 1).mean())
    print(f"\n[일중 변동] 일자 {len(share):,}일 · 하루 롱비율 분포")
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        print(f"  q{q:<5} {share.quantile(q):.3f}")
    pure = float(((share < 0.02) | (share > 0.98)).mean() * 100)
    print(f"  하루가 한 방향으로 순수한(≥98%) 날 {pure:.1f}%  → 낮을수록 타점 정보가 있다")
    sm_share = df[df.smooth != 0].groupby("day")["smooth"].apply(lambda v: (v == 1).mean())
    pure_sm = float(((sm_share < 0.02) | (sm_share > 0.98)).mean() * 100)
    print(f"  (대조) 평활 라벨은 {pure_sm:.1f}%")
    df.drop(columns=["day"]).to_csv(OUT / "raw_tb_label_series.csv", index=False)

    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)
    for nm, lab in (("원시", raw), ("평활", sm)):
        dec = base["dec"].copy()
        dec["side"] = lab
        dec["action"] = np.where(lab > 0, omega.ACTION_LONG,
                                 np.where(lab < 0, omega.ACTION_SHORT, omega.ACTION_CASH))
        dec["notional_exposure"] = np.where(lab != 0, 0.45, 0.0)
        dec["take_profit"], dec["stop_loss"] = tp_a, sl_a
        comp = dict(base)
        comp["dec"] = dec
        comp["margin"], comp["leverage"] = np.full(n, 0.225), np.full(n, 2.0)
        act = int(omega._active(dec).sum())
        _s, lg = greedy_replay(frame, {"zig075": comp}, fee=fee, slip=slip,
                               cost_mult=retest.COST_MULT, device=device)
        if lg.empty:
            print(f"[정답지 {nm}] 활성봉 {act:,} 인데 원장 0건 — 리플레이 진입 조건 확인 필요")
            continue
        lg.to_csv(OUT / f"oracle_ledger_{'raw' if nm=='원시' else 'smooth'}.csv", index=False)
        r = lg["trade_return"].to_numpy(float)
        hold = (lg["exit_i"] - lg["entry_i"]) * 5 / 60
        print(f"[정답지 {nm}] 활성 {act:,}봉 · {len(lg)}건 · 복리 {(np.prod(1+r)-1)*100:+.2f}% · "
              f"WR {(r>0).mean()*100:.1f}% · 보유중앙 {hold.median():.1f}h · "
              f"{dict(lg['reason'].value_counts())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
