"""배리어와 평활창을 둘 다 6시간에 맞춘 유일한 미측정 셀 — 스캘핑 축의 경계.

보정 실측: ret_disp(12,288)×1.50·RR 1.88 → TP 1.43%/SL 0.76%, 보유 평균 73.4봉(6.1시간),
타임아웃 0.01%. 실측 메이커 비용(진입 peg 2.76bp + TP resting / SL taker)에서
손익분기 WR 37.88%, 필요 실력 3.15pp(스트레스 3.87pp).

지금까지: 1시간 배리어 + 1~6시간 평활 → lift 0 · 1일 배리어 + 1일 평활 → lift +3.64pp.
그 사이가 비어 있었고 여기가 그 지점이다. 3.15pp 를 넘으면 스캘핑 축이 열리고,
0 에 가까우면 이 자산·이 비용에서 닫힌다.

판정은 PnL 이 아니라 **신호 WR vs 손익분기 37.88%** 로 한다 — 비용이 명시적이라 정확하다.
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
from core.event_label_engine import (combine_sample_weights, return_attribution_weights,  # noqa: E402
                                     return_dispersion_volatility, sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
MULT, RR, HORIZON = 1.50, 1.88, 1152          # 수직 배리어 느슨(4일) → 타임아웃 ~0
BE_MAKER, GAP_MAKER = 37.88, 3.15             # 실측 메이커 비용 기준
MAKER_FEE = 0.000276                          # peg 레그 2.76bp(수수료+추격 포함), 슬리피지 0
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def _fwd(col, w):
    win, res = (col == 1).astype(float), (col >= 0).astype(float)
    cw, cr = np.r_[0, np.cumsum(win)], np.r_[0, np.cumsum(res)]
    n = len(col)
    lo, hi = np.minimum(np.arange(n) + 1, n), np.minimum(np.arange(n) + 1 + w, n)
    num, den = cw[hi] - cw[lo], cr[hi] - cr[lo]
    return np.where(den >= max(w // 4, 10), num / np.maximum(den, 1), np.nan)


def main() -> int:
    device = parent._device("cpu")
    fee_t, slip_t = omega._load_fee_slip()
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
    tp_a, sl_a = (MULT * RR * vol).to_numpy(), (MULT * vol).to_numpy()
    print(f"[배리어] ret_disp×{MULT}·RR{RR}  TP 평균 {np.nanmean(tp_a)*100:.3f}% "
          f"SL {np.nanmean(sl_a)*100:.3f}%  수직 {HORIZON}봉")
    print(f"[비용] 메이커 손익분기 WR {BE_MAKER}% · 필요 실력 {GAP_MAKER}pp "
          f"(테이커 리플레이 왕복 {2*(fee_t+slip_t)*100:.3f}% / 메이커 리플레이 {2*MAKER_FEE*100:.3f}%)",
          flush=True)
    out = lift._outcomes(close, opn, tp_a, sl_a, slip_t, horizon=HORIZON)
    bwr = {}
    for col, nm in ((0, "롱"), (1, "숏")):
        d = out[:, col]
        k = d >= 0
        bwr[nm] = float((d[k] == 1).mean() * 100)
        print(f"  기저 {nm} 승률 {bwr[nm]:5.2f}%  (결착 {int(k.sum()):,} · 타임아웃 {int((~k).sum()):,})")
    base_wr = float((out[out >= 0] == 1).mean() * 100)
    print(f"  기저 합산 {base_wr:5.2f}%  → 손익분기 {BE_MAKER}% 까지 {BE_MAKER-base_wr:+.2f}pp 필요", flush=True)

    idx = np.arange(n)
    t1 = np.minimum(idx + 73, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx, t1, n),
                                 return_attribution_weights(close, idx, t1))
    rows = []
    for w_sm in (36, 73, 144):
        a = _fwd(out[:, 0], w_sm) - _fwd(out[:, 1], w_sm)
        for tau in (0.05, 0.10, 0.20):
            lab = np.where(np.isnan(a), 0, np.where(a > tau, 1, np.where(a < -tau, -1, 0))).astype(np.int64)
            if min((lab == 1).sum(), (lab == -1).sum()) < 500:
                continue
            for m in TEST_MONTHS:
                te = mon == m
                te_i = int(np.argmax(te))
                tr = np.zeros(n, dtype=bool)
                tr[:max(te_i - w_sm - 300, 0)] = True
                if tr.sum() < 5000:
                    continue
                res = []
                for seed in SEEDS:
                    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                         early_stopping=True, validation_fraction=0.15)
                    clf.fit(X[tr], lab[tr], sample_weight=wgt[tr])
                    pv = clf.predict(X[te])
                    om = out[te]
                    hit = sum(int((om[pv == s, c] == 1).sum()) for c, s in ((0, 1), (1, -1)))
                    tot = sum(int((om[pv == s, c] >= 0).sum()) for c, s in ((0, 1), (1, -1)))
                    pred = np.zeros(n, dtype=np.int64)
                    pred[te] = pv
                    comp = dict(base)
                    d0 = comp["dec"].copy()
                    d0["take_profit"], d0["stop_loss"] = tp_a, sl_a
                    comp["dec"] = d0
                    rt = oc._arm(comp, pred, tp_a, sl_a, frame, fee_t, slip_t, device)
                    rm = oc._arm(comp, pred, tp_a, sl_a, frame, MAKER_FEE, 0.0, device)
                    res.append({"swr": hit / max(tot, 1) * 100, "pnl_t": rt["pnl"],
                                "pnl_m": rm["pnl"], "n": rm["n"]})
                dd = pd.DataFrame(res)
                rows.append({"w": w_sm, "tau": tau, "월": m, "swr": dd.swr.median(),
                             "pnl_t": dd.pnl_t.median(), "pnl_m": dd.pnl_m.median(),
                             "양수_m": int((dd.pnl_m > 0).sum()), "건수": int(dd.n.median())})
            g = pd.DataFrame(rows)
            g = g[(g.w == w_sm) & (g.tau == tau)]
            lift_pp = g.swr.median() - base_wr
            ok = "✅" if lift_pp >= GAP_MAKER else "❌"
            print(f"  w={w_sm:3d}({w_sm*5/60:4.1f}h) τ={tau:.2f}  신호WR {g.swr.median():5.2f}% "
                  f"lift {lift_pp:+5.2f}pp (필요 {GAP_MAKER}) {ok}  │ PnL 테이커 {g.pnl_t.median():+7.2f}% "
                  f"메이커 {g.pnl_m.median():+7.2f}%  양수 {int((g.pnl_m>0).sum())}/{len(g)}  "
                  f"복리(메이커) {(np.prod(1+g.pnl_m/100)-1)*100:+8.2f}%  {g.건수.mean():5.1f}건", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "sixhour_matched_label.csv", index=False)
    print(f"\n판정: 신호 lift 가 {GAP_MAKER}pp(메이커 기본) 또는 3.87pp(스트레스)를 넘는가. "
          f"못 넘으면 이 자산·이 비용에서 스캘핑 축은 닫힌다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
