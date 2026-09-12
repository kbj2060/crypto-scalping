"""1시간 보유 · ATR 배리어 라벨 — 비용 장벽(손익분기 46.4%) 아래에서 사는가.

보정 실측: ATR(96)×2.0, RR 1.88 → TP 0.784% / SL 0.417%, 보유 중앙 14봉(70분).
그 크기에서 왕복비용이 배리어의 17.9% 라 손익분기 승률이 46.4% 다(현재 설계는 34.8%).
무작위 진입 기대 승률은 SL/(TP+SL) ≈ 34.7% 이므로 **+11.7pp 의 실력**이 있어야 본전이다.
그래서 이 팔의 판정은 PnL 이 아니라 먼저 '달성 승률 vs 46.4%' 로 한다.
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
from core.event_label_engine import (atr_volatility, combine_sample_weights,  # noqa: E402
                                     return_attribution_weights, sample_uniqueness_weights)
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
ATR_WIN, MULT, RR, MAXHOLD = 96, 2.0, 1.88, 36        # 보정 실측값 · 수직배리어 3시간
BE_WR = 46.4
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

    atr = atr_volatility(frame["high"], frame["low"], frame["close"], window=ATR_WIN)
    tp_a, sl_a = (MULT * RR * atr).to_numpy(), (MULT * atr).to_numpy()
    print(f"[배리어] ATR({ATR_WIN})×{MULT}·RR {RR}  TP 평균 {np.nanmean(tp_a)*100:.3f}% "
          f"SL {np.nanmean(sl_a)*100:.3f}%  수직 {MAXHOLD}봉  손익분기 WR {BE_WR}%", flush=True)
    out = lift._outcomes(close, opn, tp_a, sl_a, slip, horizon=MAXHOLD)
    for col, nm in ((0, "롱"), (1, "숏")):
        d = out[:, col]
        k = d >= 0
        print(f"  기저 {nm} 승률 {float((d[k]==1).mean())*100:5.2f}%  "
              f"(결착 {int(k.sum()):,} · 타임아웃 {int((~k).sum()):,})", flush=True)

    idx = np.arange(n)
    t1 = np.minimum(idx + MAXHOLD, n - 1)
    wgt = combine_sample_weights(sample_uniqueness_weights(idx, t1, n),
                                 return_attribution_weights(close, idx, t1))
    rows = []
    for w_sm in (12, 36, 72):
        a = _fwd(out[:, 0], w_sm) - _fwd(out[:, 1], w_sm)
        for tau in (0.05, 0.15):
            lab = np.where(np.isnan(a), 0, np.where(a > tau, 1, np.where(a < -tau, -1, 0))).astype(np.int64)
            if min((lab == 1).sum(), (lab == -1).sum()) < 500:
                continue
            for m in TEST_MONTHS:
                te = mon == m
                te_i = int(np.argmax(te))
                tr = np.zeros(n, dtype=bool)
                tr[:max(te_i - w_sm - MAXHOLD, 0)] = True
                if tr.sum() < 5000:
                    continue
                res = []
                for seed in SEEDS:
                    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                         early_stopping=True, validation_fraction=0.15)
                    clf.fit(X[tr], lab[tr], sample_weight=wgt[tr])
                    pv = clf.predict(X[te])
                    om = out[te]
                    hits = [(om[pv == s, c] == 1).sum() for c, s in ((0, 1), (1, -1))]
                    tot = [(om[pv == s, c] >= 0).sum() for c, s in ((0, 1), (1, -1))]
                    swr = float(sum(hits) / max(sum(tot), 1) * 100)
                    pred = np.zeros(n, dtype=np.int64)
                    pred[te] = pv
                    comp = dict(base)
                    d0 = comp["dec"].copy()
                    d0["take_profit"], d0["stop_loss"] = tp_a, sl_a
                    comp["dec"] = d0
                    r = oc._arm(comp, pred, tp_a, sl_a, frame, fee, slip, device)
                    res.append({"pnl": r["pnl"], "n": r["n"], "swr": swr})
                dd = pd.DataFrame(res)
                rows.append({"w": w_sm, "tau": tau, "월": m, "pnl": dd.pnl.median(),
                             "양수": int((dd.pnl > 0).sum()), "건수": int(dd.n.median()),
                             "신호WR": dd.swr.median()})
            g = pd.DataFrame(rows)
            g = g[(g.w == w_sm) & (g.tau == tau)]
            print(f"  w={w_sm:3d}({w_sm*5:3d}분) τ={tau:.2f}  신호WR 중앙 {g.신호WR.median():5.2f}% "
                  f"(손익분기 {BE_WR}%, 초과 {g.신호WR.median()-BE_WR:+5.2f}pp)  "
                  f"월PnL 중앙 {g.pnl.median():+7.2f}%  양수월 {int((g.pnl>0).sum())}/{len(g)}  "
                  f"복리 {(np.prod(1+g.pnl/100)-1)*100:+8.2f}%  평균 {g.건수.mean():4.1f}건", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "onehour_label_arms.csv", index=False)
    print(f"\n판정 순서: ① 신호WR 이 {BE_WR}% 를 넘는가 ② 그 다음에 PnL. "
          f"①을 못 넘으면 PnL 양수는 표본 잡음이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
