"""지연제거 라벨 월별 확장창 walk-forward — 레짐을 읽는가, 한쪽에 쏠린 것인가.

앞선 단일 분할(전반→후반)은 레짐 전환이 1회뿐이라 '전환을 읽었다'와 '이번엔 맞는 쪽에
있었다'를 못 가른다. 여기서는 3~8월을 차례로 표본외 검증하고, 각 월의 **시장 방향우위**와
**모델 롱비율**의 부호가 맞는지 본다. 전반(숏 유리)에서 숏을 내면 읽은 것이다.

배포 모델의 시차 상관은 동시 -0.474 / 한 달 지연 +0.358 이었다(= 한 달 뒤따름).
같은 측정을 새 라벨에 적용해 지연이 실제로 제거됐는지 확인한다.
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
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, TP, SL = 288, 0.10, 0.075, 0.040
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    t = pd.to_datetime(frame["timestamp"])
    mon = t.dt.to_period("M").astype(str).to_numpy()
    out = lift._outcomes(close, opn, TP, SL, slip)
    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)

    lw, sw = lf._fwd_wr(out[:, 0], W), lf._fwd_wr(out[:, 1], W)
    adv = lw - sw
    lab = np.where(np.isnan(adv), 0, np.where(adv > TAU, 1, np.where(adv < -TAU, -1, 0))).astype(np.int64)

    def _wr(d):
        k = d >= 0
        return float((d[k] == 1).mean() * 100) if k.sum() else np.nan

    print(f"[설정] w={W} τ={TAU} TP {TP*100:.1f}% SL {SL*100:.1f}% · 확장창 · 퍼지 {W}봉 · 시드 {len(SEEDS)}")
    print(f"{'검증월':9s} {'학습봉':>8s} {'시장우위':>9s} {'모델롱%':>7s} {'정렬':>4s} "
          f"{'PnL중앙':>9s} {'범위':>20s} {'양수':>5s} {'건수':>4s} {'WR':>6s}")
    rows = []
    for m in TEST_MONTHS:
        te = mon == m
        te_i = int(np.argmax(te))
        tr = np.zeros(len(frame), dtype=bool)
        tr[:max(te_i - W, 0)] = True                     # 확장창 + 경계 퍼지
        if tr.sum() < 5000:
            print(f"{m:9s} 학습 {int(tr.sum()):,}봉 — 부족, 건너뜀")
            continue
        mkt = _wr(out[te, 0]) - _wr(out[te, 1])
        res = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X[tr], lab[tr])
            pv = clf.predict(X[te])
            pred = np.zeros(len(frame), dtype=np.int64)
            pred[te] = pv
            r = oc._arm(base, pred, TP, SL, frame, fee, slip, device)
            res.append({"pnl": r["pnl"], "n": r["n"], "wr": r["wr"],
                        "long": int((pv > 0).sum()), "short": int((pv < 0).sum())})
        d = pd.DataFrame(res)
        lp = d["long"].median() / max(d["long"].median() + d["short"].median(), 1) * 100
        ok = "✅" if (lp - 50) * mkt > 0 else "❌"
        rows.append({"월": m, "시장우위": mkt, "모델롱%": lp, "pnl": d.pnl.median(),
                     "양수": int((d.pnl > 0).sum()), "n": int(d.n.median())})
        print(f"{m:9s} {int(tr.sum()):8,d} {mkt:+8.2f}pp {lp:6.1f}% {ok:>4s} "
              f"{d.pnl.median():+8.2f}% [{d.pnl.min():+8.2f},{d.pnl.max():+8.2f}] "
              f"{int((d.pnl>0).sum()):3d}/{len(SEEDS)} {int(d.n.median()):4d} {d.wr.median():5.1f}%", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "lagfree_walkforward.csv", index=False)
    print(f"\n부호 정렬 {int(((r['모델롱%']-50)*r['시장우위']>0).sum())}/{len(r)}개월  "
          f"(배포 모델은 3/8)")
    print(f"월 PnL 중앙 {r.pnl.median():+.2f}%  ·  양수 월 {int((r.pnl>0).sum())}/{len(r)}  ·  "
          f"복리 {(np.prod(1+r.pnl/100)-1)*100:+.2f}%")
    a, b = r["모델롱%"].to_numpy(), r["시장우위"].to_numpy()
    for lag in (-1, 0, 1):
        x, y = (a[-lag:], b[:lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag > 0 else (a, b))
        if len(x) >= 4:
            nm = "모델이 뒤따름" if lag < 0 else ("모델이 앞섬" if lag > 0 else "동시")
            print(f"  시차 {lag:+d}개월 ({nm:8s})  상관 {np.corrcoef(x, y)[0,1]:+.3f}   "
                  f"[배포: 동시 -0.474 · 뒤따름 +0.358]")
    print("\n단일 연도·전환 1회 — Fresh-Forward 다중레짐 미충족. 방향 판단용이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
