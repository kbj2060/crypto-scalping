"""지연 없는 진입 라벨 — 국소 방향우위. 라벨 4시험(상한·요구정확도·학습가능성·균형).

지그재그는 피벗이 스윙 완성 후에야 확정돼 구조적으로 뒤따른다(실측: 동시 상관 -0.474,
한 달 지연 +0.358). 대안은 진입봉 i+1 부터 앞을 보는 국소 방향우위다 -- 피벗 완성을
기다리지 않으므로 지연이 없고, w봉 평균이 개별 결과의 노이즈를 죽인다.

경계: 피쳐는 봉 i 까지, 라벨은 i+1 부터(진입도 open[i+1]).
Event-Label Boundary Contract 충족 -- 라벨 탐색 구간과 피쳐 창이 겹치지 않는다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
import research_omega461_label_oracle_ceiling_20260910 as oc  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
SPLIT = pd.Timestamp("2026-06-01")


def _fwd_wr(outcome_col, w):
    """봉 i 기준 [i+1, i+w] 구간의 배리어 승률. 결착된 것만 분모."""
    win = (outcome_col == 1).astype(np.float64)
    res = (outcome_col >= 0).astype(np.float64)
    cw, cr = np.concatenate([[0], np.cumsum(win)]), np.concatenate([[0], np.cumsum(res)])
    n = len(outcome_col)
    lo = np.minimum(np.arange(n) + 1, n)
    hi = np.minimum(np.arange(n) + 1 + w, n)
    num, den = cw[hi] - cw[lo], cr[hi] - cr[lo]
    return np.where(den >= max(w // 4, 20), num / np.maximum(den, 1), np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=float, default=0.075)
    ap.add_argument("--sl", type=float, default=0.040)
    ap.add_argument("--windows", default="288,864,2016,4032")
    ap.add_argument("--taus", default="0.05,0.10,0.20")
    args = ap.parse_args()
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    t = pd.to_datetime(frame["timestamp"])
    print(f"[프레임] {len(frame):,}봉  TP {args.tp*100:.1f}% SL {args.sl*100:.1f}%", flush=True)
    out = lift._outcomes(close, opn, args.tp, args.sl, slip)
    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)

    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)
    tr, va = (t < SPLIT).to_numpy(), (t >= SPLIT).to_numpy()
    print(f"[학습 분할] 전반 {int(tr.sum()):,}봉 → 후반 {int(va.sum()):,}봉 (레짐 반전 넘기)", flush=True)

    rows = []
    for w in [int(v) for v in args.windows.split(",")]:
        lw, sw = _fwd_wr(out[:, 0], w), _fwd_wr(out[:, 1], w)
        adv = lw - sw
        for tau in [float(v) for v in args.taus.split(",")]:
            lab = np.where(np.isnan(adv), 0, np.where(adv > tau, 1, np.where(adv < -tau, -1, 0))).astype(np.int64)
            nl, ns = int((lab == 1).sum()), int((lab == -1).sum())
            if min(nl, ns) < 500:
                continue
            m = oc._arm(base, lab, args.tp, args.sl, frame, fee, slip, device)
            bal = min(nl, ns) / max(nl, ns)
            h1n = int((lab[tr] != 0).sum())
            h2n = int((lab[va] != 0).sum())
            hbal = min(h1n, h2n) / max(h1n, h2n, 1)
            # ③ 학습 가능성: 전반 학습 → 후반 검증 (레짐 반전 통과 시험)
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import balanced_accuracy_score
            accs = []
            for seed in (615372041, 208844917, 933105268):
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                     early_stopping=True, validation_fraction=0.15)
                clf.fit(X[tr], lab[tr])
                accs.append(balanced_accuracy_score(lab[va], clf.predict(X[va])))
            rows.append({"w": w, "tau": tau, "롱": nl, "숏": ns, "롱숏균형": round(bal, 3),
                         "전후반균형": round(hbal, 3), "오라클PnL": m["pnl"], "거래": m["n"],
                         "WR": m["wr"], "전반": m["h1"], "후반": m["h2"],
                         "학습정확도": round(float(np.median(accs)), 4),
                         "정확도범위": f"{min(accs):.3f}~{max(accs):.3f}"})
            r = rows[-1]
            print(f"  w={w:5d}({w*5/60/24:4.1f}일) τ={tau:.2f}  롱{nl:6,d}/숏{ns:6,d} 균형{bal:.2f} "
                  f"전후{hbal:.2f} │ 오라클 {m['pnl']:+9.2f}% {m['n']:3d}건 WR{m['wr']:4.1f}% "
                  f"전{m['h1']:+8.2f}% 후{m['h2']:+8.2f}% │ 학습 {np.median(accs):.4f} "
                  f"[{min(accs):.3f},{max(accs):.3f}]", flush=True)

    print("\n=== ② 실제 모델 예측을 그대로 리플레이 (전반 학습 → 후반 매매) ===")
    print("시뮬레이션 열화가 아니라 실측이다. 대조군: 배포 -37.37% · 무작위 중앙 -17.41% · 오라클 +120.63%")
    from sklearn.ensemble import HistGradientBoostingClassifier
    for w, tau in ((288, 0.10), (288, 0.05), (864, 0.10)):
        lw2, sw2 = _fwd_wr(out[:, 0], w), _fwd_wr(out[:, 1], w)
        a2 = lw2 - sw2
        lab = np.where(np.isnan(a2), 0, np.where(a2 > tau, 1, np.where(a2 < -tau, -1, 0))).astype(np.int64)
        res = []
        for seed in (615372041, 208844917, 933105268, 471926350, 862017594):
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X[tr], lab[tr])
            pred = np.zeros(len(frame), dtype=np.int64)
            pred[va] = clf.predict(X[va])
            m = oc._arm(base, pred, args.tp, args.sl, frame, fee, slip, device)
            res.append({"pnl": m["h2"], "n": m["n"], "wr": m["wr"], "mdd": m["mdd"],
                        "롱": int((pred[va] > 0).sum()), "숏": int((pred[va] < 0).sum())})
        r = pd.DataFrame(res)
        bal = r["롱"].median() / max(r["롱"].median() + r["숏"].median(), 1)
        print(f"  w={w} τ={tau:.2f}  후반 PnL 중앙 {r.pnl.median():+8.2f}%  "
              f"[{r.pnl.min():+8.2f},{r.pnl.max():+8.2f}]  양수 {int((r.pnl>0).sum())}/5  "
              f"{int(r.n.median()):3d}건 WR {r.wr.median():4.1f}%  MDD {r.mdd.median():+7.2f}%  "
              f"롱비율 {bal:.0%}", flush=True)

    df = pd.DataFrame(rows)
    f = OUT / "lagfree_direction_label.csv"
    df.to_csv(f, index=False)
    print(f"\n무작위 3클래스 균형정확도 = 0.3333 · 무작위 2클래스 = 0.5")
    if len(df):
        b = df.nlargest(1, "학습정확도").iloc[0]
        print(f"[학습 최고] w={int(b.w)} τ={b.tau}  정확도 {b.학습정확도:.4f}  "
              f"오라클 {b.오라클PnL:+.2f}%  후반 {b.후반:+.2f}%  롱숏균형 {b.롱숏균형:.2f}")
    print(f"산출물: {f}")
    print("주의: 학습은 전반→후반 교차레짐 검증이다. 단일 구간·시드 3개 → 승격 근거 아님.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
