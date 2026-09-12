"""지연제거 라벨 확증 — 경계 퍼지 + 라벨 대조군 3종 + 시드 8개.

앞선 실측(+48.14%, 5/5)에는 구멍이 있었다: 라벨이 [i+1, i+w] 를 보므로 전반 끝 w봉의
라벨이 후반을 들여다본다. 여기서는 분할선 앞 w봉을 학습에서 **퍼지**한다.

대조군은 파이프라인을 전부 고정하고 **라벨만** 바꾼다 -- 그래야 이득의 출처가 라벨인지
갈린다. 특히 '단일봉 배리어 결과'는 같은 배리어·같은 지평인데 평활만 없앤 것이라,
이 짝이 '평활이 기전'이라는 주장의 직접 시험이다.
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
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
import research_omega461_label_oracle_ceiling_20260910 as oc  # noqa: E402
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
SPLIT = pd.Timestamp("2026-06-01")
W, TAU, TP, SL = 288, 0.10, 0.075, 0.040
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594,
         341778902, 726093415, 158364277]


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    t = pd.to_datetime(frame["timestamp"])
    out = lift._outcomes(close, opn, TP, SL, slip)
    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)

    va = (t >= SPLIT).to_numpy()
    split_i = int(np.argmax(va))
    tr_raw = (t < SPLIT).to_numpy()
    tr = tr_raw.copy()
    tr[max(split_i - W, 0):split_i] = False          # 경계 퍼지: 라벨이 검증구간을 보는 봉 제거
    print(f"[퍼지] 분할선 앞 {W}봉 학습 제외 — 학습 {int(tr_raw.sum()):,} → {int(tr.sum()):,}봉  "
          f"검증 {int(va.sum()):,}봉", flush=True)

    lw, sw = lf._fwd_wr(out[:, 0], W), lf._fwd_wr(out[:, 1], W)
    adv = lw - sw
    lagfree = np.where(np.isnan(adv), 0, np.where(adv > TAU, 1, np.where(adv < -TAU, -1, 0))).astype(np.int64)
    # 단일봉 배리어 결과: 같은 배리어·같은 지평, 평활만 없앰
    single = np.where(out[:, 0] == 1, 1, np.where(out[:, 1] == 1, -1, 0)).astype(np.int64)
    rng0 = np.random.default_rng(20260910)
    shuffled = lagfree.copy()
    rng0.shuffle(shuffled)                           # 누수 트립와이어: 이건 무작위여야 한다

    labels = {"① 지연제거(평활 w=288)": lagfree,
              "② 단일봉 배리어 결과(평활 없음)": single,
              "③ 라벨 셔플(트립와이어)": shuffled}
    print(f"\n대조군: 배포 -37.37% · 무작위 중앙 -17.41%(양수 0/5) · 오라클 +120.63%")
    print(f"{'라벨':32s} {'정확도':>7s} {'후반PnL 중앙':>12s} {'범위':>22s} {'양수':>5s} "
          f"{'건수':>4s} {'WR':>6s} {'MDD':>8s} {'롱%':>5s}")
    for nm, lab in labels.items():
        accs, res = [], []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X[tr], lab[tr])
            pv = clf.predict(X[va])
            accs.append(balanced_accuracy_score(lab[va], pv))
            pred = np.zeros(len(frame), dtype=np.int64)
            pred[va] = pv
            m = oc._arm(base, pred, TP, SL, frame, fee, slip, device)
            res.append({"pnl": m["h2"], "n": m["n"], "wr": m["wr"], "mdd": m["mdd"],
                        "long": int((pv > 0).sum()), "short": int((pv < 0).sum())})
        r = pd.DataFrame(res)
        lp = r["long"].median() / max(r["long"].median() + r["short"].median(), 1) * 100
        print(f"{nm:32s} {np.median(accs):7.4f} {r.pnl.median():+11.2f}% "
              f"[{r.pnl.min():+8.2f},{r.pnl.max():+8.2f}] {int((r.pnl>0).sum()):3d}/{len(SEEDS)} "
              f"{int(r.n.median()):4d} {r.wr.median():5.1f}% {r.mdd.median():+7.2f}% {lp:4.0f}%", flush=True)
        print(f"{'':32s} 고유 PnL {len(set(round(v,2) for v in r.pnl)):d}종 "
              f"(시드가 실제로 다른 모델을 만드는지)", flush=True)
    print("\n무작위 3클래스 균형정확도 0.3333. ③이 무작위 근처가 아니면 파이프라인 누수다.")
    print("단일 구간·단일 분할 — Seed-Diversity 는 충족하나 Fresh-Forward 다중구간은 미충족.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
