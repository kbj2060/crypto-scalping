"""신호 수준 예측력 — 리플레이가 아니라 '모델이 지목한 모든 봉'으로 잰다.

월 3~11건 리플레이로는 검정력이 없다. 라벨은 월 수천 봉을 신호로 내는데 단일 포지션
그리디가 대부분을 버린다. 여기서는 **지목된 모든 봉**의 배리어 승률을 그 달 기저와 대조해
'예측이 있는데 리플레이가 못 담는 것'인지 '애초에 예측이 없는 것'인지 가른다.

귀무: 그 달 안에서 결과 배열을 순환이동(예측의 시간 군집·달의 결과 구조 둘 다 보존).
겹치는 앞창 때문에 유효 표본은 봉 수보다 훨씬 작다 -- 귀무가 그 상관을 흡수한다.
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
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TAU, TP, SL = 288, 0.10, 0.075, 0.040
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
B_NULL = 400


def _wr(d):
    k = d >= 0
    return float((d[k] == 1).mean() * 100) if k.sum() >= 20 else np.nan


def main() -> int:
    device = parent._device("cpu")
    _fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    t = pd.to_datetime(frame["timestamp"])
    mon = t.dt.to_period("M").astype(str).to_numpy()
    out = lift._outcomes(close, opn, TP, SL, slip)
    bundle = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    X = parent._base_input(frame, list(bundle["base_cols"])).to_numpy(np.float32)
    lw, sw = lf._fwd_wr(out[:, 0], W), lf._fwd_wr(out[:, 1], W)
    adv = lw - sw
    lab = np.where(np.isnan(adv), 0, np.where(adv > TAU, 1, np.where(adv < -TAU, -1, 0))).astype(np.int64)

    dep_comp = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                                 dict(retest.COMPONENTS["zig075"]), device)
    dep = np.where(omega._active(dep_comp["dec"]),
                   pd.to_numeric(dep_comp["dec"]["side"], errors="raise").to_numpy(np.int64), 0)

    rng = np.random.default_rng(615372041)
    print(f"[설정] w={W} τ={TAU} · 확장창 · 퍼지 {W}봉 · 시드 {len(SEEDS)} · 귀무 B={B_NULL}")
    print(f"{'월':9s} {'측면':>4s} {'지목봉':>7s} {'커버':>6s} {'지목WR':>7s} {'기저WR':>7s} "
          f"{'lift':>8s} {'귀무q95':>8s} {'p':>6s} │ {'배포 lift':>9s} {'배포n':>6s}")
    rows = []
    for m in TEST_MONTHS:
        te = mon == m
        te_i = int(np.argmax(te))
        tr = np.zeros(len(frame), dtype=bool)
        tr[:max(te_i - W, 0)] = True
        if tr.sum() < 5000:
            continue
        preds = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed % 2**31,
                                                 early_stopping=True, validation_fraction=0.15)
            clf.fit(X[tr], lab[tr])
            preds.append(clf.predict(X[te]))
        P = np.vstack(preds)
        vote = np.sign(P.sum(axis=0)).astype(np.int64)      # 5시드 다수결
        om = out[te]
        L = int(te.sum())
        for col, s, nm in ((0, 1, "롱"), (1, -1, "숏")):
            msk = vote == s
            if msk.sum() < 200:
                print(f"{m:9s} {nm:>4s} {int(msk.sum()):7,d}  — 표본 부족")
                continue
            wp, wb = _wr(om[msk, col]), _wr(om[:, col])
            null = np.array([_wr(np.roll(om[:, col], int(k))[msk])
                             for k in rng.integers(1, L, size=B_NULL)])
            null = null[np.isfinite(null)]
            p = float((null >= wp).mean())
            dm = dep[te] == s
            dl = (_wr(om[dm, col]) - wb) if dm.sum() >= 200 else np.nan
            rows.append({"월": m, "측면": nm, "n": int(msk.sum()), "커버": msk.mean() * 100,
                         "지목WR": wp, "기저WR": wb, "lift": wp - wb,
                         "귀무q95": float(np.quantile(null, .95)), "p": p,
                         "배포lift": dl, "배포n": int(dm.sum())})
            r = rows[-1]
            print(f"{m:9s} {nm:>4s} {r['n']:7,d} {r['커버']:5.1f}% {wp:6.2f}% {wb:6.2f}% "
                  f"{r['lift']:+7.2f}pp {r['귀무q95']:7.2f}% {p:6.3f} │ "
                  f"{dl:+8.2f}pp {r['배포n']:6,d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_level_precision.csv", index=False)
    sig = df[df.p <= 0.05]
    print(f"\n[귀무 통과 {len(sig)}/{len(df)}셀 · 다중검정 기대 {len(df)*0.05:.1f}개]")
    print(f"lift 중앙 {df.lift.median():+.2f}pp · 양수 {int((df.lift>0).sum())}/{len(df)}  │  "
          f"배포 lift 중앙 {df.배포lift.median():+.2f}pp · 양수 {int((df.배포lift>0).sum())}/{int(df.배포lift.notna().sum())}")
    print("\n겹치는 앞창 때문에 유효 표본은 봉 수보다 훨씬 작다 — 판정은 p 로만 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
