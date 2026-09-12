"""① 모델의 방향 전환이 시장 전환을 앞서는가 뒤따르는가  ② 조건이 '어느 쪽'을 옮기는가.

지그재그 라벨은 사후 확정 피벗이라 구조적으로 뒤따를 수밖에 없다는 가설을 직접 잰다.
그리고 라벨을 '조건 × 방향'으로 재정의하려면, 조건이 승률이 아니라 **방향 우위**를
옮겨야 한다 -- 단독 lift 스크린(4/88, 우연 기대 4.4)은 그걸 묻지 않았다.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
B_NULL, NULL_SEED = 400, 615372041


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=float, default=0.075)
    ap.add_argument("--sl", type=float, default=0.040)
    args = ap.parse_args()
    device = parent._device("cpu")
    _fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    oc = lift._outcomes(close, opn, args.tp, args.sl, slip)
    n = len(frame)
    t = pd.to_datetime(frame["timestamp"])
    mon = t.dt.to_period("M").astype(str).to_numpy()

    base = prepare_component(frame, OUT / "preds" / "zig075" / "predictions_q075.csv",
                             dict(retest.COMPONENTS["zig075"]), device)
    side = np.where(omega._active(base["dec"]),
                    pd.to_numeric(base["dec"]["side"], errors="raise").to_numpy(np.int64), 0)

    def _wr(d):
        k = d >= 0
        return float((d[k] == 1).mean() * 100) if k.sum() else np.nan

    print("=== ① 방향 전환 타이밍 — 모델 vs 시장 ===")
    print(f"{'월':9s} {'신호봉':>7s} {'모델 롱%':>8s} {'기저 롱WR':>9s} {'기저 숏WR':>9s} {'시장 방향우위':>12s} {'정렬':>5s}")
    rows = []
    for m in sorted(set(mon)):
        k = mon == m
        sg = side[k]
        nsig = int((sg != 0).sum())
        if nsig == 0:
            continue
        lw, sw = _wr(oc[k, 0]), _wr(oc[k, 1])
        adv = lw - sw                                  # >0 이면 시장이 롱에 유리
        mlong = float((sg > 0).sum()) / nsig * 100
        ok = "✅" if (mlong - 50) * adv > 0 else "❌"   # 모델 편향과 시장 우위의 부호 일치
        rows.append({"월": m, "n": nsig, "모델롱%": mlong, "롱WR": lw, "숏WR": sw, "우위": adv})
        print(f"{m:9s} {nsig:7,d} {mlong:7.1f}% {lw:8.2f}% {sw:8.2f}% {adv:+11.2f}pp {ok:>5s}")
    r = pd.DataFrame(rows)
    print(f"\n부호 일치 {int(((r['모델롱%']-50)*r['우위']>0).sum())}/{len(r)}개월")
    for lag in (-2, -1, 0, 1, 2):
        a, b = r["모델롱%"].to_numpy(), r["우위"].to_numpy()
        if lag < 0:
            x, y = a[-lag:], b[:lag]                   # 모델이 나중 = 뒤따름
        elif lag > 0:
            x, y = a[:-lag], b[lag:]                   # 모델이 먼저 = 앞섬
        else:
            x, y = a, b
        if len(x) >= 4:
            nm = "모델이 뒤따름" if lag < 0 else ("모델이 앞섬" if lag > 0 else "동시")
            print(f"  시차 {lag:+d}개월 ({nm:8s})  상관 {np.corrcoef(x, y)[0,1]:+.3f}")

    print("\n=== ② 방향 조건부 lift — 조건이 '어느 쪽'을 옮기는가 ===")
    uncond = _wr(oc[:, 0]) - _wr(oc[:, 1])
    print(f"무조건 방향우위(롱WR-숏WR) {uncond:+.2f}pp · 순환이동 귀무 B={B_NULL}")
    rng = np.random.default_rng(NULL_SEED)
    shifts = rng.integers(1, n, size=B_NULL)
    conds = lift._conditions(frame)
    out = []
    for cname, mask in conds.items():
        idx = np.flatnonzero(mask)
        if len(idx) < 300:
            continue
        adv = _wr(oc[idx, 0]) - _wr(oc[idx, 1])
        if not np.isfinite(adv):
            continue
        null = np.array([_wr(oc[(idx + s) % n, 0]) - _wr(oc[(idx + s) % n, 1]) for s in shifts])
        null = null[np.isfinite(null)]
        p = float(min((null >= adv).mean(), (null <= adv).mean()) * 2)   # 양측
        out.append({"조건": cname, "n": len(idx), "방향우위": adv, "무조건": uncond,
                    "Δ": adv - uncond, "귀무중앙": float(np.median(null)),
                    "귀무q05": float(np.quantile(null, .05)),
                    "귀무q95": float(np.quantile(null, .95)), "p": p})
    df = pd.DataFrame(out).sort_values("Δ", key=abs, ascending=False)
    f = OUT / "direction_conditional_lift.csv"
    df.to_csv(f, index=False)
    sig = df[df.p <= 0.05]
    print(f"[귀무 통과 {len(sig)}개 / {len(df)}셀 · 다중검정 기대 {len(df)*0.05:.1f}개]")
    print(df.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n산출물: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
