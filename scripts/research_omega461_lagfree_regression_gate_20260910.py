"""지연제거 라벨을 **회귀**로 — 연속 우위를 예측하고 임계값으로 진입을 좁힌다.

이진화(τ=0.10)는 우위의 크기를 버린다. 신호 수준 측정에서 커버 10~23%일 때 +38~49pp,
커버 86~90%일 때 +0.6~3.6pp 였으므로 선택성이 핵심 축이고, 회귀는 그 축을 직접 준다.

임계값은 절대값이 아니라 **커버리지**로 맞춘다 -- 그래야 모델 우위와 선택성 효과가 갈린다.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import research_omega461_entry_condition_lift_20260910 as lift  # noqa: E402
import research_omega461_label_oracle_ceiling_20260910 as oc  # noqa: E402
import research_omega461_lagfree_direction_label_20260910 as lf  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
W, TP, SL, CTX = 288, 0.075, 0.040, 10000
TEST_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]
COVERAGES = [0.05, 0.10, 0.20, 0.40]


def _ctx_reg(y, n, seed):
    """회귀 컨텍스트: 타깃 십분위로 층화해 범위를 고르게 덮는다."""
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isfinite(y))
    if len(ok) <= n:
        return ok
    q = np.quantile(y[ok], np.linspace(0, 1, 11))
    per, parts = max(n // 10, 1), []
    for i in range(10):
        b = ok[(y[ok] >= q[i]) & (y[ok] <= q[i + 1])]
        parts.append(b if len(b) <= per else rng.choice(b, size=per, replace=False))
    return np.sort(np.concatenate(parts))


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
    adv = lf._fwd_wr(out[:, 0], W) - lf._fwd_wr(out[:, 1], W)          # 연속 타깃

    def _wr(d):
        k = d >= 0
        return float((d[k] == 1).mean() * 100) if k.sum() >= 20 else np.nan

    from tabpfn import TabPFNRegressor
    print(f"[설정] w={W} 연속타깃 · 확장창 · 퍼지 {W}봉 · ctx {CTX:,} · 시드 {len(SEEDS)}", flush=True)
    rows = []
    for m in TEST_MONTHS:
        te = mon == m
        te_i = int(np.argmax(te))
        tr = np.zeros(len(frame), dtype=bool)
        tr[:max(te_i - W, 0)] = True
        if tr.sum() < 5000:
            continue
        tri = np.flatnonzero(tr)
        om = out[te]
        preds = {"TabPFN회귀": [], "HGB회귀": []}
        for seed in SEEDS:
            sel = tri[_ctx_reg(adv[tri], CTX, seed)]
            xs, ys = X[sel], adv[sel]
            r1 = TabPFNRegressor(n_estimators=4, device="cuda", random_state=seed % 2**31,
                                 fit_mode="fit_preprocessors")
            r1.fit(xs, ys)
            preds["TabPFN회귀"].append(np.asarray(r1.predict(X[te]), dtype=np.float64))
            r2 = HistGradientBoostingRegressor(max_iter=200, random_state=seed % 2**31,
                                               early_stopping=True, validation_fraction=0.15)
            r2.fit(xs, ys)
            preds["HGB회귀"].append(np.asarray(r2.predict(X[te]), dtype=np.float64))
        for nm, ps in preds.items():
            for cov in COVERAGES:
                res = []
                for p in ps:
                    thr = np.quantile(np.abs(p), 1 - cov)
                    v = np.where(np.abs(p) > thr, np.sign(p), 0).astype(np.int64)
                    pred = np.zeros(len(frame), dtype=np.int64)
                    pred[te] = v
                    r = oc._arm(base, pred, TP, SL, frame, fee, slip, device)
                    lifts = []
                    for col, s in ((0, 1), (1, -1)):
                        k = v == s
                        if k.sum() >= 100:
                            a, b = _wr(om[k, col]), _wr(om[:, col])
                            if np.isfinite(a) and np.isfinite(b):
                                lifts.append(a - b)
                    res.append({"pnl": r["pnl"], "n": r["n"], "wr": r["wr"],
                                "lift": float(np.mean(lifts)) if lifts else np.nan,
                                "long": int((v > 0).sum())})
                d = pd.DataFrame(res)
                rows.append({"월": m, "모델": nm, "커버": cov, "신호봉": int(cov * te.sum()),
                             "lift": d.lift.median(), "pnl": d.pnl.median(),
                             "양수": int((d.pnl > 0).sum()), "건수": int(d.n.median()),
                             "롱%": d["long"].median() / max(cov * te.sum(), 1) * 100})
                r0 = rows[-1]
                print(f"  {m} {nm:10s} 커버 {cov:4.0%}  lift {r0['lift']:+7.2f}pp  "
                      f"PnL {r0['pnl']:+7.2f}%  {r0['양수']}/{len(SEEDS)}  {r0['건수']:3d}건  "
                      f"롱 {r0['롱%']:3.0f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lagfree_regression_gate.csv", index=False)
    print("\n=== 커버리지별 집계 (6개월 · 커버 매칭이라 모델 비교가 공정하다) ===")
    print(f"{'모델':11s} {'커버':>5s} {'lift중앙':>9s} {'lift양수':>8s} {'월PnL중앙':>10s} "
          f"{'PnL양수월':>9s} {'복리':>9s}")
    for nm in ("TabPFN회귀", "HGB회귀"):
        for cov in COVERAGES:
            g = df[(df.모델 == nm) & (df.커버 == cov)]
            comp = (np.prod(1 + g.pnl.to_numpy() / 100) - 1) * 100
            print(f"{nm:11s} {cov:4.0%} {g.lift.median():+8.2f}pp {int((g.lift>0).sum()):3d}/{len(g)} "
                  f"{g.pnl.median():+9.2f}% {int((g.pnl>0).sum()):3d}/{len(g)} {comp:+8.2f}%")
    print("\n대조: 지연제거 분류(τ=0.10) 복리 +9.77% · 배포 신호lift 중앙 -0.02pp")
    print("단일 연도·전환 1회 — 방향 판단용이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
