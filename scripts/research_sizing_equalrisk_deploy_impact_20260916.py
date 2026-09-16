#!/usr/bin/env python3
"""«동일위험 1/safeMAE» 를 배포하면 무엇이 어떻게 변하는가 (2026-09-16).

사용자 *"동일위험 1/safeMAE 가 배포되면 어떻게 변하는지 자세히 알려줘"*.

⭐**먼저 밝힐 것: 이미 배포돼 있다.** `live_eth_risk_sizing_policy_20260913::survival_leverage`
가 정확히 `MAX_DRAWDOWN × 100 / safe_mae_pct` = **1/safeMAE** 다.
그런데 실계좌 72왕복에서 **모델 상한이 묶은 건 1건**뿐이었다(나머지는 순자산×6·원장중앙×2).
⇒ 진짜 질문은 «배포하면»이 아니라 **«왜 안 보이나, 보이게 하려면 무엇을 바꿔야 하나»** 다.

이 스크립트가 재는 것:
  A 왜 안 보이나  survival 배수 분포 vs 순자산 상한 6배 — 누가 더 작은가
  B 상한을 풀면  EQUITY_X ∈ {6,8,10,12,25} 별 «모델이 묶는 비율」과 평균 명목 변화
  C 평균명목 보존형  현재 명목 × (ref/safeMAE) — **상한을 안 건드리고 모양만** 가져오는 판
  D 위험 환산   전체 데이터 무작위 진입에서 A~C 각 판의 SD·하위1%·청산 도달률
🔴C 가 핵심이다 — B 는 «더 크게 걸기»라 위험이 같이 오르고, 경주의 «평균명목 동일» 전제가 깨진다.

출력: tmp/eth_equalrisk_deploy_20260916/
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/eth_equalrisk_deploy_20260916"
RNG = np.random.default_rng(20260916)
HOLD, COST_BP, WARMUP, N_DRAWS = 48, 5.88, 900, 120_000
EQUITY_X_NOW, LIQ_X = 6.0, 50.0


def _mod(rel, name):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); argv, sys.argv = sys.argv, ["x"]
    try: sp.loader.exec_module(m)
    finally: sys.argv = argv
    return m


def risk(ret, w, mae):
    wn = np.asarray(w, float); wn = wn / wn.mean()
    x = ret * wn
    return dict(cv=float(wn.std() / wn.mean()), sd=float(x.std(ddof=1)),
                p1=float(np.percentile(x, 1)), mae99=float(np.percentile(mae * wn, 99)),
                liq=float((mae * wn > 1e4 / LIQ_X).mean()),
                wmax=float(wn.max()), wmin=float(wn.min()))


def selftest() -> None:
    r = np.array([10., -20., 30.]); mae = np.array([5., 40., 3.])
    a, b = risk(r, np.ones(3), mae), risk(r, np.full(3, 5.0), mae)
    assert abs(a["sd"] - b["sd"]) < 1e-9 and a["cv"] < 1e-12
    # 1/x 가중은 x 가 작을 때 커진다
    w = 1.0 / np.array([1., 2., 4.]); wn = w / w.mean()
    assert wn[0] > wn[1] > wn[2]
    # 상한을 걸면 최대 가중이 줄어든다
    cap = np.minimum(w, 0.5); cn = cap / cap.mean()
    assert cn.max() <= wn.max() + 1e-12
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest(); OUT.mkdir(parents=True, exist_ok=True)
    MQ = _mod("scripts/live_eth_mae_quantile_model_20260913.py", "MQ")
    PO = _mod("scripts/live_eth_risk_sizing_policy_20260913.py", "PO")
    svm = MQ.svm

    kl = pd.read_csv(REPO / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                     usecols=["timestamp", "open", "high", "low", "close", "volume",
                              "quote_volume", "trades"])
    kl["timestamp"] = pd.to_datetime(kl.timestamp)
    ts = kl.timestamp
    op, c, hi, lo = (kl.open.to_numpy(float), kl.close.to_numpy(float),
                     kl.high.to_numpy(float), kl.low.to_numpy(float))
    n = len(kl); lo_i, hi_i = WARMUP, n - HOLD - 2
    idx = RNG.integers(lo_i, hi_i, N_DRAWS); side = RNG.choice([1, -1], N_DRAWS)
    e = op[idx + 1]
    ret = (c[idx + HOLD] - e) / e * 1e4 * side - COST_BP
    mae = np.empty(N_DRAWS)
    for k, (x, s) in enumerate(zip(idx, side)):
        w = slice(x + 1, x + 1 + HOLD)
        mae[k] = max(0.0, ((e[k] - lo[w].min()) if s > 0 else (hi[w].max() - e[k])) / e[k] * 1e4)

    base = svm.build_features(ts, c, kl.quote_volume.to_numpy(float),
                              kl.trades.to_numpy(float), hi, lo).replace([np.inf, -np.inf], np.nan)
    art = MQ.load_model()
    r = base.iloc[idx].copy().reset_index(drop=True)
    r["log_h"] = np.log(HOLD * 5.0); r["side"] = side
    safe = MQ.safe_mae(art["models"], r[MQ.FEATURES], art["mult"])
    surv = np.array([PO.survival_leverage(float(s)) for s in safe])   # = D*100/safe_mae
    print(f"무작위 진입 {N_DRAWS:,}건 · 보유 4h · 비용 {COST_BP}bp")
    print(f"\n{'='*116}")
    print("■ A 왜 안 보이나 — survival 배수(=1/safeMAE) vs 순자산 상한 6배")
    qs = [1, 10, 25, 50, 75, 90, 99]
    print(f"  safe_mae(%)   " + " ".join(f"p{q}:{np.percentile(safe,q):6.2f}" for q in qs))
    print(f"  survival 배수  " + " ".join(f"p{q}:{np.percentile(surv,q):6.2f}" for q in qs))
    print(f"  ⇒ survival 중앙 **{np.median(surv):.2f}배** vs 순자산 상한 **{EQUITY_X_NOW:.0f}배**")
    print(f"  ⇒ survival 이 6배보다 **작은** 비율 = 모델이 묶을 수 있는 유일한 경우 "
          f"= **{(surv < EQUITY_X_NOW).mean()*100:.2f}%** (safe_mae > "
          f"{PO.MAX_DRAWDOWN*100/EQUITY_X_NOW:.2f}% 인 봉)")
    print(f"  ⇒ 나머지 {100-(surv<EQUITY_X_NOW).mean()*100:.2f}% 에서는 **순자산 상한이 항상 이긴다**"
          f" — 실계좌 72왕복 중 모델이 묶은 게 1건뿐이었던 이유가 이것이다")
    print(f"  (정책 상수: MAX_DRAWDOWN={PO.MAX_DRAWDOWN} · HARD_CAP_X={PO.HARD_CAP_X})")

    print(f"\n■ B 상한을 풀면 — EQUITY_X 를 올릴 때 «모델이 묶는 비율」과 평균 명목")
    print(f"{'EQUITY_X':>10}{'모델이 묶는 비율':>16}{'실효 배수 중앙':>14}{'평균 명목(현재=1.0)':>18}")
    eff_now = np.minimum(surv, EQUITY_X_NOW)
    for X in (6.0, 8.0, 10.0, 12.0, 25.0):
        eff = np.minimum(surv, X)
        print(f"{X:>10.0f}{(surv < X).mean()*100:>15.2f}%{np.median(eff):>14.2f}"
              f"{eff.mean()/eff_now.mean():>18.2f}")
    print("  🔴상한을 올리면 «모델이 보이기» 시작하지만 **평균 명목이 같이 커진다** — 위험도 같이 커진다")

    print(f"\n■ C ⭐평균명목 보존형 — 상한은 그대로 두고 «모양」만 1/safeMAE 로")
    ref = float(np.median(safe))
    w_shape = ref / safe                      # 중앙에서 1.0, 저변동에서 >1, 고변동에서 <1
    for clip in (None, 2.0, 1.5):
        ws = np.clip(w_shape, 1 / clip, clip) if clip else w_shape
        print(f"  배수 클립 {('±%.1f배' % clip) if clip else '없음':<8} "
              f"가중 범위 [{(ws/ws.mean()).min():.2f}, {(ws/ws.mean()).max():.2f}] "
              f"· CV {(ws/ws.mean()).std():.2f}")

    print(f"\n■ D 위험 환산 (무작위 진입 {N_DRAWS:,}건 · 평균명목 정규화)")
    ARMS = {
        "현행 고정(기준)": np.ones(N_DRAWS),
        "현행 배포 min(surv,6배)": eff_now,
        "C 보존형 1/safeMAE(클립 없음)": w_shape,
        "C 보존형 1/safeMAE(±2배 클립)": np.clip(w_shape, 0.5, 2.0),
        "C 보존형 1/safeMAE(±1.5배 클립)": np.clip(w_shape, 1 / 1.5, 1.5),
    }
    R = {k: risk(ret, v, mae) for k, v in ARMS.items()}
    fx = R["현행 고정(기준)"]
    print(f"{'판':<30}{'가중CV':>8}{'가중범위':>14}{'SD':>8}{'vs기준':>8}{'하위1%':>9}{'vs기준':>8}"
          f"{'50배청산율':>11}{'vs기준':>8}")
    for k, v in R.items():
        rng_s = f"[{v['wmin']:.2f},{v['wmax']:.2f}]"
        print(f"{k:<30}{v['cv']:>8.2f}{rng_s:>14}{v['sd']:>8.1f}"
              f"{(v['sd']/fx['sd']-1)*100:>+7.1f}%{v['p1']:>9.1f}"
              f"{(v['p1']/fx['p1']-1)*100:>+7.1f}%{v['liq']*100:>10.2f}%"
              f"{(v['liq']/fx['liq']-1)*100:>+7.1f}%")
    print("  (하위1% 의 «vs기준 −%» 는 손실이 그만큼 얕아졌다는 뜻)")
    pd.DataFrame(R).T.round(4).to_csv(OUT / "risk.csv")
    print("=" * 116)
    print(json.dumps({"surv_median": round(float(np.median(surv)), 2),
                      "bind_pct": round(float((surv < EQUITY_X_NOW).mean() * 100), 2)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
