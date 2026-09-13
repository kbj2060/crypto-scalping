"""**ML 이 사다리 깊이를 정한다** — 고정 ATR 배수 대신 학습된 MAE 분위 (2026-09-14, 사용자 요청).

사용자: *"딥러닝 모델이나 머신러닝 모델을 사용해줘."*

## 왜 MAE 분위 회귀인가 (새 모델을 안 만드는 이유)
사다리 깊이가 묻는 질문은 **«얼마나 깊이 역행할까»** 다. 그건 방향이 아니라 **크기** 문제이고,
이 저장소가 여러 번 확인한 «크기는 예측되고 방향은 안 된다»의 크기 쪽이다.
그리고 그 타깃을 학습하는 모델이 **이미 배포돼 있다** --
`live_eth_mae_quantile_model_20260913`: `HistGradientBoostingRegressor(loss="quantile")`,
22 피쳐 + log(보유시간) + 측면, **적중률(coverage)로 검증**.
새 아키텍처를 얹는 대신 **그 모델을 사다리가 필요로 하는 분위에서 다시 학습**한다
(같은 피쳐 빌더·같은 분할·같은 검증 규약).

## 깊이 ↔ 체결확률 항등식
칸을 깊이 d 에 두면 **MAE ≥ d 일 때만** 체결된다. 따라서
    d = MAE_q  =>  P(체결) = 1 − q
k 칸을 체결확률이 균등하도록 두려면 q = i/k (i=1..k−1). k=5 면 q = 0.2/0.4/0.6/0.8 이고
기대 체결 칸수는 1 + 0.8 + 0.6 + 0.4 + 0.2 = 3.0 이다. **설계로 정해지는 값**이라
고정 ATR 배수처럼 «너무 촘촘해서 다 채워지는» 사고가 구조적으로 안 난다.

## 고정 ATR 과 무엇이 다른가
ATR 배수는 **보유시간을 모른다**. MAE 는 H 에 따라 커지므로(√t 에 가깝다) 같은 변동성이라도
4시간과 24시간의 사다리는 달라야 한다. 모델은 log(H) 를 입력으로 받아 그걸 반영한다.

⚠️딥러닝이 아니라 GBM 이다. 신경망을 쓰려면 이 저장소 규약상 **아키텍처를 먼저 제안하고
사용자 확인**을 받아야 한다([[feedback_dl_architecture_requires_user_confirmation]]).
그리고 이 데이터에서 크기 예측은 atr_pct 단독이 AUC 0.8216 인데 57피쳐가 +0.001 만 더했다 --
신경망이 GBM 을 이길 사전 근거가 약하다. 먼저 GBM 으로 재고, 모자라면 그때 제안한다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_mae_quantile_model_20260913 as maq  # noqa: E402
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

# 🔴워크트리에는 `binance_data/` 가 없다(메인 체크아웃에만 있다). maq 의 KLINES 는
# 자기 ROOT 기준 상대경로라 워크트리에서 실행하면 FileNotFoundError 가 난다.
# 여기서 **명시적으로** 실제 파일을 가리킨다 -- 조용히 다른 데이터를 읽는 것보다 낫다.
_KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
assert _KL.exists(), f"5분봉 CSV 가 없다: {_KL}"
maq.KLINES = _KL

MAKER_BP, PEG_BP, EXIT_BP = 2.0, 2.95, 2.93
SEED = 20260914
WINDOWS = {"VAL(2025-09~12)⚠": ("2025-09-01", "2025-12-31"),
           "OOS(2026-01~03)": ("2026-01-01", "2026-03-31"),
           "TEST(2026-04~09)": ("2026-04-01", "2026-09-10")}


def train_depth_quantiles(k: int, out: pathlib.Path) -> dict:
    """사다리가 쓸 분위(q = i/k)를 **같은 아키텍처·같은 분할**로 학습하고 적중률을 낸다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    qs = [round(i / k, 4) for i in range(1, k)]
    df = maq._load_klines()
    panel = maq.build_panel(df)
    tr = panel[panel._ts <= maq.TRAIN_END]
    te = panel[panel._ts > maq.CALIB_END]
    print(f"패널 {len(panel):,} · 학습 {len(tr):,} · 검증 {len(te):,} · 분위 {qs}")
    models = {}
    for q in qs:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.06, max_depth=6,
                                          min_samples_leaf=200, random_state=SEED)
        m.fit(tr[maq.FEATURES], tr._mae)
        models[q] = m
    # 🔴적중률이 전부다(이 저장소 규약). 깊이 d=MAE_q 의 체결확률이 설계값 1−q 에 맞는가.
    print(f"\n표본외 적중률 — 체결확률이 설계대로인가")
    print(f"{'q':>6} {'설계 체결률':>11} {'실제 체결률':>11} {'예측깊이 중앙':>13}")
    ok = True
    for q in qs:
        d = models[q].predict(te[maq.FEATURES])
        hit = float((te._mae.to_numpy() >= d).mean())
        flag = "" if abs(hit - (1 - q)) <= 0.05 else "  🔴"
        if flag:
            ok = False
        print(f"{q:>6} {100*(1-q):>10.1f}% {100*hit:>10.1f}% {np.median(d):>12.3f}%{flag}")
    print("  (오차 5%p 이내면 사다리 깊이로 쓸 수 있다)" if ok else
          "  🔴 설계 체결률과 어긋난다 -- 이 분위로 사다리를 깔면 안 된다")
    joblib.dump({"models": models, "features": maq.FEATURES, "k": k}, out)
    print(f"저장 {out}")
    return models


def simulate(d: pd.DataFrame, idx: np.ndarray, sides: np.ndarray, *, hold_bars: int,
             depths: np.ndarray | None, k: int, atr_delta: float | None) -> dict:
    """depths(칸별 깊이 %, 행×(k−1)) 가 오면 ML 사다리, atr_delta 면 고정 ATR 사다리, 둘 다
    None 이면 단일 진입. 의도 명목 1 단위당 결과."""
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    atr = d.atr_pct.to_numpy(float)
    net, expo, fills = [], [], []
    w = hold_bars
    for r, (i, s) in enumerate(zip(idx, sides)):
        if i + w >= len(c):
            continue
        e0 = c[i]
        if k == 1:
            dep = np.zeros(1)
        elif depths is not None:
            dep = np.concatenate([[0.0], depths[r] / 100.0])      # % -> 비율
        else:
            a = atr[i]
            if not (a > 0):
                continue
            dep = np.array([j * atr_delta * a for j in range(k)])
        px = e0 * (1 - s * dep)
        seg_lo, seg_hi = lo[i + 1:i + 1 + w], hi[i + 1:i + 1 + w]
        if s > 0:
            run = np.minimum.accumulate(seg_lo); t = np.searchsorted(-run, -px, side="left")
        else:
            run = np.maximum.accumulate(seg_hi); t = np.searchsorted(run, px, side="left")
        t[0] = 0
        filled = t < w
        size = 1.0 / len(px)
        exit_px = c[i + w]
        pnl = sum(size * (s * (exit_px / px[j] - 1.0) - ((PEG_BP if j == 0 else MAKER_BP)
                                                         + EXIT_BP) / 1e4)
                  for j in range(len(px)) if filled[j])
        expo.append(float(sum(size * (w - t[j]) / w for j in range(len(px)) if filled[j])))
        net.append(1e4 * pnl)
        fills.append(int(filled.sum()))
    if not net:
        return {"n": 0}
    net = np.array(net); expo = np.array(expo)
    return {"n": len(net), "net_bp": float(net.mean()), "expo": float(expo.mean()),
            "per_expo": float(net.mean() / expo.mean()), "fills": float(np.mean(fills)),
            "worst": float(net.min()), "p05": float(np.quantile(net, 0.05))}


def _self_check() -> None:
    n = 400
    d = pd.DataFrame({"close": 100.0, "high": 100.0, "low": 100.0}, index=range(n))
    d["atr_pct"] = 0.01
    idx, sd = np.array([10]), np.array([1.0])
    r1 = simulate(d, idx, sd, hold_bars=48, depths=None, k=1, atr_delta=None)
    assert abs(r1["expo"] - 1.0) < 1e-9 and r1["fills"] == 1, r1
    # 평평하면 깊은 칸 미체결 -> ML 사다리도 노출 1/k
    dep = np.array([[0.5, 1.0, 1.5, 2.0]])                     # % 단위
    r5 = simulate(d, idx, sd, hold_bars=48, depths=dep, k=5, atr_delta=None)
    assert r5["fills"] == 1 and abs(r5["expo"] - 0.2) < 1e-9, r5
    # 깊이 0 을 주면 전부 즉시 체결 -> 노출 1.0 (사다리가 «단일»로 퇴화)
    r0 = simulate(d, idx, sd, hold_bars=48, depths=np.zeros((1, 4)), k=5, atr_delta=None)
    assert r0["fills"] == 5 and abs(r0["expo"] - 1.0) < 1e-9, r0
    print("통과 — 단일 노출 1.0 · 평평하면 1/k · 깊이 0 이면 단일로 퇴화")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hold-bars", type=int, default=48, help="5분봉 개수. 48=4시간")
    ap.add_argument("--acc", default="0.60,0.50")
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return 0

    art = ROOT / "data" / "live" / f"eth_ladder_depth_q{a.k}.joblib"
    models = train_depth_quantiles(a.k, art)

    df = maq._load_klines()
    c = df.close.to_numpy(float)
    X = svm.build_features(df.ts, c, df.quote_volume.to_numpy(float),
                           df.trades.to_numpy(float), df.high.to_numpy(float),
                           df.low.to_numpy(float))
    d = pd.DataFrame({"close": c, "high": df.high.to_numpy(float),
                      "low": df.low.to_numpy(float)})
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(288, min_periods=200).mean()
    d["atr_pct"] = (atr / c).to_numpy()
    ts = df.ts.to_numpy()
    ok = np.isfinite(X.to_numpy(float)).all(1)
    qs = sorted(models)
    print(f"\n보유 {a.hold_bars*5}분 · {a.every*5}분마다 표집 · k={a.k}\n")
    rng = np.random.default_rng(SEED)
    for acc in [float(x) for x in a.acc.split(",")]:
        print(f"=== 정확도 {acc} ===")
        print(f"{'창':>18} {'팔':>14} {'의도명목당bp':>12} {'노출':>6} {'노출당':>8} "
              f"{'체결칸':>7} {'최악':>9}")
        for wname, (w0, w1) in WINDOWS.items():
            lo_i = max(int(np.searchsorted(ts, np.datetime64(w0))), svm.WARMUP)
            hi_i = int(np.searchsorted(ts, np.datetime64(w1 + "T23:59:59"))) - a.hold_bars - 1
            if hi_i - lo_i < 1000:
                continue
            idx = np.array([i for i in range(lo_i, hi_i, a.every) if ok[i]])
            truth = np.where(c[idx + a.hold_bars] >= c[idx], 1.0, -1.0)
            sides = np.where(rng.random(len(idx)) < acc, truth, -truth)
            # ML 깊이: 그 봉·그 지평·그 측면의 예측 MAE 분위
            f = X.iloc[idx].copy()
            f["log_h"] = np.log(a.hold_bars * 5.0)
            f["side"] = sides
            depths = np.column_stack([models[q].predict(f[maq.FEATURES]) for q in qs])
            depths = np.maximum(depths, 0.0)
            arms = [("단일", None, 1, None), ("ATR 2.0", None, a.k, 2.0),
                    ("ATR 3.0", None, a.k, 3.0), ("**ML 깊이**", depths, a.k, None)]
            for lab, dep, kk, ad in arms:
                r = simulate(d, idx, sides, hold_bars=a.hold_bars, depths=dep, k=kk,
                             atr_delta=ad)
                if not r.get("n"):
                    continue
                print(f"{wname:>18} {lab:>14} {r['net_bp']:>12.2f} {r['expo']:>6.2f} "
                      f"{r['per_expo']:>8.2f} {r['fills']:>7.2f} {r['worst']:>9.1f}")
        print()
    print("⚠️노출당이 크기 매칭된 값이다. 단일 진입은 크기를 줄여도 산술 노출당이 **불변**이므로"
          " (분자·분모가 같이 준다), 사다리의 노출당 우위는 크기 효과가 아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
