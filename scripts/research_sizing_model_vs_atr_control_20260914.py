"""**배포 사이징 모델 vs 단순 ATR** — 기록이 «ATR 하나로 충분하다」고 적어놨다 (2026-09-14).

사용자: *"이전 기록에서 크기는 딥러닝이 잘 예측한다지 않았어?"* → 기록을 읽으니 반대였다:
[[omega461_label_axis_closed_size_yes_direction_no_20260911]] 은 «크기 축은 살아 있으나
**ATR 하나로 충분하다** -- 거기에 모델을 얹지 말 것(0/6 열세)» 이다. 그런데 09-13 에
**MAE 분위 모델을 배포**했다. 둘이 정합한지 이 세션에서 잰 적이 없다 -- 그걸 잰다.

## 🔴비교 기준을 바꾸면 안 된다
그 기록의 «0/6 열세」는 **상관계수**(rho, 1시간 MFE+MAE)다. 배포된 건 **분위 모델**이고
계약은 «초과율이 목표에 맞는가」다(`live_eth_mae_quantile_model_20260913` 검증 방식).
상관으로 재면 배포본을 그 계약 밖에서 평가하는 셈이라, 여기서는 **분위 계약으로** 잰다:

    ① 초과율(coverage)  -- 목표 0.1% 에 맞는가.       맞아야 «쓸 수 있다」
    ② 그 조건에서 **얼마나 타이트한가** -- 같은 위험에서 더 큰 크기를 허용하는 쪽이 이긴다
    ③ 초과가 **몰려 있는가** -- 같은 날 다 터지면 «0.1%」는 위안이 안 된다

## 대조군을 최대한 강하게 만든다
`safe_mae_atr = k_H × atr_pct` 로 두고 **k_H 를 지평마다 따로** 보정구간에서 맞춘다
(모델의 `mult` 와 같은 구간·같은 방식). 하나의 sqrt(H) 계수로 묶지 않는다 -- 묶으면
대조군을 일부러 약하게 만드는 것이고, 이 저장소가 반복해서 경계한 실수다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_mae_quantile_model_20260913 as maq  # noqa: E402

# 워크트리에는 원본 데이터가 없다 -- 본 체크아웃을 가리킨다(다른 조사 스크립트와 같은 처리).
_KL = pathlib.Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
assert _KL.exists(), _KL
maq.KLINES = _KL

EVERY = 6                      # 30분마다 표집 -- 겹침을 줄인다(같은 봉을 5지평이 공유)


def atr_pct_series(df: pd.DataFrame) -> np.ndarray:
    """사이징 워커와 같은 창(288봉=24시간)의 평균 절대변화 / 종가."""
    c = df.close.to_numpy(float)
    tr = np.abs(np.diff(c, prepend=c[0]))
    return (pd.Series(tr).rolling(288, min_periods=200).mean() / c).to_numpy()


def fit_k(p: pd.DataFrame, target: float) -> dict[float, float]:
    """지평별 k_H = quantile(실제MAE / atr_pct, 1-target). `maq.calibrate` 와 같은 꼴."""
    out = {}
    for lh, g in p.groupby("log_h"):
        r = g._mae.to_numpy() / np.maximum(g.atr_pct.to_numpy(), 1e-12)
        out[float(lh)] = float(np.quantile(r, 1.0 - target))
    return out


def report(name: str, sm: np.ndarray, mae: np.ndarray, lh: np.ndarray, day: np.ndarray,
           target: float) -> dict:
    ex = mae > sm
    # 초과가 몰렸나 -- 초과가 일어난 **날 수**로 본다(같은 날 여러 건은 한 사건에 가깝다)
    dd = len(np.unique(day[ex])) if ex.any() else 0
    return {"name": name, "n": len(sm), "exceed": float(ex.mean()), "breaches": int(ex.sum()),
            "days": dd, "med": float(np.median(sm)), "lev": float(np.median(100.0 / sm))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=maq.TARGET_EXCEED)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        # k_H 보정이 **보정구간에서** 정확히 목표 초과율을 만든다(정의상 그래야 한다)
        rng = np.random.default_rng(0)
        n = 20000
        p = pd.DataFrame({"log_h": 1.0, "atr_pct": 0.01,
                          "_mae": np.abs(rng.normal(0, 1, n))})
        k = fit_k(p, 0.10)
        sm = k[1.0] * p.atr_pct.to_numpy()
        assert abs((p._mae.to_numpy() > sm).mean() - 0.10) < 0.005, (k, sm[0])
        print("통과 — k_H 보정이 목표 초과율을 만든다")
        return 0

    art = maq.load_model()
    assert art is not None, f"배포 아티팩트가 없다: {maq.ARTIFACT}"
    models, mult = art["models"], float(art["mult"])
    df = maq._load_klines()
    atrp = atr_pct_series(df)
    p = maq.build_panel(df)
    p = p.iloc[::EVERY].copy()
    # 패널 행은 봉 인덱스를 보존한다(build_panel 이 base.copy() 를 지평×측면으로 쌓는다)
    p["atr_pct"] = atrp[p.index.to_numpy() % len(df)]
    p["ts"] = df.ts.to_numpy()[p.index.to_numpy() % len(df)]
    p = p[np.isfinite(p.atr_pct) & np.isfinite(p._mae)].copy()

    calib = p[(p.ts > maq.TRAIN_END) & (p.ts <= maq.CALIB_END)]
    test = p[p.ts > maq.CALIB_END]
    assert len(calib) > 5000 and len(test) > 5000, (len(calib), len(test))
    k = fit_k(calib, a.target)
    print(f"보정 {len(calib):,}행({maq.TRAIN_END}~{maq.CALIB_END}) · "
          f"검증 {len(test):,}행({maq.CALIB_END} 이후) · 목표 초과율 {100*a.target:.2f}%")
    print(f"ATR 계수 k_H: " + " · ".join(
        f"{int(round(np.exp(lh)))}분 {v:.1f}" for lh, v in sorted(k.items())))

    day = test.ts.to_numpy().astype("datetime64[D]")
    lh = test.log_h.to_numpy()
    mae = test._mae.to_numpy()
    rows = [report("배포 모델(q0.9×m)", maq.safe_mae(models, test, mult), mae, lh, day, a.target),
            report("ATR × k_H (대조군)",
                   np.array([k[float(x)] for x in lh]) * test.atr_pct.to_numpy(),
                   mae, lh, day, a.target)]
    print(f"\n{'팔':>18} {'초과율':>8} {'목표대비':>8} {'초과건':>7} {'초과일':>7} "
          f"{'안전MAE중앙':>11} {'허용배수중앙':>12}")
    for r in rows:
        print(f"{r['name']:>18} {100*r['exceed']:>7.3f}% {r['exceed']/a.target:>7.2f}x "
              f"{r['breaches']:>7,} {r['days']:>7} {r['med']:>10.2f}% {r['lev']:>11.1f}")

    print(f"\n지평별 (검증구간)")
    print(f"{'보유':>6} {'모델 초과율':>11} {'ATR 초과율':>11} {'모델 MAE':>9} {'ATR MAE':>9} "
          f"{'모델 배수':>9} {'ATR 배수':>9} {'승자':>6}")
    smm = maq.safe_mae(models, test, mult)
    sma = np.array([k[float(x)] for x in lh]) * test.atr_pct.to_numpy()
    for H in maq.HORIZONS_BARS:
        s = np.isclose(lh, np.log(H * 5.0))
        if not s.sum():
            continue
        em, ea = float((mae[s] > smm[s]).mean()), float((mae[s] > sma[s]).mean())
        mm, ma = float(np.median(smm[s])), float(np.median(sma[s]))
        # 둘 다 목표 안(2배 이내)이면 **더 타이트한 쪽**이 이긴다. 아니면 초과율이 먼저다.
        both_ok = em <= 2 * a.target and ea <= 2 * a.target
        win = ("모델" if mm < ma else "ATR") if both_ok else ("모델" if em < ea else "ATR")
        print(f"{H*5:>6} {100*em:>10.3f}% {100*ea:>10.3f}% {mm:>8.2f}% {ma:>8.2f}% "
              f"{100/mm:>8.1f} {100/ma:>8.1f} {win:>6}{'' if both_ok else ' ⚠초과율'}")
    print("\n⚠️둘 다 목표 초과율 안이면 **더 타이트한 쪽**(같은 위험에 더 큰 크기)이 이긴다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
