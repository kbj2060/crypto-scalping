"""**거래를 고정하고 크기만 바꾼다** — 상한을 상태의 함수로 만들면 상수를 이기는가 (2026-09-14).

1 차 설계(`research_state_dependent_cap_20260914`)는 상한이 예산 사다리·지평 선택을 통해
**거래 목록 자체를 바꿔** 「같은 노출」이 같은 비교가 아니었다(상한 2/6/10 에서 거래 816/838/803).
여기서는 그 결합을 끊는다:
  · 예산 사다리 **OFF** -- 사다리 컷만이 크기에 의존하는 청산 경로다.
  · 지평은 **정책 천장(6배)** 으로 고른다 -- 배포 구조 그대로(`effective_cap(sz, eq, None)`).
  ⇒ 실측 확인: 상한 2/6/10 에서 (진입봉, 측면, 지평, 수익률 r) 목록이 **완전히 동일**하다.

그러면 자산은 `eq *= 1 + L_t · r_t` 로만 움직인다. 순수 켈리 문제이고, 평가가 산술이라
씨드를 많이 쓸 수 있다.

## ⭐이 설계에는 검증 가능한 예측이 있다
정확도가 주입 **상수**라 `E[r|상태] = (2acc−1)·E[|이동| |상태] − 비용`, `Var ∝ E[이동²|상태]`.
성장최적 `L* ≈ E[r]/E[r²] ∝ 1/변동성` ⇒ **1/ATR 규칙이 이겨야 한다.** 안 이기면 비용과
손절이 그 구조를 지운 것이고, 그건 그 자체로 답이다.
⚠️단 이 유리한 구조는 **주입의 산물**이다. 실제로는 정확도가 변동성과 같이 움직일 수 있다.

## 🔴평균 L 을 맞추지 않으면 운영점을 못 잰다
1 차 실행에서 상태의존 팔이 전부 평균 L 11.5~12(상단)에 붙었다. 그러면 «같은 평균 L 의 상수»
가 상수 10~12 가 되는데 프런티어가 거기서 급락·요동친다 -- 그 구간의 +Δ 는 «과도한
레버리지에서 덜 망한다」는 뜻이지 «6 배 근처에서 낫다»가 아니다. 실제로 그 팔들의 **절대**
성장은 상수 6 배보다 전부 나빴다(TEST 5.4 vs 16.4bp).
⇒ 각 팔을 **평균 L = 6 으로 정규화**한 변형을 같이 낸다. 그게 운영점 비교다.

## 판정
크기 매칭: 상수 스윕이 (평균 L -> 로그성장) 프런티어를 만들고, 상태의존 팔은 **같은 평균 L**
에서 그 프런티어를 이겨야 한다. 씨드마다 그 씨드의 상수로 프런티어를 만들어 **짝지어** 뺀다.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
_sp = importlib.util.spec_from_file_location(
    "stack", ROOT / "scripts" / "research_fresh_forward_random_entry_stack_20260914.py")
H = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(H)
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

POLICY_CAP = 6.0
CONSTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
L_LO, L_HI = 0.5, 12.0
SPAN = ("2025-01-01", "2026-09-10")
SPLITS = {"TRAIN(2025-01~08)": ("2025-01-01", "2025-08-31"),
          "VAL(2025-09~12)⚠": ("2025-09-01", "2025-12-31"),
          "OOS(2026-01~03)": ("2026-01-01", "2026-03-31"),
          "TEST(2026-04~09)": ("2026-04-01", "2026-09-10")}


def growth(L: np.ndarray, r: np.ndarray) -> tuple[float, bool]:
    """거래당 로그성장. `1 + L·r <= 0` 이면 파산(그 시점까지만 센다)."""
    x = 1.0 + L * r
    bad = np.flatnonzero(x <= 0.0)
    if len(bad):
        k = bad[0]
        return (float(np.log(x[:k]).sum() / max(k, 1)) if k else -9.9), True
    return float(np.log(x).mean()), False


def _self_check() -> None:
    # 켈리 해석해: 두 결과(+b, -b) 가 확률 p 면 L* = (p(1+b) - 1)/b · 1/b 근사 -- 격자로 확인
    rng = np.random.default_rng(0)
    r = np.where(rng.random(200000) < 0.60, 0.02, -0.02)
    gs = [(L, growth(np.full(len(r), L), r)[0]) for L in np.arange(1, 30, 0.5)]
    best = max(gs, key=lambda t: t[1])[0]
    # f* = p - q = 0.2 -> 자본의 20% 를 걸어야 하고, 변동폭 2% 이므로 L* = 0.2/0.02 = 10
    assert 8.0 <= best <= 12.0, best
    # 파산 판정: L·r <= -1 이면 파산이고 그 뒤는 안 센다
    g, ru = growth(np.array([20.0, 20.0]), np.array([-0.06, 0.10]))
    assert ru and g < 0, (g, ru)
    print(f"통과 — 켈리 최적 L={best} (해석해 10) · 파산 판정")


def gen_trades(d, sm, seed: int, acc: float) -> pd.DataFrame:
    ts = d.timestamp.to_numpy()
    lo = max(int(np.searchsorted(ts, np.datetime64(SPAN[0]))), svm.WARMUP)
    hi = int(np.searchsorted(ts, np.datetime64(SPAN[1] + "T23:59:59")))
    r = H.walk(d, sm, lo, hi, acc=acc, p_entry=1.0, use_stop=True, use_ladder=False,
               use_add=False, selector=True, cap_x=POLICY_CAP, policy_cap_x=POLICY_CAP,
               rng=np.random.default_rng(seed))
    t = pd.DataFrame([x for x in r["_trades"] if "i" in x])
    t["ts"] = d.timestamp.to_numpy()[t.i.to_numpy()]
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=float, default=0.60)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check(); return 0
    d = H.load(); sm = H.safe_mae_series(d)
    c = d.close.to_numpy(float)
    X = svm.build_features(d.timestamp, c, d.quote_volume.to_numpy(float),
                           d.trades.to_numpy(float), d.high.to_numpy(float),
                           d.low.to_numpy(float))
    atr = sm["atr_pct"]
    hi_a = pd.Series(d.high.to_numpy(float)); lo_a = pd.Series(d.low.to_numpy(float))
    print(f"거래를 고정하고 크기만 바꾼다 · 정확도 {a.acc} · 씨드 {a.seeds} · "
          f"사다리 OFF · 지평 정책천장 {POLICY_CAP}배\n")

    rows = {}          # (창, 팔) -> [씨드별 (로그성장, 평균L, 파산)]
    for sd in range(a.seeds):
        t = gen_trades(d, sm, H.SEED + 1000 * sd, a.acc)
        i = t.i.to_numpy(); side = t.side.to_numpy(); r = t.r.to_numpy(); hb = t.hb.to_numpy()
        # 실제 MAE(오라클용): 그 거래의 진입~청산 구간 최대 역행폭
        rm = np.array([max(1e-3, 100.0 * ((c[ii] - lo_a[ii + 1:ii + 1 + h].min()) / c[ii]
                                          if s > 0 else
                                          (hi_a[ii + 1:ii + 1 + h].max() - c[ii]) / c[ii]))
                       for ii, s, h in zip(i, side, hb)])
        mdl = np.array([sm[(int(h) * 5, "LONG" if s > 0 else "SHORT")][ii]
                        for ii, s, h in zip(i, side, hb)])
        av = atr[i]
        tr_m = (t.ts.to_numpy() <= np.datetime64(SPLITS["TRAIN(2025-01~08)"][1] + "T23:59:59"))
        k_inv = float(np.nanmedian(av[tr_m]))                 # 학습구간 중앙 ATR
        arms = [(f"상수 {x:.0f}배" + (" (배포)" if x == 6 else ""), np.full(len(r), x))
                for x in CONSTS]
        raw = [
            ("규칙 1/ATR", np.clip(6.0 * k_inv / np.maximum(av, 1e-12), L_LO, L_HI)),
            ("모델 전권", np.clip(100.0 / np.maximum(mdl, 1e-9), L_LO, L_HI)),
            ("위험 오라클", np.clip(100.0 / rm, L_LO, L_HI)),
            ("부호 오라클(참고)", np.where(r > 0, L_HI, L_LO)),
        ]
        arms += raw
        # ⭐운영점 비교 -- 평균 L 을 6 으로 맞춘다. 척도는 **학습구간에서** 정한다
        # (창마다 다시 맞추면 그 창의 결과를 보고 크기를 정하는 셈이다).
        for nm, L in raw:
            mtr = float(np.mean(L[tr_m])) if tr_m.sum() else float(np.mean(L))
            arms.append((nm + " ·L6", np.clip(L * (6.0 / max(mtr, 1e-9)), L_LO, L_HI)))
        for wn, (w0, w1) in SPLITS.items():
            m = ((t.ts.to_numpy() >= np.datetime64(w0))
                 & (t.ts.to_numpy() <= np.datetime64(w1 + "T23:59:59")))
            if m.sum() < 100:
                continue
            for nm, L in arms:
                g, ru = growth(L[m], r[m])
                rows.setdefault((wn, nm), []).append((g, float(L[m].mean()), ru))
    cn = [f"상수 {x:.0f}배" + (" (배포)" if x == 6 else "") for x in CONSTS]
    for wn in SPLITS:
        if (wn, cn[0]) not in rows:
            continue
        n_tr = len(rows[(wn, cn[0])])
        print(f"{wn}  (씨드 {n_tr})")
        print(f"{'팔':>18} {'거래당 로그성장':>15} {'평균 L':>8} {'파산':>6} "
              f"{'Δ프런티어(짝지음)':>20}")
        for nm in [x for x in cn] + [x for x, _ in arms if x not in cn]:
            v = rows[(wn, nm)]
            g = np.array([x[0] for x in v]); mL = np.array([x[1] for x in v])
            ru = sum(x[2] for x in v)
            gap = ""
            if nm not in cn:
                dd = []
                for k in range(n_tr):
                    fr = sorted((rows[(wn, cc)][k][1], rows[(wn, cc)][k][0]) for cc in cn)
                    fx = np.array([x for x, _ in fr]); fy = np.array([y for _, y in fr])
                    dd.append(v[k][0] - float(np.interp(v[k][1], fx, fy)))
                dd = np.array(dd); se = dd.std(ddof=1) / np.sqrt(len(dd))
                mk = "✅" if dd.mean() - 2 * se > 0 else ("🔴" if dd.mean() + 2 * se < 0 else "–")
                gap = f"{1e4*dd.mean():+8.1f}±{1e4*se:<6.1f}{mk}"
            print(f"{nm:>18} {1e4*g.mean():>13.1f}bp {mL.mean():>8.2f} {ru:>4}/{n_tr} {gap:>20}")
        print()
    print("⚠️단위는 **거래당 bp 로그성장**. Δ 는 같은 평균 L 의 상수 대비(씨드 짝지음).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
