#!/usr/bin/env python3
"""진입만 연구 — **1시간 안 최고점 청산을 가정**하고 진입을 고른다 (2026-09-13).

사용자: *"진입점이 극점이기만 하면 청산은 1시간 이내 가장 좋은 청산점에서 했다고 가정하고
진입점을 잡는 거야."*

## 🔴선행 — 같은 프레이밍이 2026-09-11 에 이미 돌았다
[[eth_entry_exit_axes_closed_by_oracle_upper_bound_20260911]]: **ATR 십분위 매칭 무작위 진입**이
1시간 최고점 청산만으로 **44bp**. 증거신호 8종이 얹는 초과분은 128셀 중앙 IS −4.15 / VAL +0.55 /
OOS +0.91bp 이고, **신호를 안 쓰는 1피쳐 규칙(직전 1h 수익 하위 십분위)이 OOS 에서 앙상블을 이겼다**.
⇒ 「44bp」는 진입 실력이 아니라 **«최고점에 팔 수 있다»는 가정 자체**가 만든 값이다.

## 그래서 이번에 **새로 넣는 것만** 잰다
① 오늘 만든 **누수 없는 극점 확률**(clean-cut, 배포본은 판정창을 학습에 포함해 못 씀)
② **연속 하위값만**(ev_/trg_ 이진 제외, 2026-09-12 사용자 지시)
③ ⭐**상한을 직접 목적함수로** — MFE(1시간 최고점 수익)를 회귀해 상위를 고른다.
   지금까지는 방향/정확도를 학습하고 MFE 를 사후에 봤다. 목적함수를 상한에 맞춘 적이 없다.

## 대조군 (이게 이 실험의 전부다)
  A 무작위 진입 · **ATR 십분위 매칭**            ← 09-11 기준선(44bp)
  B **직전 1h 수익 하위 십분위** 1피쳐 규칙       ← 09-11 에서 앙상블을 이긴 규칙
  C 극점 확률 상위(누수 없는 판)
  D **MFE 회귀** 상위 (연속 하위값 41열, 전진 검증)
⚠️ATR 매칭 없이 비교하면 «변동성 큰 구간을 골랐다»를 진입 실력으로 착각한다(09-11 실측: 매칭
하나로 절반 증발). 구간 분할(2022~23 / 2024~)과 일블록 부트도 같이 낸다.

자체점검 --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_rule_direction_probability_20260912 as RD  # noqa: E402
import research_eth_stack_all_models_20260912 as S  # noqa: E402

H = 12                       # 1시간
COSTS = {"사용자 4bp": 4.0, "실측 peg 5.52bp": 5.52, "테이커 10bp": 10.0}
WARM, MIN_TRAIN = 900, 20_000


def oracle_mfe(p: pd.DataFrame, long: bool = True) -> np.ndarray:
    """진입 i+1 시가 → i+1..i+H 안 **최고점**(롱은 고가·숏은 저가) 청산 수익 bp.

    ⚠️이건 **상한**이다. 실제로 그 가격에 팔 수 있다는 보장이 없다 — 그래서 결과는
    「진입이 이만큼은 벌 수 있다」가 아니라 「이보다 더는 못 번다」로만 읽는다.
    """
    o = p["open"].to_numpy(float); hi = p["high"].to_numpy(float); lo = p["low"].to_numpy(float)
    n = len(p)
    ent = np.roll(o, -1); ent[-1] = np.nan
    best = np.full(n, np.nan)
    fwd_hi = pd.Series(hi).shift(-1).rolling(H, min_periods=H).max().shift(-(H - 1)).to_numpy()
    fwd_lo = pd.Series(lo).shift(-1).rolling(H, min_periods=H).min().shift(-(H - 1)).to_numpy()
    if long:
        best = (fwd_hi / ent - 1.0) * 1e4
    else:
        best = (1.0 - fwd_lo / ent) * 1e4
    return best


def atr_matched_random(atr: np.ndarray, sel: np.ndarray, pool: np.ndarray,
                       seed: int = 20260913) -> np.ndarray:
    """선택 집합과 **ATR 십분위 분포가 같은** 무작위 표본을 뽑는다."""
    rng = np.random.default_rng(seed)
    q = pd.qcut(pd.Series(atr[pool]), 10, labels=False, duplicates="drop").to_numpy()
    qs = pd.qcut(pd.Series(atr[pool]), 10, labels=False, duplicates="drop").to_numpy()
    want = pd.Series(qs[np.isin(pool, sel)]).value_counts()
    out = []
    for d, k in want.items():
        cand = pool[q == d]
        if len(cand) == 0:
            continue
        out.append(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.concatenate(out) if out else np.array([], int)


def report(name: str, idx: np.ndarray, mfe: np.ndarray, ts: np.ndarray, atr: np.ndarray,
           pool: np.ndarray, days: float) -> None:
    if len(idx) < 200:
        print(f"{name:<28}{len(idx):>8}  표본 부족")
        return
    v = mfe[idx]
    rnd = atr_matched_random(atr, idx, pool)
    base = float(np.nanmean(mfe[rnd])) if len(rnd) > 200 else float("nan")
    blk = idx // H
    ub = np.unique(blk)
    bm = np.array([v[blk == b].mean() for b in ub])
    rng = np.random.default_rng(20260913)
    bs = np.array([bm[rng.integers(0, len(bm), len(bm))].mean() for _ in range(1000)])
    e = pd.DatetimeIndex(ts[idx]) < pd.Timestamp("2024-01-01")
    a = float(v[e].mean()) if e.sum() > 200 else float("nan")
    b = float(v[~e].mean()) if (~e).sum() > 200 else float("nan")
    print(f"{name:<28}{len(idx):>8,}{len(idx)/days:>7.1f}{v.mean():>9.1f}{base:>10.1f}"
          f"{v.mean()-base:>9.2f}{np.percentile(bs,2.5)-base:>10.2f}{a:>9.1f}{b:>9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/materials/eth_signal_trigger_panel_ext_20260913")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    panel = Path(a.panel) / "panel_5m.parquet"
    if a.selftest:
        p = pd.DataFrame({"open": [10, 10, 10, 10, 10.0], "high": [10, 12, 11, 10, 10.0],
                          "low": [10, 9, 8, 10, 10.0], "close": [10.0] * 5})
        g = globals(); g["H"] = 2
        m = oracle_mfe(p, True)
        assert abs(m[0] - 2000.0) < 1e-6, m           # i+1..i+2 고가 12 / 진입 10 → +20%
        ms = oracle_mfe(p, False)
        # 창 i+1..i+2 의 저가는 min(9, 8) = 8 → 숏 최고점도 +20%
        assert abs(ms[0] - 2000.0) < 1e-6, ms
        assert abs(ms[2] - 0.0) < 1e-6, ms            # 창이 평평하면 0
        g["H"] = 12
        atr = np.tile(np.arange(10, dtype=float), 200)
        pool = np.arange(2000); sel = np.flatnonzero(atr >= 8)
        r = atr_matched_random(atr, sel, pool)
        assert abs(np.mean(atr[r]) - np.mean(atr[sel])) < 0.3, (np.mean(atr[r]), np.mean(atr[sel]))
        print("selftest OK — 오라클 MFE(롱/숏) · ATR 분위 매칭")
        return 0

    RD.PANEL = panel
    S.RD.PANEL = panel
    p = pd.read_parquet(panel)
    ts = p["timestamp"].to_numpy()
    atr = p["atr_pct"].to_numpy(float)
    mfe = oracle_mfe(p, True)
    n = len(p)
    ok = np.isfinite(mfe) & np.isfinite(atr)
    ok[:WARM] = False
    pool = np.flatnonzero(ok)
    days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days
    print(f"패널 {n:,}행 ({str(ts[0])[:10]} ~ {str(ts[-1])[:10]}) · 1시간 오라클 MFE\n")
    print(f"{'진입 규칙':<28}{'n':>8}{'건/일':>7}{'MFE bp':>9}{'ATR매칭':>10}{'초과':>9}"
          f"{'부트하한':>10}{'~2023':>9}{'2024~':>9}")

    report("A 무작위(= 매칭 기준선)", pool[::37], mfe, ts, atr, pool, days)
    r1h = (p["close"].to_numpy(float) / pd.Series(p["close"]).shift(H).to_numpy() - 1.0)
    q = pd.Series(r1h[pool]).rank(pct=True).to_numpy()
    report("B 직전1h수익 하위10% (09-11승자)", pool[q <= 0.10], mfe, ts, atr, pool, days)
    if "xp_bottom" in p.columns or True:
        X0, _ = S.build()
        for col, lab in (("xp_bottom", "C 극점확률 바닥 상위10%"),):
            if col in X0.columns:
                v = X0[col].to_numpy()
                m = np.isfinite(v) & ok
                thr = np.nanquantile(v[m], 0.90)
                report(lab, np.flatnonzero(m & (v >= thr)), mfe, ts, atr, pool, days)

    # D — MFE 를 직접 회귀(상한을 목적함수로). 전진 검증.
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, _ = S.build()
    ctx = [c for c in X.columns if not c.startswith(("ev_", "trg_"))]
    Xv = X[ctx].to_numpy(np.float32)
    months = pd.PeriodIndex(pd.to_datetime(ts), freq="M")
    pred = np.full(n, np.nan)
    for mo in [u for u in months.unique() if u >= pd.Period("2022-07", "M")]:
        te = (months == mo) & ok
        tr = (months < mo) & ok
        if tr.sum() < MIN_TRAIN or te.sum() < 200:
            continue
        mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6,
                                            random_state=0)
        mdl.fit(Xv[tr], mfe[tr])
        pred[te] = mdl.predict(Xv[te])
    pm = np.isfinite(pred)
    for qq, lab in ((0.90, "D MFE회귀 상위10%"), (0.99, "D MFE회귀 상위1%")):
        thr = np.nanquantile(pred[pm], qq)
        report(lab, np.flatnonzero(pm & (pred >= thr)), mfe, ts, atr, pool, days)
    # ⭐결정적 판별 — 변동성이면 롱·숏 MFE 가 **같이** 커지고, 진입 실력이면 **한쪽만** 커진다.
    # ATR 매칭은 «진입 봉» 변동성만 통제한다. MFE 는 **전방** 변동성으로 만들어지고 이 저장소
    # 스택은 그걸 잘 예측한다(스피어만 .79) — 그래서 MFE 회귀는 변동성 예측기가 되기 쉽다.
    mS = oracle_mfe(p, False)
    print(f"\n⭐판별: 변동성이면 롱·숏이 같이 커진다 · 진입 실력이면 한쪽만 커진다")
    print(f"{'집합':<22}{'n':>8}{'MFE롱':>9}{'MFE숏':>9}{'합(변동성)':>11}{'차(방향)':>10}{'차/합':>8}")
    for lab, ii in (("전체(기준)", pool),
                    ("MFE회귀 상위10%", np.flatnonzero(pm & (pred >= np.nanquantile(pred[pm], .90)))),
                    ("MFE회귀 상위1%", np.flatnonzero(pm & (pred >= np.nanquantile(pred[pm], .99)))),
                    ("MFE회귀 하위10%", np.flatnonzero(pm & (pred <= np.nanquantile(pred[pm], .10))))):
        L, Sh = float(mfe[ii].mean()), float(mS[ii].mean())
        print(f"{lab:<22}{len(ii):>8,}{L:>9.1f}{Sh:>9.1f}{L+Sh:>11.1f}{L-Sh:>10.2f}"
              f"{(L-Sh)/(L+Sh):>8.3f}")
    print(f"\n비용선: " + " · ".join(f"{k} {v}" for k, v in COSTS.items())
          + "\n⚠️MFE 는 **상한**이다 — 그 가격에 실제로 팔 수 있다는 보장이 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
