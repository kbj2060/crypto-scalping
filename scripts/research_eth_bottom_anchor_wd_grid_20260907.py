#!/usr/bin/env python3
"""바닥 앵커 — 라벨 격자 (W, D) x 앵커 격자 (k, Wc) 안정성 스크린 (2026-09-07).

사용자: *"W와 D를 격자로 훑어서 바닥 앵커 최적 셀 확정해줘"*

## ⭐격자 설계 원칙 (실행 전 고정)
W(극점 유지 창)와 D(도달 허용 봉수)는 **라벨을 바꾸는 축**이다. 셀마다 기저율이 달라지므로
셀 간 정확도를 직접 비교해 argmax 를 고르면 `btc_eth_tuning_parity_no_replacement_20260903`
§5-H("격자 축이 라벨을 바꾸면 AUC 비교 금지", kalman 확장승자 히트율 0.509->0.058)에 정확히 걸린다.

그래서 두 축을 분리한다:
  라벨 축 (W, D)  = **정의 선택**. 최적화 대상이 아니라 **안정성 확인용 배경**이다.
  앵커 축 (k, Wc) = **선택 대상**. 각 라벨 셀 안에서 순위를 매기고, 그 순위가 라벨 격자
                    전체에서 유지되는지 본다. 승자가 라벨 셀마다 바뀌면 '최적 셀'은 아티팩트다.

## 판정 (사전 고정)
각 (라벨 셀 x 앵커) 에 대해 두 대조를 동시에 요구한다:
  C1  ATR 십분위 매칭 귀무 대비 초과 > 0        (VAL·OOS 둘 다 일군집 CI 하한 > 0)
  C2  현행 first_fire_union 대비 차이 > 0        (VAL·OOS 둘 다 일군집 CI 하한 > 0)
최종 선택은 **C1∧C2 통과 라벨 셀 비율(coverage)** 이 가장 높은 앵커. 단일 셀 argmax 금지.
다중도 잣대는 시간이동 플라시보(신호별 원형이동 +-3~30일) R=10 의 통과 셀 수 분포.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_combination_screen_20260907 as S  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_combo_20260907"
W_GRID = (12, 24, 48, 96, 192)
D_GRID = (0, 1, 2, 3, 4, 6, 9, 12)
ANCHORS = [("first_fire_union", None)] + [(f"any{k}", wc) for k in (2, 3, 4) for wc in (1, 3, 12)]
MIN_N, MIN_DAYS = 30, 10
PLACEBO_R = 10
SEED = 20260907


def ext_label(x: np.ndarray, W: int, D: int, how: str) -> np.ndarray:
    """전방향 극점: 봉 b 가 [b, b+W] 의 극값인 봉이 t..t+D 안에 있는가. 뒤를 안 본다."""
    n = len(x)
    r = pd.Series(x[::-1]).rolling(W + 1, min_periods=W + 1)
    f = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
    e = np.full(n, np.nan)
    e[:n - W] = ((x[:n - W] >= f[:n - W] - 1e-12) if how == "max"
                 else (x[:n - W] <= f[:n - W] + 1e-12)).astype(float)
    # ⚠️2026-09-07 수정: 이전 판은 `out[:n-D] = a[D:]` 로 라벨을 **D만큼 밀어** [t+D, t+2D] 를 봤다.
    # a[b] = max(e[b .. b+D]) 이므로 그대로가 정답이다. 꼬리 D봉만 미정으로 둔다.
    a = pd.Series(e[::-1]).rolling(D + 1, min_periods=1).max().to_numpy()[::-1]
    out = a.astype(float).copy()
    out[n - D:] = np.nan
    return out


def build_anchor_sets(P: dict, fire: dict) -> dict:
    out = {}
    for side in ("bottom", "top"):
        f = fire[side]
        ww = S.within_windows(f)
        out[(side, "first_fire_union", None)] = S.first_fire_union(f)
        for k in (2, 3, 4):
            for wc in (1, 3, 12):
                out[(side, f"any{k}", wc)] = S.anyk_anchors(f, ww[wc], k)
    return out


def run(P: dict, fire: dict, rng: np.random.Generator, sides=("bottom", "top"),
        verbose: bool = True) -> pd.DataFrame:
    ts, dec, day = P["ts"], P["dec"], P["day"]
    sig_low, sig_high = P["low"], P["high"]
    wins = {"TRAIN": (S.TRAIN_START, S.VAL_START - pd.Timedelta(seconds=1)),
            "VAL": (S.VAL_START, S.VAL_END), "OOS": (S.OOS_START, S.OOS_END)}
    pools = {w: np.flatnonzero((ts >= a) & (ts <= b)) for w, (a, b) in wins.items()}
    anc = build_anchor_sets(P, fire)
    rows = []
    for W in W_GRID:
        for D in D_GRID:
            for side in sides:
                y = ext_label(sig_low if side == "bottom" else sig_high, W, D,
                              "min" if side == "bottom" else "max")
                base = {w: S.decile_baseline(y, dec, pools[w]) for w in wins}
                ff = {}
                for w in ("VAL", "OOS"):
                    a = anc[(side, "first_fire_union", None)]
                    p = pools[w]; a = a[(a >= p[0]) & (a <= p[-1])]
                    yy, dd, dy = y[a], dec[a], day[a]
                    ok = np.isfinite(yy) & np.isfinite(dd)
                    ff[w] = (yy[ok] - base[w][dd[ok].astype(int)], dy[ok])
                for name, wc in ANCHORS:
                    rec = {"side": side, "W": W, "D": D, "anchor": name, "Wc": wc,
                           "base_rate": float(np.nanmean(y))}
                    ok_all = True
                    for w in wins:
                        a = anc[(side, name, wc)]
                        p = pools[w]; a = a[(a >= p[0]) & (a <= p[-1])]
                        yy, dd, dy = y[a], dec[a], day[a]
                        m = np.isfinite(yy) & np.isfinite(dd)
                        yy, dd, dy = yy[m], dd[m].astype(int), dy[m]
                        if len(yy) < MIN_N or len(np.unique(dy)) < MIN_DAYS:
                            ok_all = False
                            continue
                        exc = yy - base[w][dd]
                        rec[f"{w}_n"] = len(yy); rec[f"{w}_obs"] = float(yy.mean())
                        rec[f"{w}_null"] = float(base[w][dd].mean())
                        rec[f"{w}_exc"] = float(exc.mean())
                        if w != "TRAIN":
                            lo, hi = S.day_ci(exc, dy, rng)
                            rec[f"{w}_exc_lo"] = lo
                            d2, dd2 = ff[w]
                            l2, h2 = S.diff_day_ci(exc, dy, d2, dd2, rng)
                            rec[f"{w}_vsff"] = float(exc.mean() - d2.mean()); rec[f"{w}_vsff_lo"] = l2
                    rec["evaluable"] = ok_all
                    rec["C1"] = bool(ok_all and rec.get("VAL_exc_lo", -9) > 0 and rec.get("OOS_exc_lo", -9) > 0)
                    rec["C2"] = bool(ok_all and rec.get("VAL_vsff_lo", -9) > 0 and rec.get("OOS_vsff_lo", -9) > 0)
                    rec["PASS"] = rec["C1"] and rec["C2"]
                    rows.append(rec)
        if verbose:
            print(f"  W={W} 완료", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(SEED)
    P = S.build_panel()
    sig = pd.read_parquet(S.CACHE)
    P["low"], P["high"] = sig["low"].to_numpy(float), sig["high"].to_numpy(float)
    print(f"[1/2] 실제 격자 {len(W_GRID)}x{len(D_GRID)} x 앵커 {len(ANCHORS)} ...", flush=True)
    real = run(P, P["fire"], rng)
    real.to_parquet(OUT / "wd_grid_real.parquet")
    b = real[(real.side == "bottom") & (real.anchor != "first_fire_union")]
    print(f"      바닥 평가가능 {int(b.evaluable.sum())}/{len(b)} · C1 {int(b.C1.sum())} · C2 {int(b.C2.sum())} · PASS {int(b.PASS.sum())}")

    print(f"[2/2] 시간이동 플라시보 R={PLACEBO_R} ...", flush=True)
    pl = []
    for r in range(PLACEBO_R):
        pr = run(P, S.placebo_fire(P["fire"], rng), rng, sides=("bottom",), verbose=False)
        pb = pr[pr.anchor != "first_fire_union"]
        pl.append({"rep": r, "evaluable": int(pb.evaluable.sum()), "C1": int(pb.C1.sum()),
                   "C2": int(pb.C2.sum()), "PASS": int(pb.PASS.sum())})
        print(f"      rep {r+1}/{PLACEBO_R}: {pl[-1]}", flush=True)
    pd.DataFrame(pl).to_csv(OUT / "wd_grid_placebo.csv", index=False)
    print("\n플라시보 PASS: mean %.2f max %d" % (np.mean([x["PASS"] for x in pl]), max(x["PASS"] for x in pl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
