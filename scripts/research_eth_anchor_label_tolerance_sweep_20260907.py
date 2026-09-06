#!/usr/bin/env python3
"""앵커 구조 라벨 — **시점 오차(E,D) x 가격 오차(P) x 유지창(W)** 허용범위 스윕 (2026-09-07).

사용자 지정 격자: E,D <= 3 · P <= 0.1% · W <= 24 · 최소 반등 g = 0.5%.

## 라벨 정의
바닥 앵커 t. 봉 b 가 **극점 후보** <=>
    low[b] <= min(low[b .. b+W]) * (1 + P/100)            # W 유지창, P 가격 오차
    AND (max(high[b .. b+W]) - low[b]) / low[b] * 100 >= g # 반등 게이트
라벨 y = 1  <=>  극점 후보가 **[t-E, t+D]** 안에 하나라도 있음.
  E = 앵커가 극점보다 **늦어도** 인정 · D = 앵커가 극점보다 **일러도** 인정.
천장은 high/min-max 를 뒤집는다.

⚠️2026-09-07 정정: 이전 판 `ext_label` 은 `out[:n-D] = a[D:]` 로 라벨을 **D만큼 밀어**
[t+D, t+2D] 를 봤다. 그 위에서 나온 "확정 셀 D=1~2"(부록 B)와 부록 C/E 의 구조 표는 무효다.

## 판정 (실행 전 고정)
  ① ATR 십분위 매칭 귀무 대비 초과, 일군집 CI 하한 > 0 -- VAL·OOS 두 창
  ② 현행 first_fire 대비 차이, 일군집 CI 하한 > 0     -- VAL·OOS 두 창
  ③ n >= 30 이고 서로 다른 날 >= 10
  ④ **연속 영역**: 인접 셀(E±1 · D±1 · P 한 단계)이 최소 2개 이상 같이 통과
다중도 잣대 = 시간이동 플라시보(신호별 원형이동 ±3~30일) R=10 의 통과 셀 수 분포.

## 병기 진단 (셀 선택을 숫자 하나로 하지 않기 위해)
  진입비용   open[t+1] 부터 그 극점까지 먹히는 역행 %(중앙/p75/p90)
  잔여상승   진입 후 W봉 안 최대 상승 %
  측면특이성 저점 초과 - 고점 초과 (바닥 앵커가 '저점만' 잡는가, '극점 아무거나' 잡는가)
"""
from __future__ import annotations

import itertools
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
import research_eth_bottom_anchor_wd_grid_20260907 as G  # noqa: E402
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_tolerance_sweep_20260907"
E_GRID, D_GRID = (0, 1, 2, 3), (0, 1, 2, 3)
P_GRID = (0.0, 0.05, 0.10)
W_GRID = (12, 24)
G_REBOUND = 0.5
ANCHOR, ANCHOR_WC = "any3", 3
MIN_N, MIN_DAYS = 30, 10
PLACEBO_R = 10
SEED = 20260907
WINS = {"TRAIN": ("2024-01-01", "2025-08-31 23:59:59"),
        "VAL": ("2025-09-01", "2025-12-31 23:59:59"),
        "OOS": ("2026-01-01", "2026-03-31 23:59:59")}


def load_panel():
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL); fund = B._load_funding()
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    from live_evidence_signal_dashboard_20260823 import compute_signals
    sig = compute_signals(eth, btc_df=btc[btc["timestamp"] <= tmax], funding_df=fund[fund["calc_time"] <= tmax])
    ts = pd.to_datetime(sig["timestamp"].to_numpy())
    P = {"ts": ts, "n": len(sig), "day": pd.Series(ts).dt.floor("D").to_numpy()}
    for c in ("open", "high", "low", "close"):
        P[c] = sig[c].to_numpy(float)
    P["atr"] = sig["atr_pct"].to_numpy(float)
    P["dec"] = pd.qcut(pd.Series(P["atr"]).rank(method="first"), 10, labels=False, duplicates="drop").to_numpy(float)
    P["fire"] = {s: np.stack([sig[f"{s}_{x}"].fillna(False).to_numpy(bool) for x in S.SIGNALS], axis=1)
                 for s in ("bottom", "top")}
    return P


def cand_mask(P, side, W, Ppct, g):
    """극점 후보 봉."""
    n = P["n"]; hi, lo = P["high"], P["low"]
    fmin = pd.Series(lo[::-1]).rolling(W + 1, min_periods=W + 1).min().to_numpy()[::-1]
    fmax = pd.Series(hi[::-1]).rolling(W + 1, min_periods=W + 1).max().to_numpy()[::-1]
    m = n - W
    q = np.zeros(n, bool)
    if side == "bottom":
        q[:m] = (lo[:m] <= fmin[:m] * (1 + Ppct / 100)) & (((fmax[:m] - lo[:m]) / lo[:m] * 100) >= g)
    else:
        q[:m] = (hi[:m] >= fmax[:m] * (1 - Ppct / 100)) & (((hi[:m] - fmin[:m]) / hi[:m] * 100) >= g)
    return q


def label(P, q, E, D, W):
    n = P["n"]; y = np.full(n, np.nan)
    lo, hi = max(E, 32), n - W - D - 2
    a = np.zeros(n, bool)
    for j in range(-E, D + 1):
        a[lo:hi] |= q[lo + j:hi + j]
    y[lo:hi] = a[lo:hi].astype(float)
    return y


def sweep(P, fire, rng, sides=("bottom", "top"), diag=True, verbose=True):
    ts, dec, day = P["ts"], P["dec"], P["day"]
    pools = {w: np.flatnonzero((ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))) for w, (a, b) in WINS.items()}
    anc = G.build_anchor_sets(P, fire)
    rows = []
    for side in sides:
        A = anc[(side, ANCHOR, ANCHOR_WC)]
        F = anc[(side, "first_fire_union", None)]
        for W, Ppct in itertools.product(W_GRID, P_GRID):
            q = cand_mask(P, side, W, Ppct, G_REBOUND)
            qo = cand_mask(P, "top" if side == "bottom" else "bottom", W, Ppct, G_REBOUND)
            for E, D in itertools.product(E_GRID, D_GRID):
                y = label(P, q, E, D, W)
                yo = label(P, qo, E, D, W)
                rec = {"side": side, "W": W, "P": Ppct, "E": E, "D": D, "base_rate": float(np.nanmean(y))}
                good = True
                for w in WINS:
                    p = pools[w]
                    def take(idx, yy):
                        i = idx[(idx >= p[0]) & (idx <= p[-1])]
                        v, d, dy = yy[i], dec[i], day[i]
                        k = np.isfinite(v) & np.isfinite(d)
                        return i[k], v[k], d[k].astype(int), dy[k]
                    bl = S.decile_baseline(y, dec, p)
                    ia, va, da, dya = take(A, y)
                    if len(va) < MIN_N or len(np.unique(dya)) < MIN_DAYS:
                        good = False
                        continue
                    exc = va - bl[da]
                    _, vf, df_, dyf = take(F, y)
                    excf = vf - bl[df_]
                    rec[f"{w}_n"] = len(va); rec[f"{w}_pos"] = float(va.mean()); rec[f"{w}_exc"] = float(exc.mean())
                    rec[f"{w}_vsff"] = float(exc.mean() - excf.mean())
                    if w != "TRAIN":
                        rec[f"{w}_exc_lo"], _ = S.day_ci(exc, dya, rng)
                        rec[f"{w}_vsff_lo"], _ = S.diff_day_ci(exc, dya, excf, dyf, rng)
                        if diag:
                            blo = S.decile_baseline(yo, dec, p)
                            _, vo, do, _ = take(A, yo)
                            nn = min(len(exc), len(vo))
                            sp = exc[:nn] - (vo[:nn] - blo[do[:nn]])
                            lo_, hi_ = S.day_ci(sp, dya[:nn], rng)
                            rec[f"{w}_side_gap"] = float(sp.mean()); rec[f"{w}_side_gap_lo"] = lo_
                rec["evaluable"] = good
                rec["PASS"] = bool(good and rec.get("VAL_exc_lo", -9) > 0 and rec.get("OOS_exc_lo", -9) > 0
                                   and rec.get("VAL_vsff_lo", -9) > 0 and rec.get("OOS_vsff_lo", -9) > 0)
                if diag and side == "bottom":
                    rec.update(entry_diag(P, A, q, E, D, W))
                rows.append(rec)
        if verbose:
            print(f"  {side} 완료", flush=True)
    return pd.DataFrame(rows)


def entry_diag(P, A, q, E, D, W):
    """진입 비용(open[t+1] -> 극점 역행)과 잔여 상승. 히트 앵커만."""
    n = P["n"]; O, H, L = P["open"], P["high"], P["low"]
    i = A[(A >= max(E, 32)) & (A + W + D + 2 < n)]
    mae, up = [], []
    for k in i:
        j = next((jj for jj in range(-E, D + 1) if q[k + jj]), None)
        if j is None:
            continue
        e = O[k + 1]
        end = max(k + 1, k + j)
        mae.append((e - L[k + 1:end + 1].min()) / e * 100 if end > k else 0.0)
        up.append((H[k + 1:k + 1 + W].max() - e) / e * 100)
    if len(mae) < 20:
        return {}
    mae, up = np.array(mae), np.array(up)
    return {"hit_n": len(mae), "mae_med": float(np.median(mae)), "mae_p90": float(np.percentile(mae, 90)),
            "up_med": float(np.median(up))}


def contiguous(df):
    """인접 셀(E±1 · D±1 · P 한 단계)이 2개 이상 같이 통과하는가."""
    key = {(r.side, r.W, r.P, r.E, r.D): r.PASS for r in df.itertuples()}
    pi = {p: k for k, p in enumerate(P_GRID)}
    out = []
    for r in df.itertuples():
        if not r.PASS:
            out.append(False); continue
        nb = 0
        for dE, dD, dP in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            pk = pi[r.P] + dP
            if not (0 <= pk < len(P_GRID)):
                continue
            nb += bool(key.get((r.side, r.W, P_GRID[pk], r.E + dE, r.D + dD), False))
        out.append(nb >= 2)
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    P = load_panel()
    print(f"[1/2] 실제 스윕 {len(E_GRID)}x{len(D_GRID)}x{len(P_GRID)}x{len(W_GRID)} · g={G_REBOUND}% ...", flush=True)
    real = sweep(P, P["fire"], rng)
    real["CONTIG"] = contiguous(real)
    real.to_parquet(OUT / "sweep_real.parquet")
    b = real[real.side == "bottom"]
    print(f"      바닥 PASS {int(b.PASS.sum())}/{len(b)} · 연속영역 {int(b.CONTIG.sum())}")
    print(f"[2/2] 플라시보 R={PLACEBO_R} ...", flush=True)
    pl = []
    for r in range(PLACEBO_R):
        pr = sweep(P, S.placebo_fire(P["fire"], rng), rng, sides=("bottom",), diag=False, verbose=False)
        pl.append({"rep": r, "PASS": int(pr.PASS.sum()), "evaluable": int(pr.evaluable.sum())})
        print(f"      rep {r+1}: {pl[-1]}", flush=True)
    pd.DataFrame(pl).to_csv(OUT / "placebo.csv", index=False)
    print(f"\n플라시보 PASS 평균 {np.mean([x['PASS'] for x in pl]):.2f} · 최대 {max(x['PASS'] for x in pl)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
