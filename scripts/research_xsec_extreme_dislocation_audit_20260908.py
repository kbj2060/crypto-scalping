#!/usr/bin/env python3
"""극단 이탈 되돌림 **정밀 감사** -- 유동성·종목·날짜 집중도·중앙값 (2026-09-08).

## 무엇을 감사하는가
351셀 스크린에서 `L=3 · thr 6~8% · NU=60 · H=48~96` 이 네 창 전부 크게 양수였다
(TRAIN +25~46 · VAL +94~142 · OOS +93~155 · HOLDOUT +3~37bp). 비용(5.5~12bp)보다 한 자릿수 크다.
그러나 NU=40 으로 좁히면 TRAIN 이 +0.33 으로 무너진다 -> **효과가 비유동 꼬리에만 있을 가능성**.
비유동 종목의 실제 왕복 비용은 30~100bp 라 12bp 가정이 무의미해진다.

## 감사 항목
1. **NU 사다리** 10/20/30/40/50/60 -- 유동성 상위에서도 남는가
2. **종목 집중도** -- 상위 5종목이 총손익의 몇 %인가
3. **날짜 집중도** -- 상위 5일이 몇 %인가 (한두 번의 폭락장이 전부인가)
4. **중앙값 vs 평균** -- 소수 대박에 의존하는가
5. **비용 민감도** -- 왕복 12 / 30 / 60 / 100bp 에서 순bp
6. **호가폭 대리지표** -- 진입 종목의 1분봉 |고가−저가|/종가 중앙값
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
LIQW = 288
BOOT = 3000
SEED = 20260908


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def build(L, H, thr, NU, ts, Om, Cm, liq, day_all, win_of, syms):
    past = np.full_like(Cm, np.nan); past[L:] = Cm[L:] / Cm[:-L] - 1.0
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    tid = np.arange(max(L, LIQW) + 1, len(ts) - H - 2)
    el = (liq[tid] < NU) & np.isfinite(past[tid]) & np.isfinite(fwd[tid])
    pa = np.where(el, past[tid], np.nan); nval = np.isfinite(pa).sum(1)
    p_inf = np.where(np.isfinite(pa), pa, np.inf); p_ninf = np.where(np.isfinite(pa), pa, -np.inf)
    lo_a = np.argmin(p_inf, 1); hi_a = np.argmax(p_ninf, 1)
    r = np.arange(len(tid))
    spread = p_ninf[r, hi_a] - p_inf[r, lo_a]
    port = (fwd[tid][r, lo_a] - fwd[tid][r, hi_a]) / 2.0 * 1e4
    ev = (nval >= 8) & np.isfinite(spread) & (spread >= thr) & np.isfinite(port)
    keep = np.zeros(len(tid), bool); last = np.full(len(syms), -10**9)
    for i in np.flatnonzero(ev):
        if tid[i] - last[lo_a[i]] >= H: keep[i] = True; last[lo_a[i]] = tid[i]
    return tid[keep], port[keep], lo_a[keep], hi_a[keep]


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(OUT / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w

    print("=" * 104)
    print("1) NU 사다리 -- L=3, 각 (thr,H) 에서 유동성 상위 NU 만 썼을 때")
    print("=" * 104)
    print(f"{'thr':>5}{'H':>5}{'NU':>5} {'n':>6} {'TRAIN':>20} {'VAL':>9} {'OOS':>9} {'HOLD':>9} {'중앙':>8}")
    best = None
    for thr in (0.06, 0.08):
        for H in (48, 96):
            for NU in (10, 20, 30, 40, 50, 60):
                tt, v, la, ha = build(3, H, thr, NU, ts, Om, Cm, liq, day_all, win_of, syms)
                if len(tt) < 100: continue
                ww = win_of[tt]; dd = day_all[tt]
                out = {}
                for w in SPLITS:
                    m = ww == w
                    out[w] = (v[m].mean() if m.sum() >= 20 else np.nan, int(m.sum()))
                lo, hi = day_ci(v[ww == "TRAIN"], dd[ww == "TRAIN"], rng) if (ww == "TRAIN").sum() > 40 else (np.nan, np.nan)
                print(f"{thr:>5.2f}{H:>5}{NU:>5} {len(tt):>6} "
                      f"{out['TRAIN'][0]:>+8.1f}[{lo:>+6.1f},{hi:>+6.1f}] "
                      f"{out['VAL'][0]:>+9.1f} {out['OOS'][0]:>+9.1f} "
                      f"{out['HOLDOUT_SPENT'][0]:>+9.1f} {np.median(v):>+8.1f}", flush=True)
                if NU == 60 and thr == 0.06 and H == 96:
                    best = (tt, v, la, ha)

    tt, v, la, ha = best
    ww = win_of[tt]; dd = day_all[tt]
    print("\n" + "=" * 104)
    print("2~4) 대표 셀 L=3 · thr=6% · H=96 · NU=60 집중도")
    print("=" * 104)
    tot = v.sum()
    S = pd.Series(v).groupby(pd.Series([syms[i] for i in la])).agg(["sum", "count", "mean"])
    S = S.sort_values("sum", ascending=False)
    print(f"진입 종목 수 {S.shape[0]} · 상위5 종목이 총손익의 "
          f"{S['sum'].head(5).sum()/tot:.1%} · 상위10 {S['sum'].head(10).sum()/tot:.1%}")
    print(S.head(8).round(1).to_string())
    Dg = pd.Series(v).groupby(pd.Series(dd)).sum().sort_values(ascending=False)
    print(f"\n거래일 {len(Dg)} · 상위5일이 총손익의 {Dg.head(5).sum()/tot:.1%} · 상위20일 {Dg.head(20).sum()/tot:.1%}")
    print("상위 5일:", [f"{str(k)[:10]} {x:+.0f}bp" for k, x in Dg.head(5).items()])
    print(f"\n평균 {v.mean():+.1f}bp · 중앙값 {np.median(v):+.1f}bp · 승률 {(v>0).mean():.1%} · "
          f"상위1% 제거시 평균 {v[v < np.percentile(v,99)].mean():+.1f}bp")

    print("\n" + "=" * 104)
    print("5) 비용 민감도 (네 창 평균 순bp)")
    print("=" * 104)
    for C in (12, 30, 60, 100):
        line = f"  왕복 {C:>3}bp: "
        for w in SPLITS:
            m = ww == w
            if m.sum() < 20: continue
            nv = v[m] - C; lo, hi = day_ci(nv, dd[m], rng)
            line += f"{w[:5]} {nv.mean():>+7.1f}[{lo:>+6.1f}] "
        print(line, flush=True)

    print("\n" + "=" * 104)
    print("6) 진입 종목의 유동성 위치 (liq 순위, 0=최대)")
    print("=" * 104)
    lr = liq[tt, la]
    print(f"롱 다리 liq 순위: 중앙 {np.median(lr):.0f} · 25/75분위 {np.percentile(lr,[25,75])} · "
          f"상위20 안 비율 {(lr<20).mean():.1%}")
    hr = liq[tt, ha]
    print(f"숏 다리 liq 순위: 중앙 {np.median(hr):.0f} · 상위20 안 비율 {(hr<20).mean():.1%}")
    print(json.dumps({"n": int(len(tt))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
