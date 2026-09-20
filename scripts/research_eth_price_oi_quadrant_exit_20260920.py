"""«가격↓ + OI↓ 면 숏을 거둔다» — 가격×OI 사분면을 진입 신호와 **청산 규칙** 둘로 검정한다 (2026-09-20).

사용자: *"가격이 떨어지면서 OI 가 줄어들면 숏을 거둔다 같은 신호도 포함되어 있나?"*

**절반만 했었다.** 09-20 1초 패널에서 사분면을 쟀지만 ①OI 표본이 **28시간**뿐이었고
②**진입 신호로만** 쟀다(앞 수익). 사용자의 규칙은 **청산**이다 — 이미 숏인 상태에서 언제 거둘까.
그래서 여기서는 4.7년 5분 패널(`data/binance_vision/panel/ETHUSDT.parquet`, 2022-01~2026-09,
495,936행)로 둘 다 다시 잰다.

사전등록:
  A 진입   사분면별 앞 H 수익. 판정 = 날짜블록 CI95 0배제 AND 연도셀 부호 일치
  B 청산   **하락 중에만** 조건을 걸고 ΔOI 부호로 가른다. 사용자의 규칙이 참이면
           «하락 & OI↓» 의 앞 수익이 «하락 & OI↑» 보다 **유의하게 높아야** 한다(되돌림).
  C 증분   🔴결정적 통제 — «하락» 자체가 이미 앞 수익을 갖는다(평균회귀/추세). 그러니
           **사분면 평균이 아니라 «같은 가격변화 안에서 OI 가 더하는 몫»** 을 봐야 한다.
           |Δ가격| 십분위 안에서 OI↓ − OI↑ 차이를 내고, 그 차이에 CI 를 붙인다.

🔴OI 스탬프 주의: `openInterestHist` 는 5분봉 **시작** 값이다(2026-09-19 실측). 결정 시점 t 의
   종가와 같은 스탬프의 OI 를 쓰면 OI 쪽이 더 **오래된** 정보라 미래참조가 아니다(보수적).
   앞 수익은 t+1 봉부터 센다.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/rt_probe_20260920/price_oi_quadrant.json"
RNG = np.random.default_rng(20260920)
BAR_M = 5
HOR = {"15m": 3, "1h": 12, "4h": 48, "12h": 144}     # 관측창 = 예측창 (같은 H)
R: dict = {}


def say(*a) -> None:
    print(" ".join(str(x) for x in a), flush=True)


def _day_sums(vals: np.ndarray, codes: np.ndarray, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """날별 (합, 개수). 부트스트랩을 O(관측) 이 아니라 **O(날)** 로 만든다 --
    날마다 마스크를 다시 도는 구조는 495k행에서 끝나지 않는다(첫 판에서 실제로 안 끝났다)."""
    return (np.bincount(codes, weights=vals, minlength=n_days), np.bincount(codes, minlength=n_days))


def block_boot(vals: np.ndarray, codes: np.ndarray, n_days: int, n: int = 1500) -> tuple[float, float]:
    """날짜 블록 부트 CI95. 겹치는 창이라 같은 날 관측은 독립이 아니다."""
    ssum, scnt = _day_sums(vals, codes, n_days)
    live = np.flatnonzero(scnt > 0)
    if len(live) < 20:
        return (np.nan, np.nan)
    ss, cc = ssum[live], scnt[live]
    pick = RNG.integers(0, len(live), size=(n, len(live)))
    tot, cnt = ss[pick].sum(axis=1), cc[pick].sum(axis=1)
    return tuple(np.percentile(tot / np.maximum(cnt, 1), [2.5, 97.5]))


def diff_boot(va: np.ndarray, ca: np.ndarray, vb: np.ndarray, cb: np.ndarray, n_days: int,
              n: int = 1500) -> tuple[float, float]:
    """같은 날을 함께 뽑아 두 집단 평균 차이의 CI95(짝지은 날짜 블록)."""
    sa, na = _day_sums(va, ca, n_days)
    sb, nb = _day_sums(vb, cb, n_days)
    live = np.flatnonzero((na > 0) & (nb > 0))
    if len(live) < 20:
        return (np.nan, np.nan)
    pick = RNG.integers(0, len(live), size=(n, len(live)))
    A = sa[live][pick].sum(axis=1) / np.maximum(na[live][pick].sum(axis=1), 1)
    B = sb[live][pick].sum(axis=1) / np.maximum(nb[live][pick].sum(axis=1), 1)
    return tuple(np.percentile(A - B, [2.5, 97.5]))


d = pd.read_parquet(ROOT / "data/binance_vision/panel/ETHUSDT.parquet")
d["timestamp"] = pd.to_datetime(d["timestamp"])
d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
d = d[d.sum_open_interest.notna() & d.close.notna()].reset_index(drop=True)
d["day"] = d.timestamp.dt.date.astype(str)
d["year"] = d.timestamp.dt.year
say(f"패널 {len(d):,}행 · {d.timestamp.min()} ~ {d.timestamp.max()} · 고유일 {d.day.nunique():,}")

px = d.close.to_numpy(dtype="float64")
oi = d.sum_open_interest.to_numpy(dtype="float64")
day_codes, day_uniq = pd.factorize(d.day)          # 날 -> 정수코드(부트를 벡터화하려면 필요하다)
N_DAYS = len(day_uniq)
days = d.day.to_numpy()
years = d.year.to_numpy()

for name, h in HOR.items():
    d[f"dp_{name}"] = (np.log(px) - np.log(np.r_[np.full(h, np.nan), px[:-h]])) * 1e4
    d[f"do_{name}"] = (oi - np.r_[np.full(h, np.nan), oi[:-h]]) / np.r_[np.full(h, np.nan), oi[:-h]] * 1e4
    # 앞 수익은 **다음 봉부터** h 봉 (t 에서 결정하고 t+1 에 들어간다)
    fwd = np.r_[px[h + 1:], np.full(h + 1, np.nan)]
    d[f"fwd_{name}"] = (np.log(fwd) - np.log(np.r_[px[1:], np.full(1, np.nan)])) * 1e4

# ── A. 진입 — 사분면별 앞 수익 ───────────────────────────────────────────────
say("\n" + "=" * 100)
say("## A. 진입 신호로서의 사분면 (관측 H = 예측 H, 앞 수익 bp, 날짜블록 CI95)")
say(f"{'창':6} {'사분면':22} {'n':>9} {'점유':>6} {'앞 수익':>10} {'CI95':>20} {'연도 부호':>12}")
R["entry"] = {}
for name, h in HOR.items():
    dp, do, fw = d[f"dp_{name}"].to_numpy(), d[f"do_{name}"].to_numpy(), d[f"fwd_{name}"].to_numpy()
    ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(fw)
    # 유의미한 움직임만 (|Δ가격| 중앙 초과) -- 0 근처는 사분면 자체가 뜻이 없다
    big = np.abs(dp) > np.nanmedian(np.abs(dp[ok]))
    for lab, m in (("가격↑ OI↑ 신규 롱", (dp > 0) & (do > 0)), ("가격↑ OI↓ 숏 커버", (dp > 0) & (do < 0)),
                   ("가격↓ OI↑ 신규 숏", (dp < 0) & (do > 0)), ("가격↓ OI↓ 롱 이탈", (dp < 0) & (do < 0))):
        sel = ok & big & m
        v, cc, yy = fw[sel], day_codes[sel], years[sel]
        lo, hi = block_boot(v, cc, N_DAYS)
        ysign = [int(np.sign(v[yy == y].mean())) for y in np.unique(yy) if (yy == y).sum() > 500]
        agree = f"{sum(1 for s in ysign if s > 0)}/{len(ysign)}"
        R["entry"].setdefault(name, {})[lab] = {"n": int(sel.sum()), "mean": float(v.mean()), "ci": [lo, hi], "years": agree}
        say(f"{name:6} {lab:22} {int(sel.sum()):9,} {sel.mean():6.1%} {v.mean():+10.2f} [{lo:+8.2f},{hi:+8.2f}] {agree:>12}")
    say("")

# ── B. 청산 규칙 — 하락 중에만, ΔOI 로 가른다 ────────────────────────────────
say("=" * 100)
say("## B. 청산 규칙 — «하락 중»에만 걸고 ΔOI 부호로 가른다 (사용자 규칙)")
say("   숏 보유 중이라면 앞 수익이 **양수**로 갈수록 거두는 게 맞다(되돌림).")
say(f"{'창':6} {'상태':26} {'n':>9} {'앞 수익':>10} {'CI95':>20} {'연도 부호':>10}")
R["exit"] = {}
for name, h in HOR.items():
    dp, do, fw = d[f"dp_{name}"].to_numpy(), d[f"do_{name}"].to_numpy(), d[f"fwd_{name}"].to_numpy()
    ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(fw)
    down = ok & (dp < np.nanquantile(dp[ok], 0.25))          # 하락 상위 25% = 숏이 수익 중
    for lab, m in ((f"하락 & OI↓ (거둔다)", down & (do < 0)), (f"하락 & OI↑ (유지)", down & (do > 0))):
        v, cc, yy = fw[m], day_codes[m], years[m]
        lo, hi = block_boot(v, cc, N_DAYS)
        ysign = [int(np.sign(v[yy == y].mean())) for y in np.unique(yy) if (yy == y).sum() > 300]
        R["exit"].setdefault(name, {})[lab] = {"n": int(m.sum()), "mean": float(v.mean()), "ci": [lo, hi]}
        say(f"{name:6} {lab:26} {int(m.sum()):9,} {v.mean():+10.2f} [{lo:+8.2f},{hi:+8.2f}] "
            f"{sum(1 for s in ysign if s > 0)}/{len(ysign):>8}")
    # 차이(OI↓ − OI↑)에 직접 CI
    a, b = down & (do < 0), down & (do > 0)
    lo, hi = diff_boot(fw[a], day_codes[a], fw[b], day_codes[b], N_DAYS)
    gap = fw[a].mean() - fw[b].mean()
    R["exit"][name]["차이"] = {"mean": float(gap), "ci": [float(lo), float(hi)]}
    say(f"{name:6} {'  └ 차이 (OI↓ − OI↑)':26} {'':9} {gap:+10.2f} [{lo:+8.2f},{hi:+8.2f}]")
    say("")

# ── C. 🔴결정적 통제 — 같은 가격변화 안에서 OI 가 더하는 몫 ───────────────────
say("=" * 100)
say("## C. 통제 — «하락» 자체가 이미 앞 수익을 갖는다. |Δ가격| 십분위 안에서 OI 가 더하는 몫만 본다")
say(f"{'창':6} {'하락 전체 앞 수익':>18} {'십분위 내 OI↓−OI↑':>20} {'CI95':>22}")
R["ctrl"] = {}
for name, h in HOR.items():
    dp, do, fw = d[f"dp_{name}"].to_numpy(), d[f"do_{name}"].to_numpy(), d[f"fwd_{name}"].to_numpy()
    ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(fw)
    down = ok & (dp < np.nanquantile(dp[ok], 0.25))
    base = fw[down].mean()
    # |Δ가격| 십분위 안에서만 OI 부호로 가른다(하락 «크기»를 고정)
    q = pd.qcut(pd.Series(dp[down]).rank(method="first"), 10, labels=False).to_numpy()
    fwd_d, do_d = fw[down], do[down]
    code_d = day_codes[down]
    per_cell, av, ac, bv, bc = [], [], [], [], []
    for c in range(10):
        cm = q == c
        a, b = cm & (do_d < 0), cm & (do_d > 0)
        if a.sum() > 200 and b.sum() > 200:
            per_cell.append(fwd_d[a].mean() - fwd_d[b].mean())
            av.append(fwd_d[a]); ac.append(code_d[a]); bv.append(fwd_d[b]); bc.append(code_d[b])
    gap = float(np.mean(per_cell)) if per_cell else np.nan
    lo, hi = (diff_boot(np.concatenate(av), np.concatenate(ac),
                        np.concatenate(bv), np.concatenate(bc), N_DAYS) if av else (np.nan, np.nan))
    R["ctrl"][name] = {"base": float(base), "gap": gap, "ci": [lo, hi], "cells": [round(x, 2) for x in per_cell]}
    say(f"{name:6} {base:+18.2f} {gap:+20.2f} [{lo:+9.2f},{hi:+9.2f}]")
    say(f"       십분위별 차이: {[round(x, 1) for x in per_cell]}")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
say("\n->" + str(OUT))

# ── D. 1시간 창이 살아남았다 — 견고성 셋 ─────────────────────────────────────
say("\n" + "=" * 100)
say("## D. 견고성 — 1h 창의 «하락 중 OI↓ − OI↑» 차이")
name, h = "1h", HOR["1h"]
dp, do, fw = d[f"dp_{name}"].to_numpy(), d[f"do_{name}"].to_numpy(), d[f"fwd_{name}"].to_numpy()
ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(fw)
down = ok & (dp < np.nanquantile(dp[ok], 0.25))
a, b = down & (do < 0), down & (do > 0)
say("\n### D-1 연도별 (통제 전 차이)")
for y in np.unique(years[down]):
    ya, yb = a & (years == y), b & (years == y)
    if ya.sum() > 300 and yb.sum() > 300:
        say(f"  {y}  OI↓ {fw[ya].mean():+7.2f} · OI↑ {fw[yb].mean():+7.2f} · 차이 {fw[ya].mean() - fw[yb].mean():+7.2f}"
            f"  (n {int(ya.sum()):,}/{int(yb.sum()):,})")
say("\n### D-2 OI 를 한 봉 더 지연 (초보수적 -- 스탬프 규약 의심 해소)")
oi_lag = np.r_[np.nan, oi[:-1]]
do_lag = (oi_lag - np.r_[np.full(h, np.nan), oi_lag[:-h]]) / np.r_[np.full(h, np.nan), oi_lag[:-h]] * 1e4
ok2 = np.isfinite(dp) & np.isfinite(do_lag) & np.isfinite(fw)
down2 = ok2 & (dp < np.nanquantile(dp[ok2], 0.25))
a2, b2 = down2 & (do_lag < 0), down2 & (do_lag > 0)
lo, hi = diff_boot(fw[a2], day_codes[a2], fw[b2], day_codes[b2], N_DAYS)
say(f"  차이 {fw[a2].mean() - fw[b2].mean():+7.2f} [{lo:+.2f},{hi:+.2f}]  (n {int(a2.sum()):,}/{int(b2.sum()):,})")
say("\n### D-3 «거둔다»의 값 — 숏 보유 중 1시간 더 들고 갈 때의 기대 (부호 뒤집어 숏 손익)")
say(f"  하락 & OI↑ (유지) : 숏 손익 {-fw[b].mean():+6.2f}bp  ⇒ 더 벌린다")
say(f"  하락 & OI↓ (거둔다): 숏 손익 {-fw[a].mean():+6.2f}bp  ⇒ 되돌린다")
say(f"  거둘 때 피하는 손실 = {-(-fw[a].mean()):+6.2f}bp · 두 상태의 격차 {(-fw[b].mean()) - (-fw[a].mean()):+6.2f}bp")
say(f"  ⇒ OI↓ 에서 거두면 기대 −1.24bp 를 피하고 청산비용 ~0.7bp 를 낸다 ⇒ 트리거당 순 ~+0.5bp. 얇다.")
say(f"  ⇒ 값의 대부분은 «거두기»가 아니라 «OI↑ 면 계속 들고 있기»에 있다(+2.28bp/시간).")
R["robust"] = {"lag1": {"gap": float(fw[a2].mean() - fw[b2].mean()), "ci": [lo, hi]},
               "short_pnl_hold_oi_up": float(-fw[b].mean()), "short_pnl_hold_oi_down": float(-fw[a].mean())}
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
