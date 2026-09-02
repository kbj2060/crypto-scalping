#!/usr/bin/env python3
"""B3: 진입 정책 대조군 5종 (2026-09-03).

후보 정책: 트리거마다 양팔 지정가(매수 −3·ATR / 매도 +3·ATR, 대기 6봉) → 1슬롯 순차.

기계+트리거 검정 (모델 없는 기본 정책에)
  ① 무작위 봉      -- 같은 개수의 무작위 봉에서 같은 기계. 트리거가 값을 더하나?
  ② 순환이동 플라시보 -- 트리거 위치만 원형이동. 클러스터 구조 보존, 가격 정렬만 깨짐.
  ③ 모멘텀 뒤집기   -- 지정가를 반대편에(매수를 위, 매도를 아래) = 페이드가 아니라 추격.
                     페이드가 진짜면 이건 져야 한다.
모델 검정 (필터 정책에)
  ④ 시간블록 군집 부트스트랩 -- 같은 급락에서 여러 체결이 상관되므로 **일 단위 블록**으로 리샘플
  ⑤ 무작위 5시드 부호 일치

⚠️HOLDOUT도 표시하되 판정은 VAL/OOS로 한다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)
from research_eth_entry_direction_oracle_ceiling_20260903 import (  # noqa: E402
    DEPTH, WAIT, NOTIONAL, COST, trail_out)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT = ROOT / "tmp/eth_entry_b3_controls_20260903"
B = 20
RNG = np.random.default_rng(20260903)
WINDOWS = (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START))


def log(m): print(f"[b3] {m}", flush=True)


def simulate(trig, atr, h, l, c, hz_map, n, momentum=False):
    """trig: (pos, side, signal). 양팔 지정가 → 체결 목록. 반환 DataFrame."""
    rows = []
    for i, sd, sname in trig:
        a = atr[i]
        if not (np.isfinite(a) and a > 0): continue
        H = hz_map[sname]
        for armv, s in ((1, sd), (0, -sd)):
            # 페이드: 롱이면 아래 매수. 모멘텀: 롱이면 위 매수(추격)
            below = (s > 0) if not momentum else (s < 0)
            lim = c[i] * (1 - DEPTH * a) if below else c[i] * (1 + DEPTH * a)
            j = None
            for k in range(i + 1, min(i + 1 + WAIT, n)):
                if (l[k] <= lim) if below else (h[k] >= lim):
                    j = k; break
            if j is None or j + H >= n: continue
            mv = trail_out(s, lim, a, h[j:j + H], l[j:j + H], c[j:j + H])
            rows.append({"fill_i": j, "exit_i": j + H, "arm": armv,
                         "y": float(mv * NOTIONAL - COST * NOTIONAL)})
    return pd.DataFrame(rows)


def slot1(df):
    if df.empty: return np.array([])
    d = df.sort_values("fill_i")
    taken, busy = [], []
    for fi, ei, y in zip(d.fill_i, d.exit_i, d.y):
        busy = [b for b in busy if b > fi]
        if not busy:
            taken.append(y); busy.append(ei)
    return np.asarray(taken, float)


def mstat(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (0, 0.0)
    return (len(v), float(v.mean() * 1e4))


def main() -> int:
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    src_k = load_klines(); ind = build_indicator_frame(src_k)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(kl["timestamp"])
    h, l, c = (kl[k].to_numpy(float) for k in ("high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(ts)}
    n = len(kl)
    at = pd.DataFrame({"ts": src_k["timestamp"], "atr": ind["atr_pct"].to_numpy()})
    at = at[at.ts.isin(pos_of)]
    atr = np.full(n, np.nan); atr[[pos_of[t] for t in at.ts]] = at.atr.to_numpy()

    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    hz = {k: int(v["horizon"]) for k, v in cfg.items()}
    trig = []
    for name in cfg:
        d = pd.read_csv(SRC / f"{name}_causal_fires.csv", parse_dates=["timestamp"])
        d = d[d.timestamp.isin(pos_of)]
        for t, s in zip(d.timestamp, d.side):
            trig.append((pos_of[t], 1 if s == "bottom" else -1, name))
    trig.sort()
    log(f"트리거 {len(trig):,} | klines {n:,}봉")

    def win(df, lo, hi):
        lo_i, hi_i = int(ts.searchsorted(lo)), int(ts.searchsorted(hi))
        return df[(df.fill_i >= lo_i) & (df.fill_i < hi_i)]

    real = simulate(trig, atr, h, l, c, hz, n)
    log(f"실제 체결 {len(real):,}")
    base = {}
    for wn, lo, hi in WINDOWS + (("HOLDOUT", HOLDOUT_START, ts.max()),):
        v = slot1(win(real, lo, hi)); nn, m = mstat(v); base[wn] = m
        log(f"  실제 {wn:8s} 1슬롯 n={nn:4d} {m:+7.2f}bp")

    # ---- ③ 모멘텀 뒤집기 ----
    log("\n=== ③ 모멘텀 뒤집기 (지정가를 반대편에 = 페이드 아닌 추격) ===")
    mom = simulate(trig, atr, h, l, c, hz, n, momentum=True)
    for wn, lo, hi in WINDOWS + (("HOLDOUT", HOLDOUT_START, ts.max()),):
        v = slot1(win(mom, lo, hi)); nn, m = mstat(v)
        log(f"  모멘텀 {wn:8s} 1슬롯 n={nn:4d} {m:+7.2f}bp   (실제 {base[wn]:+7.2f})")

    # ---- ① 무작위 봉 ----
    log(f"\n=== ① 무작위 봉 대조군 (B={B}) ===")
    valid = np.flatnonzero(np.isfinite(atr) & (atr > 0))
    valid = valid[(valid > 300) & (valid < n - 300)]
    names = [t[2] for t in trig]
    rnd = {wn: [] for wn, _, _ in WINDOWS}
    for b in range(B):
        pos = RNG.choice(valid, size=len(trig), replace=False)
        sides = RNG.choice([1, -1], size=len(trig))
        rt = sorted(zip(pos.tolist(), sides.tolist(), names))
        df = simulate(rt, atr, h, l, c, hz, n)
        for wn, lo, hi in WINDOWS:
            _, m = mstat(slot1(win(df, lo, hi))); rnd[wn].append(m)
    for wn, _, _ in WINDOWS:
        a = np.array(rnd[wn])
        log(f"  {wn:5s} 무작위 평균 {a.mean():+6.2f}bp (95% [{np.quantile(a,.025):+.2f},{np.quantile(a,.975):+.2f}]) "
            f"vs 실제 {base[wn]:+6.2f}  → 백분위 {float((a<base[wn]).mean()):.0%}")

    # ---- ② 순환이동 플라시보 ----
    log(f"\n=== ② 순환이동 플라시보 (B={B}) ===")
    sh = {wn: [] for wn, _, _ in WINDOWS}
    for b in range(B):
        off = int(RNG.integers(3000, n - 3000))
        st = sorted(((p + off) % n, s, nm) for p, s, nm in trig)
        st = [(p, s, nm) for p, s, nm in st if 300 < p < n - 300]
        df = simulate(st, atr, h, l, c, hz, n)
        for wn, lo, hi in WINDOWS:
            _, m = mstat(slot1(win(df, lo, hi))); sh[wn].append(m)
    for wn, _, _ in WINDOWS:
        a = np.array(sh[wn])
        log(f"  {wn:5s} 이동 평균 {a.mean():+6.2f}bp (95% [{np.quantile(a,.025):+.2f},{np.quantile(a,.975):+.2f}]) "
            f"vs 실제 {base[wn]:+6.2f}  → p={float((a>=base[wn]).mean()):.3f}")

    # ---- ④ 시간블록 군집 부트스트랩 (모델 없는 기본 정책) ----
    log("\n=== ④ 시간블록(일 단위) 군집 부트스트랩, 기본 정책 ===")
    for wn, lo, hi in WINDOWS:
        w = win(real, lo, hi)
        v = slot1(w)
        d2 = w.sort_values("fill_i").iloc[:len(v)].copy(); d2["y2"] = v
        d2["day"] = (d2.fill_i // 288).astype(int)
        days = d2.day.unique()
        bs = []
        for _ in range(2000):
            pick = RNG.choice(days, size=len(days), replace=True)
            vals = np.concatenate([d2.loc[d2.day == dd, "y2"].to_numpy() for dd in pick])
            bs.append(vals.mean() * 1e4)
        bs = np.array(bs)
        log(f"  {wn:5s} {base[wn]:+6.2f}bp  95%CI [{np.quantile(bs,.025):+.2f}, {np.quantile(bs,.975):+.2f}]  "
            f"블록 {len(days)}일 · P(>0)={float((bs>0).mean()):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"base": base, "random_bar": {k: list(map(float, v)) for k, v in rnd.items()},
               "circular_shift": {k: list(map(float, v)) for k, v in sh.items()}, "B": B},
              open(OUT / "controls.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
