#!/usr/bin/env python3
"""진입 방향 모델(안 B)의 오라클 천장 진단 (2026-09-03).

사용자 결정: 방향을 규칙(신호 side)이 아니라 **모델**이 정하는 안 B로 간다.
설계 전에 천장부터 잰다 -- 체결 시점에서 방향을 완벽히 고를 수 있다면 얼마나 좋아지는가,
그리고 오라클은 실제로 몇 %나 뒤집는가.

  신호방향   : 신호가 제안한 방향 그대로 (현행, 안 A의 상한이기도 함)
  항상뒤집기 : 반대 방향 (기존 flip 대조군)
  ⭐오라클    : 건별로 max(롱, 숏, 0) -- 0은 기권
  오라클(뒤집기없음) : max(신호방향, 0) -- 기권만 허용, 뒤집기 불가 = **안 A의 오라클 천장**

⭐**완전 반사실**: 같은 가격 경로에서 롱/숏 결과가 **둘 다 계산된다.** 보통의 매매 학습과 달리
off-policy 보정이 필요 없다 -- 두 팔의 결과를 모두 안다. 이건 안 B에 유리한 구조적 사실이다.

진입은 지정가(깊이 3.0xATR, 대기 6봉). 청산은 **전 신호 공통** SL3.0/ARM1.0/Trail0.1을 쓴다 --
신호별 최적 설정을 쓰면 진단에 설정선택 편향이 섞이기 때문이다. horizon만 신호 고유값.
비용 10bp(보수). 1슬롯 미적용(진단이므로 체결 전수를 본다).
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

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT_DIR = ROOT / "tmp/eth_entry_direction_oracle_20260903"
DEPTH, WAIT = 3.0, 6
SL, ARM, TRAIL = 3.0, 1.0, 0.1      # 전 신호 공통 (설정선택 편향 배제)
MARGIN, LEV = 0.30, 3.0
NOTIONAL = MARGIN * LEV
COST = 0.0010


def log(m): print(f"[dir_oracle] {m}", flush=True)


def trail_out(side, e, a, hi, lo, cl):
    """side=+1 롱 / -1 숏. 반환: price_move (부호 있음, 수수료 전)."""
    if side > 0:
        stop = e * (1 - SL * a); peak = e; armed = False
        for k in range(len(cl)):
            if lo[k] <= stop:
                return stop / e - 1.0
            if hi[k] > peak:
                peak = hi[k]
                if not armed and (peak - e) / e >= ARM * a:
                    armed = True
                if armed:
                    stop = max(stop, peak * (1 - TRAIL * a))
        return cl[-1] / e - 1.0
    stop = e * (1 + SL * a); peak = e; armed = False
    for k in range(len(cl)):
        if hi[k] >= stop:
            return 1.0 - stop / e
        if lo[k] < peak:
            peak = lo[k]
            if not armed and (e - peak) / e >= ARM * a:
                armed = True
            if armed:
                stop = min(stop, peak * (1 + TRAIL * a))
    return 1.0 - cl[-1] / e


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(kl["timestamp"])
    o, h, l, c = (kl[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(ts)}
    n = len(kl)
    acct_cost = COST * NOTIONAL

    rows = []
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        f = SRC / f"{name}_causal_fires.csv"
        if not f.exists(): continue
        d = pd.read_csv(f, parse_dates=["timestamp"])
        d = d[d.timestamp.isin(pos_of)].copy()
        d["i"] = [pos_of[t] for t in d.timestamp]
        nf = 0
        for i, side_s, atr in zip(d["i"], d["side"], d["atr_pct"]):
            i = int(i)
            if not (np.isfinite(atr) and atr > 0): continue
            sig = 1 if side_s == "bottom" else -1
            lim = c[i] * (1 - DEPTH * atr) if sig > 0 else c[i] * (1 + DEPTH * atr)
            j = None
            for k in range(i + 1, min(i + 1 + WAIT, n)):
                if (l[k] <= lim) if sig > 0 else (h[k] >= lim):
                    j = k; break
            if j is None or j + H >= n: continue
            hi_, lo_, cl_ = h[j:j + H], l[j:j + H], c[j:j + H]
            rl = trail_out(+1, lim, atr, hi_, lo_, cl_) * NOTIONAL - acct_cost
            rs = trail_out(-1, lim, atr, hi_, lo_, cl_) * NOTIONAL - acct_cost
            rows.append({"signal": name, "ts": ts[j], "sig_dir": sig, "atr": atr,
                         "bars_to_fill": j - i,
                         "long_ret": rl, "short_ret": rs,
                         "sig_ret": rl if sig > 0 else rs,
                         "flip_ret": rs if sig > 0 else rl})
            nf += 1
        log(f"{name:26s} 체결 {nf:,}")

    df = pd.DataFrame(rows)
    df["split"] = np.where(df.ts < VAL_START, "TRAIN",
                    np.where(df.ts < OOS_START, "VAL",
                    np.where(df.ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    df["oracle_ret"] = np.maximum(np.maximum(df.long_ret, df.short_ret), 0.0)
    df["oracleA_ret"] = np.maximum(df.sig_ret, 0.0)      # 기권만 허용(뒤집기 없음) = 안 A 천장
    df["oracle_flips"] = (df.short_ret > df.long_ret) & (df.short_ret > 0) & (df.sig_dir > 0) | \
                         (df.long_ret > df.short_ret) & (df.long_ret > 0) & (df.sig_dir < 0)
    df["oracle_abstains"] = (df.long_ret <= 0) & (df.short_ret <= 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "fills_both_directions.csv", index=False)

    def stat(v):
        v = np.asarray(v)
        if len(v) == 0: return (0, 0.0, 0.0)
        w, ls = v[v > 0].sum(), -v[v < 0].sum()
        return (len(v), float(v.mean() * 1e4), float(w / ls) if ls > 0 else float("inf"))

    log(f"\n총 체결 {len(df):,}  " + " ".join(f"{k} {int(v):,}" for k, v in df.split.value_counts().items()))
    log("\n=== 정책별 건당 bp / PF ===")
    print(f"{'구간':9s} {'n':>6s} | " + " | ".join(f"{x:>16s}" for x in
          ("신호방향", "항상뒤집기", "오라클A(기권만)", "오라클B(뒤집기)")))
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        cells = []
        for col in ("sig_ret", "flip_ret", "oracleA_ret", "oracle_ret"):
            nn, m, pf = stat(w[col]); cells.append(f"{m:+7.2f}bp PF{pf:5.2f}")
        print(f"{wn:9s} {len(w):6,} | " + " | ".join(f"{x:>16s}" for x in cells))

    log("\n=== 오라클이 실제로 얼마나 뒤집나 ===")
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        log(f"  {wn:8s} 뒤집기 {float(w.oracle_flips.mean()):.1%} · 기권 {float(w.oracle_abstains.mean()):.1%} "
            f"· 신호방향유지 {float((~w.oracle_flips & ~w.oracle_abstains).mean()):.1%}")

    log("\n=== 뒤집기의 값어치 (오라클B − 오라클A) ===")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        gain = float((w.oracle_ret - w.oracleA_ret).mean() * 1e4)
        base = float(w.oracleA_ret.mean() * 1e4)
        log(f"  {wn:8s} +{gain:.2f}bp  (오라클A {base:.2f}bp 대비 +{gain/max(base,1e-9)*100:.0f}%)")
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
