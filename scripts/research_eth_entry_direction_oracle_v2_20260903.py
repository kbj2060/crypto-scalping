#!/usr/bin/env python3
"""진입 방향 오라클 천장 v2 -- **실현 가능한 반사실**로 정정 (2026-09-03).

v1의 결함: 롱 체결가에서 숏 수익을 계산했다. 지정가는 한쪽으로만 걸리므로
(bottom 신호 → 3 ATR **아래 매수** 지정가) 체결되면 롱이고, 그 가격에서 숏이 될 수 없다.
실현 불가능한 반사실이라 천장이 부풀었다.

정정: 방향 선택은 **발주 시점의 선택**이다. 두 팔은 각자 다른 체결 사건을 갖는다.
  신호방향 팔 : bottom → close*(1 − 3·atr)에 매수 지정가, 체결되면 롱
  역방향  팔 : bottom → close*(1 + 3·atr)에 매도 지정가, 체결되면 숏
둘 다 대기 WAIT봉. 둘 다 체결될 수도(내렸다 올랐다), 둘 다 미체결일 수도 있다.

⭐이 형태는 **완전 반사실이 여전히 성립한다** — 같은 가격 경로에서 두 팔의 결과가 모두
계산되므로 off-policy 보정이 필요 없다. 다만 이제 "체결 여부"까지 반사실에 포함된다.

청산은 전 신호 공통 SL3.0/ARM1.0/Trail0.1(설정선택 편향 배제), horizon만 신호 고유.
비용 10bp. 1슬롯 미적용(진단).
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
    DEPTH, WAIT, SL, ARM, TRAIL, NOTIONAL, COST, trail_out)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT_DIR = ROOT / "tmp/eth_entry_direction_oracle_v2_20260903"


def log(m): print(f"[dir_v2] {m}", flush=True)


def arm(side, i, atr, H, h, l, c, n):
    """side=+1 매수지정가(아래) / -1 매도지정가(위). 반환 (체결여부, 체결봉, 순수익)."""
    lim = c[i] * (1 - DEPTH * atr) if side > 0 else c[i] * (1 + DEPTH * atr)
    j = None
    for k in range(i + 1, min(i + 1 + WAIT, n)):
        if (l[k] <= lim) if side > 0 else (h[k] >= lim):
            j = k; break
    if j is None or j + H >= n:
        return (False, -1, 0.0)
    mv = trail_out(side, lim, atr, h[j:j + H], l[j:j + H], c[j:j + H])
    return (True, j, float(mv * NOTIONAL - COST * NOTIONAL))


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(kl["timestamp"])
    h, l, c = (kl[k].to_numpy(float) for k in ("high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(ts)}
    n = len(kl)

    rows = []
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        f = SRC / f"{name}_causal_fires.csv"
        if not f.exists(): continue
        d = pd.read_csv(f, parse_dates=["timestamp"])
        d = d[d.timestamp.isin(pos_of)].copy()
        d["i"] = [pos_of[t] for t in d.timestamp]
        cnt = [0, 0, 0]
        for i, side_s, atr in zip(d["i"], d["side"], d["atr_pct"]):
            i = int(i)
            if not (np.isfinite(atr) and atr > 0): continue
            sig = 1 if side_s == "bottom" else -1
            fS, jS, rS = arm(sig, i, atr, H, h, l, c, n)       # 신호방향 팔
            fF, jF, rF = arm(-sig, i, atr, H, h, l, c, n)      # 역방향 팔
            if not (fS or fF): continue
            cnt[0] += fS; cnt[1] += fF; cnt[2] += (fS and fF)
            rows.append({"signal": name, "ts": ts[i], "sig_dir": sig, "atr": atr,
                         "sig_filled": fS, "flip_filled": fF,
                         "sig_ret": rS if fS else np.nan, "flip_ret": rF if fF else np.nan,
                         "bars_to_fill_sig": (jS - i) if fS else -1,
                         "bars_to_fill_flip": (jF - i) if fF else -1})
        log(f"{name:26s} 신호방향체결 {cnt[0]:5,} · 역방향체결 {cnt[1]:5,} · 양쪽 {cnt[2]:4,}")

    df = pd.DataFrame(rows)
    df["split"] = np.where(df.ts < VAL_START, "TRAIN",
                    np.where(df.ts < OOS_START, "VAL",
                    np.where(df.ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    # 정책별 건당 수익 (미체결 팔은 거래 없음 = 0)
    df["p_sig"]  = df.sig_ret.fillna(0.0)
    df["p_flip"] = df.flip_ret.fillna(0.0)
    df["p_oracleA"] = np.maximum(df.p_sig, 0.0)                       # 신호방향 or 기권
    df["p_oracleB"] = np.maximum(np.maximum(df.p_sig, df.p_flip), 0.0)  # 두 팔 + 기권
    df["best_is_flip"] = (df.p_flip > df.p_sig) & (df.p_flip > 0)
    df["both_lose"] = (df.p_sig <= 0) & (df.p_flip <= 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "both_arms.csv", index=False)

    def stat(v):
        v = np.asarray(v, float)
        w, ls = v[v > 0].sum(), -v[v < 0].sum()
        return float(v.mean() * 1e4), (float(w / ls) if ls > 0 else float("inf"))

    log(f"\n트리거 {len(df):,}건 (최소 한 팔 체결) | 신호방향 체결 {int(df.sig_filled.sum()):,} "
        f"· 역방향 체결 {int(df.flip_filled.sum()):,} · 양쪽 {int((df.sig_filled&df.flip_filled).sum()):,}")
    log("\n=== 정책별 건당 bp / PF (미체결=0) ===")
    print(f"{'구간':9s} {'n':>6s} | " + " | ".join(f"{x:>17s}" for x in
          ("신호방향", "역방향만", "오라클A(기권만)", "오라클B(양팔)")))
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        cells = []
        for col in ("p_sig", "p_flip", "p_oracleA", "p_oracleB"):
            m, pf = stat(w[col]); cells.append(f"{m:+7.2f}bp PF{pf:5.2f}")
        print(f"{wn:9s} {len(w):6,} | " + " | ".join(f"{x:>17s}" for x in cells))

    log("\n=== 오라클B의 선택 분포 ===")
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        log(f"  {wn:8s} 역방향 {float(w.best_is_flip.mean()):.1%} · 둘다손실(기권) "
            f"{float(w.both_lose.mean()):.1%} · 신호방향 {float((~w.best_is_flip & ~w.both_lose).mean()):.1%}")

    log("\n=== 역방향 선택의 순수 값어치 (오라클B − 오라클A) ===")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = df[df.split == wn]
        if not len(w): continue
        g = float((w.p_oracleB - w.p_oracleA).mean() * 1e4)
        b = float(w.p_oracleA.mean() * 1e4)
        log(f"  {wn:8s} +{g:.2f}bp  (오라클A {b:.2f}bp 대비 +{g/max(b,1e-9)*100:.0f}%)")
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
