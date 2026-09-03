#!/usr/bin/env python3
"""증거신호 유지 규칙: **TP에서 끊기(A) vs HORIZON 채우기(B)** (2026-09-03).

사용자 질문: "증거신호 피쳐로 들어갈 때, 익절한 경우 다시 미발동으로 돌아가는지 아니면
익절해도 horizon을 다 채우고 기다리는지에 따라 어떻게 변하는지."

⭐**저장소에 두 규칙이 실제로 공존한다**:
  A `_fill_until_tp_or_horizon` (`live_evidence_signal_dashboard_20260823.py:690`)
    -- 대시보드 칩/votes/net_score. 라벨이 확정되면(TP 터치, fib는 MAE) **즉시 끈다**.
       2026-08-30 주석: "라벨이 확정된 사건은 더 이상 지금 유효한 증거가 아니다."
  B `build_eth_evidence_signal_material_tensor_20260902.py`
    -- DL/RL 재료 텐서. proba를 **그 신호 자신의 HORIZON 내내 유지**한다(TP 무시).

둘 다 인과적이다(시점 i의 값은 i까지의 정보만 쓴다). 차이는 **TP 이후 구간을 신호로 볼
것인가**이고, 그 구간이 정보를 담는지는 실증 문제다.

이 스크립트가 재는 것:
  ① 유지 구간 중 **TP 이후가 몇 %**인가 (분쟁 구간의 크기 -- 작으면 논쟁 자체가 무의미)
  ② TP 이후 구간이 **전방수익과 관계가 있는가** (IC / 조건부 평균)
  ③ A/B 각각의 전체 IC 비교
부호 규약은 텐서와 같다: +가 bottom(롱-우호), -가 top.
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
from scipy.stats import spearmanr  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT = ROOT / "tmp/eth_evidence_tp_truncation_20260903"
HIT_RESOLUTION = {"fib_extension_exhaustion": "touch_and_mae"}      # 나머지는 "touch"
FIB_K_LOSS_MULT = 2.0
FWD = [12, 24, 48]                                                   # 전방수익 창(봉)


def log(m): print(f"[tptrunc] {m}", flush=True)


def resolve_bar(i: int, side: str, k: float, hz: int, mode: str,
                c: np.ndarray, h: np.ndarray, lo: np.ndarray, ap: float, n: int) -> int:
    """라벨이 확정되는 봉. 없으면 i+hz(만료). 대시보드 `_fill_until_tp_or_horizon`과 동일."""
    end = min(i + hz, n - 1)
    if not np.isfinite(ap):
        return end
    target = k * ap
    if mode == "touch":
        level = c[i] * (1 - target) if side == "top" else c[i] * (1 + target)
    else:
        adv = FIB_K_LOSS_MULT * target
        level = c[i] * (1 + adv) if side == "top" else c[i] * (1 - adv)
    for b in range(i + 1, end + 1):
        if mode == "touch":
            done = (lo[b] <= level) if side == "top" else (h[b] >= level)
        else:
            done = (h[b] >= level) if side == "top" else (lo[b] <= level)
        if done:
            return b
    return end


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)
    ts = pd.DatetimeIndex(kl["timestamp"])
    pos_of = {t: i for i, t in enumerate(ts)}
    c, h, lo = (kl[x].to_numpy(float) for x in ("close", "high", "low"))
    n = len(kl)
    split = np.where(ts < VAL_START, "TRAIN", np.where(ts < OOS_START, "VAL",
                     np.where(ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    fwd = {f: np.concatenate([(c[f:] - c[:-f]) / c[:-f], np.full(f, np.nan)]) for f in FWD}
    log(f"klines {n:,}봉 {ts.min()} ~ {ts.max()}")

    rows, agg = [], {}
    for name, cf in cfg.items():
        k, hz = float(cf["k"]), int(cf["horizon"])
        mode = HIT_RESOLUTION.get(name, "touch")
        fires = pd.read_csv(SRC / f"{name}_causal_fires.csv", parse_dates=["timestamp"])
        pr = pd.read_csv(SRC / f"{name}_causal_proba.csv", parse_dates=["timestamp"])
        fires = fires.merge(pr[["timestamp", "proba"]], on="timestamp", how="left")
        fires = fires[fires.timestamp.isin(pos_of)].reset_index(drop=True)

        A = np.zeros(n); B = np.zeros(n); postTP = np.zeros(n, bool); heldB = np.zeros(n, bool)
        n_tp = 0
        for t, side, ap, p in zip(fires.timestamp, fires.side, fires.atr_pct, fires["proba"]):
            i = pos_of[t]
            sgn = 1.0 if side == "bottom" else -1.0
            v = sgn * (float(p) if np.isfinite(p) else 0.5)
            r = resolve_bar(i, side, k, hz, mode, c, h, lo, float(ap), n)
            e = min(i + hz, n - 1)
            A[i:r + 1] = v                                  # TP에서 끊기
            B[i:e + 1] = v                                  # HORIZON 채우기
            heldB[i:e + 1] = True
            if r < e:
                postTP[r + 1:e + 1] = True                   # 분쟁 구간
                n_tp += 1

        held_b = int(heldB.sum()); post = int(postTP.sum())
        d = {"signal": name, "hz": hz, "fires": len(fires), "tp_resolved": n_tp,
             "tp_rate": round(n_tp / max(len(fires), 1), 4),
             "bars_held_B": held_b, "bars_held_A": int((A != 0).sum()),
             "post_tp_bars": post,
             "post_tp_share": round(post / max(held_b, 1), 4),
             "coverage_B": round(held_b / n, 4), "coverage_A": round((A != 0).sum() / n, 4)}
        for f in FWD:
            y = fwd[f]
            ok = np.isfinite(y)
            mA = ok & (A != 0); mB = ok & (B != 0); mP = ok & postTP
            d[f"ic{f}_A"] = round(spearmanr(A[mA], y[mA])[0], 4) if mA.sum() > 200 else np.nan
            d[f"ic{f}_B"] = round(spearmanr(B[mB], y[mB])[0], 4) if mB.sum() > 200 else np.nan
            d[f"ic{f}_postTP"] = round(spearmanr(B[mP], y[mP])[0], 4) if mP.sum() > 200 else np.nan
        rows.append(d)
        agg[name] = (A, B, postTP)
        log(f"  {name:28s} 발동 {len(fires):5,} · TP확정 {n_tp:5,} ({n_tp/max(len(fires),1)*100:5.1f}%) "
            f"· 유지봉 B {held_b:6,} → A {int((A!=0).sum()):6,} · TP이후 {post/max(held_b,1)*100:5.1f}%")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "per_signal.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== ① 분쟁 구간 크기 ===")
    print(R[["signal", "hz", "fires", "tp_rate", "bars_held_B", "bars_held_A",
             "post_tp_share", "coverage_B", "coverage_A"]].to_string(index=False))
    print("\n=== ② TP 이후 구간이 정보를 담는가 (IC, 전방 24봉) ===")
    print(R[["signal", "ic24_A", "ic24_B", "ic24_postTP"]].to_string(index=False))
    print("\n=== ③ 창별 IC 비교 ===")
    for f in FWD:
        a = R[f"ic{f}_A"].abs().mean(); b = R[f"ic{f}_B"].abs().mean(); p = R[f"ic{f}_postTP"].abs().mean()
        print(f"  전방 {f:2d}봉  |IC| 평균  A(TP끊기) {a:.4f} · B(HORIZON) {b:.4f} · TP이후만 {p:.4f}")

    # 스플릿별 안정성 (합성: 8신호 signed 평균)
    print("\n=== ④ 스플릿별 (8신호 signed 합, 전방 24봉 IC) ===")
    SA = np.sum([agg[s][0] for s in agg], axis=0)
    SB = np.sum([agg[s][1] for s in agg], axis=0)
    y = fwd[24]
    for wn in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        m = (split == wn) & np.isfinite(y)
        ia = spearmanr(SA[m & (SA != 0)], y[m & (SA != 0)])[0] if (m & (SA != 0)).sum() > 200 else np.nan
        ib = spearmanr(SB[m & (SB != 0)], y[m & (SB != 0)])[0] if (m & (SB != 0)).sum() > 200 else np.nan
        print(f"  {wn:8s} A {ia:+.4f} (n={int((m&(SA!=0)).sum()):,})  ·  B {ib:+.4f} (n={int((m&(SB!=0)).sum()):,})")
    json.dump({"note": "A=TP에서 끊기(대시보드), B=HORIZON 채우기(재료 텐서)",
               "fwd_windows": FWD}, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
