#!/usr/bin/env python3
"""발동 후 **짧은 구간 원시 수익** — "천장→아래, 바닥→위가 몇 봉은 대부분 맞는다"의 검정 (2026-09-05).

사용자 관찰: *"천장 신호가 뜨면 아래로, 바닥이면 위로 몇 봉 동안은 대부분 맞던데 왜 그런거지?"*

앞선 측정은 전부 **트레일링 청산(5.0 SL/1.5 ARM/0.1 trail, 최대 200봉) + 10bp 비용** 아래였다.
"짧게 h봉만 들고 나온다"는 다른 명제이므로 따로 잰다. 청산 구조·비용 가정을 **빼고** 원시로 본다.

  진입   open[i+1] (규칙과 동일)
  청산   **고정 h봉 뒤 종가** (h ∈ 1,2,3,4,6,8,12,18,24,48) — 트레일링 없음
  방향   페이드(= 칩 방향: 바닥 발동 → 롱, 천장 발동 → 숏)
  지표   평균 bp · 중앙값 · P(>0) · P(>10bp, 테이커 왕복 비용 초과) · 일군집 CI
  ⭐대조 같은 창·같은 방향의 **무제한 기저**(모든 봉에서 같은 방향으로 h봉 보유) — 하락장에서는
        무작위 숏도 맞으므로 이걸 빼야 신호의 몫이 남는다(§7-1 같은 측면 귀무).
측면별(바닥/천장)로 나눠 보고, 창별로 따로 본다. HOLDOUT 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_sh", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OUT = ROOT / "data/research/eth_fire_short_horizon_fade_20260905"
HS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 48)
WINDOWS = ("TRAIN", "VAL", "OOS")
COST = 10.0
rng = np.random.default_rng(20260905)


def log(m): print(f"[short] {m}", flush=True)


def day_ci(v, t, B=1000):
    d = pd.DatetimeIndex(pd.to_datetime(t)).normalize().to_numpy()
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}
    out = [v[np.concatenate([idx[x] for x in u[rng.integers(0, len(u), len(u))]])].mean() for _ in range(B)]
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, sd, split, ts, bidx = B["pos"], B["sd"], B["split"], B["ts"], B["bidx"]
    o, c = B["o"], B["c"]
    n = len(c)
    fade_sign = np.where(sd == 1, 1.0, -1.0)          # 바닥 발동 → 페이드 = 롱
    ts_all = pd.DatetimeIndex(B["bar"]["timestamp"].to_numpy()); p_first = B["p_first"]
    allb = B["bar"]["pos"].to_numpy() - p_first        # 모든 봉의 seg 인덱스
    split_all = B["bar"]["split"].to_numpy()
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "horizons": list(HS), "cost_bp": COST,
           "holdout_touched": False, "entry": "open[i+1]", "exit": "close[i+h]", "rows": {}}
    for h in HS:
        ok = (bidx + h) < n
        r = fade_sign * (c[np.where(ok, bidx + h, 0)] - o[bidx + 1]) / o[bidx + 1] * 1e4
        rec = {}
        for w in WINDOWS:
            m = ok & (split == w)
            if m.sum() < 100:
                continue
            d = {"n": int(m.sum()), "mean_bp": round(float(r[m].mean()), 2), "median_bp": round(float(np.median(r[m])), 2),
                 "p_pos": round(float((r[m] > 0).mean()), 3), "p_gt_cost": round(float((r[m] > COST).mean()), 3),
                 "day_ci95": day_ci(r[m], ts[m]), "net_after_cost_bp": round(float(r[m].mean() - COST), 2)}
            # 같은 창·같은 방향 무제한 기저 (모든 봉)
            bm = (split_all == w) & ((allb + h) < n) & ((allb + 1) < n)
            base = {}
            for sgn, nm in ((1.0, "long"), (-1.0, "short")):
                bb = allb[bm]
                rb = sgn * (c[bb + h] - o[bb + 1]) / o[bb + 1] * 1e4
                base[nm] = {"mean_bp": round(float(rb.mean()), 2), "p_pos": round(float((rb > 0).mean()), 3)}
            d["baseline_all_bars"] = base
            # 측면별 + 같은 측면 기저 대비 초과
            d["by_side"] = {}
            for sv, nm, bkey in ((1, "bottom_fire(fade=long)", "long"), (0, "top_fire(fade=short)", "short")):
                mm = m & (sd == sv)
                if mm.sum() < 50:
                    continue
                d["by_side"][nm] = {"n": int(mm.sum()), "mean_bp": round(float(r[mm].mean()), 2),
                                    "p_pos": round(float((r[mm] > 0).mean()), 3),
                                    "excess_mean_bp": round(float(r[mm].mean() - base[bkey]["mean_bp"]), 2),
                                    "excess_p_pos_pp": round(float((r[mm] > 0).mean() - base[bkey]["p_pos"]) * 100, 1),
                                    "day_ci95": day_ci(r[mm], ts[mm])}
            rec[w] = d
        rep["rows"][f"h{h}"] = rec
        log(f"  h={h:>2}봉: " + " | ".join(
            f"{w} 평균 {rec[w]['mean_bp']:>6}bp P(>0) {rec[w]['p_pos']:.3f} (기저 롱 {rec[w]['baseline_all_bars']['long']['p_pos']:.3f}/숏 {rec[w]['baseline_all_bars']['short']['p_pos']:.3f}) "
            f"P(>10bp) {rec[w]['p_gt_cost']:.3f} 비용후 {rec[w]['net_after_cost_bp']:>6}" for w in WINDOWS if w in rec))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    print("\n=== 측면별 초과분 (같은 방향 전봉 기저 대비) ===")
    for h in HS:
        for w in WINDOWS:
            d = rep["rows"][f"h{h}"].get(w)
            if not d:
                continue
            print(f"  h={h:>2} {w:5s} " + " · ".join(
                f"{k.split('(')[0]:11s} 평균 {v['mean_bp']:>6}(초과 {v['excess_mean_bp']:>6}) P(>0) {v['p_pos']:.3f}(초과 {v['excess_p_pos_pp']:>+5.1f}pp)"
                for k, v in d["by_side"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
