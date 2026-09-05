#!/usr/bin/env python3
"""확신도 사이징 — **축 선택 민감도 · 하나빼기 · 집중도** (2026-09-05).

C7(`research_eth_composite_conviction_sizing_20260905.py`)의 확신도는 13축을 TRAIN 갭 기준으로 골랐다.
그 선택이 결과를 만든 것인지, 아니면 축을 아무렇게나 골라도 나오는 것인지 가른다.
  S1 전체축 (선택 없음, 사용 가능한 모든 상태축)
  S2 사전등록 "작동" 기준 축 (TRAIN CI>0 ∧ VAL·OOS 갭>0) — 참고용(VAL/OOS 사용 선택이라 낙관적)
  S3 13축에서 하나씩 빼기 (단일 축 의존 확인)
  S4 무작위 K축 표집(K=13, B=200) — "아무 13축이나 골라도 되는가"
  S5 일손익 집중도 — 상위 5일 제거 후에도 Δ가 남는가 (이긴 날 비율 0.47의 해석)
전부 위험사이징(라이브 계약) 위에서, β=0.5, R 대비 일별 짝비교.
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C7 = _load("c7s", "scripts/research_eth_composite_conviction_sizing_20260905.py")
C1M = _load("comp1_s", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
build, pf, cand_of, day_paired = C1M.build, C1M.pf, C1M.cand_of, C1M.day_paired
CELL, CAP, WINDOWS = C1M.CELL, C1M.CAP, C1M.WINDOWS
AXES13, BETA, W_LO, W_HI, MIN_AXES = C7.AXES_TRAIN, 0.5, C7.W_LO, C7.W_HI, C7.MIN_AXES
AXES_WORKS = ["vwap_dev_z", "rsi_c", "atr_pct", "ax_market_beta_move", "vol_z", "ax_activity", "delta_z", "ax_btc_shock", "atr_percentile_864"]
RISK_FRAC, NOTIONAL_CAP, B_RAND = 0.004, 0.5, 200
OUT = ROOT / "data/research/eth_conviction_axis_sensitivity_20260905"
rng = np.random.default_rng(20260905)


def log(m): print(f"[sens] {m}", flush=True)


def ranks(B):
    """축별 TRAIN ECDF 백분위 (aligned는 지속방향 부호 적용)."""
    cs, tr = B["cont_sign"], B["split"] == "TRAIN"; n = len(cs); out = {}
    for name, (kind, raw) in B["S"].items():
        x = raw * cs if kind == "aligned" else raw.astype(float)
        fin = np.isfinite(x); ref = np.sort(x[fin & tr])
        if len(ref) < 500:
            continue
        r = np.full(n, np.nan); r[fin] = np.searchsorted(ref, x[fin], side="right") / len(ref)
        out[name] = r
    return out


def conv(R, names):
    arrs = [R[k] for k in names if k in R]
    M = np.vstack(arrs); ok = np.isfinite(M)
    cnt = ok.sum(0); acc = np.where(ok, M, 0.0).sum(0)
    return np.where(cnt >= min(MIN_AXES, len(arrs)), acc / np.maximum(cnt, 1), 0.5)


def arm(B, sizer, c, base):
    pos, split, ts, cont_bp, cont_ex = B["pos"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"]
    w = np.clip(1.0 + BETA * (2.0 * c - 1.0), W_LO, W_HI); res = {}
    for win in WINDOWS:
        m = split == win
        r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * sizer[m] * w[m]))
        res[win] = {"exp_bp": r["stats"]["exp_bp"], "daily_mean_bp": r["stats"]["daily_mean_bp"],
                    "daily_sharpe_ann": r["stats"]["daily_sharpe_ann"], "vs_R": day_paired(r["pnl"], r["ts"], base[win]["pnl"], base[win]["ts"]),
                    "_pnl": r["pnl"], "_ts": r["ts"]}
    return res


def strip_priv(d):
    return {w: {k: v for k, v in x.items() if not k.startswith("_")} for w, x in d.items()}


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = build()
    pos, split, ts, cont_bp, cont_ex = B["pos"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"]
    atr_pct = B["atr"] / B["entry"]
    sizer = np.minimum(RISK_FRAC / (CELL[0] * atr_pct), NOTIONAL_CAP)
    base = {w: pf(cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w] * sizer[split == w])) for w in WINDOWS}
    R = ranks(B); allax = sorted(R)
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "n_axes_available": len(allax), "axes_available": allax,
           "R_baseline": {w: base[w]["stats"] for w in WINDOWS}, "holdout_touched": False}
    main13 = arm(B, sizer, conv(R, AXES13), base)
    rep["S0_main13"] = strip_priv(main13)
    rep["S1_all_axes"] = strip_priv(arm(B, sizer, conv(R, allax), base))
    rep["S2_works9"] = strip_priv(arm(B, sizer, conv(R, AXES_WORKS), base))
    rep["S3_leave_one_out"] = {}
    for k in AXES13:
        rest = [x for x in AXES13 if x != k]
        a = arm(B, sizer, conv(R, rest), base)
        rep["S3_leave_one_out"][k] = {w: {"vs_R": a[w]["vs_R"]["diff_bp_day"], "ci": a[w]["vs_R"]["ci95"]} for w in ("VAL", "OOS")}
    # S4 무작위 13축
    vals = {w: [] for w in WINDOWS}
    for _ in range(B_RAND):
        pick = list(rng.choice(allax, size=min(13, len(allax)), replace=False))
        a = arm(B, sizer, conv(R, pick), base)
        for w in WINDOWS:
            vals[w].append(a[w]["vs_R"]["diff_bp_day"])
    rep["S4_random13"] = {w: {"mean": round(float(np.mean(vals[w])), 3), "p05": round(float(np.percentile(vals[w], 5)), 3),
                              "p95": round(float(np.percentile(vals[w], 95)), 3),
                              "obs_main13": main13[w]["vs_R"]["diff_bp_day"],
                              "percentile_of_main13": round(float((np.asarray(vals[w]) < main13[w]["vs_R"]["diff_bp_day"]).mean() * 100), 1)} for w in WINDOWS}
    # S5 집중도: 일 차이 상위 5일 제거
    rep["S5_concentration"] = {}
    for w in WINDOWS:
        a = main13[w]
        def dser(p, t):
            return pd.Series(np.asarray(p, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(t))).normalize()).groupby(level=0).sum() / CAP
        A, Bs = dser(a["_pnl"], a["_ts"]), dser(base[w]["pnl"], base[w]["ts"])
        days = A.index.union(Bs.index); d = (A.reindex(days).fillna(0) - Bs.reindex(days).fillna(0))
        srt = d.sort_values(ascending=False)
        rep["S5_concentration"][w] = {"mean_all": round(float(d.mean()), 3), "n_days": int(len(d)),
                                      "mean_drop_top5": round(float(d.drop(srt.index[:5]).mean()), 3),
                                      "mean_drop_top10": round(float(d.drop(srt.index[:10]).mean()), 3),
                                      "median": round(float(d.median()), 3), "win_frac": round(float((d > 0).mean()), 3),
                                      "top5_share_of_total": round(float(srt.iloc[:5].sum() / d.sum()), 3) if d.sum() != 0 else None}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for k in ("S0_main13", "S1_all_axes", "S2_works9"):
        log(f"  {k}: " + " | ".join(f"{w} Δ={rep[k][w]['vs_R']['diff_bp_day']}{rep[k][w]['vs_R']['ci95']} 샤프={rep[k][w]['daily_sharpe_ann']}" for w in WINDOWS))
    log(f"  S4 무작위13축: {rep['S4_random13']}")
    log(f"  S5 집중도: {rep['S5_concentration']}")


if __name__ == "__main__":
    main()
