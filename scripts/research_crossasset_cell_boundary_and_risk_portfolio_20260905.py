#!/usr/bin/env python3
"""교차자산 지속 규칙 — **격자 경계 확장 · 위험정규화 셀 선택 · 다자산 포트폴리오** (2026-09-05).

1차 셀 보정(`research_crossasset_cell_calibration_and_portfolio_20260905.py`)의 결함 둘을 고친다.
  ⚠️(a) **격자 경계**: 네 자산 전부 SL=8.0(상한)·trail=0.05(하한)을 골랐다 — §5.19 §5-A "격자 경계를 의심할 것".
       SL 3~16, arm 1~5, trail 0.02~0.2로 확장한다.
  ⚠️(b) **위험 미정규화**: SL이 넓으면 건당 bp가 그냥 커진다(위험을 더 진 것뿐). 사전등록 계약대로
       `notional = min(0.004/(sl·atr_pct), 0.5)`로 **자기자본 bp**를 계산하면 SL 축이 상쇄된다.
       셀 선택 잣대도 **일 샤프**(척도 무관)로 바꾼다.

  A 셀 선택   자산별 TRAIN에서 **일 샤프 최대**(단 TRAIN 일CI 하한 > 0 인 셀 중) → VAL/OOS 1회 확인.
             경계 점검: 고른 셀의 각 축이 격자 내부인지 보고. 같은 셀 무작위 진입 귀무 대비 초과분 병기.
  B 포트폴리오 자산별 위험정규화 일손익 → 상관행렬, 동일가중 합산 샤프. **총 노출 매칭**을 위해
             자산당 동시보유 CAP ∈ {1,2,5}를 전부 돌린다(ETH 단독 CAP5 = 5슬롯 vs 3자산 CAP2 = 6슬롯).
             A4 크로스심볼 캡(총 notional ≤ 1.5 equity)과의 정합도 계산해 보고.
HOLDOUT 차단. 연구/개발 점수.
"""
from __future__ import annotations

import importlib.util
import itertools
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


XA = _load("xa_b", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
CC = _load("cc_b", "scripts/research_crossasset_cell_calibration_and_portfolio_20260905.py")
V2 = _load("hev2_b", "scripts/research_homer_entry_v2_20260904.py")
C1M = _load("comp1_b", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, portfolio, day_boot = V2.sim_exit, V2.portfolio, V2.day_boot
cand_of = C1M.cand_of
OUT = ROOT / "data/research/crossasset_cell_boundary_risk_portfolio_20260905"
FWD, COST, SPLITS = XA.FWD, XA.COST, XA.SPLITS
SL_G = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
ARM_G = (1.0, 1.5, 2.0, 3.0, 5.0)
TRAIL_G = (0.02, 0.05, 0.10, 0.20)
CAPS = (1, 2, 5)
RISK_FRAC, NOTIONAL_CAP, B_NULL, NULL_POOL, B_BOOT = 0.004, 0.5, 200, 9000, 1000
rng = np.random.default_rng(20260905)


def log(m): print(f"[bnd] {m}", flush=True)


def eq_pnl(price_bp, atr_pct, sl):
    """자기자본 bp = 가격 bp × notional(위험 0.4% 고정). SL이 넓을수록 notional이 작아져 위험이 맞춰진다."""
    return price_bp * np.minimum(RISK_FRAC / (sl * atr_pct), NOTIONAL_CAP)


def run_pf(pnl, ts, eb, xb, cap):
    r = portfolio(cand_of(ts, eb, xb, pnl), cap)
    if r is None:
        return None
    t = r["trades"]; p = t["pnl_bp"].to_numpy(); tt = t["timestamp"].to_numpy()
    s = pd.Series(p / cap, index=pd.DatetimeIndex(pd.to_datetime(tt)).normalize()).groupby(level=0).sum()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
    sd = float(s.std(ddof=1))
    return {"n": int(len(p)), "exp_bp": round(float(p.mean()), 4), "win_rate": round(float((p > 0).mean()), 3),
            "daily_mean_bp": round(float(s.mean()), 3), "daily_sharpe_ann": round(float(s.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None,
            "max_dd_bp": round(float((np.cumsum(p) - np.maximum.accumulate(np.cumsum(p))).min()), 1),
            "_pnl": p, "_ts": tt, "_daily": s}


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "grid": {"sl": SL_G, "arm": ARM_G, "trail": TRAIL_G},
           "selection": "TRAIN 일 샤프 최대 (TRAIN 일CI 하한>0 인 셀 중), 위험정규화 자기자본 bp",
           "risk_contract": {"risk_frac": RISK_FRAC, "notional_cap": NOTIONAL_CAP}, "cost_bp": COST,
           "holdout_excluded": True, "assets": {}}
    daily = {w: {} for w in SPLITS}
    for sym, ref in XA.ASSETS.items():
        D = CC.prep(sym, ref); tsi = pd.DatetimeIndex(D["ts"])
        M = {w: np.asarray((tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))) for w, (a, b) in SPLITS.items()}
        atr_pct = D["atr"][D["i"]] / D["o"][D["i"] + 1]
        grid = {}
        for cell in itertools.product(SL_G, ARM_G, TRAIL_G):
            ret, ex = sim_exit(D["o"][D["i"] + 1], D["atr"][D["i"]], D["sign"], D["h"][D["ix"]], D["l"][D["ix"]], D["c"][D["ix"]], *cell)
            p = eq_pnl(ret * 1e4 - COST, atr_pct, cell[0])
            m = M["TRAIN"]
            r = run_pf(p[m], D["ts"][m], D["i"][m] + 1, D["i"][m] + 1 + ex[m], 5)
            if r is None:
                continue
            lo, hi = day_boot(r["_pnl"], r["_ts"], B_BOOT, rng)
            grid["_".join(map(str, cell))] = {"daily_sharpe_ann": r["daily_sharpe_ann"], "exp_bp": r["exp_bp"],
                                              "daily_mean_bp": r["daily_mean_bp"], "day_ci95": [round(lo, 4), round(hi, 4)], "n": r["n"]}
        elig = {k: v for k, v in grid.items() if v["day_ci95"][0] > 0 and v["daily_sharpe_ann"] is not None}
        pick = max(elig, key=lambda k: elig[k]["daily_sharpe_ann"]) if elig else max(grid, key=lambda k: grid[k]["daily_sharpe_ann"] or -9)
        cell = tuple(float(x) for x in pick.split("_"))
        interior = {"sl": SL_G[0] < cell[0] < SL_G[-1], "arm": ARM_G[0] < cell[1] < ARM_G[-1], "trail": TRAIL_G[0] < cell[2] < TRAIL_G[-1]}
        A = {"symbol": sym, "n_fires": int(len(D["i"])), "picked_cell": cell, "cells_eligible_TRAIN": len(elig), "n_cells": len(grid),
             "boundary_interior": interior, "all_interior": all(interior.values()),
             "eth_default_cell_TRAIN": grid.get("5.0_1.5_0.1"), "top5_TRAIN": dict(sorted(elig.items(), key=lambda kv: -kv[1]["daily_sharpe_ann"])[:5]), "windows": {}}
        ret, ex = sim_exit(D["o"][D["i"] + 1], D["atr"][D["i"]], D["sign"], D["h"][D["ix"]], D["l"][D["ix"]], D["c"][D["ix"]], *cell)
        p_eq = eq_pnl(ret * 1e4 - COST, atr_pct, cell[0])
        for w in SPLITS:
            m = M[w]
            if m.sum() < 100:
                continue
            rec = {}
            for cap in CAPS:
                r = run_pf(p_eq[m], D["ts"][m], D["i"][m] + 1, D["i"][m] + 1 + ex[m], cap)
                if r is None:
                    continue
                lo, hi = day_boot(r["_pnl"], r["_ts"], B_BOOT, rng)
                rec[f"cap{cap}"] = {k: v for k, v in r.items() if not k.startswith("_")} | {"day_ci95": [round(lo, 4), round(hi, 4)]}
                daily[w].setdefault(cap, {})[sym] = r["_daily"]
            # 같은 셀 무작위 진입 귀무 (cap5)
            a, b = SPLITS[w]
            wm = (pd.DatetimeIndex(D["ts_all"]) >= pd.Timestamp(a)) & (pd.DatetimeIndex(D["ts_all"]) < pd.Timestamp(b))
            pool = np.flatnonzero(wm & np.isfinite(D["atr"]) & (np.arange(D["n"]) + 1 + FWD < D["n"]))
            if len(pool) > NULL_POOL:
                pool = np.sort(rng.choice(pool, NULL_POOL, replace=False))
            pix = (pool + 1)[:, None] + np.arange(FWD)
            nv = []
            for _ in range(B_NULL):
                parts = []
                for sgn in (-1.0, 1.0):
                    cnt = int((m & (D["sign"] == sgn)).sum())
                    if cnt == 0:
                        continue
                    k = rng.choice(len(pool), size=min(cnt, len(pool)), replace=False)
                    pr, pe = sim_exit(D["o"][pool[k] + 1], D["atr"][pool[k]], np.full(len(k), sgn), D["h"][pix[k]], D["l"][pix[k]], D["c"][pix[k]], *cell)
                    pq = eq_pnl(pr * 1e4 - COST, D["atr"][pool[k]] / D["o"][pool[k] + 1], cell[0])
                    parts.append(cand_of(D["ts_all"][pool[k]], pool[k] + 1, pool[k] + 1 + pe, pq))
                q = portfolio(pd.concat(parts, ignore_index=True), 5); nv.append(q["exp_bp"] if q else np.nan)
            nv = np.asarray(nv, float); obs = rec.get("cap5", {}).get("exp_bp", np.nan)
            rec["null_same_cell"] = {"mean_bp": round(float(np.nanmean(nv)), 4), "excess_bp": round(float(obs - np.nanmean(nv)), 4),
                                     "percentile_of_obs": round(float((nv < obs).mean() * 100), 1)}
            A["windows"][w] = rec
        A["usable"] = all(A["windows"].get(w, {}).get("cap5", {}).get("exp_bp", -9) > 0 and
                          A["windows"].get(w, {}).get("null_same_cell", {}).get("excess_bp", -9) > 0 for w in ("VAL", "OOS"))
        out["assets"][sym] = A
        log(f"{sym}: 셀 {cell} 내부={interior} 적격 {len(elig)}/{len(grid)} · " +
            " · ".join(f"{w} exp={A['windows'][w]['cap5']['exp_bp']:.3f} 샤프={A['windows'][w]['cap5']['daily_sharpe_ann']} 초과={A['windows'][w]['null_same_cell']['excess_bp']:.3f}" for w in A["windows"]) + f" · 쓸만={A['usable']}")
    # 포트폴리오
    port = {}
    usable = [s for s, a in out["assets"].items() if a.get("usable")]
    for w in SPLITS:
        port[w] = {"usable_assets": usable, "by_cap": {}}
        for cap in CAPS:
            dd = daily[w].get(cap, {})
            if not dd:
                continue
            idx = pd.date_range(min(s.index.min() for s in dd.values()), max(s.index.max() for s in dd.values()), freq="D")
            df = pd.DataFrame({k: v for k, v in dd.items()}).reindex(idx).fillna(0.0)
            def sh(s):
                sd = s.std(ddof=1); return round(float(s.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None
            e_all = df.mean(axis=1); e_us = df[usable].mean(axis=1) if usable else None
            port[w]["by_cap"][f"cap{cap}"] = {
                "corr": df.corr().round(3).to_dict(), "per_asset_sharpe": {c: sh(df[c]) for c in df},
                "per_asset_daily_mean": {c: round(float(df[c].mean()), 3) for c in df},
                "all4_sharpe": sh(e_all), "all4_daily_mean": round(float(e_all.mean()), 3),
                "usable_sharpe": sh(e_us) if e_us is not None else None,
                "usable_daily_mean": round(float(e_us.mean()), 3) if e_us is not None else None,
                "eth_only_sharpe": sh(df["ETHUSDT"]) if "ETHUSDT" in df else None}
    out["portfolio"] = port
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for w in SPLITS:
        for cap in CAPS:
            p = port[w]["by_cap"].get(f"cap{cap}")
            if p:
                log(f"  [{w} cap{cap}] ETH단독 샤프 {p['eth_only_sharpe']} · 쓸만({len(usable)}자산) 샤프 {p['usable_sharpe']} 일평균 {p['usable_daily_mean']} · 자산별 {p['per_asset_sharpe']}")


if __name__ == "__main__":
    main()
