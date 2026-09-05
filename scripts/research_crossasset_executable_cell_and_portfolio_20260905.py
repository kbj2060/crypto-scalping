#!/usr/bin/env python3
"""교차자산 지속 규칙 — **실행가능성 하한을 건 셀 선택 + 최종 다자산 포트폴리오** (2026-09-05).

앞 라운드(방향뒤집기 대조군)에서 격자가 **arm=0.25 · trail=0.01(둘 다 하한)**으로 밀렸고 샤프 12~17이
나왔다. 뒤집기 대조군은 통과했다(real−flip CI 두 창 > 0). 그런데 **trail = 0.01×ATR ≈ 0.25bp**로
호가 스프레드(≈0.5~1bp)보다 좁다 — 그 가격에 스톱이 체결된다는 가정 자체가 성립하지 않는다.
⇒ **방향뒤집기 대조군은 필요조건이지 충분조건이 아니다.** 미세구조 실행가능성 하한이 따로 필요하다.

하한(사전 고정): 탐색을 **이미 라이브 섀도우로 검증 중인 값보다 느슨한 쪽으로만** 허용한다.
    trail ≥ 0.10×ATR  (배포 R의 값. 중앙 atr_pct 기준 ≈2.5bp)
    arm   ≥ 1.0×ATR   (memory `feedback_trailing_stop_low_arm_noise_harvest_artifact`: ARM→0 코너 금지)
    sl 는 자유 (3~16)
선택: TRAIN에서 (real−flip 짝비교 CI 하한 > 0) ∧ (일CI 하한 > 0) 인 셀 중 **일 샤프 최대**.
보고: 셀의 절대 bp 환산(trail_bp/arm_bp/sl_bp 중앙), 평균 보유 봉, 승률, 뒤집기 대조, 같은셀 무작위 귀무,
      자산별 위험정규화 일손익 상관, 자산당 CAP {1,2,5} 동일가중 합산 샤프.
HOLDOUT 차단. 연구/개발 점수 — 승격은 전진 섀도우.
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


XA = _load("xa_e", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
CC = _load("cc_e", "scripts/research_crossasset_cell_calibration_and_portfolio_20260905.py")
BR = _load("br_e", "scripts/research_crossasset_cell_boundary_and_risk_portfolio_20260905.py")
V2 = _load("hev2_e", "scripts/research_homer_entry_v2_20260904.py")
C1M = _load("comp1_e", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, day_boot, portfolio = V2.sim_exit, V2.day_boot, V2.portfolio
eq_pnl, run_pf = BR.eq_pnl, BR.run_pf
day_paired, cand_of = C1M.day_paired, C1M.cand_of
OUT = ROOT / "data/research/crossasset_executable_cell_final_20260905"
FWD, COST, SPLITS, CAPS = XA.FWD, XA.COST, XA.SPLITS, (1, 2, 5)
SL_G, ARM_G, TRAIL_G = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0), (1.0, 1.5, 2.0, 3.0), (0.10, 0.20)
MIN_TRAIL, MIN_ARM = 0.10, 1.0
B_BOOT, B_NULL, NULL_POOL = 1000, 200, 9000
rng = np.random.default_rng(20260905)


def log(m): print(f"[exec] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
           "executability_floor": {"min_trail_atr": MIN_TRAIL, "min_arm_atr": MIN_ARM,
                                   "rationale": "배포 R(0.1 trail/1.5 arm)보다 타이트한 쪽으로는 탐색하지 않는다 — trail 0.01×ATR ≈ 0.25bp는 스프레드 미만"},
           "grid": {"sl": SL_G, "arm": ARM_G, "trail": TRAIL_G}, "cost_bp": COST,
           "selection": "TRAIN real−flip 짝비교 CI 하한>0 ∧ 일CI 하한>0 인 셀 중 일 샤프 최대(자기자본 bp)",
           "holdout_excluded": True, "assets": {}}
    daily = {w: {c: {} for c in CAPS} for w in SPLITS}
    for sym, ref in XA.ASSETS.items():
        D = CC.prep(sym, ref); tsi = pd.DatetimeIndex(D["ts"])
        M = {w: np.asarray((tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))) for w, (a, b) in SPLITS.items()}
        atr_pct = D["atr"][D["i"]] / D["o"][D["i"] + 1]
        e0, a0, s0, H, L, C = D["o"][D["i"] + 1], D["atr"][D["i"]], D["sign"], D["h"][D["ix"]], D["l"][D["ix"]], D["c"][D["ix"]]
        mt = M["TRAIN"]; grid = {}
        for cell in itertools.product(SL_G, ARM_G, TRAIL_G):
            rr, xr = sim_exit(e0, a0, s0, H, L, C, *cell); rf, xf = sim_exit(e0, a0, -s0, H, L, C, *cell)
            pr = eq_pnl(rr * 1e4 - COST, atr_pct, cell[0]); pfl = eq_pnl(rf * 1e4 - COST, atr_pct, cell[0])
            R = run_pf(pr[mt], D["ts"][mt], D["i"][mt] + 1, D["i"][mt] + 1 + xr[mt], 5)
            Fp = run_pf(pfl[mt], D["ts"][mt], D["i"][mt] + 1, D["i"][mt] + 1 + xf[mt], 5)
            if R is None or Fp is None:
                continue
            dp = day_paired(R["_pnl"], R["_ts"], Fp["_pnl"], Fp["_ts"], B=300)
            lo, hi = day_boot(R["_pnl"], R["_ts"], 300, rng)
            grid["_".join(map(str, cell))] = {"sharpe": R["daily_sharpe_ann"], "exp_bp": R["exp_bp"], "win_rate": R["win_rate"],
                                              "flip_exp_bp": Fp["exp_bp"], "real_minus_flip": dp["diff_bp_day"], "rmf_ci95": dp["ci95"],
                                              "day_ci95": [round(lo, 4), round(hi, 4)], "mean_hold_bars": round(float(xr[mt].mean()), 1)}
        elig = {k: v for k, v in grid.items() if v["rmf_ci95"][0] > 0 and v["day_ci95"][0] > 0 and v["sharpe"] is not None}
        A = {"symbol": sym, "n_cells": len(grid), "n_eligible": len(elig), "grid_TRAIN": grid,
             "eth_default_5_1.5_0.1": grid.get("5.0_1.5_0.1")}
        if not elig:
            A["usable"] = False; A["picked_cell"] = None; out["assets"][sym] = A
            log(f"{sym}: 적격 셀 0/{len(grid)} — 사용 불가"); continue
        pick = max(elig, key=lambda k: elig[k]["sharpe"]); cell = tuple(float(x) for x in pick.split("_"))
        med_atr = float(np.median(atr_pct))
        A |= {"picked_cell": cell, "median_atr_pct": round(med_atr, 5),
              "absolute_levels_bp_at_median_atr": {"sl_bp": round(cell[0] * med_atr * 1e4, 1), "arm_bp": round(cell[1] * med_atr * 1e4, 1), "trail_bp": round(cell[2] * med_atr * 1e4, 2)},
              "boundary_interior": {"sl": SL_G[0] < cell[0] < SL_G[-1], "arm": ARM_G[0] < cell[1] < ARM_G[-1], "trail": TRAIL_G[0] < cell[2] < TRAIL_G[-1]},
              "windows": {}}
        rr, xr = sim_exit(e0, a0, s0, H, L, C, *cell); rf, xf = sim_exit(e0, a0, -s0, H, L, C, *cell)
        pr = eq_pnl(rr * 1e4 - COST, atr_pct, cell[0]); pfl = eq_pnl(rf * 1e4 - COST, atr_pct, cell[0])
        for w in SPLITS:
            m = M[w]
            if m.sum() < 100:
                continue
            rec = {}
            for cap in CAPS:
                R = run_pf(pr[m], D["ts"][m], D["i"][m] + 1, D["i"][m] + 1 + xr[m], cap)
                Fp = run_pf(pfl[m], D["ts"][m], D["i"][m] + 1, D["i"][m] + 1 + xf[m], cap)
                if R is None:
                    continue
                lo, hi = day_boot(R["_pnl"], R["_ts"], B_BOOT, rng)
                rec[f"cap{cap}"] = {k: v for k, v in R.items() if not k.startswith("_")} | {
                    "day_ci95": [round(lo, 4), round(hi, 4)], "mean_hold_bars": round(float(xr[m].mean()), 1),
                    "flip_exp_bp": Fp["exp_bp"] if Fp else None,
                    "real_minus_flip": day_paired(R["_pnl"], R["_ts"], Fp["_pnl"], Fp["_ts"]) if Fp else None}
                daily[w][cap][sym] = R["_daily"]
            # 같은 셀 무작위 진입 귀무 (cap5)
            a, b = SPLITS[w]
            wm = (pd.DatetimeIndex(D["ts_all"]) >= pd.Timestamp(a)) & (pd.DatetimeIndex(D["ts_all"]) < pd.Timestamp(b))
            pool = np.flatnonzero(wm & np.isfinite(D["atr"]) & (np.arange(D["n"]) + 1 + FWD < D["n"]))
            if len(pool) > NULL_POOL:
                pool = np.sort(rng.choice(pool, NULL_POOL, replace=False))
            pix = (pool + 1)[:, None] + np.arange(FWD); nv = []
            for _ in range(B_NULL):
                parts = []
                for sgn in (-1.0, 1.0):
                    cnt = int((m & (D["sign"] == sgn)).sum())
                    if cnt == 0:
                        continue
                    k = rng.choice(len(pool), size=min(cnt, len(pool)), replace=False)
                    q, qe = sim_exit(D["o"][pool[k] + 1], D["atr"][pool[k]], np.full(len(k), sgn), D["h"][pix[k]], D["l"][pix[k]], D["c"][pix[k]], *cell)
                    parts.append(cand_of(D["ts_all"][pool[k]], pool[k] + 1, pool[k] + 1 + qe,
                                         eq_pnl(q * 1e4 - COST, D["atr"][pool[k]] / D["o"][pool[k] + 1], cell[0])))
                z = portfolio(pd.concat(parts, ignore_index=True), 5); nv.append(z["exp_bp"] if z else np.nan)
            nv = np.asarray(nv, float); obs = rec.get("cap5", {}).get("exp_bp", np.nan)
            rec["null_same_cell"] = {"mean_bp": round(float(np.nanmean(nv)), 4), "excess_bp": round(float(obs - np.nanmean(nv)), 4),
                                     "percentile_of_obs": round(float((nv < obs).mean() * 100), 1)}
            A["windows"][w] = rec
        A["usable"] = all(A["windows"].get(w, {}).get("cap2", {}).get("day_ci95", [-9])[0] > 0 and
                          (A["windows"].get(w, {}).get("cap2", {}).get("real_minus_flip") or {}).get("ci95", [-9])[0] > 0 and
                          A["windows"].get(w, {}).get("null_same_cell", {}).get("excess_bp", -9) > 0 for w in ("VAL", "OOS"))
        out["assets"][sym] = A
        log(f"{sym}: 셀 {cell} = SL {A['absolute_levels_bp_at_median_atr']['sl_bp']}bp/ARM {A['absolute_levels_bp_at_median_atr']['arm_bp']}bp/TRAIL {A['absolute_levels_bp_at_median_atr']['trail_bp']}bp · 적격 {len(elig)}/{len(grid)} · " +
            " · ".join(f"{w} exp={A['windows'][w]['cap2']['exp_bp']:.3f} CI{A['windows'][w]['cap2']['day_ci95']} 샤프={A['windows'][w]['cap2']['daily_sharpe_ann']} 보유={A['windows'][w]['cap2']['mean_hold_bars']}봉 "
                       f"r−f={A['windows'][w]['cap2']['real_minus_flip']['diff_bp_day']}{A['windows'][w]['cap2']['real_minus_flip']['ci95']} 초과={A['windows'][w]['null_same_cell']['excess_bp']:.3f}" for w in A["windows"]) + f" · 쓸만={A['usable']}")
    usable = [s for s, a in out["assets"].items() if a.get("usable")]
    port = {}
    for w in SPLITS:
        port[w] = {"usable_assets": usable, "by_cap": {}}
        for cap in CAPS:
            dd = daily[w][cap]
            if not dd:
                continue
            idx = pd.date_range(min(s.index.min() for s in dd.values()), max(s.index.max() for s in dd.values()), freq="D")
            df = pd.DataFrame(dd).reindex(idx).fillna(0.0)
            def sh(s):
                sd = s.std(ddof=1); return round(float(s.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None
            row = {"corr": df.corr().round(3).to_dict(), "per_asset_sharpe": {c: sh(df[c]) for c in df},
                   "per_asset_daily_mean": {c: round(float(df[c].mean()), 3) for c in df},
                   "eth_only_sharpe": sh(df["ETHUSDT"]) if "ETHUSDT" in df else None,
                   "eth_only_daily_mean": round(float(df["ETHUSDT"].mean()), 3) if "ETHUSDT" in df else None}
            # 자산별 usable 여부와 무관하게, 셀이 선택된 전 자산의 동일가중 합산도 항상 계산한다
            #   (개별 자산의 3~4개월 창 CI는 원래 넓다 — 합산이 좁아지는지가 분산의 요점)
            for tag, cols in (("combo_all", list(df.columns)), ("usable", usable)):
                if not cols:
                    continue
                e2 = df[cols].mean(axis=1)
                b2 = np.array([e2.to_numpy()[rng.integers(0, len(e2), len(e2))].mean() for _ in range(B_BOOT)])
                row[tag + "_assets"] = cols
                row[tag + "_sharpe"] = sh(e2); row[tag + "_daily_mean"] = round(float(e2.mean()), 3)
                row[tag + "_daily_ci95"] = [round(float(np.percentile(b2, 2.5)), 3), round(float(np.percentile(b2, 97.5)), 3)]
                row[tag + "_pos_day_frac"] = round(float((e2 > 0).mean()), 3)
                row[tag + "_worst_day"] = round(float(e2.min()), 2); row[tag + "_worst5_sum"] = round(float(e2.nsmallest(5).sum()), 2)
            if False:
                e = df[usable].mean(axis=1)
                bo = np.array([e.to_numpy()[rng.integers(0, len(e), len(e))].mean() for _ in range(B_BOOT)])
                row |= {"usable_sharpe": sh(e), "usable_daily_mean": round(float(e.mean()), 3),
                        "usable_daily_ci95": [round(float(np.percentile(bo, 2.5)), 3), round(float(np.percentile(bo, 97.5)), 3)],
                        "usable_pos_day_frac": round(float((e > 0).mean()), 3),
                        "usable_worst_day": round(float(e.min()), 2), "usable_worst5_sum": round(float(e.nsmallest(5).sum()), 2)}
            port[w]["by_cap"][f"cap{cap}"] = row
    out["portfolio"] = port
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for w in SPLITS:
        for cap in CAPS:
            p = port[w]["by_cap"].get(f"cap{cap}")
            if p and "combo_all_sharpe" in p:
                log(f"  [{w} cap{cap}] ETH단독 샤프 {p['eth_only_sharpe']} (일 {p['eth_only_daily_mean']}) · {p['combo_all_assets']} 합산 샤프 {p['combo_all_sharpe']} 일 {p['combo_all_daily_mean']} CI {p['combo_all_daily_ci95']} 양의날 {p['combo_all_pos_day_frac']} 최악일 {p['combo_all_worst_day']}")
        p2 = port[w]["by_cap"].get("cap2")
        if p2: log(f"  [{w}] 자산별 샤프 {p2['per_asset_sharpe']} · 상관 " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != k} for k, v in p2['corr'].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
