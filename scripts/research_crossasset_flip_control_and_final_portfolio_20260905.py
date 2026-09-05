#!/usr/bin/env python3
"""교차자산 지속 규칙 — **방향뒤집기 대조군 + 격자 하향 확장 + 최종 다자산 포트폴리오** (2026-09-05).

앞 라운드에서 자산별 TRAIN 선택 셀이 전부 **arm=1.0(격자 하한)·trail=0.02(격자 하한)** 코너로 갔다.
이건 이 저장소가 이미 아티팩트로 기록해 둔 자리다 — 「낮은 ARM은 방향예측과 무관하게 봉 노이즈만으로
거의 항상 무장·수확해 승률 97%를 만든다」(memory `feedback_trailing_stop_low_arm_noise_harvest_artifact`,
호메로스 §5.8). 그 기록이 명시한 처방은 하나다: **방향뒤집기(direction-flip) 대조군을 격자 전체에 적용**.

  A 격자 하향 확장  arm ∈ {0.25,0.5,0.75,1.0,1.5,2.0,3.0} × trail ∈ {0.01,0.02,0.03,0.05,0.10,0.20} × sl {3~16}
                   → 코너가 계속 밀리는지 확인(밀리면 아티팩트 강한 증거).
  B 방향뒤집기      같은 셀·같은 발동 봉에서 방향만 뒤집은 팔(= 페이드). **진짜 신호라면 real이 flipped를
                   VAL·OOS 두 창에서 명확히 이겨야 한다.** real−flipped 일별 짝비교 CI로 판정.
  C 셀 선택 재정의  TRAIN에서 (일 샤프 최대)가 아니라 **(real−flipped TRAIN 짝비교 CI 하한 > 0) 인 셀 중
                   일 샤프 최대**로 바꾼다 — 아티팩트 코너를 구조적으로 배제한다.
  D 최종 포트폴리오 C의 셀로 자산별 위험정규화 일손익 → 상관·동일가중 샤프, 자산당 CAP {1,2,5}.
전부 자기자본 bp(notional = min(0.004/(sl·atr_pct), 0.5)). HOLDOUT 차단. 연구/개발 점수.
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


XA = _load("xa_f", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
CC = _load("cc_f", "scripts/research_crossasset_cell_calibration_and_portfolio_20260905.py")
BR = _load("br_f", "scripts/research_crossasset_cell_boundary_and_risk_portfolio_20260905.py")
V2 = _load("hev2_f", "scripts/research_homer_entry_v2_20260904.py")
C1M = _load("comp1_f", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, day_boot = V2.sim_exit, V2.day_boot
eq_pnl, run_pf = BR.eq_pnl, BR.run_pf
day_paired = C1M.day_paired
OUT = ROOT / "data/research/crossasset_flip_control_final_20260905"
FWD, COST, SPLITS, CAPS = XA.FWD, XA.COST, XA.SPLITS, (1, 2, 5)
SL_G = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
ARM_G = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TRAIL_G = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
B_BOOT = 1000
rng = np.random.default_rng(20260905)


def log(m): print(f"[flip] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
           "grid": {"sl": SL_G, "arm": ARM_G, "trail": TRAIL_G, "n_cells": len(SL_G) * len(ARM_G) * len(TRAIL_G)},
           "selection": "TRAIN real−flipped 일별 짝비교 CI 하한 > 0 인 셀 중 일 샤프 최대 (자기자본 bp)",
           "holdout_excluded": True, "assets": {}}
    daily = {w: {c: {} for c in CAPS} for w in SPLITS}
    for sym, ref in XA.ASSETS.items():
        D = CC.prep(sym, ref); tsi = pd.DatetimeIndex(D["ts"])
        M = {w: np.asarray((tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))) for w, (a, b) in SPLITS.items()}
        atr_pct = D["atr"][D["i"]] / D["o"][D["i"] + 1]
        e0, a0, s0, H, L, C = D["o"][D["i"] + 1], D["atr"][D["i"]], D["sign"], D["h"][D["ix"]], D["l"][D["ix"]], D["c"][D["ix"]]
        grid = {}
        mt = M["TRAIN"]
        for cell in itertools.product(SL_G, ARM_G, TRAIL_G):
            rr, xr = sim_exit(e0, a0, s0, H, L, C, *cell)
            rf, xf = sim_exit(e0, a0, -s0, H, L, C, *cell)
            pr = eq_pnl(rr * 1e4 - COST, atr_pct, cell[0]); pfl = eq_pnl(rf * 1e4 - COST, atr_pct, cell[0])
            R = run_pf(pr[mt], D["ts"][mt], D["i"][mt] + 1, D["i"][mt] + 1 + xr[mt], 5)
            Fp = run_pf(pfl[mt], D["ts"][mt], D["i"][mt] + 1, D["i"][mt] + 1 + xf[mt], 5)
            if R is None or Fp is None:
                continue
            dp = day_paired(R["_pnl"], R["_ts"], Fp["_pnl"], Fp["_ts"], B=300)   # 격자 단계는 가벼운 부트스트랩
            lo, hi = day_boot(R["_pnl"], R["_ts"], 300, rng)
            grid["_".join(map(str, cell))] = {"sharpe": R["daily_sharpe_ann"], "exp_bp": R["exp_bp"], "win_rate": R["win_rate"],
                                              "flip_exp_bp": Fp["exp_bp"], "flip_win_rate": Fp["win_rate"],
                                              "real_minus_flip": dp["diff_bp_day"], "rmf_ci95": dp["ci95"], "day_ci95": [round(lo, 4), round(hi, 4)]}
        elig = {k: v for k, v in grid.items() if v["rmf_ci95"][0] > 0 and v["day_ci95"][0] > 0 and v["sharpe"] is not None}
        # 아티팩트 진단: 샤프 최상위 셀이 뒤집기 대조를 통과하는가
        best_sharpe = max(grid, key=lambda k: grid[k]["sharpe"] or -9)
        pick = max(elig, key=lambda k: elig[k]["sharpe"]) if elig else None
        A = {"symbol": sym, "n_cells": len(grid), "n_eligible_flip_and_ci": len(elig),
             "best_sharpe_cell_ignoring_flip": {"cell": best_sharpe, **grid[best_sharpe]},
             "picked_cell": pick, "picked": grid.get(pick), "eth_default_5_1.5_0.1": grid.get("5.0_1.5_0.1"), "windows": {}}
        if pick is None:
            A["usable"] = False; out["assets"][sym] = A
            log(f"{sym}: 뒤집기 대조 통과 셀 0/{len(grid)} — 사용 불가 (최고샤프 셀 {best_sharpe} 승률 {grid[best_sharpe]['win_rate']} real−flip {grid[best_sharpe]['real_minus_flip']}{grid[best_sharpe]['rmf_ci95']})")
            continue
        cell = tuple(float(x) for x in pick.split("_"))
        A["picked_cell_tuple"] = cell
        A["boundary_interior"] = {"sl": SL_G[0] < cell[0] < SL_G[-1], "arm": ARM_G[0] < cell[1] < ARM_G[-1], "trail": TRAIL_G[0] < cell[2] < TRAIL_G[-1]}
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
                    "day_ci95": [round(lo, 4), round(hi, 4)],
                    "flip_exp_bp": Fp["exp_bp"] if Fp else None, "flip_sharpe": Fp["daily_sharpe_ann"] if Fp else None,
                    "real_minus_flip": day_paired(R["_pnl"], R["_ts"], Fp["_pnl"], Fp["_ts"]) if Fp else None}
                daily[w][cap][sym] = R["_daily"]
            A["windows"][w] = rec
        A["usable"] = all((A["windows"].get(w, {}).get("cap2", {}).get("exp_bp", -9) > 0 and
                           (A["windows"].get(w, {}).get("cap2", {}).get("real_minus_flip") or {}).get("ci95", [-9])[0] > 0) for w in ("VAL", "OOS"))
        out["assets"][sym] = A
        log(f"{sym}: 셀 {cell} 내부 {A['boundary_interior']} 적격 {len(elig)}/{len(grid)} · " +
            " · ".join(f"{w} exp={A['windows'][w]['cap2']['exp_bp']:.3f} 샤프={A['windows'][w]['cap2']['daily_sharpe_ann']} 승률={A['windows'][w]['cap2']['win_rate']} "
                       f"real−flip={A['windows'][w]['cap2']['real_minus_flip']['diff_bp_day']}{A['windows'][w]['cap2']['real_minus_flip']['ci95']}" for w in A["windows"]) + f" · 쓸만={A['usable']}")
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
                   "eth_only_sharpe": sh(df["ETHUSDT"]) if "ETHUSDT" in df else None}
            if usable:
                e = df[usable].mean(axis=1)
                bo = np.array([e.to_numpy()[rng.integers(0, len(e), len(e))].mean() for _ in range(B_BOOT)])
                row |= {"usable_sharpe": sh(e), "usable_daily_mean": round(float(e.mean()), 3),
                        "usable_daily_ci95": [round(float(np.percentile(bo, 2.5)), 3), round(float(np.percentile(bo, 97.5)), 3)],
                        "usable_pos_day_frac": round(float((e > 0).mean()), 3)}
            port[w]["by_cap"][f"cap{cap}"] = row
    out["portfolio"] = port
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for w in SPLITS:
        for cap in CAPS:
            p = port[w]["by_cap"].get(f"cap{cap}")
            if p and "usable_sharpe" in p:
                log(f"  [{w} cap{cap}] ETH단독 {p['eth_only_sharpe']} · {usable} 합산 샤프 {p['usable_sharpe']} 일평균 {p['usable_daily_mean']} CI {p['usable_daily_ci95']} 양의날 {p['usable_pos_day_frac']}")


if __name__ == "__main__":
    main()
