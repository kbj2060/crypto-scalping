#!/usr/bin/env python3
"""ETH 복합 알고리즘 2라운드 — **조기 손절(C4)** · ATR 교란 점검(C5) · 위험기반 사이징(C6) (2026-09-05).

1라운드(`research_eth_composite_direction_trend_pullback_20260905.py`)에서 나온 두 관찰의 후속:
  (a) C3 되돌림 지정가: **체결된(=되돌림이 온) 건의 시장가 성과가 −1 ~ −13bp**로 전체(+4.4/+6.8)보다 훨씬 나쁘다.
      "되돌림을 기다려 진입"은 역선택이라 실패했지만, 뒤집으면 **인과적으로 실행 가능한 규칙**이 된다 —
      진입 후 첫 N봉 안에 k×ATR 역행하면 **조기 손절**(현행 손절은 5.0×ATR).
  (b) C2 상태축: atr_pct·atr_percentile_864·activity의 Q5−Q1 갭이 세 창 전부 양수. 그런데 **청산 배리어가
      ATR 배수로 정의**돼 있으므로 bp 손익은 ATR에 기계적으로 비례한다 → 교란 의심(09-04 청산맵 ATR 아티팩트와 같은 부류).

  C4 조기 손절   k ∈ {0.15,0.2,0.3,0.4,0.5,0.75,1.0} × N ∈ {1,2,3,4,6} 시간제한 타이트 스톱.
                 대조군 = 시간제한 없는 상시 SL 축소(sl ∈ {0.5,1,2,3,5}). **TRAIN으로만 셀 선택 → VAL/OOS 1회 확인.**
  C5 ATR 교란    C2의 상태 갭을 **ATR 정규화 손익**(bp / atr_pct = ATR 배수)으로 다시 계산. 갭이 사라지면 아티팩트.
  C6 위험 사이징 사전등록 라이브 계약(위험 0.4%/건, notional = 0.004/(5·atr_pct) 상한 0.5, 레버리지 3)로
                 자기자본 기준 손익을 계산 → 저ATR을 크게 잡으므로 ATR 상태 우위가 뒤집히는지 확인.

판정은 1라운드 사전등록과 동일: VAL·OOS 두 창 모두 R 대비 일별 짝비교 CI 하한 > 0.
⚠️C4는 1라운드 결과를 보고 세운 **사후 가설**이다 — TRAIN 선택 + VAL/OOS 1회 확인이라도 전진 섀도우 없이는 승격 근거가 아니다.
HOLDOUT 미접촉.
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


C1M = _load("comp1", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
build, pf, cand_of, day_paired, gap_day_ci, sim_exit = C1M.build, C1M.pf, C1M.cand_of, C1M.day_paired, C1M.gap_day_ci, C1M.sim_exit
CELL, FWD, COST, CAP, WINDOWS = C1M.CELL, C1M.FWD, C1M.COST, C1M.CAP, C1M.WINDOWS
OUT = ROOT / "data/research/eth_composite_early_stop_20260905"
STOP_K = (0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
STOP_N = (1, 2, 3, 4, 6)
SL_CTRL = (0.5, 1.0, 2.0, 3.0, 5.0)
RISK_FRAC, LEV, NOTIONAL_CAP = 0.004, 3.0, 0.5
rng = np.random.default_rng(20260905)


def log(m): print(f"[c2round] {m}", flush=True)


def adverse_touch(B, k, N):
    """진입(open[i+1]) 후 첫 N봉 안에 k×ATR **역행**했는가 (인과: 완결 봉 고가/저가). 반환 (touched, bar_off 1..N)."""
    bidx, atr, cs, h, l = B["bidx"], B["atr"], B["cont_sign"], B["h"], B["l"]
    ref = B["entry"]; stop = ref - cs * k * atr
    tou = np.zeros(len(bidx), bool); off = np.zeros(len(bidx), int)
    for step in range(1, N + 1):
        b = bidx + step
        hit = np.where(cs > 0, l[b] <= stop, h[b] >= stop) & ~tou
        off = np.where(hit, step, off); tou = tou | hit
    return tou, off, stop


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = build()
    pos, split, ts, cont_bp, cont_ex, atr, cs = B["pos"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"], B["atr"], B["cont_sign"]
    entry = B["entry"]; atr_pct = atr / entry
    base = {w: pf(cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "n": int(len(pos)), "holdout_touched": False,
           "note": "C4는 1라운드 결과를 보고 세운 사후 가설 — TRAIN 선택 후 VAL/OOS 1회 확인",
           "baseline": {w: base[w]["stats"] for w in WINDOWS}}

    # ---------------- C4 조기 손절
    log("C4 조기 손절 격자 …")
    cells = {}
    for k in STOP_K:
        for N in STOP_N:
            tou, off, stop = adverse_touch(B, k, N)
            pnl_stop = cs * (stop - entry) / entry * 1e4 - COST         # = −k·atr_pct·1e4 − COST
            pnl = np.where(tou, pnl_stop, cont_bp)
            ex_bar = np.where(tou, pos + off, pos + 1 + cont_ex)        # 조기 청산은 슬롯을 일찍 비운다
            rec = {"touch_rate_all": round(float(tou.mean()), 3), "windows": {}}
            for w in WINDOWS:
                m = split == w
                r = pf(cand_of(ts[m], pos[m] + 1, ex_bar[m], pnl[m]))
                if r is None:
                    continue
                rec["windows"][w] = {**{x: r["stats"][x] for x in ("n", "exp_bp", "win_rate", "day_ci95", "per_day", "daily_mean_bp", "daily_sharpe_ann", "max_dd_bp")},
                                     "touch_rate": round(float(tou[m].mean()), 3),
                                     "vs_R": day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
            cells[f"k{k}_N{N}"] = rec
    rep["C4_early_stop"] = {"cells": cells}
    # 대조군: 시간제한 없는 상시 SL 축소 (같은 셀의 arm/trail 유지)
    ctrl = {}
    for sl in SL_CTRL:
        st0 = B["bidx"] + 1; idx = st0[:, None] + np.arange(FWD)
        ret, ex = sim_exit(entry, atr, cs, B["h"][idx], B["l"][idx], B["c"][idx], sl, CELL[1], CELL[2])
        p = ret * 1e4 - COST; rec = {}
        for w in WINDOWS:
            m = split == w
            r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + ex[m], p[m]))
            if r is None:
                continue
            rec[w] = {**{x: r["stats"][x] for x in ("n", "exp_bp", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                      "vs_R": day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
        ctrl[f"sl{sl}"] = rec
    rep["C4_control_plain_sl"] = ctrl
    # TRAIN 선택 (일 평균 자본손익 최대, 사전 고정 잣대) → VAL/OOS 1회 확인
    ok = {c: v for c, v in cells.items() if "TRAIN" in v["windows"]}
    pick = max(ok, key=lambda c: ok[c]["windows"]["TRAIN"]["daily_mean_bp"])
    pick_sharpe = max(ok, key=lambda c: ok[c]["windows"]["TRAIN"]["daily_sharpe_ann"] or -9)
    rep["C4_train_pick"] = {"by_daily_mean": pick, "by_sharpe": pick_sharpe,
                            "confirm": {c: {w: cells[c]["windows"].get(w) for w in WINDOWS} for c in {pick, pick_sharpe}}}

    # ---------------- C5 ATR 교란 점검 (ATR 배수 손익으로 상태 갭 재계산)
    log("C5 ATR 교란 점검 …")
    cont_atr = (cont_bp + COST) / (atr_pct * 1e4) - COST / (atr_pct * 1e4)    # = 순수 ATR 배수 (비용은 bp라 ATR로 환산)
    cont_atr = (cont_bp / 1e4) / atr_pct                                      # 비용 포함 ATR 배수 (해석 단순화)
    c5 = {}
    for name, (kind, raw) in B["S"].items():
        x = raw * cs if kind == "aligned" else raw.astype(float)
        fin = np.isfinite(x); tr = split == "TRAIN"
        if (fin & tr).sum() < 500:
            continue
        qs = np.quantile(x[fin & tr], [0.2, 0.4, 0.6, 0.8]); qi = np.where(fin, np.digitize(x, qs), -1)
        d = {}
        for w in WINDOWS:
            m = (split == w) & fin; hi = m & (qi == 4); lo = m & (qi == 0)
            if hi.sum() < 30 or lo.sum() < 30:
                continue
            d[w] = {"gap_bp": round(float(cont_bp[hi].mean() - cont_bp[lo].mean()), 2),
                    "gap_atr_mult": round(float(cont_atr[hi].mean() - cont_atr[lo].mean()), 4),
                    "gap_atr_ci95": gap_day_ci(cont_atr[hi], ts[hi], cont_atr[lo], ts[lo]),
                    "atr_pct_hi": round(float(atr_pct[hi].mean()), 5), "atr_pct_lo": round(float(atr_pct[lo].mean()), 5)}
        c5[name] = d
    rep["C5_atr_normalized_state_gap"] = c5

    # ---------------- C6 위험기반 사이징 (사전등록 라이브 계약)
    log("C6 위험기반 사이징 …")
    notional = np.minimum(RISK_FRAC / (CELL[0] * atr_pct), NOTIONAL_CAP)      # 위험 0.4% = notional × 5×atr_pct
    eq_bp = cont_bp * notional                                                # 자기자본 대비 bp
    c6 = {"notional_median": round(float(np.median(notional)), 4), "notional_capped_frac": round(float((notional >= NOTIONAL_CAP).mean()), 3), "windows": {}}
    for w in WINDOWS:
        m = split == w
        r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], eq_bp[m]))
        c6["windows"][w] = {**{x: r["stats"][x] for x in ("n", "exp_bp", "day_ci95", "daily_mean_bp", "daily_sharpe_ann", "max_dd_bp")}}
        # ATR 상태 갭을 자기자본 기준으로
        for nm in ("atr_pct", "atr_percentile_864", "ax_activity"):
            if nm not in B["S"]:
                continue
            x = B["S"][nm][1].astype(float); fin = np.isfinite(x); tr = split == "TRAIN"
            qs = np.quantile(x[fin & tr], [0.2, 0.4, 0.6, 0.8]); qi = np.where(fin, np.digitize(x, qs), -1)
            hi = m & fin & (qi == 4); lo = m & fin & (qi == 0)
            if hi.sum() > 30 and lo.sum() > 30:
                c6["windows"][w].setdefault("state_gap_equity", {})[nm] = {
                    "gap_equity_bp": round(float(eq_bp[hi].mean() - eq_bp[lo].mean()), 2),
                    "gap_price_bp": round(float(cont_bp[hi].mean() - cont_bp[lo].mean()), 2),
                    "ci95_equity": gap_day_ci(eq_bp[hi], ts[hi], eq_bp[lo], ts[lo])}
    rep["C6_risk_sizing"] = c6

    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for c in {pick, pick_sharpe}:
        for w in WINDOWS:
            v = cells[c]["windows"].get(w)
            if v:
                log(f"  C4 {c} {w}: exp={v['exp_bp']}bp n={v['n']} 일평균={v['daily_mean_bp']} 샤프={v['daily_sharpe_ann']} "
                    f"터치율={v['touch_rate']} ΔR={v['vs_R']['diff_bp_day']}{v['vs_R']['ci95']} 이긴날={v['vs_R']['win_day_frac']}")


if __name__ == "__main__":
    main()
