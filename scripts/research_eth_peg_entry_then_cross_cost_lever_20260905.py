#!/usr/bin/env python3
"""**비용 지렛대** — 메이커 peg 진입 후 미체결 시 크로스 (2026-09-05).

R의 gross는 +14.9(TRAIN)/+14.4(VAL)/+16.8(OOS)bp다. 비용 10bp를 09-04 실측 메이커 왕복 **7.8bp**
(peg 2.76bp/leg 진입 + 테이커 청산)로 낮추면 net이 **+32~50%** 오른다 — 이 세션에서 확인된 가장 큰 남은 지렛대.
그런데 C3(되돌림 지정가)가 보인 대로 **지정가는 역선택**을 부른다. 그 실험은 k=0.10×ATR(≈2.5bp)부터
시작했는데 실제 peg 거리는 그보다 **한 자릿수 작다**(스프레드 ≈0.5~1bp = 0.02~0.04×ATR).

## 핵심 차이 — 미체결이면 **크로스한다**(거래를 버리지 않는다)
C3은 미체결 = 거래 없음이라 **선택 효과**가 생겼다. 실제 집행 알고리즘은 한 봉 걸어두고 안 되면 넘긴다.
그러면 **모든 발동이 거래되므로 역선택이 원천적으로 없다** — 남는 건 가격·수수료 효과뿐이다.

    A 기준(현행)  o[i+1] 시장가 · 비용 10bp
    B peg→크로스  지정가 = o[i+1] − sign·k·ATR, 봉 i+1 안 체결되면 그 봉 **종가에 크로스**
                  체결: 진입가 = 지정가, 비용 **7.8bp**(peg 진입 + 테이커 청산)
                  크로스: 진입가 = c[i+1], 비용 10bp
    k ∈ {0.01,0.02,0.03,0.05,0.10,0.20}×ATR (중앙 atr_pct 0.25% 기준 ≈ 0.25/0.5/0.75/1.25/2.5/5.0bp)
    ⚠️청산 관리는 **양 팔 모두 봉 i+2부터**(체결 이전 고가/저가 크레딧 금지). 현행 R(i+1부터)도 앵커로 병기.
    대조: 체결분만 시장가(C3식 선택 효과 분리) · 비용을 10으로 고정한 변형(가격 효과만)
판정: VAL·OOS 두 창 모두 A 대비 일별 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
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


C1 = _load("c1_peg", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OUT = ROOT / "data/research/eth_peg_entry_cost_lever_20260905"
KS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
COST_TAKER, COST_PEG = 10.0, 7.8
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[peg] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, split, ts, bidx = B["pos"], B["split"], B["ts"], B["bidx"]
    cont_bp, cont_ex, atr, cs = B["cont_bp"], B["cont_ex"], B["atr"], B["cont_sign"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]; n = len(c); FWD = C1.FWD
    ref = o[bidx + 1]; atr_pct = atr / ref
    med = float(np.median(atr_pct))
    # 양 팔 공통: 봉 i+2부터 관리 (체결 이전 크레딧 금지)
    st2 = bidx + 2
    ok = (st2 + FWD) < n
    ix2 = np.where(ok[:, None], st2[:, None] + np.arange(FWD), 0)
    H2, L2, C2 = h[ix2], l[ix2], c[ix2]

    def arm(entry_px, cost):
        r, ex = C1.sim_exit(entry_px, atr, cs, H2, L2, C2, *C1.CELL)
        return r * 1e4 - cost, ex

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "median_atr_pct": round(med, 5),
           "cost_taker_bp": COST_TAKER, "cost_peg_roundtrip_bp": COST_PEG, "holdout_touched": False,
           "n_usable": int(ok.sum()), "arms": {}}
    # 앵커: 현행 R (봉 i+1부터 관리, 시장가, 10bp) -- 배포본 그 자체
    anchor = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    rep["anchor_deployed_R"] = {w: anchor[w]["stats"] for w in WINDOWS}
    # A: 시장가 + 봉 i+2부터 관리 (비교 기준 -- 노출 시점 차이를 제거)
    pA, exA = arm(ref, COST_TAKER)
    base = {}
    for w in WINDOWS:
        m = ok & (split == w)
        base[w] = C1.pf(C1.cand_of(ts[m], pos[m] + 2, pos[m] + 2 + exA[m], pA[m]))
    rep["arms"]["A_market_from_i2"] = {w: {**{k: base[w]["stats"][k] for k in ("n", "exp_bp", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                                           "vs_anchor": C1.day_paired(base[w]["pnl"], base[w]["ts"], anchor[w]["pnl"], anchor[w]["ts"])} for w in WINDOWS}
    log("A(시장가·i+2부터): " + " | ".join(f"{w} exp={rep['arms']['A_market_from_i2'][w]['exp_bp']:>6} (앵커 {anchor[w]['stats']['exp_bp']:>6})" for w in WINDOWS))
    for k in KS:
        lim = ref - cs * k * atr
        b1 = bidx + 1
        filled = np.where(cs > 0, l[b1] <= lim, h[b1] >= lim)
        px_fill = np.where(cs > 0, np.minimum(lim, ref), np.maximum(lim, ref))     # 시가가 이미 지정가 너머면 시가 체결
        entry_px = np.where(filled, px_fill, c[b1])                                 # 미체결 → 그 봉 종가에 크로스
        cost = np.where(filled, COST_PEG, COST_TAKER)
        pB, exB = arm(entry_px, cost)
        pB10, _ = arm(entry_px, COST_TAKER)                                         # 가격 효과만 (비용 고정)
        rec = {"limit_bp_at_median_atr": round(k * med * 1e4, 2), "fill_rate_all": round(float(filled[ok].mean()), 3)}
        for w in WINDOWS:
            m = ok & (split == w)
            rB = C1.pf(C1.cand_of(ts[m], pos[m] + 2, pos[m] + 2 + exB[m], pB[m]))
            r10 = C1.pf(C1.cand_of(ts[m], pos[m] + 2, pos[m] + 2 + exB[m], pB10[m]))
            mf = m & filled
            rSel = C1.pf(C1.cand_of(ts[mf], pos[mf] + 2, pos[mf] + 2 + exA[mf], pA[mf])) if mf.sum() > 100 else None
            rec[w] = {"fill_rate": round(float(filled[m].mean()), 3),
                      "peg_exp_bp": rB["stats"]["exp_bp"], "peg_day_ci95": rB["stats"]["day_ci95"],
                      "peg_sharpe": rB["stats"]["daily_sharpe_ann"],
                      "price_only_exp_bp_cost10": r10["stats"]["exp_bp"],
                      "vs_A": C1.day_paired(rB["pnl"], rB["ts"], base[w]["pnl"], base[w]["ts"]),
                      "vs_anchor": C1.day_paired(rB["pnl"], rB["ts"], anchor[w]["pnl"], anchor[w]["ts"]),
                      "selection_filled_only_market_exp_bp": rSel["stats"]["exp_bp"] if rSel else None,
                      "mean_price_edge_bp": round(float((cs[m] * (ref[m] - entry_px[m]) / ref[m]).mean() * 1e4), 2)}
        rep["arms"][f"B_peg{k}"] = rec
        log(f"  k={k:<5} (지정가 {rec['limit_bp_at_median_atr']:>4}bp) 체결률 {rec['fill_rate_all']:.3f} · " + " | ".join(
            f"{w} exp={rec[w]['peg_exp_bp']:>6} 가격이득 {rec[w]['mean_price_edge_bp']:>5}bp ΔA={rec[w]['vs_A']['diff_bp_day']:>6}{str(rec[w]['vs_A']['ci95']):>16} 체결분시장가 {rec[w]['selection_filled_only_market_exp_bp']}" for w in WINDOWS))
    P = [nm for nm, rec in rep["arms"].items() if nm.startswith("B_")
         and rec["VAL"]["vs_A"]["ci95"][0] > 0 and rec["OOS"]["vs_A"]["ci95"][0] > 0]
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 A 대비 짝비교 CI 하한 > 0", "passes": P, "n_pass": len(P)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · 통과 {len(P)} {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
