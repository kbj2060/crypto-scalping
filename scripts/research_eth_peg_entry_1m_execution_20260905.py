#!/usr/bin/env python3
"""비용 지렛대 2 — **1분봉 집행 모델**로 peg 진입 재측정 (2026-09-05).

1차(`research_eth_peg_entry_then_cross_cost_lever_20260905.py`)는 미체결 시 **5분봉 종가**에 크로스했다.
그건 최대 페널티 가정이다 — 실제 집행 알고리즘은 수십 초 안에 넘긴다. 그 결과 수수료를 2.2bp 아끼고도
**평균 가격 이득이 −0.9bp**로 상쇄됐다(미체결 5%가 강한 지속 구간이라 5분 뒤 크로스가 비싸다).

여기서는 **1분봉**으로 대기 시간을 나눈다:
    진입봉(5분) 시가에 지정가 = o − sign·k·ATR 게시 → **W분(W ∈ 1,2,3) 안 체결되면 그 1분봉 종가에 크로스**
    체결: 진입가 = 지정가, 비용 **7.8bp**(peg 진입 + 테이커 청산)
    크로스: 진입가 = 해당 1분봉 종가, 비용 10bp
    ⚠️모든 발동이 거래된다 — 역선택 없음. 청산 관리는 A·B 모두 **5분봉 i+2부터**(체결 이전 크레딧 금지).
    파리티: 1분봉을 5분으로 재집계해 5분봉 OHLC와 일치하는지 먼저 검증(불일치 시 중단).
판정: VAL·OOS 두 창 모두 A(시장가·같은 관리 시점·10bp) 대비 일별 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_p1", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "data/research/eth_peg_entry_1m_execution_20260905"
# ⚠️k=0.0은 **동어반복**이라 판정에서 제외한다 -- 지정가 = 봉 시가면 "저가 ≤ 시가"가 정의상 항상 참이라
#   체결률이 1.000이 나온다. 그건 "시장가에 항상 메이커 수수료로 체결된다"를 가정한 것이지 측정이 아니다.
#   (참고용으로 계산은 하되 verdict에서 뺀다 -- 상한선 표시.)
KS, WS = (0.0, 0.01, 0.02, 0.03, 0.05), (1, 2, 3)
DEGENERATE_K = 0.0
COST_TAKER, COST_PEG, WINDOWS = 10.0, 7.8, ("TRAIN", "VAL", "OOS")


def log(m): print(f"[peg1m] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, split, ts, bidx = B["pos"], B["split"], B["ts"], B["bidx"]
    cont_bp, cont_ex, atr, cs = B["cont_bp"], B["cont_ex"], B["atr"], B["cont_sign"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]; n = len(c); FWD = C1.FWD
    bar_ts = pd.DatetimeIndex(B["bar"]["timestamp"].to_numpy()); p_first = B["p_first"]
    t0_seg = bar_ts[0]

    m1 = pd.read_csv(KL1, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    j0 = int(np.searchsorted(m1["timestamp"].to_numpy(), np.datetime64(t0_seg)))
    assert m1["timestamp"].iloc[j0] == t0_seg, "1분봉 시작 정렬 실패"
    need = n * 5 + 10
    seg1 = m1.iloc[j0:j0 + need].reset_index(drop=True)
    assert np.all(np.diff(seg1["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int) == 1), "1분봉 구간 비연속"
    o1, h1, l1, c1 = (seg1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    # 파리티: 1분봉 5개 재집계 == 5분봉
    chk = np.arange(200, min(n - 5, 5000))
    a5 = o1[chk * 5]; hi5 = np.stack([h1[chk * 5 + k] for k in range(5)]).max(0); lo5 = np.stack([l1[chk * 5 + k] for k in range(5)]).min(0)
    par = max(float(np.abs(a5 - o[chk]).max()), float(np.abs(hi5 - h[chk]).max()), float(np.abs(lo5 - l[chk]).max()))
    log(f"1분↔5분 재집계 파리티 |Δ|max {par:.2e} (0이어야 함)")
    assert par < 1e-8, "1분/5분 불일치 -- 중단"

    st2 = bidx + 2; ok = (st2 + FWD) < n
    ix2 = np.where(ok[:, None], st2[:, None] + np.arange(FWD), 0)
    H2, L2, C2 = h[ix2], l[ix2], c[ix2]
    ref = o[bidx + 1]; med = float(np.median(atr / ref))

    def run(entry_px, cost):
        r, ex = C1.sim_exit(entry_px, atr, cs, H2, L2, C2, *C1.CELL)
        return r * 1e4 - cost, ex

    anchor = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    pA, exA = run(ref, COST_TAKER)
    base = {w: C1.pf(C1.cand_of(ts[ok & (split == w)], pos[ok & (split == w)] + 2, pos[ok & (split == w)] + 2 + exA[ok & (split == w)], pA[ok & (split == w)])) for w in WINDOWS}
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "median_atr_pct": round(med, 5), "parity_1m_5m": par,
           "cost_taker_bp": COST_TAKER, "cost_peg_bp": COST_PEG, "holdout_touched": False,
           "anchor_deployed_R": {w: anchor[w]["stats"] for w in WINDOWS},
           "A_market_from_i2": {w: base[w]["stats"] for w in WINDOWS}, "arms": {}}
    m1base = bidx * 5 + 5                                              # 진입 5분봉의 첫 1분봉 인덱스
    for k, W in itertools.product(KS, WS):
        lim = ref - cs * k * atr
        filled = np.zeros(len(bidx), bool)
        for u in range(W):
            b = m1base + u
            hit = np.where(cs > 0, l1[b] <= lim, h1[b] >= lim) & ~filled
            filled |= hit
        px_fill = np.where(cs > 0, np.minimum(lim, o1[m1base]), np.maximum(lim, o1[m1base]))
        entry_px = np.where(filled, px_fill, c1[m1base + W - 1])       # 미체결 → W번째 1분봉 종가에 크로스
        cost = np.where(filled, COST_PEG, COST_TAKER)
        pB, exB = run(entry_px, cost)
        rec = {"limit_bp_at_median_atr": round(k * med * 1e4, 2), "wait_minutes": W, "fill_rate_all": round(float(filled[ok].mean()), 3)}
        for w in WINDOWS:
            m = ok & (split == w)
            rB = C1.pf(C1.cand_of(ts[m], pos[m] + 2, pos[m] + 2 + exB[m], pB[m]))
            rec[w] = {"fill_rate": round(float(filled[m].mean()), 3), "exp_bp": rB["stats"]["exp_bp"],
                      "day_ci95": rB["stats"]["day_ci95"], "daily_sharpe_ann": rB["stats"]["daily_sharpe_ann"],
                      "mean_price_edge_bp": round(float((cs[m] * (ref[m] - entry_px[m]) / ref[m]).mean() * 1e4), 2),
                      "mean_cost_bp": round(float(cost[m].mean()), 2),
                      "vs_A": C1.day_paired(rB["pnl"], rB["ts"], base[w]["pnl"], base[w]["ts"]),
                      "vs_anchor": C1.day_paired(rB["pnl"], rB["ts"], anchor[w]["pnl"], anchor[w]["ts"])}
        rep["arms"][f"k{k}_W{W}m"] = rec
        log(f"  k={k:<5}({rec['limit_bp_at_median_atr']:>4}bp) W={W}분 체결 {rec['fill_rate_all']:.3f} · " + " | ".join(
            f"{w} exp={rec[w]['exp_bp']:>6} 가격 {rec[w]['mean_price_edge_bp']:>5} 비용 {rec[w]['mean_cost_bp']:>5} ΔA={rec[w]['vs_A']['diff_bp_day']:>6}{str(rec[w]['vs_A']['ci95']):>16}" for w in WINDOWS))
    P = [nm for nm, r in rep["arms"].items()
         if not nm.startswith(f"k{DEGENERATE_K}_") and r["VAL"]["vs_A"]["ci95"][0] > 0 and r["OOS"]["vs_A"]["ci95"][0] > 0]
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 A 대비 CI 하한 > 0 (k=0.0 동어반복 팔 제외)",
                      "excluded_degenerate": [nm for nm in rep["arms"] if nm.startswith(f"k{DEGENERATE_K}_")],
                      "degenerate_note": "지정가=봉 시가면 '저가<=시가'가 항상 참 -> 체결률 1.000. 상한선일 뿐 측정이 아니다.",
                      "passes": P, "n_pass": len(P)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s · 통과 {len(P)} {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
