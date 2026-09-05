#!/usr/bin/env python3
"""페이드 구제 시도 — **분할 매수**와 **비대칭 TP/SL 브래킷** (2026-09-05).

사용자: *"맞는 경우가 더 자주 있다면 분할 매수나 틀려도 손해를 덜 보는 방향으로 가면
맞는 수익을 극대화할 수 있지 않을까?"*

부록 §15가 잰 것: 페이드 승률 0.54~0.56(기저 0.50), 손익비 0.74~0.92, 평균 ≈ 0, 비용 후 −6.5~−14.2bp.
두 제안은 성격이 다르다.

  A **분할 매수**(불리하게 움직이면 추가 진입) — 진입가 평균을 낮춘다. 단 **평균 진입가가 좋아지는 건
    가격이 반대로 갔을 때뿐**이고, 그 부분집합이 바로 지속(=페이드 실패) 사건이다. 역선택 검정.
    (⚠️순수한 **사이징**은 수익률의 선형변환이라 기대값 부호를 못 바꾼다 — 분산·낙폭만 바뀐다.
     분할 매수가 다른 이유는 **조건부 진입가**를 바꿔 비선형이기 때문이다.)
  B **비대칭 브래킷**(타이트 손절 + 짧은 익절) — 왼쪽 두꺼운 꼬리를 자른다. 기대값을 실제로 바꾼다.
    격자 TP ∈ {0.25,0.5,0.75,1.0,1.5}×ATR × SL ∈ {0.25,0.5,0.75,1.0,1.5,2.0,3.0}×ATR × 만기 {12,48}봉.
    ⭐**방향뒤집기 대조군 필수**(§5.8) + ⭐**절대 bp 하한 병기**(§5.29 — TP/SL을 중앙 atr_pct로 환산).

규격: 8종 raw 첫발동(GAP12) 합집합 · 진입 open[i+1] · 비용 10bp · 봉 내 순서는 비관(손절 먼저).
HOLDOUT 미접촉.
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


C1 = _load("c1_sc", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OUT = ROOT / "data/research/eth_fade_scalein_bracket_20260905"
COST, WINDOWS = 10.0, ("TRAIN", "VAL", "OOS")
TP_G, SL_G, T_G = (0.25, 0.5, 0.75, 1.0, 1.5), (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), (12, 48)
SCALE_D, SCALE_N, SCALE_H = (0.25, 0.5, 1.0), 6, (3, 6, 12)
rng = np.random.default_rng(20260905)


def log(m): print(f"[fade] {m}", flush=True)


def day_ci(v, t, B=600):
    d = pd.DatetimeIndex(pd.to_datetime(t)).normalize().to_numpy()
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}
    out = [v[np.concatenate([idx[x] for x in u[rng.integers(0, len(u), len(u))]])].mean() for _ in range(B)]
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def bracket(entry, atr, sign, H, L, C, tp, sl, T):
    """고정 TP/SL 브래킷. 봉 내 순서는 비관(손절 먼저 → 익절). 만기는 종가."""
    n = len(entry); tp_px = entry + sign * tp * atr; sl_px = entry - sign * sl * atr
    fav = np.where(sign[:, None] > 0, H, L); adv = np.where(sign[:, None] > 0, L, H)
    out = np.zeros(n); ex = np.full(n, T - 1); done = np.zeros(n, bool)
    for t in range(T):
        live = ~done
        hs = live & np.where(sign > 0, adv[:, t] <= sl_px, adv[:, t] >= sl_px)
        out = np.where(hs, sign * (sl_px - entry) / entry, out); ex = np.where(hs, t, ex); done |= hs
        live = ~done
        ht = live & np.where(sign > 0, fav[:, t] >= tp_px, fav[:, t] <= tp_px)
        out = np.where(ht, sign * (tp_px - entry) / entry, out); ex = np.where(ht, t, ex); done |= ht
    return np.where(done, out, sign * (C[:, T - 1] - entry) / entry), ex


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    sd, split, ts, bidx, atr = B["sd"], B["split"], B["ts"], B["bidx"], B["atr"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]; n = len(c)
    entry = o[bidx + 1]; atr_pct = atr / entry
    fs = np.where(sd == 1, 1.0, -1.0)                       # 페이드 부호
    med_atr = float(np.median(atr_pct))
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cost_bp": COST, "median_atr_pct": round(med_atr, 5),
           "holdout_touched": False, "A_scale_in": {}, "B_bracket": {}}

    # ---------------- A. 분할 매수 (불리 방향 d×ATR에서 2차 진입, N봉 유효, 절반씩)
    log("A. 분할 매수 …")
    for d in SCALE_D:
        p2 = entry - fs * d * atr
        filled = np.zeros(len(entry), bool)
        for step in range(1, SCALE_N + 1):
            b = bidx + step
            hit = np.where(fs > 0, l[b] <= p2, h[b] >= p2) & ~filled
            filled |= hit
        eff = np.where(filled, (entry + p2) / 2.0, entry)     # 체결 레그 평균가
        size = np.where(filled, 1.0, 0.5)                     # 2레그 vs 1레그
        for H_ in SCALE_H:
            ok = (bidx + H_) < n
            r_base = fs * (c[np.where(ok, bidx + H_, 0)] - entry) / entry * 1e4 - COST
            r_sc = fs * (c[np.where(ok, bidx + H_, 0)] - eff) / eff * 1e4 - COST
            rec = {"fill_rate_2nd_leg": round(float(filled[ok].mean()), 3)}
            for w in WINDOWS:
                m = ok & (split == w)
                if m.sum() < 100:
                    continue
                rec[w] = {"n": int(m.sum()),
                          "base_mean_bp": round(float(r_base[m].mean()), 2),
                          "scalein_mean_bp_per_notional": round(float(r_sc[m].mean()), 2),
                          "scalein_size_weighted_bp": round(float((r_sc[m] * size[m]).sum() / size[m].sum()), 2),
                          "p_pos_base": round(float((r_base[m] > 0).mean()), 3),
                          "p_pos_scalein": round(float((r_sc[m] > 0).mean()), 3),
                          # ⭐역선택: 2차 레그가 체결된(= 불리하게 움직인) 건들의 기본 성과
                          "base_mean_when_2nd_filled": round(float(r_base[m & filled].mean()), 2),
                          "base_mean_when_not_filled": round(float(r_base[m & ~filled].mean()), 2)}
            rep["A_scale_in"][f"d{d}_H{H_}"] = rec
            if H_ == 6:
                log(f"  d={d}×ATR 2차체결률 {rec['fill_rate_2nd_leg']:.3f} · " + " | ".join(
                    f"{w} 기본 {rec[w]['base_mean_bp']:>6} → 분할 {rec[w]['scalein_mean_bp_per_notional']:>6} "
                    f"(체결됐을때 기본 {rec[w]['base_mean_when_2nd_filled']:>7} vs 미체결 {rec[w]['base_mean_when_not_filled']:>6})" for w in WINDOWS if w in rec))

    # ---------------- B. 비대칭 브래킷 (+ 방향뒤집기 대조)
    log("B. 비대칭 TP/SL 브래킷 …")
    best = []
    for tp, sl, T in itertools.product(TP_G, SL_G, T_G):
        ok = (bidx + T) < n
        ix = (bidx + 1)[:, None] + np.arange(T); ix = np.where(ok[:, None], ix, 0)
        H_, L_, C_ = h[ix], l[ix], c[ix]
        rf, _ = bracket(entry, atr, fs, H_, L_, C_, tp, sl, T)          # 페이드
        rc, _ = bracket(entry, atr, -fs, H_, L_, C_, tp, sl, T)         # 뒤집기 대조(=지속)
        pf_, pc_ = rf * 1e4 - COST, rc * 1e4 - COST
        rec = {"tp_bp_at_median_atr": round(tp * med_atr * 1e4, 1), "sl_bp_at_median_atr": round(sl * med_atr * 1e4, 1)}
        for w in WINDOWS:
            m = ok & (split == w)
            if m.sum() < 100:
                continue
            x = pf_[m]; wn = x > 0
            rec[w] = {"n": int(m.sum()), "fade_mean_bp": round(float(x.mean()), 2), "win_rate": round(float(wn.mean()), 3),
                      "payoff": round(float(x[wn].mean() / -x[~wn].mean()), 3) if wn.any() and (~wn).any() else None,
                      "day_ci95": day_ci(x, ts[m]), "flip_cont_mean_bp": round(float(pc_[m].mean()), 2)}
        key = f"tp{tp}_sl{sl}_T{T}"
        rep["B_bracket"][key] = rec
        if all(w in rec for w in WINDOWS):
            best.append((min(rec[w]["fade_mean_bp"] for w in WINDOWS), key, rec))
    best.sort(reverse=True)
    rep["B_top10_by_worst_window"] = [{"cell": k, **{w: r[w] for w in WINDOWS}} for _, k, r in best[:10]]
    rep["B_n_cells_all_three_positive"] = sum(1 for v, _, _ in best if v > 0)
    log(f"  격자 {len(rep['B_bracket'])}셀 · 세 창 모두 평균>0인 셀 {rep['B_n_cells_all_three_positive']}개")
    for v, k, r in best[:6]:
        log(f"    {k:22s} TP {r['tp_bp_at_median_atr']:>5}bp/SL {r['sl_bp_at_median_atr']:>5}bp · " + " | ".join(
            f"{w} 페이드 {r[w]['fade_mean_bp']:>6}{str(r[w]['day_ci95']):>16} 승률 {r[w]['win_rate']:.3f} 손익비 {r[w]['payoff']} · 뒤집기 {r[w]['flip_cont_mean_bp']:>6}" for w in WINDOWS))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
