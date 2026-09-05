#!/usr/bin/env python3
"""ETH 복합 알고리즘 3라운드 — **복합 확신도 사이징(C7)** (2026-09-05).

1~2라운드 판독:
  · 필터는 전부 R에 진다(거래 수가 줄어 자본 대비 일손익이 깎인다). 살아남은 형태는 **사이징**뿐이다
    (C1 size_prop_m_same: VAL +1.8 / OOS +2.3 bp/일, 둘 다 CI 0 포함).
  · C2 상태축 중 **TRAIN 갭 일CI 하한 > 0**인 13축이 있다. 축 선택은 **TRAIN만** 쓴다(VAL/OOS 갭은 보지 않은 것으로 취급).
  · C5/C6: ATR 계열 우위는 대부분(≈85%) 배리어가 ATR 배수라서 생기는 기계적 스케일이고, 위험기반 사이징이 그걸 상쇄한다.
    → 확신도 사이징은 **위험 사이징 위에** 얹어야 한다(둘 다 보고).

C7 설계 (전부 TRAIN에서 고정 → VAL/OOS 1회 확인)
  확신도 c = TRAIN ECDF 기준 백분위 순위의 **평균**(13축, 결측 축은 제외, 유효 8축 미만이면 c=0.5)
  사이징   w = clip(1 + β(2c − 1), 0.2, 1.8), β = 0.5 (주), 0.3/0.8 (민감도, 선택 아님)
  대조군   ① 확신도 무작위 순열 사이징 B=200 → 관측 Δ의 백분위 ② 필터형(c≥0.5, c≥0.7) ③ 축 1개씩 단독 사이징
  기준     R(동일 가중) 및 위험사이징 R. 판정: VAL·OOS 두 창 모두 일별 짝비교 CI 하한 > 0.

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


C1M = _load("comp1c", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
build, pf, cand_of, day_paired = C1M.build, C1M.pf, C1M.cand_of, C1M.day_paired
CELL, CAP, WINDOWS = C1M.CELL, C1M.CAP, C1M.WINDOWS
OUT = ROOT / "data/research/eth_composite_conviction_sizing_20260905"
# ── TRAIN 갭 일CI 하한 > 0 인 축 (1라운드 C2의 **TRAIN 열만** 보고 확정, VAL/OOS 미사용)
AXES_TRAIN = ["vwap_dev_z", "rsi_c", "di_spread", "bb_pctb_c", "atr_pct", "ax_market_beta_move", "vol_z",
              "ax_activity", "delta_z", "ax_btc_shock", "atr_percentile_864", "ret3_z", "er24"]
BETA_MAIN, BETAS = 0.5, (0.3, 0.5, 0.8)
W_LO, W_HI, MIN_AXES = 0.2, 1.8, 8
RISK_FRAC, NOTIONAL_CAP = 0.004, 0.5
B_NULL = 200
rng = np.random.default_rng(20260905)


def log(m): print(f"[c7] {m}", flush=True)


def conviction(B):
    """TRAIN ECDF 백분위 순위의 평균. 결측 축 제외, 유효 축 < MIN_AXES면 0.5(중립)."""
    cs, split = B["cont_sign"], B["split"]; tr = split == "TRAIN"
    n = len(cs); acc = np.zeros(n); cnt = np.zeros(n); per_axis = {}
    for name in AXES_TRAIN:
        if name not in B["S"]:
            log(f"  ⚠️축 없음: {name}"); continue
        kind, raw = B["S"][name]
        x = raw * cs if kind == "aligned" else raw.astype(float)
        fin = np.isfinite(x); ref = np.sort(x[fin & tr])
        if len(ref) < 500:
            log(f"  ⚠️TRAIN 표본 부족: {name}"); continue
        r = np.full(n, np.nan)
        r[fin] = np.searchsorted(ref, x[fin], side="right") / len(ref)      # TRAIN ECDF 백분위 (0~1)
        per_axis[name] = r
        acc = np.where(fin, acc + r, acc); cnt = np.where(fin, cnt + 1, cnt)
    c = np.where(cnt >= MIN_AXES, acc / np.maximum(cnt, 1), 0.5)
    return c, cnt, per_axis


def diff_point(pnl_a, ts_a, pnl_b, ts_b, cap=CAP):
    """짝비교 점추정만 (귀무 순열용 — 부트스트랩 생략해 200배 빠르다)."""
    def g(p, t):
        return pd.Series(np.asarray(p, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(t))).normalize()).groupby(level=0).sum() / cap
    A, Bs = g(pnl_a, ts_a), g(pnl_b, ts_b); d = A.reindex(A.index.union(Bs.index)).fillna(0.0) - Bs.reindex(A.index.union(Bs.index)).fillna(0.0)
    return float(d.mean())


def weights(c, beta):
    return np.clip(1.0 + beta * (2.0 * c - 1.0), W_LO, W_HI)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = build()
    pos, split, ts, cont_bp, cont_ex, atr = B["pos"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"], B["atr"]
    atr_pct = atr / B["entry"]
    notional = np.minimum(RISK_FRAC / (CELL[0] * atr_pct), NOTIONAL_CAP)
    c, cnt, per_axis = conviction(B)
    log(f"확신도: 유효축 중앙 {np.median(cnt):.0f} · c 평균 {c.mean():.3f} · TRAIN 분포 {np.percentile(c[split=='TRAIN'],[5,50,95]).round(3)}")
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "axes_selected_on_TRAIN": AXES_TRAIN, "beta_main": BETA_MAIN,
           "weight_clip": [W_LO, W_HI], "min_axes": MIN_AXES, "holdout_touched": False,
           "conviction_axis_count_median": float(np.median(cnt)), "arms": {}, "buckets": {}}
    for scheme in ("equal", "risk"):
        sizer = np.ones(len(pos)) if scheme == "equal" else notional
        base = {}
        for w in WINDOWS:
            m = split == w
            base[w] = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * sizer[m]))
        rep["arms"][scheme] = {"R": {w: base[w]["stats"] for w in WINDOWS}}
        # 확신도 5분위 버킷 (TRAIN 경계)
        qs = np.quantile(c[split == "TRAIN"], [0.2, 0.4, 0.6, 0.8]); qi = np.digitize(c, qs)
        rep["buckets"][scheme] = {}
        for w in WINDOWS:
            m = split == w; d = {}
            for q in range(5):
                sel = m & (qi == q)
                if sel.sum() < 30:
                    continue
                d[f"Q{q+1}"] = {"n": int(sel.sum()), "row_bp": round(float((cont_bp[sel] * sizer[sel]).mean()), 3)}
            rep["buckets"][scheme][w] = d
        arms = {}
        for beta in BETAS:
            wt = weights(c, beta)
            arms[f"size_conviction_b{beta}"] = wt
        arms["filter_c>=0.5"] = np.where(c >= 0.5, 1.0, 0.0)
        arms["filter_c>=0.7"] = np.where(c >= 0.7, 1.0, 0.0)
        for name, r_ax in per_axis.items():
            arms[f"solo_{name}_b0.5"] = weights(np.where(np.isfinite(r_ax), r_ax, 0.5), 0.5)
        res = {}
        for nm, wt in arms.items():
            rec = {}
            for w in WINDOWS:
                m = split == w; keep = m & (wt > 0)
                if keep.sum() < 100:
                    continue
                r = pf(cand_of(ts[keep], pos[keep] + 1, pos[keep] + 1 + cont_ex[keep], cont_bp[keep] * sizer[keep] * wt[keep]))
                if r is None:
                    continue
                rec[w] = {**{x: r["stats"][x] for x in ("n", "exp_bp", "day_ci95", "daily_mean_bp", "daily_sharpe_ann", "max_dd_bp")},
                          "mean_weight": round(float(wt[keep].mean()), 3),
                          "vs_R": day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
            res[nm] = rec
        # 무작위 순열 사이징 귀무 (주 팔 β=0.5)
        null = {}
        for w in WINDOWS:
            m = split == w; obs = res[f"size_conviction_b{BETA_MAIN}"].get(w, {}).get("vs_R", {}).get("diff_bp_day")
            if obs is None:
                continue
            vals = []
            for _ in range(B_NULL):
                perm = rng.permutation(c[m])
                wt = weights(perm, BETA_MAIN)
                r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * sizer[m] * wt))
                vals.append(diff_point(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"]))
            v = np.asarray(vals, float)
            null[w] = {"obs_diff": obs, "null_mean": round(float(np.nanmean(v)), 2), "null_p95": round(float(np.nanpercentile(v, 95)), 2),
                       "percentile_of_obs": round(float((v < obs).mean() * 100), 1)}
        res["_permutation_null_b0.5"] = null
        rep["arms"][scheme].update(res)

    # 판정
    P = []
    for scheme in ("equal", "risk"):
        for nm, rec in rep["arms"][scheme].items():
            if not isinstance(rec, dict) or "VAL" not in rec or not isinstance(rec.get("VAL"), dict) or "vs_R" not in rec["VAL"]:
                continue
            v, o = rec["VAL"]["vs_R"], rec["OOS"]["vs_R"]
            if v["ci95"][0] > 0 and o["ci95"][0] > 0:
                P.append({"scheme": scheme, "arm": nm, "VAL": v, "OOS": o})
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 vs_R 짝비교 CI 하한 > 0", "n_pass": len(P), "passes": P,
                      "verdict": "R 단독 유지" if not P else "후보 존재"}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for scheme in ("equal", "risk"):
        a = rep["arms"][scheme]
        for nm in [f"size_conviction_b{BETA_MAIN}", "filter_c>=0.5", "filter_c>=0.7"]:
            r = a.get(nm, {})
            if "VAL" in r:
                log(f"  [{scheme}] {nm}: VAL Δ={r['VAL']['vs_R']['diff_bp_day']}{r['VAL']['vs_R']['ci95']} 이긴날={r['VAL']['vs_R']['win_day_frac']} | "
                    f"OOS Δ={r['OOS']['vs_R']['diff_bp_day']}{r['OOS']['vs_R']['ci95']} 이긴날={r['OOS']['vs_R']['win_day_frac']}")
        log(f"  [{scheme}] 순열귀무: {a.get('_permutation_null_b0.5')}")
    log(f"  판정: {rep['verdict']['verdict']} ({rep['verdict']['n_pass']}개)")


if __name__ == "__main__":
    main()
