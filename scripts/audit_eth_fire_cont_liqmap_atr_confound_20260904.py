#!/usr/bin/env python3
"""청산맵 근접 효과의 교란 점검 -- ATR 교란 · 원시 % 거리 · 시간이동 플라시보 (2026-09-04).

`research_eth_fire_cont_liqmap_distance_20260904.py`가 "지속 방향 최근접 클러스터가 **가까울수록** 지속이 잘 된다"
(ρ≈−0.24, 세 구간)를 냈다. 거리를 ATR 단위로 쟀으므로 **고ATR 봉이 기계적으로 'near'가 된다** -- 경제라벨은 고정
10bp 비용 때문에 고ATR을 선호하므로(§5.22) 근접 효과가 실은 ATR 효과일 수 있다. 세 가지로 가른다:
  A  ATR 통제: 순위 부분상관(d_cont_atr ⟂ atr_pct), ATR 삼분위 안에서 near/mid/far 표
  B  원시 % 거리 삼분위(ATR 정규화 없이)
  C  시간이동 플라시보: 시간별 레벨 표를 ±7일 이동해 잘못된 시점의 맵으로 같은 거리·삼분위를 계산
     (§5.14 오프셋 플라시보 규약). 실제 맵만 단조 패턴을 보여야 한다.
컷은 전부 TRAIN에서 적합. HOLDOUT 미접촉.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp/eth_fire_cont_liqmap_20260904"
OUT = ROOT / "data/research/eth_fire_cont_liqmap_distance_20260904/atr_confound_audit.json"
WINDOWS = ("TRAIN", "VAL", "OOS")


def tert(x, tr):
    _, e = pd.qcut(x[tr], 3, retbins=True, duplicates="drop"); e = np.r_[-np.inf, e[1:-1], np.inf]
    return pd.cut(x, bins=e, labels=["near", "mid", "far"][: len(e) - 1]).astype(str)


def partial_spearman(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    def resid(a, b):
        A = np.c_[np.ones(len(b)), b]; beta = np.linalg.lstsq(A, a, rcond=None)[0]; return a - A @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def main():
    M = pd.read_parquet(TMP / "fire_cont_rows_with_levels.parquet"); H = pd.read_parquet(TMP / "hourly_levels.parquet").sort_values("timestamp")
    tr = (M["split"] == "TRAIN").to_numpy(); rep = {"windows": {}}
    M["atr_t"] = tert(M["atr_pct"], tr); M["dpct_t"] = tert(M["d_cont_pct"], tr)
    for w in WINDOWS:
        S = M.loc[M["split"] == w]; R = {}
        R["rho_dcont_atr_units"] = round(float(spearmanr(S["d_cont_atr"], S["net_bp"]).correlation), 4)
        R["rho_dcont_pct"] = round(float(spearmanr(S["d_cont_pct"], S["net_bp"]).correlation), 4)
        R["rho_atr_pct"] = round(float(spearmanr(S["atr_pct"], S["net_bp"]).correlation), 4)
        R["partial_rho_dcont_atr_units_given_atr"] = round(partial_spearman(S["d_cont_atr"].to_numpy(), S["net_bp"].to_numpy(), S["atr_pct"].to_numpy()), 4)
        R["partial_rho_dcont_pct_given_atr"] = round(partial_spearman(S["d_cont_pct"].to_numpy(), S["net_bp"].to_numpy(), S["atr_pct"].to_numpy()), 4)
        R["within_atr_tercile"] = {a: {t_: [round(float(S.loc[(S["atr_t"] == a) & (S["d_cont_atr_t"].astype(str) == t_), "net_bp"].mean()), 2),
                                            int(((S["atr_t"] == a) & (S["d_cont_atr_t"].astype(str) == t_)).sum())] for t_ in ("near", "mid", "far")}
                                   for a in ("near", "mid", "far")}
        R["raw_pct_tertile"] = {t_: [round(float(S.loc[S["dpct_t"] == t_, "net_bp"].mean()), 2), int((S["dpct_t"] == t_).sum())] for t_ in ("near", "mid", "far")}
        rep["windows"][w] = R
    # C 시간이동 플라시보
    base_cols = ["pos", "is_downside", "timestamp", "split", "net_bp", "fire_side", "atr_pct"]
    plc = {}
    for shift_d in (-7, 7, -30, 30):
        Hs = H.copy(); Hs["timestamp"] = Hs["timestamp"] + pd.Timedelta(days=shift_d)
        X = pd.merge_asof(M[base_cols].sort_values("timestamp"), Hs.sort_values("timestamp"), on="timestamp", direction="backward")
        sup = X["sup_dist_pct"].abs().fillna(5.0); res = X["res_dist_pct"].abs().fillna(5.0)
        X["d"] = np.where(X["fire_side"] == "bottom", sup, res) / 100.0 / X["atr_pct"]
        trx = (X["split"] == "TRAIN").to_numpy(); X["t"] = tert(X["d"], trx)
        plc[f"shift_{shift_d}d"] = {w: {"rho": round(float(spearmanr(X.loc[X["split"] == w, "d"], X.loc[X["split"] == w, "net_bp"]).correlation), 4),
                                        **{t_: round(float(X.loc[(X["split"] == w) & (X["t"] == t_), "net_bp"].mean()), 2) for t_ in ("near", "mid", "far")}} for w in WINDOWS}
    rep["placebo_time_shift"] = plc
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    for w in WINDOWS:
        R = rep["windows"][w]
        print(f"{w:>5s} ρ(d_atr) {R['rho_dcont_atr_units']:+.3f}  ρ(d_pct) {R['rho_dcont_pct']:+.3f}  ρ(atr) {R['rho_atr_pct']:+.3f}  "
              f"partial ρ(d_atr|atr) {R['partial_rho_dcont_atr_units_given_atr']:+.3f}  partial ρ(d_pct|atr) {R['partial_rho_dcont_pct_given_atr']:+.3f}")
        print(f"      raw % tertile near/mid/far: {R['raw_pct_tertile']}")
        for a in ("near", "mid", "far"):
            print(f"      ATR {a:>4s}: " + "  ".join(f"{t_} {R['within_atr_tercile'][a][t_][0]:+6.2f} (n{R['within_atr_tercile'][a][t_][1]})" for t_ in ("near", "mid", "far")))
    print("\n[placebo time-shifted level tables: rho / near / mid / far]")
    for k, v in plc.items():
        print(f"  {k:>10s}: " + " | ".join(f"{w} ρ{v[w]['rho']:+.3f} {v[w]['near']:+5.1f}/{v[w]['mid']:+5.1f}/{v[w]['far']:+5.1f}" for w in WINDOWS))


if __name__ == "__main__":
    main()
