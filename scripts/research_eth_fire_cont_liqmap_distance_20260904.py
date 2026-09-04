#!/usr/bin/env python3
"""발동 봉 지속(continuation) 규칙 × **청산맵 지지/저항 거리** (2026-09-04).

사용자 지시: "청산맵의 지지/저항과 레짐 분류로 업그레이드". 이 스크립트는 청산맵 축.

사전 가설(결과 보기 전): H_liq1 -- 지속 방향에 있는 최근접 청산 클러스터가 **멀수록** 지속이 잘 된다
(가까우면 자석/벽에 막혀 1.5 ATR 무장 전에 멈춤). 바닥 발동→지속=숏→지지선(아래) 거리, 천장 발동→지속=롱→
저항선(위) 거리. 반대 방향(페이드 쪽) 레벨 거리도 같이 보고한다(2026-08-27 confluence 발견: 지지선 근접
바닥 발동은 반등 lift가 높았다 → 지속은 약해야 한다는 같은 부호의 예측).

데이터/인과성: 라이브 `compute_spliced_levels()`를 1시간봉(24봉 tail)으로 매 시간 재계산하고 h+1h로 스탬프해
5분봉 발동에 merge_asof(backward) -- `research_eth_evidence_signal_liquidation_confluence_20260827.py`의 규약
그대로(룩어헤드 없음). 거리는 ATR 단위(거리%/atr_pct). 삼분위 컷은 **TRAIN에서만** 적합해 VAL/OOS에 적용.

⚠️HOLDOUT 미접촉. 셀·GAP·비용·한도는 상속(선택 없음). 이 결과는 진단이며 승격 주장이 아니다.
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


CM = _load("cont_mod", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")
from live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402

portfolio, day_boot, stats_of, load_fires = CM.portfolio, CM.day_boot, CM.stats_of, CM.load_fires
FRAME, KL, MAX_CONC, B_BOOT = CM.FRAME, CM.KL, CM.MAX_CONC, CM.B_BOOT
OUT = ROOT / "data/research/eth_fire_cont_liqmap_distance_20260904"
TMP = ROOT / "tmp/eth_fire_cont_liqmap_20260904"
LOOKBACK_HOURLY_BARS = 24
LV_START, LV_END = pd.Timestamp("2024-03-20"), pd.Timestamp("2026-04-01")
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[liq] {m}", flush=True)


def hourly_levels():
    cache = TMP / "hourly_levels.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    kl = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close", "volume"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    kl = kl.loc[(kl["timestamp"] >= LV_START) & (kl["timestamp"] < LV_END)]
    hourly = kl.set_index("timestamp").resample("1h").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    n = len(hourly); close = hourly["close"].to_numpy(); ts = hourly["timestamp"]; rows = []; t0 = time.time()
    for h in range(LOOKBACK_HOURLY_BARS - 1, n):
        window = hourly.iloc[h - LOOKBACK_HOURLY_BARS + 1: h + 1]
        lv = compute_spliced_levels(window, float(close[h]))
        sup = lv.get("support_levels") or []; res = lv.get("resistance_levels") or []
        rows.append({"timestamp": ts.iloc[h] + pd.Timedelta(hours=1), "hour_close": float(close[h]),
                     "sup_dist_pct": sup[0]["distance_pct"] if sup else np.nan, "sup_w": sup[0]["weight_pct"] if sup else np.nan, "n_sup": len(sup),
                     "res_dist_pct": res[0]["distance_pct"] if res else np.nan, "res_w": res[0]["weight_pct"] if res else np.nan, "n_res": len(res)})
        if (h + 1) % 3000 == 0:
            log(f"  hourly {h+1}/{n} ({time.time()-t0:.0f}s)")
    H = pd.DataFrame(rows); TMP.mkdir(parents=True, exist_ok=True); H.to_parquet(cache, index=False)
    log(f"hourly levels {len(H):,} ({time.time()-t0:.0f}s) · has_sup {H['sup_dist_pct'].notna().mean():.3f} has_res {H['res_dist_pct'].notna().mean():.3f}")
    return H


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260904)
    H = hourly_levels()
    D = pd.read_parquet(FRAME, columns=["pos", "is_downside", "side", "timestamp", "split", "entry", "atr", "net_bp", "net_bp_flip", "exit_off"])
    F = load_fires(); Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"])
    key = D.set_index(["pos", "is_downside"])
    # 지속 트레이드 = 발동 봉의 반대 측면 행 (net_bp=지속, net_bp_flip=페이드, exit_off=지속 청산)
    R = key.reindex(pd.MultiIndex.from_arrays([Fp["pos"].to_numpy(), 1 - Fp["is_downside"].to_numpy().astype(int)], names=["pos", "is_downside"]))
    R = R.dropna(subset=["net_bp"]).reset_index()
    R["fire_side"] = np.where(R["is_downside"] == 0, "bottom", "top")      # 지속이 숏(is_downside 0)이면 바닥 발동
    R = R.sort_values("timestamp").reset_index(drop=True)
    M = pd.merge_asof(R, H.sort_values("timestamp"), on="timestamp", direction="backward")
    M["atr_pct"] = M["atr"] / M["entry"]
    sup = M["sup_dist_pct"].abs().fillna(5.0); res = M["res_dist_pct"].abs().fillna(5.0)
    M["d_cont_pct"] = np.where(M["fire_side"] == "bottom", sup, res); M["d_fade_pct"] = np.where(M["fire_side"] == "bottom", res, sup)
    M["d_cont_atr"] = M["d_cont_pct"] / 100.0 / M["atr_pct"]; M["d_fade_atr"] = M["d_fade_pct"] / 100.0 / M["atr_pct"]
    M["cont_none"] = np.where(M["fire_side"] == "bottom", M["sup_dist_pct"].isna(), M["res_dist_pct"].isna())
    M["lvl_age_min"] = (M["timestamp"] - M["timestamp"]).dt.total_seconds()  # placeholder (스탬프는 h+1h)
    tr = M["split"] == "TRAIN"
    rep = {"holdout_touched": False, "n_rows": int(len(M)), "cont_level_missing_share": round(float(M["cont_none"].mean()), 4),
           "windows": {}, "tertile_edges_train_atr": {}}
    log(f"지속 행 {len(M):,} · 레벨 결측(지속방향) {M['cont_none'].mean():.3f} · d_cont_atr 중앙 {M['d_cont_atr'].median():.2f} ({time.time()-t0:.0f}s)")
    for dcol in ("d_cont_atr", "d_fade_atr"):
        _, edges = pd.qcut(M.loc[tr, dcol], 3, retbins=True, duplicates="drop"); edges = np.r_[-np.inf, edges[1:-1], np.inf]
        M[f"{dcol}_t"] = pd.cut(M[dcol], bins=edges, labels=["near", "mid", "far"][: len(edges) - 1])
        rep["tertile_edges_train_atr"][dcol] = [round(float(x), 3) for x in edges[1:-1]]
    for w in WINDOWS:
        S = M.loc[M["split"] == w]; Rw = {"n": int(len(S))}
        for dcol in ("d_cont_atr", "d_fade_atr"):
            T = {}
            for t_ in ("near", "mid", "far"):
                s = S.loc[S[f"{dcol}_t"] == t_]
                if len(s) < 30:
                    continue
                cand = pd.DataFrame({"timestamp": s["timestamp"].to_numpy(), "pos": s["pos"].to_numpy(), "p": 1.0, "entry_bar": s["pos"].to_numpy() + 1,
                                     "exit_bar": s["pos"].to_numpy() + 1 + s["exit_off"].to_numpy(), "pnl_bp": s["net_bp"].to_numpy()})
                r = portfolio(cand, MAX_CONC); lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
                T[t_] = {"n": int(len(s)), "cont_row_bp": round(float(s["net_bp"].mean()), 2), "fade_row_bp": round(float(s["net_bp_flip"].mean()), 2),
                         "p_fade_gt_cont": round(float((s["net_bp_flip"] > s["net_bp"]).mean()), 3), "pf_exp_bp": round(r["exp_bp"], 2),
                         "pf_n": r["n"], "day_ci95": [round(lo, 2), round(hi, 2)],
                         "by_fire_side": {fs: round(float(s.loc[s["fire_side"] == fs, "net_bp"].mean()), 2) for fs in ("bottom", "top")}}
            Rw[dcol] = T
        # 연속형: 스피어만(d_cont_atr, cont net) 참고
        from scipy.stats import spearmanr
        Rw["spearman_dcont_vs_cont"] = round(float(spearmanr(S["d_cont_atr"], S["net_bp"]).correlation), 4)
        Rw["spearman_dfade_vs_cont"] = round(float(spearmanr(S["d_fade_atr"], S["net_bp"]).correlation), 4)
        rep["windows"][w] = Rw
        log(f"{w}: " + " | ".join(f"{t_} {Rw['d_cont_atr'].get(t_, {}).get('pf_exp_bp')}" for t_ in ("near", "mid", "far")) + f" · ρ(d_cont) {Rw['spearman_dcont_vs_cont']}")
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    M.to_parquet(TMP / "fire_cont_rows_with_levels.parquet", index=False)
    print(f"\ntertile edges (TRAIN, ATR units): {rep['tertile_edges_train_atr']}")
    for dcol in ("d_cont_atr", "d_fade_atr"):
        print(f"\n[{dcol}]  pf_exp_bp (row cont / fade, P(f>c)) [dayCI]  n")
        for w in WINDOWS:
            line = f"{w:>5s}"
            for t_ in ("near", "mid", "far"):
                x = rep["windows"][w][dcol].get(t_)
                if x:
                    line += f" | {t_:>4s} {x['pf_exp_bp']:+6.2f} ({x['cont_row_bp']:+5.1f}/{x['fade_row_bp']:+5.1f}, {x['p_fade_gt_cont']:.3f}) {x['day_ci95']} n{x['n']}"
            print(line)
    print("\n[by fire side, d_cont tertile, cont row bp]")
    for w in WINDOWS:
        print(w, {t_: rep["windows"][w]["d_cont_atr"][t_]["by_fire_side"] for t_ in rep["windows"][w]["d_cont_atr"]})
    log(f"완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
