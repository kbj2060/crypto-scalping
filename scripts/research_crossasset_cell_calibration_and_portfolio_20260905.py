#!/usr/bin/env python3
"""교차자산 지속 규칙 — **자산별 청산 셀 보정 + 다자산 포트폴리오** (2026-09-05).

교차자산 재현에서 드러난 것: BTC는 지속이 −1.7~−2.2bp인데 **같은 창의 무작위 진입도 −2.2~−4.2bp**다.
즉 BTC에서 실패한 건 신호가 아니라 **ETH에서 고른 청산 셀(5.0 SL/1.5 ARM/0.1 trail ×ATR, 200봉) + 10bp**가
그 자산의 움직임 분포에 안 맞는 것이다(§5.19 포팅 프로토콜: 셀은 자산별로 재선정해야 한다).

이 스크립트가 하는 것
  1) 자산별 셀 격자 sweep — **TRAIN만 보고** 셀 선택 (일평균 자본 bp 최대, 단 TRAIN 일CI 하한 > 0 인 셀 중에서)
  2) 선택 셀로 VAL/OOS **1회** 확인 + **같은 셀의 같은 측면 무작위 진입 귀무** 대비 초과분(셀 효과 제거)
  3) 자산별 일손익 **상관행렬** → 다자산 합산 포트폴리오(동일 위험 배분)의 샤프가 단일 자산보다 나은가
판정: 자산이 "쓸 만하다" = VAL·OOS 둘 다 (지속 exp>0) ∧ (같은측면 귀무 대비 초과 > 0). 합산은 보고 전용.
⚠️셀을 자산별로 고르는 순간 다중성이 생긴다 — 그래서 초과분(귀무 대비)을 같이 본다. 승격은 전진 섀도우.
HOLDOUT은 로드 단계에서 차단.
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


XA = _load("xa_cell", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
V2 = _load("hev2_cell", "scripts/research_homer_entry_v2_20260904.py")
C1M = _load("comp1_cell", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, portfolio, day_boot = V2.sim_exit, V2.portfolio, V2.day_boot
pf, cand_of = C1M.pf, C1M.cand_of
OUT = ROOT / "data/research/crossasset_cell_calibration_20260905"
FWD, COST, CAP, SPLITS = XA.FWD, XA.COST, XA.CAP, XA.SPLITS
SL_G, ARM_G, TRAIL_G = (3.0, 4.0, 5.0, 6.0, 8.0), (1.0, 1.5, 2.0, 3.0), (0.05, 0.10, 0.20)
B_NULL, NULL_POOL = 200, 9000
rng = np.random.default_rng(20260905)


def log(m): print(f"[cell] {m}", flush=True)


def prep(sym, ref):
    kl = XA.load_kl(sym); r = XA.load_kl(ref)
    sig = XA.DASH.compute_signals(kl.copy(), btc_df=r, funding_df=None)
    n = len(kl); c = kl["close"].to_numpy(float); h = kl["high"].to_numpy(float); l = kl["low"].to_numpy(float); o = kl["open"].to_numpy(float)
    prev = np.r_[np.nan, c[:-1]]; trr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(trr).rolling(14, min_periods=14).mean().to_numpy(); ts_all = kl["timestamp"].to_numpy()
    rows = []
    for s in XA.SIGNALS:
        for side, sd in (("bottom", 1), ("top", 0)):
            col = f"{side}_{s}"
            if col in sig.columns:
                rows.append(pd.DataFrame({"i": np.flatnonzero(XA.first_fire_mask(sig[col].fillna(False).to_numpy(bool))), "is_downside": sd}))
    F = pd.concat(rows, ignore_index=True).sort_values("i").drop_duplicates(["i", "is_downside"]).reset_index(drop=True)
    F = F.loc[(F["i"].to_numpy() + 1 + FWD < n) & np.isfinite(atr[F["i"].to_numpy()])].reset_index(drop=True)
    i = F["i"].to_numpy(); sd = F["is_downside"].to_numpy().astype(int)
    return dict(n=n, o=o, h=h, l=l, c=c, atr=atr, ts_all=ts_all, i=i, sd=sd,
                sign=np.where(sd == 1, -1.0, 1.0), ts=ts_all[i], ix=(i + 1)[:, None] + np.arange(FWD))


def eval_cell(D, cell, mask):
    i, ix, sign, atr, o = D["i"], D["ix"], D["sign"], D["atr"], D["o"]
    ret, ex = sim_exit(o[i + 1], atr[i], sign, D["h"][ix], D["l"][ix], D["c"][ix], *cell)
    p = ret * 1e4 - COST
    return pf(cand_of(D["ts"][mask], i[mask] + 1, i[mask] + 1 + ex[mask], p[mask])), p, ex


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "grid": {"sl": SL_G, "arm": ARM_G, "trail": TRAIL_G},
           "cost_bp": COST, "max_concurrent": CAP, "holdout_excluded": True, "assets": {}}
    daily = {}
    for sym, ref in XA.ASSETS.items():
        D = prep(sym, ref); tsi = pd.DatetimeIndex(D["ts"])
        M = {w: np.asarray((tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))) for w, (a, b) in SPLITS.items()}
        grid = {}
        for cell in itertools.product(SL_G, ARM_G, TRAIL_G):
            r, _, _ = eval_cell(D, cell, M["TRAIN"])
            if r:
                grid["_".join(map(str, cell))] = {"exp_bp": r["stats"]["exp_bp"], "daily_mean_bp": r["stats"]["daily_mean_bp"],
                                                  "daily_sharpe_ann": r["stats"]["daily_sharpe_ann"], "day_ci95": r["stats"]["day_ci95"], "n": r["stats"]["n"]}
        elig = {k: v for k, v in grid.items() if v["day_ci95"][0] > 0}
        pick = max(elig, key=lambda k: elig[k]["daily_mean_bp"]) if elig else max(grid, key=lambda k: grid[k]["daily_mean_bp"])
        cell = tuple(float(x) for x in pick.split("_"))
        A = {"symbol": sym, "n_fires": int(len(D["i"])), "cells_eligible_TRAIN": len(elig), "n_cells": len(grid),
             "picked_cell_TRAIN": cell, "eth_default_cell_TRAIN": grid.get("5.0_1.5_0.1"), "windows": {}}
        for w in SPLITS:
            m = M[w]
            if m.sum() < 100:
                continue
            r, p, ex = eval_cell(D, cell, m)
            # 같은 셀·같은 측면 무작위 진입 귀무
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
                    parts.append(cand_of(D["ts_all"][pool[k]], pool[k] + 1, pool[k] + 1 + pe, pr * 1e4 - COST))
                q = portfolio(pd.concat(parts, ignore_index=True), CAP); nv.append(q["exp_bp"] if q else np.nan)
            nv = np.asarray(nv, float); obs = r["stats"]["exp_bp"]
            A["windows"][w] = {**{x: r["stats"][x] for x in ("n", "exp_bp", "win_rate", "day_ci95", "per_day", "daily_mean_bp", "daily_sharpe_ann", "max_dd_bp")},
                               "null_same_cell_mean_bp": round(float(np.nanmean(nv)), 2), "excess_over_null_bp": round(obs - float(np.nanmean(nv)), 2),
                               "percentile_of_obs": round(float((nv < obs).mean() * 100), 1)}
            s = pd.Series(r["pnl"] / CAP, index=pd.DatetimeIndex(pd.to_datetime(r["ts"])).normalize()).groupby(level=0).sum()
            daily.setdefault(w, {})[sym] = s
        A["usable"] = all(A["windows"].get(w, {}).get("exp_bp", -9) > 0 and A["windows"].get(w, {}).get("excess_over_null_bp", -9) > 0 for w in ("VAL", "OOS"))
        out["assets"][sym] = A
        log(f"{sym}: 셀 {cell} (TRAIN 적격 {len(elig)}/{len(grid)}) · " +
            " · ".join(f"{w} exp={A['windows'][w]['exp_bp']:.2f} 초과={A['windows'][w]['excess_over_null_bp']:.2f} 샤프={A['windows'][w]['daily_sharpe_ann']}" for w in A["windows"]) +
            f" · 쓸만={A['usable']}")
    # 다자산 합산 (동일 위험 배분: 자산별 일손익 평균)
    port = {}
    for w, dd in daily.items():
        df = pd.DataFrame(dd)
        idx = pd.date_range(min(s.index.min() for s in dd.values()), max(s.index.max() for s in dd.values()), freq="D")
        df = df.reindex(idx).fillna(0.0)
        corr = df.corr().round(3).to_dict()
        eq = df.mean(axis=1)
        def sh(s):
            return round(float(s.mean() / s.std(ddof=1) * np.sqrt(365)), 2) if s.std(ddof=1) > 0 else None
        port[w] = {"corr": corr, "per_asset_sharpe": {c: sh(df[c]) for c in df}, "per_asset_daily_mean": {c: round(float(df[c].mean()), 2) for c in df},
                   "equal_weight_sharpe": sh(eq), "equal_weight_daily_mean": round(float(eq.mean()), 2),
                   "usable_only_sharpe": None}
        us = [s for s in df.columns if out["assets"][s].get("usable")]
        if us:
            e2 = df[us].mean(axis=1); port[w]["usable_only_sharpe"] = sh(e2)
            port[w]["usable_only_daily_mean"] = round(float(e2.mean()), 2); port[w]["usable_assets"] = us
    out["portfolio"] = port
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for w, p in port.items():
        log(f"  [{w}] 자산 샤프 {p['per_asset_sharpe']} · 동일가중 샤프 {p['equal_weight_sharpe']} (일평균 {p['equal_weight_daily_mean']}) · 쓸만한 것만 {p.get('usable_assets')} 샤프 {p['usable_only_sharpe']}")


if __name__ == "__main__":
    main()
