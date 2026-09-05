#!/usr/bin/env python3
"""교차자산 지속 규칙 — **측면 비대칭이 신호인가 시장 드리프트인가** (2026-09-05).

교차자산 재현(`research_crossasset_fire_continuation_replication_20260905.py`)에서 VAL/OOS의 엣지가
**바닥 발동(=지속 숏) 다리에 몰려** 있었다(ETH OOS 숏 +13.8 vs 롱 +0.2, SOL VAL +13.6 vs −1.0,
XRP OOS +4.2 vs −4.6, BTC OOS +1.8 vs −5.2). 같은 기간 시장이 내렸다면 이건 신호가 아니라 방향 베타다.

측정: 창별로 **같은 측면의 무작위 진입 귀무**를 따로 만들어(롱 전용 / 숏 전용) R의 각 다리와 비교한다.
  초과 = R(측면) − 무작위(같은 측면).  이게 0 근처면 그 다리는 **드리프트지 신호가 아니다.**
같이 보고: 창 시장 드리프트(bp), 측면별 승률.
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


XA = _load("xa_side", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
V2 = _load("hev2_side", "scripts/research_homer_entry_v2_20260904.py")
C1M = _load("comp1_side", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, portfolio, day_boot = V2.sim_exit, V2.portfolio, V2.day_boot
pf, cand_of = C1M.pf, C1M.cand_of
DASH = XA.DASH
OUT = ROOT / "data/research/crossasset_side_drift_audit_20260905"
CELL, FWD, COST, CAP, SPLITS, B_NULL = XA.CELL, XA.FWD, XA.COST, XA.CAP, XA.SPLITS, 300
NULL_POOL = 12000
rng = np.random.default_rng(20260905)


def log(m): print(f"[side] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "assets": {}}
    for sym, ref in XA.ASSETS.items():
        kl = XA.load_kl(sym); r = XA.load_kl(ref)
        sig = DASH.compute_signals(kl.copy(), btc_df=r, funding_df=None)
        n = len(kl); c = kl["close"].to_numpy(float); h = kl["high"].to_numpy(float); l = kl["low"].to_numpy(float); o = kl["open"].to_numpy(float)
        prev = np.r_[np.nan, c[:-1]]; trr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
        atr_all = pd.Series(trr).rolling(14, min_periods=14).mean().to_numpy(); ts_all = kl["timestamp"].to_numpy()
        rows = []
        for s in XA.SIGNALS:
            for side, sd in (("bottom", 1), ("top", 0)):
                col = f"{side}_{s}"
                if col not in sig.columns:
                    continue
                ff = XA.first_fire_mask(sig[col].fillna(False).to_numpy(bool))
                rows.append(pd.DataFrame({"i": np.flatnonzero(ff), "is_downside": sd}))
        F = pd.concat(rows, ignore_index=True).sort_values("i").drop_duplicates(["i", "is_downside"]).reset_index(drop=True)
        ok = (F["i"].to_numpy() + 1 + FWD < n) & np.isfinite(atr_all[F["i"].to_numpy()])
        F = F.loc[ok].reset_index(drop=True)
        i = F["i"].to_numpy(); sd = F["is_downside"].to_numpy().astype(int)
        cont_sign = np.where(sd == 1, -1.0, 1.0)                       # 바닥 발동 -> 숏
        st = i + 1; ix = st[:, None] + np.arange(FWD)
        ret, ex = sim_exit(o[st], atr_all[i], cont_sign, h[ix], l[ix], c[ix], *CELL)
        pnl = ret * 1e4 - COST; ts = ts_all[i]; tsi = pd.DatetimeIndex(ts)
        A = {"symbol": sym, "windows": {}}
        for w, (a, b) in SPLITS.items():
            m = (tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))
            wm = (pd.DatetimeIndex(ts_all) >= pd.Timestamp(a)) & (pd.DatetimeIndex(ts_all) < pd.Timestamp(b))
            if m.sum() < 100 or wm.sum() < 500:
                continue
            cw = c[wm]; drift = float(cw[-1] / cw[0] - 1.0) * 1e4
            pool = np.flatnonzero(wm & np.isfinite(atr_all) & (np.arange(n) + 1 + FWD < n))
            if len(pool) > NULL_POOL:
                pool = np.sort(rng.choice(pool, NULL_POOL, replace=False))
            pst = pool + 1; pix = pst[:, None] + np.arange(FWD)
            R = {"n": int(m.sum()), "market_drift_bp": round(drift, 1), "sides": {}}
            for sgn, nm in ((-1.0, "short(bottom_fire)"), (1.0, "long(top_fire)")):
                sel = m & (cont_sign == sgn)
                if sel.sum() < 50:
                    continue
                rr = pf(cand_of(ts[sel], i[sel] + 1, i[sel] + 1 + ex[sel], pnl[sel]))
                pr, pe = sim_exit(o[pst], atr_all[pool], np.full(len(pool), sgn), h[pix], l[pix], c[pix], *CELL)
                pp = pr * 1e4 - COST
                vals = []
                for _ in range(B_NULL):
                    k = rng.choice(len(pool), size=min(int(sel.sum()), len(pool)), replace=False)
                    x = cand_of(ts_all[pool[k]], pool[k] + 1, pool[k] + 1 + pe[k], pp[k])
                    q = portfolio(x, CAP); vals.append(q["exp_bp"] if q else np.nan)
                v = np.asarray(vals, float); obs = rr["stats"]["exp_bp"]
                R["sides"][nm] = {"n": int(sel.sum()), "R_exp_bp": obs, "R_win_rate": rr["stats"]["win_rate"],
                                  "R_day_ci95": rr["stats"]["day_ci95"],
                                  "null_same_side_mean_bp": round(float(np.nanmean(v)), 2),
                                  "null_same_side_p95": round(float(np.nanpercentile(v, 95)), 2),
                                  "excess_over_same_side_null_bp": round(obs - float(np.nanmean(v)), 2),
                                  "percentile_of_R": round(float((v < obs).mean() * 100), 1)}
            A["windows"][w] = R
        out["assets"][sym] = A
        log(f"{sym}: " + " | ".join(
            f"{w} 드리프트{A['windows'][w]['market_drift_bp']:.0f}bp " +
            " ".join(f"{k.split('(')[0]} R={s['R_exp_bp']:.1f} 귀무={s['null_same_side_mean_bp']:.1f} 초과={s['excess_over_same_side_null_bp']:.1f}(%{s['percentile_of_R']:.0f})"
                     for k, s in A["windows"][w]["sides"].items()) for w in A["windows"]))
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")


if __name__ == "__main__":
    main()
