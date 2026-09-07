#!/usr/bin/env python3
"""자산 확장 — 지속 규칙(R)을 보유 59심볼 전체에 적용 (2026-09-06).

근거: `eth_composite_direction_trend_pullback_results_20260905.md` §7-4·§10 권고1 —
ETH 안에서 R을 정교화하는 116팔이 전부 실패했고, **두 확인 창을 모두 통과한 유일한 구성이
3자산 합산**이었다(TRAIN 6.95 / VAL 3.52 / OOS 4.18 샤프, 자산간 일손익 상관 0.05~0.29).
근거가 적합이 아니라 **구조**(무상관 분산)이므로 이번 세션이 막은 방향예측 벽에 걸리지 않는다.

## 자유도 0 계약 (교차자산 사전등록 상속)
셀을 자산별로 재선정하지 **않는다** — §7-2에서 TRAIN 재선정이 VAL에서 뒤집혔다(과적합).
**ETH 배포 셀 5.0/1.5/0.1을 59자산 공통으로 쓴다.** 모집단·진입·청산·비용 전부 ETH와 동일:
8종 raw 첫발동(GAP12, 측면별) · 진입 `open[pos+1]` · `sim_exit` 200봉 · 10bp ·
위험 0.4%/건 `notional = 0.004/(5·atr_pct)` 상한 0.5 · 자산당 동시보유 2.

## 팔 (선택 다중성을 분리해서 보기 위해)
  P0  ETH 단독                     기준
  P1  ETH+XRP+SOL                  현행 배포(사전등록됨)
  P2  TRAIN 선별 부분집합            **TRAIN에서만** 선별(일CI 하한>0 ∧ 뒤집기 우위>0), VAL/OOS 미열람
  P3  **전 자산 동일가중**            선택 자체가 없음 -> 다중성 0. ⭐가장 정직한 팔
판정: VAL·OOS **두 창** 일손익 CI 하한 > 0.

⚠️ HOLDOUT(≥2026-04-01) 미접촉. 이건 연구 점수이고 승격은 전진 섀도우다.
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


M = _load("order_mod", "scripts/research_eth_fire_ordering_model_20260906.py")
TS, V2 = M.TS, M.V2
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402

OUT = ROOT / "tmp/eth_asset_expansion_20260906"
CELL, FWD, COST, GAP = (5.0, 1.5, 0.1), 200, 10.0, 12
RISK, SL_MULT, NOTIONAL_CAP, CONC = 0.004, 5.0, 0.5, 2
TRAIN_S, TRAIN_E = pd.Timestamp("2024-05-01"), pd.Timestamp("2025-09-01")
VAL_E, OOS_E = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
MIN_TRADES = 200
RNG = np.random.default_rng(20260906)


def log(m): print(f"[expand] {m}", flush=True)


def day_ci(v, d, B=1500):
    ud = np.unique(d)
    if len(ud) < 5:
        return (float("nan"), float("nan"))
    idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def load_kl(sym):
    p = ROOT / f"binance_data/klines/{sym}/{sym}-5m-api.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    d = d.loc[(d["timestamp"] >= TRAIN_S - pd.Timedelta(days=30)) & (d["timestamp"] < OOS_E)].reset_index(drop=True)
    if len(d) < 20000:
        return None
    d["timestamp"] = M._ns(d["timestamp"])
    return d


def run_asset(sym, btc):
    """R 규칙 per-trade 손익 + 위험정규화 notional. 자유도 0."""
    seg = load_kl(sym)
    if seg is None:
        return None
    try:
        sig = compute_signals(seg, btc_df=btc, funding_df=None)
    except Exception as ex:                                                    # noqa: BLE001
        log(f"{sym}: compute_signals 실패 {type(ex).__name__}"); return None
    h, l, c = (seg[k].to_numpy(float) for k in ("high", "low", "close"))
    atr_pct = pd.Series(h - l).rolling(14).mean().to_numpy() / c
    recs = []
    for nm, _ in SIGNAL_ORDER:
        for side, isdn in (("bottom", 1), ("top", 0)):
            raw = sig[f"{side}_{nm}"].fillna(False).to_numpy(bool)
            for i in np.flatnonzero(TS.first_fire(raw, gap=GAP)):
                recs.append((i, isdn))
    if not recs:
        return None
    F = pd.DataFrame(recs, columns=["kp", "is_downside"]).drop_duplicates(["kp", "is_downside"])
    F = F.sort_values("kp").reset_index(drop=True)
    kp = F["kp"].to_numpy(); sd_ = F["is_downside"].to_numpy().astype(int)
    ts_arr = seg["timestamp"].to_numpy()
    ok = (kp >= 300) & (kp + FWD + 2 < len(seg)) & (ts_arr[kp] >= TRAIN_S) & (ts_arr[kp] < OOS_E) & np.isfinite(atr_pct[kp])
    kp, sd_ = kp[ok], sd_[ok]
    if len(kp) < MIN_TRADES:
        return None
    # 같은 봉 양측 첫발동이면 둘 다 스킵 (러너 규약)
    dup = pd.Series(kp).duplicated(keep=False).to_numpy()
    kp, sd_ = kp[~dup], sd_[~dup]
    cont = np.where(sd_ == 1, -1.0, 1.0)                 # 지속: 바닥→숏, 천장→롱
    O = seg["open"].to_numpy(float); H, L, C = h, l, c
    off = np.arange(FWD + 1); e = kp + 1
    atr = atr_pct[kp] * C[kp]
    Hm = H[e[:, None] + off[None, :]]; Lm = L[e[:, None] + off[None, :]]; Cm = C[e[:, None] + off[None, :]]
    ret, _ = V2.sim_exit(O[e], atr, cont, Hm, Lm, Cm, *CELL)
    bp = ret * 1e4 - COST
    notional = np.minimum(RISK / (SL_MULT * atr_pct[kp]), NOTIONAL_CAP)
    d = pd.DataFrame({"ts": pd.to_datetime(ts_arr[kp]), "bp": bp, "notional": notional, "kp": kp})
    d["equity_bp"] = d["bp"] * d["notional"]             # 자기자본 대비 bp
    # 자산당 동시보유 CONC: 순차 진입, 슬롯 찬 건 스킵 (러너 규약 근사)
    keep = np.zeros(len(d), bool); open_until = []
    for i, k in enumerate(d["kp"].to_numpy()):
        open_until = [x for x in open_until if x > k]
        if len(open_until) < CONC:
            keep[i] = True; open_until.append(k + FWD)
    d = d[keep].reset_index(drop=True)
    d["split"] = np.where(d.ts < TRAIN_E, "TRAIN", np.where(d.ts < VAL_E, "VAL", "OOS"))
    d["day"] = d.ts.dt.floor("D")
    d["symbol"] = sym
    return d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sorted(p.name for p in (ROOT / "binance_data/klines").iterdir()
                  if (p / f"{p.name}-5m-api.csv").exists())
    btc = load_kl("BTCUSDT")
    log(f"후보 심볼 {len(syms)}개")
    t0 = time.time(); parts = []
    for i, s in enumerate(syms, 1):
        d = run_asset(s, btc)
        if d is not None:
            parts.append(d)
        if i % 10 == 0:
            log(f"{i}/{len(syms)} 처리 ({time.time()-t0:.0f}s) · 적격 {len(parts)}")
    A = pd.concat(parts, ignore_index=True)
    A.to_parquet(OUT / "per_trade.parquet", index=False)
    log(f"적격 자산 {A.symbol.nunique()}개 · 거래 {len(A):,}건 ({time.time()-t0:.0f}s)")

    # 자산별 요약 + TRAIN 선별
    rows = []
    for s, g in A.groupby("symbol"):
        r = {"symbol": s, "n": len(g)}
        for w in ("TRAIN", "VAL", "OOS"):
            gg = g[g.split == w]
            if len(gg) < 30:
                r[f"{w}_bp"] = np.nan; r[f"{w}_ci"] = (np.nan, np.nan); continue
            dd = gg.groupby("day")["equity_bp"].sum()
            r[f"{w}_bp"] = float(dd.mean()); r[f"{w}_n_day"] = int(len(dd))
            r[f"{w}_ci"] = day_ci(gg["equity_bp"].to_numpy(), gg["day"].to_numpy())
            r[f"{w}_sharpe"] = float(dd.mean() / dd.std() * np.sqrt(365)) if dd.std() > 0 else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    sel = S[(S["TRAIN_ci"].apply(lambda t: t[0] > 0 if isinstance(t, tuple) else False))]["symbol"].tolist()
    log(f"TRAIN 선별(일CI 하한>0): {len(sel)}개 — {sel}")

    ARMS = {"P0 ETH 단독": ["ETHUSDT"], "P1 ETH+XRP+SOL": ["ETHUSDT", "XRPUSDT", "SOLUSDT"],
            "P2 TRAIN선별": sel, "P3 전자산 동일가중": sorted(A.symbol.unique())}
    res = {"n_symbols_eligible": int(A.symbol.nunique()), "train_selected": sel,
           "per_asset": json.loads(S.to_json(orient="records"))}
    print("\n" + "=" * 96)
    print(f"{'팔':<20}{'자산수':>6}{'창':<6}{'일평균 자기자본bp':>18}{'일CI95':>26}{'샤프':>8}{'거래/일':>8}")
    for arm, syms_ in ARMS.items():
        sub = A[A.symbol.isin(syms_)]
        r = {"arm": arm, "n_assets": len(syms_)}
        for w in ("VAL", "OOS"):
            gg = sub[sub.split == w]
            if gg.empty:
                continue
            dd = gg.groupby("day")["equity_bp"].sum()
            allday = pd.date_range(gg.day.min(), gg.day.max(), freq="D")
            dd = dd.reindex(allday, fill_value=0.0)
            lo, hi = day_ci(dd.to_numpy(), dd.index.to_numpy())
            sh = float(dd.mean() / dd.std() * np.sqrt(365)) if dd.std() > 0 else np.nan
            r[w] = {"bp_day": float(dd.mean()), "ci": [lo, hi], "sharpe": sh,
                    "trades_day": float(len(gg) / len(dd))}
            print(f"{arm:<20}{len(syms_):>6}{w:<6}{dd.mean():>18.2f}"
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>26}{sh:>8.2f}{len(gg)/len(dd):>8.1f}")
        r["pass"] = bool(r.get("VAL", {}).get("ci", [0])[0] > 0 and r.get("OOS", {}).get("ci", [0])[0] > 0)
        res[arm] = r
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print("\n두 창 동시 통과 팔:", [k for k, v in res.items() if isinstance(v, dict) and v.get("pass")])
    print(f"\n산출물: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
