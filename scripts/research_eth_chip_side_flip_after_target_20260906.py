#!/usr/bin/env python3
"""증거신호 8종 — **목표 달성 후 호라이즌 전 측면 뒤바뀜**을 허용 vs 잠금 (2026-09-06).

사용자: *"8가지 모두 목표가 달성 이후 호라이즌 전에 신호가 이전과 뒤바뀌는 것과 뒤바뀌지 않게 만든 것과 성능 테스트."*

현행 규약(`live_evidence_signal_dashboard_20260823.compute_signals`)에서 `bottom_*_fill`과 `top_*_fill`은
**완전히 독립**으로 채워진다. 그래서 바닥 발동이 목표에 닿아 꺼진 뒤에도, 그 발동의 호라이즌이 끝나기 전에
천장 발동이 나면 칩·표가 그대로 반대편으로 넘어간다(실측 kalman 21.0% · taker 17.6%의 활성 봉에서 양측 동시).

## 팔 (전부 R 규약 상속 — 자유도 0)
    A   현행: 모든 첫발동(GAP=12, 신호·측면별)
    B1  ⭐**목표 달성 후 잠금**: 같은 신호의 반대 측면이 직전 H봉 안에 발동했고 **그 목표가 이미 닿았으면**
        이번 발동을 버린다(원 호라이즌이 끝날 때까지 측면을 유지한다는 뜻)
    B2  **전면 잠금**: 목표 도달 여부와 무관하게, 반대 측면이 직전 H봉 안에 발동했으면 버린다
    F1  **버려진 것만**: B1이 제거한 발동들 — 이게 좋으면 잠그면 손해다(분해)
    H, K는 신호별 자기 값(SUSTAIN_BARS_OVERRIDE / K_OVERRIDE, 두 모듈 수동 동기화값을 **소스에서 파싱해 검증**)

## 손익 (R 그대로)
    방향 지속(신호 반대) · 진입 open[i+1] · sim_exit(5.0/1.5/0.1 ×ATR14) 200봉 · 10bp · 동시 5 슬롯
    신호별 개별 + 합집합((봉,측면) 중복제거, R의 실제 모집단) 둘 다.
판정: VAL·OOS 두 창 모두 A 대비 일별 짝비교 CI 하한 > 0. HOLDOUT(≥2026-04-01) 로드 단계에서 차단.
"""
from __future__ import annotations

import importlib.util
import json
import re
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


XA = _load("xa_flip", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
C1 = _load("c1_flip", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
V2 = _load("v2_flip", "scripts/research_homer_entry_v2_20260904.py")
OUT = ROOT / "data/research/eth_chip_side_flip_after_target_20260906"
CELL, FWD, COST, GAP, CAP = (5.0, 1.5, 0.1), 200, 10.0, 12, 5
SPLITS = XA.SPLITS
# 수동 동기화 상수 -- 아래에서 **소스 파싱으로 검증**한다(어긋나면 즉시 중단)
HORIZON = {"taker_delta_z_climax": 24, "short_term_return_z": 12, "liquidity_sweep": 30,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
K = {"taker_delta_z_climax": 2.00, "short_term_return_z": 1.75, "liquidity_sweep": 4.00,
     "orthogonal_combo": 3.571, "smt_divergence": 4.20, "fib_extension_exhaustion": 2.35,
     "demarker_extreme": 0.70, "kalman_deviation_meanrev": 2.5}
rng = np.random.default_rng(20260906)


def log(m): print(f"[flip] {m}", flush=True)


def verify_constants():
    d = (ROOT / "scripts/live_evidence_signal_dashboard_20260823.py").read_text(encoding="utf-8")
    ko = {a: float(b) for a, b in re.findall(r'"(\w+)":\s*([\d.]+)', re.search(r"K_OVERRIDE = \{(.*?)\}", d, re.S).group(1))}
    so = {a: int(b) for a, b in re.findall(r'"(\w+)":\s*(\d+)', re.search(r"SUSTAIN_BARS_OVERRIDE = \{(.*?)\}", d, re.S).group(1))}
    assert ko == K and so == HORIZON, f"상수 불일치\nK: {ko}\nH: {so}"
    log(f"상수 파리티 OK (K·H 각 {len(K)}종, 라이브 정본 소스와 일치)")


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    verify_constants()
    kl = XA.load_kl("ETHUSDT"); btc = XA.load_kl("BTCUSDT")       # HOLDOUT 차단 포함
    sig = XA.DASH.compute_signals(kl.copy(), btc_df=btc, funding_df=None)
    n = len(kl)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    atr_pct = atr / c
    ts_all = kl["timestamp"].to_numpy()

    def target_touched(j, side, H_, k_, before):
        """봉 j 발동의 K×ATR 목표가에 min(j+H, before)까지 닿았는가 (고가/저가, `_tp_touched` 규약)."""
        if not np.isfinite(atr_pct[j]):
            return False
        lvl = c[j] * (1 + k_ * atr_pct[j]) if side == "bottom" else c[j] * (1 - k_ * atr_pct[j])
        end = min(j + H_, before - 1, n - 1)
        if end <= j:
            return False
        seg_h, seg_l = h[j + 1:end + 1], l[j + 1:end + 1]
        return bool((seg_h >= lvl).any()) if side == "bottom" else bool((seg_l <= lvl).any())

    rows = []
    for s in XA.SIGNALS:
        raw = {sd: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for sd in ("bottom", "top")}
        ff = {sd: XA.first_fire_mask(raw[sd], GAP) for sd in ("bottom", "top")}
        H_, k_ = HORIZON[s], K[s]
        for sd in ("bottom", "top"):
            opp = "top" if sd == "bottom" else "bottom"
            opp_idx = np.flatnonzero(raw[opp])
            for i in np.flatnonzero(ff[sd]):
                prior = opp_idx[(opp_idx < i) & (opp_idx > i - H_)]       # 직전 H봉 안 반대 측면 발동(원시)
                rev = len(prior) > 0
                rev_tp = bool(rev and any(target_touched(int(j), opp, H_, k_, i) for j in prior))
                rows.append({"signal": s, "i": int(i), "is_downside": 1 if sd == "bottom" else 0,
                             "rev_within_h": rev, "rev_tp_touched": rev_tp})
    F = pd.DataFrame(rows)
    ok = (F["i"].to_numpy() + 1 + FWD < n) & np.isfinite(atr[F["i"].to_numpy()])
    F = F.loc[ok].reset_index(drop=True)
    i_ = F["i"].to_numpy(); sd_ = F["is_downside"].to_numpy()
    sign = np.where(sd_ == 1, -1.0, 1.0)                                  # 지속: 바닥 발동 → 숏
    ix = (i_ + 1)[:, None] + np.arange(FWD)
    ret, ex = V2.sim_exit(o[i_ + 1], atr[i_], sign, h[ix], l[ix], c[ix], *CELL)
    F["pnl_bp"] = ret * 1e4 - COST; F["exit_off"] = ex; F["ts"] = ts_all[i_]
    tsi = pd.DatetimeIndex(F["ts"]); F["split"] = "NONE"
    for w, (a, b) in SPLITS.items():
        F.loc[(tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b)), "split"] = w
    log(f"첫발동 {len(F):,} · 반대측면 H봉내 {F['rev_within_h'].mean():.3f} · 그중 목표달성 {F['rev_tp_touched'].mean():.3f}")

    def pf_of(G):
        if len(G) < 60:
            return None
        return C1.pf(C1.cand_of(G["ts"].to_numpy(), G["i"].to_numpy() + 1,
                                G["i"].to_numpy() + 1 + G["exit_off"].to_numpy(), G["pnl_bp"].to_numpy()))

    def arms_of(G):
        return {"A_current": G, "B1_lock_after_target": G.loc[~G["rev_tp_touched"]],
                "B2_lock_always": G.loc[~G["rev_within_h"]], "F1_only_flipped": G.loc[G["rev_tp_touched"]]}

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cell": CELL, "cost_bp": COST, "gap": GAP,
           "max_concurrent": CAP, "horizon": HORIZON, "k": K, "holdout_excluded": True,
           "n_first_fires": int(len(F)), "per_signal": {}, "pooled": {}}
    # ── 신호별
    for s in XA.SIGNALS:
        Gs = F.loc[F["signal"] == s]
        rec = {"n": int(len(Gs)), "rev_within_h_rate": round(float(Gs["rev_within_h"].mean()), 3),
               "flip_after_target_rate": round(float(Gs["rev_tp_touched"].mean()), 3), "windows": {}}
        for w in SPLITS:
            Gw = Gs.loc[Gs["split"] == w]; base = pf_of(Gw)
            if base is None:
                continue
            d = {"A_n": base["stats"]["n"], "A_exp_bp": base["stats"]["exp_bp"], "A_day_ci95": base["stats"]["day_ci95"]}
            for nm, G in arms_of(Gw).items():
                if nm == "A_current":
                    continue
                r = pf_of(G)
                d[nm] = {"n": r["stats"]["n"], "exp_bp": r["stats"]["exp_bp"], "day_ci95": r["stats"]["day_ci95"],
                         "vs_A": C1.day_paired(r["pnl"], r["ts"], base["pnl"], base["ts"])} if r else {"skip": "n<60"}
            rec["windows"][w] = d
        rep["per_signal"][s] = rec
    # ── 합집합 (R의 실제 모집단: (봉,측면) 중복제거)
    for w in SPLITS:
        Gw = F.loc[F["split"] == w]
        d = {}
        for nm, G in arms_of(Gw).items():
            U = G.sort_values("i").drop_duplicates(["i", "is_downside"])
            r = pf_of(U)
            d[nm] = {"n": r["stats"]["n"], "exp_bp": r["stats"]["exp_bp"], "win_rate": r["stats"]["win_rate"],
                     "day_ci95": r["stats"]["day_ci95"], "per_day": r["stats"]["per_day"],
                     "daily_mean_bp": r["stats"]["daily_mean_bp"], "daily_sharpe_ann": r["stats"]["daily_sharpe_ann"],
                     "_pnl": r["pnl"], "_ts": r["ts"]} if r else None
        for nm in ("B1_lock_after_target", "B2_lock_always", "F1_only_flipped"):
            if d.get(nm) and d.get("A_current"):
                d[nm]["vs_A"] = C1.day_paired(d[nm]["_pnl"], d[nm]["_ts"], d["A_current"]["_pnl"], d["A_current"]["_ts"])
        rep["pooled"][w] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in d.items() if v}
        log(f"  합집합 {w}: " + " | ".join(
            f"{nm.split('_')[0]} n={d[nm]['n']:>5} exp={d[nm]['exp_bp']:>6}" + (f" ΔA={d[nm]['vs_A']['diff_bp_day']:>6}{d[nm]['vs_A']['ci95']}" if "vs_A" in d[nm] else "")
            for nm in ("A_current", "B1_lock_after_target", "B2_lock_always", "F1_only_flipped") if d.get(nm)))
    P = [nm for nm in ("B1_lock_after_target", "B2_lock_always")
         if all(rep["pooled"].get(w, {}).get(nm, {}).get("vs_A", {}).get("ci95", [-9])[0] > 0 for w in ("VAL", "OOS"))]
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 A 대비 CI 하한 > 0", "passes": P, "n_pass": len(P)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · 통과 {len(P)} {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
