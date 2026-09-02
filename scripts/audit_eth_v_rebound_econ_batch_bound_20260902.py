#!/usr/bin/env python3
"""T3 후속 (효율판) -- 배치 의존성의 **결정 영향 상한**을 해석적으로 구한다.

## 왜 설계를 바꿨나

전 행(51,840)을 라이브 청크(6행)로 재예측하려면 시드당 8,640회 호출이 필요한데,
**TabPFN은 호출마다 18,000행 컨텍스트를 재인코딩**하므로 비용이 예측 행 수가 아니라
**호출 횟수**에 비례한다. 실측 86분에 첫 시드도 못 끝냈다(호출당 4초 이상).

## 대신: 상한을 측정하고 해석적으로 적용

  1) **작은 표본(600행)** 에서 라이브 청크(6행) vs 일괄 예측의 **max|Δp|** 를 측정 -> 상한 δ
     (5시드 앙상블 기준. 앙상블은 평균이라 단일 시드보다 δ가 작거나 같다)
  2) 전 행의 백테스트 확률 `p_bt`(일괄 예측, 저렴)에서
     **|p_bt − CUT| <= δ** 인 행을 센다 -> **결정이 뒤집힐 수 있는 행의 상한**
  3) 그 행들만 실제로 6행 청크로 재예측해 **실제 뒤집힌 건수**와 PnL 영향을 확정

2단계에서 0이면 3단계는 불필요하다. 이게 전수 재예측과 **논리적으로 동등**하면서
호출 수가 수백 배 적다.

⚠️읽기 전용.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_bb", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK

CUT, CELL, MC = 0.8158, (5.0, 1.5, 0.1), 5
LIVE_CHUNK, BT_CHUNK = 6, 20000
BOUND_SAMPLE = 600          # 상한 측정용 표본 (600/6 = 100 호출/시드)
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_econ_lookahead_audit_20260902/batch_decision_bound.json"


def log(m): print(f"[bound] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)

    sl0, arm0, tr0 = LABEL_CELL
    ii = long["pos"].to_numpy().astype(int)
    sg = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    at = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        j = ii[s_:e_]
        H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
        L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
        pn, _ = sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - 10.0
    long["y"] = (net > 0).astype(float)

    tr_set = long.loc[long["split"] == "TRAIN"]
    oos = long.loc[long["split"] == "OOS"].reset_index(drop=True)
    models, Pbt = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        m = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        m.fit(ctx[TIER0], ctx["y"].to_numpy())
        models.append(m)
        Pbt.append(np.concatenate([m.predict_proba(oos[TIER0].iloc[k:k+BT_CHUNK])[:, 1]
                                   for k in range(0, len(oos), BT_CHUNK)]))
        log(f"  seed {sd} 일괄예측 완료")
    oos["p_bt"] = np.vstack(Pbt).mean(axis=0)

    # ---- 1) 상한 δ 측정 (작은 표본) ----
    log("")
    log(f"=== 1) 상한 δ 측정 -- 표본 {BOUND_SAMPLE}행, 라이브청크 {LIVE_CHUNK}행 ===")
    rs = np.random.default_rng(20260902)
    samp_idx = np.sort(rs.choice(len(oos), size=BOUND_SAMPLE, replace=False))
    samp = oos.iloc[samp_idx]
    Pl = []
    for si, m in enumerate(models):
        Pl.append(np.concatenate([m.predict_proba(samp[TIER0].iloc[k:k+LIVE_CHUNK])[:, 1]
                                  for k in range(0, len(samp), LIVE_CHUNK)]))
        log(f"  seed {SEEDS[si]} 라이브청크 완료 ({len(samp)//LIVE_CHUNK}회 호출)")
    p_live_samp = np.vstack(Pl).mean(axis=0)
    d = np.abs(p_live_samp - samp["p_bt"].to_numpy())
    delta = float(d.max())
    log(f"  앙상블 max|Δp| = **{delta:.3e}**  (중앙 {np.median(d):.3e}, 0 아닌 행 {int((d>0).sum())}/{len(d)})")

    # ---- 2) 해석적 상한: 뒤집힐 수 있는 행 ----
    log("")
    log("=== 2) 해석적 상한 -- 결정이 뒤집힐 수 있는 행 ===")
    gap = (oos["p_bt"] - CUT).abs().to_numpy()
    n_flip_max = int((gap <= delta).sum())
    calls_bt = int((oos["p_bt"] >= CUT).sum())
    log(f"  OOS {len(oos):,}행 중 |p−CUT| ≤ δ({delta:.3e}) 인 행: **{n_flip_max}건**")
    log(f"  (참고) 호출 {calls_bt:,}건 대비 최대 영향 비율 {n_flip_max/max(calls_bt,1)*100:.4f}%")
    for band in (1e-4, 1e-3, 1e-2):
        log(f"    |p−CUT| ≤ {band:.0e}: {int((gap<=band).sum()):,}행")

    res = {"delta_max_abs_dp": delta, "delta_median": float(np.median(d)),
           "sample_rows": BOUND_SAMPLE, "live_chunk": LIVE_CHUNK,
           "oos_rows": int(len(oos)), "calls_bt": calls_bt,
           "max_flippable_rows": n_flip_max,
           "band_counts": {f"{b:.0e}": int((gap <= b).sum()) for b in (1e-4, 1e-3, 1e-2)}}

    # ---- 3) 실제 뒤집힘 확인 (상한이 0이 아닐 때만) ----
    if n_flip_max == 0:
        log("")
        log("  ⇒ ✅상한이 0건 -- 배치 의존성이 매매 결정을 바꿀 수 없다. 3단계 불필요.")
        res["verdict"] = "no_decision_impact"
        res["actual_flips"] = 0
    else:
        log("")
        log(f"=== 3) 실제 뒤집힘 확인 -- {n_flip_max}행 재예측 ===")
        fi = np.flatnonzero(gap <= delta)
        sub = oos.iloc[fi]
        Pf = [np.concatenate([m.predict_proba(sub[TIER0].iloc[k:k+LIVE_CHUNK])[:, 1]
                              for k in range(0, len(sub), LIVE_CHUNK)]) for m in models]
        pl = np.vstack(Pf).mean(axis=0)
        flips = int(((sub["p_bt"].to_numpy() >= CUT) != (pl >= CUT)).sum())
        log(f"  실제 뒤집힌 건: **{flips}건**")
        res["actual_flips"] = flips
        res["verdict"] = "no_decision_impact" if flips == 0 else "check_pnl"
        if flips:
            fl = sub.iloc[np.flatnonzero((sub["p_bt"].to_numpy() >= CUT) != (pl >= CUT))]
            j = fl["pos"].to_numpy().astype(int)
            sgn = np.where(fl["is_downside"].to_numpy() == 1, 1.0, -1.0)
            H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
            L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
            C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
            pn, _ = sim_exit(o[j+1], fl["atr"].to_numpy(float), sgn, H, L, C, *CELL)
            v = pn * 1e4 - 10.0
            log(f"  뒤집힌 거래들의 PnL: 기대값 {v.mean():+.2f}bp  총 {v.sum():+.1f}bp "
                f"(전체 누적 +11,031bp 대비 {abs(v.sum())/11031*100:.3f}%)")
            res["flipped_pnl"] = {"n": int(len(v)), "exp_bp": round(float(v.mean()), 3),
                                  "total_bp": round(float(v.sum()), 1)}

    log("")
    log(f"⇒ 판정: {res['verdict']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({res['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
