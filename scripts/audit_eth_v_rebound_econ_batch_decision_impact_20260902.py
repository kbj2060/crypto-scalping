#!/usr/bin/env python3
"""T3 후속 -- TabPFN 배치 의존성이 **실제 매매 결정**을 바꾸는지 정량화.

## 배경

`audit_eth_v_rebound_econ_lookahead_contamination_20260902.py` T3에서 청크 크기에 따라
확률이 최대 **1.284e-4** 달라졌다(순서 역전은 0 -> batch-statistic 정규화로 추정).
백테스트는 20,000행씩, **라이브는 6행씩**(현재 봉 x 양방향 x SCORE_TAIL_BARS) 예측하므로
백테스트/라이브 불일치 가능성이 실재한다.

## 그런데 확률 차이 != 결정 차이

임계값이 p>=0.8158이므로, 결정이 바뀌려면 확률이 임계값의 1.3e-4 이내에 있어야 한다.
**몇 건이 실제로 뒤집히는지**와 **그 건들이 PnL에 얼마나 영향을 주는지**를 직접 센다.

  · 라이브와 같은 청크(6행)로 전 OOS/HOLDOUT을 예측 -> 백테스트 청크(20,000)와 호출 집합 비교
  · 추가/누락된 호출 수와 그 거래들의 실현 PnL
  · 임계값 근방(+-1e-3) 밀도

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


_pf = _load("pf_bi", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK

CUT, CELL, MC = 0.8158, (5.0, 1.5, 0.1), 5
LIVE_CHUNK, BT_CHUNK = 6, 20000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_econ_lookahead_audit_20260902/batch_decision_impact.json"


def log(m): print(f"[batch] {m}", flush=True)


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
    log(f"OOS {len(oos):,}행 -- 두 청크 크기로 5시드 앙상블 예측")

    Pbt, Pl = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        m = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        m.fit(ctx[TIER0], ctx["y"].to_numpy())
        Pbt.append(np.concatenate([m.predict_proba(oos[TIER0].iloc[k:k+BT_CHUNK])[:, 1]
                                   for k in range(0, len(oos), BT_CHUNK)]))
        Pl.append(np.concatenate([m.predict_proba(oos[TIER0].iloc[k:k+LIVE_CHUNK])[:, 1]
                                  for k in range(0, len(oos), LIVE_CHUNK)]))
        log(f"  seed {sd} 완료")
    oos["p_bt"] = np.vstack(Pbt).mean(axis=0)
    oos["p_live"] = np.vstack(Pl).mean(axis=0)

    d = np.abs(oos["p_live"] - oos["p_bt"])
    log("")
    log(f"확률 차이: max {d.max():.3e}  중앙 {np.median(d):.3e}  평균 {d.mean():.3e}")
    near = int(((oos["p_bt"] - CUT).abs() <= 1e-3).sum())
    log(f"임계값 ±1e-3 근방 행: {near:,} / {len(oos):,} ({near/len(oos)*100:.4f}%)")

    a = set(np.flatnonzero((oos["p_bt"] >= CUT).to_numpy()))
    b = set(np.flatnonzero((oos["p_live"] >= CUT).to_numpy()))
    only_bt, only_live = a - b, b - a
    log("")
    log(f"호출 집합: 백테스트청크 {len(a):,}  라이브청크 {len(b):,}  "
        f"공통 {len(a & b):,}")
    log(f"  백테스트에만 {len(only_bt)}건 / 라이브에만 {len(only_live)}건  "
        f"(불일치율 {(len(only_bt)+len(only_live))/max(len(a),1)*100:.4f}%)")

    def pnl_of(idxset):
        if not idxset:
            return None
        s = oos.iloc[sorted(idxset)]
        j = s["pos"].to_numpy().astype(int)
        sgn = np.where(s["is_downside"].to_numpy() == 1, 1.0, -1.0)
        H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
        L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
        pn, _ = sim_exit(o[j+1], s["atr"].to_numpy(float), sgn, H, L, C, *CELL)
        v = pn * 1e4 - 10.0
        return {"n": int(len(v)), "exp_bp": round(float(v.mean()), 3),
                "total_bp": round(float(v.sum()), 1)}

    res = {"prob_diff": {"max": float(d.max()), "median": float(np.median(d)),
                         "mean": float(d.mean())},
           "near_threshold_rows": near, "oos_rows": int(len(oos)),
           "calls_bt": len(a), "calls_live": len(b), "common": len(a & b),
           "only_bt": pnl_of(only_bt), "only_live": pnl_of(only_live)}
    for k, v in (("백테스트에만", res["only_bt"]), ("라이브에만", res["only_live"])):
        if v:
            log(f"  {k} {v['n']}건: 기대값 {v['exp_bp']:+.2f}bp  총 {v['total_bp']:+.1f}bp")

    # 포트폴리오 수준 영향
    log("")
    log("=== 포트폴리오 수준 영향 ===")
    pf_res = {}
    for nm, col in (("백테스트청크", "p_bt"), ("라이브청크", "p_live")):
        s = oos.loc[oos[col] >= CUT]
        j = s["pos"].to_numpy().astype(int)
        sgn = np.where(s["is_downside"].to_numpy() == 1, 1.0, -1.0)
        H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
        L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
        pn, ex = sim_exit(o[j+1], s["atr"].to_numpy(float), sgn, H, L, C, *CELL)
        cand = pd.DataFrame({"timestamp": s["timestamp"].to_numpy(), "entry_bar": j+1,
                             "exit_bar": j+1+ex, "pnl_bp": pn*1e4-10.0})
        r = portfolio(cand, MC)
        pf_res[nm] = {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in r.items() if k not in ("idx", "pnl", "ts")}
        log(f"  {nm}: n={r['n']:,}  기대값 {r['exp_bp']:+.3f}bp  총 {r['total_bp']:+.0f}bp")
    diff = pf_res["라이브청크"]["exp_bp"] - pf_res["백테스트청크"]["exp_bp"]
    log(f"  ⇒ 기대값 차이 {diff:+.4f}bp "
        f"({'✅무시 가능' if abs(diff) < 0.1 else '⚠️유의미'})")
    res["portfolio"] = pf_res
    res["portfolio_exp_diff_bp"] = round(diff, 4)
    res["verdict"] = "negligible" if abs(diff) < 0.1 else "material"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({res['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
