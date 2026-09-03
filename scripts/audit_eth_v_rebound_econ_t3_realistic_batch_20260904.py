#!/usr/bin/env python3
"""T3 후속 -- **라이브 실제 배치 크기(1~3행)** vs **원본 백테스트 배치(창 전체)**를 직접 비교.

원본 T3(`audit_eth_v_rebound_econ_lookahead_contamination_20260902.py`)는 청크
{20,100,500,2000}을 "2000행 일괄"과 비교했다. 그런데 실제 배치 크기 두 극단을 안 쟀다:
  · 원본 백테스트/HOLDOUT은 청크 20,000~40,000을 쓰고(`research_eth_v_rebound_direct_
    economic_label_20260902.py:246`, `..._ensemble_portfolio_sim_20260902.py:CHUNK=40000`) --
    VAL/OOS/HOLDOUT 창(1,383~1,987행)이 전부 그 이하라 **사실상 창 전체를 한 배치**로 채점했다.
  · 라이브(`compute_signal()`)는 사이클당 후보 **1~3개**뿐이다.
그래서 "2000 vs 20"이 아니라 "**창 전체 vs 1~3행**"이 진짜 비교 대상이다.

판정: 이 격차가 **임계값(0.8221) 판정을 뒤집을 수 있는 후보가 있는지** 직접 센다.
뒤집는 후보가 있으면 코드 수정(라이브를 큰 컨텍스트와 함께 채점) 필요, 없으면 문서화로 충분.

⚠️읽기 전용. 원본 감사와 같은 로딩 절차를 그대로 재사용(자체 재구현 금지 -- 미묘한 차이 방지).
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


_pf = _load("pf_audit2", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit = _pf.sim_exit
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
CUT = 0.8221                                  # 라이브 임계값(3시드 재보정본)
OUT = ROOT / "data/research/eth_v_rebound_econ_t3_realistic_batch_20260904/report.json"


def log(m): print(f"[t3b] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    log("프레임/라벨 재구성...")
    _s1.VAL_END = OOS_END
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
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)
    log(f"  long {len(long):,}행 -- 라벨(y) 계산 중...")

    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
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
        pn, ex = sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - 10.0
    long["y"] = (net > 0).astype(float)
    log(f"  라벨 완료")

    from tabpfn import TabPFNClassifier
    tr_set = long.loc[long["split"] == "TRAIN"]
    rng = np.random.default_rng(SEEDS[0])
    ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
    clf = TabPFNClassifier(device="cuda", random_state=SEEDS[0], ignore_pretraining_limits=True)
    clf.fit(ctx[TIER0], ctx["y"].to_numpy())

    oos = long.loc[long["split"] == "OOS"].reset_index(drop=True)
    log(f"  OOS 전체 {len(oos):,}행")

    # ⭐batch_1을 51,840행 전체에 걸면 각 호출이 컨텍스트(18,000행)를 다시 인코딩해야 해서
    # (오늘 진입모델 실측: 호출당 ~69초, 쿼리행수와 거의 무관) 51,840 x 69초 ≈ 46일이 된다.
    # 그래서 **큰 배치(2000)로 먼저 스크리닝**해 임계값(0.8221) 근처 행만 골든셋으로 추리고,
    # 그 골든셋에만 batch_1을 건다 -- 판정을 뒤집을 수 있는 건 애초에 그 근처 행뿐이다.
    log("  1단계: 배치 2000으로 스크리닝 채점...")
    p_screen = np.concatenate([clf.predict_proba(oos[TIER0].iloc[k:k+2000])[:, 1]
                               for k in range(0, len(oos), 2000)])
    MARGIN = 0.01
    near = np.flatnonzero(np.abs(p_screen - CUT) <= MARGIN)
    log(f"  임계값 ±{MARGIN} 근처(골든셋): {len(near):,}행 / 전체 {len(oos):,}행")
    if len(near) > 30:
        near = near[np.argsort(np.abs(p_screen[near] - CUT))[:30]]
        log(f"  → 상위 30개(경계에 가장 가까운 순)로 축소 -- GPU시간 절감, 20~2000배치 사이 이미 격차가 안정적이었음")

    log(f"  2단계: 골든셋 {len(near)}행을 batch=1로 개별 채점 (예상 ~{len(near)*70/60:.0f}분)...")
    p1 = np.array([clf.predict_proba(oos[TIER0].iloc[[i_]])[:, 1][0] for i_ in near])

    d = np.abs(p1 - p_screen[near])
    call_screen = p_screen[near] >= CUT
    call_1 = p1 >= CUT
    n_flip = int((call_screen != call_1).sum())

    results = {"margin": MARGIN,
               "n_near_threshold_total": int(len(np.flatnonzero(np.abs(p_screen - CUT) <= MARGIN))),
               "n_tested_batch1": int(len(near)),
               "max_abs_diff_at_boundary": float(d.max()) if len(d) else None,
               "mean_abs_diff_at_boundary": float(d.mean()) if len(d) else None}
    if len(d):
        log(f"  경계 근처 batch1 vs batch2000: max|Δp|={d.max():.3e} mean={d.mean():.3e}")
    else:
        log("  경계 근처 행 없음")

    flips = {"n_flip": n_flip, "n_tested": int(len(near)),
             "flip_pct": round(n_flip / max(len(near), 1) * 100, 4)}
    log(f"\n=== ⭐임계값(0.8221) 판정이 실제로 뒤집히는가 (경계 근처 {len(near)}행 기준) ===")
    log(f"  뒤집힌 행 {n_flip}/{len(near)} ({n_flip/max(len(near),1)*100:.2f}%)")
    if n_flip:
        idx = np.flatnonzero(call_screen != call_1)[:8]
        for i_ in idx:
            log(f"    예시 pos={near[i_]} batch2000_p={p_screen[near[i_]]:.4f} batch1_p={p1[i_]:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"near_threshold": results, "threshold_flips": flips,
                               "cut": CUT, "n_oos": len(oos),
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
