#!/usr/bin/env python3
"""E0 경제라벨 라이브 후보의 **동결 컨텍스트 아티팩트** 생성.

배포판(`live_eth_sweep_v_rebound_signal_20260829.py`)이 `tabpfn_train_context_frozen_*.csv`
하나를 쓰는 것과 같은 방식이되, 이 후보는 **5시드 앙상블**이므로 시드별 컨텍스트를 한 파일에
`seed` 컬럼으로 담는다(5 x 18,000 = 90,000행). 라이브에서 시드별로 나눠 5개 모델을 적합한다.

라벨: E0_binary = (open[i+1] 진입 -> 트레일링(5.0/1.5/0.1) 청산 -> 비용 10bp) > 0
피쳐: Tier0 23 (154피쳐는 전 구성에서 악화되어 제외)
학습 구간: TRAIN < 2025-09-01 (VAL/OOS/HOLDOUT 전부 미포함)

⚠️이 스크립트는 **아티팩트 생성 전용**이다. 성능 주장을 하지 않는다 --
근거는 `docs/homer/v_rebound_open_issues_20260901.md` 20절.

Run on the server via handoff, then pull the CSV.
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


_pf = _load("pf_ctx", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit = _pf.sim_exit
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK
COST = _pf.COST_TAKER

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
OUTDIR = ROOT / "data/labels/eth_5m_v_rebound_econ_label_20260902"


def log(m): print(f"[ctx] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    _s1.VAL_END = TRAIN_END          # TRAIN 구간만 만들면 충분
    log("building TRAIN frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long = long.loc[long["timestamp"] < TRAIN_END].reset_index(drop=True)
    assert long["timestamp"].max() < TRAIN_END, "TRAIN 경계 위반"

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
        net[s_:e_] = pn * 1e4 - COST
    long["label"] = (net > 0).astype(float)
    log(f"TRAIN {len(long):,}행  라벨률 {long['label'].mean():.4f}")

    parts = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        idx = np.sort(rng.choice(len(long), size=min(CONTEXT_N, len(long)), replace=False))
        p_ = long.iloc[idx][["timestamp", "label"] + TIER0].copy()
        p_.insert(0, "seed", sd)
        parts.append(p_)
        log(f"  seed {sd}: {len(p_):,}행 라벨률 {p_['label'].mean():.4f}")
    ctx = pd.concat(parts, ignore_index=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv = OUTDIR / "tabpfn_train_context_frozen_econ_5seed_20260902.csv"
    ctx.to_csv(csv, index=False)
    rep = {"artifact": str(csv.relative_to(ROOT)), "rows": int(len(ctx)),
           "seeds": SEEDS, "context_n_per_seed": CONTEXT_N,
           "label": {"kind": "E0_binary", "definition":
                     "(open[i+1] 진입 -> 트레일링 SL5.0/ARM1.5/Trail0.1 x ATR -> 비용 10bp) > 0",
                     "train_label_rate": round(float(long["label"].mean()), 5)},
           "features": TIER0, "n_features": len(TIER0),
           "train_range": [str(long["timestamp"].min()), str(long["timestamp"].max())],
           "train_pool_rows": int(len(long)),
           "serving": {"threshold": 0.8158, "cell_sl_arm_trail": list(LABEL_CELL),
                       "max_concurrent": 5, "entry": "open[i+1] (시장가)",
                       "note": "임계값/셀/한도는 VAL에서 선정, OOS·HOLDOUT 각 1회 검증"},
           "evidence": "docs/homer/v_rebound_open_issues_20260901.md 20절"}
    (OUTDIR / "context_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"saved -> {csv}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
