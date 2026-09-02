#!/usr/bin/env python3
"""BTC 경제라벨 후보의 **동결 컨텍스트 아티팩트** 생성 (5시드).

ETH판(`build_eth_v_rebound_econ_frozen_context_20260902.py`)의 BTC 대응.
라벨 셀은 1단계가 TRAIN에서만 확정한 BTC 전용 값을 리포트에서 읽는다(하드코딩 금지 --
셀이 바뀌면 아티팩트도 따라가야 한다).

⚠️BTC 전용 값: SL은 ETH(5.0)의 3.2배인 16.0×ATR. BTC ATR이 16bp로 ETH(~23bp)보다 30% 작아
같은 ATR 배수라도 절대폭이 좁기 때문. 격자를 3차까지 확장해 **내부 최적**임을 확인했다
(20/30에서 다시 나빠짐).

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


_pf = _load("pf_ctx_btc", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_sc = _load("sc_ctx_btc", "scripts/research_btc_v_rebound_econ_label_screen_20260902.py")
sim_exit, FORWARD_BARS = _pf.sim_exit, _pf.FORWARD_BARS
CONTEXT_N, SEEDS, CHUNK = _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK
TIER0 = _sc.TIER0
COST = 10.0

SCREEN = ROOT / "data/research/btc_v_rebound_econ_label_screen_20260902/report_extended2.json"
PORTFOLIO = ROOT / "data/research/btc_v_rebound_econ_portfolio_20260902/report.json"
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
OUTDIR = ROOT / "data/labels/btc_5m_v_rebound_econ_label_20260902"


def log(m): print(f"[btc-ctx] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    scr = json.loads(SCREEN.read_text())
    CELL = tuple(scr["selected_cell"]["cell"])
    log(f"BTC 라벨 셀 (1단계 TRAIN 확정): {CELL}")

    serving = {}
    if PORTFOLIO.exists():
        pf = json.loads(PORTFOLIO.read_text())
        vs = pf.get("val_selection", {})
        serving = {"threshold": vs.get("cut"), "trade_cell": vs.get("cell"),
                   "max_concurrent": vs.get("max_concurrent"),
                   "passed": pf.get("passed")}
        log(f"2단계 서빙 규격: {serving}")
    else:
        log("⚠️2단계 리포트 없음 -- 서빙 규격은 비워둔다")

    long, meta = _sc.build_long()
    df = meta.pop("df")
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    nb = len(df)
    long = long.dropna(subset=TIER0).reset_index(drop=True)
    long = long.loc[(long["bar_idx"] + FORWARD_BARS + 1 < nb)
                    & (long["timestamp"] < TRAIN_END)].reset_index(drop=True)
    assert long["timestamp"].max() < TRAIN_END, "TRAIN 경계 위반"

    sl0, arm0, tr0 = CELL
    ii = long["bar_idx"].to_numpy().astype(int)
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
    csv = OUTDIR / "tabpfn_train_context_frozen_econ_5seed_btc_20260902.csv"
    ctx.to_csv(csv, index=False)
    rep = {"artifact": str(csv.relative_to(ROOT)), "asset": "BTCUSDT", "rows": int(len(ctx)),
           "seeds": SEEDS, "context_n_per_seed": CONTEXT_N,
           "label": {"kind": "E0_binary",
                     "definition": f"(open[i+1] 진입 -> 트레일링 SL{sl0}/ARM{arm0}/Trail{tr0} x ATR -> 비용 {COST}bp) > 0",
                     "cell_selected_on": "TRAIN only, 3차 격자확장으로 내부최적 확인",
                     "train_label_rate": round(float(long["label"].mean()), 5)},
           "features": TIER0, "n_features": len(TIER0),
           "train_range": [str(long["timestamp"].min()), str(long["timestamp"].max())],
           "train_pool_rows": int(len(long)),
           "btc_vs_eth": {"btc_atr_median_bp": 16.0, "eth_atr_median_bp": 23.0,
                          "btc_label_sl": sl0, "eth_label_sl": 5.0,
                          "note": "BTC ATR이 30% 작아 같은 ATR 배수라도 절대폭이 좁다 -- SL 재튜닝 필요했음"},
           "serving": serving,
           "evidence": "docs/experiments/btc_v_rebound_econ_label_20260902.md"}
    (OUTDIR / "context_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"saved -> {csv}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
