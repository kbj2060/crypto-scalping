#!/usr/bin/env python3
"""FINAL -- the combination that was recommended but never actually measured, plus the only
trading use BTC's regime currently has. 2026-09-02, user: "최종적으로 eth, btc 레짐을 어떻게
매매에 이용할 수 있을까".

TWO GAPS THIS CLOSES
  1. Part A concluded "gate per signal", Part B concluded "top-1 or top-2". Nobody ever ran
     top-k WITH the per-signal gate applied. Both top-2 members (short_term_return_z,
     liquidity_sweep) are chop-beneficiaries per Part A, so the combination is plausible -- but
     gating cuts trade count, and Part A already showed chop gating trades total return for
     per-trade quality. Measured, not assumed.
  2. BTC's new regime classifier has NO consumer: BTC has no evidence signals on the dashboard,
     so its ribbon is display-only. The natural trading use is as a CROSS-ASSET condition on ETH
     trades -- "only take the ETH reversal when BTC is also chopping". Never tested.

ARMS (1-slot sequential portfolio, configs FROZEN from the VAL-only selection so OOS stays honest)
    top{1,2}_plain      no regime condition (the Part B baseline)
    top{1,2}_ethchop    ETH S12_K3 predicts chop
    top{1,2}_btcchop    BTC S24_K3 predicts chop      <- the only current trading use for BTC's model
    top{1,2}_bothchop   both assets chopping
Direction-flip control on every arm; VAL/OOS split reported separately.

⚠️ HOLDOUT excluded (all signals spent theirs). Regime predictions over this window are in-sample
for both regime models -- disclosed, and it applies equally to every arm so the COMPARISON is fair.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, SIGNALS, VAL_START, build_regime_pred,
)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize,
)

BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_label_train_20260902/plot_series.parquet"
VAL_SEL = ROOT / "tmp/eth_ensemble_val_selected_oos_eval_20260902/config_stability.csv"
OUT_DIR = ROOT / "tmp/eth_final_regime_gated_ensemble_20260902"
TOPK = ["short_term_return_z", "liquidity_sweep"]     # VAL-bp order from the selection-bias study


def log(m: str) -> None:
    print(f"[final_gate] {m}", flush=True)


def main() -> int:
    st = pd.read_csv(VAL_SEL).set_index("signal")
    cfgs = {}
    for n in TOPK:
        sl, arm, tr = st.loc[n, "val_only"].replace("SL", "").replace("ARM", "").replace("Tr", "").split("/")
        cfgs[n] = {"sl": float(sl), "arm": float(arm), "trail": float(tr)}
    log(f"frozen VAL-selected configs: {cfgs}")

    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = pd.DatetimeIndex(ts)

    eth_reg = build_regime_pred()
    eth_chop = set(eth_reg.loc[eth_reg["regime"] == 2, "timestamp"])
    btc = pd.read_parquet(BTC_REGIME)
    btc_chop = set(btc.loc[btc["new_pred"] == 2, "timestamp"])
    log(f"ETH chop bars {len(eth_chop):,} | BTC chop bars {len(btc_chop):,}")

    tabs = {}
    for name in TOPK:
        cfg, b = SIGNALS[name], cfgs[name]
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, atr, cfg["horizon"],
                                  b["sl"], b["arm"], b["trail"])
            t["signal"] = name
            t["decision_pos"] = [int(idx.get_loc(x)) for x in t["decision_ts"]]
            t["eth_chop"] = t["decision_ts"].isin(eth_chop)
            t["btc_chop"] = t["decision_ts"].isin(btc_chop)
            tabs[(name, lb)] = t
        r = tabs[(name, "real")]
        log(f"{name}: {len(r)} outcomes | eth_chop {r.eth_chop.mean():.3f} btc_chop {r.btc_chop.mean():.3f} "
            f"both {(r.eth_chop & r.btc_chop).mean():.3f}")

    prio = {n: i for i, n in enumerate(TOPK)}
    rows = []
    for lb in ("real", "flip"):
        allc = pd.concat([tabs[(n, lb)] for n in TOPK], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START))):
            w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
            for k in (1, 2):
                base = w[w["signal"].isin(TOPK[:k])]
                gates = {"plain": np.ones(len(base), bool),
                         "ethchop": base["eth_chop"].to_numpy(),
                         "btcchop": base["btc_chop"].to_numpy(),
                         "bothchop": (base["eth_chop"] & base["btc_chop"]).to_numpy()}
                for gname, g in gates.items():
                    s = summarize(sequential_portfolio(base[g], prio), f"top{k}_{gname}")
                    s.update({"window": wn, "kind": lb}); rows.append(s)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "final_gated_ensemble.csv", index=False)
    pd.set_option("display.width", 240)

    for wn in ("VAL", "OOS"):
        log(f"\n=== {wn} (real) ===")
        t = df[(df.kind == "real") & (df.window == wn)].set_index("arm")
        t = t.assign(ret_per_dd=(t["total_bp"] / t["max_dd"].abs()).round(0))
        print(t[["n", "total_bp", "mean_bp", "pf", "profit_wr", "max_dd", "ret_per_dd"]].round(3).to_string())

    log("\n=== 방향뒤집기 대조 (양 창 모두 통과해야 ✅) ===")
    p = df.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    for a in p.index:
        gv = p.loc[a, ("VAL", "real")] - p.loc[a, ("VAL", "flip")]
        go = p.loc[a, ("OOS", "real")] - p.loc[a, ("OOS", "flip")]
        ok = gv > 0 and go > 0 and p.loc[a, ("VAL", "real")] > 0 and p.loc[a, ("OOS", "real")] > 0
        print(f"  {'✅' if ok else '❌'} {a:16s} VAL {p.loc[a,('VAL','real')]:+8.1f} (gap {gv:+8.1f}) | "
              f"OOS {p.loc[a,('OOS','real')]:+8.1f} (gap {go:+8.1f})")
    log(f"\nWrote {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
