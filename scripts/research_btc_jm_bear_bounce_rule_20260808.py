"""JM-bear contrarian bounce RULE line, BTC 5m (2026-08-08).

Mechanism basis (measured on TRAIN in the 2026-08-08 JM detector work): bars the causal
Jump Model (k3 lam32) labels bear show fwd-24h return +0.52% -- a contrarian bounce
pattern.  The ML regime-expert axis is closed (D2 gate OOS -19.5%; JM/czz MoE failed VAL
gates outright); this line tests the residual hypothesis with a FIXED RULE instead of a
trained model: hypothesis class = 6 pre-registered variants, no features, no fitting.

Rules (all long-only; exits via the corrected TB label tp/sl moves at the entry bar +
288-bar horizon through the causal replay -- identical cost model to every 2026-08 line):
  bear_entry     enter at the first bar of each causal JM-bear run
  bear_period    enter every 288 bars while JM-bear persists
  bear_entry_czz / bear_period_czz   same but additionally require czz4 bear wave
  bull_entry     control: first bar of each JM-bull run (momentum analog, fwd24h +0.17%)
  bull_period    control: every 288 bars while JM-bull persists
Selection: VAL only (n_trades>=15, pnl>0, >=3/4 positive months); best VAL PnL earns ONE
OOS read; adopt iff OOS pnl>0 AND >=2/3 OOS months positive.  Fresh-forward flags as usual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_btc_regime_conditioned_entry_20260808 import load_all  # noqa: E402
from train_eval_btc_jm_regime_moe_20260808 import load_jm  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import replay  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_jm_bear_bounce_20260808"
PERIOD = 288


def rule_side_state(name: str, jm: dict, n: int) -> np.ndarray:
    reg = jm["jm_lam32"]
    czz = jm["czz4"]
    side = np.zeros(n, dtype=np.int64)
    target = 0 if name.startswith("bear") else 2
    in_run = reg == target
    run_entry = in_run & ~np.roll(in_run, 1)
    run_entry[0] = in_run[0]
    if "_entry" in name:
        side[run_entry] = 1
    else:
        bars_in = np.zeros(n, dtype=np.int64)
        cnt = 0
        for i in range(n):
            cnt = cnt + 1 if in_run[i] else 0
            bars_in[i] = cnt
        side[(bars_in > 0) & ((bars_in - 1) % PERIOD == 0)] = 1
    if name.endswith("_czz"):
        side[czz != 0] = np.where(czz[czz != 0] == 0, side[czz != 0], 0)
    return side


RULES = ["bear_entry", "bear_period", "bear_entry_czz", "bear_period_czz", "bull_entry", "bull_period"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    jm = load_jm(ts)
    months = ts.dt.to_period("M").astype(str).to_numpy()

    if args.stage == "val":
        v_idx = np.flatnonzero(val_mask)
        table = []
        for name in RULES:
            side = rule_side_state(name, jm, len(panel))
            ss = np.zeros(len(panel), dtype=np.int64)
            ss[v_idx] = side[v_idx]
            rres = replay(panel, ss, tp_moves, sl_moves, val_mask)
            mon = {m: replay(panel, ss, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                   for m in sorted(set(months[v_idx]))}
            rec = {"rule": name, **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                   "monthly": mon, "n_pos_months": int(sum(v_ > 0 for v_ in mon.values()))}
            table.append(rec)
            print(json.dumps({k: rec[k] for k in ("rule", "n_trades", "pnl_pct", "n_pos_months")}), flush=True)
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0 and r["n_pos_months"] >= 3]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"table": table, "selected": None if best is None else best["rule"],
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- VAL gate failed"}))
            return 1
        name = prior["selected"]
        o_idx = np.flatnonzero(oos_mask)
        side = rule_side_state(name, jm, len(panel))
        ss = np.zeros(len(panel), dtype=np.int64)
        ss[o_idx] = side[o_idx]
        rres = replay(panel, ss, tp_moves, sl_moves, oos_mask)
        mon = {m: replay(panel, ss, tp_moves, sl_moves, oos_mask & (months == m)).get("pnl_pct", 0.0)
               for m in sorted(set(months[o_idx]))}
        out = {"stage": "oos", "rule": name, **rres, "monthly": mon,
               "n_pos_months": int(sum(v_ > 0 for v_ in mon.values())),
               "adopted": bool((rres.get("pnl_pct") or 0) > 0 and sum(v_ > 0 for v_ in mon.values()) >= 2),
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
