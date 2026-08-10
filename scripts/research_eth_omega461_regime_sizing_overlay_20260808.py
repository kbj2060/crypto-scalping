"""TRANSFER test: BTC's adopted regime SIZING overlay applied to the live ETH Omega4.6.1
greedy-router stack (contract docs/experiments/eth_regime_sizing_overlay_transfer_20260808.json).

The map (czz_trend: margin_fraction *= {bear 0.5, chop 1.0, bull 1.5} at the ENTRY-SIGNAL bar)
has literal multipliers and no fitted parameters and was selected on BTC, so running it unchanged
on ETH is a genuine out-of-sample transfer test of the RULE.

Implementation notes that matter:
  * `greedy_replay` / `prepare_component` are IMPORTED UNMODIFIED from
    replay_omega4_6_1_greedy_router_20260706.py -- no copy, so model/artifact paths cannot drift
    and the identity arm is a true regression control.
  * The overlay is applied by RE-RUNNING the replay with a modified per-bar margin array, never by
    rescaling a saved ledger: ETH's exit head consumes notional/leverage in pos_values, so margin
    changes exits and therefore the ledger itself (this is the BTC provenance bug, avoided here).
  * Futures sizing contract untouched: notional = margin x leverage inside the replay; TP/SL stay
    price-move targets.

Arms: identity (regression gate) / czz_trend (only adoption candidate) / czz_contra (falsification
probe, never adoptable). One pass, no retuning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as gr  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import causal_zigzag_regime  # noqa: E402

ETH_PRICE_PANEL = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
OUT_DIR = ROOT / "tmp/eth_regime_sizing_overlay_20260808"
WIN_START, WIN_END = "2026-01-01", "2026-06-30"
BASELINE_EXPECTED = {"pnl": 77.11, "mdd": -15.48, "trades": 37}
REG_TOL_PNL, REG_TOL_MDD = 0.5, 0.5
MAPS = {
    "identity": None,
    "czz_trend": {0: 0.5, 1: 1.0, 2: 1.5},
    "czz_contra": {0: 1.5, 1: 1.0, 2: 0.5},
}
BLOCK_DAYS, N_BOOT, SEED = 21, 2000, 903174


def czz_state_for(frame: pd.DataFrame) -> np.ndarray:
    """causal 4% directional-change regime, computed on the FULL ETH price history so the state at
    the window start is not a warmup artifact, then merged onto the replay frame's bars."""
    hist = pd.read_csv(ETH_PRICE_PANEL, usecols=["timestamp", "close"], low_memory=False)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    cdir = causal_zigzag_regime(hist["close"].to_numpy(dtype=np.float64), 0.04)
    st = pd.DataFrame({"timestamp": hist["timestamp"],
                       "czz4": np.where(cdir > 0, 2, np.where(cdir < 0, 0, 1)).astype(np.int64)})
    ts = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
    merged = pd.merge_asof(pd.DataFrame({"timestamp": ts}), st, on="timestamp",
                           direction="backward", tolerance=pd.Timedelta("10min"))
    n_missing = int(merged["czz4"].isna().sum())
    if n_missing:
        print(json.dumps({"warn_regime_rows_unmatched": n_missing}), flush=True)
    return merged["czz4"].fillna(1).to_numpy(dtype=np.int64)


def metrics_from_returns(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "calmar": 0.0, "worst5": 0.0, "wr": 0.0}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    pnl = float((curve[-1] - 1.0) * 100.0)
    mdd = float(dd.min() * 100.0)
    return {"pnl": round(pnl, 2), "mdd": round(mdd, 2), "trades": int(len(returns)),
            "calmar": round(pnl / abs(mdd), 2) if mdd < -1e-9 else None,
            "worst5": round(float(np.sort(returns)[:5].sum() * 100.0), 2),
            "wr": round(float((returns > 0).mean()), 3)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE
    frame = retest.load_frame_current(WIN_START, WIN_END)
    fee, slip = omega._load_fee_slip()
    print(f"ETH frame {len(frame)} bars {frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)

    base_components = {}
    for name, cfg in retest.COMPONENTS.items():
        # prepare_component asserts exact timestamp equality; align the frozen per-bar parent
        # predictions to the requested window (same handling as the 20260808 rerun audit)
        pred_csv = gr.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        full = pd.read_csv(pred_csv)
        full["timestamp"] = pd.to_datetime(full["timestamp"])
        keep = full["timestamp"].isin(frame["timestamp"])
        if int(keep.sum()) != len(frame):
            raise RuntimeError(f"{name}: prediction rows covering window = {int(keep.sum())} != frame rows {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{name}_{cfg['q_tag']}.csv"
        full.loc[keep.to_numpy()].reset_index(drop=True).to_csv(tmp, index=False)
        base_components[name] = gr.prepare_component(frame, tmp, cfg, device)
        print(f"{name}: prepared", flush=True)

    czz = czz_state_for(frame)
    print(json.dumps({"czz_bar_occupancy": {k: round(float((czz == v).mean()), 3)
                                            for k, v in (("bear", 0), ("chop", 1), ("bull", 2))}}), flush=True)

    results, ledgers = {}, {}
    for arm, mult in MAPS.items():
        comps = {}
        for name, c in base_components.items():
            c2 = dict(c)
            if mult is not None:
                m = np.vectorize(mult.get)(czz).astype(np.float64)
                margin = np.asarray(c["margin"], dtype=np.float64).copy()
                c2["margin"] = margin * m[: len(margin)]
            comps[name] = c2
        _, ledger = gr.greedy_replay(frame, comps, fee=fee, slip=slip,
                                     cost_mult=retest.COST_MULT, device=device)
        ledger = ledger.copy()
        ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
        ledgers[arm] = ledger
        r = ledger["trade_return"].to_numpy()
        res = metrics_from_returns(r)
        if mult is not None and len(ledger):
            ent_i = ledger["entry_signal_i"].to_numpy(dtype=int)
            res["mean_applied_mult"] = round(float(np.mean([mult[int(czz[i])] for i in ent_i])), 3)
            res["entry_regime_counts"] = {k: int((czz[ent_i] == v).sum())
                                          for k, v in (("bear", 0), ("chop", 1), ("bull", 2))}
        res["monthly"] = {}
        for mth, grp in ledger.groupby(ledger["entry_timestamp_dt"].dt.to_period("M")):
            res["monthly"][str(mth)] = metrics_from_returns(grp["trade_return"].to_numpy())
        results[arm] = res
        print(json.dumps({"arm": arm, **{k: res[k] for k in ("pnl", "mdd", "trades", "calmar", "worst5")}}), flush=True)

    # gate 0 regression
    ident = results["identity"]
    reg_ok = (abs(ident["pnl"] - BASELINE_EXPECTED["pnl"]) <= REG_TOL_PNL
              and abs(ident["mdd"] - BASELINE_EXPECTED["mdd"]) <= REG_TOL_MDD
              and ident["trades"] == BASELINE_EXPECTED["trades"])

    # paired time-block bootstrap: same block draw for every arm
    rng = np.random.default_rng(SEED)
    t0, t1 = pd.Timestamp(WIN_START), pd.Timestamp(WIN_END)
    edges = pd.date_range(t0, t1 + pd.Timedelta(days=BLOCK_DAYS), freq=f"{BLOCK_DAYS}D")
    blocks = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    per_arm_blocks = {arm: [ledgers[arm].loc[(ledgers[arm]["entry_timestamp_dt"] >= a)
                                             & (ledgers[arm]["entry_timestamp_dt"] < b),
                                             "trade_return"].to_numpy() for a, b in blocks]
                      for arm in MAPS}
    boot = {arm: [] for arm in MAPS}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        for arm in MAPS:
            r = np.concatenate([per_arm_blocks[arm][j] for j in pick]) if len(blocks) else np.array([])
            boot[arm].append(metrics_from_returns(r)["mdd"])
    boot = {a: np.array(v) for a, v in boot.items()}
    p_better = {a: round(float(np.mean(boot[a] > boot["identity"])), 3) for a in MAPS if a != "identity"}

    def monthly_better(arm: str) -> int:
        cnt = 0
        for mth, mi in results["identity"]["monthly"].items():
            ma = results[arm]["monthly"].get(mth)
            if ma and ma["mdd"] > mi["mdd"]:
                cnt += 1
        return cnt

    cand = "czz_trend"
    gates = {
        "gate_0_regression": bool(reg_ok),
        "gate_1_mdd": bool(results[cand]["mdd"] > ident["mdd"]),
        "gate_2_consistency": bool(monthly_better(cand) >= 4),
        "gate_3_bootstrap": bool(p_better[cand] >= 0.70),
        "gate_4_pnl_guardrail": bool(results[cand]["pnl"] >= 0.60 * ident["pnl"]),
        "gate_5_not_starvation": bool(abs(results[cand]["trades"] - ident["trades"]) <= 0.20 * ident["trades"]),
        "gate_6_falsification": not (results["czz_contra"]["mdd"] > ident["mdd"] and p_better["czz_contra"] >= 0.70),
    }
    out = {"contract": "eth_regime_sizing_overlay_transfer_20260808",
           "window": [WIN_START, WIN_END], "results": results,
           "monthly_mdd_better_count": {a: monthly_better(a) for a in MAPS if a != "identity"},
           "paired_bootstrap_P_mdd_better": p_better,
           "gates": gates, "adopted": bool(all(gates.values()))}
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2))
    for arm in MAPS:
        ledgers[arm].to_csv(OUT_DIR / f"ledger_{arm}.csv", index=False)
    print(json.dumps({k: out[k] for k in ("monthly_mdd_better_count", "paired_bootstrap_P_mdd_better", "gates", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
