#!/usr/bin/env python3
"""ETH V_REBOUND 9-trigger candidate pool -- feeder-ROLE audit + GAP/threshold re-screen.

Follow-up to feedback_signal_feeder_role_needs_own_grid_screen_20260901 memory: a signal's
own-task-calibrated trigger threshold (and, for continuous/z-score signals, its dedup GAP) must
NOT be assumed good when that signal is reused as a candidate-pool feeder for a DIFFERENT
downstream objective (here: V_REBOUND's giveback/persistence label). BTC's analogous 8-trigger
widening round found exactly this for kalman_deviation_meanrev (its TOP side got monotonically
WORSE as the z-cutoff tightened 2.0->4.0) while demarker_extreme's net-new contribution held up.
That BTC screen was never re-run for ETH's LIVE 9-trigger pool -- this script does that, for ALL 4
of ETH's continuous/threshold-based signals (kalman_deviation_meanrev, demarker_extreme,
taker_delta_z_climax, short_term_return_z), not just kalman, per this round's broader "are all 9
triggers' roles still correct" request. The other 5 triggers (liquidity_sweep, orthogonal_combo,
smt_divergence, fib_extension_exhaustion, local_extreme) are discrete/event-type (no single scalar
threshold to sweep) -- Part 1's net-new audit still covers them, just not Part 2/3's grid sweep.

**Part 0 -- discovered while reading the live code (not previously documented)**: neither
build_eth_5m_v_rebound_multitrigger_labels_20260831.py (the label builder) nor
live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() (the live server) apply ANY
cluster-dedup to kalman_deviation_meanrev or demarker_extreme before OR-ing them into the 9-trigger
union -- both use compute_signals()'s raw per-bar bottom_X/top_X columns directly. This is exactly
BTC's REJECTED-round pattern (raw union of state-type indicators that can stay pinned past their
threshold for many consecutive bars), never fixed on the ETH side. Part 1 below quantifies how much
this actually costs today; Part 2/3 re-screen GAP and threshold together to find a better setting.

**Part 1 -- net-new role-fit audit (all 9 triggers, CURRENT live settings)**: a trigger's published
"of_total_fired" rate (docs/homer/README.md; also embedded in
data/labels/eth_5m_v_rebound_multitrigger_20260831/report.json) rides on candidates other triggers
would have caught anyway. The question this project actually cares about -- "is this signal's ROLE
in the ensemble correct" -- is answered by its NET-NEW contribution: candidates where THIS trigger
fired and NONE of the other 8 did. A trigger with a healthy net-new count and a net-new rate at or
above the pool baseline is doing real, non-redundant work; one with net-new rate far below baseline
is mostly dead weight riding on overlap.

**Part 2/3 -- GAP-dedup x threshold-strictness grid, net-new-only, CPU-only (no TabPFN/GPU)**:
reuses compute_outcome_fields()/label_side() VERBATIM from research_btc_v_rebound_gridscreen_
20260901.py (asset-agnostic vectorized port of realized_outcome(), already self-check-validated
there against the row-by-row reference) and cluster_dedup() VERBATIM from research_btc_demarker_
extreme_metalabel_tabpfn_20260901.py. For each of the 4 continuous signals x side x GAP x threshold,
recomputes the net-new-vs-other-8(current) population and its V_REBOUND label rate. Other 8
triggers are held at their CURRENTLY DEPLOYED settings throughout (one-factor-at-a-time sweep,
matching BTC's own methodology) -- this is a diagnostic screen, not a joint re-optimization of the
whole 9-trigger pool at once.

Fresh-forward discipline: truncated to timestamp < 2026-01-01 (TRAIN+VAL only, OOS/HOLDOUT never
loaded into any grid-search decision) -- same convention as research_btc_v_rebound_gridscreen_
20260901.py, so any future re-validation of a chosen setting still has clean OOS/HOLDOUT. This is a
SCREENING pass only: no TabPFN retraining, no live code change, no economic/cost-gate backtest.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_multitrigger_feeder_role_screen_20260901.py
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

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_btc_v_rebound_gridscreen_20260901 import compute_outcome_fields, label_side  # noqa: E402
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402

ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
REF_SCRIPT = ROOT / "scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py"
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/feeder_role_screen_report.json"

VAL_END = pd.Timestamp("2026-01-01", tz="UTC")  # == OOS start; never loaded past this point
RANDOM_STATE = 20260901

NAMED8 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
          "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ALL9 = NAMED8 + ["local_extreme"]
LOCAL_EXTREME_W = 6

# current live thresholds (compute_signals(), verbatim) -- used as the "held fixed" background
# when re-screening one signal at a time, and as GAP=0/native-threshold anchor in each grid.
CURRENT_THRESHOLD = {
    "kalman_deviation_meanrev": 2.0, "demarker_extreme": 0.10,
    "taker_delta_z_climax": 2.0, "short_term_return_z": 2.5,
}
CONTINUOUS_COL = {
    "kalman_deviation_meanrev": "kalman_dev_z", "demarker_extreme": "dem",
    "taker_delta_z_climax": "delta_z", "short_term_return_z": "ret3_z",
}
# grid of bottom-side cutoffs (top mirrors: same distance from the neutral point, opposite sign for
# z-scores; 1-x for demarker's [0,1] scale). z-signal grid matches BTC's own kalman sweep (2.0->4.0)
# exactly, for direct cross-asset comparability of the finding.
THRESHOLD_GRID_BOTTOM = {
    "kalman_deviation_meanrev": [2.0, 2.5, 3.0, 3.5, 4.0],
    "taker_delta_z_climax": [2.0, 2.5, 3.0, 3.5, 4.0],
    "short_term_return_z": [2.0, 2.5, 3.0, 3.5, 4.0],
    "demarker_extreme": [0.10, 0.075, 0.05, 0.025, 0.01],
}
GAP_GRID = [0, 6, 12, 24, 48, 96]  # 0 == no dedup, reproduces today's live raw behavior exactly
MIN_NET_NEW_N = 50  # floor below which a grid point's rate is noise, not a candidate recommendation


def log(msg: str) -> None:
    print(f"[feeder_role_screen] {msg}", flush=True)


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_feeder_screen_20260901", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ref():
    spec = importlib.util.spec_from_file_location("v_rebound_ref_feeder_screen_20260901", REF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sig, fields): sig = compute_signals() output (all 9 triggers' CURRENT raw bottom_X/
    top_X booleans + continuous kalman_dev_z/dem/delta_z/ret3_z + atr from add_causal_columns),
    truncated to TRAIN+VAL (timestamp < VAL_END); fields = compute_outcome_fields(sig), the
    per-bar forward-window arithmetic label_side() consumes."""
    eth = load_klines(ETH_CSV)
    btc = load_klines(BTC_CSV)
    log(f"ETH {len(eth)} bars {eth['timestamp'].iloc[0]} .. {eth['timestamp'].iloc[-1]}")
    log(f"BTC {len(btc)} bars {btc['timestamp'].iloc[0]} .. {btc['timestamp'].iloc[-1]}")

    impl = load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()

    n_before = len(sig)
    sig = sig.loc[sig["timestamp"] < VAL_END].reset_index(drop=True)
    log(f"truncated to TRAIN+VAL (< {VAL_END}): {n_before} -> {len(sig)} rows "
        f"({sig['timestamp'].iloc[-1]} last bar; OOS/HOLDOUT never loaded)")

    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    W = LOCAL_EXTREME_W
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        seg_lo, seg_hi = low[i - W:i + W + 1], high[i - W:i + W + 1]
        if low[i] == seg_lo.min():
            local_low[i] = True
        if high[i] == seg_hi.max():
            local_high[i] = True
    sig["bottom_local_extreme"] = local_low
    sig["top_local_extreme"] = local_high

    fields = compute_outcome_fields(sig)
    return sig, fields


def status_array(fields: pd.DataFrame, is_down: bool) -> np.ndarray:
    lab = label_side(fields, is_down=is_down)
    valid, label_raw = lab["valid"].to_numpy(), lab["label_raw"].to_numpy()
    return np.where(~valid, "invalid",
             np.where(label_raw == 1, "v_rebound", np.where(label_raw == 0, "chop", "ambiguous")))


def self_check(sig: pd.DataFrame, status_down: np.ndarray, status_up: np.ndarray, n_sample: int = 500) -> dict:
    """Cross-validates compute_outcome_fields()+label_side() (imported verbatim from the BTC
    gridscreen, itself already self-check-validated there) against realized_outcome()'s own
    row-by-row reference implementation, called directly on THIS sig frame -- same discipline as
    research_btc_v_rebound_gridscreen_20260901.py::self_check, ported to check both sides here."""
    ref = load_ref()
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(sig)
    mismatches = []
    n_checked = 0
    for is_down, status in ((True, status_down), (False, status_up)):
        checkable = np.flatnonzero(status != "invalid")
        checkable = checkable[checkable > 0]  # ref fn does atr.iloc[idx-1], wraps at idx=0
        sample = rng.choice(checkable, size=min(n_sample, len(checkable)), replace=False)
        label_map = {"V자반등": "v_rebound", "지지/횡보": "chop", "애매(제외권)": "ambiguous"}
        for idx in sample:
            ref_out = ref.realized_outcome(sig, int(idx), is_down)
            n_checked += 1
            if ref_out is None or ref_out["partial_window"]:
                if status[idx] != "invalid":
                    mismatches.append({"idx": int(idx), "is_down": is_down, "reason": "ref_none_or_partial_but_mine_valid"})
                continue
            ref_status = label_map[ref_out["outcome"]]
            if ref_status != status[idx]:
                mismatches.append({"idx": int(idx), "is_down": is_down, "mine": status[idx], "ref": ref_status})
    return {"n_checked": n_checked, "n_mismatches": len(mismatches), "mismatches_sample": mismatches[:10]}


def build_trigger_matrix(sig: pd.DataFrame, is_down: bool) -> dict[str, np.ndarray]:
    prefix = "bottom" if is_down else "top"
    return {name: sig[f"{prefix}_{name}"].fillna(False).to_numpy() for name in ALL9}


def rate_of_fired(status: np.ndarray, fired_mask: np.ndarray) -> dict:
    pool = status[fired_mask]
    denom = int((pool != "invalid").sum())
    n_v = int((pool == "v_rebound").sum())
    return {"n_fired": int(fired_mask.sum()), "n_labeled_denom": denom, "n_v_rebound": n_v,
            "rate": round(n_v / denom, 4) if denom else None}


def part1_role_audit(sig: pd.DataFrame, status_down: np.ndarray, status_up: np.ndarray) -> dict:
    """Net-new contribution of each of the 9 triggers, both sides + combined, against the OTHER 8
    triggers' CURRENT live settings -- answers "is this signal's role in the ensemble correct"
    directly, independent of any grid sweep below."""
    out = {}
    for side, status in (("bottom", status_down), ("top", status_up)):
        M = build_trigger_matrix(sig, is_down=(side == "bottom"))
        stacked = np.stack([M[name] for name in ALL9], axis=1)
        n_fired_row = stacked.sum(axis=1)
        side_out = {}
        for j, name in enumerate(ALL9):
            total = rate_of_fired(status, M[name])
            net_new_mask = M[name] & (n_fired_row == 1)
            net_new = rate_of_fired(status, net_new_mask)
            side_out[name] = {
                "total_fired": total, "net_new": net_new,
                "overlap_rate": round(1 - net_new["n_fired"] / total["n_fired"], 4) if total["n_fired"] else None,
            }
        out[side] = side_out

    combined = {}
    for name in ALL9:
        b, t = out["bottom"][name], out["top"][name]
        tot_n = b["total_fired"]["n_labeled_denom"] + t["total_fired"]["n_labeled_denom"]
        tot_v = b["total_fired"]["n_v_rebound"] + t["total_fired"]["n_v_rebound"]
        nn_n = b["net_new"]["n_labeled_denom"] + t["net_new"]["n_labeled_denom"]
        nn_v = b["net_new"]["n_v_rebound"] + t["net_new"]["n_v_rebound"]
        combined[name] = {
            "total_fired_rate": round(tot_v / tot_n, 4) if tot_n else None,
            "total_fired_n": b["total_fired"]["n_fired"] + t["total_fired"]["n_fired"],
            "net_new_rate": round(nn_v / nn_n, 4) if nn_n else None,
            "net_new_n": b["net_new"]["n_fired"] + t["net_new"]["n_fired"],
            "overlap_rate_combined": round(1 - (b["net_new"]["n_fired"] + t["net_new"]["n_fired"]) /
                                            (b["total_fired"]["n_fired"] + t["total_fired"]["n_fired"]), 4),
        }
    out["combined"] = combined
    pool_baseline_n = int(((status_down == "v_rebound") | (status_down == "chop")).sum() +
                           ((status_up == "v_rebound") | (status_up == "chop")).sum())
    pool_baseline_v = int((status_down == "v_rebound").sum() + (status_up == "v_rebound").sum())
    out["pool_baseline_rate"] = round(pool_baseline_v / pool_baseline_n, 4)
    return out


def other8_fixed_masks(sig: pd.DataFrame, is_down: bool, exclude: str) -> np.ndarray:
    prefix = "bottom" if is_down else "top"
    others = [n for n in ALL9 if n != exclude]
    M = np.zeros(len(sig), dtype=bool)
    for name in others:
        M |= sig[f"{prefix}_{name}"].fillna(False).to_numpy()
    return M


def threshold_bool(sig: pd.DataFrame, name: str, side: str, cutoff: float) -> np.ndarray:
    col = sig[CONTINUOUS_COL[name]].to_numpy()
    if name == "demarker_extreme":
        return (col <= cutoff) if side == "bottom" else (col >= (1.0 - cutoff))
    return (col <= -cutoff) if side == "bottom" else (col >= cutoff)


def part23_grid_screen(sig: pd.DataFrame, status_down: np.ndarray, status_up: np.ndarray) -> dict:
    out = {}
    for name in CONTINUOUS_COL:
        col = sig[CONTINUOUS_COL[name]].to_numpy()
        finite = np.isfinite(col)
        name_out = {}
        for side in ("bottom", "top"):
            status = status_down if side == "bottom" else status_up
            others_fixed = other8_fixed_masks(sig, is_down=(side == "bottom"), exclude=name)
            extremeness = col
            most_negative = (side == "bottom") if name != "demarker_extreme" else (side == "bottom")
            grid_rows = []
            for cutoff in THRESHOLD_GRID_BOTTOM[name]:
                raw = threshold_bool(sig, name, side, cutoff) & finite
                idx_raw = np.flatnonzero(raw)
                for gap in GAP_GRID:
                    if gap == 0 or len(idx_raw) == 0:
                        kept_idx = idx_raw
                    else:
                        kept_idx = cluster_dedup(idx_raw, extremeness[idx_raw], most_negative=most_negative, gap=gap)
                    deduped = np.zeros(len(sig), dtype=bool)
                    deduped[kept_idx] = True
                    net_new_mask = deduped & ~others_fixed
                    r = rate_of_fired(status, net_new_mask)
                    grid_rows.append({
                        "cutoff": cutoff, "gap": gap,
                        "n_fired_post_dedup": int(deduped.sum()),
                        "n_net_new": r["n_fired"], "n_labeled_denom": r["n_labeled_denom"],
                        "n_v_rebound": r["n_v_rebound"], "net_new_rate": r["rate"],
                    })
            name_out[side] = grid_rows
        out[name] = name_out
    return out


def summarize_recommendations(grid: dict, audit: dict) -> dict:
    """For each continuous signal x side, picks the grid point maximizing net_new_rate among points
    with n_net_new >= MIN_NET_NEW_N, and compares it to the CURRENT setting (cutoff=native, gap=0)
    found in the same grid (both computed identically -> apples-to-apples, no external benchmark
    population mismatch)."""
    rec = {}
    for name, sides in grid.items():
        rec[name] = {}
        for side, rows in sides.items():
            current_cutoff = CURRENT_THRESHOLD[name]
            current_row = next(r for r in rows if r["cutoff"] == current_cutoff and r["gap"] == 0)
            eligible = [r for r in rows if r["n_net_new"] >= MIN_NET_NEW_N]
            best = max(eligible, key=lambda r: (r["net_new_rate"] if r["net_new_rate"] is not None else -1)) if eligible else None
            rec[name][side] = {
                "current": current_row,
                "best_eligible": best,
                "improves_over_current": (
                    best is not None and current_row["net_new_rate"] is not None and best["net_new_rate"] is not None
                    and best["net_new_rate"] > current_row["net_new_rate"]
                    and not (best["cutoff"] == current_row["cutoff"] and best["gap"] == current_row["gap"])
                ),
            }
    return rec


def main() -> int:
    t0 = time.time()
    sig, fields = build_base()

    log("computing V_REBOUND status (bottom/top) via compute_outcome_fields+label_side (verbatim BTC port)...")
    status_down = status_array(fields, is_down=True)
    status_up = status_array(fields, is_down=False)
    log(f"  bottom: v_rebound={int((status_down=='v_rebound').sum())} chop={int((status_down=='chop').sum())} "
        f"ambiguous={int((status_down=='ambiguous').sum())} invalid={int((status_down=='invalid').sum())}")
    log(f"  top:    v_rebound={int((status_up=='v_rebound').sum())} chop={int((status_up=='chop').sum())} "
        f"ambiguous={int((status_up=='ambiguous').sum())} invalid={int((status_up=='invalid').sum())}")

    log("self-check vs realized_outcome() row-by-row reference (500/side random sample)...")
    sc = self_check(sig, status_down, status_up, n_sample=500)
    log(f"  self-check: {sc['n_checked']} checked, {sc['n_mismatches']} mismatches")
    if sc["n_mismatches"]:
        log(f"  MISMATCH SAMPLE: {json.dumps(sc['mismatches_sample'][:5], default=str, ensure_ascii=False)}")

    log("=== PART 1: net-new role-fit audit, all 9 triggers, current live settings ===")
    audit = part1_role_audit(sig, status_down, status_up)
    log(f"  pool baseline v_rebound rate (of labeled): {audit['pool_baseline_rate']}")
    for name in ALL9:
        c = audit["combined"][name]
        log(f"  {name:24s} total_fired_rate={c['total_fired_rate']} (n={c['total_fired_n']:6d})  "
            f"net_new_rate={c['net_new_rate']} (n={c['net_new_n']:6d})  overlap={c['overlap_rate_combined']}")

    log("=== PART 2/3: GAP x threshold grid screen, 4 continuous signals, net-new-only ===")
    grid = part23_grid_screen(sig, status_down, status_up)
    rec = summarize_recommendations(grid, audit)
    for name, sides in rec.items():
        for side, r in sides.items():
            cur, best = r["current"], r["best_eligible"]
            log(f"  {name}/{side}: current(gap=0,cutoff={cur['cutoff']}) rate={cur['net_new_rate']} n={cur['n_net_new']}  "
                f"| best(gap={best['gap'] if best else None},cutoff={best['cutoff'] if best else None}) "
                f"rate={best['net_new_rate'] if best else None} n={best['n_net_new'] if best else None}  "
                f"IMPROVES={r['improves_over_current']}")

    report = {
        "signal": "v_rebound", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_and_role_audit_only": True, "tabpfn_training_done": False,
            "economic_cost_gate_done": False, "live_code_changed": False,
            "holdout_touched": False, "oos_touched": False,
            "raw_frame_truncated_at": str(VAL_END),
            "population": "TRAIN+VAL only (timestamp < 2026-01-01)",
        },
        "part0_discovered_raw_undeduped_pattern": {
            "note": ("Neither build_eth_5m_v_rebound_multitrigger_labels_20260831.py nor "
                     "live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() apply any "
                     "cluster-dedup to kalman_deviation_meanrev or demarker_extreme before "
                     "OR-ing into the 9-trigger union -- both consume compute_signals()'s raw "
                     "per-bar booleans directly. Matches BTC's REJECTED raw-8-trigger pattern, "
                     "never fixed on the ETH side."),
        },
        "self_check_vs_reference_impl": sc,
        "part1_net_new_role_audit": audit,
        "part23_gap_threshold_grid": grid,
        "recommendations": rec,
        "min_net_new_n_floor": MIN_NET_NEW_N,
        "threshold_grids": THRESHOLD_GRID_BOTTOM,
        "gap_grid": GAP_GRID,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
