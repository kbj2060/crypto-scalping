#!/usr/bin/env python3
"""ETH V_REBOUND 9-trigger feeder-role ranking, RE-DONE under held_up stratification.

Follow-up to the same-day held_up circularity proof
(research_eth_v_rebound_local_extreme_circularity_check_20260901.py): that script proved
local_extreme[i]=True mechanically implies low[i+1:i+W+1].min()>=low[i] ("held_up", W=6=FAST_BARS,
the exact window fast_move uses), and that this property ALONE inflates V_REBOUND label rates
4.2-4.8x regardless of which trigger fired. docs/homer/v_rebound_open_issues_20260901.md #4/#6 flags
as unresolved: "local_extreme=best trigger" and the kalman/demarker GAP=12 dedup net-new comparison
(eth_v_rebound_feeder_role_audit_and_dedup_fix_20260901) were both ranked using the pooled (unstrat-
ified) net_new_rate -- which is exactly the metric proven vulnerable to held_up. This script re-runs
both rankings SPLIT by held_up so they can be judged apples-to-apples instead.

local_extreme is CENTERED (backward_only AND forward_only/held_up) by construction, so 100% of its
candidates have held_up=True -- it structurally cannot appear in a held_up=False bucket. The fair
question is therefore not "is local_extreme's pooled rate highest" but "is local_extreme's held_up=
True rate still highest among the OTHER triggers' own held_up=True subsets" -- if yes, it retains
genuine edge beyond held_up; if it's now unremarkable, its rank was mostly the held_up artifact.

Two parts:
  PART A -- current LIVE 9-trigger settings (no dedup change), net-new-vs-other-8 audit
            (identical methodology to research_eth_v_rebound_multitrigger_feeder_role_screen_
            20260901.py::part1_role_audit), split by held_up True/False.
  PART B -- re-checks the ALREADY-CHOSEN kalman_deviation_meanrev/demarker_extreme GAP=12 dedup
            setting (eth_v_rebound_feeder_role_audit_and_dedup_fix_20260901's Part 2/3 result) under
            the same held_up split, to see whether the reported +62~82%/+49~80% net-new improvement
            survives once held_up is controlled for.

Screening-only: TRAIN+VAL population (timestamp < 2026-01-01), no TabPFN/GPU, no live code change,
OOS/HOLDOUT never touched. Self-checked against realized_outcome() row-by-row reference, verbatim
discipline from both parent scripts.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_feeder_role_held_up_stratified_20260901.py
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
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/feeder_role_held_up_stratified_report.json"

VAL_END = pd.Timestamp("2026-01-01", tz="UTC")  # TRAIN+VAL only; OOS/HOLDOUT never loaded
RANDOM_STATE = 20260901
W = 6  # LOCAL_EXTREME_W == FAST_BARS -- the exact overlap window proven in the circularity check

NAMED8 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
          "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ALL9 = NAMED8 + ["local_extreme"]

CONTINUOUS_COL = {"kalman_deviation_meanrev": "kalman_dev_z", "demarker_extreme": "dem"}
CURRENT_THRESHOLD = {"kalman_deviation_meanrev": 2.0, "demarker_extreme": 0.10}
DEDUP_GAP = 12  # value chosen in eth_v_rebound_feeder_role_audit_and_dedup_fix_20260901


def log(msg: str) -> None:
    print(f"[held_up_stratified_role_audit] {msg}", flush=True)


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_held_up_stratified_20260901", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ref():
    spec = importlib.util.spec_from_file_location("v_rebound_ref_held_up_stratified_20260901", REF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_min(arr: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = i + lo_off, i + hi_off
        if a < 0 or b >= n:
            continue
        out[i] = arr[a:b + 1].min()
    return out


def window_max(arr: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = i + lo_off, i + hi_off
        if a < 0 or b >= n:
            continue
        out[i] = arr[a:b + 1].max()
    return out


def build_base():
    eth = load_klines(ETH_CSV)
    btc = load_klines(BTC_CSV)
    log(f"ETH {len(eth)} bars {eth['timestamp'].iloc[0]} .. {eth['timestamp'].iloc[-1]}")

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

    log("computing held_up (forward_only) arrays -- verbatim mechanics from circularity_check...")
    fwd_low_min = window_min(low, 1, W)
    fwd_high_max = window_max(high, 1, W)
    held_up_bottom = fwd_low_min >= low
    held_up_top = fwd_high_max <= high
    valid_fwd_bottom = ~np.isnan(fwd_low_min)
    valid_fwd_top = ~np.isnan(fwd_high_max)

    # self-check: local_extreme (centered) must be a strict subset of held_up on its own side --
    # if this ever fails, the held_up computation here has diverged from local_extreme's own.
    le_subset_bottom = int((local_low & ~held_up_bottom).sum())
    le_subset_top = int((local_high & ~held_up_top).sum())
    log(f"  self-check: local_extreme candidates NOT satisfying held_up -- bottom={le_subset_bottom}, top={le_subset_top} (must be 0)")

    return sig, fields, held_up_bottom, held_up_top, valid_fwd_bottom, valid_fwd_top, (le_subset_bottom, le_subset_top)


def status_array(fields: pd.DataFrame, is_down: bool) -> np.ndarray:
    lab = label_side(fields, is_down=is_down)
    valid, label_raw = lab["valid"].to_numpy(), lab["label_raw"].to_numpy()
    return np.where(~valid, "invalid",
             np.where(label_raw == 1, "v_rebound", np.where(label_raw == 0, "chop", "ambiguous")))


def rate_of_fired(status: np.ndarray, fired_mask: np.ndarray) -> dict:
    pool = status[fired_mask]
    denom = int((pool != "invalid").sum())
    n_v = int((pool == "v_rebound").sum())
    return {"n_fired": int(fired_mask.sum()), "n_labeled_denom": denom, "n_v_rebound": n_v,
            "rate": round(n_v / denom, 4) if denom else None}


def self_check(sig: pd.DataFrame, status_down: np.ndarray, status_up: np.ndarray, n_sample: int = 500) -> dict:
    ref = load_ref()
    rng = np.random.default_rng(RANDOM_STATE)
    mismatches = []
    n_checked = 0
    for is_down, status in ((True, status_down), (False, status_up)):
        checkable = np.flatnonzero(status != "invalid")
        checkable = checkable[checkable > 0]
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


def other8_fixed_masks(sig: pd.DataFrame, is_down: bool, exclude: str) -> np.ndarray:
    prefix = "bottom" if is_down else "top"
    others = [n for n in ALL9 if n != exclude]
    M = np.zeros(len(sig), dtype=bool)
    for name in others:
        M |= sig[f"{prefix}_{name}"].fillna(False).to_numpy()
    return M


def part_a_current_live_stratified(sig, status_down, status_up, held_up_bottom, held_up_top,
                                    valid_fwd_bottom, valid_fwd_top) -> dict:
    out = {"bottom": {}, "top": {}}
    for side, status, held_up, valid_fwd in (
        ("bottom", status_down, held_up_bottom, valid_fwd_bottom),
        ("top", status_up, held_up_top, valid_fwd_top),
    ):
        M = {name: sig[f"{side}_{name}"].fillna(False).to_numpy() for name in ALL9}
        stacked = np.stack([M[name] for name in ALL9], axis=1)
        n_fired_row = stacked.sum(axis=1)
        for name in ALL9:
            net_new_mask = M[name] & (n_fired_row == 1) & valid_fwd
            hu_true = rate_of_fired(status, net_new_mask & held_up)
            hu_false = rate_of_fired(status, net_new_mask & ~held_up)
            n_tot = hu_true["n_fired"] + hu_false["n_fired"]
            out[side][name] = {
                "net_new_held_up_true": hu_true,
                "net_new_held_up_false": hu_false,
                "held_up_incidence_within_net_new": round(hu_true["n_fired"] / n_tot, 4) if n_tot else None,
            }

    combined = {}
    for name in ALL9:
        b, t = out["bottom"][name], out["top"][name]
        for key in ("net_new_held_up_true", "net_new_held_up_false"):
            bn, bv = b[key]["n_labeled_denom"], b[key]["n_v_rebound"]
            tn, tv = t[key]["n_labeled_denom"], t[key]["n_v_rebound"]
            combined.setdefault(name, {})[key] = {
                "n_labeled_denom": bn + tn, "n_v_rebound": bv + tv,
                "rate": round((bv + tv) / (bn + tn), 4) if (bn + tn) else None,
            }
    out["combined"] = combined
    return out


def part_b_gap12_dedup_stratified(sig, status_down, status_up, held_up_bottom, held_up_top,
                                   valid_fwd_bottom, valid_fwd_top) -> dict:
    out = {}
    for name in ("kalman_deviation_meanrev", "demarker_extreme"):
        col = sig[CONTINUOUS_COL[name]].to_numpy()
        cutoff = CURRENT_THRESHOLD[name]
        finite = np.isfinite(col)
        name_out = {}
        for side in ("bottom", "top"):
            status = status_down if side == "bottom" else status_up
            held_up = held_up_bottom if side == "bottom" else held_up_top
            valid_fwd = valid_fwd_bottom if side == "bottom" else valid_fwd_top
            others_fixed = other8_fixed_masks(sig, is_down=(side == "bottom"), exclude=name)

            if name == "demarker_extreme":
                raw = (col <= cutoff) if side == "bottom" else (col >= (1.0 - cutoff))
            else:
                raw = (col <= -cutoff) if side == "bottom" else (col >= cutoff)
            raw = raw & finite
            idx_raw = np.flatnonzero(raw)
            most_negative = (side == "bottom")
            kept_idx = (cluster_dedup(idx_raw, col[idx_raw], most_negative=most_negative, gap=DEDUP_GAP)
                        if len(idx_raw) else idx_raw)
            deduped = np.zeros(len(sig), dtype=bool)
            deduped[kept_idx] = True

            net_new_mask = deduped & ~others_fixed & valid_fwd
            hu_true = rate_of_fired(status, net_new_mask & held_up)
            hu_false = rate_of_fired(status, net_new_mask & ~held_up)
            n_tot = hu_true["n_fired"] + hu_false["n_fired"]
            name_out[side] = {
                "n_fired_post_dedup": int(deduped.sum()),
                "net_new_held_up_true": hu_true,
                "net_new_held_up_false": hu_false,
                "held_up_incidence_within_net_new": round(hu_true["n_fired"] / n_tot, 4) if n_tot else None,
            }
        out[name] = name_out
    return out


def main() -> int:
    t0 = time.time()
    sig, fields, held_up_bottom, held_up_top, valid_fwd_bottom, valid_fwd_top, le_subset = build_base()

    status_down = status_array(fields, is_down=True)
    status_up = status_array(fields, is_down=False)

    log("self-check vs realized_outcome() row-by-row reference (500/side random sample)...")
    sc = self_check(sig, status_down, status_up, n_sample=500)
    log(f"  self-check: {sc['n_checked']} checked, {sc['n_mismatches']} mismatches")
    if sc["n_mismatches"]:
        log(f"  MISMATCH SAMPLE: {json.dumps(sc['mismatches_sample'][:5], default=str, ensure_ascii=False)}")

    log("=== PART A: current live 9-trigger net-new audit, split by held_up ===")
    part_a = part_a_current_live_stratified(sig, status_down, status_up, held_up_bottom, held_up_top,
                                             valid_fwd_bottom, valid_fwd_top)
    for name in ALL9:
        c = part_a["combined"][name]
        t_rate = c["net_new_held_up_true"]["rate"]
        t_n = c["net_new_held_up_true"]["n_labeled_denom"]
        f_rate = c["net_new_held_up_false"]["rate"]
        f_n = c["net_new_held_up_false"]["n_labeled_denom"]
        log(f"  {name:24s} held_up=True  rate={t_rate} (n={t_n:6d})   held_up=False rate={f_rate} (n={f_n:6d})")

    log("=== ranking check: among held_up=True subsets ONLY, where does local_extreme fall? ===")
    ranked = sorted(
        ((name, part_a["combined"][name]["net_new_held_up_true"]["rate"],
          part_a["combined"][name]["net_new_held_up_true"]["n_labeled_denom"])
         for name in ALL9 if part_a["combined"][name]["net_new_held_up_true"]["rate"] is not None),
        key=lambda x: -x[1],
    )
    for rank, (name, rt, n) in enumerate(ranked, 1):
        log(f"  #{rank} {name:24s} held_up=True net_new_rate={rt} (n={n})")

    log("=== PART B: kalman/demarker GAP=12 dedup re-check, split by held_up ===")
    part_b = part_b_gap12_dedup_stratified(sig, status_down, status_up, held_up_bottom, held_up_top,
                                            valid_fwd_bottom, valid_fwd_top)
    for name, sides in part_b.items():
        for side, r in sides.items():
            t, f = r["net_new_held_up_true"], r["net_new_held_up_false"]
            log(f"  {name}/{side}: held_up=True rate={t['rate']}(n={t['n_labeled_denom']})  "
                f"held_up=False rate={f['rate']}(n={f['n_labeled_denom']})  "
                f"incidence={r['held_up_incidence_within_net_new']}")

    report = {
        "signal": "v_rebound_feeder_role_held_up_stratified", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "tabpfn_training_done": False, "economic_cost_gate_done": False,
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "population": "TRAIN+VAL only (timestamp < 2026-01-01)",
            "purpose": ("Re-rank the 9-trigger pool's net-new contribution SPLIT by held_up, "
                        "since docs/homer/v_rebound_open_issues_20260901.md #4/#6 flagged the "
                        "unstratified ranking (local_extreme=best, kalman/demarker GAP=12 dedup "
                        "improvement) as potentially confounded by held_up."),
        },
        "self_check_vs_reference_impl": sc,
        "local_extreme_centered_subset_of_held_up_self_check": {
            "bottom_violations": le_subset[0], "top_violations": le_subset[1],
            "note": "must be 0 -- local_extreme is CENTERED (backward AND forward), a strict subset of held_up (forward-only) by construction",
        },
        "part_a_current_live_held_up_stratified": part_a,
        "part_a_held_up_true_ranking": [{"rank": i + 1, "name": n, "rate": r, "n": nn} for i, (n, r, nn) in enumerate(ranked)],
        "part_b_gap12_dedup_held_up_stratified": part_b,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
