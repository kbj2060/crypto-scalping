#!/usr/bin/env python3
"""RESEARCH ONLY -- v3 candidate fix for the zig075 SHORT/sustained-uptrend veto's ranging-regime
misfire found in [[eth_zig075_short_veto_paired_onoff_causal_test_20260817]] (real_g0 lost
20.18pp to the veto in a genuine chop window 2025-05-12..07-07; N=20 random-direction paired
comparison confirmed it at t=-26.58, near-deterministic).

=== v2 attempt (research_eth_odyssey4_zig075_veto_breakout_confirm_fix_20260817.py, NOT this
file -- REJECTED, recorded for the record) ===
v2's companion condition was `close_t >= rolling_max(close, WEEK_BARS)_t` ("this bar sets a new
>=1-week high", zero new tunable parameters). It DID eliminate the ranging-window harm (real_g0
-17.59% -> +2.59%, identical to no-veto) but activation collapsed ~60x EVERYWHERE including 2025-Q3
(43.0% -> 0.70% of bars), which gutted the veto's own reason for existing: Q3 real_g0 PnL fell from
+20.17% (original veto) all the way back to -15.86% (= no veto at all, veto_bars 19->0). v2 doesn't
fix the veto, it just disables it almost everywhere -- rejected.

=== v3 (this file) design discipline (matched to the ORIGINAL detector's own documented
discipline in research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py's module
docstring Step 1-4 -- read in full before writing this, not re-derived from memory) ===
The original detector (score = rolling(2016,'1 week').mean(dual_momentum>0) > p90(2025-Q1+Q2-only))
earned CONFIRMED status specifically because it added ZERO new free parameters: WEEK_BARS=2016
reuses dual_momentum's OWN existing close.shift(2016) lookback (not invented), and
DETECTOR_PERCENTILE=0.90 is a round, unswept decile convention. v3 is held to the SAME bar, using
a SOFTER companion condition than v2's exact-new-high (which proved too strict): `close_t >=
rolling_mean(close, WEEK_BARS)_t`, i.e. "this bar's close is at or above its own trailing 1-week
average" -- reuses the SAME WEEK_BARS=2016 window already in the original detector (no new
lookback parameter) and needs no threshold of its own (no new tunable number, "mean" is
parameter-free the same way "max" was). Ad-hoc single-window sanity check (real_g0 only, not the
full pre-registered protocol below, done to decide whether v3 was worth a full run before spending
the compute): Q3 +20.17%->+21.41% (preserved, marginally better), VAL/OOS-Q1/OOS-Q2 unchanged,
ranging-harm window -17.59%->+0.97% (recovers ~19 of the 20.18pp harm). Promising enough to run
the full protocol below.

=== Verification protocol (pre-registered before reading any window's own N=20 random-seed
result -- the ad-hoc real_g0-only sanity check above was already seen, which this line's own
"read the result before deciding to proceed" discipline treats as a screening step, not the
confirmatory read; the N=20 paired numbers below are the first time those are examined) ===
- G0 consistency: the NEW detector's activation on 2025-Q1+Q2 (the calibration sample) must still
  correctly identify Q3 (2025's confirmed sustained-uptrend quarter, context tier, the veto's
  entire raison d'etre) as MORE active than Q1/Q2 -- if the fix kills Q3's activation, it defeats
  the veto's purpose and is rejected regardless of what it does to the ranging windows.
- Primary target: does the fix reduce/eliminate the -20.18pp real_g0 harm and the N=20 paired
  t=-26.58 harm in the ranging window 2025-05-12..07-07 (found AFTER this fix was designed --
  the fix's OWN parameters were fixed by the design-discipline paragraph above BEFORE this
  window's activation rate under the new detector was computed)?
- Secondary: does it preserve non-worse in VAL/OOS-Q1/OOS-Q2 (the original CONFIRMED gate) and in
  the other two ranging windows (2025-03-10..05-05, 2026-02-09..04-06)?
- Full N=20 paired WITH/WITHOUT-veto re-run (SAME seeds as
  research_eth_odyssey4_random_direction_veto_paired_onoff_20260817.py, for direct before/after
  comparison) using the NEW breakout-confirmed mask in place of the original mask.

Reuses (imports, never duplicates) every helper already built in this line. Only new code: the
breakout-confirmation mask computation (_rolling_close_high_confirmed, mirroring _rolling_dual_
momentum_score's exact read/sort/dedup discipline) and the AND-combination with the unmodified
original score/threshold.

fresh_forward_bar_by_bar=true (plain backward .rolling(), no negative shift -- close_t compared
only to close_{t-WEEK_BARS+1..t}). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. Does NOT touch
trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env. Does NOT
modify any imported module. No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_odyssey4_random_direction_risk_management_ablation_20260817 as abl  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_zig075_veto_v3_mean_confirm_fix_20260817"
N_SEEDS = 20
WEEK_BARS = guard.WEEK_BARS  # reused verbatim, not re-chosen (2016)

RANGING_CANDIDATES = [
    {"key": "ranging_2025_05_12_to_07_07", "start": "2025-05-12", "end": "2025-07-07",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2025_03_10_to_05_05", "start": "2025-03-10", "end": "2025-05-05",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2026_02_09_to_04_06", "start": "2026-02-09", "end": "2026-04-06",
     "oof": False, "split": "oos", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
]
DOWNTREND_WINDOW_KEYS = ("val", "oos_q1", "oos_q2")
CONTEXT_WINDOW_KEYS = ("2025q1", "2025q2", "2025q3")  # Q3 = the veto's original raison d'etre


def log(msg: str) -> None:
    print(msg, flush=True)


def _rolling_close_high_confirmed(base_csv: Path) -> pd.DataFrame:
    """v3: mirrors guard._rolling_dual_momentum_score's exact read/sort/dedup discipline. New
    column: True iff this bar's close is >= the trailing WEEK_BARS-bar rolling MEAN of close
    (softer than v2's exact-new-high, which proved too strict -- see module docstring) -- zero
    new tunable parameters, same window as the original detector. Function name kept as
    `_rolling_close_high_confirmed` / column name `breakout_confirmed` for drop-in compatibility
    with build_v2_mask_for_frame and every downstream caller in this line -- only the aggregation
    inside changed from .max() to .mean()."""
    frame = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="raise")
    rolling_mean = close.rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    frame["breakout_confirmed"] = close >= rolling_mean
    return frame[["timestamp", "breakout_confirmed"]]


def build_v2_mask_for_frame(aligned_frame: pd.DataFrame, window_name: str, score_by_base: dict,
                             breakout_by_base: dict, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask_v1_original, mask_v2_breakout_confirmed) for the same aligned_frame."""
    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    score = score_by_base[base_csv]
    breakout = breakout_by_base[base_csv]
    merged = aligned_frame[["timestamp"]].merge(score, on="timestamp", how="left").merge(breakout, on="timestamp", how="left")
    if len(merged) != len(aligned_frame):
        raise RuntimeError(f"{window_name}: mask merge failed (row count mismatch)")
    raw_score = merged["sustained_uptrend_score"]
    mask_v1 = (raw_score > threshold).fillna(False).to_numpy(dtype=bool)
    breakout_flag = merged["breakout_confirmed"].fillna(False).to_numpy(dtype=bool)
    mask_v2 = mask_v1 & breakout_flag
    return mask_v1, mask_v2


def load_custom_window(cand: dict) -> dict[str, Any]:
    frame = sweep.load_frame(cand["start"], cand["end"], base_csv=cand["base_csv"], wide24_csv=cand["wide24_csv"])
    frame, n_dropped = gate._drop_route_nan(frame)
    gate.WINDOW_DEFS[cand["key"]] = {"split": cand["split"], "base_csv": cand["base_csv"], "wide24_csv": cand["wide24_csv"]}
    return {"frame": frame, "oof": cand["oof"], "tier": "ranging_retest", "route_nan_dropped": n_dropped}


def paired_v1_v2(aligned_frame: pd.DataFrame, components: dict, mask_v1: np.ndarray, mask_v2: np.ndarray,
                  *, fee: float, slip: float, device) -> tuple[dict, dict, dict]:
    """Runs greedy_replay_entry_veto three times on the SAME components: WITHOUT any veto,
    WITH the original (v1) mask, WITH the breakout-confirmed (v2) mask."""
    components["zig075"].pop("short_entry_veto_mask", None)
    diag_none, ledger_none = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_none = mfe_width._duration_gated(ledger_none, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    components["zig075"]["short_entry_veto_mask"] = mask_v1
    diag_v1, ledger_v1 = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_v1 = mfe_width._duration_gated(ledger_v1, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    components["zig075"]["short_entry_veto_mask"] = mask_v2
    diag_v2, ledger_v2 = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=abl.sweep.COST_MULT, device=device)
    wg_v2 = mfe_width._duration_gated(ledger_v2, aligned_frame, abl.greedy.DURATION_THRESHOLD)

    return (
        {"pnl": wg_none["pnl"], "mdd": wg_none["mdd"], "trades": wg_none["trades"]},
        {"pnl": wg_v1["pnl"], "mdd": wg_v1["mdd"], "trades": wg_v1["trades"], "veto_bars": diag_v1.get("veto_bars", 0)},
        {"pnl": wg_v2["pnl"], "mdd": wg_v2["mdd"], "trades": wg_v2["trades"], "veto_bars": diag_v2.get("veto_bars", 0)},
    )


def run_window(window_key: str, windows: dict, score_by_base: dict, breakout_by_base: dict, threshold: float,
               device, fee: float, slip: float, seeds: list[int]) -> dict[str, Any]:
    log(f"=== window={window_key} ===")

    aligned_frame, real_components, _prep_diag = guard.prepare_regime_aware_components(
        window_key, windows, score_by_base, threshold, OUT_DIR, device)
    mask_v1, mask_v2 = build_v2_mask_for_frame(aligned_frame, window_key, score_by_base, breakout_by_base, threshold)
    log(f"  activation rate: v1={mask_v1.mean():.4f} ({int(mask_v1.sum())} bars)  "
        f"v2={mask_v2.mean():.4f} ({int(mask_v2.sum())} bars)  "
        f"v2/v1 ratio={mask_v2.sum() / max(mask_v1.sum(), 1):.3f}")

    none_r, v1_r, v2_r = paired_v1_v2(aligned_frame, real_components, mask_v1, mask_v2, fee=fee, slip=slip, device=device)
    log(f"  real_g0: NONE={none_r['pnl']:+.2f}%  V1(orig)={v1_r['pnl']:+.2f}%(veto_bars={v1_r['veto_bars']})  "
        f"V2(fix)={v2_r['pnl']:+.2f}%(veto_bars={v2_r['veto_bars']})")

    seed_rows = []
    for i, seed in enumerate(seeds):
        aligned_frame_r, components_r = abl.build_ablation_components(
            window_key, windows, score_by_base, threshold, OUT_DIR, device,
            side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed))
        none_s, v1_s, v2_s = paired_v1_v2(aligned_frame_r, components_r, mask_v1, mask_v2, fee=fee, slip=slip, device=device)
        seed_rows.append({"seed": seed, "none_pnl": none_s["pnl"], "v1_pnl": v1_s["pnl"], "v2_pnl": v2_s["pnl"],
                           "v1_delta": v1_s["pnl"] - none_s["pnl"], "v2_delta": v2_s["pnl"] - none_s["pnl"],
                           "v1_veto_bars": v1_s["veto_bars"], "v2_veto_bars": v2_s["veto_bars"]})
        if (i + 1) % 5 == 0:
            log(f"  ...random seeds done: {i + 1}/{len(seeds)}")

    sdf = pd.DataFrame(seed_rows)
    v1_delta = sdf["v1_delta"].to_numpy()
    v2_delta = sdf["v2_delta"].to_numpy()

    def _t(arr):
        if arr.std(ddof=1) == 0:
            return float("nan")
        return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))

    log(f"  random v1(orig) delta: mean={v1_delta.mean():+.2f}pp std={v1_delta.std(ddof=1):.2f}pp t={_t(v1_delta):+.2f}")
    log(f"  random v2(fix)  delta: mean={v2_delta.mean():+.2f}pp std={v2_delta.std(ddof=1):.2f}pp t={_t(v2_delta):+.2f}")

    return {
        "window": window_key, "v1_active_frac": float(mask_v1.mean()), "v2_active_frac": float(mask_v2.mean()),
        "real_g0_none_pnl": none_r["pnl"], "real_g0_v1_pnl": v1_r["pnl"], "real_g0_v2_pnl": v2_r["pnl"],
        "real_g0_v1_veto_bars": v1_r["veto_bars"], "real_g0_v2_veto_bars": v2_r["veto_bars"],
        "random_v1_mean_delta": float(v1_delta.mean()), "random_v1_std_delta": float(v1_delta.std(ddof=1)), "random_v1_t": _t(v1_delta),
        "random_v2_mean_delta": float(v2_delta.mean()), "random_v2_std_delta": float(v2_delta.std(ddof=1)), "random_v2_t": _t(v2_delta),
        "seed_rows": seed_rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = abl.DEVICE
    fee, slip = abl.omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = dict(gate.load_all_windows())
    for cand in RANGING_CANDIDATES:
        windows[cand["key"]] = load_custom_window(cand)

    log("=== stage=detector_build (v1 unmodified, v3 = v1 AND close>=rolling_week_mean; internal var names still say v2, holds the v3 mask) ===")
    score_by_base, _robustness, threshold = guard.build_detector()
    breakout_by_base = {sweep.BASE_2025: _rolling_close_high_confirmed(sweep.BASE_2025),
                         sweep.BASE_2026: _rolling_close_high_confirmed(sweep.BASE_2026)}

    seed_sequence = np.random.SeedSequence(2026081799)  # SAME seeds as the paired on/off test
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"N_SEEDS={N_SEEDS} seeds (identical to the paired on/off test): {seeds}")

    log("\n=== stage=context_tier_sanity (Q1/Q2/Q3 -- must NOT kill Q3's activation) ===")
    context_rows = []
    for wkey in CONTEXT_WINDOW_KEYS:
        w = windows[wkey]
        split = gate.WINDOW_DEFS[wkey]["split"]
        q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
        aligned_frame, _ = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
        mask_v1, mask_v2 = build_v2_mask_for_frame(aligned_frame, wkey, score_by_base, breakout_by_base, threshold)
        context_rows.append({"window": wkey, "v1_active_frac": float(mask_v1.mean()), "v2_active_frac": float(mask_v2.mean())})
        log(f"  {wkey}: v1={mask_v1.mean():.4f}  v2={mask_v2.mean():.4f}")
    cdf = pd.DataFrame(context_rows)
    print(cdf.to_string(index=False))
    cdf.to_csv(OUT_DIR / "context_tier_activation_sanity.csv", index=False)
    q3_v2_highest = cdf.loc[cdf.window == "2025q3", "v2_active_frac"].iloc[0] == cdf["v2_active_frac"].max()
    log(f"  Q3 remains the highest-activation quarter under v2: {q3_v2_highest}")

    all_keys = list(DOWNTREND_WINDOW_KEYS) + [c["key"] for c in RANGING_CANDIDATES]
    results = [run_window(wkey, windows, score_by_base, breakout_by_base, threshold, device, fee, slip, seeds) for wkey in all_keys]

    log("\n\n=== FINAL: v1(original) vs v2(breakout-confirmed fix) ===")
    rows = []
    for r in results:
        rows.append({
            "window": r["window"], "v1_active_frac": round(r["v1_active_frac"], 4), "v2_active_frac": round(r["v2_active_frac"], 4),
            "real_g0_v1_pnl": round(r["real_g0_v1_pnl"], 2), "real_g0_v2_pnl": round(r["real_g0_v2_pnl"], 2),
            "real_g0_v1_veto_bars": r["real_g0_v1_veto_bars"], "real_g0_v2_veto_bars": r["real_g0_v2_veto_bars"],
            "random_v1_mean_delta": round(r["random_v1_mean_delta"], 2), "random_v1_t": round(r["random_v1_t"], 2),
            "random_v2_mean_delta": round(r["random_v2_mean_delta"], 2), "random_v2_t": round(r["random_v2_t"], 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "v1_vs_v2_summary.csv", index=False)

    detail_rows = []
    for r in results:
        for sr in r["seed_rows"]:
            detail_rows.append({"window": r["window"], **sr})
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "per_seed_v1_v2_detail.csv", index=False)

    log(f"\nwrote {OUT_DIR / 'v1_vs_v2_summary.csv'} and {OUT_DIR / 'per_seed_v1_v2_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
