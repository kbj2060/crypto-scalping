#!/usr/bin/env python3
"""DIAGNOSIS ONLY -- Odyssey4 #6: bar-level loss-mechanism diagnosis for zig075 LONG entries during
detected sustained downtrends.

=== Why this script exists ===
docs/experiments/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.md (Odyssey4
execution log #5, CONFIRMED) tested a mirror-image entry veto (skip zig075 LONG entries when a
symmetric "sustained downtrend" detector is active) WITHOUT first running the bar-level loss-
mechanism diagnosis that motivated the original SHORT/uptrend veto (Odyssey3 execution log #1,
scripts/research_eth_omega461_zig075_sustained_uptrend_diagnosis_20260814.py: "is this an exit
problem the model could have fixed, or a pure entry-timing problem no exit policy could salvage?").
That doc's own "정직한 한계" flagged this gap explicitly. This script closes it, structured as the
exact mirror of the original diagnosis:
  (1) reason/count/hold-bar breakdown of zig075 LONG trades, split by whether the entry SIGNAL bar
      coincided with the downtrend detector (the veto's own condition) -- reusing the ALREADY
      SIMULATED Odyssey4-baseline ledgers written by execution log #5's own script (diagnostic reuse,
      not a new simulation).
  (2) a FRESH, causal, bar-by-bar walk of zig075's own exit-head probability across the holding
      window of every LONG trade found in (1), reconstructing move/mfe/mae/take_profit/stop_loss/
      pos_values bar-by-bar EXACTLY as replay_omega4_6_1_greedy_router_20260706.greedy_replay does
      (imported unmodified from the original diagnosis script), self-checked against each trade's own
      recorded exit reason and trade_return before any aggregate is trusted.
  (3) a counterfactual exit-threshold sweep on the detector-overlap subset (mirrors the original
      diagnosis's Finding 3): would ANY exit_head threshold, applied post-hoc, have improved these
      trades as cleanly as the entry veto did -- i.e. is entry veto the ONLY viable lever, or does an
      exit-side alternative also exist?
  (4) a NEW addition vs the original diagnosis: direction/quality confidence (dir_p_long,
      quality_for_action, read from the raw prediction CSV at each trade's entry signal bar) compared
      between winners and losers within the detector-overlap subset -- checks whether the model's own
      internal confidence signals could have separated these trades (if they can't, that is direct
      support for why an EXTERNAL regime veto was needed instead of a model-internal selection rule,
      mirroring the stated rationale in the SHORT/uptrend veto's own design docstring).

=== Compliance (diagnostic carve-out, NOT a promotion/test claim) ===
Layer 1 reads the Odyssey4-baseline ledgers (already written by execution log #5's own G0b stage) as
INPUT -- explicitly allowed for diagnostic/historical-reproduction use per repo policy, NOT a
promotion/model-selection/test claim; execution log #5's CONFIRMED verdict already stands on its own
fresh-forward replay, this script only explains it. Layer 2 is a FRESH causal per-bar computation (no
future bar ever read). Layer 3 is a pure post-hoc counterfactual over already-computed bar-by-bar
probabilities, informational only -- no new design is proposed or gated here. Layer 4 reads the raw
prediction CSV (already-computed model outputs, not future information) at each trade's own entry
signal bar only. No retraining, no GPU (DEVICE=cpu). Does NOT touch trading_bot.py /
trading_bot_modules/omega4_6_1_live.py / trading_bot_modules/runtime_config.py / .env. Does NOT
modify any imported module -- all of the original diagnosis script, the LONG-veto candidate script,
the guard module, and the gate module are imported and read only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_zig075_sustained_uptrend_diagnosis_20260814 as origdiag  # noqa: E402
import research_eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815 as lveto  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_long_downtrend_loss_diagnosis_20260815"
BASELINE_LEDGER_DIR = lveto.OUT_DIR  # execution log #5's own output dir; ledgers named
# portfolio_ledger_{window}_odyssey4_baseline.csv (written by that script's G0b stage).
DEVICE = portfolio.DEVICE
EXPECTED_DOWNTREND_THRESHOLD = 0.9712301587301587  # locked by execution log #5's report.json
# ["detector_downtrend_long_veto"]["threshold_used"] -- reused, asserted, never re-derived here.

MFE_FAVORABLE_EPS = 0.0
COUNTERFACTUAL_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)


def log(msg: str) -> None:
    print(f"[zig075_long_downtrend_diag] {msg}", flush=True)


# =========================================================================================================
# Layer 1: reason/count breakdown, reusing the already-written Odyssey4-baseline ledgers.
# =========================================================================================================


def _load_baseline_ledger(wname: str) -> pd.DataFrame:
    path = BASELINE_LEDGER_DIR / f"portfolio_ledger_{wname}_odyssey4_baseline.csv"
    if not path.exists():
        raise RuntimeError(f"expected Odyssey4-baseline ledger missing (should have been written by "
                            f"execution log #5's script): {path}")
    return pd.read_csv(path)


def _zig075_long_subset(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger[(ledger["source_component"] == "zig075") & (ledger["side"] == 1)].reset_index(drop=True)


# =========================================================================================================
# Layer 1b: alignment identical to the LONG-veto candidate script (both h48qual+zig075 q_tags) so
# entry_i/exit_i line up with the ledger being re-read -- ALSO returns aligned_paths (unlike
# origdiag._prepare_zig075, which discards them) since Layer 4 needs the raw prediction CSV path.
# =========================================================================================================


def _prepare_zig075_with_paths(window_name: str, windows: dict[str, Any], out_dir: Path, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    zig075 = prep(aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)
    return aligned_frame, zig075, aligned_paths["zig075"]


# =========================================================================================================
# Layer 4: direction/quality confidence at each trade's entry signal bar, read from the raw (aligned)
# prediction CSV -- NOT part of comp["dec"], which only carries the derived quality_score/confidence,
# not the raw dir_p_long/quality_for_action columns needed for the winner-vs-loser comparison.
# =========================================================================================================


def _load_raw_confidence(pred_csv: Path, *, oof: bool) -> pd.DataFrame:
    prefix = omega._tabm_prefix(oof)
    cols = ["timestamp", f"{prefix}dir_p_long", f"{prefix}dir_p_short", f"{prefix}quality_for_action", f"{prefix}dir_confidence"]
    frame = pd.read_csv(pred_csv, usecols=[c for c in cols if c != "timestamp"] + ["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    out = pd.DataFrame({
        "timestamp": frame["timestamp"],
        "dir_p_long": pd.to_numeric(frame[f"{prefix}dir_p_long"], errors="raise"),
        "dir_p_short": pd.to_numeric(frame[f"{prefix}dir_p_short"], errors="raise"),
        "quality_for_action": pd.to_numeric(frame[f"{prefix}quality_for_action"], errors="raise"),
        "dir_confidence": pd.to_numeric(frame[f"{prefix}dir_confidence"], errors="raise"),
    })
    return out


# =========================================================================================================
# Layer 3: counterfactual exit-threshold sweep, mirroring the original diagnosis's Finding 3. Reuses
# each trade's already-walked bar-by-bar probability trajectory (Layer 2 output) -- no new per-bar
# model queries.
# =========================================================================================================


def _counterfactual_at_threshold(walk: dict[str, Any], threshold: float) -> float | None:
    """Returns the move-based improvement (candidate_exit_move - actual_final_move, + = better) if
    this threshold would have fired at some bar before the trade's actual exit; None if it never
    would have fired (probability never reached threshold on a bar where the model was even queried)."""
    final_move = walk["bars"][-1]["move"]
    for b in walk["bars"]:
        if b["exit_prob"] is not None and b["exit_prob"] >= threshold:
            return float(b["move"] - final_move)
    return None


def counterfactual_sweep(walks: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr in COUNTERFACTUAL_THRESHOLDS:
        improvements = [_counterfactual_at_threshold(w, thr) for w in walks]
        fired = [imp for imp in improvements if imp is not None]
        out[f"{thr:.2f}"] = {
            "n_fired": len(fired),
            "n_total": len(walks),
            "mean_improvement": float(np.mean(fired)) if fired else None,
            "sum_improvement": float(np.sum(fired)) if fired else None,
        }
    return out


# =========================================================================================================
# Per-window diagnosis.
# =========================================================================================================


def diagnose_window(wname: str, windows: dict[str, Any], device: torch.device, *, fee: float, slip: float, cost_mult: float,
                     score_by_base_down: dict[Path, pd.DataFrame], threshold_down: float) -> dict[str, Any]:
    log(f"=== window={wname} ===")
    ledger = _load_baseline_ledger(wname)
    long_trades = _zig075_long_subset(ledger)
    log(f"  zig075 LONG trades in Odyssey4-baseline ledger: {len(long_trades)}")

    aligned_frame, comp, pred_csv = _prepare_zig075_with_paths(wname, windows, OUT_DIR, device)
    mask, n_nan = lveto._downtrend_mask_for_frame(aligned_frame, wname, score_by_base_down, threshold_down)
    oof = windows[wname]["oof"]
    conf = _load_raw_confidence(pred_csv, oof=oof)
    if not conf["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{wname}: raw confidence CSV timestamp order != aligned_frame (alignment drift)")

    overlap_flags = long_trades["entry_signal_i"].astype(int).apply(lambda i: bool(mask[i]))
    long_trades = long_trades.assign(detector_overlap=overlap_flags)

    walks = [origdiag._walk_trade_probabilities(aligned_frame, comp, row, fee=fee, slip=slip, cost_mult=cost_mult, device=device)
              for _, row in long_trades.iterrows()]
    reason_mismatches = [w for w in walks if not w["reason_match"]]
    return_mismatches = [w for w in walks if not w["trade_return_match"]]
    self_check_pass = not reason_mismatches and not return_mismatches
    if not self_check_pass:
        log(f"  WARNING: self-check failures -- reason_mismatches={len(reason_mismatches)} return_mismatches={len(return_mismatches)}")

    def _subset_stats(idx: list[int]) -> dict[str, Any]:
        if not idx:
            return {"n_trades": 0}
        sub = long_trades.iloc[idx]
        sub_walks = [walks[i] for i in idx]
        reasons = sub["reason"].value_counts().to_dict()
        sl_walks = [w for w in sub_walks if w["recorded_reason"] == "stop_loss"]
        sl_ratio = [abs(w["mfe"]) / abs(w["stop_loss"]) if w["stop_loss"] else None for w in sl_walks]
        sl_ratio = [r for r in sl_ratio if r is not None]
        conf_rows = conf.iloc[sub["entry_signal_i"].astype(int).to_numpy()]
        winners_mask = (sub["trade_return"] > 0).to_numpy()
        return {
            "n_trades": int(len(sub)),
            "sum_trade_return": float(sub["trade_return"].sum()),
            "win_rate": float(winners_mask.mean()),
            "reason_counts": {str(k): int(v) for k, v in reasons.items()},
            "exit_head_reason_count": int((sub["reason"] == "exit_head").sum()),
            "n_ever_favorable_mfe_gt_0": int(sum(1 for w in sub_walks if w["ever_favorable"])),
            "n_stop_loss_trades": len(sl_walks),
            "n_stop_loss_never_favorable": int(sum(1 for w in sl_walks if not w["ever_favorable"])),
            "stop_loss_mfe_over_sl_distance_median": (float(np.median(sl_ratio)) if sl_ratio else None),
            "take_profit_share": float((sub["reason"] == "take_profit").mean()),
            "dir_p_long_winners": {"min": float(conf_rows.loc[winners_mask.astype(bool), "dir_p_long"].min()) if winners_mask.any() else None,
                                    "max": float(conf_rows.loc[winners_mask.astype(bool), "dir_p_long"].max()) if winners_mask.any() else None,
                                    "mean": float(conf_rows.loc[winners_mask.astype(bool), "dir_p_long"].mean()) if winners_mask.any() else None},
            "dir_p_long_losers": {"min": float(conf_rows.loc[~winners_mask.astype(bool), "dir_p_long"].min()) if (~winners_mask).any() else None,
                                   "max": float(conf_rows.loc[~winners_mask.astype(bool), "dir_p_long"].max()) if (~winners_mask).any() else None,
                                   "mean": float(conf_rows.loc[~winners_mask.astype(bool), "dir_p_long"].mean()) if (~winners_mask).any() else None},
            "quality_for_action_winners": {"min": float(conf_rows.loc[winners_mask.astype(bool), "quality_for_action"].min()) if winners_mask.any() else None,
                                            "max": float(conf_rows.loc[winners_mask.astype(bool), "quality_for_action"].max()) if winners_mask.any() else None,
                                            "mean": float(conf_rows.loc[winners_mask.astype(bool), "quality_for_action"].mean()) if winners_mask.any() else None},
            "quality_for_action_losers": {"min": float(conf_rows.loc[~winners_mask.astype(bool), "quality_for_action"].min()) if (~winners_mask).any() else None,
                                           "max": float(conf_rows.loc[~winners_mask.astype(bool), "quality_for_action"].max()) if (~winners_mask).any() else None,
                                           "mean": float(conf_rows.loc[~winners_mask.astype(bool), "quality_for_action"].mean()) if (~winners_mask).any() else None},
        }

    overlap_idx = [i for i, ov in enumerate(long_trades["detector_overlap"]) if ov]
    non_overlap_idx = [i for i, ov in enumerate(long_trades["detector_overlap"]) if not ov]
    overlap_stats = _subset_stats(overlap_idx)
    non_overlap_stats = _subset_stats(non_overlap_idx)
    log(f"  overlap: n={overlap_stats['n_trades']} sum_return={overlap_stats.get('sum_trade_return')} "
        f"win_rate={overlap_stats.get('win_rate')} exit_head_count={overlap_stats.get('exit_head_reason_count')}")
    log(f"  non_overlap: n={non_overlap_stats['n_trades']} sum_return={non_overlap_stats.get('sum_trade_return')} "
        f"win_rate={non_overlap_stats.get('win_rate')}")

    overlap_walks = [walks[i] for i in overlap_idx]
    cf = counterfactual_sweep(overlap_walks) if overlap_walks else {}
    if cf:
        for thr, v in cf.items():
            log(f"    counterfactual thr={thr}: fired={v['n_fired']}/{v['n_total']} mean_improvement={v['mean_improvement']}")

    return {
        "n_long_trades": int(len(long_trades)),
        "detector_active_frac": float(mask.mean()),
        "self_check": {"reason_mismatches": len(reason_mismatches), "return_mismatches": len(return_mismatches), "all_pass": self_check_pass},
        "overlap_subset": overlap_stats,
        "non_overlap_subset": non_overlap_stats,
        "overlap_counterfactual_threshold_sweep": cf,
        "overlap_trade_entry_signal_i": [int(x) for x in long_trades.iloc[overlap_idx]["entry_signal_i"]] if overlap_idx else [],
    }


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Odyssey4 #6 -- bar-level loss-mechanism diagnosis for zig075 LONG entries during "
            "detected sustained downtrends, closing the gap execution log #5 (CONFIRMED entry veto) "
            "left open: is the downtrend-overlap subset a pure entry-timing problem (no exit policy "
            "could have salvaged it, supporting entry veto as the right lever) or could an exit-side "
            "threshold have achieved a similar effect? Mirrors Odyssey3 execution log #1's structure "
            "(reason breakdown + bar-by-bar MFE/exit-probability walk + counterfactual threshold "
            "sweep), plus a new direction/quality-confidence winner-vs-loser comparison."
        ),
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=downtrend_detector (reused from execution log #5, asserted unchanged) ===")
    score_by_base_down, robustness_thresholds_down, threshold_down = lveto.build_downtrend_detector()
    if abs(threshold_down - EXPECTED_DOWNTREND_THRESHOLD) > 1e-9:
        report["stage_reached"] = "detector_build"
        report["gate_pass"] = False
        report["note"] = f"recomputed downtrend p90 threshold {threshold_down!r} != locked execution-log-#5 value {EXPECTED_DOWNTREND_THRESHOLD!r} -- data drift, aborting."
        _write_report(report)
        log("stage=ABORT downtrend threshold drift")
        return 1
    log(f"  threshold_down={threshold_down:.10f} == locked execution-log-#5 value")

    results: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        results[wname] = diagnose_window(wname, windows, device, fee=fee, slip=slip, cost_mult=sweep.COST_MULT,
                                          score_by_base_down=score_by_base_down, threshold_down=threshold_down)
    report["by_window"] = results

    all_self_checks_pass = all(results[w]["self_check"]["all_pass"] for w in gate.ALL_WINDOWS)
    report["all_self_checks_pass"] = all_self_checks_pass

    total_overlap = sum(results[w]["overlap_subset"].get("n_trades", 0) for w in gate.ALL_WINDOWS)
    total_overlap_never_favorable_sl = sum(results[w]["overlap_subset"].get("n_stop_loss_never_favorable", 0) or 0 for w in gate.ALL_WINDOWS)
    total_overlap_sl = sum(results[w]["overlap_subset"].get("n_stop_loss_trades", 0) or 0 for w in gate.ALL_WINDOWS)
    total_overlap_exit_head = sum(results[w]["overlap_subset"].get("exit_head_reason_count", 0) or 0 for w in gate.ALL_WINDOWS)
    report["headline"] = {
        "total_overlap_trades_all_windows": total_overlap,
        "total_overlap_stop_loss_trades": total_overlap_sl,
        "total_overlap_stop_loss_never_favorable": total_overlap_never_favorable_sl,
        "exit_head_ever_fires_in_overlap_subset_any_window": bool(total_overlap_exit_head > 0),
    }
    log("=== stage=headline ===")
    log(f"  {report['headline']}")

    report["stage_reached"] = "done"
    report["gate_pass"] = bool(all_self_checks_pass)
    _write_report(report)
    log(f"stage=done all_self_checks_pass={all_self_checks_pass}")
    return 0 if all_self_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
