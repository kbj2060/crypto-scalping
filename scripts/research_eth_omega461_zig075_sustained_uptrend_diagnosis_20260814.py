#!/usr/bin/env python3
"""DIAGNOSIS ONLY -- Odyssey3 #1 step 1: mechanism diagnosis for zig075's 2025-Q3 SHORT weakness.

=== Why this script exists ===
docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md "다음 점검 대상 #1"
asks whether the regime-aware uptrend guard already applied to h48qual's exit_head (Odyssey2 #11,
scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py) can also be applied
to zig075, whose 2025-Q3 SHORT PnL is -0.517 (original) / -0.500 (h48qual-liveATR variant, zig075
itself untouched either way) per docs/experiments/
eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md (Odyssey2 #10). That doc
found zig075's Q3 trade COUNT barely moves (16->18, vs h48qual's 8->18) -- i.e. zig075's problem is
NOT the turnover-inflation mechanism #11's guard fixes for h48qual. This script asks the prerequisite
question directly, at the ledger+raw-probability level, before any intervention is designed: are
zig075's Q3 SHORT losses caused by the exit_head being too insensitive (rides to stop_loss when it
could have cut losses earlier), or by the entries themselves being wrong-way bets in a persistent
uptrend that no exit policy could have salvaged (post-entry intervention structurally cannot help)?

=== Method ===
Two layers, both diagnostic only (see "Compliance" below for why ledger reuse is fine here but would
NOT be fine for a promotion/test claim):
  (1) Reason/count/hold-bar breakdown of zig075 SHORT trades in 2025q1/q2/q3, reusing the ALREADY
      SIMULATED ledgers written tonight by eth_omega461_multiwindow_confirmation_gate_20260814.py's
      G0b stage (portfolio_ledger_{2025q1,2025q2,2025q3}_{baseline_both_original,
      asymmetric_tabm_liveatr}.csv) -- no new simulation for this layer, exactly the CSVs the task
      names as reusable.
  (2) A FRESH, causal, bar-by-bar walk of zig075's own exit-head probability (rs._predict_exit_prob_
      one, the same raw-probability call site every script in this lineage uses, including
      research_eth_omega461_exit_sweep_20260721.py which #15 imports as `sweep` and reuses for its
      own grid) across the holding window of every zig075 SHORT trade found in (1), reconstructing
      move/mfe/mae/take_profit/stop_loss/pos_values bar-by-bar EXACTLY as replay_omega4_6_1_greedy_
      router_20260706.greedy_replay does, self-checked against each trade's own recorded exit
      reason and trade_return (see _walk_trade_probabilities docstring) before any aggregate is
      trusted. This layer is NEW computation (not read from any ledger) -- it answers "what did the
      model actually see/decide, bar by bar" independent of what the ledger's summary reason column
      alone can show, and in particular whether the position was EVER favorable (mfe>0) during the
      hold (a trade that never went favorable had no window in which ANY exit policy -- more
      sensitive threshold or otherwise -- could have improved on stop_loss; the loss is a pure
      function of the entry timing, not of exit-head behaviour).

=== Compliance (diagnostic carve-out, NOT a promotion/test claim) ===
This script's layer (1) reads pre-existing trade ledgers as input. Per repo policy (Fresh-Forward
Validation/OOS/Test Rule) that is explicitly allowed for diagnostic/accounting/historical-
reproduction use ("저장 원장 기반 replay는 diagnostic ... 전용이다") and explicitly NOT usable as
promotion/model-selection/test evidence -- this script makes no such claim; it only characterizes an
already-decided, already-shadow-deployed baseline's existing behaviour, exactly the same use this
project's docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md
(Odyssey2 #10) made of the identical ledgers. Layer (2) IS a fresh causal per-bar computation (no
future bar ever read; entry/exit bar indices and trade-level constants -- notional/leverage/margin_
fraction/side -- are taken from the ledger, but the exit-head probability at each bar is computed
fresh, forward, using only that bar's own already-closed state). No retraining, no GPU (DEVICE=cpu,
matching every script in this lineage). Does NOT touch trading_bot.py /
trading_bot_modules/omega4_6_1_live.py / trading_bot_modules/runtime_config.py / .env. Does NOT
modify any imported module -- eth_omega461_multiwindow_confirmation_gate_20260814.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
research_eth_omega461_exit_sweep_20260721.py, train_eval_omega4_2_risk_sidecar_20260622.py,
train_omega1_regime3_expert_direction_head_volpca_20260602.py are all imported and read only.
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
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_sustained_uptrend_diagnosis_20260814"
EXISTING_LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_multiwindow_confirmation_gate_20260814"
DEVICE = portfolio.DEVICE

QUARTERS = ("2025q1", "2025q2", "2025q3")  # 2025q3 is the primary target; q1/q2 are context (are
# these quarters' zig075 SHORT trades governed by the SAME structural exit_head-never-fires pattern,
# or is Q3 special?).
VARIANTS = ("baseline_both_original", "asymmetric_tabm_liveatr")  # zig075 config is IDENTICAL in
# both (gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["zig075"] == gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR
# ["zig075"] == portfolio._component_cfg("zig075"), no bundle_override in either) -- only h48qual
# differs between variants, which changes shared-slot availability and therefore WHICH zig075
# signals get to actually open a position (see "union" step below), not zig075's own decision logic.

# "Ever favorable" cutoff for a SHORT position's max-favorable-excursion (mfe) during its hold:
# mfe > 0 means price dipped at least slightly below the (slippage-adjusted) entry price at some bar
# before the trade finally closed -- i.e. there existed at least one bar where SOME exit policy
# (more sensitive threshold, trailing stop, anything) could have locked in a gain or a smaller loss
# than what stop_loss eventually produced. mfe<=0 means the position was underwater (or flat) on
# every single bar of its life -- no exit policy, however aggressive, had anything to exit INTO.
MFE_FAVORABLE_EPS = 0.0


def log(msg: str) -> None:
    print(f"[zig075_diag] {msg}", flush=True)


# =========================================================================================================
# Layer 1: reason/count/hold-bar breakdown, pure reuse of tonight's already-written ledgers.
# =========================================================================================================


def _load_existing_ledger(wname: str, variant: str) -> pd.DataFrame:
    path = EXISTING_LEDGER_DIR / f"portfolio_ledger_{wname}_{variant}.csv"
    if not path.exists():
        raise RuntimeError(f"expected pre-existing ledger missing (should have been written by tonight's "
                            f"eth_omega461_multiwindow_confirmation_gate_20260814.py G0b run): {path}")
    return pd.read_csv(path)


def _zig075_short_subset(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger[(ledger["source_component"] == "zig075") & (ledger["side"] == -1)].reset_index(drop=True)


def reason_breakdown(wname: str, variant: str) -> dict[str, Any]:
    ledger = _load_existing_ledger(wname, variant)
    sub = _zig075_short_subset(ledger)
    holds = (sub["exit_i"] - sub["entry_i"]).to_numpy(dtype=np.float64)
    return {
        "ledger_path": str(EXISTING_LEDGER_DIR / f"portfolio_ledger_{wname}_{variant}.csv"),
        "n_trades": int(len(sub)),
        "sum_trade_return": float(sub["trade_return"].sum()),
        "reason_counts": {str(k): int(v) for k, v in sub["reason"].value_counts().to_dict().items()},
        "exit_head_reason_count": int((sub["reason"] == "exit_head").sum()),
        "hold_bars_mean": float(holds.mean()) if len(holds) else None,
        "hold_bars_median": float(np.median(holds)) if len(holds) else None,
        "hold_bars_max": int(holds.max()) if len(holds) else None,
    }


def _union_trades(wname: str) -> pd.DataFrame:
    """Union of zig075-SHORT rows across both variants, deduped by entry_signal_i (zig075's own
    config is identical between variants -- a signal at the same entry_signal_i is the same zig075
    decision either way; it only actually OPENS a position in the ledger where the shared slot
    happened to be free, which the two ledgers can disagree on -- see module docstring). This union
    is the fullest available diagnostic sample of "zig075 Q3 SHORT trades that actually executed in
    at least one already-simulated variant tonight", still purely a re-read of existing ledgers."""
    frames = []
    for variant in VARIANTS:
        ledger = _load_existing_ledger(wname, variant)
        sub = _zig075_short_subset(ledger).copy()
        sub["source_variant"] = variant
        frames.append(sub)
    combined = pd.concat(frames, ignore_index=True)
    deduped = combined.drop_duplicates(subset=["entry_signal_i", "exit_i"], keep="first").sort_values("entry_signal_i").reset_index(drop=True)
    return deduped


# =========================================================================================================
# Layer 2: fresh causal per-bar exit-probability walk over each identified trade's holding window.
# =========================================================================================================


def _prepare_zig075(window_name: str, windows: dict[str, Any], out_dir: Path, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}  # both
    # h48qual+zig075 q_tags, matching exactly what produced the existing ledgers' row alignment/
    # indexing (see module docstring) -- zig075-only alignment would intersect a DIFFERENT (larger)
    # timestamp set and desync entry_i/exit_i from the ledgers being re-read above.
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else __import__("replay_omega4_6_1_greedy_router_20260706").prepare_component
    zig075 = prep(aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)
    return aligned_frame, zig075


def _walk_trade_probabilities(
    frame: pd.DataFrame, comp: dict[str, Any], trade: pd.Series, *, fee: float, slip: float, cost_mult: float, device: torch.device,
) -> dict[str, Any]:
    """Reconstructs, bar-by-bar from entry_i to exit_i inclusive, EXACTLY the same move/mfe/mae/
    exit-probability computation replay_omega4_6_1_greedy_router_20260706.greedy_replay performs
    while a position is open -- including greedy_replay's own short-circuit (TP/SL checked BEFORE
    the exit-head model is ever queried; a bar where move already satisfies take_profit or
    stop_loss never reaches the _predict_exit_prob_one call, exactly matching live behaviour). Self-
    checks the reconstruction against the trade's OWN recorded (reason, trade_return) before
    returning -- if either check fails, the trade is flagged rather than silently trusted."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    slip_eff = float(slip) * float(cost_mult)
    fee_eff = float(fee) * float(cost_mult)
    side = int(trade["side"])
    entry_i = int(trade["entry_i"])
    entry_signal_i = int(trade["entry_signal_i"])
    exit_i = int(trade["exit_i"])
    notional = float(trade["notional"])
    leverage_v = float(trade["leverage"])
    recorded_reason = str(trade["reason"])
    recorded_trade_return = float(trade["trade_return"])

    take_profit = float(comp["dec"]["take_profit"].iloc[entry_signal_i])
    stop_loss = float(comp["dec"]["stop_loss"].iloc[entry_signal_i])
    dec_side = int(comp["dec"]["side"].iloc[entry_signal_i])
    if dec_side != side:
        raise RuntimeError(f"entry_signal_i={entry_signal_i}: comp['dec'] side={dec_side} != ledger side={side}")

    entry_price = arrays["open"][entry_i] * (1 + slip_eff if side > 0 else 1 - slip_eff)

    mfe = mae = 0.0
    bars: list[dict[str, Any]] = []
    final_reason = ""
    for i in range(entry_i, exit_i + 1):
        move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if side > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
        mfe, mae = max(mfe, move), min(mae, move)
        reason = ""
        if take_profit > 0.0 and move >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and move <= -abs(stop_loss):
            reason = "stop_loss"
        prob = None
        if not reason:
            hold = max(i - entry_i, 0)
            giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
            expert = hard.EXPERT_NAMES[int(comp["route"][i])]
            pos_values = [float(side), float(hold), float(move), float(mfe), float(mae),
                          float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                          float(move + abs(stop_loss)), float(notional), float(leverage_v),
                          float(notional * leverage_v), float(take_profit), float(stop_loss)]
            prob = rs._predict_exit_prob_one(comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert, pos_values=pos_values, device=device)
            if prob >= float(comp["exit_threshold"]):
                reason = "exit_head"
        bars.append({"i": int(i), "bar_offset": int(i - entry_i), "move": float(move), "mfe_so_far": float(mfe), "mae_so_far": float(mae), "exit_prob": (float(prob) if prob is not None else None), "tp_sl_shortcircuit": bool(prob is None)})
        if reason:
            final_reason = reason
            break

    # --- self-check: does the walk reproduce the ledger's own recorded outcome? ---
    reason_match = bool(final_reason == recorded_reason)
    exit_px = arrays["close"][exit_i] * (1 - slip_eff if side > 0 else 1 + slip_eff)
    raw_exit = (exit_px - entry_price) / entry_price if side > 0 else (entry_price - exit_px) / entry_price
    before_frac = 1.0 - fee_eff * notional  # entry-fee-adjusted fraction of entry_equity (see script docstring derivation)
    recon_trade_return = before_frac * (1.0 + raw_exit * notional - fee_eff * notional) - 1.0
    trade_return_match = bool(abs(recon_trade_return - recorded_trade_return) < 1e-6)

    probs = [b["exit_prob"] for b in bars if b["exit_prob"] is not None]
    return {
        "entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": exit_i,
        "entry_timestamp": str(trade["entry_timestamp"]), "exit_timestamp": str(trade["exit_timestamp"]),
        "recorded_reason": recorded_reason, "reconstructed_reason": final_reason, "reason_match": reason_match,
        "recorded_trade_return": recorded_trade_return, "reconstructed_trade_return": float(recon_trade_return), "trade_return_match": trade_return_match,
        "hold_bars": int(exit_i - entry_i),
        "mfe": float(mfe), "mae": float(mae), "ever_favorable": bool(mfe > MFE_FAVORABLE_EPS),
        "max_exit_prob": (float(max(probs)) if probs else None), "mean_exit_prob": (float(np.mean(probs)) if probs else None),
        "n_bars_prob_queried": int(len(probs)), "n_bars_total": int(len(bars)),
        "take_profit": take_profit, "stop_loss": stop_loss,
        "bars": bars,
    }


def diagnose_window(wname: str, windows: dict[str, Any], device: torch.device, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    log(f"=== window={wname} ===")
    breakdown = {variant: reason_breakdown(wname, variant) for variant in VARIANTS}
    for variant, b in breakdown.items():
        log(f"  {variant:26s} n={b['n_trades']:3d} sum={b['sum_trade_return']:+.4f} reasons={b['reason_counts']} exit_head_count={b['exit_head_reason_count']}")

    union = _union_trades(wname)
    log(f"  union of both variants' zig075-SHORT trades (deduped by entry_signal_i): {len(union)}")

    aligned_frame, comp = _prepare_zig075(wname, windows, OUT_DIR, device)
    walks = [_walk_trade_probabilities(aligned_frame, comp, row, fee=fee, slip=slip, cost_mult=cost_mult, device=device) for _, row in union.iterrows()]

    reason_mismatches = [w for w in walks if not w["reason_match"]]
    return_mismatches = [w for w in walks if not w["trade_return_match"]]
    if reason_mismatches or return_mismatches:
        log(f"  WARNING: self-check failures -- reason_mismatches={len(reason_mismatches)} return_mismatches={len(return_mismatches)}")

    n_ever_favorable = sum(1 for w in walks if w["ever_favorable"])
    max_prob_overall = max((w["max_exit_prob"] for w in walks if w["max_exit_prob"] is not None), default=None)
    mean_of_max_probs = float(np.mean([w["max_exit_prob"] for w in walks if w["max_exit_prob"] is not None])) if any(w["max_exit_prob"] is not None for w in walks) else None
    n_trades_with_any_prob_query = sum(1 for w in walks if w["n_bars_prob_queried"] > 0)
    stop_loss_walks = [w for w in walks if w["recorded_reason"] == "stop_loss"]
    n_sl_never_favorable = sum(1 for w in stop_loss_walks if not w["ever_favorable"])

    log(f"  trades_ever_favorable(mfe>0)={n_ever_favorable}/{len(walks)}  "
        f"stop_loss_trades_NEVER_favorable={n_sl_never_favorable}/{len(stop_loss_walks)}  "
        f"max_exit_prob_overall={max_prob_overall}  mean_of_per_trade_max_prob={mean_of_max_probs}  "
        f"trades_where_model_was_ever_queried(no_immediate_tp_sl)={n_trades_with_any_prob_query}/{len(walks)}")

    return {
        "reason_breakdown": breakdown,
        "union_trade_count": int(len(union)),
        "walks": walks,
        "self_check": {"reason_mismatches": len(reason_mismatches), "return_mismatches": len(return_mismatches), "all_pass": bool(not reason_mismatches and not return_mismatches)},
        "summary": {
            "n_trades": len(walks),
            "n_ever_favorable_mfe_gt_0": n_ever_favorable,
            "n_stop_loss_trades": len(stop_loss_walks),
            "n_stop_loss_trades_never_favorable": n_sl_never_favorable,
            "max_exit_prob_overall": max_prob_overall,
            "mean_of_per_trade_max_exit_prob": mean_of_max_probs,
            "n_trades_model_ever_queried": n_trades_with_any_prob_query,
            "n_trades_total": len(walks),
        },
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
            "Odyssey3 #1 step 1 -- diagnoses whether zig075's 2025-Q3 SHORT weakness (Odyssey2 #10, "
            "-0.517/-0.500) is an exit problem (exit_head too insensitive, rides to stop_loss when "
            "it could exit earlier) or an entry-timing problem (positions never favorable, no exit "
            "policy could have helped) before any intervention is designed."
        ),
        "note_on_ledger_reuse": (
            "Layer 1 (reason/count breakdown) reuses tonight's already-written multiwindow-gate "
            "ledgers as INPUT -- explicitly a diagnostic/historical-reproduction use per repo policy, "
            "NOT a promotion/test claim (see module docstring). Layer 2 (per-bar exit-probability "
            "walk) is a FRESH causal computation, not read from any ledger."
        ),
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    results: dict[str, Any] = {}
    for wname in QUARTERS:
        results[wname] = diagnose_window(wname, windows, device, fee=fee, slip=slip, cost_mult=sweep.COST_MULT)
    report["by_quarter"] = results

    all_self_checks_pass = all(results[w]["self_check"]["all_pass"] for w in QUARTERS)
    report["all_self_checks_pass"] = all_self_checks_pass

    q3 = results["2025q3"]["summary"]
    q1 = results["2025q1"]["summary"]
    q2 = results["2025q2"]["summary"]
    exit_head_never_fires_any_quarter = all(
        results[w]["reason_breakdown"][v]["exit_head_reason_count"] == 0
        for w in QUARTERS for v in VARIANTS
    )
    report["headline"] = {
        "exit_head_never_fires_any_quarter_any_variant": exit_head_never_fires_any_quarter,
        "2025q3": q3, "2025q1": q1, "2025q2": q2,
    }
    log("=== stage=headline ===")
    log(f"  exit_head reason NEVER appears in ANY quarter/variant for zig075 SHORT: {exit_head_never_fires_any_quarter}")
    log(f"  2025q3: {q3}")
    log(f"  2025q1: {q1}")
    log(f"  2025q2: {q2}")

    report["stage_reached"] = "done"
    report["gate_pass"] = bool(all_self_checks_pass)
    _write_report(report)
    log(f"stage=done all_self_checks_pass={all_self_checks_pass}")
    return 0 if all_self_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
