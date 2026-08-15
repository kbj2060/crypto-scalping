#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey4 candidate: zig075 LONG entry veto during detected sustained downtrends.

=== Why this script exists ===
Odyssey4's zig075-SHORT entry veto (docs/experiments/eth_omega461_zig075_short_entry_veto_
sustained_uptrend_20260814.md) CONFIRMED strict on VAL + single-touch OOS-Q1/OOS-Q2, and is the
current reference baseline (docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_
20260814.md G0 table: "향후 신규 후보는 이 표를 G0 기준으로 삼는다"). That candidate manages SHORT
beta exposure during a persistent up-move using an external, causal, already-locked regime detector
(rolling 1-week fraction of dual_momentum>0, threshold=p90 of a 2025-Q1+Q2-only calibration sample).
This script tests the MIRROR-IMAGE hypothesis on the other side of the book: does a symmetric
"sustained downtrend" detector (same recipe, dm<0 instead of dm>0) usefully veto zig075 LONG entries?

=== Honest scope note (unlike the SHORT/uptrend candidate) ===
The original zig075-SHORT veto was motivated by a bar-level loss-mechanism diagnosis (Odyssey3
execution log #1: Q3 SHORT losses concentrated 10/19 trades and -0.4089/-0.5440 union loss in
detector-active bars, median MFE only 41% of SL distance). NO equivalent diagnosis has been run for
zig075 LONG entries during downtrends -- this candidate is derived purely by structural symmetry with
a validated template, not by a confirmed loss mechanism specific to LONG. If this candidate fails, the
principled next step is to run the LONG-side equivalent of that diagnosis (dir_p_long/quality_for_
action/MFE decomposition on downtrend-overlapping LONG losers), not to sweep this detector's
parameters further.

=== The detector (new number, zero new DEGREES OF FREEDOM vs the locked recipe) ===
Reuses, verbatim, every methodological choice already locked by research_eth_omega461_regime_aware_
exit_head_uptrend_guard_20260814.build_detector: WEEK_BARS=2016 (dual_momentum's own existing
close.shift(2016) lookback), DETECTOR_PERCENTILE=0.90 ("top decile" convention), calibration window
=2025-Q1+Q2 ONLY (Q3/VAL/OOS never touched). The ONLY change is the inequality direction:
score = rolling(2016).mean(dual_momentum < 0) instead of (dual_momentum > 0). This produces a NEW
threshold constant (this script is the first to compute it -- there is no prior locked value to
assert equality against, unlike the SHORT/uptrend script which re-derives and checks against an
already-published number). The resulting threshold is logged and should be treated as newly locked
for any future reuse, exactly as 0.8025793650793651 was locked by the original guard script.

=== The intervention ===
Layered ON TOP of the Odyssey4 baseline (Odyssey3's h48qual regime-aware exit guard, UNCHANGED, +
zig075 SHORT sustained-uptrend entry veto, UNCHANGED). In the flat-state entry loop, iff the
candidate entry is (component == "zig075" AND side == LONG) and the sustained-downtrend detector is
ACTIVE at the signal bar, skip that entry. Nothing else changes: zig075 SHORT veto logic/threshold
untouched, h48qual (LONG/SHORT + its own exit guard) untouched, all model heads / thresholds / TP/SL
/ sizing / priority / caps untouched.
Baseline for ALL comparisons = the locked Odyssey4 baseline (G0 table in
docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md), NOT the bare Odyssey3
baseline -- this candidate is additive on top of the already-confirmed SHORT veto, per that
contract's own "future candidates use this table as G0" instruction.

=== Verification protocol (pre-registered before running, mirrors the zig075-SHORT script) ===
- G0a: the IMPORTED zveto.greedy_replay_entry_veto (zig075 SHORT veto attached, unmodified) must
  reproduce the locked Odyssey4 G0 table on val+oos_q1 -- environment/data drift check.
- G0b: THIS script's replay copy (dual veto machinery present: SHORT veto attached as in Odyssey4,
  LONG veto NOT attached) must reproduce the Odyssey4 G0 table on ALL 6 windows -- proves the copy
  is faithful outside the intentionally-added LONG-veto block, and doubles as the baseline tuples
  for the verdict.
- Candidate: LONG veto at the primary (p90) downtrend threshold, all 6 windows, single execution,
  SHORT veto still attached throughout (Odyssey4 baseline is never disabled).
- Verdict: gate.summarize_multiwindow (with_gate PnL AND MDD non-worse) vs the Odyssey4 baseline,
  strict + relaxed(3pp), VAL gate first, then OOS-Q1+OOS-Q2 single touch. 2025q1/q2/q3 stay
  context-tier.
- Robustness (context only, same p75/p95 percentiles of the SAME downtrend calibration sample):
  LONG veto threshold at p75/p95 on the three 2025 quarters. SHORT veto stays at its own locked p90
  everywhere; h48qual's exit guard stays at p90 everywhere.

fresh_forward_bar_by_bar=true (single causal forward pass, i increasing; detector is a plain
backward-looking rolling mean; veto reads mask[i] at the signal bar only).
trade_ledgers_used_as_input=false (ledgers are write-only outputs). saved_parent_exit_timestamps_
used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module (guard module,
zveto module both imported and read only). No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
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
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as zveto  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
VETO_COMPONENT = "zig075"

# Same recipe as guard.build_detector, mirrored: WEEK_BARS/percentile/calibration window are REUSED
# verbatim, not re-chosen. Only the sign of the dual_momentum comparison flips.
WEEK_BARS = guard.WEEK_BARS
DETECTOR_PERCENTILE = guard.DETECTOR_PERCENTILE
CALIBRATION_START = guard.CALIBRATION_START
CALIBRATION_END = guard.CALIBRATION_END
ROBUSTNESS_PERCENTILES = guard.ROBUSTNESS_PERCENTILES

# G0 reference -- the locked Odyssey4 baseline (Odyssey3 + zig075 SHORT sustained-uptrend entry
# veto), copied verbatim from docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_
# 20260814.md G0 table ("진입거부 p90 적용" column). This candidate is additive on top of this
# table, per that contract's own instruction that future candidates use it as G0.
G0_ODYSSEY4 = {
    "2025q1": ({"pnl": 97.70, "mdd": -20.62, "trades": 28}, {"pnl": 44.98, "mdd": -20.62, "trades": 20}),
    "2025q2": ({"pnl": 65.83, "mdd": -14.17, "trades": 31}, {"pnl": 5.62, "mdd": -23.59, "trades": 19}),
    "2025q3": ({"pnl": -10.63, "mdd": -29.66, "trades": 23}, {"pnl": 20.17, "mdd": -19.72, "trades": 17}),
    "val": ({"pnl": 41.13, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
    "oos_q2": ({"pnl": -9.55, "mdd": -20.76, "trades": 13}, {"pnl": -12.69, "mdd": -20.76, "trades": 10}),
}


def log(msg: str) -> None:
    print(f"[zig075_long_entry_veto_downtrend] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


# =====================================================================================================
# Downtrend detector construction -- mirror of guard._rolling_dual_momentum_score /
# guard.build_detector with the dual_momentum inequality flipped. Same base CSVs, same rolling
# window, same calibration sample, same percentile set. Column name changed to avoid any ambiguity
# with the uptrend score living in a separate dict.
# =====================================================================================================


def _rolling_dual_momentum_downtrend_score(base_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "dual_momentum"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    dm = pd.to_numeric(frame["dual_momentum"], errors="raise")
    dm_neg = (dm < 0).astype(float)
    frame["sustained_downtrend_score"] = dm_neg.rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    return frame[["timestamp", "sustained_downtrend_score"]]


def build_downtrend_detector() -> tuple[dict[Path, pd.DataFrame], dict[str, float], float]:
    """Returns (score_by_base_csv, robustness_thresholds, primary_threshold). Calibration sample is
    2025 Q1+Q2 ONLY -- identical calibration window to the uptrend detector; Q3/VAL/OOS never used."""
    score_2025 = _rolling_dual_momentum_downtrend_score(sweep.BASE_2025)
    score_2026 = _rolling_dual_momentum_downtrend_score(sweep.BASE_2026)
    calib_mask = (score_2025["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (score_2025["timestamp"] <= pd.Timestamp(CALIBRATION_END))
    calib = score_2025.loc[calib_mask, "sustained_downtrend_score"].dropna()
    thresholds = {f"p{int(p * 100)}": float(calib.quantile(p)) for p in ROBUSTNESS_PERCENTILES}
    primary = thresholds[f"p{int(DETECTOR_PERCENTILE * 100)}"]
    score_by_base = {sweep.BASE_2025: score_2025, sweep.BASE_2026: score_2026}
    return score_by_base, thresholds, primary


def _downtrend_mask_for_frame(aligned_frame: pd.DataFrame, window_name: str, score_by_base: dict[Path, pd.DataFrame], threshold: float) -> tuple[np.ndarray, int]:
    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    score = score_by_base[base_csv]
    merged = aligned_frame[["timestamp"]].merge(score, on="timestamp", how="left")
    if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{window_name}: downtrend detector score merge failed (row count/order mismatch)")
    raw = merged["sustained_downtrend_score"]
    n_nan = int(raw.isna().sum())
    mask = (raw > threshold).fillna(False).to_numpy(dtype=bool)
    return mask, n_nan


def _attach_long_veto_mask(components: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Return a components dict whose zig075 entry ADDITIONALLY carries the LONG downtrend veto
    mask, on top of whatever it already carries (the SHORT uptrend veto mask, attached separately by
    zveto._attach_veto_mask). Shallow-copies only the zig075 dict -- never mutates the input."""
    out = dict(components)
    zig = dict(out[VETO_COMPONENT])
    zig["long_entry_veto_mask"] = mask
    out[VETO_COMPONENT] = zig
    return out


# =====================================================================================================
# Renamed copy of zveto.greedy_replay_entry_veto (itself a copy of guard.greedy_replay_regime_aware_
# exit_guard). h48qual's exit guard and zig075's SHORT/uptrend veto are BOTH fully preserved
# unchanged -- the only new logic is the LONG/downtrend veto block, mirroring the SHORT block exactly
# but checking side > 0 and a differently-named mask key.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_dual_entry_veto(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
    trailing_activate_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Identical to zveto.greedy_replay_entry_veto (h48qual regime-aware exit guard + zig075
    SHORT/uptrend entry veto fully preserved), plus ONE new rule in the flat-state entry loop: if a
    component carries a 'long_entry_veto_mask' and its candidate entry this bar is LONG while
    mask[i] is True at the signal bar, that entry is skipped too. No LONG mask attached ->
    byte-identical to zveto.greedy_replay_entry_veto's own behaviour."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    armed = False
    trailing_enabled = trailing_activate_frac is not None and trailing_trail_frac is not None
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars_short = 0
    veto_bars_long = 0
    veto_events: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == guard_component:
                guard_hold_bars += 1
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason and trailing_enabled:
                if (not armed) and take_profit > 0.0 and mfe >= float(trailing_activate_frac) * take_profit:
                    armed = True
                if armed and mfe > 0.0 and move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                use_guard = False
                mask = comp.get("sustained_uptrend_mask")
                if active_comp == guard_component and mask is not None and bool(mask[i]):
                    use_guard = True
                if use_guard:
                    guard_active_bars += 1
                    prob = rs._predict_exit_prob_one(
                        comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=int(i),
                        expert=expert, pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
                    default_prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if (prob >= active_threshold) != (default_prob >= float(comp["exit_threshold"])):
                        guard_decision_differs_bars += 1
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp["exit_threshold"])
                if prob >= active_threshold:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        # flat: try priority order
        for name in greedy.PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            # --- zig075 SHORT entry veto (Odyssey4 baseline, unchanged vs zveto.greedy_replay_entry_veto) ---
            short_veto_mask = comp.get("short_entry_veto_mask")
            if short_veto_mask is not None and side < 0 and bool(short_veto_mask[i]):
                veto_bars_short += 1
                veto_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "component": name, "veto_side": "short"})
                continue
            # --- zig075 LONG entry veto: only new logic vs zveto.greedy_replay_entry_veto ---
            long_veto_mask = comp.get("long_entry_veto_mask")
            if long_veto_mask is not None and side > 0 and bool(long_veto_mask[i]):
                veto_bars_long += 1
                veto_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "component": name, "veto_side": "long"})
                continue
            # --- end entry veto block ---
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            armed = False
            break

    diag = {
        "reason_counts": reasons,
        f"{guard_component}_hold_bars": guard_hold_bars,
        f"{guard_component}_guard_active_bars": guard_active_bars,
        f"{guard_component}_guard_decision_differs_bars": guard_decision_differs_bars,
        "veto_bars_short": veto_bars_short,
        "veto_bars_long": veto_bars_long,
        "veto_events": veto_events,
    }
    return diag, pd.DataFrame(rows)


def _ledger_diff(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    return zveto._ledger_diff(baseline, candidate)


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
            "Odyssey4 candidate -- zig075 LONG entry veto during detected sustained downtrends. "
            "Mirror-image of the CONFIRMED zig075 SHORT/uptrend entry veto (2026-08-14): same "
            "detector recipe (rolling 1-week fraction of dual_momentum, threshold=p90 of "
            "2025-Q1+Q2-only calibration), inequality flipped (dual_momentum<0), applied to LONG "
            "entries. Layered ON TOP of the locked Odyssey4 baseline (h48qual regime exit guard + "
            "zig075 SHORT/uptrend veto), not the bare Odyssey3 baseline. Zero new free parameters "
            "in the RECIPE; the resulting threshold constant is newly computed (no prior locked "
            "value exists for the downtrend score) and logged for future reuse. Unlike the "
            "SHORT/uptrend candidate, this design is NOT backed by a bar-level loss-mechanism "
            "diagnosis of zig075 LONG entries -- it tests the symmetry hypothesis; see module "
            "docstring 'Honest scope note'."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    # =================================================================================================
    # stage=load_windows
    # =================================================================================================
    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=detector_build -- uptrend detector reused verbatim (drives h48qual's exit guard AND
    # zig075's existing SHORT veto); downtrend detector newly built here (drives the new LONG veto).
    # =================================================================================================
    log("=== stage=detector_build ===")
    score_by_base_up, robustness_thresholds_up, threshold_up = guard.build_detector()
    if abs(threshold_up - zveto.EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        report["stage_reached"] = "detector_build"
        report["gate_pass"] = False
        report["note"] = f"recomputed uptrend p90 threshold {threshold_up!r} != locked Odyssey4 value {zveto.EXPECTED_PRIMARY_THRESHOLD!r} -- data drift, aborting."
        _write_report(report)
        log("stage=ABORT uptrend threshold drift")
        return 1
    log(f"  uptrend thresholds (Q1+Q2-only): {robustness_thresholds_up}  primary(p90)={threshold_up:.10f} == locked Odyssey4 value")

    score_by_base_down, robustness_thresholds_down, threshold_down = build_downtrend_detector()
    log(f"  downtrend thresholds (Q1+Q2-only, NEWLY COMPUTED): {robustness_thresholds_down}  primary(p90)={threshold_down:.10f}")
    report["detector_uptrend_short_veto"] = {
        "reused_from": "research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector (unmodified import)",
        "threshold_used": threshold_up,
        "matches_locked_odyssey4_value": True,
    }
    report["detector_downtrend_long_veto"] = {
        "feature": "dual_momentum (features/engineering.py _dual_momentum, unmodified)",
        "aggregation": f"rolling({WEEK_BARS}, min_periods={WEEK_BARS}).mean() of (dual_momentum < 0)",
        "calibration_window": [CALIBRATION_START, CALIBRATION_END],
        "calibration_excludes_2025q3": True,
        "percentile_primary": DETECTOR_PERCENTILE,
        "thresholds_q1q2_only": robustness_thresholds_down,
        "threshold_used": threshold_down,
        "newly_computed_this_script": True,
        "new_free_parameters": 0,
        "note": "recipe (week_bars/percentile/calibration window) is verbatim-reused from the locked uptrend detector; only the resulting numeric threshold differs because the input series (dual_momentum<0 instead of >0) differs.",
    }

    # =================================================================================================
    # stage=G0a -- the IMPORTED zveto.greedy_replay_entry_veto (SHORT veto attached, unmodified) must
    # reproduce the locked Odyssey4 G0 table on val+oos_q1.
    # =================================================================================================
    log("=== stage=G0a_odyssey4_baseline_via_zveto_module ===")
    g0a: dict[str, Any] = {}
    prepared: dict[str, tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]] = {}
    for wname in ("val", "oos_q1"):
        aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base_up, threshold_up, OUT_DIR, device)
        short_mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base_up, threshold_up)
        veto_components = zveto._attach_veto_mask(components, short_mask)
        prepared[wname] = (aligned_frame, veto_components, prep_diag)
        diag, ledger = zveto.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_ODYSSEY4[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0a[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg}}
        log(f"  {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg}")
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0a_odyssey4_baseline_via_zveto_module"] = {"windows": g0a, "pass": g0a_pass}
    log(f"stage=G0a_result pass={g0a_pass}")

    # =================================================================================================
    # stage=G0b -- copy-fidelity check AND baseline production: THIS script's dual-veto replay copy,
    # SHORT veto attached (as in Odyssey4) but LONG veto NOT attached, must reproduce the Odyssey4 G0
    # table on ALL 6 windows.
    # =================================================================================================
    log("=== stage=G0b_copy_fidelity_all6_short_veto_only ===")
    g0b: dict[str, Any] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}
    for wname in gate.ALL_WINDOWS:
        if wname not in prepared:
            aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base_up, threshold_up, OUT_DIR, device)
            short_mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base_up, threshold_up)
            veto_components = zveto._attach_veto_mask(components, short_mask)
            prepared[wname] = (aligned_frame, veto_components, prep_diag)
        aligned_frame, veto_components, prep_diag = prepared[wname]
        diag, ledger = greedy_replay_dual_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_odyssey4_baseline.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_ODYSSEY4[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg},
                      "veto_bars_long_expected_zero": int(diag["veto_bars_long"])}
        baseline_runs[wname] = {"no_gate": no_gate, "with_gate": with_gate, "ledger": ledger, "ledger_path": str(ledger_path)}
        log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} match={ok_wg} "
            f"veto_bars_short={diag['veto_bars_short']} veto_bars_long={diag['veto_bars_long']}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] and g0b[w]["veto_bars_long_expected_zero"] == 0 for w in gate.ALL_WINDOWS)
    report["g0b_copy_fidelity_all6_short_veto_only"] = {"windows": g0b, "pass": g0b_pass}
    log(f"stage=G0b_result pass={g0b_pass}")

    g0_pass = bool(g0a_pass and g0b_pass)
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (Odyssey4 baseline reproduction and/or copy fidelity). Aborting before trusting any candidate number."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=entry_overlap_static -- cheap pre-replay diagnostic: per window, how many bars carry an
    # active zig075 LONG signal, and how many coincide with the downtrend detector.
    # =================================================================================================
    log("=== stage=entry_overlap_static ===")
    overlap: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, veto_components, prep_diag = prepared[wname]
        zig = veto_components[VETO_COMPONENT]
        side = pd.to_numeric(zig["dec"]["side"], errors="raise").to_numpy()
        active = omega._active(zig["dec"])
        active = active.to_numpy() if hasattr(active, "to_numpy") else np.asarray(active)
        mask, _ = _downtrend_mask_for_frame(aligned_frame, wname, score_by_base_down, threshold_down)
        long_sig = (side > 0) & active.astype(bool)
        overlap[wname] = {
            "zig075_long_signal_bars": int(long_sig.sum()),
            "long_signal_bars_detector_active": int((long_sig & mask).sum()),
            "detector_active_frac": float(mask.mean()),
        }
        log(f"  {wname:8s} long_signal_bars={overlap[wname]['zig075_long_signal_bars']:5d}  "
            f"overlap_with_detector={overlap[wname]['long_signal_bars_detector_active']:5d}  "
            f"detector_active={overlap[wname]['detector_active_frac'] * 100:5.1f}%")
    report["entry_overlap_static"] = overlap

    # =================================================================================================
    # stage=candidate_run -- LONG veto at the primary (p90) downtrend threshold, all 6 windows, single
    # execution. SHORT veto (Odyssey4 baseline) stays attached throughout.
    # =================================================================================================
    log("=== stage=candidate_run (LONG veto @ p90 downtrend, all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, veto_components, prep_diag = prepared[wname]
        mask, _ = _downtrend_mask_for_frame(aligned_frame, wname, score_by_base_down, threshold_down)
        dual_veto_components = _attach_long_veto_mask(veto_components, mask)
        diag, ledger = greedy_replay_dual_entry_veto(aligned_frame, dual_veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_zig075_long_entry_veto_p90.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        diff = _ledger_diff(baseline_runs[wname]["ledger"], ledger)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "odyssey4_baseline": {"no_gate": baseline_runs[wname]["no_gate"], "with_gate": baseline_runs[wname]["with_gate"], "ledger_path": baseline_runs[wname]["ledger_path"]},
            "long_entry_veto_p90": {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)},
            "detector_diag": overlap[wname],
            "veto_replay_diag": {k: v for k, v in diag.items() if k != "veto_events"},
            "veto_events": diag["veto_events"],
            "ledger_diff": diff,
        }
        b_ng, b_wg = baseline_runs[wname]["no_gate"], baseline_runs[wname]["with_gate"]
        log(f"  {wname:8s} baseline  no_gate={b_ng['pnl']:7.2f}%/{b_ng['mdd']:7.2f}%/{b_ng['trades']:3d}  with_gate={b_wg['pnl']:7.2f}%/{b_wg['mdd']:7.2f}%/{b_wg['trades']:3d}")
        log(f"  {wname:8s} veto_p90  no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
            f"veto_bars_long={diag['veto_bars_long']}  removed={diff['n_removed']}(ret {diff['removed_return_sum']:+.4f})  added={diff['n_added']}(ret {diff['added_return_sum']:+.4f})")
    report["comparison"] = comparison

    # =================================================================================================
    # stage=robustness -- LONG veto threshold at p75/p95 (pre-registered percentiles of the SAME
    # downtrend Q1+Q2-only sample), 2025 quarters only, context tier. SHORT veto stays at p90.
    # =================================================================================================
    log("=== stage=robustness (LONG veto @ p75/p95 downtrend, 2025 quarters, context only) ===")
    robustness: dict[str, Any] = {}
    for plabel in ("p75", "p95"):
        thr = robustness_thresholds_down[plabel]
        robustness[plabel] = {"threshold": thr}
        for wname in gate.CONTEXT_WINDOWS:
            aligned_frame, veto_components, prep_diag = prepared[wname]
            mask, _ = _downtrend_mask_for_frame(aligned_frame, wname, score_by_base_down, thr)
            dual_veto_components = _attach_long_veto_mask(veto_components, mask)
            diag, ledger = greedy_replay_dual_entry_veto(aligned_frame, dual_veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            diff = _ledger_diff(baseline_runs[wname]["ledger"], ledger)
            robustness[plabel][wname] = {"no_gate": no_gate, "with_gate": with_gate,
                                         "veto_bars_long": diag["veto_bars_long"], "n_removed": diff["n_removed"], "n_added": diff["n_added"],
                                         "removed_return_sum": diff["removed_return_sum"], "added_return_sum": diff["added_return_sum"]}
            log(f"  {plabel} {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  removed={diff['n_removed']} added={diff['n_added']}")
    report["robustness_context_only"] = robustness

    # =================================================================================================
    # stage=summarize -- VAL gate + OOS-Q1/OOS-Q2 single touch vs the Odyssey4 baseline, strict and
    # relaxed(3pp). 2025 quarters context-only.
    # =================================================================================================
    log("=== stage=summarize ===")
    baseline_tuples = {w: (baseline_runs[w]["no_gate"], baseline_runs[w]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["long_entry_veto_p90"]["no_gate"], comparison[w]["long_entry_veto_p90"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    val_gate_pass_strict = bool(summary_strict["rows"]["val"]["with_gate_pass"])
    val_gate_pass_relaxed = bool(summary_relaxed["rows"]["val"]["with_gate_pass"])
    log(f"  VAL gate: strict={val_gate_pass_strict} relaxed={val_gate_pass_relaxed}")
    log(f"  OOS single touch: strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']}")

    q3_base = comparison["2025q3"]["odyssey4_baseline"]
    q3_veto = comparison["2025q3"]["long_entry_veto_p90"]
    q3_effect = {
        "no_gate": {"baseline_pnl": q3_base["no_gate"]["pnl"], "veto_pnl": q3_veto["no_gate"]["pnl"],
                    "baseline_mdd": q3_base["no_gate"]["mdd"], "veto_mdd": q3_veto["no_gate"]["mdd"]},
        "with_gate": {"baseline_pnl": q3_base["with_gate"]["pnl"], "veto_pnl": q3_veto["with_gate"]["pnl"],
                      "baseline_mdd": q3_base["with_gate"]["mdd"], "veto_mdd": q3_veto["with_gate"]["mdd"]},
    }
    log(f"  2025q3 effect: {q3_effect}")

    report["summary"] = {
        "val_gate_pass_strict": val_gate_pass_strict,
        "val_gate_pass_relaxed": val_gate_pass_relaxed,
        "multiwindow_strict_mdd0pp": summary_strict,
        "multiwindow_relaxed_mdd3pp": summary_relaxed,
        "q3_effect_context_tier_never_gated": q3_effect,
    }
    report["stage_reached"] = "summarize"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']} val_strict={val_gate_pass_strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
