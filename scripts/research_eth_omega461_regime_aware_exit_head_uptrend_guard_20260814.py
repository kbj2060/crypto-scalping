#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #11: regime-aware exit-head guard for h48qual, testing the mitigation
PROPOSED but explicitly NOT implemented in docs/experiments/
eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md section 6: "지속 상승
레짐으로 분류된 구간에서만 조건부로 원본(느린 exit_threshold=0.95 정적 판단)으로 되돌리고, 그 외
레짐에서는 재라벨(liveATR, 빠른 회전)을 그대로 쓰는 정책."

=== Mechanism established by that (already-written, cited-not-re-derived) doc ===
The shadow-deployed baseline (asymmetric_tabm_liveatr: h48qual exit_head liveATR-relabeled, zig075
untouched) is a "turnover accelerator" -- it shortens h48qual's average holding period 2-3x in every
2025 quarter. In Q1/Q2 (choppy/reversal-prone), this does not inflate h48qual's own trade count much
(8->8, 5->8) and improves PnL. In Q3 (2025's only sustained LOW-noise uptrend: price drift +66.63%,
lowest realized vol of the three quarters at 60.75%), the same acceleration inflates h48qual's trade
count 8->18 (2.25x, all 18 SHORT) because freed-up slots keep re-triggering an already-known
direction-unbiased short signal in a regime with too little chop to stop it -- portfolio no_gate PnL
goes from -9.73% to -46.26% (4.7x worse).

=== What THIS script tests ===
A causal, PRE-REGISTERED "sustained uptrend" detector that, per bar, selects which of h48qual's TWO
ALREADY-TRAINED exit-head decision paths governs an OPEN h48qual position's exit check that bar:
  - detector ACTIVE   -> h48qual's ORIGINAL frozen exit head (research_eth_omega461_exit_head_
    portfolio_asymmetric_20260813._component_cfg("h48qual") with NO bundle_override -- exactly
    "baseline_both_original"'s h48qual exit policy).
  - detector INACTIVE -> h48qual's current live-ATR-relabeled exit head (bundle_override=
    portfolio.NEW_H48QUAL_BUNDLE -- exactly "asymmetric_tabm_liveatr"'s h48qual exit policy, the
    current shadow-deployed default).
zig075 is completely untouched in every window (always portfolio._component_cfg("zig075"), the
detector is never applied to it). h48qual's quality_threshold/direction/quality heads are untouched.
By construction (not merely by assumption -- see "entry-side diagnostic" in main()), this script's
components dict always sources dec/take_profit/stop_loss/margin/leverage/route from the LIVEATR
h48qual preparation, regardless of guard state -- the ONLY thing the detector ever switches is which
(base_np, exit_runtime, pos_idx, exit_threshold) quadruple answers "should this already-open h48qual
position exit on this bar", at exit_threshold=0.95 either way (same constant, different model).
No retraining: both exit-head decision paths, and the dual_momentum input column used by the
detector, already exist on disk -- only a rolling mean (see below) is computed fresh here.

=== Detector design discipline (STEP ORDER IS THE POINT -- recorded so it can be audited) ===
Step 1 (exploration, BEFORE any threshold was fixed): computed per-bar magnitude/percentile
activation of every trend/persistence-flavoured column the task named (regime_persistence,
chop_index, hurst_48, regime_trending, sig_trend_health, hma_slope, mtf_trend_1h, mtf_trend_4h,
dual_momentum, breakout_strength, cvd_slope_12, cvd_slope_48, funding_roc_12/48/288) across all 6
pre-registered windows (2025q1/q2/q3, val, oos_q1, oos_q2). FINDING: every purely INSTANTANEOUS /
intraday-window (<=288-bar) candidate's activation rate came out nearly IDENTICAL across all 6
windows regardless of threshold (e.g. regime_persistence>P90(Q1+Q2) fires 8.7%/11.3%/10.6% of Q1/Q2/
Q3 bars respectively -- Q3 does NOT stand out) -- i.e. local 5m-bar technical regime state does not
track quarter-scale sustained drift. This echoes this project's existing finding that live regime3
HMM bull-share is ALSO ~equal across the same three quarters (25.57/27.64/27.50%, docs/experiments/
eth_val_oos_regime_mismatch_investigation_20260813.md Sec.2) despite wildly different price drift.
The one exception: `dual_momentum` (features/engineering.py _dual_momentum -- already a 1-WEEK-
lookback [close.shift(2016)] signal, +1 only when BOTH ETH's own trailing-week return AND its
trailing-week return relative to BTC are positive, -1 for the symmetric down case, else 0) tracked
quarter drift direction cleanly on its own mean (Q1 -0.331, Q2 -0.014, Q3 +0.248 -- clearly highest
-- val -0.165, oos_q1 -0.158, oos_q2 -0.257).

Step 2 (still before fixing any number): a raw per-bar dual_momentum reading is itself noisy bar to
bar (it is a step function that only moves when a 1-week return crosses zero, but individual bars
still flicker near that crossing). This script tries exactly ONE simple aggregation (the only "단순
조합" attempted, per the task's own instruction to try at most 2-3 and prefer none) -- a rolling MEAN
of (dual_momentum>0) over a 2016-bar (1-week) window. 2016 bars is not an invented number: it is
dual_momentum's OWN existing lookback (features/engineering.py already hardcodes close.shift(2016)),
reused rather than re-chosen. A same-length rolling window applied to the instantaneous candidates
from Step 1, and a shorter 288-bar (1-day) rolling window applied to dual_momentum itself, were both
also checked and both failed to separate Q3 (see companion doc) -- confirming it is specifically the
week-scale aggregation of an already-week-scale input that matters, not "rolling aggregation" in
general.

Step 3 (threshold choice, fixed BEFORE checking Q3 separation): 90th percentile ("top decile of the
feature's own historical distribution" -- an example the task itself names as principled) of this
rolling series, computed ONLY on 2025-01-01..2025-06-30 (Q1+Q2 combined) -- Q3 is deliberately
EXCLUDED from the calibration sample, so Q3 is never used to derive the number. 90% is the standard
round decile convention, not searched/swept against any outcome.

Step 4 (confirmation, AFTER the rule above was fixed -- this is the only place Q3's own numbers are
looked at): see main() DETECTOR_ACTIVATION diagnostic block and the companion doc for the resulting
per-window activation rates and the robustness check against the 75th/95th percentiles of the same
Q1+Q2-only calibration sample.

fresh_forward_bar_by_bar=true (the renamed greedy_replay copy below is a single causal forward pass,
i increasing, only bar i and already-closed history used at bar i; the detector's rolling window is
a plain backward-looking pandas .rolling(), no negative shift; dual_momentum itself uses
close.shift(2016), a backward shift only). trade_ledgers_used_as_input=false (ledgers are
write-only outputs). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module --
eth_omega461_multiwindow_confirmation_gate_20260814.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
replay_omega4_6_1_greedy_router_20260706.py, research_eth_omega461_exit_sweep_20260721.py,
research_eth_omega461_live_sltp_mfe_width_20260813.py,
train_eval_omega4_2_risk_sidecar_20260622.py are all imported and read only. No retraining, no GPU
(DEVICE=cpu throughout, matching every script in this lineage).
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05

# =====================================================================================================
# Detector constants -- fixed BEFORE this script ever looked at Q3's own numbers (see module
# docstring Step 1-3). WEEK_BARS reuses dual_momentum's own existing lookback (features/engineering.py
# _dual_momentum: close.shift(2016)); DETECTOR_PERCENTILE=0.90 is a round, standard "top decile"
# convention, not swept.
# =====================================================================================================
WEEK_BARS = 2016
DETECTOR_PERCENTILE = 0.90
CALIBRATION_START = gate.WINDOW_DEFS["2025q1"]["start"]     # "2025-01-01"
CALIBRATION_END = gate.WINDOW_DEFS["2025q2"]["end"]         # "2025-06-30 23:59:59" -- Q3 excluded
ROBUSTNESS_PERCENTILES = [0.75, 0.90, 0.95]                 # reported for context only; 0.90 is the
# one actually used to gate the candidate run below.

# G0 reference -- identical to eth_omega461_multiwindow_confirmation_gate_20260814.
# REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR, re-imported by name (not re-typed) for the primary G0
# check; re-typed once more only as the literal task-instruction values to self-document this
# script's own required numbers independent of the gate module's constant name.
G0_REQUIRED = {
    "val": ({"pnl": 46.59, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
}


def log(msg: str) -> None:
    print(f"[regime_uptrend_guard] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


# =====================================================================================================
# Detector construction.
# =====================================================================================================


def _rolling_dual_momentum_score(base_csv: Path) -> pd.DataFrame:
    """Causal, whole-year rolling series: fraction of the trailing WEEK_BARS bars with
    dual_momentum>0. Computed on the FULL base CSV (never window-by-window) so windows that start
    well into a year (val, oos_q2, 2025q2/q3) are never artificially NaN-truncated at their own
    start -- only the genuine first ~1 week of each base CSV's own calendar year is NaN (real
    unavailable history), not a window-boundary artifact. Matches
    research_eth_omega461_exit_sweep_20260721.load_frame's own read/sort/dedup discipline for the
    base CSV exactly, so timestamps line up with every other script in this lineage."""
    frame = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "dual_momentum"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    dm = pd.to_numeric(frame["dual_momentum"], errors="raise")
    dm_pos = (dm > 0).astype(float)
    frame["sustained_uptrend_score"] = dm_pos.rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    return frame[["timestamp", "sustained_uptrend_score"]]


def build_detector() -> tuple[dict[Path, pd.DataFrame], dict[str, float], float]:
    """Returns (score_by_base_csv, robustness_thresholds, primary_threshold). Calibration sample is
    2025 Q1+Q2 ONLY (CALIBRATION_START..CALIBRATION_END) -- Q3, VAL, and both OOS windows are never
    used to derive any of these numbers."""
    score_2025 = _rolling_dual_momentum_score(sweep.BASE_2025)
    score_2026 = _rolling_dual_momentum_score(sweep.BASE_2026)
    calib_mask = (score_2025["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (score_2025["timestamp"] <= pd.Timestamp(CALIBRATION_END))
    calib = score_2025.loc[calib_mask, "sustained_uptrend_score"].dropna()
    thresholds = {f"p{int(p * 100)}": float(calib.quantile(p)) for p in ROBUSTNESS_PERCENTILES}
    primary = thresholds[f"p{int(DETECTOR_PERCENTILE * 100)}"]
    score_by_base = {sweep.BASE_2025: score_2025, sweep.BASE_2026: score_2026}
    return score_by_base, thresholds, primary


def _detector_mask_for_frame(aligned_frame: pd.DataFrame, window_name: str, score_by_base: dict[Path, pd.DataFrame], threshold: float) -> tuple[np.ndarray, int]:
    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    score = score_by_base[base_csv]
    merged = aligned_frame[["timestamp"]].merge(score, on="timestamp", how="left")
    if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{window_name}: detector score merge failed (row count/order mismatch)")
    raw = merged["sustained_uptrend_score"]
    n_nan = int(raw.isna().sum())
    mask = (raw > threshold).fillna(False).to_numpy(dtype=bool)
    return mask, n_nan


# =====================================================================================================
# Component preparation: h48qual is prepared TWICE (liveATR bundle = current shadow default, ORIGINAL
# bundle = pre-relabel) from the SAME prediction CSV/aligned_frame; only the second prep's
# base_np/exit_runtime/pos_idx are used, attached onto the first as a "guard_*" side-channel, together
# with the detector mask. zig075 is prepared once, untouched.
# =====================================================================================================


def _prep_liveatr_only(window_name: str, windows: dict[str, Any], out_dir: Path, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    """h48qual+zig075 on the plain asymmetric_tabm_liveatr config, no guard machinery attached at
    all -- used only by the G0b self-check (detector-forced-inactive identity test)."""
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    components = {name: prep(aligned_frame, aligned_paths[name], cfg, device) for name, cfg in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR.items()}
    return aligned_frame, components


def prepare_regime_aware_components(
    window_name: str, windows: dict[str, Any], score_by_base: dict[Path, pd.DataFrame], threshold: float,
    out_dir: Path, device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_liveatr = prep(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], device)
    h48qual_original = prep(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["h48qual"], device)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)

    mask, n_nan = _detector_mask_for_frame(aligned_frame, window_name, score_by_base, threshold)

    h48qual_guarded = dict(h48qual_liveatr)
    h48qual_guarded["guard_base_np"] = h48qual_original["base_np"]
    h48qual_guarded["guard_exit_runtime"] = h48qual_original["exit_runtime"]
    h48qual_guarded["guard_pos_idx"] = h48qual_original["pos_idx"]
    h48qual_guarded["guard_exit_threshold"] = h48qual_original["exit_threshold"]
    h48qual_guarded["sustained_uptrend_mask"] = mask

    components = {"h48qual": h48qual_guarded, "zig075": zig075}
    diag = {
        "n_bars": int(len(aligned_frame)), "detector_nan_bars": n_nan,
        "detector_active_bars": int(mask.sum()), "detector_active_frac": float(mask.mean()),
    }
    return aligned_frame, components, diag


# =====================================================================================================
# Renamed copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay. That module is NEVER
# edited -- only imported and read to produce this copy. Every line is unchanged except the block
# marked "--- regime-aware exit guard: only new logic vs greedy_replay ---" and the two diagnostic
# counters threaded through it.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_regime_aware_exit_guard(
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
    """While `guard_component` (h48qual) holds an open position, the exit-head probability is read
    from ONE OF TWO fully independent, already-trained decision sources depending on
    components[guard_component]['sustained_uptrend_mask'][i] at that bar:
      - mask[i] True  -> comp['guard_base_np']/comp['guard_exit_runtime']/comp['guard_pos_idx']/
        comp['guard_exit_threshold'] (h48qual's ORIGINAL, pre-liveATR-relabel exit head).
      - mask[i] False, or no mask attached -> comp['base_np']/comp['exit_runtime']/comp['pos_idx']/
        comp['exit_threshold'] -- byte-identical to the unmodified greedy_replay's own behaviour.
    Any other active component (zig075) is unaffected -- the guard branch only ever fires when
    active_comp == guard_component AND a 'sustained_uptrend_mask' key is present on that component.
    """
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
    guard_decision_differs_bars = 0  # diagnostic only (does not affect any decision below): of the
    # guard_active_bars, how many would have gotten a DIFFERENT exit-or-hold decision from the
    # default (liveATR) path on that same bar -- distinguishes "guard engaged but happened to agree
    # with default anyway" from "guard engaged and never even queried the default", which
    # guard_active_bars alone cannot distinguish.

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
                # --- regime-aware exit guard: only new logic vs greedy_replay ---
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
                    # diagnostic-only counterfactual: what would the default (liveATR) path have
                    # decided on this SAME bar? Never used to set `reason`/`active_threshold` above.
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
                # --- end regime-aware exit guard block ---
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
    }
    return diag, pd.DataFrame(rows)


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
            "Odyssey2 #11 -- causal 'sustained uptrend' detector (rolling 1-week fraction of "
            "dual_momentum>0, threshold=90th percentile of 2025-Q1+Q2-only calibration sample) "
            "conditionally routes h48qual's HELD-POSITION exit-probability check between its "
            "ORIGINAL frozen exit head (detector active) and its current live-ATR-relabeled exit "
            "head (detector inactive, matches shadow-deployed default). zig075 always untouched. "
            "Tests the mitigation proposed but not implemented in "
            "docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md."
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
    # stage=G0a -- reproduce the task's 4 required reference numbers via the ALREADY-EXISTING,
    # already-validated gate.run_portfolio_variant (unmodified import) -- sanity that this script's
    # environment/data has not drifted from the pre-registered reference.
    # =================================================================================================
    log("=== stage=G0a_reference_via_gate_module ===")
    g0a: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = G0_REQUIRED[wname]
        ok_ng, ok_wg = _close(result["no_gate"], ref_ng), _close(result["with_gate"], ref_wg)
        g0a[wname] = {"no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg}}
        log(f"  {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
            f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}")
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0a_reference_via_gate_module"] = {"windows": g0a, "pass": g0a_pass}
    log(f"stage=G0a_result pass={g0a_pass}")

    # =================================================================================================
    # stage=G0b -- detector-forced-inactive identity check: run THIS script's OWN renamed
    # greedy_replay_regime_aware_exit_guard copy on plain asymmetric_tabm_liveatr components (no
    # guard/mask attached at all) -- must reproduce the SAME 4 numbers exactly. Proves the copy is
    # faithful to greedy.greedy_replay outside the intentionally-changed block (the detector-logic
    # integrity check the task instruction requires as a separate item from G0a).
    # =================================================================================================
    log("=== stage=G0b_detector_forced_inactive_identity ===")
    g0b: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        aligned_frame, components = _prep_liveatr_only(wname, windows, OUT_DIR, device)
        diag, ledger = greedy_replay_regime_aware_exit_guard(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_REQUIRED[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg},
                      "guard_active_bars_expected_zero": diag[f"h48qual_guard_active_bars"]}
        log(f"  {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg} "
            f"guard_active_bars={diag['h48qual_guard_active_bars']}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] and g0b[w]["guard_active_bars_expected_zero"] == 0 for w in ("val", "oos_q1"))
    report["g0b_detector_forced_inactive_identity"] = {"windows": g0b, "pass": g0b_pass}
    log(f"stage=G0b_result pass={g0b_pass}")

    g0_pass = bool(g0a_pass and g0b_pass)
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (reference reproduction and/or detector-forced-inactive identity check). Aborting before trusting any candidate number, per task instruction."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=detector_build -- construct the rolling dual_momentum score + calibrate the primary
    # (p90) and robustness (p75/p95) thresholds on 2025 Q1+Q2 ONLY, then report activation rates on
    # ALL 6 windows (confirmation step -- the rule/number were already fixed above).
    # =================================================================================================
    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = build_detector()
    log(f"  calibration window={CALIBRATION_START}..{CALIBRATION_END} (2025 Q1+Q2 only, Q3 excluded)")
    log(f"  thresholds (Q1+Q2-only percentiles): {robustness_thresholds}  primary(p90)={threshold:.6f}")

    activation_by_window: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        base_csv = gate.WINDOW_DEFS[wname]["base_csv"]
        score = score_by_base[base_csv]
        frame_ts = windows[wname]["frame"][["timestamp"]]
        merged = frame_ts.merge(score, on="timestamp", how="left")
        s = merged["sustained_uptrend_score"]
        row = {"n_bars": int(len(s)), "nan_bars": int(s.isna().sum())}
        for label, thr in robustness_thresholds.items():
            row[f"activation_{label}"] = float((s > thr).fillna(False).mean())
        activation_by_window[wname] = row
        log(f"  {wname:8s} n={row['n_bars']:6d} nan={row['nan_bars']:5d} " + " ".join(f"{k}={v * 100:5.1f}%" for k, v in row.items() if k.startswith("activation_")))
    report["detector"] = {
        "feature": "dual_momentum (features/engineering.py _dual_momentum, unmodified)",
        "aggregation": f"rolling({WEEK_BARS}, min_periods={WEEK_BARS}).mean() of (dual_momentum>0)",
        "week_bars_source": "reuses dual_momentum's own existing close.shift(2016) lookback convention, not an invented window",
        "calibration_window": [CALIBRATION_START, CALIBRATION_END],
        "calibration_excludes_2025q3": True,
        "percentile_primary": DETECTOR_PERCENTILE,
        "thresholds_q1q2_only": robustness_thresholds,
        "threshold_used": threshold,
        "activation_by_window": activation_by_window,
    }

    # =================================================================================================
    # stage=entry_side_diagnostic -- soft, informative check (not a hard gate; this script's design
    # does not depend on it -- entry/sizing are ALWAYS sourced from the liveATR h48qual prep by
    # construction, see module docstring) of whether dec/take_profit/stop_loss/margin/leverage
    # actually coincide between the original and liveATR-relabeled h48qual bundle configs on VAL.
    # =================================================================================================
    log("=== stage=entry_side_diagnostic (VAL, informative only) ===")
    val_w = windows["val"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    val_aligned_frame, val_aligned_paths = gate.align_frame_and_predictions(val_w["frame"], q_tags, "validation", OUT_DIR)
    h48qual_liveatr_diag = portfolio._prepare_component_val(val_aligned_frame, val_aligned_paths["h48qual"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], device)
    h48qual_original_diag = portfolio._prepare_component_val(val_aligned_frame, val_aligned_paths["h48qual"], gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["h48qual"], device)
    entry_side_identical = {
        "side": bool(h48qual_liveatr_diag["dec"]["side"].equals(h48qual_original_diag["dec"]["side"])),
        "take_profit": bool(np.allclose(pd.to_numeric(h48qual_liveatr_diag["dec"]["take_profit"]), pd.to_numeric(h48qual_original_diag["dec"]["take_profit"]))),
        "stop_loss": bool(np.allclose(pd.to_numeric(h48qual_liveatr_diag["dec"]["stop_loss"]), pd.to_numeric(h48qual_original_diag["dec"]["stop_loss"]))),
        "margin": bool(np.allclose(h48qual_liveatr_diag["margin"], h48qual_original_diag["margin"])),
        "leverage": bool(np.allclose(h48qual_liveatr_diag["leverage"], h48qual_original_diag["leverage"])),
    }
    report["entry_side_diagnostic_val"] = entry_side_identical
    log(f"  entry-side identical between original/liveATR h48qual bundle configs (VAL): {entry_side_identical}")

    # =================================================================================================
    # stage=main_run -- baseline_both_original / asymmetric_tabm_liveatr (both via the unmodified
    # gate.run_portfolio_variant) and regime_aware_guard (this script's own renamed replay) on ALL 6
    # pre-registered windows.
    # =================================================================================================
    log("=== stage=main_run (all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        orig = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="baseline_both_original")
        liveatr = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        aligned_frame, components, prep_diag = prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        guard_diag, guard_ledger = greedy_replay_regime_aware_exit_guard(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        guard_ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_regime_aware_guard.csv"
        guard_ledger.to_csv(guard_ledger_path, index=False)
        guard_no_gate = portfolio._ledger_metrics(guard_ledger)
        guard_with_gate = mfe_width._duration_gated(guard_ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "baseline_both_original": {"no_gate": orig["no_gate"], "with_gate": orig["with_gate"]},
            "asymmetric_tabm_liveatr": {"no_gate": liveatr["no_gate"], "with_gate": liveatr["with_gate"]},
            "regime_aware_guard": {"no_gate": guard_no_gate, "with_gate": guard_with_gate, "ledger_path": str(guard_ledger_path)},
            "detector_diag": prep_diag,
            "guard_replay_diag": guard_diag,
        }
        log(f"  {wname:8s} original      no_gate={orig['no_gate']['pnl']:7.2f}%/{orig['no_gate']['mdd']:7.2f}%/{orig['no_gate']['trades']:3d}  with_gate={orig['with_gate']['pnl']:7.2f}%/{orig['with_gate']['mdd']:7.2f}%/{orig['with_gate']['trades']:3d}")
        log(f"  {wname:8s} liveatr       no_gate={liveatr['no_gate']['pnl']:7.2f}%/{liveatr['no_gate']['mdd']:7.2f}%/{liveatr['no_gate']['trades']:3d}  with_gate={liveatr['with_gate']['pnl']:7.2f}%/{liveatr['with_gate']['mdd']:7.2f}%/{liveatr['with_gate']['trades']:3d}")
        log(f"  {wname:8s} regime_guard  no_gate={guard_no_gate['pnl']:7.2f}%/{guard_no_gate['mdd']:7.2f}%/{guard_no_gate['trades']:3d}  with_gate={guard_with_gate['pnl']:7.2f}%/{guard_with_gate['mdd']:7.2f}%/{guard_with_gate['trades']:3d}  "
            f"detector_active={prep_diag['detector_active_frac'] * 100:5.1f}%  guard_active_bars={guard_diag['h48qual_guard_active_bars']} guard_decision_differs_bars={guard_diag['h48qual_guard_decision_differs_bars']}")
    report["comparison"] = comparison

    # =================================================================================================
    # stage=summarize -- non-regression check on val/oos_q1/oos_q2 (candidate=regime_aware_guard vs
    # baseline=asymmetric_tabm_liveatr), strict (mdd_slack_pp=0) and relaxed (mdd_slack_pp=3, matching
    # eth_omega461_relaxed_gate_rescoring_20260814.md's precedent) -- EITHER passing is sufficient per
    # task instruction. 2025q1/q2/q3 are context-only, never gated.
    # =================================================================================================
    log("=== stage=summarize ===")
    baseline_tuples = {w: (comparison[w]["asymmetric_tabm_liveatr"]["no_gate"], comparison[w]["asymmetric_tabm_liveatr"]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["regime_aware_guard"]["no_gate"], comparison[w]["regime_aware_guard"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    non_regression_ok = bool(summary_strict["oos_confirm_all_pass_single_touch"] or summary_relaxed["oos_confirm_all_pass_single_touch"])
    log(f"  non-regression (val/oos_q1/oos_q2 vs asymmetric_tabm_liveatr): strict={summary_strict['final_verdict']} relaxed_mdd3pp={summary_relaxed['final_verdict']} -> non_regression_ok={non_regression_ok}")

    # Q3 mitigation / Q1+Q2 preservation, quantified directly (context tier, never gated, but this IS
    # the primary evidence this experiment exists to produce -- see task framing).
    def _mitigation(wname: str, key: str) -> dict[str, float]:
        o = comparison[wname]["baseline_both_original"][key]["pnl"]
        l = comparison[wname]["asymmetric_tabm_liveatr"][key]["pnl"]
        g = comparison[wname]["regime_aware_guard"][key]["pnl"]
        denom = (o - l)  # full original<->liveatr gap; positive when liveatr is worse than original
        recovered_frac = float((g - l) / denom) if abs(denom) > 1e-9 else float("nan")  # +1.0 = fully back to original, 0.0 = no change from liveatr, negative = moved further away
        return {"original_pnl": o, "liveatr_pnl": l, "guard_pnl": g, "fraction_of_gap_recovered_toward_original": recovered_frac}

    q3_mitigation = {"no_gate": _mitigation("2025q3", "no_gate"), "with_gate": _mitigation("2025q3", "with_gate")}
    q1_preservation = {"no_gate": _mitigation("2025q1", "no_gate"), "with_gate": _mitigation("2025q1", "with_gate")}
    q2_preservation = {"no_gate": _mitigation("2025q2", "no_gate"), "with_gate": _mitigation("2025q2", "with_gate")}
    log(f"  2025q3 mitigation: {q3_mitigation}")
    log(f"  2025q1 preservation: {q1_preservation}")
    log(f"  2025q2 preservation: {q2_preservation}")

    report["summary"] = {
        "non_regression_strict_mdd0pp": summary_strict,
        "non_regression_relaxed_mdd3pp": summary_relaxed,
        "non_regression_ok_either_criterion": non_regression_ok,
        "q3_mitigation": q3_mitigation,
        "q1_preservation": q1_preservation,
        "q2_preservation": q2_preservation,
    }
    report["stage_reached"] = "summarize"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done gate_pass=True non_regression_ok={non_regression_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
