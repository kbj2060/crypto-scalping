#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #18 (evidence-signal injection, Candidate C from docs/experiments/
eth_omega461_evidence_signal_injection_research_20260814.md): a short-position counter-evidence
exit overlay for h48qual, targeting the same 2025-Q3 turnover-acceleration vulnerability as
Odyssey2 #11 (docs/experiments/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.md) but
via a DIFFERENT, complementary signal source and granularity.

=== Precedent this candidate is INCREMENTAL to, not a re-derivation of ===
docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md
established the mechanism: the shadow-deployed baseline (asymmetric_tabm_liveatr) is a "turnover
accelerator" for h48qual -- it shortens average hold 2-3x every 2025 quarter. In Q1/Q2 this barely
inflates trade count and helps PnL; in Q3 (2025's only sustained LOW-noise uptrend) it inflates
h48qual trade count 8->18 (2.25x, all 18 SHORT) because freed-up slots keep re-triggering an
already-known direction-unbiased short signal with too little chop to stop it -- portfolio no_gate
PnL -9.73%->-46.26% (4.7x worse).

Odyssey2 #11 already tested ONE mitigation for this: a WEEKLY-scale regime detector
(rolling(2016bar) fraction of dual_momentum>0, threshold=P90 of a 2025-Q1+Q2-only calibration
sample) that routes h48qual's held-position exit check back to its ORIGINAL (pre-liveATR) exit
head whenever "sustained uptrend" is detected -- recovering 82.4% of the Q3 no_gate gap, VAL/
OOS-Q1/OOS-Q2 non-regressing (mostly byte-identical, since the detector essentially never fires in
those windows). This candidate (#18) is a DIFFERENT, orthogonal signal source: bar-level order-flow
exhaustion evidence (not a slow structural trend gauge), from a genuinely independent evidence-study
lineage (docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md,
Spearman-stable across two structurally different 5-month windows). #18 must be read and reported
relative to #11 -- it is a complementary candidate for finer-grained coverage #11's weekly detector
may miss (localized selling-exhaustion inside stretches that are not obviously trending on a weekly
scale), not a re-invention of the same idea.

=== What THIS script tests ===
A causal per-bar "bottom evidence" signal (orthogonal_combo: adaptive Williams-%R/Slow-%K both in
their own rolling-864-bar bottom decile AND a same-bar net-aggressive-sell-volume z-score <= -2,
formula reused UNMODIFIED from scripts/analyze_eth_creative_reversal_evidence_signals_20260814.py
build_signals()/add_creative_indicators() and scripts/backtest_eth_slowk_williamsr_persistence_
confluence_20260814.py compute_indicators() -- no new formula, no re-tuned threshold, exactly the
already cross-window-validated definition) is applied as a FORCED-EXIT trigger while h48qual holds
an OPEN SHORT position: if the signal fires on bar i, the position exits on bar i regardless of the
exit_head's own probability (reason="evidence_veto"), checked in the SAME priority slot as
exit_head (after take_profit/stop_loss, i.e. hard barriers still take priority). LONG h48qual
positions and zig075 (both sides) are completely untouched -- this candidate is scoped exactly to
the failure mode it targets ("bad SHORT re-entries churning in a clean uptrend").

Chose orthogonal_combo alone (not "OR taker_sell_climax") as the trigger: taker_sell_climax
(delta_z<=-2 alone) is a strict SUPERSET of orthogonal_combo (delta_z<=-2 AND both oscillators in
their bottom decile) -- ORing them is mathematically identical to using taker_sell_climax alone,
which would silently swap in the noisier/lower-precision signal (34.4% precision, lift 2.75-3.12x)
for a hard-forced-exit action instead of the strongest, most stable, most conservative one
(orthogonal_combo: 43.9% precision, lift 3.51-3.92x bottom, rare/high-confidence -- 0.66% of 2025
bars per the dry-run check recorded in this session). No new hyperparameter was introduced or
searched: both the oscillator-percentile threshold (0.10, i.e. bottom/top decile) and the
order-flow z-threshold (2.0) are the exact values already validated across two independent 5-month
windows in the evidence-study lineage, not chosen or tuned by this script.

fresh_forward_bar_by_bar=true (renamed greedy_replay copy below is a single causal forward pass, i
increasing, only bar i and already-closed history used at bar i; compute_indicators/
add_creative_indicators are rolling/shift-only, no negative shift, verified by direct inspection of
both functions before reuse here). trade_ledgers_used_as_input=false (ledgers are write-only
outputs). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module --
eth_omega461_multiwindow_confirmation_gate_20260814.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
replay_omega4_6_1_greedy_router_20260706.py, research_eth_omega461_exit_sweep_20260721.py,
research_eth_omega461_live_sltp_mfe_width_20260813.py,
train_eval_omega4_2_risk_sidecar_20260622.py, analyze_eth_creative_reversal_evidence_signals_
20260814.py, backtest_eth_slowk_williamsr_persistence_confluence_20260814.py are all imported and
read only. No retraining, no GPU (DEVICE=cpu, matching every script in this lineage).
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
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_evidence_veto_exit_overlay_20260814"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05

# =====================================================================================================
# Signal constants -- BOTH values reused unmodified from the already cross-window-validated
# evidence-study lineage (docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_
# 20260814.md), not re-derived or tuned here. No calibration sample needed (unlike Odyssey2 #11's
# dual_momentum detector, which had to pick a NEW threshold) -- the trigger IS the already-validated
# orthogonal_combo definition, verbatim.
# =====================================================================================================
OSCILLATOR_PERCENTILE_WINDOW = 864  # scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py
OSCILLATOR_DECILE = 0.10
DELTA_Z_WINDOW = 288  # scripts/analyze_eth_creative_reversal_evidence_signals_20260814.py
DELTA_Z_THRESHOLD = -2.0

G0_REQUIRED = {
    "val": ({"pnl": 46.59, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
}
REFERENCE_2025Q_BASELINE_BOTH_ORIGINAL_WITH_GATE = gate.REFERENCE_2025Q_BASELINE_BOTH_ORIGINAL_WITH_GATE


def log(msg: str) -> None:
    print(f"[evidence_veto_exit] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP, check_trades: bool = True) -> bool:
    ok = bool(abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp)
    if check_trades and "trades" in expected:
        ok = ok and int(actual["trades"]) == int(expected["trades"])
    return ok


# =====================================================================================================
# Signal construction -- reuses compute_indicators (Williams-%R/Slow-%K rolling-864 percentile) and
# add_creative_indicators (delta_z, net aggressive-sell/buy volume z-score) UNMODIFIED, computed on
# the FULL base CSV per year (matching Odyssey2 #11's own "never window-by-window" discipline so
# windows starting mid-year are never artificially NaN-truncated at their own start).
# =====================================================================================================


def _evidence_veto_score(base_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ind = compute_indicators(df)
    ind = add_creative_indicators(ind)
    bottom_orthogonal_combo = (ind["p_fast"] <= OSCILLATOR_DECILE) & (ind["p_slow"] <= OSCILLATOR_DECILE) & (ind["delta_z"] <= DELTA_Z_THRESHOLD)
    out = ind[["timestamp"]].copy()
    out["evidence_veto"] = bottom_orthogonal_combo.fillna(False).to_numpy(dtype=bool)
    out["delta_z_nan"] = ind["delta_z"].isna().to_numpy()
    return out


def build_signal() -> dict[Path, pd.DataFrame]:
    score_2025 = _evidence_veto_score(sweep.BASE_2025)
    score_2026 = _evidence_veto_score(sweep.BASE_2026)
    return {sweep.BASE_2025: score_2025, sweep.BASE_2026: score_2026}


def _veto_mask_for_frame(aligned_frame: pd.DataFrame, window_name: str, score_by_base: dict[Path, pd.DataFrame]) -> tuple[np.ndarray, int]:
    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    score = score_by_base[base_csv]
    merged = aligned_frame[["timestamp"]].merge(score, on="timestamp", how="left")
    if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{window_name}: evidence-veto score merge failed (row count/order mismatch)")
    n_nan = int(merged["delta_z_nan"].sum())
    mask = merged["evidence_veto"].fillna(False).to_numpy(dtype=bool)
    return mask, n_nan


# =====================================================================================================
# Component preparation: h48qual sourced ENTIRELY from the current shadow-deployed liveATR bundle
# (entry/TP/SL/sizing/exit-head all unchanged) with the evidence-veto mask attached as a side-channel.
# zig075 untouched. This candidate never sources anything from the pre-liveATR bundle (unlike #11,
# which switches between two trained exit heads) -- it only ever ADDS a forced-exit trigger on top of
# the existing liveATR decision path for h48qual SHORT positions.
# =====================================================================================================


def _prep_liveatr_only(window_name: str, windows: dict[str, Any], out_dir: Path, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    components = {name: prep(aligned_frame, aligned_paths[name], cfg, device) for name, cfg in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR.items()}
    return aligned_frame, components


def prepare_evidence_veto_components(
    window_name: str, windows: dict[str, Any], score_by_base: dict[Path, pd.DataFrame], out_dir: Path, device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component

    h48qual_liveatr = prep(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], device)
    zig075 = prep(aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)

    mask, n_nan = _veto_mask_for_frame(aligned_frame, window_name, score_by_base)

    h48qual_guarded = dict(h48qual_liveatr)
    h48qual_guarded["evidence_veto_mask"] = mask

    components = {"h48qual": h48qual_guarded, "zig075": zig075}
    diag = {"n_bars": int(len(aligned_frame)), "evidence_score_nan_bars": n_nan,
            "veto_active_bars": int(mask.sum()), "veto_active_frac": float(mask.mean())}
    return aligned_frame, components, diag


# =====================================================================================================
# Renamed copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay. That module is NEVER
# edited -- only imported and read to produce this copy. Every line is unchanged except the block
# marked "--- evidence veto: only new logic vs greedy_replay ---" and the two diagnostic counters
# threaded through it.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_evidence_veto_exit(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    veto_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
    """While `veto_component` (h48qual) holds an OPEN SHORT position (pos<0), if
    components[veto_component]['evidence_veto_mask'][i] is True on bar i, the position is force-
    exited on bar i with reason="evidence_veto" -- checked in the same priority slot as exit_head
    (i.e. AFTER take_profit/stop_loss/trailing, so hard barriers are never overridden). LONG
    positions on veto_component, and any other active component (zig075), are unaffected -- the veto
    branch only ever fires when active_comp == veto_component AND pos < 0 AND a 'evidence_veto_mask'
    key is present on that component.
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
    trailing_enabled = False
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    veto_short_hold_bars = 0
    veto_active_bars = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == veto_component and pos < 0:
                veto_short_hold_bars += 1
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
                pass  # trailing not used by this candidate (mirrors greedy_replay default off)
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                # --- evidence veto: only new logic vs greedy_replay ---
                use_veto = False
                mask = comp.get("evidence_veto_mask")
                if active_comp == veto_component and pos < 0 and mask is not None and bool(mask[i]):
                    use_veto = True
                if use_veto:
                    veto_active_bars += 1
                    reason = "evidence_veto"
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if prob >= float(comp["exit_threshold"]):
                        reason = "exit_head"
                # --- end evidence veto block ---
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
        f"{veto_component}_short_hold_bars": veto_short_hold_bars,
        f"{veto_component}_veto_active_bars": veto_active_bars,
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
            "Odyssey2 #18 -- short-position counter-evidence exit overlay for h48qual. While "
            "h48qual holds an OPEN SHORT, orthogonal_combo (adaptive Williams-%R/Slow-%K bottom "
            "decile AND net-aggressive-sell-volume z<=-2, formula reused unmodified from the "
            "evidence-study lineage, no new threshold) forces an immediate exit (reason=evidence_"
            "veto), checked in the same priority slot as exit_head. LONG h48qual and zig075 (both "
            "sides) are untouched. Targets the same 2025-Q3 turnover-acceleration vulnerability as "
            "Odyssey2 #11 (dual_momentum weekly regime guard) via a different, complementary "
            "bar-level order-flow signal -- must be read as incremental to #11, not a re-derivation."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "trigger_signal": "orthogonal_combo (bottom side only) -- (p_fast<=0.10)&(p_slow<=0.10)&(delta_z<=-2.0)",
        "trigger_scope": "h48qual SHORT positions only; LONG h48qual and zig075 (both sides) untouched",
    }

    # =================================================================================================
    # stage=load_windows
    # =================================================================================================
    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=G0a -- reproduce reference numbers via the already-validated gate module (unmodified import)
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
    # stage=G0b -- veto-forced-inactive identity check: this script's own renamed replay copy on
    # plain asymmetric_tabm_liveatr components (no mask attached at all) must reproduce the SAME 4
    # reference numbers exactly -- proves the copy is faithful to greedy.greedy_replay outside the
    # intentionally-changed block.
    # =================================================================================================
    log("=== stage=G0b_veto_forced_inactive_identity ===")
    g0b: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        aligned_frame, components = _prep_liveatr_only(wname, windows, OUT_DIR, device)
        diag, ledger = greedy_replay_evidence_veto_exit(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_REQUIRED[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg},
                      "veto_active_bars_expected_zero": diag["h48qual_veto_active_bars"]}
        log(f"  {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg} "
            f"veto_active_bars={diag['h48qual_veto_active_bars']}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] and g0b[w]["veto_active_bars_expected_zero"] == 0 for w in ("val", "oos_q1"))
    report["g0b_veto_forced_inactive_identity"] = {"windows": g0b, "pass": g0b_pass}
    log(f"stage=G0b_result pass={g0b_pass}")

    g0_pass = bool(g0a_pass and g0b_pass)
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (reference reproduction and/or veto-forced-inactive identity check). Aborting before trusting any candidate number."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=signal_build -- construct the evidence-veto signal (no calibration needed -- reuses the
    # already-validated orthogonal_combo definition verbatim), report activation rate on ALL 6 windows.
    # =================================================================================================
    log("=== stage=signal_build ===")
    score_by_base = build_signal()

    activation_by_window: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        base_csv = gate.WINDOW_DEFS[wname]["base_csv"]
        score = score_by_base[base_csv]
        frame_ts = windows[wname]["frame"][["timestamp"]]
        merged = frame_ts.merge(score, on="timestamp", how="left")
        row = {"n_bars": int(len(merged)), "nan_bars": int(merged["delta_z_nan"].sum()),
               "activation_rate": float(merged["evidence_veto"].fillna(False).mean())}
        activation_by_window[wname] = row
        log(f"  {wname:8s} n={row['n_bars']:6d} nan={row['nan_bars']:5d} activation_rate={row['activation_rate'] * 100:5.2f}%")
    report["signal"] = {
        "definition": "orthogonal_combo bottom: (p_fast<=0.10)&(p_slow<=0.10)&(delta_z<=-2.0), reused unmodified from analyze_eth_creative_reversal_evidence_signals_20260814.py / backtest_eth_slowk_williamsr_persistence_confluence_20260814.py",
        "no_calibration_needed": "threshold values are the already cross-window-validated evidence-study definition, not tuned by this script",
        "activation_by_window": activation_by_window,
    }

    # =================================================================================================
    # stage=main_run -- baseline_both_original / asymmetric_tabm_liveatr (both via the unmodified
    # gate.run_portfolio_variant) and evidence_veto_guard (this script's own renamed replay) on ALL 6
    # pre-registered windows.
    # =================================================================================================
    log("=== stage=main_run (all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        orig = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="baseline_both_original")
        liveatr = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        aligned_frame, components, prep_diag = prepare_evidence_veto_components(wname, windows, score_by_base, OUT_DIR, device)
        veto_diag, veto_ledger = greedy_replay_evidence_veto_exit(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        veto_ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_evidence_veto_guard.csv"
        veto_ledger.to_csv(veto_ledger_path, index=False)
        veto_no_gate = portfolio._ledger_metrics(veto_ledger)
        veto_with_gate = mfe_width._duration_gated(veto_ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "baseline_both_original": {"no_gate": orig["no_gate"], "with_gate": orig["with_gate"]},
            "asymmetric_tabm_liveatr": {"no_gate": liveatr["no_gate"], "with_gate": liveatr["with_gate"]},
            "evidence_veto_guard": {"no_gate": veto_no_gate, "with_gate": veto_with_gate, "ledger_path": str(veto_ledger_path)},
            "signal_diag": prep_diag,
            "guard_replay_diag": veto_diag,
        }
        log(f"  {wname:8s} original      no_gate={orig['no_gate']['pnl']:7.2f}%/{orig['no_gate']['mdd']:7.2f}%/{orig['no_gate']['trades']:3d}  with_gate={orig['with_gate']['pnl']:7.2f}%/{orig['with_gate']['mdd']:7.2f}%/{orig['with_gate']['trades']:3d}")
        log(f"  {wname:8s} liveatr       no_gate={liveatr['no_gate']['pnl']:7.2f}%/{liveatr['no_gate']['mdd']:7.2f}%/{liveatr['no_gate']['trades']:3d}  with_gate={liveatr['with_gate']['pnl']:7.2f}%/{liveatr['with_gate']['mdd']:7.2f}%/{liveatr['with_gate']['trades']:3d}")
        log(f"  {wname:8s} evidence_veto no_gate={veto_no_gate['pnl']:7.2f}%/{veto_no_gate['mdd']:7.2f}%/{veto_no_gate['trades']:3d}  with_gate={veto_with_gate['pnl']:7.2f}%/{veto_with_gate['mdd']:7.2f}%/{veto_with_gate['trades']:3d}  "
            f"veto_active_bars={veto_diag['h48qual_veto_active_bars']} short_hold_bars={veto_diag['h48qual_short_hold_bars']}")
    report["comparison"] = comparison

    # =================================================================================================
    # stage=summarize -- non-regression check on val/oos_q1/oos_q2 (candidate=evidence_veto_guard vs
    # baseline=asymmetric_tabm_liveatr), strict (mdd_slack_pp=0) and relaxed (mdd_slack_pp=3), matching
    # the multiwindow gate module's own pre-registered criterion. 2025q1/q2/q3 are context-only.
    # =================================================================================================
    log("=== stage=summarize ===")
    baseline_tuples = {w: (comparison[w]["asymmetric_tabm_liveatr"]["no_gate"], comparison[w]["asymmetric_tabm_liveatr"]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["evidence_veto_guard"]["no_gate"], comparison[w]["evidence_veto_guard"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    non_regression_ok = bool(summary_strict["oos_confirm_all_pass_single_touch"] or summary_relaxed["oos_confirm_all_pass_single_touch"])
    log(f"  non-regression (val/oos_q1/oos_q2 vs asymmetric_tabm_liveatr): strict={summary_strict['final_verdict']} relaxed_mdd3pp={summary_relaxed['final_verdict']} -> non_regression_ok={non_regression_ok}")

    # Q3 mitigation / Q1+Q2 preservation, quantified the same way as Odyssey2 #11's own _mitigation
    # helper (same formula, so the two candidates' recovery fractions are directly comparable).
    def _mitigation(wname: str, key: str) -> dict[str, float]:
        o = comparison[wname]["baseline_both_original"][key]["pnl"]
        l = comparison[wname]["asymmetric_tabm_liveatr"][key]["pnl"]
        g = comparison[wname]["evidence_veto_guard"][key]["pnl"]
        denom = (o - l)
        recovered_frac = float((g - l) / denom) if abs(denom) > 1e-9 else float("nan")
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
        "comparison_note": "recovered_frac formula identical to Odyssey2 #11's _mitigation helper -- directly comparable to #11's reported 82.4% (no_gate) Q3 recovery.",
    }
    report["stage_reached"] = "summarize"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done gate_pass=True non_regression_ok={non_regression_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
