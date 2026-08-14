#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey4 #3: evidence-signal SIZING modifier for zig075 SHORT (layered on the
locked Odyssey4 macro trend-veto, not a replacement for it).

=== Where this sits relative to what already exists ===
Odyssey4's regime-level entry veto (research_eth_omega461_zig075_short_entry_veto_sustained_
uptrend_20260814, CONFIRMED) blocks a zig075 SHORT entry outright when the week-scale sustained-
uptrend detector is active. That answers "is this the WRONG REGIME to short at all" (macro).
This script asks a DIFFERENT, complementary question for the bars the macro veto does NOT
already block: "does THIS SPECIFIC bar have independent microstructure evidence of an actual top
forming" (micro). It does not touch the macro veto's own logic and does not reject any entry the
macro veto lets through -- it only SCALES the position size (margin_fraction) of zig075 SHORT
entries that lack corroborating evidence, exactly the "rule-based post-entry-adjacent sizing
overlay" tier already scoped (but not implemented) by docs/experiments/eth_omega461_evidence_
signal_injection_research_20260814.md's own Tier-1 recommendation, and matching this repo's
existing precedent that post-entry/sizing-side interventions (Odyssey2/3 exit_head changes,
Odyssey4 macro veto) are the axis that has actually worked, unlike entry-side quality-gate
retraining (29 failed attempts catalogued in Odyssey1/2).

=== Why sizing, not another entry veto ===
Metalabeling/quality-gating on top of a picker's OWN internal confidence (quality_for_action,
dir_confidence, ensemble epistemic uncertainty -- Odyssey1 candidates A1-A9) failed every time
because the gating signal was derived from the SAME skill-less source as the picker itself --
circular. These evidence signals are the opposite: computed purely from OHLCV + taker_buy_base,
never seen by zig075's training pipeline in any form, independently validated for cross-regime
rank stability (Spearman 0.976 bottom / 0.924 top between a 2025-09~2026-02 downtrend window and
a disjoint 2026-03~07 calmer window -- docs/experiments/eth_evidence_signal_ranking_stability_
mar_jul_2026_20260814.md) -- a genuinely external, non-circular information source.

A cheap diagnostic (this session, not saved as its own artifact) cross-referenced TOP evidence-
signal presence against ACTUAL FILLED zig075 SHORT entries in the Odyssey3/4 baseline ledger:
  2025q1: evidence-confirmed entries win 100% (n=3) vs 57.1% unconfirmed (n=7)
  2025q2: evidence-confirmed entries win 66.7% (n=3) vs 30.8% unconfirmed (n=13)
  2025q3: evidence-confirmed entries win 12.5% (n=8) vs 11.1% unconfirmed (n=9) -- NO discrimination
Small samples, directional only -- but the pattern is mechanistically coherent and is exactly why
this is a SIZING signal layered inside the macro veto rather than a replacement for it: evidence
confirmation is a real quality signal precisely in the regimes where the macro veto does NOT
already intervene (Q1/Q2), and (correctly) provides no extra information in the pathological
regime the macro veto already handles (Q3) -- the two mechanisms operate on different bars by
construction, so stacking them is complementary rather than redundant.

=== The intervention (fixed BEFORE looking at any candidate outcome) ===
For a zig075 SHORT candidate entry that the macro veto does NOT block:
  - TOP evidence signal active at the signal bar (any of: orthogonal_combo top, taker_buy_climax,
    volume_wick_climax_high, vwap_extreme_high, momentum/cvd divergence top, bollinger_pctb
    extreme-high, liquidity_sweep_high -- the exact, UNMODIFIED formulas from analyze_eth_
    creative_reversal_evidence_signals_20260814.build_signals and analyze_eth_broad_evidence_
    signal_sweep_20260814.reversal_signals, minus A5 BTC-lead which needs data this repo's base
    CSVs don't carry) -> margin_fraction UNCHANGED (x1.0).
  - No TOP evidence signal active -> margin_fraction x0.5 (SIZE_MULT_UNCONFIRMED, a round,
    one-sided, NOT swept constant -- halving is the plainest possible "reduce, don't reject"
    convention, chosen before this script ever computed a candidate ledger; NOT tuned against
    zig075 P&L, which is exactly the target outcome this discipline exists to protect against).
Only the SIGN (SHORT), only zig075, only when the macro veto already lets the bar through.
zig075 LONG, h48qual (any side), TP/SL, exit logic, leverage caps, priority are all untouched.
This CANNOT change which trades happen or how they exit (TP/SL are price-move triggers,
independent of notional) -- only the PnL/MDD magnitude of the SAME trade set changes. That is a
strong structural safety property: this candidate cannot reproduce the repeated-reentry whipsaw
damage pattern, because it never removes or adds a trade.

=== Data source note ===
The evidence-signal research's own scripts source data/eth_5m_1year.csv, which only covers
2023-12-31..2026-02-17 -- missing the back half of OOS-Q1 and all of OOS-Q2. This script instead
computes the IDENTICAL formulas from sweep.BASE_2025 / sweep.BASE_2026 (the same base CSVs the
whole multiwindow gate lineage already uses), which carry taker_buy_base end-to-end through
2026-07-20 -- full 6-window coverage, verified before writing this script (`head`/`grep` on both
CSVs). Formulas are copied by unmodified function import, not reimplemented, so values are
identical wherever the two sources overlap.

=== Verification protocol ===
- G0: this script's own sizing-aware replay copy, with the sizing multiplier forced to 1.0
  everywhere (evidence check disabled), must reproduce the Odyssey4 baseline (veto-only) numbers
  EXACTLY on all 6 windows -- proves the copy is faithful outside the one new block, and doubles
  as this candidate's comparison baseline (same code path, ceteris paribus).
- Structural sanity check: candidate trade COUNT and entry/exit timestamps must be byte-identical
  to the Odyssey4 baseline on every window (only trade_return magnitude may differ) -- verified
  directly, not assumed.
- Candidate: sizing enabled, all 6 windows, single execution.
- Verdict: gate.summarize_multiwindow vs the Odyssey4 baseline (with_gate PnL/MDD non-worse),
  strict(0pp) + relaxed(3pp), VAL gate first then OOS-Q1+OOS-Q2 single touch. 2025 quarters stay
  context tier.

fresh_forward_bar_by_bar=true (evidence frame is a plain backward rolling/shift computation over
the full base CSV, never window-sliced; sizing reads the signal bar only).
trade_ledgers_used_as_input=false (compared against, never fed in). saved_parent_exit_timestamps_
used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / .env / any live or shadow script. Imports
existing modules read-only. No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
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
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as vetomod  # noqa: E402
import backtest_eth_slowk_williamsr_persistence_confluence_20260814 as osc  # noqa: E402
import analyze_eth_broad_evidence_signal_sweep_20260814 as broad_ev  # noqa: E402
import analyze_eth_creative_reversal_evidence_signals_20260814 as creative_ev  # noqa: E402
import analyze_eth_deep_evidence_signal_sweep_round2_20260814 as deep_ev  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_short_evidence_sizing_20260814_v2_masterranking"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
SIZE_MULT_UNCONFIRMED = 0.5  # fixed before any candidate outcome was computed; not swept.


def log(msg: str) -> None:
    print(f"[evidence_sizing] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


# =====================================================================================================
# Evidence-signal frame: unmodified formula imports, computed on the FULL base CSV (never
# window-sliced) so no window boundary ever truncates the rolling warm-up artificially.
# =====================================================================================================


def build_evidence_confirmed_mask(base_csv: Path) -> pd.DataFrame:
    """Master-ranking-filtered version (revision 2, same day): restricted to exactly the TOP-side
    "상위 5개" (top 5) that docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_
    20260814.md found held their rank across two INDEPENDENT, disjoint out-of-sample regimes
    (2025-09~2026-02 downtrend vs 2026-03~07 calm window, Spearman 0.924 top) -- liquidity_sweep,
    short_term_return_z (was MISSING from revision 1 -- a completeness bug fix, not tuning),
    orthogonal_combo, volume_wick_climax, taker_buy_climax (orderflow alone). Explicitly DROPPED
    vs revision 1: momentum_divergence (that doc found it WORSE than random out-of-sample, 0.88->
    0.77 top side -- explicitly flagged "don't trust divergence"), cvd_divergence and vwap_extreme
    (never confirmed in the stability doc's stated top-5/top-6, unranked in this summary), and
    bollinger_pctb (stable but rank 6, one below the "top 5" cut the user asked for). This signal-
    SET change is licensed by information external to zig075's P&L (the independently-published
    cross-regime ranking) -- SIZE_MULT_UNCONFIRMED itself is deliberately left unchanged from
    revision 1 (0.5, fixed before revision 1 ever saw a candidate outcome); re-deriving THAT number
    now, having just seen revision-1's zig075-specific result, would be exactly the post-hoc
    target-outcome tuning this project's discipline forbids."""
    raw = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    frame = osc.compute_indicators(raw).reset_index(drop=True)
    frame = broad_ev.add_broad_indicators(frame)
    frame = creative_ev.add_creative_indicators(frame)
    frame = deep_ev.add_short_term_and_patterns(frame)
    top_signals = creative_ev.build_signals(frame, "top")
    sig = pd.DataFrame({
        "orthogonal_combo": top_signals["orthogonal_combo (adaptive_OB AND taker_buy_climax)"].fillna(False),
        "taker_buy_climax": top_signals["taker_buy_climax (delta_z>=2)"].fillna(False),
        "volume_wick_climax_high": top_signals["volume_wick_climax_high (vol_z>=2, upper_wick>=.5)"].fillna(False),
        "liquidity_sweep_high": frame["sweep_high"].fillna(False),
        "short_term_return_z_high": (frame["ret3_z"] >= 2.5).fillna(False),
    })
    any_top = sig.any(axis=1)
    out = pd.DataFrame({"timestamp": frame["timestamp"], "evidence_confirmed": any_top})
    return out


def build_confirmed_by_base() -> dict[Path, pd.DataFrame]:
    return {sweep.BASE_2025: build_evidence_confirmed_mask(sweep.BASE_2025), sweep.BASE_2026: build_evidence_confirmed_mask(sweep.BASE_2026)}


def _confirmed_mask_for_frame(aligned_frame: pd.DataFrame, window_name: str, confirmed_by_base: dict[Path, pd.DataFrame]) -> np.ndarray:
    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    src = confirmed_by_base[base_csv]
    merged = aligned_frame[["timestamp"]].merge(src, on="timestamp", how="left")
    if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{window_name}: evidence-signal merge failed (row count/order mismatch)")
    return merged["evidence_confirmed"].fillna(False).to_numpy(dtype=bool)


# =====================================================================================================
# Renamed copy of vetomod.greedy_replay_entry_veto. Only new block: when a flat-state zig075 SHORT
# candidate SURVIVES the macro veto check, scale row_margin by SIZE_MULT_UNCONFIRMED unless the
# evidence-confirmed mask says otherwise. Nothing else changed vs the parent function.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_entry_veto_with_sizing(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
    sizing_enabled: bool = True,
) -> tuple[dict, pd.DataFrame]:
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
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars = 0
    sized_down_bars = 0
    sized_down_events: list[dict] = []

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

        for name in greedy.PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            veto_mask = comp.get("short_entry_veto_mask")
            if veto_mask is not None and side < 0 and bool(veto_mask[i]):
                veto_bars += 1
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            # --- evidence-signal sizing: only new logic vs greedy_replay_entry_veto ---
            evidence_mask = comp.get("evidence_confirmed_mask")
            if sizing_enabled and name == "zig075" and side < 0 and evidence_mask is not None and not bool(evidence_mask[i]):
                sized_down_bars += 1
                sized_down_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i])})
                row_margin *= SIZE_MULT_UNCONFIRMED
            # --- end evidence-signal sizing block ---
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
        "veto_bars": veto_bars,
        "sized_down_bars": sized_down_bars,
        "sized_down_events": sized_down_events,
    }
    return diag, pd.DataFrame(rows)


def _trade_identity_check(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    cols = ["entry_signal_i", "entry_i", "exit_i", "side", "source_component", "reason"]
    if len(baseline) != len(candidate):
        return {"identical": False, "reason": f"trade count differs: {len(baseline)} vs {len(candidate)}"}
    b, c = baseline[cols].reset_index(drop=True), candidate[cols].reset_index(drop=True)
    same = b.equals(c)
    return {"identical": bool(same), "n_trades": int(len(baseline))}


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
            "Odyssey4 #3 -- evidence-signal SIZING modifier for zig075 SHORT, layered inside the "
            "locked Odyssey4 macro trend-veto (not a replacement). When the macro veto does not "
            "already block a zig075 SHORT entry, margin_fraction is halved (SIZE_MULT_UNCONFIRMED="
            f"{SIZE_MULT_UNCONFIRMED}, fixed before any candidate outcome, not swept) unless an "
            "external, non-circular TOP microstructure evidence signal (OHLCV+taker_buy_base only, "
            "never seen by zig075's own training) confirms at the signal bar. Cannot change which "
            "trades occur or how they exit -- TP/SL are price-move triggers independent of notional."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "size_mult_unconfirmed": SIZE_MULT_UNCONFIRMED,
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=detector_build (macro veto, reused verbatim from Odyssey4) ===")
    score_by_base, _th, threshold = guard.build_detector()
    log(f"  macro veto threshold (p90, locked)={threshold:.10f}")

    log("=== stage=evidence_signal_build (full base CSVs, unmodified formula imports) ===")
    confirmed_by_base = build_confirmed_by_base()
    for base_csv, df in confirmed_by_base.items():
        log(f"  {base_csv.name}: n={len(df)}  evidence_confirmed_frac={df['evidence_confirmed'].mean() * 100:.2f}%")

    prepared: dict[str, tuple] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}

    log("=== stage=G0_odyssey4_baseline_reproduction (sizing disabled, all 6 windows) ===")
    g0: dict[str, Any] = {}
    g0_pass = True
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
        components = vetomod._attach_veto_mask(components, mask)
        prepared[wname] = (aligned_frame, components, prep_diag)
        diag, ledger = greedy_replay_entry_veto_with_sizing(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, sizing_enabled=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = vetomod.G0_ODYSSEY3[wname]
        # compare against the Odyssey4 CANDIDATE numbers (veto applied), not the Odyssey3 no-veto numbers
        ok = True  # placeholder, real check below via re-derivation
        baseline_runs[wname] = {"no_gate": no_gate, "with_gate": with_gate, "ledger": ledger}
        g0[wname] = {"no_gate": no_gate, "with_gate": with_gate, "veto_bars": diag["veto_bars"], "sized_down_bars_expected_zero": diag["sized_down_bars"]}
        if diag["sized_down_bars"] != 0:
            g0_pass = False
        log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} sized_down_bars={diag['sized_down_bars']}")

    # Independently confirm baseline_runs matches the ALREADY-PUBLISHED Odyssey4 report numbers
    # (val/oos_q1/oos_q2 from that experiment's report.json summary; 2025q* from its comparison
    # table) -- re-typed once as literal reference values, not re-derived from another module.
    ODYSSEY4_REFERENCE_WITH_GATE = {
        "2025q1": (44.98, -20.62, 20), "2025q2": (5.62, -23.59, 19), "2025q3": (20.17, -19.72, 17),
        "val": (77.31, -21.76, 26), "oos_q1": (67.25, -15.48, 19), "oos_q2": (-12.69, -20.76, 10),
    }
    for wname, (pnl, mdd, trades) in ODYSSEY4_REFERENCE_WITH_GATE.items():
        actual = baseline_runs[wname]["with_gate"]
        ok = abs(actual["pnl"] - pnl) <= G0_TOLERANCE_PP and abs(actual["mdd"] - mdd) <= G0_TOLERANCE_PP and int(actual["trades"]) == trades
        g0[wname]["matches_published_odyssey4"] = bool(ok)
        g0_pass = g0_pass and ok
        log(f"  {wname:8s} matches published Odyssey4 report: {ok}")

    report["g0"] = {"windows": g0, "pass": g0_pass}
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (Odyssey4 baseline reproduction and/or sizing-disabled identity check). Aborting."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    log("=== stage=candidate_run (sizing enabled, all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = prepared[wname]
        ev_mask = _confirmed_mask_for_frame(aligned_frame, wname, confirmed_by_base)
        comp2 = dict(components)
        zig = dict(comp2["zig075"])
        zig["evidence_confirmed_mask"] = ev_mask
        comp2["zig075"] = zig
        diag, ledger = greedy_replay_entry_veto_with_sizing(aligned_frame, comp2, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, sizing_enabled=True)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_evidence_sizing.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        identity = _trade_identity_check(baseline_runs[wname]["ledger"], ledger)
        zig_short_bars = int(((pd.to_numeric(components["zig075"]["dec"]["side"], errors="raise") < 0)).sum())
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "odyssey4_baseline": {"no_gate": baseline_runs[wname]["no_gate"], "with_gate": baseline_runs[wname]["with_gate"]},
            "evidence_sizing": {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)},
            "evidence_confirmed_frac_over_zig075_short_signal_bars": float(ev_mask[pd.to_numeric(components["zig075"]["dec"]["side"], errors="raise").to_numpy() < 0].mean()) if zig_short_bars else None,
            "sized_down_bars": diag["sized_down_bars"],
            "trade_identity_vs_baseline": identity,
        }
        b_ng, b_wg = baseline_runs[wname]["no_gate"], baseline_runs[wname]["with_gate"]
        log(f"  {wname:8s} baseline  no_gate={b_ng['pnl']:7.2f}%/{b_ng['mdd']:7.2f}%/{b_ng['trades']:3d}  with_gate={b_wg['pnl']:7.2f}%/{b_wg['mdd']:7.2f}%/{b_wg['trades']:3d}")
        log(f"  {wname:8s} sizing    no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
            f"sized_down_bars={diag['sized_down_bars']}  trade_identity={identity['identical']}")
    report["comparison"] = comparison

    if not all(comparison[w]["trade_identity_vs_baseline"]["identical"] for w in gate.ALL_WINDOWS):
        report["stage_reached"] = "candidate_run"
        report["gate_pass"] = False
        report["note"] = "trade identity check failed -- sizing changed which trades occurred, violating the structural safety property. Aborting before trusting the verdict."
        _write_report(report)
        log("stage=ABORT trade identity check failed")
        return 1

    log("=== stage=summarize (vs Odyssey4 baseline) ===")
    baseline_tuples = {w: (baseline_runs[w]["no_gate"], baseline_runs[w]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["evidence_sizing"]["no_gate"], comparison[w]["evidence_sizing"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    val_gate_pass_strict = bool(summary_strict["rows"]["val"]["with_gate_pass"])
    log(f"  VAL gate: strict={val_gate_pass_strict}")
    log(f"  OOS single touch: strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']}")

    report["summary"] = {
        "val_gate_pass_strict": val_gate_pass_strict,
        "multiwindow_strict_mdd0pp": summary_strict,
        "multiwindow_relaxed_mdd3pp": summary_relaxed,
    }
    report["stage_reached"] = "summarize"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']} val_strict={val_gate_pass_strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
