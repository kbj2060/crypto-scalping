#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 post-entry literature scouting (docs/experiments/
eth_omega461_post_entry_literature_scouting_20260814.md), rank-1 candidate ("저비용 실전판" row of
that document's priority table): a "queue pressure" post-processing exit rule for h48qual's
exit_head, motivated by the Gittins-index retirement formulation (Dhankhar/Mishra/Bodas,
arXiv:2405.01157) -- the "retirement value" of holding a position is approximated, without any
retraining, by whether the OTHER shared-slot component (zig075) currently wants the slot.

Neither h48qual's nor zig075's exit_head MODEL is changed (unlike Odyssey2 #4/GBDT and #5/TCN,
which swapped the exit_head model itself and both failed the component-level gate by exiting
"unconditionally" faster). h48qual keeps the exact same TabM live-ATR-relabeled exit head that is
the current confirmed Odyssey2 baseline (tmp/causal_regen_20260516/
eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/true_3head_tabm_bundle.pt). Only
h48qual's fixed EXIT_THRESHOLD=0.95 (trading_bot_modules/omega4_6_1_live.py's live constant) is
made CONDITIONAL: on any bar h48qual holds the shared position slot, if zig075 has its own
dir_action != CASH AND quality_for_action >= zig075's live quality_threshold(0.75) at that same bar
(i.e. zig075 would take a trade this bar if the slot were free -- "queue pressure"), h48qual's
exit_head threshold is lowered to a candidate value (swept over {0.80, 0.85, 0.90}); otherwise
0.95 is used exactly as today. zig075's own decisions/exit logic are never touched or read as an
input to ITS OWN exits -- zig075 is only ever the SOURCE of the pressure signal used to modulate
h48qual's threshold.

=== Why a renamed copy of greedy_replay, not the GBDT-style duck-typing wrapper ===
GBDT/TCN (research_eth_omega461_gbdt_exit_head_val_20260813.py /
research_eth_omega461_tcn_exit_head_val_20260813.py) both replaced the h48qual exit_head MODEL
object, so they could inject at existing call sites (directly for GBDT since
_predict_exit_prob_one's single-row contract already fit; via a renamed
greedy_replay_windowed copy for TCN, which needed a window instead of a single row). This
experiment does not replace any model -- it changes which THRESHOLD is compared against the
existing TabM model's probability, and that decision depends on a SECOND component's state
(zig075) at the same bar, information greedy_replay's per-bar loop already has in `components`
but never looks at while a position is open. Per the coordinator's explicit instruction, the fix
is a renamed, logic-preserving COPY of replay_omega4_6_1_greedy_router_20260706.greedy_replay --
greedy_replay_queue_pressure below -- with the exit-head threshold selection block (and ONLY that
block) made conditional. replay_omega4_6_1_greedy_router_20260706.py itself is never edited, only
imported and read to produce this copy (verified via `git diff` before/after, see the companion
experiment doc's compliance section).

=== G0 self-check (runs first, unconditionally) ===
Re-derives the two ALREADY-PUBLISHED portfolio-level reference numbers (baseline_both_original
36.82/-24.34/29, asymmetric_h48qual_liveatr_zig075_original 46.59/-21.70/35) through this exact
script's import chain via research_eth_omega461_exit_head_portfolio_asymmetric_20260813.run_variant
(100% pre-existing code, calls the UNMODIFIED greedy.greedy_replay) -- same pattern priority #4/#5
used. If G0 fails, this script aborts before computing any queue-pressure number.

=== G0b harness self-consistency (this script's own addition, beyond the task's literal G0 scope) ===
Because G0 only exercises the UNMODIFIED greedy_replay, it cannot catch a copy-paste bug introduced
while writing greedy_replay_queue_pressure. As a second, cheap self-check, this script also runs
greedy_replay_queue_pressure itself in a DEGENERATE mode (queue_pressure_threshold=0.95, i.e.
identical to the always-on baseline threshold) and asserts it reproduces the exact same
46.59/-21.70/35 reference -- proving the copy is behaviourally faithful outside the one block that
was intentionally changed. This run is also where the "queue pressure frequency under the
baseline's own actual holding pattern" diagnostic is read off (unconfounded by the intervention
itself, since threshold=0.95 everywhere makes the holding pattern identical to the real baseline).

=== VAL gate: PORTFOLIO level only ===
Unlike Odyssey2 #4 (GBDT) / #5 (TCN), there is no component-level gate here. "Queue pressure" is
only DEFINED in the shared-slot portfolio context (it is a function of zig075's state, which does
not exist in a component-standalone h48qual-only replay) -- so a standalone-component counterfactual
for this specific intervention has no meaning, and the coordinator's instructions restrict the gate
to portfolio PnL+MDD non-worse vs the TabM live-ATR baseline (threshold always 0.95).

fresh_forward_bar_by_bar=true (greedy_replay_queue_pressure is a single causal forward pass, i
increasing, only bar i and already-closed history used at bar i -- the queue-pressure mask read at
bar i is zig075's OWN bar-i prediction, never a future row). trade_ledgers_used_as_input=false
(ledgers are written-only outputs). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. direction_head/quality_head/encoder are frozen and unchanged for
BOTH components (bit-identical to the current confirmed baseline) -- only h48qual's exit_head
THRESHOLD (never its model weights) is made conditional. VAL window 2025-10-01..2025-12-31
(research_eth_omega461_exit_sweep_20260721.VAL_START/VAL_END). OOS (2026-01-01..2026-03-31) is
opened once, only if a VAL candidate passes the gate.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT touch zig075's own exit logic/model/threshold.
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
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_queue_pressure_exit_threshold_20260814"
CANDIDATE_THRESHOLDS = [0.80, 0.85, 0.90]
G0_TOLERANCE_PP = 0.05  # percentage points on pnl/mdd; trades must match exactly (deterministic replay)

G0_REFERENCE = {
    # Published in docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md
    # ("후속 2 -- 포트폴리오 레벨 검증"), reused as the required G0 reference per this experiment's
    # coordinator instructions (portfolio level only).
    "baseline_both_original": {"pnl": 36.82, "mdd": -24.34, "trades": 29},
    "asymmetric_tabm_liveatr": {"pnl": 46.59, "mdd": -21.70, "trades": 35},
}
# Published in the same doc's "후속 3 -- OOS 단일 확인" section. Used only as a cross-check on this
# script's own OOS baseline recomputation (see stage=OOS_single_touch below), not as an input.
OOS_BASELINE_REFERENCE = {"pnl": 93.27, "mdd": -15.48, "trades": 24}
OOS_CAVEAT_TEXT = (
    "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by the TabM-liveATR baseline "
    "and the queue-pressure candidate here (queue pressure only modulates h48qual's exit_head "
    "EXIT_THRESHOLD -- it never touches quality_threshold or the direction/quality heads, which are "
    "frozen in both variants), was itself OOS-pnl-primary selected against a frame spanning "
    "2026-01-01..2026-02-28 -- the first two of this OOS window's three months (see "
    "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
    "relative comparison (candidate vs baseline within this run) remains meaningful because both "
    "share the identical contaminated entry-selection layer; the absolute OOS PnL/MDD figures below "
    "are not clean unbiased forward performance and must not be over-interpreted as such."
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _close(actual: dict[str, Any], expected: dict[str, Any], tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


@torch.no_grad()
def greedy_replay_queue_pressure(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    queue_pressure_component: str = "h48qual",
    queue_pressure_threshold: float = 0.95,
    trailing_activate_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Renamed copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay. Logic is 100%
    identical EXCEPT the exit-head threshold compared against `prob` is now conditional: while
    `queue_pressure_component` (h48qual) holds the position, if
    components[queue_pressure_component]['queue_pressure_mask'][i] is True -- zig075 has its own
    dir_action != CASH AND quality_for_action >= zig075's quality_threshold at bar i, i.e. it would
    take a trade this bar if the shared slot were free -- `queue_pressure_threshold` is used in
    place of the component's own fixed comp['exit_threshold'] (0.95) for that one probability
    comparison. Any other active component (zig075) is completely unaffected: it always uses its
    own comp['exit_threshold'] exactly as the original, unmodified greedy_replay does, because the
    conditional branch below only ever fires when active_comp == queue_pressure_component. If no
    'queue_pressure_mask' key is present on the active component (e.g. zig075, or h48qual outside
    this experiment), or queue_pressure_threshold == comp['exit_threshold'], behaviour is
    byte-identical to the unmodified greedy_replay (see this script's G0b self-check).

    replay_omega4_6_1_greedy_router_20260706.py itself is NEVER edited by this script -- only
    imported and read, to produce this copy. Every line below is unchanged from that function
    except the block marked "--- queue pressure: only new logic vs greedy_replay ---" and the two
    diagnostic counters (qp_hold_bars/qp_pressure_bars) threaded through it.
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
    qp_hold_bars = 0
    qp_pressure_bars = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == queue_pressure_component:
                qp_hold_bars += 1
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
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(notional), float(leverage_v),
                                float(notional * leverage_v), float(take_profit), float(stop_loss)],
                    device=device,
                )
                # --- queue pressure: only new logic vs greedy_replay ---
                active_threshold = comp["exit_threshold"]
                pressure_mask = comp.get("queue_pressure_mask")
                if active_comp == queue_pressure_component and pressure_mask is not None and bool(pressure_mask[i]):
                    qp_pressure_bars += 1
                    active_threshold = queue_pressure_threshold
                # --- end queue pressure block ---
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
        f"{queue_pressure_component}_hold_bars": qp_hold_bars,
        f"{queue_pressure_component}_pressure_bars": qp_pressure_bars,
        "queue_pressure_threshold_used": float(queue_pressure_threshold),
    }
    return diag, pd.DataFrame(rows)


def _zig075_pressure_mask(pred_csv: Path, zig075_dec: pd.DataFrame, *, oof: bool, quality_threshold: float) -> tuple[np.ndarray, int]:
    """Queue pressure = zig075's OWN entry gate (dir_action != CASH AND quality_for_action >=
    zig075's live quality_threshold), read directly from the threshold-independent raw columns
    every *_predictions_qXXX.csv already carries (same columns
    research_eth_omega461_regime_specific_quality_threshold_20260813.build_final_action reads).
    Cross-checked against zig075_dec['side'] != 0 -- built by prepare_component/
    _prepare_component_val from the SAME file's already-threshold-baked final_action column -- as a
    plumbing self-check (the two are mathematically the same quantity, not two different
    definitions of pressure, since zig075's own *_predictions_q075.csv was generated with
    quality_threshold=0.75 baked into final_action already). Verified empirically to match with 0
    mismatches on the VAL file before being relied on here."""
    prefix = omega._tabm_prefix(oof)
    raw = pd.read_csv(pred_csv)
    dir_action = pd.to_numeric(raw[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    qfa = pd.to_numeric(raw[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    mask_raw = (dir_action != 0) & (qfa >= float(quality_threshold))
    mask_dec = pd.to_numeric(zig075_dec["side"], errors="raise").to_numpy(dtype=np.int64) != 0
    if len(mask_raw) != len(mask_dec):
        raise RuntimeError(f"pressure mask length mismatch: raw={len(mask_raw)} dec={len(mask_dec)}")
    mismatches = int((mask_raw != mask_dec).sum())
    return mask_raw, mismatches


def _align_frame_and_oos_predictions(oos_frame: pd.DataFrame, q_tags: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Local copy of research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.
    _align_frame_and_oos_predictions -- logic (including the WIDE24_2026 95-bar / 0.37%
    Regime3-route-probability coverage gap fix on 2026-02-28 16:05..23:55, discovered and documented
    by that script) is copied verbatim; only OUT_DIR differs, so this script writes its own aligned
    CSVs instead of reusing (and side-effecting into) that other experiment's output directory."""
    n_route_bad = int((~np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)).sum())
    if n_route_bad:
        oos_frame = oos_frame[np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
        print(f"  dropped {n_route_bad} bars with non-finite Regime3 route probabilities (WIDE24_2026 coverage gap)", flush=True)
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(oos_frame["timestamp"])
    for cname, q_tag in q_tags.items():
        pred_csv = sweep.EXT_PRED_DIR / cname / f"oos_predictions_{q_tag}.csv"
        df = pd.read_csv(pred_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[cname] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = oos_frame[oos_frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    for cname, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{cname}: OOS alignment failed after timestamp intersection")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = OUT_DIR / f"_aligned_oos_{cname}_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        aligned_paths[cname] = out_path
    return aligned_frame, aligned_paths


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 post-entry literature scouting rank-1 candidate: h48qual's exit_head MODEL is "
            "unchanged (TabM live-ATR relabel, the current confirmed baseline); only its fixed "
            "EXIT_THRESHOLD=0.95 is made conditional -- lowered to a swept candidate value on bars "
            "h48qual holds the shared slot AND zig075 has its own dir_action!=CASH & "
            "quality_for_action>=0.75 at that bar ('queue pressure'). zig075's own exit logic/model/"
            "threshold are never touched."
        ),
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_window": [sweep.OOS_START, sweep.OOS_END],
        "candidate_thresholds": CANDIDATE_THRESHOLDS,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    print("=== stage=G0_self_check (portfolio level, task-scoped) ===", flush=True)
    val_frame_raw = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    fee, slip = omega._load_fee_slip()
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    val_frame, aligned_pred_paths = portfolio._align_frame_and_predictions(val_frame_raw, q_tags)
    print(f"  VAL aligned rows={len(val_frame)} (from raw {len(val_frame_raw)})", flush=True)

    portfolio_baseline = portfolio.run_variant(
        "g0_baseline_both_original",
        {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    portfolio_tabm_liveatr = portfolio.run_variant(
        "g0_asymmetric_h48qual_liveatr_zig075_original",
        {"h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    g0_ok_baseline = _close(portfolio_baseline, G0_REFERENCE["baseline_both_original"])
    g0_ok_tabm = _close(portfolio_tabm_liveatr, G0_REFERENCE["asymmetric_tabm_liveatr"])
    g0_pass = bool(g0_ok_baseline and g0_ok_tabm)
    print(f"stage=G0_result baseline_match={g0_ok_baseline} tabm_liveatr_match={g0_ok_tabm} pass={g0_pass}", flush=True)

    report["g0"] = {
        "baseline_both_original": {"actual": portfolio_baseline, "reference": G0_REFERENCE["baseline_both_original"], "match": g0_ok_baseline},
        "asymmetric_tabm_liveatr": {"actual": portfolio_tabm_liveatr, "reference": G0_REFERENCE["asymmetric_tabm_liveatr"], "match": g0_ok_tabm},
        "tolerance_pp": G0_TOLERANCE_PP,
        "pass": g0_pass,
    }
    if not g0_pass:
        report["stage_reached"] = "G0_self_check"
        report["gate_pass"] = False
        report["note"] = "G0 failed -- this harness does not reproduce the published portfolio reference numbers. Aborting before evaluating any queue-pressure candidate."
        _write_report(report)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    print("=== stage=prepare_components_and_queue_pressure_mask ===", flush=True)
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    h48qual_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["h48qual"], h48qual_cfg, portfolio.DEVICE)
    zig075_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["zig075"], zig075_cfg, portfolio.DEVICE)

    pressure_mask, mismatches = _zig075_pressure_mask(
        aligned_pred_paths["zig075"], zig075_prepped["dec"], oof=True,
        quality_threshold=sweep.COMPONENTS["zig075"]["quality_threshold"],
    )
    print(f"  queue_pressure_mask (VAL): total_bars={len(pressure_mask)} raw_pressure_bars={int(pressure_mask.sum())} cross_check_mismatches={mismatches}", flush=True)
    if mismatches != 0:
        raise RuntimeError(f"queue_pressure_mask cross-check failed on VAL: {mismatches} mismatches between raw dir_action/quality_for_action derivation and dec['side'] derivation")
    h48qual_prepped["queue_pressure_mask"] = pressure_mask
    components = {"h48qual": h48qual_prepped, "zig075": zig075_prepped}
    report["queue_pressure_mask_cross_check_val"] = {"total_bars": int(len(pressure_mask)), "raw_pressure_bars": int(pressure_mask.sum()), "mismatches_vs_dec_side": int(mismatches)}

    print("=== stage=G0b_harness_self_consistency (degenerate threshold=0.95) ===", flush=True)
    diag95, ledger95 = greedy_replay_queue_pressure(
        val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE,
        queue_pressure_component="h48qual", queue_pressure_threshold=0.95,
    )
    metrics95 = portfolio._ledger_metrics(ledger95)
    ledger95.to_csv(OUT_DIR / "portfolio_ledger_val_g0b_degenerate_thr095.csv", index=False)
    g0b_ok = _close(metrics95, G0_REFERENCE["asymmetric_tabm_liveatr"])
    hold95, press95 = int(diag95["h48qual_hold_bars"]), int(diag95["h48qual_pressure_bars"])
    baseline_pressure_freq = (press95 / hold95 * 100.0) if hold95 else 0.0
    print(f"  G0b degenerate(0.95): pnl={metrics95['pnl']:.2f}% mdd={metrics95['mdd']:.2f}% trades={metrics95['trades']} match={g0b_ok}", flush=True)
    print(f"  baseline-policy queue pressure frequency (VAL): {press95}/{hold95} h48qual hold-bars = {baseline_pressure_freq:.2f}%", flush=True)

    report["g0b_harness_self_consistency"] = {
        "actual": metrics95, "reference": G0_REFERENCE["asymmetric_tabm_liveatr"], "match": g0b_ok,
        "note": "greedy_replay_queue_pressure run with queue_pressure_threshold=0.95 (degenerate, identical to comp['exit_threshold']) must reproduce the TabM live-ATR baseline exactly -- proves the copy is faithful outside the intentionally-changed block.",
    }
    report["queue_pressure_frequency_diagnostic"] = {
        "definition": "fraction of bars h48qual holds the shared position slot where zig075 independently has dir_action!=CASH & quality_for_action>=0.75 at that same bar",
        "measured_under": "baseline policy (exit_threshold always 0.95, i.e. h48qual's ACTUAL VAL holding pattern under the current confirmed baseline, unconfounded by the intervention itself)",
        "h48qual_hold_bars": hold95, "h48qual_pressure_bars": press95, "pressure_frequency_pct": baseline_pressure_freq,
    }
    if not g0b_ok:
        report["stage_reached"] = "G0b_harness_self_consistency"
        report["gate_pass"] = False
        report["note"] = "G0b failed -- greedy_replay_queue_pressure does not reproduce the TabM live-ATR baseline when queue_pressure_threshold==0.95 (degenerate case). This indicates a bug in the copy, aborting before trusting any candidate number."
        _write_report(report)
        print("stage=ABORT G0b failed", flush=True)
        return 1

    print("=== stage=VAL_candidate_sweep ===", flush=True)
    val_candidates: dict[str, Any] = {}
    for c in CANDIDATE_THRESHOLDS:
        diag_c, ledger_c = greedy_replay_queue_pressure(
            val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE,
            queue_pressure_component="h48qual", queue_pressure_threshold=c,
        )
        metrics_c = portfolio._ledger_metrics(ledger_c)
        ledger_c.to_csv(OUT_DIR / f"portfolio_ledger_val_qp_thr{c:.2f}.csv", index=False)
        hold_c, press_c = int(diag_c["h48qual_hold_bars"]), int(diag_c["h48qual_pressure_bars"])
        freq_c = (press_c / hold_c * 100.0) if hold_c else 0.0
        gate_pnl = float(metrics_c["pnl"]) >= float(portfolio_tabm_liveatr["pnl"])
        gate_mdd = float(metrics_c["mdd"]) >= float(portfolio_tabm_liveatr["mdd"])
        gate_pass_c = bool(gate_pnl and gate_mdd)
        val_candidates[f"{c:.2f}"] = {
            "metrics": metrics_c, "h48qual_hold_bars": hold_c, "h48qual_pressure_bars": press_c,
            "pressure_frequency_pct": freq_c, "gate_pnl_nonworse": gate_pnl, "gate_mdd_nonworse": gate_mdd, "gate_pass": gate_pass_c,
        }
        print(f"  threshold={c:.2f}: pnl={metrics_c['pnl']:.2f}% mdd={metrics_c['mdd']:.2f}% trades={metrics_c['trades']} pressure_freq={freq_c:.2f}% gate_pass={gate_pass_c}", flush=True)

    passing = [c for c in CANDIDATE_THRESHOLDS if val_candidates[f"{c:.2f}"]["gate_pass"]]
    winner = max(passing, key=lambda c: val_candidates[f"{c:.2f}"]["metrics"]["pnl"]) if passing else None
    print(f"stage=VAL_gate_result passing={passing} winner={winner}", flush=True)

    report["val_baseline_portfolio_tabm_liveatr"] = portfolio_tabm_liveatr
    report["val_gate_rule"] = (
        "queue-pressure candidate PORTFOLIO PnL AND MDD both non-worse than the TabM live-ATR "
        "baseline (exit_threshold always 0.95) on VAL. Portfolio level only -- no component-level "
        "gate, because queue pressure is only defined in the shared-slot portfolio context (it is a "
        "function of zig075's state, which does not exist in a component-standalone h48qual-only "
        "replay)."
    )
    report["val_candidates"] = val_candidates
    report["val_passing_thresholds"] = passing
    report["val_winner"] = winner

    if winner is None:
        report["oos_opened"] = False
        report["stage_reached"] = "VAL_candidate_sweep"
        report["gate_pass"] = False
        report["note"] = "No candidate threshold beat the TabM live-ATR baseline on VAL (portfolio PnL+MDD both non-worse) -- OOS NOT opened, per this project's methodology discipline. Negative pilot result."
        _write_report(report)
        print("stage=done (negative result, OOS not opened)", flush=True)
        return 0

    print(f"=== stage=OOS_single_touch winner={winner:.2f} ===", flush=True)
    print("*** MANDATORY CAVEAT ***", flush=True)
    print(OOS_CAVEAT_TEXT, flush=True)
    oos_frame_raw = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"  OOS frame rows={len(oos_frame_raw)} range=[{oos_frame_raw['timestamp'].min()}, {oos_frame_raw['timestamp'].max()}]", flush=True)
    oos_frame, oos_aligned_paths = _align_frame_and_oos_predictions(oos_frame_raw, q_tags)
    print(f"  OOS aligned rows={len(oos_frame)} (from raw {len(oos_frame_raw)})", flush=True)

    # OOS baseline: TabM live-ATR asymmetric config, threshold always 0.95, UNMODIFIED greedy.greedy_replay.
    oos_components_baseline = {
        "h48qual": greedy.prepare_component(oos_frame, oos_aligned_paths["h48qual"], h48qual_cfg, portfolio.DEVICE),
        "zig075": greedy.prepare_component(oos_frame, oos_aligned_paths["zig075"], zig075_cfg, portfolio.DEVICE),
    }
    _diag_oos_base, ledger_oos_base = greedy.greedy_replay(oos_frame, oos_components_baseline, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE)
    metrics_oos_base = portfolio._ledger_metrics(ledger_oos_base)
    ledger_oos_base.to_csv(OUT_DIR / "portfolio_ledger_oos_baseline_tabm_liveatr.csv", index=False)
    oos_baseline_cross_check = _close(metrics_oos_base, OOS_BASELINE_REFERENCE)
    print(f"  OOS baseline: pnl={metrics_oos_base['pnl']:.2f}% mdd={metrics_oos_base['mdd']:.2f}% trades={metrics_oos_base['trades']} cross_check_vs_published(93.27/-15.48/24)={oos_baseline_cross_check}", flush=True)

    # OOS candidate: same components + queue pressure mask, winner threshold.
    oos_pressure_mask, oos_mismatches = _zig075_pressure_mask(
        oos_aligned_paths["zig075"], oos_components_baseline["zig075"]["dec"], oof=False,
        quality_threshold=sweep.COMPONENTS["zig075"]["quality_threshold"],
    )
    print(f"  queue_pressure_mask (OOS): total_bars={len(oos_pressure_mask)} raw_pressure_bars={int(oos_pressure_mask.sum())} cross_check_mismatches={oos_mismatches}", flush=True)
    if oos_mismatches != 0:
        raise RuntimeError(f"queue_pressure_mask cross-check failed on OOS: {oos_mismatches} mismatches")
    oos_components_candidate = dict(oos_components_baseline)
    oos_components_candidate["h48qual"] = dict(oos_components_baseline["h48qual"])
    oos_components_candidate["h48qual"]["queue_pressure_mask"] = oos_pressure_mask
    diag_oos_cand, ledger_oos_cand = greedy_replay_queue_pressure(
        oos_frame, oos_components_candidate, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE,
        queue_pressure_component="h48qual", queue_pressure_threshold=winner,
    )
    metrics_oos_cand = portfolio._ledger_metrics(ledger_oos_cand)
    ledger_oos_cand.to_csv(OUT_DIR / f"portfolio_ledger_oos_qp_thr{winner:.2f}.csv", index=False)
    hold_oos, press_oos = int(diag_oos_cand["h48qual_hold_bars"]), int(diag_oos_cand["h48qual_pressure_bars"])
    freq_oos = (press_oos / hold_oos * 100.0) if hold_oos else 0.0
    print(f"  OOS candidate({winner:.2f}): pnl={metrics_oos_cand['pnl']:.2f}% mdd={metrics_oos_cand['mdd']:.2f}% trades={metrics_oos_cand['trades']} pressure_freq={freq_oos:.2f}%", flush=True)

    oos_gate_pnl = float(metrics_oos_cand["pnl"]) >= float(metrics_oos_base["pnl"])
    oos_gate_mdd = float(metrics_oos_cand["mdd"]) >= float(metrics_oos_base["mdd"])
    oos_survives = bool(oos_gate_pnl and oos_gate_mdd)
    print(f"stage=OOS_result survives={oos_survives}", flush=True)

    report.update({
        "oos_opened": True,
        "oos_frame_rows_raw": int(len(oos_frame_raw)), "oos_frame_rows_aligned": int(len(oos_frame)),
        "oos_baseline_portfolio_tabm_liveatr": metrics_oos_base,
        "oos_baseline_cross_check_reference": OOS_BASELINE_REFERENCE, "oos_baseline_cross_check_match": oos_baseline_cross_check,
        "queue_pressure_mask_cross_check_oos": {"total_bars": int(len(oos_pressure_mask)), "raw_pressure_bars": int(oos_pressure_mask.sum()), "mismatches_vs_dec_side": int(oos_mismatches)},
        "oos_candidate_threshold": winner,
        "oos_candidate_queue_pressure": metrics_oos_cand,
        "oos_candidate_h48qual_hold_bars": hold_oos, "oos_candidate_h48qual_pressure_bars": press_oos,
        "oos_candidate_pressure_frequency_pct": freq_oos,
        "oos_gate_pnl_nonworse": oos_gate_pnl, "oos_gate_mdd_nonworse": oos_gate_mdd, "oos_survives": oos_survives,
        "oos_caveat_quality_threshold_contamination": OOS_CAVEAT_TEXT,
        "oos_caveat_source_doc": "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md",
        "stage_reached": "OOS_single_touch",
        "gate_pass": True,
    })
    _write_report(report)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
