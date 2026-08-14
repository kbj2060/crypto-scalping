#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #13: literature-scouting rank-4 candidate
(docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md, section 3 table, "Selective
Conformal Risk Control" (SCRC), Xu, Guo, Wei, arXiv:2512.12844, v1 2025-12-14 / v2 2026-04-27 -- read
directly via WebFetch of both the arXiv abstract page and the full HTML paper (arxiv.org/html/2512.12844v2)
before writing this script, not from the scouting doc's one-paragraph summary).

=== Paper mechanism (as confirmed from the paper itself, not the scouting doc) ===
Abstract (verbatim, fetched): "we propose Selective Conformal Risk Control (SCRC), a unified framework
that integrates conformal prediction with selective classification. The framework formulates uncertainty
control as a two-stage problem: the first stage selects confident samples for prediction, and the second
stage applies conformal risk control on the selected subset to construct calibrated prediction sets." The
scouting doc's summary is CONFIRMED accurate at this level.

Exact setting (classification with SET-valued predictions, K classes): base classifier f(x) in [0,1]^K,
a separate selection/confidence score g(x) in [0,1] (feature-only -- depends on X, not Y, "necessary for
symmetric selection" per the paper), two thresholds (lambda_1, lambda_2). Selective rule:
  C(x) = reject-symbol         if g(x) < 1-lambda_1        (NOT selected -- no prediction made)
  C(x) = {k : f(x)_k >= 1-lambda_2}   otherwise             (selected -- risk-controlled prediction SET)
Two algorithms:
  SCRC-T (transductive): lambda_1 is computed JOINTLY over calibration+test features (a symmetric
    function of all n+1 points, preserving exchangeability under selection, Lemma 1/Theorem 2) --
    EXACT finite-sample distribution-free guarantees on both conditional risk (<=alpha) and coverage
    (acceptance rate >=xi), but requires seeing the test point's features at calibration time (an
    online/per-test recomputation).
  SCRC-I (inductive/calibration-only): lambda_1 and lambda_2 are both fixed from calibration data ALONE
    (no test-time access), using a Dvoretzky-Kiefer-Wolfowitz lower confidence bound on the empirical
    selection rate for lambda_1 and a Hoeffding correction (over a lambda_1 x lambda_2 grid) for lambda_2
    -- a PAC-style ("with probability >=1-delta") guarantee, not the exact finite-sample one; the paper's
    own experiments (CIFAR-10, Diabetic Retinopathy Detection) report SCRC-I as "slightly more
    conservative" (marginally larger prediction sets) than SCRC-T, with the gap shrinking as the
    calibration set grows.
Relationship to the rank-2 candidate (Joshi, Wang, Hassani, Dobriban, "Risk-Controlled Post-Processing of
Decision Policies", arXiv:2605.06479 -- this project's Odyssey2 #8, docs/experiments/
eth_omega461_risk_controlled_post_processing_exit_fallback_20260814.md): the fetched SCRC related-work
section (conformal-prediction block, selective-classification block, and an explicit "integration" block
citing Fisch et al. 2022 "calibrated selective classification", Bao et al. 2024 "selective conditional
conformal prediction", Gazin et al. 2024) contains NO citation of Joshi/Wang/Hassani/Dobriban or of any
threshold-based decision-policy-post-processing paper -- confirmed absent, not merely unmentioned in the
fetched excerpt (the fetch explicitly enumerated the full related-work block). This is temporally
consistent (SCRC v1 2025-12-14 predates Joshi et al. 2026-05-07; SCRC v2 2026-04-27 also predates it) --
SCRC could not have cited Joshi, and nothing in the fetched Joshi/Wang summary already in this project's
#8 script/doc cites SCRC either. The two lines are independent, parallel conformal-risk-control-adjacent
work, not sequential/derivative. Mechanically they differ in a load-bearing way: Joshi et al. is a
SINGLE-STAGE decision-policy problem (one score Delta(x), one threshold tau, choose between exactly two
ACTIONS pi0(x)/pi*(x) at every x); SCRC is a genuine TWO-STAGE classification-with-abstention problem (a
separate selection stage that can reject a point entirely BEFORE any risk-controlled construction is
attempted on it, with its own independent threshold/guarantee). This project's #8 already implemented
Joshi et al.'s single-stage mechanism end-to-end (VAL-won, OOS-portfolio-reversed). This script's job is
to add the genuinely-missing SECOND STAGE -- explicit selection BEFORE risk control -- not to re-run #8.

=== Honest scoping (stated up front, not glossed over) ===
This script does NOT literally transplant SCRC's classification-with-prediction-SETS formalism (there is
no natural K-class "prediction set" analog for a single continuous hold/exit decision at a bar) or its
exact DKW/Hoeffding SCRC-I calibration algorithm (that machinery is specific to bounding a selection RATE
and a set-membership risk over a lambda_1 x lambda_2 grid via concentration inequalities -- transplanting
it verbatim onto a ~13k-bar/tens-of-trades trading calibration set the way #8 already found this project's
existing sample sizes to be delicate about would add complexity without a validated point of contact).
What IS transplanted, faithfully and load-bearingly, is the paper's DEFINING STRUCTURAL claim: a stage-1
selection step (accept/reject based on a feature-only confidence score g(x), evaluated and calibrated
BEFORE stage 2) that gates a stage-2 risk-controlled step (evaluated and calibrated ONLY on the
stage-1-accepted subset). Stage 2's actual risk-control MATH is Joshi et al.'s already-implemented,
already-verified Algorithm 1 (research_eth_omega461_risk_controlled_exit_fallback_20260814._risk_
controlled_action / _calibrate_threshold / _bumped_risk, reused UNMODIFIED via import, zero
reimplementation, zero retraining) -- this script's only new math is the stage-1 selection gate and the
population restriction it imposes on stage 2's calibration set. This is declared explicitly as a
reinterpretation, not a literal SCRC-T/SCRC-I port, exactly the same honesty standard #8 applied to its
own mapping of Joshi et al.'s LLM-routing worked example onto this project's TabM/GBDT pair.

=== Mapping onto this project (calibration-only, zero retraining, zero new features) ===
- g(x) [stage-1 selection score]: TabM baseline exit_head's OWN causal probability prob_baseline(x) --
  feature-only (does not depend on the y-rule/ground truth used in stage 2, matching the paper's
  "L^(1) depends only on X" requirement for a valid, symmetric selection rule).
- select_threshold [stage-1 cutoff, "1-lambda_1" in the paper's notation]: fixed (not calibrated) at
  sweep.BASELINE_EXIT_THRESHOLD (0.95) -- the SAME confidence cutoff this project already uses to decide
  a real TabM exit trigger, not a new hyperparameter. Concretely: "selected" (accepted into stage 2) is
  EXACTLY the set of bars where a0=1 (TabM's own exit_head has ALREADY, independently, decided with its
  own established conviction cutoff to exit) -- i.e. this operationalizes the scouting doc's "exit_head
  확률을 1단계 선별 기준으로, 선별된 '확신 있는 exit 신호'에만" as literally as possible. Deliberately NOT
  calibrated adaptively on VAL (unlike the paper's own lambda_1): this project's own eth_val_oos_regime_
  mismatch_investigation_20260813.md diagnosed a "3x-stacked selection-bias" root cause (risk-sizing ->
  quality_threshold -> new-candidate-threshold, all fit against the SAME ~26k-bar VAL window) behind this
  session's repeated VAL-win/OOS-reversal pattern (queue-pressure #7, risk-controlled #8, regime-threshold
  #12 all independently hit this). Adding a freely-VAL-calibrated select_threshold on top of the eps/
  tau_hat calibration #8 already performs would be a FOURTH such layer on the same small window -- fixing
  it at an existing canonical value removes one full degree of VAL-fitting freedom relative to a literal
  SCRC port, by design, not by oversight.
  A SECOND, explicitly diagnostic-only (never gated) select_threshold=0.50 run is also reported, to make
  the two-stage vs #8's single-stage behavior visible by direct contrast (see "diagnostic" section below).
- Stage 2 (risk control on the selected subset): IDENTICAL mechanism to #8 -- pi0=TabM baseline
  (EXIT_THRESHOLD=0.95), pi*=already-trained GBDT exit_head (Odyssey2 #4, gbdt_exit_bundle.pkl) at its own
  0.95 threshold, g(hold,x)=p_gbdt(x)/g(exit,x)=1-p_gbdt(x), Delta(x)=g(pi0(x),x)-g(pi*(x),x), y-rule =
  pos_giveback>=0.65 OR pos_unrealized<=-0.010 (causal, backward-looking-only, same 98.1%-of-positive-
  class rule #4/#8 already documented) -- ALL reused unmodified via import from research_eth_omega461_
  risk_controlled_exit_fallback_20260814 (rc_mod below). eps grid (rc_mod.EPS_FRACTIONS = [0.90,0.70,
  0.50], pre-registered before seeing any PnL) is applied to the baseline's OWN bumped mismatch rate
  computed ONLY over the select_threshold-accepted subset (the paper's Z_{lambda_1-bar} restriction --
  "conformal risk control on the SELECTED subset", not the full held-bar population #8 calibrated on).
  Because "selected" == {a0=1} at the primary select_threshold, EVERY selected bar already has a0=1 by
  construction -- Delta(x) can therefore only ever justify switching astar=0 (GBDT disagrees, thinks the
  TabM exit call was premature) over a0=1 (TabM's own confident exit call), i.e. this candidate's
  intervention can only ever CANCEL a TabM exit that GBDT confidently disputes -- it structurally CANNOT
  trigger a fresh early exit on a bar TabM itself was not confident about (#8 could do both; see
  #8's own risk_at_tau_hat / switch_bars diagnostics for contrast). This narrowing is the direct,
  concrete, testable consequence of adding a real stage-1 gate -- not asserted, verified by construction
  and by the diagnostic run's contrast below.

=== Compliance (same standard as every Odyssey2 script this session) ===
fresh_forward_bar_by_bar=true (both renamed-copy replay loops are single causal forward passes, i
increasing only, the y-rule uses only the position's own already-realized giveback/unrealized-PnL as of
bar i, and stage-1 selection is evaluated from the SAME already-causal prob_baseline(x) the baseline
already computes at that bar -- no new lookahead of any kind). trade_ledgers_used_as_input=false (ledgers
are written-only outputs). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
direction_head/quality_head/quality_threshold/encoder frozen and unchanged for BOTH components -- only
h48qual's exit_head DECISION (never its TabM weights, never GBDT's weights) is made conditional on the
two-stage select_threshold/tau_hat rule. zig075 is not touched in any way (never given a fallback runtime,
never selected/gated). Uses eth_omega461_multiwindow_confirmation_gate_20260814.load_all_windows /
align_frame_and_predictions / summarize_multiwindow for the official OOS-Q1+OOS-Q2 single-touch
confirmation (imported, unmodified -- the required infra per this session's methodology update).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env. Does NOT
modify research_eth_omega461_exit_sweep_20260721.py, replay_omega4_6_1_greedy_router_20260706.py,
research_eth_omega461_risk_controlled_exit_fallback_20260814.py, or eth_omega461_multiwindow_confirmation_
gate_20260814.py -- all four are imported and read only. Seed-Diversity Ensemble Promotion Gate: not
applicable (deterministic post-processing threshold policy over two already-frozen, already-trained
artifacts, no seed-ensemble promotion claim). Omega Artifact Integrity Promotion Gate: not applicable (no
new parent prediction artifact created or promoted).
"""
from __future__ import annotations

import sys
import json
import pickle
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
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import research_eth_omega461_gbdt_exit_head_val_20260813 as gbdt_val  # noqa: E402
import research_eth_omega461_risk_controlled_exit_fallback_20260814 as rc_mod  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as mw_gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_selective_conformal_risk_control_20260814"

SELECT_THRESHOLD_PRIMARY = sweep.BASELINE_EXIT_THRESHOLD  # 0.95 -- pre-registered stage-1 cutoff, see
# module docstring. "Selected" == {a0=1} at this value: only TabM's own already-confident exit calls
# enter stage 2.
SELECT_THRESHOLD_DIAGNOSTIC = 0.50  # diagnostic-only, NEVER gated -- the probability decision-boundary
# midpoint, chosen only to materially widen the selected set (so stage 2 can also be reached on bars
# where a0=0, letting the run show the "can trigger fresh early exits too" behavior #8 exhibited, by
# contrast with the primary/gated 0.95 cutoff's structurally narrower "cancel-only" behavior).
SELECT_THRESHOLD_NEVER_SELECT = 1.01  # sentinel > max possible probability -- degenerate case for G0c
# (stage-1 neutralized independently of stage-2's own tau=NEVER_SWITCH sentinel).
G0_TOLERANCE_PP = 0.05


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP, check_trades: bool = True) -> bool:
    return rc_mod._close(actual, expected, tol_pp=tol_pp, check_trades=check_trades)


# =====================================================================================================
# Stage-1 selection gate wrapping stage-2's risk control (rc_mod._risk_controlled_action, Joshi et al.
# Theorem 3.1, reused UNMODIFIED -- this function adds ONLY the selection gate around it).
# =====================================================================================================


def _selective_risk_controlled_action(
    prob_baseline: float, prob_fallback: float, *, selected: bool, baseline_threshold: float,
    fallback_threshold: float, tau: float, y_rule: int,
) -> dict[str, Any]:
    """Stage 1 (selection): a bar enters stage 2 only if `selected` (decided by the CALLER from
    g(x)=prob_baseline >= select_threshold, evaluated BEFORE this function is called and, critically,
    BEFORE prob_fallback is even scored on non-selected bars -- see the replay-loop callers below, which
    skip the GBDT forward pass entirely when not selected; this is a genuine two-stage evaluation order,
    not just a decision-logic distinction). Non-selected bars pass straight through to the baseline
    action a0 -- GBDT/Delta is never consulted. This mirrors the paper's semantics (rejected samples get
    no risk-controlled treatment, "deferred to a downstream fallback" the paper explicitly leaves out of
    scope) adapted to a continuous exit-timing setting where the only coherent "downstream fallback" for
    an already-open position is to keep the existing baseline decision (there is no abstain-from-deciding
    option once a bar must resolve to hold-or-exit).

    Stage 2 (risk control), reached only when selected=True: delegates to rc_mod._risk_controlled_action
    verbatim (Delta(x)=g(pi0(x),x)-g(pi*(x),x), switch iff Delta>=tau) -- reused, not reimplemented."""
    a0 = bool(prob_baseline >= baseline_threshold)
    if not selected:
        return {"a0": a0, "astar": None, "delta": float("nan"), "switched": False, "final_exit": a0, "selected": False, "y": int(y_rule)}
    rc = rc_mod._risk_controlled_action(
        prob_baseline, prob_fallback, baseline_threshold=baseline_threshold, fallback_threshold=fallback_threshold,
        tau=tau, y_rule=y_rule,
    )
    rc["selected"] = True
    return rc


# =====================================================================================================
# Renamed copy #1 (component level): research_eth_omega461_exit_sweep_20260721.replay_exit_variant, via
# rc_mod.replay_exit_variant_risk_controlled's already-established diff block (that script is never
# edited -- only imported and read). Every line below is unchanged from rc_mod's version EXCEPT the
# blocks marked "--- selective: ... ---" (select_threshold param, lazy fallback scoring gated on
# selection, "selected" logging).
# =====================================================================================================


@torch.no_grad()
def replay_exit_variant_selective_risk_controlled(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    fallback_loaded_models: dict[str, tuple] | None = None,
    fallback_threshold: float = rc_mod.FALLBACK_THRESHOLD,
    select_threshold: float = SELECT_THRESHOLD_PRIMARY,  # --- selective: new param ---
    tau: float = rc_mod.TAU_NEVER_SWITCH,
    trailing_activate_frac: float | None = None,
    trailing_retain_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trailing_enabled = trailing_activate_frac is not None and (
        trailing_retain_frac is not None or trailing_trail_frac is not None)
    if trailing_retain_frac is not None and trailing_trail_frac is not None:
        raise ValueError("pass either trailing_retain_frac (proportional) or trailing_trail_frac (fixed distance)")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    fallback_exit_runtime = None
    if fallback_loaded_models is not None:
        _fb_base_np, fallback_exit_runtime, fb_pos_idx = rs._prepare_exit_runtime(base_x, fallback_loaded_models)
        if fb_pos_idx != pos_idx:
            raise RuntimeError("selective-risk-controlled: fallback pos_idx does not match baseline pos_idx")
    # --- selective: log now also carries "selected" (stage-1 accept/reject per bar) ---
    rc_log: dict[str, list[Any]] = {"delta": [], "y": [], "a0": [], "astar": [], "switched": [], "selected": []}

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif trailing_enabled and (not armed) and mfe >= float(trailing_activate_frac) * take_profit and take_profit > 0.0:
                armed = True
            if not reason and trailing_enabled and armed and mfe > 0.0:
                if trailing_retain_frac is not None:
                    if move <= float(trailing_retain_frac) * mfe:
                        reason = "trailing_stop"
                elif move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                expert = hard.EXPERT_NAMES[int(route[i])]
                pos_values = [
                    float(pos), float(hold), float(move), float(mfe), float(mae),
                    giveback_clipped, float(take_profit - move), float(move + abs(stop_loss)),
                    float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                ]
                prob = rs._predict_exit_prob_one(base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert, pos_values=pos_values, device=device)
                exit_prob = float(prob)
                final_exit = bool(prob >= float(exit_threshold))
                # --- selective: stage-1 gate, then (only if selected) stage-2 fallback scoring+decision ---
                if fallback_exit_runtime is not None:
                    selected = bool(prob >= float(select_threshold))
                    if selected:
                        prob_fb = rs._predict_exit_prob_one(base_np, fallback_exit_runtime, pos_idx, row_i=int(i), expert=expert, pos_values=pos_values, device=device)
                    else:
                        prob_fb = float("nan")
                    y_rule = 1 if (giveback_clipped >= rc_mod.Y_RULE_GIVEBACK or float(move) <= rc_mod.Y_RULE_UNREALIZED) else 0
                    rc = _selective_risk_controlled_action(
                        prob, prob_fb, selected=selected, baseline_threshold=float(exit_threshold), fallback_threshold=float(fallback_threshold),
                        tau=float(tau), y_rule=y_rule,
                    )
                    final_exit = rc["final_exit"]
                    rc_log["delta"].append(rc["delta"]); rc_log["y"].append(rc["y"])
                    rc_log["a0"].append(int(rc["a0"])); rc_log["astar"].append(int(rc["astar"]) if rc["astar"] is not None else -1)
                    rc_log["switched"].append(int(rc["switched"])); rc_log["selected"].append(int(rc["selected"]))
                # --- end selective decision ---
                if final_exit:
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                armed = False
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
        armed = False

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
        "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
        "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
        "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
        "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
        "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
    }
    if fallback_exit_runtime is not None:
        metrics["scrc_log"] = rc_log
    return metrics, ledger


# =====================================================================================================
# Renamed copy #2 (portfolio level): replay_omega4_6_1_greedy_router_20260706.greedy_replay, via
# rc_mod.greedy_replay_risk_controlled's already-established diff block. Every line below is unchanged
# EXCEPT the blocks marked "--- selective: ... ---".
# =====================================================================================================


@torch.no_grad()
def greedy_replay_selective_risk_controlled(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    risk_component: str = "h48qual",
    select_threshold: float = SELECT_THRESHOLD_PRIMARY,  # --- selective: new param ---
    tau: float = rc_mod.TAU_NEVER_SWITCH,
    trailing_activate_frac: float | None = None,
    trailing_trail_frac: float | None = None,
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
    trailing_enabled = trailing_activate_frac is not None and trailing_trail_frac is not None
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    rc_hold_bars = 0
    rc_selected_bars = 0  # --- selective: new counter (stage-1 accepts, subset of rc_hold_bars) ---
    rc_switch_bars = 0
    rc_switch_to_exit = 0
    rc_switch_to_hold = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
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
                prob = rs._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=pos_values, device=device,
                )
                final_exit = bool(prob >= comp["exit_threshold"])
                # --- selective: stage-1 gate, then (only if selected) stage-2 fallback scoring+decision ---
                fallback_runtime = comp.get("fallback_exit_runtime")
                if active_comp == risk_component and fallback_runtime is not None:
                    rc_hold_bars += 1
                    selected = bool(prob >= float(select_threshold))
                    if selected:
                        rc_selected_bars += 1
                        prob_fb = rs._predict_exit_prob_one(
                            comp["base_np"], fallback_runtime, comp["pos_idx"], row_i=int(i), expert=expert,
                            pos_values=pos_values, device=device,
                        )
                    else:
                        prob_fb = float("nan")
                    y_rule = 1 if (giveback_clipped >= rc_mod.Y_RULE_GIVEBACK or move <= rc_mod.Y_RULE_UNREALIZED) else 0
                    rc = _selective_risk_controlled_action(
                        prob, prob_fb, selected=selected, baseline_threshold=float(comp["exit_threshold"]),
                        fallback_threshold=float(comp.get("fallback_exit_threshold", rc_mod.FALLBACK_THRESHOLD)),
                        tau=float(tau), y_rule=y_rule,
                    )
                    final_exit = rc["final_exit"]
                    if rc["switched"]:
                        rc_switch_bars += 1
                        if rc["astar"] and not rc["a0"]:
                            rc_switch_to_exit += 1
                        elif rc["a0"] and not rc["astar"]:
                            rc_switch_to_hold += 1
                # --- end selective decision ---
                if final_exit:
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
        "rc_hold_bars": rc_hold_bars, "rc_selected_bars": rc_selected_bars, "rc_switch_bars": rc_switch_bars,
        "rc_switch_to_exit_bars": rc_switch_to_exit, "rc_switch_to_hold_bars": rc_switch_to_hold,
        "select_threshold_used": float(select_threshold), "tau_used": float(tau),
    }
    return diag, pd.DataFrame(rows)


# =====================================================================================================
# OOS multiwindow hooks -- following eth_omega461_multiwindow_confirmation_gate_20260814's own worked
# examples (_prep_asymmetric_components / _risk_controlled_variant inside that module's main()) as the
# intended pattern for a candidate script's own per-window evaluator. Uses ONLY the module's public API
# (load_all_windows/align_frame_and_predictions/summarize_multiwindow/WINDOW_DEFS/ALL_WINDOWS) -- does
# not modify or reach into that module's private helpers.
# =====================================================================================================


def _prep_asymmetric_components(window_name: str, windows: dict, h48qual_cfg: dict, zig075_cfg: dict, device: torch.device, out_dir: Path):
    w = windows[window_name]
    split = mw_gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {"h48qual": sweep.COMPONENTS["h48qual"]["q_tag"], "zig075": sweep.COMPONENTS["zig075"]["q_tag"]}
    aligned_frame, aligned_paths = mw_gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    h48qual_prepped = prep(aligned_frame, aligned_paths["h48qual"], h48qual_cfg, device)
    zig075_prepped = prep(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)
    return aligned_frame, aligned_paths, h48qual_prepped, zig075_prepped


def _scrc_variant(
    window_name: str, windows: dict, h48qual_cfg: dict, zig075_cfg: dict, device: torch.device, out_dir: Path,
    fee: float, slip: float, base_cols: list[str], gbdt_models: dict[str, Any], select_threshold: float, tau: float,
) -> tuple[tuple[dict, dict], tuple[dict, dict], dict]:
    aligned_frame, _aligned_paths, h48qual_prepped, zig075_prepped = _prep_asymmetric_components(window_name, windows, h48qual_cfg, zig075_cfg, device, out_dir)
    components_baseline = {"h48qual": h48qual_prepped, "zig075": zig075_prepped}
    _diag_b, ledger_b = greedy.greedy_replay(aligned_frame, components_baseline, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    h48qual_scrc = rc_mod._gbdt_portfolio_fallback(dict(h48qual_prepped), base_cols, gbdt_models, device)
    components_candidate = {"h48qual": h48qual_scrc, "zig075": zig075_prepped}
    diag_c, ledger_c = greedy_replay_selective_risk_controlled(
        aligned_frame, components_candidate, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
        risk_component="h48qual", select_threshold=select_threshold, tau=tau,
    )
    ledger_b.to_csv(out_dir / f"portfolio_ledger_{window_name}_baseline.csv", index=False)
    ledger_c.to_csv(out_dir / f"portfolio_ledger_{window_name}_scrc_candidate.csv", index=False)
    baseline = (portfolio._ledger_metrics(ledger_b), mfe_width._duration_gated(ledger_b, aligned_frame, greedy.DURATION_THRESHOLD))
    candidate = (portfolio._ledger_metrics(ledger_c), mfe_width._duration_gated(ledger_c, aligned_frame, greedy.DURATION_THRESHOLD))
    return baseline, candidate, diag_c


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 #13 -- Selective Conformal Risk Control (Xu, Guo, Wei, arXiv:2512.12844). Two "
            "explicit stages: stage 1 selects 'confident exit signal' bars (g(x)=TabM's own exit "
            "probability >= select_threshold, fixed at the existing EXIT_THRESHOLD=0.95, i.e. selected== "
            "{a0=1}); stage 2 applies #8's already-verified Joshi et al. Algorithm 1 risk-controlled "
            "GBDT-fallback threshold policy (reused unmodified) ONLY to that selected subset. Structural "
            "difference from #8: #8 evaluated/calibrated Delta(x) at EVERY held bar unconditionally; this "
            "candidate evaluates/calibrates it ONLY on stage-1-selected bars, which by construction can "
            "only ever cancel an already-confident TabM exit call, never trigger a fresh one."
        ),
        "paper_citation": "Xu, Guo, Wei, Selective Conformal Risk Control, arXiv:2512.12844 (v1 2025-12-14, v2 2026-04-27)",
        "paper_relationship_to_rank2_joshi_et_al": (
            "Fetched SCRC's related-work section directly (arxiv.org/html/2512.12844v2): it cites "
            "Fisch et al. 2022 (calibrated selective classification), Bao et al. 2024 (selective "
            "conditional conformal prediction), Gazin et al. 2024 (informativeness constraints) -- NO "
            "citation of Joshi/Wang/Hassani/Dobriban arXiv:2605.06479 or any decision-policy-post-"
            "processing paper. Temporally consistent: SCRC v1/v2 (2025-12-14/2026-04-27) both predate "
            "Joshi et al. (2026-05-07), so SCRC could not have cited it; nothing in this project's own #8 "
            "reading of Joshi et al. cited SCRC either. Independent, parallel lines, not sequential -- "
            "mechanically distinguished by SCRC being a genuine two-stage select-then-control "
            "classification-with-abstention problem vs Joshi et al.'s single-stage two-action "
            "decision-policy threshold problem (see script docstring for the full argument).",
        ),
        "select_threshold_primary": SELECT_THRESHOLD_PRIMARY,
        "select_threshold_diagnostic_not_gated": SELECT_THRESHOLD_DIAGNOSTIC,
        "eps_fractions_preregistered": rc_mod.EPS_FRACTIONS,
        "y_rule": {"pos_giveback_ge": rc_mod.Y_RULE_GIVEBACK, "pos_unrealized_le": rc_mod.Y_RULE_UNREALIZED},
        "fallback_model": "GBDT exit_head (Odyssey2 #4), " + str(rc_mod.GBDT_BUNDLE),
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_confirm_windows": list(mw_gate.OOS_CONFIRM_WINDOWS),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    if not rc_mod.GBDT_BUNDLE.exists():
        raise FileNotFoundError(f"GBDT bundle not found: {rc_mod.GBDT_BUNDLE}")
    with open(rc_mod.GBDT_BUNDLE, "rb") as f:
        gbdt_bundle = pickle.load(f)
    gbdt_models = gbdt_bundle["models"]

    device = portfolio.DEVICE
    fee, slip = omega._load_fee_slip()
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    gbdt_component_loaded = gbdt_val._gbdt_loaded_models(base_cols, gbdt_models, device)

    print("=== stage=load_windows (eth_omega461_multiwindow_confirmation_gate_20260814, reused) ===", flush=True)
    windows = mw_gate.load_all_windows()

    # ======================================================================================
    # stage=G0 -- literal reproduction of the task-specified reference numbers, via the
    # multiwindow gate module's own already-verified run_portfolio_variant + reference dict
    # (single source of truth, no retyping).
    # ======================================================================================
    print("=== stage=G0_portfolio_via_gate_module (val + oos_q1, asymmetric_tabm_liveatr) ===", flush=True)
    g0_portfolio: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = mw_gate.run_portfolio_variant(wname, windows, mw_gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng, ok_wg = _close(result["no_gate"], ref_ng), _close(result["with_gate"], ref_wg)
        g0_portfolio[wname] = {"no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
                                "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg}}
        print(f"  {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
              f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}", flush=True)
    g0_portfolio_pass = all(g0_portfolio[w]["no_gate"]["match"] and g0_portfolio[w]["with_gate"]["match"] for w in ("val", "oos_q1"))

    print("=== stage=G0_component (h48cons._evaluate_val, vs rc_mod.G0_REFERENCE) ===", flush=True)
    g0_component = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    g0_ok_component_baseline = _close(g0_component["baseline"], rc_mod.G0_REFERENCE["component_baseline_original"])
    g0_ok_component_tabm = _close(g0_component["h48cons_relabel"], rc_mod.G0_REFERENCE["component_tabm_liveatr"])
    print(f"  component baseline_original={g0_component['baseline']} match={g0_ok_component_baseline}", flush=True)
    print(f"  component tabm_liveatr={g0_component['h48cons_relabel']} match={g0_ok_component_tabm}", flush=True)
    g0_component_pass = bool(g0_ok_component_baseline and g0_ok_component_tabm)

    g0_pass = bool(g0_portfolio_pass and g0_component_pass)
    report["g0"] = {"portfolio_val_oosq1_via_gate_module": g0_portfolio, "portfolio_pass": g0_portfolio_pass,
                     "component": {"baseline_original": {"actual": g0_component["baseline"], "reference": rc_mod.G0_REFERENCE["component_baseline_original"], "match": g0_ok_component_baseline},
                                   "tabm_liveatr": {"actual": g0_component["h48cons_relabel"], "reference": rc_mod.G0_REFERENCE["component_tabm_liveatr"], "match": g0_ok_component_tabm}},
                     "component_pass": g0_component_pass, "pass": g0_pass}
    print(f"stage=G0_result pass={g0_pass}", flush=True)
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed to exactly reproduce the task-specified reference numbers. Aborting before trusting any candidate number."
        _write_report(report)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    # ======================================================================================
    # stage=G0b -- tau=NEVER_SWITCH (stage 2 neutralized), select_threshold=PRIMARY: must
    # reproduce the baseline exactly AND doubles as the calibration-log capture (unconfounded
    # by the intervention, same trick #8's own G0b used).
    # ======================================================================================
    print("=== stage=G0b_tau_never_switch (stage-2 neutralized, select_threshold=primary; also captures calibration log) ===", flush=True)
    val_frame_raw = windows["val"]["frame"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    val_pred_h48qual = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{h48qual_cfg['q_tag']}.csv"
    comp_prepped = sweep.prep_component("h48qual", h48qual_cfg, val_frame_raw, val_pred_h48qual, oof=True)
    m_g0b_component, _ledger_g0b_component = replay_exit_variant_selective_risk_controlled(
        comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
        risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
        fallback_loaded_models=gbdt_component_loaded, fallback_threshold=rc_mod.FALLBACK_THRESHOLD,
        select_threshold=SELECT_THRESHOLD_PRIMARY, tau=rc_mod.TAU_NEVER_SWITCH,
    )
    g0b_ok_component = _close(m_g0b_component, rc_mod.G0_REFERENCE["component_tabm_liveatr"])
    print(f"  component G0b: pnl={m_g0b_component['pnl']:.2f}% mdd={m_g0b_component['mdd']:.2f}% trades={m_g0b_component['trades']} match={g0b_ok_component}", flush=True)

    val_frame, aligned_pred_paths = mw_gate.align_frame_and_predictions(val_frame_raw, q_tags, "validation", OUT_DIR)
    h48qual_prepped_portfolio = portfolio._prepare_component_val(val_frame, aligned_pred_paths["h48qual"], h48qual_cfg, device)
    h48qual_prepped_portfolio = rc_mod._gbdt_portfolio_fallback(h48qual_prepped_portfolio, base_cols, gbdt_models, device)
    zig075_prepped_portfolio = portfolio._prepare_component_val(val_frame, aligned_pred_paths["zig075"], zig075_cfg, device)
    components_val = {"h48qual": h48qual_prepped_portfolio, "zig075": zig075_prepped_portfolio}
    diag_g0b_portfolio, ledger_g0b_portfolio = greedy_replay_selective_risk_controlled(
        val_frame, components_val, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
        risk_component="h48qual", select_threshold=SELECT_THRESHOLD_PRIMARY, tau=rc_mod.TAU_NEVER_SWITCH,
    )
    m_g0b_portfolio = portfolio._ledger_metrics(ledger_g0b_portfolio)
    ledger_g0b_portfolio.to_csv(OUT_DIR / "portfolio_ledger_val_g0b_tau_never_switch.csv", index=False)
    g0b_ok_portfolio = _close(m_g0b_portfolio, rc_mod.G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])
    g0b_portfolio_with_gate = mfe_width._duration_gated(ledger_g0b_portfolio, val_frame, greedy.DURATION_THRESHOLD)
    print(f"  portfolio G0b: no_gate pnl={m_g0b_portfolio['pnl']:.2f}% mdd={m_g0b_portfolio['mdd']:.2f}% trades={m_g0b_portfolio['trades']} match={g0b_ok_portfolio} "
          f"with_gate={g0b_portfolio_with_gate['pnl']:.2f}%/{g0b_portfolio_with_gate['mdd']:.2f}%/{g0b_portfolio_with_gate['trades']} "
          f"rc_hold_bars={diag_g0b_portfolio['rc_hold_bars']} rc_selected_bars={diag_g0b_portfolio['rc_selected_bars']} rc_switch_bars={diag_g0b_portfolio['rc_switch_bars']}", flush=True)
    g0b_pass = bool(g0b_ok_component and g0b_ok_portfolio and diag_g0b_portfolio["rc_switch_bars"] == 0)

    # ======================================================================================
    # stage=G0c -- select_threshold=NEVER_SELECT (stage 1 neutralized independently), tau
    # irrelevant (never reached). Must ALSO reproduce the baseline exactly. Proves stage-1's
    # own plumbing is faithful, independent of stage-2's already-proven fidelity above.
    # ======================================================================================
    print("=== stage=G0c_select_threshold_never_select (stage-1 neutralized) ===", flush=True)
    m_g0c_component, _ = replay_exit_variant_selective_risk_controlled(
        comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
        risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
        fallback_loaded_models=gbdt_component_loaded, fallback_threshold=rc_mod.FALLBACK_THRESHOLD,
        select_threshold=SELECT_THRESHOLD_NEVER_SELECT, tau=rc_mod.TAU_NEVER_SWITCH,
    )
    g0c_ok_component = _close(m_g0c_component, rc_mod.G0_REFERENCE["component_tabm_liveatr"])
    diag_g0c_portfolio, ledger_g0c_portfolio = greedy_replay_selective_risk_controlled(
        val_frame, components_val, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
        risk_component="h48qual", select_threshold=SELECT_THRESHOLD_NEVER_SELECT, tau=rc_mod.TAU_NEVER_SWITCH,
    )
    m_g0c_portfolio = portfolio._ledger_metrics(ledger_g0c_portfolio)
    g0c_ok_portfolio = _close(m_g0c_portfolio, rc_mod.G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])
    print(f"  component G0c match={g0c_ok_component}  portfolio G0c match={g0c_ok_portfolio} rc_selected_bars={diag_g0c_portfolio['rc_selected_bars']}(expect 0)", flush=True)
    g0c_pass = bool(g0c_ok_component and g0c_ok_portfolio and diag_g0c_portfolio["rc_selected_bars"] == 0)

    report["g0b_stage2_neutralized"] = {
        "component_tau_never_switch": {"actual": m_g0b_component, "reference": rc_mod.G0_REFERENCE["component_tabm_liveatr"], "match": g0b_ok_component},
        "portfolio_tau_never_switch_no_gate": {"actual": m_g0b_portfolio, "reference": rc_mod.G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"], "match": g0b_ok_portfolio},
        "portfolio_tau_never_switch_with_gate": g0b_portfolio_with_gate,
        "rc_switch_bars_expected_zero": diag_g0b_portfolio["rc_switch_bars"], "pass": g0b_pass,
    }
    report["g0c_stage1_neutralized"] = {
        "component": {"actual": m_g0c_component, "match": g0c_ok_component},
        "portfolio": {"actual": m_g0c_portfolio, "match": g0c_ok_portfolio},
        "rc_selected_bars_expected_zero": diag_g0c_portfolio["rc_selected_bars"], "pass": g0c_pass,
    }
    g0bc_pass = bool(g0b_pass and g0c_pass)
    print(f"stage=G0bc_result pass={g0bc_pass}", flush=True)
    if not g0bc_pass:
        report["stage_reached"] = "G0bc"
        report["gate_pass"] = False
        report["note"] = "G0b/G0c self-consistency failed -- the selective-risk-controlled copies do not reproduce the baseline in their respective degenerate (stage-2-neutralized / stage-1-neutralized) modes. Aborting."
        _write_report(report)
        print("stage=ABORT G0bc failed", flush=True)
        return 1

    # ======================================================================================
    # stage=calibration -- Algorithm 1 (rc_mod._calibrate_threshold, reused unmodified) on the
    # SELECTED-subset-only calibration log (the paper's Z_{lambda_1-bar} restriction).
    # ======================================================================================
    print("=== stage=calibration (Algorithm 1, VAL only, selected-subset-only population) ===", flush=True)
    scrc_log = m_g0b_component["scrc_log"]
    sel_mask = np.asarray(scrc_log["selected"], dtype=bool)
    n_total_held = len(sel_mask)
    n_selected = int(sel_mask.sum())
    delta_all = np.asarray(scrc_log["delta"], dtype=np.float64)
    y_all = np.asarray(scrc_log["y"], dtype=np.int64)
    a0_all = np.asarray(scrc_log["a0"], dtype=np.int64)
    astar_all = np.asarray(scrc_log["astar"], dtype=np.int64)
    delta_sel, y_sel, a0_sel, astar_sel = delta_all[sel_mask], y_all[sel_mask], a0_all[sel_mask], astar_all[sel_mask]
    a0_all_ones = bool(np.all(a0_sel == 1))  # sanity check: at SELECT_THRESHOLD_PRIMARY==EXIT_THRESHOLD, selected must == {a0=1}
    baseline_mismatch_bumped_sel = rc_mod._bumped_risk(delta_sel, y_sel, a0_sel, astar_sel, rc_mod.TAU_NEVER_SWITCH)
    disagreement_bars_sel = int((a0_sel != astar_sel).sum())
    print(f"  n_total_held_bars={n_total_held} n_selected={n_selected} ({n_selected / max(n_total_held, 1) * 100:.2f}% of held bars) "
          f"selected_all_have_a0_eq_1={a0_all_ones} baseline_bumped_mismatch_rate(selected-only)={baseline_mismatch_bumped_sel:.4f} "
          f"disagreement_bars(a0!=astar, selected-only)={disagreement_bars_sel} ({disagreement_bars_sel / max(n_selected, 1) * 100:.2f}%)", flush=True)

    eps_grid = [round(baseline_mismatch_bumped_sel * f, 6) for f in rc_mod.EPS_FRACTIONS]
    calibration_results: dict[str, Any] = {}
    for frac, eps in zip(rc_mod.EPS_FRACTIONS, eps_grid):
        cal = rc_mod._calibrate_threshold(delta_sel, y_sel, a0_sel, astar_sel, eps)
        m_min_paper_feasibility = int(np.ceil(1.0 / eps) - 1) if eps > 0 else None  # paper's own m>=ceil(1/alpha)-1 feasibility floor (Section on limitations, fetched)
        cal["paper_m_min_feasibility_floor"] = m_min_paper_feasibility
        cal["n_selected_clears_paper_floor"] = bool(m_min_paper_feasibility is not None and n_selected >= m_min_paper_feasibility)
        calibration_results[f"{frac:.2f}"] = cal
        print(f"  eps_frac={frac:.2f} eps={eps:.4f} -> tau_hat={cal['tau_hat']:.4f} risk_at_tau_hat={cal['risk_at_tau_hat']:.4f} "
              f"feasible={cal['feasible_count']}/{cal['grid_size']} paper_m_min={m_min_paper_feasibility} n_selected_clears_floor={cal['n_selected_clears_paper_floor']}", flush=True)

    report["calibration"] = {
        "n_total_held_bars": n_total_held, "n_selected": n_selected, "selection_rate_pct": n_selected / max(n_total_held, 1) * 100.0,
        "selected_all_have_a0_eq_1_sanity_check": a0_all_ones,
        "baseline_bumped_mismatch_rate_selected_subset": baseline_mismatch_bumped_sel,
        "disagreement_bars_a0_ne_astar_selected_subset": disagreement_bars_sel,
        "eps_fractions": rc_mod.EPS_FRACTIONS, "eps_grid": eps_grid, "candidates": calibration_results,
        "note_vs_rc8_calibration_population": f"#8 calibrated on ALL {n_total_held} held VAL bars; this candidate calibrates on only the {n_selected} stage-1-selected bars ({n_selected / max(n_total_held, 1) * 100:.2f}%) -- the direct, load-bearing consequence of adding a real stage-1 gate.",
    }

    # ---- diagnostic-only (never gated): select_threshold=0.50, single eps_frac=0.90 point, for contrast ----
    print(f"=== stage=diagnostic_select_threshold_{SELECT_THRESHOLD_DIAGNOSTIC:.2f} (NOT gated, context only) ===", flush=True)
    m_diag_component, _ = replay_exit_variant_selective_risk_controlled(
        comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
        risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
        fallback_loaded_models=gbdt_component_loaded, fallback_threshold=rc_mod.FALLBACK_THRESHOLD,
        select_threshold=SELECT_THRESHOLD_DIAGNOSTIC, tau=rc_mod.TAU_NEVER_SWITCH,
    )
    diag_log = m_diag_component["scrc_log"]
    diag_sel_mask = np.asarray(diag_log["selected"], dtype=bool)
    n_selected_diag = int(diag_sel_mask.sum())
    delta_diag, y_diag, a0_diag, astar_diag = (np.asarray(diag_log[k], dtype=np.float64 if k == "delta" else np.int64)[diag_sel_mask] for k in ("delta", "y", "a0", "astar"))
    a0_diag_frac_positive = float(a0_diag.mean()) if n_selected_diag else 0.0
    baseline_mismatch_bumped_diag = rc_mod._bumped_risk(delta_diag, y_diag, a0_diag.astype(np.int64), astar_diag.astype(np.int64), rc_mod.TAU_NEVER_SWITCH)
    eps_diag = round(baseline_mismatch_bumped_diag * 0.90, 6)
    cal_diag = rc_mod._calibrate_threshold(delta_diag, y_diag, a0_diag.astype(np.int64), astar_diag.astype(np.int64), eps_diag)
    print(f"  select_threshold={SELECT_THRESHOLD_DIAGNOSTIC:.2f}: n_selected={n_selected_diag} ({n_selected_diag / max(n_total_held, 1) * 100:.2f}% of held bars, "
          f"vs {n_selected}={n_selected / max(n_total_held, 1) * 100:.2f}% at primary 0.95) a0_fraction_within_selected={a0_diag_frac_positive:.4f} "
          f"(< 1.0 confirms selection is genuinely wider than {{a0=1}} here) tau_hat(eps_frac=0.90)={cal_diag['tau_hat']:.4f}", flush=True)
    report["diagnostic_select_threshold_050_not_gated"] = {
        "select_threshold": SELECT_THRESHOLD_DIAGNOSTIC, "n_selected": n_selected_diag,
        "selection_rate_pct": n_selected_diag / max(n_total_held, 1) * 100.0,
        "a0_fraction_within_selected": a0_diag_frac_positive,
        "baseline_bumped_mismatch_rate": baseline_mismatch_bumped_diag,
        "eps_frac_0_90_calibration": cal_diag,
        "note": "Diagnostic only, NEVER gated. Shows selection is genuinely wider than {a0=1} at this threshold (a0_fraction_within_selected<1), i.e. stage 2 here CAN be reached on bars TabM itself was not yet confident about -- contrast with the primary 0.95 threshold, which by construction restricts stage 2 to cancel-only.",
    }

    # ======================================================================================
    # stage=VAL_candidate_sweep -- primary select_threshold=0.95, 3 pre-registered eps
    # fractions, dual gate criterion (identical to #8's).
    # ======================================================================================
    print("=== stage=VAL_candidate_sweep (select_threshold=primary=0.95) ===", flush=True)
    val_candidates: dict[str, Any] = {}
    for frac in rc_mod.EPS_FRACTIONS:
        key = f"{frac:.2f}"
        tau_hat = calibration_results[key]["tau_hat"]
        m_comp, ledger_comp = replay_exit_variant_selective_risk_controlled(
            comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
            risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
            exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
            cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
            fallback_loaded_models=gbdt_component_loaded, fallback_threshold=rc_mod.FALLBACK_THRESHOLD,
            select_threshold=SELECT_THRESHOLD_PRIMARY, tau=tau_hat,
        )
        diag_port, ledger_port = greedy_replay_selective_risk_controlled(
            val_frame, components_val, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            risk_component="h48qual", select_threshold=SELECT_THRESHOLD_PRIMARY, tau=tau_hat,
        )
        m_port_no_gate = portfolio._ledger_metrics(ledger_port)
        m_port_with_gate = mfe_width._duration_gated(ledger_port, val_frame, greedy.DURATION_THRESHOLD)
        ledger_comp.to_csv(OUT_DIR / f"component_ledger_val_eps{frac:.2f}.csv", index=False)
        ledger_port.to_csv(OUT_DIR / f"portfolio_ledger_val_eps{frac:.2f}.csv", index=False)

        gate_component_pnl = float(m_comp["pnl"]) >= float(rc_mod.G0_REFERENCE["component_tabm_liveatr"]["pnl"])
        gate_component_mdd = float(m_comp["mdd"]) >= float(rc_mod.G0_REFERENCE["component_tabm_liveatr"]["mdd"])
        gate_portfolio_pnl = float(m_port_no_gate["pnl"]) >= float(mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["val"][0]["pnl"])
        gate_portfolio_mdd = float(m_port_no_gate["mdd"]) >= float(mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["val"][0]["mdd"])
        gate_original_pass = bool(gate_component_pnl and gate_component_mdd and gate_portfolio_pnl and gate_portfolio_mdd)

        gate_relaxed_main = float(m_port_with_gate["pnl"]) > float(g0b_portfolio_with_gate["pnl"])
        gate_relaxed_mdd = (float(m_port_with_gate["mdd"]) - float(g0b_portfolio_with_gate["mdd"])) >= -3.0
        gate_relaxed_guardrail = rc_mod._guardrail_ok(float(rc_mod.G0_REFERENCE["component_tabm_liveatr"]["pnl"]), float(m_comp["pnl"]))
        gate_relaxed_pass = bool(gate_relaxed_main and gate_relaxed_mdd and gate_relaxed_guardrail)

        val_candidates[key] = {
            "eps_frac": frac, "tau_hat": tau_hat, "select_threshold": SELECT_THRESHOLD_PRIMARY,
            "component_no_gate": m_comp, "portfolio_no_gate": m_port_no_gate, "portfolio_with_gate": m_port_with_gate,
            "rc_diag": {k: v for k, v in diag_port.items() if k != "reason_counts"},
            "gate_original": {"component_pnl_nonworse": gate_component_pnl, "component_mdd_nonworse": gate_component_mdd,
                               "portfolio_pnl_nonworse": gate_portfolio_pnl, "portfolio_mdd_nonworse": gate_portfolio_mdd, "pass": gate_original_pass},
            "gate_relaxed": {"portfolio_with_gate_pnl_improved": gate_relaxed_main, "portfolio_with_gate_mdd_within_3pp": gate_relaxed_mdd,
                              "component_guardrail_ok": gate_relaxed_guardrail, "pass": gate_relaxed_pass},
            "passes_any": bool(gate_original_pass or gate_relaxed_pass),
        }
        print(f"  eps_frac={frac:.2f} tau_hat={tau_hat:.4f}: component_no_gate={m_comp['pnl']:.2f}%/{m_comp['mdd']:.2f}%/{m_comp['trades']} "
              f"portfolio_no_gate={m_port_no_gate['pnl']:.2f}%/{m_port_no_gate['mdd']:.2f}%/{m_port_no_gate['trades']} "
              f"portfolio_with_gate={m_port_with_gate['pnl']:.2f}%/{m_port_with_gate['mdd']:.2f}%/{m_port_with_gate['trades']} "
              f"selected_bars={diag_port['rc_selected_bars']} switch_bars={diag_port['rc_switch_bars']}(->exit:{diag_port['rc_switch_to_exit_bars']},->hold:{diag_port['rc_switch_to_hold_bars']}) "
              f"gate_original={gate_original_pass} gate_relaxed={gate_relaxed_pass}", flush=True)

    report["val_baseline_portfolio_no_gate"] = mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["val"][0]
    report["val_baseline_portfolio_with_gate"] = g0b_portfolio_with_gate
    report["val_baseline_component_no_gate"] = rc_mod.G0_REFERENCE["component_tabm_liveatr"]
    report["val_candidates"] = val_candidates

    passing_original = [k for k, v in val_candidates.items() if v["gate_original"]["pass"]]
    passing_relaxed = [k for k, v in val_candidates.items() if v["gate_relaxed"]["pass"]]
    passing_any = [k for k, v in val_candidates.items() if v["passes_any"]]
    winner = max(passing_any, key=lambda k: val_candidates[k]["portfolio_with_gate"]["pnl"]) if passing_any else None
    print(f"stage=VAL_gate_result passing_original={passing_original} passing_relaxed={passing_relaxed} passing_any={passing_any} winner={winner}", flush=True)
    report["val_passing_original"] = passing_original
    report["val_passing_relaxed"] = passing_relaxed
    report["val_passing_any"] = passing_any
    report["val_winner"] = winner

    if winner is None:
        report["oos_opened"] = False
        report["stage_reached"] = "VAL_candidate_sweep"
        report["gate_pass"] = False
        report["note"] = "No eps candidate passed either gate criterion on VAL -- OOS NOT opened, per this project's methodology discipline. Negative pilot result."
        _write_report(report)
        print("stage=done (negative result, OOS not opened)", flush=True)
        return 0

    # ======================================================================================
    # stage=OOS_multiwindow_single_touch -- eth_omega461_multiwindow_confirmation_gate_20260814
    # required infra: OOS-Q1+OOS-Q2 opened together, once, tau_hat frozen from VAL. 2025 Q1-Q3
    # shown as context only.
    # ======================================================================================
    winner_tau = val_candidates[winner]["tau_hat"]
    print(f"=== stage=OOS_multiwindow_single_touch winner=eps_frac{winner} tau_hat={winner_tau:.4f} select_threshold={SELECT_THRESHOLD_PRIMARY} ===", flush=True)
    baseline_tuples: dict[str, tuple] = {}
    candidate_tuples: dict[str, tuple] = {}
    per_window_diag: dict[str, Any] = {}
    for wname in mw_gate.ALL_WINDOWS:
        baseline, candidate, diag_c = _scrc_variant(wname, windows, h48qual_cfg, zig075_cfg, device, OUT_DIR, fee, slip, base_cols, gbdt_models, SELECT_THRESHOLD_PRIMARY, winner_tau)
        baseline_tuples[wname] = baseline
        candidate_tuples[wname] = candidate
        per_window_diag[wname] = {k: v for k, v in diag_c.items() if k != "reason_counts"}
        b_ng, b_wg = baseline
        c_ng, c_wg = candidate
        print(f"  {wname}: baseline no_gate={b_ng['pnl']:.2f}%/{b_ng['mdd']:.2f}%/{b_ng['trades']} with_gate={b_wg['pnl']:.2f}%/{b_wg['mdd']:.2f}%/{b_wg['trades']}  |  "
              f"candidate no_gate={c_ng['pnl']:.2f}%/{c_ng['mdd']:.2f}%/{c_ng['trades']} with_gate={c_wg['pnl']:.2f}%/{c_wg['mdd']:.2f}%/{c_wg['trades']} "
              f"selected={diag_c['rc_selected_bars']} switch={diag_c['rc_switch_bars']}", flush=True)

    val_baseline_check = _close(baseline_tuples["val"][0], mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["val"][0])
    oosq1_baseline_check = _close(baseline_tuples["oos_q1"][0], mw_gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["oos_q1"][0])
    print(f"  cross-check baseline ledgers vs G0 reference: val={val_baseline_check} oos_q1={oosq1_baseline_check}", flush=True)

    summary_strict = mw_gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = mw_gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    print(f"stage=OOS_result verdict_strict={summary_strict['final_verdict']} verdict_relaxed_mdd3pp={summary_relaxed['final_verdict']}", flush=True)

    report.update({
        "oos_opened": True, "oos_winner_eps_frac": winner, "oos_winner_tau_hat": winner_tau,
        "oos_select_threshold": SELECT_THRESHOLD_PRIMARY,
        "multiwindow_by_window_diag": per_window_diag,
        "multiwindow_baseline_cross_checks": {"val_no_gate_match": val_baseline_check, "oos_q1_no_gate_match": oosq1_baseline_check},
        "multiwindow_summary_strict_mdd0pp": summary_strict,
        "multiwindow_summary_relaxed_mdd3pp": summary_relaxed,
        "final_verdict_strict": summary_strict["final_verdict"],
        "final_verdict_relaxed": summary_relaxed["final_verdict"],
        "stage_reached": "OOS_multiwindow_single_touch",
        "gate_pass": True,
    })
    _write_report(report)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
