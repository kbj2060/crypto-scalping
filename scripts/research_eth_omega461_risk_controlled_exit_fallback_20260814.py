#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #8: literature-scouting rank-2 candidate
(docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md, section 3 table, "Risk-
Controlled Post-Processing of Decision Policies", Joshi/Wang/Hassani/Dobriban, arXiv:2605.06479,
2026-05-07 -- read directly via WebFetch before writing this script, not from the scouting doc's
one-paragraph summary).

=== Paper mechanism (as confirmed from the paper itself, not the scouting doc) ===
Setting: a "stakeholder-reluctant-to-change" baseline policy pi0(x) and a fitted fallback/oracle
policy pi*(x). A violation event ell(a,y)>=c (loss of action a under true outcome y exceeds a
threshold). Population-level optimization: maximize agreement with baseline subject to a MARGINAL
chance constraint, max_pi P(pi(X)=pi0(X)) s.t. P(ell(pi(X),Y)>=c)<=epsilon. Theorem 3.1: the
optimal policy has THRESHOLD structure on an oracle score Delta(x) := g(pi0(x),x) - g(pi*(x),x)
(g(a,x) = conditional violation risk of taking action a at x) -- switch to the fallback action
pi*(x) iff Delta(x)>=tau, else keep pi0(x). Finite-sample calibration (their Algorithm 1): given a
calibration set and a FITTED score Delta_hat (fit on a training sample, held fixed), build a
threshold grid T_n = {0, top} union {Delta_hat_i}, compute for each candidate t the "bumped"
empirical violation risk R_hat_n^+(t) = (sum_i 1{ell(pi_hat(X_i,t),Y_i)>=c} + 1)/(n+1) (a split-
conformal-style +1/(n+1) correction), take the feasible set F_n = {t : R_hat_n^+(t)<=epsilon}, and
select tau_hat = max(F_n) (or 0 if F_n is empty) -- i.e. the LARGEST threshold (= least switching,
maximal agreement with baseline) that still empirically satisfies the risk budget. Their Theorem
4.2 (general fitted-fallback case, i.i.d. regularity conditions): E[violation risk of the deployed
policy] <= epsilon + C3*log(n+1)/(n+1) -- an O(log n / n) EXCESS risk above the target budget. A
sharper special case (not used here, see note below) applies when the fallback is "exact-safe"
(ell(pi_safe(x),y)<c for ALL x,y): then exact risk control (no excess term) holds under
exchangeability, plus a high-probability near-optimality bound on the agreement objective. This
project's GBDT fallback is NOT exact-safe (it can itself violate), so only the general O(log n/n)
guarantee (Theorem 4.2) applies here, not the exact zero-excess special case -- stated honestly in
this script's report and the companion doc, not glossed over.

=== Mapping onto this project (calibration-only, zero retraining) ===
- pi0 = current confirmed h48qual exit_head baseline: TabM live-ATR-relabel exit_head, fixed
  EXIT_THRESHOLD=0.95 ("the policy stakeholders are reluctant to change" -- literally this
  project's own framing of it in prior docs).
- pi* = the ALREADY-TRAINED GBDT exit_head (tmp/causal_regen_20260516/
  eth_omega461_gbdt_exit_head_liveatr_20260813/h48qual/gbdt_exit_bundle.pkl, Odyssey2 #4, rejected
  as a full replacement because it hurt component-level economics) at its own natural threshold
  0.95 -- same convention as the TabM baseline, not a new hyperparameter. GBDT was picked over TCN
  per the coordinator's instruction (lighter, and already the more thoroughly diagnosed of the two
  in this project's history).
- g(a,x): rather than fitting a NEW separate risk model (which the paper's LLM-routing worked
  example does via two logistic regressions predicting each policy's own error probability), this
  script reuses GBDT's OWN probability p_gbdt(x) (frozen, already trained, independent of TabM's
  decision rule) as the shared risk-of-mismatch estimate: g(hold,x)=p_gbdt(x), g(exit,x)=1-p_gbdt(x).
  This is a direct, non-circular use of an INDEPENDENT model (different model class, different
  decision boundary than TabM) as the arbiter -- not TabM scoring itself. Delta(x) then collapses
  to exactly 0 whenever pi0(x) and pi*(x) AGREE (no reason to ever switch) and is large in magnitude
  exactly where they DISAGREE and GBDT is confident -- i.e. this concretely operationalizes the
  "fallback and baseline strongly disagree" signal the coordinator named as one candidate example
  for "risky context", now with the paper's exact threshold-selection machinery wrapped around it
  instead of an ad hoc disagreement cutoff.
- y (ground truth, calibration-only, VAL data): the SAME "pos_giveback>=0.65 OR pos_unrealized<=
  -0.010" rule documented as 98.1% of the live-ATR exit label's positive class
  (docs/experiments/eth_omega461_gbdt_exit_head_20260813.md) -- computed causally at each bar
  directly from the SAME pos_giveback/pos_unrealized values already being fed into both models
  (zero new feature engineering, zero lookahead: both quantities are backward-looking functions of
  the position's own realized path up to and including bar i).
- ell(a,y)=1{a!=y} (0-1 mismatch loss), c=1 -- "violation" = the chosen action disagrees with this
  causal rule at that bar.
- epsilon (risk budget): pre-registered (BEFORE running the script / seeing any PnL) as three
  fractions {0.90, 0.70, 0.50} of the baseline's OWN bumped mismatch rate against the rule
  (measured under tau=TAU_NEVER_SWITCH, i.e. the baseline's actual VAL holding pattern, unconfounded
  by the intervention -- same "measured under baseline's own policy" discipline
  eth_omega461_queue_pressure_exit_threshold_20260814.py used for its pressure-frequency diagnostic).

Calibration happens ONLY on VAL (2025-10-01..2025-12-31); OOS (2026-01-01..2026-03-31) is opened at
most once, only for the single VAL winner, with tau_hat FROZEN from VAL (no OOS recalibration).

=== Dual judging criterion (per coordinator instruction, both reported for every VAL candidate) ===
(a) ORIGINAL 4-metric gate (matches Odyssey2 #4/#5's own criterion): component NO_GATE PnL+MDD
    AND portfolio NO_GATE PnL+MDD all non-worse than the TabM live-ATR baseline.
(b) RELAXED gate (docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md, agreed with the
    user earlier this session): portfolio WITH_GATE PnL strictly improved (main criterion) AND
    portfolio WITH_GATE MDD within 3pp of baseline AND a guardrail -- component NO_GATE PnL must not
    flip sign or worsen more than 50% relative to baseline (this experiment's exit_head decision
    source does change, conditionally, in a narrow subset of bars, so the guardrail applies exactly
    as it did to GBDT/TCN).
If ANY VAL candidate passes EITHER criterion, OOS is opened once for the single best (by portfolio
WITH_GATE PnL) passing candidate.

=== Implementation pattern (same "renamed copy, only the exit-head decision block changed" style as
Odyssey2 #4/#5/#7) ===
research_eth_omega461_exit_sweep_20260721.py and replay_omega4_6_1_greedy_router_20260706.py are
NEVER edited -- only imported and read. This script defines renamed copies
(replay_exit_variant_risk_controlled / greedy_replay_risk_controlled) with exactly one block changed
(the exit-head probability-vs-threshold comparison), verified faithful via a degenerate-tau ("G0b")
self-check that must reproduce the untouched baseline numbers exactly. The GBDT duck-typed wrapper
itself (GBDTExitHeadWrapper) is reused unmodified from research_eth_omega461_gbdt_exit_head_val_
20260813.py -- this script only composes it differently (attaches it ALONGSIDE the TabM runtime
under a new key, rather than replacing exit_runtime wholesale, since both probabilities are needed
at every bar here).

fresh_forward_bar_by_bar=true (both replay copies are single causal forward passes, i increasing,
only bar i and already-closed history used at bar i; the causal rule-label y(x) uses only the
position's own already-realized giveback/unrealized-PnL as of bar i). trade_ledgers_used_as_input=
false (ledgers are written-only outputs). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. direction_head/quality_head/quality_threshold/encoder frozen and
unchanged for BOTH components -- only h48qual's exit_head DECISION (never its TabM weights) is made
conditional on the calibrated Delta(x)>=tau_hat rule. zig075 is not touched in any way.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
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
import train_eval_omega461_gbdt_exit_head_liveatr_20260813 as gbdt_train  # noqa: E402
import research_eth_omega461_gbdt_exit_head_val_20260813 as gbdt_val  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814"
GBDT_BUNDLE = gbdt_train.OUT_DIR / "h48qual" / "gbdt_exit_bundle.pkl"

FALLBACK_THRESHOLD = 0.95  # GBDT's own natural decision threshold -- same convention as the TabM
# baseline's EXIT_THRESHOLD, not a new hyperparameter (matches how Odyssey2 #4's own VAL comparison
# scored GBDT at exit_threshold=sweep.BASELINE_EXIT_THRESHOLD).
Y_RULE_GIVEBACK = 0.65
Y_RULE_UNREALIZED = -0.010  # pos_giveback>=Y_RULE_GIVEBACK OR pos_unrealized<=Y_RULE_UNREALIZED --
# documented as 98.1% of the live-ATR exit label's positive class (mfe_giveback_exit 75.6% +
# adverse_unreal_exit 22.5%, docs/experiments/eth_omega461_gbdt_exit_head_20260813.md). Computed
# causally at each held bar from the SAME giveback/move values already fed into pos_values -- zero
# new feature engineering, zero lookahead.
TAU_NEVER_SWITCH = 10.0  # sentinel > max possible |Delta|=1 (Delta in [-1,1] by construction, see
# _risk_controlled_action docstring) -- degenerate case, must reproduce the untouched baseline
# exactly (this script's G0b self-check).
EPS_FRACTIONS = [0.90, 0.70, 0.50]  # PRE-REGISTERED before running / seeing any PnL number --
# fractions of the baseline's own bumped mismatch-vs-rule rate (aggressive to mild), same 3-point
# grid size as Odyssey2 #7's queue-pressure threshold sweep.
G0_TOLERANCE_PP = 0.05

G0_REFERENCE = {
    # Published in docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md, reused
    # verbatim as the G0 reference by Odyssey2 #4/#5/#7 -- same numbers, same source.
    "component_baseline_original": {"pnl": 5.45, "mdd": -11.62, "trades": 29},
    "component_tabm_liveatr": {"pnl": 9.23, "mdd": -7.59, "trades": 63},
    "portfolio_baseline_both_original": {"pnl": 36.82, "mdd": -24.34, "trades": 29},
    "portfolio_asymmetric_tabm_liveatr": {"pnl": 46.59, "mdd": -21.70, "trades": 35},
}
# NOTE (discovered while building THIS script, verified by direct computation against the exact
# 35-trade canonical ledger on disk): docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.
# md's rescoring table shows "baseline with_gate PnL+54.88%/MDD-31.11%" uniformly for every VAL row
# (regime-threshold/GBDT/TCN/queue-pressure alike) -- but this is actually
# mfe_width._duration_gated(portfolio_ledger_baseline_both_original.csv, ...) (BOTH components on
# their ORIGINAL frozen exit heads, no_gate 36.82%/-24.34%/29 trades), NOT
# asymmetric_tabm_liveatr's own with_gate (no_gate 46.59%/-21.70%/35 trades -- the actual baseline
# GBDT/TCN/queue-pressure/this experiment all compare their OWN no_gate numbers against). The two
# baselines' with_gate values differ substantially (54.88/-31.11 vs the freshly-computed value this
# script establishes below) because they are different 29-trade vs 35-trade ledgers. This did not
# change any PAST verdict (GBDT/TCN failed on the component guardrail either way; regime-threshold
# failed the PnL-improvement criterion against either number; queue-pressure passed against either
# number) -- recorded here only so this script does not repeat the mix-up: this experiment's own
# relaxed-gate baseline is computed FRESH from asymmetric_tabm_liveatr's ledger via G0b below
# (portfolio_tau_never_switch_with_gate in the report), not hardcoded against a value that turned
# out to belong to a different ledger. Kept as BASELINE_BOTH_ORIGINAL_WITH_GATE purely for citation/
# context in the report, never used as a gate threshold.
BASELINE_BOTH_ORIGINAL_WITH_GATE_CONTEXT_ONLY = {"pnl": 54.88, "mdd": -31.11}
OOS_BASELINE_REFERENCE_NO_GATE = {"pnl": 93.27, "mdd": -15.48, "trades": 24}
# Same caveat for OOS: docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md's "대기압력
# (0.80) OOS" row baseline with_gate (67.25/-15.48) may be subject to the identical baseline-ledger
# mix-up. This script does not hardcode-check against it (never gated on it); the OOS relaxed-gate
# reference is computed FRESH from this run's own OOS TabM-live-ATR baseline ledger.
OOS_BASELINE_BOTH_ORIGINAL_WITH_GATE_CONTEXT_ONLY = {"pnl": 67.25, "mdd": -15.48}
OOS_CAVEAT_TEXT = (
    "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by the TabM-liveATR baseline "
    "and the risk-controlled candidate here (this experiment only modulates h48qual's exit_head "
    "DECISION conditionally -- it never touches quality_threshold or the direction/quality heads, "
    "which are frozen in both variants), was itself OOS-pnl-primary selected against a frame "
    "spanning 2026-01-01..2026-02-28 -- the first two of this OOS window's three months (see "
    "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
    "relative comparison (candidate vs baseline within this run) remains meaningful because both "
    "share the identical contaminated entry-selection layer; the absolute OOS PnL/MDD figures below "
    "are not clean unbiased forward performance and must not be over-interpreted as such."
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP, check_trades: bool = True) -> bool:
    ok = bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
    )
    if check_trades and "trades" in expected:
        ok = ok and int(actual["trades"]) == int(expected["trades"])
    return ok


# =====================================================================================================
# Core mechanism: Theorem 3.1's threshold policy + Algorithm 1's finite-sample calibration
# (arXiv:2605.06479, Joshi/Wang/Hassani/Dobriban) -- pure functions, unit-testable in isolation,
# called identically from both renamed-copy replay loops below so the actual NEW math exists in
# exactly one place (the surrounding loop bodies are kept as close to byte-identical copies of the
# originals as possible, per this project's established "renamed copy, minimal diff" discipline).
# =====================================================================================================


def _risk_controlled_action(
    prob_baseline: float, prob_fallback: float, *, baseline_threshold: float, fallback_threshold: float,
    tau: float, y_rule: int,
) -> dict[str, Any]:
    """Delta(x) = g(pi0(x),x) - g(pi*(x),x), g(hold,x)=prob_fallback, g(exit,x)=1-prob_fallback
    (GBDT's own probability used as the shared, TabM-independent risk-of-mismatch estimate for
    BOTH candidate actions -- see module docstring "g(a,x)" note). Delta in [-1,1]: it is exactly 0
    whenever pi0 and pi* agree (a0==astar), and its sign/magnitude only matters when they disagree.
    Switch to the fallback's action iff Delta>=tau (Theorem 3.1)."""
    a0 = bool(prob_baseline >= baseline_threshold)
    astar = bool(prob_fallback >= fallback_threshold)
    g_hold, g_exit = float(prob_fallback), 1.0 - float(prob_fallback)
    g0 = g_exit if a0 else g_hold
    gstar = g_exit if astar else g_hold
    delta = float(g0 - gstar)
    switched = bool(delta >= float(tau))
    final_exit = astar if switched else a0
    return {"a0": a0, "astar": astar, "delta": delta, "switched": switched, "final_exit": bool(final_exit), "y": int(y_rule)}


def _bumped_risk(delta: np.ndarray, y: np.ndarray, a0: np.ndarray, astar: np.ndarray, tau: float) -> float:
    """R_hat_n^+(t) = (sum_i 1{action(X_i,t) != y_i} + 1) / (n+1) -- the paper's "bumped" (split-
    conformal-style +1/(n+1)) empirical violation risk of the nested policy pi_hat(.,t)."""
    switched = delta >= float(tau)
    action = np.where(switched, astar, a0)
    mismatch = action != y
    return float((int(mismatch.sum()) + 1) / (len(delta) + 1))


def _calibrate_threshold(delta: np.ndarray, y: np.ndarray, a0: np.ndarray, astar: np.ndarray, eps: float) -> dict[str, Any]:
    """Algorithm 1: T_n = {0, top} union {Delta_hat_i}; F_n = {t in T_n : R_hat_n^+(t) <= eps};
    tau_hat = max(F_n), or 0 if F_n is empty (paper's stated fallback rule). No filtering of the
    grid to nonnegative Delta values -- this follows the paper's literal T_n definition exactly
    (observed Delta_i can be negative; a negative tau would mean switching even where the fallback
    is assessed as slightly worse, which the algorithm does not forbid a priori -- reported as a
    diagnostic if it ever occurs, not silently avoided)."""
    n = len(delta)
    top = float(delta.max()) + 1.0 if n else 1.0
    grid = np.unique(np.concatenate([[0.0], delta.astype(np.float64), [top]]))
    risk_by_t: dict[float, float] = {}
    feasible: list[float] = []
    for t in grid:
        r = _bumped_risk(delta, y, a0, astar, float(t))
        risk_by_t[float(t)] = r
        if r <= eps:
            feasible.append(float(t))
    tau_hat = max(feasible) if feasible else 0.0
    return {
        "eps": float(eps), "tau_hat": float(tau_hat), "n_calibration_bars": int(n),
        "grid_size": int(len(grid)), "feasible_count": int(len(feasible)),
        "risk_at_tau_hat": _bumped_risk(delta, y, a0, astar, tau_hat),
        "risk_at_never_switch_sentinel": _bumped_risk(delta, y, a0, astar, top),
        "tau_hat_is_negative": bool(tau_hat < 0.0),
    }


def _guardrail_ok(component_baseline_pnl: float, component_candidate_pnl: float) -> bool:
    """Relaxed-gate guardrail (docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md):
    component (no_gate) PnL must not flip sign nor worsen more than 50% relative to baseline.
    Formula validated against that doc's own worked examples: GBDT 9.23->2.72 = -70.5% relative
    (below the 50% floor of 4.615 -> guardrail FAIL, matches doc); TCN 9.23->-7.74 sign flip ->
    guardrail FAIL, matches doc."""
    if component_baseline_pnl <= 0:
        return component_candidate_pnl >= component_baseline_pnl
    return bool(component_candidate_pnl > 0 and component_candidate_pnl >= component_baseline_pnl * 0.5)


# =====================================================================================================
# GBDT fallback runtime construction -- 100% reuse of research_eth_omega461_gbdt_exit_head_val_
# 20260813's unmodified GBDTExitHeadWrapper / _gbdt_loaded_models / _inject_gbdt_exit_runtime, only
# composed differently: this experiment needs the GBDT runtime ALONGSIDE the TabM one (both
# probabilities scored at every bar), not as a wholesale replacement of exit_runtime.
# =====================================================================================================


def _gbdt_component_fallback_runtime(base_x: pd.DataFrame, base_cols: list[str], gbdt_models: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[int]]:
    gbdt_loaded = gbdt_val._gbdt_loaded_models(base_cols, gbdt_models, device)
    _fb_base_np, fb_exit_runtime, fb_pos_idx = rs._prepare_exit_runtime(base_x, gbdt_loaded)
    return fb_exit_runtime, fb_pos_idx


def _gbdt_portfolio_fallback(prepped: dict[str, Any], base_cols: list[str], gbdt_models: dict[str, Any], device: torch.device) -> dict[str, Any]:
    injected = gbdt_val._inject_gbdt_exit_runtime(prepped, gbdt_models, device, base_cols)
    out = dict(prepped)
    out["fallback_exit_runtime"] = injected["exit_runtime"]
    out["fallback_exit_threshold"] = FALLBACK_THRESHOLD
    return out


# =====================================================================================================
# Renamed copy #1 (component level): research_eth_omega461_exit_sweep_20260721.replay_exit_variant.
# That module is never edited -- only imported and read to produce this copy. Every line below is
# unchanged EXCEPT the block marked "--- risk-controlled: only new logic vs replay_exit_variant ---".
# =====================================================================================================


@torch.no_grad()
def replay_exit_variant_risk_controlled(
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
    fallback_threshold: float = FALLBACK_THRESHOLD,
    tau: float = TAU_NEVER_SWITCH,
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
    # --- risk-controlled: only new logic vs replay_exit_variant (setup) ---
    fallback_exit_runtime = None
    if fallback_loaded_models is not None:
        _fb_base_np, fallback_exit_runtime, fb_pos_idx = rs._prepare_exit_runtime(base_x, fallback_loaded_models)
        if fb_pos_idx != pos_idx:
            raise RuntimeError("risk-controlled: fallback pos_idx does not match baseline pos_idx")
    rc_log: dict[str, list[Any]] = {"delta": [], "y": [], "a0": [], "astar": [], "switched": []}
    # --- end risk-controlled setup ---

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
                # --- risk-controlled: only new logic vs replay_exit_variant (decision) ---
                if fallback_exit_runtime is not None:
                    prob_fb = rs._predict_exit_prob_one(base_np, fallback_exit_runtime, pos_idx, row_i=int(i), expert=expert, pos_values=pos_values, device=device)
                    y_rule = 1 if (giveback_clipped >= Y_RULE_GIVEBACK or float(move) <= Y_RULE_UNREALIZED) else 0
                    rc = _risk_controlled_action(
                        prob, prob_fb, baseline_threshold=float(exit_threshold), fallback_threshold=float(fallback_threshold),
                        tau=float(tau), y_rule=y_rule,
                    )
                    final_exit = rc["final_exit"]
                    rc_log["delta"].append(rc["delta"]); rc_log["y"].append(rc["y"])
                    rc_log["a0"].append(int(rc["a0"])); rc_log["astar"].append(int(rc["astar"])); rc_log["switched"].append(int(rc["switched"]))
                # --- end risk-controlled decision ---
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
        metrics["risk_controlled_log"] = rc_log
    return metrics, ledger


# =====================================================================================================
# Renamed copy #2 (portfolio level): replay_omega4_6_1_greedy_router_20260706.greedy_replay. That
# module is never edited -- only imported and read to produce this copy. Every line below is
# unchanged EXCEPT the block marked "--- risk-controlled: only new logic vs greedy_replay ---".
# =====================================================================================================


@torch.no_grad()
def greedy_replay_risk_controlled(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    risk_component: str = "h48qual",
    tau: float = TAU_NEVER_SWITCH,
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
    # --- risk-controlled: only new logic vs greedy_replay (counters) ---
    rc_hold_bars = 0
    rc_switch_bars = 0
    rc_switch_to_exit = 0
    rc_switch_to_hold = 0
    # --- end counters ---

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
                # --- risk-controlled: only new logic vs greedy_replay (decision) ---
                fallback_runtime = comp.get("fallback_exit_runtime")
                if active_comp == risk_component and fallback_runtime is not None:
                    rc_hold_bars += 1
                    prob_fb = rs._predict_exit_prob_one(
                        comp["base_np"], fallback_runtime, comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    y_rule = 1 if (giveback_clipped >= Y_RULE_GIVEBACK or move <= Y_RULE_UNREALIZED) else 0
                    rc = _risk_controlled_action(
                        prob, prob_fb, baseline_threshold=float(comp["exit_threshold"]),
                        fallback_threshold=float(comp.get("fallback_exit_threshold", FALLBACK_THRESHOLD)),
                        tau=float(tau), y_rule=y_rule,
                    )
                    final_exit = rc["final_exit"]
                    if rc["switched"]:
                        rc_switch_bars += 1
                        if rc["astar"] and not rc["a0"]:
                            rc_switch_to_exit += 1
                        elif rc["a0"] and not rc["astar"]:
                            rc_switch_to_hold += 1
                # --- end risk-controlled decision ---
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
        "rc_hold_bars": rc_hold_bars, "rc_switch_bars": rc_switch_bars,
        "rc_switch_to_exit_bars": rc_switch_to_exit, "rc_switch_to_hold_bars": rc_switch_to_hold,
        "tau_used": float(tau),
    }
    return diag, pd.DataFrame(rows)


# =====================================================================================================
# OOS alignment helper -- local copy of research_eth_omega461_exit_head_portfolio_asymmetric_
# oos_confirm_20260813._align_frame_and_oos_predictions (logic copied verbatim, including the
# WIDE24_2026 95-bar/0.37% Regime3-route-probability coverage-gap fix that script discovered), same
# pattern eth_omega461_queue_pressure_exit_threshold_20260814.py used -- only OUT_DIR differs, so
# this script writes its own aligned CSVs instead of side-effecting into another experiment's dir.
# =====================================================================================================


def _align_frame_and_oos_predictions(oos_frame: pd.DataFrame, q_tags: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Path]]:
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


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 #8 -- Risk-Controlled Post-Processing of Decision Policies (Joshi/Wang/"
            "Hassani/Dobriban, arXiv:2605.06479). h48qual exit_head MODEL stays the TabM live-ATR "
            "baseline; the already-trained GBDT exit_head is used as a calibrated fallback that "
            "only overrides the baseline's hold/exit decision when Delta(x)=g(pi0(x),x)-g(pi*(x),x) "
            ">= tau_hat, tau_hat selected via the paper's Algorithm 1 on VAL only. zig075 untouched."
        ),
        "paper_citation": "Joshi, Wang, Hassani, Dobriban, arXiv:2605.06479 (submitted 2026-05-07)",
        "fallback_model": "GBDT exit_head (Odyssey2 #4), tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_liveatr_20260813/h48qual/gbdt_exit_bundle.pkl",
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_window": [sweep.OOS_START, sweep.OOS_END],
        "eps_fractions_preregistered": EPS_FRACTIONS,
        "y_rule": {"pos_giveback_ge": Y_RULE_GIVEBACK, "pos_unrealized_le": Y_RULE_UNREALIZED},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    if not GBDT_BUNDLE.exists():
        raise FileNotFoundError(f"GBDT bundle not found: {GBDT_BUNDLE}")
    with open(GBDT_BUNDLE, "rb") as f:
        gbdt_bundle = pickle.load(f)
    gbdt_models = gbdt_bundle["models"]

    device = portfolio.DEVICE
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    gbdt_component_loaded = gbdt_val._gbdt_loaded_models(base_cols, gbdt_models, device)

    # ======================================================================================
    # stage=G0_self_check -- component (h48cons._evaluate_val, unmodified) + portfolio
    # (portfolio.run_variant x2, unmodified) reproduction of already-published reference
    # numbers, EXACTLY as Odyssey2 #4/#5/#7 did.
    # ======================================================================================
    print("=== stage=G0_self_check (published reference reproduction, unmodified harnesses) ===", flush=True)
    g0_component = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    g0_ok_component_baseline = _close(g0_component["baseline"], G0_REFERENCE["component_baseline_original"])
    g0_ok_component_tabm = _close(g0_component["h48cons_relabel"], G0_REFERENCE["component_tabm_liveatr"])
    print(f"  component baseline_original={g0_component['baseline']} match={g0_ok_component_baseline}", flush=True)
    print(f"  component tabm_liveatr={g0_component['h48cons_relabel']} match={g0_ok_component_tabm}", flush=True)

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
        {"h48qual": h48qual_cfg, "zig075": zig075_cfg},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    g0_ok_portfolio_baseline = _close(portfolio_baseline, G0_REFERENCE["portfolio_baseline_both_original"])
    g0_ok_portfolio_tabm = _close(portfolio_tabm_liveatr, G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])
    print(f"  portfolio baseline_both_original={portfolio_baseline} match={g0_ok_portfolio_baseline}", flush=True)
    print(f"  portfolio asymmetric_tabm_liveatr={portfolio_tabm_liveatr} match={g0_ok_portfolio_tabm}", flush=True)

    g0_pass_task_scope = bool(g0_ok_component_baseline and g0_ok_component_tabm and g0_ok_portfolio_baseline and g0_ok_portfolio_tabm)
    report["g0"] = {
        "component_baseline_original": {"actual": g0_component["baseline"], "reference": G0_REFERENCE["component_baseline_original"], "match": g0_ok_component_baseline},
        "component_tabm_liveatr": {"actual": g0_component["h48cons_relabel"], "reference": G0_REFERENCE["component_tabm_liveatr"], "match": g0_ok_component_tabm},
        "portfolio_baseline_both_original": {"actual": portfolio_baseline, "reference": G0_REFERENCE["portfolio_baseline_both_original"], "match": g0_ok_portfolio_baseline},
        "portfolio_asymmetric_tabm_liveatr": {"actual": portfolio_tabm_liveatr, "reference": G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"], "match": g0_ok_portfolio_tabm},
        "pass": g0_pass_task_scope,
    }
    if not g0_pass_task_scope:
        report["stage_reached"] = "G0_self_check"
        report["gate_pass"] = False
        report["note"] = "G0 failed -- this harness does not reproduce published reference numbers. Aborting."
        _write_report(report)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    # ======================================================================================
    # stage=G0b_self_consistency_and_calibration_log -- run BOTH renamed-copy replay functions
    # in the degenerate tau=TAU_NEVER_SWITCH mode (never switch, must reproduce G0 numbers
    # exactly) WITH the fallback runtime attached, so this single pair of calls both (a) proves
    # the copies are faithful outside the new block, and (b) yields the (Delta, y, a0, astar)
    # calibration log at every h48qual-held VAL bar, measured under the baseline's own actual
    # holding pattern (unconfounded by the intervention itself).
    # ======================================================================================
    print("=== stage=G0b_self_consistency_and_calibration_log (component level, tau=never-switch) ===", flush=True)
    val_pred_h48qual = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{h48qual_cfg['q_tag']}.csv"
    comp_prepped = sweep.prep_component("h48qual", h48qual_cfg, val_frame_raw, val_pred_h48qual, oof=True)
    m_g0b_component, _ledger_g0b_component = replay_exit_variant_risk_controlled(
        comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
        risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
        fallback_loaded_models=gbdt_component_loaded, fallback_threshold=FALLBACK_THRESHOLD, tau=TAU_NEVER_SWITCH,
    )
    g0b_ok_component = _close(m_g0b_component, G0_REFERENCE["component_tabm_liveatr"])
    print(f"  component G0b(tau=never): pnl={m_g0b_component['pnl']:.2f}% mdd={m_g0b_component['mdd']:.2f}% trades={m_g0b_component['trades']} match={g0b_ok_component}", flush=True)

    h48qual_prepped_portfolio = portfolio._prepare_component_val(val_frame, aligned_pred_paths["h48qual"], h48qual_cfg, device)
    h48qual_prepped_portfolio = _gbdt_portfolio_fallback(h48qual_prepped_portfolio, base_cols, gbdt_models, device)
    zig075_prepped_portfolio = portfolio._prepare_component_val(val_frame, aligned_pred_paths["zig075"], zig075_cfg, device)
    components_val = {"h48qual": h48qual_prepped_portfolio, "zig075": zig075_prepped_portfolio}
    diag_g0b_portfolio, ledger_g0b_portfolio = greedy_replay_risk_controlled(
        val_frame, components_val, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
        risk_component="h48qual", tau=TAU_NEVER_SWITCH,
    )
    m_g0b_portfolio = portfolio._ledger_metrics(ledger_g0b_portfolio)
    ledger_g0b_portfolio.to_csv(OUT_DIR / "portfolio_ledger_val_g0b_tau_never_switch.csv", index=False)
    g0b_ok_portfolio = _close(m_g0b_portfolio, G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])
    print(f"  portfolio G0b(tau=never): pnl={m_g0b_portfolio['pnl']:.2f}% mdd={m_g0b_portfolio['mdd']:.2f}% trades={m_g0b_portfolio['trades']} match={g0b_ok_portfolio} rc_hold_bars={diag_g0b_portfolio['rc_hold_bars']} rc_switch_bars={diag_g0b_portfolio['rc_switch_bars']}", flush=True)

    g0b_portfolio_with_gate = mfe_width._duration_gated(ledger_g0b_portfolio, val_frame, greedy.DURATION_THRESHOLD)
    # No prior published reference exists for asymmetric_tabm_liveatr's OWN with_gate number (see
    # BASELINE_BOTH_ORIGINAL_WITH_GATE_CONTEXT_ONLY note above -- the only previously-published
    # "baseline with_gate" figure, 54.88/-31.11, turned out to belong to a DIFFERENT 29-trade
    # ledger). This script establishes it fresh here and uses it as the actual relaxed-gate
    # reference throughout (val_baseline_portfolio_tabm_liveatr_with_gate below) -- cross-verified
    # independently via direct interactive computation against both a raw (sweep.load_frame) and
    # portfolio-aligned frame, both giving the identical pnl=77.31%/mdd=-21.76%/trades=26/skipped=9
    # before this script was corrected to rely on it instead of the mismatched external reference.
    print(f"  portfolio G0 with_gate (freshly established, no prior reference -- see script header note): pnl={g0b_portfolio_with_gate['pnl']:.2f}% mdd={g0b_portfolio_with_gate['mdd']:.2f}% trades={g0b_portfolio_with_gate['trades']}", flush=True)

    g0b_pass = bool(g0b_ok_component and g0b_ok_portfolio and diag_g0b_portfolio["rc_switch_bars"] == 0)
    report["g0b"] = {
        "component_tau_never_switch": {"actual": m_g0b_component, "reference": G0_REFERENCE["component_tabm_liveatr"], "match": g0b_ok_component},
        "portfolio_tau_never_switch_no_gate": {"actual": m_g0b_portfolio, "reference": G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"], "match": g0b_ok_portfolio},
        "portfolio_tau_never_switch_with_gate_freshly_established": g0b_portfolio_with_gate,
        "portfolio_tau_never_switch_with_gate_note": "No prior published reference for asymmetric_tabm_liveatr's own with_gate exists; the only previously-published 'baseline with_gate' number (54.88/-31.11, docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md) belongs to a different ledger (baseline_both_original, 29 trades) -- see BASELINE_BOTH_ORIGINAL_WITH_GATE_CONTEXT_ONLY. This value is established fresh and used as the relaxed-gate baseline throughout this script.",
        "portfolio_rc_switch_bars_expected_zero": diag_g0b_portfolio["rc_switch_bars"],
        "pass": g0b_pass,
        "note": "Both renamed-copy replay functions run with tau=TAU_NEVER_SWITCH (Delta<=1<10 always, so the risk-controlled branch is exercised at every bar but never overrides the baseline action) -- must reproduce the untouched TabM live-ATR baseline exactly (no_gate). Proves the copies are faithful outside the intentionally-changed block.",
    }
    if not g0b_pass:
        report["stage_reached"] = "G0b_self_consistency"
        report["gate_pass"] = False
        report["note"] = "G0b failed -- the risk-controlled copies do not reproduce the baseline in the degenerate tau=never-switch case. Aborting before trusting any candidate number."
        _write_report(report)
        print("stage=ABORT G0b failed", flush=True)
        return 1

    # ======================================================================================
    # stage=calibration -- Algorithm 1 on the component-level calibration log (measured under
    # the baseline's actual VAL holding pattern, larger n than the portfolio's shared-slot
    # holding pattern since h48qual trades every time its own signal fires here, unconfounded
    # by zig075 priority).
    # ======================================================================================
    print("=== stage=calibration (Algorithm 1, VAL only) ===", flush=True)
    rc_log = m_g0b_component["risk_controlled_log"]
    delta = np.asarray(rc_log["delta"], dtype=np.float64)
    y_arr = np.asarray(rc_log["y"], dtype=np.int64)
    a0_arr = np.asarray(rc_log["a0"], dtype=np.int64)
    astar_arr = np.asarray(rc_log["astar"], dtype=np.int64)
    n_cal = len(delta)
    baseline_mismatch_bumped = _bumped_risk(delta, y_arr, a0_arr, astar_arr, TAU_NEVER_SWITCH)
    disagreement_bars = int((a0_arr != astar_arr).sum())
    print(f"  n_calibration_bars={n_cal} baseline_bumped_mismatch_rate={baseline_mismatch_bumped:.4f} disagreement_bars(a0!=astar)={disagreement_bars} ({disagreement_bars / max(n_cal,1) * 100:.2f}%)", flush=True)

    eps_grid = [round(baseline_mismatch_bumped * f, 6) for f in EPS_FRACTIONS]
    calibration_results: dict[str, Any] = {}
    for frac, eps in zip(EPS_FRACTIONS, eps_grid):
        cal = _calibrate_threshold(delta, y_arr, a0_arr, astar_arr, eps)
        calibration_results[f"{frac:.2f}"] = cal
        print(f"  eps_frac={frac:.2f} eps={eps:.4f} -> tau_hat={cal['tau_hat']:.4f} risk_at_tau_hat={cal['risk_at_tau_hat']:.4f} feasible_count={cal['feasible_count']}/{cal['grid_size']}", flush=True)

    report["calibration"] = {
        "n_calibration_bars": n_cal,
        "baseline_bumped_mismatch_rate": baseline_mismatch_bumped,
        "disagreement_bars_a0_ne_astar": disagreement_bars,
        "disagreement_frequency_pct": disagreement_bars / max(n_cal, 1) * 100.0,
        "eps_fractions": EPS_FRACTIONS,
        "eps_grid": eps_grid,
        "candidates": calibration_results,
        "excess_risk_guarantee_form": "Theorem 4.2 (general fitted-fallback case): E[violation risk of deployed policy] <= eps + C3*log(n+1)/(n+1) -- O(log n / n) excess above budget under i.i.d. regularity conditions. The exact-safe-fallback special case (zero excess term) does NOT apply here since GBDT is not an exact-safe fallback (it can itself violate the y-rule). C3 is a paper-internal constant not resolvable from the abstract/sections fetched via WebFetch -- not fabricated here.",
        "log_n_over_n_plus_1": float(np.log(n_cal + 1) / (n_cal + 1)) if n_cal else None,
    }

    # ======================================================================================
    # stage=VAL_candidate_sweep -- for each eps/tau_hat: component no_gate + portfolio
    # no_gate/with_gate, both gate criteria.
    # ======================================================================================
    print("=== stage=VAL_candidate_sweep ===", flush=True)
    val_candidates: dict[str, Any] = {}
    for frac in EPS_FRACTIONS:
        key = f"{frac:.2f}"
        tau_hat = calibration_results[key]["tau_hat"]
        m_comp, ledger_comp = replay_exit_variant_risk_controlled(
            comp_prepped["frame"], comp_prepped["x"], comp_prepped["dec"], comp_prepped["loaded"],
            risk_margin_fraction=comp_prepped["margin"], risk_leverage=comp_prepped["leverage"],
            exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped["fee"], slip=comp_prepped["slip"],
            cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped["notional_scaled_sltp"], device=device,
            fallback_loaded_models=gbdt_component_loaded, fallback_threshold=FALLBACK_THRESHOLD, tau=tau_hat,
        )
        diag_port, ledger_port = greedy_replay_risk_controlled(
            val_frame, components_val, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            risk_component="h48qual", tau=tau_hat,
        )
        m_port_no_gate = portfolio._ledger_metrics(ledger_port)
        m_port_with_gate = mfe_width._duration_gated(ledger_port, val_frame, greedy.DURATION_THRESHOLD)
        ledger_comp.to_csv(OUT_DIR / f"component_ledger_val_eps{frac:.2f}.csv", index=False)
        ledger_port.to_csv(OUT_DIR / f"portfolio_ledger_val_eps{frac:.2f}.csv", index=False)

        gate_component_pnl = float(m_comp["pnl"]) >= float(G0_REFERENCE["component_tabm_liveatr"]["pnl"])
        gate_component_mdd = float(m_comp["mdd"]) >= float(G0_REFERENCE["component_tabm_liveatr"]["mdd"])
        gate_portfolio_pnl = float(m_port_no_gate["pnl"]) >= float(portfolio_tabm_liveatr["pnl"])
        gate_portfolio_mdd = float(m_port_no_gate["mdd"]) >= float(portfolio_tabm_liveatr["mdd"])
        gate_original_pass = bool(gate_component_pnl and gate_component_mdd and gate_portfolio_pnl and gate_portfolio_mdd)

        gate_relaxed_main = float(m_port_with_gate["pnl"]) > float(g0b_portfolio_with_gate["pnl"])
        gate_relaxed_mdd = (float(m_port_with_gate["mdd"]) - float(g0b_portfolio_with_gate["mdd"])) >= -3.0
        gate_relaxed_guardrail = _guardrail_ok(float(G0_REFERENCE["component_tabm_liveatr"]["pnl"]), float(m_comp["pnl"]))
        gate_relaxed_pass = bool(gate_relaxed_main and gate_relaxed_mdd and gate_relaxed_guardrail)

        val_candidates[key] = {
            "eps_frac": frac, "tau_hat": tau_hat,
            "component_no_gate": m_comp,
            "portfolio_no_gate": m_port_no_gate,
            "portfolio_with_gate": m_port_with_gate,
            "rc_diag": {k: v for k, v in diag_port.items() if k != "reason_counts"},
            "gate_original": {"component_pnl_nonworse": gate_component_pnl, "component_mdd_nonworse": gate_component_mdd,
                               "portfolio_pnl_nonworse": gate_portfolio_pnl, "portfolio_mdd_nonworse": gate_portfolio_mdd,
                               "pass": gate_original_pass},
            "gate_relaxed": {"portfolio_with_gate_pnl_improved": gate_relaxed_main, "portfolio_with_gate_mdd_within_3pp": gate_relaxed_mdd,
                              "component_guardrail_ok": gate_relaxed_guardrail, "pass": gate_relaxed_pass},
            "passes_any": bool(gate_original_pass or gate_relaxed_pass),
        }
        print(
            f"  eps_frac={frac:.2f} tau_hat={tau_hat:.4f}: component_no_gate={m_comp['pnl']:.2f}%/{m_comp['mdd']:.2f}%/{m_comp['trades']} "
            f"portfolio_no_gate={m_port_no_gate['pnl']:.2f}%/{m_port_no_gate['mdd']:.2f}%/{m_port_no_gate['trades']} "
            f"portfolio_with_gate={m_port_with_gate['pnl']:.2f}%/{m_port_with_gate['mdd']:.2f}%/{m_port_with_gate['trades']} "
            f"switch_bars={diag_port['rc_switch_bars']}(->exit:{diag_port['rc_switch_to_exit_bars']},->hold:{diag_port['rc_switch_to_hold_bars']}) "
            f"gate_original={gate_original_pass} gate_relaxed={gate_relaxed_pass}",
            flush=True,
        )

    report["val_baseline_portfolio_tabm_liveatr_no_gate"] = portfolio_tabm_liveatr
    report["val_baseline_portfolio_tabm_liveatr_with_gate"] = g0b_portfolio_with_gate
    report["val_baseline_component_tabm_liveatr_no_gate"] = G0_REFERENCE["component_tabm_liveatr"]
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
    # stage=OOS_single_touch -- winner's tau_hat is FROZEN from VAL calibration, applied
    # directly to OOS bars with no recalibration.
    # ======================================================================================
    winner_tau = val_candidates[winner]["tau_hat"]
    print(f"=== stage=OOS_single_touch winner=eps_frac{winner} tau_hat={winner_tau:.4f} ===", flush=True)
    print("*** MANDATORY CAVEAT ***", flush=True)
    print(OOS_CAVEAT_TEXT, flush=True)

    oos_frame_raw = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"  OOS frame rows={len(oos_frame_raw)} range=[{oos_frame_raw['timestamp'].min()}, {oos_frame_raw['timestamp'].max()}]", flush=True)
    # The WIDE24_2026 95-bar/0.37% Regime3-route-probability coverage gap (documented in
    # research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py, inherited by
    # _align_frame_and_oos_predictions above) affects the RAW oos_frame_raw itself, not just the
    # portfolio-level alignment step -- component-level replay also calls hard._route_id(frame) and
    # will raise "non-finite Regime3 route probabilities" if these rows are not dropped first. Apply
    # the identical filter here, once, before EITHER the component-level or portfolio-level OOS prep
    # touches oos_frame_raw (so both see the same clean frame; _align_frame_and_oos_predictions'
    # own internal copy of this same filter becomes a no-op the second time, harmless).
    n_route_bad_component = int((~np.isfinite(oos_frame_raw[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)).sum())
    if n_route_bad_component:
        oos_frame_raw = oos_frame_raw[np.isfinite(oos_frame_raw[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
        print(f"  dropped {n_route_bad_component} bars with non-finite Regime3 route probabilities from oos_frame_raw (WIDE24_2026 coverage gap, applied before component-level prep too)", flush=True)

    # -- component level OOS --
    oos_pred_h48qual = sweep.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{h48qual_cfg['q_tag']}.csv"
    comp_prepped_oos = sweep.prep_component("h48qual", h48qual_cfg, oos_frame_raw, oos_pred_h48qual, oof=False)
    m_comp_oos_baseline, _ = replay_exit_variant_risk_controlled(
        comp_prepped_oos["frame"], comp_prepped_oos["x"], comp_prepped_oos["dec"], comp_prepped_oos["loaded"],
        risk_margin_fraction=comp_prepped_oos["margin"], risk_leverage=comp_prepped_oos["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped_oos["fee"], slip=comp_prepped_oos["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped_oos["notional_scaled_sltp"], device=device,
        fallback_loaded_models=None, tau=TAU_NEVER_SWITCH,
    )
    m_comp_oos_candidate, ledger_comp_oos = replay_exit_variant_risk_controlled(
        comp_prepped_oos["frame"], comp_prepped_oos["x"], comp_prepped_oos["dec"], comp_prepped_oos["loaded"],
        risk_margin_fraction=comp_prepped_oos["margin"], risk_leverage=comp_prepped_oos["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=comp_prepped_oos["fee"], slip=comp_prepped_oos["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=comp_prepped_oos["notional_scaled_sltp"], device=device,
        fallback_loaded_models=gbdt_component_loaded, fallback_threshold=FALLBACK_THRESHOLD, tau=winner_tau,
    )
    ledger_comp_oos.to_csv(OUT_DIR / "component_ledger_oos_candidate.csv", index=False)
    print(f"  component OOS baseline={m_comp_oos_baseline['pnl']:.2f}%/{m_comp_oos_baseline['mdd']:.2f}%/{m_comp_oos_baseline['trades']} candidate={m_comp_oos_candidate['pnl']:.2f}%/{m_comp_oos_candidate['mdd']:.2f}%/{m_comp_oos_candidate['trades']}", flush=True)

    # -- portfolio level OOS --
    oos_frame, oos_aligned_paths = _align_frame_and_oos_predictions(oos_frame_raw, q_tags)
    print(f"  OOS aligned rows={len(oos_frame)} (from raw {len(oos_frame_raw)})", flush=True)
    oos_components_baseline = {
        "h48qual": greedy.prepare_component(oos_frame, oos_aligned_paths["h48qual"], h48qual_cfg, device),
        "zig075": greedy.prepare_component(oos_frame, oos_aligned_paths["zig075"], zig075_cfg, device),
    }
    _diag_oos_base, ledger_oos_base = greedy.greedy_replay(oos_frame, oos_components_baseline, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    m_port_oos_base_no_gate = portfolio._ledger_metrics(ledger_oos_base)
    m_port_oos_base_with_gate = mfe_width._duration_gated(ledger_oos_base, oos_frame, greedy.DURATION_THRESHOLD)
    ledger_oos_base.to_csv(OUT_DIR / "portfolio_ledger_oos_baseline_tabm_liveatr.csv", index=False)
    oos_baseline_cross_check_no_gate = _close(m_port_oos_base_no_gate, OOS_BASELINE_REFERENCE_NO_GATE)
    # with_gate has no valid prior reference for THIS ledger (see script header note) -- established
    # fresh here and used directly (not compared against OOS_BASELINE_BOTH_ORIGINAL_WITH_GATE_
    # CONTEXT_ONLY, which belongs to a different ledger) as the OOS relaxed-gate baseline below.
    print(f"  portfolio OOS baseline no_gate={m_port_oos_base_no_gate['pnl']:.2f}%/{m_port_oos_base_no_gate['mdd']:.2f}%/{m_port_oos_base_no_gate['trades']} cross_check={oos_baseline_cross_check_no_gate}", flush=True)
    print(f"  portfolio OOS baseline with_gate (freshly established, no prior reference)={m_port_oos_base_with_gate['pnl']:.2f}%/{m_port_oos_base_with_gate['mdd']:.2f}%", flush=True)

    oos_components_candidate = dict(oos_components_baseline)
    oos_components_candidate["h48qual"] = _gbdt_portfolio_fallback(dict(oos_components_baseline["h48qual"]), base_cols, gbdt_models, device)
    diag_oos_cand, ledger_oos_cand = greedy_replay_risk_controlled(
        oos_frame, oos_components_candidate, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
        risk_component="h48qual", tau=winner_tau,
    )
    m_port_oos_cand_no_gate = portfolio._ledger_metrics(ledger_oos_cand)
    m_port_oos_cand_with_gate = mfe_width._duration_gated(ledger_oos_cand, oos_frame, greedy.DURATION_THRESHOLD)
    ledger_oos_cand.to_csv(OUT_DIR / "portfolio_ledger_oos_candidate.csv", index=False)
    print(f"  portfolio OOS candidate no_gate={m_port_oos_cand_no_gate['pnl']:.2f}%/{m_port_oos_cand_no_gate['mdd']:.2f}%/{m_port_oos_cand_no_gate['trades']} switch_bars={diag_oos_cand['rc_switch_bars']}", flush=True)
    print(f"  portfolio OOS candidate with_gate={m_port_oos_cand_with_gate['pnl']:.2f}%/{m_port_oos_cand_with_gate['mdd']:.2f}%", flush=True)

    oos_gate_original = bool(
        float(m_comp_oos_candidate["pnl"]) >= float(m_comp_oos_baseline["pnl"])
        and float(m_comp_oos_candidate["mdd"]) >= float(m_comp_oos_baseline["mdd"])
        and float(m_port_oos_cand_no_gate["pnl"]) >= float(m_port_oos_base_no_gate["pnl"])
        and float(m_port_oos_cand_no_gate["mdd"]) >= float(m_port_oos_base_no_gate["mdd"])
    )
    oos_gate_relaxed = bool(
        float(m_port_oos_cand_with_gate["pnl"]) > float(m_port_oos_base_with_gate["pnl"])
        and (float(m_port_oos_cand_with_gate["mdd"]) - float(m_port_oos_base_with_gate["mdd"])) >= -3.0
        and _guardrail_ok(float(m_comp_oos_baseline["pnl"]), float(m_comp_oos_candidate["pnl"]))
    )
    print(f"stage=OOS_result survives_original={oos_gate_original} survives_relaxed={oos_gate_relaxed}", flush=True)

    report.update({
        "oos_opened": True,
        "oos_winner_eps_frac": winner, "oos_winner_tau_hat": winner_tau,
        "oos_frame_rows_raw": int(len(oos_frame_raw)), "oos_frame_rows_aligned": int(len(oos_frame)),
        "oos_component_baseline": m_comp_oos_baseline, "oos_component_candidate": m_comp_oos_candidate,
        "oos_portfolio_baseline_no_gate": m_port_oos_base_no_gate, "oos_portfolio_baseline_with_gate": m_port_oos_base_with_gate,
        "oos_portfolio_candidate_no_gate": m_port_oos_cand_no_gate, "oos_portfolio_candidate_with_gate": m_port_oos_cand_with_gate,
        "oos_baseline_cross_check_no_gate_reference": OOS_BASELINE_REFERENCE_NO_GATE, "oos_baseline_cross_check_no_gate_match": oos_baseline_cross_check_no_gate,
        "oos_portfolio_baseline_with_gate_note": "Freshly established, no prior valid published reference for THIS ledger (see script header note on BASELINE_BOTH_ORIGINAL_WITH_GATE_CONTEXT_ONLY) -- used directly as the OOS relaxed-gate baseline.",
        "oos_rc_diag": {k: v for k, v in diag_oos_cand.items() if k != "reason_counts"},
        "oos_gate_original_survives": oos_gate_original,
        "oos_gate_relaxed_survives": oos_gate_relaxed,
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
