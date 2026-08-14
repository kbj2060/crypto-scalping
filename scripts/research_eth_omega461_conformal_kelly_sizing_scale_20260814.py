#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #9 (literature scouting #6, rank-3 candidate, SIZING axis):
Conformal Kelly position-sizing scale (Ryan, arXiv:2608.01494 "Conformal Kelly: Conformal
Prediction Intervals as the Scale in Fractional Kelly Position Sizing", 2026-08-02), reinterpreted
for this project's classification + separate risk-sizing-GBM architecture.

=== Paper mechanism (verified by direct PDF fetch, not the literature-scouting summary) ===
The paper is a regression/ETF-momentum setting: for each asset, sigma_hat = q_eff / z, with
z=1.2816 (a disclosed leftover-alpha inconsistency) and q_eff a slow, per-asset, UNWEIGHTED
rolling+expanding-anchor conformal quantile of the ABSOLUTE forecast residual (nonconformity score
s = |R_H - mu_hat|), q_eff = q_roll^(1-lambda) * q_anchor^lambda, lambda=0.3, q_roll = 75th
pct of the last W=500 landed daily residuals, q_anchor = 75th pct of ALL landed residuals to date
(refreshed every 21 rows, held stale between refreshes -- "slow is the point"). Kelly fraction
f = kappa * mu_hat / sigma_hat^2, kappa=0.15, winsorised +-0.75 per asset. The paper's own
strongest, most load-bearing empirical finding (Sec 5.1/6.1/6.5) is that EVERY device that makes
the interval adapt FASTER loses (-0.7 to -5.3 pp/yr), including replacing the per-asset quantile
with model-ensemble/epistemic-disagreement dispersion (Sec 6.5: "epistemic uncertainty is ~1.5%
of predictive uncertainty ... not adopted"), and pooling calibration across assets also loses
(Sec 6.1). This directly confirms the literature-scouting doc's summary ("simplest per-asset
rolling quantile beats model-ensemble dispersion") -- verified against the source, not assumed.

CORRECTION to the literature-scouting doc (found by direct fetch, Step 1(c) of this task):
that doc characterised the paper's true-holdout result as "40 configurations, many underperforming
a pre-registered holdout". That is not what the paper reports. The "40" figures in the paper are
BOTH from the DEV window (2016-2021, the sealed AGENT-SEARCH window): (a) "40+ configurations"
of the miscoverage-based LEVERAGE-CUT idea (a different mechanism than the sizing scale used
here) that were tested and rejected purely on DEV, and (b) a 40-way circular-shift PLACEBO
significance test of that same leverage-cut dial, also purely on DEV. The actual pre-registered
true-OOS "lockbox" (2022-01-01..2024-09-20, 683 days, sealed before any DEV work began) tested
only TWO pre-registered configurations (Config A/B, with two predict_tail variants each -- 4
number-combinations total, not 40). BOTH underperformed: DEV net log growth 28.13%/25.52% fell to
lockbox 8.47%/7.01% (predict_tail=False primary variant) -- "roughly 30% of its development
value" -- while realized conformal COVERAGE transferred almost exactly (0.745 vs 0.750 nominal).
Both configs ranked LAST of 11 entries on lockbox Sharpe and Calmar. The paper's own conclusion
(Sec 12): "marginal calibration transferred out of sample and the economic value of the sizing
rule did not." This is a STRONGER, more precise warning than the literature-scouting summary
implied (not "many of 40 configs failed" but "the calibration mechanism itself is honest and
transfers; the Kelly-growth payoff evaporates on true OOS even with zero multiple-comparison
opportunity on the lockbox side, in exactly the VAL-wins-OOS-reverses shape this sub-project has
hit repeatedly"). Full detail + PDF text extraction: docs/experiments/
eth_omega461_conformal_kelly_sizing_scale_20260814.md.

=== This project's reinterpretation (Step 2) ===
No forecaster produces a point return prediction mu_hat here; direction/quality/exit_head are
frozen classifiers and margin_fraction is already the output of a separate risk-sizing GBM
(risk_sidecar.pkl) via train_eval_omega4_2_risk_sidecar_20260622._risk_margins (score -> sigmoid
scale -> margin_fraction). Verified empirically (both live h48qual/zig075 sidecar pkls) that
risk_target_mode="net" and target_mae_penalty=0.0 for BOTH components -- i.e. `score` IS trained,
with no unit transform, to predict `net_per_notional` (realized trade PnL per unit notional,
already logged verbatim on every replay_exit_variant ledger row). This gives an exact,
unit-consistent analogue of the paper's residual with zero re-derivation needed:
    nonconformity score for closed trade k = | net_per_notional_k(realized) - score_k(predicted at entry) |
This is the design chosen (Step 2's "가장 자연스러워 보이는 안"), adopted as-is: it reuses the
EXACT quantity the sizing GBM was already trained against, requires no synthetic mu_hat, and
needs no unit rescaling. Rejected alternative: recomputing a full mu_hat/sigma_hat^2 Kelly formula
from scratch (would redefine sizing wholesale, conflicting with "existing sizing model output
treated as the base -- only a multiplicative scale is layered on top", and would double-count the
GBM's own already-fitted edge estimate against a second, from-scratch Kelly numerator).

margin_fraction's effect on TP/SL is fully severed because notional_scaled_sltp=False for BOTH
live sidecars (verified from the pkls) -- TP/SL are pure price-move thresholds, independent of
margin/notional. entry timing is therefore always scale-invariant (entries are `dec['side']!=0`,
margin-independent). CORRECTION (found by direct empirical comparison after the first run, not
assumed away): exit timing is NOT fully scale-invariant -- research_eth_omega461_exit_sweep_
20260721.replay_exit_variant's exit_head call passes `notional`/`leverage`/`notional*leverage` as
POSITION-STATE INPUT FEATURES to the exit_head model itself (train_eval_omega4_2_risk_sidecar_
20260622._predict_exit_prob_one's `pos_values`, independent of the notional_scaled_sltp flag,
which only gates TP/SL). Scaling margin therefore CAN shift an exit_head-triggered exit by a few
bars, which can cascade (via the shared-slot portfolio mechanism, the same "slot recirculation"
dynamic this sub-project has repeatedly documented in #7/#8) into later trades. Measured directly
(candidate vs baseline component ledgers, entry_signal_i/exit_i, all 6 window/component pairs):
zig075 is IDENTICAL to baseline in all 3 scored windows (val/oos_q1/oos_q2); h48qual is identical
in oos_q2, has 2 exits shifted by <=4 bars in oos_q1 (same 28 entries, same reasons), and has a
real mid-window reshuffle in val starting at the 30th trade (63 entries in both, but several
exit_i/entry_i pairs differ from there on). This means the causal residual-history pool (built
from a fresh BASELINE-scale=1.0 shadow pass, not from the actual scaled run's own realized trades)
is a disclosed, second-order APPROXIMATION rather than an exact fixed point: still fully causal/
leakage-free (baseline exit timestamps are themselves realized bar-by-bar, strictly before the
later bars they inform), but not bit-identical to what the scaled run's OWN history would show.
An exact fixed point (iteratively re-simulate the scaled run, rebuild the pool from ITS OWN
realized trades, repeat to convergence) was not implemented -- it would need to interleave scale
computation inside replay_exit_variant's own bar loop, which the task instructions prohibit
modifying; given the divergence is empirically small (0 of 2 components affected in 4 of 6
scored windows, and the VAL verdict below is a clean, unanimous 3-of-3-grid-candidate rejection
that this second-order effect is very unlikely to overturn), this approximation is disclosed
rather than eliminated. See report["ledger_divergence_from_baseline"] for the exact per-window
counts.

Per-asset (here: per-COMPONENT, h48qual and zig075 kept STRICTLY separate -- no pooling), matching
the paper's own strongest anti-pooling finding (Sec 6.1). q_eff is EXPANDING-ANCHOR ONLY (no
separate fast W-trade rolling leg): the paper's W=500 *daily* landed-score window has no
non-degenerate analogue here (this project's OOF/OOS windows carry a few dozen trades total, not
thousands), and the paper's own Sec 5.1 finding is that slower is (weakly) always better, up to
and including a FULLY FROZEN sigma losing only 1.5pp to the best rolling variant on DEV -- so
"anchor-only, no rolling leg" is the direct, paper-endorsed limit of "make W as large as possible"
under this project's trade-count constraint, not an ad hoc simplification.
    q_eff(t) = 75th-pct{ residual_k : exit_timestamp_k < t }, pooled causally over
               [pre-VAL calibration set] union [this run's own realized trades closed before t]
    q_ref    = q_eff frozen at VAL_START (2025-10-01) -- the calibration-only anchor, never
               re-anchored later (a further "slow is the point" simplification)
    kelly_scale(t) = clip( (q_ref / max(q_eff(t), eps))^2 , scale_floor, scale_cap )
        exponent=2 matches the paper's f ~ mu_hat/sigma_hat^2 with sigma_hat ~ q_eff linearly
        (so f ~ 1/q_eff^2); NOT itself a grid axis (adopted verbatim from the paper).
    margin_fraction_scaled[i] = margin_fraction_raw[i] * kelly_scale(timestamp_i)
This is CLAUDE.md Futures Risk Sizing Contract compliant BY CONSTRUCTION (Step 3): the scale
multiplies margin_fraction directly (train_eval_omega4_2_risk_sidecar_20260622._risk_margins(...)
output), leverage is never touched, and TP/SL are derived from notional AFTER this point
(notional_scaled_sltp=False here means TP/SL don't even re-read notional at all, so there is no
leverage-double-counting path to begin with -- verified, not assumed).

(scale_floor, scale_cap) is the ONLY grid axis, calibrated on VAL only, 3 candidates (small grid
per task instruction): narrow (0.85,1.20), medium (0.70,1.40), wide (0.50,2.00) -- symmetric in
log-space around 1.0.

Pre-VAL calibration set (Step 4): built from THIS SCRIPT'S OWN fresh bar-by-bar replay of
2025q1+2025q2+2025q3 (the multiwindow gate's own "context" windows, oof=True/train_predictions,
scale FIXED at 1.0 -- candidate==baseline there, since a window cannot calibrate itself: avoids
bootstrap circularity, and mirrors how a live deployment would also start at scale=1.0 with an
empty trade history). VAL onward (2025-10-01..2026-06-30, i.e. val -> oos_q1 -> oos_q2, which are
calendar-CONTIGUOUS with each other and with 2025q3) is walked as ONE continuous causal pass, the
running residual pool carried forward window to window exactly as a live bot's own memory would be
-- never reset, never fed from a stale saved ledger file on disk.

Reused UNMODIFIED (imported only): eth_omega461_multiwindow_confirmation_gate_20260814
(load_all_windows, align_frame_and_predictions, run_portfolio_variant, summarize_multiwindow,
COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR, _close),
research_eth_omega461_exit_sweep_20260721 (replay_exit_variant, COMPONENTS, COST_MULT, load_frame),
research_eth_omega461_exit_head_portfolio_asymmetric_20260813 (_ledger_metrics),
research_eth_omega461_live_sltp_mfe_width_20260813 (_duration_gated, _as_router_component),
replay_omega4_6_1_greedy_router_20260706 (greedy_replay, DURATION_THRESHOLD).
NEW code in this file: _prep_component_with_score (renamed copy of research_eth_omega461_exit_
sweep_20260721.prep_component with ONE added returned field, `score` -- the only reason a copy is
needed is that field), the walk-forward driver, the kelly-scale math, and a small portfolio-replay
wrapper for pre-built (scaled) components. direction_head/quality_head/exit_head decision logic is
NEVER touched -- h48qual/zig075 labels, quality_threshold, and exit_head weights are all frozen
identically to the already-certified asymmetric_tabm_liveatr baseline; only margin_fraction is
conditionally rescaled post-hoc.

=== Compliance ===
fresh_forward_bar_by_bar=true: every ledger (baseline AND candidate, every window) comes from an
unmodified single forward call to replay_exit_variant (i increasing, only row i and already-closed
history used at bar i). trade_ledgers_used_as_input=false in the PROHIBITED sense (no pre-existing
ledger CSV is ever loaded from disk as an input) -- but the whole POINT of this experiment (task
instruction, Step 4) is that the Kelly scale causally consumes THIS RUN's OWN freshly-generated
realized-trade history bar-by-bar/window-by-window as it walks forward, exactly as a live bot's
own memory would; this is sanctioned by the task and is not the "old saved ledger as promotion
evidence" pattern the flag exists to catch -- see kelly_calibration_uses_this_runs_own_fresh_
trade_history=true in the report for an explicit, separate marker of this distinction.
saved_parent_exit_timestamps_used=false (no exit timestamp is read from any file this script did
not itself just write in this same process). future_rows_used_for_entry=false (residual pool at
bar i is filtered to exit_timestamp < timestamp[i], strictly). Does NOT touch trading_bot.py,
trading_bot_modules/omega4_6_1_live.py, runtime_config.py, .env. Does NOT modify any imported
module. No retraining (risk_sidecar GBMs are loaded frozen; only their margin_fraction OUTPUT is
post-hoc scaled). No GPU.

quality_threshold caveat (inherited, same as Odyssey2 #8): quality_threshold (h48qual=0.50,
zig075=0.75), shared identically by baseline and candidate here, was itself OOS-pnl-primary
selected against 2026-01-01..02-28 (see docs/experiments/
eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The relative comparison
(candidate vs baseline within this run) remains meaningful since both share the identical
contaminated entry-selection layer; absolute OOS PnL/MDD are not clean unbiased forward
performance and must not be over-interpreted as such.
"""
from __future__ import annotations

import json
import pickle
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

import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_conformal_kelly_sizing_scale_20260814"
DEVICE = portfolio.DEVICE
COMPONENT_NAMES = ("h48qual", "zig075")
COMP_CFGS = gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR  # reused unmodified -- the certified baseline

ALPHA = 0.25  # 75% conformal interval, matches paper Sec 3 exactly
EXPONENT = 2.0  # f ~ mu/sigma^2, sigma ~ q_eff linearly => scale ~ 1/q_eff^2 (adopted from paper, not gridded)
EPS = 1.0e-6
MIN_POOL_FOR_SCALE = 3  # below this, force scale=1.0 (quantile of <3 points is not trustworthy)
CALIBRATION_WINDOWS = ("2025q1", "2025q2", "2025q3")
SCORED_WINDOWS_IN_ORDER = ("val", "oos_q1", "oos_q2")  # calendar-contiguous continuation of 2025q3
SCALE_GRID: dict[str, tuple[float, float]] = {
    "narrow_0.85_1.20": (0.85, 1.20),
    "medium_0.70_1.40": (0.70, 1.40),
    "wide_0.50_2.00": (0.50, 2.00),
}
MDD_SLACK_RELAXED_PP = 3.0


def log(msg: str) -> None:
    print(f"[conformal_kelly] {msg}", flush=True)


# =========================================================================================
# Renamed copy of research_eth_omega461_exit_sweep_20260721.prep_component (byte-identical
# except ONE added field in the returned dict: `score`). Not reused via import because the
# original does not expose `score`, and this experiment needs it to build residuals -- the
# task instruction explicitly sanctions a renamed copy for this kind of minimal addition.
# =========================================================================================
def _prep_component_with_score(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path, *, oof: bool) -> dict[str, Any]:
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    src_raw = pd.read_csv(pred_csv)
    for c in src_raw.columns:
        if str(src_raw[c].dtype).lower().startswith("str"):
            src_raw[c] = src_raw[c].astype(object)
    src_raw["timestamp"] = pd.to_datetime(src_raw["timestamp"])
    keep_ts = set(src_raw["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src = src_raw[src_raw["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if len(src) != len(frame) or not src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"{name}: prediction/frame timestamp mismatch ({len(src)} vs {len(frame)})")

    x = base_sweep.parent._base_input(frame, base_cols)
    dec_base = base_sweep.parent._to_decisions(src, oof=oof)
    dec, _atr_diag = base_sweep.atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr_pct = base_sweep.atr_eval._atr_pct(frame, cfg["atr_window"])
    fee, slip = base_sweep.omega._load_fee_slip()
    loaded = base_sweep.parent._load_payloads(models, device=DEVICE)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)

    features = base_sweep.rs._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = base_sweep.rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = base_sweep.rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    if pkl["risk_target_mode"] != "net" or float(pkl["target_mae_penalty"]) != 0.0:
        raise RuntimeError(f"{name}: sidecar risk_target_mode/target_mae_penalty changed from the verified net/0.0 -- residual definition (score==net_per_notional prediction) no longer holds, abort")

    mapping = pkl["selected_mapping"]
    margin_kwargs = {k: mapping[k] for k in base_sweep.rs.MARGIN_CFG_KEYS}
    margin = base_sweep.rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **margin_kwargs)
    leverage = None
    if pkl["dynamic_leverage"]:
        lev_kwargs = {k: mapping[k] for k in base_sweep.rs.LEVERAGE_CFG_KEYS}
        leverage = base_sweep.rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **lev_kwargs)
    if bool(pkl["notional_scaled_sltp"]):
        raise RuntimeError(f"{name}: notional_scaled_sltp changed from the verified False -- margin scaling would now also move TP/SL, invalidating the single-baseline-pass trade-timing argument, abort")

    return dict(
        frame=frame, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
        fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"], score=score,
    )


def _component_ledger(p: dict[str, Any], *, margin_override: np.ndarray | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    margin = margin_override if margin_override is not None else p["margin"]
    m, ledger = base_sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=margin, risk_leverage=p["leverage"],
        exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )
    if len(ledger):
        ledger = ledger.copy()
        ledger["score_at_entry"] = p["score"][ledger["entry_signal_i"].to_numpy(dtype=np.int64)]
        ledger["residual"] = (ledger["net_per_notional"] - ledger["score_at_entry"]).abs()
        ledger["exit_timestamp_dt"] = pd.to_datetime(ledger["exit_timestamp"])
    return {k: v for k, v in m.items() if k != "exit_reasons"} | {"exit_reasons": m["exit_reasons"]}, ledger


def _q_eff_lookup(pool_res_sorted_by_time: np.ndarray) -> dict[int, float]:
    """residual quantile as a function of n (number of pool trades available), computed once per
    distinct n actually needed rather than once per bar."""
    cache: dict[int, float] = {}

    def get(n: int) -> float:
        if n not in cache:
            cache[n] = float(np.quantile(pool_res_sorted_by_time[:n], 1.0 - ALPHA)) if n >= MIN_POOL_FOR_SCALE else float("nan")
        return cache[n]
    return get  # type: ignore[return-value]


def _causal_scale_ratio(frame_ts: np.ndarray, pool_ts_sorted: np.ndarray, pool_res_sorted: np.ndarray, q_ref: float) -> np.ndarray:
    """Vectorised causal (q_ref/q_eff(t))^EXPONENT, unclipped, per bar. n_avail(t) = count of pool
    trades whose exit_timestamp is STRICTLY before t (side='left' on a time-sorted pool). Returns
    ratio=1.0 (neutral) wherever n_avail < MIN_POOL_FOR_SCALE."""
    n_avail = np.searchsorted(pool_ts_sorted, frame_ts, side="left")
    getter = _q_eff_lookup(pool_res_sorted)
    ratio = np.ones(len(frame_ts), dtype=np.float64)
    for n in np.unique(n_avail):
        n = int(n)
        if n < MIN_POOL_FOR_SCALE:
            continue
        q_eff = getter(n)
        r = (float(q_ref) / max(q_eff, EPS)) ** EXPONENT
        ratio[n_avail == n] = r
    return ratio


def _portfolio_replay(frame: pd.DataFrame, comp_ps: dict[str, dict[str, Any]], *, fee: float, slip: float) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Sibling of gate.run_portfolio_variant, for PRE-BUILT (possibly scale-modified) component
    dicts rather than building them from a comp_cfg -- gate.run_portfolio_variant cannot be reused
    unmodified here because it always builds margin from scratch via portfolio._component_cfg with
    no scaling hook. Same greedy_replay/_ledger_metrics/_duration_gated call shape."""
    router_components = {name: mfe_width._as_router_component(p, exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD) for name, p in comp_ps.items()}
    _diag, ledger = greedy.greedy_replay(frame, router_components, fee=fee, slip=slip, cost_mult=base_sweep.COST_MULT, device=DEVICE)
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate, ledger


def run_component_walk_forward(name: str, windows: dict[str, Any], aligned: dict[str, tuple[pd.DataFrame, dict[str, Path]]]) -> dict[str, Any]:
    """One continuous causal pass for ONE component across all 6 windows in calendar order.
    Returns baseline (scale=1.0, identical for every grid candidate) ledgers/metrics/p-dicts for
    every window, plus the unclipped causal ratio array per SCORED window (grid-independent), so
    the caller can cheaply produce each of the 3 SCALE_GRID candidates by clipping.
    """
    cfg = COMP_CFGS[name]
    per_window: dict[str, Any] = {}
    calib_pool_ts: list[pd.Timestamp] = []
    calib_pool_res: list[float] = []

    for wname in CALIBRATION_WINDOWS:
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        p = _prep_component_with_score(name, cfg, aligned_frame, aligned_paths[name], oof=w["oof"])
        m_base, ledger_base = _component_ledger(p)
        per_window[wname] = {"p_baseline": p, "metrics_baseline": m_base, "ledger_baseline": ledger_base, "aligned_frame": aligned_frame}
        if len(ledger_base):
            calib_pool_ts.extend(ledger_base["exit_timestamp_dt"].tolist())
            calib_pool_res.extend(ledger_base["residual"].tolist())
        log(f"{name} calib_window={wname} rows={len(aligned_frame)} trades={m_base['trades']} pnl={m_base['pnl']:.2f}% mdd={m_base['mdd']:.2f}%")

    order = np.argsort(calib_pool_ts) if calib_pool_ts else np.array([], dtype=np.int64)
    pool_ts_sorted = np.array(calib_pool_ts, dtype="datetime64[ns]")[order] if calib_pool_ts else np.array([], dtype="datetime64[ns]")
    pool_res_sorted = np.array(calib_pool_res, dtype=np.float64)[order] if calib_pool_res else np.array([], dtype=np.float64)
    n_calib = int(len(pool_res_sorted))
    q_ref = float(np.quantile(pool_res_sorted, 1.0 - ALPHA)) if n_calib >= MIN_POOL_FOR_SCALE else float("nan")
    log(f"{name} PRE-VAL calibration set: n_calib_trades={n_calib} q_ref(75th pct |residual|)={q_ref}")

    running_ts = list(pool_ts_sorted)
    running_res = list(pool_res_sorted)
    for wname in SCORED_WINDOWS_IN_ORDER:
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        p = _prep_component_with_score(name, cfg, aligned_frame, aligned_paths[name], oof=w["oof"])
        m_base, ledger_base = _component_ledger(p)

        # Causal pool for THIS window's own scale array must include not only prior windows'
        # closed trades but also THIS window's own baseline-ledger trades that close before a
        # given later bar in the SAME window (e.g. a bar in December should see June, September,
        # AND early-October trades). This uses the BASELINE (scale=1.0) ledger's exit timestamps
        # as a disclosed, second-order approximation of "what would already be known" (exit
        # timing is NOT perfectly scale-invariant here -- see module docstring correction); it is
        # still strictly causal -- no future information crosses into any bar's own entry
        # decision, since searchsorted below still enforces exit_timestamp < bar_timestamp
        # strictly, using timestamps that were themselves realized bar-by-bar in a single forward
        # pass.
        combined_ts = running_ts + (ledger_base["exit_timestamp_dt"].tolist() if len(ledger_base) else [])
        combined_res = running_res + (ledger_base["residual"].tolist() if len(ledger_base) else [])
        order2 = np.argsort(combined_ts) if combined_ts else np.array([], dtype=np.int64)
        pool_ts_sorted2 = np.array(combined_ts, dtype="datetime64[ns]")[order2] if combined_ts else np.array([], dtype="datetime64[ns]")
        pool_res_sorted2 = np.array(combined_res, dtype=np.float64)[order2] if combined_res else np.array([], dtype=np.float64)
        frame_ts = aligned_frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        raw_ratio = _causal_scale_ratio(frame_ts, pool_ts_sorted2, pool_res_sorted2, q_ref) if n_calib >= MIN_POOL_FOR_SCALE else np.ones(len(frame_ts), dtype=np.float64)

        per_window[wname] = {
            "p_baseline": p, "metrics_baseline": m_base, "ledger_baseline": ledger_base, "aligned_frame": aligned_frame,
            "raw_ratio": raw_ratio, "n_pool_at_window_start": int(len(running_ts)),
        }
        log(f"{name} scored_window={wname} rows={len(aligned_frame)} baseline_trades={m_base['trades']} baseline_pnl={m_base['pnl']:.2f}% "
            f"n_pool_at_start={len(running_ts)} raw_ratio_range=[{raw_ratio.min():.3f},{raw_ratio.max():.3f}]")

        if len(ledger_base):
            running_ts.extend(ledger_base["exit_timestamp_dt"].tolist())
            running_res.extend(ledger_base["residual"].tolist())

    return {"per_window": per_window, "n_calib": n_calib, "q_ref": q_ref}


def candidate_component_result(walk: dict[str, Any], wname: str, bounds: tuple[float, float]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Clip the (grid-independent) raw ratio to `bounds`, rescale margin, replay once. For
    CALIBRATION_WINDOWS (no raw_ratio stored -- scale fixed at 1.0 by design), returns baseline."""
    win = walk["per_window"][wname]
    p = win["p_baseline"]
    if "raw_ratio" not in win:
        return win["metrics_baseline"], win["ledger_baseline"], dict(p)
    scale_arr = np.clip(win["raw_ratio"], bounds[0], bounds[1])
    margin_scaled = p["margin"] * scale_arr
    m_cand, ledger_cand = _component_ledger(p, margin_override=margin_scaled)
    p_cand = dict(p)
    p_cand["margin"] = margin_scaled
    return m_cand, ledger_cand, p_cand


def _ledger_divergence(ledger_baseline: pd.DataFrame, ledger_candidate: pd.DataFrame) -> dict[str, Any]:
    """Quantifies how far a scaled candidate's own component-level trade timing diverges from the
    baseline (scale=1.0) shadow ledger used to build its causal calibration pool -- see module
    docstring correction on exit_head reading notional as an input feature. same_trade_count=false
    or n_exit_i_diff>0 means the single-baseline-pass approximation is not exact for this window;
    still fully causal (no future information), just not a bit-identical fixed point."""
    if len(ledger_baseline) != len(ledger_candidate):
        return {"same_trade_count": False, "n_baseline_trades": int(len(ledger_baseline)), "n_candidate_trades": int(len(ledger_candidate)), "n_entry_i_diff": None, "n_exit_i_diff": None}
    n = len(ledger_baseline)
    if n == 0:
        return {"same_trade_count": True, "n_baseline_trades": 0, "n_candidate_trades": 0, "n_entry_i_diff": 0, "n_exit_i_diff": 0}
    entry_diff = int((ledger_baseline["entry_signal_i"].to_numpy() != ledger_candidate["entry_signal_i"].to_numpy()).sum())
    exit_diff = int((ledger_baseline["exit_i"].to_numpy() != ledger_candidate["exit_i"].to_numpy()).sum())
    return {"same_trade_count": True, "n_baseline_trades": int(n), "n_candidate_trades": int(n), "n_entry_i_diff": entry_diff, "n_exit_i_diff": exit_diff}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = base_sweep.omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": __doc__,
        "paper_citation": "Ryan, R.J. (2026-08-02) 'Conformal Kelly: Conformal Prediction Intervals as the Scale in Fractional Kelly Position Sizing', arXiv:2608.01494",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "kelly_calibration_uses_this_runs_own_fresh_trade_history": True,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "alpha": ALPHA, "exponent": EXPONENT, "eps": EPS, "min_pool_for_scale": MIN_POOL_FOR_SCALE,
        "scale_grid": {k: list(v) for k, v in SCALE_GRID.items()},
        "oos_caveat_quality_threshold_contamination": (
            "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by baseline and every "
            "candidate here, was itself OOS-pnl-primary selected against 2026-01-01..02-28 (see "
            "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
            "relative comparison (candidate vs baseline) remains meaningful; absolute OOS PnL/MDD are "
            "not clean unbiased forward performance."
        ),
    }

    # =====================================================================================
    # stage=load_and_align -- reuse gate.load_all_windows / gate.align_frame_and_predictions
    # unmodified for all 6 windows.
    # =====================================================================================
    log("=== stage=load_and_align ===")
    windows = gate.load_all_windows()
    q_tags = {name: base_sweep.COMPONENTS[name]["q_tag"] for name in COMPONENT_NAMES}
    aligned: dict[str, tuple[pd.DataFrame, dict[str, Path]]] = {}
    for wname, wd in gate.WINDOW_DEFS.items():
        # NOTE: 2025q1/q2/q3 all share split="train", so align_frame_and_predictions's own
        # out_dir/f"_aligned_{split}_{cname}_predictions.csv" naming collides across those three
        # windows if called once-per-window-upfront (unlike the gate module's own usage pattern,
        # which always writes-then-immediately-reads a given window before moving to the next).
        # Fixed here (not in the shared, unmodified gate module) with a per-window subdirectory.
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(windows[wname]["frame"], q_tags, wd["split"], OUT_DIR / wname)
        aligned[wname] = (aligned_frame, aligned_paths)
        log(f"  aligned {wname}: rows={len(aligned_frame)}")

    # =====================================================================================
    # stage=G0 -- baseline (scale=1.0) must exactly reproduce the already-published
    # asymmetric_tabm_liveatr reference numbers, via TWO independent code paths: (a) the
    # unmodified gate.run_portfolio_variant, (b) this script's own walk-forward machinery
    # with scale forced to (1.0, 1.0) [a no-op clip].
    # =====================================================================================
    log("=== stage=G0 ===")
    g0a: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=DEVICE, out_dir=OUT_DIR, variant_label="g0a_gate_module")
        ref_ng, ref_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng, ok_wg = gate._close(result["no_gate"], ref_ng), gate._close(result["with_gate"], ref_wg)
        g0a[wname] = {"no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg}}
        log(f"  G0a(gate module) {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
            f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}")
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))

    log("Running this script's own walk-forward machinery (needed for G0b AND for the actual candidate results below)...")
    walks: dict[str, dict[str, Any]] = {name: run_component_walk_forward(name, windows, aligned) for name in COMPONENT_NAMES}

    g0b: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        comp_ps_identity = {name: candidate_component_result(walks[name], wname, (1.0, 1.0))[2] for name in COMPONENT_NAMES}
        no_gate, with_gate, _ledger = _portfolio_replay(aligned[wname][0], comp_ps_identity, fee=fee, slip=slip)
        ref_ng, ref_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng, ok_wg = gate._close(no_gate, ref_ng), gate._close(with_gate, ref_wg)
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg}}
        log(f"  G0b(own machinery, scale forced 1.0) {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] for w in ("val", "oos_q1"))

    g0_pass = bool(g0a_pass and g0b_pass)
    report["g0"] = {"gate_module_path": g0a, "own_machinery_path": g0b, "pass": bool(g0_pass)}
    log(f"stage=G0_result pass={g0_pass}")
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed to reproduce the already-published asymmetric_tabm_liveatr reference numbers on val/oos_q1 -- aborting per task instruction before VAL/OOS scoring."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
        return 1

    report["calibration"] = {name: {"n_calib_trades": walks[name]["n_calib"], "q_ref": walks[name]["q_ref"]} for name in COMPONENT_NAMES}

    # =====================================================================================
    # stage=VAL -- 3-candidate grid, both original (4-metric non-worse) and relaxed
    # (with_gate PnL improves, MDD within 3pp) criteria, vs asymmetric_tabm_liveatr baseline.
    # =====================================================================================
    log("=== stage=VAL_grid ===")
    base_no_gate_val, base_with_gate_val, _ = _portfolio_replay(
        aligned["val"][0], {name: candidate_component_result(walks[name], "val", (1.0, 1.0))[2] for name in COMPONENT_NAMES}, fee=fee, slip=slip)
    log(f"  baseline VAL: no_gate={base_no_gate_val} with_gate={base_with_gate_val}")

    val_rows: dict[str, Any] = {}
    for grid_name, bounds in SCALE_GRID.items():
        comp_ps = {}
        comp_metrics = {}
        comp_divergence = {}
        for name in COMPONENT_NAMES:
            m_cand, ledger_cand, p_cand = candidate_component_result(walks[name], "val", bounds)
            comp_ps[name] = p_cand
            comp_metrics[name] = m_cand
            comp_divergence[name] = _ledger_divergence(walks[name]["per_window"]["val"]["ledger_baseline"], ledger_cand)
        no_gate, with_gate, _ledger = _portfolio_replay(aligned["val"][0], comp_ps, fee=fee, slip=slip)
        pass_original = (no_gate["pnl"] >= base_no_gate_val["pnl"] and no_gate["mdd"] >= base_no_gate_val["mdd"] and
                          with_gate["pnl"] >= base_with_gate_val["pnl"] and with_gate["mdd"] >= base_with_gate_val["mdd"])
        pass_relaxed = (with_gate["pnl"] > base_with_gate_val["pnl"] and
                         (with_gate["mdd"] - base_with_gate_val["mdd"]) >= -abs(MDD_SLACK_RELAXED_PP))
        val_rows[grid_name] = {
            "bounds": list(bounds), "portfolio_no_gate": no_gate, "portfolio_with_gate": with_gate,
            "component_metrics": comp_metrics, "pass_original_gate": bool(pass_original), "pass_relaxed_gate": bool(pass_relaxed),
            "ledger_divergence_from_baseline": comp_divergence,
        }
        log(f"  VAL candidate={grid_name} bounds={bounds}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} pass_original={pass_original} pass_relaxed={pass_relaxed}")

    passing_original = [k for k, v in val_rows.items() if v["pass_original_gate"]]
    passing_relaxed = [k for k, v in val_rows.items() if v["pass_relaxed_gate"]]
    passing_any = sorted(set(passing_original) | set(passing_relaxed), key=lambda k: val_rows[k]["portfolio_with_gate"]["pnl"], reverse=True)
    val_winner = passing_any[0] if passing_any else None
    report["val"] = {
        "baseline_no_gate": base_no_gate_val, "baseline_with_gate": base_with_gate_val, "candidates": val_rows,
        "passing_original_gate": passing_original, "passing_relaxed_gate": passing_relaxed, "val_winner": val_winner,
    }
    log(f"stage=VAL_result passing_original={passing_original} passing_relaxed={passing_relaxed} val_winner={val_winner}")

    if val_winner is None:
        report["stage_reached"] = "VAL"
        report["oos_opened"] = False
        report["final_verdict"] = "REJECTED_VAL_GATE"
        report["gate_pass"] = True
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
        log("stage=done -- no VAL candidate passed either gate, OOS NOT opened (negative result).")
        return 0

    # =====================================================================================
    # stage=OOS_single_touch -- the ONE VAL-selected candidate, all 6 windows, via
    # gate.summarize_multiwindow (oos_q1+oos_q2 opened together, single touch).
    # =====================================================================================
    log(f"=== stage=OOS_single_touch candidate={val_winner} bounds={SCALE_GRID[val_winner]} ===")
    bounds = SCALE_GRID[val_winner]
    baseline_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    candidate_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    component_table: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        comp_ps_base = {name: candidate_component_result(walks[name], wname, (1.0, 1.0))[2] for name in COMPONENT_NAMES}
        no_gate_b, with_gate_b, _ = _portfolio_replay(aligned[wname][0], comp_ps_base, fee=fee, slip=slip)
        baseline_results[wname] = (no_gate_b, with_gate_b)

        comp_ps_cand = {}
        comp_metrics_cand = {}
        for name in COMPONENT_NAMES:
            m_cand, _ledger_cand, p_cand = candidate_component_result(walks[name], wname, bounds)
            comp_ps_cand[name] = p_cand
            comp_metrics_cand[name] = m_cand
        no_gate_c, with_gate_c, _ = _portfolio_replay(aligned[wname][0], comp_ps_cand, fee=fee, slip=slip)
        candidate_results[wname] = (no_gate_c, with_gate_c)
        component_table[wname] = comp_metrics_cand
        log(f"  {wname} [{gate.WINDOW_DEFS[wname]['tier']}]: baseline no_gate={no_gate_b['pnl']:.2f}%/{no_gate_b['mdd']:.2f}%/{no_gate_b['trades']} with_gate={with_gate_b['pnl']:.2f}%/{with_gate_b['mdd']:.2f}%/{with_gate_b['trades']}  |  "
            f"candidate no_gate={no_gate_c['pnl']:.2f}%/{no_gate_c['mdd']:.2f}%/{no_gate_c['trades']} with_gate={with_gate_c['pnl']:.2f}%/{with_gate_c['mdd']:.2f}%/{with_gate_c['trades']}")

    summary_strict = gate.summarize_multiwindow(baseline_results, candidate_results, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_results, candidate_results, mdd_slack_pp=MDD_SLACK_RELAXED_PP)
    log(f"OOS single-touch verdict: strict(mdd0pp)={summary_strict['final_verdict']} relaxed(mdd{MDD_SLACK_RELAXED_PP}pp)={summary_relaxed['final_verdict']}")

    report["oos_opened"] = True
    report["val_winner"] = val_winner
    report["val_winner_bounds"] = list(bounds)
    report["six_window_component_metrics"] = component_table
    report["six_window_summary_strict_mdd0pp"] = summary_strict
    report["six_window_summary_relaxed_mdd3pp"] = summary_relaxed
    report["final_verdict"] = summary_strict["final_verdict"] if summary_strict["oos_confirm_all_pass_single_touch"] else (
        summary_relaxed["final_verdict"] if summary_relaxed["oos_confirm_all_pass_single_touch"] else "REJECTED_SIGN_MISMATCH")
    report["paper_oos_replication_warning_applied"] = (
        "The source paper's own true-holdout (lockbox) result showed calibration coverage transferring "
        "almost exactly while the Kelly-growth economic payoff collapsed to ~30% of its development-window "
        "value and ranked last of 11 comparators on risk-adjusted return -- the calibration mechanism "
        "generalises, the sizing profitability does not. That is treated here as a directly-relevant prior "
        "for this candidate, not a generic caveat: this sub-project has independently hit the identical "
        "VAL-wins/OOS-reverses shape on 3 of its last 4 post-entry candidates (queue-pressure, "
        "risk-controlled, and both regime-threshold variants). A VAL pass here is evidence the mechanism is "
        "not obviously broken, not evidence it will hold on oos_q1+oos_q2."
    )
    report["stage_reached"] = "oos_single_touch"
    report["gate_pass"] = True
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
    log(f"stage=done final_verdict={report['final_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
