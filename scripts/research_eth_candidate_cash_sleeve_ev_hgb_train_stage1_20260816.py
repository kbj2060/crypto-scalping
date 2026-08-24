#!/usr/bin/env python3
"""RESEARCH ONLY -- Stage 1 REAL training for the "cash-sleeve EV-HGB" candidate
(docs/experiments/eth_candidate_cash_sleeve_ev_hgb_20260816.md), gated on the cheap_gate
(oracle headroom, decisively not-negative) and the IC-check (ambiguous middle zone, 12/50
feature/target combos survive VAL+OOS-Q1 sign-consistency + noise floor + price-contamination
screens). This is the first script in the sub-project that actually FITS a model.

=== Scope discipline (per orchestrating-session instruction) ===
VAL window ONLY (2025-10-01..12-31). OOS-Q1 (2026-01-01..03-31) is NOT loaded, NOT read, NOT
used anywhere in this script -- it is reserved untouched for a later fresh-forward walk-forward
test. Using it now (even just to "peek" at a metric) would be a form of leakage into model/
hyperparameter selection, which is exactly the failure mode the Fresh-Forward Rule exists to
prevent. Purged CV within VAL only; no OOS-Q1 gate check, no OOS-Q1 metric of any kind below.

=== What is reused vs. what is new ===
Reused, unmodified, imported (not reimplemented):
  - eth_omega461_multiwindow_confirmation_gate_20260814 (window/prediction loading,
    align_frame_and_predictions)
  - research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 (h48qual regime-aware
    exit guard component prep + detector)
  - research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814
    (zig075 SHORT entry veto mask attach + the exact locked Odyssey4-baseline greedy replay,
    greedy_replay_entry_veto -- gives the real account-level `held` ledger)
  - research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816 (build_primary_ledger,
    _held_mask, run_cash_sleeve_oracle -- the EXACT 3x-cost-stressed fallback-trade label
    simulation; called directly here, not copy-pasted, to avoid divergence -- its output is
    cross-checked byte-for-byte against the already-published oracle CSV below)
  - research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816 (CANDIDATE_FEATURES / ALL_FEATURES
    -- the exact same 25-feature market/momentum/volatility/regime-router list already IC-screened)
  - train_eval_omega1_2_tabm_diffusion_risk_20260603._source_state (the exact function that
    already turns an h48qual/zig075 raw ThreeHeadTabM prediction CSV into the "tabm_*" trace
    feature set -- this is also literally the source BTC's live cash-sleeve adapter
    (trading_bot_modules/omega1_2_3_cash_sleeve.py._trace_features) draws its own tabm_* features
    from, so reusing it here is the closest available ETH equivalent of BTC's "primary-trace"
    feature group, not an invented substitute)
  - core.event_label_engine.purged_kfold_splits (AFML Ch.7 purge+embargo utility -- reused as-is,
    see "Purged CV design" below for why it fits this event structure)

New in this script: (1) the "cash-state-history" feature block (primary_cash_streak,
primary_active_roll_12/48, time_since_primary_exit, last_primary_active_len, last_primary_side --
names/tanh-scaling mirror BTC's Omega123CashSleeveAdapter._history_features exactly for
consistency), derived causally from the SAME `held`/`ledger` objects the cheap_gate script
already builds; (2) the purged-CV + label-permutation-null training/eval loop; (3) the two
pre-specified metrics (OOF Spearman IC, decision-relevant ev_min selection quality) computed on
BOTH the real models and the permutation-null models, so the real result is judged against a null
distribution rather than reported as a bare number.

=== Purged CV design ===
Each CASH-bar "event" i has a fallback-trade label (long_net[i], short_net[i]) that depends on
price action from bar i+1 up to AT MOST bar i+MAX_HOLD_BARS (=192, cheap_gate.MAX_HOLD_BARS --
the label's own vertical/takeover barrier, exactly the same quantity Lopez de Prado's t1 refers
to). core/event_label_engine.purged_kfold_splits(event_idx, t1_idx, n_bars, n_splits, embargo_frac)
already implements exactly this: for each fold, purge any TRAIN event whose [event_idx, t1_idx]
interval overlaps [test_fold_start, test_fold_end + embargo]. It fits this event structure exactly
(one label horizon per event) so it is reused as-is rather than rewritten -- the only design choice
made here is what to pass as t1_idx: a conservative fixed upper bound
`min(event_idx + MAX_HOLD_BARS, n_bars_in_window - 1)` for EVERY event (not each event's own
actual exit_j, which can be earlier if TP/SL fired first) -- this guarantees the realized purge
gap is always >= 192 bars regardless of how early any individual trade actually exited, matching
the orchestrating session's literal instruction ("purge gap >= 192 bars ... any train bar within
192 bars of a validation fold's start/end boundary must be dropped"). embargo_frac=0.0 is used
because the +192 is already folded into t1_idx; per-fold realized gaps are computed and logged
below as a direct empirical check (not just trusted by construction).

=== Permutation-label null ===
For each of N_PERM=30 repeats: for each fold, shuffle (long_net_train, short_net_train) with the
SAME within-fold random permutation (preserves each row's own long/short pairing, matching how a
real market state's long/short outcomes co-vary), refit both HGB models on the shuffled labels
with the REAL (unshuffled) features, predict on the real held-out test fold, and score against the
REAL (unshuffled) test-fold labels. Out-of-fold predictions are pooled across all 5 folds per
repeat exactly the way the real model's OOF predictions are pooled, so each of the 30 null repeats
yields ONE pooled-OOF null value per metric, directly comparable to the real model's one pooled-OOF
value.

fresh_forward_bar_by_bar=true (primary ledger + `held` come from an unmodified single causal
bar-by-bar replay; features are each bar's own already-computed causal columns / already-trained
frozen TabM outputs). trade_ledgers_used_as_input=false (ledger informs ONLY the CASH/held
structural fact and, causally, only ALREADY-CLOSED trades feed last_primary_active_len/
last_primary_side -- see _cash_state_history_features). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false for features; labels (long_net/short_net) remain the cheap_gate
script's own oracle simulation, which by design and by its own documentation uses each bar's
REALIZED future path -- this is unchanged from the cheap_gate/IC-check framing and is exactly what
an EV regression model is supposed to learn to approximate causally at inference time (the model
itself only ever sees bar-i features, never future price data).

Does NOT touch trading_bot.py / trading_bot_modules/ / runtime_config.py / .env. Does NOT modify
any imported module. CPU only (DEVICE=cpu), single default seed (N>=5 seed reproduction is a later,
separate stage), conda env quant_ai.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import core.event_label_engine as label_engine  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816 as cheap_gate  # noqa: E402
import research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816 as ic_check  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816"
DEVICE = guard.DEVICE
WNAME = "val"  # OOS-Q1 is deliberately never loaded/used in this script -- see module docstring.

SEED = 20260816
N_SPLITS = 5
N_PERM = 30
EV_MIN = cheap_gate.EV_MIN            # 0.002, reused not re-typed
MAX_HOLD_BARS = cheap_gate.MAX_HOLD_BARS  # 192, reused not re-typed
HGB_KWARGS = dict(max_iter=140, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0)  # BTC
# production config, unmodified port (train_eval_omega1_2_3_cash_sleeve_upgrade_20260615.py:213)


def log(msg: str) -> None:
    print(f"[cash_sleeve_train_stage1] {msg}", flush=True)


# =====================================================================================================
# Feature group 3: cash-state-history -- new logic, causal, derived only from `held`/`ledger`
# (mirrors trading_bot_modules/omega1_2_3_cash_sleeve.py Omega123CashSleeveAdapter._history_features
# naming/tanh-scaling for consistency: primary_cash_streak and time_since_primary_exit both use
# tanh(x/144.0) exactly as BTC's adapter does -- and are literally the same underlying quantity in
# BTC's own implementation too (both read self.primary_cash_streak), kept as two named columns here
# for naming parity, not because they differ).
# =====================================================================================================
def _cash_state_history_features(n: int, held: np.ndarray, ledger: pd.DataFrame) -> pd.DataFrame:
    cash_streak = np.zeros(n, dtype=np.float64)
    streak = 0
    for i in range(n):
        streak = 0 if held[i] else streak + 1
        cash_streak[i] = streak

    held_f = pd.Series(held.astype(np.float64))
    # shift(1): "how active was primary over the trailing 12/48 bars BEFORE this bar" -- excludes
    # bar i itself (which is always held[i]=False on every CASH bar being scored here, so including
    # it would only dilute the signal with a constant).
    active_roll_12 = held_f.shift(1).rolling(12, min_periods=1).mean().fillna(0.0).to_numpy()
    active_roll_48 = held_f.shift(1).rolling(48, min_periods=1).mean().fillna(0.0).to_numpy()

    sorted_ledger = ledger.sort_values("exit_i").reset_index(drop=True)
    exit_is = sorted_ledger["exit_i"].to_numpy(dtype=np.int64)
    entry_is = sorted_ledger["entry_i"].to_numpy(dtype=np.int64)
    sides = sorted_ledger["side"].to_numpy(dtype=np.int64)
    last_active_len = np.zeros(n, dtype=np.float64)
    last_side = np.zeros(n, dtype=np.float64)
    cur_len, cur_side, j = 0.0, 0.0, 0
    for i in range(n):
        # only trades ALREADY CLOSED strictly before bar i are causally known at bar i's decision time
        while j < len(exit_is) and exit_is[j] < i:
            cur_len = float(exit_is[j] - entry_is[j] + 1)
            cur_side = float(sides[j])
            j += 1
        last_active_len[i] = cur_len
        last_side[i] = cur_side

    return pd.DataFrame(
        {
            "primary_cash_streak": np.tanh(cash_streak / 144.0),
            "time_since_primary_exit": np.tanh(cash_streak / 144.0),
            "primary_active_roll_12": active_roll_12,
            "primary_active_roll_48": active_roll_48,
            "last_primary_active_len": np.tanh(last_active_len / 288.0),
            "last_primary_side": last_side,
        }
    )


# =====================================================================================================
# Feature group 2: primary-trace -- reused omega._source_state (the exact function this lineage
# already uses to turn a raw ThreeHeadTabM prediction CSV into tabm_* trace features), applied to
# both h48qual and zig075's raw prediction CSVs, prefixed per-component. Plus the sustained-uptrend
# detector mask (same detector that gates both h48qual's regime-aware exit guard AND zig075's SHORT
# entry veto -- literally the same boolean array in both places, included once).
# =====================================================================================================
def _primary_trace_features(
    wname: str, windows: dict[str, Any], score_by_base, threshold: float, aligned_frame: pd.DataFrame,
) -> pd.DataFrame:
    w = windows[wname]
    split = gate.WINDOW_DEFS[wname]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    af_check, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
    if not af_check["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{wname}: primary-trace re-derived aligned_frame timestamp mismatch vs primary ledger's own aligned_frame")

    mask, n_nan = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
    out = pd.DataFrame({"sustained_uptrend_detector_active": mask.astype(np.float64)})

    for name in ("h48qual", "zig075"):
        raw = pd.read_csv(aligned_paths[name])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        if not raw["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{wname}/{name}: aligned prediction CSV timestamp mismatch")
        trace = omega._source_state(raw, oof=bool(w["oof"]))
        trace = trace.add_prefix(f"{name}_")
        out = pd.concat([out.reset_index(drop=True), trace.reset_index(drop=True)], axis=1)
    return out


# =====================================================================================================
# Metrics
# =====================================================================================================
def _ranking_metrics(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    ic, _ = spearmanr(pred, real)
    return {
        "spearman_ic": float(ic) if np.isfinite(ic) else float("nan"),
        "r2": float(r2_score(real, pred)),
        "mae": float(mean_absolute_error(real, pred)),
    }


def _decision_metrics(long_pred: np.ndarray, short_pred: np.ndarray, long_real: np.ndarray, short_real: np.ndarray,
                       *, ev_min: float = EV_MIN, ev_offset: float = 0.0) -> dict[str, Any]:
    pred_max_ev = np.maximum(long_pred, short_pred) - ev_offset
    side_is_long = long_pred >= short_pred
    realized_model_side = np.where(side_is_long, long_real, short_real)
    selected = pred_max_ev > ev_min
    n = len(pred_max_ev)
    n_sel = int(selected.sum())
    mean_edge_all = float(realized_model_side.mean())
    if n_sel > 0:
        frac_clearing = float((realized_model_side[selected] > ev_min).mean())
        mean_edge_sel = float(realized_model_side[selected].mean())
    else:
        frac_clearing, mean_edge_sel = float("nan"), float("nan")
    return {
        "n_bars": int(n),
        "n_selected": n_sel,
        "selected_frac": float(n_sel / n) if n else 0.0,
        "frac_selected_clearing_ev_min_realized": frac_clearing,
        "mean_edge_selected_pct": mean_edge_sel * 100.0 if n_sel else float("nan"),
        "mean_edge_all_bars_model_side_pct": mean_edge_all * 100.0,
        "selected_minus_unconditional_pp": (mean_edge_sel - mean_edge_all) * 100.0 if n_sel else float("nan"),
    }


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    m = HistGradientBoostingRegressor(**HGB_KWARGS, random_state=int(seed))
    m.fit(x_train, y_train)
    return m.predict(x_test)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    fee, slip = omega._load_fee_slip()
    log(f"fee={fee} slip={slip} (real ETH backtest constants, reused via omega._load_fee_slip)")

    log("=== stage=load_windows (VAL only ever used below -- OOS-Q1 intentionally never touched) ===")
    windows = gate.load_all_windows()
    score_by_base, robustness_thresholds, threshold = guard.build_detector()

    log(f"=== stage=primary_ledger window={WNAME} (reused cheap_gate.build_primary_ledger, unmodified) ===")
    aligned_frame, ledger, diag = cheap_gate.build_primary_ledger(WNAME, windows, score_by_base, threshold, fee, slip)
    no_gate = cheap_gate.portfolio._ledger_metrics(ledger)
    ref = cheap_gate.G0_ODYSSEY4_NO_GATE[WNAME]
    g0_ok = cheap_gate._close(no_gate, ref)
    log(f"  G0 sanity vs locked Odyssey4 baseline: actual={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} "
        f"reference={ref['pnl']:.2f}%/{ref['mdd']:.2f}%/{ref['trades']} match={g0_ok}")
    if not g0_ok:
        raise RuntimeError("G0 sanity check failed -- primary ledger reproduction diverged from the locked Odyssey4 baseline")

    n_bars = len(aligned_frame)
    held = cheap_gate._held_mask(n_bars, ledger)
    valid_n = n_bars - 2
    log(f"  n_bars={n_bars} valid_n={valid_n} primary_held_bars={int(held[:valid_n].sum())} "
        f"cash_bars={int((~held[:valid_n]).sum())}")

    log("=== stage=cash_sleeve_oracle_labels (re-running cheap_gate.run_cash_sleeve_oracle directly, not copy-pasted) ===")
    cash_df, valid_n2 = cheap_gate.run_cash_sleeve_oracle(aligned_frame, held, fee, slip)
    assert valid_n2 == valid_n
    published = pd.read_csv(cheap_gate.OUT_DIR / f"cash_sleeve_oracle_bars_{WNAME}.csv", parse_dates=["timestamp"])
    if len(published) != len(cash_df) or not np.array_equal(published["i"].to_numpy(), cash_df["i"].to_numpy()):
        raise RuntimeError("cash_df row set diverged from the already-published cheap_gate oracle CSV")
    if not np.allclose(published["long_net"].to_numpy(), cash_df["long_net"].to_numpy(), rtol=1e-9, atol=1e-12) or \
       not np.allclose(published["short_net"].to_numpy(), cash_df["short_net"].to_numpy(), rtol=1e-9, atol=1e-12):
        raise RuntimeError("cash_df long_net/short_net diverged numerically from the already-published cheap_gate oracle CSV")
    log(f"  {len(cash_df)} CASH bars, labels cross-checked byte-for-byte against the already-published oracle CSV -- OK")

    log("=== stage=feature_panel ===")
    idx = cash_df["i"].to_numpy(dtype=np.int64)
    market_frame = aligned_frame.iloc[idx][[f for f, _g in ic_check.ALL_FEATURES]].reset_index(drop=True)
    market_frame = market_frame.apply(pd.to_numeric, errors="coerce")

    trace_full = _primary_trace_features(WNAME, windows, score_by_base, threshold, aligned_frame)
    trace_frame = trace_full.iloc[idx].reset_index(drop=True)

    history_full = _cash_state_history_features(n_bars, held, ledger)
    history_frame = history_full.iloc[idx].reset_index(drop=True)

    x_full = pd.concat([market_frame, trace_frame, history_frame], axis=1)
    feature_cols = list(x_full.columns)
    n_constant = int((x_full.std(numeric_only=True) < 1e-12).sum())
    log(f"  feature groups: market={len(market_frame.columns)} primary_trace={len(trace_frame.columns)} "
        f"cash_state_history={len(history_frame.columns)} total={len(feature_cols)} (n_near_constant={n_constant})")

    x_arr = x_full.to_numpy(dtype=np.float64)  # HistGradientBoostingRegressor natively supports NaN
    y_long = cash_df["long_net"].to_numpy(dtype=np.float64)
    y_short = cash_df["short_net"].to_numpy(dtype=np.float64)
    n_events = len(cash_df)

    log("=== stage=purged_cv_setup (core.event_label_engine.purged_kfold_splits, reused unmodified) ===")
    event_idx = idx.copy()
    t1_idx = np.minimum(event_idx + MAX_HOLD_BARS, n_bars - 1)  # conservative per-event horizon upper bound
    folds = list(label_engine.purged_kfold_splits(event_idx, t1_idx, n_bars, n_splits=N_SPLITS, embargo_frac=0.0))
    assert len(folds) == N_SPLITS

    long_pred_oof = np.full(n_events, np.nan, dtype=np.float64)
    short_pred_oof = np.full(n_events, np.nan, dtype=np.float64)
    perm_long_oof = np.full((N_PERM, n_events), np.nan, dtype=np.float64)
    perm_short_oof = np.full((N_PERM, n_events), np.nan, dtype=np.float64)

    fold_diag: list[dict[str, Any]] = []
    for k, (train_mask, test_mask) in enumerate(folds):
        train_ev, test_ev = event_idx[train_mask], event_idx[test_mask]
        gap_before = float(test_ev.min() - train_ev[train_ev < test_ev.min()].max()) if np.any(train_ev < test_ev.min()) else float("nan")
        gap_after = float(train_ev[train_ev > test_ev.max()].min() - test_ev.max()) if np.any(train_ev > test_ev.max()) else float("nan")
        fold_diag.append({
            "fold": k, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "test_event_range": [int(test_ev.min()), int(test_ev.max())],
            "purge_gap_before_bars": gap_before, "purge_gap_after_bars": gap_after,
        })
        log(f"  fold={k} n_train={train_mask.sum()} n_test={test_mask.sum()} "
            f"purge_gap_before={gap_before} purge_gap_after={gap_after}")
        if np.isfinite(gap_before) and gap_before < MAX_HOLD_BARS:
            raise RuntimeError(f"fold {k}: purge gap_before={gap_before} < {MAX_HOLD_BARS} bars")
        if np.isfinite(gap_after) and gap_after < MAX_HOLD_BARS:
            raise RuntimeError(f"fold {k}: purge gap_after={gap_after} < {MAX_HOLD_BARS} bars")

        x_tr, x_te = x_arr[train_mask], x_arr[test_mask]
        yl_tr, ys_tr = y_long[train_mask], y_short[train_mask]

        long_pred_oof[test_mask] = _fit_predict(x_tr, yl_tr, x_te, seed=SEED + k * 10 + 1)
        short_pred_oof[test_mask] = _fit_predict(x_tr, ys_tr, x_te, seed=SEED + k * 10 + 2)

        for r in range(N_PERM):
            perm_rng = np.random.default_rng(SEED * 1_000_003 + k * 1009 + r)
            perm = perm_rng.permutation(len(yl_tr))
            perm_long_oof[r, test_mask] = _fit_predict(x_tr, yl_tr[perm], x_te, seed=SEED + k * 10 + 1 + (r + 1) * 100_000)
            perm_short_oof[r, test_mask] = _fit_predict(x_tr, ys_tr[perm], x_te, seed=SEED + k * 10 + 2 + (r + 1) * 100_000)
        log(f"    fold {k}: {N_PERM} label-permutation repeats done")

    assert not np.isnan(long_pred_oof).any() and not np.isnan(short_pred_oof).any(), "OOF coverage incomplete"
    assert not np.isnan(perm_long_oof).any() and not np.isnan(perm_short_oof).any(), "permutation OOF coverage incomplete"

    log("=== stage=metrics (real vs permutation-null) ===")
    real_long_metrics = _ranking_metrics(long_pred_oof, y_long)
    real_short_metrics = _ranking_metrics(short_pred_oof, y_short)
    real_decision = _decision_metrics(long_pred_oof, short_pred_oof, y_long, y_short)
    log(f"  REAL long: ic={real_long_metrics['spearman_ic']:.4f} r2={real_long_metrics['r2']:.4f} mae={real_long_metrics['mae']:.5f}")
    log(f"  REAL short: ic={real_short_metrics['spearman_ic']:.4f} r2={real_short_metrics['r2']:.4f} mae={real_short_metrics['mae']:.5f}")
    log(f"  REAL decision: n_selected={real_decision['n_selected']} ({real_decision['selected_frac']:.4f} of CASH bars) "
        f"frac_clearing_ev_min={real_decision['frac_selected_clearing_ev_min_realized']} "
        f"mean_edge_selected_pct={real_decision['mean_edge_selected_pct']:.4f} "
        f"vs unconditional_pct={real_decision['mean_edge_all_bars_model_side_pct']:.4f}")

    null_long_ic = np.array([_ranking_metrics(perm_long_oof[r], y_long)["spearman_ic"] for r in range(N_PERM)])
    null_short_ic = np.array([_ranking_metrics(perm_short_oof[r], y_short)["spearman_ic"] for r in range(N_PERM)])
    null_decisions = [
        _decision_metrics(perm_long_oof[r], perm_short_oof[r], y_long, y_short) for r in range(N_PERM)
    ]
    null_mean_edge_sel = np.array([d["mean_edge_selected_pct"] for d in null_decisions], dtype=np.float64)
    null_frac_clear = np.array([d["frac_selected_clearing_ev_min_realized"] for d in null_decisions], dtype=np.float64)
    null_sel_minus_uncond = np.array([d["selected_minus_unconditional_pp"] for d in null_decisions], dtype=np.float64)
    null_n_selected = np.array([d["n_selected"] for d in null_decisions], dtype=np.float64)

    def _z(real: float, null: np.ndarray) -> float:
        null_f = null[np.isfinite(null)]
        if len(null_f) < 2 or not np.isfinite(real):
            return float("nan")
        s = float(null_f.std())
        return float((real - null_f.mean()) / s) if s > 1e-12 else float("nan")

    def _pctile_rank(real: float, null: np.ndarray) -> float:
        null_f = null[np.isfinite(null)]
        if len(null_f) == 0 or not np.isfinite(real):
            return float("nan")
        return float((null_f <= real).mean())

    verdict = {
        "long_ic": {"real": real_long_metrics["spearman_ic"], "null_mean": float(np.nanmean(null_long_ic)),
                    "null_std": float(np.nanstd(null_long_ic)), "z": _z(real_long_metrics["spearman_ic"], null_long_ic),
                    "percentile_rank": _pctile_rank(real_long_metrics["spearman_ic"], null_long_ic)},
        "short_ic": {"real": real_short_metrics["spearman_ic"], "null_mean": float(np.nanmean(null_short_ic)),
                     "null_std": float(np.nanstd(null_short_ic)), "z": _z(real_short_metrics["spearman_ic"], null_short_ic),
                     "percentile_rank": _pctile_rank(real_short_metrics["spearman_ic"], null_short_ic)},
        "mean_edge_selected_pct": {"real": real_decision["mean_edge_selected_pct"], "null_mean": float(np.nanmean(null_mean_edge_sel)),
                                    "null_std": float(np.nanstd(null_mean_edge_sel)),
                                    "z": _z(real_decision["mean_edge_selected_pct"], null_mean_edge_sel),
                                    "percentile_rank": _pctile_rank(real_decision["mean_edge_selected_pct"], null_mean_edge_sel)},
        "frac_selected_clearing_ev_min": {"real": real_decision["frac_selected_clearing_ev_min_realized"], "null_mean": float(np.nanmean(null_frac_clear)),
                                           "null_std": float(np.nanstd(null_frac_clear)),
                                           "z": _z(real_decision["frac_selected_clearing_ev_min_realized"], null_frac_clear),
                                           "percentile_rank": _pctile_rank(real_decision["frac_selected_clearing_ev_min_realized"], null_frac_clear)},
        "selected_minus_unconditional_pp": {"real": real_decision["selected_minus_unconditional_pp"], "null_mean": float(np.nanmean(null_sel_minus_uncond)),
                                             "null_std": float(np.nanstd(null_sel_minus_uncond)),
                                             "z": _z(real_decision["selected_minus_unconditional_pp"], null_sel_minus_uncond),
                                             "percentile_rank": _pctile_rank(real_decision["selected_minus_unconditional_pp"], null_sel_minus_uncond)},
    }
    for k, v in verdict.items():
        log(f"  {k}: real={v['real']} null_mean={v['null_mean']:.5f} null_std={v['null_std']:.5f} z={v['z']} pct_rank={v['percentile_rank']}")

    # Pre-specified pass bar (documented BEFORE looking at the numbers, in the module docstring's
    # metric design + orchestrating-session instruction): both IC z-scores >= 2.0 (real IC clearly
    # outside the null distribution) AND the decision-relevant selected-vs-unconditional edge is
    # positive with z >= 2.0. All must hold -- picking whichever metric looks best after the fact is
    # exactly the failure mode this stage is designed to catch.
    beats_null = bool(
        np.isfinite(verdict["long_ic"]["z"]) and verdict["long_ic"]["z"] >= 2.0
        and np.isfinite(verdict["short_ic"]["z"]) and verdict["short_ic"]["z"] >= 2.0
        and np.isfinite(verdict["selected_minus_unconditional_pp"]["z"]) and verdict["selected_minus_unconditional_pp"]["z"] >= 2.0
    )
    log(f"=== verdict: beats_permutation_null (pre-specified z>=2.0 on both IC + decision-edge)={beats_null} ===")

    # Persist outputs
    oof_df = pd.DataFrame({
        "i": event_idx, "timestamp": cash_df["timestamp"].to_numpy(),
        "long_net": y_long, "short_net": y_short,
        "long_pred": long_pred_oof, "short_pred": short_pred_oof,
    })
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_diag).to_csv(OUT_DIR / "fold_purge_diagnostics.csv", index=False)
    pd.DataFrame({
        "perm": np.arange(N_PERM), "long_ic": null_long_ic, "short_ic": null_short_ic,
        "mean_edge_selected_pct": null_mean_edge_sel, "frac_selected_clearing_ev_min": null_frac_clear,
        "selected_minus_unconditional_pp": null_sel_minus_uncond, "n_selected": null_n_selected,
    }).to_csv(OUT_DIR / "permutation_null.csv", index=False)

    report: dict[str, Any] = {
        "type": "stage1_real_training_purged_cv_plus_permutation_null",
        "candidate": "eth_candidate_cash_sleeve_ev_hgb",
        "scope": "VAL window ONLY (2025-10-01..12-31); OOS-Q1 never loaded/used in this script",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "labels_use_realized_future_price_path_by_design": True,
        "seed": SEED, "n_splits": N_SPLITS, "n_perm": N_PERM, "ev_min": EV_MIN, "max_hold_bars": MAX_HOLD_BARS,
        "hgb_kwargs": HGB_KWARGS,
        "n_cash_bars": int(n_events),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_near_constant_features": n_constant,
        "g0_sanity_check": {"actual": no_gate, "reference": ref, "match": g0_ok},
        "fold_purge_diagnostics": fold_diag,
        "real_long_metrics": real_long_metrics,
        "real_short_metrics": real_short_metrics,
        "real_decision_metrics": real_decision,
        "null_long_ic": {"mean": float(np.nanmean(null_long_ic)), "std": float(np.nanstd(null_long_ic)), "values": null_long_ic.tolist()},
        "null_short_ic": {"mean": float(np.nanmean(null_short_ic)), "std": float(np.nanstd(null_short_ic)), "values": null_short_ic.tolist()},
        "verdict_vs_null": verdict,
        "pre_specified_pass_rule": "z>=2.0 on long_ic AND short_ic AND selected_minus_unconditional_pp (all three, not any one)",
        "beats_permutation_null": beats_null,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
