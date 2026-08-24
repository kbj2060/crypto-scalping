#!/usr/bin/env python3
"""0단계 진단 -- meta-label rank correlation on direction_head's REAL (imperfect) predictions.
(2026-08-15, Odyssey2 #23 -- the one open thread left by contract #22.)

=== What this settles ===
docs/experiments/eth_zigzag_metalabel_evidence_signal_rank_correlation_20260814.md found evidence-
signal agreement predicts the realized quality of a zigzag_action bet in 5/5 windows (rho +0.026 to
+0.061, n=3-5k per window). But that used `zigzag_action` itself as the primary signal, and
zigzag_action is an ORACLE -- it needs forward price confirmation to construct. That document's own
"next step" section, and contract #22's single remaining open issue, both name the same follow-up:
rerun the identical correlation with `direction_head`'s ACTUAL, imperfect prediction
(`omega1_regime3_expertdq_oof_dir_action`) as the primary bet.

The stakes are pre-registered and asymmetric:
  POSITIVE -> "when the model's real prediction is wrong, evidence-signal disagreement detects it"
              is the first genuine practical evidence in this whole lineage, and the only thing that
              would justify reopening the entry-side anti-goal for discussion.
  NEGATIVE -> the evidence-signal -> Omega axis is closed permanently, with all five injection
              forms plus the meta-label idea ruled out.

=== PRE-REGISTERED (fixed before any result was inspected) ===
Primary population: every bar where `dir_action` in {1=LONG, 2=SHORT} -- direction_head's own bet
  BEFORE the quality gate. Reported per component (h48qual, zig075) independently; neither is
  selected after the fact.
Secondary population: bars where `final_action` is active (the quality-gated subset that is actually
  traded). Smaller n, reported alongside, never used to overturn the primary.
Simulation: core.causal_futures_backtest.simulate_single_position with side FIXED to dir_action's
  own call, and TP=1.6xATR / SL=1.0xATR / 48 bars / 3x / margin 0.30 / round trip 0.1% -- byte-for-
  byte the same constants as the oracle version, so the two are directly comparable. The simulation
  is chronological and NON-OVERLAPPING (a new decision is only taken when flat), exactly as in the
  oracle run.
Agreement score: bottom_votes if the bet is LONG, top_votes if SHORT (0-6), from the unmodified
  `_compute_signal_frame` of the top-6 confluence backtest. No new threshold anywhere.
Windows: all 6 (2025q1/q2/q3, val, oos_q1, oos_q2). The oracle version could only test 5 because
  zigzag_action labels stop at 2026-02-28; prediction CSVs have no such gap, so oos_q2 is included.
PRIMARY METRIC: Spearman(agreement_score, trade_return) per window.
KILL CRITERION: unless >= 4 of the 6 windows show POSITIVE rho for a given component, that
  component's meta-label idea is CLOSED. (Same 4-of-6 bar contract #22 named.)

MANDATORY CONTROLS -- this sub-project has now been burned three times by an effect that was real
but already contained in something simpler/free, so a bare positive rho is not accepted as a result:
  (1) PERMUTATION: agreement scores shuffled across the window's own trades, 200 replicates, fixed
      seed -> empirical p-value for the observed rho.
  (2) SHORT-TERM REVERSAL BENCHMARK: `reversal_support` = -side * ret3_z at the decision bar (for a
      LONG, a more negative recent 15-min return is more reversal support). Note ret3_z is LITERALLY
      one of the six components of the agreement score, so this is the sharpest possible version of
      the "is it just reversal?" question.
  (3) PARTIAL correlation of agreement with trade_return CONTROLLING for reversal_support (computed
      on ranks). If the partial collapses toward zero, the agreement score is a repackaging of its
      own reversal component and adds nothing.

Causal status: the TP/SL walk-forward is causal (entry at decision_i+1 open, bar-by-bar resolution),
and unlike the oracle version EVERY input here is causal too -- dir_action is a real model output
available at the decision bar, and the evidence signals are rolling/shift only. This is therefore a
strictly stronger test than the oracle version. It remains a DIAGNOSTIC: no training, no deployment,
no live file touched, and per contract #22 a positive result licenses only a DISCUSSION of the
entry-side anti-goal, never an injection.

trade_ledgers_used_as_input=false (ledgers are fresh outputs of this run).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
from backtest_eth_evidence_signal_top6_confluence_20260814 import (  # noqa: E402
    HORIZON_BARS, LEVERAGE, MARGIN_FRACTION, ROUNDTRIP_COST_RATE, SL_ATR_MULT, TP_ATR_MULT,
    _compute_signal_frame,
)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_direction_head_metalabel_evidence_signal_rank_correlation_20260815"
COMPONENTS = ("h48qual", "zig075")
WINDOWS = ("2025q1", "2025q2", "2025q3", "val", "oos_q1", "oos_q2")
PRED_PREFIX = "omega1_regime3_expertdq_oof_"       # train/validation splits (out-of-fold)
PRED_PREFIX_OOS = "omega1_regime3_expertdq_"       # oos split (no _oof_ infix)
SPLIT_FILE_PREFIX = {"train": "train_predictions", "validation": "validation_predictions", "oos": "oos_predictions"}
PERM_REPS, PERM_SEED = 200, 20260815
KILL_MIN_POSITIVE_WINDOWS = 4
# published oracle-version results, for direct side-by-side comparison (not recomputed here)
ORACLE_RHO = {"2025q1": 0.058, "2025q2": 0.028, "2025q3": 0.061, "val": 0.042, "oos_q1": 0.026, "oos_q2": None}


def log(msg: str) -> None:
    print(f"[dirhead_metalabel] {msg}", flush=True)


def _ret3_z(base_csv: Path) -> pd.DataFrame:
    """ret3_z exactly as _compute_signal_frame defines it internally (close/close.shift(3)-1,
    z-scored over 288 bars), recomputed here because that function does not return it."""
    raw = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "close"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    r = raw["close"] / raw["close"].shift(3) - 1.0
    z = (r - r.rolling(288, min_periods=288).mean()) / r.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    return pd.DataFrame({"timestamp": raw["timestamp"], "ret3_z": z})


def _partial_spearman(a: np.ndarray, y: np.ndarray, ctrl: np.ndarray) -> float:
    """Spearman partial correlation of a with y controlling for ctrl (residual method on ranks)."""
    ok = np.isfinite(a) & np.isfinite(y) & np.isfinite(ctrl)
    if ok.sum() < 20:
        return float("nan")
    ra, ry, rc = (stats.rankdata(v[ok]) for v in (a, y, ctrl))
    rc_c = rc - rc.mean()
    denom = float((rc_c ** 2).sum())
    if denom <= 0:
        return float("nan")
    res_a = (ra - ra.mean()) - rc_c * float((rc_c * (ra - ra.mean())).sum() / denom)
    res_y = (ry - ry.mean()) - rc_c * float((rc_c * (ry - ry.mean())).sum() / denom)
    if res_a.std() == 0 or res_y.std() == 0:
        return float("nan")
    return float(np.corrcoef(res_a, res_y)[0, 1])


def run_population(frame: pd.DataFrame, *, start, end, side_col: str, rng: np.random.Generator) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    has_decision = (frame[side_col] != 0).to_numpy() & frame["atr_pct"].notna().to_numpy()
    decision_indices = np.flatnonzero(eligible & has_decision)
    if len(decision_indices) < 20:
        return {"n_decision_bars": int(len(decision_indices)), "skipped": "too_few_decision_bars"}

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices,
        scores=frame[side_col].to_numpy(dtype=np.float64)[decision_indices],
        tp_moves=(TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices],
        sl_moves=(SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices],
        upper_threshold=0.5, lower_threshold=-0.5, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    if ledger.empty:
        return {"n_decision_bars": int(len(decision_indices)), "n_trades": 0}

    by_ts = frame.set_index("timestamp")
    agreement = by_ts["agreement_score"].reindex(ledger["decision_timestamp"]).to_numpy(dtype=float)
    reversal = by_ts["reversal_support"].reindex(ledger["decision_timestamp"]).to_numpy(dtype=float)
    ret = ledger["trade_return"].to_numpy(dtype=float)

    ok = np.isfinite(agreement) & np.isfinite(ret)
    n = int(ok.sum())
    if n < 20 or np.unique(agreement[ok]).size < 2:
        return {"n_decision_bars": int(len(decision_indices)), "n_trades": int(len(ledger)),
                "skipped": "too_few_trades_or_no_variation"}

    rho, pval = stats.spearmanr(agreement[ok], ret[ok])
    perm = np.array([stats.spearmanr(rng.permutation(agreement[ok]), ret[ok]).statistic for _ in range(PERM_REPS)])
    rev_ok = np.isfinite(reversal) & ok
    rev_rho = float(stats.spearmanr(reversal[rev_ok], ret[rev_ok]).statistic) if rev_ok.sum() >= 20 else float("nan")

    n_agree = int((agreement[ok] >= 1).sum())
    return {
        "n_decision_bars": int(len(decision_indices)), "n_trades": int(len(ledger)), "n_used": n,
        "win_rate": float((ret[ok] > 0).mean()),
        "long_share": float((ledger["side"] > 0).mean()) if "side" in ledger.columns else float("nan"),
        "n_agreement_ge1": n_agree,
        "mean_return_agreement_ge1": float(ret[ok][agreement[ok] >= 1].mean()) if n_agree else float("nan"),
        "mean_return_agreement_eq0": float(ret[ok][agreement[ok] == 0].mean()) if n - n_agree else float("nan"),
        "spearman_rho": float(rho), "p_value": float(pval),
        "permutation": {"reps": PERM_REPS, "mean": float(perm.mean()), "std": float(perm.std()),
                        "empirical_p_two_sided": float((np.abs(perm) >= abs(float(rho))).mean())},
        "reversal_benchmark_rho": rev_rho,
        "partial_rho_controlling_reversal": _partial_spearman(agreement, ret, reversal),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PERM_SEED)
    log("=== stage=build_signal_frames ===")
    signal_by_base = {sweep.BASE_2025: _compute_signal_frame(sweep.BASE_2025),
                      sweep.BASE_2026: _compute_signal_frame(sweep.BASE_2026)}
    ret3z_by_base = {sweep.BASE_2025: _ret3_z(sweep.BASE_2025), sweep.BASE_2026: _ret3_z(sweep.BASE_2026)}

    report: dict[str, Any] = {
        "design": "Meta-label rank correlation using direction_head's REAL (imperfect) dir_action as the primary bet, "
                  "replacing the oracle zigzag_action of the 2026-08-14 version. Same simulation constants, same "
                  "agreement score, all 6 windows.",
        "pre_registered": {"primary_population": "dir_action in {LONG,SHORT}", "secondary_population": "final_action active",
                           "components": list(COMPONENTS), "windows": list(WINDOWS),
                           "kill_criterion": f">= {KILL_MIN_POSITIVE_WINDOWS} of 6 windows positive rho, else CLOSED",
                           "controls": ["permutation", "reversal_benchmark", "partial_controlling_reversal"],
                           "perm_reps": PERM_REPS, "perm_seed": PERM_SEED},
        "all_inputs_causal": True,
        "diagnostic_only": "A positive result licenses only a DISCUSSION of the entry-side anti-goal (contract #22); "
                           "it is not an injection mandate. No training, no deployment.",
        "oracle_version_rho_for_comparison": ORACLE_RHO,
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "results": {},
    }

    for comp in COMPONENTS:
        q_tag = sweep.COMPONENTS[comp]["q_tag"]
        log(f"=== component={comp} (q_tag={q_tag}) ===")
        for wname in WINDOWS:
            wd = gate.WINDOW_DEFS[wname]
            pred_path = sweep.EXT_PRED_DIR / comp / f"{SPLIT_FILE_PREFIX[wd['split']]}_{q_tag}.csv"
            # OOF splits (train/validation) carry the "_oof_" infix; the OOS split does not (it is
            # not out-of-fold). Resolve per file rather than assuming one naming convention.
            head = pd.read_csv(pred_path, nrows=0)
            prefix = PRED_PREFIX if f"{PRED_PREFIX}dir_action" in head.columns else PRED_PREFIX_OOS
            if f"{prefix}dir_action" not in head.columns:
                raise RuntimeError(f"{pred_path}: neither '{PRED_PREFIX}dir_action' nor '{PRED_PREFIX_OOS}dir_action' present")
            pred = pd.read_csv(pred_path, usecols=["timestamp", f"{prefix}dir_action", f"{prefix}final_action"])
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            pred = pred.rename(columns={f"{prefix}dir_action": "dir_action", f"{prefix}final_action": "final_action"})

            frame = signal_by_base[wd["base_csv"]].merge(pred, on="timestamp", how="left")
            frame = frame.merge(ret3z_by_base[wd["base_csv"]], on="timestamp", how="left")
            for col in ("dir_action", "final_action"):
                frame[col] = frame[col].fillna(0).astype(int)
                frame[f"side_{col}"] = np.where(frame[col] == 1, 1, np.where(frame[col] == 2, -1, 0))

            side_primary = frame["side_dir_action"].to_numpy()
            frame["agreement_score"] = np.where(side_primary > 0, frame["bottom_votes"],
                                                np.where(side_primary < 0, frame["top_votes"], np.nan))
            frame["reversal_support"] = -side_primary * frame["ret3_z"].to_numpy()

            res_primary = run_population(frame, start=wd["start"], end=wd["end"], side_col="side_dir_action", rng=rng)

            side_secondary = frame["side_final_action"].to_numpy()
            frame_sec = frame.copy()
            frame_sec["agreement_score"] = np.where(side_secondary > 0, frame["bottom_votes"],
                                                    np.where(side_secondary < 0, frame["top_votes"], np.nan))
            frame_sec["reversal_support"] = -side_secondary * frame["ret3_z"].to_numpy()
            res_secondary = run_population(frame_sec, start=wd["start"], end=wd["end"], side_col="side_final_action", rng=rng)

            report["results"][f"{comp}_{wname}"] = {"primary_dir_action": res_primary, "secondary_final_action": res_secondary}
            if "spearman_rho" in res_primary:
                log(f"  {wname:8s} PRIMARY  trades={res_primary['n_trades']:5d} wr={res_primary['win_rate']:.3f} "
                    f"rho={res_primary['spearman_rho']:+.4f} p={res_primary['p_value']:.4f} "
                    f"perm_p={res_primary['permutation']['empirical_p_two_sided']:.3f} | "
                    f"reversal_rho={res_primary['reversal_benchmark_rho']:+.4f} "
                    f"partial={res_primary['partial_rho_controlling_reversal']:+.4f} "
                    f"(oracle was {ORACLE_RHO.get(wname)})")
            else:
                log(f"  {wname:8s} PRIMARY  skipped: {res_primary}")
            if "spearman_rho" in res_secondary:
                log(f"  {wname:8s} SECOND   trades={res_secondary['n_trades']:5d} rho={res_secondary['spearman_rho']:+.4f} "
                    f"p={res_secondary['p_value']:.4f} partial={res_secondary['partial_rho_controlling_reversal']:+.4f}")
            else:
                log(f"  {wname:8s} SECOND   skipped: {res_secondary.get('skipped', res_secondary)}")

    verdicts = {}
    for comp in COMPONENTS:
        rhos = {w: report["results"][f"{comp}_{w}"]["primary_dir_action"].get("spearman_rho") for w in WINDOWS}
        computed = {w: v for w, v in rhos.items() if v is not None}
        n_pos = sum(1 for v in computed.values() if v > 0)
        partials = [report["results"][f"{comp}_{w}"]["primary_dir_action"].get("partial_rho_controlling_reversal") for w in WINDOWS]
        partials = [v for v in partials if v is not None and np.isfinite(v)]
        verdicts[comp] = {
            "rho_by_window": rhos, "windows_computed": len(computed), "windows_positive": n_pos,
            "partial_positive_windows": int(sum(1 for v in partials if v > 0)), "partial_windows_computed": len(partials),
            "passes_kill_criterion": bool(n_pos >= KILL_MIN_POSITIVE_WINDOWS),
        }
        log(f"VERDICT {comp}: {n_pos}/{len(computed)} windows positive -> pass={verdicts[comp]['passes_kill_criterion']} "
            f"| partial positive in {verdicts[comp]['partial_positive_windows']}/{verdicts[comp]['partial_windows_computed']}")
    report["verdicts"] = verdicts
    report["overall"] = ("SURVIVES" if any(v["passes_kill_criterion"] for v in verdicts.values())
                         else "CLOSED_METALABEL_FAILS_ON_REAL_PREDICTIONS")
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done OVERALL={report['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
