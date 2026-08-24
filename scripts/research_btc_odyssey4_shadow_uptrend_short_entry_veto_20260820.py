#!/usr/bin/env python3
"""RESEARCH ONLY -- BTC h48qual+swingtransition SHORT entry veto during detected sustained
uptrends. Transfers the ETH Odyssey4 entry-veto mechanism
(research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py) to BTC's
single-component architecture (no dual router, no zig075).

Background: eth_odyssey4_shadow_full_reseed_causal_isolation_20260820.md confirmed the ETH
Odyssey4 shadow's 6/6 seed-sign-stability is a genuine effect of the entry-veto+exit-guard
mechanism itself, not an artifact of which component happened to be held fixed. A cheap
pre-check (research_btc_odyssey4_shadow_uptrend_short_variance_diagnostic_20260820.py,
ledger-only reaggregation, no new training/replay) found a MIXED signal on BTC: of the 3
baseline sign-flip windows, only 2025q2 was resolved with control-group specificity (random-
exclusion success 8.6% vs non-uptrend-SHORT-exclusion success 0%); 2025q3/oos_q1 were not
resolved, and val/oos_q1 std got WORSE than the random-exclusion control. Both that diagnostic's
author-agent and this script's author judged "the full mechanism looks premature" -- the user
was told this directly and explicitly chose to build and test it anyway (2026-08-20 session,
"그래도 BTC 전체 메커니즘 구축").

Scope note -- entry-veto only, NOT exit-guard: ETH's exit guard switches an in-position h48qual
trade between TWO already-trained exit_head bundles (original vs the 2026-08-13 liveATR-relabel
variant) once a sustained uptrend is detected. BTC has no second h48qual exit_head variant
trained -- building one would be a full labeling+training project, not a "wire up existing
pieces" mechanism build, and was not part of what was approved this session. This script
therefore implements the entry-veto half only (cheap: reuses existing bundles/predictions with
ZERO retraining) and leaves the exit-guard half as a distinct, larger, not-yet-approved
follow-up if this axis is ever reopened.

Detector -- verbatim reuse of the ETH Odyssey3/4 SustainedUptrendDetector's own formula and
structural constants (WEEK_BARS=2016, DETECTOR_PERCENTILE=0.90; see
research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py's own docstring for why
these particular numbers were fixed once and never swept -- they are tied to bar granularity,
5-min bars, shared across assets, not asset-specific), applied to BTC's OWN dual_momentum
feature and calibrated on BTC's OWN 2025-Q1+Q2-only sample -- neither the ETH threshold
(0.8025793650793651) nor yesterday's diagnostic placeholder (0.604167, which used the FULL 2025
year rather than Q1+Q2-only and was explicitly marked "진단용 placeholder, 프로덕션 아님") is
reused. Score is computed on BTC's own dedicated per-year feature CSVs (same files yesterday's
diagnostic read dual_momentum from: data/splits/year_oos/btc_features_{2025,2026}_
swingtransition.csv -- confirmed to span the full calendar year with zero gaps/NaN in
dual_momentum itself), never window-by-window, for the same reason the ETH script does this:
windows starting mid-year (val, oos_q2) must not get an artificial NaN-truncation at their own
start.

Veto mechanism: BTC's replay engine (train_eval_omega4_2_risk_sidecar_btc_20260708.py::
_replay_with_risk, imported and NEVER edited) gates flat-state entries on
omega._active(dec) = (action != CASH) & (side != 0) & (notional_exposure > 0). Because BTC is a
SINGLE-component replay (no greedy multi-candidate router the way ETH's dual is), the veto needs
no replay-loop copy at all -- it is a pure dec-preprocessing step:
dec.loc[(side == -1) & detector_mask, "side"] = 0
applied before dec is handed to the untouched _replay_with_risk. This is simpler than ETH's veto
(which required a renamed replay-loop copy to intercept one candidate among several competing
components) and touches zero shared/tested replay code.

No new training: reuses the exact 5 seed bundles + their own frozen prediction CSVs already
produced for btc_live_promotion_seed_robustness_eval_5seed_20260819.py (260620_original +
750703416/160125165/626578270/179796523). Risk sidecar stays frozen/shared across all 5 seeds,
same explicit simplification as that script (and as the ETH/Ilias1 N=5 axes).

G0 fidelity gate: stage=g0_no_veto re-runs this script's own copy of the per-seed/per-window
replay loop with the detector built but the veto never applied (apply_veto=False), and must
reproduce btc_live_promotion_seed_robustness_20260819_eval/report.json bit-for-bit (pnl/mdd/
trades, all 5 seeds x 6 windows, veto_bars==0 everywhere) before stage=candidate_run's veto
numbers are trusted at all. Aborts (no candidate_run, no report written beyond the G0 failure
detail) if G0 does not pass.

Fresh-Forward: fresh_forward_bar_by_bar=true (each window's entries are that seed bundle's own
causal predictions for that window; exits are _replay_with_risk's single forward bar-by-bar
loop; the veto reads the detector mask at the signal bar only, itself a backward-looking rolling
mean of already-closed history -- no lookahead). trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. This IS a genuine
causal replay (unlike yesterday's diagnostic, which was pure ledger reaggregation) -- but per
CLAUDE.md's Seed-Diversity Ensemble Promotion Gate, still N=5-seed RESEARCH/candidate evidence
on a single index-order seed pairing, not itself a promotion claim.
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

import btc_live_promotion_seed_robustness_eval_5seed_20260819 as ref_eval  # noqa: E402
import apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806 as scale_ref  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar  # noqa: E402

parent_script = ref_eval.parent_script
omega = ref_eval.omega
DEVICE = torch.device("cpu")

OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_odyssey4_shadow_uptrend_short_entry_veto_20260820"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_20260819_eval/report.json"
G0_TOLERANCE_PP = 0.05

# Reused verbatim from the ETH detector (structural constants, not asset-specific).
WEEK_BARS = 2016
DETECTOR_PERCENTILE = 0.90
CALIBRATION_START = ref_eval.WINDOW_DEFS["2025q1"]["start"]
CALIBRATION_END = ref_eval.WINDOW_DEFS["2025q2"]["end"]

BASE_2025 = ROOT / "data/splits/year_oos/btc_features_2025_swingtransition.csv"
BASE_2026 = ROOT / "data/splits/year_oos/btc_features_2026_swingtransition.csv"
SPLIT_TO_BASE = {"train": BASE_2025, "validation": BASE_2025, "oos": BASE_2026}


def log(msg: str) -> None:
    print(f"[btc_uptrend_short_veto] {msg}", flush=True)


# =====================================================================================================
# Detector construction -- BTC's own dual_momentum, BTC's own 2025-Q1+Q2-only calibration sample.
# =====================================================================================================


def _rolling_dual_momentum_score(base_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(base_csv, usecols=["timestamp", "dual_momentum"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    dm = pd.to_numeric(frame["dual_momentum"], errors="raise")
    dm_pos = (dm > 0).astype(float)
    frame["sustained_uptrend_score"] = dm_pos.rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    return frame[["timestamp", "sustained_uptrend_score"]]


def build_detector() -> tuple[dict[Path, pd.DataFrame], float]:
    score_2025 = _rolling_dual_momentum_score(BASE_2025)
    score_2026 = _rolling_dual_momentum_score(BASE_2026)
    calib_mask = (score_2025["timestamp"] >= pd.Timestamp(CALIBRATION_START)) & (score_2025["timestamp"] <= pd.Timestamp(CALIBRATION_END))
    calib = score_2025.loc[calib_mask, "sustained_uptrend_score"].dropna()
    if len(calib) < WEEK_BARS:
        raise RuntimeError(f"calibration sample too small: {len(calib)} rows")
    threshold = float(calib.quantile(DETECTOR_PERCENTILE))
    return {BASE_2025: score_2025, BASE_2026: score_2026}, threshold


def _detector_mask_for_window(frame: pd.DataFrame, split: str, score_by_base: dict[Path, pd.DataFrame], threshold: float) -> tuple[np.ndarray, int]:
    score = score_by_base[SPLIT_TO_BASE[split]]
    merged = frame[["timestamp"]].merge(score, on="timestamp", how="left")
    if len(merged) != len(frame) or not merged["timestamp"].reset_index(drop=True).equals(frame["timestamp"].reset_index(drop=True)):
        raise RuntimeError("detector score merge failed (row count/order mismatch)")
    raw = merged["sustained_uptrend_score"]
    n_nan = int(raw.isna().sum())
    mask = (raw > threshold).fillna(False).to_numpy(dtype=bool)
    return mask, n_nan


# =====================================================================================================
# Per-seed/per-window replay -- follows btc_live_promotion_seed_robustness_eval_5seed_20260819.py's
# main() inner loop exactly (that module is imported, never edited). Only new logic: the veto block,
# marked below.
# =====================================================================================================


def _replay_one(seed_label: str, wname: str, wd: dict, frame_full: pd.DataFrame, pkl: dict, score_by_base, threshold: float, fee: float, slip: float, *, apply_veto: bool):
    bundle_dir = ref_eval._bundle_dir_for(seed_label)
    bundle = torch.load(bundle_dir / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(bundle["models"], device=DEVICE)

    pred_path = bundle_dir / f"{ref_eval._PRED_SPLIT_PREFIX[wd['split']]}_{ref_eval.Q_TAG}.csv"
    pred_full = pd.read_csv(pred_path)
    pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"])
    if len(pred_full) != len(frame_full) or not pred_full["timestamp"].equals(frame_full["timestamp"]):
        raise RuntimeError(f"{seed_label}/{wname}: prediction/frame timestamp mismatch")

    win_mask = (frame_full["timestamp"] >= wd["start"]) & (frame_full["timestamp"] <= wd["end"])
    frame = frame_full.loc[win_mask].reset_index(drop=True)
    pred = pred_full.loc[win_mask].reset_index(drop=True)

    missing = sorted(set(base_cols) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{seed_label}/{wname}: frame missing base_cols: {missing[:20]}")
    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=bool(wd["oof"]))
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, **ref_eval.ATR_KWARGS)
    atr = atr_eval._atr_pct(frame, ref_eval.ATR_KWARGS["atr_window"])

    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = (
        sidecar._predict_side_split_models(pkl["model"], x_all, side_all)
        if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    )
    mapping = pkl["selected_mapping"]
    base_margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    base_leverage = (
        sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
        if pkl["dynamic_leverage"] else np.ones(len(dec))
    )
    margin, leverage = scale_ref._scaled_margin_leverage(dec, base_margin, base_leverage, long_scale=ref_eval.LONG_SCALE, short_scale=ref_eval.SHORT_SCALE)

    # --- BTC SHORT entry veto during detected sustained uptrend: only new logic vs
    # btc_live_promotion_seed_robustness_eval_5seed_20260819.py's inner loop ---
    veto_bars = 0
    if apply_veto:
        det_mask, _n_nan = _detector_mask_for_window(frame, wd["split"], score_by_base, threshold)
        veto_target = (side_all == -1) & det_mask
        veto_bars = int(veto_target.sum())
        if veto_bars:
            dec = dec.copy()
            dec.loc[veto_target, "side"] = 0
    # --- end veto block ---

    _m, ledger = sidecar._replay_with_risk(
        frame, x, dec, loaded,
        risk_margin_fraction=margin, risk_leverage=leverage, exit_threshold=ref_eval.EXIT_THRESHOLD,
        fee=fee, slip=slip, cost_mult=ref_eval.COST_MULT,
        notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=DEVICE,
    )
    metrics = scale_ref._compound_metrics(ledger)
    return metrics, ledger, veto_bars


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("stage=prepare_frames")
    frames = parent_script._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=ref_eval.DIRECTION_LABEL_DIR,
        quality_mode="quality_label_action",
        quality_label_dir=ref_eval.QUALITY_LABEL_DIR,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    raw_by_split = {"train": frames["train_raw"], "validation": frames["val_raw"], "oos": frames["oos_raw"]}
    for split, df in raw_by_split.items():
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    with open(ref_eval.SIDECAR_PKL, "rb") as f:
        pkl = pickle.load(f)
    fee, slip = omega._load_fee_slip()

    log("stage=detector_build")
    score_by_base, threshold = build_detector()
    log(f"  threshold(BTC own 2025-Q1+Q2-only p90) = {threshold:.6f}  "
        f"(yesterday's diagnostic placeholder was 0.604167, full-2025-year p90 -- not directly comparable)")

    reference_doc = json.loads(REFERENCE_REPORT.read_text())
    reference = reference_doc["windows"]
    baseline_flip = set(reference_doc.get("sign_flip_windows", []))
    log(f"  baseline (no-veto) sign_flip_windows = {sorted(baseline_flip)}")

    log("stage=g0_no_veto (must reproduce btc_live_promotion_seed_robustness_20260819_eval/report.json exactly)")
    g0_rows = []
    g0_pass = True
    for seed_label in ref_eval.SEED_LABELS:
        for wname, wd in ref_eval.WINDOW_DEFS.items():
            frame_full = raw_by_split[wd["split"]].reset_index(drop=True)
            metrics, _ledger, veto_bars = _replay_one(seed_label, wname, wd, frame_full, pkl, score_by_base, threshold, fee, slip, apply_veto=False)
            ref = reference[seed_label][wname]
            ok = (
                abs(metrics["pnl"] - ref["pnl"]) <= G0_TOLERANCE_PP
                and abs(metrics["mdd"] - ref["mdd"]) <= G0_TOLERANCE_PP
                and int(metrics["trades"]) == int(ref["trades"])
            )
            g0_pass = g0_pass and ok and veto_bars == 0
            g0_rows.append({"seed": seed_label, "window": wname, "match": ok, "veto_bars_expected_zero": veto_bars})
            log(f"  {seed_label:20} {wname:8} match={ok} pnl={metrics['pnl']:9.3f} (ref {ref['pnl']:9.3f}) trades={metrics['trades']:3d} (ref {ref['trades']:3d})")
    if not g0_pass:
        log("G0 FAILED -- aborting before candidate_run.")
        (OUT_DIR / "report.json").write_text(json.dumps({"g0_pass": False, "g0_rows": g0_rows}, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
        return 1
    log("G0 PASSED -- wrapper reproduces the existing N=5 baseline bit-for-bit with veto off.")

    log("stage=candidate_run (veto on, all 5 seeds x 6 windows)")
    all_results: dict[str, dict[str, Any]] = {}
    veto_diag: dict[str, dict[str, int]] = {}
    for seed_label in ref_eval.SEED_LABELS:
        windows: dict[str, Any] = {}
        veto_diag[seed_label] = {}
        for wname, wd in ref_eval.WINDOW_DEFS.items():
            frame_full = raw_by_split[wd["split"]].reset_index(drop=True)
            metrics, ledger, veto_bars = _replay_one(seed_label, wname, wd, frame_full, pkl, score_by_base, threshold, fee, slip, apply_veto=True)
            windows[wname] = dict(metrics)
            veto_diag[seed_label][wname] = veto_bars
            ledger.to_csv(OUT_DIR / f"ledger_{seed_label}_{wname}_veto.csv", index=False)
            log(f"  {seed_label:20} {wname:8} pnl={metrics['pnl']:9.3f} mdd={metrics['mdd']:8.3f} trades={metrics['trades']:3d} veto_bars={veto_bars}")
        all_results[seed_label] = windows

    log("")
    log("=== veto activation (should be seed-identical -- zero free parameters) ===")
    veto_seed_identical = {}
    for wname in ref_eval.WINDOW_DEFS:
        counts = {s: veto_diag[s][wname] for s in ref_eval.SEED_LABELS}
        identical = len(set(counts.values())) == 1
        veto_seed_identical[wname] = identical
        log(f"  {wname:8} {counts}  seed_identical={identical}")

    log("")
    log("=== 5-시드 부호일치 (veto ON) ===")
    sign_flip_windows = []
    resolved_windows = []
    newly_broken_windows = []
    for wname in ref_eval.WINDOW_DEFS:
        pnls = [all_results[s][wname]["pnl"] for s in ref_eval.SEED_LABELS]
        signs = {p >= 0 for p in pnls}
        consistent = len(signs) == 1
        if not consistent:
            sign_flip_windows.append(wname)
            if wname not in baseline_flip:
                newly_broken_windows.append(wname)
        elif wname in baseline_flip:
            resolved_windows.append(wname)
        log(f"  {wname:8} {[round(p, 2) for p in pnls]} consistent={consistent}")

    log("")
    log(f"baseline_flip={sorted(baseline_flip)} -> veto_flip={sorted(sign_flip_windows)}  resolved={resolved_windows}  newly_broken={newly_broken_windows}")

    report = {
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope_note": "entry-veto only, no exit-guard (BTC has no second h48qual exit_head variant to switch to)",
        "detector_threshold_btc_2025q1q2_only_p90": threshold,
        "detector_week_bars": WEEK_BARS, "detector_percentile": DETECTOR_PERCENTILE,
        "g0_pass": g0_pass, "g0_rows": g0_rows,
        "veto_activation_by_window": veto_diag, "veto_seed_identical": veto_seed_identical,
        "windows": all_results,
        "baseline_sign_flip_windows": sorted(baseline_flip),
        "veto_sign_flip_windows": sign_flip_windows,
        "resolved_windows": resolved_windows,
        "newly_broken_windows": newly_broken_windows,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
