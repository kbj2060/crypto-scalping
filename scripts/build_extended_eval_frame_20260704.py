#!/usr/bin/env python3
"""Build the extended (2026 Jan-Jun) evaluation frame the frozen Omega6 v2 winner's L2 needs,
WITHOUT rebuilding the full alpha7-candidates lineage. The L2 TabM bundles need 172 non-position
input columns; after the base-feature extension and the three regime3 overlay extensions
(2026-07-04 session), exactly 66 columns remained uncovered -- all model-prediction features:

  - m7_* (44) + sig_whale/sig_oi_divergence/sig_ai_squeeze (3): produced by
    ensemble.seven_model_ensemble.SevenModelEnsemble + strategies.elite_builder enrichment,
    exactly as pipeline/augment_m7_dataset.py does (same functions imported and reused).
  - ai_* / patchtst_median / patchtst_regime_sim / tide_vol_* / dlinear_smf_* (17): produced by
    ensemble.ensemble_router's PatchTST/TiDE/DLinear forecasters' predict_batch (TimesNet not
    needed -- none of its output columns are L2 inputs).
  - pred_patchtst / conf_patchtst (2): derived from PatchTST outputs exactly as trading_bot.py
    line ~2152 does live: pred=clip(ai_dir_edge,-1,1), conf=clip(patchtst_regime_sim,0,1).

CRITICAL consistency gate: every generated column is compared against the existing alpha7 eval
candidates CSV (tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/
trade_candidates_2026_alpha6_current_tail111_exact.csv) on the overlapping Jan-Feb range. If the
saved model artifacts on disk have drifted since that file was built (git status shows several
M7 pickles modified), the diffs will be large and the frame must NOT be used to score the frozen
model (out-of-distribution inputs) -- the script reports per-column diffs and writes the frame
plus a consistency report either way, but marks `consistency_pass` false above tolerance.

Output: tmp/causal_regen_20260516/extended_eval_frame_20260704/frame.parquet + report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT_DIR = ROOT / "tmp/causal_regen_20260516/extended_eval_frame_20260704"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
WIDE24 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
CMAMBA = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv"
RISK = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
EXISTING_EVAL = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
PRIMARY_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt"

CONSISTENCY_ATOL = 1e-4  # generated-vs-original tolerance on overlap; prediction pipelines have float noise


def load_base() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2026, parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for path in (WIDE24, CMAMBA, RISK):
        overlay = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        cols = [c for c in overlay.columns if c != "timestamp"]
        frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    return frame


def add_nf_features(frame: pd.DataFrame) -> pd.DataFrame:
    from ensemble.ensemble_router import PatchTSTForecaster, TiDEVolatilityForecaster, DLinearOFIForecaster

    out = frame.copy()
    for name, forecaster in (
        ("PatchTST", PatchTSTForecaster()),
        ("TiDE", TiDEVolatilityForecaster()),
        ("DLinear", DLinearOFIForecaster()),
    ):
        if not forecaster.available:
            raise RuntimeError(f"NF forecaster unavailable: {name} (check data/nf_* model dirs)")
        print(f"NF batch: {name}", flush=True)
        res = forecaster.predict_batch(out, chunk_size=512)
        if res.empty:
            raise RuntimeError(f"NF forecaster produced empty output: {name}")
        for col in res.columns:
            out[col] = res[col].to_numpy()
    # trading_bot.py live-derivation contract (line ~2152)
    out["pred_patchtst"] = np.clip(pd.to_numeric(out["ai_dir_edge"], errors="raise"), -1.0, 1.0)
    out["conf_patchtst"] = np.clip(pd.to_numeric(out["patchtst_regime_sim"], errors="raise"), 0.0, 1.0)
    return out


def add_m7_features(frame: pd.DataFrame) -> pd.DataFrame:
    from ensemble.seven_model_ensemble import SevenModelEnsemble
    from features.high_order_state import add_high_order_state_features
    from strategies.elite_builder import (
        EliteSignals,
        compute_new_elite_signals,
        compute_regime,
        compute_synthetic_alphas,
        compute_volatility_models,
        row_to_market_row,
    )
    import augment_m7_dataset as m7mod

    work = frame.copy()
    work = m7mod._derive_prereq_features(work)
    work = compute_synthetic_alphas(work)
    work = compute_regime(work)
    work = compute_volatility_models(work)
    work = compute_new_elite_signals(work)
    work = add_high_order_state_features(work)

    elite = EliteSignals()
    if "smart_money_flow" in work.columns:
        smf_std = (
            work["smart_money_flow"].rolling(window=576, min_periods=10).std()
            .fillna(work["smart_money_flow"].expanding(min_periods=1).std())
            .fillna(1.0)
        )
    else:
        smf_std = pd.Series(1.0, index=work.index)
    elite_keys = [
        "sig_whale", "sig_oi_divergence", "sig_ai_squeeze", "sig_orderblock",
        "sig_liq_squeeze", "sig_net_taker", "sig_hurst_ofi",
        "sig_funding_cascade", "sig_multifractal", "sig_cluster_fib",
        "sig_top_trader_squeeze", "sig_btc_corr_breakout",
        "sig_garch_regime", "sig_ou_mean_rev", "sig_jump_rebound", "sig_evt_tail",
    ]
    for k in elite_keys:
        if k not in work.columns:
            work[k] = 0.0
    records = work.to_dict("records")
    for i in range(len(records)):
        cur = row_to_market_row(records[i])
        prev = row_to_market_row(records[i - 1]) if i > 0 else cur
        sigs = elite.compute_all(current=cur, prev=prev, smf_std=float(smf_std.iloc[i]))
        for k in elite_keys:
            if k in sigs:
                work.at[i, k] = float(sigs[k])
        if i % 10000 == 0:
            print(f"elite signals {i}/{len(records)}", flush=True)

    ensemble = SevenModelEnsemble()
    m7 = ensemble.predict_batch(work)
    print(f"m7 columns generated: {len(m7.columns)}", flush=True)

    out = frame.copy()
    for col in m7.columns:
        out[col] = m7[col].to_numpy()
    for k in ("sig_whale", "sig_oi_divergence", "sig_ai_squeeze"):
        out[k] = work[k].to_numpy()
    return out


def main() -> int:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_base()
    print(f"base+overlays: {len(frame)} rows, {len(frame.columns)} cols", flush=True)

    frame = add_nf_features(frame)
    print(f"after NF: {len(frame.columns)} cols", flush=True)
    frame = add_m7_features(frame)
    print(f"after M7: {len(frame.columns)} cols", flush=True)

    bundle = torch.load(PRIMARY_BUNDLE, map_location="cpu", weights_only=False)
    pos_cols = set(bundle["pos_cols"])
    input_cols = set(dict(bundle["models"])["bull"]["input_columns"])
    needed = sorted(input_cols - pos_cols)
    still_missing = [c for c in needed if c not in frame.columns]
    if still_missing:
        raise RuntimeError(f"extended frame still missing L2 inputs: {still_missing}")
    print("all 172 non-position L2 input columns present", flush=True)

    existing = pd.read_csv(EXISTING_EVAL, low_memory=False)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    generated_cols = [c for c in needed if c in existing.columns]
    merged = existing[["timestamp", *generated_cols]].merge(
        frame[["timestamp", *generated_cols]], on="timestamp", suffixes=("_old", "_new"), how="inner"
    )
    diffs = {}
    for c in generated_cols:
        old_v = pd.to_numeric(merged[f"{c}_old"], errors="coerce")
        new_v = pd.to_numeric(merged[f"{c}_new"], errors="coerce")
        both = old_v.notna() & new_v.notna()
        if both.any():
            scale = max(float(old_v[both].abs().median()), 1e-6)
            diffs[c] = float(np.max(np.abs(old_v[both] - new_v[both])) / scale)
    bad = {c: d for c, d in sorted(diffs.items(), key=lambda kv: -kv[1]) if d > CONSISTENCY_ATOL}
    consistency_pass = len(bad) == 0
    print(f"consistency: {len(diffs)} columns compared on {len(merged)} overlap rows; {len(bad)} exceed tol", flush=True)
    for c, d in list(bad.items())[:20]:
        print(f"  DRIFT {c}: rel max diff {d:.4g}", flush=True)

    frame.to_parquet(OUT_DIR / "frame.parquet", index=False)
    report = {
        "rows": int(len(frame)),
        "range": [str(frame["timestamp"].min()), str(frame["timestamp"].max())],
        "l2_inputs_covered": True,
        "consistency_pass": consistency_pass,
        "consistency_tolerance_rel": CONSISTENCY_ATOL,
        "overlap_rows": int(len(merged)),
        "columns_compared": len(diffs),
        "columns_exceeding_tolerance": bad,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "columns_exceeding_tolerance"}, indent=2), flush=True)
    print(f"consistency_pass={consistency_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
