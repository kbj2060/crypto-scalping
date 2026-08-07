#!/usr/bin/env python3
"""Build the extended (2026-01-01..06-30) evaluation frame for the M7-FREE Omega6 L2 retrain
(scripts/train_eval_omega6_nom7_tabm_3head_20260704.py). Unlike
scripts/build_extended_eval_frame_20260704.py, this does NOT run SevenModelEnsemble at all --
only the base 142 features + 3 regime3 overlays (wide24/cmamba/stability-risk, all already
extended and reproducibility-verified) + NF forecaster columns (PatchTST/TiDE/DLinear, a
separate still-functioning pipeline in ensemble/ensemble_router.py) + sig_whale/sig_oi_divergence/
sig_ai_squeeze (from strategies/elite_builder.py's EliteSignals, unchanged/reproducible,
independent of SevenModelEnsemble).
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/extended_eval_frame_nom7_20260704"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
WIDE24 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
CMAMBA = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv"
RISK = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
EXISTING_EVAL = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"

CONSISTENCY_ATOL = 1e-3


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
            raise RuntimeError(f"NF forecaster unavailable: {name}")
        print(f"NF batch: {name}", flush=True)
        res = forecaster.predict_batch(out, chunk_size=512)
        if res.empty:
            raise RuntimeError(f"NF forecaster produced empty output: {name}")
        for col in res.columns:
            out[col] = res[col].to_numpy()
    out["pred_patchtst"] = np.clip(pd.to_numeric(out["ai_dir_edge"], errors="raise"), -1.0, 1.0)
    out["conf_patchtst"] = np.clip(pd.to_numeric(out["patchtst_regime_sim"], errors="raise"), 0.0, 1.0)
    return out


def add_sig_features(frame: pd.DataFrame) -> pd.DataFrame:
    """sig_whale/sig_oi_divergence/sig_ai_squeeze from EliteSignals -- independent of
    SevenModelEnsemble, needs the same prereq derivation + synthetic-alpha/regime/volatility
    precompute steps augment_m7_dataset.py runs before calling EliteSignals row-by-row."""
    import augment_m7_dataset as m7mod
    from strategies.elite_builder import EliteSignals, compute_new_elite_signals, compute_regime, compute_synthetic_alphas, compute_volatility_models, row_to_market_row
    from features.high_order_state import add_high_order_state_features

    work = m7mod._derive_prereq_features(frame)
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
    keys = ["sig_whale", "sig_oi_divergence", "sig_ai_squeeze"]
    for k in keys:
        work[k] = 0.0
    records = work.to_dict("records")
    for i in range(len(records)):
        cur = row_to_market_row(records[i])
        prev = row_to_market_row(records[i - 1]) if i > 0 else cur
        sigs = elite.compute_all(current=cur, prev=prev, smf_std=float(smf_std.iloc[i]))
        for k in keys:
            if k in sigs:
                work.at[i, k] = float(sigs[k])
        if i % 10000 == 0:
            print(f"sig signals {i}/{len(records)}", flush=True)

    out = frame.copy()
    for k in keys:
        out[k] = work[k].to_numpy()
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_base()
    print(f"base+overlays: {len(frame)} rows, {len(frame.columns)} cols", flush=True)
    frame = add_nf_features(frame)
    print(f"after NF: {len(frame.columns)} cols", flush=True)
    frame = add_sig_features(frame)
    print(f"after sig_*: {len(frame.columns)} cols", flush=True)

    existing = pd.read_csv(EXISTING_EVAL, low_memory=False)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    generated_cols = [c for c in frame.columns if c not in ("timestamp",) and c in existing.columns and not c.startswith("m7_")]
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
    print(f"consistency: {len(diffs)} columns, {len(merged)} overlap rows, {len(bad)} exceed tol", flush=True)
    for c, d in list(bad.items())[:20]:
        print(f"  DRIFT {c}: rel max diff {d:.4g}", flush=True)

    frame.to_parquet(OUT_DIR / "frame.parquet", index=False)
    report = {
        "rows": int(len(frame)),
        "range": [str(frame["timestamp"].min()), str(frame["timestamp"].max())],
        "consistency_pass": consistency_pass,
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
