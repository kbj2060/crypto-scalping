#!/usr/bin/env python3
"""RESEARCH ONLY -- even-cheaper pre-training check for the "cash-sleeve EV-HGB" candidate
(docs/experiments/eth_candidate_cash_sleeve_ev_hgb_20260816.md), one step before any HGB training is
justified: Spearman rank correlation (IC) between existing CAUSAL features (h48qual/zig075's own
base+wide24 feature panel) and the ORACLE fallback-trade edge already computed by
research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816.py, for every PRIMARY-CASH bar in VAL and
OOS-Q1.

Does NOT train any model (no HGB, no purged CV). Does NOT touch trading_bot.py / any live/production
code / runtime_config.py / .env. Read-only research: loads the same aligned per-window feature frame
the cheap_gate script itself built (via the SAME unmodified loaders,
eth_omega461_multiwindow_confirmation_gate_20260814.load_all_windows +
research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.prepare_regime_aware_components /
build_detector), joins it against the two already-published oracle CSVs by the exact same bar index
`i` those CSVs were written with (verified by a timestamp-equality assertion below, not assumed), and
computes:
  1. Spearman IC of each candidate feature vs. max_net = max(long_net, short_net) and vs.
     net_diff = long_net - short_net, separately per window.
  2. Spearman IC of each candidate feature vs. raw ETH close price over the same bars (this repo's own
     established "raw feature price-trend contamination" check --
     see docs memory feedback_raw_feature_price_trend_contamination -- flags features whose apparent
     edge-IC is really just co-drift with price).
  3. A bootstrap 95% CI on each real IC (resampling bar-pairs with replacement) and a label-shuffle
     null distribution, both to gauge whether the real IC is distinguishable from noise.

Candidate features are a hand-picked subset of the h48qual/zig075 base+wide24 panel
(data/splits/year_oos/training_features_2025.csv + 2026_rebuilt.csv, overlaid with
regime3_current_sensitive_wide24 columns) -- the exact panel `train_eval_omega1_2_tabm_3head_20260603.py`
sources its own feature_cols from (via train_eval_omega1_2_tabm_diffusion_risk_20260603._load_omega_
frames / _numeric_feature_cols), restricted here to momentum/trend, volatility, and regime/router-prob
groups per the orchestrating session's explicit request (not an exhaustive scan of the full ~140-185
column panel -- this is a cheap screen, not a feature-selection study).

fresh_forward_bar_by_bar=true (feature values used are each bar's own already-computed causal feature,
matching the PRIMARY pipeline's real-time contract; the oracle targets remain the same hindsight
oracle values the cheap_gate script itself computed and labeled as unachievable). No retraining, no
GPU (DEVICE=cpu), conda env quant_ai.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402

CHEAP_GATE_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816"
DEVICE = guard.DEVICE

WINDOWS = ("val", "oos_q1")

# Hand-picked subset of the base+wide24 feature panel, per the orchestrating session's explicit
# groups (momentum/trend, volatility, regime/router prob) -- NOT an exhaustive scan.
CANDIDATE_FEATURES: dict[str, list[str]] = {
    "momentum_trend": [
        "dual_momentum", "mtf_trend_1h", "mtf_trend_4h", "mean_reversion_z", "breakout_strength",
        "macd_hist", "hma_slope", "turtle_signal", "kalman_velocity", "rsi",
    ],
    "volatility": [
        "garman_klass_vol", "realized_vol_ratio", "rogers_satchell_vol", "parkinson_vol",
        "atr_pct_rank_288", "garch_vol_z", "volatility_z", "bb_width_z", "compression_score",
    ],
    "regime_router": [
        "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
        "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
        "regime3_current_sensitive_wide24_entropy", "regime3_current_sensitive_wide24_margin",
    ],
}
ALL_FEATURES = [(feat, group) for group, feats in CANDIDATE_FEATURES.items() for feat in feats]

PRICE_CONTAM_THRESHOLD = 0.5
N_SHUFFLES = 200
N_BOOTSTRAP = 500
RNG_SEED = 20260816


def log(msg: str) -> None:
    print(f"[ic_check] {msg}", flush=True)


def spearman_ic(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 10:
        return float("nan"), n
    r, _ = spearmanr(x[mask], y[mask])
    return float(r), n


def bootstrap_ci(x: np.ndarray, y: np.ndarray, *, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    m = len(x)
    if m < 30:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n, dtype=np.float64)
    for b in range(n):
        idx = rng.integers(0, m, size=m)
        r, _ = spearmanr(x[idx], y[idx])
        boots[b] = r if np.isfinite(r) else 0.0
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.std(boots))


def shuffle_null_std(x: np.ndarray, y: np.ndarray, *, n: int, rng: np.random.Generator) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return float("nan"), float("nan")
    out = np.empty(n, dtype=np.float64)
    for s in range(n):
        yp = rng.permutation(y)
        r, _ = spearmanr(x, yp)
        out[s] = r if np.isfinite(r) else 0.0
    return float(np.mean(out)), float(np.std(out))


def verdict_for(ic: float, ci_lo: float, ci_hi: float, null_std: float, price_ic: float) -> str:
    if not np.isfinite(ic):
        return "insufficient_data"
    if np.isfinite(price_ic) and abs(price_ic) > PRICE_CONTAM_THRESHOLD:
        return "price_trend_contaminated"
    noise_like = (np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= 0.0 <= ci_hi)
    below_null = np.isfinite(null_std) and abs(ic) < 2.0 * null_std
    if noise_like or below_null:
        return "indistinguishable_from_noise"
    if abs(ic) < 0.02:
        return "negligible"
    if abs(ic) < 0.05:
        return "weak"
    if abs(ic) < 0.10:
        return "moderate"
    return "strong"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    log("=== stage=load_windows (same unmodified loader the cheap_gate script used) ===")
    windows = gate.load_all_windows()
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    log(f"  detector threshold(p90)={threshold:.10f}")

    all_rows: list[dict[str, Any]] = []
    per_window_frames: dict[str, pd.DataFrame] = {}
    for wname in WINDOWS:
        log(f"=== window={wname} ===")
        aligned_frame, _components, _diag = guard.prepare_regime_aware_components(
            wname, windows, score_by_base, threshold, OUT_DIR, DEVICE
        )
        oracle = pd.read_csv(CHEAP_GATE_DIR / f"cash_sleeve_oracle_bars_{wname}.csv", parse_dates=["timestamp"])
        idx = oracle["i"].to_numpy(dtype=np.int64)
        af_ts = aligned_frame["timestamp"].to_numpy()[idx]
        oc_ts = oracle["timestamp"].to_numpy()
        if not np.array_equal(af_ts, oc_ts):
            n_bad = int((af_ts != oc_ts).sum())
            raise RuntimeError(f"{wname}: aligned_frame/oracle timestamp mismatch at i-index join ({n_bad} rows differ)")
        log(f"  join sanity OK: {len(oracle)} CASH bars, i-index join verified bit-exact against aligned_frame timestamps")

        feat_frame = aligned_frame.iloc[idx].reset_index(drop=True)
        missing = [f for f, _g in ALL_FEATURES if f not in feat_frame.columns]
        if missing:
            raise RuntimeError(f"{wname}: candidate features missing from aligned_frame: {missing}")

        close = pd.to_numeric(feat_frame["close"], errors="raise").to_numpy(dtype=np.float64)
        max_net = oracle[["long_net", "short_net"]].max(axis=1).to_numpy(dtype=np.float64)
        net_diff = (oracle["long_net"] - oracle["short_net"]).to_numpy(dtype=np.float64)
        targets = {"max_net": max_net, "net_diff": net_diff}

        joined = feat_frame[["timestamp", "close"] + [f for f, _g in ALL_FEATURES]].copy()
        joined["max_net"] = max_net
        joined["net_diff"] = net_diff
        per_window_frames[wname] = joined

        for feat, group in ALL_FEATURES:
            fvals = pd.to_numeric(feat_frame[feat], errors="coerce").to_numpy(dtype=np.float64)
            price_ic, price_n = spearman_ic(fvals, close)
            for tname, tvals in targets.items():
                ic, n = spearman_ic(fvals, tvals)
                lo, hi, boot_std = bootstrap_ci(fvals, tvals, n=N_BOOTSTRAP, rng=rng)
                null_mean, null_std = shuffle_null_std(fvals, tvals, n=N_SHUFFLES, rng=rng)
                verdict = verdict_for(ic, lo, hi, null_std, price_ic)
                all_rows.append(
                    {
                        "window": wname,
                        "feature_group": group,
                        "feature": feat,
                        "target": tname,
                        "n_bars": n,
                        "ic": ic,
                        "ic_boot_ci_lo": lo,
                        "ic_boot_ci_hi": hi,
                        "ic_boot_std": boot_std,
                        "price_contam_ic": price_ic,
                        "shuffle_null_mean": null_mean,
                        "shuffle_null_std": null_std,
                        "verdict": verdict,
                    }
                )
        log(f"  {wname}: {len(ALL_FEATURES)} features x 2 targets scored")

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "ic_results.csv", index=False)
    for wname, joined in per_window_frames.items():
        joined.to_csv(OUT_DIR / f"feature_target_join_{wname}.csv", index=False)

    # VAL/OOS-Q1 consistency check: same feature+target, sign-consistent AND both windows NOT
    # "indistinguishable_from_noise"/"price_trend_contaminated"/"insufficient_data"/"negligible".
    consistency_rows: list[dict[str, Any]] = []
    clean_verdicts = {"weak", "moderate", "strong"}
    for (feat, target), grp in results.groupby(["feature", "target"]):
        by_w = {r["window"]: r for _, r in grp.iterrows()}
        if set(by_w.keys()) != set(WINDOWS):
            continue
        val_r, oos_r = by_w["val"], by_w["oos_q1"]
        both_clean = (val_r["verdict"] in clean_verdicts) and (oos_r["verdict"] in clean_verdicts)
        same_sign = np.isfinite(val_r["ic"]) and np.isfinite(oos_r["ic"]) and (np.sign(val_r["ic"]) == np.sign(oos_r["ic"])) and val_r["ic"] != 0
        consistency_rows.append(
            {
                "feature": feat,
                "target": target,
                "val_ic": val_r["ic"],
                "val_verdict": val_r["verdict"],
                "oos_q1_ic": oos_r["ic"],
                "oos_q1_verdict": oos_r["verdict"],
                "same_sign_both_windows": bool(same_sign),
                "both_windows_clean_and_above_noise": bool(both_clean),
                "promising": bool(both_clean and same_sign),
            }
        )
    consistency = pd.DataFrame(consistency_rows).sort_values(["promising", "target", "feature"], ascending=[False, True, True])
    consistency.to_csv(OUT_DIR / "val_oos_consistency.csv", index=False)

    promising = consistency[consistency["promising"]]
    verdict_summary = {
        "n_feature_target_pairs_tested": int(len(consistency)),
        "n_promising_pairs": int(len(promising)),
        "promising_pairs": promising.to_dict(orient="records"),
        "overall_verdict": "SIGNAL_CANDIDATE_FOUND" if len(promising) > 0 else "DECISIVELY_NEGATIVE_NO_CLEAN_CONSISTENT_IC",
    }
    (OUT_DIR / "ic_check_summary.json").write_text(json.dumps(verdict_summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"=== overall_verdict === {verdict_summary['overall_verdict']} (n_promising_pairs={len(promising)}/{len(consistency)})")
    log(f"outputs: {OUT_DIR / 'ic_results.csv'}, {OUT_DIR / 'val_oos_consistency.csv'}, {OUT_DIR / 'ic_check_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
