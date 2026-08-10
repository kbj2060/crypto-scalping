"""Final decision scorecard for the JM regime3 redesign, against the three stated criteria.

The whole redesign has so far been ranked on balanced accuracy against the ADX/slope/BB rule
label. That measures ONE thing: how faithfully a detector reproduces a rule. It says nothing about
whether the detector is steady, and nothing at all about whether it is early -- a detector that
switches to "bull" only after the rally has already happened scores identically to one that called
it beforehand. So the three criteria are scored directly, each with its own instrument:

  1. CONSISTENT   - reproducible across initialisations (std of balanced accuracy over 5 random
                    seeds) and stable out of sample (VAL -> OOS drift). A config whose lead is
                    smaller than its own seed noise, or that decays from VAL to OOS, is not
                    consistent whatever its headline number.

  2. DOESN'T FLIP - median run length, flip rate, and the WHIPSAW SHARE: the fraction of regime
                    runs shorter than 6 bars (30 min). Median run alone hides this -- a detector
                    can post a 13-bar median and still be a stripe pattern if its runs are
                    bimodal, which is exactly what the ETH m=6 chart showed.

  3. EARLY, NOT LAGGING - measured against the retrospective zigzag oracle using this project's
                    existing lag framework (scripts/audit_btc_regime_classifier_lag_20260808.py):
                      LAG PROFILE   argmax over k of agreement(detector_t, oracle_{t-k}).
                                    k > 0 means the detector is a delayed copy; k ~ 0 is a
                                    genuine nowcaster.
                      DETECTION LAG bars from each oracle pivot until the detector shows the new
                                    direction. Lower is earlier.
                      WAVE Q1       agreement with the oracle inside the FIRST fifth of each wave.
                                    A lagged copy is near zero here by construction: it has not
                                    caught up yet. This is the most direct read on "did it know
                                    before the move".
                    Plus the forward-return separation already collected, which is a pure
                    look-ahead statistic: it asks whether the CURRENT regime call predicts the
                    NEXT hour, so a lagging detector cannot score on it.

The oracle is retrospective by construction and is used only as a scoring reference, never as an
input. Everything is measured on VAL and OOS separately; OOS is reported, never selected on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM, SOURCES,
    _class_proba, _num, _read, _state_class_matrix, causal_decode_V, fit_jm, labels_for,
    quantile_matched_label_config, reference_label_quantiles, run_lengths, slice_window,
    softmax_states, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.ranked_jm_feature_selection_20260810 import load_pool, rankings_for  # noqa: E402
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

SEEDS = (7529, 481003, 26611, 903174, 155827)
ORACLE_THETA = 0.04          # 4% zigzag waves: the scale the project's oracle work settled on
WHIPSAW_MAX_BARS = 6         # runs shorter than 30 min count as whipsaw
LAG_PROFILE_K = np.arange(-288, 289, 12)   # +-24h in 1h steps
SUP = ROOT / "data/ensemble/supervised"

# Candidates: (label, kind, spec). "jm" refits from a feature-set spec; "csv" reads emitted probs.
CANDIDATES = {
    "btc": [
        ("INCUMBENT live HMM wide24", "csv",
         SUP / "btc_regime3_current_hmm_sensitive_wide24_20260708/btc_features_{y}_regime3_current_sensitive_hmm_wide24.csv"),
        ("INCUMBENT JM lam4 wide24", "csv", SUP / "btc_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv"),
        ("INCUMBENT JM lam2 wide24", "csv", SUP / "btc_regime3_current_hmm_jmlam2_20260810_{y}_maskedname.csv"),
        ("ranked m12 f_rank std k3 lpd2", "jm", ("f_rank", 12, "standard", 3, 2.0, 0.25)),
        ("ranked m8 mrmr rob k3 lpd0.1", "jm", ("mrmr", 8, "robust", 3, 0.1, 0.25)),
        ("ranked m12 f_rank std k3 lpd4", "jm", ("f_rank", 12, "standard", 3, 4.0, 0.25)),
        ("ranked m12 f_rank std k3 lpd8", "jm", ("f_rank", 12, "standard", 3, 8.0, 0.5)),
    ],
    "eth": [
        ("INCUMBENT JM lam4 wide24", "csv", SUP / "eth_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv"),
        ("ranked m6 f_rank std k3 lpd2", "jm", ("f_rank", 6, "standard", 3, 2.0, 0.5)),
        ("ranked m6 f_rank std k3 lpd8", "jm", ("f_rank", 6, "standard", 3, 8.0, 0.5)),
        ("ranked m8 mrmr rob k3 lpd0.1", "jm", ("mrmr", 8, "robust", 3, 0.1, 2.0)),
        ("ranked m12 mrmr std k3 lpd1", "jm", ("mrmr", 12, "standard", 3, 1.0, 1.0)),
        ("ranked m12 mrmr std k3 lpd4", "jm", ("mrmr", 12, "standard", 3, 4.0, 1.0)),
    ],
}


# ---------------------------------------------------------------- criterion 3 instruments
def lag_profile(direction: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    """agreement(detector_t, oracle_{t-k}); argmax k > 0 means the detector trails the oracle."""
    best_k, best_a = 0, -1.0
    curve = {}
    for k in LAG_PROFILE_K:
        shifted = np.roll(oracle, int(k))
        if k > 0:
            shifted[:k] = 0
        elif k < 0:
            shifted[k:] = 0
        m = mask & (shifted != 0) & (direction != 0)
        a = float((direction[m] == shifted[m]).mean()) if m.sum() > 500 else np.nan
        curve[int(k)] = a
        if np.isfinite(a) and a > best_a:
            best_k, best_a = int(k), a
    return {"peak_lag_bars": best_k, "peak_agreement": best_a,
            "agreement_at_zero": curve.get(0), "curve": curve}


def detection_lag(direction: np.ndarray, oracle: np.ndarray, pivots: list[int],
                  lo: int, hi: int, horizon: int = 576) -> dict:
    """Bars from each oracle pivot until the detector first shows the new wave direction."""
    lags = []
    for p in pivots:
        if not (lo <= p <= hi - horizon):
            continue
        target = oracle[min(p + 1, len(oracle) - 1)]
        if target == 0:
            continue
        hit = np.flatnonzero(direction[p: p + horizon] == target)
        if len(hit):
            lags.append(int(hit[0]))
    return {"median_bars": float(np.median(lags)) if lags else None,
            "mean_bars": round(float(np.mean(lags)), 1) if lags else None,
            "n_pivots": len(lags)}


def wave_position(pivots: list[int], n: int) -> np.ndarray:
    pos = np.full(n, np.nan)
    bounds = list(pivots) + [n - 1]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b > a:
            pos[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
    return pos


def wave_quintiles(direction: np.ndarray, oracle: np.ndarray, pos: np.ndarray,
                   mask: np.ndarray) -> dict:
    out = {}
    for q in range(5):
        m = mask & (pos >= q / 5) & (pos < (q + 1) / 5) & (oracle != 0) & (direction != 0)
        out[f"Q{q + 1}"] = float((direction[m] == oracle[m]).mean()) if m.sum() > 200 else None
    return out


def to_direction(pred: np.ndarray) -> np.ndarray:
    """bull -> +1, bear -> -1, chop -> 0 (chop bars are excluded from lag scoring)."""
    d = np.zeros(len(pred), dtype=np.int8)
    d[pred == 0] = 1
    d[pred == 1] = -1
    return d


# ---------------------------------------------------------------- prediction sources
def predictions_from_csv(asset: str, tmpl: Path, frames: dict) -> dict[str, np.ndarray] | None:
    out = {}
    for year, frame in frames.items():
        p = Path(str(tmpl).replace("{y}", year))
        if not p.exists():
            return None
        pf = pd.read_csv(p)
        cols = None
        for prefix in (f"{PREFIX_STEM}_wide24_", f"{PREFIX_STEM}_hmm_wide24_"):
            c = [f"{prefix}{n}_prob" for n in CLASSES3]
            if all(x in pf.columns for x in c):
                cols = c
                break
        if cols is None:
            return None
        pf["timestamp"] = pd.to_datetime(pf["timestamp"])
        merged = frame[["timestamp"]].merge(pf[["timestamp"] + cols], on="timestamp", how="left")
        proba = merged[cols].to_numpy()
        pred = np.full(len(frame), -1, dtype=np.int64)
        ok = np.isfinite(proba).all(axis=1)
        pred[ok] = np.argmax(proba[ok], axis=1)
        out[year] = pred
    return out


def predictions_from_jm(asset: str, spec, labels_fit: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    ranking, m, scaler, k, lpd, temp_ratio = spec
    pool = load_pool(asset, scaler)
    idx = [int(i) for i in rankings_for(asset, scaler)[ranking][:m]]
    lam = lpd * m
    mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=k, lam=lam, seed=seed, n_init=3, n_iter=10)
    V = {y: causal_decode_V(pool[f"x_{y}"][:, idx], mu, lam) for y in ("2024", "2025", "2026")}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    sp = {y: softmax_states(v, temp_ratio * spread) for y, v in V.items()}
    state_class = _state_class_matrix(sp[FIT_YEAR], labels_fit)
    return {y: np.argmax(_class_proba(sp[y], state_class), axis=1).astype(np.int64) for y in sp}


# ---------------------------------------------------------------- scoring
def score_candidate(asset: str, name: str, preds_by_seed: list[dict], frames: dict,
                    labels: dict, oracle: dict, pivots: dict, pos: dict) -> dict:
    out = {"candidate": name, "n_seeds": len(preds_by_seed), "windows": {}}
    for wname in ("val", "oos"):
        yr, start, end = EVAL_WINDOWS[wname]
        mask = slice_window(frames[yr]["timestamp"], start, end)
        close = _num(frames[yr], "close").ffill().bfill().to_numpy()
        per_seed = []
        for preds in preds_by_seed:
            pred = preds[yr]
            valid = mask & (pred >= 0)
            wm = window_metrics(pred[valid], labels[yr][valid], close[valid])
            rl = run_lengths(pred[valid])
            wm["whipsaw_share"] = float((rl < WHIPSAW_MAX_BARS).mean()) if len(rl) else 0.0
            wm["runs_per_week"] = float(len(rl) / max(valid.sum() / 2016.0, 1e-9))
            d = to_direction(pred)
            d[~mask] = 0
            lp = lag_profile(d, oracle[yr], mask)
            idxs = np.flatnonzero(mask)
            dl = detection_lag(d, oracle[yr], pivots[yr], int(idxs[0]), int(idxs[-1]))
            wq = wave_quintiles(d, oracle[yr], pos[yr], mask)
            per_seed.append({**wm, "lag_peak_bars": lp["peak_lag_bars"],
                             "lag_agreement_at_zero": lp["agreement_at_zero"],
                             "detection_lag_median": dl["median_bars"],
                             "detection_lag_n_pivots": dl["n_pivots"],
                             "wave_Q1": wq["Q1"], "wave_Q5": wq["Q5"]})
        agg = {}
        for key in per_seed[0]:
            vals = [s[key] for s in per_seed if s[key] is not None and not isinstance(s[key], dict)]
            if not vals or isinstance(vals[0], dict):
                continue
            arr = np.asarray(vals, dtype=float)
            agg[key] = float(arr.mean())
            if len(arr) > 1:
                agg[f"{key}_std"] = float(arr.std(ddof=1))
        out["windows"][wname] = agg
    v, o = out["windows"]["val"], out["windows"]["oos"]
    out["consistency_seed_std"] = v.get("balanced_accuracy_std", 0.0)
    out["consistency_val_oos_drift"] = o["balanced_accuracy"] - v["balanced_accuracy"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    report = {}
    for asset in args.assets:
        frames = {y: _read(p) for y, p in SOURCES[asset].items()}
        ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
        cfg = (dict(LABEL_CONFIGS[LABEL_MODE]) if asset == "eth"
               else quantile_matched_label_config(frames[FIT_YEAR], ref_q))
        labels = {y: labels_for(f, cfg) for y, f in frames.items()}
        oracle, pivots, pos = {}, {}, {}
        for y, f in frames.items():
            close = _num(f, "close").ffill().bfill().to_numpy()
            d, pv = zigzag_oracle(close, ORACLE_THETA)
            oracle[y], pivots[y] = d, pv
            pos[y] = wave_position(pv, len(close))
        print(f"\n=== {asset.upper()}   oracle theta={ORACLE_THETA:.0%}: "
              + ", ".join(f"{y}={len(pivots[y])} pivots" for y in pivots))

        rows = []
        for name, kind, spec in CANDIDATES[asset]:
            if kind == "csv":
                preds = predictions_from_csv(asset, spec, frames)
                if preds is None:
                    print(f"  [skip] {name}: missing or unreadable source")
                    continue
                preds_by_seed = [preds]
            else:
                preds_by_seed = [predictions_from_jm(asset, spec, labels[FIT_YEAR], s) for s in seeds]
            rows.append(score_candidate(asset, name, preds_by_seed, frames, labels,
                                        oracle, pivots, pos))
            r = rows[-1]
            v, o = r["windows"]["val"], r["windows"]["oos"]
            print(f"  {name:<32} VAL bal={v['balanced_accuracy']:.4f} | OOS bal={o['balanced_accuracy']:.4f} "
                  f"run={o['median_run_bars']:>4.0f} whip={o['whipsaw_share']:.2f} "
                  f"lagpeak={o['lag_peak_bars']:>5.0f} detlag={o.get('detection_lag_median', float('nan')):>6.1f} "
                  f"Q1={o.get('wave_Q1', float('nan')):.3f} sep_t={o['economic_separation_tstat']:+.2f}")
        report[asset] = rows

    path = OUT_DIR / "decision_scorecard.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nscorecard -> {path}")


if __name__ == "__main__":
    main()
