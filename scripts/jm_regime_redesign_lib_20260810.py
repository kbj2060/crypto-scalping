"""Shared library for the JM-only regime3 classifier redesign (2026-08-10).

Why this exists
---------------
Both live-candidate JM regime detectors (BTC/ETH regime3-current) were built by taking the
12-state sticky-HMM pipeline and swapping only the model: they still consume `wide24`, a feature
panel designed for a Gaussian HMM, and they inherited `lambda=4` from a completely different
panel (the 2026-08-08 zigzag-oracle detector zoo). That is the wrong shape of input for a Jump
Model, for three concrete reasons:

  1. A JM is k-means plus a switching penalty. Its state assignment is plain squared Euclidean
     distance in the scaled feature space, so EVERY dimension carries equal weight. `wide24`
     holds many near-duplicate trend columns (rsi, macd_hist, hma_slope, mtf_trend_1h,
     mtf_trend_4h, state7_trend_score, state7_directional_return_48, ...), so the effective
     metric is silently over-weighted toward "trend" purely by column count. A Gaussian HMM with
     a full covariance does not care about that redundancy; a JM does.
  2. `state12` columns are tanh-squashed and then RobustScaler'd, i.e. transformed twice, which
     compresses the variance of exactly the bars (tail moves) that mark a regime change.
  3. The per-bar cost `||x - mu||^2` grows with the feature count d, so a fixed lambda means
     something different for a 6-dim panel than for a 24-dim one. Comparing panels at a fixed
     absolute lambda compares nothing. This module therefore parameterises the penalty as
     `lambda = lambda_per_dim * d`, which is the only way panel comparison is meaningful.

So the redesign searches, per asset and independently, over JM-native feature panels x scaler x
K x lambda_per_dim. Nothing about the OUTPUT contract changes: the same
`regime3_current_sensitive_wide24_*` column names, the same 3 classes, the same ADX/slope/BB rule
label used to fit the state->class matrix, so a winner is a drop-in replacement downstream.

Split discipline (project Fresh-Forward rule)
--------------------------------------------
  FIT     : the 2024 file only (JM centroids, scaler, medians, state->class matrix).
  SELECT  : 2025-09-01..2025-12-31, the project's validation window.
  REPORT  : 2026-01-01..2026-03-31, the project's OOS window -- never touched by selection.
Each calendar-year file is causally decoded from its own first bar, so the VAL/OOS slices always
sit on 8 / 0 months of DP warm-up respectively; windows are sliced AFTER decoding, never before.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_regime3_current_hmm_wide24_20260529 import (  # noqa: E402
    CLASSES3, FEATURE_SETS, LABEL_CONFIGS, _class_proba, _labels, _num,
    _state_class_matrix, _with_features,
)
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402

LABEL_MODE = "balancedish_adx16_slope15_bb012"
PREFIX_STEM = "regime3_current_sensitive"

# 5m bars: 72 = 6h, 288 = 24h, 864 = 72h. The halflife ladder used by the JM regime literature
# (Nystrup/Lindstrom/Madsen 2020-21; Shu/Yu/Kolm 2024, arXiv:2402.05272) rescaled to 5m.
HALFLIVES = (72, 288, 864)
FWD_BARS = 12  # 1h forward return, the horizon used for state ordering + economic separation

SOURCES = {
    "btc": {
        "2024": ROOT / "data/splits/year_oos/btc_features_2024.csv",
        "2025": ROOT / "data/splits/year_oos/btc_features_2025.csv",
        "2026": ROOT / "data/splits/year_oos/btc_features_2026.csv",
    },
    "eth": {
        "2024": ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
        "2025": ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
        "2026": ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
    },
}
FIT_YEAR = "2024"
EVAL_WINDOWS = {
    # name -> (year file, start, end inclusive). "val" is the only window selection may read.
    "val": ("2025", "2025-09-01", "2025-12-31"),
    "oos": ("2026", "2026-01-01", "2026-03-31"),
    "full_2025": ("2025", None, None),
    "full_2026": ("2026", None, None),
}
SELECTION_WINDOW = "val"

# Gates applied before a config may be declared a winner. Both encode lessons already paid for on
# this project: a detector that flickers is unusable downstream regardless of its agreement rate
# (project-jump-model-regime-detector-20260808), and 3-state regime work here has repeatedly
# produced configs that win on accuracy by collapsing a class to near-zero coverage
# (project-regime-3class-literature-validation-20260809).
MIN_MEDIAN_RUN_BARS = 12   # >= 1h of persistence
MIN_CLASS_COVERAGE = 0.05  # every one of bull/bear/chop present on >= 5% of VAL bars


# --------------------------------------------------------------------------------------------
# feature panels
# --------------------------------------------------------------------------------------------
def _ewm_panel(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Causal EWM return / downside-deviation / Sortino ladder -- the JM literature core."""
    close = _num(frame, "close").ffill().bfill()
    logc = np.log(close.clip(lower=1e-12))
    lr = logc.diff().fillna(0.0)
    out: dict[str, np.ndarray] = {}
    for hl in HALFLIVES:
        mean = lr.ewm(halflife=hl, adjust=False).mean()
        dd = np.sqrt((lr.clip(upper=0.0) ** 2).ewm(halflife=hl, adjust=False).mean())
        out[f"jm_ret_hl{hl}"] = mean.to_numpy()
        out[f"jm_dd_hl{hl}"] = dd.to_numpy()
        out[f"jm_sortino_hl{hl}"] = (mean / (dd + 1e-9)).to_numpy()
    return out


def _perp_panel(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Perp-market state that spot-only JM literature has no analogue for."""
    hl = HALFLIVES[1]  # 24h
    out = {}
    for col, name in (("funding_pressure", "jm_funding"), ("oi_change_rate", "jm_oi"),
                      ("net_taker_ratio", "jm_taker")):
        s = _num(frame, col).fillna(0.0)
        out[f"{name}_hl{hl}"] = s.ewm(halflife=hl, adjust=False).mean().to_numpy()
    return out


PANEL_DIMS = {
    "jm6": 6,
    "jm9": 9,
    "jm9_perp": 12,
    "state12": 12,
    "wide24": 24,
    "wide24_decorr": None,  # data-derived on the 2024 fit window
}


def build_panel(frame: pd.DataFrame, panel: str, decorr_cols: list[str] | None = None):
    """Return (feature DataFrame, column names) for `panel`, computed causally from `frame`.

    `wide24_decorr` needs the column list chosen on the fit window; pass it via `decorr_cols`
    for every non-fit window so the panel stays frozen.
    """
    if panel in ("wide24", "state12"):
        cols = list(FEATURE_SETS[panel])
        work = _with_features(frame, cols)
        return work[cols].copy(), cols
    if panel == "wide24_decorr":
        cols = list(FEATURE_SETS["wide24"])
        work = _with_features(frame, cols)
        if decorr_cols is None:
            raise ValueError("wide24_decorr requires decorr_cols (choose them on the fit window)")
        return work[decorr_cols].copy(), list(decorr_cols)
    if panel not in ("jm6", "jm9", "jm9_perp"):
        raise ValueError(f"unknown panel: {panel}")
    feats = _ewm_panel(frame)
    if panel == "jm6":
        feats = {k: v for k, v in feats.items() if "sortino" not in k}
    if panel == "jm9_perp":
        feats.update(_perp_panel(frame))
    cols = list(feats.keys())
    return pd.DataFrame(feats, index=frame.index)[cols], cols


def choose_decorr_cols(fit_frame: pd.DataFrame, max_abs_rho: float = 0.7) -> list[str]:
    """Greedy |Pearson rho| prune of wide24 on the FIT window only.

    Keeps columns in the panel's declared order and drops any column correlated above the
    threshold with an already-kept one -- deterministic, and it isolates "is wide24's problem
    just redundancy?" from "is wide24's problem the wrong feature family?".
    """
    cols = list(FEATURE_SETS["wide24"])
    work = _with_features(fit_frame, cols)
    x = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    corr = x.corr().abs().fillna(0.0)
    kept: list[str] = []
    for c in cols:
        if all(corr.loc[c, k] <= max_abs_rho for k in kept):
            kept.append(c)
    return kept


# --------------------------------------------------------------------------------------------
# label bases
# --------------------------------------------------------------------------------------------
# The ADX/slope/BB rule label defines what "bull/bear/chop" MEAN to the downstream sidecar, and
# its thresholds (adx>=16 trend, adx<12 weak, |ema_slope|>1.5e-4, bb_width<0.012 forces chop) were
# calibrated on ETH and then copied to BTC and SOL unchanged. They do not port: measured on the
# 2024 fit windows the same rule yields bull/bear/chop = .124/.124/.752 on ETH but .091/.089/.820
# on BTC, because BTC's bb_width distribution sits lower (median .0068 vs .0082) so the tight-BB
# chop override fires far more often. A detector scored against a label that is 82% one class is
# being graded on a nearly degenerate target, which is a per-asset defect worth separating from
# the feature/hyperparameter question. So every config is scored under BOTH bases:
#   "frozen"    - the live thresholds verbatim; a winner here is contract-identical.
#   "qmatched"  - each threshold moved to the quantile position it occupies in ETH's own 2024
#                 distribution, evaluated against the asset's own 2024 distribution. One idea,
#                 no free knobs: it makes the rule mean the same thing on every asset. On ETH it
#                 reduces to the frozen label by construction (ETH is the reference).
LABEL_REF_ASSET = "eth"
LABEL_BASES = ("frozen", "qmatched")


def _label_inputs(frame: pd.DataFrame):
    from scripts.experiment_regime3_current_hmm_wide24_20260529 import _adx

    close = _num(frame, "close")
    ema21 = close.ewm(span=21, adjust=False).mean()
    slope = ((ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(_num(frame, "high"), _num(frame, "low"), close)
    bb = _num(frame, "bb_width", np.nan)
    if bb.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    return adx.fillna(0.0), slope, bb.fillna(0.0)


def reference_label_quantiles(ref_fit_frame: pd.DataFrame) -> dict[str, float]:
    """Where the live thresholds sit in the REFERENCE asset's own 2024 distributions."""
    cfg = LABEL_CONFIGS[LABEL_MODE]
    adx, slope, bb = _label_inputs(ref_fit_frame)
    return {
        "trend_adx_min": float((adx <= cfg["trend_adx_min"]).mean()),
        "weak_adx_max": float((adx <= cfg["weak_adx_max"]).mean()),
        "slope_min": float((slope.abs() <= cfg["slope_min"]).mean()),
        "tight_bb_max": float((bb <= cfg["tight_bb_max"]).mean()),
    }


def quantile_matched_label_config(fit_frame: pd.DataFrame, ref_q: dict[str, float]) -> dict:
    """Read those same quantile positions off THIS asset's 2024 distributions."""
    adx, slope, bb = _label_inputs(fit_frame)
    return {
        "trend_adx_min": float(adx.quantile(ref_q["trend_adx_min"])),
        "weak_adx_max": float(adx.quantile(ref_q["weak_adx_max"])),
        "slope_min": float(slope.abs().quantile(ref_q["slope_min"])),
        "tight_bb_max": float(bb.quantile(ref_q["tight_bb_max"])),
        "prefix_stem": PREFIX_STEM,
    }


def labels_for(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    from scripts.experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded

    return _current_labels3_thresholded(frame, cfg)


def make_scaler(kind: str):
    if kind == "robust":
        return RobustScaler(quantile_range=(5.0, 95.0))
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"unknown scaler: {kind}")


# Winsorization bounds, frozen on the fit window and applied to every window. This is not
# cosmetic for a JM: assignment is squared Euclidean distance, so a single heavy-tailed column
# dominates the metric for every bar in its tail. Measured directly -- jm9's Sortino ratio
# (ewm_return / downside_deviation, unbounded when the denominator collapses) drove the typical
# cross-state cost spread to ~3.1e5 under StandardScaler, which pins the decode into one state for
# an entire year and makes lambda irrelevant. Clipping at the fit window's 1st/99th percentile is
# applied uniformly to every panel so the panel comparison is not decided by tail handling.
WINSOR_Q = (0.01, 0.99)


def fit_scale(x_fit: pd.DataFrame, scaler_kind: str):
    raw = x_fit.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = raw.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    lo = filled.quantile(WINSOR_Q[0])
    hi = filled.quantile(WINSOR_Q[1])
    scaler = make_scaler(scaler_kind)
    clipped = filled.clip(lower=lo, upper=hi, axis=1)
    return scaler.fit_transform(clipped).astype(np.float64), scaler, medians, (lo, hi)


def apply_scale(x: pd.DataFrame, scaler, medians: pd.Series, clip_bounds) -> np.ndarray:
    lo, hi = clip_bounds
    raw = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = raw.fillna(medians).fillna(0.0).clip(lower=lo, upper=hi, axis=1)
    return scaler.transform(filled).astype(np.float64)


# --------------------------------------------------------------------------------------------
# jump model: fit (offline DP + centroid updates) and causal soft decode
# --------------------------------------------------------------------------------------------
def _cost_matrix(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-bar squared distance to each centroid, (n, k).

    Expanded as ||x||^2 - 2 x.mu + ||mu||^2 so the work is one BLAS gemm. The obvious broadcast
    form `((x[:,None,:] - mu[None,:,:])**2).sum(2)` materialises an (n, k, d) temporary, which at
    n=105k, k=4, d=130 is 438 MB per call -- and this is called on every descent iteration and
    every decode, across every worker process. Measured on that shape: 2.15s broadcast vs 0.236s
    gemm, agreeing to 5e-16 relative.
    """
    d2 = (np.einsum("ij,ij->i", x, x)[:, None]
          - 2.0 * (x @ mu.T)
          + np.einsum("ij,ij->i", mu, mu)[None, :])
    # The expansion can land a hair below zero for a point sitting on its own centroid; squared
    # distance cannot be negative, and k-means++ samples proportional to it, so clamp.
    return np.maximum(d2, 0.0, out=d2)


def offline_dp(cost: np.ndarray, lam: float) -> np.ndarray:
    """Viterbi-style DP for a constant jump penalty. Plain-Python inner loop on purpose: k is 2-4,
    so numpy's per-call overhead dominates and this is ~6x faster than the vectorised form."""
    c = cost.tolist()
    n, k = len(c), len(c[0])
    idx = range(k)
    v = list(c[0])
    back = [[0] * k for _ in range(n)]
    for t in range(1, n):
        m = min(v)
        ai = v.index(m)
        sw = m + lam
        ct = c[t]
        bt = back[t]
        nv = [0.0] * k
        for s in idx:
            vs = v[s]
            if vs <= sw:
                nv[s] = ct[s] + vs
                bt[s] = s
            else:
                nv[s] = ct[s] + sw
                bt[s] = ai
        v = nv
    states = [0] * n
    states[-1] = v.index(min(v))
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1][states[t + 1]]
    return np.asarray(states, dtype=np.int8)


def fit_jm(x: np.ndarray, k: int, lam: float, seed: int, n_init: int = 5, n_iter: int = 15):
    """Coordinate descent: optimal jump-penalised state path given centroids, centroids re-set to
    state means. k-means++ init, `n_init` restarts, best total objective wins."""
    rng = np.random.default_rng(seed)
    best_obj, best_mu = np.inf, None
    for _ in range(n_init):
        mu = [x[rng.integers(len(x))]]
        while len(mu) < k:
            d2 = np.min(_cost_matrix(x, np.asarray(mu)), axis=1)
            tot = d2.sum()
            p = d2 / tot if tot > 0 else np.full(len(x), 1.0 / len(x))
            mu.append(x[rng.choice(len(x), p=p)])
        mu = np.asarray(mu, dtype=np.float64)
        prev = None
        states = None
        for _ in range(n_iter):
            states = offline_dp(_cost_matrix(x, mu), lam)
            for s in range(k):
                m = states == s
                if m.sum() > 10:
                    mu[s] = x[m].mean(axis=0)
            if prev is not None and (states == prev).all():
                break
            prev = states
        obj = float(((x - mu[states]) ** 2).sum() + lam * (np.diff(states) != 0).sum())
        if obj < best_obj:
            best_obj, best_mu = obj, mu.copy()
    return best_mu, best_obj


def causal_decode_V(x: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    """Forward-only DP: V_t(s) = ||x_t - mu_s||^2 + min(V_{t-1}(s), min_s' V_{t-1}(s') + lambda),
    re-based to min_s V_t(s) = 0 at every step (numerical drift control).

    Returns the (n, k) relative running-cost matrix. Everything downstream -- the hard path and
    the soft state distribution at any temperature -- is a pure function of this, so a config is
    decoded ONCE and then read out at every temperature for free.
    """
    c = _cost_matrix(x, mu).tolist()
    n, k = len(c), len(c[0])
    idx = range(k)
    out = np.zeros((n, k), dtype=np.float64)
    v = list(c[0])
    m = min(v)
    v = [a - m for a in v]
    out[0] = v
    for t in range(1, n):
        m = min(v)
        sw = m + lam
        ct = c[t]
        v = [ct[s] + (v[s] if v[s] <= sw else sw) for s in idx]
        m = min(v)
        v = [a - m for a in v]
        out[t] = v
    return out


def softmax_states(V: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(-(V_t - min_s V_t)/temperature). The DP's switch clamp bounds the cross-state
    spread of V to roughly lambda, which is why usable temperatures live on lambda's scale.

    Temperature leaves the HARD state path untouched (it is a monotone map of -V), but it is NOT
    inert for the emitted class prediction: the class probability the sidecar consumes is the
    mixture `state_prob @ state_class`, so re-weighting the states re-weights that mixture and can
    move the class argmax. It is a real search axis here, not a post-hoc cosmetic -- the earlier
    lambda sweep's argmax-invariance claim holds for the state path only.
    """
    z = np.exp(-V / float(temperature))
    return z / np.clip(z.sum(axis=1, keepdims=True), 1e-300, None)


def causal_decode_soft(x: np.ndarray, mu: np.ndarray, lam: float, temperature: float):
    """(hard states, soft state probabilities) -- convenience wrapper over causal_decode_V."""
    V = causal_decode_V(x, mu, lam)
    probs = softmax_states(V, temperature)
    return np.argmax(probs, axis=1).astype(np.int8), probs


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------
def run_lengths(pred: np.ndarray) -> np.ndarray:
    if len(pred) == 0:
        return np.asarray([], dtype=np.int64)
    change = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(pred)]))
    return np.diff(bounds)


def fwd_log_return(close: np.ndarray, bars: int = FWD_BARS) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) > bars:
        out[:-bars] = np.log(close[bars:]) - np.log(close[:-bars])
    return out


def window_metrics(pred: np.ndarray, y: np.ndarray, close: np.ndarray) -> dict:
    """Detector quality on one window.

    `balanced_accuracy` is fidelity to the ADX/slope/BB rule label the downstream sidecar's
    state->class contract is defined against, and is the selection metric. `economic_separation`
    (bull-predicted minus bear-predicted mean forward 1h log-return, with a Welch t-stat) is the
    mechanism check reported alongside: agreement with a rule label is not by itself evidence the
    detector separates anything tradeable.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    rl = run_lengths(pred)
    counts = np.bincount(pred, minlength=len(CLASSES3)).astype(float)
    cov = counts / max(len(pred), 1)
    fwd = fwd_log_return(close)
    bull_m = (pred == 0) & np.isfinite(fwd)
    bear_m = (pred == 1) & np.isfinite(fwd)
    if bull_m.sum() > 30 and bear_m.sum() > 30:
        a, b = fwd[bull_m], fwd[bear_m]
        sep = float(a.mean() - b.mean())
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        tstat = float(sep / se) if se > 0 else 0.0
    else:
        sep, tstat = 0.0, 0.0
    return {
        "rows": int(len(pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "median_run_bars": float(np.median(rl)) if len(rl) else 0.0,
        "mean_run_bars": float(rl.mean()) if len(rl) else 0.0,
        "coverage": {CLASSES3[i]: float(cov[i]) for i in range(len(CLASSES3))},
        "min_class_coverage": float(cov.min()),
        "economic_separation_fwd1h": sep,
        "economic_separation_tstat": tstat,
    }


def passes_gates(m: dict) -> bool:
    return m["median_run_bars"] >= MIN_MEDIAN_RUN_BARS and m["min_class_coverage"] >= MIN_CLASS_COVERAGE


def slice_window(ts: pd.Series, start: str | None, end: str | None) -> np.ndarray:
    mask = np.ones(len(ts), dtype=bool)
    if start is not None:
        mask &= (ts >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        mask &= (ts <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_numpy()
    return mask


def order_states_by_return(states: np.ndarray, close: np.ndarray, k: int):
    """Diagnostic only: label states bear/chop/bull by mean forward 1h return on the fit window.
    The class mapping actually used downstream is the state->class matrix, not this ordering."""
    fwd = fwd_log_return(close)
    means = {}
    for s in range(k):
        m = states == s
        means[s] = float(np.nanmean(fwd[m])) if m.any() else 0.0
    return sorted(range(k), key=lambda s: means[s]), means


def load_asset_frames(asset: str) -> dict[str, pd.DataFrame]:
    return {year: _read(path) for year, path in SOURCES[asset].items()}


__all__ = [
    "CLASSES3", "LABEL_MODE", "PREFIX_STEM", "HALFLIVES", "FWD_BARS", "SOURCES", "FIT_YEAR",
    "EVAL_WINDOWS", "SELECTION_WINDOW", "MIN_MEDIAN_RUN_BARS", "MIN_CLASS_COVERAGE", "PANEL_DIMS",
    "LABEL_CONFIGS", "build_panel", "choose_decorr_cols", "fit_scale", "apply_scale", "fit_jm",
    "causal_decode_soft", "causal_decode_V", "softmax_states", "offline_dp", "window_metrics",
    "passes_gates", "slice_window",
    "order_states_by_return", "load_asset_frames", "run_lengths", "fwd_log_return",
    "LABEL_BASES", "LABEL_REF_ASSET", "reference_label_quantiles", "quantile_matched_label_config",
    "labels_for", "_labels", "_state_class_matrix", "_class_proba", "_read",
]
