"""Odyssey ETH live core -- regime3 "current" HMM live routing features.

Vendored from `trading_bot_modules/omega4_6_2_source_parent_live.py`'s `Regime3CurrentLiveFeatures`
class (95 lines) plus the raw-state feature functions it calls, `_with_raw_state12`/`_with_raw_state7`
(scripts/retrain_clean_regime_hmm_raw_state12_20260517.py / _raw_state7_20260517.py). Neither of these
touches the cmamba/risk-overlay columns (`RISK_COLS`/`CMAMBA_PREFIX`) that live in the same source
files -- traced every call site inside `.append()`/`_with_raw_state*` and confirmed neither is
reachable from them, import-time or runtime, so this module deliberately excludes that dead code. See
docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md for the full trace.

`GaussianStateModel` (the HMM class the regime3 joblib artifact is pickled with) IS vendored here,
verbatim, from `scripts/retrain_clean_regime_hmm_20260517.py`. This required more than copying the
class definition: pickle resolves a class by its exact module path recorded at *save* time, so the
original artifact (pickled under `scripts.retrain_clean_regime_hmm_20260517.GaussianStateModel`)
cannot be unpickled by an identically-named class living in a different module -- copying the class
alone would have been a correctness trap (confirmed empirically: `joblib.load()` reports the
producing class by its exact module path). The artifact was therefore migrated once (mechanical
parameter copy, not a retrain) by `scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py`
into a sibling file pickled under THIS module's class instead:
`data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/
regime3_current_sensitive_hmm_wide24_2024_odyssey_native.joblib`. Verified bit-identical
`filter_proba()` output against the original artifact/class before and after migration (same
script; see docs/experiments/eth_odyssey_live_cleanroom_dependency_rewrite_20260816.md). This
module now has zero import dependency on any `scripts/*` training script -- confirmed by blocking
`scripts.retrain_clean_regime_hmm_20260517` from being importable at all and re-running the same
import test that motivated this file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =====================================================================================================
# scripts/retrain_clean_regime_hmm_20260517.py -- GaussianStateModel, copied verbatim (a hand-rolled
# sticky Gaussian HMM with its own numpy forward-backward filtering; not sklearn/hmmlearn). Only
# `filter_proba` is called at live inference time, but `fit`/`_init_params`/`_forward_backward` are
# kept too so this is a complete, honest copy rather than an inference-only subset.
# =====================================================================================================


@dataclass
class GaussianStateModel:
    n_states: int
    n_iter: int
    seed: int
    min_var: float = 1e-4
    sticky: float = 0.94

    def __post_init__(self) -> None:
        self.pi_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.log_likelihood_: list[float] = []

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
        m = np.max(a, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out

    def _init_params(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n = len(x)
        score = x[:, 0] if x.shape[1] else np.arange(n, dtype=float)
        order = np.argsort(score)
        centers = []
        for frac in np.linspace(0.04, 0.96, self.n_states):
            centers.append(x[order[int(np.clip(frac * (n - 1), 0, n - 1))]])
        self.mu_ = np.asarray(centers, dtype=np.float64)
        global_var = np.var(x, axis=0).clip(min=self.min_var)
        self.var_ = np.tile(global_var[None, :], (self.n_states, 1))
        self.pi_ = np.ones(self.n_states, dtype=np.float64) / self.n_states
        self.A_ = np.full(
            (self.n_states, self.n_states),
            (1.0 - self.sticky) / max(self.n_states - 1, 1),
            dtype=np.float64,
        )
        np.fill_diagonal(self.A_, self.sticky)
        self.A_ += rng.random(self.A_.shape) * 1e-4
        self.A_ /= self.A_.sum(axis=1, keepdims=True)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        assert self.mu_ is not None and self.var_ is not None
        diff = x[:, None, :] - self.mu_[None, :, :]
        var = np.maximum(self.var_, self.min_var)
        return -0.5 * np.sum((diff * diff) / var[None, :, :] + np.log(2.0 * np.pi * var[None, :, :]), axis=2)

    def _forward_backward(self, log_emit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        assert self.pi_ is not None and self.A_ is not None
        n = log_emit.shape[0]
        log_a = np.zeros_like(log_emit)
        log_b = np.zeros_like(log_emit)
        log_trans = np.log(self.A_ + 1e-300)
        log_a[0] = np.log(self.pi_ + 1e-300) + log_emit[0]
        for t in range(1, n):
            log_a[t] = log_emit[t] + self._logsumexp(log_a[t - 1][:, None] + log_trans, axis=0)
        for t in range(n - 2, -1, -1):
            log_b[t] = self._logsumexp(log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :], axis=1)
        ll = float(self._logsumexp(log_a[-1], axis=0))
        gamma = np.exp(log_a + log_b - ll)
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)
        return gamma, log_a, log_b, ll

    def fit(self, x: np.ndarray) -> "GaussianStateModel":
        x = np.asarray(x, dtype=np.float64)
        self._init_params(x)
        assert self.pi_ is not None and self.A_ is not None
        prev_ll = -np.inf
        for _ in range(int(self.n_iter)):
            log_emit = self._log_emission(x)
            gamma, log_a, log_b, ll = self._forward_backward(log_emit)
            log_trans = np.log(self.A_ + 1e-300)
            log_xi_sum = np.full((self.n_states, self.n_states), -np.inf, dtype=np.float64)
            for t in range(len(x) - 1):
                lx = log_a[t][:, None] + log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :] - ll
                log_xi_sum = np.logaddexp(log_xi_sum, lx)
            xi_sum = np.exp(log_xi_sum)
            self.pi_ = gamma[0] / np.clip(gamma[0].sum(), 1e-300, None)
            self.A_ = xi_sum / np.clip(xi_sum.sum(axis=1, keepdims=True), 1e-300, None)
            self.A_ = 0.015 / max(self.n_states - 1, 1) + 0.985 * self.A_
            self.A_ /= self.A_.sum(axis=1, keepdims=True)
            wsum = gamma.sum(axis=0) + 1e-300
            self.mu_ = (gamma.T @ x) / wsum[:, None]
            assert self.var_ is not None
            for s in range(self.n_states):
                diff = x - self.mu_[s]
                self.var_[s] = ((gamma[:, s][:, None] * diff * diff).sum(axis=0) / wsum[s]).clip(min=self.min_var)
            self.log_likelihood_.append(ll)
            if abs(ll - prev_ll) < 1e-4 * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        return self

    def filter_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.pi_ is not None and self.A_ is not None
        log_emit = self._log_emission(np.asarray(x, dtype=np.float64))
        log_trans = np.log(self.A_ + 1e-300)
        log_alpha = np.zeros_like(log_emit)
        log_alpha[0] = np.log(self.pi_ + 1e-300) + log_emit[0]
        log_alpha[0] -= self._logsumexp(log_alpha[0], axis=0)
        for t in range(1, len(log_emit)):
            log_alpha[t] = log_emit[t] + self._logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            log_alpha[t] -= self._logsumexp(log_alpha[t], axis=0)
        out = np.exp(log_alpha)
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
        return out

from trading_bot_modules.odyssey_tabm_core import ROUTE_COLS  # noqa: E402 -- single source of truth, not re-declared here.

# =====================================================================================================
# scripts/retrain_clean_regime_hmm_raw_state7_20260517.py -- STATE7_COLS / _num / _zscore /
# _with_raw_state7, copied verbatim.
# =====================================================================================================

STATE7_COLS = [
    "state7_trend_score",
    "state7_trend_efficiency_48",
    "state7_directional_return_48",
    "state7_volatility_state",
    "state7_sign_flip_rate_24",
    "state7_range_compression",
    "state7_flow_alignment",
]


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean().ffill()
    std = s.rolling(window, min_periods=min_periods).std().ffill().replace(0, np.nan)
    return ((s - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _with_raw_state7(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = _num(out, "close").ffill()
    high = _num(out, "high").ffill()
    low = _num(out, "low").ffill()
    diff_abs = close.diff().abs()
    net_change_48 = close - close.shift(48)
    er_48 = (net_change_48.abs() / (diff_abs.rolling(48, min_periods=8).sum() + 1e-12)).fillna(0.0)
    ret_48 = (close / close.shift(48) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_vol = ret.rolling(24, min_periods=4).std().fillna(0.0)
    vol_state = _zscore(raw_vol, 288, 24)

    ret_sign = np.sign(ret.where(ret.abs() >= 1e-8, np.nan)).ffill().fillna(0.0)
    sign_flip_24 = (
        (ret_sign != ret_sign.shift(1))
        .astype(float)
        .rolling(24, min_periods=4)
        .mean()
        .fillna(0.0)
    )

    range_48 = ((high.rolling(48, min_periods=8).max() - low.rolling(48, min_periods=8).min()) / close.abs().clip(lower=1e-12)).fillna(0.0)
    range_compression = -_zscore(range_48, 288, 24)
    range_compression = 0.50 * range_compression - 0.35 * _num(out, "bb_width_z").fillna(0.0)
    range_compression += 0.15 * np.tanh((_num(out, "chop_index", 50.0).fillna(50.0) - 50.0) / 20.0)

    trend_score = (
        0.34 * np.tanh(_num(out, "mtf_trend_1h").fillna(0.0) / 0.0010)
        + 0.24 * np.tanh(_num(out, "mtf_trend_4h").fillna(0.0) / 0.0007)
        + 0.16 * np.tanh(_num(out, "hma_slope").fillna(0.0))
        + 0.12 * np.tanh(_num(out, "breakout_strength").fillna(0.0))
        + 0.08 * np.tanh(_num(out, "dual_momentum").fillna(0.0))
        - 0.06 * np.tanh(_num(out, "mean_reversion_z").fillna(0.0) / 2.0)
    )
    flow_raw = (
        0.40 * np.tanh(_num(out, "net_taker_ratio").fillna(0.0))
        + 0.28 * np.tanh(_num(out, "smart_money_flow").fillna(0.0))
        + 0.18 * np.tanh(_num(out, "taker_acceleration").fillna(0.0))
        + 0.14 * np.tanh(_num(out, "ofi_acceleration").fillna(0.0))
    )
    flow_alignment = np.sign(trend_score) * flow_raw

    out["state7_trend_score"] = np.clip(trend_score, -3.0, 3.0)
    out["state7_trend_efficiency_48"] = np.clip(er_48, 0.0, 1.0)
    out["state7_directional_return_48"] = np.tanh(ret_48 / 0.01)
    out["state7_volatility_state"] = np.tanh(vol_state / 3.0)
    out["state7_sign_flip_rate_24"] = np.clip(sign_flip_24, 0.0, 1.0)
    out["state7_range_compression"] = np.tanh(range_compression / 3.0)
    out["state7_flow_alignment"] = np.clip(flow_alignment, -3.0, 3.0)
    for col in STATE7_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# =====================================================================================================
# scripts/retrain_clean_regime_hmm_raw_state12_20260517.py -- RAW5_COLS / _with_raw_state12, copied
# verbatim.
# =====================================================================================================

RAW5_COLS = [
    "state12_log_return",
    "state12_garman_klass_vol",
    "state12_net_taker_ratio",
    "state12_oi_change_rate",
    "state12_chop_index",
]
STATE12_COLS = STATE7_COLS + RAW5_COLS


def _with_raw_state12(frame: pd.DataFrame) -> pd.DataFrame:
    out = _with_raw_state7(frame)
    out["state12_log_return"] = np.tanh(_num(out, "log_return").fillna(0.0) / 0.003)
    out["state12_garman_klass_vol"] = np.tanh(_num(out, "garman_klass_vol").fillna(0.0) / 0.00002)
    out["state12_net_taker_ratio"] = np.tanh(_num(out, "net_taker_ratio").fillna(0.0))
    out["state12_oi_change_rate"] = np.tanh(_num(out, "oi_change_rate").fillna(0.0) / 0.01)
    out["state12_chop_index"] = np.tanh((_num(out, "chop_index", 50.0).fillna(50.0) - 50.0) / 20.0)
    for col in STATE12_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# =====================================================================================================
# scripts/retrain_clean_regime_hmm_20260517.py -- _class_proba, copied verbatim.
# =====================================================================================================


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    out = state_prob @ state_class
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out


# =====================================================================================================
# trading_bot_modules/omega4_6_2_source_parent_live.py -- Regime3CurrentLiveFeatures, copied verbatim
# (no cmamba/risk references existed in the original class to begin with -- see module docstring).
# =====================================================================================================

CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
FORBIDDEN_FEATURE_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
FORBIDDEN_FEATURE_NAMES = {"tp_sl_action_score"}

# Odyssey-native re-pickle of the original artifact (same fitted parameters, mechanical parameter
# copy -- see scripts/migrate_regime3_hmm_artifact_to_odyssey_native_20260816.py and the module
# docstring), pickled under THIS module's vendored GaussianStateModel so joblib.load() never needs
# scripts.retrain_clean_regime_hmm_20260517 to be importable.
DEFAULT_CURRENT_REGIME_PATH = (
    ROOT
    / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
    / "regime3_current_sensitive_hmm_wide24_2024_odyssey_native.joblib"
)


class Regime3CurrentLiveFeatures:
    def __init__(self, *, current_path: str | Path) -> None:
        self.current_payload = joblib.load(Path(current_path))

    @staticmethod
    def _reject_forbidden(cols: list[str], tag: str) -> None:
        bad = [
            c
            for c in cols
            if c in FORBIDDEN_FEATURE_NAMES
            or any(str(c).startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
        ]
        if bad:
            raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")

    @staticmethod
    def _require_finite_frame(raw: pd.DataFrame, tag: str) -> None:
        bad = [str(c) for c in raw.columns if bool(raw[c].isna().any())]
        if bad:
            raise RuntimeError(f"{tag} non-finite model inputs: {bad[:40]}")

    @staticmethod
    def _impute_training_medians(raw: pd.DataFrame, payload: dict[str, Any], tag: str) -> pd.DataFrame:
        medians = payload.get("feature_medians")
        if medians is None:
            raise RuntimeError(f"{tag} payload missing feature_medians")
        fill = pd.Series({str(k): float(v) for k, v in dict(medians).items()})
        missing = [str(c) for c in raw.columns if str(c) not in fill.index]
        if missing:
            raise RuntimeError(f"{tag} feature_medians missing columns: {missing[:40]}")
        return raw.fillna(fill.reindex(raw.columns)).fillna(0.0)

    @staticmethod
    def _finite_latest(frame: pd.DataFrame, cols: list[str], tag: str) -> None:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{tag} missing columns: {missing[:40]}")
        if not len(frame):
            raise RuntimeError(f"{tag} empty frame")
        latest = frame.iloc[-1]
        bad = []
        for col in cols:
            try:
                val = float(latest[col])
            except Exception:
                bad.append(col)
                continue
            if not np.isfinite(val):
                bad.append(col)
        if bad:
            raise RuntimeError(f"{tag} non-finite latest columns: {bad[:40]}")

    @staticmethod
    def _with_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = _with_raw_state12(frame.copy())
        for col in cols:
            if col not in out.columns:
                raise RuntimeError(f"missing current HMM feature column: {col}")
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _append_current(self, frame: pd.DataFrame) -> pd.DataFrame:
        payload = self.current_payload
        cols = list(payload["feature_cols"])
        self._reject_forbidden(cols, "Regime3 current")
        work = self._with_features(frame, cols)
        raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = self._impute_training_medians(raw, payload, "Regime3 current")
        self._require_finite_frame(raw, "Regime3 current")
        xz = payload["scaler"].transform(raw)
        state = payload["model"].filter_proba(xz)
        proba = _class_proba(state, np.asarray(payload["state_class_matrix"], dtype=np.float64))
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)

        out = frame.copy()
        for i, name in enumerate(payload["classes"]):
            out[f"{CURRENT_PREFIX}{name}_prob"] = proba[:, i]
        sorted_p = np.sort(proba, axis=1)
        out[f"{CURRENT_PREFIX}confidence"] = proba.max(axis=1)
        out[f"{CURRENT_PREFIX}margin"] = sorted_p[:, -1] - sorted_p[:, -2]
        out[f"{CURRENT_PREFIX}entropy"] = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1) / np.log(3.0)
        return out

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._append_current(frame)
        self._finite_latest(
            out,
            ROUTE_COLS + [f"{CURRENT_PREFIX}confidence", f"{CURRENT_PREFIX}entropy", f"{CURRENT_PREFIX}margin"],
            "Regime3 current",
        )
        return out
