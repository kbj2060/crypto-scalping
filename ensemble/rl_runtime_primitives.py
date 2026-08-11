"""Lightweight runtime primitives for DSAC scripts.

This module avoids importing pytorch_lightning/torchvision dependency chains.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from features.schema import STATE_ALPHA, STATE_CONF, STATE_ELITE, STATE_PRED, STATE_SYNTH

REGIME_COLS = ["regime_chop", "regime_whipsaw", "regime_bull", "regime_bear", "regime_normal"]
HMM_N_STATES = 4
HMM_DIM = HMM_N_STATES + 1
MTF_DIM = 3
FEATURE_DIM = len(STATE_PRED) + len(STATE_ELITE) + len(STATE_ALPHA) + 1 + HMM_DIM + len(STATE_SYNTH)
STATE_DIM = FEATURE_DIM + 5 + MTF_DIM
STACK_N = 2


class OnlineHMMDetector:
    N_STATES = 4
    OBS_DIM = 3
    MIN_STD = 1e-3
    WINDOW = 512

    def __init__(self):
        self.A = np.full((self.N_STATES, self.N_STATES), 0.05 / (self.N_STATES - 1))
        np.fill_diagonal(self.A, 0.85)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(self.N_STATES) / self.N_STATES
        self.mu = np.array(
            [[0.8, -0.5, 0.3], [-0.8, -0.5, -0.3], [0.0, 1.5, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        self.sigma = np.array(
            [[0.5, 0.4, 0.5], [0.5, 0.4, 0.5], [1.0, 0.6, 0.8], [0.3, 0.3, 0.3]],
            dtype=np.float64,
        )
        self._obs_buffer: deque = deque(maxlen=self.WINDOW)
        self._alpha: np.ndarray = self.pi.copy()
        self._obs_mean = np.zeros(self.OBS_DIM)
        self._obs_std = np.ones(self.OBS_DIM)

    def reset_episode(self):
        self._alpha = self.pi.copy()
        self._obs_buffer.clear()

    def _extract_obs(self, row: dict) -> np.ndarray:
        raw = np.array(
            [float(row.get("log_return", 0.0)), float(row.get("garch_vol_z", 0.0)), float(row.get("oi_change_rate", 0.0))],
            dtype=np.float64,
        )
        return (raw - self._obs_mean) / (self._obs_std + 1e-8)

    def _emission_log_prob(self, obs: np.ndarray) -> np.ndarray:
        diff = obs[None, :] - self.mu
        var = np.maximum(self.sigma**2, self.MIN_STD**2)
        return -0.5 * np.sum((diff**2) / var + np.log(2 * np.pi * var), axis=1)

    def _forward_step(self, obs: np.ndarray) -> np.ndarray:
        log_emit = self._emission_log_prob(obs)
        predicted = self._alpha @ self.A
        log_joint = np.log(predicted + 1e-300) + log_emit
        log_joint -= log_joint.max()
        alpha_new = np.exp(log_joint)
        alpha_new /= alpha_new.sum() + 1e-300
        self._alpha = alpha_new
        return alpha_new

    def fit(self, df: pd.DataFrame, n_iter: int = 30) -> None:
        needed = ["log_return", "garch_vol_z", "oi_change_rate"]
        raw_mat = np.zeros((len(df), 3), dtype=np.float64)
        for i, col in enumerate(needed):
            if col in df.columns:
                raw_mat[:, i] = df[col].fillna(0).values
        self._obs_mean = raw_mat.mean(axis=0)
        self._obs_std = raw_mat.std(axis=0).clip(min=1e-6)

        obs_seq = (raw_mat - self._obs_mean) / (self._obs_std + 1e-8)
        t_len = len(obs_seq)
        for _ in range(n_iter):
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(t_len)])
            log_alpha = np.zeros((t_len, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, t_len):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-300)) + log_emit[t, j]

            log_beta = np.zeros((t_len, self.N_STATES))
            for t in range(t_len - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(np.log(self.A[i, :] + 1e-300) + log_emit[t + 1] + log_beta[t + 1])

            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((t_len - 1, self.N_STATES, self.N_STATES))
            for t in range(t_len - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (
                            log_alpha[t, i] + np.log(self.A[i, j] + 1e-300) + log_emit[t + 1, j] + log_beta[t + 1, j]
                        )
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
            xi = np.exp(log_xi)

            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            self.A = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                self.mu[s] = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff = obs_seq - self.mu[s]
                self.sigma[s] = np.sqrt((w[:, None] * diff**2).sum(axis=0) / w.sum()).clip(self.MIN_STD)

        self._alpha = gamma[-1]
        self._obs_buffer.extend(obs_seq[-self.WINDOW :].tolist())

    def get_features(self, row: dict) -> np.ndarray:
        obs = self._extract_obs(row)
        probs = self._forward_step(obs)
        ent = float(-np.sum(probs * np.log(probs + 1e-300)))
        ent_n = ent / np.log(self.N_STATES + 1e-8)
        self._obs_buffer.append(obs.tolist())
        return np.concatenate([probs, [ent_n]]).astype(np.float32)

    def update_online(self, n_iter: int = 5) -> None:
        if len(self._obs_buffer) < 64:
            return
        obs_seq = np.array(self._obs_buffer, dtype=np.float64)
        t_len = len(obs_seq)
        a_old = self.A.copy()
        for _ in range(n_iter):
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(t_len)])
            log_alpha = np.zeros((t_len, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, t_len):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-300)) + log_emit[t, j]

            log_beta = np.zeros((t_len, self.N_STATES))
            for t in range(t_len - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(np.log(self.A[i, :] + 1e-300) + log_emit[t + 1] + log_beta[t + 1])

            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((t_len - 1, self.N_STATES, self.N_STATES))
            for t in range(t_len - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (
                            log_alpha[t, i] + np.log(self.A[i, j] + 1e-300) + log_emit[t + 1, j] + log_beta[t + 1, j]
                        )
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
            xi = np.exp(log_xi)

            a_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            a_new /= a_new.sum(axis=1, keepdims=True) + 1e-300
            self.A = 0.8 * a_old + 0.2 * a_new

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                new_mu = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff = obs_seq - new_mu
                new_sigma = np.sqrt((w[:, None] * diff**2).sum(axis=0) / w.sum()).clip(self.MIN_STD)
                self.mu[s] = 0.85 * self.mu[s] + 0.15 * new_mu
                self.sigma[s] = 0.85 * self.sigma[s] + 0.15 * new_sigma

        self._alpha = gamma[-1]


class MultiTimeframeFeatures:
    _RET_SCALE = 50.0
    _VOL_1H_WINDOW = 4

    def __init__(self, close_arr: np.ndarray, w1h: int = 1, w4h: int = 4):
        self.w1h = w1h
        self.w4h = w4h
        self._cache = self._precompute(close_arr.astype(np.float64))

    @staticmethod
    def _linreg_slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((x - xm) * (y - ym)).sum() / denom
        price_range = max(y.max() - y.min(), abs(ym) * 0.001, 1e-8)
        return float(np.clip(slope * n / price_range, -1.0, 1.0))

    @staticmethod
    def _logret_slope(logret: np.ndarray) -> float:
        if len(logret) < 2:
            return 0.0
        mean_ret = logret.mean()
        if abs(mean_ret) < 1e-3:
            return 0.0
        return float(np.clip(mean_ret * 100.0, -1.0, 1.0))

    def _precompute(self, close: np.ndarray) -> np.ndarray:
        t_len = len(close)
        out = np.zeros((t_len, MTF_DIM), dtype=np.float32)
        logret = np.zeros(t_len, dtype=np.float64)
        logret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-8))
        for i in range(t_len):
            sv = max(0, i - self._VOL_1H_WINDOW + 1)
            lr1w = logret[sv : i + 1]
            trend1 = self._logret_slope(lr1w) if len(lr1w) >= 2 else 0.0
            s4 = max(0, i - self.w4h + 1)
            c4 = close[s4 : i + 1]
            ret4 = float(np.tanh((c4[-1] / c4[0] - 1) * self._RET_SCALE)) if len(c4) > 1 else 0.0
            trend4 = self._linreg_slope(c4) if len(c4) >= 3 else 0.0
            align = float(np.sign(trend1) * np.sign(trend4)) if (trend1 != 0 and trend4 != 0) else 0.0
            out[i] = [ret4, trend4, align]
        return out

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._cache):
            return np.zeros(MTF_DIM, dtype=np.float32)
        return self._cache[idx]
