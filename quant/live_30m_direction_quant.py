#!/usr/bin/env python3
"""
Read-only data pipeline for live DuckDB sources + Binance close merge,
then define a geometric/vector/calculus objective for close prediction.

Target: predict zeta_px N minutes into the future (default 30 min).

Key guarantees:
- DuckDB connections are opened with read_only=True.
- Only SELECT statements are used; no mutation SQL is executed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple, Union
from urllib.parse import urlencode
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree
from scipy.optimize import minimize


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_DB_MEM_CACHE: Dict[str, Dict[str, object]] = {}


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class FrameSpec:
    db_path: str
    alias_prefix: str


def _discover_single_table(con: duckdb.DuckDBPyConnection) -> Tuple[str, str]:
    tables = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type='BASE TABLE'
          AND table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name
        """
    ).fetchall()
    if not tables:
        raise RuntimeError("No base table found in DuckDB")
    if len(tables) > 1:
        raise RuntimeError(f"Expected 1 table, found {len(tables)}: {tables}")
    return tables[0][0], tables[0][1]


def _read_table_read_only(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        schema, table = _discover_single_table(con)
        df = con.execute(f'SELECT * FROM "{schema}"."{table}"').fetchdf()
    finally:
        con.close()

    if "ts" not in df.columns:
        raise RuntimeError(f"{db_path} table has no 'ts' column")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def _read_table_cached_read_only(db_path: str) -> pd.DataFrame:
    """
    Read DuckDB table with memory cache.
    Cache invalidates when file mtime changes.
    """
    mtime = os.path.getmtime(db_path)
    cached = _DB_MEM_CACHE.get(db_path)
    if cached is not None and float(cached.get("mtime", -1.0)) == float(mtime):
        return cached["df"].copy()
    df = _read_table_read_only(db_path)
    _DB_MEM_CACHE[db_path] = {"mtime": float(mtime), "df": df}
    return df.copy()


def _anonymize_columns(df: pd.DataFrame, prefix: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = [c for c in df.columns if c != "ts"]
    mapping: Dict[str, str] = {c: f"{prefix}_{i:03d}" for i, c in enumerate(cols, start=1)}
    return df.rename(columns=mapping), mapping


def _dt_to_ms(dt: pd.Timestamp) -> int:
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return int(dt.tz_convert("UTC").timestamp() * 1000)


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[list]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    url = f"{BINANCE_KLINES_URL}?{urlencode(params)}"
    with urlopen(url, timeout=20) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _fetch_binance_close_series(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str = "1m",
) -> pd.DataFrame:
    start_ms = _dt_to_ms(start_ts.floor("min"))
    end_ms = _dt_to_ms(end_ts.ceil("min"))

    rows: List[list] = []
    cursor = start_ms
    # Binance max 1000 klines per call.
    while cursor <= end_ms:
        batch = _fetch_klines(symbol=symbol, interval=interval, start_ms=cursor, end_ms=end_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + 60_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    if not rows:
        raise RuntimeError("No klines fetched from Binance")

    kdf = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ])
    kdf["ts"] = pd.to_datetime(kdf["open_time"].astype("int64"), unit="ms", utc=True)
    kdf["close"] = pd.to_numeric(kdf["close"], errors="coerce")
    kdf = kdf[["ts", "close"]].dropna().sort_values("ts").reset_index(drop=True)
    return kdf


def _resample_bars(df: pd.DataFrame, close_col: str, bar_minutes: int) -> pd.DataFrame:
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if "ts" not in df.columns:
        raise ValueError("DataFrame must include ts column")

    work = df.copy()
    work["bar_ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.floor(f"{bar_minutes}min")
    work = work.dropna(subset=["bar_ts"]).reset_index(drop=True)

    numeric_cols = set(work.select_dtypes(include=[np.number]).columns.tolist())
    agg: Dict[str, str] = {}
    for c in work.columns:
        if c in {"ts", "bar_ts"}:
            continue
        if c == close_col:
            agg[c] = "last"
        elif c in numeric_cols:
            agg[c] = "mean"
        else:
            agg[c] = "last"

    out = work.groupby("bar_ts", as_index=False).agg(agg).rename(columns={"bar_ts": "ts"})
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def build_merged_dataset(
    micro_db_path: str,
    tail_db_path: str,
    symbol: str,
    close_alias: str = "zeta_px",
    bar_minutes: int = 30,
    horizon_minutes: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str], str, str]:
    """
    Returns
    -------
    merged : DataFrame with current features, current close (close_alias),
             and future close (target_alias).
    micro_map, tail_map : column rename dictionaries.
    close_alias : column name for *current* close.
    target_alias : column name for *future* close (horizon_minutes later).
    """
    micro_raw = _read_table_read_only(micro_db_path)
    tail_raw = _read_table_read_only(tail_db_path)

    micro_renamed, micro_map = _anonymize_columns(micro_raw, "mvec")
    tail_renamed, tail_map = _anonymize_columns(tail_raw, "trsk")

    merged = pd.merge(micro_renamed, tail_renamed, on="ts", how="inner")
    if merged.empty:
        raise RuntimeError("Merged micro/tail frame is empty on ts")

    close_df = _fetch_binance_close_series(
        symbol=symbol,
        start_ts=merged["ts"].min(),
        end_ts=merged["ts"].max(),
        interval="1m",
    )

    merged = pd.merge(merged, close_df, on="ts", how="inner")
    merged = merged.rename(columns={"close": close_alias})
    merged = merged.sort_values("ts").reset_index(drop=True)

    # 30분 바 데이터로 리샘플링
    merged = _resample_bars(merged, close_col=close_alias, bar_minutes=bar_minutes)

    # ── horizon_minutes 뒤 zeta_px 타겟 생성 (bar 단위 shift) ──
    if horizon_minutes % bar_minutes != 0:
        raise ValueError("horizon_minutes must be a multiple of bar_minutes")
    step_ahead = max(1, horizon_minutes // bar_minutes)
    target_alias = f"{close_alias}_{horizon_minutes}m"
    merged[target_alias] = merged[close_alias].shift(-step_ahead)
    # 미래 값이 없는 마지막 step_ahead 행 제거
    merged = merged.dropna(subset=[target_alias]).reset_index(drop=True)

    return merged, micro_map, tail_map, close_alias, target_alias


def _safe_grad(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, t)


def geometric_vector_calculus_objective(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    t_seconds: Sequence[float],
    w_data: float = 1.0,
    w_angle: float = 0.2,
    w_curvature: float = 0.2,
    w_path: float = 0.1,
    w_smooth: float = 0.1,
) -> float:
    """
    Non-linear functional objective using geometry/vector/calculus terms.

    J = w_data * ∫(y-ŷ)^2 dt
        + w_angle * E[(arccos(<T,T^>/||T|| ||T^||))^2]
        + w_curvature * E[(kappa-kappa^)^2]
        + w_path * (L(y)-L(ŷ))^2
        + w_smooth * ∫(d²ŷ/dt²)^2 dt

    where:
    T      = (1, dy/dt),
    kappa  = y'' / (1 + (y')^2)^(3/2),
    L(y)   = ∫ sqrt(1 + (dy/dt)^2) dt.
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    t = np.asarray(t_seconds, dtype=float)

    if y.shape != yhat.shape or y.shape != t.shape:
        raise ValueError("y_true, y_pred, t_seconds must have identical shape")
    if len(y) < 3:
        raise ValueError("Need at least 3 points for derivative/curvature terms")

    # Ensure finite + ordered + unique time coordinates for stable derivatives.
    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(t)
    y, yhat, t = y[valid], yhat[valid], t[valid]
    order = np.argsort(t)
    y, yhat, t = y[order], yhat[order], t[order]
    packed = pd.DataFrame({"t": t, "y": y, "yhat": yhat}).groupby("t", as_index=False).mean()
    y = packed["y"].to_numpy(dtype=float)
    yhat = packed["yhat"].to_numpy(dtype=float)
    t = packed["t"].to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 unique time points after cleanup")

    dy = _safe_grad(y, t)
    dyhat = _safe_grad(yhat, t)
    d2y = _safe_grad(dy, t)
    d2yhat = _safe_grad(dyhat, t)

    # Data fidelity via integral of squared error
    int_data = _integrate((y - yhat) ** 2, t)

    # Vector/tangent angle mismatch
    T = np.stack([np.ones_like(dy), dy], axis=1)
    That = np.stack([np.ones_like(dyhat), dyhat], axis=1)
    dot = np.sum(T * That, axis=1)
    norm = np.linalg.norm(T, axis=1) * np.linalg.norm(That, axis=1) + 1e-12
    cosang = np.clip(dot / norm, -1.0, 1.0)
    angle_term = float(np.mean(np.arccos(cosang) ** 2))

    # Curvature mismatch (differential geometry)
    kappa = d2y / np.power(1.0 + dy ** 2, 1.5)
    kappahat = d2yhat / np.power(1.0 + dyhat ** 2, 1.5)
    curvature_term = float(np.mean((kappa - kappahat) ** 2))

    # Arc-length mismatch
    L_true = _integrate(np.sqrt(1.0 + dy ** 2), t)
    L_pred = _integrate(np.sqrt(1.0 + dyhat ** 2), t)
    path_term = (L_true - L_pred) ** 2

    # Smoothness regularization via integral of squared acceleration
    smooth_term = _integrate(d2yhat ** 2, t)

    return (
        w_data * float(int_data)
        + w_angle * angle_term
        + w_curvature * curvature_term
        + w_path * path_term
        + w_smooth * smooth_term
    )


def data_adaptive_geo_vector_objective(
    X: np.ndarray,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    t_seconds: Sequence[float],
    return_terms: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    """
    Data-adaptive objective using calculus + geometry + vector terms.

    Curve embedding:
      r_true(t) = (t, y(t), z(t)),  r_pred(t) = (t, y_hat(t), z(t))
    where z(t) is the first principal trajectory from X.

    Objective:
      J = λ1 * ∫ w(t) (y-ŷ)^2 dt
        + λ2 * E[θ(t)^2]
        + λ3 * E[(κ-κ̂)^2]
        + λ4 * (L-L̂)^2
        + λ5 * ∫ ||z''(t)|| |y-ŷ| dt
        + λ6 * ∫ (ŷ''(t))^2 dt

    with data-driven weights λi from empirical scales.
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    t = np.asarray(t_seconds, dtype=float)
    Xv = np.asarray(X, dtype=float)

    if y.shape != yhat.shape or y.shape != t.shape:
        raise ValueError("y_true, y_pred, t_seconds must have identical shape")
    if Xv.ndim != 2 or len(Xv) != len(y):
        raise ValueError("X must be 2D with same row count as y")
    if len(y) < 5:
        raise ValueError("Need at least 5 points for stable geometry terms")

    # Clean and sort
    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(t)
    y, yhat, t, Xv = y[valid], yhat[valid], t[valid], Xv[valid]
    order = np.argsort(t)
    y, yhat, t, Xv = y[order], yhat[order], t[order], Xv[order]

    # Time uniqueness by averaging duplicates
    temp = pd.DataFrame({"t": t, "y": y, "yhat": yhat})
    temp["_idx"] = np.arange(len(temp))
    grp = temp.groupby("t", as_index=False).mean(numeric_only=True)
    keep_idx = grp["_idx"].round().astype(int).clip(0, len(Xv) - 1).to_numpy()
    t = grp["t"].to_numpy(dtype=float)
    y = grp["y"].to_numpy(dtype=float)
    yhat = grp["yhat"].to_numpy(dtype=float)
    Xv = Xv[keep_idx]
    if len(y) < 5:
        raise ValueError("Need at least 5 unique time points after cleanup")

    # First principal trajectory z(t) from X (vector/geometric manifold axis)
    Xc = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = Xc - np.mean(Xc, axis=0, keepdims=True)
    std = np.std(Xc, axis=0, keepdims=True) + 1e-12
    Xn = Xc / std
    _, _, vt = np.linalg.svd(Xn, full_matrices=False)
    pc1 = vt[0]
    z = Xn @ pc1

    # Derivatives
    dy = _safe_grad(y, t)
    dyhat = _safe_grad(yhat, t)
    d2y = _safe_grad(dy, t)
    d2yhat = _safe_grad(dyhat, t)
    dz = _safe_grad(z, t)
    d2z = _safe_grad(dz, t)

    # 3D tangent vectors of embedded curves
    T = np.stack([np.ones_like(dy), dy, dz], axis=1)
    That = np.stack([np.ones_like(dyhat), dyhat, dz], axis=1)
    dot = np.sum(T * That, axis=1)
    nrm = np.linalg.norm(T, axis=1) * np.linalg.norm(That, axis=1) + 1e-12
    cosang = np.clip(dot / nrm, -1.0, 1.0)
    theta2 = np.arccos(cosang) ** 2

    # 3D curvature: ||r' x r''|| / ||r'||^3
    r1 = T
    r2 = np.stack([np.zeros_like(d2y), d2y, d2z], axis=1)
    r2h = np.stack([np.zeros_like(d2yhat), d2yhat, d2z], axis=1)
    kappa = np.linalg.norm(np.cross(r1, r2), axis=1) / (np.linalg.norm(r1, axis=1) ** 3 + 1e-12)
    kappah = np.linalg.norm(np.cross(That, r2h), axis=1) / (np.linalg.norm(That, axis=1) ** 3 + 1e-12)

    # Integral weight from local signal roughness (data-adaptive)
    med_dy = float(np.median(np.abs(dy)) + 1e-12)
    med_d2y = float(np.median(np.abs(d2y)) + 1e-12)
    w_t = 1.0 + (np.abs(dy) / med_dy) + (np.abs(d2y) / med_d2y)

    err = y - yhat
    term_data = _integrate(w_t * (err ** 2), t)
    term_angle = float(np.mean(theta2))
    term_curv = float(np.mean((kappa - kappah) ** 2))
    L_true = _integrate(np.linalg.norm(T, axis=1), t)
    L_pred = _integrate(np.linalg.norm(That, axis=1), t)
    term_path = float((L_true - L_pred) ** 2)
    term_manifold = _integrate(np.abs(d2z) * np.abs(err), t)
    term_smooth = _integrate(d2yhat ** 2, t)

    # Data-derived lambdas
    var_y = float(np.var(y) + 1e-12)
    var_dy = float(np.var(dy) + 1e-12)
    var_d2y = float(np.var(d2y) + 1e-12)
    mean_abs_d2z = float(np.mean(np.abs(d2z)) + 1e-12)
    mean_abs_dy = float(np.mean(np.abs(dy)) + 1e-12)
    l1 = 1.0
    l2 = var_dy / var_y
    l3 = var_d2y / var_dy
    l4 = 1.0 / (L_true + 1e-12)
    l5 = mean_abs_d2z / mean_abs_dy
    l6 = med_d2y / med_dy
    lsum = l1 + l2 + l3 + l4 + l5 + l6 + 1e-12
    l1, l2, l3, l4, l5, l6 = (l1 / lsum, l2 / lsum, l3 / lsum, l4 / lsum, l5 / lsum, l6 / lsum)

    obj = (
        l1 * term_data
        + l2 * term_angle
        + l3 * term_curv
        + l4 * term_path
        + l5 * term_manifold
        + l6 * term_smooth
    )

    if return_terms:
        return float(obj), {
            "lambda_data": float(l1),
            "lambda_angle": float(l2),
            "lambda_curvature": float(l3),
            "lambda_path": float(l4),
            "lambda_manifold": float(l5),
            "lambda_smooth": float(l6),
            "term_data": float(term_data),
            "term_angle": float(term_angle),
            "term_curvature": float(term_curv),
            "term_path": float(term_path),
            "term_manifold": float(term_manifold),
            "term_smooth": float(term_smooth),
        }
    return float(obj)


# ─────────────────────────────────────────────────────────────
#  Phase-Space Geodesic Predictor
# ─────────────────────────────────────────────────────────────

def _takens_embed(price: np.ndarray, dim: int = 5, tau: int = 1) -> np.ndarray:
    """
    Takens delay embedding: reconstruct phase space from 1D price series.
    Φ(t) = [p(t), p(t-τ), p(t-2τ), ..., p(t-(d-1)τ)]
    Returns (N - (dim-1)*tau, dim) array.
    """
    n = len(price)
    rows = n - (dim - 1) * tau
    if rows <= 0:
        raise ValueError(f"Not enough data for embedding: {n} pts, dim={dim}, tau={tau}")
    out = np.empty((rows, dim), dtype=float)
    for d in range(dim):
        out[:, d] = price[d * tau : d * tau + rows]
    return out


def _estimate_tau(price: np.ndarray, max_lag: int = 50) -> int:
    """
    Estimate delay τ from first zero-crossing of autocorrelation.
    """
    p = price - np.mean(price)
    var = np.dot(p, p)
    if var < 1e-15:
        return 1
    for lag in range(1, min(max_lag, len(price) // 2)):
        acf = np.dot(p[lag:], p[:-lag]) / var
        if acf <= 0:
            return max(1, lag)
    return max(1, max_lag // 2)


def _frenet_serret_3d(
    curve: np.ndarray, dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Frenet-Serret frame for a 3D (or higher dim reduced to 3) curve.

    Returns:
        T: unit tangent (N, 3)
        N: unit normal  (N, 3)
        B: binormal     (N, 3)
        kappa: curvature (N,)
        torsion: torsion (N,)
    """
    # Use only first 3 dims if higher
    c = curve[:, :3].copy() if curve.shape[1] >= 3 else np.hstack(
        [curve, np.zeros((len(curve), 3 - curve.shape[1]))]
    )
    n = len(c)

    # First and second derivatives via central differences
    r1 = np.gradient(c, dt, axis=0)    # dr/dt
    r2 = np.gradient(r1, dt, axis=0)   # d²r/dt²
    r3 = np.gradient(r2, dt, axis=0)   # d³r/dt³

    # Tangent
    r1_norm = np.linalg.norm(r1, axis=1, keepdims=True) + 1e-15
    T = r1 / r1_norm

    # Curvature: κ = ||r' × r''|| / ||r'||³
    cross12 = np.cross(r1, r2)
    cross12_norm = np.linalg.norm(cross12, axis=1) + 1e-15
    kappa = cross12_norm / (r1_norm.ravel() ** 3 + 1e-15)

    # Normal: N = (T' / ||T'||)  (more stable than cross product method)
    dT = np.gradient(T, dt, axis=0)
    dT_norm = np.linalg.norm(dT, axis=1, keepdims=True) + 1e-15
    N = dT / dT_norm

    # Binormal: B = T × N
    B = np.cross(T, N)

    # Torsion: τ = (r' × r'') · r''' / ||r' × r''||²
    dot_r3 = np.sum(cross12 * r3, axis=1)
    torsion = dot_r3 / (cross12_norm ** 2 + 1e-15)

    return T, N, B, kappa, torsion


def _osculating_circle_extrapolate(
    T: np.ndarray, N: np.ndarray, kappa: np.ndarray,
    last_pos: np.ndarray, arc_length: float,
) -> np.ndarray:
    """
    Extrapolate along the osculating circle from the last point.
    Uses the formula: p(s) = p₀ + R·sin(s/R)·T + R·(1−cos(s/R))·N
    where R = 1/κ is the osculating radius.
    """
    kappa_safe = max(abs(float(kappa)), 1e-10)
    R = 1.0 / kappa_safe
    theta = arc_length / R
    # Clamp θ to avoid wild extrapolation
    theta = np.clip(theta, -math.pi / 4, math.pi / 4)
    p_new = last_pos + R * math.sin(theta) * T + R * (1.0 - math.cos(theta)) * N
    return p_new


def phase_space_geodesic_predictor(
    price_series: np.ndarray,
    features: np.ndarray,
    y_now: np.ndarray,
    horizon: int = 30,
    embed_dim: int = 5,
    k_neighbors: int = 10,
    manifold_alpha: float = 0.3,
) -> np.ndarray:
    """
    Phase-Space Geodesic Predictor:
      1. Takens delay embedding of price → phase space trajectory
      2. Frenet-Serret frame → curvature κ, torsion τ
      3. Osculating circle extrapolation → geometric prediction
      4. Feature manifold k-NN correction → final prediction

    NOT a linear equation — uses differential geometry of the price trajectory.

    Parameters
    ----------
    price_series : (N,) current close prices (original scale)
    features     : (N, D) feature matrix
    y_now        : (N,) current close prices (same as price_series usually)
    horizon      : prediction horizon in bars
    embed_dim    : Takens embedding dimension
    k_neighbors  : k for manifold correction
    manifold_alpha : weight of manifold correction vs geometric prediction
    """
    n = len(price_series)
    if n < embed_dim * 3:
        return y_now.copy()  # fallback

    # --- Step 1: Takens embedding ---
    tau = _estimate_tau(price_series)
    embed = _takens_embed(price_series, dim=embed_dim, tau=tau)
    offset = n - len(embed)  # index offset due to embedding

    # --- Step 2: PCA to 3D for Frenet-Serret ---
    embed_c = embed - np.mean(embed, axis=0, keepdims=True)
    embed_std = np.std(embed_c, axis=0, keepdims=True) + 1e-12
    embed_n = embed_c / embed_std
    try:
        U, S, Vt = np.linalg.svd(embed_n, full_matrices=False)
        curve_3d = embed_n @ Vt[:3].T  # project onto first 3 PCs
    except np.linalg.LinAlgError:
        return y_now.copy()

    # --- Step 3: Frenet-Serret frame ---
    T_vec, N_vec, B_vec, kappa, torsion = _frenet_serret_3d(curve_3d, dt=1.0)

    # --- Step 4: Osculating circle extrapolation ---
    # For each point, estimate future price using local geometry
    y_geo = np.full(n, np.nan, dtype=float)

    # Average speed in phase space (for arc length estimation)
    speeds = np.linalg.norm(np.diff(curve_3d, axis=0), axis=1)
    avg_speed = float(np.median(speeds)) + 1e-15

    for i in range(len(curve_3d)):
        orig_i = i + offset
        if orig_i >= n:
            break

        # Local curvature and frame at this point
        local_T = T_vec[i]
        local_N = N_vec[i]
        local_kappa = kappa[i]
        local_pos = curve_3d[i]

        # Arc length to travel = speed × horizon
        local_speed = speeds[min(i, len(speeds) - 1)] if i < len(speeds) else avg_speed
        arc = local_speed * horizon

        # Extrapolate in phase space
        future_pos = _osculating_circle_extrapolate(
            T=local_T, N=local_N, kappa=local_kappa,
            last_pos=local_pos, arc_length=arc,
        )

        # Back-project to price: the first PC component encodes most price variance
        # future_pos is in 3D PCA space → price change ≈ Vt[0] · future_pos
        future_embed_approx = future_pos @ Vt[:3]  # back to embed space (normalized)
        current_embed_approx = local_pos @ Vt[:3]
        delta_embed = future_embed_approx - current_embed_approx

        # Price delta from the first delay coordinate (most recent price)
        price_delta = delta_embed[0] * float(embed_std[0, 0])
        y_geo[orig_i] = price_series[orig_i] + price_delta

    # Fill any remaining NaN with naive (current price)
    nan_mask = np.isnan(y_geo)
    y_geo[nan_mask] = y_now[nan_mask]

    # --- Step 5: Feature manifold k-NN correction ---
    feat_c = features - np.mean(features, axis=0, keepdims=True)
    feat_s = np.std(feat_c, axis=0, keepdims=True) + 1e-12
    feat_n = feat_c / feat_s

    tree = cKDTree(feat_n)
    k = min(k_neighbors, n - 1)

    # For each point, find k-nearest neighbors and compute
    # manifold-weighted correction from their actual future prices
    y_manifold = np.full(n, np.nan, dtype=float)
    for i in range(n):
        dists, idxs = tree.query(feat_n[i], k=k + 1)
        # Exclude self
        mask = idxs != i
        nn_dists = dists[mask][:k]
        nn_idxs = idxs[mask][:k]

        if len(nn_idxs) == 0:
            y_manifold[i] = y_now[i]
            continue

        # Gaussian kernel weights: w = exp(−α · d²)
        weights = np.exp(-manifold_alpha * nn_dists ** 2)
        w_sum = weights.sum() + 1e-15

        # Weighted average of neighbors' (observed price - current price)
        nn_deltas = price_series[nn_idxs] - y_now[nn_idxs]
        correction = float(np.dot(weights, nn_deltas) / w_sum)
        y_manifold[i] = y_now[i] + correction

    nan_mask2 = np.isnan(y_manifold)
    y_manifold[nan_mask2] = y_now[nan_mask2]

    # --- Step 6: Blend geometric + manifold ---
    # Geometric prediction is the primary signal, manifold provides correction
    blend_weight = 0.7  # 70% geometric, 30% manifold
    y_pred = blend_weight * y_geo + (1.0 - blend_weight) * y_manifold

    return y_pred


def make_lookback_matrix(
    X: np.ndarray,
    y: np.ndarray,
    t_seconds: np.ndarray,
    lookback_minutes: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build samples from previous lookback window for each timestamp."""
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(X) != len(y) or len(y) != len(t_seconds):
        raise ValueError("X, y, t_seconds must have same number of rows")
    if len(X) < lookback_minutes:
        raise ValueError("Not enough rows for requested lookback window")

    out_X: List[np.ndarray] = []
    out_y: List[float] = []
    out_t: List[float] = []
    start = lookback_minutes - 1
    for i in range(start, len(X)):
        window = X[i - lookback_minutes + 1 : i + 1, :]
        out_X.append(window.reshape(-1))
        out_y.append(float(y[i]))
        out_t.append(float(t_seconds[i]))
    return np.vstack(out_X), np.asarray(out_y, dtype=float), np.asarray(out_t, dtype=float)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = y_true - y_pred
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(np.abs(e) / np.clip(np.abs(y_true), 1e-12, None)) * 100.0)
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2, "corr": corr}


def _fit_ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0) + 1e-8
    z_train = (X_train - mu) / sd
    z_eval = (X_eval - mu) / sd
    z_train = np.concatenate([np.ones((len(z_train), 1)), z_train], axis=1)
    z_eval = np.concatenate([np.ones((len(z_eval), 1)), z_eval], axis=1)
    reg = np.eye(z_train.shape[1], dtype=float)
    reg[0, 0] = 0.0  # bias no penalty
    w = np.linalg.solve(z_train.T @ z_train + l2 * reg, z_train.T @ y_train)
    return z_eval @ w


def compute_live_quant_snapshot(
    micro_db_path: str,
    tail_db_path: str,
    close_df: pd.DataFrame,
    current_price: float,
    lookback_minutes: int = 15,
    horizon_minutes: int = 30,
    bar_minutes: int = 1,
    top_k_features: int = 25,
    max_history_rows: int = 3000,
) -> Dict[str, float | str | int]:
    """
    Build one live quant-card snapshot.
    Data source: micro/tail DuckDB(read-only) + provided close series.
    """
    if "ts" not in close_df.columns or "close" not in close_df.columns:
        raise ValueError("close_df must have: ts, close")

    cdf = close_df.copy()
    cdf["ts"] = pd.to_datetime(cdf["ts"], utc=True, errors="coerce")
    cdf["close"] = pd.to_numeric(cdf["close"], errors="coerce")
    cdf = cdf.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)
    if len(cdf) < (lookback_minutes + horizon_minutes + 30):
        raise ValueError("close_df rows are insufficient")

    micro_raw = _read_table_cached_read_only(micro_db_path)
    tail_raw = _read_table_cached_read_only(tail_db_path)
    micro_renamed, _ = _anonymize_columns(micro_raw, "mvec")
    tail_renamed, _ = _anonymize_columns(tail_raw, "trsk")

    merged = pd.merge(micro_renamed, tail_renamed, on="ts", how="inner")
    merged = pd.merge(merged, cdf[["ts", "close"]], on="ts", how="inner")
    merged = merged.rename(columns={"close": "zeta_px"})
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged = _resample_bars(merged, close_col="zeta_px", bar_minutes=bar_minutes)

    if horizon_minutes % bar_minutes != 0:
        raise ValueError("horizon_minutes must be multiple of bar_minutes")
    step_ahead = max(1, horizon_minutes // bar_minutes)
    if max_history_rows > 0:
        keep_rows = int(max_history_rows) + step_ahead + 2
        if len(merged) > keep_rows:
            merged = merged.tail(keep_rows).reset_index(drop=True)
    target_alias = f"zeta_px_{horizon_minutes}m"
    merged[target_alias] = merged["zeta_px"].shift(-step_ahead)
    merged = merged.dropna(subset=[target_alias]).reset_index(drop=True)
    if len(merged) < max(220, lookback_minutes + 50):
        raise ValueError("merged rows are insufficient")

    feature_cols = [c for c in merged.columns if c not in {"ts", target_alias}]
    X_base = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_target_raw = merged[target_alias].to_numpy(dtype=float)
    y_now_raw = merged["zeta_px"].to_numpy(dtype=float)
    t_epoch = (merged["ts"].astype("int64") / 1e9).to_numpy(dtype=float)
    t_seconds = t_epoch - t_epoch[0]

    # V3 pipeline: multi-scale + pattern features
    X_ms, offset = _build_multiscale_features(X_base, y_now_raw, t_seconds)
    y_target = y_target_raw[offset:]
    y_now = y_now_raw[offset:]
    delta_true = y_target - y_now
    direction = (delta_true > 0).astype(float)
    n = len(X_ms)
    if n < 300:
        raise ValueError("insufficient samples for V3")

    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)
    X_tr, X_va = X_ms[:tr_end], X_ms[tr_end:va_end]
    y_tr, y_va = direction[:tr_end], direction[tr_end:va_end]

    # Mutual-information feature selection
    mi_scores = _mutual_information(X_tr, y_tr, n_bins=30)
    top_k = min(int(max(5, top_k_features)), X_ms.shape[1])
    top_idx = np.argsort(-mi_scores)[:top_k]
    X_sel = X_ms[:, top_idx]
    X_tr_s, X_va_s = X_sel[:tr_end], X_sel[tr_end:va_end]

    # Time-decay weights
    half_life = max(10.0, tr_end * 0.3)
    time_weights = np.exp(np.log(2) * np.arange(tr_end) / half_life)
    # Class-imbalance correction (up/down prior drift)
    pos_rate = float(np.mean(y_tr))
    neg_rate = float(1.0 - pos_rate)
    cls_weights = np.where(
        y_tr > 0.5,
        0.5 / max(pos_rate, 1e-6),
        0.5 / max(neg_rate, 1e-6),
    )
    time_weights = time_weights * cls_weights
    time_weights /= np.mean(time_weights)

    # Logistic (L-BFGS-B)
    best_lg = {"name": "logistic", "l2": None, "acc": -1.0, "prob": None}
    for l2 in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        pv = _fit_logistic_lbfgs(X_tr_s, y_tr, X_va_s, l2=l2, sample_weights=time_weights)
        acc = float(np.mean((pv >= 0.5) == y_va))
        if acc > best_lg["acc"]:
            pa = _fit_logistic_lbfgs(X_tr_s, y_tr, X_sel, l2=l2, sample_weights=time_weights)
            best_lg = {"name": "logistic", "l2": l2, "acc": acc, "prob": pa}
    prob_log = best_lg["prob"] if best_lg["prob"] is not None else np.full(n, float(np.mean(y_tr)))

    # Kernel logistic (RFF + L-BFGS-B)
    n_rff = min(200, max(32, top_k * 4))
    best_kl = {"name": "kernel_log", "gamma": None, "l2": None, "acc": -1.0, "prob": None}
    for g in [0.01, 0.1, 0.5, 1.0]:
        for l2 in [0.01, 0.1, 1.0]:
            try:
                mu_s = np.mean(X_tr_s, axis=0)
                sd_s = np.std(X_tr_s, axis=0) + 1e-8
                Z_tr_n = (X_tr_s - mu_s) / sd_s
                Z_va_n = (X_va_s - mu_s) / sd_s
                Z_all_n = (X_sel - mu_s) / sd_s
                Ztr_rff, W_rff, b_rff = _rff_transform(Z_tr_n, n_components=n_rff, gamma=g)
                Zva_rff = _rff_apply(Z_va_n, W_rff, b_rff, n_rff)
                pv = _fit_logistic_lbfgs(Ztr_rff, y_tr, Zva_rff, l2=l2, sample_weights=time_weights)
                acc = float(np.mean((pv >= 0.5) == y_va))
                if acc > best_kl["acc"]:
                    Zall_rff = _rff_apply(Z_all_n, W_rff, b_rff, n_rff)
                    pa = _fit_logistic_lbfgs(Ztr_rff, y_tr, Zall_rff, l2=l2, sample_weights=time_weights)
                    best_kl = {"name": "kernel_log", "gamma": g, "l2": l2, "acc": acc, "prob": pa}
            except Exception:
                continue
    prob_kl = best_kl["prob"] if best_kl["prob"] is not None else np.full(n, float(np.mean(y_tr)))

    # kNN classifier
    best_knn = {"name": "knn", "k": None, "alpha": None, "acc": -1.0, "prob": None}
    for k in [10, 20, 50, 100]:
        for a in [0.1, 0.5, 1.0, 2.0]:
            pv = _knn_classify_proba(X_tr_s, y_tr, X_va_s, k=k, alpha=a)
            acc = float(np.mean((pv >= 0.5) == y_va))
            if acc > best_knn["acc"]:
                pa = _knn_classify_proba(X_tr_s, y_tr, X_sel, k=k, alpha=a)
                best_knn = {"name": "knn", "k": k, "alpha": a, "acc": acc, "prob": pa}
    prob_knn = best_knn["prob"] if best_knn["prob"] is not None else np.full(n, float(np.mean(y_tr)))

    # baseline + ensemble weight search on validation
    prob_base = np.full(n, float(np.mean(y_tr)))
    mdl = {"baseline": prob_base, "logistic": prob_log, "kernel_log": prob_kl, "knn": prob_knn}
    mdl_va = {k: v[tr_end:va_end] for k, v in mdl.items()}
    names = list(mdl.keys())
    best_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    best_acc = -1.0
    for w0 in range(11):
        for w1 in range(11 - w0):
            for w2 in range(11 - w0 - w1):
                w3 = 10 - w0 - w1 - w2
                w = np.array([w0, w1, w2, w3], dtype=float) / 10.0
                p = sum(w[i] * mdl_va[names[i]] for i in range(4))
                acc = float(np.mean(((p >= 0.5).astype(float)) == y_va))
                if acc > best_acc:
                    best_acc = acc
                    best_w = w.copy()

    prob_ens_raw = sum(best_w[i] * mdl[names[i]] for i in range(4))
    # Temperature scaling on validation for probability calibration.
    # Optimize Brier score to reduce over/under-confidence.
    eps = 1e-6
    logit_va = np.log(np.clip(prob_ens_raw[tr_end:va_end], eps, 1 - eps) / np.clip(1.0 - prob_ens_raw[tr_end:va_end], eps, 1 - eps))
    best_temp = 1.0
    best_brier = float("inf")
    for temp in [0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
        p_cal = _sigmoid(logit_va / temp)
        brier = float(np.mean((p_cal - y_va) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_temp = float(temp)
    logit_all = np.log(np.clip(prob_ens_raw, eps, 1 - eps) / np.clip(1.0 - prob_ens_raw, eps, 1 - eps))
    prob_ens = _sigmoid(logit_all / best_temp)
    pred_ens = (prob_ens >= 0.5).astype(float)

    # latest direction signal
    p_last = float(prob_ens[-1])
    direction_last = "UP" if p_last >= 0.5 else "DOWN"
    margin_conf = float(np.clip(abs(p_last - 0.5) * 2.0, 0.0, 1.0))

    prob_up = float(np.clip(p_last, 0.01, 0.99))
    prob_down = float(1.0 - prob_up)

    # expected move from train conditional mean delta
    d_tr = delta_true[:tr_end]
    dir_tr = direction[:tr_end]
    up_mean = float(np.mean(d_tr[dir_tr == 1])) if np.any(dir_tr == 1) else 0.0
    dn_mean = float(np.mean(d_tr[dir_tr == 0])) if np.any(dir_tr == 0) else 0.0
    exp_delta = prob_up * up_mean + prob_down * dn_mean
    current = float(current_price if current_price > 0 else y_now[-1])
    pred_price = float(current + exp_delta)
    expected_return_pct = float((exp_delta / max(abs(current), 1e-12)) * 100.0)

    # OOS metrics on test
    y_te = direction[va_end:]
    p_te = prob_ens[va_end:]
    yb_te = (prob_base[va_end:] >= 0.5).astype(float)
    pb_te = prob_base[va_end:]
    if len(y_te) >= 8:
        model_c = _clf_metrics(y_te, (p_te >= 0.5).astype(float), p_te)
        base_c = _clf_metrics(y_te, yb_te, pb_te)
    else:
        model_c = _clf_metrics(direction, pred_ens, prob_ens)
        base_c = _clf_metrics(direction, (prob_base >= 0.5).astype(float), prob_base)

    # Confidence = margin + recent quality correction.
    # This avoids near-zero confidence in noisy 1m horizon while still
    # penalizing poorly calibrated or weak recent performance.
    if len(y_te) >= 20:
        recent_n = min(120, len(y_te))
        y_recent = y_te[-recent_n:]
        p_recent = p_te[-recent_n:]
        recent_acc = float(np.mean((p_recent >= 0.5).astype(float) == y_recent))
        recent_brier = float(np.mean((p_recent - y_recent) ** 2))
    else:
        recent_acc = float(model_c["accuracy"])
        recent_brier = float(np.mean((prob_ens - direction) ** 2))
    perf_conf = float(np.clip((recent_acc - 0.5) / 0.25, 0.0, 1.0))
    calib_conf = float(np.clip(1.0 - (recent_brier / 0.25), 0.0, 1.0))
    edge_conf = float(np.clip((float(model_c["accuracy"]) - float(base_c["accuracy"])) / 0.15, 0.0, 1.0))
    quality_conf = float(np.clip(0.5 * perf_conf + 0.3 * calib_conf + 0.2 * edge_conf, 0.0, 1.0))
    conf = float(np.clip(0.6 * margin_conf + 0.4 * quality_conf, 0.0, 1.0))
    # Adaptive neutral band: when recent quality is weak, require larger directional edge.
    neutral_band = float(np.clip(0.12 + 0.25 * max(0.0, 0.52 - recent_acc), 0.10, 0.22))
    if margin_conf < neutral_band or conf < 0.18:
        signal = "HOLD"
        direction_out = "NEUTRAL"
    else:
        signal = "LONG" if direction_last == "UP" else "SHORT"
        direction_out = direction_last

    out = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "signal": signal,
        "direction": direction_out,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "pred_price_horizon": pred_price,
        "target_alias": target_alias,
        "current_price": current,
        "expected_return_pct": float(expected_return_pct),
        "confidence": conf,
        "confidence_margin": margin_conf,
        "confidence_quality": quality_conf,
        "neutral_band": neutral_band,
        "calibration_temp": float(best_temp),
        "bar_minutes": int(bar_minutes),
        "lookback_minutes": int(lookback_minutes),
        "horizon_minutes": int(horizon_minutes),
        "model_family": "V3_PATTERN_ENSEMBLE",
        "top_k_features": int(top_k),
        "l2": float(best_lg.get("l2") or 0.0),
        "alpha": float(best_knn.get("alpha") or 0.0),
        "accuracy_model": float(model_c["accuracy"]),
        "accuracy_baseline": float(base_c["accuracy"]),
        "win_rate_model": float(model_c["accuracy"] * 100.0),
        "win_rate_baseline": float(base_c["accuracy"] * 100.0),
        "auc_model": float(model_c["auc_roc"]),
        "auc_baseline": float(base_c["auc_roc"]),
        "model_active": bool(best_w[0] < 0.999),
        "ensemble_weights": {
            "baseline": float(best_w[0]),
            "logistic": float(best_w[1]),
            "kernel_log": float(best_w[2]),
            "knn": float(best_w[3]),
        },
        # backward compatibility keys
        "rmse_model": float(1.0 - model_c["accuracy"]),
        "rmse_naive": float(1.0 - base_c["accuracy"]),
        "r2_model": float(model_c["auc_roc"]),
        "r2_naive": float(base_c["auc_roc"]),
    }
    # Backward compatibility for existing dashboard id
    if horizon_minutes == 30:
        out["pred_price_30m"] = pred_price
    return out


# ─────────────────────────────────────────────────────────────
#  Geometric Multi-Scale Objective
# ─────────────────────────────────────────────────────────────

def geometric_multiscale_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_seconds: np.ndarray,
    features: np.ndarray,
    embed_dim: int = 5,
    return_terms: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Geometric Multi-Scale Functional Objective.

    J = λ₁ · ∫ w(t)(y−ŷ)² dt              # weighted data fidelity
      + λ₂ · E[‖T − T̂‖²]                  # Frenet tangent mismatch
      + λ₃ · E[(κ−κ̂)² + (τ−τ̂)²]          # curvature + torsion mismatch
      + λ₄ · W₁(P_y, P_ŷ)                  # Wasserstein-1 distribution
      + λ₅ · ∫‖Φ_true(t) − Φ_pred(t)‖² dt # phase space trajectory integral
      + λ₆ · (1 − cos(Θ_phase))            # phase portrait angle mismatch
      + λ₇ · ∫(ŷ″)² dt                     # smoothness regularization

    Uses Frenet-Serret frame, Takens embedding, Wasserstein distance.
    """
    y = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_pred, dtype=float)
    t = np.asarray(t_seconds, dtype=float)

    if y.shape != yh.shape or y.shape != t.shape:
        raise ValueError("y_true, y_pred, t_seconds must have identical shape")
    n = len(y)
    if n < 10:
        raise ValueError("Need at least 10 points")

    # Clean
    valid = np.isfinite(y) & np.isfinite(yh) & np.isfinite(t)
    y, yh, t = y[valid], yh[valid], t[valid]
    order = np.argsort(t)
    y, yh, t = y[order], yh[order], t[order]
    n = len(y)

    # --- Derivatives ---
    dy = np.gradient(y, t)
    dyh = np.gradient(yh, t)
    d2y = np.gradient(dy, t)
    d2yh = np.gradient(dyh, t)

    # ── Term 1: Weighted data fidelity ──
    # Weight by local volatility (roughness)
    med_dy = float(np.median(np.abs(dy)) + 1e-12)
    w_t = 1.0 + np.abs(dy) / med_dy
    err = y - yh
    term_data = _integrate(w_t * err ** 2, t)

    # ── Term 2: Frenet tangent vector mismatch ──
    # Build 3D curves from (t, y, dy) and (t, yh, dyh)
    curve_true = np.column_stack([t, y, dy])
    curve_pred = np.column_stack([t, yh, dyh])
    T_true, N_true, _, k_true, tau_true = _frenet_serret_3d(curve_true)
    T_pred, N_pred, _, k_pred, tau_pred = _frenet_serret_3d(curve_pred)
    term_tangent = float(np.mean(np.sum((T_true - T_pred) ** 2, axis=1)))

    # ── Term 3: Curvature + torsion mismatch ──
    term_curvtors = float(np.mean((k_true - k_pred) ** 2 + (tau_true - tau_pred) ** 2))

    # ── Term 4: Wasserstein-1 distance (distribution shape) ──
    term_wass = float(wasserstein_distance(y, yh))

    # ── Term 5: Phase space trajectory integral ──
    # Takens embedding of y_true and y_pred
    tau_est = max(1, _estimate_tau(y))
    edim = min(embed_dim, max(2, n // (3 * tau_est)))
    try:
        embed_true = _takens_embed(y, dim=edim, tau=tau_est)
        embed_pred = _takens_embed(yh, dim=edim, tau=tau_est)
        m = min(len(embed_true), len(embed_pred))
        embed_true, embed_pred = embed_true[:m], embed_pred[:m]
        t_embed = t[(edim - 1) * tau_est : (edim - 1) * tau_est + m]
        phase_diff_sq = np.sum((embed_true - embed_pred) ** 2, axis=1)
        term_phase_traj = _integrate(phase_diff_sq, t_embed)
    except ValueError:
        term_phase_traj = 0.0

    # ── Term 6: Phase portrait angle mismatch ──
    # In the (y, dy/dt) phase plane, compare orbit directions
    phase_true = np.column_stack([y, dy])
    phase_pred = np.column_stack([yh, dyh])
    p_true_norm = np.linalg.norm(phase_true, axis=1, keepdims=True) + 1e-12
    p_pred_norm = np.linalg.norm(phase_pred, axis=1, keepdims=True) + 1e-12
    cos_phase = np.clip(
        np.sum((phase_true / p_true_norm) * (phase_pred / p_pred_norm), axis=1),
        -1.0, 1.0,
    )
    term_phase_angle = float(np.mean(1.0 - cos_phase))

    # ── Term 7: Smoothness regularization ──
    term_smooth = _integrate(d2yh ** 2, t)

    # ── Data-adaptive λ weights ──
    var_y = float(np.var(y) + 1e-12)
    var_dy = float(np.var(dy) + 1e-12)
    var_d2y = float(np.var(d2y) + 1e-12)
    scale_y = float(np.std(y) + 1e-12)

    raw = np.array([
        1.0,                        # λ₁ data
        var_dy / var_y,             # λ₂ tangent
        var_d2y / var_dy,           # λ₃ curv+tors
        1.0 / scale_y,             # λ₄ wasserstein
        1.0,                        # λ₅ phase traj
        var_dy / var_y,             # λ₆ phase angle
        var_d2y / (var_dy + 1e-12), # λ₇ smooth
    ])
    lam = raw / (raw.sum() + 1e-12)

    obj = (
        lam[0] * term_data
        + lam[1] * term_tangent
        + lam[2] * term_curvtors
        + lam[3] * term_wass
        + lam[4] * term_phase_traj
        + lam[5] * term_phase_angle
        + lam[6] * term_smooth
    )

    if return_terms:
        return float(obj), {
            "λ_data": float(lam[0]),
            "λ_tangent": float(lam[1]),
            "λ_curvtors": float(lam[2]),
            "λ_wasserstein": float(lam[3]),
            "λ_phase_traj": float(lam[4]),
            "λ_phase_angle": float(lam[5]),
            "λ_smooth": float(lam[6]),
            "term_data": float(term_data),
            "term_tangent": float(term_tangent),
            "term_curvtors": float(term_curvtors),
            "term_wasserstein": float(term_wass),
            "term_phase_traj": float(term_phase_traj),
            "term_phase_angle": float(term_phase_angle),
            "term_smooth": float(term_smooth),
        }
    return float(obj)


# ─────────────────────────────────────────────────────────────
#  Multi-Scale Differential Feature Engineering
# ─────────────────────────────────────────────────────────────

def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling standard deviation using cumulative sums."""
    n = len(arr)
    if n < window:
        return np.zeros(n)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    cs2 = np.concatenate([[0.0], np.cumsum(arr ** 2)])
    s = cs[window:] - cs[:-window]
    s2 = cs2[window:] - cs2[:-window]
    var = np.maximum(s2 / window - (s / window) ** 2, 0.0)
    return np.concatenate([np.zeros(window - 1), np.sqrt(var)])


def _build_multiscale_features(
    X_base: np.ndarray,
    y_now: np.ndarray,
    t_seconds: np.ndarray,
    windows: Tuple[int, ...] = (1, 3, 5, 10, 15),
) -> Tuple[np.ndarray, int]:
    """
    Rich multi-scale differential geometry features.

    From raw features and price, compute:
    1. Current base features (microstructure + tail_risk + close)
    2. Multi-scale log returns + signed-sqrt (non-linear)
    3. Multi-scale acceleration
    4. Multi-scale realized volatility
    5. Price curvature κ and velocity dp/dt
    6. Feature momentum (Δfeatures at multiple scales)
    7. Cross-features: momentum × volatility (non-linear interaction)

    Returns (features_matrix, offset) where offset rows are trimmed from start.
    """
    max_w = max(windows)
    n = len(X_base)
    if n <= max_w + 2:
        raise ValueError(f"Need more data than max_window+2, got {n}")

    out_n = n - max_w
    feats: List[np.ndarray] = []

    # 1. Current base features
    feats.append(X_base[max_w:])

    # 2. Multi-scale log returns + signed sqrt
    for w in windows:
        p_now = y_now[max_w:]
        p_past = y_now[max_w - w : n - w]
        ret = np.log(np.maximum(p_now, 1e-12) / np.maximum(p_past, 1e-12))
        feats.append(ret.reshape(-1, 1))
        feats.append((np.sign(ret) * np.sqrt(np.abs(ret))).reshape(-1, 1))

    # 3. Multi-scale acceleration (change in momentum)
    dp = np.diff(y_now)
    for w in [3, 5, 10]:
        if w < max_w:
            accel = (dp[max_w - 1 : n - 1] - dp[max_w - 1 - w : n - 1 - w]) / (w + 1e-12)
            feats.append(accel.reshape(-1, 1))

    # 4. Multi-scale realized volatility
    log_rets = np.diff(np.log(np.maximum(y_now, 1e-12)))
    for w in [5, 10, 15]:
        rvol = _rolling_std(log_rets, w)
        feats.append(rvol[max_w - 1 :].reshape(-1, 1))

    # 5. Price curvature and velocity
    dt = np.diff(t_seconds)
    dt = np.where(dt == 0, 1.0, dt)
    dp_dt = dp / dt
    dp_dt_full = np.concatenate([[dp_dt[0]], dp_dt])
    d2p_dt = np.diff(dp_dt_full)
    d2p_full = np.concatenate([[d2p_dt[0]], d2p_dt])
    kappa = d2p_full / np.power(1.0 + dp_dt_full ** 2, 1.5)
    feats.append(kappa[max_w:].reshape(-1, 1))
    feats.append(dp_dt_full[max_w:].reshape(-1, 1))

    # 6. Feature momentum at scales 1 and 5
    for fd_w in [1, 5]:
        if fd_w <= max_w:
            feat_delta = X_base[max_w:] - X_base[max_w - fd_w : n - fd_w]
            feats.append(feat_delta)

    # 7. Cross-features: short_ret × vol, long_ret × vol (non-linear)
    short_ret = np.log(np.maximum(y_now[max_w:], 1e-12) / np.maximum(y_now[max_w - 1 : n - 1], 1e-12))
    long_ret = np.log(np.maximum(y_now[max_w:], 1e-12) / np.maximum(y_now[max_w - 10 : n - 10], 1e-12))
    rvol10 = _rolling_std(log_rets, 10)[max_w - 1 :]
    feats.append((short_ret * rvol10).reshape(-1, 1))
    feats.append((long_ret * rvol10).reshape(-1, 1))
    feats.append((short_ret * kappa[max_w:]).reshape(-1, 1))

    # ━━ 8. Autoregressive (AR) pattern features ━━
    # 과거 방향(상승/하락)의 패턴이 미래 방향을 예측하는 핵심 신호
    price_rets = np.diff(y_now)  # 1-step price change
    direction_1 = (price_rets > 0).astype(float)  # 1=up, 0=down/flat

    for win_ar in [3, 5, 10, 15]:
        # 최근 win_ar분간 상승 비율
        cs_dir = np.concatenate([[0.0], np.cumsum(direction_1)])
        if len(cs_dir) > win_ar:
            up_ratio = (cs_dir[win_ar:] - cs_dir[:-win_ar]) / win_ar
            # align to max_w offset
            up_ratio_aligned = up_ratio[max_w - 1:] if len(up_ratio) > max_w - 1 else up_ratio
            if len(up_ratio_aligned) >= out_n:
                feats.append(up_ratio_aligned[:out_n].reshape(-1, 1))

    # 방향 연속 횟수 (consecutive up/down count)
    consec = np.zeros(len(direction_1))
    for i in range(1, len(direction_1)):
        if direction_1[i] == direction_1[i - 1]:
            consec[i] = consec[i - 1] + 1
        else:
            consec[i] = 1
    consec_signed = consec * (2 * direction_1 - 1)  # positive=up streak, negative=down
    consec_aligned = consec_signed[max_w - 1:]
    if len(consec_aligned) >= out_n:
        feats.append(consec_aligned[:out_n].reshape(-1, 1))

    # 직전 1/3/5분의 return sign 이력
    for lag in [1, 3, 5]:
        if lag < max_w:
            sign_lag = np.sign(y_now[max_w:] - y_now[max_w - lag: n - lag])
            feats.append(sign_lag.reshape(-1, 1))

    X_out = np.hstack(feats)
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
    return X_out, max_w


# ─────────────────────────────────────────────────────────────
#  Kernel Ridge Regression via Random Fourier Features
# ─────────────────────────────────────────────────────────────

def _rff_transform(
    X: np.ndarray, n_components: int = 500, gamma: float = 1.0, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Fourier Features approximating Gaussian RBF kernel."""
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.randn(d, n_components) * np.sqrt(2.0 * gamma)
    b = rng.uniform(0, 2.0 * math.pi, n_components)
    Z = np.sqrt(2.0 / n_components) * np.cos(X @ W + b)
    return Z, W, b


def _rff_apply(X: np.ndarray, W: np.ndarray, b: np.ndarray, n_components: int) -> np.ndarray:
    return np.sqrt(2.0 / n_components) * np.cos(X @ W + b)


def _fit_kernel_ridge(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
    gamma: float = 1.0, l2: float = 1.0, n_rff: int = 500,
) -> np.ndarray:
    """Kernel ridge regression via Random Fourier Features (non-linear)."""
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0) + 1e-8
    Z_tr_raw = (X_train - mu) / sd
    Z_ev_raw = (X_eval - mu) / sd
    Z_tr, W, b = _rff_transform(Z_tr_raw, n_components=n_rff, gamma=gamma)
    Z_ev = _rff_apply(Z_ev_raw, W, b, n_rff)
    # Add bias column
    Z_tr = np.concatenate([np.ones((len(Z_tr), 1)), Z_tr], axis=1)
    Z_ev = np.concatenate([np.ones((len(Z_ev), 1)), Z_ev], axis=1)
    reg = np.eye(Z_tr.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.solve(Z_tr.T @ Z_tr + l2 * reg, Z_tr.T @ y_train)
    return Z_ev @ w


# ─────────────────────────────────────────────────────────────
#  k-NN Analogy Predictor (uses actual future deltas)
# ─────────────────────────────────────────────────────────────

def _knn_analog_predictor(
    X_train: np.ndarray, delta_train: np.ndarray,
    X_eval: np.ndarray, y_now_eval: np.ndarray,
    k: int = 20, alpha: float = 0.5,
) -> np.ndarray:
    """
    Find k nearest past analogies in feature space
    and use their actual future deltas (weighted by distance).

    This is a non-parametric, non-linear predictor that captures
    complex patterns through local manifold structure.
    """
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0) + 1e-8
    Z_tr = (X_train - mu) / sd
    Z_ev = (X_eval - mu) / sd

    tree = cKDTree(Z_tr)
    k_safe = min(k, len(Z_tr) - 1)

    pred_delta = np.zeros(len(Z_ev))
    for i in range(len(Z_ev)):
        dists, idxs = tree.query(Z_ev[i], k=k_safe)
        if k_safe == 1:
            dists = np.array([dists])
            idxs = np.array([idxs])
        # Gaussian kernel weights
        weights = np.exp(-alpha * dists ** 2)
        w_sum = weights.sum() + 1e-15
        pred_delta[i] = float(np.dot(weights, delta_train[idxs]) / w_sum)

    return y_now_eval + pred_delta


# ─────────────────────────────────────────────────────────────
#  Pattern-Based Classification (v3)
# ─────────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Estimate mutual information between each feature and binary target."""
    n, d = X.shape
    mi = np.zeros(d)
    y_bool = y.astype(bool)
    for j in range(d):
        xj = X[:, j]
        # discretize continuous feature into bins
        edges = np.percentile(xj[np.isfinite(xj)], np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            continue
        bins = np.digitize(xj, edges[:-1]) - 1
        bins = np.clip(bins, 0, len(edges) - 2)
        # joint and marginal counts
        n_b = len(edges) - 1
        p_y1 = y.mean()
        p_y0 = 1.0 - p_y1
        for b in range(n_b):
            mask_b = (bins == b)
            p_b = mask_b.mean()
            if p_b < 1e-12:
                continue
            p_b_y1 = (mask_b & y_bool).mean()
            p_b_y0 = (mask_b & ~y_bool).mean()
            if p_b_y1 > 1e-12:
                mi[j] += p_b_y1 * np.log(p_b_y1 / (p_b * p_y1 + 1e-15) + 1e-15)
            if p_b_y0 > 1e-12:
                mi[j] += p_b_y0 * np.log(p_b_y0 / (p_b * p_y0 + 1e-15) + 1e-15)
    return mi


def _logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray,
                   l2: float, sample_w: np.ndarray) -> float:
    """Weighted logistic loss + L2 regularization."""
    p = _sigmoid(X @ w)
    eps = 1e-12
    loss = -np.mean(sample_w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    loss += 0.5 * l2 * np.dot(w[1:], w[1:])  # no penalty on bias
    return loss


def _logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray,
                   l2: float, sample_w: np.ndarray) -> np.ndarray:
    """Gradient of weighted logistic loss."""
    p = _sigmoid(X @ w)
    grad = X.T @ (sample_w * (p - y)) / len(y)
    grad[1:] += l2 * w[1:]
    return grad


def _fit_logistic_lbfgs(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
    l2: float = 1.0, sample_weights: np.ndarray = None,
) -> np.ndarray:
    """Logistic regression via L-BFGS-B solver. Returns P(y=1) for X_eval."""
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0) + 1e-8
    Z_tr = np.concatenate([np.ones((len(X_train), 1)), (X_train - mu) / sd], axis=1)
    Z_ev = np.concatenate([np.ones((len(X_eval), 1)), (X_eval - mu) / sd], axis=1)

    if sample_weights is None:
        sw = np.ones(len(y_train))
    else:
        sw = sample_weights / sample_weights.mean()  # normalize

    w0 = np.zeros(Z_tr.shape[1])
    result = minimize(
        _logistic_loss, w0,
        args=(Z_tr, y_train, l2, sw),
        jac=_logistic_grad,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-8},
    )
    return _sigmoid(Z_ev @ result.x)


def _knn_classify_proba(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, k: int = 20, alpha: float = 0.5,
) -> np.ndarray:
    """k-NN classifier returning P(y=1) via distance-weighted vote."""
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0) + 1e-8
    Z_tr, Z_ev = (X_train - mu) / sd, (X_eval - mu) / sd
    tree = cKDTree(Z_tr)
    k_safe = min(k, len(Z_tr) - 1)
    probs = np.zeros(len(Z_ev))
    for i in range(len(Z_ev)):
        dists, idxs = tree.query(Z_ev[i], k=k_safe)
        if k_safe == 1:
            dists, idxs = np.array([dists]), np.array([idxs])
        weights = np.exp(-alpha * dists ** 2)
        probs[i] = float(np.dot(weights, y_train[idxs]) / (weights.sum() + 1e-15))
    return np.clip(probs, 0.001, 0.999)


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Manual AUC-ROC computation."""
    si = np.argsort(-y_prob)
    ys = y_true[si]
    np_, nn_ = float(np.sum(y_true == 1)), float(np.sum(y_true == 0))
    if np_ == 0 or nn_ == 0:
        return 0.5
    tp_c, fp_c = 0.0, 0.0
    tpr_a, fpr_a = [0.0], [0.0]
    for j in range(len(ys)):
        if ys[j] == 1:
            tp_c += 1
        else:
            fp_c += 1
        tpr_a.append(tp_c / np_)
        fpr_a.append(fp_c / nn_)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trapz(tpr_a, fpr_a))


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
    auc = _roc_auc(y_true, y_prob)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction classification v3")
    parser.add_argument("--micro-db", default="/home/llewyn/crypto-scalping/data/live/microstructure.duckdb")
    parser.add_argument("--tail-db", default="/home/llewyn/crypto-scalping/data/live/tail_risk.duckdb")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--close-alias", default="zeta_px")
    parser.add_argument("--bar-minutes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--top-k-features", type=int, default=25, help="MI 기반 상위 K 피처만 사용")
    parser.add_argument("--output-csv", default="/home/llewyn/crypto-scalping/analysis/merged_geometric_dataset.csv")
    parser.add_argument("--pred-csv", default="/home/llewyn/crypto-scalping/analysis/pred_30m_direction.csv")
    args = parser.parse_args()

    merged, micro_map, tail_map, close_alias, target_alias = build_merged_dataset(
        micro_db_path=args.micro_db, tail_db_path=args.tail_db,
        symbol=args.symbol, close_alias=args.close_alias,
        bar_minutes=args.bar_minutes, horizon_minutes=args.horizon,
    )

    feature_cols = [c for c in merged.columns if c not in {"ts", target_alias}]
    X_base = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_target_raw = merged[target_alias].to_numpy(dtype=float)
    y_now_raw = merged[close_alias].to_numpy(dtype=float)
    t_seconds = (merged["ts"].astype("int64") / 1e9).to_numpy(dtype=float)
    t_seconds = t_seconds - t_seconds[0]

    # ━━ 1. Multi-scale features (with AR) ━━
    print("[...] Building multi-scale + AR features...")
    X_ms, offset = _build_multiscale_features(X_base, y_now_raw, t_seconds)
    y_target = y_target_raw[offset:]
    y_now = y_now_raw[offset:]
    ts_series = merged["ts"].iloc[offset:].reset_index(drop=True)
    delta_true = y_target - y_now
    direction = (delta_true > 0).astype(float)
    n = len(X_ms)
    print(f"[...] Raw features: {X_ms.shape[1]} dims, {n} samples, up={direction.mean()*100:.1f}%")

    # ━━ 2. Train/Val/Test split ━━
    tr_end, va_end = int(n * 0.70), int(n * 0.85)
    X_tr, X_va = X_ms[:tr_end], X_ms[tr_end:va_end]
    y_tr, y_va, y_te = direction[:tr_end], direction[tr_end:va_end], direction[va_end:]

    # ━━ 3. Feature Selection (Mutual Information) ━━
    print("[...] Computing Mutual Information for feature selection...")
    mi_scores = _mutual_information(X_tr, y_tr, n_bins=30)
    top_k = min(args.top_k_features, X_ms.shape[1])
    top_idx = np.argsort(-mi_scores)[:top_k]
    X_sel = X_ms[:, top_idx]
    X_tr_s, X_va_s = X_sel[:tr_end], X_sel[tr_end:va_end]
    print(f"[...] Selected top-{top_k} features (MI range: {mi_scores[top_idx[0]]:.6f} ~ {mi_scores[top_idx[-1]]:.6f})")

    # ━━ 4. Time decay weights (최근 데이터 우선) ━━
    half_life = tr_end * 0.3
    time_weights = np.exp(np.log(2) * np.arange(tr_end) / half_life)
    time_weights /= time_weights.mean()

    # ━━ 5. Logistic Ridge (L-BFGS-B) ━━
    print("[...] Training Logistic (L-BFGS-B)...")
    best_lg = {"l2": None, "acc": 0.0, "prob": None}
    for l2 in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        pv = _fit_logistic_lbfgs(X_tr_s, y_tr, X_va_s, l2=l2, sample_weights=time_weights)
        acc = float(np.mean((pv >= 0.5) == y_va))
        if acc > best_lg["acc"]:
            pa = _fit_logistic_lbfgs(X_tr_s, y_tr, X_sel, l2=l2, sample_weights=time_weights)
            best_lg = {"l2": l2, "acc": acc, "prob": pa}
    prob_log = best_lg["prob"] if best_lg["prob"] is not None else np.full(n, y_tr.mean())

    # ━━ 6. Kernel Logistic (RFF + L-BFGS-B) ━━
    print("[...] Training Kernel Logistic (RFF + L-BFGS-B)...")
    n_rff = min(200, top_k * 4)
    best_kl = {"g": None, "l2": None, "acc": 0.0, "prob": None}
    for g in [0.01, 0.1, 0.5, 1.0]:
        for l2 in [0.01, 0.1, 1.0]:
            try:
                mu_s = np.mean(X_tr_s, axis=0)
                sd_s = np.std(X_tr_s, axis=0) + 1e-8
                Z_tr_n = (X_tr_s - mu_s) / sd_s
                Z_va_n = (X_va_s - mu_s) / sd_s
                Z_all_n = (X_sel - mu_s) / sd_s
                Ztr_rff, W_rff, b_rff = _rff_transform(Z_tr_n, n_components=n_rff, gamma=g)
                Zva_rff = _rff_apply(Z_va_n, W_rff, b_rff, n_rff)
                pv = _fit_logistic_lbfgs(Ztr_rff, y_tr, Zva_rff, l2=l2, sample_weights=time_weights)
                acc = float(np.mean((pv >= 0.5) == y_va))
                if acc > best_kl["acc"]:
                    Zall_rff = _rff_apply(Z_all_n, W_rff, b_rff, n_rff)
                    pa = _fit_logistic_lbfgs(Ztr_rff, y_tr, Zall_rff, l2=l2, sample_weights=time_weights)
                    best_kl = {"g": g, "l2": l2, "acc": acc, "prob": pa}
            except Exception:
                continue
    prob_kl = best_kl["prob"] if best_kl["prob"] is not None else np.full(n, y_tr.mean())

    # ━━ 7. k-NN Classifier ━━
    print("[...] Running k-NN classifier...")
    best_knn = {"k": None, "a": None, "acc": 0.0, "prob": None}
    for k in [10, 20, 50, 100]:
        for a in [0.1, 0.5, 1.0, 2.0]:
            pv = _knn_classify_proba(X_tr_s, y_tr, X_va_s, k=k, alpha=a)
            acc = float(np.mean((pv >= 0.5) == y_va))
            if acc > best_knn["acc"]:
                pa = _knn_classify_proba(X_tr_s, y_tr, X_sel, k=k, alpha=a)
                best_knn = {"k": k, "a": a, "acc": acc, "prob": pa}
    prob_knn = best_knn["prob"] if best_knn["prob"] is not None else np.full(n, y_tr.mean())

    # ━━ 8. Ensemble ━━
    print("[...] Optimizing ensemble...")
    prob_base = np.full(n, y_tr.mean())
    mdl = {"baseline": prob_base, "logistic": prob_log, "kernel_log": prob_kl, "knn": prob_knn}
    mdl_va = {k: v[tr_end:va_end] for k, v in mdl.items()}
    mnames = list(mdl.keys())

    best_w = np.array([1.0, 0.0, 0.0, 0.0])
    best_ea = 0.0
    for w0 in range(11):
        for w1 in range(11 - w0):
            for w2 in range(11 - w0 - w1):
                w3 = 10 - w0 - w1 - w2
                w = np.array([w0, w1, w2, w3], dtype=float) / 10.0
                p = sum(w[i] * mdl_va[mnames[i]] for i in range(4))
                ac = float(np.mean(((p >= 0.5).astype(float)) == y_va))
                if ac > best_ea:
                    best_w, best_ea = w.copy(), ac

    prob_ens = sum(best_w[i] * mdl[mnames[i]] for i in range(4))
    pred_ens = (prob_ens >= 0.5).astype(float)

    # ━━ 9. Metrics ━━
    am = {}
    for name in mnames:
        pred = (mdl[name] >= 0.5).astype(float)
        am[name] = {"all": _clf_metrics(direction, pred, mdl[name]),
                     "test": _clf_metrics(y_te, pred[va_end:], mdl[name][va_end:])}
    am["ensemble"] = {"all": _clf_metrics(direction, pred_ens, prob_ens),
                       "test": _clf_metrics(y_te, pred_ens[va_end:], prob_ens[va_end:])}
    m_tr = _clf_metrics(y_tr, pred_ens[:tr_end], prob_ens[:tr_end])
    m_va = _clf_metrics(y_va, pred_ens[tr_end:va_end], prob_ens[tr_end:va_end])

    # ━━ 10. Output ━━
    merged.to_csv(args.output_csv, index=False)
    pdf = pd.DataFrame({"ts": ts_series, "y_now": y_now, "y_target": y_target,
                         "delta": delta_true, "dir_true": direction,
                         "prob_ens": prob_ens, "pred_ens": pred_ens})
    for nm in mnames:
        pdf[f"prob_{nm}"] = mdl[nm]
    pdf.to_csv(args.pred_csv, index=False)

    print("\n" + "=" * 70)
    print("  Direction Classification Pipeline v3 (Pattern-Based)")
    print("=" * 70)
    print(f"[OK] samples={n}  raw_features={X_ms.shape[1]}  selected={top_k}")
    print(f"[OK] horizon={args.horizon}min  up={direction.mean()*100:.1f}%  down={(1-direction.mean())*100:.1f}%")
    print(f"[OK] logistic l2={best_lg['l2']}")
    print(f"[OK] kernel_log g={best_kl['g']} l2={best_kl['l2']} rff={n_rff}")
    print(f"[OK] knn k={best_knn['k']} a={best_knn['a']}")
    ws = "  ".join(f"{mnames[i]}={best_w[i]:.2f}" for i in range(4))
    print(f"[OK] ensemble: {ws}")

    print(f"\n-- Top-{top_k} Features (by MI) --")
    for rank, idx in enumerate(top_idx[:10]):
        print(f"  #{rank+1:2d}  col_{idx:03d}  MI={mi_scores[idx]:.6f}")

    print("\n-- Test Set Metrics --")
    for nm in mnames + ["ensemble"]:
        m = am[nm]["test"]
        print(f"  [{nm:12s}]  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

    print("\n-- Ensemble Split --")
    for nm, m in [("train", m_tr), ("valid", m_va), ("test", am["ensemble"]["test"])]:
        print(f"  [{nm}]  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}")

    # ━━ 11. 최신 방향 예측 출력 ━━
    print("\n" + "=" * 70)
    print("  ★ 최신 방향 예측 (Latest Direction Predictions) ★")
    print("=" * 70)
    last_n = 10
    for i in range(max(0, n - last_n), n):
        ts_str = str(ts_series.iloc[i])[:19]
        p_ens = prob_ens[i]
        dir_pred = "▲ 상승" if p_ens >= 0.5 else "▼ 하락"
        conf = abs(p_ens - 0.5) * 200  # 0~100% confidence
        actual = ""
        if not np.isnan(delta_true[i]):
            actual_dir = "▲" if delta_true[i] > 0 else "▼"
            correct = "✓" if (delta_true[i] > 0) == (p_ens >= 0.5) else "✗"
            actual = f"  실제={actual_dir} {correct}"
        # per-model
        p_lg = prob_log[i]
        p_kl = prob_kl[i]
        p_kn = prob_knn[i]
        print(f"  {ts_str}  now={y_now[i]:.2f}  → {dir_pred}  prob={p_ens:.4f}  conf={conf:.1f}%{actual}")
        print(f"    logistic={p_lg:.4f}  kernel={p_kl:.4f}  knn={p_kn:.4f}")

    # 마지막 행 = 가장 최신 예측
    last_p = prob_ens[-1]
    last_dir = "▲ 상승" if last_p >= 0.5 else "▼ 하락"
    last_conf = abs(last_p - 0.5) * 200
    last_ts = str(ts_series.iloc[-1])[:19]
    print(f"\n  ★★★ 최종 예측: {last_ts} 기준 {args.horizon}분 후 → {last_dir} (확률={last_p:.4f}, 확신도={last_conf:.1f}%) ★★★")

    print("\n[Micro map]")
    for k, v in micro_map.items():
        print(f"  {k} -> {v}")
    print("[Tail map]")
    for k, v in tail_map.items():
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
