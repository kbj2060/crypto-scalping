#!/usr/bin/env python3
"""Research-only competing-risk rescue exit for live ETH Omega4.6.1.

This experiment does not modify the live adapter, runtime configuration, or artifacts.  It tests
whether training on direction candidates before the quality gate provides useful exit information.

Protocol
--------
* TRAIN: 2025-01-01..2025-09-30.
* VAL:   2025-10-01..2025-12-31.  The frozen parent OOF predictions begin on 2025-10-01, so the
  project-default 2025-09-01 VAL start cannot be used without overlapping parent training.
* OOS:   2026-01-01..2026-03-31, loaded only after a candidate clears the predeclared VAL gate.
* A: frozen SLTP + exit_head@0.95 baseline.
* B: hazard models trained on post-quality candidate episodes.
* C: hazard models trained on pre-quality direction candidate episodes.
* D: same pre-quality episodes, weighted toward the live quality distribution.

The target is a seven-class competing-risk distribution: TP/SL first within 12, 48, or 384 bars,
plus right-censoring at 384 bars.  A bootstrap ensemble supplies uncertainty.  The rescue layer may
only exit when every ensemble member assigns sufficient SL probability and even the most optimistic
continuation-value estimate is below immediate liquidation by a margin.  Otherwise it abstains and
the frozen SLTP lifecycle remains the owner.  The frozen exit head is omitted from this harness
because the audited live threshold 0.95 is empirically inert; this makes the baseline exactly the
observed pure-SLTP behavior and avoids expensive no-op neural inference.

Training labels use future rows only to describe each historical episode outcome.  Inference uses
only the current and already-closed bars.  Saved trade ledgers and saved exit timestamps are never
inputs.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "eth_omega461_competing_risk_rescue_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"
HORIZON = 384
TIME_BINS = (12, 48)
N_BOOTSTRAPS = 3
SEED = 260724
EXIT_THRESHOLD = 0.95
FEATURE_MARKET_COLS = (
    "log_return",
    "volatility_z",
    "atr_pct_rank_288",
    "rsi",
    "mean_reversion_z",
    "regime_persistence",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "regime_trending",
    "sig_trend_health",
    "last_funding_rate",
    "funding_z_score",
    "funding_pressure",
    "funding_oi_divergence",
    "ou_funding_z",
    "ou_halflife",
)
POSITION_COLS = (
    "pos_side",
    "pos_hold_bars",
    "pos_unrealized",
    "pos_mfe",
    "pos_mae",
    "pos_giveback",
    "pos_dist_to_tp",
    "pos_dist_to_sl",
    "pos_notional",
    "pos_leverage",
    "pos_exposure",
    "pos_tp",
    "pos_sl",
)
SOURCE_SUFFIXES = (
    "router_confidence",
    "router_margin",
    "dir_p_cash",
    "dir_p_long",
    "dir_p_short",
    "dir_confidence",
    "dir_side_edge",
    "dir_trade_prob",
    "quality_p_cash",
    "quality_p_long",
    "quality_p_short",
    "quality_for_action",
)
CLASS_NAMES = (
    "tp_soon",
    "tp_mid",
    "tp_late",
    "sl_soon",
    "sl_mid",
    "sl_late",
    "censored",
)
SL_CLASSES = frozenset({3, 4, 5})


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _load_aligned_source(frame: pd.DataFrame, pred_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(pred_csv, low_memory=False)
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    keep = set(src["timestamp"])
    aligned_frame = frame[frame["timestamp"].isin(keep)].reset_index(drop=True)
    src = src[src["timestamp"].isin(set(aligned_frame["timestamp"]))].reset_index(drop=True)
    if len(src) != len(aligned_frame) or not src["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"prediction/frame timestamp mismatch: {len(src)} vs {len(aligned_frame)}")
    for col in src.columns:
        if str(src[col].dtype).lower().startswith("str"):
            src[col] = src[col].astype(object)
    return aligned_frame, src


def prepare_split(
    name: str,
    cfg: dict[str, Any],
    frame: pd.DataFrame,
    pred_csv: Path,
    *,
    oof: bool,
    pre_quality: bool,
) -> dict[str, Any]:
    frame, src = _load_aligned_source(frame, pred_csv)
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle["base_cols"])
    source_for_dec = src.copy()
    prefix = _prefix(oof)
    if pre_quality:
        source_for_dec[f"{prefix}final_action"] = pd.to_numeric(
            source_for_dec[f"{prefix}dir_action"], errors="raise"
        ).to_numpy(dtype=np.int64)

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(source_for_dec, oof=oof)
    dec, _ = atr_eval._apply_atr_safety_sltp(
        dec_base,
        frame,
        atr_window=cfg["atr_window"],
        tp_mult=cfg["tp_mult"],
        sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"],
        min_sl=cfg["min_sl"],
        max_tp=cfg["max_tp"],
        max_sl=cfg["max_sl"],
    )
    atr_pct = atr_eval._atr_pct(frame, cfg["atr_window"])
    loaded = parent._load_payloads(bundle["models"], device=sweep.DEVICE)
    with open(cfg["sidecar_pkl"], "rb") as handle:
        sidecar = pickle.load(handle)
    risk_features = rs._risk_feature_frame(
        frame,
        source_for_dec,
        dec,
        base_cols,
        atr_pct=atr_pct,
        feature_mode=sidecar["risk_feature_mode"],
    )
    risk_x, _ = rs._feature_matrix(risk_features, sidecar["feature_columns"])
    sides = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    if sidecar["side_split_model"]:
        score = rs._predict_side_split_models(sidecar["model"], risk_x, sides)
    else:
        score = np.asarray(sidecar["model"].predict(risk_x), dtype=np.float64)
    mapping = sidecar["selected_mapping"]
    margin = rs._risk_margins(
        dec,
        score,
        train_q50=sidecar["train_score_q50"],
        train_iqr=sidecar["train_score_iqr"],
        **{key: mapping[key] for key in rs.MARGIN_CFG_KEYS},
    )
    if sidecar["dynamic_leverage"]:
        leverage = rs._risk_leverage(
            dec,
            score,
            train_q50=sidecar["train_score_q50"],
            train_iqr=sidecar["train_score_iqr"],
            **{key: mapping[key] for key in rs.LEVERAGE_CFG_KEYS},
        )
    else:
        leverage = np.ones(len(dec), dtype=np.float64)
    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(x, loaded)
    feature_arrays: dict[str, np.ndarray] = {}
    for col in FEATURE_MARKET_COLS:
        feature_arrays[f"market_{col}"] = (
            pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if col in frame
            else np.zeros(len(frame), dtype=np.float64)
        )
    for col in hard.ROUTE_COLS:
        feature_arrays[f"route_{col}"] = (
            pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if col in frame
            else np.zeros(len(frame), dtype=np.float64)
        )
    for suffix in SOURCE_SUFFIXES:
        col = f"{prefix}{suffix}"
        feature_arrays[f"source_{suffix}"] = pd.to_numeric(src[col], errors="raise").to_numpy(dtype=np.float64)
    return {
        "name": name,
        "frame": frame,
        "src": src,
        "prefix": prefix,
        "x": x,
        "dec": dec,
        "margin": np.asarray(margin, dtype=np.float64),
        "leverage": np.asarray(leverage, dtype=np.float64),
        "loaded": loaded,
        "base_np": base_np,
        "exit_runtime": exit_runtime,
        "pos_idx": pos_idx,
        "route": hard._route_id(frame),
        "fee": omega._load_fee_slip()[0],
        "slip": omega._load_fee_slip()[1],
        "notional_scaled_sltp": bool(sidecar["notional_scaled_sltp"]),
        "quality_threshold": float(cfg["quality_threshold"]),
        "pre_quality": bool(pre_quality),
        "feature_arrays": feature_arrays,
    }


def _move(close: float, *, side: int, entry_price: float, slip_eff: float) -> float:
    if side > 0:
        return float((close * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12))
    return float((entry_price - close * (1.0 + slip_eff)) / max(entry_price, 1.0e-12))


def _class_id(cause: str, remaining: int) -> int:
    if cause == "censored":
        return 6
    offset = 0 if cause == "take_profit" else 3
    if remaining <= TIME_BINS[0]:
        return offset
    if remaining <= TIME_BINS[1]:
        return offset + 1
    return offset + 2


def _feature_row(
    prepared: dict[str, Any],
    *,
    row_i: int,
    side: int,
    hold: int,
    move: float,
    mfe: float,
    mae: float,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
) -> dict[str, float]:
    giveback = (mfe - move) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
    row = {name: float(values[row_i]) for name, values in prepared["feature_arrays"].items()}
    row.update(
        {
            "pos_side": float(side),
            "pos_hold_bars": float(hold),
            "pos_unrealized": float(move),
            "pos_mfe": float(mfe),
            "pos_mae": float(mae),
            "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
            "pos_dist_to_tp": float(take_profit - move),
            "pos_dist_to_sl": float(move + abs(stop_loss)),
            "pos_notional": float(notional),
            "pos_leverage": float(leverage),
            "pos_exposure": float(notional * leverage),
            "pos_tp": float(take_profit),
            "pos_sl": float(stop_loss),
        }
    )
    return row


def build_episode_dataset(prepared: dict[str, Any], *, quality_weighted: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = prepared["frame"]
    dec = prepared["dec"]
    src = prepared["src"]
    prefix = prepared["prefix"]
    arrays = {col: pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(prepared["fee"]) * sweep.COST_MULT
    slip_eff = float(prepared["slip"]) * sweep.COST_MULT
    rows: list[dict[str, Any]] = []
    episode_id = 0
    missed_entries = 0
    i = 0
    last_i = len(frame) - 2
    while i < last_i:
        if not bool(active[i]):
            i += 1
            continue
        drow = dec.iloc[i]
        side = int(drow.get("side", 0) or 0)
        if side == 0:
            i += 1
            continue
        filled, entry_price, _entry_fee, _route = omega._try_execution(
            arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            missed_entries += 1
            i += 1
            continue
        entry_i = min(i + 1, len(frame) - 1)
        margin = float(prepared["margin"][i])
        leverage = float(prepared["leverage"][i])
        notional = margin * leverage
        if notional <= 0.0:
            i += 1
            continue
        take_profit = float(drow.get("take_profit", 0.0) or 0.0)
        stop_loss = float(drow.get("stop_loss", 0.0) or 0.0)
        if prepared["notional_scaled_sltp"]:
            take_profit *= notional
            stop_loss *= notional
        end_limit = min(entry_i + HORIZON, last_i)
        cause = "censored"
        end_i = end_limit
        trajectory: list[tuple[int, float, float, float]] = []
        mfe = 0.0
        mae = 0.0
        for row_i in range(entry_i, end_limit + 1):
            current_move = _move(float(arrays["close"][row_i]), side=side, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, current_move)
            mae = min(mae, current_move)
            trajectory.append((row_i, current_move, mfe, mae))
            if stop_loss > 0.0 and current_move <= -abs(stop_loss):
                cause, end_i = "stop_loss", row_i
                break
            if take_profit > 0.0 and current_move >= take_profit:
                cause, end_i = "take_profit", row_i
                break
        terminal_move = float(trajectory[-1][1])
        episode_len = len(trajectory)
        entry_quality = float(pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").iloc[i])
        proximity_weight = 1.0
        if quality_weighted:
            ratio = entry_quality / max(float(prepared["quality_threshold"]), 1.0e-8)
            proximity_weight = float(np.clip(ratio, 0.10, 1.0) ** 2)
        for row_i, current_move, row_mfe, row_mae in trajectory:
            remaining = max(end_i - row_i, 0)
            feature = _feature_row(
                prepared,
                row_i=row_i,
                side=side,
                hold=max(row_i - entry_i, 0),
                move=current_move,
                mfe=row_mfe,
                mae=row_mae,
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            feature.update(
                {
                    "episode_id": int(episode_id),
                    "entry_signal_i": int(i),
                    "entry_timestamp": frame["timestamp"].iloc[i],
                    "label": int(_class_id(cause, remaining)),
                    "cause": cause,
                    "terminal_move": terminal_move,
                    "remaining_bars": int(remaining),
                    "sample_weight": float(proximity_weight / max(episode_len, 1)),
                    "entry_quality": entry_quality,
                }
            )
            rows.append(feature)
        episode_id += 1
        i = max(end_i + 1, i + 1)
    if not rows:
        raise RuntimeError(f"{prepared['name']}: empty competing-risk episode dataset")
    data = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diag = {
        "component": prepared["name"],
        "pre_quality": bool(prepared["pre_quality"]),
        "quality_weighted": bool(quality_weighted),
        "rows": int(len(data)),
        "episodes": int(data["episode_id"].nunique()),
        "missed_entries": int(missed_entries),
        "cause_counts_episodes": {
            str(key): int(value)
            for key, value in data.groupby("episode_id", sort=False)["cause"].first().value_counts().to_dict().items()
        },
        "class_counts_rows": {
            CLASS_NAMES[int(key)]: int(value) for key, value in data["label"].value_counts().sort_index().to_dict().items()
        },
        "entry_quality_quantiles": {
            str(q): float(data.groupby("episode_id", sort=False)["entry_quality"].first().quantile(q))
            for q in (0.10, 0.50, 0.90)
        },
    }
    return data, diag


def _feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "episode_id",
        "entry_signal_i",
        "entry_timestamp",
        "label",
        "cause",
        "terminal_move",
        "remaining_bars",
        "sample_weight",
        "entry_quality",
    }
    return [col for col in data.columns if col not in excluded]


def fit_bootstrap_ensemble(data: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    feature_cols = _feature_columns(data)
    x = data[feature_cols].to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=np.int64)
    episode = data["episode_id"].to_numpy(dtype=np.int64)
    base_weight = data["sample_weight"].to_numpy(dtype=np.float64)
    class_weight = np.ones(len(CLASS_NAMES), dtype=np.float64)
    for label in np.unique(y):
        class_weight[int(label)] = float(len(y) / (len(np.unique(y)) * max(np.sum(y == label), 1)))
    weights = base_weight * class_weight[y]
    payoff_by_class = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for label in np.unique(y):
        mask = y == label
        payoff_by_class[int(label)] = float(np.average(data.loc[mask, "terminal_move"], weights=base_weight[mask]))

    groups = np.unique(episode)
    rng = np.random.default_rng(int(seed))
    models: list[HistGradientBoostingClassifier] = []
    for boot in range(N_BOOTSTRAPS):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        index_parts = [np.flatnonzero(episode == group) for group in sampled_groups]
        idx = np.concatenate(index_parts)
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(seed + boot),
        )
        model.fit(x[idx], y[idx], sample_weight=weights[idx])
        models.append(model)
    return {
        "models": models,
        "feature_columns": feature_cols,
        "payoff_by_class": payoff_by_class,
        "class_names": CLASS_NAMES,
        "horizon": HORIZON,
        "time_bins": TIME_BINS,
    }


def _aligned_proba(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(CLASS_NAMES)), dtype=np.float64)
    for idx, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, idx]
    return out


def predict_rescue(bundle: dict[str, Any], row: dict[str, float]) -> tuple[float, float]:
    x = np.asarray([[float(row[col]) for col in bundle["feature_columns"]]], dtype=np.float64)
    expected_values: list[float] = []
    sl_probs: list[float] = []
    for model in bundle["models"]:
        probs = _aligned_proba(model, x)[0]
        expected_values.append(float(np.dot(probs, bundle["payoff_by_class"])))
        sl_probs.append(float(sum(probs[idx] for idx in SL_CLASSES)))
    return float(max(expected_values)), float(min(sl_probs))


def precompute_rescue_path(
    prepared: dict[str, Any],
    bundle: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    entry_i: int,
    side: int,
    entry_price: float,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
    slip_eff: float,
) -> dict[int, tuple[float, float]]:
    """Batch the model work for one causal position path.

    The path state at bar t depends only on entry state and bars <= t.  Computing those states in
    one forward loop is equivalent to recomputing them bar by bar, while avoiding thousands of
    single-row tree calls.  Future labels/outcomes are not included in the feature rows.
    """
    end_i = min(int(entry_i) + HORIZON, len(arrays["close"]) - 2)
    indices: list[int] = []
    rows: list[dict[str, float]] = []
    mfe = 0.0
    mae = 0.0
    for row_i in range(int(entry_i), end_i + 1):
        current_move = _move(
            float(arrays["close"][row_i]), side=side, entry_price=entry_price, slip_eff=slip_eff
        )
        mfe = max(mfe, current_move)
        mae = min(mae, current_move)
        indices.append(row_i)
        rows.append(
            _feature_row(
                prepared,
                row_i=row_i,
                side=side,
                hold=max(row_i - int(entry_i), 0),
                move=current_move,
                mfe=mfe,
                mae=mae,
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
        )
        if stop_loss > 0.0 and current_move <= -abs(stop_loss):
            break
        if take_profit > 0.0 and current_move >= take_profit:
            break
    x = np.asarray(
        [[float(row[col]) for col in bundle["feature_columns"]] for row in rows], dtype=np.float64
    )
    if bundle.get("kind") == "distributional_continuation":
        continuation_ucb = np.asarray(bundle["models"]["q90"].predict(x), dtype=np.float64)
        sl_probability_lcb = np.ones(len(x), dtype=np.float64)
    else:
        expected_by_model: list[np.ndarray] = []
        sl_by_model: list[np.ndarray] = []
        for model in bundle["models"]:
            probs = _aligned_proba(model, x)
            expected_by_model.append(probs @ bundle["payoff_by_class"])
            sl_by_model.append(probs[:, list(sorted(SL_CLASSES))].sum(axis=1))
        continuation_ucb = np.max(np.vstack(expected_by_model), axis=0)
        sl_probability_lcb = np.min(np.vstack(sl_by_model), axis=0)
    return {
        int(row_i): (float(continuation_ucb[idx]), float(sl_probability_lcb[idx]))
        for idx, row_i in enumerate(indices)
    }


def _counterfactual_cause(
    arrays: dict[str, np.ndarray],
    *,
    start_i: int,
    side: int,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    slip_eff: float,
) -> str:
    end_i = min(start_i + HORIZON, len(arrays["close"]) - 2)
    for row_i in range(start_i, end_i + 1):
        current_move = _move(float(arrays["close"][row_i]), side=side, entry_price=entry_price, slip_eff=slip_eff)
        if stop_loss > 0.0 and current_move <= -abs(stop_loss):
            return "stop_loss"
        if take_profit > 0.0 and current_move >= take_profit:
            return "take_profit"
    return "censored"


@torch.no_grad()
def replay_router(
    frame: pd.DataFrame,
    components: dict[str, dict[str, Any]],
    *,
    rescue_bundles: dict[str, dict[str, Any]] | None,
    sl_probability_min: float,
    value_margin: float,
    persistence: int,
    entry_notional_multiplier: float = 1.0,
    max_entry_notional: float | None = None,
    chop_soft_size_threshold: float | None = None,
    rearm_mode: str = "none",
    rearm_bars: int = 0,
    cost_mult: float = sweep.COST_MULT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if rearm_mode not in {"none", "cooldown", "signal_reset"}:
        raise ValueError(f"unsupported rearm_mode={rearm_mode!r}")
    arrays = {col: pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close")}
    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    active_masks = {name: omega._active(comp["dec"]) for name, comp in components.items()}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    active_name = ""
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    rescue_streak = 0
    rescue_path: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    rescue_counterfactual: Counter[str] = Counter()
    entry_chop_probability = float("nan")
    entry_sizing_multiplier = 1.0
    rearm_block_until_i = -1
    rearm_wait_for_signal_reset = False
    rearm_blocked_bars = 0
    rearm_blocked_signal_bars = 0

    for i in range(0, len(frame) - 2):
        if pos != 0:
            comp = components[active_name]
            current_move = _move(float(arrays["close"][i]), side=pos, entry_price=entry_price, slip_eff=slip_eff)
            mfe = max(mfe, current_move)
            mae = min(mae, current_move)
            eq = cash * (1.0 + current_move * notional)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            continuation_ucb = float("nan")
            sl_probability_lcb = float("nan")
            if take_profit > 0.0 and current_move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and current_move <= -abs(stop_loss):
                reason = "stop_loss"
            elif rescue_bundles is not None:
                continuation_ucb, sl_probability_lcb = rescue_path.get(i, (float("nan"), float("nan")))
                rescue_hit = (
                    sl_probability_lcb >= float(sl_probability_min)
                    and current_move >= continuation_ucb + float(value_margin)
                )
                rescue_streak = rescue_streak + 1 if rescue_hit else 0
                if rescue_streak >= int(persistence):
                    reason = "hazard_rescue"
                    rescue_counterfactual[
                        _counterfactual_cause(
                            arrays,
                            start_i=i,
                            side=pos,
                            entry_price=entry_price,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            slip_eff=slip_eff,
                        )
                    ] += 1
            if reason:
                filled, exit_price, exit_fee, _route = omega._try_execution(
                    arrays, i, pos, entry=False, fee_base=fee_eff, slip_base=slip_eff
                )
                if not filled:
                    continue
                raw_exit = (
                    (exit_price - entry_price) / max(entry_price, 1.0e-12)
                    if pos > 0
                    else (entry_price - exit_price) / max(entry_price, 1.0e-12)
                )
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                reasons[reason] += 1
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                        "exit_timestamp": str(frame["timestamp"].iloc[i]),
                        "side": int(pos),
                        "source_component": active_name,
                        "reason": reason,
                        "trade_return": float(trade_return),
                        "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                        "mae_price_move": float(mae),
                        "mfe_price_move": float(mfe),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "entry_chop_probability": float(entry_chop_probability),
                        "entry_sizing_multiplier": float(entry_sizing_multiplier),
                        "continuation_ucb": continuation_ucb,
                        "sl_probability_lcb": sl_probability_lcb,
                    }
                )
                pos = 0
                active_name = ""
                rescue_streak = 0
                rescue_path = {}
                if reason == "hazard_rescue":
                    rearm_block_until_i = i + int(rearm_bars) + 1
                    rearm_wait_for_signal_reset = rearm_mode == "signal_reset"
                continue
            continue

        any_active = any(bool(active_masks[name][i]) for name in greedy.PRIORITY)
        if rearm_wait_for_signal_reset and not any_active:
            rearm_wait_for_signal_reset = False
        rearm_blocked = i < rearm_block_until_i or rearm_wait_for_signal_reset
        if rearm_blocked:
            rearm_blocked_bars += 1
            rearm_blocked_signal_bars += int(any_active)
            continue

        for name in greedy.PRIORITY:
            comp = components[name]
            if not bool(active_masks[name][i]):
                continue
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0:
                continue
            row_margin = float(comp["margin"][i])
            row_leverage = float(comp["leverage"][i])
            if row_margin <= 0.0 or row_leverage <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            base_row_notional = row_notional
            row_notional *= float(entry_notional_multiplier)
            if max_entry_notional is not None:
                row_notional = min(row_notional, float(max_entry_notional))
            entry_chop_probability = float("nan")
            entry_sizing_multiplier = row_notional / max(base_row_notional, 1.0e-12)
            if chop_soft_size_threshold is not None:
                entry_chop_probability = float(
                    frame["regime3_current_sensitive_wide24_chop_prob"].iloc[i]
                )
                threshold = float(chop_soft_size_threshold)
                chop_multiplier = (
                    1.0
                    if entry_chop_probability < threshold
                    else max(0.0, 1.0 - (entry_chop_probability - threshold) / (1.0 - threshold))
                )
                row_notional *= chop_multiplier
                entry_sizing_multiplier = row_notional / max(base_row_notional, 1.0e-12)
            row_leverage = row_notional / max(row_margin, 1.0e-12)
            filled, px, entry_fee, _route = omega._try_execution(
                arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff
            )
            if not filled:
                continue
            pos = side
            active_name = name
            entry_price = float(px)
            entry_equity = cash
            entry_i = min(i + 1, len(frame) - 1)
            entry_signal_i = i
            margin_fraction = row_margin
            leverage = row_leverage
            notional = row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * entry_fee * notional
            mfe = 0.0
            mae = 0.0
            rescue_streak = 0
            rescue_path = (
                precompute_rescue_path(
                    comp,
                    rescue_bundles[name],
                    arrays,
                    entry_i=entry_i,
                    side=pos,
                    entry_price=entry_price,
                    notional=notional,
                    leverage=leverage,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    slip_eff=slip_eff,
                )
                if rescue_bundles is not None
                else {}
            )
            break

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("empty router replay ledger")
    returns = ledger["trade_return"].to_numpy(dtype=np.float64)
    log_growth = np.log1p(np.clip(returns, -0.999999, None))
    tail_excess = np.maximum(-ledger["mae_price_move"].to_numpy(dtype=np.float64) * ledger["notional"].to_numpy(dtype=np.float64) - 0.02, 0.0)
    liquidation_excess = np.maximum(-ledger["mae_price_move"].to_numpy(dtype=np.float64) * ledger["leverage"].to_numpy(dtype=np.float64) - 0.12, 0.0)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(np.mean(returns > 0.0)),
        "log_growth_sum": float(log_growth.sum()),
        "tail_excess_sum": float(tail_excess.sum()),
        "liquidation_excess_sum": float(liquidation_excess.sum()),
        "log_risk_utility": float((log_growth - tail_excess - 0.25 * liquidation_excess).sum()),
        "exit_reasons": {str(key): int(value) for key, value in reasons.items()},
        "rescue_counterfactual_causes": {str(key): int(value) for key, value in rescue_counterfactual.items()},
        "rearm_mode": rearm_mode,
        "rearm_bars": int(rearm_bars),
        "rearm_blocked_bars": int(rearm_blocked_bars),
        "rearm_blocked_signal_bars": int(rearm_blocked_signal_bars),
    }
    return metrics, ledger


def _clears_val(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate["log_risk_utility"] > baseline["log_risk_utility"]
        and candidate["pnl"] >= 0.90 * baseline["pnl"]
        and candidate["mdd"] >= baseline["mdd"]
        and candidate["rescue_counterfactual_causes"].get("take_profit", 0)
        <= candidate["rescue_counterfactual_causes"].get("stop_loss", 0)
    )


def _prediction_path(split: str, name: str, cfg: dict[str, Any]) -> Path:
    return sweep.EXT_PRED_DIR / name / f"{split}_predictions_{cfg['q_tag']}.csv"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    models_by_variant: dict[str, dict[str, dict[str, Any]]] = {"B_post_quality": {}, "C_pre_quality": {}, "D_pre_quality_weighted": {}}
    expected_model_paths = {
        (variant, name): OUT_DIR / f"model_{name}_{variant}.pkl"
        for variant in models_by_variant
        for name in sweep.COMPONENTS
    }
    reuse_models = (OUT_DIR / "dataset_diagnostics.json").exists() and all(
        path.exists() for path in expected_model_paths.values()
    )
    dataset_diag: dict[str, dict[str, Any]] = (
        json.loads((OUT_DIR / "dataset_diagnostics.json").read_text(encoding="utf-8"))
        if reuse_models
        else {}
    )
    val_components: dict[str, dict[str, Any]] = {}
    if reuse_models:
        print(f"stage=frames val={len(val_frame)} reuse_models=true", flush=True)
        for (variant, name), path in expected_model_paths.items():
            with open(path, "rb") as handle:
                models_by_variant[variant][name] = pickle.load(handle)
    else:
        train_frame = sweep.load_frame(
            TRAIN_START, TRAIN_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025
        )
        print(f"stage=frames train={len(train_frame)} val={len(val_frame)} reuse_models=false", flush=True)
    for name, cfg in sweep.COMPONENTS.items():
        if not reuse_models:
            print(f"stage=prepare component={name} split=TRAIN post_quality", flush=True)
            train_post = prepare_split(
                name, cfg, train_frame, _prediction_path("train", name, cfg), oof=True, pre_quality=False
            )
            print(f"stage=prepare component={name} split=TRAIN pre_quality", flush=True)
            train_pre = prepare_split(
                name, cfg, train_frame, _prediction_path("train", name, cfg), oof=True, pre_quality=True
            )
            for variant, prepared, weighted in (
                ("B_post_quality", train_post, False),
                ("C_pre_quality", train_pre, False),
                ("D_pre_quality_weighted", train_pre, True),
            ):
                print(f"stage=dataset component={name} variant={variant}", flush=True)
                data, diag = build_episode_dataset(prepared, quality_weighted=weighted)
                dataset_diag[f"{name}:{variant}"] = diag
                data.to_csv(OUT_DIR / f"train_dataset_{name}_{variant}.csv.gz", index=False, compression="gzip")
                print(
                    f"stage=fit component={name} variant={variant} episodes={diag['episodes']} rows={diag['rows']}",
                    flush=True,
                )
                bundle = fit_bootstrap_ensemble(data, seed=SEED + len(models_by_variant[variant]) * 100)
                models_by_variant[variant][name] = bundle
                with open(OUT_DIR / f"model_{name}_{variant}.pkl", "wb") as handle:
                    pickle.dump(bundle, handle)
        print(f"stage=prepare component={name} split=VAL", flush=True)
        val_components[name] = prepare_split(
            name,
            cfg,
            val_frame,
            _prediction_path("validation", name, cfg),
            oof=True,
            pre_quality=False,
        )

    (OUT_DIR / "dataset_diagnostics.json").write_text(
        json.dumps(dataset_diag, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print("stage=val_baseline", flush=True)
    baseline, baseline_ledger = replay_router(
        val_frame,
        val_components,
        rescue_bundles=None,
        sl_probability_min=1.0,
        value_margin=1.0,
        persistence=1,
    )
    baseline_ledger.to_csv(OUT_DIR / "validation_ledger_A_baseline.csv", index=False)

    val_rows: list[dict[str, Any]] = [{"variant": "A_baseline", "sl_probability_min": None, "value_margin": None, "persistence": None, **baseline, "clears_val": True}]
    winners: list[dict[str, Any]] = []
    for variant, bundles in models_by_variant.items():
        for sl_probability_min in (0.60, 0.70, 0.80):
            for value_margin in (0.0, 0.0025, 0.0050):
                for persistence in (1, 3):
                    print(
                        f"stage=val variant={variant} psl={sl_probability_min} margin={value_margin} persistence={persistence}",
                        flush=True,
                    )
                    metrics, ledger = replay_router(
                        val_frame,
                        val_components,
                        rescue_bundles=bundles,
                        sl_probability_min=sl_probability_min,
                        value_margin=value_margin,
                        persistence=persistence,
                    )
                    clears = _clears_val(metrics, baseline)
                    row = {
                        "variant": variant,
                        "sl_probability_min": sl_probability_min,
                        "value_margin": value_margin,
                        "persistence": persistence,
                        **metrics,
                        "clears_val": clears,
                    }
                    val_rows.append(row)
                    if clears:
                        winners.append(row)
                    if clears:
                        ledger.to_csv(
                            OUT_DIR / f"validation_ledger_{variant}_psl{sl_probability_min:.2f}_m{value_margin:.4f}_p{persistence}.csv",
                            index=False,
                        )

    val_table = pd.DataFrame(val_rows).sort_values(
        ["clears_val", "log_risk_utility", "mdd", "pnl"], ascending=[False, False, False, False]
    )
    val_table.to_csv(OUT_DIR / "validation_ranking.csv", index=False)
    selected = max(winners, key=lambda row: (row["log_risk_utility"], row["mdd"], row["pnl"])) if winners else None
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "val_rejected" if selected is None else "val_selected_oos_pending",
        "protocol": {
            "train": [TRAIN_START, TRAIN_END],
            "validation": [sweep.VAL_START, sweep.VAL_END],
            "oos": [sweep.OOS_START, sweep.OOS_END],
            "validation_boundary_exception": "Frozen OOF predictions start 2025-10-01; September overlaps parent training.",
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "training_future_rows_used_for_labels_only": True,
            "barrier_observation": "close_based_frozen_replay_contract",
            "frozen_exit_head_used": False,
            "frozen_exit_head_omission_reason": "EXIT_THRESHOLD=0.95 is empirically inert on the audited live baseline.",
        },
        "dataset_diagnostics": dataset_diag,
        "validation_baseline": baseline,
        "selected": selected,
    }
    if selected is None:
        (OUT_DIR / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
        )
        print("stage=done no_val_winner; OOS not opened", flush=True)
        return 0

    print(f"stage=oos_confirm selected={selected}", flush=True)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    oos_components: dict[str, dict[str, Any]] = {}
    for name, cfg in sweep.COMPONENTS.items():
        oos_components[name] = prepare_split(
            name,
            cfg,
            oos_frame,
            _prediction_path("oos", name, cfg),
            oof=False,
            pre_quality=False,
        )
    oos_baseline, oos_baseline_ledger = replay_router(
        oos_frame,
        oos_components,
        rescue_bundles=None,
        sl_probability_min=1.0,
        value_margin=1.0,
        persistence=1,
    )
    oos_candidate, oos_candidate_ledger = replay_router(
        oos_frame,
        oos_components,
        rescue_bundles=models_by_variant[str(selected["variant"])],
        sl_probability_min=float(selected["sl_probability_min"]),
        value_margin=float(selected["value_margin"]),
        persistence=int(selected["persistence"]),
    )
    oos_baseline_ledger.to_csv(OUT_DIR / "oos_ledger_A_baseline.csv", index=False)
    oos_candidate_ledger.to_csv(OUT_DIR / "oos_ledger_selected_candidate.csv", index=False)
    report["status"] = "oos_confirmed"
    report["oos_baseline"] = oos_baseline
    report["oos_candidate"] = oos_candidate
    report["oos_pass"] = _clears_val(oos_candidate, oos_baseline)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "oos_pass": report["oos_pass"]}, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
