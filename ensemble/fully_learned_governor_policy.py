from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


FEATURE_COLS = [
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
    "log_return",
    "volatility_z",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "liquidity_vacuum",
    "execution_quality",
    "jump_z",
    "jump_flag",
    "evt_tail_flag",
    "evt_excess_z",
    "funding_abs",
    "funding_pressure",
    "funding_price_divergence",
    "long_squeeze_risk",
    "crowding_pressure",
    "smart_money_flow",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "whale_conviction",
    "m7_gate_block",
    "m7_tail_risk",
    "m7_expected_ret",
    "m7_composite_score",
    "m7_confidence",
    "m7_qwidth",
    "pred_patchtst",
    "conf_patchtst",
    "ai_dir_edge",
    "ai_dir_p_up",
    "ai_dir_p_down",
    "ai_dir_p_flat",
    "ai_dir_entropy",
    "patchtst_median",
    "patchtst_regime_sim",
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_raw",
    "tide_vol_zscore",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "ai_flow_flip_prob",
    "ai_flow_slope",
    "dlinear_smf_ema",
    "dlinear_smf_slope",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "trade_intensity",
    "big_trade_ratio",
    "whale_retail_ratio",
    "squeeze_power",
    "breakout_strength",
    "regime_bull_id",
    "regime_bear_id",
    "regime_chop_id",
    "regime_whipsaw_id",
    "regime_normal_id",
]


@dataclass(frozen=True)
class FullyLearnedGovernorConfig:
    notional_buckets: tuple[float, ...] = (0.35, 0.55, 0.80, 1.10, 1.50, 2.10, 3.00)
    leverage_buckets: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 5.0)
    take_profit_buckets: tuple[float, ...] = (0.010, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 1.000)
    stop_loss_buckets: tuple[float, ...] = (0.006, 0.010, 0.016, 0.024, 0.035, 0.055)
    max_hold_buckets: tuple[int, ...] = (3, 6, 12, 24, 48, 96, 288, 864)
    cooldown_buckets: tuple[int, ...] = (0, 1, 3, 6, 12, 24, 48)
    max_train_horizon_bars: int = 864
    fee: float = 0.0005
    slip: float = 0.0002
    cash_score: float = 0.0000
    adverse_penalty: float = 0.85
    size_penalty: float = 0.018
    hold_penalty: float = 0.004
    turnover_bonus: float = 0.0015
    max_margin_fraction: float = 1.0


@dataclass(frozen=True)
class FullyLearnedGovernorDecision:
    action: int
    side: int
    notional_exposure: float
    leverage: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    cooldown_bars: int
    position_fraction: float
    quality_score: float
    confidence: float

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_num(s: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series([default])


def _close_array(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _momentum(close: np.ndarray, bars: int) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.float64)
    b = int(max(1, bars))
    if len(close) > b:
        out[b:] = close[b:] / np.maximum(close[:-b], 1e-12) - 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_features(
    frame: pd.DataFrame,
    *,
    side_hint: int | np.ndarray = 0,
    close: np.ndarray | None = None,
    feature_cols: Sequence[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    target_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)
    close_arr = _close_array(frame) if close is None else np.asarray(close, dtype=np.float64)
    out = pd.DataFrame(index=frame.index)
    n = len(frame)
    out["side_hint"] = np.full(n, int(side_hint), dtype=np.float64) if np.isscalar(side_hint) else np.asarray(side_hint, dtype=np.float64)
    for bars, name in ((6048, "mom_21d"), (864, "mom_3d"), (288, "mom_1d")):
        if name in frame.columns:
            mom = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            mom = _momentum(close_arr, bars)
        if len(mom) != n:
            mom = np.resize(mom, n)
        out[name] = mom
        out[f"abs_{name}"] = np.abs(mom)
    for col in target_cols:
        if col in out.columns:
            continue
        if col.startswith("regime_") and col.endswith("_id"):
            regime = col.replace("regime_", "").replace("_id", "")
            raw_col = f"regime_{regime}"
            out[col] = _safe_num(frame[raw_col]) if raw_col in frame.columns else 0.0
            continue
        if col in frame.columns:
            out[col] = _safe_num(frame[col])
        else:
            if strict:
                raise RuntimeError(f"missing fully learned feature column: {col}")
            out[col] = 0.0
    return out.reindex(columns=target_cols).replace([np.inf, -np.inf], np.nan)


def _classifier(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.040,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _regressor(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.040,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _first_hit(path: np.ndarray, tp: float, sl: float, hold: int) -> int:
    m = min(int(hold), len(path))
    if m <= 1:
        return 0
    p = path[:m]
    hit = np.flatnonzero((p >= float(tp)) | (p <= -abs(float(sl))))
    return int(hit[0]) if hit.size else int(m - 1)


def _best_label_for_path(raw_ret: np.ndarray, cfg: FullyLearnedGovernorConfig) -> tuple[int, int, int, int, int, int, int, float]:
    best_score = float(cfg.cash_score)
    best = (ACTION_CASH, 0, 0, 0, 0, 0, 0, best_score)
    cost = 2.0 * float(cfg.fee + cfg.slip)
    for action, side in ((ACTION_LONG, 1.0), (ACTION_SHORT, -1.0)):
        side_ret = np.nan_to_num(raw_ret * side, nan=0.0, posinf=0.0, neginf=0.0)
        for ni, notional in enumerate(cfg.notional_buckets):
            exp_path = side_ret * float(notional)
            for ti, tp in enumerate(cfg.take_profit_buckets):
                for si, sl in enumerate(cfg.stop_loss_buckets):
                    for hi, hold in enumerate(cfg.max_hold_buckets):
                        exit_i = _first_hit(exp_path, float(tp), float(sl), int(hold))
                        sample = exp_path[: exit_i + 1]
                        pnl = float(exp_path[exit_i] - cost * float(notional))
                        adverse = max(0.0, -float(np.min(sample)))
                        hold_frac = float(exit_i + 1) / max(float(cfg.max_train_horizon_bars), 1.0)
                        for li, lev in enumerate(cfg.leverage_buckets):
                            margin = float(notional) / max(float(lev), 1e-8)
                            if margin > float(cfg.max_margin_fraction) + 1e-9:
                                continue
                            liq_buffer = 0.70 / max(float(lev), 1.0)
                            liq_penalty = 2.5 * max(0.0, adverse - liq_buffer)
                            score = (
                                pnl
                                - float(cfg.adverse_penalty) * adverse
                                - float(cfg.size_penalty) * (float(notional) / max(cfg.notional_buckets)) ** 2
                                - float(cfg.hold_penalty) * hold_frac
                                - liq_penalty
                                + float(cfg.turnover_bonus) / max(float(exit_i + 1), 1.0) ** 0.35
                            )
                            if score > best_score:
                                # Cooldown is labeled from immediate post-exit reversal risk, not fixed at runtime.
                                next_slice = side_ret[exit_i + 1 : min(len(side_ret), exit_i + 49)]
                                if len(next_slice) == 0:
                                    cool_i = 0
                                else:
                                    reversal = max(0.0, -float(np.min(next_slice)) * float(notional))
                                    continuation = max(0.0, float(np.max(next_slice)) * float(notional))
                                    if continuation > reversal + 0.012:
                                        cool_i = 0
                                    elif reversal > 0.030:
                                        cool_i = len(cfg.cooldown_buckets) - 1
                                    elif reversal > 0.018:
                                        cool_i = min(len(cfg.cooldown_buckets) - 1, 4)
                                    else:
                                        cool_i = min(len(cfg.cooldown_buckets) - 1, 2)
                                best_score = score
                                best = (action, ni, li, ti, si, hi, cool_i, best_score)
    return best


def _choose_leverage_index(adverse: np.ndarray, notional: float, cfg: FullyLearnedGovernorConfig) -> tuple[np.ndarray, np.ndarray]:
    adverse_arr = np.asarray(adverse, dtype=np.float64)
    best_penalty = np.full(adverse_arr.shape, np.inf, dtype=np.float64)
    best_idx = np.zeros(adverse_arr.shape, dtype=np.int64)
    for li, lev in enumerate(cfg.leverage_buckets):
        lev_f = float(lev)
        margin = float(notional) / max(lev_f, 1e-8)
        if margin > float(cfg.max_margin_fraction) + 1e-9:
            continue
        liq_buffer = 0.70 / max(lev_f, 1.0)
        penalty = 2.5 * np.maximum(0.0, adverse_arr - liq_buffer)
        take = penalty < best_penalty
        best_penalty[take] = penalty[take]
        best_idx[take] = int(li)
    best_penalty = np.where(np.isfinite(best_penalty), best_penalty, 1e9)
    return best_idx, best_penalty


def _vectorized_labels(raw_ret: np.ndarray, cfg: FullyLearnedGovernorConfig) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_ret, dtype=np.float64)
    n = raw.shape[0]
    best_score = np.full(n, float(cfg.cash_score), dtype=np.float64)
    labels = {
        "action": np.full(n, ACTION_CASH, dtype=np.int64),
        "notional": np.zeros(n, dtype=np.int64),
        "leverage": np.zeros(n, dtype=np.int64),
        "take_profit": np.zeros(n, dtype=np.int64),
        "stop_loss": np.zeros(n, dtype=np.int64),
        "max_hold": np.zeros(n, dtype=np.int64),
        "cooldown": np.zeros(n, dtype=np.int64),
        "quality": best_score.copy(),
    }
    cost = 2.0 * float(cfg.fee + cfg.slip)
    row_idx = np.arange(n)
    max_notional = max(cfg.notional_buckets)
    hold_norm = max(float(cfg.max_train_horizon_bars), 1.0)

    for action, side in ((ACTION_LONG, 1.0), (ACTION_SHORT, -1.0)):
        side_ret = np.nan_to_num(raw * side, nan=0.0, posinf=0.0, neginf=0.0)
        for ni, notional in enumerate(cfg.notional_buckets):
            notional_f = float(notional)
            exp_path = side_ret * notional_f
            cum_min = np.minimum.accumulate(exp_path, axis=1)
            for ti, tp in enumerate(cfg.take_profit_buckets):
                tp_f = float(tp)
                for si, sl in enumerate(cfg.stop_loss_buckets):
                    hit = (exp_path >= tp_f) | (exp_path <= -abs(float(sl)))
                    has_hit = hit.any(axis=1)
                    first_hit = np.where(has_hit, hit.argmax(axis=1), exp_path.shape[1] - 1).astype(np.int64)
                    for hi, hold in enumerate(cfg.max_hold_buckets):
                        hold_i = max(0, min(int(hold), exp_path.shape[1]) - 1)
                        exit_i = np.where(first_hit <= hold_i, first_hit, hold_i).astype(np.int64)
                        pnl = exp_path[row_idx, exit_i] - cost * notional_f
                        adverse = np.maximum(0.0, -cum_min[row_idx, exit_i])
                        li_idx, liq_penalty = _choose_leverage_index(adverse, notional_f, cfg)
                        hold_frac = (exit_i.astype(np.float64) + 1.0) / hold_norm
                        score = (
                            pnl
                            - float(cfg.adverse_penalty) * adverse
                            - float(cfg.size_penalty) * (notional_f / max_notional) ** 2
                            - float(cfg.hold_penalty) * hold_frac
                            - liq_penalty
                            + float(cfg.turnover_bonus) / np.maximum(exit_i.astype(np.float64) + 1.0, 1.0) ** 0.35
                        )
                        take = score > best_score
                        if not np.any(take):
                            continue
                        next_end = np.minimum(exp_path.shape[1], exit_i + 49)
                        cool_i = np.zeros(n, dtype=np.int64)
                        for r in np.flatnonzero(take):
                            if exit_i[r] + 1 >= next_end[r]:
                                ci = 0
                            else:
                                next_slice = exp_path[r, exit_i[r] + 1 : next_end[r]]
                                reversal = max(0.0, -float(np.min(next_slice)))
                                continuation = max(0.0, float(np.max(next_slice)))
                                if continuation > reversal + 0.012:
                                    ci = 0
                                elif reversal > 0.030:
                                    ci = len(cfg.cooldown_buckets) - 1
                                elif reversal > 0.018:
                                    ci = min(len(cfg.cooldown_buckets) - 1, 4)
                                else:
                                    ci = min(len(cfg.cooldown_buckets) - 1, 2)
                            cool_i[r] = int(ci)
                        best_score[take] = score[take]
                        labels["action"][take] = int(action)
                        labels["notional"][take] = int(ni)
                        labels["leverage"][take] = li_idx[take]
                        labels["take_profit"][take] = int(ti)
                        labels["stop_loss"][take] = int(si)
                        labels["max_hold"][take] = int(hi)
                        labels["cooldown"][take] = cool_i[take]
                        labels["quality"][take] = score[take]
    return labels


def build_training_set(
    frame: pd.DataFrame,
    *,
    cfg: FullyLearnedGovernorConfig,
    stride_bars: int = 3,
    batch_size: int = 512,
    feature_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    close = _close_array(frame)
    h = int(cfg.max_train_horizon_bars)
    valid = np.arange(0, max(0, len(frame) - h - 1), max(1, int(stride_bars)), dtype=np.int64)
    if valid.size == 0:
        raise ValueError("no fully learned governor candidates")
    full_x = prepare_features(frame, side_hint=0, close=close, feature_cols=feature_cols)
    x = full_x.iloc[valid].reset_index(drop=True)
    y = {
        "action": np.zeros(valid.size, dtype=np.int64),
        "notional": np.zeros(valid.size, dtype=np.int64),
        "leverage": np.zeros(valid.size, dtype=np.int64),
        "take_profit": np.zeros(valid.size, dtype=np.int64),
        "stop_loss": np.zeros(valid.size, dtype=np.int64),
        "max_hold": np.zeros(valid.size, dtype=np.int64),
        "cooldown": np.zeros(valid.size, dtype=np.int64),
        "quality": np.zeros(valid.size, dtype=np.float64),
    }
    horizons = np.arange(1, h + 1, dtype=np.int64)
    for start in range(0, valid.size, int(batch_size)):
        end = min(start + int(batch_size), valid.size)
        idx = valid[start:end]
        fut = close[idx[:, None] + horizons[None, :]]
        raw_ret = fut / np.maximum(close[idx][:, None], 1e-12) - 1.0
        batch_labels = _vectorized_labels(raw_ret, cfg)
        for key, vals in batch_labels.items():
            y[key][start:end] = vals
    meta = {"candidates": int(valid.size), "stride_bars": int(stride_bars), "max_train_horizon_bars": int(h)}
    return x, y, meta


def _weighted_fit_classifier(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    if np.unique(y).size < 2:
        return None
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    return model


def train_policy(
    x: pd.DataFrame,
    y: Mapping[str, np.ndarray],
    *,
    cfg: FullyLearnedGovernorConfig,
    random_state: int = 42,
    feature_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    model_feature_cols = list(feature_cols) if feature_cols is not None else list(x.columns)
    action_weights = np.where(np.asarray(y["action"]) == ACTION_CASH, 0.35, 1.0)
    quality_weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64)), 0.03, 1.0)
    weights = np.maximum(action_weights, quality_weights)
    trade_mask = np.asarray(y["action"]) != ACTION_CASH
    x_trade = x.loc[trade_mask].copy()
    trade_weights = weights[trade_mask]
    bundle: dict[str, Any] = {
        "model_type": "fully_learned_governor_policy_v1",
        "feature_cols": model_feature_cols,
        "config": asdict(cfg),
        "action_model": _weighted_fit_classifier(_classifier(random_state), x, np.asarray(y["action"]), weights),
        "quality_model": _regressor(random_state + 99),
        "default_bucket_indexes": {
            key: int(pd.Series(np.asarray(y[key])[trade_mask]).mode().iloc[0]) if np.any(trade_mask) else 0
            for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
        },
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict()
            for key, vals in y.items()
            if key != "quality"
        },
    }
    bundle["quality_model"].fit(x, np.asarray(y["quality"], dtype=np.float64), histgradientboostingregressor__sample_weight=weights)
    for offset, key in enumerate(("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"), start=1):
        model = _weighted_fit_classifier(_classifier(random_state + offset), x_trade, np.asarray(y[key])[trade_mask], trade_weights)
        if model is not None:
            bundle[f"{key}_model"] = model
    bundle["label_distribution"]["quality_mean"] = float(np.mean(y["quality"]))
    bundle["label_distribution"]["quality_p95"] = float(np.quantile(y["quality"], 0.95))
    return bundle


def _bucket_expectation(model: Any, x: pd.DataFrame, buckets: tuple[float, ...]) -> tuple[float, float, int]:
    proba = model.predict_proba(x)[0]
    classes = np.asarray(model.classes_, dtype=int)
    vals = np.asarray([buckets[int(c)] for c in classes], dtype=np.float64)
    cls = int(classes[int(np.argmax(proba))])
    return float(np.sum(proba * vals)), float(np.max(proba)), cls


def _bucket_expectation_batch(model: Any, x: pd.DataFrame, buckets: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    vals = np.asarray([buckets[int(c)] for c in classes], dtype=np.float64)
    return proba @ vals, np.max(proba, axis=1)


def _bucket_or_default_batch(
    bundle: Mapping[str, Any],
    key: str,
    x: pd.DataFrame,
    buckets: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    model_key = f"{key}_model"
    if model_key in bundle:
        return _bucket_expectation_batch(bundle[model_key], x, buckets)
    default_idx = int(dict(bundle.get("default_bucket_indexes", {})).get(key, 0))
    default = float(buckets[int(np.clip(default_idx, 0, len(buckets) - 1))])
    return np.full(len(x), default, dtype=np.float64), np.ones(len(x), dtype=np.float64)


def predict_policy_frame(
    bundle: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    close: np.ndarray | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(bundle.get("config", {})))
    feature_cols = list(bundle.get("feature_cols") or FEATURE_COLS)
    if set(feature_cols).issubset(frame.columns):
        x = frame.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
        if "side_hint" in x.columns:
            x["side_hint"] = 0.0
    else:
        if strict:
            missing = [c for c in feature_cols if c not in frame.columns]
            raise RuntimeError(f"missing fully learned feature columns: {missing[:30]}")
        x = prepare_features(frame, side_hint=0, close=close, feature_cols=feature_cols)

    action_proba = bundle["action_model"].predict_proba(x)
    action_classes = np.asarray(bundle["action_model"].classes_, dtype=int)
    action_idx = np.argmax(action_proba, axis=1)
    action = action_classes[action_idx]
    action_conf = np.max(action_proba, axis=1)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    quality = bundle["quality_model"].predict(x) if "quality_model" in bundle else np.zeros(len(x), dtype=np.float64)

    x_side = x.copy()
    x_side["side_hint"] = side.astype(np.float64)
    notional, c1 = _bucket_or_default_batch(bundle, "notional", x_side, cfg.notional_buckets)
    leverage, c2 = _bucket_or_default_batch(bundle, "leverage", x_side, cfg.leverage_buckets)
    take_profit, c3 = _bucket_or_default_batch(bundle, "take_profit", x_side, cfg.take_profit_buckets)
    stop_loss, c4 = _bucket_or_default_batch(bundle, "stop_loss", x_side, cfg.stop_loss_buckets)
    max_hold, c5 = _bucket_or_default_batch(bundle, "max_hold", x_side, tuple(float(v) for v in cfg.max_hold_buckets))
    cooldown, c6 = _bucket_or_default_batch(bundle, "cooldown", x_side, tuple(float(v) for v in cfg.cooldown_buckets))

    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, min(cfg.notional_buckets), max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    confidence = np.mean(np.vstack([action_conf, c1, c2, c3, c4, c5, c6]), axis=0)
    cash = action == ACTION_CASH
    out = pd.DataFrame(
        {
            "action": action.astype(np.int64),
            "side": side.astype(np.int64),
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": take_profit.astype(np.float64),
            "stop_loss": stop_loss.astype(np.float64),
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "quality_score": np.asarray(quality, dtype=np.float64),
            "confidence": confidence.astype(np.float64),
        },
        index=frame.index,
    )
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def predict_policy(bundle: Mapping[str, Any], row: pd.Series | pd.DataFrame, *, close: np.ndarray | None = None) -> FullyLearnedGovernorDecision:
    frame = row.to_frame().T if isinstance(row, pd.Series) else row.tail(1).copy()
    dec = predict_policy_frame(bundle, frame, close=close).iloc[-1]
    return FullyLearnedGovernorDecision(
        action=int(dec.action),
        side=int(dec.side),
        notional_exposure=float(dec.notional_exposure),
        leverage=float(dec.leverage),
        take_profit=float(dec.take_profit),
        stop_loss=float(dec.stop_loss),
        max_hold_bars=int(dec.max_hold_bars),
        cooldown_bars=int(dec.cooldown_bars),
        position_fraction=float(dec.position_fraction),
        quality_score=float(dec.quality_score),
        confidence=float(dec.confidence),
    )
