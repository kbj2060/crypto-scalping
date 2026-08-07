#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FEATURE_COLS,
    FullyLearnedGovernorConfig,
    build_training_set,
    train_policy,
)
from scripts.train_eval_alphazero_style_governor_2026 import (  # noqa: E402
    AZExitModel,
    EXIT_ACTIONS,
    PVBundle,
    _predict_pv,
    _rollout_exit_targets,
    _train_pv,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    _date_codes,
    _exit_probability_vec,
    _fill_price,
    backtest_no_limit_exit,
    collect_exit_samples,
    train_exit_model,
)
from scripts.train_eval_muzero_style_governor_2026 import (  # noqa: E402
    ENTRY_ACTIONS,
    MZBundle,
    _make_targets,
    _planned_decisions,
    _plan_scores,
    _train_muzero,
)
from scripts.train_eval_zero_style_risk_overlay_2026 import (  # noqa: E402
    MZRiskBundle,
    RISK_ACTIONS,
    RISK_SCALES,
    _apply_scale,
    _mz_entry_decisions,
    _predict_mz_risk,
    _risk_targets,
    _state_frame,
    _train_mz_risk,
)


DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026_ledger.csv"


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return ["", ""]
    return [str(ts.min()), str(ts.max())]


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    if "timestamp" not in a.columns or "timestamp" not in b.columns:
        return 0
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _audit(train_csv: Path, eval_csv: Path, policy: dict[str, Any]) -> dict[str, Any]:
    train_raw = pd.read_csv(train_csv, usecols=["timestamp"])
    eval_raw = pd.read_csv(eval_csv, usecols=["timestamp"])
    t1_raw = pd.to_datetime(train_raw["timestamp"], errors="coerce")
    t2_raw = pd.to_datetime(eval_raw["timestamp"], errors="coerce")
    t1 = t1_raw.dropna()
    t2 = t2_raw.dropna()
    overlap = set(t1.astype("int64").tolist()) & set(t2.astype("int64").tolist())
    return {
        "train_rows": int(len(train_raw)),
        "eval_rows": int(len(eval_raw)),
        "train_valid_timestamp_rows": int(len(t1)),
        "eval_valid_timestamp_rows": int(len(t2)),
        "train_range": [str(t1.min()), str(t1.max())],
        "eval_range": [str(t2.min()), str(t2.max())],
        "timestamp_overlap_rows": int(len(overlap)),
        "train_duplicate_timestamps": int(t1_raw.duplicated().sum()),
        "eval_duplicate_timestamps": int(t2_raw.duplicated().sum()),
        "policy_feature_count": int(len(policy.get("feature_cols", []))),
        "label_distribution": policy.get("label_distribution", {}),
    }


def _score(bt: dict[str, Any], *, mdd_weight: float = 3.0) -> float:
    tpd = float(bt.get("trades_per_day", 0.0) or 0.0)
    sparse_penalty = 60.0 * max(0.0, 4.0 - tpd)
    return float(bt.get("pnl", 0.0) or 0.0) + float(mdd_weight) * float(bt.get("mdd", 0.0) or 0.0) - sparse_penalty


def _policy_config_from_bundle(path: Path) -> FullyLearnedGovernorConfig:
    if not path.exists():
        return FullyLearnedGovernorConfig()
    bundle = joblib.load(path)
    cfg = dict(bundle.get("config", {}) or {})
    allowed = set(FullyLearnedGovernorConfig.__dataclass_fields__.keys())
    return FullyLearnedGovernorConfig(**{k: v for k, v in cfg.items() if k in allowed})


def _train_clean_policy(
    train_df: pd.DataFrame,
    *,
    cfg: FullyLearnedGovernorConfig,
    stride_bars: int,
    batch_size: int,
    seed: int,
    train_csv: Path,
    model_out: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=int(stride_bars), batch_size=int(batch_size))
    bundle = train_policy(x, y, cfg=cfg, random_state=int(seed))
    bundle["train_csv"] = str(train_csv)
    bundle["clean_scope"] = {
        "train_range": _range(train_df),
        "fit_rows": int(len(train_df)),
        "stride_bars": int(stride_bars),
        "max_train_horizon_bars": int(cfg.max_train_horizon_bars),
        "seed": int(seed),
    }
    bundle["training_meta"] = meta
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    return bundle, meta


def _train_clean_exit_model(
    train_df: pd.DataFrame,
    policy: dict[str, Any],
    entry_cfg: dict[str, Any],
    *,
    fee: float,
    slip: float,
    max_samples: int,
    seed: int,
    model_out: Path,
) -> tuple[Any, dict[str, Any]]:
    x, y, meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(fee),
        slip=float(slip),
        entry_stride=36,
        min_age=3,
        max_age=144,
        age_stride=24,
        future_horizon=72,
        exit_edge=0.0015,
        adverse_gap=0.012,
        max_samples=int(max_samples),
        seed=int(seed),
    )
    model = train_exit_model(x, y, seed=int(seed))
    payload = {"model": model, "sample_meta": meta, "entry_config": dict(entry_cfg), "model_cols": list(MODEL_COLS)}
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_out)
    return model, meta


def _select_controls_on_validation(
    val_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    base_risk_cfg: dict[str, Any],
    *,
    fee: float,
    slip: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for max_daily in (12, 16):
        for daily_loss in (0.025, 0.04):
            for daily_dd in (0.025, 0.035):
                for cooldown in (12, 24):
                    risk_cfg = dict(base_risk_cfg)
                    risk_cfg.update(
                        {
                            "max_daily_trades": int(max_daily),
                            "daily_loss_limit": float(daily_loss),
                            "daily_dd_limit": float(daily_dd),
                            "loss_cooldown_bars": int(cooldown),
                        }
                    )
                    pre = _base_frame(val_df, policy, entry_cfg)
                    for th in (0.45, 0.55, 0.65):
                        for age in (3, 6, 12):
                            exit_cfg = {"exit_threshold": float(th), "min_exit_age": int(age)}
                            bt = backtest_no_limit_exit(
                                val_df,
                                policy,
                                exit_model,
                                entry_config=entry_cfg,
                                risk_config=risk_cfg,
                                exit_threshold=float(th),
                                min_exit_age=int(age),
                                fee=float(fee),
                                slip=float(slip),
                                precomputed=pre,
                            )
                            rows.append(
                                {
                                    "name": f"exit{th:.2f}_age{age}_max{max_daily}_dd{daily_dd}_loss{daily_loss}_cd{cooldown}",
                                    "entry_config": dict(entry_cfg),
                                    "risk_config": risk_cfg,
                                    "exit_config": exit_cfg,
                                    "eval": _compact(bt),
                                    "score": _score(bt),
                                }
                            )
    ranked = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    return ranked[0], ranked


def _apply_az_risk(dec: pd.DataFrame, state: pd.DataFrame, bundle: PVBundle, device: str) -> pd.DataFrame:
    x = state.reindex(columns=bundle.feature_cols).to_numpy(dtype=np.float32)
    probs, values = _predict_pv(bundle, x, device)
    idx = np.argmax(probs, axis=1)
    idx = np.where(values < -0.15, 3, idx)
    return _apply_scale(dec, idx)


def _run_simple(
    name: str,
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    bt = backtest_no_limit_exit(
        df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(fee),
        slip=float(slip),
        precomputed=precomputed,
    )
    return {"name": name, "eval": _compact(bt), "score": _score(bt)}


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return max(len(df) / 288.0, 1e-8)
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(float(default)).to_numpy(dtype=np.float64)


def realistic_ledger_replay(
    df: pd.DataFrame,
    exit_model: Any,
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    funding_mult: float,
    impact_per_notional: float,
    partial_fill_ratio: float,
    maintenance_margin: float,
    liquidation_fee: float,
) -> dict[str, Any]:
    base_feat, decisions, close, fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    day_codes = _date_codes(df)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    cooldowns = pd.to_numeric(decisions["cooldown_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    high = _num_col(df, "high", 0.0)
    low = _num_col(df, "low", 0.0)
    if not np.any(high):
        high = close.copy()
    if not np.any(low):
        low = close.copy()
    funding = _num_col(df, "last_funding_rate", 0.0)
    if not np.any(np.isfinite(funding)) or np.nanmax(np.abs(funding)) == 0.0:
        funding = _num_col(df, "funding_rate", 0.0)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    model_cooldown = 0
    cooldown_left = 0
    loss_cooldown_left = 0
    loss_streak = 0
    peak_unrealized = 0.0
    entry_quality = 0.0
    entry_confidence = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    funding_paid = fee_paid = slippage_paid = impact_paid = liquidation_paid = 0.0
    partial_fill_events = liquidations = 0
    exits: dict[str, int] = {}
    entry_blocks: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    open_ledger: dict[str, Any] | None = None
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def block(reason: str) -> None:
        entry_blocks[reason] = entry_blocks.get(reason, 0) + 1

    def impact_slip(n: float) -> float:
        return float(slip) + float(impact_per_notional) * min(max(float(n), 0.0), float(risk_cfg.get("max_notional", 3.6)))

    def fill_ratio(n: float) -> float:
        pressure = max(0.0, float(n) / max(float(risk_cfg.get("max_notional", 3.6)), 1e-12) - 0.35)
        ratio = float(partial_fill_ratio) - 0.08 * pressure
        return float(np.clip(ratio, 0.72, 1.0))

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        s = impact_slip(notional)
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - s) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + s)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str, *, forced_raw: float | None = None) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, cooldown_left, model_cooldown
        nonlocal trades, wins, loss_streak, loss_cooldown_left, daily_trades, peak_unrealized
        nonlocal fee_paid, slippage_paid, impact_paid, liquidation_paid, liquidations, open_ledger
        s = impact_slip(notional)
        if forced_raw is None:
            exit_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, s, entry=False)
            raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        else:
            raw = float(forced_raw)
        before = cash
        cash = cash * (1.0 + raw * notional)
        fee_cost = before * float(fee) * notional
        cash -= fee_cost
        fee_paid += fee_cost
        slippage_paid += before * float(slip) * notional
        impact_paid += before * max(0.0, s - float(slip)) * notional
        if reason == "liquidation":
            liq_fee = before * float(liquidation_fee) * notional
            cash -= liq_fee
            liquidation_paid += liq_fee
            liquidations += 1
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_cfg.get("loss_cooldown_bars", 0)))
        exits[reason] = exits.get(reason, 0) + 1
        if open_ledger is not None:
            rec = dict(open_ledger)
            rec.update(
                {
                    "exit_idx": int(i),
                    "exit_timestamp": str(df["timestamp"].iloc[int(np.clip(i, 0, len(df) - 1))]) if "timestamp" in df.columns else "",
                    "exit_reason": reason,
                    "exit_equity": float(cash),
                    "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                    "peak_unrealized_pct": float(peak_unrealized * 100.0),
                }
            )
            ledger.append(rec)
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        cooldown_left = int(model_cooldown)
        model_cooldown = 0
        peak_unrealized = 0.0
        open_ledger = None

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            rate = float(funding[int(np.clip(i, 0, len(funding) - 1))])
            f_cost = cash * notional * abs(rate) / 96.0 * float(funding_mult)
            cash -= f_cost
            funding_paid += f_cost
            liq_buffer = max(0.02, 1.0 / max(leverage, 1.0) - float(maintenance_margin))
            if pos > 0:
                adverse_raw = (float(low[i]) * (1.0 - impact_slip(notional)) - entry_price) / max(entry_price, 1e-12)
            else:
                adverse_raw = (entry_price - float(high[i]) * (1.0 + impact_slip(notional))) / max(entry_price, 1e-12)
            if adverse_raw <= -liq_buffer:
                close_position(i, "liquidation", forced_raw=-liq_buffer)
                continue
            peak_unrealized = max(peak_unrealized, unreal)
            age = i - entry_idx
            if age >= int(exit_cfg["min_exit_age"]):
                row_vec = np.zeros(len(MODEL_COLS), dtype=np.float32)
                row_vec[: len(FEATURE_COLS)] = base_values[int(i)]
                row_vec[0] = float(pos)
                j = len(FEATURE_COLS)
                current_side = int(sides[int(i)])
                ctx = (
                    float(pos),
                    float(age),
                    float(np.log1p(max(age, 0))),
                    float(unreal),
                    float(peak_unrealized),
                    float(peak_unrealized - unreal),
                    float(notional),
                    float(leverage),
                    float(entry_quality),
                    float(entry_confidence),
                    float(current_side == pos),
                    float(current_side == -pos),
                    float(qualities[int(i)]),
                    float(confs[int(i)]),
                )
                row_vec[j : j + len(ctx)] = np.asarray(ctx, dtype=np.float32)
                if _exit_probability_vec(exit_model, row_vec) >= float(exit_cfg["exit_threshold"]):
                    close_position(i, "exit_governor")
                    continue
            continue

        if cooldown_left > 0:
            cooldown_left -= 1
            block("model_cooldown")
            continue
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(risk_cfg.get("max_daily_trades", 999999)):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(risk_cfg.get("daily_loss_limit", 0.0))):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(risk_cfg.get("daily_dd_limit", 0.0))):
            block("daily_dd_lock")
            continue
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            block("cash_signal")
            continue

        n = float(notionals[i])
        if account_dd >= float(risk_cfg.get("global_dd_cut", 999.0)):
            n *= float(risk_cfg.get("global_dd_mult", 1.0))
        if loss_streak >= int(risk_cfg.get("loss_streak_soft", 999999)):
            steps = loss_streak - int(risk_cfg.get("loss_streak_soft", 999999)) + 1
            n *= float(risk_cfg.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(risk_cfg.get("daily_profit_boost_start", 999.0)):
            n *= float(risk_cfg.get("daily_profit_boost_mult", 1.0))
        n = float(np.clip(n, 0.0, float(risk_cfg.get("max_notional", 3.6))))
        ratio = fill_ratio(n)
        if ratio < 0.999:
            partial_fill_events += 1
        n *= ratio
        if n <= 1e-8:
            block("zero_notional")
            continue
        pos = int(sides[i])
        s = impact_slip(n)
        entry_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, s, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = float(leverages[i])
        model_cooldown = int(cooldowns[i])
        fee_cost = cash * float(fee) * notional
        cash -= fee_cost
        fee_paid += fee_cost
        slippage_paid += cash * float(slip) * notional
        impact_paid += cash * max(0.0, s - float(slip)) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        peak_unrealized = 0.0
        entry_quality = float(qualities[i])
        entry_confidence = float(confs[i])
        open_ledger = {
            "entry_idx": int(i),
            "entry_timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
            "side": "LONG" if pos > 0 else "SHORT",
            "notional_exposure": float(notional),
            "leverage": float(leverage),
            "entry_equity": float(entry_equity),
            "fill_ratio": float(ratio),
        }

    if pos != 0:
        close_position(len(df) - 2, "forced_end")
    entries = max(long_entries + short_entries, 1)
    return {
        "eval": {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "trades_per_day": float(trades / _days(df)),
            "wr": float(wins / max(trades, 1)),
            "avg_notional": float(notional_sum / entries),
            "avg_leverage": float(leverage_sum / entries),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "liquidations": int(liquidations),
            "partial_fill_events": int(partial_fill_events),
            "fee_paid_equity": float(fee_paid),
            "slippage_paid_proxy": float(slippage_paid),
            "impact_paid_proxy": float(impact_paid),
            "funding_paid_equity": float(funding_paid),
            "liquidation_paid_equity": float(liquidation_paid),
            "entry_blocks": entry_blocks,
            "exits": exits,
        },
        "ledger": ledger,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-scope MuZero/AZ baseline re-audit with untouched 2026 OOS replay.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--source-policy", type=Path, default=ROOT / "data/ensemble/supervised/hf_entry_grid/hf_v4_balanced_h144.pkl")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--split-date", type=str, default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--policy-stride", type=int, default=3)
    p.add_argument("--policy-batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--samples", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.3e-3)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.010)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    all_2025 = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split = pd.Timestamp(args.split_date)
    ts = pd.to_datetime(all_2025["timestamp"], errors="coerce")
    train_df = all_2025.loc[ts < split].reset_index(drop=True)
    val_df = all_2025.loc[ts >= split].reset_index(drop=True)
    if train_df.empty or val_df.empty or eval_df.empty:
        raise ValueError("empty train/validation/eval split")

    entry_cfg0, risk_cfg0, _ = _load_selected(args.selection_report)
    cfg = _policy_config_from_bundle(args.source_policy)
    policy_path = args.model_dir / "hf_v4_clean_train_to_2025_10.pkl"
    policy, policy_meta = _train_clean_policy(
        train_df,
        cfg=cfg,
        stride_bars=int(args.policy_stride),
        batch_size=int(args.policy_batch_size),
        seed=int(args.seed),
        train_csv=args.train_csv,
        model_out=policy_path,
    )

    exit_path = args.model_dir / "hf_no_limit_exit_clean_train_to_2025_10.pkl"
    base_exit_model, exit_meta = _train_clean_exit_model(
        train_df,
        policy,
        dict(entry_cfg0),
        fee=float(args.fee),
        slip=float(args.slip),
        max_samples=int(args.samples),
        seed=int(args.seed) + 1,
        model_out=exit_path,
    )
    selected_controls, control_rows = _select_controls_on_validation(
        val_df,
        policy,
        base_exit_model,
        dict(entry_cfg0),
        dict(risk_cfg0),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    entry_cfg = dict(selected_controls["entry_config"])
    risk_cfg = dict(selected_controls["risk_config"])
    exit_cfg_base = dict(selected_controls["exit_config"])

    train_pre = _base_frame(train_df, policy, entry_cfg)
    val_pre_base = _base_frame(val_df, policy, entry_cfg)
    eval_pre_base = _base_frame(eval_df, policy, entry_cfg)

    train_feat, train_dec, _, _ = train_pre
    x, x_next, pi, value, reward, mz_label_meta = _make_targets(
        train_df,
        train_dec,
        train_feat.reindex(columns=FEATURE_COLS),
        search_horizon=int(args.horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=0.012,
        max_samples=int(args.samples),
        seed=int(args.seed) + 2,
    )
    mz_entry_net, mz_entry_mean, mz_entry_std, mz_entry_train_meta = _train_muzero(
        x,
        x_next,
        pi,
        value,
        reward,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed) + 2,
    )
    mz_entry = MZBundle(mz_entry_net, mz_entry_mean, mz_entry_std, list(FEATURE_COLS), ENTRY_ACTIONS)
    mz_entry_path = args.model_dir / "mz_entry_clean_train_to_2025_10.pt"
    torch.save(
        {
            "type": "clean_muzero_entry",
            "state_dict": mz_entry_net.state_dict(),
            "mean": mz_entry_mean.astype(np.float32),
            "std": mz_entry_std.astype(np.float32),
            "feature_cols": list(FEATURE_COLS),
            "actions": list(ENTRY_ACTIONS),
            "label_meta": mz_label_meta,
            "train_meta": mz_entry_train_meta,
        },
        mz_entry_path,
    )

    exit_x, exit_pi, exit_value, az_exit_label_meta = _rollout_exit_targets(
        train_df,
        policy,
        entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        horizon=int(args.horizon),
        max_samples=int(args.samples),
        seed=int(args.seed) + 3,
    )
    az_exit_net, az_exit_mean, az_exit_std, az_exit_train_meta = _train_pv(
        exit_x,
        exit_pi,
        exit_value,
        n_actions=len(EXIT_ACTIONS),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=1.5e-3,
        device=device,
        seed=int(args.seed) + 3,
    )
    az_exit_bundle = PVBundle(az_exit_net, az_exit_mean, az_exit_std, list(MODEL_COLS), EXIT_ACTIONS)
    az_exit_model = AZExitModel(az_exit_bundle, device)
    az_exit_path = args.model_dir / "az_exit_clean_train_to_2025_10.pt"
    torch.save(
        {
            "type": "clean_alphazero_exit",
            "exit": {
                "state_dict": az_exit_net.state_dict(),
                "mean": az_exit_mean.astype(np.float32),
                "std": az_exit_std.astype(np.float32),
                "feature_cols": list(MODEL_COLS),
                "actions": list(EXIT_ACTIONS),
                "meta": az_exit_train_meta,
            },
            "label_meta": az_exit_label_meta,
        },
        az_exit_path,
    )

    train_feat_mz, train_dec_mz, _, _, train_scores, train_probs, train_vals = _mz_entry_decisions(train_df, policy, entry_cfg, mz_entry, device=device)
    train_state0 = _state_frame(train_feat_mz, train_dec_mz, train_scores, train_probs, train_vals)
    risk_x, risk_x_next, risk_pi, risk_value, risk_reward, risk_label_meta = _risk_targets(
        train_df,
        train_state0,
        train_dec_mz,
        horizon=int(args.horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed) + 4,
    )
    az_risk_net, az_risk_mean, az_risk_std, az_risk_train_meta = _train_pv(
        risk_x,
        risk_pi,
        risk_value,
        n_actions=len(RISK_SCALES),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=1.2e-3,
        device=device,
        seed=int(args.seed) + 4,
    )
    az_risk_bundle = PVBundle(az_risk_net, az_risk_mean, az_risk_std, list(train_state0.columns), RISK_ACTIONS)
    train_dec_after_az = _apply_az_risk(train_dec_mz, train_state0, az_risk_bundle, device)
    train_state2 = _state_frame(train_feat_mz, train_dec_after_az, train_scores, train_probs, train_vals)
    s2_x, s2_x_next, s2_pi, s2_value, s2_reward, s2_label_meta = _risk_targets(
        train_df,
        train_state2,
        train_dec_after_az,
        horizon=int(args.horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed) + 5,
    )
    s2_net, s2_mean, s2_std, s2_train_meta = _train_mz_risk(
        s2_x,
        s2_x_next,
        s2_pi,
        s2_value,
        s2_reward,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=1.2e-3,
        device=device,
        seed=int(args.seed) + 5,
    )
    stage2_mz = MZRiskBundle(s2_net, s2_mean, s2_std, list(train_state2.columns), RISK_ACTIONS)
    az_risk_path = args.model_dir / "az_risk_clean_train_to_2025_10.pt"
    stage2_path = args.model_dir / "stage2_mz_clean_train_to_2025_10.pt"
    torch.save(
        {
            "type": "clean_alphazero_risk",
            "state_dict": az_risk_net.state_dict(),
            "mean": az_risk_mean.astype(np.float32),
            "std": az_risk_std.astype(np.float32),
            "feature_cols": list(train_state0.columns),
            "actions": list(RISK_ACTIONS),
            "scales": RISK_SCALES.astype(np.float32),
            "label_meta": risk_label_meta,
            "train_meta": az_risk_train_meta,
        },
        az_risk_path,
    )
    torch.save(
        {
            "type": "clean_stage2_muzero_risk",
            "state_dict": s2_net.state_dict(),
            "mean": s2_mean.astype(np.float32),
            "std": s2_std.astype(np.float32),
            "feature_cols": list(train_state2.columns),
            "actions": list(RISK_ACTIONS),
            "scales": RISK_SCALES.astype(np.float32),
            "label_meta": s2_label_meta,
            "train_meta": s2_train_meta,
        },
        stage2_path,
    )

    def build_stream(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        feat, dec_mz, close, fill, scores, probs, vals = _mz_entry_decisions(df, policy, entry_cfg, mz_entry, device=device)
        state0 = _state_frame(feat, dec_mz, scores, probs, vals)
        dec_az = _apply_az_risk(dec_mz, state0, az_risk_bundle, device)
        state2 = _state_frame(feat, dec_az, scores, probs, vals)
        return feat, dec_az, close, fill, state2

    val_feat, val_dec_az, val_close, val_fill, val_state2 = build_stream(val_df)
    eval_feat, eval_dec_az, eval_close, eval_fill, eval_state2 = build_stream(eval_df)
    val_base = _run_simple(
        "clean_stage1_mz_azrisk_azexit",
        val_df,
        policy,
        az_exit_model,
        entry_cfg,
        risk_cfg,
        {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
        (val_feat, val_dec_az, val_close, val_fill),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    eval_base = _run_simple(
        "clean_stage1_mz_azrisk_azexit",
        eval_df,
        policy,
        az_exit_model,
        entry_cfg,
        risk_cfg,
        {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
        (eval_feat, eval_dec_az, eval_close, eval_fill),
        fee=float(args.fee),
        slip=float(args.slip),
    )

    val_rows = [val_base]
    stage_cache: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    val_x2 = val_state2.reindex(columns=stage2_mz.feature_cols).to_numpy(dtype=np.float32)
    for gamma in (0.55, 0.70):
        for prior in (0.0, 0.08, 0.16):
            for depth in (1,):
                stage_cache[(gamma, prior, depth)] = _predict_mz_risk(stage2_mz, val_x2, device=device, gamma=gamma, prior_weight=prior, depth=depth)
                scores, probs, _vals = stage_cache[(gamma, prior, depth)]
                for score_floor in (-0.20, 0.00, 0.12):
                    idx = np.where(scores.max(axis=1) < float(score_floor), 3, np.argmax(scores, axis=1))
                    dec = _apply_scale(val_dec_az, idx)
                    name = f"clean_stage2_mz_g{gamma:.2f}_p{prior:.2f}_d{depth}_sf{score_floor:.2f}"
                    val_rows.append(
                        _run_simple(
                            name,
                            val_df,
                            policy,
                            az_exit_model,
                            entry_cfg,
                            risk_cfg,
                            {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
                            (val_feat, dec, val_close, val_fill),
                            fee=float(args.fee),
                            slip=float(args.slip),
                        )
                    )
    selected_stage = sorted(val_rows, key=lambda r: float(r["score"]), reverse=True)[0]
    selected_name = selected_stage["name"]

    def reconstruct_eval_dec(name: str) -> pd.DataFrame:
        if name == "clean_stage1_mz_azrisk_azexit":
            return eval_dec_az
        parts = name.split("_")
        gamma = float(parts[3].replace("g", ""))
        prior = float(parts[4].replace("p", ""))
        depth = int(parts[5].replace("d", ""))
        score_floor = float(parts[6].replace("sf", ""))
        eval_x2 = eval_state2.reindex(columns=stage2_mz.feature_cols).to_numpy(dtype=np.float32)
        scores, _, _ = _predict_mz_risk(stage2_mz, eval_x2, device=device, gamma=gamma, prior_weight=prior, depth=depth)
        idx = np.where(scores.max(axis=1) < float(score_floor), 3, np.argmax(scores, axis=1))
        return _apply_scale(eval_dec_az, idx)

    eval_dec_selected = reconstruct_eval_dec(selected_name)
    eval_pre_selected = (eval_feat, eval_dec_selected, eval_close, eval_fill)
    eval_selected = _run_simple(
        selected_name,
        eval_df,
        policy,
        az_exit_model,
        entry_cfg,
        risk_cfg,
        {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
        eval_pre_selected,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    cost_stress = {
        f"cost_{m:g}x": _run_simple(
            selected_name,
            eval_df,
            policy,
            az_exit_model,
            entry_cfg,
            risk_cfg,
            {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
            eval_pre_selected,
            fee=float(args.fee) * m,
            slip=float(args.slip) * m,
        )["eval"]
        for m in (1.0, 2.0, 3.0)
    }
    realistic = realistic_ledger_replay(
        eval_df,
        az_exit_model,
        risk_cfg,
        {"exit_threshold": 0.45, "min_exit_age": exit_cfg_base["min_exit_age"]},
        eval_pre_selected,
        fee=float(args.fee),
        slip=float(args.slip),
        funding_mult=1.0,
        impact_per_notional=0.00008,
        partial_fill_ratio=0.92,
        maintenance_margin=0.005,
        liquidation_fee=0.003,
    )
    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(realistic["ledger"]).to_csv(args.ledger_out, index=False)

    report = {
        "type": "clean_scope_muzero_az_reaudit_2026",
        "note": "Base policy, exit, MuZero entry, AZ risk, and Stage2 MuZero sleeve are trained only on data before split-date. Stage2 is selected on validation and replayed once on 2026 OOS.",
        "config": {
            "split_date": str(args.split_date),
            "fee": float(args.fee),
            "slip": float(args.slip),
            "device": device,
            "epochs": int(args.epochs),
            "samples": int(args.samples),
            "policy_stride": int(args.policy_stride),
        },
        "data_audit": {
            "source_audit_full_train_vs_eval": _audit(args.train_csv, args.eval_csv, policy),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
            "train_range": _range(train_df),
            "validation_range": _range(val_df),
            "eval_range": _range(eval_df),
            "train_validation_overlap": _overlap(train_df, val_df),
            "train_eval_overlap": _overlap(train_df, eval_df),
            "validation_eval_overlap": _overlap(val_df, eval_df),
        },
        "artifact_manifest": {
            "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
            "base_exit": {"path": str(exit_path), "sha256": _sha256(exit_path)},
            "mz_entry": {"path": str(mz_entry_path), "sha256": _sha256(mz_entry_path)},
            "az_exit": {"path": str(az_exit_path), "sha256": _sha256(az_exit_path)},
            "az_risk": {"path": str(az_risk_path), "sha256": _sha256(az_risk_path)},
            "stage2_mz": {"path": str(stage2_path), "sha256": _sha256(stage2_path)},
        },
        "policy_training_meta": policy_meta,
        "exit_sample_meta": exit_meta,
        "control_selection": {
            "selected": selected_controls,
            "top10": sorted(control_rows, key=lambda r: float(r["score"]), reverse=True)[:10],
        },
        "train_meta": {
            "mz_entry": mz_entry_train_meta,
            "az_exit": az_exit_train_meta,
            "az_risk": az_risk_train_meta,
            "stage2_mz": s2_train_meta,
        },
        "label_meta": {
            "mz_entry": mz_label_meta,
            "az_exit": az_exit_label_meta,
            "az_risk": risk_label_meta,
            "stage2_mz": s2_label_meta,
        },
        "validation_stage_rows": sorted(val_rows, key=lambda r: float(r["score"]), reverse=True),
        "eval_stage1": eval_base,
        "eval_selected": eval_selected,
        "cost_stress": cost_stress,
        "realistic_replay": realistic["eval"],
        "ledger_out": str(args.ledger_out),
        "decision": {
            "selected_stage_name": selected_name,
            "validation_selected_eval": selected_stage["eval"],
            "simple_oos": eval_selected["eval"],
            "realistic_oos": realistic["eval"],
            "cost_1x_2x_3x_survival": {
                k: bool(float(v["pnl"]) > 0.0) for k, v in cost_stress.items()
            },
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "selected_stage": selected_name,
                "simple_oos": eval_selected["eval"],
                "realistic_oos": realistic["eval"],
                "cost_stress": {k: v["pnl"] for k, v in cost_stress.items()},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
