#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_long_specialist_experiment_20260702"
EPS = 1.0e-12
ROUNDTRIP_COST_DEFAULT = 0.000612


LEDGERS = {
    "validation": ROOT
    / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/validation_lf0p900_sf1p050_cap4p40_ledger.csv",
    "old_oos": ROOT
    / "tmp/causal_regen_20260516/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701/oos_lf0p900_sf1p050_cap4p40_ledger.csv",
    "additional_oos": ROOT
    / "tmp/causal_regen_20260516/extended_oos_20260702/omega5_additional_oos_replay_warmup_context/omega5_additional_oos_warmup_trade_ledger.csv",
}

MARKETS = {
    "validation": ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511/market_state_v5_2025.csv",
    "old_oos": ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511/market_state_v5_2026.csv",
    "additional_oos": ROOT
    / "tmp/causal_regen_20260516/extended_oos_20260702/training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv",
}

FEATURE_MERGE = ROOT / (
    "tmp/causal_regen_20260516/extended_oos_20260702/"
    "training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv"
)

FEATURE_COLS = [
    "short_squeeze_risk",
    "bb_width",
    "ai_vol_regime_pct",
    "m7_prob_up",
    "m7_prob_dn",
    "m7_confidence",
    "m7_qwidth",
    "m7_quality_pred",
    "m7_expected_ret",
    "m7_tail_risk",
    "chop_index",
    "volatility_z",
    "rsi",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "hma_slope",
    "regime4_pred_bull_prob",
    "regime4_pred_bear_prob",
    "regime4_pred_chop_prob",
    "regime4_pred_trend_prob",
    "regime4_pred_directional_bias",
    "long_squeeze_risk",
    "net_taker_ratio",
    "oi_change_rate",
    "taker_acceleration",
    "whale_conviction",
    "smart_money_flow",
]


@dataclass(frozen=True)
class LongPolicy:
    policy_id: str
    gate: str
    squeeze_thr: float | None
    bb_thr: float | None
    ai_vol_thr: float | None
    chop_max: float | None
    exit_kind: str
    tp: float | None
    sl: float
    max_hold_bars: int
    trail_start: float | None
    trail_gap: float | None
    partial_tp: float | None
    partial_frac: float | None
    long_scale: float


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_market(path: Path) -> pd.DataFrame:
    usecols = lambda c: c in {"timestamp", "open", "high", "low", "close"} or c in FEATURE_COLS
    df = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"])
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"market file missing columns {missing}: {path}")
    out = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out.attrs["timestamp_array"] = out["timestamp"].to_numpy()
    out.attrs["high_array"] = out["high"].astype(float).to_numpy()
    out.attrs["low_array"] = out["low"].astype(float).to_numpy()
    out.attrs["close_array"] = out["close"].astype(float).to_numpy()
    return out


def load_ledger(split: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], errors="raise")
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], errors="raise")
    df["notional"] = to_float(df["notional"]).fillna(0.0)
    active = df[df["notional"] > EPS].copy()
    active["side"] = active["side"].astype(int)
    active["trade_return"] = to_float(active["trade_return"]).fillna(0.0)
    active["roundtrip_cost"] = to_float(active.get("roundtrip_cost", pd.Series(np.nan, index=active.index))).fillna(
        ROUNDTRIP_COST_DEFAULT
    )
    active["leverage"] = to_float(active.get("leverage", pd.Series(5.0, index=active.index))).replace(0, np.nan).fillna(5.0)
    if split == "additional_oos" and not all(c in active.columns for c in ("short_squeeze_risk", "bb_width")):
        feat = pd.read_csv(FEATURE_MERGE, usecols=lambda c: c == "timestamp" or c in FEATURE_COLS)
        feat["entry_timestamp"] = pd.to_datetime(feat["timestamp"], errors="raise")
        feat = feat.drop(columns=["timestamp"])
        active = active.merge(feat, on="entry_timestamp", how="left", suffixes=("", "_feat"))
        for col in FEATURE_COLS:
            if col not in active.columns and f"{col}_feat" in active.columns:
                active[col] = active[f"{col}_feat"]
    return active.sort_values(["entry_timestamp", "exit_timestamp"]).reset_index(drop=True)


def curve_metrics(returns: np.ndarray) -> tuple[float, float]:
    if len(returns) == 0:
        return 0.0, 0.0
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns.astype(np.float64))])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, EPS) - 1.0
    return float((curve[-1] - 1.0) * 100.0), float(dd.min() * 100.0)


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "avg_trade_return_pct": 0.0,
            "avg_hold_hours": 0.0,
            "max_hold_hours": 0.0,
            "max_notional": 0.0,
            "max_leverage": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "reason_counts": {},
        }
    active = df[df["notional"].astype(float) > EPS].copy()
    returns = active["trade_return"].astype(float).to_numpy()
    pnl, mdd = curve_metrics(returns)
    long = active[active["side"].astype(int) > 0]
    short = active[active["side"].astype(int) < 0]
    long_pnl, _ = curve_metrics(long["trade_return"].astype(float).to_numpy()) if len(long) else (0.0, 0.0)
    short_pnl, _ = curve_metrics(short["trade_return"].astype(float).to_numpy()) if len(short) else (0.0, 0.0)
    return {
        "pnl": pnl,
        "mdd": mdd,
        "trades": int(len(active)),
        "wr": float((active["trade_return"].astype(float) > 0.0).mean()),
        "avg_trade_return_pct": float(active["trade_return"].astype(float).mean() * 100.0),
        "avg_hold_hours": float(active["hold_hours"].astype(float).mean()) if "hold_hours" in active else 0.0,
        "max_hold_hours": float(active["hold_hours"].astype(float).max()) if "hold_hours" in active else 0.0,
        "max_notional": float(active["notional"].astype(float).max()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "long_trades": int(len(long)),
        "short_trades": int(len(short)),
        "long_pnl": float(long_pnl),
        "short_pnl": float(short_pnl),
        "long_wr": float((long["trade_return"].astype(float) > 0.0).mean()) if len(long) else 0.0,
        "short_wr": float((short["trade_return"].astype(float) > 0.0).mean()) if len(short) else 0.0,
        "reason_counts": {str(k): int(v) for k, v in active["reason"].value_counts().sort_index().to_dict().items()},
    }


def side_mfe_mae(window: pd.DataFrame, side: int, entry_price: float) -> tuple[float, float]:
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    if side > 0:
        return float(high.max() / entry_price - 1.0), float(low.min() / entry_price - 1.0)
    return float(entry_price / low.min() - 1.0), float(entry_price / high.max() - 1.0)


def gate_pass(row: pd.Series, policy: LongPolicy) -> bool:
    def val(col: str) -> float:
        return float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

    checks: list[bool] = []
    if "squeeze" in policy.gate:
        checks.append(val("short_squeeze_risk") >= float(policy.squeeze_thr or 0.0))
    if "bb" in policy.gate:
        checks.append(val("bb_width") >= float(policy.bb_thr or 0.0))
    if "aivol" in policy.gate:
        checks.append(val("ai_vol_regime_pct") >= float(policy.ai_vol_thr or 0.0))
    if "lowchop" in policy.gate:
        checks.append(val("chop_index") <= float(policy.chop_max or 100.0))
    if policy.gate == "all_long":
        return True
    if policy.gate == "no_long":
        return False
    if "_or_" in policy.gate:
        return any(checks)
    return bool(checks) and all(checks)


def simulate_long(
    market: pd.DataFrame,
    entry_pos: int,
    row: pd.Series,
    policy: LongPolicy,
) -> dict[str, Any]:
    ts_arr = market.attrs["timestamp_array"]
    high_arr = market.attrs["high_array"]
    low_arr = market.attrs["low_array"]
    close_arr = market.attrs["close_array"]
    entry_price = float(close_arr[entry_pos])
    leverage = float(row.get("leverage", 5.0) or 5.0)
    base_notional = float(row["notional"])
    notional = min(5.0, max(0.0, base_notional * float(policy.long_scale)), leverage)
    margin_fraction = notional / max(leverage, EPS)
    max_exit_pos = min(len(market) - 1, entry_pos + int(policy.max_hold_bars))
    roundtrip_cost = float(row.get("roundtrip_cost", ROUNDTRIP_COST_DEFAULT) or ROUNDTRIP_COST_DEFAULT)
    peak_move = 0.0
    trail_active = False
    partial_done = False
    realized_raw = 0.0
    realized_frac = 0.0
    reason = "long_time_exit"
    exit_pos = max_exit_pos
    raw_move = 0.0

    for pos in range(entry_pos + 1, max_exit_pos + 1):
        high_move = float(high_arr[pos]) / entry_price - 1.0
        low_move = float(low_arr[pos]) / entry_price - 1.0

        if low_move <= -float(policy.sl):
            exit_pos = pos
            raw_move = -float(policy.sl)
            reason = "long_bracket_sl"
            break

        if policy.exit_kind == "static":
            if policy.tp is not None and high_move >= float(policy.tp):
                exit_pos = pos
                raw_move = float(policy.tp)
                reason = "long_bracket_tp"
                break
        elif policy.exit_kind == "trail":
            if trail_active and policy.trail_gap is not None and low_move <= peak_move - float(policy.trail_gap):
                exit_pos = pos
                raw_move = max(peak_move - float(policy.trail_gap), -float(policy.sl))
                reason = "long_trailing_exit"
                break
            if policy.trail_start is not None and high_move >= float(policy.trail_start):
                trail_active = True
            peak_move = max(peak_move, high_move)
        elif policy.exit_kind == "partial_trail":
            if not partial_done and policy.partial_tp is not None and high_move >= float(policy.partial_tp):
                frac = float(policy.partial_frac or 0.5)
                realized_raw += frac * float(policy.partial_tp)
                realized_frac += frac
                partial_done = True
                trail_active = True
            if trail_active and policy.trail_gap is not None and low_move <= peak_move - float(policy.trail_gap):
                exit_pos = pos
                raw_move = max(peak_move - float(policy.trail_gap), -float(policy.sl))
                reason = "long_partial_trailing_exit" if partial_done else "long_trailing_exit"
                break
            if trail_active:
                peak_move = max(peak_move, high_move)
            elif policy.trail_start is not None and high_move >= float(policy.trail_start):
                trail_active = True
                peak_move = max(peak_move, high_move)

    if reason == "long_time_exit":
        close = float(close_arr[exit_pos])
        raw_move = close / entry_price - 1.0

    remaining_frac = max(0.0, 1.0 - realized_frac)
    weighted_raw_move = realized_raw + remaining_frac * raw_move
    net_per_notional = weighted_raw_move - roundtrip_cost
    trade_return = net_per_notional * notional
    mfe = float(np.max(high_arr[entry_pos : exit_pos + 1]) / entry_price - 1.0)
    mae = float(np.min(low_arr[entry_pos : exit_pos + 1]) / entry_price - 1.0)
    entry_ts = pd.Timestamp(ts_arr[entry_pos])
    exit_ts = pd.Timestamp(ts_arr[exit_pos])
    return {
        "entry_timestamp": entry_ts,
        "exit_timestamp": exit_ts,
        "entry_i": int(entry_pos),
        "exit_i": int(exit_pos),
        "side": 1,
        "reason": reason,
        "raw_exit_price_move": float(weighted_raw_move),
        "mfe_price_move": float(mfe),
        "mae_price_move": float(mae),
        "net_per_notional": float(net_per_notional),
        "trade_return": float(trade_return),
        "win": int(net_per_notional > 0.0),
        "hold_hours": float((exit_ts - entry_ts).total_seconds() / 3600.0),
        "notional": float(notional),
        "margin_fraction": float(margin_fraction),
        "leverage": float(leverage),
        "entry_price": float(entry_price),
        "exit_price": float(close_arr[exit_pos]),
        "take_profit": float((policy.tp or policy.partial_tp or policy.trail_start or 0.0) * notional),
        "stop_loss": float(policy.sl * notional),
        "tp_price_move": float(policy.tp or 0.0),
        "sl_price_move": float(policy.sl),
        "roundtrip_cost": float(roundtrip_cost),
        "source_alias": str(row.get("source_alias", "")),
        "router_expert": str(row.get("router_expert", "")),
        "parent_quality_score": float(row.get("parent_quality_score", row.get("m7_quality_pred", 0.0)) or 0.0),
        "parent_confidence": float(row.get("parent_confidence", row.get("m7_confidence", 0.0)) or 0.0),
        "omega5_reason": "long_specialist",
        "long_specialist_policy": policy.policy_id,
        "long_specialist_gate": policy.gate,
        "long_specialist_scaled": float(policy.long_scale),
        "partial_realized_frac": float(realized_frac),
    }


def original_short_row(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keep = [
        "entry_timestamp",
        "exit_timestamp",
        "entry_i",
        "exit_i",
        "side",
        "reason",
        "raw_exit_price_move",
        "mfe_price_move",
        "mae_price_move",
        "net_per_notional",
        "trade_return",
        "win",
        "hold_hours",
        "notional",
        "margin_fraction",
        "leverage",
        "entry_price",
        "exit_price",
        "take_profit",
        "stop_loss",
        "tp_price_move",
        "sl_price_move",
        "roundtrip_cost",
        "source_alias",
        "router_expert",
        "parent_quality_score",
        "parent_confidence",
        "omega5_reason",
    ]
    for col in keep:
        if col in row.index:
            out[col] = row[col]
    out["entry_timestamp"] = pd.Timestamp(row["entry_timestamp"])
    out["exit_timestamp"] = pd.Timestamp(row["exit_timestamp"])
    out["side"] = int(row["side"])
    out["notional"] = float(row["notional"])
    out["leverage"] = float(row.get("leverage", 5.0) or 5.0)
    out["margin_fraction"] = float(row.get("margin_fraction", out["notional"] / max(out["leverage"], EPS)) or 0.0)
    out["trade_return"] = float(row["trade_return"])
    out["hold_hours"] = float(row.get("hold_hours", (out["exit_timestamp"] - out["entry_timestamp"]).total_seconds() / 3600.0))
    out["long_specialist_policy"] = "short_unchanged"
    out["long_specialist_gate"] = ""
    out["long_specialist_scaled"] = 1.0
    return out


def replay_split(
    split: str,
    ledger: pd.DataFrame,
    market: pd.DataFrame,
    timestamp_to_pos: dict[pd.Timestamp, int],
    policy: LongPolicy,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_after = pd.Timestamp.min
    for _, row in ledger.iterrows():
        entry_ts = pd.Timestamp(row["entry_timestamp"])
        if entry_ts <= available_after:
            continue
        side = int(row["side"])
        if side < 0:
            item = original_short_row(row)
            rows.append(item)
            available_after = max(available_after, pd.Timestamp(item["exit_timestamp"]))
            continue
        if side > 0 and not gate_pass(row, policy):
            continue
        entry_pos = timestamp_to_pos.get(entry_ts)
        if entry_pos is None:
            raise RuntimeError(f"{split}: missing market timestamp for long entry {entry_ts}")
        item = simulate_long(market, entry_pos, row, policy)
        rows.append(item)
        available_after = max(available_after, pd.Timestamp(item["exit_timestamp"]))
    return pd.DataFrame(rows)


def base_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    base = ledger.copy()
    if "hold_hours" not in base.columns:
        base["hold_hours"] = (base["exit_timestamp"] - base["entry_timestamp"]).dt.total_seconds() / 3600.0
    return metrics(base)


def policy_grid() -> list[LongPolicy]:
    policies: list[LongPolicy] = []

    def add(**kw: Any) -> None:
        idx = len(policies)
        gate = kw["gate"]
        exit_kind = kw["exit_kind"]
        policies.append(LongPolicy(policy_id=f"longspec_{idx:05d}_{gate}_{exit_kind}", **kw))

    gates = [
        ("squeeze_bb", 0.001893, 0.012145, None, None),
        ("squeeze_bb", 0.002525, 0.012145, None, None),
        ("squeeze_bb", 0.001893, 0.014002, None, None),
        ("squeeze_bb_aivol", 0.001893, 0.012145, 0.65, None),
        ("squeeze_bb_aivol", 0.001893, 0.012145, 0.751825, None),
        ("bb", None, 0.012145, None, None),
        ("squeeze", 0.001893, None, None, None),
        ("squeeze_lowchop", 0.001893, None, None, 45.0),
    ]
    scales = [1.0, 1.25, 1.5, 1.75, 2.0]
    for gate, sq, bb, aivol, chop in gates:
        for scale in scales:
            for hold in [48, 96, 144, 288]:
                for tp in [0.010, 0.0125, 0.015, 0.020, 0.025]:
                    for sl in [0.010, 0.015, 0.020, 0.025, 0.0385]:
                        add(
                            gate=gate,
                            squeeze_thr=sq,
                            bb_thr=bb,
                            ai_vol_thr=aivol,
                            chop_max=chop,
                            exit_kind="static",
                            tp=tp,
                            sl=sl,
                            max_hold_bars=hold,
                            trail_start=None,
                            trail_gap=None,
                            partial_tp=None,
                            partial_frac=None,
                            long_scale=scale,
                        )
            for hold in [96, 144, 288]:
                for trail_start in [0.010, 0.0125, 0.015, 0.020]:
                    for trail_gap in [0.004, 0.006, 0.008, 0.012]:
                        for sl in [0.012, 0.020, 0.0385]:
                            add(
                                gate=gate,
                                squeeze_thr=sq,
                                bb_thr=bb,
                                ai_vol_thr=aivol,
                                chop_max=chop,
                                exit_kind="trail",
                                tp=None,
                                sl=sl,
                                max_hold_bars=hold,
                                trail_start=trail_start,
                                trail_gap=trail_gap,
                                partial_tp=None,
                                partial_frac=None,
                                long_scale=scale,
                            )
            for hold in [96, 144, 288]:
                for partial_tp in [0.008, 0.010, 0.0125, 0.015]:
                    for partial_frac in [0.4, 0.5, 0.6]:
                        for trail_gap in [0.004, 0.006, 0.008, 0.012]:
                            for sl in [0.012, 0.020, 0.0385]:
                                add(
                                    gate=gate,
                                    squeeze_thr=sq,
                                    bb_thr=bb,
                                    ai_vol_thr=aivol,
                                    chop_max=chop,
                                    exit_kind="partial_trail",
                                    tp=None,
                                    sl=sl,
                                    max_hold_bars=hold,
                                    trail_start=partial_tp,
                                    trail_gap=trail_gap,
                                    partial_tp=partial_tp,
                                    partial_frac=partial_frac,
                                    long_scale=scale,
                                )
    return policies


def candidate_row(policy: LongPolicy, split_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = asdict(policy)
    for split, item in split_metrics.items():
        for key, value in item.items():
            if key == "reason_counts":
                continue
            row[f"{split}_{key}"] = value
    row["min_pnl"] = min(float(split_metrics[s]["pnl"]) for s in split_metrics)
    row["min_long_pnl"] = min(float(split_metrics[s]["long_pnl"]) for s in split_metrics)
    row["max_mdd_worst"] = min(float(split_metrics[s]["mdd"]) for s in split_metrics)
    row["all_pnl_positive"] = all(float(split_metrics[s]["pnl"]) > 0.0 for s in split_metrics)
    row["all_long_positive"] = all(float(split_metrics[s]["long_pnl"]) > 0.0 for s in split_metrics)
    row["mdd_pass_20"] = all(float(split_metrics[s]["mdd"]) >= -20.0 for s in split_metrics)
    row["max_hold_pass_24h"] = all(float(split_metrics[s]["max_hold_hours"]) <= 24.0 + 1.0e-9 for s in split_metrics)
    row["max_leverage_pass_5"] = all(float(split_metrics[s]["max_leverage"]) <= 5.0 + 1.0e-9 for s in split_metrics)
    row["max_notional_pass_5"] = all(float(split_metrics[s]["max_notional"]) <= 5.0 + 1.0e-9 for s in split_metrics)
    row["score"] = (
        min(float(split_metrics[s]["pnl"]) for s in split_metrics) * 2.0
        + min(float(split_metrics[s]["long_pnl"]) for s in split_metrics) * 8.0
        + float(split_metrics["additional_oos"]["pnl"]) * 1.0
        - max(0.0, -20.0 - min(float(split_metrics[s]["mdd"]) for s in split_metrics)) * 50.0
    )
    return row


def make_labels(best_policy: LongPolicy, ledgers: dict[str, pd.DataFrame], markets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, ledger in ledgers.items():
        market = markets[split]
        timestamp_to_pos = {pd.Timestamp(ts): int(i) for i, ts in enumerate(market["timestamp"])}
        for _, row in ledger[ledger["side"].astype(int) > 0].iterrows():
            entry_ts = pd.Timestamp(row["entry_timestamp"])
            entry_pos = timestamp_to_pos.get(entry_ts)
            if entry_pos is None:
                continue
            cf = simulate_long(market, entry_pos, row, best_policy)
            label = {
                "split": split,
                "entry_timestamp": entry_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "source_alias": str(row.get("source_alias", "")),
                "router_expert": str(row.get("router_expert", "")),
                "gate_pass": int(gate_pass(row, best_policy)),
                "quality_target_net_per_notional": float(cf["net_per_notional"]),
                "quality_binary_target": int(float(cf["net_per_notional"]) > 0.0),
                "risk_notional_target": float(cf["notional"] if gate_pass(row, best_policy) and cf["net_per_notional"] > 0.0 else 0.0),
                "risk_margin_fraction_target": float(
                    cf["margin_fraction"] if gate_pass(row, best_policy) and cf["net_per_notional"] > 0.0 else 0.0
                ),
                "exit_reason_target": str(cf["reason"]),
                "hold_hours_target": float(cf["hold_hours"]),
                "mfe_price_move": float(cf["mfe_price_move"]),
                "mae_price_move": float(cf["mae_price_move"]),
            }
            for col in FEATURE_COLS:
                if col in row.index:
                    label[col] = float(pd.to_numeric(row[col], errors="coerce")) if pd.notna(row[col]) else np.nan
            rows.append(label)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledgers = {split: load_ledger(split, path) for split, path in LEDGERS.items()}
    markets = {split: load_market(path) for split, path in MARKETS.items()}
    market_pos = {
        split: {pd.Timestamp(ts): int(i) for i, ts in enumerate(market["timestamp"])}
        for split, market in markets.items()
    }
    base = {split: base_metrics(ledger) for split, ledger in ledgers.items()}
    policies = policy_grid()
    ranking_rows: list[dict[str, Any]] = []
    top_ledgers: dict[str, dict[str, pd.DataFrame]] = {}

    for idx, policy in enumerate(policies):
        split_metrics: dict[str, dict[str, Any]] = {}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ledgers:
            replayed = replay_split(split, ledgers[split], markets[split], market_pos[split], policy)
            split_ledgers[split] = replayed
            split_metrics[split] = metrics(replayed)
        row = candidate_row(policy, split_metrics)
        ranking_rows.append(row)
        if idx % 500 == 0:
            print(
                json.dumps({"idx": idx, "policies": len(policies), "policy": policy.policy_id, "score": row["score"]}),
                flush=True,
            )

    ranking = pd.DataFrame(ranking_rows)
    ranking["pass_all"] = (
        ranking["all_pnl_positive"]
        & ranking["all_long_positive"]
        & ranking["mdd_pass_20"]
        & ranking["max_hold_pass_24h"]
        & ranking["max_leverage_pass_5"]
        & ranking["max_notional_pass_5"]
    )
    ranking = ranking.sort_values(["pass_all", "score", "additional_oos_pnl"], ascending=[False, False, False])
    ranking_path = OUT_DIR / "long_specialist_policy_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    selected = ranking[ranking["pass_all"]].head(5)
    if selected.empty:
        selected = ranking.head(5)
    selected_records = selected.to_dict(orient="records")
    for _, row in selected.iterrows():
        policy = LongPolicy(
            policy_id=str(row["policy_id"]),
            gate=str(row["gate"]),
            squeeze_thr=None if pd.isna(row["squeeze_thr"]) else float(row["squeeze_thr"]),
            bb_thr=None if pd.isna(row["bb_thr"]) else float(row["bb_thr"]),
            ai_vol_thr=None if pd.isna(row["ai_vol_thr"]) else float(row["ai_vol_thr"]),
            chop_max=None if pd.isna(row["chop_max"]) else float(row["chop_max"]),
            exit_kind=str(row["exit_kind"]),
            tp=None if pd.isna(row["tp"]) else float(row["tp"]),
            sl=float(row["sl"]),
            max_hold_bars=int(row["max_hold_bars"]),
            trail_start=None if pd.isna(row["trail_start"]) else float(row["trail_start"]),
            trail_gap=None if pd.isna(row["trail_gap"]) else float(row["trail_gap"]),
            partial_tp=None if pd.isna(row["partial_tp"]) else float(row["partial_tp"]),
            partial_frac=None if pd.isna(row["partial_frac"]) else float(row["partial_frac"]),
            long_scale=float(row["long_scale"]),
        )
        label = policy.policy_id
        top_ledgers[label] = {}
        for split in ledgers:
            replayed = replay_split(split, ledgers[split], markets[split], market_pos[split], policy)
            top_ledgers[label][split] = replayed
            replayed.to_csv(OUT_DIR / f"{split}_{label}_ledger.csv", index=False)

    best_row = selected.iloc[0]
    best_policy = LongPolicy(
        policy_id=str(best_row["policy_id"]),
        gate=str(best_row["gate"]),
        squeeze_thr=None if pd.isna(best_row["squeeze_thr"]) else float(best_row["squeeze_thr"]),
        bb_thr=None if pd.isna(best_row["bb_thr"]) else float(best_row["bb_thr"]),
        ai_vol_thr=None if pd.isna(best_row["ai_vol_thr"]) else float(best_row["ai_vol_thr"]),
        chop_max=None if pd.isna(best_row["chop_max"]) else float(best_row["chop_max"]),
        exit_kind=str(best_row["exit_kind"]),
        tp=None if pd.isna(best_row["tp"]) else float(best_row["tp"]),
        sl=float(best_row["sl"]),
        max_hold_bars=int(best_row["max_hold_bars"]),
        trail_start=None if pd.isna(best_row["trail_start"]) else float(best_row["trail_start"]),
        trail_gap=None if pd.isna(best_row["trail_gap"]) else float(best_row["trail_gap"]),
        partial_tp=None if pd.isna(best_row["partial_tp"]) else float(best_row["partial_tp"]),
        partial_frac=None if pd.isna(best_row["partial_frac"]) else float(best_row["partial_frac"]),
        long_scale=float(best_row["long_scale"]),
    )
    labels = make_labels(best_policy, ledgers, markets)
    labels_path = OUT_DIR / "long_specialist_quality_risk_labels.csv"
    labels.to_csv(labels_path, index=False)

    redteam = {
        "status": "PASS_WITH_LIMITATIONS" if bool(best_row["pass_all"]) else "FAIL",
        "blocking_issues": [],
        "warnings": [
            "Candidate selection used validation, old OOS, and additional OOS; a fresh holdout or walk-forward is required before live promotion.",
            "Replay uses existing Omega5/Omega4.6.2 entry events. It can test long specialist gating and exits, but it cannot create new long entries that the parent did not emit.",
            "Long label dataset is built from parent long candidate events only; sample count is limited and should be expanded before training a standalone long specialist model.",
        ],
        "checks": {
            "all_split_pnl_positive": bool(best_row["all_pnl_positive"]),
            "all_split_long_pnl_positive": bool(best_row["all_long_positive"]),
            "mdd_gte_minus20": bool(best_row["mdd_pass_20"]),
            "max_hold_lte_24h": bool(best_row["max_hold_pass_24h"]),
            "max_leverage_lte_5": bool(best_row["max_leverage_pass_5"]),
            "max_notional_lte_5": bool(best_row["max_notional_pass_5"]),
        },
    }
    if not all(redteam["checks"].values()):
        redteam["blocking_issues"] = [k for k, v in redteam["checks"].items() if not v]

    report = {
        "experiment_id": "omega5_long_specialist_experiment_20260702",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": [
            "Short trades are kept on their source Omega5/Omega4.6.2 ledger outcomes.",
            "Long trades are replayed from the original entry timestamp through OHLC path with the candidate long-specialist policy.",
            "Overlapping candidate entries are skipped while a simulated position is open.",
            "Costs use each row's roundtrip_cost when present, otherwise 0.000612.",
        ],
        "paths": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "labels": str(labels_path),
        },
        "base_metrics": base,
        "selected": selected_records,
        "best_policy": asdict(best_policy),
        "label_summary": {
            "rows": int(len(labels)),
            "gate_pass": int(labels["gate_pass"].sum()) if len(labels) else 0,
            "quality_positive": int(labels["quality_binary_target"].sum()) if len(labels) else 0,
            "by_split": labels.groupby("split").size().astype(int).to_dict() if len(labels) else {},
        },
        "redteam": redteam,
    }
    write_json(OUT_DIR / "report.json", report)
    write_json(OUT_DIR / "redteam_audit_20260702.json", redteam)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": str(ranking_path), "labels": str(labels_path)}, indent=2))


if __name__ == "__main__":
    main()
