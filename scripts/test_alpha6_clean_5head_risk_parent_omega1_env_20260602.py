#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import __main__
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    PolicyConfig,
    _ConstantClassifier,
    _ConstantRegressor,
    _predict_policy,
)
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import _json_default  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import ACTIVE_TEMPLATE  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_risk_allocator_20260602 import (  # noqa: E402
    ACTION_CASH,
    _load_variant_frames,
    _num,
)
from scripts.train_eval_alpha6_parent_dsac_risk4_allocator_20260602 import (  # noqa: E402
    _assert_no_forbidden_features,
)


MODEL_ID = "alpha6_clean_5head_risk_parent_omega1_env_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_BUNDLE = ROOT / "tmp/causal_regen_20260516/alpha6_clean_5head_risk_parent_20260602/current_tail111_clean_no_forbidden_bundle.joblib"

setattr(__main__, "_ConstantClassifier", _ConstantClassifier)
setattr(__main__, "_ConstantRegressor", _ConstantRegressor)


def _cfg_from_bundle(bundle: dict[str, Any]) -> PolicyConfig:
    raw = dict(bundle.get("config") or {})
    allowed = set(PolicyConfig.__dataclass_fields__.keys())
    return PolicyConfig(**{k: v for k, v in raw.items() if k in allowed})


def _predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(bundle["feature_cols"])
    _assert_no_forbidden_features(cols, where="alpha6_5head_bundle.feature_cols")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Alpha6 5-head feature contract mismatch. Missing columns: {missing[:30]}")
    x_raw = frame.loc[:, cols].copy()
    for col in cols:
        x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce")
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    x = bundle["pipeline"].transform(x_raw)
    policy_frame = frame.copy()
    if "atr14_pct" not in policy_frame.columns:
        policy_frame["atr14_pct"] = 0.003
    return _predict_policy(bundle["models"], x, policy_frame, _cfg_from_bundle(bundle))


def _to_decisions(pred: pd.DataFrame, *, threshold: float, leverage: float) -> pd.DataFrame:
    action = pd.to_numeric(pred["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    quality = pd.to_numeric(pred["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    active = (action != ACTION_CASH) & (quality >= float(threshold))
    side = np.where(action == 1, 1, np.where(action == 2, -1, 0)).astype(np.int64)
    side = np.where(active, side, 0).astype(np.int64)
    notional = pd.to_numeric(pred["notional"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    tp = pd.to_numeric(pred["take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    sl = pd.to_numeric(pred["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    lev = float(leverage)
    out = pd.DataFrame(
        {
            "action": np.where(active, action, 0).astype(np.int64),
            "side": side,
            "notional_exposure": np.where(active, notional, 0.0),
            "leverage": np.where(active, lev, 1.0),
            "position_fraction": np.where(active, notional / max(lev, 1e-8), 0.0),
            "take_profit": np.where(active, tp, 0.0),
            "stop_loss": np.where(active, np.abs(sl), 0.0),
            "max_hold_bars": np.zeros(len(pred), dtype=np.int64),
            "cooldown_bars": np.zeros(len(pred), dtype=np.int64),
            "quality_score": quality,
            "confidence": pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        }
    )
    return out


def _simulate_direct(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    i: int,
    row: pd.Series,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, dict[str, Any]]:
    action = int(row.get("action", 0) or 0)
    side = int(row.get("side", 0) or 0)
    notional = float(row.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "exit_i": int(i)}
    entry_i = min(int(i) + 1, len(frame) - 1)
    entry_px = float(arrays["open"][entry_i])
    if entry_px <= 0.0:
        return 0.0, {"active": 0, "exit_i": int(i)}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    entry = entry_px * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    tp = max(float(row.get("take_profit", 0.0) or 0.0), 1e-8)
    sl = max(abs(float(row.get("stop_loss", 0.0) or 0.0)), 1e-8)
    end_i = len(frame) - 1
    exit_fill: float | None = None
    exit_reason = "end"
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = float(arrays["high"][j]) / max(entry, 1e-12) - 1.0
            adverse = float(arrays["low"][j]) / max(entry, 1e-12) - 1.0
        else:
            favorable = entry / max(float(arrays["low"][j]), 1e-12) - 1.0
            adverse = entry / max(float(arrays["high"][j]), 1e-12) - 1.0
        if adverse <= -sl:
            trigger_px = entry * max(1.0 - sl, 1e-8) if side > 0 else entry / max(1.0 - sl, 1e-8)
            exit_fill = trigger_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            exit_reason = "stop_loss"
            end_i = j
            break
        if favorable >= tp:
            trigger_px = entry * (1.0 + tp) if side > 0 else entry / max(1.0 + tp, 1e-8)
            exit_fill = trigger_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            exit_reason = "take_profit"
            end_i = j
            break
    if exit_fill is None:
        exit_px = float(arrays["close"][end_i])
        exit_fill = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    qty = notional / max(entry, 1e-12)
    exit_notional = qty * max(float(exit_fill), 0.0)
    gross = exit_notional - notional if side > 0 else notional - exit_notional
    net = float(gross - fee_eff * notional - fee_eff * exit_notional)
    return net, {"active": 1, "exit_i": int(end_i), "win": int(net > 0.0), "exit_reason": exit_reason}


def _replay(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    arrays = {k: _num(frame, k) for k in ("open", "high", "low", "close")}
    active = (
        pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int).to_numpy() != 0
    ) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int).to_numpy() != 0
    ) & (
        pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy() > 0
    )
    next_allowed = 0
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    trades = 0
    wins = 0
    exits: dict[str, int] = {}
    long_entries = 0
    short_entries = 0
    exposure_sum = 0.0
    for i in range(len(frame) - 3):
        if i < next_allowed or not bool(active[i]):
            continue
        reward, meta = _simulate_direct(frame, arrays, i, dec.iloc[i], fee=fee, slip=slip, cost_mult=cost_mult)
        if int(meta.get("active", 0)) != 1:
            continue
        trades += 1
        wins += int(reward > 0.0)
        side = int(dec.iloc[i].get("side", 0) or 0)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        exposure_sum += float(dec.iloc[i].get("notional_exposure", 0.0) or 0.0)
        reason = str(meta.get("exit_reason", "unknown"))
        exits[reason] = exits.get(reason, 0) + 1
        equity += float(reward) * 100.0
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
        next_allowed = max(i + 1, int(meta.get("exit_i", i)))
    return {
        "pnl": float(equity),
        "mdd": float(mdd),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "exits": exits,
    }


def _score(row: dict[str, Any]) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p00")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--thresholds", type=int, default=30)
    ap.add_argument("--leverage", type=float, default=float(ACTIVE_TEMPLATE["leverage"]))
    ap.add_argument("--cost-mult", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = OUT_DIR / str(args.variant) / args.bundle.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.bundle)
    _assert_no_forbidden_features(list(bundle["feature_cols"]), where="alpha6_5head_bundle.feature_cols")
    _train_df, val_df, oos_df, _train_src, _val_src, _oos_src, overlay = _load_variant_frames(str(args.variant))
    val_pred = _predict_bundle(bundle, val_df)
    oos_pred = _predict_bundle(bundle, oos_df)
    parent_cfg = joblib.load(v31.DEFAULT_PARENT)["config"]
    fee = float(parent_cfg["fee"])
    slip = float(parent_cfg["slip"])
    q = pd.to_numeric(val_pred["quality_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if q.empty:
        raise RuntimeError("empty Alpha6 5-head quality predictions")
    thresholds = np.unique(np.quantile(q.to_numpy(dtype=np.float64), np.linspace(0.50, 0.995, max(2, int(args.thresholds)))))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for th in thresholds:
        val_dec = _to_decisions(val_pred, threshold=float(th), leverage=float(args.leverage))
        oos_dec = _to_decisions(oos_pred, threshold=float(th), leverage=float(args.leverage))
        val_bt = _replay(val_df, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_bt = _replay(oos_df, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        row = {
            "threshold": float(th),
            "split": "val",
            "variant": "alpha6_5head_risk_parent",
            "cost": int(args.cost_mult),
            **val_bt,
            "selection_score": _score(val_bt),
        }
        rows.append(row)
        rows.append(
            {
                "threshold": float(th),
                "split": "oos",
                "variant": "alpha6_5head_risk_parent",
                "cost": int(args.cost_mult),
                **oos_bt,
                "selection_score": _score(oos_bt),
            }
        )
        if best is None or row["selection_score"] > best["val"]["selection_score"]:
            best = {"threshold": float(th), "val": row, "oos": rows[-1]}
    assert best is not None
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Alpha6 clean 5-head parent owns action/quality/notional/TP/SL. Leverage is fixed to Omega1 template because this Alpha6 head family has no leverage head. Replay uses Omega1 runtime-style Cost3 direct accounting with no max-hold/cooldown.",
        "atr14_pct_source": "Omega1 frame column if present; otherwise explicit constant 0.003 because Alpha6 5-head TP/SL heads are ATR-multiple buckets.",
        "selection_basis": "VAL threshold selection; OOS report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "bundle": str(args.bundle),
        "feature_count": int(len(bundle["feature_cols"])),
        "forbidden_feature_count": 0,
        "leverage": float(args.leverage),
        "best": best,
        "overlay": overlay,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
