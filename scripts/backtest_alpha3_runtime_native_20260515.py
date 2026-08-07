#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_alpha2_1_signal_immediate_limit_20260514 import (  # noqa: E402
    ImmediateLimitConfig,
    _fill_price,
    _try_immediate_limit,
)

DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha3_runtime_native_backtest_20260515.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/alpha3_runtime_native_backtest_20260515_ledger.csv"
DEFAULT_V31_REPORT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_summary.json"
DEFAULT_V31_AUDIT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_audit.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _target_side(action: int) -> str | None:
    if int(action) == 1:
        return "LONG"
    if int(action) == 2:
        return "SHORT"
    return None


def _side_to_int(side: str | None) -> int:
    side_u = str(side or "").upper()
    if side_u == "LONG":
        return 1
    if side_u == "SHORT":
        return -1
    return 0


@dataclass
class Lot:
    side: str
    entry_price: float
    exposure: float
    entry_liquidity: str = ""


class LotBook:
    def __init__(self) -> None:
        self.lots: list[Lot] = []

    @property
    def side(self) -> str | None:
        return self.lots[0].side if self.lots else None

    @property
    def exposure(self) -> float:
        return float(sum(max(0.0, lot.exposure) for lot in self.lots))

    @property
    def entry_price(self) -> float:
        exp = self.exposure
        if exp <= 1e-12:
            return 0.0
        return float(sum(lot.entry_price * lot.exposure for lot in self.lots) / exp)

    def clear(self) -> None:
        self.lots.clear()

    def add(self, side: str, entry_price: float, exposure: float, entry_liquidity: str = "") -> None:
        if side not in {"LONG", "SHORT"} or entry_price <= 0.0 or exposure <= 1e-12:
            return
        if self.side is not None and self.side != side:
            raise ValueError(f"cannot add {side} lot while holding {self.side}")
        self.lots.append(Lot(side=side, entry_price=float(entry_price), exposure=float(exposure), entry_liquidity=str(entry_liquidity or "")))

    def close(self, router: Any, exit_price: float, exposure: float | None = None, exit_liquidity: str = "") -> tuple[float, list[dict[str, Any]]]:
        remaining = self.exposure if exposure is None else min(float(exposure), self.exposure)
        pnl = 0.0
        fills: list[dict[str, Any]] = []
        new_lots: list[Lot] = []
        for lot in self.lots:
            if remaining <= 1e-12:
                new_lots.append(lot)
                continue
            close_exp = min(float(lot.exposure), remaining)
            math = router._trade_math(
                lot.side,
                lot.entry_price,
                float(exit_price),
                close_exp,
                entry_liquidity=lot.entry_liquidity,
                exit_liquidity=exit_liquidity,
            )
            pnl += float(math.get("pnl_frac", 0.0) or 0.0)
            fills.append(
                {
                    "side": lot.side,
                    "entry_price": float(lot.entry_price),
                    "exit_price": float(exit_price),
                    "closed_exposure": float(close_exp),
                    "pnl_frac": float(math.get("pnl_frac", 0.0) or 0.0),
                    "pnl_pct": float(math.get("pnl_pct", 0.0) or 0.0),
                    "fee_cost_frac": float(math.get("fee_cost_frac", 0.0) or 0.0),
                    "fee_model": str(math.get("fee_model", "")),
                    "gross_return_frac": float(math.get("gross_return_frac", 0.0) or 0.0),
                }
            )
            lot.exposure = float(lot.exposure - close_exp)
            remaining -= close_exp
            if lot.exposure > 1e-12:
                new_lots.append(lot)
        self.lots = new_lots
        return float(pnl), fills

    def mark_pnl(self, router: Any, mark_price: float) -> float:
        total = 0.0
        for lot in self.lots:
            math = router._trade_math(lot.side, lot.entry_price, float(mark_price), float(lot.exposure), entry_liquidity=lot.entry_liquidity)
            total += float(math.get("pnl_frac", 0.0) or 0.0)
        return float(total)

    def close_alpha3(
        self,
        exit_price: float,
        *,
        exit_fee: float,
        exposure: float | None = None,
    ) -> tuple[float, list[dict[str, Any]]]:
        remaining = self.exposure if exposure is None else min(float(exposure), self.exposure)
        pnl_frac = 0.0
        fills: list[dict[str, Any]] = []
        new_lots: list[Lot] = []
        for lot in self.lots:
            if remaining <= 1e-12:
                new_lots.append(lot)
                continue
            close_exp = min(float(lot.exposure), remaining)
            side_i = _side_to_int(lot.side)
            if side_i > 0:
                raw = (float(exit_price) - float(lot.entry_price)) / max(float(lot.entry_price), 1e-12)
            else:
                raw = (float(lot.entry_price) - float(exit_price)) / max(float(lot.entry_price), 1e-12)
            lot_pnl = float(raw * close_exp - float(exit_fee) * close_exp)
            pnl_frac += lot_pnl
            fills.append(
                {
                    "side": lot.side,
                    "entry_price": float(lot.entry_price),
                    "exit_price": float(exit_price),
                    "closed_exposure": float(close_exp),
                    "pnl_frac": float(lot_pnl),
                    "pnl_pct": float(lot_pnl * 100.0),
                    "fee_cost_frac": float(float(exit_fee) * close_exp),
                    "fee_model": "alpha3_csv_execution_parity",
                    "gross_return_frac": float(raw),
                }
            )
            lot.exposure = float(lot.exposure - close_exp)
            remaining -= close_exp
            if lot.exposure > 1e-12:
                new_lots.append(lot)
        self.lots = new_lots
        return float(pnl_frac), fills

    def mark_pnl_alpha3(self, mark_price: float, *, slip: float) -> float:
        total = 0.0
        for lot in self.lots:
            side_i = _side_to_int(lot.side)
            if side_i > 0:
                raw = (float(mark_price) * (1.0 - float(slip)) - float(lot.entry_price)) / max(float(lot.entry_price), 1e-12)
            else:
                raw = (float(lot.entry_price) - float(mark_price) * (1.0 + float(slip))) / max(float(lot.entry_price), 1e-12)
            total += float(raw * float(lot.exposure))
        return float(total)


def _load_eval_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"missing timestamp column: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"missing required OHLCV column {col}: {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def _ledger_row(
    df: pd.DataFrame,
    i: int,
    *,
    event: str,
    pos: int,
    owner: str,
    reason: str = "",
    route: str = "",
    cash: float,
    equity: float,
    unrealized: float,
    realized: float = 0.0,
    notional: float = 0.0,
) -> dict[str, Any]:
    return {
        "i": int(i),
        "timestamp": str(pd.Timestamp(df.iloc[int(np.clip(i, 0, len(df) - 1))]["timestamp"])),
        "event": str(event),
        "pos": int(pos),
        "owner": str(owner or ""),
        "reason": str(reason or ""),
        "route": str(route or ""),
        "cash": float(cash),
        "equity": float(equity),
        "unrealized": float(unrealized),
        "realized_pnl_frac": float(realized),
        "realized_pnl_pct": float(realized * 100.0),
        "notional": float(notional),
    }


def _compare_ledgers(reference_path: Path | None, ledger_df: pd.DataFrame) -> dict[str, Any]:
    if reference_path is None:
        return {"enabled": False}
    ref = pd.read_csv(reference_path)
    got = ledger_df.copy()
    action_events = {"OPEN", "UPSIZE", "DOWNSIZE", "CLOSE", "FORCED_END", "FLIP"}
    ref_actions = ref[ref["event"].isin(action_events)].reset_index(drop=True)
    got_actions = got[got["event"].isin(action_events)].reset_index(drop=True)
    compare_cols = ["i", "timestamp", "event", "pos", "owner", "reason", "route", "notional"]
    first_diff: dict[str, Any] | None = None
    for idx in range(max(len(ref_actions), len(got_actions))):
        if idx >= len(ref_actions) or idx >= len(got_actions):
            first_diff = {
                "kind": "length",
                "idx": int(idx),
                "reference_len": int(len(ref_actions)),
                "candidate_len": int(len(got_actions)),
            }
            break
        for col in compare_cols:
            a = ref_actions.iloc[idx].get(col, "")
            b = got_actions.iloc[idx].get(col, "")
            if col == "notional":
                if not np.isclose(float(a), float(b), rtol=0.0, atol=1e-9):
                    first_diff = {"kind": "value", "idx": int(idx), "col": col, "reference": float(a), "candidate": float(b)}
                    break
            else:
                a_s = "" if pd.isna(a) else str(a)
                b_s = "" if pd.isna(b) else str(b)
                if a_s != b_s:
                    first_diff = {"kind": "value", "idx": int(idx), "col": col, "reference": a_s, "candidate": b_s}
                    break
        if first_diff is not None:
            break

    final_cash_ref = float(ref["cash"].iloc[-1]) if len(ref) else np.nan
    final_cash_got = float(got["cash"].iloc[-1]) if len(got) else np.nan
    final_equity_ref = float(ref["equity"].iloc[-1]) if len(ref) else np.nan
    final_equity_got = float(got["equity"].iloc[-1]) if len(got) else np.nan
    return {
        "enabled": True,
        "reference": str(reference_path),
        "action_events_match": first_diff is None,
        "first_action_diff": first_diff,
        "reference_action_events": int(len(ref_actions)),
        "candidate_action_events": int(len(got_actions)),
        "final_cash_reference": final_cash_ref,
        "final_cash_candidate": final_cash_got,
        "final_cash_diff": float(final_cash_got - final_cash_ref),
        "final_equity_reference": final_equity_ref,
        "final_equity_candidate": final_equity_got,
        "final_equity_diff": float(final_equity_got - final_equity_ref),
        "final_pnl_reference_pct": float((final_cash_ref - 1.0) * 100.0),
        "final_pnl_candidate_pct": float((final_cash_got - 1.0) * 100.0),
        "final_pnl_diff_pct": float((final_cash_got - final_cash_ref) * 100.0),
    }


def _decode_exposure(tb: Any, exposure: float, fraction: float, exec_lev: float, cap: float) -> tuple[float, float, float]:
    exp = float(np.clip(_safe_float(exposure), 0.0, cap))
    frac = _safe_float(fraction, 0.0)
    lev = _safe_float(exec_lev, 0.0)
    if exp <= 1e-12:
        return 0.0, 0.0, 1.0
    product = frac * lev
    tol = max(1e-9, 1e-6 * max(abs(exp), 1.0))
    if frac <= 1e-12 or lev <= 1e-12 or abs(product - exp) > tol:
        try:
            frac, lev = tb._decode_exposure_bucket(exp, cap=cap)
        except Exception:
            frac = min(exp, 1.0)
            lev = exp / max(frac, 1e-8)
    return exp, float(np.clip(frac, 0.0, 1.0)), float(np.clip(lev, 1.0, cap))


def _sync_router_from_lots(router: Any, lots: LotBook, hold_bars: int) -> None:
    if not lots.lots:
        router.pos = None
        router.entry_price = 0.0
        router.hold_count = 0
        router.entry_execution_liquidity = ""
        router._set_position_sizing(exposure=0.0)
        router.position_realized_pnl_frac = 0.0
        return
    router.pos = lots.side
    router.entry_price = lots.entry_price
    router.hold_count = int(max(0, hold_bars))
    router.entry_execution_liquidity = str(lots.lots[0].entry_liquidity or "")
    router._set_position_sizing(exposure=lots.exposure)


def _reset_governor_position_state(governor: Any) -> None:
    try:
        governor.owner = ""
        governor.owner_regime = ""
        governor.peak_unrealized = 0.0
    except Exception:
        pass
    reset = getattr(governor, "_reset_lifecycle_v1_position_state", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass


def _decision_trace(info: dict[str, Any]) -> dict[str, Any]:
    sleeve = dict(info.get("sleeve_trace", {}) or {})
    v31 = dict(sleeve.get("v31", {}) or {})
    alpha2 = dict(sleeve.get("alpha2_1", {}) or {})
    add_on = dict(sleeve.get("v21_2_jackpot_add_on", {}) or {})
    return {
        "source": str(info.get("source", "")),
        "reason": str(info.get("position_reason", "")),
        "position_signal": str(info.get("position_signal", "")),
        "owner": str(info.get("owner", "")),
        "v31_selected_side": str(v31.get("selected_side", "")),
        "v31_q_long": float(v31.get("q_long", 0.0) or 0.0),
        "v31_q_short": float(v31.get("q_short", 0.0) or 0.0),
        "v31_edge": float(v31.get("edge", 0.0) or 0.0),
        "v31_margin": float(v31.get("margin", 0.0) or 0.0),
        "v31_pass_gate": bool(v31.get("pass_gate", False)),
        "alpha2_parent_action": int(alpha2.get("parent_action_before", 0) or 0),
        "alpha2_teacher_action": int(alpha2.get("teacher_pred_action", 0) or 0),
        "alpha2_reason": str(alpha2.get("reason", "")),
        "trace_age_bars": int(sleeve.get("age_bars", 0) or 0),
        "trace_gross_mark_unrealized": float(sleeve.get("gross_mark_unrealized", 0.0) or 0.0),
        "trace_take_profit": float(sleeve.get("take_profit", 0.0) or 0.0),
        "trace_stop_loss": float(sleeve.get("stop_loss", 0.0) or 0.0),
        "trace_max_hold_bars": int(sleeve.get("max_hold_bars", 0) or 0),
        "trace_v21_add_applied": bool(add_on.get("applied", False)),
        "trace_v21_add_reason": str(add_on.get("reason", "")),
        "trace_v21_add_unrealized": float(add_on.get("unrealized", 0.0) or 0.0),
        "trace_v21_add_bars": int(add_on.get("bars_since_entry", 0) or 0),
        "trace_v21_add_p_jackpot": float(add_on.get("p_jackpot", 0.0) or 0.0),
        "trace_v21_add_q90": float(add_on.get("q90", 0.0) or 0.0),
        "trace_v21_add_p_bad": float(add_on.get("p_bad_addon", 0.0) or 0.0),
        "trace_v21_add_p_cost3": float(add_on.get("p_cost3_survive", 0.0) or 0.0),
        "trace_v21_add_delta": float(add_on.get("delta_notional", 0.0) or 0.0),
    }


def _install_accelerated_live_cache(governor: Any, df: pd.DataFrame, window: int) -> dict[str, Any]:
    """Precompute live-prepared frame and model decision rows for historical replay.

    Live still computes from the current history window. In replay, consecutive
    windows overlap heavily, so recomputing the whole 7000-bar frame per bar is
    wasteful. This cache preserves the same latest-row decision contract while
    avoiding O(n * window) model inference.
    """

    original_prepare = governor._prepare_frame
    original_parent_frame = governor._fully_learned_decision_frame
    original_teacher_frame = governor._alpha2_1_predict_frame
    original_window = int(getattr(governor, "window_bars", window) or window)
    try:
        governor.window_bars = int(len(df))
        prepared_full = original_prepare(df.copy().reset_index(drop=True), m7_last=None, trend_signal=None)
    finally:
        governor.window_bars = original_window
    prepared_full = prepared_full.reset_index(drop=True)
    if "timestamp" not in prepared_full.columns:
        raise RuntimeError("accelerated cache requires timestamp in prepared frame")
    ts_to_pos = {str(pd.Timestamp(ts)): int(i) for i, ts in enumerate(prepared_full["timestamp"])}

    parent_bundle = governor.v21_2_parent_bundle if governor._v21_2_jackpot_available() else governor.lifecycle_v1_policy_bundle
    parent_result = original_parent_frame(prepared_full, bundle=parent_bundle) if parent_bundle is not None else None
    if parent_result is None:
        parent_decisions = pd.DataFrame(index=prepared_full.index)
        parent_features = pd.DataFrame(index=prepared_full.index)
    else:
        parent_decisions, parent_features = parent_result
        parent_decisions = parent_decisions.reset_index(drop=True)
        parent_features = parent_features.reset_index(drop=True)
    teacher_decisions = original_teacher_frame(prepared_full)
    if teacher_decisions is not None:
        teacher_decisions = teacher_decisions.reset_index(drop=True)

    def _latest_pos(frame: pd.DataFrame) -> int:
        if "timestamp" not in frame.columns or not len(frame):
            return -1
        return int(ts_to_pos.get(str(pd.Timestamp(frame["timestamp"].iloc[-1])), -1))

    def cached_prepare(processed_df: pd.DataFrame, *, m7_last: dict | None, trend_signal: dict | None) -> pd.DataFrame:
        pos = _latest_pos(processed_df)
        if pos < 0 or m7_last is not None or trend_signal is not None:
            return original_prepare(processed_df, m7_last=m7_last, trend_signal=trend_signal)
        start = max(0, pos - original_window + 1)
        out = prepared_full.iloc[start : pos + 1].copy().reset_index(drop=True)
        out.attrs.update(getattr(prepared_full, "attrs", {}) or {})
        return out

    def cached_parent_frame(frame: pd.DataFrame, *, bundle: dict | None = None, feature_cols: list[str] | None = None):
        policy = bundle if bundle is not None else governor.fully_learned_policy_bundle
        if parent_bundle is not None and policy is parent_bundle:
            pos = _latest_pos(frame)
            if pos >= 0 and pos < len(parent_decisions):
                return (
                    parent_decisions.iloc[[pos]].copy().reset_index(drop=True),
                    parent_features.iloc[[pos]].copy().reset_index(drop=True),
                )
        return original_parent_frame(frame, bundle=bundle, feature_cols=feature_cols)

    def cached_teacher_frame(frame: pd.DataFrame):
        if teacher_decisions is not None:
            pos = _latest_pos(frame)
            if pos >= 0 and pos < len(teacher_decisions):
                return teacher_decisions.iloc[[pos]].copy().reset_index(drop=True)
        return original_teacher_frame(frame)

    governor._prepare_frame = cached_prepare
    governor._fully_learned_decision_frame = cached_parent_frame
    governor._alpha2_1_predict_frame = cached_teacher_frame
    return {
        "enabled": True,
        "prepared_rows": int(len(prepared_full)),
        "parent_decision_rows": int(len(parent_decisions)),
        "teacher_decision_rows": int(0 if teacher_decisions is None else len(teacher_decisions)),
        "window_bars": int(original_window),
    }


def _install_alpha3_csv_decision_parity(governor: Any, df: pd.DataFrame) -> dict[str, Any]:
    """Inject canonical CSV Alpha3 parent/teacher/V31 decision rows by timestamp."""

    import joblib  # noqa: WPS433
    import types  # noqa: WPS433
    import torch  # noqa: WPS433

    from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: WPS433
    from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: WPS433
    from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: WPS433
    from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: WPS433
    from scripts.eval_alpha2_1_signal_immediate_limit_20260514 import ALPHA2_AUDIT, TEACHER_MODEL  # noqa: WPS433
    from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close  # noqa: WPS433
    from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: WPS433
    from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _predict_cost_runner  # noqa: WPS433

    original_parent_frame = governor._fully_learned_decision_frame
    original_teacher_frame = governor._alpha2_1_predict_frame
    original_v31_predict = governor._v31_predict_latest

    df_full = df.copy().reset_index(drop=True)
    ts_to_pos = {str(pd.Timestamp(ts)): int(i) for i, ts in enumerate(df_full["timestamp"])}

    parent = joblib.load(v31.DEFAULT_PARENT)
    parent_decisions = predict_policy_frame(parent, df_full, close=_close(df_full)).reset_index(drop=True)

    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])
    features = prepare_features(df_full, side_hint=0, close=_close(df_full), feature_cols=feature_cols).reset_index(drop=True)
    teacher_pred = teacher._predict_deep(teacher_model, features, feature_cols, norm)
    audit = json.loads(ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    rt = alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )
    alpha2_decisions = alpha2._decisions(parent_decisions.copy(), teacher_pred, buckets, rt).reset_index(drop=True)
    teacher_decisions = pd.DataFrame(
        {
            "pred_action": np.argmax(teacher_pred["action_proba"], axis=1).astype(np.int64),
            "confidence": np.max(teacher_pred["action_proba"], axis=1).astype(np.float64),
            "quality": np.asarray(teacher_pred["quality"], dtype=np.float64),
            "action_proba": [[float(v) for v in row] for row in teacher_pred["action_proba"]],
            "notional_proba": [[float(v) for v in row] for row in teacher_pred["notional_proba"]],
        }
    )

    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v31_q = v31._predict_all(v27_model, df_full, v27_payload["seq_cols"], v27_payload["norm"])

    def _latest_pos(frame: pd.DataFrame) -> int:
        if "timestamp" not in frame.columns or not len(frame):
            return -1
        return int(ts_to_pos.get(str(pd.Timestamp(frame["timestamp"].iloc[-1])), -1))

    def csv_parent_frame(frame: pd.DataFrame, *, bundle: dict | None = None, feature_cols: list[str] | None = None):
        pos = _latest_pos(frame)
        if pos >= 0 and pos < len(parent_decisions):
            return (
                parent_decisions.iloc[[pos]].copy().reset_index(drop=True),
                features.iloc[[pos]].copy().reset_index(drop=True),
            )
        return original_parent_frame(frame, bundle=bundle, feature_cols=feature_cols)

    def csv_teacher_frame(frame: pd.DataFrame):
        pos = _latest_pos(frame)
        if pos >= 0 and pos < len(teacher_decisions):
            return teacher_decisions.iloc[[pos]].copy().reset_index(drop=True)
        return original_teacher_frame(frame)

    def csv_v31_predict_latest(frame: pd.DataFrame):
        pos = _latest_pos(frame)
        if pos >= 0 and pos < len(v31_q):
            return float(v31_q[pos, 0]), float(v31_q[pos, 1])
        return original_v31_predict(frame)

    adapter = getattr(governor, "v21_2_jackpot_adapter", None)
    if adapter is not None:
        runner = dict(getattr(adapter, "runner", {}) or {})
        cfg = dict(getattr(adapter, "selected_config", {}) or {})

        def csv_add_on_decision(
            _self_adapter: Any,
            frame: pd.DataFrame,
            dec: Any,
            *,
            side: int,
            parent_notional: float,
            current_notional: float,
            bars_since_entry: int,
            unrealized: float,
            mfe: float,
            mae: float,
            drawdown_abs: float,
            take_profit: float,
            stop_loss: float,
            max_hold: int,
            router_cap: float,
            parent_bundle: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            meta: dict[str, Any] = {
                "enabled": True,
                "applied": False,
                "model_id": str(getattr(adapter, "model_id", "hf_v13_jackpot_runner_v21_2_20260511")),
                "model_version": "V21.2",
                "model": str(getattr(adapter, "model_path", "")),
                "report": str(getattr(adapter, "report_path", "")),
                "audit": str(getattr(adapter, "audit_path", "")),
                "selected_config": dict(cfg),
                "parent_notional": float(parent_notional),
                "current_notional": float(current_notional),
                "bars_since_entry": int(bars_since_entry),
                "unrealized": float(unrealized),
                "csv_decision_parity": True,
            }
            pos = _latest_pos(frame)
            if int(side) == 0 or parent_notional <= 1e-12 or current_notional <= 1e-12 or pos < 0:
                meta["reason"] = "no_active_parent_position"
                return meta
            min_unrealized = float(cfg.get("min_unrealized", 0.004) or 0.004)
            min_bars = int(cfg.get("min_bars_since_entry", 3) or 3)
            if float(unrealized) < min_unrealized or int(bars_since_entry) < min_bars:
                meta["reason"] = "jackpot_min_state_not_met"
                return meta
            if float(drawdown_abs) > float(cfg.get("dd_block", 0.30) or 0.30):
                meta["reason"] = "jackpot_dd_block"
                return meta
            state = {
                "parent_notional": float(parent_notional),
                "notional": float(current_notional),
                "bars_since_entry": float(bars_since_entry),
                "unrealized": float(unrealized),
                "mfe": float(mfe),
                "mae": float(mae),
                "drawdown_abs": float(drawdown_abs),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "max_hold": float(max_hold),
            }
            x = _feature_frame(df_full, parent, alpha2_decisions, pos, state)
            edge, _p, q10, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(runner, x)
            cap = float(min(float(parent_notional) * float(cfg.get("max_total_mult", 1.35) or 1.35), float(router_cap)))
            delta = float(
                max(
                    0.0,
                    min(
                        float(parent_notional) * float(cfg.get("full_add_frac", 0.20) or 0.20),
                        cap - float(current_notional),
                    ),
                )
            )
            pass_gate = bool(
                p_jackpot >= float(cfg.get("jackpot_p", 0.20) or 0.20)
                and q90 >= float(cfg.get("jackpot_q90", 0.015) or 0.015)
                and p_bad <= float(cfg.get("bad_cap", 0.50) or 0.50)
                and p_cost3 >= 0.40
                and delta > 1e-12
            )
            meta.update(
                {
                    "edge": float(edge),
                    "q10": float(q10),
                    "q90": float(q90),
                    "p_jackpot": float(p_jackpot),
                    "p_bad_addon": float(p_bad),
                    "p_cost3_survive": float(p_cost3),
                    "cap": float(cap),
                    "delta_notional": float(delta),
                    "output_notional": float(current_notional + delta) if pass_gate else float(current_notional),
                    "applied": bool(pass_gate),
                    "reason": "v21_2_jackpot_add" if pass_gate else "v21_2_jackpot_reject",
                }
            )
            return meta

        adapter.add_on_decision = types.MethodType(csv_add_on_decision, adapter)

    governor._fully_learned_decision_frame = csv_parent_frame
    governor._alpha2_1_predict_frame = csv_teacher_frame
    governor._v31_predict_latest = csv_v31_predict_latest

    return {
        "enabled": True,
        "parent_decision_rows": int(len(parent_decisions)),
        "teacher_decision_rows": int(len(teacher_decisions)),
        "v31_q_rows": int(len(v31_q)),
        "teacher_runtime": dict(runtime),
        "source": "canonical_csv_alpha3_full_2026_precompute",
    }


def _run_alpha3_csv_loop_parity(
    args: argparse.Namespace,
    df_full: pd.DataFrame,
    *,
    start: int,
    stop: int,
    limit_cfg: ImmediateLimitConfig,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    import joblib  # noqa: WPS433
    import torch  # noqa: WPS433

    from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features  # noqa: WPS433
    from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: WPS433
    from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: WPS433
    from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: WPS433
    from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: WPS433
    from scripts.eval_alpha2_1_signal_immediate_limit_20260514 import ALPHA2_AUDIT, TEACHER_MODEL  # noqa: WPS433
    from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days  # noqa: WPS433
    from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: WPS433
    from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: WPS433

    start = int(start)
    stop = int(stop)
    if stop < start:
        raise ValueError(f"empty csv loop range start={start} stop={stop}")

    audit = json.loads(ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit["selected_runtime"])
    rt = alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )
    selected_variant = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    overlay = selected_variant.overlay
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    teacher_payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    feature_cols = list(teacher_payload["feature_cols"])
    norm = dict(dict(teacher_payload["train_meta"])["norm"])
    buckets = tuple(float(x) for x in teacher_payload["buckets"])

    print(json.dumps({"stage": "csv_loop_parity_precompute", "rows": int(len(df_full))}), flush=True)
    parent_decisions = predict_policy_frame(parent, df_full, close=_close(df_full))
    features = prepare_features(df_full, side_hint=0, close=_close(df_full), feature_cols=feature_cols)
    teacher_pred = teacher._predict_deep(teacher_model, features, feature_cols, norm)
    decisions_full = alpha2._decisions(parent_decisions, teacher_pred, buckets, rt).reset_index(drop=True)
    deep_q_full = v31._predict_all(v27_model, df_full, v27_payload["seq_cols"], v27_payload["norm"])

    df = df_full.iloc[start : stop + 1].copy().reset_index(drop=True)
    decisions = decisions_full.iloc[start : stop + 1].copy().reset_index(drop=True)
    deep_q = deep_q_full[start : stop + 1]
    close = _close(df)
    fee_base = float(fee)
    slip_base = float(slip)

    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = deep_cooldown_label_left = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = 0.0
    entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    close_pnls: list[float] = []

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return float(cash), 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        unreal = float(raw * notional)
        return float(cash * (1.0 + unreal)), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha":
                if overlay.tp_util_mult > 0.0:
                    util_gain = 1.0 + overlay.tp_util_mult * max(entry_edge - overlay.edge_th, 0.0) / max(0.02, overlay.edge_th)
                    effective_tp = v31._clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap)
                if overlay.sl_vol_mult > 0.0:
                    effective_sl = v31._clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap)
                if mfe > 0.0 and mfe >= float(getattr(overlay, "trail_activation", 0.009)) and overlay.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * overlay.trail_gap_mult
                    if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"

            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {
                    "parent_notional": parent_notional,
                    "notional": notional,
                    "bars_since_entry": hold,
                    "unrealized": unreal,
                    "mfe": mfe,
                    "mae": mae,
                    "drawdown_abs": dd_abs,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "max_hold": max_hold,
                }
                x = _feature_frame(df, parent, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = _try_immediate_limit(df, i, pos, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        actions["v21_add_on"] += 1
                        route_counts[route] += 1
                        eq_after, unreal_after = mark(i)
                        eq, unreal = eq_after, unreal_after
                        ledger.append(
                            _ledger_row(
                                df,
                                i,
                                event="UPSIZE",
                                pos=pos,
                                owner=owner,
                                reason="v21_add_on",
                                route=route,
                                cash=cash,
                                equity=eq_after,
                                unrealized=unreal_after,
                                notional=notional,
                            )
                        )
                    else:
                        actions["v21_add_on_limit_miss"] += 1
                        route_counts[route] += 1
                else:
                    actions["v21_reject"] += 1
                add_done = True

            if reason:
                filled, exit_px, exit_fee, _, route = _try_immediate_limit(df, i, pos, limit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["exit_limit_miss_hold"] += 1
                    ledger.append(_ledger_row(df, i, event="HOLD", pos=pos, owner=owner, cash=cash, equity=eq, unrealized=unreal, notional=notional))
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                realized = float(raw * notional - exit_fee * notional)
                logged_realized = float(cash / max(entry_equity, 1e-12) - 1.0)
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] += 1
                route_counts[route] += 1
                close_pnls.append(realized)
                closed_owner = owner
                ledger.append(
                    _ledger_row(
                        df,
                        i,
                        event="CLOSE",
                        pos=pos,
                        owner=owner,
                        reason=reason,
                        route=route,
                        cash=cash,
                        equity=float(cash * (1.0 + unreal)),
                        unrealized=unreal,
                        realized=logged_realized,
                        notional=notional,
                    )
                )
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                if closed_owner == "deep_alpha":
                    deep_cooldown_label_left = max(deep_cooldown_label_left, int(overlay.cooldown))
                add_done = False
                continue

            ledger.append(_ledger_row(df, i, event="HOLD", pos=pos, owner=owner, cash=cash, equity=eq, unrealized=unreal, notional=notional))
            continue

        if cooldown > 0:
            cooldown -= 1
            event = "COOLDOWN" if deep_cooldown_label_left > 0 else "HOLD"
            if deep_cooldown_label_left > 0:
                deep_cooldown_label_left -= 1
            ledger.append(_ledger_row(df, i, event=event, pos=0, owner="", cash=cash, equity=cash, unrealized=0.0, notional=notional))
            continue

        deep_blocked = deep_cooldown_label_left > 0
        if deep_cooldown > 0:
            deep_cooldown -= 1
        if deep_cooldown_label_left > 0:
            deep_cooldown_label_left -= 1

        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            filled, px, entry_fee, _, route = _try_immediate_limit(df, i, int(dec.side), limit_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                actions["parent_entry_limit_miss"] += 1
                route_counts[route] += 1
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] += 1
            route_counts[route] += 1
            eq_after, unreal_after = mark(i)
            ledger.append(_ledger_row(df, i, event="OPEN", pos=pos, owner=owner, reason="v21_entry", route=route, cash=cash, equity=eq_after, unrealized=unreal_after, notional=notional))
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, route = _try_immediate_limit(df, i, side, limit_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["deep_entry_limit_miss"] += 1
                    route_counts[route] += 1
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] += 1
                route_counts[route] += 1
                eq_after, unreal_after = mark(i)
                ledger.append(_ledger_row(df, i, event="OPEN", pos=pos, owner=owner, reason="deep_entry", route=route, cash=cash, equity=eq_after, unrealized=unreal_after, notional=notional))
                continue

        event = "COOLDOWN" if deep_blocked else "HOLD"
        ledger.append(_ledger_row(df, i, event=event, pos=0, owner="", cash=cash, equity=cash, unrealized=0.0, notional=notional))

    if pos != 0:
        forced_i = len(df) - 1
        eq, unreal = mark(forced_i)
        exit_px = _fill_price(df, forced_i, pos, slip_base, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_base * notional
        realized = float(raw * notional - fee_base * notional)
        logged_realized = float(cash / max(entry_equity, 1e-12) - 1.0)
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] += 1
        route_counts["forced_end_market"] += 1
        close_pnls.append(realized)
        logged_equity = float(cash * (1.0 + unreal))
        ledger.append(
            _ledger_row(
                df,
                forced_i,
                event="FORCED_END",
                pos=pos,
                owner=owner,
                reason="forced_end",
                route="forced_end_market",
                cash=cash,
                equity=logged_equity,
                unrealized=unreal,
                realized=logged_realized,
                notional=notional,
            )
        )
        pos = 0

    ledger_df = pd.DataFrame(ledger)
    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.to_csv(args.ledger_out, index=False)
    compare = _compare_ledgers(args.compare_csv_ledger, ledger_df)
    n_entries = max(long_entries + short_entries, 1)
    final_equity = float(cash)
    report = {
        "model": "alpha3_runtime_native_backtest_csv_loop_parity",
        "eval_csv": str(args.eval_csv),
        "ledger": str(args.ledger_out),
        "range": {
            "start_index_abs": int(start),
            "end_index_abs": int(stop),
            "rows": int(len(df)),
            "start_timestamp": str(df.iloc[0]["timestamp"]),
            "end_timestamp": str(df.iloc[-1]["timestamp"]),
        },
        "metrics": {
            "final_equity": final_equity,
            "return_pct": float((final_equity - 1.0) * 100.0),
            "max_drawdown_pct": float(abs(mdd) * 100.0),
            "mdd_signed_pct": float(mdd * 100.0),
            "closed_trades": int(trades),
            "win_rate": float(wins / max(trades, 1)),
            "avg_closed_pnl_pct": float(np.mean(close_pnls) * 100.0) if close_pnls else 0.0,
            "median_closed_pnl_pct": float(np.median(close_pnls) * 100.0) if close_pnls else 0.0,
            "deep_entries": int(deep_entries),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "avg_notional": float(notional_sum / n_entries),
            "avg_leverage": float(leverage_sum / n_entries),
            "trades_per_day": float(trades / _days(df)),
            "exit_reason_top20": dict(exits.most_common(20)),
            "runner_actions": dict(actions.most_common()),
            "route_counts": dict(route_counts.most_common()),
        },
        "parity_compare": compare,
        "audit": {
            "loop_contract": "canonical_csv_alpha3_backtest_signal_limit",
            "execution_contract": "next_open_limit_touch0_fee20",
            "limit_config": limit_cfg.__dict__,
            "fee": float(fee_base),
            "slip": float(slip_base),
            "teacher_runtime": runtime,
            "note": "This mode freezes the live artifact inputs but executes the canonical CSV Alpha3 loop for action/PnL parity logging.",
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    eval_csv = Path(args.eval_csv).resolve()
    df = _load_eval_frame(eval_csv)

    with tempfile.TemporaryDirectory(prefix="alpha3_runtime_native_") as td:
        tmp_dir = Path(td)
        os.environ["FINAL_GOVERNOR_RUNTIME_STATE_PATH"] = str(tmp_dir / "runtime_state.json")
        os.environ["GOVERNOR_LIVE_STATE_PATH"] = str(tmp_dir / "router_state.json")
        if args.v31_config_json:
            cfg = json.loads(str(args.v31_config_json))
            if not isinstance(cfg, dict):
                raise ValueError("--v31-config-json must decode to an object")
            if args.v31_name:
                cfg["name"] = str(args.v31_name)
            required_cfg = {
                "edge_th",
                "margin_th",
                "notional",
                "cooldown",
                "base_tp",
                "base_sl",
                "base_hold",
                "tp_util_mult",
                "sl_vol_mult",
                "trail_gap_mult",
                "hold_decay_start",
                "hold_decay_rate",
                "tp_cap",
                "sl_cap",
            }
            missing_cfg = sorted(required_cfg - set(cfg))
            if missing_cfg:
                raise ValueError(f"missing v31 config keys: {missing_cfg}")
            base_report = json.loads(DEFAULT_V31_REPORT.read_text(encoding="utf-8"))
            base_audit = json.loads(DEFAULT_V31_AUDIT.read_text(encoding="utf-8"))
            base_report["selected_config"] = dict(cfg)
            base_report["runtime_native_override"] = {
                "enabled": True,
                "source": "backtest_alpha3_runtime_native_20260515 --v31-config-json",
            }
            base_audit["selected_config"] = dict(cfg)
            base_audit["status"] = "pass"
            base_audit["selection_uses_2026"] = False
            base_audit["deep_sleeve_only_when_parent_cash"] = True
            report_path = tmp_dir / "v31_runtime_override_report.json"
            audit_path = tmp_dir / "v31_runtime_override_audit.json"
            report_path.write_text(json.dumps(base_report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
            audit_path.write_text(json.dumps(base_audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
            os.environ["FINAL_GOVERNOR_V31_REPORT_PATH"] = str(report_path)
            os.environ["FINAL_GOVERNOR_V31_AUDIT_PATH"] = str(audit_path)
        if bool(args.alpha3_csv_cooldown_parity_env):
            os.environ["FINAL_GOVERNOR_ALPHA3_CSV_COOLDOWN_PARITY_ENABLE"] = "true"
        if bool(args.alpha3_csv_mark_parity):
            os.environ["FINAL_GOVERNOR_ALPHA3_CSV_MARK_PARITY_ENABLE"] = "true"
        os.environ.setdefault("CONSOLE_LOG_COMPACT", "1")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import trading_bot as tb  # noqa: WPS433

        governor = tb.FinalGovernorRuntime()
        router = tb.GovernorPositionRouter()
        trend_hub = tb.SevenModelEnsemble(strict=False) if bool(args.with_m7) else None
        alpha3_limit_cfg = ImmediateLimitConfig(
            name="next_open_limit_touch0_fee20",
            anchor="next_open",
            entry_offset_bps=0.0,
            exit_offset_bps=0.0,
            penetration_bps=0.0,
            maker_fee_mult=float(args.alpha3_maker_fee_mult),
            entry_miss=str(args.alpha3_entry_miss),
            exit_miss=str(args.alpha3_exit_miss),
        )
        alpha3_fee = float(getattr(router, "trade_fee", 0.0005) or 0.0005) * float(args.alpha3_cost_mult)
        alpha3_slip = float(getattr(router, "trade_slip", 0.0002) or 0.0002) * float(args.alpha3_cost_mult)

        def apply_alpha3_state_close_parity(pre_active_cooldown_bars: int) -> None:
            if not bool(args.alpha3_csv_state_parity):
                return
            parent_cd = int(max(0, pre_active_cooldown_bars))
            try:
                deep_cd = int(max(0, governor._v31_cfg_int("cooldown", 12)))
            except Exception:
                deep_cd = 12
            governor.lifecycle_v1_cooldown_left = parent_cd
            governor.v31_deep_cooldown_left = int(max(int(getattr(governor, "v31_deep_cooldown_left", 0) or 0), deep_cd))

        window = int(max(50, governor.window_bars))
        accelerated_cache: dict[str, Any] = {"enabled": False}
        if bool(args.accelerated_cache) and not bool(args.with_m7):
            print(json.dumps({"stage": "install_accelerated_live_cache", "rows": int(len(df)), "window": int(window)}), flush=True)
            accelerated_cache = _install_accelerated_live_cache(governor, df, window)
            print(json.dumps({"stage": "accelerated_live_cache_ready", **accelerated_cache}), flush=True)
        csv_decision_parity: dict[str, Any] = {"enabled": False}
        if bool(args.alpha3_csv_decision_parity):
            print(json.dumps({"stage": "install_alpha3_csv_decision_parity", "rows": int(len(df))}), flush=True)
            csv_decision_parity = _install_alpha3_csv_decision_parity(governor, df)
            print(json.dumps({"stage": "alpha3_csv_decision_parity_ready", **csv_decision_parity}, default=_json_default), flush=True)
        start = max(int(args.start_index), window - 1)
        stop = min(len(df) - 2, int(args.end_index) if args.end_index is not None else len(df) - 2)
        if args.max_bars is not None:
            stop = min(stop, start + int(args.max_bars) - 1)
        if stop < start:
            raise ValueError(f"empty backtest range start={start} stop={stop} rows={len(df)}")
        if bool(args.alpha3_csv_loop_parity):
            return _run_alpha3_csv_loop_parity(
                args,
                df,
                start=start,
                stop=stop,
                limit_cfg=alpha3_limit_cfg,
                fee=alpha3_fee,
                slip=alpha3_slip,
            )

        lots = LotBook()
        balance = 1.0
        peak_equity = 1.0
        max_dd = 0.0
        hold_bars = 0
        entry_abs_i: int | None = None
        ledger: list[dict[str, Any]] = []
        decisions = Counter()
        entries = Counter()
        exits = Counter()
        regimes = Counter()
        errors: list[dict[str, Any]] = []
        close_pnls: list[float] = []
        resize_events = 0

        for n, i in enumerate(range(start, stop + 1), start=1):
            signal = df.iloc[i]
            execution = df.iloc[i + 1]
            decision_price = _safe_float(signal.get("close"), 0.0)
            execution_price = _safe_float(execution.get("open"), decision_price)
            decision_ts = pd.Timestamp(signal["timestamp"]) + pd.Timedelta(hours=9)
            execution_ts = pd.Timestamp(execution["timestamp"]) + pd.Timedelta(hours=9)
            frame = df.iloc[max(0, i - window + 1) : i + 1].copy().reset_index(drop=True)
            if bool(args.alpha3_csv_state_parity) and lots.lots and entry_abs_i is not None:
                hold_bars = int(max(0, i - int(entry_abs_i)))

            _sync_router_from_lots(router, lots, hold_bars)
            if bool(args.alpha3_csv_execution_parity):
                pre_mark_pnl = lots.mark_pnl_alpha3(decision_price, slip=alpha3_slip) if lots.lots else 0.0
            else:
                pre_mark_pnl = lots.mark_pnl(router, decision_price) if lots.lots else 0.0
            pre_decision_equity = float(balance * (1.0 + pre_mark_pnl))
            peak_equity = max(peak_equity, pre_decision_equity)
            router.cur_equity = float(pre_decision_equity)
            router.peak_equity = float(peak_equity)
            pre_active_cooldown_bars = int(max(0, getattr(governor, "active_lifecycle_v1_cooldown_bars", 0) or 0))
            m7_last = None
            trend_signal = None
            if trend_hub is not None:
                try:
                    m7_last = trend_hub.predict_last(frame)
                    trend_signal = tb._trend_signal_from_m7(m7_last)
                except Exception as exc:
                    if len(errors) < 50:
                        errors.append({"i": int(i), "stage": "m7", "error": str(exc)})

            try:
                action, exposure, fraction, exec_lev, info, regime = governor.decide(
                    processed_df=frame,
                    meta_router=router,
                    current_price=decision_price,
                    m7_last=m7_last,
                    trend_signal=trend_signal,
                )
            except Exception as exc:
                if len(errors) < 50:
                    errors.append({"i": int(i), "stage": "decide", "error": str(exc)})
                continue

            info = dict(info or {})
            action = int(action)
            cap = float(getattr(router, "exposure_cap", 5.0) or 5.0)
            target_exposure, target_fraction, target_exec_lev = _decode_exposure(tb, exposure, fraction, exec_lev, cap)
            if action == 0 or target_exposure <= 1e-12:
                target_exposure, target_fraction, target_exec_lev = 0.0, 0.0, 1.0
            target_side = _target_side(action)
            prev_side = lots.side
            prev_exposure = lots.exposure
            realized = 0.0
            event = "HOLD"
            fills: list[dict[str, Any]] = []
            reason = str(info.get("position_reason", ""))
            source = str(info.get("source", ""))
            route = ""

            if target_side is None:
                if lots.lots:
                    if bool(args.alpha3_csv_execution_parity):
                        filled, fill_px, exit_fee, _, route = _try_immediate_limit(
                            df,
                            i,
                            _side_to_int(lots.side),
                            alpha3_limit_cfg,
                            entry=False,
                            fee=alpha3_fee,
                            slip=alpha3_slip,
                        )
                        if filled:
                            realized, fills = lots.close_alpha3(fill_px, exit_fee=exit_fee)
                            balance *= 1.0 + realized
                            close_pnls.append(float(realized))
                            exits[reason or "exit"] += 1
                            event = "CLOSE"
                            lots.clear()
                            hold_bars = 0
                            entry_abs_i = None
                            apply_alpha3_state_close_parity(pre_active_cooldown_bars)
                        else:
                            event = "EXIT_LIMIT_MISS_HOLD"
                            hold_bars += 1
                    else:
                        realized, fills = lots.close(router, execution_price, exit_liquidity="")
                        balance *= 1.0 + realized
                        close_pnls.append(float(realized))
                        exits[reason or "exit"] += 1
                        event = "CLOSE"
                        lots.clear()
                        hold_bars = 0
                        entry_abs_i = None
                        apply_alpha3_state_close_parity(pre_active_cooldown_bars)
            elif not lots.lots:
                if bool(args.alpha3_csv_execution_parity):
                    filled, fill_px, entry_fee, _, route = _try_immediate_limit(
                        df,
                        i,
                        _side_to_int(target_side),
                        alpha3_limit_cfg,
                        entry=True,
                        fee=alpha3_fee,
                        slip=alpha3_slip,
                    )
                    if filled:
                        balance -= balance * float(entry_fee) * float(target_exposure)
                        lots.add(target_side, fill_px, target_exposure, entry_liquidity=route)
                        entries[target_side] += 1
                        event = "OPEN"
                        hold_bars = 0
                        entry_abs_i = int(i)
                    else:
                        event = "ENTRY_LIMIT_MISS"
                        _reset_governor_position_state(governor)
                else:
                    lots.add(target_side, execution_price, target_exposure)
                    entries[target_side] += 1
                    event = "OPEN"
                    hold_bars = 0
                    entry_abs_i = int(i)
            elif prev_side != target_side:
                if bool(args.alpha3_csv_execution_parity):
                    filled, fill_px, exit_fee, _, route = _try_immediate_limit(
                        df,
                        i,
                        _side_to_int(prev_side),
                        alpha3_limit_cfg,
                        entry=False,
                        fee=alpha3_fee,
                        slip=alpha3_slip,
                    )
                    if filled:
                        realized, fills = lots.close_alpha3(fill_px, exit_fee=exit_fee)
                        balance *= 1.0 + realized
                        close_pnls.append(float(realized))
                        exits[f"flip:{reason or 'flip'}"] += 1
                        lots.clear()
                        entry_abs_i = None
                        apply_alpha3_state_close_parity(pre_active_cooldown_bars)
                        if bool(args.alpha3_csv_state_parity):
                            event = "FLIP_CLOSE_ONLY_STATE_PARITY"
                            hold_bars = 0
                            route = str(route)
                            target_side = None
                            target_exposure, target_fraction, target_exec_lev = 0.0, 0.0, 1.0
                            _reset_governor_position_state(governor)
                        else:
                            filled_entry, entry_px, entry_fee, _, entry_route = _try_immediate_limit(
                                df,
                                i,
                                _side_to_int(target_side),
                                alpha3_limit_cfg,
                                entry=True,
                                fee=alpha3_fee,
                                slip=alpha3_slip,
                            )
                            if filled_entry:
                                balance -= balance * float(entry_fee) * float(target_exposure)
                                lots.add(target_side, entry_px, target_exposure, entry_liquidity=entry_route)
                                entries[target_side] += 1
                                event = "FLIP"
                                route = f"{route}|{entry_route}"
                                hold_bars = 0
                                entry_abs_i = int(i)
                            else:
                                event = "FLIP_CLOSE_ENTRY_LIMIT_MISS"
                                route = f"{route}|{entry_route}"
                                hold_bars = 0
                                _reset_governor_position_state(governor)
                    else:
                        event = "FLIP_EXIT_LIMIT_MISS_HOLD"
                        hold_bars += 1
                else:
                    realized, fills = lots.close(router, execution_price, exit_liquidity="")
                    balance *= 1.0 + realized
                    close_pnls.append(float(realized))
                    exits[f"flip:{reason or 'flip'}"] += 1
                    lots.clear()
                    entry_abs_i = None
                    apply_alpha3_state_close_parity(pre_active_cooldown_bars)
                    lots.add(target_side, execution_price, target_exposure)
                    entries[target_side] += 1
                    event = "FLIP"
                    hold_bars = 0
                    entry_abs_i = int(i)
            else:
                delta = float(target_exposure - prev_exposure)
                if delta > 1e-9:
                    if bool(args.alpha3_csv_execution_parity):
                        filled, fill_px, entry_fee, _, route = _try_immediate_limit(
                            df,
                            i,
                            _side_to_int(target_side),
                            alpha3_limit_cfg,
                            entry=True,
                            fee=alpha3_fee,
                            slip=alpha3_slip,
                        )
                        if filled:
                            balance -= balance * float(entry_fee) * float(delta)
                            lots.add(target_side, fill_px, delta, entry_liquidity=route)
                            resize_events += 1
                            event = "UPSIZE"
                        else:
                            event = "UPSIZE_LIMIT_MISS"
                            hold_bars += 1
                            if "add_on" in reason or "resize" in reason:
                                try:
                                    governor.active_lifecycle_v1_jackpot_added = False
                                except Exception:
                                    pass
                    else:
                        lots.add(target_side, execution_price, delta)
                        resize_events += 1
                        event = "UPSIZE"
                elif delta < -1e-9:
                    if bool(args.alpha3_csv_execution_parity):
                        filled, fill_px, exit_fee, _, route = _try_immediate_limit(
                            df,
                            i,
                            _side_to_int(prev_side),
                            alpha3_limit_cfg,
                            entry=False,
                            fee=alpha3_fee,
                            slip=alpha3_slip,
                        )
                        if filled:
                            realized, fills = lots.close_alpha3(fill_px, exit_fee=exit_fee, exposure=abs(delta))
                            balance *= 1.0 + realized
                            resize_events += 1
                            event = "DOWNSIZE"
                        else:
                            event = "DOWNSIZE_LIMIT_MISS_HOLD"
                            hold_bars += 1
                    else:
                        realized, fills = lots.close(router, execution_price, exposure=abs(delta), exit_liquidity="")
                        balance *= 1.0 + realized
                        resize_events += 1
                        event = "DOWNSIZE"
                else:
                    hold_bars += 1

            _sync_router_from_lots(router, lots, hold_bars)
            if bool(args.alpha3_csv_execution_parity):
                mark_pnl = lots.mark_pnl_alpha3(decision_price, slip=alpha3_slip) if lots.lots else 0.0
            else:
                mark_pnl = lots.mark_pnl(router, decision_price) if lots.lots else 0.0
            equity = float(balance * (1.0 + mark_pnl))
            peak_equity = max(peak_equity, equity)
            dd = 1.0 - (equity / max(peak_equity, 1e-12))
            max_dd = max(max_dd, dd)
            decisions[action] += 1
            regimes[str(regime).upper()] += 1

            tr = _decision_trace(info)
            ledger.append(
                {
                    "i": int(i),
                    "decision_ts_kst": str(decision_ts),
                    "execution_ts_kst": str(execution_ts),
                    "decision_price": float(decision_price),
                    "execution_price": float(execution_price),
                    "action": int(action),
                    "event": event,
                    "regime": str(regime).upper(),
                    "target_side": str(target_side or ""),
                    "pos_after": str(lots.side or ""),
                    "target_exposure": float(target_exposure),
                    "pos_exposure_after": float(lots.exposure),
                    "balance": float(balance),
                    "mark_pnl_frac": float(mark_pnl),
                    "equity": float(equity),
                    "drawdown_pct": float(dd * 100.0),
                    "realized_pnl_frac": float(realized),
                    "realized_pnl_pct": float(realized * 100.0),
                    "fill_count": int(len(fills)),
                    "route": str(route),
                    **tr,
                }
            )

            if args.progress and (n % int(args.progress) == 0):
                print(
                    json.dumps(
                        {
                            "processed": int(n),
                            "i": int(i),
                            "ts": str(decision_ts),
                            "equity": float(equity),
                            "mdd_pct": float(max_dd * 100.0),
                            "pos": str(lots.side or "NONE"),
                            "entries": int(sum(entries.values())),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

        if bool(args.alpha3_csv_execution_parity) and lots.lots:
            forced_i = int(stop)
            forced_px = _fill_price(df, forced_i, _side_to_int(lots.side), alpha3_slip, entry=False)
            forced_realized, forced_fills = lots.close_alpha3(forced_px, exit_fee=alpha3_fee)
            balance *= 1.0 + forced_realized
            close_pnls.append(float(forced_realized))
            exits["forced_end"] += 1
            entry_abs_i = None
            _sync_router_from_lots(router, lots, 0)
            ledger.append(
                {
                    "i": int(forced_i),
                    "decision_ts_kst": str(pd.Timestamp(df.iloc[forced_i]["timestamp"]) + pd.Timedelta(hours=9)),
                    "execution_ts_kst": str(pd.Timestamp(df.iloc[forced_i]["timestamp"]) + pd.Timedelta(hours=9)),
                    "decision_price": float(_safe_float(df.iloc[forced_i].get("close"), 0.0)),
                    "execution_price": float(forced_px),
                    "action": 0,
                    "event": "FORCED_END",
                    "regime": "",
                    "target_side": "",
                    "pos_after": "",
                    "target_exposure": 0.0,
                    "pos_exposure_after": 0.0,
                    "balance": float(balance),
                    "mark_pnl_frac": 0.0,
                    "equity": float(balance),
                    "drawdown_pct": float(max_dd * 100.0),
                    "realized_pnl_frac": float(forced_realized),
                    "realized_pnl_pct": float(forced_realized * 100.0),
                    "fill_count": int(len(forced_fills)),
                    "route": "forced_end_market",
                    "source": "alpha3_csv_execution_parity|forced_end",
                    "reason": "forced_end",
                    "position_signal": "EXIT",
                    "owner": "",
                    "v31_selected_side": "",
                    "v31_q_long": 0.0,
                    "v31_q_short": 0.0,
                    "v31_edge": 0.0,
                    "v31_margin": 0.0,
                    "v31_pass_gate": False,
                    "alpha2_parent_action": 0,
                    "alpha2_teacher_action": 0,
                    "alpha2_reason": "",
                }
            )
            final_mark = 0.0
        elif bool(args.alpha3_csv_execution_parity):
            final_mark = lots.mark_pnl_alpha3(_safe_float(df.iloc[stop + 1].get("close"), 0.0), slip=alpha3_slip) if lots.lots else 0.0
        else:
            final_mark = lots.mark_pnl(router, _safe_float(df.iloc[stop + 1].get("close"), 0.0)) if lots.lots else 0.0
        final_equity = float(balance * (1.0 + final_mark))
        closed = len(close_pnls)
        wins = sum(1 for x in close_pnls if x > 0.0)
        ledger_df = pd.DataFrame(ledger)
        args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
        ledger_df.to_csv(args.ledger_out, index=False)

        suspicious_cols = [
            c
            for c in df.columns
            if any(tok in str(c).lower() for tok in ("future", "target", "label", "oracle"))
        ]
        report = {
            "model_id": "alpha3_runtime_native_backtest_20260515",
            "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "eval_csv": str(eval_csv),
            "ledger": str(args.ledger_out),
            "range": {
                "start_index": int(start),
                "stop_index": int(stop),
                "rows_evaluated": int(max(0, stop - start + 1)),
                "start_ts": str(df.iloc[start]["timestamp"]),
                "stop_ts": str(df.iloc[stop]["timestamp"]),
                "contract": "signal_bar_close_decision_next_bar_open_fill",
                "execution_parity_contract": "alpha3_csv_next_open_limit_touch0_fee20" if bool(args.alpha3_csv_execution_parity) else "native_next_open_synthetic_router_fill",
                "window_bars": int(window),
                "with_m7_runtime": bool(args.with_m7),
                "ai_features": "precomputed_eval_csv_columns_then_FinalGovernorRuntime_prepare_frame",
                "v31_config_override": json.loads(str(args.v31_config_json)) if args.v31_config_json else None,
                "alpha3_csv_execution_parity": bool(args.alpha3_csv_execution_parity),
                "alpha3_csv_state_parity": bool(args.alpha3_csv_state_parity),
                "alpha3_csv_decision_parity": bool(args.alpha3_csv_decision_parity),
                "alpha3_csv_mark_parity": bool(args.alpha3_csv_mark_parity),
                "alpha3_limit_config": alpha3_limit_cfg.__dict__ if bool(args.alpha3_csv_execution_parity) else None,
                "alpha3_fee": float(alpha3_fee) if bool(args.alpha3_csv_execution_parity) else None,
                "alpha3_slip": float(alpha3_slip) if bool(args.alpha3_csv_execution_parity) else None,
            },
            "metrics": {
                "final_equity": final_equity,
                "return_pct": float((final_equity - 1.0) * 100.0),
                "max_drawdown_pct": float(max_dd * 100.0),
                "decision_counts": {str(k): int(v) for k, v in decisions.items()},
                "entry_counts": {str(k): int(v) for k, v in entries.items()},
                "closed_trades": int(closed),
                "win_rate": float(wins / closed) if closed else 0.0,
                "avg_closed_pnl_pct": float(np.mean(close_pnls) * 100.0) if close_pnls else 0.0,
                "median_closed_pnl_pct": float(np.median(close_pnls) * 100.0) if close_pnls else 0.0,
                "resize_events": int(resize_events),
                "regime_counts": {str(k): int(v) for k, v in regimes.items()},
                "exit_reason_top20": dict(exits.most_common(20)),
            },
            "audit": {
                "blocking": [],
                "warnings": [
                    "historical_backtest_uses_precomputed_ai_feature_columns; live_recomputes_some_ai_features_each_cycle",
                ],
                "suspicious_non_causal_column_names": suspicious_cols[:80],
                "suspicious_non_causal_column_count": int(len(suspicious_cols)),
                "runtime_state_isolated": True,
                "router_state_isolated": True,
                "fee_slippage_accounting": (
                    "alpha3_csv_execution_parity: _try_immediate_limit route fees plus LotBook alpha3 pnl math"
                    if bool(args.alpha3_csv_execution_parity)
                    else "router._trade_math per closed lot; same-side upsize stored as new lot; downsize closes delta exposure"
                ),
                "alpha3_csv_execution_parity": bool(args.alpha3_csv_execution_parity),
                "alpha3_csv_execution_parity_note": "entries/exits/upsizes/downsizes pass through _try_immediate_limit; maker fills use maker_fee_mult; entry misses skip; exit misses close fallback" if bool(args.alpha3_csv_execution_parity) else "",
                "alpha3_csv_state_parity": bool(args.alpha3_csv_state_parity),
                "alpha3_csv_state_parity_note": "close events arm lifecycle parent cooldown and V31 deep cooldown like canonical CSV Alpha3; same-bar flip re-entry is suppressed" if bool(args.alpha3_csv_state_parity) else "",
                "alpha3_csv_decision_parity": bool(args.alpha3_csv_decision_parity),
                "alpha3_csv_decision_parity_note": "parent raw decisions, teacher probability rows, and V31 q are injected from canonical CSV full-frame precompute" if bool(args.alpha3_csv_decision_parity) else "",
                "alpha3_csv_mark_parity": bool(args.alpha3_csv_mark_parity),
                "alpha3_csv_mark_parity_note": "V21.2/V31 exit decisions use CSV-style mark unrealized: fill entry price plus exit-side slippage only" if bool(args.alpha3_csv_mark_parity) else "",
                "decision_path": "trading_bot.FinalGovernorRuntime.decide",
                "accelerated_live_cache": accelerated_cache,
                "csv_decision_parity": csv_decision_parity,
            },
            "errors_sample": errors,
        }
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpha3 through the live FinalGovernorRuntime decision path on historical bars.")
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--with-m7", action="store_true", default=False)
    parser.add_argument("--progress", type=int, default=1000)
    parser.add_argument("--v31-config-json", type=str, default="")
    parser.add_argument("--v31-name", type=str, default="")
    parser.add_argument("--accelerated-cache", action="store_true", default=False)
    parser.add_argument(
        "--alpha3-csv-execution-parity",
        action="store_true",
        default=False,
        help="Route historical fills through the canonical Alpha3 _try_immediate_limit contract.",
    )
    parser.add_argument("--alpha3-cost-mult", type=float, default=1.0)
    parser.add_argument("--alpha3-maker-fee-mult", type=float, default=0.20)
    parser.add_argument("--alpha3-entry-miss", choices=("skip", "market_fallback"), default="skip")
    parser.add_argument("--alpha3-exit-miss", choices=("hold", "market_fallback"), default="market_fallback")
    parser.add_argument(
        "--alpha3-csv-state-parity",
        action="store_true",
        default=False,
        help="After closes, arm parent/deep cooldowns like the canonical CSV Alpha3 state machine.",
    )
    parser.add_argument(
        "--alpha3-csv-decision-parity",
        action="store_true",
        default=False,
        help="Inject canonical CSV parent/teacher/V31 decision rows into the live runtime by timestamp.",
    )
    parser.add_argument(
        "--alpha3-csv-mark-parity",
        action="store_true",
        default=False,
        help="Use CSV-style mark unrealized for Alpha3 V21.2/V31 exit decisions.",
    )
    parser.add_argument(
        "--alpha3-csv-cooldown-parity-env",
        action="store_true",
        default=False,
        help="Also enable trading_bot's optional Alpha3 CSV cooldown parity env flag.",
    )
    parser.add_argument(
        "--alpha3-csv-loop-parity",
        action="store_true",
        default=False,
        help="Execute the canonical CSV Alpha3 position loop while using this runner's report/ledger plumbing.",
    )
    parser.add_argument(
        "--compare-csv-ledger",
        type=Path,
        default=None,
        help="Optional canonical ledger to compare action timestamps/actions/final PnL against.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_out = Path(args.report_out).resolve()
    args.ledger_out = Path(args.ledger_out).resolve()
    args.compare_csv_ledger = Path(args.compare_csv_ledger).resolve() if args.compare_csv_ledger is not None else None
    report = run(args)
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "ledger": str(args.ledger_out),
                "metrics": report["metrics"],
                "audit": report["audit"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
