#!/usr/bin/env python3
"""Cleanroom rewrite of live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py -- IDENTICAL decision
behavior, but built entirely on new Odyssey-owned modules instead of the shared production
Omega4.6.1 live stack.

=== Why this file exists ===
The 08-14 shadow imports `trading_bot_modules.omega4_6_1_live` (shared with real ETH live order
execution in trading_bot.py) and `trading_bot_modules.runtime_config` (pulls in a completely
unrelated Omega5 system and raises at import time on Omega5 env-var mismatches that have nothing to
do with ETH/Odyssey). Tracing the full import graph found ETH Odyssey pulling in ~6,850 lines across
8 files, of which SOL/BTC training-script imports (886 lines) and a risk-sidecar training-script
import (1,697 lines, zero references) were 100% dead weight, plus a large Omega5-oriented adapter
module dragged in for two symbols. Full trace: docs/experiments/eth_odyssey_live_cleanroom_
dependency_rewrite_20260816.md.

This file replaces that import graph with three new, Odyssey-owned, ETH-only modules
(`trading_bot_modules.odyssey_tabm_core`, `.odyssey_regime3_live`, `.odyssey_live_adapter`) that
were verified bit-identical against the production adapter on 400 real historical entry decisions
and 800 exit-probability evaluations (see the same doc) before this file was written. Everything
else -- state schema, OUT_DIR, the shared sustained-uptrend detector, the h48qual regime-aware exit
guard, the zig075 SHORT entry veto -- is copied verbatim from the 08-14 shadow so this file is a
drop-in replacement: pointing the same systemd unit at this script resumes from the same
`data/live/eth_odyssey4_shadow/state.json`, no shadow history is lost.

RESEARCH/SHADOW ONLY -- order_submission_supported=False, activation_allowed=False, matching every
other shadow bot in this repo. Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py
/ trading_bot_modules/runtime_config.py / .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from trading_bot_modules.odyssey_tabm_core import load_fee_slip  # noqa: E402
from trading_bot_modules.odyssey_live_adapter import (  # noqa: E402
    DURATION_THRESHOLD as ODYSSEY_LIVE_DURATION_THRESHOLD,
    EXIT_THRESHOLD,
    ODYSSEY_H48QUAL_BUNDLE_PATH,
    ODYSSEY_H48QUAL_SIDECAR_PATH,
    ODYSSEY_ZIG075_BUNDLE_PATH,
    ODYSSEY_ZIG075_SIDECAR_PATH,
    OdysseyLiveAdapter,
    _Component,
    _ComponentConfig,
)

KST = ZoneInfo("Asia/Seoul")
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.jsonl"
DASHBOARD_STATE_PATH = ROOT / "data/live/dashboard_state.json"
OUT_DIR = ROOT / "data/live/eth_odyssey4_shadow"
STATE_PATH = OUT_DIR / "state.json"
TRADES_PATH = OUT_DIR / "closed_trades.jsonl"
EQUITY_PATH = OUT_DIR / "equity_curve.jsonl"
VETO_EVENTS_PATH = OUT_DIR / "zig075_short_veto_events.jsonl"

# =====================================================================================================
# Detector constants -- copied verbatim from live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py.
# NEVER recalibrated live. Shared by BOTH the h48qual exit guard AND the zig075 entry veto -- one
# instance, one .update() per bar.
# =====================================================================================================
DETECTOR_WEEK_BARS = 2016
DETECTOR_PERCENTILE = 0.90
DETECTOR_THRESHOLD = 0.8025793650793651
DETECTOR_CALIBRATION_WINDOW = ("2025-01-01", "2025-06-30 23:59:59")
DETECTOR_SOURCE_REPORT = (
    "tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814/report.json"
)
FALLBACK_BASE_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

# h48qual: exit_head swapped for the 2026-08-13 liveATR-relabel retrain -- same artifact selection as
# the 08-14 shadow. Used for entry_decision (always) and exit_probability (only when the detector is
# inactive).
H48QUAL_NEW_BUNDLE_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
    "/h48qual/true_3head_tabm_bundle.pt"
)
H48QUAL_SHIM_SIDECAR_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar"
    "/risk_sidecar.pkl"
)
# h48qual ORIGINAL (pre-liveATR-relabel) exit head -- used only via .exit_probability, only when the
# detector is active on an open h48qual position. Never used for entry/sizing. Sourced from Odyssey's
# own artifact-path constants (odyssey_live_adapter.py), NOT trading_bot_modules.runtime_config --
# decoupled on purpose so nothing about this shadow's config can be affected by (or affect) the
# unrelated Omega5 system that runtime_config.py also configures.
H48QUAL_ORIGINAL_BUNDLE_PATH = ROOT / ODYSSEY_H48QUAL_BUNDLE_PATH
H48QUAL_ORIGINAL_SIDECAR_PATH = ROOT / ODYSSEY_H48QUAL_SIDECAR_PATH
# zig075: fully original live artifacts. The detector NEVER touches zig075's direction/quality/
# TP/SL/sizing -- only whether a SHORT signal is allowed to queue at all.
COMPONENTS_OVERRIDE = {
    "h48qual": {
        "bundle": H48QUAL_NEW_BUNDLE_PATH,
        "sidecar": H48QUAL_SHIM_SIDECAR_PATH,
        "quality_threshold": 0.50,
    },
    "zig075": {
        "bundle": ROOT / ODYSSEY_ZIG075_BUNDLE_PATH,
        "sidecar": ROOT / ODYSSEY_ZIG075_SIDECAR_PATH,
        "quality_threshold": 0.75,
    },
}
PRIORITY = ("h48qual", "zig075")
BUFFER_ROWS = 3000


class SustainedUptrendDetector:
    """Identical to live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py's class of the same name
    -- copied, not imported, to keep this shadow a fully self-contained deployment unit (matches this
    repo's precedent of each shadow script being independently copyable to the server)."""

    def __init__(self, *, threshold: float, week_bars: int) -> None:
        self.threshold = float(threshold)
        self.week_bars = int(week_bars)
        self._window: deque[float] = deque(maxlen=self.week_bars)

    def seed(self, dual_momentum_values: np.ndarray) -> None:
        self._window.clear()
        for v in dual_momentum_values:
            self._window.append(1.0 if float(v) > 0.0 else 0.0)

    def update(self, dual_momentum_value: float) -> tuple[float | None, bool]:
        self._window.append(1.0 if float(dual_momentum_value) > 0.0 else 0.0)
        if len(self._window) < self.week_bars:
            return None, False
        score = sum(self._window) / self.week_bars
        return score, bool(score > self.threshold)

    def __len__(self) -> int:
        return len(self._window)


def now_kst() -> pd.Timestamp:
    return pd.Timestamp.now(tz=KST)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def build_adapter(device: str = "cpu") -> OdysseyLiveAdapter:
    return OdysseyLiveAdapter(
        h48qual_bundle="", h48qual_sidecar="", zig075_bundle="", zig075_sidecar="",
        device=device,
        components_override=COMPONENTS_OVERRIDE,
        priority=PRIORITY,
    )


def build_guard_component(device: torch.device) -> _Component:
    cfg = _ComponentConfig(
        "h48qual_guard_original", H48QUAL_ORIGINAL_BUNDLE_PATH, H48QUAL_ORIGINAL_SIDECAR_PATH,
        quality_threshold=0.50,
    )
    return _Component(cfg, device=device)


def evaluate_exit_guarded(
    adapter: OdysseyLiveAdapter, guard_component: _Component, frame: pd.DataFrame, *,
    source_component: str, side: int, hold_bars: int, unrealized_move: float, mfe: float, mae: float,
    notional: float, leverage: float, take_profit: float, stop_loss: float,
    bar_high_move: float | None, bar_low_move: float | None, detector_active: bool,
) -> tuple[bool, str, float, dict[str, Any]]:
    """Byte-identical to live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py's function of the same
    name -- the h48qual exit guard is completely unchanged by this rewrite; only the module each
    piece is sourced from is new (see decide_and_queue_entry below)."""
    if source_component != "h48qual" or not detector_active:
        should_exit, reason, exit_prob = adapter.evaluate_exit(
            frame, source_component=source_component, side=side, hold_bars=hold_bars,
            unrealized_move=unrealized_move, mfe=mfe, mae=mae, notional=notional, leverage=leverage,
            take_profit=take_profit, stop_loss=stop_loss, bar_high_move=bar_high_move, bar_low_move=bar_low_move,
        )
        return should_exit, reason, exit_prob, {"guard_engaged": False, "decision_differs": None, "default_prob": None}

    frame = adapter.regime3_current.append(frame)
    tp_move = unrealized_move if bar_high_move is None else bar_high_move
    sl_move = unrealized_move if bar_low_move is None else bar_low_move
    if stop_loss > 0.0 and sl_move <= -abs(stop_loss):
        return True, "stop_loss", 0.0, {"guard_engaged": True, "decision_differs": None, "default_prob": None}
    if take_profit > 0.0 and tp_move >= take_profit:
        return True, "take_profit", 0.0, {"guard_engaged": True, "decision_differs": None, "default_prob": None}

    prob = guard_component.exit_probability(
        frame, side=side, hold_bars=hold_bars, unrealized_move=unrealized_move, mfe=mfe, mae=mae,
        notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss,
    )
    should_exit = bool(prob >= EXIT_THRESHOLD)
    reason = "exit_head_guard_original" if should_exit else "hold"
    default_prob = adapter.components["h48qual"].exit_probability(
        frame, side=side, hold_bars=hold_bars, unrealized_move=unrealized_move, mfe=mfe, mae=mae,
        notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss,
    )
    decision_differs = bool(should_exit != (default_prob >= EXIT_THRESHOLD))
    return should_exit, reason, prob, {
        "guard_engaged": True, "decision_differs": decision_differs, "default_prob": float(default_prob),
    }


def decide_and_queue_entry(
    adapter: OdysseyLiveAdapter, frame: pd.DataFrame, bar_ts: pd.Timestamp, detector_active: bool, score: float | None,
) -> tuple[dict | None, dict[str, Any]]:
    """Odyssey4 #1's ONLY new decision-point, unchanged from the 08-14 shadow: adapter.decide_entry
    runs exactly as before. If the result is a zig075 SHORT (source_component=='zig075', side<0) AND
    the sustained-uptrend detector is active on this bar, the entry is NOT queued -- logged as vetoed
    instead. Every other decision queues exactly as the baseline would."""
    decision = adapter.decide_entry(frame)
    veto_diag = {"vetoed": False}
    if decision is None:
        return None, veto_diag
    if decision.source_component == "zig075" and decision.side < 0 and detector_active:
        veto_diag = {"vetoed": True, "detector_score": score}
        return None, veto_diag
    pending = {
        "kind": "enter", "signal_bar_ts": bar_ts.isoformat(), "side": decision.side,
        "source_component": decision.source_component, "margin_fraction": decision.margin_fraction,
        "leverage": decision.leverage, "notional_exposure": decision.notional_exposure,
        "take_profit": decision.take_profit, "stop_loss": decision.stop_loss,
    }
    return pending, veto_diag


def omega461_eth_position() -> tuple[int, str | None]:
    state = load_json(DASHBOARD_STATE_PATH, {})
    pos = state.get("position") or {}
    current = str(pos.get("current", "")).upper()
    side = 1 if current == "LONG" else (-1 if current == "SHORT" else 0)
    return side, pos.get("opened_at")


def seed_buffer(path: Path, rows: int) -> tuple[pd.DataFrame, int]:
    approx_bytes = rows * 16000
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - approx_bytes))
        f.readline()
        data = f.read().decode("utf-8", errors="ignore")
    out = []
    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = payload.get("values")
        if isinstance(values, dict):
            out.append(values)
    return pd.DataFrame(out), size


def read_new_rows(path: Path, offset: int) -> tuple[int, list[dict]]:
    size = path.stat().st_size
    if offset < 0 or offset > size:
        offset = size
    rows: list[dict] = []
    with path.open("rb") as f:
        f.seek(offset)
        while True:
            pos_before = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                return pos_before, rows
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            values = payload.get("values")
            if isinstance(values, dict):
                rows.append(values)
        return f.tell(), rows


def seed_detector(detector: SustainedUptrendDetector, buffer: pd.DataFrame, last_processed_ts: pd.Timestamp | None) -> None:
    if last_processed_ts is not None:
        seed_rows = buffer[buffer["timestamp"] <= last_processed_ts]
    else:
        seed_rows = buffer.iloc[:-1] if len(buffer) > 0 else buffer
    if "dual_momentum" in seed_rows.columns and len(seed_rows) > 0:
        live_dm = pd.to_numeric(seed_rows["dual_momentum"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        live_dm = np.asarray([], dtype=float)

    fallback_dm = np.asarray([], dtype=float)
    shortfall = detector.week_bars - len(live_dm)
    if shortfall > 0 and FALLBACK_BASE_CSV.exists():
        earliest_ts = seed_rows["timestamp"].min() if len(seed_rows) else (buffer["timestamp"].min() if len(buffer) else None)
        fb = pd.read_csv(FALLBACK_BASE_CSV, usecols=["timestamp", "dual_momentum"], low_memory=False)
        fb["timestamp"] = pd.to_datetime(fb["timestamp"])
        fb = fb.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if earliest_ts is not None:
            fb = fb[fb["timestamp"] < pd.Timestamp(earliest_ts)]
        fallback_dm = pd.to_numeric(fb["dual_momentum"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[-shortfall:]
        print(f"[init] detector warm-up: live snapshot buffer alone covered {len(live_dm)}/{detector.week_bars} bars, "
              f"topping up {len(fallback_dm)} bars from {FALLBACK_BASE_CSV.name}", flush=True)

    detector.seed(np.concatenate([fallback_dm, live_dm]))
    print(f"[init] detector window seeded with {len(detector)}/{detector.week_bars} bars "
          f"(threshold={detector.threshold:.6f})", flush=True)


def try_fill_pending(state: dict, frame: pd.DataFrame, fee: float, slip: float) -> dict:
    pending = state.get("pending")
    if pending is None:
        return state
    ts_list = frame["timestamp"] if "timestamp" in frame.columns else None
    if ts_list is None or len(frame) < 2:
        return state
    signal_ts = pd.Timestamp(pending["signal_bar_ts"])
    idx = frame.index[frame["timestamp"] == signal_ts]
    if len(idx) == 0 or idx[0] + 1 >= len(frame):
        return state
    fill_row = frame.iloc[idx[0] + 1]
    fill_open = float(fill_row["open"])
    fill_ts = pd.Timestamp(fill_row["timestamp"])

    if pending["kind"] == "enter":
        side = int(pending["side"])
        entry_price = fill_open * (1 + slip if side > 0 else 1 - slip)
        state["position"] = {
            "side": side, "source_component": pending["source_component"], "entry_price": entry_price,
            "entry_ts": fill_ts.isoformat(), "notional_exposure": pending["notional_exposure"],
            "margin_fraction": pending["margin_fraction"], "leverage": pending["leverage"],
            "take_profit": pending["take_profit"], "stop_loss": pending["stop_loss"],
            "hold_bars": 0, "mfe": 0.0, "mae": 0.0, "_prev_pnl_frac": 0.0,
        }
        print(f"[fill] ENTER side={side} src={pending['source_component']} price={entry_price:.2f} "
              f"notional={pending['notional_exposure']:.3f} ts={fill_ts}", flush=True)
    else:
        pos = state["position"]
        side = int(pos["side"])
        exit_price = fill_open * (1 - slip if side > 0 else 1 + slip)
        entry_price = float(pos["entry_price"])
        raw_move = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        notional = float(pos["notional_exposure"])
        realized_frac = raw_move * notional - float(pos.get("_prev_pnl_frac", 0.0)) - fee * notional
        state["equity"] = float(state.get("equity", 1.0)) * (1.0 + realized_frac)
        closed = {
            "entry_ts": pos["entry_ts"], "exit_ts": fill_ts.isoformat(), "side": side,
            "source_component": pos["source_component"], "entry_price": entry_price, "exit_price": exit_price,
            "reason": pending["reason"], "notional_exposure": notional, "raw_price_move": raw_move,
            "trade_return_frac": realized_frac, "hold_bars": pos["hold_bars"],
        }
        append_jsonl(TRADES_PATH, closed)
        print(f"[fill] EXIT side={side} src={pos['source_component']} reason={pending['reason']} "
              f"price={exit_price:.2f} trade_return={realized_frac*100:.3f}% equity={state['equity']:.4f} ts={fill_ts}", flush=True)
        state["position"] = None
    state["pending"] = None
    return state


def process_bar(
    state: dict, frame: pd.DataFrame, adapter: OdysseyLiveAdapter, guard_component: _Component,
    detector: SustainedUptrendDetector, fee: float, slip: float,
) -> dict:
    bar = frame.iloc[-1]
    bar_ts = pd.Timestamp(bar["timestamp"])
    score, detector_active = detector.update(float(pd.to_numeric(bar["dual_momentum"])))
    state["last_detector_score"] = score
    state["last_detector_active"] = bool(detector_active)
    state["detector_bars_seen"] = int(state.get("detector_bars_seen", 0)) + 1

    # Diagnostic-only: entry_decision() computes quality_score before thresholding (see
    # odyssey_live_adapter.py _Component.entry_decision), but decide_entry()'s priority loop
    # discards it for any component whose signal doesn't qualify. Read it directly here, every bar
    # regardless of position/pending state, purely for dashboard display -- does not affect entry
    #/exit/veto decisions, which still flow through adapter.decide_entry()/evaluate_exit_guarded
    # unchanged. Regime3CurrentLiveFeatures.append() is a pure transform (returns a new frame,
    # never mutates self) so calling it here in addition to decide_entry()'s own internal call is
    # safe -- confirmed by reading trading_bot_modules/odyssey_regime3_live.py. Ported from
    # live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py (added there 2026-08-16 by a
    # concurrent session, "feat: show live h48qual/zig075 quality scores on the Odyssey4 shadow
    # dashboard") so this cleanroom script doesn't regress that dashboard feature.
    regime3_frame = adapter._with_regime3(frame)
    for alias in ("h48qual", "zig075"):
        diag = adapter.components[alias].entry_decision(regime3_frame)
        state[f"last_{alias}_quality_score"] = float(diag["quality_score"])
        state[f"last_{alias}_quality_threshold"] = float(adapter.components[alias].cfg.quality_threshold)
    position = state.get("position")

    if position is not None and state.get("pending") is None:
        side = int(position["side"])
        entry_price = float(position["entry_price"])
        close_px, high_px, low_px = float(bar["close"]), float(bar["high"]), float(bar["low"])
        unrealized_move = ((close_px - entry_price) / entry_price if side > 0
                            else (entry_price - close_px) / entry_price)
        if side > 0:
            best_move = (high_px - entry_price) / entry_price
            worst_move = (low_px - entry_price) / entry_price
        else:
            best_move = (entry_price - low_px) / entry_price
            worst_move = (entry_price - high_px) / entry_price
        position["hold_bars"] = int(position.get("hold_bars", 0)) + 1
        position["mfe"] = max(float(position.get("mfe", 0.0)), unrealized_move)
        position["mae"] = min(float(position.get("mae", 0.0)), unrealized_move)

        if position["source_component"] == "h48qual":
            state["h48qual_hold_bars"] = int(state.get("h48qual_hold_bars", 0)) + 1

        should_exit, reason, exit_prob, exit_diag = evaluate_exit_guarded(
            adapter, guard_component, frame, source_component=position["source_component"], side=side,
            hold_bars=position["hold_bars"], unrealized_move=unrealized_move, mfe=position["mfe"],
            mae=position["mae"], notional=position["notional_exposure"], leverage=position["leverage"],
            take_profit=position["take_profit"], stop_loss=position["stop_loss"],
            bar_high_move=best_move, bar_low_move=worst_move, detector_active=detector_active,
        )
        if exit_diag["guard_engaged"]:
            state["h48qual_guard_active_bars"] = int(state.get("h48qual_guard_active_bars", 0)) + 1
            if exit_diag["decision_differs"]:
                state["h48qual_guard_decision_differs_bars"] = int(state.get("h48qual_guard_decision_differs_bars", 0)) + 1

        notional = float(position["notional_exposure"])
        mark_frac = unrealized_move * notional - float(position.get("_prev_pnl_frac", 0.0))
        state["equity"] = float(state.get("equity", 1.0)) * (1.0 + mark_frac)
        position["_prev_pnl_frac"] = unrealized_move * notional
        score_str = "nan" if score is None else f"{score:.4f}"
        print(f"[bar {bar_ts}] hold pos side={side} src={position['source_component']} "
              f"unreal={unrealized_move*100:.3f}% mfe={position['mfe']*100:.2f}% mae={position['mae']*100:.2f}% "
              f"exit_prob={exit_prob:.3f} reason={reason} detector_score={score_str} detector_active={detector_active} "
              f"guard_engaged={exit_diag['guard_engaged']} decision_differs={exit_diag['decision_differs']}", flush=True)
        if should_exit:
            state["pending"] = {"kind": "exit", "signal_bar_ts": bar_ts.isoformat(), "reason": reason}
    elif position is None and state.get("pending") is None:
        pending, veto_diag = decide_and_queue_entry(adapter, frame, bar_ts, detector_active, score)
        if veto_diag["vetoed"]:
            state["zig075_short_veto_bars"] = int(state.get("zig075_short_veto_bars", 0)) + 1
            event = {"bar_ts": bar_ts.isoformat(), "detector_score": veto_diag["detector_score"]}
            append_jsonl(VETO_EVENTS_PATH, event)
            print(f"[veto {bar_ts}] zig075 SHORT signal SKIPPED (sustained-uptrend detector active, "
                  f"score={veto_diag['detector_score']:.4f})", flush=True)
        elif pending is not None:
            state["pending"] = pending
            print(f"[signal {bar_ts}] ENTER queued side={pending['side']} src={pending['source_component']} "
                  f"notional={pending['notional_exposure']:.3f} tp={pending['take_profit']:.4f} sl={pending['stop_loss']:.4f}", flush=True)

    peak = max(float(state.get("peak_equity", 1.0)), float(state["equity"]))
    state["peak_equity"] = peak
    state["mdd"] = min(float(state.get("mdd", 0.0)), float(state["equity"]) / peak - 1.0)
    omega_side, _ = omega461_eth_position()
    state["last_processed_bar_ts"] = bar_ts.isoformat()
    state["last_decision_at_kst"] = now_kst().isoformat()
    append_jsonl(EQUITY_PATH, {
        "bar_ts": bar_ts.isoformat(), "equity": state["equity"], "mdd": state["mdd"],
        "position_side": (state.get("position") or {}).get("side", 0),
        "real_live_omega461_eth_side": omega_side,
        "detector_score": score, "detector_active": bool(detector_active),
    })
    return state


def run(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("[init] loading Odyssey4 cleanroom adapter (h48qual q050 = liveATR relabel for entry+"
          "default exit, ORIGINAL exit_head substituted while detector active; zig075 q075 = fully "
          "original except SHORT entries are skipped while detector active; regime3 = original live "
          "HMM) -- research/shadow only, see module docstring. code_lineage=cleanroom_20260816 "
          "(trading_bot_modules.odyssey_tabm_core/.odyssey_regime3_live/.odyssey_live_adapter, "
          "verified bit-identical to the 08-14 shadow's production-adapter path before cutover)", flush=True)
    device_obj = torch.device(args.device)
    adapter = build_adapter(device=args.device)
    guard_component = build_guard_component(device=device_obj)
    print(f"[init] duration_threshold={adapter.duration_threshold} (live default "
          f"{ODYSSEY_LIVE_DURATION_THRESHOLD}, unmodified)", flush=True)
    print(f"[init] guard component loaded: alias={guard_component.cfg.alias} "
          f"base_cols={len(guard_component.base_cols)} quality_threshold={guard_component.cfg.quality_threshold} "
          f"experts={sorted(guard_component.loaded)}", flush=True)
    fee, slip = load_fee_slip()
    detector = SustainedUptrendDetector(threshold=DETECTOR_THRESHOLD, week_bars=DETECTOR_WEEK_BARS)

    state = load_json(STATE_PATH, {})
    if not state:
        buffer, offset = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        seed_detector(detector, buffer, None)
        state = {
            "schema_version": "eth_odyssey4_shadow.v1", "live_forward_only": True,
            "order_submission_supported": False, "activation_allowed": False,
            "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "candidate": "Odyssey4: h48qual regime-aware exit-head guard (Odyssey3, unchanged) + "
                          "zig075 SHORT sustained-uptrend entry veto (Odyssey4 #1, CONFIRMED). One "
                          "shared detector = rolling 1-week (2016-bar) fraction of dual_momentum>0, "
                          f"threshold={DETECTOR_THRESHOLD:.6f} (90th pct of 2025-Q1+Q2-only "
                          "calibration, reused verbatim, zero new free parameters).",
            "code_lineage": "cleanroom_20260816",
            "research_doc_h48qual_guard": "docs/experiments/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.md",
            "research_doc_zig075_veto": "docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md",
            "contract": "docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md",
            "supersedes": ["live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py (same decisions, old dependency graph)"],
            "detector_week_bars": DETECTOR_WEEK_BARS, "detector_threshold": DETECTOR_THRESHOLD,
            "started_at_kst": now_kst().isoformat(), "snapshot_offset": offset,
            "position": None, "pending": None, "last_processed_bar_ts": None,
            "equity": 1.0, "peak_equity": 1.0, "mdd": 0.0,
            "last_detector_score": None, "last_detector_active": False, "detector_bars_seen": 0,
            "h48qual_hold_bars": 0, "h48qual_guard_active_bars": 0, "h48qual_guard_decision_differs_bars": 0,
            "zig075_short_veto_bars": 0,
        }
        print(f"[init] seeded buffer with {len(buffer)} rows, snapshot_offset={offset}", flush=True)
    else:
        buffer, _ = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        last_processed = state.get("last_processed_bar_ts")
        seed_detector(detector, buffer, pd.Timestamp(last_processed) if last_processed else None)
        print(f"[resume] loaded prior state, last_processed_bar_ts={state.get('last_processed_bar_ts')}, "
              f"equity={state.get('equity')}, h48qual_guard_active_bars={state.get('h48qual_guard_active_bars')}, "
              f"zig075_short_veto_bars={state.get('zig075_short_veto_bars')}", flush=True)
        state.setdefault("code_lineage", "cleanroom_20260816")

    offset = int(state["snapshot_offset"])
    end_at = pd.Timestamp(args.end_at_kst, tz=KST) if args.end_at_kst else None

    while end_at is None or now_kst() < end_at:
        offset, new_rows = read_new_rows(SNAPSHOT_PATH, offset)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            buffer = pd.concat([buffer, new_df], ignore_index=True)
            buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
            buffer = buffer.sort_values("timestamp").drop_duplicates("timestamp")
            buffer = buffer.tail(BUFFER_ROWS).reset_index(drop=True)
        state["snapshot_offset"] = offset

        if len(buffer) >= 250:
            last_processed = state.get("last_processed_bar_ts")
            last_processed_ts = pd.Timestamp(last_processed) if last_processed else None
            new_bars = buffer[buffer["timestamp"] > last_processed_ts] if last_processed_ts is not None else buffer.tail(1)
            for i in range(len(new_bars)):
                bar_ts = pd.Timestamp(new_bars.iloc[i]["timestamp"])
                frame = buffer[buffer["timestamp"] <= bar_ts]
                if frame[["open", "high", "low", "close"]].isna().any().any():
                    print(f"[skip] bar {bar_ts} has NaN OHLC", flush=True)
                    state["last_processed_bar_ts"] = bar_ts.isoformat()
                    continue
                try:
                    frame = adapter.regime3_current.append(frame)
                    if len(frame) > 1:
                        _numeric_cols = frame.select_dtypes(include=[np.number]).columns
                        _hist_idx = frame.index[:-1]
                        frame.loc[_hist_idx, _numeric_cols] = (
                            frame.loc[_hist_idx, _numeric_cols].ffill().bfill()
                        )
                    state = try_fill_pending(state, frame, fee, slip)
                    state = process_bar(state, frame, adapter, guard_component, detector, fee, slip)
                except Exception as exc:  # noqa: BLE001 - log and keep the shadow loop alive
                    print(f"[error] bar {bar_ts}: {exc!r}", flush=True)
                    state["last_processed_bar_ts"] = bar_ts.isoformat()

        write_json(STATE_PATH, state)
        time.sleep(max(30.0, float(args.poll_seconds)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=float, default=90.0)
    ap.add_argument("--end-at-kst", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
