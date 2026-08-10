#!/usr/bin/env python3
"""LIVE shadow test of the ETH regime3 HMM->JM(lambda=4) swap candidate against the REAL live
Omega4.6.1 ETH position. NO real orders are placed -- this only tracks a hypothetical Omega4.6.1
position built with the SAME h48qual/zig075 TabM parents and risk sidecars, but with the live
12-state sticky HMM regime3 classifier replaced by the JM(k=3, lambda=4) classifier, for observation
only. See project-eth-regime3-jm-lam4-swap-retrain-20260809.md for why this candidate is CLOSED for
live promotion -- n=11 not statistically significant (p_mean=0.594 on effect_size_report) and the
6-month extension is untrusted due to a label-regeneration defect. [CORRECTED 2026-08-10: an earlier
claim here that "the real production risk gate rejects both sidecars outright" was FALSE -- it used
a mistaken -8.00% MDD floor (the base script's argparse default) instead of the -25.00% the currently
-live sidecars were actually built under (see their own recorded selection_rule). Retrained under the
correct -25.00% floor, both components clear the gate with an eligible mapping -- no relaxation
needed. The sidecars below are therefore the CORRECTGATE artifacts (out-suffix ..._correctgate_
20260810), genuinely gate-compliant, not relaxed.] Still shadow-only because of the standing
significance and dataset-lineage gaps, not a sizing-artifact problem.

Built by directly reusing already-live, already-validated pieces rather than reimplementing anything:
  - entry/exit decision + sidecar sizing: Omega461LiveAdapter, imported verbatim from
    trading_bot_modules.omega4_6_1_live -- the SAME class the real live ETH executor runs every
    cycle. Only the JM-swapped bundles/sidecars are passed in via components_override, and
    adapter.regime3_current is swapped to a JM-based instance (see below) after construction --
    the adapter class itself is untouched.
  - raw feature panel: _num/_with_features/_class_proba below are copied verbatim (not imported --
    see the comment above Regime3CurrentLiveFeaturesJM for why) from
    experiment_regime3_current_hmm_wide24_20260529 -- the exact functions the JM regime3 classifier
    was built and validated with (build_eth_regime3_jm_lam4_20260809.py).
  - JM online causal decode: causal_decode_soft below is copied verbatim from
    build_eth_regime3_jm_lam4_20260809 -- the exact DP recursion used to build the offline JM
    regime3 CSVs this candidate's backtest results are based on.
  - raw market data: tails data/live/decision_feature_snapshot.jsonl, the same live 5m feature
    stream trading_bot.py's real live loop writes (same source the Sigma6 regime-tiebreak shadow
    bot already reads for its own shadow tracking).
  - real live ETH position for reference only: data/live/dashboard_state.json's `position` block.

Entries and exits both queue for the NEXT bar's open, matching omega4_6_1_live.py's own documented
execution-delay model (its evaluate_exit docstring: "the fill itself still executes at the next
bar's open"). This is causal, forward-only, and reads no ledger or future row -- Fresh-Forward-
compliant live-forward observation, not a backtest.

RESEARCH/SHADOW ONLY -- order_submission_supported=False, activation_allowed=False, matching every
other shadow bot in this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _load_fee_slip  # noqa: E402
from trading_bot_modules.omega4_6_1_live import (  # noqa: E402
    Omega461LiveAdapter, DURATION_THRESHOLD as _UNUSED_DEFAULT_DURATION_THRESHOLD,
)
from trading_bot_modules.omega4_6_2_source_parent_live import (  # noqa: E402
    DEFAULT_CURRENT_REGIME_PATH,
)

KST = ZoneInfo("Asia/Seoul")
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.jsonl"
DASHBOARD_STATE_PATH = ROOT / "data/live/dashboard_state.json"
JM_REGIME_PATH = ROOT / "data/ensemble/supervised/eth_regime3_current_jm_jmlam4_20260809_2024.joblib"
OUT_DIR = ROOT / "data/live/eth_jmlam4_regime_swap_shadow"
STATE_PATH = OUT_DIR / "state.json"
TRADES_PATH = OUT_DIR / "closed_trades.jsonl"
EQUITY_PATH = OUT_DIR / "equity_curve.jsonl"

# The trusted, corrected (102-feature-pinned) JM candidate from the 2mo n=11 fresh-forward result --
# see project-eth-regime3-jm-lam4-swap-retrain-20260809.md "FINAL CORRECTED RESULT". Sidecars are the
# CORRECTED-GATE sidecars (2026-08-10): retrained under the REAL production risk gate (notional
# 0.45-0.95, MDD floor -25.00% -- matching the currently-live sidecars' own recorded
# selection_rule, tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_
# {h48qual_q050,zig075_q075}_precomputed_20260630/report.json). The earlier "RELAXED-gate" sidecars
# (out-suffix ..._matched_20260809 / ..._matched_fixedthreshold_20260809) were built after a mistaken
# real-gate check used -8.00% MDD (the base script's argparse default) instead of the -25.00% the
# live model was actually built under; that mistaken check is why this module's docstring/log lines
# used to say "real production risk gate rejects both outright" -- false, see
# project-eth-regime3-jm-lam4-swap-retrain-20260809.md's 2026-08-10 correction section. Both
# components now clear the correct gate outright with an eligible mapping (no relaxation needed) and
# happen to select the SAME risk mapping as the earlier "matched" artifacts, so backtest numbers are
# unchanged -- only the artifact's own gate-compliance status changed from failing to passing.
JM_COMPONENTS_OVERRIDE = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_h48qual_ext/true_3head_tabm_bundle.pt",
        "sidecar": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q070_correctgate_20260810/risk_sidecar.pkl",
        "quality_threshold": 0.70,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_zig075/true_3head_tabm_bundle.pt",
        "sidecar": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q080_correctgate_20260810/risk_sidecar.pkl",
        "quality_threshold": 0.80,
    },
}
JM_PRIORITY = ("h48qual", "zig075")
BUFFER_ROWS = 3000  # ~10.4 days of 5m bars, comfortably covers atr_window=192 and all rolling features


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


# _num/_with_features/_class_proba/causal_decode_soft are copied verbatim (not imported) from
# experiment_regime3_current_hmm_wide24_20260529.py and build_eth_regime3_jm_lam4_20260809.py -- both
# transitively import train_regime3_hmm_mamba_20260529.py, which does `from mamba_ssm import Mamba`
# unconditionally at module level even though these specific functions never touch Mamba. mamba_ssm
# is not installed in this runtime; inlining these small pure numpy/pandas helpers avoids pulling in
# an unrelated heavy/uninstalled dependency for a shadow-only observation loop.
def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _with_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = _with_raw_state12(frame.copy())
    for col in cols:
        if col in out.columns:
            out[col] = _num(out, col).fillna(0.0)
        else:
            raise ValueError(f"missing current HMM feature column: {col}")
    return out


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    proba = state_prob @ state_class
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-300, None)
    return proba


def causal_decode_soft(x: np.ndarray, mu: np.ndarray, lam: float, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    n, k = len(x), len(mu)
    cost = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    states = np.zeros(n, dtype=np.int8)
    probs = np.zeros((n, k), dtype=np.float64)
    V = cost[0].copy()
    states[0] = int(V.argmin())
    rel0 = V - V.min()
    probs[0] = np.exp(-rel0 / temperature)
    probs[0] /= probs[0].sum()
    for t in range(1, n):
        switch = V.min() + lam
        V = cost[t] + np.minimum(V, switch)
        V -= V.min()
        states[t] = int(V.argmin())
        probs[t] = np.exp(-V / temperature)
        probs[t] /= probs[t].sum()
    return states, probs


class Regime3CurrentLiveFeaturesJM:
    """Live/online JM(k=3, lambda=4) regime3 classifier, drop-in replacement for
    Regime3CurrentLiveFeatures (trading_bot_modules.omega4_6_2_source_parent_live) -- same
    .append(frame)->frame interface and the SAME output column names (regime3_current_sensitive_
    wide24_{bull,bear,chop}_prob/_confidence/_margin/_entropy), since Omega461LiveAdapter and the
    sidecar feature builders read those exact names regardless of which model produced them.
    Recomputes the causal DP from scratch over the whole buffer each call, exactly matching how
    Regime3CurrentLiveFeatures.filter_proba() re-filters the whole window each call -- both are
    stateless-across-calls, causal-within-the-given-window."""

    def __init__(self, *, jm_path: Path) -> None:
        payload = joblib.load(Path(jm_path))
        self.cols: list[str] = list(payload["feature_cols"])
        self.medians = pd.Series({str(k): float(v) for k, v in dict(payload["feature_medians"]).items()})
        self.scaler = payload["scaler"]
        self.mu = np.asarray(payload["jm_mu"], dtype=np.float64)
        self.lam = float(payload["jm_lambda"])
        self.temperature = float(payload["jm_temperature"])
        self.state_class = np.asarray(payload["state_class_matrix"], dtype=np.float64)
        self.classes: list[str] = list(payload["classes"])
        self.prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise RuntimeError("Regime3CurrentLiveFeaturesJM received empty frame")
        work = _with_features(frame, self.cols)
        raw = work[self.cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = raw.fillna(self.medians.reindex(raw.columns)).fillna(0.0)
        if raw.isna().any().any():
            raise RuntimeError("Regime3CurrentLiveFeaturesJM non-finite model inputs after imputation")
        x_obs = self.scaler.transform(raw)
        _, state_prob = causal_decode_soft(x_obs, self.mu, self.lam, self.temperature)
        proba = _class_proba(state_prob, self.state_class)
        out = frame.copy()
        for i, name in enumerate(self.classes):
            out[f"{self.prefix}{name}_prob"] = proba[:, i]
        sorted_p = np.sort(proba, axis=1)
        out[f"{self.prefix}confidence"] = sorted_p[:, -1]
        out[f"{self.prefix}margin"] = sorted_p[:, -1] - sorted_p[:, -2]
        out[f"{self.prefix}entropy"] = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1) / np.log(len(self.classes))
        latest = out.iloc[-1]
        for col in (f"{self.prefix}bull_prob", f"{self.prefix}bear_prob", f"{self.prefix}chop_prob",
                    f"{self.prefix}confidence", f"{self.prefix}margin", f"{self.prefix}entropy"):
            if not np.isfinite(float(latest[col])):
                raise RuntimeError(f"Regime3CurrentLiveFeaturesJM non-finite latest {col}")
        return out


def build_jm_adapter(device: str = "cpu") -> Omega461LiveAdapter:
    adapter = Omega461LiveAdapter(
        h48qual_bundle="", h48qual_sidecar="", zig075_bundle="", zig075_sidecar="",  # unused, components_override wins
        current_regime_path=DEFAULT_CURRENT_REGIME_PATH,  # placeholder load only, replaced below
        device=device,
        components_override=JM_COMPONENTS_OVERRIDE,
        priority=JM_PRIORITY,
    )
    adapter.regime3_current = Regime3CurrentLiveFeaturesJM(jm_path=JM_REGIME_PATH)
    return adapter


def omega461_eth_position() -> tuple[int, str | None]:
    state = load_json(DASHBOARD_STATE_PATH, {})
    pos = state.get("position") or {}
    current = str(pos.get("current", "")).upper()
    side = 1 if current == "LONG" else (-1 if current == "SHORT" else 0)
    return side, pos.get("opened_at")


def seed_buffer(path: Path, rows: int) -> tuple[pd.DataFrame, int]:
    approx_bytes = rows * 16000  # ~14-15KB/line observed
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


@dataclass
class PendingAction:
    kind: str  # "enter" or "exit"
    signal_bar_ts: str
    side: int = 0
    source_component: str = ""
    margin_fraction: float = 0.0
    leverage: float = 0.0
    notional_exposure: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    reason: str = ""


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
        return state  # target (next) bar hasn't printed yet
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
    else:  # exit
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


def process_bar(state: dict, frame: pd.DataFrame, adapter: Omega461LiveAdapter, fee: float, slip: float) -> dict:
    bar = frame.iloc[-1]
    bar_ts = pd.Timestamp(bar["timestamp"])
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

        should_exit, reason, exit_prob = adapter.evaluate_exit(
            frame, source_component=position["source_component"], side=side, hold_bars=position["hold_bars"],
            unrealized_move=unrealized_move, mfe=position["mfe"], mae=position["mae"],
            notional=position["notional_exposure"], leverage=position["leverage"],
            take_profit=position["take_profit"], stop_loss=position["stop_loss"],
            bar_high_move=best_move, bar_low_move=worst_move,
        )
        notional = float(position["notional_exposure"])
        mark_frac = unrealized_move * notional - float(position.get("_prev_pnl_frac", 0.0))
        state["equity"] = float(state.get("equity", 1.0)) * (1.0 + mark_frac)
        position["_prev_pnl_frac"] = unrealized_move * notional
        print(f"[bar {bar_ts}] hold pos side={side} src={position['source_component']} "
              f"unreal={unrealized_move*100:.3f}% mfe={position['mfe']*100:.2f}% mae={position['mae']*100:.2f}% "
              f"exit_prob={exit_prob:.3f} reason={reason}", flush=True)
        if should_exit:
            state["pending"] = {"kind": "exit", "signal_bar_ts": bar_ts.isoformat(), "reason": reason}
    elif position is None and state.get("pending") is None:
        decision = adapter.decide_entry(frame)
        if decision is not None:
            state["pending"] = {
                "kind": "enter", "signal_bar_ts": bar_ts.isoformat(), "side": decision.side,
                "source_component": decision.source_component, "margin_fraction": decision.margin_fraction,
                "leverage": decision.leverage, "notional_exposure": decision.notional_exposure,
                "take_profit": decision.take_profit, "stop_loss": decision.stop_loss,
            }
            print(f"[signal {bar_ts}] ENTER queued side={decision.side} src={decision.source_component} "
                  f"notional={decision.notional_exposure:.3f} tp={decision.take_profit:.4f} sl={decision.stop_loss:.4f}", flush=True)

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
    })
    return state


def run(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("[init] loading JM-swapped Omega4.6.1 adapter (h48qual q070 / zig075 q080, CORRECTGATE "
          "sidecars -- shadow-only due to n=11 significance + dataset-lineage gaps, not a sizing "
          "problem, see module docstring)", flush=True)
    adapter = build_jm_adapter(device=args.device)
    fee, slip = _load_fee_slip()

    state = load_json(STATE_PATH, {})
    if not state:
        buffer, offset = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        state = {
            "schema_version": "eth_jmlam4_regime_swap_shadow.v1", "live_forward_only": True,
            "order_submission_supported": False, "activation_allowed": False,
            "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "candidate": "ETH regime3 HMM->JM(lambda=4) swap, h48qual q070 + zig075 q080, "
                          "pinned102 base_cols, CORRECTGATE sidecars (real MDD floor -25%)",
            "closed_line_memory": "project-eth-regime3-jm-lam4-swap-retrain-20260809.md",
            "started_at_kst": now_kst().isoformat(), "snapshot_offset": offset,
            "position": None, "pending": None, "last_processed_bar_ts": None,
            "equity": 1.0, "peak_equity": 1.0, "mdd": 0.0,
        }
        print(f"[init] seeded buffer with {len(buffer)} rows, snapshot_offset={offset}", flush=True)
    else:
        buffer, _ = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        print(f"[resume] loaded prior state, last_processed_bar_ts={state.get('last_processed_bar_ts')}, "
              f"equity={state.get('equity')}", flush=True)

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

        if len(buffer) >= 250:  # enough history for atr_window=192 + rolling feature warm-up
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
                    # adapter.decide_entry/evaluate_exit read regime3_current_sensitive_wide24_*
                    # columns directly off frame -- they don't compute them internally, the caller
                    # must inject them first (same contract the BTC multislot shadow loop follows:
                    # processed = adapter.regime3_current.append(processed)). Missing this call meant
                    # every decision hit "non-finite input features in history window".
                    frame = adapter.regime3_current.append(frame)
                    # seed_buffer() re-reads up to BUFFER_ROWS from decision_feature_snapshot.jsonl
                    # on every resume, which can reach back far enough to catch old gaps in that
                    # monitoring log (e.g. a stretch of incomplete rows from an earlier snapshot-
                    # writer incident, unrelated to the current bar). entry_decision/evaluate_exit
                    # require the ENTIRE history window to be finite, so a single old NaN cell deep
                    # in the buffer would otherwise block every future decision until it eventually
                    # ages out. Backfill/forward-fill only the historical rows (never the latest,
                    # current bar) so an old gap can't permanently wedge the shadow -- if the latest
                    # row itself is bad, this leaves it untouched and the existing finite-input
                    # guards still fire.
                    if len(frame) > 1:
                        _numeric_cols = frame.select_dtypes(include=[np.number]).columns
                        _hist_idx = frame.index[:-1]
                        frame.loc[_hist_idx, _numeric_cols] = (
                            frame.loc[_hist_idx, _numeric_cols].ffill().bfill()
                        )
                    state = try_fill_pending(state, frame, fee, slip)
                    state = process_bar(state, frame, adapter, fee, slip)
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
