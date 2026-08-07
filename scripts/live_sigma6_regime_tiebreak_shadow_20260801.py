#!/usr/bin/env python3
"""LIVE shadow test of the Sigma6-filtered + Omega4.6.1 "regime_tiebreak" two-leg ETH portfolio
(see project-sigma6-omega461-regime-tiebreak-conflict-rule-20260801.md). NO real orders are placed
-- this only tracks a hypothetical Sigma6-filtered position and a hypothetical regime_tiebreak-
weighted combination against the REAL live Omega4.6.1 ETH position, purely for observation.

This is the first time Sigma3-1h/Sigma6 has ever run live (every prior result in this project came
from an offline batch tape, see train_sigma3_1h_hgb_persist_20260801.py's docstring) -- built by
directly reusing already-live, already-validated pieces rather than reimplementing anything:
  - feature engineering: resample_1h()/compute_features(), imported verbatim from
    build_1h_trendscan_dataset_20260705.py (the exact functions that built Sigma3-1h's training data)
  - regime probabilities: Regime3CurrentLiveFeatures, imported from
    trading_bot_modules.omega4_6_2_source_parent_live -- the SAME class and SAME model artifact
    (regime3_current_hmm_sensitive_balancedish_20260530/..._wide24_2024.joblib) Sigma6's offline
    research already used (confirmed by path equality), and the same class the real live Omega4.6.1
    adapter runs every cycle -- not a reimplementation.
  - raw market data: tails data/live/decision_feature_snapshot.jsonl, the same live 5m feature
    stream trading_bot.py's real live loop writes (same source scripts/run_omega5_live_only_shadow_
    loop_20260702.py already reads for its own shadow tracking).
  - Omega4.6.1's real live ETH position: read from data/live/dashboard_state.json's top-level
    `position` block (ETH IS Omega4.6.1's real executing strategy, unlike SOL/BTC which have a
    separate omega4_6_1_shadow_<asset>_state.json because a different strategy executes there).

Position/exit mechanics replicate sigma6_filtered_trades() from
research_eth_sigma6_walkforward_omega461_joint_portfolio_20260801.py bar-for-bar (WINNER config:
thr=0.60, lev=3.0, sl_atr=1.5, reg_mode=not_chop, reg_thr=0.50, stab_thr=0.0 i.e. stability gate
off; BASE_KW: margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144h, cooldown=3h), except
timestamp-keyed instead of index-keyed (robust to bot downtime/gaps, unlike a sequential loop).
Entries execute at the NEXT hour's open after the signal bar, matching the offline convention.

regime_tiebreak weighting (the validated rule): every completed hour, if a conflict exists (our
Sigma6-filtered position is open AND opposite side to Omega4.6.1's real live position), the bar's
PnL contribution to the COMBINED-portfolio-weighted equity is zeroed unless our side agrees with the
regime (bull_prob >= bear_prob for LONG, bear_prob > bull_prob for SHORT); otherwise weight=1. The
underlying Sigma6-filtered position itself is untouched by this -- weighting is an allocation-layer
decision on top of it, exactly as validated in eval_sigma6_omega_rule_and_meta_allocation_20260801.py.

Both a RAW (unweighted, standalone Sigma6-filtered leg) and a WEIGHTED (regime_tiebreak-applied)
equity curve are tracked so the tiebreak's own effect stays visible. RESEARCH/SHADOW ONLY -- no
order_submission_supported, no activation_allowed, matching every other shadow bot in this repo.
Per CLAUDE.md Fresh-Forward rule this IS genuinely causal live-forward data (unlike the VAL/OOS
backtest windows), but is still just one new shadow bot's first run -- not a promotion claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

from build_1h_trendscan_dataset_20260705 import resample_1h, compute_features  # noqa: E402
from trading_bot_modules.omega4_6_2_source_parent_live import (  # noqa: E402
    Regime3CurrentLiveFeatures, DEFAULT_CURRENT_REGIME_PATH, CURRENT_PREFIX,
)

KST = ZoneInfo("Asia/Seoul")
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.jsonl"
DASHBOARD_STATE_PATH = ROOT / "data/live/dashboard_state.json"
MODEL_DIR = ROOT / "data/ensemble/supervised/sigma3_1h_hgb_live_20260801"
OUT_DIR = ROOT / "data/live/sigma6_regime_tiebreak_shadow"
STATE_PATH = OUT_DIR / "state.json"
TRADES_PATH = OUT_DIR / "closed_trades.jsonl"
EQUITY_PATH = OUT_DIR / "equity_curve.jsonl"

# WINNER config from project-sigma6-regime-filter-leave-one-window-out-CANDIDATE-20260801.md
QUALITY_THR = 0.60
REG_THR = 0.50
LEVERAGE, MARGIN = 3.0, 0.30
NOTIONAL = MARGIN * LEVERAGE
SL_ATR, TRAIL_ATR, MIN_PROFIT_ATR = 1.5, 5.0, 2.0
MAX_HOLD_HOURS, COOLDOWN_HOURS = 144, 3
FEE, SLIP = 0.00020, 0.00050
DEFAULT_DIR_THR = 0.45  # Sigma3-1h's own tape-quality gate before Sigma6's stricter 0.60 re-gate

BUFFER_DAYS = 10  # rolling 5m history kept in memory; comfortably covers the longest 48h rolling window


def now_kst() -> datetime:
    return datetime.now(tz=KST)


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


def seed_buffer(path: Path, days: int) -> tuple[pd.DataFrame, int]:
    """Read the tail of the (huge, continuously-appended) snapshot file to seed the rolling 5m
    buffer without loading the whole multi-hundred-MB file."""
    approx_bytes = days * 24 * 12 * 16000  # ~14-15KB/line observed (293+ value keys per row), margin
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - approx_bytes))
        f.readline()  # drop partial first line
        data = f.read().decode("utf-8", errors="ignore")
    rows = []
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
            rows.append(values)
    df = pd.DataFrame(rows)
    return df, size


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
class Position:
    side: int
    entry_price: float
    entry_atr: float
    hold_start_ts: str  # signal bar ts (matches sigma6_filtered_trades hold_start indexing)
    entry_ts: str
    peak_unreal: float = 0.0


@dataclass
class PendingEntry:
    side: int
    signal_bar_ts: str
    entry_atr: float


def build_regime(buffer: pd.DataFrame, regime_model: Regime3CurrentLiveFeatures) -> pd.DataFrame:
    frame = buffer.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out = regime_model.append(frame)
    return out[["timestamp", f"{CURRENT_PREFIX}bull_prob", f"{CURRENT_PREFIX}bear_prob",
                f"{CURRENT_PREFIX}chop_prob"]]


def regime_at_or_before(regime_df: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float, float] | None:
    sub = regime_df[regime_df["timestamp"] <= ts]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return (float(row[f"{CURRENT_PREFIX}bull_prob"]), float(row[f"{CURRENT_PREFIX}bear_prob"]),
            float(row[f"{CURRENT_PREFIX}chop_prob"]))


def sigma3_signal(feat_row: pd.Series, model, feature_cols: list[str]) -> tuple[int, float]:
    """Returns (side, atr_pct) after Sigma3-1h's own 0.45 gate AND Sigma6's stricter 0.60 re-gate
    (apply_quality_threshold equivalent -- dir probs and quality probs are identical in this
    pipeline, see replay_omega6_v2_variants_20260704.apply_quality_threshold)."""
    x = feat_row[feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    cls = list(model.classes_)
    col_for = {c: i for i, c in enumerate(cls)}
    pc = proba[col_for.get(0, -1)] if 0 in col_for else 0.0
    pl = proba[col_for.get(1, -1)] if 1 in col_for else 0.0
    ps = proba[col_for.get(2, -1)] if 2 in col_for else 0.0
    probs = np.array([pc, pl, ps])
    dir_action = int(probs.argmax())
    qual = probs[dir_action] if dir_action > 0 else pc
    final_action = dir_action if (dir_action != 0 and qual >= QUALITY_THR) else 0
    side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)
    return side, float(feat_row["atr_pct"])


def omega461_eth_position() -> tuple[int, str | None]:
    """Returns (side, entry_price_str) for the REAL live Omega4.6.1 ETH position (ETH has no
    separate shadow-state file because Omega4.6.1 IS the real executing strategy there)."""
    state = load_json(DASHBOARD_STATE_PATH, {})
    pos = state.get("position") or {}
    current = str(pos.get("current", "")).upper()
    side = 1 if current == "LONG" else (-1 if current == "SHORT" else 0)
    return side, pos.get("opened_at")


def try_fill_pending_entry(state: dict, r: pd.DataFrame) -> dict:
    """Fill a queued pending entry as soon as its target bar's OPEN is observable -- i.e. once that
    hour's first 5m sub-bar has closed (~5min after the hour starts), NOT once the whole target hour
    is complete. Matches the offline backtest's own entry_price=open[i+1] convention exactly (same
    price), just without the extra ~55min of unnecessary latency that came from previously requiring
    the target bar to also be fully complete before acting on already-known information (see
    project-tau1-name-spec-20260801.md follow-up: 'signal_close_next_open' contract, fill on the
    entry bar's own open, not on the bar after it)."""
    pending = state.get("pending_entry")
    if pending is None or state.get("position") is not None:
        return state
    target_ts = pd.Timestamp(pending["target_bar_ts"])
    open_lookup = r.set_index("timestamp")["open"]
    if target_ts not in open_lookup.index:
        return state  # target hour's own open hasn't printed yet
    open_px = float(open_lookup.loc[target_ts])
    side = int(pending["side"])
    entry_price = open_px * (1 + SLIP if side > 0 else 1 - SLIP)
    state["position"] = {
        "side": side, "entry_price": entry_price, "entry_atr": float(pending["entry_atr"]),
        "hold_start_ts": pending["signal_bar_ts"], "entry_ts": target_ts.isoformat(), "peak_unreal": 0.0,
    }
    state["pending_entry"] = None
    print(f"[fill] pending entry side={side} filled at open={open_px:.4f} "
          f"(target_bar={target_ts}, signal_bar={pending['signal_bar_ts']})", flush=True)
    return state


def process_bar(state: dict, bar_ts: pd.Timestamp, feat_row: pd.Series, regime_df: pd.DataFrame,
                 model, feature_cols: list[str], is_catchup: bool = False) -> dict:
    """Advance the state machine by exactly one newly-completed 1h bar (signal generation + exit
    checks -- both require the FULL bar, unlike entry fills which try_fill_pending_entry() handles
    separately and earlier, as soon as just the open is known).

    is_catchup=True means this bar is being processed late (bot downtime/backlog), NOT right after
    it completed. omega461_eth_position() only ever exposes Omega4.6.1's CURRENT position -- there is
    no historical position log in dashboard_state.json -- so applying the regime_tiebreak conflict
    rule to a backlog bar would score it against a position that didn't exist yet at that bar's own
    time (future information leaking into a past decision). Fixed 2026-08-07: catch-up bars fall back
    to weight=1.0 (no tiebreak penalty applied) and are flagged via tiebreak_stale in the equity row."""
    bar_iso = bar_ts.isoformat()
    reg = regime_at_or_before(regime_df, bar_ts)
    bull, bear, chop = reg if reg is not None else (0.0, 0.0, 1.0)
    close_px = float(feat_row["close"])

    position = state.get("position")
    cooldown_until = state.get("cooldown_until_ts")

    weight = 1.0
    tiebreak_stale = False
    closed_trade = None

    # if a position is open, mark-to-market on this bar's close and check exit conditions
    if position is not None:
        side = int(position["side"])
        entry_price = float(position["entry_price"])
        entry_atr = max(float(position["entry_atr"]), 1e-9)
        raw = (close_px * (1 - SLIP) - entry_price) / entry_price if side > 0 else \
              (entry_price - close_px * (1 + SLIP)) / entry_price
        unreal = raw * NOTIONAL
        peak_unreal = max(float(position.get("peak_unreal", 0.0)), unreal)
        position["peak_unreal"] = peak_unreal
        hold_start = pd.Timestamp(position["hold_start_ts"])
        hold_hours = (bar_ts - hold_start).total_seconds() / 3600.0

        reason = ""
        if unreal <= -SL_ATR * entry_atr:
            reason = "stop"
        elif peak_unreal >= MIN_PROFIT_ATR * entry_atr and (peak_unreal - unreal) >= TRAIL_ATR * entry_atr:
            reason = "trail"
        elif hold_hours >= MAX_HOLD_HOURS:
            reason = "time"

        # bar delta for equity tracking: this bar's move only (prev bar's unreal -> this bar's unreal
        # or realized pnl at close), computed against the position's equity fraction (NOTIONAL-scaled)
        prev_unreal = float(position.get("_prev_unreal", 0.0))
        bar_delta = (unreal - prev_unreal) if reason == "" else (
            (unreal - FEE * NOTIONAL) - prev_unreal  # realized this bar: unrealized move + exit fee
        )
        position["_prev_unreal"] = unreal if reason == "" else 0.0

        omega_side, _ = omega461_eth_position()
        conflict = (omega_side != 0) and (omega_side == -side)
        if conflict:
            if is_catchup:
                tiebreak_stale = True  # no historical Omega position available for this backlog bar
            else:
                regime_side = 1 if bull >= bear else -1
                weight = 1.0 if side == regime_side else 0.0

        state["raw_equity"] = float(state.get("raw_equity", 1.0)) + bar_delta
        state["weighted_equity"] = float(state.get("weighted_equity", 1.0)) + weight * bar_delta

        if reason:
            exit_price = close_px * (1 - SLIP if side > 0 else 1 + SLIP)
            closed_trade = {
                "entry_ts": position["entry_ts"], "exit_ts": bar_iso, "side": side,
                "entry_price": entry_price, "exit_price": exit_price, "reason": reason,
                "notional": NOTIONAL, "realized_pnl_frac": unreal - FEE * NOTIONAL,
                "hold_hours": round(hold_hours, 2),
            }
            state["position"] = None
            state["cooldown_until_ts"] = (bar_ts + timedelta(hours=COOLDOWN_HOURS)).isoformat()
    else:
        state["raw_equity"] = float(state.get("raw_equity", 1.0))
        state["weighted_equity"] = float(state.get("weighted_equity", 1.0))

    # 3) if flat (and not pending), evaluate this bar's entry signal for the NEXT bar to execute
    if state.get("position") is None and state.get("pending_entry") is None:
        in_cooldown = cooldown_until is not None and bar_ts < pd.Timestamp(cooldown_until)
        if not in_cooldown:
            gated_ok = chop < REG_THR  # not_chop regime filter, WINNER config
            side, atr_pct = sigma3_signal(feat_row, model, feature_cols)
            if gated_ok and side != 0:
                target_bar_ts = (bar_ts + timedelta(hours=1)).isoformat()
                state["pending_entry"] = {"side": side, "signal_bar_ts": bar_iso, "entry_atr": atr_pct,
                                           "target_bar_ts": target_bar_ts}

    for key in ("raw_equity", "weighted_equity"):
        peak_key, mdd_key = f"peak_{key}", f"mdd_{key}"
        peak = max(float(state.get(peak_key, 1.0)), float(state[key]))
        state[peak_key] = peak
        state[mdd_key] = min(float(state.get(mdd_key, 0.0)), float(state[key]) / peak - 1.0)

    state["last_processed_bar_ts"] = bar_iso
    state["last_decision_bar_ts"] = bar_iso
    state["last_decision_at_kst"] = now_kst().isoformat()
    equity_row = {
        "bar_ts": bar_iso, "raw_equity": state["raw_equity"], "weighted_equity": state["weighted_equity"],
        "position_side": (position or {}).get("side", 0), "bull_prob": bull, "bear_prob": bear,
        "chop_prob": chop, "weight_applied": weight, "is_catchup": is_catchup, "tiebreak_stale": tiebreak_stale,
    }
    append_jsonl(EQUITY_PATH, equity_row)
    if closed_trade is not None:
        append_jsonl(TRADES_PATH, closed_trade)
    return state


def run(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    model = joblib.load(MODEL_DIR / "model.joblib")
    meta = json.loads((MODEL_DIR / "feature_cols.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    regime_model = Regime3CurrentLiveFeatures(current_path=DEFAULT_CURRENT_REGIME_PATH)

    state = load_json(STATE_PATH, {})
    if not state:
        buffer, offset = seed_buffer(SNAPSHOT_PATH, BUFFER_DAYS)
        state = {
            "schema_version": "sigma6_regime_tiebreak_shadow.v1", "live_forward_only": True,
            "order_submission_supported": False, "activation_allowed": False,
            "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
            "started_at_kst": now_kst().isoformat(), "snapshot_offset": offset,
            "position": None, "pending_entry": None, "cooldown_until_ts": None,
            "last_processed_bar_ts": None, "raw_equity": 1.0, "weighted_equity": 1.0,
            "peak_raw_equity": 1.0, "peak_weighted_equity": 1.0, "mdd_raw_equity": 0.0,
            "mdd_weighted_equity": 0.0,
        }
        print(f"[init] seeded buffer with {len(buffer)} rows, snapshot_offset={offset}", flush=True)
    else:
        buffer, _ = seed_buffer(SNAPSHOT_PATH, BUFFER_DAYS)
        print(f"[resume] loaded prior state, last_processed_bar_ts={state.get('last_processed_bar_ts')}", flush=True)

    offset = int(state["snapshot_offset"])
    cutoff = pd.Timestamp.now(tz=timezone.utc).tz_localize(None) - pd.Timedelta(days=BUFFER_DAYS)

    end_at = pd.Timestamp(args.end_at_kst) if args.end_at_kst else None
    while end_at is None or now_kst() < end_at:
        offset, new_rows = read_new_rows(SNAPSHOT_PATH, offset)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            buffer = pd.concat([buffer, new_df], ignore_index=True)
            buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
            buffer = buffer.sort_values("timestamp").drop_duplicates("timestamp")
            buffer = buffer[buffer["timestamp"] >= cutoff].reset_index(drop=True)
        state["snapshot_offset"] = offset

        if len(buffer) >= 300:  # enough history for the rolling feature windows
            r = resample_1h(buffer)
            state = try_fill_pending_entry(state, r)
            feats = compute_features(r)
            last_bar_time = pd.to_datetime(buffer["timestamp"]).max()
            # a 1h bar timestamped bar_ts is "complete" once we have data through bar_ts+55min
            complete_mask = feats["timestamp"] + pd.Timedelta(minutes=55) <= last_bar_time
            complete = feats[complete_mask]
            last_processed = state.get("last_processed_bar_ts")
            last_processed_ts = pd.Timestamp(last_processed) if last_processed else None
            new_bars = complete[complete["timestamp"] > last_processed_ts] if last_processed_ts is not None else complete.tail(1)
            if len(new_bars) > 0:
                regime_df = build_regime(buffer, regime_model)
                n_new = len(new_bars)
                if n_new > 1:
                    print(f"[catchup] {n_new} backlog bars to process -- all but the last will run with "
                          f"tiebreak weighting disabled (no historical Omega position available)", flush=True)
                for i, (_, feat_row) in enumerate(new_bars.iterrows()):
                    bar_ts = feat_row["timestamp"]
                    if feat_row[feature_cols].isna().any():
                        print(f"[skip] bar {bar_ts} has NaN features (insufficient warm-up history)", flush=True)
                        state["last_processed_bar_ts"] = bar_ts.isoformat()
                        continue
                    is_catchup = i < n_new - 1
                    state = process_bar(state, bar_ts, feat_row, regime_df, model, feature_cols, is_catchup=is_catchup)
                    print(f"[bar {bar_ts}] pos={state.get('position')} pending={state.get('pending_entry')} "
                          f"raw_eq={state['raw_equity']:.4f} weighted_eq={state['weighted_equity']:.4f}"
                          f"{' [catchup]' if is_catchup else ''}", flush=True)

        write_json(STATE_PATH, state)
        time.sleep(max(30.0, float(args.poll_seconds)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=float, default=90.0)
    ap.add_argument("--end-at-kst", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
