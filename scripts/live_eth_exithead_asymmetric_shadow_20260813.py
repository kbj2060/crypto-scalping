#!/usr/bin/env python3
"""LIVE shadow test of the ETH Omega4.6.1 asymmetric exit-head-swap candidate against the REAL live
Omega4.6.1 ETH position. NO real orders are placed -- this only tracks a hypothetical Omega4.6.1
position built with the SAME live h48qual/zig075 TabM parents, risk sidecars, and HMM regime3
classifier, except h48qual's exit_head is swapped for the 2026-08-13 overnight "live ATR barrier
relabel" retrain (direction_head/quality_head/encoder frozen -- only exit_head differs) while
zig075 stays fully original. See
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md (follow-up 1-3) for why
this is the ONLY candidate from tonight's overnight exit-head-redesign track that survived both
portfolio-level VAL and a single confirmatory OOS check without a sign flip (VAL PnL +36.82% ->
+46.59%, MDD -24.34% -> -21.70%; OOS PnL +49.32% -> +93.27%, MDD -16.20% -> -15.48% -- see that
document's "OOS absolute numbers are contaminated by quality_threshold selection bias, relative
comparison still valid" caveat before quoting the OOS numbers standalone). Per this sub-project's
standing convention (see the multislot-capacity contracts' "G5 outcome ceiling"), the next step for
even the best overnight result is a shadow build, not a live cutover -- this script is that shadow.

Built by directly reusing already-live, already-validated pieces rather than reimplementing
anything, following the exact structural pattern of
scripts/live_eth_jmlam4_regime_swap_shadow_20260809.py:
  - entry/exit decision + sidecar sizing: Omega461LiveAdapter, imported verbatim from
    trading_bot_modules.omega4_6_1_live -- the SAME class the real live ETH executor runs every
    cycle. Only h48qual's bundle/sidecar are swapped in via components_override; zig075 is passed
    through unchanged (still required in components_override because passing components_override
    to Omega461LiveAdapter.__init__ replaces BOTH default components wholesale -- there is no
    partial-override mode, confirmed by reading trading_bot_modules/omega4_6_1_live.py). The
    adapter class itself is untouched.
  - regime3 classifier: UNCHANGED from the JM shadow bot's swap -- this candidate never touched
    regime3, so Omega461LiveAdapter's own default `self.regime3_current =
    _Regime3CurrentLiveFeatures(...)` (the real live 12-state sticky HMM, from
    trading_bot_modules.omega4_6_2_source_parent_live) is used as-is. No post-construction
    override, unlike the JM shadow bot.
  - raw market data: tails data/live/decision_feature_snapshot.jsonl, the same live 5m feature
    stream trading_bot.py's real live loop writes.
  - real live ETH position for reference only: data/live/dashboard_state.json's `position` block
    (omega461_eth_position(), same pattern as the JM shadow bot).

Artifact-lineage note (read before touching COMPONENTS_OVERRIDE): Omega461LiveAdapter's
_Component.__init__ unconditionally calls
trading_bot_modules.omega4_6_1_runtime_contract.validate_sidecar_lineage, which hard-requires the
sidecar's own report.json (`risk_model.precomputed_prediction_dir`) to point at the EXACT directory
the paired bundle lives in, plus train/validation/oos_predictions_q050.csv present in that same
directory. Pairing the new exit-head bundle directly with the untouched original h48qual sidecar
path therefore raises "invalid artifact lineage" on ANY machine (verified 2026-08-13) -- this is
not a relaxable default, and editing trading_bot_modules/omega4_6_1_live.py or
omega4_6_1_runtime_contract.py to bypass it is out of scope (live files must not change). Instead:
  1. tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/
     gained 3 new sibling files: byte-for-byte copies of the original live h48qual bundle
     directory's train/validation/oos_predictions_q050.csv. This is valid (not fabricated data)
     because a direct torch.load comparison (2026-08-13) confirmed the new bundle's encoder/
     direction_head/quality_head state_dict tensors are bit-identical to the original live h48qual
     bundle for all three regime experts (bull/bear/chop) -- only exit_head.weight/exit_head.bias
     differ -- and these CSVs contain only router/direction_head/quality_head columns (verified via
     header inspection), no exit_head content.
  2. A new directory,
     tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar/,
     holds risk_sidecar.pkl (an unmodified byte-for-byte copy of the real live h48qual sidecar,
     md5 verified identical -- the risk-sizing model itself is 100% unchanged) plus a report.json
     copied from the original with exactly one field rewritten: risk_model.precomputed_prediction_dir
     now points (as a repo-relative path, resolved dynamically via Omega461LiveAdapter's own ROOT,
     so it works on any machine) at the new bundle directory in (1) instead of the original.
Full rationale, verification commands, and output: docs/experiments/
eth_omega461_exit_head_asymmetric_shadow_20260813.md.

Entries and exits both queue for the NEXT bar's open, matching omega4_6_1_live.py's own documented
execution-delay model (its evaluate_exit docstring: "the fill itself still executes at the next
bar's open"). This is causal, forward-only, and reads no ledger or future row -- Fresh-Forward-
compliant live-forward observation, not a backtest.

RESEARCH/SHADOW ONLY -- order_submission_supported=False, activation_allowed=False, matching every
other shadow bot in this repo. Must run where data/live/decision_feature_snapshot.jsonl and
data/live/dashboard_state.json are actively updated by the real trading_bot.py loop (the live
trading server, reached via scripts/ops/handoff.sh) -- confirmed 2026-08-13 that this dev checkout's
copies of those files are multi-day-stale, and separately that the real live h48qual/zig075
sidecars' report.json files hardcode absolute /home/llewyn/crypto-scalping/... paths that only
resolve on the server, so even zig075's fully-original components_override entry fails artifact-
lineage validation on this dev machine. See the experiment doc's "실행 위치" section.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _load_fee_slip  # noqa: E402
from trading_bot_modules.omega4_6_1_live import (  # noqa: E402
    DURATION_THRESHOLD as OMEGA4_6_1_LIVE_DURATION_THRESHOLD,
    Omega461LiveAdapter,
)
from trading_bot_modules.runtime_config import (  # noqa: E402
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
)

KST = ZoneInfo("Asia/Seoul")
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.jsonl"
DASHBOARD_STATE_PATH = ROOT / "data/live/dashboard_state.json"
OUT_DIR = ROOT / "data/live/eth_exithead_asymmetric_shadow"
STATE_PATH = OUT_DIR / "state.json"
TRADES_PATH = OUT_DIR / "closed_trades.jsonl"
EQUITY_PATH = OUT_DIR / "equity_curve.jsonl"

# h48qual: exit_head swapped for the 2026-08-13 "live ATR barrier relabel" retrain (direction_head/
# quality_head/encoder frozen -- verified bit-identical to the live bundle via torch.load, see module
# docstring). Sidecar is a byte-identical copy of the real live h48qual risk_sidecar.pkl, repointed
# only at the lineage-metadata layer (see module docstring point 2) -- same quality_threshold=0.50 as
# live, unchanged.
H48QUAL_NEW_BUNDLE_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
    "/h48qual/true_3head_tabm_bundle.pt"
)
H48QUAL_SHIM_SIDECAR_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar"
    "/risk_sidecar.pkl"
)
# zig075: fully original live artifacts, included explicitly (not omitted) so this override block
# alone documents exactly what changed vs. real live -- only h48qual's bundle/sidecar above differ.
COMPONENTS_OVERRIDE = {
    "h48qual": {
        "bundle": H48QUAL_NEW_BUNDLE_PATH,
        "sidecar": H48QUAL_SHIM_SIDECAR_PATH,
        "quality_threshold": 0.50,
    },
    "zig075": {
        "bundle": ROOT / FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
        "sidecar": ROOT / FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
        "quality_threshold": 0.75,
    },
}
PRIORITY = ("h48qual", "zig075")  # same order as real live
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


def build_adapter(device: str = "cpu") -> Omega461LiveAdapter:
    return Omega461LiveAdapter(
        h48qual_bundle="", h48qual_sidecar="", zig075_bundle="", zig075_sidecar="",  # unused, components_override wins
        device=device,
        components_override=COMPONENTS_OVERRIDE,
        priority=PRIORITY,
        # duration_threshold/scale_map/base_template/expert_scales all omitted deliberately -- this
        # lets Omega461LiveAdapter apply its own live defaults (DURATION_THRESHOLD=0.005417, real
        # ETH SCALE_MAP/BASE_TEMPLATE/EXPERT_SCALES), matching this repo's currently-live ETH .env
        # (no FINAL_GOVERNOR_OMEGA4_6_1_ETH_* overrides set as of 2026-08-13) -- nothing in tonight's
        # exit-head experiment touched these, so the shadow must not touch them either.
    )


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
    print("[init] loading asymmetric exit-head-swap Omega4.6.1 adapter (h48qual q050 = new liveATR "
          "relabel exit_head, zig075 q075 = fully original live, regime3 = original live HMM) -- "
          "research/shadow only, the only overnight exit-head candidate that survived VAL+OOS without "
          "a sign flip, see module docstring", flush=True)
    adapter = build_adapter(device=args.device)
    print(f"[init] duration_threshold={adapter.duration_threshold} (live default "
          f"{OMEGA4_6_1_LIVE_DURATION_THRESHOLD}, unmodified)", flush=True)
    fee, slip = _load_fee_slip()

    state = load_json(STATE_PATH, {})
    if not state:
        buffer, offset = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        state = {
            "schema_version": "eth_exithead_asymmetric_shadow.v1", "live_forward_only": True,
            "order_submission_supported": False, "activation_allowed": False,
            "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "candidate": "ETH Omega4.6.1 asymmetric exit-head swap: h48qual=2026-08-13 liveATR-"
                          "relabel exit_head retrain (direction/quality/encoder frozen, bit-"
                          "identical to live), zig075=fully original live, regime3=original live "
                          "HMM (unchanged)",
            "closed_line_doc": "docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md",
            "shadow_doc": "docs/experiments/eth_omega461_exit_head_asymmetric_shadow_20260813.md",
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
                    # columns directly off frame and also recompute+overwrite them internally via
                    # Omega461LiveAdapter._with_regime3 -- but this explicit pre-append is still
                    # required so the ffill/bfill NaN-backfill step just below (which runs on
                    # whatever numeric columns already exist in `frame`) can also reach and patch
                    # any old gaps in the regime3 columns themselves, not just the raw OHLCV/feature
                    # columns. Confirmed load-bearing in the JM shadow bot this script is templated
                    # from (scripts/live_eth_jmlam4_regime_swap_shadow_20260809.py): "Missing this
                    # call meant every decision hit 'non-finite input features in history window'."
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
