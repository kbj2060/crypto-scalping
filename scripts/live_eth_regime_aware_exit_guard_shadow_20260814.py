#!/usr/bin/env python3
"""LIVE shadow test of the ETH Omega4.6.1 regime-aware exit-head guard candidate (Odyssey2 #11)
against the REAL live Omega4.6.1 ETH position. NO real orders are placed -- this only tracks a
hypothetical Omega4.6.1 position, built on top of the ALREADY-DEPLOYED shadow baseline
(scripts/live_eth_exithead_asymmetric_shadow_20260813.py: h48qual exit_head swapped for the
2026-08-13 liveATR-relabel retrain, zig075 fully original), with ONE further conditional change:
while a HELD h48qual position is evaluated for exit, a causal "sustained uptrend" detector decides,
bar by bar, whether that check reads h48qual's ORIGINAL frozen exit head (pre-liveATR-relabel) or
the liveATR-relabel exit head (the baseline shadow's default). See
docs/experiments/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.md for the full backtest
this replicates live, and docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_
vulnerability_20260814.md for the failure mode this candidate partially mitigates (the liveATR
relabel is a "turnover accelerator" that inflates h48qual's trade count and worsens PnL specifically
in sustained low-noise uptrends, e.g. 2025 Q3 no_gate (original->liveATR-relabel, single consistent
gate setting -- verified directly against tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_
head_uptrend_guard_20260814/report.json's comparison["2025q3"], NOT the vulnerability doc's own
headline table, which mixes with_gate-original against no_gate-liveATR trade counts):
PnL -35.54%->-46.26%, MDD -49.79%->-56.94%, trades 25->38).

That backtest's own verdict is explicitly NOT "promotable" -- it is "worth adding as a shadow
observation candidate": Q1/Q2/VAL/OOS-Q1/OOS-Q2 show zero side effects (most windows produce a
byte-identical ledger to the baseline shadow), and 2025 Q3 (in-sample) shows partial mitigation
(no_gate 82.4% / with_gate 32.9% of the original<->liveATR gap recovered), but NO forward OOS window
has yet contained a genuine sustained-uptrend regime, so the live-forward behavior of this exact
policy is completely unobserved. This script exists to observe it, not to promote it.

=== Mechanism (byte-for-byte the same rule as the backtest's greedy_replay_regime_aware_exit_guard
guard branch, see that function's docstring) ===
Per bar, while h48qual holds an open position:
  - detector ACTIVE   -> exit-head probability read from `GUARD_COMPONENT` (h48qual's ORIGINAL,
    pre-liveATR-relabel exit head -- bundle/sidecar = the exact paths trading_bot.py's REAL live
    adapter uses by default, i.e. FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_{BUNDLE,SIDECAR}_PATH).
  - detector INACTIVE (including the whole WEEK_BARS warm-up period) -> exit-head probability read
    from `adapter.components["h48qual"]` (the liveATR-relabel exit head -- byte-identical to the
    ALREADY-DEPLOYED live_eth_exithead_asymmetric_shadow_20260813.py baseline).
EXIT_THRESHOLD (0.95) and the TP/SL barriers ahead of it are identical in both branches -- confirmed
2026-08-13 (see live_eth_exithead_asymmetric_shadow_20260813.py's own docstring) that the two h48qual
bundles' encoder/direction_head/quality_head weights and 102-column base_cols are bit-identical;
independently re-confirmed here via a fresh torch.load diff before writing this script (only
exit_head.weight/exit_head.bias differ, for all three bull/bear/chop experts). zig075 and h48qual's
own direction/quality/entry/sizing are NEVER touched by the detector -- entry/sizing always come from
`adapter.decide_entry`, exactly as in the baseline shadow, regardless of detector state.

Detector: rolling WEEK_BARS(=2016, i.e. 1 week of 5m bars -- reuses dual_momentum's OWN existing
close.shift(2016) lookback, not an invented window)-bar mean of (dual_momentum>0), active when that
score exceeds DETECTOR_THRESHOLD (90th percentile of the 2025-Q1+Q2-ONLY calibration sample computed
by research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector() --
DETECTOR_THRESHOLD below is the exact float copied verbatim from that already-executed run's
tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814/report.json
["detector"]["threshold_used"], NOT re-derived here). `dual_momentum` (features/engineering.py
_dual_momentum) is already a live-computed column in every data/live/decision_feature_snapshot.jsonl
row (trading_bot.py's own FeatureEngineer pipeline computes it every cycle) -- this script reads it
off the same buffer the other two shadows already tail, it does not recompute it from raw OHLCV.

This script deliberately does NOT import the research script's build_detector() / re-read its
report.json / re-read the 234MB+125MB 2025/2026 base CSVs at live startup -- that would add a heavy,
fragile runtime dependency (large gitignored data files that may not be present/synced on the live
trading server) to an always-on production-adjacent process just to re-derive one already-computed,
already-validated scalar. WEEK_BARS/DETECTOR_THRESHOLD are hardcoded constants instead, with their
provenance cited above -- this is "reuse the number", not "reuse the computation", per the task's
explicit instruction not to recalibrate.

SustainedUptrendDetector below is the LIVE, O(1)-per-bar, incremental reimplementation of that
backtest's _rolling_dual_momentum_score (a full-CSV vectorized pandas .rolling(2016,
min_periods=2016).mean()) -- self-verified to match the batch computation exactly (same threshold
crossings, same NaN/warm-up bars) by
scripts/verify_eth_regime_aware_exit_guard_shadow_detector_20260814.py (run offline, BEFORE this
script was deployed; see that script's own output / the deployment log entry in
docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md for the result). When the
detector is inactive for every bar of a run (e.g. during the WEEK_BARS warm-up right after a cold
start, before decision_feature_snapshot.jsonl has produced 2016 rows worth of history since this
script started tailing it -- mitigated by seed_detector()'s buffer/base-CSV warm-up below, but not
provably impossible on a very short buffer), this script's behavior collapses to being byte-identical
to live_eth_exithead_asymmetric_shadow_20260813.py, by construction (evaluate_exit_guarded's
early-return branch).

Built by directly reusing already-live, already-validated pieces, following the exact structural
pattern of live_eth_exithead_asymmetric_shadow_20260813.py (itself templated from
live_eth_jmlam4_regime_swap_shadow_20260809.py):
  - entry/exit decision + sidecar sizing: Omega461LiveAdapter, imported verbatim from
    trading_bot_modules.omega4_6_1_live -- the SAME class the real live ETH executor runs every
    cycle, constructed with the exact same components_override as the baseline shadow (h48qual =
    liveATR relabel + artifact-lineage shim sidecar, zig075 = fully original). The ONLY new piece is
    a second, standalone `_Component` (also imported verbatim from the same module, not
    reimplemented) built from h48qual's ORIGINAL bundle/sidecar, used ONLY via its .exit_probability
    method when the detector is active -- it never participates in adapter.components / entry
    routing / priority.
  - raw market data: tails data/live/decision_feature_snapshot.jsonl, same as both prior shadows.
  - real live ETH position for reference only: data/live/dashboard_state.json's `position` block.

Artifact-lineage note: identical situation to live_eth_exithead_asymmetric_shadow_20260813.py for the
liveATR h48qual component (needs the report.json-repointing shim sidecar, already built and pushed to
the server for that shadow -- reused verbatim here, not rebuilt). The ORIGINAL h48qual component
needs NO shim: FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_{BUNDLE,SIDECAR}_PATH is the exact bundle/sidecar
pair trading_bot.py's real live adapter already constructs by default (components_override=None
branch of Omega461LiveAdapter.__init__) with quality_threshold=0.50 -- an unmodified, already-live,
already-lineage-valid pair, so validate_sidecar_lineage passes on the server the same way it already
does for the real live production adapter. (This pair's report.json hardcodes an absolute path that
only resolves on the server -- same reason live_eth_exithead_asymmetric_shadow_20260813.py's zig075
component only constructs on the server, not dev -- so this script, like both precedents, must run on
the server.)

Entries and exits both queue for the NEXT bar's open, matching omega4_6_1_live.py's own documented
execution-delay model. Causal, forward-only, reads no ledger or future row -- Fresh-Forward-compliant
live-forward observation, not a backtest.

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

from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _load_fee_slip  # noqa: E402
from trading_bot_modules.omega4_6_1_live import (  # noqa: E402
    DURATION_THRESHOLD as OMEGA4_6_1_LIVE_DURATION_THRESHOLD,
    EXIT_THRESHOLD,
    Omega461LiveAdapter,
    _Component,
    _ComponentConfig,
)
from trading_bot_modules.runtime_config import (  # noqa: E402
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH,
    FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_SIDECAR_PATH,
)

KST = ZoneInfo("Asia/Seoul")
SNAPSHOT_PATH = ROOT / "data/live/decision_feature_snapshot.jsonl"
DASHBOARD_STATE_PATH = ROOT / "data/live/dashboard_state.json"
OUT_DIR = ROOT / "data/live/eth_regime_aware_exit_guard_shadow"
STATE_PATH = OUT_DIR / "state.json"
TRADES_PATH = OUT_DIR / "closed_trades.jsonl"
EQUITY_PATH = OUT_DIR / "equity_curve.jsonl"

# =====================================================================================================
# Detector constants -- copied verbatim from the already-executed, already-validated backtest run
# (see module docstring). NEVER recalibrated live.
# =====================================================================================================
DETECTOR_WEEK_BARS = 2016  # research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.WEEK_BARS
DETECTOR_PERCENTILE = 0.90  # ...DETECTOR_PERCENTILE (documentation only, not recomputed here)
DETECTOR_THRESHOLD = 0.8025793650793651  # report.json["detector"]["threshold_used"], full precision
DETECTOR_CALIBRATION_WINDOW = ("2025-01-01", "2025-06-30 23:59:59")  # documentation only
DETECTOR_SOURCE_REPORT = (
    "tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814/report.json"
)
# Fallback warm-up source if the live snapshot buffer alone doesn't yet cover a full
# DETECTOR_WEEK_BARS window (e.g. right after this shadow's own first-ever cold start on a
# short-lived snapshot log) -- matches research_eth_omega461_exit_sweep_20260721.BASE_2026 exactly
# (not imported, to avoid pulling that module's own heavier transitive import chain into an
# always-on process just for a path constant).
FALLBACK_BASE_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

# h48qual: exit_head swapped for the 2026-08-13 liveATR-relabel retrain -- byte-identical setup to the
# already-deployed baseline shadow (live_eth_exithead_asymmetric_shadow_20260813.py). This is what
# adapter.components["h48qual"] uses for BOTH entry_decision (always) and exit_probability (only when
# the detector is inactive).
H48QUAL_NEW_BUNDLE_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
    "/h48qual/true_3head_tabm_bundle.pt"
)
H48QUAL_SHIM_SIDECAR_PATH = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_asymmetric_shadow_20260813_h48qual_sidecar"
    "/risk_sidecar.pkl"
)
# h48qual ORIGINAL (pre-liveATR-relabel) exit head -- the exact bundle/sidecar trading_bot.py's real
# live adapter uses by default. Used ONLY for its .exit_probability method, ONLY when the detector is
# active on a bar where an open position's source_component == "h48qual". Never used for entry/sizing.
H48QUAL_ORIGINAL_BUNDLE_PATH = ROOT / FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH
H48QUAL_ORIGINAL_SIDECAR_PATH = ROOT / FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_SIDECAR_PATH
# zig075: fully original live artifacts, untouched by the detector in every state.
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
BUFFER_ROWS = 3000  # ~10.4 days of 5m bars, comfortably covers atr_window=192 + DETECTOR_WEEK_BARS


class SustainedUptrendDetector:
    """Live, O(1)-per-bar, causal reimplementation of research_eth_omega461_regime_aware_exit_head_
    uptrend_guard_20260814._rolling_dual_momentum_score's rolling(WEEK_BARS,
    min_periods=WEEK_BARS).mean() of (dual_momentum>0), evaluated one bar at a time instead of
    vectorized over a whole CSV. Self-verified to match the batch computation bar-for-bar by
    scripts/verify_eth_regime_aware_exit_guard_shadow_detector_20260814.py."""

    def __init__(self, *, threshold: float, week_bars: int) -> None:
        self.threshold = float(threshold)
        self.week_bars = int(week_bars)
        self._window: deque[float] = deque(maxlen=self.week_bars)

    def seed(self, dual_momentum_values: np.ndarray) -> None:
        """(Re)initialize the rolling window from historical values, given OLDEST-FIRST. Replaces any
        existing window contents -- call once, before the first live .update()."""
        self._window.clear()
        for v in dual_momentum_values:
            self._window.append(1.0 if float(v) > 0.0 else 0.0)

    def update(self, dual_momentum_value: float) -> tuple[float | None, bool]:
        """Push ONE new bar's dual_momentum reading (must be called exactly once per processed live
        bar, in strictly increasing timestamp order, regardless of position state -- the rolling
        window is a pure function of the bar sequence, not of trading state). Returns (score, active)
        for THAT bar: score is None (the live analogue of NaN) until the window holds `week_bars`
        observations, matching rolling(min_periods=week_bars); active is always False while score is
        None. `active` uses a strict `>` against the threshold, matching the backtest's own
        `(raw > threshold)`."""
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


def build_adapter(device: str = "cpu") -> Omega461LiveAdapter:
    return Omega461LiveAdapter(
        h48qual_bundle="", h48qual_sidecar="", zig075_bundle="", zig075_sidecar="",  # unused, components_override wins
        device=device,
        components_override=COMPONENTS_OVERRIDE,
        priority=PRIORITY,
        # duration_threshold/scale_map/base_template/expert_scales all omitted deliberately, exactly
        # as in live_eth_exithead_asymmetric_shadow_20260813.py -- this lets Omega461LiveAdapter apply
        # its own live ETH defaults, unmodified by tonight's detector work.
    )


def build_guard_component(device: torch.device) -> _Component:
    """Standalone h48qual ORIGINAL (pre-liveATR-relabel) component -- imported verbatim from
    trading_bot_modules.omega4_6_1_live, NOT part of any Omega461LiveAdapter.components dict (so it
    never participates in entry routing/priority). Used only via .exit_probability, only when the
    detector is active on an open h48qual position."""
    cfg = _ComponentConfig(
        "h48qual_guard_original", H48QUAL_ORIGINAL_BUNDLE_PATH, H48QUAL_ORIGINAL_SIDECAR_PATH,
        quality_threshold=0.50,
    )
    return _Component(cfg, device=device)


def evaluate_exit_guarded(
    adapter: Omega461LiveAdapter,
    guard_component: _Component,
    frame: pd.DataFrame,
    *,
    source_component: str,
    side: int,
    hold_bars: int,
    unrealized_move: float,
    mfe: float,
    mae: float,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
    bar_high_move: float | None,
    bar_low_move: float | None,
    detector_active: bool,
) -> tuple[bool, str, float, dict[str, Any]]:
    """Byte-identical to adapter.evaluate_exit(...) UNLESS source_component=='h48qual' AND
    detector_active -- in that one case, the exit-head probability is read from guard_component
    (h48qual's ORIGINAL exit head) instead of adapter.components['h48qual'] (the liveATR relabel).
    TP/SL barriers and EXIT_THRESHOLD (0.95) are identical in both branches -- only the *source
    model* for the exit-head probability changes. Mirrors
    scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.
    greedy_replay_regime_aware_exit_guard's guard branch exactly (same TP/SL-first order, same
    threshold, same "only the probability source switches" contract), rebuilt for the live adapter's
    one-bar-at-a-time evaluate_exit contract instead of that backtest's batched arrays.

    Returns (should_exit, reason, exit_prob, diag). diag={"guard_engaged": bool,
    "decision_differs": bool | None, "default_prob": float | None} -- decision_differs/default_prob
    are a diagnostic-only counterfactual (computed only when guard_engaged, never used to choose
    `reason`/`exit_prob` above): what would the liveATR default path have decided on this SAME bar?
    Mirrors the backtest's own guard_decision_differs_bars counter.
    """
    if source_component != "h48qual" or not detector_active:
        should_exit, reason, exit_prob = adapter.evaluate_exit(
            frame, source_component=source_component, side=side, hold_bars=hold_bars,
            unrealized_move=unrealized_move, mfe=mfe, mae=mae, notional=notional, leverage=leverage,
            take_profit=take_profit, stop_loss=stop_loss, bar_high_move=bar_high_move, bar_low_move=bar_low_move,
        )
        return should_exit, reason, exit_prob, {"guard_engaged": False, "decision_differs": None, "default_prob": None}

    # --- guard branch: TP/SL identical to Omega461LiveAdapter.evaluate_exit; only the exit-head
    # probability source changes (h48qual's ORIGINAL exit head instead of the liveATR relabel). ---
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

    # Diagnostic-only counterfactual: what would the default (liveATR) h48qual component have
    # decided on this same bar? Never used to set `should_exit`/`reason`/`prob` above.
    default_prob = adapter.components["h48qual"].exit_probability(
        frame, side=side, hold_bars=hold_bars, unrealized_move=unrealized_move, mfe=mfe, mae=mae,
        notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss,
    )
    decision_differs = bool(should_exit != (default_prob >= EXIT_THRESHOLD))
    return should_exit, reason, prob, {
        "guard_engaged": True, "decision_differs": decision_differs, "default_prob": float(default_prob),
    }


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


def seed_detector(detector: SustainedUptrendDetector, buffer: pd.DataFrame, last_processed_ts: pd.Timestamp | None) -> None:
    """Prime the incremental rolling window from already-collected buffer history so the detector
    isn't stuck inactive for a fresh DETECTOR_WEEK_BARS-bar warm-up on every process restart. Seeds
    with every buffer row up to (and including) last_processed_bar_ts on resume, or every row except
    the final one on a cold start (the final row is processed fresh as the first live bar and pushes
    its own flag then, via the normal update() call inside process_bar). Tops up from
    FALLBACK_BASE_CSV's tail if the live snapshot buffer alone doesn't cover a full
    DETECTOR_WEEK_BARS window -- matches the task's explicit fallback instruction; never touches 2025
    data (irrelevant to a live-2026 warm-up) and never re-derives DETECTOR_THRESHOLD."""
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


def process_bar(
    state: dict, frame: pd.DataFrame, adapter: Omega461LiveAdapter, guard_component: _Component,
    detector: SustainedUptrendDetector, fee: float, slip: float,
) -> dict:
    bar = frame.iloc[-1]
    bar_ts = pd.Timestamp(bar["timestamp"])
    # Detector advances exactly once per processed bar, unconditionally (position-state-independent),
    # matching the backtest's rolling window being a pure function of the bar sequence.
    score, detector_active = detector.update(float(pd.to_numeric(bar["dual_momentum"])))
    state["last_detector_score"] = score
    state["last_detector_active"] = bool(detector_active)
    state["detector_bars_seen"] = int(state.get("detector_bars_seen", 0)) + 1
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
        decision = adapter.decide_entry(frame)  # always liveATR h48qual + original zig075, detector-independent
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
        "detector_score": score, "detector_active": bool(detector_active),
    })
    return state


def run(args: argparse.Namespace) -> None:
    ensure_dir(OUT_DIR)
    print("[init] loading regime-aware exit-head guard Omega4.6.1 adapter (h48qual q050 = liveATR "
          "relabel exit_head for entry+default exit, ORIGINAL exit_head substituted only while the "
          "sustained-uptrend detector is active; zig075 q075 = fully original live; regime3 = "
          "original live HMM) -- research/shadow only, Odyssey2 #11, see module docstring", flush=True)
    device_obj = torch.device(args.device)
    adapter = build_adapter(device=args.device)
    guard_component = build_guard_component(device=device_obj)
    print(f"[init] duration_threshold={adapter.duration_threshold} (live default "
          f"{OMEGA4_6_1_LIVE_DURATION_THRESHOLD}, unmodified)", flush=True)
    print(f"[init] guard component loaded: alias={guard_component.cfg.alias} "
          f"base_cols={len(guard_component.base_cols)} quality_threshold={guard_component.cfg.quality_threshold} "
          f"experts={sorted(guard_component.loaded)}", flush=True)
    fee, slip = _load_fee_slip()
    detector = SustainedUptrendDetector(threshold=DETECTOR_THRESHOLD, week_bars=DETECTOR_WEEK_BARS)

    state = load_json(STATE_PATH, {})
    if not state:
        buffer, offset = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        seed_detector(detector, buffer, None)
        state = {
            "schema_version": "eth_regime_aware_exit_guard_shadow.v1", "live_forward_only": True,
            "order_submission_supported": False, "activation_allowed": False,
            "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "candidate": "ETH Omega4.6.1 regime-aware exit-head guard (Odyssey2 #11): while h48qual "
                          "holds a position, its exit-head probability check conditionally routes "
                          "between the ORIGINAL frozen exit head (sustained-uptrend detector active) "
                          "and the 2026-08-13 liveATR-relabel exit head (detector inactive, matches "
                          "the eth_exithead_asymmetric_shadow baseline). Detector = rolling 1-week "
                          f"(2016-bar) fraction of dual_momentum>0, threshold={DETECTOR_THRESHOLD:.6f} "
                          "(90th pct of 2025-Q1+Q2-only calibration, reused verbatim). zig075 and "
                          "h48qual's direction/quality/entry/sizing are untouched in every state.",
            "research_doc": "docs/experiments/eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.md",
            "baseline_shadow_doc": "docs/experiments/eth_omega461_exit_head_asymmetric_shadow_20260813.md",
            "detector_week_bars": DETECTOR_WEEK_BARS, "detector_threshold": DETECTOR_THRESHOLD,
            "started_at_kst": now_kst().isoformat(), "snapshot_offset": offset,
            "position": None, "pending": None, "last_processed_bar_ts": None,
            "equity": 1.0, "peak_equity": 1.0, "mdd": 0.0,
            "last_detector_score": None, "last_detector_active": False, "detector_bars_seen": 0,
            "h48qual_hold_bars": 0, "h48qual_guard_active_bars": 0, "h48qual_guard_decision_differs_bars": 0,
        }
        print(f"[init] seeded buffer with {len(buffer)} rows, snapshot_offset={offset}", flush=True)
    else:
        buffer, _ = seed_buffer(SNAPSHOT_PATH, BUFFER_ROWS)
        buffer["timestamp"] = pd.to_datetime(buffer["timestamp"])
        last_processed = state.get("last_processed_bar_ts")
        seed_detector(detector, buffer, pd.Timestamp(last_processed) if last_processed else None)
        print(f"[resume] loaded prior state, last_processed_bar_ts={state.get('last_processed_bar_ts')}, "
              f"equity={state.get('equity')}, h48qual_guard_active_bars={state.get('h48qual_guard_active_bars')}", flush=True)

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
                    # columns directly off frame and also recompute+overwrite them internally --
                    # this explicit pre-append is still required so the ffill/bfill NaN-backfill step
                    # just below can also reach and patch any old gaps in the regime3 columns
                    # themselves, not just the raw OHLCV/feature columns. Same pattern as both
                    # precedent shadow scripts (load-bearing there too).
                    frame = adapter.regime3_current.append(frame)
                    # Backfill/forward-fill only the historical rows (never the latest, current bar)
                    # so an old gap in decision_feature_snapshot.jsonl can't permanently wedge the
                    # shadow -- identical rationale/implementation to both precedent shadow scripts.
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
