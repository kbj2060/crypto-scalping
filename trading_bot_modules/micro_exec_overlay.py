"""MicroExec v1 — 1m-cadence execution-timing overlay on the live microstructure_1m signals.

Design doc: docs/micro_scalp_1m_design_20260718.md (section 4). This module does NOT generate
positions. Given a Layer-1 intent ("open LONG within the next K minutes"), it decides each
minute whether to execute now, keep waiting, or flag the minute as vetoed (toxic book state).
At the deadline the intent is always executed (Layer-1 authority is never overridden).

Signal: contrarian composite S = mean of causally-z-scored {-tbr_dev, -nif_whale, -nif_retail,
-obi}. High S = taker-sell/ask-pressure extreme = good
minute to BUY (and -S symmetric for SELL). Validated in
scripts/analyze_microstructure_edge_20260718.py (per-component daily-block |t| 3..7.6, all
contrarian) — v1 deliberately uses no trained model.

Causality: a microstructure row labeled ts=T is written at wall ~T+60..75s, so it is usable
from decision minute D = T+2min onward. prepare_overlay_frame() returns the frame indexed by
that first-usable decision minute; live callers reading the scanner's in-memory cache are
strictly fresher and may treat the newest cached row as current.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AVAIL_SHIFT_MIN = 2  # ts=T row first usable at decision T+2min (see module docstring)

# (source column, contrarian sign): +1*sign*value enters the long-favoring composite.
# v1 evaluated two candidates in the paired replay (replay_micro_exec_overlay_20260718.py):
# a 6-component set adding {queue_bias_m15, obi_m15, tbr_dev_m15} (long-only positive, shorts
# flat) and this fast 4-component set (positive on BOTH sides, simpler) — fast set adopted.
_COMPONENTS = [
    ("tbr_dev", -1.0),
    ("nif_whale", -1.0),
    ("nif_retail", -1.0),
    ("obi", -1.0),
]
_Z_MIN_PERIODS = 1440  # 1 day of expanding history before z-scores are emitted


@dataclass
class MicroExecConfig:
    exec_z: float = 1.28          # composite z needed to execute before the deadline (~top 10%)
    deadline_min: int = 10        # K: force execution this many minutes after the intent
    veto_enabled: bool = True
    veto_toxicity_z: float = 2.0
    veto_spoofing: float = 0.5
    veto_queue_collapse: bool = True


@dataclass
class PlacementConfig:
    """v1.5 maker-placement policy (docs/micro_scalp_1m_design_20260718.md section 4.2).

    ETHUSDT perp book is ~1 tick wide (orderbook_decision_snapshots: spread median 0.053bps),
    so the economics are fee-differential dominated: maker 2.0 vs taker 4.5 bps/side. The
    contrarian composite modulates HOW PATIENT the limit order is, not whether to trade:
      side*score >= urgent_z  -> favorable move imminent: join top-of-book (taker instead if
                                 short-term momentum is already running in the intent
                                 direction -- waiting would chase)
      side*score <= patient_z -> price expected to drift toward us: rest deeper by
                                 deep_frac * recent 15m range
      otherwise               -> join top-of-book
    Deadline and stop-loss-style urgent exits are always taker (never risk missing Layer-1).
    """
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 4.5
    urgent_z: float = 1.0
    patient_z: float = -0.5
    momentum_taker_bps: float = 8.0   # 3m move in intent direction that forces cross-spread
    deep_frac: float = 0.5            # depth of patient limit as fraction of 15m range
    join_offset_bps: float = 0.08     # half-spread + 1 tick: conservative join-bid price line


def choose_placement(pcfg: PlacementConfig, *, side: int, score: float,
                     mom3_bps: float, range15_bps: float) -> tuple[str, float]:
    """Order placement for one minute of an active intent.

    Returns (mode, limit_offset_bps): mode "taker" executes at market now; mode "limit" rests
    side*offset bps below (buy) / above (sell) the current price. Offset includes the
    conservative join line so a fill requires strict trade-through of that level.
    """
    s = side * score if np.isfinite(score) else 0.0
    if s >= pcfg.urgent_z:
        if side * mom3_bps >= pcfg.momentum_taker_bps:
            return "taker", 0.0
        return "limit", pcfg.join_offset_bps
    if s <= pcfg.patient_z:
        depth = max(0.0, pcfg.deep_frac * range15_bps)
        return "limit", pcfg.join_offset_bps + depth
    return "limit", pcfg.join_offset_bps


def _expanding_z(s: pd.Series) -> pd.Series:
    mu = s.expanding(min_periods=_Z_MIN_PERIODS).mean().shift(1)
    sd = s.expanding(min_periods=_Z_MIN_PERIODS).std().shift(1)
    return (s - mu) / sd.replace(0.0, np.nan)


def prepare_overlay_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: quality-filtered microstructure_1m rows indexed by naive-UTC bar label ts.

    Returns a frame indexed by the first decision minute each row is usable at (ts + 2min)
    with columns: score (contrarian-long composite z), veto, plus the veto reasons.
    """
    m = raw.sort_index()
    d = pd.DataFrame(index=m.index)
    d["tbr_dev"] = m["taker_buy_ratio"] - 0.5
    d["nif_whale"] = m["nif_whale"]
    d["nif_retail"] = m["nif_retail"]
    d["obi"] = m["obi"]

    z = pd.DataFrame({col: _expanding_z(d[col]) * sign for col, sign in _COMPONENTS})
    out = pd.DataFrame(index=m.index)
    out["score"] = z.mean(axis=1)

    out["toxicity_z"] = _expanding_z(m["shadow_toxicity_score"])
    out["veto_toxicity"] = out["toxicity_z"] > MicroExecConfig.veto_toxicity_z
    out["veto_spoofing"] = m["spoofing_score"].astype(float) >= MicroExecConfig.veto_spoofing
    out["veto_queue_collapse"] = m["shadow_queue_collapse"].astype(float) >= 1.0
    out["veto"] = out[["veto_toxicity", "veto_spoofing", "veto_queue_collapse"]].any(axis=1)

    out.index = out.index + pd.Timedelta(minutes=AVAIL_SHIFT_MIN)
    return out


def decide_minute(cfg: MicroExecConfig, *, side: int, score: float, veto: bool,
                  minutes_since_intent: int) -> str:
    """One live decision. side: +1 long intent, -1 short. Returns EXECUTE / WAIT / VETO_WAIT.

    The deadline always executes: the overlay may only cost delay, never a Layer-1 trade.
    """
    if minutes_since_intent >= cfg.deadline_min:
        return "EXECUTE"
    if cfg.veto_enabled and veto:
        return "VETO_WAIT"
    if np.isfinite(score) and side * score >= cfg.exec_z:
        return "EXECUTE"
    return "WAIT"
