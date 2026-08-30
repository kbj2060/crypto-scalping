from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, web
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Macro calendar (2026-08-26) needs FRED/EIA/Finnhub API keys from .env -- no other endpoint in
# this file has needed a real secret before, so .env was never loaded here until now.
load_dotenv(REPO_ROOT / ".env")
# Reuses the exact, already-verified signal formulas from the standalone CLI dashboard rather
# than re-deriving them here -- see that module's docstring for formula provenance (each formula
# transcribed verbatim from the 2026-08-14 research scripts). compute_signals/bars_since_last_true
# are pure functions (no I/O, no sleep) -- that makes them correctness-safe to call from an async
# handler, but NOT free: 2026-08-25 perf pass moved the compute_signals() call itself behind
# asyncio.to_thread (see load_evidence_signals()) so its pandas rolling-window work doesn't block
# the event loop for its duration, matching the pattern load_liquidation_5m_signal()/
# load_liquidation_direction_signal() already used.
from scripts.live_evidence_signal_dashboard_20260823 import (  # noqa: E402
    FETCH_LIMIT as EVIDENCE_FETCH_LIMIT,
    FUNDING_HISTORY_LIMIT as EVIDENCE_FUNDING_HISTORY_LIMIT,
    FUNDING_Z_MIN_PERIODS as EVIDENCE_FUNDING_Z_MIN_PERIODS,
    FUNDING_Z_WINDOW as EVIDENCE_FUNDING_Z_WINDOW,
    PCTRANK_WINDOW as EVIDENCE_PCTRANK_WINDOW,
    SIGNAL_ORDER as EVIDENCE_SIGNAL_ORDER,
    bars_since_last_true,
    compute_signals,
)
# 유동성스윕 반등예측 event-triggered signal (2026-08-29, TabPFN Tier0+rsi model -- see
# docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md). Own klines fetch +
# a frozen historical TabPFN context, NOT from trading_bot.py's dashboard_state.json -- computed
# dashboard-side so it never touches the live bot.
from scripts.live_eth_sweep_v_rebound_signal_20260829 import compute_eth_sweep_v_rebound_signal  # noqa: E402
# taker_delta_z_climax / short_term_return_z evidence-signal chips REPLACED in-place with their
# TabPFN meta-label models' live probability (2026-08-30, user decision -- unlike V_REBOUND above,
# these stay in the "증거 신호" row and reuse the klines/compute_signals() this endpoint already
# computed each cycle, rather than becoming new standalone "모델 내부 지표" chips with their own
# fetch+cache). See docs/experiments/eth_taker_delta_climax_metalabel_20260829.md.
from scripts.live_evidence_signal_metalabel_20260829 import compute_evidence_signal_metalabels  # noqa: E402
# 2026-08-30: liquidity_sweep now trained on the SAME Tier0+rsi schema as taker/short_term_
# return_z/dalton_rule2_balance_edge (standard touch-based-MFE redo, replacing the V_REBOUND-model
# relay bridge this import used to be) -- it lives in METALABEL_SIGNALS above and is handled by
# compute_evidence_signal_metalabels() like every other signal there, no separate import/call.
# 베이시스 청산압박 model indicator (replaces 독성/toxicity, 2026-08-27) -- own live spot+perp
# klines fetch each cache cycle (no persistent collector), same "computed here, not bot state"
# category as OI 급변 above. See that module's docstring for the liquidation-crowding validation.
from scripts.live_spot_perp_basis_signal_20260827 import compute_basis_liquidation_signal  # noqa: E402
# 5-minute liquidation $ aggregate for the Snapshot tab's liquidation gauge (2026-08-25) -- reads
# tail_risk.duckdb's own per-minute persisted history read-only, same "computed here, not from
# trading_bot.py's dashboard_state.json" category as OI 급변 above. See that module's docstring for
# why this can't just be 5x the trailing 1-minute value the bot already exposes.
from scripts.live_liquidation_5m_signal_20260825 import compute_liquidation_5m_signal  # noqa: E402
# Directional-only liquidation tilt reading (liq_net_z_12, contrarian sign convention) -- model-
# indicator tier like OI 급변, NOT an evidence-signal-tier chip. See that module's docstring for
# the pre-registered formula it reuses and why it carries no PnL/economic claim.
from scripts.live_liquidation_direction_signal_20260825 import compute_liquidation_direction_signal  # noqa: E402
# Snapshot-tab liquidation map (estimated support/resistance, self-hosted Coinglass-heatmap
# alternative, 2026-08-24) -- discretionary reading aid only, NOT wired to trading_bot.py.
# 2026-08-25: switched from the event-driven state machine (compute_event_driven_levels(),
# resets on break/drift, staleness had no reliable lever -- see memory
# eth_liquidation_map_staleness_tuning_rejected_20260825) to a plain fixed-lookback rolling
# recompute (compute_liquidation_levels()) at user request, to replicate Coinglass's own
# liquidation-heatmap LOGIC: recomputed fresh each time rather than a sticky level set -- see
# eth_liquidation_map_dwell_duration_metric_rejected_20260825 for why event-driven's win-rate edge
# over this variant was already known to be thin/inconsistent, so this isn't a downgrade. A rolling
# window also has no "staleness" concept at all (always reflects the latest
# LIQUIDATION_MAP_LOOKBACK_HOURS as of the last cache refresh), which incidentally resolves the
# earlier "85시간 too long" complaint the event-driven variant's reset-staleness produced.
# 2026-08-25 (later same day): switched again from 48h to 168h (7d) fixed lookback -- TRAIN/OOS
# intrabar dwell validation (research_eth_liquidation_map_fixed7d_dwell_intrabar_break_test_20260825
# vs. ..._fixed48h_..._20260825) showed 168h's support side OOS pairWR=0.574 (up from TRAIN 0.519,
# no TRAIN->OOS reversal) was the best support result of the three variants tested that day
# (event-driven 0.349, 48h-fixed 0.476); no longer literally mirrors Coinglass's own 48h default,
# hence the constant is named LOOKBACK_HOURS not COINGLASS_LOOKBACK_HOURS.
# 2026-08-26: switched compute_liquidation_levels()/compute_heatmap_history() (both sides priced
# off close) to compute_spliced_levels()/compute_spliced_heatmap_history() (support priced off
# (high+low)/2, resistance still off close -- two independent passes spliced together, not a
# shared-bins merge, see that function's own docstring for why). 20-seed-averaged, 4-fold
# (bear/choppy/bull) backtest: support pairWR 0.469->0.590 (+0.12, far outside the ~0.03-0.04
# seed-noise floor), resistance 0.538->0.524 (within noise, at/above the close-only baseline in
# 3 of 4 folds) -- see eth_liquidation_map_spliced_hybrid_confirmed_20260826 memory for the full
# validation chain (single-split -> 4-fold walk-forward -> regime characterization -> seed
# robustness). compute_liquidation_levels()/compute_heatmap_history() are unchanged and still
# importable (other research scripts still use them as the close-only reference) -- only this
# dashboard entry point moved.
from scripts.live_liquidation_map_20260824 import compute_spliced_levels, compute_spliced_heatmap_history  # noqa: E402
# Regime overlay (bull/bear/chop probability per 5-min bar) for the Snapshot tab's liquidation-map
# chart. 2026-08-26: swapped from the wide24 HMM+linear-calibration model to an independently
# trained HistGradientBoostingClassifier (OOS balanced_accuracy 0.9189 vs wide24's 0.7691) -- see
# live_regime_gbm3_signal_20260826.py's docstring and memory
# eth_regime_hierarchical_whipsaw_circularity_rejected_20260826 for the full history (a whipsaw
# sub-class was investigated at length and dropped -- every feature set tried left it too
# imprecise for a dashboard chip). Loaded independently of whatever trading_bot.py's live regime
# routing currently uses. See that module's docstring for why DAYS_BACK isn't shortened and why
# this is expensive enough to need its own cache.
# 2026-08-27: reverted from the 2-class trend/chop GBM2 model (2026-08-27, built for a low-flip
# discretionary display) back to GBM3 here -- a same-day cost-gated backtest found GBM2's much
# broader chop definition (55-57% of all bars vs GBM3's narrower slice) dilutes a liquidation-
# confluence filter's selectivity (see eth_evidence_signal_liquidation_confluence_gbm2gate_rejected_
# 20260827 memory) and the user asked to match the GBM3-based analysis. GBM2 remains a valid,
# separately-loadable model for anything that specifically wants a low-flip label; it is simply not
# what this dashboard endpoint serves right now.
from scripts.live_regime_gbm3_signal_20260826 import compute_regime_gbm3_signal as compute_regime_wide24_signal  # noqa: E402
# Session-open volatility risk alert for the evidence-signal chip row (2026-08-26) -- pure
# calendar/clock computation (pandas_market_calendars), no price data, so it needs no cache of its
# own; computed fresh on every evidence-signal refresh. See that module's docstring for the
# same-day empirical research (NYSE open real effect, LSE/JPX marginal) behind the chosen windows.
from scripts.live_session_volatility_alert_20260826 import compute_session_volatility_alert  # noqa: E402
# US macro/corporate event calendar for the Snapshot tab (2026-08-26) -- see that module's
# docstring for the 6 sources (FRED/FOMC-static/Fed Chair HTML/EIA-rule-based/Finnhub/Treasury) and their
# individual caveats. Independent of evidence-signal's klines fetch -- own cache below.
from scripts.live_macro_calendar_20260826 import compute_macro_calendar, compute_macro_event_alert  # noqa: E402
# 2026-08-31: per-coin registry for the 4 Snapshot-tab signals wired to BTC this session (basis
# liquidation, liquidation direction, liquidation 5m, liquidation map) -- see
# docs/eth_dashboard_multicoin_expansion_design_20260831.md section 6. Evidence signals/regime/
# specialized-detector (EVIDENCE_SIGNAL_SYMBOL etc. below) are untouched -- those are trained ML
# models with no BTC-trained artifact yet, not something a symbol swap alone can serve.
from scripts.coin_config import COIN_CONFIG  # noqa: E402

EVIDENCE_SIGNAL_SYMBOL = "ETHUSDT"
EVIDENCE_SIGNAL_INTERVAL = "5m"
EVIDENCE_SIGNAL_CACHE_SECONDS = 60
# Live PREVIEW of the currently-forming bar (2026-08-26), NOT the confirmed signal above -- see
# load_evidence_signals_provisional() docstring. Short TTL so a ~10s frontend poll gets a fresh
# read each time; still cached to protect against a multi-tab request burst.
EVIDENCE_SIGNAL_PROVISIONAL_CACHE_SECONDS = 8
EVIDENCE_SIGNAL_HISTORY_BARS = 48  # 4h strip for the Snapshot tab's per-bar activity graph
EVIDENCE_SIGNAL_BTC_SYMBOL = "BTCUSDT"  # smt_divergence's cross-asset non-confirmation leg, 2026-08-24
EVIDENCE_SIGNAL_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"  # orthogonal_combo's bottom-leg funding_z input (2026-08-27; formerly funding_oscillator_combo's own leg)
LIQUIDATION_MAP_INTERVAL = "1h"  # symbol now comes from COIN_CONFIG[asset]["binance_symbol"] (2026-08-31)
LIQUIDATION_MAP_LOOKBACK_HOURS = 24  # 2026-08-25: 168h->24h. Both event-driven (floor=ceiling=Nh
                                      # sweep, research_eth_liquidation_map_event_driven_window_
                                      # sweep_20260825) and this stateless mechanism (research_eth_
                                      # liquidation_map_fixed24h_dwell_intrabar_test_20260825, vs.
                                      # the same fixed48h/fixed7d siblings) show no validated OOS
                                      # edge at any lookback tried -- so the choice among 24h/48h/
                                      # 168h is a UX preference (staleness/reactivity), not a
                                      # statistical one. Stateless (this mechanism) was inferred to
                                      # match Coinglass's own likely implementation more closely than
                                      # event-driven -- see eth_liquidation_map_coinglass_visual_
                                      # logic_replication_20260825 for the reasoning (their lookback
                                      # is an hours-selector dropdown, not a reset-sensitivity
                                      # control, and sweep-darkening is explainable by real OI
                                      # depleting under a stateless recompute, no freeze-until-reset
                                      # state machine required).
LIQUIDATION_MAP_FETCH_LIMIT = 48  # 24h window + buffer for occasional dropped/duplicate bars -- no
                                   # state machine to bootstrap, so no need for a long fetch. Also
                                   # covers compute_heatmap_history()'s own need (lookback+display+
                                   # slack ~= 24+6+3 = 33h) with room to spare, so unchanged.
LIQUIDATION_MAP_CACHE_SECONDS = 300  # structure moves slowly -- no need to recompute every tick
# 2026-08-26: a full recompute (15-day fetch + FeatureEngineer + HMM filter) takes ~10-20s, and the
# HMM itself is sticky (0.90) so regime rarely flips bar-to-bar -- a 5-min cache doesn't meaningfully
# stale the reading. See live_regime_wide24_signal_20260826.py's module docstring.
REGIME_WIDE24_CACHE_SECONDS = 300
MACRO_CALENDAR_CACHE_SECONDS = 6 * 3600  # calendar dates change at most once/day -- no reason to
                                          # re-hit 3 external APIs every evidence-signal cycle
# 2026-08-26 (user request): given its own dedicated constant rather than reusing
# EVIDENCE_SIGNAL_CACHE_SECONDS, so speeding this up doesn't also speed up the unrelated OI/
# liquidation-direction signals that happen to share that constant. Unlike the evidence signals,
# this one is a genuinely incremental accumulator (compute_liquidation_5m_signal() sums whatever
# 1-minute rows have landed in the current BAR_MINUTES=15 window so far, see its docstring) rather
# than a bar-close-only reading -- so a shorter cache here means real reduced staleness (up to 1s
# lag behind a new duckdb row instead of up to 60s), not a "provisional/unconfirmed" reading like
# the evidence-signal preview needed its own separate endpoint for. 2026-08-27: dropped 10s->1s per
# user request -- safe to poll this tightly since compute_liquidation_5m_signal() only reads a local
# duckdb (no external API/rate-limit exposure); note the duckdb itself only gains a new row once a
# minute (tail_risk_interceptor.py's insert cadence), so this mostly tightens worst-case staleness
# rather than surfacing meaningfully new data every second.
LIQUIDATION_5M_SIGNAL_CACHE_SECONDS = 1
LIQUIDATION_MAP_DISPLAY_HOURS = 6  # 2026-08-25 user request: density-history snapshot count for the
                                    # chart's time-varying heatmap overlay (see compute_heatmap_
                                    # history() docstring) -- matches the Snapshot-tab chart's own
                                    # visible-candle window (4h->6h same day, "4시간은 너무 작다")
                                    # (dashboard/live/app.js's SNAPSHOT_CHART_MAX_CANDLES) so every
                                    # column the chart can show has a real snapshot behind it.

# The 6 model-internal indicators (microstructure/tail_risk) only ever have their LATEST reading
# persisted by trading_bot.py -- no history is stored anywhere. Rather than touch the live bot
# (would need a bot restart, open-position risk) or the browser (resets every page load), this
# dashboard SERVER -- already polling data/live/dashboard_state.json every EVENT_POLL_SECONDS in
# publish_dashboard_events() -- keeps its own small in-memory sample buffer, gated to a much
# coarser interval than that poll. It survives page refreshes and new browser sessions as long as
# THIS SERVER PROCESS stays up; it resets only on a dashboard-server restart (e.g. a deploy), not
# on every page load like the old client-only accumulation did. Raw values only -- the tone/hint
# thresholds stay in app.js (single source of truth), applied to this history same as to the
# live reading, so there is no second copy of that classification logic to drift out of sync.
MODEL_INDICATOR_SAMPLE_SECONDS = 300  # 5 min, matching the evidence-signal strip's bar cadence
MODEL_INDICATOR_HISTORY_MAX = 48  # 4h at the sample interval above -- same window as evidence signals
LIVE_DIR = REPO_ROOT / "data" / "live"
DASHBOARD_DIR = REPO_ROOT / "dashboard" / "live"
# 2026-08-27: tail_risk_interceptor.py's event-triggered sibling of dashboard_state.json's
# tail_risk block (see its _write_liq_burst_state() docstring) -- written the instant a new
# liquidation event arrives, not on a 10s timer, for sub-few-second "sudden liquidation" alerting.
LIQ_BURST_STATE_PATH = LIVE_DIR / "liq_burst_state.json"
# Shadow-only, no order submission -- standalone loop separate from trading_bot.py's
# single-slot BTC shadow (see scripts/run_btc_multislot_shadow_loop_20260807.py).
BTC_MULTISLOT_SHADOW_STATE_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_state_20260807.json"
BTC_MULTISLOT_SHADOW_LEDGER_PATH = REPO_ROOT / "data" / "ensemble" / "omega4_6_1_btc_multislot_shadow_ledger_20260807.csv"
BTC_MULTISLOT_SHADOW_BAR_SECONDS = 300
# Odyssey4: h48qual regime-aware exit-head guard (Odyssey3, unchanged) + zig075 SHORT
# sustained-uptrend entry veto (Odyssey4 #1, CONFIRMED), shadow-only -- supersedes the retired
# eth-jmlam4-shadow (regime3 HMM->JM swap, never reproduced at N=5 seeds) and eth-exithead-shadow
# (h48qual exit_head liveATR relabel baseline, subsequently found NOT robust to walk-forward
# retraining) as of 2026-08-14 (see scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py
# and docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md).
ETH_ODYSSEY4_SHADOW_STATE_PATH = REPO_ROOT / "data" / "live" / "eth_odyssey4_shadow" / "state.json"
ETH_ODYSSEY4_SHADOW_TRADES_PATH = REPO_ROOT / "data" / "live" / "eth_odyssey4_shadow" / "closed_trades.jsonl"
ETH_ODYSSEY4_SHADOW_BAR_SECONDS = 300
MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT", "xrp": "XRPUSDT"}
EVENT_POLL_SECONDS = 2.5
MARKET_HISTORY_CACHE_SECONDS = 300
SCALP_SHADOW_MODEL_ID = "eth_micro_scalp_source_stable_opportunity_moe_v4_20260718"
SCALP_SHADOW_STATE_SCHEMA = "eth_micro_scalp_v4.shadow_bot_step.v1"
SCALP_SHADOW_SUMMARY_SCHEMA = "eth_micro_scalp_v4.shadow_bot.v1"
SCALP_SHADOW_OBSERVER_SCHEMA = "eth_micro_scalp_v3.fresh_forward_observer.v1"
SCALP_SHADOW_FEES_BP = (2.0, 4.5, 5.5, 9.0)
SCALP_SHADOW_DISPLAY_FEE_BP = 4.5
SCALP_SHADOW_ASSETS = {
    "eth": {
        "asset": "eth",
        "model_id": SCALP_SHADOW_MODEL_ID,
        "state_schema": SCALP_SHADOW_STATE_SCHEMA,
        "summary_schema": SCALP_SHADOW_SUMMARY_SCHEMA,
        "observer_schema": SCALP_SHADOW_OBSERVER_SCHEMA,
        "state_file": "eth_micro_scalp_v4_shadow_state.json",
        "database_file": "eth_micro_scalp_v4_shadow.duckdb",
        "symbol": "ETHUSDT",
        "require_asset_contract": False,
    },
    "btc": {
        "asset": "btc",
        "model_id": "btc_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
        "state_schema": "cross_asset_micro_scalp.shadow_bot_step.v1",
        "summary_schema": "cross_asset_micro_scalp.shadow_bot.v1",
        "observer_schema": "cross_asset_micro_scalp.shadow_observer.v1",
        "state_file": "btc_micro_scalp_shadow_state.json",
        "database_file": "btc_micro_scalp_shadow.duckdb",
        "symbol": "BTCUSDT",
        "require_asset_contract": True,
    },
    "sol": {
        "asset": "sol",
        "model_id": "sol_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
        "state_schema": "cross_asset_micro_scalp.shadow_bot_step.v1",
        "summary_schema": "cross_asset_micro_scalp.shadow_bot.v1",
        "observer_schema": "cross_asset_micro_scalp.shadow_observer.v1",
        "state_file": "sol_micro_scalp_shadow_state.json",
        "database_file": "sol_micro_scalp_shadow.duckdb",
        "symbol": "SOLUSDT",
        "require_asset_contract": True,
    },
}
SCALP_REUSE_MODES = {
    "eth_lifecycle": {
        "asset": "eth",
        "mode": "eth_lifecycle",
        "model_id": "eth_micro_scalp_dynamic_lifecycle_shadow_v1_20260718",
        "state_schema": "micro_scalp_reuse.shadow_bot_step.v1",
        "summary_schema": "micro_scalp_reuse.shadow_bot.v1",
        "observer_schema": "micro_scalp_reuse.shadow_observer.v1",
        "state_file": "eth_micro_scalp_lifecycle_shadow_state.json",
        "database_file": "eth_micro_scalp_lifecycle_shadow.duckdb",
        "symbol": "ETHUSDT",
        "require_asset_contract": True,
    },
    "sol_entry": {
        "asset": "sol",
        "mode": "sol_entry",
        "model_id": "sol_micro_scalp_entry_only_shadow_v1_20260718",
        "state_schema": "micro_scalp_reuse.shadow_bot_step.v1",
        "summary_schema": "micro_scalp_reuse.shadow_bot.v1",
        "observer_schema": "micro_scalp_reuse.shadow_observer.v1",
        "state_file": "sol_micro_scalp_entry_shadow_state.json",
        "database_file": "sol_micro_scalp_entry_shadow.duckdb",
        "symbol": "SOLUSDT",
        "require_asset_contract": True,
    },
}


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def make_etag(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]
    return f'W/"{prefix}-{digest}"'


def etag_matches(request: web.Request, etag: str) -> bool:
    candidates = request.headers.get("If-None-Match", "")
    return any(candidate.strip() in {"*", etag} for candidate in candidates.split(","))


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def row_ts(row: dict) -> float:
    raw = row.get("closed_at") or row.get("ts") or row.get("opened_at") or ""
    try:
        return time.mktime(time.strptime(str(raw).replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0


def strategy_tag(row: dict) -> str:
    basis = f"{row.get('source', '')} {row.get('raw_source', '')}".upper()
    if "SNIPER" in basis:
        return "SNIPER"
    if "TREND" in basis:
        return "TREND"
    if "MICRO" in basis or "WNC" in basis:
        return "MICRO"
    if "GOVERNOR" in basis or "CASH" in basis:
        return "GOVERNOR"
    if "COMPACT" in basis:
        return "COMPACT"
    if "CONTROLLER" in basis:
        return "CONTROLLER"
    return "GOVERNOR"


def pnl_pct(row: dict) -> float:
    if row.get("pnl_pct") is not None:
        return float(row.get("pnl_pct") or 0.0)
    if row.get("pnl_frac") is not None:
        return float(row.get("pnl_frac") or 0.0) * 100.0
    return 0.0


def equity_series(rows: list[dict], source_filter: str) -> list[dict]:
    closes = [r for r in rows if str(r.get("kind", "")).upper() == "CLOSE"]
    if source_filter != "ALL":
        closes = [r for r in closes if strategy_tag(r) == source_filter]
    closes.sort(key=row_ts)

    equity = 1.0
    out = []
    for idx, row in enumerate(closes, start=1):
        trade_pnl = pnl_pct(row)
        equity *= 1.0 + trade_pnl / 100.0
        out.append(
            {
                **row,
                "chart_index": idx,
                "pnl_pct": trade_pnl,
                "equity": equity,
                "cumulative_return_pct": (equity - 1.0) * 100.0,
                "ts": row.get("closed_at") or row.get("ts"),
            }
        )
    return out


def utc_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "T")
    if text.endswith("Z") or "+" in text[10:]:
        return text
    return text + "Z"


def utc_age_minutes(value: Any, *, offset_seconds: float = 0.0) -> float | None:
    """`offset_seconds` shifts `value` forward before computing age -- use this to convert an
    OHLCV bar's OPEN-time label (the raw kline convention, e.g. the last-bar timestamps this repo's
    shadow scripts store) into its CLOSE time (open + bar duration) for a freshness/staleness
    reading, since a bar is only actually usable/complete once it closes. Without this, "age" reads
    a full bar duration older than the data actually is. Default 0.0 preserves every existing
    caller's behavior exactly."""
    encoded = utc_iso(value)
    if encoded is None:
        return None
    try:
        timestamp = datetime.fromisoformat(encoded.replace("Z", "+00:00"))
    except ValueError:
        return None
    if offset_seconds:
        timestamp = timestamp + timedelta(seconds=offset_seconds)
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 60.0)


def _evidence_last_fired_ts(series: pd.Series, latest: pd.Series) -> str | None:
    """Exact UTC timestamp of the bar where `series` was last True, derived from
    bars_since_last_true()'s bar-offset via the fixed 5-minute evidence-signal bar spacing --
    None if it never fired in the loaded lookback."""
    bars = bars_since_last_true(series)
    if bars is None:
        return None
    return utc_iso(latest["timestamp"] - pd.Timedelta(minutes=5 * bars))


def _require_scalp_contract(condition: bool, field: str) -> None:
    if not condition:
        raise RuntimeError(f"scalp shadow contract mismatch: {field}")


def scalp_shadow_payload(
    live_dir: Path,
    asset: str = "eth",
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configs = SCALP_SHADOW_ASSETS if configs is None else configs
    config = configs.get(asset)
    _require_scalp_contract(config is not None, "asset")
    state_path = live_dir / config["state_file"]
    database_path = live_dir / config["database_file"]
    state = load_json(state_path)
    _require_scalp_contract(isinstance(state, dict), "state")
    summary = state.get("summary") or {}
    stream = state.get("stream") or {}
    _require_scalp_contract(state.get("schema_version") == config["state_schema"], "state.schema_version")
    _require_scalp_contract(state.get("model_id") == config["model_id"], "state.model_id")
    _require_scalp_contract(state.get("activation_allowed") is False, "state.activation_allowed")
    _require_scalp_contract(state.get("order_submission_supported") is False, "state.order_submission_supported")
    _require_scalp_contract(summary.get("schema_version") == config["summary_schema"], "summary.schema_version")
    _require_scalp_contract(summary.get("model_id") == config["model_id"], "summary.model_id")
    if config.get("require_asset_contract"):
        _require_scalp_contract(state.get("asset") == config["asset"], "state.asset")
        _require_scalp_contract(summary.get("asset") == config["asset"], "summary.asset")
        _require_scalp_contract(summary.get("symbol") == config["symbol"], "summary.symbol")
        if config.get("mode"):
            _require_scalp_contract(state.get("mode") == config["mode"], "state.mode")
            _require_scalp_contract(summary.get("mode") == config["mode"], "summary.mode")
    _require_scalp_contract(summary.get("performance_eligible") is False, "summary.performance_eligible")
    _require_scalp_contract(summary.get("order_submission_supported") is False, "summary.order_submission_supported")
    _require_scalp_contract(summary.get("fixed_holding_period_used") is False, "summary.fixed_holding_period_used")
    _require_scalp_contract(float(summary.get("unit_notional", 0.0)) == 1.0, "summary.unit_notional")
    _require_scalp_contract(
        summary.get("evidence_class") == "counterfactual completed-close-to-next-completed-close",
        "summary.evidence_class",
    )
    _require_scalp_contract(summary.get("fresh_forward_bar_by_bar") is True, "summary.fresh_forward_bar_by_bar")
    _require_scalp_contract(summary.get("trade_ledgers_used_as_input") is False, "summary.trade_ledgers_used_as_input")
    _require_scalp_contract(summary.get("saved_parent_exit_timestamps_used") is False, "summary.saved_parent_exit_timestamps_used")
    _require_scalp_contract(summary.get("future_rows_used_for_entry") is False, "summary.future_rows_used_for_entry")
    expected_fee_keys = {f"{fee:.2f}bp_per_notional_change" for fee in SCALP_SHADOW_FEES_BP}
    _require_scalp_contract(set((summary.get("fee_scenarios") or {}).keys()) == expected_fee_keys, "summary.fee_scenarios")
    _require_scalp_contract(database_path.exists(), "database")

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        metadata = connection.execute(
            """
            SELECT schema_version, model_id, model_sha256, fresh_start_utc,
                   order_submission_supported
            FROM observer_metadata WHERE singleton = true
            """
        ).fetchone()
        _require_scalp_contract(metadata is not None, "observer_metadata")
        _require_scalp_contract(metadata[0] == config["observer_schema"], "observer_metadata.schema_version")
        _require_scalp_contract(metadata[1] == config["model_id"], "observer_metadata.model_id")
        _require_scalp_contract(metadata[2] == state.get("model_sha256"), "observer_metadata.model_sha256")
        _require_scalp_contract(bool(metadata[4]) is False, "observer_metadata.order_submission_supported")

        decision_count = int(connection.execute("SELECT count(*) FROM decisions").fetchone()[0])
        latest = connection.execute(
            """
            SELECT timestamp, close, target_position
            FROM decisions ORDER BY timestamp DESC LIMIT 1
            """
        ).fetchone()
        pnl_rows = connection.execute(
            """
            SELECT fee_bp, decision_timestamp, settlement_timestamp,
                   previous_position, position, turnover, gross_return,
                   cost_return, net_return, equity, causal_settlement
            FROM shadow_pnl
            ORDER BY fee_bp, decision_timestamp
            """
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT d.timestamp, d.close, d.available, d.previous_position,
                   d.target_position, d.position_change, p.settlement_timestamp,
                   p.net_return, p.equity
            FROM decisions AS d
            LEFT JOIN shadow_pnl AS p
              ON p.decision_timestamp = d.timestamp AND p.fee_bp = ?
            ORDER BY d.timestamp DESC LIMIT 6
            """,
            [SCALP_SHADOW_DISPLAY_FEE_BP],
        ).fetchall()
    finally:
        connection.close()

    by_fee: dict[float, list[tuple[Any, ...]]] = {fee: [] for fee in SCALP_SHADOW_FEES_BP}
    for row in pnl_rows:
        fee = float(row[0])
        _require_scalp_contract(fee in by_fee, "shadow_pnl.fee_bp")
        _require_scalp_contract(bool(row[9]) and bool(row[9] > 0.0), "shadow_pnl.equity")
        _require_scalp_contract(row[10] is True, "shadow_pnl.causal_settlement")
        by_fee[fee].append(row)
    settled_counts = {len(rows) for rows in by_fee.values()}
    _require_scalp_contract(len(settled_counts) == 1, "shadow_pnl.partial_fee_set")
    settled_intervals = next(iter(settled_counts), 0)
    _require_scalp_contract(decision_count == int(summary.get("decision_count", -1)), "summary.decision_count")
    _require_scalp_contract(settled_intervals == int(summary.get("settled_intervals", -1)), "summary.settled_intervals")

    scenarios = []
    for fee in SCALP_SHADOW_FEES_BP:
        rows = by_fee[fee]
        equities = [float(row[9]) for row in rows]
        peak = 1.0
        max_drawdown = 0.0
        for equity in equities:
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        scenarios.append(
            {
                "fee_bp": fee,
                "compounded_return_pct": ((equities[-1] - 1.0) * 100.0) if equities else 0.0,
                "gross_return_pct": sum(float(row[6]) for row in rows) * 100.0,
                "cost_pct": sum(float(row[7]) for row in rows) * 100.0,
                "max_drawdown_pct": max_drawdown * 100.0,
            }
        )

    display_rows = by_fee[SCALP_SHADOW_DISPLAY_FEE_BP]
    displayed = next(row for row in scenarios if row["fee_bp"] == SCALP_SHADOW_DISPLAY_FEE_BP)
    positioned_intervals = sum(1 for row in display_rows if int(row[4]) != 0)
    position_changes = sum(1 for row in display_rows if float(row[5]) > 0.0)
    equity_rows = display_rows[-180:]
    latest_decision_utc = utc_iso(latest[0]) if latest is not None else None
    latest_feature_completed_utc = utc_iso(stream.get("latest_feature_completed_at_utc"))
    return {
        "contract": {
            "asset": config["asset"],
            "mode": config.get("mode"),
            "symbol": config["symbol"],
            "model_id": config["model_id"],
            "model_sha256": state.get("model_sha256"),
            "parent_model_id": summary.get("parent_model_id"),
            "research_policy_enabled": summary.get("research_policy_enabled", True),
            "dynamic_exit_enabled": summary.get("dynamic_exit_enabled"),
            "evidence_class": summary.get("evidence_class"),
            "actual_execution": False,
            "performance_eligible": False,
            "order_submission_supported": False,
            "fixed_holding_period_used": False,
            "unit_notional": float(summary.get("unit_notional")),
            "display_fee_bp": SCALP_SHADOW_DISPLAY_FEE_BP,
        },
        "health": {
            "latest_feature_completed_utc": latest_feature_completed_utc,
            "stream_age_minutes": utc_age_minutes(latest_feature_completed_utc),
            "latest_decision_utc": latest_decision_utc,
        },
        "summary": {
            "decision_count": decision_count,
            "settled_intervals": settled_intervals,
            "unsettled_decisions": max(0, decision_count - settled_intervals),
            "positioned_intervals": positioned_intervals,
            "position_changes": position_changes,
            "high_risk_bars": int(summary.get("high_risk_bars", 0)),
            "dynamic_exit_enabled": summary.get("dynamic_exit_enabled"),
            "pnl_sample_ready": positioned_intervals > 0,
            "current_position": int(latest[2]) if latest is not None else 0,
            "latest_close": float(latest[1]) if latest is not None else None,
            **displayed,
        },
        "fee_scenarios": scenarios,
        "equity": [
            {
                "ts": utc_iso(row[1]),
                "settlement_ts": utc_iso(row[2]),
                "position": int(row[4]),
                "net_return_pct": float(row[8]) * 100.0,
                "equity": float(row[9]),
                "cumulative_return_pct": (float(row[9]) - 1.0) * 100.0,
            }
            for row in equity_rows
        ],
        "recent_decisions": [
            {
                "ts": utc_iso(row[0]),
                "close": float(row[1]),
                "available": bool(row[2]),
                "previous_position": int(row[3]),
                "target_position": int(row[4]),
                "position_change": int(row[5]),
                "settlement_ts": utc_iso(row[6]),
                "net_return_pct": float(row[7]) * 100.0 if row[7] is not None else None,
                "equity": float(row[8]) if row[8] is not None else None,
            }
            for row in recent_rows
        ],
    }


def btc_multislot_shadow_payload() -> dict[str, Any]:
    state = load_json(BTC_MULTISLOT_SHADOW_STATE_PATH) or {}
    slots = state.get("slots") if isinstance(state.get("slots"), list) else []
    last_bar = state.get("last_bar")
    age_minutes = utc_age_minutes(last_bar, offset_seconds=BTC_MULTISLOT_SHADOW_BAR_SECONDS)
    total_trades = 0
    cumulative_return_pct: float | None = None
    recent_trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    if BTC_MULTISLOT_SHADOW_LEDGER_PATH.exists():
        with BTC_MULTISLOT_SHADOW_LEDGER_PATH.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        total_trades = len(rows)
        equity = 1.0
        for row in rows:
            try:
                return_frac = float(row.get("trade_return_net") or 0.0)
            except (TypeError, ValueError):
                continue
            equity *= 1.0 + return_frac
            equity_curve.append(
                {
                    "ts": row.get("exit_timestamp"),
                    "slot": row.get("slot"),
                    "side": row.get("side"),
                    "trade_return_pct": return_frac * 100.0,
                    "cumulative_return_pct": (equity - 1.0) * 100.0,
                }
            )
        cumulative_return_pct = (equity - 1.0) * 100.0
        equity_curve = equity_curve[-500:]
        for row in rows[-5:]:
            try:
                return_pct = float(row["trade_return_net"]) * 100.0
            except (KeyError, TypeError, ValueError):
                return_pct = None
            recent_trades.append(
                {
                    "slot": row.get("slot"),
                    "side": row.get("side"),
                    "entry_timestamp": row.get("entry_timestamp"),
                    "exit_timestamp": row.get("exit_timestamp"),
                    "entry_price": row.get("entry_price"),
                    "exit_price": row.get("exit_price"),
                    "trade_return_pct": return_pct,
                    "reason": row.get("reason"),
                }
            )
        recent_trades.reverse()
    return {
        "last_bar": last_bar,
        "age_minutes": age_minutes,
        "stale": age_minutes is None or age_minutes >= (BTC_MULTISLOT_SHADOW_BAR_SECONDS / 60.0) * 3,
        "bar_seconds": BTC_MULTISLOT_SHADOW_BAR_SECONDS,
        "slot_count": len(slots),
        "open_slots": sum(1 for s in slots if s),
        "slots": slots,
        "total_trades": total_trades,
        "cumulative_return_pct": cumulative_return_pct,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
    }


def eth_odyssey4_shadow_payload() -> dict[str, Any]:
    state = load_json(ETH_ODYSSEY4_SHADOW_STATE_PATH) or {}
    last_bar = state.get("last_processed_bar_ts")
    age_minutes = utc_age_minutes(last_bar, offset_seconds=ETH_ODYSSEY4_SHADOW_BAR_SECONDS)
    position = state.get("position") if isinstance(state.get("position"), dict) else None
    trades = parse_jsonl(ETH_ODYSSEY4_SHADOW_TRADES_PATH)
    total_trades = len(trades)
    equity = state.get("equity")
    cumulative_return_pct = (float(equity) - 1.0) * 100.0 if isinstance(equity, (int, float)) else None
    mdd_pct = float(state["mdd"]) * 100.0 if isinstance(state.get("mdd"), (int, float)) else None
    recent_trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    running_equity = 1.0
    for row in trades:
        try:
            return_frac = float(row["trade_return_frac"])
        except (KeyError, TypeError, ValueError):
            continue
        running_equity *= 1.0 + return_frac
        equity_curve.append(
            {
                "ts": row.get("exit_ts"),
                "side": row.get("side"),
                "trade_return_pct": return_frac * 100.0,
                "cumulative_return_pct": (running_equity - 1.0) * 100.0,
            }
        )
    equity_curve = equity_curve[-500:]
    for row in trades[-5:]:
        try:
            return_pct = float(row["trade_return_frac"]) * 100.0
        except (KeyError, TypeError, ValueError):
            return_pct = None
        recent_trades.append(
            {
                "source_component": row.get("source_component"),
                "side": row.get("side"),
                "entry_ts": row.get("entry_ts"),
                "exit_ts": row.get("exit_ts"),
                "entry_price": row.get("entry_price"),
                "exit_price": row.get("exit_price"),
                "trade_return_pct": return_pct,
                "reason": row.get("reason"),
            }
        )
    recent_trades.reverse()
    return {
        "last_bar": last_bar,
        "age_minutes": age_minutes,
        "stale": age_minutes is None or age_minutes >= (ETH_ODYSSEY4_SHADOW_BAR_SECONDS / 60.0) * 3,
        "bar_seconds": ETH_ODYSSEY4_SHADOW_BAR_SECONDS,
        "position_side": (position or {}).get("side", 0),
        "position_source_component": (position or {}).get("source_component"),
        "position": position,
        "total_trades": total_trades,
        "cumulative_return_pct": cumulative_return_pct,
        "mdd_pct": mdd_pct,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
        "candidate": state.get("candidate"),
        "order_submission_supported": state.get("order_submission_supported", False),
        "h48qual_guard_active_bars": state.get("h48qual_guard_active_bars", 0),
        "zig075_short_veto_bars": state.get("zig075_short_veto_bars", 0),
        "h48qual_quality_score": state.get("last_h48qual_quality_score"),
        "h48qual_quality_threshold": state.get("last_h48qual_quality_threshold"),
        "zig075_quality_score": state.get("last_zig075_quality_score"),
        "zig075_quality_threshold": state.get("last_zig075_quality_threshold"),
    }


def no_cache(resp: web.StreamResponse) -> web.StreamResponse:
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def json_response(request: web.Request, payload: Any, etag: str) -> web.Response:
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if etag_matches(request, etag):
        return web.Response(status=web.HTTPNotModified.status_code, headers=headers)

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    response = web.Response(body=body, content_type="application/json", headers=headers)
    response.enable_compression()
    return response


def make_app() -> web.Application:
    @web.middleware
    async def static_asset_headers(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        response = await handler(request)
        asset_name = Path(request.path).name
        if asset_name in {"app.js", "styles.css"}:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            response.enable_compression()
        elif asset_name.endswith(".ttf"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    app = web.Application(middlewares=[static_asset_headers])
    json_cache: dict[Path, tuple[tuple[int, int] | None, Any]] = {}
    trade_cache: dict[str, Any] = {
        "signature": None,
        "rows": [],
        "payloads": {},
    }
    event_clients: set[asyncio.Queue[str]] = set()
    latest_event_state: dict[str, Any] | None = None
    latest_event_tickers: dict[str, dict[str, Any]] = {}
    market_history_cache: dict[str, tuple[float, list[dict[str, float | int]]]] = {}
    market_history_locks = {asset: asyncio.Lock() for asset in MARKET_SYMBOLS}
    evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None, "frames": None}
    evidence_signal_lock = asyncio.Lock()
    evidence_signal_provisional_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    evidence_signal_provisional_lock = asyncio.Lock()
    v_rebound_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    v_rebound_lock = asyncio.Lock()
    # 2026-08-31: keyed by asset (was a single shared slot) so an ETH and a BTC request don't
    # evict each other's cached reading -- same per-asset dict/lock shape as market_history_cache/
    # market_history_locks above.
    basis_liquidation_cache: dict[str, dict[str, Any]] = {}
    basis_liquidation_locks = {asset: asyncio.Lock() for asset in COIN_CONFIG}
    liquidation_5m_cache: dict[str, dict[str, Any]] = {}
    liquidation_5m_locks = {asset: asyncio.Lock() for asset in COIN_CONFIG}
    liquidation_direction_cache: dict[str, dict[str, Any]] = {}
    liquidation_direction_locks = {asset: asyncio.Lock() for asset in COIN_CONFIG}
    liquidation_map_cache: dict[str, dict[str, Any]] = {}
    liquidation_map_locks = {asset: asyncio.Lock() for asset in COIN_CONFIG}
    regime_wide24_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    regime_wide24_lock = asyncio.Lock()
    macro_calendar_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    macro_calendar_lock = asyncio.Lock()
    model_indicator_history: deque = deque(maxlen=MODEL_INDICATOR_HISTORY_MAX)
    model_indicator_sample_state: dict[str, float] = {"last_sample_at": 0.0}

    def load_json_cached(path: Path, signature: tuple[int, int] | None = None) -> Any:
        if signature is None:
            signature = file_signature(path)
        cached = json_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        payload = load_json(path)
        json_cache[path] = (signature, payload)
        return payload

    def cached_trade_rows(path: Path, signature: tuple[int, int] | None = None) -> list[dict]:
        if signature is None:
            signature = file_signature(path)
        if trade_cache["signature"] == signature:
            return trade_cache["rows"]

        rows = []
        for row in parse_jsonl(path):
            raw_source = row.get("source", "")
            tagged = {**row, "raw_source": raw_source}
            tagged["source"] = strategy_tag(tagged)
            rows.append(tagged)
        rows.sort(key=row_ts)
        trade_cache.update(signature=signature, rows=rows, payloads={})
        return rows

    def dashboard_state_payload() -> tuple[dict[str, Any], str]:
        state_path = LIVE_DIR / "dashboard_state.json"
        governor_path = LIVE_DIR / "dashboard_state_governor.json"
        dsac_path = LIVE_DIR / "dashboard_state_dsac_compact.json"
        state_sig = file_signature(state_path)
        governor_sig = file_signature(governor_path)
        dsac_sig = file_signature(dsac_path)
        etag = make_etag("state", state_sig, governor_sig, dsac_sig)
        return {
            "state": load_json_cached(state_path, state_sig),
            "compactState": load_json_cached(governor_path, governor_sig) or load_json_cached(dsac_path, dsac_sig),
        }, etag

    async def fetch_market_ticker(session: ClientSession, asset: str, symbol: str) -> tuple[str, dict[str, Any] | None]:
        try:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": symbol},
            ) as response:
                if response.status != web.HTTPOk.status_code:
                    return asset, None
                payload = await response.json()
        except (asyncio.TimeoutError, OSError, ValueError):
            return asset, None
        try:
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError):
            return asset, None
        if price <= 0:
            return asset, None
        return asset, {"symbol": symbol, "price": price, "ts": datetime.now(timezone.utc).isoformat()}

    async def load_market_history_from_evidence_cache(asset: str) -> list[dict[str, float | int]]:
        """ETH/BTC candle-chart data (2026-08-26, user request to de-duplicate) sliced straight out
        of evidence_signal_cache["frames"] instead of a separate Binance klines fetch -- that cache
        already holds EVIDENCE_FETCH_LIMIT (1500) 5m closed bars for both symbols, a strict superset
        of the 100 bars this used to fetch on its own, refreshed every EVIDENCE_SIGNAL_CACHE_SECONDS
        (60s, 5x more often than the old MARKET_HISTORY_CACHE_SECONDS=300s cache as a side effect).
        No separate cache of its own needed -- slicing 100 rows out of an already-in-memory frame is
        cheap enough to just redo on every call."""
        if evidence_signal_cache["frames"] is None:
            await load_evidence_signals()  # first call this process -- warm up the shared history
        frames = evidence_signal_cache["frames"]
        if frames is None:
            raise web.HTTPBadGateway(reason="market_history_upstream_error")
        closed_df, btc_df, _funding_df = frames
        src = closed_df if asset == "eth" else btc_df
        if src is None or src.empty:
            raise web.HTTPBadGateway(reason="market_history_upstream_error")
        return [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for _, row in src.tail(100).iterrows()
        ]

    async def load_market_history(asset: str) -> list[dict[str, float | int]]:
        if asset in ("eth", "btc"):
            return await load_market_history_from_evidence_cache(asset)
        # sol: no evidence-signal history exists for it (those only ever cover ETH+BTC), so this
        # keeps its own independent fetch -- unchanged from before.
        cached = market_history_cache.get(asset)
        now = time.monotonic()
        if cached and now - cached[0] < MARKET_HISTORY_CACHE_SECONDS:
            return cached[1]
        async with market_history_locks[asset]:
            cached = market_history_cache.get(asset)
            now = time.monotonic()
            if cached and now - cached[0] < MARKET_HISTORY_CACHE_SECONDS:
                return cached[1]
            async with ClientSession(timeout=ClientTimeout(total=3)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol": MARKET_SYMBOLS[asset], "interval": "5m", "limit": 100},
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="market_history_upstream_error")
                    rows = await response.json()
            candles = [
                {
                    "time": int(row[0]) // 1000,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                }
                for row in rows
            ]
            market_history_cache[asset] = (time.monotonic(), candles)
            return candles

    async def load_evidence_signals() -> dict[str, Any]:
        """Informational-only reversal-evidence-signal readout for the Snapshot tab -- NOT a
        trading signal (see docstring in the imported module). Mirrors load_market_history()'s
        cache/lock pattern but needs a much longer klines window (EVIDENCE_FETCH_LIMIT bars, to
        warm up orthogonal_combo's EVIDENCE_PCTRANK_WINDOW-bar percentile-rank window) than the
        chart's own /api/market-history (limit=100), so it gets its own cache rather than sharing
        market_history_cache."""
        now = time.monotonic()
        cached = evidence_signal_cache["payload"]
        if cached is not None and now - evidence_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with evidence_signal_lock:
            cached = evidence_signal_cache["payload"]
            if cached is not None and time.monotonic() - evidence_signal_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": EVIDENCE_SIGNAL_SYMBOL,
                        "interval": EVIDENCE_SIGNAL_INTERVAL,
                        "limit": EVIDENCE_FETCH_LIMIT,
                    },
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="evidence_signal_upstream_error")
                    raw = await response.json()
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
            df = pd.DataFrame(raw, columns=cols)
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
                df[c] = df[c].astype("float64")
            df["close_time"] = df["close_time"].astype("int64")
            df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar

            # BTC leg for smt_divergence (2026-08-24) -- failure here must never take down the
            # ETH-only signals, so it's caught and logged, not raised; compute_signals() degrades
            # smt_divergence to not-fired when btc_df is None.
            btc_df = None
            try:
                async with ClientSession(timeout=ClientTimeout(total=10)) as btc_session:
                    async with btc_session.get(
                        "https://fapi.binance.com/fapi/v1/klines",
                        params={
                            "symbol": EVIDENCE_SIGNAL_BTC_SYMBOL,
                            "interval": EVIDENCE_SIGNAL_INTERVAL,
                            "limit": EVIDENCE_FETCH_LIMIT,
                        },
                    ) as btc_response:
                        if btc_response.status == web.HTTPOk.status_code:
                            braw = await btc_response.json()
                            bdf = pd.DataFrame(braw, columns=cols)
                            # open/close cast+retained too (2026-08-26) so load_market_history()
                            # can slice BTC candles straight out of this cache instead of its own
                            # separate klines fetch -- compute_signals() itself still only reads
                            # btc_df's high/low (smt_divergence), the extra columns are unused by it.
                            for c in ("open", "high", "low", "close"):
                                bdf[c] = bdf[c].astype("float64")
                            bdf["close_time"] = bdf["close_time"].astype("int64")
                            bdf["timestamp"] = pd.to_datetime(bdf["open_time"].astype("int64"), unit="ms", utc=True)
                            bdf = bdf.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
                            if len(bdf) and int(bdf.iloc[-1]["close_time"]) >= now_ms:
                                bdf = bdf.iloc[:-1].reset_index(drop=True)
                            btc_df = bdf[["timestamp", "open", "high", "low", "close"]]
            except Exception as btc_exc:  # noqa: BLE001 -- ETH signals must still render this cycle
                print(f"evidence-signal BTC leg failed (smt_divergence family will read as "
                      f"not-fired this cycle): {btc_exc}", flush=True)

            # Funding leg for orthogonal_combo's bottom leg (2026-08-25, merged 2026-08-27 --
            # formerly a separate funding_oscillator_combo signal) -- same fail-soft pattern as
            # the BTC leg above; compute_signals() degrades orthogonal_combo's bottom to its
            # delta_z-only pre-2026-08-27 formula when funding_df is None. funding_z is
            # rolling-z-scored here (not in compute_signals) to match scripts/live_evidence_signal_
            # dashboard_20260823.py::fetch_funding_history verbatim (EVIDENCE_FUNDING_Z_WINDOW/
            # EVIDENCE_FUNDING_Z_MIN_PERIODS imported from there).
            funding_df = None
            try:
                async with ClientSession(timeout=ClientTimeout(total=10)) as funding_session:
                    async with funding_session.get(
                        EVIDENCE_SIGNAL_FUNDING_URL,
                        params={"symbol": EVIDENCE_SIGNAL_SYMBOL, "limit": EVIDENCE_FUNDING_HISTORY_LIMIT},
                    ) as funding_response:
                        if funding_response.status == web.HTTPOk.status_code:
                            fraw = await funding_response.json()
                            fdf = pd.DataFrame(fraw)
                            fdf["calc_time"] = pd.to_datetime(fdf["fundingTime"].astype("int64"), unit="ms", utc=True)
                            fdf["fundingRate"] = fdf["fundingRate"].astype("float64")
                            fdf = fdf.sort_values("calc_time").drop_duplicates("calc_time", keep="last").reset_index(drop=True)
                            fmean = fdf["fundingRate"].rolling(EVIDENCE_FUNDING_Z_WINDOW, min_periods=EVIDENCE_FUNDING_Z_MIN_PERIODS).mean()
                            fstd = fdf["fundingRate"].rolling(EVIDENCE_FUNDING_Z_WINDOW, min_periods=EVIDENCE_FUNDING_Z_MIN_PERIODS).std()
                            fdf["funding_z"] = (fdf["fundingRate"] - fmean) / fstd.replace(0.0, float("nan"))
                            funding_df = fdf[["calc_time", "funding_z"]]
            except Exception as funding_exc:  # noqa: BLE001 -- ETH signals must still render this cycle
                print(f"evidence-signal funding leg failed (orthogonal_combo's bottom leg will "
                      f"degrade to delta_z-only this cycle): {funding_exc}", flush=True)

            sig = await asyncio.to_thread(compute_signals, df, btc_df=btc_df, funding_df=funding_df)
            latest = sig.iloc[-1] if len(sig) else None
            warmed_up = latest is not None and pd.notna(latest.get("p_fast")) and pd.notna(latest.get("p_slow"))
            # taker_delta_z_climax / short_term_return_z / liquidity_sweep / orthogonal_combo /
            # smt_divergence REPLACED with their TabPFN meta-label models' live probability
            # (2026-08-30/31; dalton_rule2_balance_edge removed 2026-08-31, see METALABEL_SIGNALS'
            # own module docstring) -- reuses this cycle's already-fetched `df` and already-computed
            # `latest` fire state, no separate fetch/compute_signals() call. Fail-soft: a GPU/TabPFN
            # hiccup must not block the other signals from rendering.
            metalabels: dict[str, dict] = {}
            if warmed_up:
                try:
                    metalabels = await asyncio.to_thread(compute_evidence_signal_metalabels, df, latest)
                except Exception as metalabel_exc:  # noqa: BLE001
                    print(f"evidence-signal metalabel leg failed (taker_delta_z_climax/"
                          f"short_term_return_z/liquidity_sweep/orthogonal_combo/smt_divergence will "
                          f"read as not-fired this cycle): {metalabel_exc}", flush=True)
            signals_payload = []
            for name, description in EVIDENCE_SIGNAL_ORDER:
                bcol, tcol = f"bottom_{name}", f"top_{name}"
                # _active = per-signal sustain window (2026-08-24 default 20min/4 bars; 2026-08-30:
                # taker_delta_z_climax/short_term_return_z instead use their own trained HORIZON,
                # 2h/1h) -- rolling-max of the raw bcol/tcol firing column, not a new/looser firing
                # condition (see compute_signals() docstring, SUSTAIN_BARS_OVERRIDE). last_fired_ts
                # always reads the RAW column so it keeps reporting the true original firing bar
                # even while _active keeps the chip lit.
                bacol, tacol = f"{bcol}_active", f"{tcol}_active"
                entry = {
                    "name": name,
                    "description": description,
                    "bottom_fired": bool(latest[bacol]) if warmed_up else None,
                    "bottom_last_fired_ts": _evidence_last_fired_ts(sig[bcol], latest) if warmed_up else None,
                    "top_fired": bool(latest[tacol]) if warmed_up else None,
                    "top_last_fired_ts": _evidence_last_fired_ts(sig[tcol], latest) if warmed_up else None,
                    # Oldest-to-newest, for the Snapshot tab's activity-strip graph (one cell/bar).
                    "bottom_history": sig[bacol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                    "top_history": sig[tacol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                }
                if name in metalabels:
                    entry["model_proba"] = metalabels[name]["proba"]
                    entry["model_side"] = metalabels[name]["side"]
                signals_payload.append(entry)
            # session_volatility_alert/macro_event_alert moved to /api/session-alerts (2026-08-27)
            # -- they need much faster polling than this endpoint's 5min client-side cadence, see
            # api_session_alerts()'s docstring.
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "latest_bar_utc": utc_iso(latest["timestamp"]) if latest is not None else None,
                "price": float(latest["close"]) if latest is not None else None,
                "bars_loaded": int(len(sig)),
                "warmed_up": bool(warmed_up),
                "net_score": int(latest["net_score"]) if warmed_up else None,
                "bottom_votes": int(latest["bottom_votes"]) if warmed_up else None,
                "top_votes": int(latest["top_votes"]) if warmed_up else None,
                "signals": signals_payload,
            }
            # Closed-bar frames only (forming bar already dropped above) -- reused by
            # load_evidence_signals_provisional() so its ~10s poll doesn't re-pull the full
            # EVIDENCE_FETCH_LIMIT-bar history, just the one new forming-bar row.
            evidence_signal_cache["frames"] = (
                df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]],
                btc_df,
                funding_df,
            )
            evidence_signal_cache["ts"] = time.monotonic()
            evidence_signal_cache["payload"] = payload
            return payload

    async def load_evidence_signals_provisional() -> dict[str, Any]:
        """Live PREVIEW of the CURRENTLY-FORMING 5m bar's evidence-signal state (2026-08-26, user
        request: "값이 아직 채워지지 않았다면 진행 중인 값으로 대체"). load_evidence_signals()
        above is UNCHANGED and remains the only signal with a real validated lift track record
        (3.51x etc., measured on closed bars only) -- this function is a methodologically DIFFERENT
        reading: it appends the in-progress bar's current (partial) O/H/L/C/volume onto the last
        confirmed closed-bar history and reruns compute_signals(), so the result can flicker
        (high/low/volume keep changing until the bar closes) and has no lift validation of its own.
        The frontend MUST render this with an explicit "미확정(진행중)" marker, never merged into
        or replacing the confirmed reading -- see scripts/live_evidence_signal_dashboard_20260823.py
        module docstring for why the 9 formulas fundamentally need closed-bar aggregates (a bar's
        FULL volume, a bar's FINAL high/low), not "current price".
        Cheap by design: reuses evidence_signal_cache["frames"] (closed-bar history, refreshed at
        its own 60s cadence) instead of re-fetching EVIDENCE_FETCH_LIMIT bars every ~10s -- only
        fetches the 1-2 most recent bars to read the forming bar's live state."""
        now = time.monotonic()
        cached = evidence_signal_provisional_cache["payload"]
        if cached is not None and now - evidence_signal_provisional_cache["ts"] < EVIDENCE_SIGNAL_PROVISIONAL_CACHE_SECONDS:
            return cached
        async with evidence_signal_provisional_lock:
            cached = evidence_signal_provisional_cache["payload"]
            if cached is not None and time.monotonic() - evidence_signal_provisional_cache["ts"] < EVIDENCE_SIGNAL_PROVISIONAL_CACHE_SECONDS:
                return cached

            def _store(payload: dict[str, Any]) -> dict[str, Any]:
                evidence_signal_provisional_cache["ts"] = time.monotonic()
                evidence_signal_provisional_cache["payload"] = payload
                return payload

            if evidence_signal_cache["frames"] is None:
                await load_evidence_signals()  # first call this process -- warm up closed-bar history
            frames = evidence_signal_cache["frames"]
            if frames is None:
                return _store({"available": False, "reason": "confirmed_history_not_ready"})
            closed_df, btc_df, funding_df = frames

            async def _fetch_forming_bar(symbol: str) -> pd.DataFrame | None:
                """1-row frame for `symbol`'s currently-forming bar, or None on any failure/bar-
                boundary -- fail-soft by design (never raises) so a BTC-leg hiccup degrades
                smt_divergence's BTC side back to closed-only data this cycle instead of taking
                down the whole provisional preview, mirroring load_evidence_signals()'s own
                BTC-leg fail-soft handling above."""
                try:
                    async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                        async with session.get(
                            "https://fapi.binance.com/fapi/v1/klines",
                            params={"symbol": symbol, "interval": EVIDENCE_SIGNAL_INTERVAL, "limit": 2},
                        ) as response:
                            if response.status != web.HTTPOk.status_code:
                                return None
                            raw = await response.json()
                except Exception as exc:  # noqa: BLE001 -- forming-bar preview must never 500
                    print(f"evidence-signal provisional forming-bar fetch failed ({symbol}): {exc}", flush=True)
                    return None
                cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
                recent = pd.DataFrame(raw, columns=cols)
                for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
                    recent[c] = recent[c].astype("float64")
                recent["close_time"] = recent["close_time"].astype("int64")
                recent["timestamp"] = pd.to_datetime(recent["open_time"].astype("int64"), unit="ms", utc=True)
                now_ms = int(time.time() * 1000)
                forming = recent[recent["close_time"] >= now_ms]
                if forming.empty:
                    return None
                return forming.iloc[[-1]][["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]

            eth_forming = await _fetch_forming_bar(EVIDENCE_SIGNAL_SYMBOL)
            if (
                eth_forming is None
                or closed_df.empty
                or eth_forming.iloc[-1]["timestamp"] <= closed_df.iloc[-1]["timestamp"]
            ):
                # Fetch failed, or right at a bar boundary -- no bar currently forming, nothing to preview.
                return _store({"available": False, "reason": "no_forming_bar"})
            combined = pd.concat([closed_df, eth_forming], ignore_index=True)

            # BTC leg (2026-08-26, user request "BTC도 진행값으로") -- same forming-bar substitution
            # as ETH, kept in its own fail-soft branch: if this fetch fails or lands right on a bar
            # boundary, smt_divergence's BTC side just falls back to the last CONFIRMED BTC bar
            # (still correct, just one bar less "live" than its ETH side that cycle) rather than
            # failing the whole preview.
            btc_combined = btc_df
            if btc_df is not None and len(btc_df):
                btc_forming = await _fetch_forming_bar(EVIDENCE_SIGNAL_BTC_SYMBOL)
                if btc_forming is not None and btc_forming.iloc[-1]["timestamp"] > btc_df.iloc[-1]["timestamp"]:
                    btc_combined = pd.concat(
                        [btc_df, btc_forming[["timestamp", "high", "low"]]], ignore_index=True,
                    )

            sig = await asyncio.to_thread(compute_signals, combined, btc_df=btc_combined, funding_df=funding_df)
            latest = sig.iloc[-1]
            warmed_up = bool(pd.notna(latest.get("p_fast")) and pd.notna(latest.get("p_slow")))
            signals_payload = [
                {
                    "name": name,
                    # Deliberately the RAW (non-sustained) columns, not the confirmed path's
                    # *_active rolling-max -- this is meant to read "is it true on the forming bar
                    # right now", not a smoothed multi-bar window.
                    "bottom_fired": bool(latest[f"bottom_{name}"]) if warmed_up else None,
                    "top_fired": bool(latest[f"top_{name}"]) if warmed_up else None,
                }
                for name, _description in EVIDENCE_SIGNAL_ORDER
            ]
            bar_open = eth_forming.iloc[0]["timestamp"]
            elapsed_s = max(0, int((datetime.now(timezone.utc) - bar_open.to_pydatetime()).total_seconds()))
            return _store({
                "available": True,
                "bar_open_utc": utc_iso(bar_open),
                "bar_elapsed_seconds": elapsed_s,
                "price": float(latest["close"]),
                "warmed_up": warmed_up,
                # 2026-08-26: whether the BTC leg (smt_divergence's cross-asset check) got a live
                # forming bar this cycle, or fell back to the last CONFIRMED BTC bar -- exposed so
                # the frontend/anyone reading the raw payload can tell, same transparency principle
                # as the "미확정" labeling itself (never silently claim more liveness than delivered).
                "btc_leg_live": bool(btc_combined is not btc_df),
                "net_score": int(latest["net_score"]) if warmed_up else None,
                "bottom_votes": int(latest["bottom_votes"]) if warmed_up else None,
                "top_votes": int(latest["top_votes"]) if warmed_up else None,
                "signals": signals_payload,
            })

    async def load_v_rebound_signal() -> dict[str, Any]:
        """유동성스윕 반등예측 event-triggered signal -- see
        scripts/live_eth_sweep_v_rebound_signal_20260829.py docstring for the VAL/OOS/holdout-
        validated TabPFN model and why this is computed HERE (dashboard-side) rather than by
        trading_bot.py. Each call re-fits TabPFN on its frozen historical context (~3s measured
        on this server's GPU, 2026-08-29) -- asyncio.to_thread so that never stalls the event loop,
        same reasoning as load_evidence_signals() above."""
        now = time.monotonic()
        cached = v_rebound_cache["payload"]
        if cached is not None and now - v_rebound_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with v_rebound_lock:
            cached = v_rebound_cache["payload"]
            if cached is not None and time.monotonic() - v_rebound_cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(compute_eth_sweep_v_rebound_signal)
            v_rebound_cache["ts"] = time.monotonic()
            v_rebound_cache["payload"] = payload
            return payload

    async def load_basis_liquidation_signal(asset: str = "eth") -> dict[str, Any]:
        """베이시스 청산압박 model indicator -- see scripts/live_spot_perp_basis_signal_20260827.py
        docstring for the liquidation-crowding validation (exploratory, ~1 month) and why this is
        computed HERE (dashboard-side, own live spot+perp klines fetch) rather than by
        trading_bot.py. asyncio.to_thread so the two blocking HTTP calls inside
        compute_basis_liquidation_signal() never stall this process's event loop, same reasoning
        as load_evidence_signals() above.

        asset: 2026-08-31, BTC added -- the underlying validation (basis_z48 extreme ->
        forward liquidation-volume tilt) was only ever measured on ETH; BTC's reading is exposed
        with the same exploratory caveat, not a re-validated one (see design doc section 6.5)."""
        now = time.monotonic()
        cache = basis_liquidation_cache.setdefault(asset, {"ts": 0.0, "payload": None})
        cached = cache["payload"]
        if cached is not None and now - cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with basis_liquidation_locks[asset]:
            cached = cache["payload"]
            if cached is not None and time.monotonic() - cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(
                compute_basis_liquidation_signal, symbol=COIN_CONFIG[asset]["binance_symbol"]
            )
            cache["ts"] = time.monotonic()
            cache["payload"] = payload
            return payload

    async def load_liquidation_5m_signal(asset: str = "eth") -> dict[str, Any]:
        """Liquidation $ aggregate (BAR_MINUTES=15 rolling bar, despite the module's "_5m" filename
        -- widened 2026-08-25, see that script's docstring) for the Snapshot tab's liquidation
        gauge. Underlying duckdb gets a new row once per minute (tail_risk_interceptor.py's own
        insert cadence) and compute_liquidation_5m_signal() sums whatever's landed in the current
        bar so far -- a genuine incremental accumulator, not a bar-close-only reading -- so
        LIQUIDATION_5M_SIGNAL_CACHE_SECONDS (10s, 2026-08-26 user request, own dedicated constant)
        gives real reduced staleness rather than just re-serving an unchanged value. Same
        asyncio.to_thread reasoning as load_evidence_signals() above.

        asset: 2026-08-31, BTC added -- see coin_config.py for BTC's separate tail-risk file."""
        now = time.monotonic()
        cache = liquidation_5m_cache.setdefault(asset, {"ts": 0.0, "payload": None})
        cached = cache["payload"]
        if cached is not None and now - cache["ts"] < LIQUIDATION_5M_SIGNAL_CACHE_SECONDS:
            return cached
        async with liquidation_5m_locks[asset]:
            cached = cache["payload"]
            if cached is not None and time.monotonic() - cache["ts"] < LIQUIDATION_5M_SIGNAL_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(compute_liquidation_5m_signal, coin=asset)
            cache["ts"] = time.monotonic()
            cache["payload"] = payload
            return payload

    async def load_liquidation_direction_signal(asset: str = "eth") -> dict[str, Any]:
        """Directional-only liquidation tilt (liq_net_z_12, contrarian sign) -- model-indicator
        tier, no PnL/economic claim. See scripts/live_liquidation_direction_signal_20260825.py
        docstring. Same 60s cache reasoning as load_liquidation_5m_signal() above (underlying data
        updates once per minute).

        asset: 2026-08-31, BTC added -- see coin_config.py for BTC's separate tail-risk file."""
        now = time.monotonic()
        cache = liquidation_direction_cache.setdefault(asset, {"ts": 0.0, "payload": None})
        cached = cache["payload"]
        if cached is not None and now - cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
            return cached
        async with liquidation_direction_locks[asset]:
            cached = cache["payload"]
            if cached is not None and time.monotonic() - cache["ts"] < EVIDENCE_SIGNAL_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(compute_liquidation_direction_signal, coin=asset)
            cache["ts"] = time.monotonic()
            cache["payload"] = payload
            return payload

    async def load_liquidation_map(asset: str = "eth") -> dict[str, Any]:
        """Snapshot-tab liquidation map (estimated support/resistance) -- see
        scripts/live_liquidation_map_20260824.py docstring for the estimation methodology and its
        caveats. Mirrors load_evidence_signals()'s klines-fetch/cache pattern (own cache, since
        this needs a much longer 1h lookback than the chart's own /api/market-history).

        asset: 2026-08-31, BTC added. compute_spliced_levels()/compute_spliced_heatmap_history()
        take a plain OHLCV dataframe -- no code change needed there, only the klines fetch below
        swaps symbol. BIN_WIDTH_PCT/LOOKBACK_HOURS/etc. in that module are still ETH-tuned
        constants (see design doc section 5) -- BTC's map uses the same constants, unvalidated for
        BTC's own liquidity/volatility."""
        now = time.monotonic()
        cache = liquidation_map_cache.setdefault(asset, {"ts": 0.0, "payload": None})
        cached = cache["payload"]
        if cached is not None and now - cache["ts"] < LIQUIDATION_MAP_CACHE_SECONDS:
            return cached
        async with liquidation_map_locks[asset]:
            cached = cache["payload"]
            if cached is not None and time.monotonic() - cache["ts"] < LIQUIDATION_MAP_CACHE_SECONDS:
                return cached
            async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": COIN_CONFIG[asset]["binance_symbol"],
                        "interval": LIQUIDATION_MAP_INTERVAL,
                        "limit": LIQUIDATION_MAP_FETCH_LIMIT,
                    },
                ) as response:
                    if response.status != web.HTTPOk.status_code:
                        raise web.HTTPBadGateway(reason="liquidation_map_upstream_error")
                    raw = await response.json()
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
            df = pd.DataFrame(raw, columns=cols)
            for c in ("high", "low", "close", "volume"):
                df[c] = df[c].astype("float64")
            df["close_time"] = df["close_time"].astype("int64")
            df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar

            current_price = float(df["close"].iloc[-1]) if len(df) else 0.0
            payload = await asyncio.to_thread(
                compute_spliced_levels, df.tail(LIQUIDATION_MAP_LOOKBACK_HOURS).reset_index(drop=True), current_price
            )
            # Full df (not the LIQUIDATION_MAP_LOOKBACK_HOURS-trimmed tail above) -- the earliest
            # snapshot in the history still needs its own full LIQUIDATION_MAP_LOOKBACK_HOURS window,
            # so compute_spliced_heatmap_history() needs lookback+display+slack hours of input, not
            # just the single latest snapshot's lookback (see LIQUIDATION_MAP_FETCH_LIMIT's comment).
            payload["heatmap_history"] = await asyncio.to_thread(
                compute_spliced_heatmap_history, df, current_price, LIQUIDATION_MAP_LOOKBACK_HOURS, LIQUIDATION_MAP_DISPLAY_HOURS
            )
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            cache["ts"] = time.monotonic()
            cache["payload"] = payload
            return payload

    async def load_regime_wide24() -> dict[str, Any]:
        """wide24 HMM regime overlay for the Snapshot tab's liquidation-map chart -- see
        scripts/live_regime_wide24_signal_20260826.py docstring. compute_regime_wide24_signal()
        itself never raises (degrades to warmed_up=False), and its own fetch/compute is blocking
        (requests + pandas/HMM), so it's offloaded via asyncio.to_thread same as
        compute_liquidation_levels() above rather than converted to aiohttp -- keeps the ported
        logic identical to the validated scratchpad script it came from."""
        now = time.monotonic()
        cached = regime_wide24_cache["payload"]
        if cached is not None and now - regime_wide24_cache["ts"] < REGIME_WIDE24_CACHE_SECONDS:
            return cached
        async with regime_wide24_lock:
            cached = regime_wide24_cache["payload"]
            if cached is not None and time.monotonic() - regime_wide24_cache["ts"] < REGIME_WIDE24_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(compute_regime_wide24_signal)
            regime_wide24_cache["ts"] = time.monotonic()
            regime_wide24_cache["payload"] = payload
            return payload

    async def load_macro_calendar() -> dict[str, Any]:
        """US macro/corporate event calendar for the Snapshot tab -- see scripts/live_macro_
        calendar_20260826.py docstring for the 6 sources. compute_macro_calendar() is blocking
        (requests, no aiohttp) and never raises (each source degrades independently), same
        asyncio.to_thread pattern as load_regime_wide24() above."""
        now = time.monotonic()
        cached = macro_calendar_cache["payload"]
        if cached is not None and now - macro_calendar_cache["ts"] < MACRO_CALENDAR_CACHE_SECONDS:
            return cached
        async with macro_calendar_lock:
            cached = macro_calendar_cache["payload"]
            if cached is not None and time.monotonic() - macro_calendar_cache["ts"] < MACRO_CALENDAR_CACHE_SECONDS:
                return cached
            payload = await asyncio.to_thread(
                compute_macro_calendar, os.getenv("FRED_API_KEY"), os.getenv("EIA_API_KEY"), os.getenv("FINNHUB_API_KEY")
            )
            macro_calendar_cache["ts"] = time.monotonic()
            macro_calendar_cache["payload"] = payload
            return payload

    async def publish_dashboard_events(app: web.Application) -> None:
        nonlocal latest_event_state, latest_event_tickers
        last_state_etag = ""
        timeout = ClientTimeout(total=2)
        async with ClientSession(timeout=timeout) as session:
            while True:
                started = time.monotonic()
                try:
                    state_payload, state_etag = dashboard_state_payload()
                    if started - model_indicator_sample_state["last_sample_at"] >= MODEL_INDICATOR_SAMPLE_SECONDS:
                        model_indicator_sample_state["last_sample_at"] = started
                        raw_state = (state_payload or {}).get("state") or {}
                        model_indicator_history.append({
                            "sampled_at": datetime.now(timezone.utc).isoformat(),
                            "microstructure": raw_state.get("microstructure") or {},
                            "tail_risk": raw_state.get("tail_risk") or {},
                        })
                    ticker_rows = await asyncio.gather(
                        *(fetch_market_ticker(session, asset, symbol) for asset, symbol in MARKET_SYMBOLS.items())
                    )
                    latest_event_tickers = {
                        asset: ticker for asset, ticker in ticker_rows if ticker is not None
                    }
                    state_changed = state_etag != last_state_etag
                    if state_changed:
                        latest_event_state = state_payload
                        last_state_etag = state_etag
                    payload = {
                        "state": state_payload if state_changed else None,
                        "tickers": latest_event_tickers,
                    }
                    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    for queue in tuple(event_clients):
                        if queue.full():
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        queue.put_nowait(encoded)
                except Exception as exc:  # noqa: BLE001 -- one bad cycle (a malformed state file, a
                    # transient ticker/gather hiccup, ...) must not permanently kill this loop: it is
                    # the sole source of SSE pushes for every connected browser tab (Ops/Snapshot
                    # alike), so an uncaught exception here would silently freeze everyone's live
                    # updates until the next full server restart. asyncio.CancelledError subclasses
                    # BaseException, not Exception, so server shutdown (stop_dashboard_events's
                    # task.cancel()) still propagates through this unaffected.
                    print(f"publish_dashboard_events cycle failed (will retry next cycle): {exc}", flush=True)
                await asyncio.sleep(max(0.0, EVENT_POLL_SECONDS - (time.monotonic() - started)))

    async def start_dashboard_events(app: web.Application) -> None:
        app["dashboard_event_task"] = asyncio.create_task(publish_dashboard_events(app))

    async def stop_dashboard_events(app: web.Application) -> None:
        task = app["dashboard_event_task"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def supervised_processes(specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        pids: dict[str, int] = {}
        for proc_dir in Path("/proc").glob("[0-9]*"):
            if len(pids) == len(specs):
                break
            try:
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            for name, signature in specs:
                if name not in pids and signature in cmdline:
                    pids[name] = int(proc_dir.name)
        return [
            {"name": name, "status": "RUNNING", "pid": pids[name]}
            if name in pids
            else {"name": name, "status": "STOPPED", "pid": None}
            for name, _ in specs
        ]

    async def index(_: web.Request) -> web.Response:
        raise web.HTTPFound("/dashboard/live/")

    async def dashboard_index(_: web.Request) -> web.FileResponse:
        response = web.FileResponse(DASHBOARD_DIR / "index.html")
        response.enable_compression()
        return no_cache(response)

    async def api_state(request: web.Request) -> web.Response:
        payload, etag = dashboard_state_payload()
        return json_response(request, payload, etag)

    async def api_events(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=web.HTTPOk.status_code,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        event_clients.add(queue)
        initial_payload = {"state": latest_event_state, "tickers": latest_event_tickers}
        try:
            await response.write(f"data: {json.dumps(initial_payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8"))
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20)
                    await response.write(f"data: {payload}\n\n".encode("utf-8"))
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            event_clients.discard(queue)
        return response

    async def api_market_history(request: web.Request) -> web.Response:
        asset = request.query.get("asset", "").lower()
        if asset not in MARKET_SYMBOLS:
            raise web.HTTPBadRequest(reason="unsupported_market_history_asset")
        candles = await load_market_history(asset)
        return web.json_response({"asset": asset, "candles": candles}, headers={"Cache-Control": "no-cache"})

    async def api_model_indicator_history(request: web.Request) -> web.Response:
        return web.json_response(
            {"samples": list(model_indicator_history), "sample_interval_seconds": MODEL_INDICATOR_SAMPLE_SECONDS},
            headers={"Cache-Control": "no-cache"},
        )

    async def api_evidence_signals(request: web.Request) -> web.Response:
        try:
            payload = await load_evidence_signals()
        except web.HTTPBadGateway:
            return web.json_response(
                {"error": "evidence_signal_upstream_error", "detail": "Binance klines fetch failed."},
                status=web.HTTPBadGateway.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_evidence_signals_provisional(request: web.Request) -> web.Response:
        try:
            payload = await load_evidence_signals_provisional()
        except web.HTTPBadGateway:
            return web.json_response(
                {"available": False, "error": "evidence_signal_provisional_upstream_error"},
                status=web.HTTPBadGateway.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_v_rebound_signal(request: web.Request) -> web.Response:
        payload = await load_v_rebound_signal()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    def _query_coin_asset(request: web.Request) -> str:
        """Shared `?asset=` parsing for the 4 Snapshot-tab signals wired to multiple coins
        (2026-08-31) -- raises the same 400 shape as api_market_history()'s existing
        unsupported-asset check."""
        asset = request.query.get("asset", "eth").lower()
        if asset not in COIN_CONFIG:
            raise web.HTTPBadRequest(reason="unsupported_asset")
        return asset

    async def api_basis_liquidation_signal(request: web.Request) -> web.Response:
        payload = await load_basis_liquidation_signal(_query_coin_asset(request))
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_liquidation_5m_signal(request: web.Request) -> web.Response:
        payload = await load_liquidation_5m_signal(_query_coin_asset(request))
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_liquidation_direction_signal(request: web.Request) -> web.Response:
        payload = await load_liquidation_direction_signal(_query_coin_asset(request))
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_liquidation_map(request: web.Request) -> web.Response:
        try:
            payload = await load_liquidation_map(_query_coin_asset(request))
        except web.HTTPBadGateway:
            return web.json_response(
                {"error": "liquidation_map_upstream_error", "detail": "Binance klines fetch failed."},
                status=web.HTTPBadGateway.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_regime_wide24(request: web.Request) -> web.Response:
        payload = await load_regime_wide24()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_macro_calendar(request: web.Request) -> web.Response:
        payload = await load_macro_calendar()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_liq_burst_state(request: web.Request) -> web.Response:
        # load_json_cached() keys off (mtime, size), not a timer -- so this serves the freshest
        # write tail_risk_interceptor.py has made (event-triggered, see its _write_liq_burst_state()
        # docstring) without needing its own cache TTL/lock here.
        payload = load_json_cached(LIQ_BURST_STATE_PATH)
        if not payload:
            return web.json_response({"available": False}, headers={"Cache-Control": "no-cache"})
        # tail_risk_interceptor.py's _write_liq_burst_state() never sets "available" itself (it
        # always writes on success) -- the frontend's renderLiqBurstAlert() checks payload.available
        # to distinguish this from the {"available": False} fallback above, so stamp it here.
        return web.json_response({**payload, "available": True}, headers={"Cache-Control": "no-cache"})

    async def api_session_alerts(request: web.Request) -> web.Response:
        """Split out of /api/evidence-signals (2026-08-27, user report: badges only updated on a
        manual page reload) -- both alerts were piggybacking on that endpoint's payload, which the
        FRONTEND only re-fetches every EVIDENCE_POLL_MS=5min (fine for 5-min-candle-driven evidence
        signals, much too slow for a +-30min event window someone is watching approach in real
        time). This endpoint is cheap (pure calendar math + a read of load_macro_calendar()'s own
        6h-cached event list, no new external I/O) so the frontend can poll it far more often
        without adding real load."""
        macro_cal = await load_macro_calendar()
        payload = {
            "session_volatility_alert": compute_session_volatility_alert(),
            "macro_event_alert": compute_macro_event_alert(macro_cal.get("events", [])),
        }
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_trades(request: web.Request) -> web.Response:
        source_filter = request.query.get("source", "ALL").upper()
        journal_path = LIVE_DIR / "trade_journal.jsonl"
        signature = file_signature(journal_path)
        etag = make_etag("trades", signature, source_filter)
        if etag_matches(request, etag):
            return json_response(request, None, etag)

        rows = cached_trade_rows(journal_path, signature)
        payloads = trade_cache["payloads"]
        if source_filter not in payloads:
            payloads[source_filter] = {
                "rows": rows,
                "equity": equity_series(rows, source_filter),
            }
        return json_response(request, payloads[source_filter], etag)

    async def api_ops_status(request: web.Request) -> web.Response:
        ops_dir = LIVE_DIR / "ops_watchdog"
        health_path = ops_dir / "health_snapshot.json"
        heartbeat_path = ops_dir / "watchdog_heartbeat.json"
        state_path = ops_dir / "state.json"
        health_sig = file_signature(health_path)
        heartbeat_sig = file_signature(heartbeat_path)
        state_sig = file_signature(state_path)
        etag = make_etag("ops-status", health_sig, heartbeat_sig, state_sig)
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "health": load_json_cached(health_path, health_sig) or {},
            "heartbeat": load_json_cached(heartbeat_path, heartbeat_sig) or {},
            # Match the managed process itself, not the supervisor that launched it --
            # these run under systemd now (previously bash _supervise.sh), and matching
            # the old bash wrapper's argv left this permanently reporting STOPPED after
            # the migration even though everything was healthy.
            "supervisors": supervised_processes([
                ("trading_bot", "trading_bot.py"),
                ("ops_watchdog", "ops_watchdog.py"),
            ]),
        }
        return json_response(request, payload, etag)

    async def api_scalp_shadow(request: web.Request) -> web.Response:
        asset = request.query.get("asset", "eth").lower()
        config = SCALP_SHADOW_ASSETS.get(asset)
        if config is None:
            return web.json_response(
                {"error": "unsupported_scalp_shadow_asset"},
                status=web.HTTPBadRequest.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        state_path = LIVE_DIR / config["state_file"]
        database_path = LIVE_DIR / config["database_file"]
        etag = make_etag(
            "scalp-shadow",
            asset,
            file_signature(state_path),
            file_signature(database_path),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        try:
            payload = scalp_shadow_payload(LIVE_DIR, asset)
        except Exception as exc:
            print(f"Scalp shadow dashboard contract error: {exc}", flush=True)
            return web.json_response(
                {
                    "error": "scalp_shadow_contract_error",
                    "detail": "Scalp shadow data contract is unavailable.",
                },
                status=web.HTTPServiceUnavailable.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return json_response(request, payload, etag)

    async def api_scalp_reuse_shadow(request: web.Request) -> web.Response:
        mode = request.query.get("mode", "eth_lifecycle").lower()
        config = SCALP_REUSE_MODES.get(mode)
        if config is None:
            return web.json_response(
                {"error": "unsupported_scalp_reuse_mode"},
                status=web.HTTPBadRequest.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        state_path = LIVE_DIR / config["state_file"]
        database_path = LIVE_DIR / config["database_file"]
        etag = make_etag(
            "scalp-reuse-shadow",
            mode,
            file_signature(state_path),
            file_signature(database_path),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        try:
            payload = scalp_shadow_payload(LIVE_DIR, mode, SCALP_REUSE_MODES)
        except Exception as exc:
            print(f"Scalp reuse shadow dashboard contract error: {exc}", flush=True)
            return web.json_response(
                {
                    "error": "scalp_reuse_shadow_contract_error",
                    "detail": "Scalp reuse shadow data contract is unavailable.",
                },
                status=web.HTTPServiceUnavailable.status_code,
                headers={"Cache-Control": "no-cache"},
            )
        return json_response(request, payload, etag)

    async def api_btc_multislot_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "btc-multislot-shadow",
            file_signature(BTC_MULTISLOT_SHADOW_STATE_PATH),
            file_signature(BTC_MULTISLOT_SHADOW_LEDGER_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, btc_multislot_shadow_payload(), etag)

    async def api_eth_odyssey4_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "eth-odyssey4-shadow",
            file_signature(ETH_ODYSSEY4_SHADOW_STATE_PATH),
            file_signature(ETH_ODYSSEY4_SHADOW_TRADES_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, eth_odyssey4_shadow_payload(), etag)

    app.router.add_get("/", index)
    app.router.add_get("/dashboard/live", dashboard_index)
    app.router.add_get("/dashboard/live/", dashboard_index)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/market-history", api_market_history)
    app.router.add_get("/api/evidence-signals", api_evidence_signals)
    app.router.add_get("/api/evidence-signals-provisional", api_evidence_signals_provisional)
    app.router.add_get("/api/v-rebound-signal", api_v_rebound_signal)
    app.router.add_get("/api/basis-liquidation-signal", api_basis_liquidation_signal)
    app.router.add_get("/api/liquidation-5m-signal", api_liquidation_5m_signal)
    app.router.add_get("/api/liquidation-direction-signal", api_liquidation_direction_signal)
    app.router.add_get("/api/liquidation-map", api_liquidation_map)
    app.router.add_get("/api/regime-wide24", api_regime_wide24)
    app.router.add_get("/api/macro-calendar", api_macro_calendar)
    app.router.add_get("/api/liq-burst-state", api_liq_burst_state)
    app.router.add_get("/api/session-alerts", api_session_alerts)
    app.router.add_get("/api/model-indicator-history", api_model_indicator_history)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/ops-status", api_ops_status)
    app.router.add_get("/api/scalp-shadow", api_scalp_shadow)
    app.router.add_get("/api/scalp-reuse-shadow", api_scalp_reuse_shadow)
    app.router.add_get("/api/btc-multislot-shadow", api_btc_multislot_shadow)
    app.router.add_get("/api/eth-odyssey4-shadow", api_eth_odyssey4_shadow)
    app.router.add_static("/dashboard/live/", DASHBOARD_DIR, show_index=True)
    app.router.add_static("/data/live/", LIVE_DIR, show_index=False)
    app.on_startup.append(start_dashboard_events)
    app.on_cleanup.append(stop_dashboard_events)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic live dashboard server.")
    parser.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8787")))
    args = parser.parse_args()

    print(f"Serving dashboard at http://{args.host}:{args.port}/dashboard/live/", flush=True)
    web.run_app(
        make_app(),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
