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
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
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
# BTC 코인 페이지의 메인 증거신호 패널(2026-09-02) -- 이전엔 코인탭과 무관하게 항상 ETH의
# /api/evidence-signals를 보여줬다(사용자 신고: "비트코인 페이지에 이더리움 증거신호가 나온다").
# ETH의 compute_signals()를 재사용하되 BTC 자체 그리드스크린 K/HORIZON·TabPFN 모델로 채점하는
# compute_btc_evidence_signals_panel()을 새로 추가(기존 compute_btc_evidence_signals()는 섀도우
# 러너 전용 다른 모양이라 그대로 둠). 자세한 내용은 그 함수 docstring 참고.
from scripts.live_btc_evidence_signal_metalabel_20260902 import compute_btc_evidence_signals_panel  # noqa: E402
# 2026-09-03: XRP 증거신호 5종. XRP 페이지는 그동안 ETH 신호를 그대로 보여주고 있었다
# (BTC에서 사용자가 신고했던 것과 같은 버그의 XRP판). 자산별 라우팅으로 해소한다.
from scripts.live_xrp_evidence_signal_metalabel_20260903 import compute_xrp_evidence_signals_panel  # noqa: E402
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
# BTC-native regime scorer (2026-09-02). Until now the Snapshot tab's BTC ribbon was a hard-coded
# grey "model not available" band -- app.js gated the ribbon on activeSnapshotAsset === "eth" to stop
# ETH's classifier being drawn over BTC candles (memory eth-dashboard-btc-regime-classifier-not-
# trained-todo-20260831). This is the BTC model that guard was waiting for: same GBM config and the
# same 136 feature_cols as the ETH scorer, trained on BTC's canonical features against a BTC-native
# label (S=24 scale + K=3 confirm) picked by re-screening the grid on BTC rather than porting ETH's
# choice -- ETH's S12_K3 scores only 3/10 on BTC. Same return contract, so it caches/serves
# identically. See scripts/live_regime_btc_signal_20260902.py and
# docs/experiments/btc_regime_s24k3_label_train_20260902.md.
from scripts.live_regime_btc_signal_20260902 import compute_regime_btc_signal  # noqa: E402
# 2026-09-03: XRP 레짐(S96_K9, 같은 날 S48_K6에서 교체 -- 격자 경계 감사).
# 자산마다 교차자산 슬롯이 다르다 -- XRP는 BTC를 넣는다
# (BTC 캐노니컬은 ETH가 들어있다). live_regime_xrp_signal_20260903.py docstring 참조.
from scripts.live_regime_xrp_signal_20260903 import compute_regime_xrp_signal  # noqa: E402
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
V_REBOUND_ECON_SHADOW_STATE_PATH = REPO_ROOT / "data" / "live" / "v_rebound_econ_shadow_state.json"
BTC_EVIDENCE_SHADOW_STATE_PATH = REPO_ROOT / "data" / "live" / "btc_evidence_signal_shadow_state.json"
BTC_EVIDENCE_CTX_REPORT_PATH = REPO_ROOT / "data" / "labels" / "btc_5m_evidence_signal_live_contexts_20260902" / "contexts_report.json"
ETH_ODYSSEY4_SHADOW_TRADES_PATH = REPO_ROOT / "data" / "live" / "eth_odyssey4_shadow" / "closed_trades.jsonl"
ETH_ODYSSEY4_SHADOW_BAR_SECONDS = 300
MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT", "xrp": "XRPUSDT", "hype": "HYPEUSDT"}
EVENT_POLL_SECONDS = 2.5
# 2026-09-03 perf pass -- background cache prewarm (see prewarm_loop()). The cycle is deliberately
# shorter than the SHORTEST TTL it needs to stay ahead of that matters for a coin switch
# (EVIDENCE_SIGNAL_CACHE_SECONDS=60), so a switch lands on a warm cache rather than triggering the
# recompute itself. JOB_GAP staggers the jobs so 25 (asset, endpoint) pairs never hit Binance or
# the shared GPU at once; START_DELAY keeps the warm-up from competing with the first real page
# load right after a deploy/restart.
# How long a cached payload may keep being served while its (expensive) replacement computes.
# Sized to comfortably cover a worst-case TabPFN refit plus a couple of prewarm cycles: in normal
# operation prewarm_loop() keeps everything far fresher than this, so the grace only matters while
# upstream is actually struggling -- exactly when hanging the page would be worst.
STALE_GRACE_SECONDS = 600
PREWARM_CYCLE_SECONDS = 45
PREWARM_JOB_GAP_SECONDS = 0.4
PREWARM_START_DELAY_SECONDS = 5
PREWARM_IDLE_POLL_SECONDS = 10  # how often to re-check for a viewer while standing down
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


def btc_evidence_shadow_payload() -> dict[str, Any]:
    """BTC 증거신호 7종의 섀도우(관측) 원장.

    ⚠️ETH 증거신호 칩과 **다른 자산·다른 파라미터**다. 2026-09-01 그리드스크린이 BTC에서
    HIT정의/H/K/GAP을 독자 재선정했다(ETH 값과 전부 다름).
    ⚠️주문 없음. BTC는 경제성 게이트를 통과한 모델이 아직 없어, 이 러너는 **가상 매매 성과를
    주장하지 않고** 라이브 hit률이 학습 hit률을 재현하는지만 관측한다.
    근거: docs/experiments/btc_evidence_signal_and_shadow_20260902.md
    """
    state = load_json(BTC_EVIDENCE_SHADOW_STATE_PATH) or {}
    ctx = (load_json(BTC_EVIDENCE_CTX_REPORT_PATH) or {}).get("signals", {})
    ledger = state.get("ledger") if isinstance(state.get("ledger"), list) else []
    pending = state.get("pending") if isinstance(state.get("pending"), list) else []

    holdout_auc = {"demarker_extreme": 0.7286, "kalman_deviation_meanrev": 0.6709,
                   "short_term_return_z": 0.6443, "taker_delta_climax": 0.6276,
                   "orthogonal_combo": 0.5933, "fib_extension_exhaustion": 0.5657,
                   "liquidity_sweep": 0.5214}
    per: list[dict[str, Any]] = []
    for name, info in sorted(ctx.items(), key=lambda kv: -holdout_auc.get(kv[0], 0.0)):
        if "error" in info:
            continue
        rows = [r for r in ledger if r.get("signal") == name and r.get("hit") is not None]
        hits = [int(r["hit"]) for r in rows]
        live = (sum(hits) / len(hits)) if hits else None
        train = info.get("hit_rate")
        per.append({
            "signal": name, "n_resolved": len(hits),
            "live_hit_rate": round(live, 4) if live is not None else None,
            "train_hit_rate": train,
            "delta": round(live - train, 4) if (live is not None and train is not None) else None,
            "holdout_auc": holdout_auc.get(name),
            "n_pending": sum(1 for q in pending if q.get("signal") == name),
            "btc_params": info.get("btc_params", {}),
        })
    return {
        "asset": "BTCUSDT", "started_utc": state.get("started_utc"),
        "cycles": state.get("cycles", 0),
        "total_resolved": len([r for r in ledger if r.get("hit") is not None]),
        "total_pending": len(pending),
        "per_signal": per,
        "recent": [
            {"signal": r.get("signal"), "side": r.get("side"), "proba": r.get("proba"),
             "hit": r.get("hit"), "bar_utc": r.get("bar_utc")}
            for r in ledger[-8:]
        ],
        "note": "관측용 섀도우 -- 주문 없음. BTC는 경제성 통과 모델이 아직 없다.",
    }


def _locked_bp(p: dict[str, Any]) -> float | None:
    """섀도우 포지션의 손절선이 이미 확정한 손익(bp, 왕복비용 10bp 차감). 현재가 불필요."""
    try:
        entry, stop = float(p["entry"]), float(p["stop"])
        sgn = 1.0 if p.get("side") == "long" else -1.0
        return round(sgn * (stop - entry) / entry * 1e4 - 10.0, 2)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


COIN_INDICATOR_CACHE_SECONDS = 20
# nif_whale은 간헐적이라 최신 1행만 보면 절반이 빈 값이다 -- 이 창 안의 마지막 값을 쓴다.
MICRO_LOOKBACK_MIN = 15
# ETH 톤 스트립과 같은 모양(app.js MICRO_HISTORY_MAX=48, 5분 간격 = 4시간)
MICRO_STRIP_SAMPLES = 48


def coin_indicators_payload(asset: str) -> dict[str, Any]:
    """코인별 **실시간 지표**(수급흐름/리테일수급/청산캐스케이드) — 2026-09-03.

    그 전까지 이 세 지표는 `trading_bot.py`의 dashboard_state만 읽었는데 **봇은 ETH만 돌린다**.
    그래서 XRP/BTC 탭에서도 ETH 값이 그대로 보이고 있었다(사용자 신고 "비트코인 페이지에
    이더리움 증거신호가 나온다"와 같은 계열).

    ⭐XRP/HYPE는 전용 워커가 microstructure까지 모으므로(`supervisor_xrp_worker.sh`:
    "microstructure + tail-risk + OI/long-short-ratio, all three") **실제 그 코인 값**을 줄 수 있다.
    tail_risk는 COIN_CONFIG에 5코인 전부 있다.

    ⚠️`hawkes_active`는 봇 내부 상태라 다른 코인에는 없다. Z 기반 "주의" 티어까지만 판정 가능하고
    "위험"(hawkes) 티어는 뜨지 않는다 -- `hawkes_available: False`로 명시해 UI가 숨기지 않고
    사실대로 표시하게 한다.
    """
    cfg = COIN_CONFIG.get(asset) or {}
    out: dict[str, Any] = {"asset": asset, "warmed_up": False, "error": None,
                           "micro": None, "tail": None, "hawkes_available": False}
    try:
        mpath, mtable = cfg.get("microstructure_db_path"), cfg.get("microstructure_table")
        if mpath and Path(mpath).exists():
            # ⚠️`nif_whale`은 **대형 체결이 있는 분에만** 계산된다 -- XRP 실측 24시간 기준
            # whale 49.7% / retail 93.7%만 비null이다. 최신 1행만 보면 whale이 절반은 빈 값이
            # 되어 화면이 대부분 "값 없음"이 된다. 그래서 **최근 MICRO_LOOKBACK_MIN분 안의
            # 마지막 비null 값**을 쓰고 **몇 분 전 값인지 함께** 내려준다(오래된 값을 지금 값인
            # 것처럼 보여주지 않기 위해서다).
            con = duckdb.connect(str(mpath), read_only=True)
            try:
                rows = con.execute(f"select ts, nif_whale, nif_retail from {mtable} "
                                   f"order by ts desc limit {MICRO_LOOKBACK_MIN}").fetchall()
            finally:
                con.close()
            if rows:
                latest_ts = rows[0][0]

                def _last(col_idx):
                    for rr in rows:
                        if rr[col_idx] is not None:
                            age = None
                            try:
                                age = round((latest_ts - rr[0]).total_seconds() / 60.0, 1)
                            except (TypeError, AttributeError):
                                pass
                            return float(rr[col_idx]), str(rr[0]), age
                    return None, None, None

                w, w_ts, w_age = _last(1)
                rt, rt_ts, rt_age = _last(2)
                out["micro"] = {"ts": str(latest_ts),
                                "nif_whale": w, "nif_whale_ts": w_ts, "nif_whale_age_min": w_age,
                                "nif_retail": rt, "nif_retail_ts": rt_ts, "nif_retail_age_min": rt_age,
                                "lookback_min": MICRO_LOOKBACK_MIN}
            # ⭐톤 스트립: ETH는 48샘플 x 5분(4시간)을 쓴다(app.js MICRO_HISTORY_MAX=48).
            # 다른 코인도 **같은 모양**으로 만들어야 칩/스트립이 ETH와 똑같아 보인다.
            # 1분 테이블에서 5분마다 하나씩 뽑는다. 임계값은 classifyIndicators와 동일(+-0.05).
            con = duckdb.connect(str(mpath), read_only=True)
            try:
                hrows = con.execute(f"select ts, nif_whale, nif_retail from {mtable} "
                                    f"order by ts desc limit {MICRO_STRIP_SAMPLES * 5}").fetchall()
            finally:
                con.close()
            if hrows:
                picked = list(reversed(hrows))[::5][-MICRO_STRIP_SAMPLES:]

                def _tone(v):
                    if v is None:
                        return "neutral"
                    return "good" if v > 0.05 else ("bad" if v < -0.05 else "neutral")

                out["micro"]["whale_history"] = [_tone(r[1]) for r in picked]
                out["micro"]["retail_history"] = [_tone(r[2]) for r in picked]
                out["micro"]["history_ts"] = [str(r[0]) for r in picked]
        tpath, ttable = cfg.get("tail_risk_db_path"), cfg.get("tail_risk_table")
        if tpath and Path(tpath).exists():
            con = duckdb.connect(str(tpath), read_only=True)
            try:
                r = con.execute(f"select ts, long_usd_1m, short_usd_1m, mu_long, sigma_long, "
                                f"mu_short, sigma_short from {ttable} order by ts desc limit 1").fetchone()
            finally:
                con.close()
            if r:
                def _z(v, mu, sd):
                    try:
                        return float((v - mu) / sd) if sd and sd > 0 else 0.0
                    except (TypeError, ValueError):
                        return 0.0
                out["tail"] = {"ts": str(r[0]), "z_long": _z(r[1], r[3], r[4]),
                               "z_short": _z(r[2], r[5], r[6]),
                               "hawkes_active": False}
                con = duckdb.connect(str(tpath), read_only=True)
                try:
                    hr = con.execute(f"select ts, long_usd_1m, short_usd_1m, mu_long, sigma_long, "
                                     f"mu_short, sigma_short from {ttable} "
                                     f"order by ts desc limit {MICRO_STRIP_SAMPLES * 5}").fetchall()
                finally:
                    con.close()
                if hr:
                    picked = list(reversed(hr))[::5][-MICRO_STRIP_SAMPLES:]
                    # ⚠️hawkes가 없으니 Z만으로 판정한다 -> "위험"(bad) 티어는 나오지 않는다.
                    out["tail"]["cascade_history"] = [
                        ("warn" if max(_z(x[1], x[3], x[4]), _z(x[2], x[5], x[6])) >= 2.0 else "good")
                        for x in picked]
                    out["tail"]["history_ts"] = [str(x[0]) for x in picked]
        out["warmed_up"] = bool(out["micro"] or out["tail"])
        if not out["warmed_up"]:
            out["error"] = "no_coin_indicator_data"
    except Exception as e:                                     # noqa: BLE001 -- 절대 raise 안 함
        out["error"] = f"coin_indicators_error: {e}"
    return out


def v_rebound_econ_shadow_payload() -> dict[str, Any]:
    """V자반등 **경제라벨** 후보의 섀도우(가상) 원장. 주문은 내지 않는다 -- 표시 전용.

    근거: docs/model_contracts/eth_v_rebound_econ_label_autotrade_spec_20260902.md
    러너: scripts/live_eth_v_rebound_econ_shadow_runner_20260902.py
    ⚠️배포 대시보드 칩(매 봉 giveback 모델)과는 **다른 모델**이다 -- 라벨 정의부터 다르다.
    """
    state = load_json(V_REBOUND_ECON_SHADOW_STATE_PATH) or {}
    ledger = state.get("ledger") if isinstance(state.get("ledger"), list) else []
    positions = state.get("positions") if isinstance(state.get("positions"), list) else []

    pnls: list[float] = []
    for row in ledger:
        try:
            pnls.append(float(row["pnl_bp"]))
        except (KeyError, TypeError, ValueError):
            continue
    n = len(pnls)
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    equity: list[dict[str, Any]] = []
    run = 0.0
    peak = 0.0
    mdd = 0.0
    for row, v in zip([r for r in ledger if "pnl_bp" in r], pnls):
        run += v
        peak = max(peak, run)
        mdd = min(mdd, run - peak)
        equity.append({"ts": row.get("exit_utc"), "cum_bp": round(run, 2)})

    # ── 사람이 바로 읽을 수 있는 해석값 (대시보드가 원시 숫자만 나열하지 않도록) ──
    # 목표 표본: HOLDOUT 빈도(13.18건/일) x 2주. 이만큼은 모여야 백테스트와 대조가 의미 있다.
    HOLDOUT_EXP_BP, HOLDOUT_PER_DAY = 6.09, 13.18
    target = int(round(HOLDOUT_PER_DAY * 14))
    days = 0.0
    started = state.get("started_utc")
    if started:
        try:
            days = max((datetime.now(timezone.utc)
                        - datetime.fromisoformat(str(started))).total_seconds() / 86400.0, 0.0)
        except (TypeError, ValueError):
            days = 0.0
    exp = (sum(pnls) / n) if n else None
    if n < 30:
        verdict = {"tone": "neutral", "headline": "표본이 아직 적습니다",
                   "detail": f"{n}건 청산 · 판단에는 {target}건 정도가 필요합니다"}
    elif exp is None:
        verdict = {"tone": "neutral", "headline": "기록 없음", "detail": ""}
    elif exp <= 0:
        verdict = {"tone": "bad", "headline": "백테스트에 미달합니다",
                   "detail": f"건당 {exp:+.2f}bp — 백테스트 기대 {HOLDOUT_EXP_BP:+.2f}bp"}
    elif exp < HOLDOUT_EXP_BP * 0.5:
        verdict = {"tone": "warn", "headline": "백테스트보다 약합니다",
                   "detail": f"건당 {exp:+.2f}bp — 백테스트 기대의 "
                             f"{exp / HOLDOUT_EXP_BP * 100:.0f}% 수준"}
    elif exp <= HOLDOUT_EXP_BP * 1.5:
        verdict = {"tone": "good", "headline": "백테스트와 비슷합니다",
                   "detail": f"건당 {exp:+.2f}bp — 백테스트 기대 {HOLDOUT_EXP_BP:+.2f}bp"}
    else:
        verdict = {"tone": "warn", "headline": "백테스트보다 지나치게 좋습니다",
                   "detail": f"건당 {exp:+.2f}bp — 계측이 느슨하지 않은지 먼저 의심할 것"}

    return {
        "started_utc": state.get("started_utc"),
        "open_positions": [
            {
                "side": p.get("side"), "entry": p.get("entry"), "stop": p.get("stop"),
                "best": p.get("best"), "armed": bool(p.get("armed")),
                "proba": p.get("proba"), "opened_utc": p.get("opened_utc"),
                "bars_held": p.get("bars_held"),
                # 손절선이 이미 확정한 손익. 무장 전이면 최대손실, 무장 후면 확보이익이 될 수 있다.
                # 현재가 없이도 계산되고, "최악이어도 얼마"라는 직관적 의미를 준다.
                "locked_bp": _locked_bp(p),
            }
            for p in positions[-10:]
        ],
        "n_open": len(positions),
        "closed_trades": n,
        "exp_bp": round(sum(pnls) / n, 2) if n else None,
        "total_bp": round(sum(pnls), 1) if n else None,
        "win_rate": round(len(wins) / n, 4) if n else None,
        "payoff": (round((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)), 3)
                   if wins and losses else None),
        "max_dd_bp": round(mdd, 1) if n else None,
        "consec_loss": state.get("consec_loss"),
        "equity_curve": equity[-300:],
        "recent_trades": [
            {
                "side": r.get("side"), "entry_utc": r.get("entry_utc"),
                "exit_utc": r.get("exit_utc"), "entry": r.get("entry"),
                "exit": r.get("exit"), "pnl_bp": r.get("pnl_bp"),
                "reason": r.get("reason"), "proba": r.get("proba"),
            }
            for r in ledger[-8:]
        ],
        "backtest_reference": {"oos_exp_bp": 7.98, "holdout_exp_bp": 6.09,
                               "holdout_win_rate": 0.780, "holdout_payoff": 0.346,
                               "holdout_trades_per_day": 13.18},
        "days_running": days,
        "trades_per_day": round(n / days, 2) if (n and days > 0) else None,
        "target_trades": target,
        "progress_pct": round(min(n / target, 1.0) * 100, 1) if target else None,
        "verdict": verdict,
        "note": "섀도우 -- 주문 없음. 배포 칩(매 봉 giveback)과 다른 모델.",
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
    market_history_cache: dict[str, dict[str, Any]] = {}
    market_history_locks = {asset: asyncio.Lock() for asset in MARKET_SYMBOLS}
    evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None, "frames": None}
    evidence_signal_lock = asyncio.Lock()
    evidence_signal_provisional_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    evidence_signal_provisional_lock = asyncio.Lock()
    btc_evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    btc_evidence_signal_lock = asyncio.Lock()
    xrp_evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    xrp_evidence_signal_lock = asyncio.Lock()
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
    regime_btc_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    regime_xrp_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    regime_xrp_lock = asyncio.Lock()
    coin_indicator_cache: dict[str, dict] = {a: {"ts": 0.0, "payload": None} for a in COIN_CONFIG}
    coin_indicator_locks = {a: asyncio.Lock() for a in COIN_CONFIG}
    regime_btc_lock = asyncio.Lock()
    macro_calendar_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    macro_calendar_lock = asyncio.Lock()
    model_indicator_history: deque = deque(maxlen=MODEL_INDICATOR_HISTORY_MAX)
    model_indicator_sample_state: dict[str, float] = {"last_sample_at": 0.0}

    # ---------------------------------------------------------------------------------
    # 2026-09-03 perf pass -- stale-while-revalidate + background prewarm.
    #
    # Why: every Snapshot-tab cache declared above is keyed by asset and filled LAZILY, so
    # the first visit to a coin paid the full cold cost of ~6 endpoints at once (a Binance
    # klines fetch + pandas each, and for the evidence panels a TabPFN refit measured at up
    # to 43s under GPU contention -- see the 2026-09-03 GPU-contention finding). With 5 coins
    # in the switcher that cold surface is 5x what it was when this was ETH-only, which is
    # exactly the "코인이 늘수록 대시보드가 느려진다" symptom this pass targets.
    #
    # Worse, several client poll intervals were set EQUAL to the server TTL they mirror
    # (evidence 60s TTL vs a 300s client poll; liq-map/regime 300s vs 300s), so even the
    # already-visited coin missed the cache on essentially every poll and paid the cold cost
    # again. Two mechanisms now keep the REQUEST path off the COMPUTE path entirely:
    #
    #   swr_cached()   a request never blocks on a recompute once it has any payload to
    #                  serve -- it returns the stale one immediately and refreshes behind it.
    #   prewarm_loop() a background task walks every (asset, endpoint) pair and refreshes
    #                  each BEFORE its TTL expires, so the steady-state request path is a
    #                  plain in-memory dict read.
    #
    # The per-endpoint fetch/compute bodies themselves are UNCHANGED -- they were only moved
    # into `produce()` closures so both mechanisms can drive the same code.
    # ---------------------------------------------------------------------------------
    refresh_tasks: dict[str, asyncio.Task] = {}

    def _schedule_refresh(key: str, cache: dict, lock: asyncio.Lock, ttl: float, produce) -> None:
        """Kick one background refresh for `key`, deduped while a previous one is in flight."""
        task = refresh_tasks.get(key)
        if task is not None and not task.done():
            return
        async def _run() -> None:
            try:
                async with lock:
                    if time.monotonic() - cache.get("ts", 0.0) < ttl:
                        return  # another path refreshed it while this one queued on the lock
                    payload = await produce()
                    cache["payload"] = payload
                    cache["ts"] = time.monotonic()
            except Exception as exc:  # noqa: BLE001 -- a failed BACKGROUND refresh must never
                # reach the client: the stale payload it was refreshing is still being served,
                # so the correct behaviour is to log it and let the next cycle retry. Letting
                # this raise would only kill an orphan task and lose the reason.
                print(f"cache refresh failed for {key} (still serving stale): {exc}", flush=True)
        refresh_tasks[key] = asyncio.create_task(_run())

    async def swr_cached(key: str, cache: dict, lock: asyncio.Lock, ttl: float, produce,
                         *, max_stale: float = 0.0) -> Any:
        """Fresh -> serve it. Stale -> serve stale NOW and refresh behind it. Cold -> await.

        `max_stale` is how far PAST `ttl` a payload may be served while its replacement is still
        being computed, and it defaults to 0 -- i.e. this behaves exactly like the double-checked
        lock it replaced unless a call site opts in. Only endpoints whose recompute is genuinely
        expensive (a Binance round trip, or a TabPFN refit that has been measured at 43s under GPU
        contention) opt in; for the endpoints that just read a local duckdb, blocking was already
        fast and returns FRESHER data, so they keep doing that.

        Past ttl + max_stale the payload is treated as cold again: at that age, blocking for a
        current reading beats silently serving something long out of date."""
        now = time.monotonic()
        age = now - cache.get("ts", 0.0)
        payload = cache.get("payload")
        if payload is not None and age < ttl:
            return payload
        if payload is not None and age < ttl + max_stale:
            _schedule_refresh(key, cache, lock, ttl, produce)
            return payload
        async with lock:
            payload = cache.get("payload")
            if payload is not None and time.monotonic() - cache.get("ts", 0.0) < ttl:
                return payload
            payload = await produce()
            cache["payload"] = payload
            cache["ts"] = time.monotonic()
            return payload

    # Shared, connection-pooled session for this process's Binance calls. Every klines/funding
    # fetch used to open its OWN `async with ClientSession(...)` (7 of them), which means a fresh
    # TCP+TLS handshake per call and no keep-alive reuse across endpoints or across coins -- the
    # single biggest avoidable fixed cost once one coin switch fans out to 6 endpoints at once.
    http_session: dict[str, ClientSession | None] = {"session": None}

    def binance_session() -> ClientSession:
        session = http_session["session"]
        if session is None or session.closed:
            raise web.HTTPServiceUnavailable(reason="http_session_unavailable")
        return session

    async def fetch_binance_json(url: str, params: dict, *, timeout: float = 10.0,
                                 error_reason: str | None = None) -> Any:
        """GET `url` on the shared pooled session and return the decoded JSON.

        `error_reason` reproduces what each per-endpoint block used to do on a non-200: raise
        HTTPBadGateway with that reason. Passing None instead returns None on a non-200, for the
        two legs (BTC/funding, forming-bar preview) that are documented as fail-soft."""
        async with binance_session().get(url, params=params,
                                         timeout=ClientTimeout(total=timeout)) as response:
            if response.status != web.HTTPOk.status_code:
                if error_reason is None:
                    return None
                raise web.HTTPBadGateway(reason=error_reason)
            return await response.json()

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
        async def produce() -> list[dict[str, float | int]]:
            rows = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {"symbol": MARKET_SYMBOLS[asset], "interval": "5m", "limit": 100},
                timeout=3.0, error_reason="market_history_upstream_error",
            )
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
            return candles

        return await swr_cached(
            f"market_history:{asset}",
            market_history_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            market_history_locks[asset], MARKET_HISTORY_CACHE_SECONDS, produce,
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_evidence_signals() -> dict[str, Any]:
        """Informational-only reversal-evidence-signal readout for the Snapshot tab -- NOT a
        trading signal (see docstring in the imported module). Mirrors load_market_history()'s
        cache/lock pattern but needs a much longer klines window (EVIDENCE_FETCH_LIMIT bars, to
        warm up orthogonal_combo's EVIDENCE_PCTRANK_WINDOW-bar percentile-rank window) than the
        chart's own /api/market-history (limit=100), so it gets its own cache rather than sharing
        market_history_cache."""
        async def produce() -> dict[str, Any]:
            raw = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {
                    "symbol": EVIDENCE_SIGNAL_SYMBOL,
                    "interval": EVIDENCE_SIGNAL_INTERVAL,
                    "limit": EVIDENCE_FETCH_LIMIT,
                },
                error_reason="evidence_signal_upstream_error",
            )
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
                braw = await fetch_binance_json(
                    "https://fapi.binance.com/fapi/v1/klines",
                    {
                        "symbol": EVIDENCE_SIGNAL_BTC_SYMBOL,
                        "interval": EVIDENCE_SIGNAL_INTERVAL,
                        "limit": EVIDENCE_FETCH_LIMIT,
                    },
                )
                if braw is not None:
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
                fraw = await fetch_binance_json(
                    EVIDENCE_SIGNAL_FUNDING_URL,
                    {"symbol": EVIDENCE_SIGNAL_SYMBOL, "limit": EVIDENCE_FUNDING_HISTORY_LIMIT},
                )
                if fraw is not None:
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
            # smt_divergence / fib_extension_exhaustion REPLACED with their TabPFN meta-label
            # models' live probability (2026-08-30/31; dalton_rule2_balance_edge removed 2026-08-31,
            # see METALABEL_SIGNALS' own module docstring) -- reuses this cycle's already-fetched
            # `df` and already-computed `latest` fire state, no separate fetch/compute_signals()
            # call. Fail-soft: a GPU/TabPFN hiccup must not block the other signals from rendering.
            metalabels: dict[str, dict] = {}
            if warmed_up:
                try:
                    metalabels = await asyncio.to_thread(compute_evidence_signal_metalabels, df, sig)
                except Exception as metalabel_exc:  # noqa: BLE001
                    print(f"evidence-signal metalabel leg failed (taker_delta_z_climax/"
                          f"short_term_return_z/liquidity_sweep/orthogonal_combo/smt_divergence/"
                          f"fib_extension_exhaustion will read as not-fired this cycle): {metalabel_exc}", flush=True)
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
                    # 2026-08-31 fix: switched from the _active (sustain-window) column to the RAW
                    # bcol/tcol -- using _active here made smt_divergence's 72-bar/6h sustain (which
                    # exceeds this 48-bar/4h strip) look permanently stuck on (user report). See
                    # eth_dashboard_evidence_signal_history_strip_sustain_window_bug_20260831.
                    # 2026-09-01 (user follow-up): a single raw-fire bar was too subtle to read at a
                    # glance, so this now sends the "fill" column instead (active from the fire bar
                    # through whichever comes first, this signal's own K*ATR take-profit price or
                    # its trained HORIZON -- see compute_signals()'s _fill_until_tp_or_horizon).
                    # User explicitly confirmed this may fill the ENTIRE visible strip when the
                    # horizon runs that long (no cap at EVIDENCE_SIGNAL_HISTORY_BARS this time). The
                    # true raw column rides along separately in bottom_raw_fire/top_raw_fire purely
                    # so the frontend can force a visible segment boundary at each actual re-fire
                    # even mid-fill (app.js::toneStripSvg) -- otherwise a second real trigger inside
                    # an already-active fill window would silently disappear into one block again.
                    "bottom_history": sig[f"{bcol}_fill"].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                    "top_history": sig[f"{tcol}_fill"].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                    "bottom_raw_fire": sig[bcol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                    "top_raw_fire": sig[tcol].tail(EVIDENCE_SIGNAL_HISTORY_BARS).fillna(False).astype(bool).tolist() if warmed_up else [],
                }
                if name in metalabels:
                    entry["model_proba"] = metalabels[name]["proba"]
                    entry["model_side"] = metalabels[name]["side"]
                    entry["model_tp_price"] = metalabels[name].get("tp_price")
                    # 2026-09-01: 저ATR 경고 (표시 전용, 모델/발동 로직과 무관) -- 발동봉 ATR이
                    # 이 신호 자신의 발동시 ATR 중앙값보다 낮으면 low_atr=True. 저변동 구간에선
                    # SL/ARM/Trail이 ATR 배수로 줄어드는데 왕복비용은 고정이라 방향이 맞아도
                    # 수수료를 못 넘기는 비율이 커진다. 근거/실측:
                    # docs/homer/evidence_signal_economics_tuning_protocol.md
                    # 2026-09-03: 익절가 도달 여부. 그 전까지 칩은 발동 후 horizon_bars 동안
                    # (smt_divergence는 6시간) 무조건 유지되며 목표 달성을 보지 않았다 --
                    # 이미 끝난 움직임을 "활성"으로 띄워 늦은 진입을 유도할 수 있었다.
                    entry["model_tp_touched"] = metalabels[name].get("tp_touched")
                    entry["model_bars_since_fire"] = metalabels[name].get("bars_since_fire")
                    entry["model_atr_bp"] = metalabels[name].get("atr_bp")
                    entry["model_atr_median_bp"] = metalabels[name].get("atr_median_bp")
                    entry["model_low_atr"] = metalabels[name].get("low_atr")
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
            return payload

        return await swr_cached(
            "evidence_signal", evidence_signal_cache, evidence_signal_lock,
            EVIDENCE_SIGNAL_CACHE_SECONDS, produce,
            max_stale=STALE_GRACE_SECONDS,
        )

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
                    raw = await fetch_binance_json(
                        "https://fapi.binance.com/fapi/v1/klines",
                        {"symbol": symbol, "interval": EVIDENCE_SIGNAL_INTERVAL, "limit": 2},
                    )
                    if raw is None:
                        return None
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

    async def load_btc_evidence_signals() -> dict[str, Any]:
        """BTC 코인 페이지의 메인 증거신호 패널(2026-09-02) -- load_evidence_signals()의 BTC판.
        compute_btc_evidence_signals_panel()이 klines 페치+지표 계산+TabPFN 채점을 전부 자체
        처리하므로(BTC 전용 그리드스크린 K/HORIZON, live_btc_evidence_signal_metalabel_20260902.py
        참고) 여기서는 캐시/락만 감싼다 -- load_evidence_signals()처럼 별도 klines 페치 단계가
        없다. 첫 호출은 TabPFN 7개를 새로 적합(수 초)하지만 이후 같은 프로세스 안에서는
        캐시된 모델을 재사용(그 함수의 _load_models() 참고)하므로 이 EVIDENCE_SIGNAL_CACHE_SECONDS
        (ETH와 동일 60초) 캐시는 매 사이클의 재적합 비용이 아니라 klines 재페치+추론 비용만 아낀다."""
        return await swr_cached(
            "btc_evidence_signal", btc_evidence_signal_cache, btc_evidence_signal_lock, EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_btc_evidence_signals_panel),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_xrp_evidence_signals() -> dict[str, Any]:
        """XRP 코인 페이지의 메인 증거신호 패널(2026-09-03) -- load_btc_evidence_signals()의 XRP판.
        서빙 5종(liquidity_sweep/fib_extension_exhaustion은 HOLDOUT AUC가 무작위 미만이라 제외).
        구조/캐시 정책은 BTC판과 동일하다."""
        return await swr_cached(
            "xrp_evidence_signal", xrp_evidence_signal_cache, xrp_evidence_signal_lock, EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_xrp_evidence_signals_panel),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_v_rebound_signal() -> dict[str, Any]:
        """유동성스윕 반등예측 event-triggered signal -- see
        scripts/live_eth_sweep_v_rebound_signal_20260829.py docstring for the VAL/OOS/holdout-
        validated TabPFN model and why this is computed HERE (dashboard-side) rather than by
        trading_bot.py. Each call re-fits TabPFN on its frozen historical context (~3s measured
        on this server's GPU, 2026-08-29) -- asyncio.to_thread so that never stalls the event loop,
        same reasoning as load_evidence_signals() above."""
        return await swr_cached(
            "v_rebound", v_rebound_cache, v_rebound_lock, EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_eth_sweep_v_rebound_signal),
            max_stale=STALE_GRACE_SECONDS,
        )

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
        return await swr_cached(
            f"basis_liquidation:{asset}", basis_liquidation_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            basis_liquidation_locks[asset], EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_basis_liquidation_signal, symbol=COIN_CONFIG[asset]["binance_symbol"]),
            max_stale=STALE_GRACE_SECONDS,
        )

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
        return await swr_cached(
            f"liquidation_5m:{asset}",
            liquidation_5m_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            liquidation_5m_locks[asset], LIQUIDATION_5M_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_liquidation_5m_signal, coin=asset),
        )

    async def load_liquidation_direction_signal(asset: str = "eth") -> dict[str, Any]:
        """Directional-only liquidation tilt (liq_net_z_12, contrarian sign) -- model-indicator
        tier, no PnL/economic claim. See scripts/live_liquidation_direction_signal_20260825.py
        docstring. Same 60s cache reasoning as load_liquidation_5m_signal() above (underlying data
        updates once per minute).

        asset: 2026-08-31, BTC added -- see coin_config.py for BTC's separate tail-risk file."""
        return await swr_cached(
            f"liquidation_direction:{asset}",
            liquidation_direction_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            liquidation_direction_locks[asset], EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_liquidation_direction_signal, coin=asset),
        )

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
        async def produce() -> dict[str, Any]:
            raw = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {
                    "symbol": COIN_CONFIG[asset]["binance_symbol"],
                    "interval": LIQUIDATION_MAP_INTERVAL,
                    "limit": LIQUIDATION_MAP_FETCH_LIMIT,
                },
                error_reason="liquidation_map_upstream_error",
            )
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
            return payload

        return await swr_cached(
            f"liquidation_map:{asset}",
            liquidation_map_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            liquidation_map_locks[asset], LIQUIDATION_MAP_CACHE_SECONDS, produce,
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_wide24() -> dict[str, Any]:
        """wide24 HMM regime overlay for the Snapshot tab's liquidation-map chart -- see
        scripts/live_regime_wide24_signal_20260826.py docstring. compute_regime_wide24_signal()
        itself never raises (degrades to warmed_up=False), and its own fetch/compute is blocking
        (requests + pandas/HMM), so it's offloaded via asyncio.to_thread same as
        compute_liquidation_levels() above rather than converted to aiohttp -- keeps the ported
        logic identical to the validated scratchpad script it came from."""
        return await swr_cached(
            "regime_wide24", regime_wide24_cache, regime_wide24_lock, REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_regime_wide24_signal),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_btc() -> dict[str, Any]:
        """BTC regime overlay for the Snapshot tab when its coin switcher is on BTC -- the BTC twin
        of load_regime_wide24() above, same cache TTL, same asyncio.to_thread offload (the scorer
        fetches from Binance and runs FeatureEngineer, both blocking), and the same never-raises
        contract (degrades to warmed_up=False so the ribbon shows its waiting state rather than
        breaking the chart)."""
        return await swr_cached(
            "regime_btc", regime_btc_cache, regime_btc_lock, REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_regime_btc_signal),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_xrp() -> dict[str, Any]:
        """XRP 3-class 레짐(S96_K9, 2026-09-03) -- load_regime_btc()의 XRP판.
        같은 캐시 TTL / asyncio.to_thread / never-raises 계약."""
        return await swr_cached(
            "regime_xrp", regime_xrp_cache, regime_xrp_lock, REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_regime_xrp_signal),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_macro_calendar() -> dict[str, Any]:
        """US macro/corporate event calendar for the Snapshot tab -- see scripts/live_macro_
        calendar_20260826.py docstring for the 6 sources. compute_macro_calendar() is blocking
        (requests, no aiohttp) and never raises (each source degrades independently), same
        asyncio.to_thread pattern as load_regime_wide24() above."""
        return await swr_cached(
            "macro_calendar", macro_calendar_cache, macro_calendar_lock, MACRO_CALENDAR_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_macro_calendar, os.getenv("FRED_API_KEY"), os.getenv("EIA_API_KEY"), os.getenv("FINNHUB_API_KEY")),
            max_stale=STALE_GRACE_SECONDS,
        )

    def prewarm_jobs() -> list[tuple[str, Any]]:
        """(label, coroutine-factory) for the panels worth warming ahead of a coin switch.

        This list is deliberately EXACTLY the set that opts into swr_cached(max_stale=...): the
        endpoints whose recompute is a Binance round trip and/or GPU work. The three that are
        excluded from both -- liquidation_5m, liquidation_direction, coin_indicator -- read a local
        duckdb, which is already fast enough on demand AND is the reason to stay away: those files
        (tail_risk*.duckdb, microstructure.duckdb) have a LIVE WRITER, trading_bot.py among them,
        and a reader takes a real lock against it (2026-08-24: "Conflicting lock is held", absorbed
        only by a retry ladder). Warming 5 coins x 2 of those every cycle would multiply that
        contention against the live-money process to buy back a few ms on a local read. Not worth it.

        ETH-first because that is the tab the dashboard opens on, so a cold start warms the visible
        coin before spending time on the other four (COIN_CONFIG/MARKET_SYMBOLS both start at eth)."""
        jobs: list[tuple[str, Any]] = [
            ("evidence_signal", load_evidence_signals),
            ("regime_wide24", load_regime_wide24),
            ("regime_btc", load_regime_btc),
            ("regime_xrp", load_regime_xrp),
            ("btc_evidence_signal", load_btc_evidence_signals),
            ("xrp_evidence_signal", load_xrp_evidence_signals),
            ("v_rebound", load_v_rebound_signal),
        ]
        for asset in COIN_CONFIG:
            jobs.append((f"liquidation_map:{asset}", lambda a=asset: load_liquidation_map(a)))
            jobs.append((f"basis_liquidation:{asset}", lambda a=asset: load_basis_liquidation_signal(a)))
        for asset in MARKET_SYMBOLS:
            jobs.append((f"market_history:{asset}", lambda a=asset: load_market_history(a)))
        return jobs

    async def prewarm_loop(app: web.Application) -> None:
        """Keep every coin's panel caches warm so a coin switch is a memory read, not a compute.

        This is the half of the 2026-09-03 perf pass that removes the cost rather than hiding it:
        swr_cached() already stops a request BLOCKING on a recompute, but only this stops the
        recompute from being triggered by a user action at all. Runs the jobs SEQUENTIALLY with a
        small gap -- warming 5 coins x 5 endpoints in parallel would stampede both Binance's rate
        limit and the single shared GPU the evidence/V-rebound panels contend for (see the
        2026-09-03 GPU-contention finding), which is the very thing that makes them slow.

        Each job is just the normal loader: if its cache is still fresh this is a dict lookup and
        costs nothing, so the loop self-throttles to only the endpoints that actually went stale."""
        await asyncio.sleep(PREWARM_START_DELAY_SECONDS)  # let the first real page load win the race
        while True:
            started = time.monotonic()
            # Only warm while somebody is actually watching. Four of these jobs are TabPFN refits
            # sharing ONE GPU with the live shadow runners and whatever research jobs other sessions
            # are running (2026-09-03 contention finding: a dashboard TabPFN call measured at 43s
            # while a research job held the GPU). Warming caches nobody is going to read would add
            # exactly that contention for no benefit -- and a coin switch can only happen while a
            # browser is connected anyway, which is the case this whole loop exists to serve.
            # event_clients is the SSE client set; app.js drops its connection when the tab is
            # hidden, so this also stands down for a backgrounded tab.
            if not event_clients:
                await asyncio.sleep(PREWARM_IDLE_POLL_SECONDS)
                continue
            for label, factory in prewarm_jobs():
                try:
                    await factory()
                except Exception as exc:  # noqa: BLE001 -- one unreachable coin (a collector that
                    # has not backfilled yet, a Binance blip) must not stop the other coins from
                    # being warmed, and must never take down the loop for the whole process.
                    print(f"prewarm {label} failed (will retry next cycle): {exc}", flush=True)
                await asyncio.sleep(PREWARM_JOB_GAP_SECONDS)
            await asyncio.sleep(max(0.0, PREWARM_CYCLE_SECONDS - (time.monotonic() - started)))

    async def start_http_session(app: web.Application) -> None:
        http_session["session"] = ClientSession(
            timeout=ClientTimeout(total=10),
            connector=TCPConnector(limit=32, ttl_dns_cache=300, keepalive_timeout=60),
        )

    async def stop_http_session(app: web.Application) -> None:
        session = http_session["session"]
        http_session["session"] = None
        if session is not None and not session.closed:
            await session.close()

    async def start_prewarm(app: web.Application) -> None:
        app["prewarm_task"] = asyncio.create_task(prewarm_loop(app))

    async def stop_prewarm(app: web.Application) -> None:
        task = app.get("prewarm_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Background refreshes swr_cached() kicked off hold a reference to the pooled session that
        # stop_http_session() is about to close, so cancel them here rather than leaving pending
        # tasks to be destroyed mid-request at loop shutdown.
        for pending in tuple(refresh_tasks.values()):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*refresh_tasks.values(), return_exceptions=True)
        refresh_tasks.clear()

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

    async def api_btc_evidence_signals(request: web.Request) -> web.Response:
        payload = await load_btc_evidence_signals()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_xrp_evidence_signals(request: web.Request) -> web.Response:
        payload = await load_xrp_evidence_signals()
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

    async def api_regime_btc(request: web.Request) -> web.Response:
        payload = await load_regime_btc()
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})


    async def load_coin_indicators(asset: str) -> dict[str, Any]:
        return await swr_cached(
            f"coin_indicator:{asset}", coin_indicator_cache[asset],
            coin_indicator_locks[asset], COIN_INDICATOR_CACHE_SECONDS,
            lambda: asyncio.to_thread(coin_indicators_payload, asset),
        )

    async def api_coin_indicators(request: web.Request) -> web.Response:
        payload = await load_coin_indicators(_query_coin_asset(request))
        return web.json_response(payload, headers={"Cache-Control": "no-cache"})

    async def api_regime_xrp(request: web.Request) -> web.Response:
        payload = await load_regime_xrp()
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

    async def api_btc_evidence_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "btc-evidence-shadow",
            file_signature(BTC_EVIDENCE_SHADOW_STATE_PATH),
            file_signature(BTC_EVIDENCE_CTX_REPORT_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, btc_evidence_shadow_payload(), etag)

    async def api_v_rebound_econ_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "v-rebound-econ-shadow",
            file_signature(V_REBOUND_ECON_SHADOW_STATE_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, v_rebound_econ_shadow_payload(), etag)

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
    app.router.add_get("/api/btc-evidence-signals", api_btc_evidence_signals)
    app.router.add_get("/api/xrp-evidence-signals", api_xrp_evidence_signals)
    app.router.add_get("/api/v-rebound-signal", api_v_rebound_signal)
    app.router.add_get("/api/basis-liquidation-signal", api_basis_liquidation_signal)
    app.router.add_get("/api/liquidation-5m-signal", api_liquidation_5m_signal)
    app.router.add_get("/api/liquidation-direction-signal", api_liquidation_direction_signal)
    app.router.add_get("/api/liquidation-map", api_liquidation_map)
    app.router.add_get("/api/regime-wide24", api_regime_wide24)
    app.router.add_get("/api/regime-btc", api_regime_btc)
    app.router.add_get("/api/regime-xrp", api_regime_xrp)
    app.router.add_get("/api/coin-indicators", api_coin_indicators)
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
    app.router.add_get("/api/v-rebound-econ-shadow", api_v_rebound_econ_shadow)
    app.router.add_get("/api/btc-evidence-shadow", api_btc_evidence_shadow)
    # show_index=False to match /data/live/ below: this directory is reachable from the
    # public tunnel, and a listing advertised every file sitting in it (e.g. the
    # *.bak_pre_live_tab_removal_20260831 snapshots) rather than just the three the page
    # actually loads. The explicit /dashboard/live[/] routes above already serve the page
    # itself, and nothing fetches a listing, so this only removes the enumeration.
    app.router.add_static("/dashboard/live/", DASHBOARD_DIR, show_index=False)
    app.router.add_static("/data/live/", LIVE_DIR, show_index=False)
    # start_http_session FIRST: prewarm_loop and every klines fetch below it need the pooled
    # session to exist. stop_http_session LAST for the mirror-image reason.
    app.on_startup.append(start_http_session)
    app.on_startup.append(start_dashboard_events)
    app.on_startup.append(start_prewarm)
    app.on_cleanup.append(stop_prewarm)
    app.on_cleanup.append(stop_dashboard_events)
    app.on_cleanup.append(stop_http_session)
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
