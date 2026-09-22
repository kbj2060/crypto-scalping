from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import csv
import functools
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from aiohttp import ClientSession, ClientTimeout, TCPConnector, WSMsgType, web
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trading_bot_modules.duckdb_access import duckdb_path_lock  # noqa: E402 -- sys.path 먼저
# Macro calendar (2026-08-26) needs FRED/EIA/Finnhub API keys from .env -- no other endpoint in
# this file has needed a real secret before, so .env was never loaded here until now.
load_dotenv(REPO_ROOT / ".env")
# 주의: 이건 import 시점에 .env를 os.environ에 한 번 굽는다. 실행 중인 프로세스는 옛 값을
# 계속 들고 있으므로 .env를 고쳤으면 대시보드를 재기동해야 반영된다(2026-09-10 바이낸스 키
# 교체 때 실제로 걸렸다). dashboard/ 아래가 바뀌면 deploy_watcher.sh가 알아서 재기동한다.
# Reuses the exact, already-verified signal formulas from the standalone CLI dashboard rather
# than re-deriving them here -- see that module's docstring for formula provenance (each formula
# transcribed verbatim from the 2026-08-14 research scripts). compute_signals/bars_since_last_true
# are pure functions (no I/O, no sleep) -- that makes them correctness-safe to call from an async
# handler, but NOT free: 2026-08-25 perf pass moved the compute_signals() call itself behind
# asyncio.to_thread (see load_evidence_signals()) so its pandas rolling-window work doesn't block
# the event loop for its duration, matching the pattern load_liquidation_5m_signal()/
# load_liquidation_direction_signal() already used.
# 2026-09-16 증거신호 칩 제거 후 남은 것: klines 창 길이 상수와 «마지막 발동 이후 봉 수» 헬퍼.
# compute_signals/SIGNAL_ORDER/PCTRANK/FUNDING_* 는 8종 계산 전용이라 같이 내렸다.
# (모듈 자체는 남긴다 -- 극점 탐지기가 compute_signals 를 직접 import 한다.)
from scripts.live_trade_tape_collector_20260916 import (  # noqa: E402
    RETAIL_MAX_USD,
    WHALE_MIN_USD,
    TakerOrderAggregator,
)
from scripts.live_evidence_signal_dashboard_20260823 import (  # noqa: E402
    FETCH_LIMIT as EVIDENCE_FETCH_LIMIT,
    bars_since_last_true,
)
# 유동성스윕 반등예측 event-triggered signal (2026-08-29, TabPFN Tier0+rsi model -- see
# docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md). Own klines fetch +
# a frozen historical TabPFN context, NOT from trading_bot.py's dashboard_state.json -- computed
# dashboard-side so it never touches the live bot.
# 2026-09-09 극점 탐지기: 증거신호 8종을 피쳐로 쓴 "이 봉이 ±60분 국소 극점일 확률" 모델.
# 표본외 161일 정밀도 강 66.0%(1.46건/일) · 중 53.2% · 약 32.1% (발동봉 기저 24.2% · 무작위 봉 2.9%).
# 🔴강한 추세 구간(ret144 7일 분위 상하 20%)에서는 콜을 억제한다 -- 게이트 없이는 순 -3.36bp,
#   중립 구간만 쓰면 +2.27bp. 표시 전용이고 매매 트리거가 아니다.
# 2026-09-09 청산맵 신호 마커: 차트(72봉)와 **같은 타임스탬프 격자**로 증거신호 종수 + 이벤트
# 트리거를 내보낸다. 신호마다 이력 창이 48봉으로 제각각이라 그대로 얹으면 정렬이 어긋난다.
from scripts.live_eth_chart_markers_20260909 import compute_chart_markers  # noqa: E402
# taker_delta_z_climax / short_term_return_z evidence-signal chips REPLACED in-place with their
# TabPFN meta-label models' live probability (2026-08-30, user decision -- unlike V_REBOUND above,
# these stay in the "증거 신호" row and reuse the klines/compute_signals() this endpoint already
# computed each cycle, rather than becoming new standalone "모델 내부 지표" chips with their own
# fetch+cache). See docs/experiments/eth_taker_delta_climax_metalabel_20260829.md.
# 2026-09-14 사용자 결정: **BTC/XRP 증거신호 계산을 내렸다. 우선 ETH 만 한다.**
# 엔드포인트(/api/btc-evidence-signals, /api/xrp-evidence-signals)·로더·워커를 전부 제거했고,
# 프런트는 EVIDENCE_SIGNAL_SUPPORTED_ASSETS = ["eth"] 게이트로 «미지원»을 표시한다.
# 이유: 이 서버는 RTX 3070 Ti 8GB 한 장인데 파이썬 9개가 CUDA 컨텍스트를 들고 VRAM 여유가
# 420MiB 였고, 실효 성능이 이론치의 10%(2.0/21.7 TFLOPS)였다. 자산당 TabPFN 7개(BTC)·5개(XRP)가
# 그 압박의 일부였다. 되살리려면 `git log -S compute_btc_evidence_signals_panel` 로 이 커밋을 찾아
# 되돌리고 워커를 다시 등록하면 된다 -- 계산 모듈(scripts/live_*_evidence_signal_metalabel_*.py)은
# 지우지 않았다.
# 2026-08-30: liquidity_sweep now trained on the SAME Tier0+rsi schema as taker/short_term_
# return_z/dalton_rule2_balance_edge (standard touch-based-MFE redo, replacing the V_REBOUND-model
# relay bridge this import used to be) -- it lives in METALABEL_SIGNALS above and is handled by
# compute_evidence_signal_metalabels() like every other signal there, no separate import/call.
# 베이시스 청산압박 model indicator (replaces 독성/toxicity, 2026-08-27) -- own live spot+perp
# klines fetch each cache cycle (no persistent collector), same "computed here, not bot state"
# category as OI 급변 above. See that module's docstring for the liquidation-crowding validation.
# 5-minute liquidation $ aggregate for the Snapshot tab's liquidation gauge (2026-08-25) -- reads
# tail_risk.duckdb's own per-minute persisted history read-only, same "computed here, not from
# trading_bot.py's dashboard_state.json" category as OI 급변 above. See that module's docstring for
# why this can't just be 5x the trailing 1-minute value the bot already exposes.
from scripts.live_liquidation_5m_signal_20260825 import (  # noqa: E402
    compute_liquidation_5m_signal, compute_liquidation_5m_history)
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
# 2026-09-20 «미시 참고» 카드의 계산부(순수 함수 + 자체점검). 서버는 값을 모아 넣기만 한다.
from dashboard import micro_ref as mref  # noqa: E402
# 2026-09-21 «상황 읽기 · 30분» -- 사용자가 고른 트레이더 읽기를 규칙으로. 순수 함수 + 자체점검.
from dashboard import situation as sit  # noqa: E402
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
# BTC-native regime scorer (2026-09-02). Until now the Snapshot tab's BTC ribbon was a hard-coded
# grey "model not available" band -- app.js gated the ribbon on activeSnapshotAsset === "eth" to stop
# ETH's classifier being drawn over BTC candles (memory eth-dashboard-btc-regime-classifier-not-
# trained-todo-20260831). This is the BTC model that guard was waiting for: same GBM config and the
# same 136 feature_cols as the ETH scorer, trained on BTC's canonical features against a BTC-native
# label (S=24 scale + K=3 confirm) picked by re-screening the grid on BTC rather than porting ETH's
# choice -- ETH's S12_K3 scores only 3/10 on BTC. Same return contract, so it caches/serves
# identically. See scripts/live_regime_btc_signal_20260902.py and
# docs/experiments/btc_regime_s24k3_label_train_20260902.md.
# 2026-09-03: XRP 레짐(S96_K9, 같은 날 S48_K6에서 교체 -- 격자 경계 감사).
# 자산마다 교차자산 슬롯이 다르다 -- XRP는 BTC를 넣는다
# (BTC 캐노니컬은 ETH가 들어있다). live_regime_xrp_signal_20260903.py docstring 참조.
# Session-open volatility risk alert for the evidence-signal chip row (2026-08-26) -- pure
# calendar/clock computation (pandas_market_calendars), no price data, so it needs no cache of its
# own; computed fresh on every evidence-signal refresh. See that module's docstring for the
# same-day empirical research (NYSE open real effect, LSE/JPX marginal) behind the chosen windows.
from scripts.live_session_volatility_alert_20260826 import compute_session_volatility_alert  # noqa: E402
# US macro/corporate event calendar for the Snapshot tab (2026-08-26) -- see that module's
# docstring for the 6 sources (FRED/FOMC-static/Fed Chair HTML/EIA-rule-based/Finnhub/Treasury) and their
# individual caveats. Independent of evidence-signal's klines fetch -- own cache below.
from scripts.live_macro_calendar_20260826 import compute_macro_event_alert  # noqa: E402
# 2026-08-31: per-coin registry for the 4 Snapshot-tab signals wired to BTC this session (basis
# liquidation, liquidation direction, liquidation 5m, liquidation map) -- see
# docs/eth_dashboard_multicoin_expansion_design_20260831.md section 6. Evidence signals/regime/
# specialized-detector (EVIDENCE_SIGNAL_SYMBOL etc. below) are untouched -- those are trained ML
# models with no BTC-trained artifact yet, not something a symbol swap alone can serve.
from scripts.coin_config import COIN_CONFIG  # noqa: E402
# 2026-09-10: 거래소 계정 자체(수동 매매 포함)를 읽는 유일한 경로. trade_journal.jsonl은
# trading_bot.py가 스스로 결정한 것만 담고, 그 봇은 지금 account.enabled=false(페이퍼)다.
from scripts.live_binance_account_20260910 import fetch_account  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    EXIT_VOL_WINDOW, STOP_LOSS_PCT, build_entry_plan, build_exit_plan, build_stop_plan,
    exec_enabled, load_filters, realized_vol_bpm, resolve_exit_position)
from scripts.live_manual_peg_execute_20260912 import run_entry, run_exit  # noqa: E402
# 2026-09-13 보유시간 조건부 위험 사이징. 계산은 사이징 워커가 하고 여기서는 상태파일만
# 읽는다(요청 경로 계산 금지 -- 2026-09-10 스레드 풀 고갈 실장애).
from scripts.live_eth_risk_sizing_policy_20260913 import (  # noqa: E402
    HARD_CAP_X, entry_notional, exit_fraction_required)
from scripts.live_eth_trade_plan_20260913 import (  # noqa: E402
    EXCHANGE_MAX_LEVERAGE, LEVERAGE_STEPS, PRESCRIBE_ACC, plan_now, recommend_hold)
# 2026-09-04: PWA 웹푸시. 사용자가 "다른 작업 중이라 신호를 계속 놓친다"고 해서 추가했다.
# 이 파일은 구독 등록/해지/테스트발송만 담당하고, 실제로 무엇을 언제 보낼지 판단하는 것은
# scripts/live_push_notifier_20260904.py(별도 데몬)다 -- 대시보드 서버는 조회가 있을 때만
# 계산하므로(load_chart_klines_frames()의 60초 캐시) 아무도 안 보고 있으면 트리거 자체가 돌지 않는다.
from scripts.push_webpush_20260904 import (  # noqa: E402
    add_subscription,
    broadcast,
    load_subscriptions,
    remove_subscription,
    subscription_id,
)

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
# 2026-09-16 300 -> 60. **더 자주 해도 새 값이 나오지 않는다** -- 이 계산의 입력은 1시간봉
# 하나뿐이고(LIQUIDATION_MAP_INTERVAL="1h"), 진행 중인 봉은 버린다. 즉 입력은 정시에 한 번만
# 바뀐다. 300초는 그 변화보다 이미 12배 자주 도는 값이었다.
# 그래도 줄인 이유는 **정시 직후의 지연**이다: 새 시간봉이 닫힌 뒤 화면에 뜨기까지 최대
# 5분이었던 것이 1분이 된다. 비용은 실측 0.35초/회(대시보드 최고는 evidence_signal 8.22초)라
# 분당 0.6% 듀티다. 더 줄이는 건 낭비다 -- 입력이 안 바뀐다.
LIQUIDATION_MAP_CACHE_SECONDS = 60
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
LIQUIDATION_MAP_DISPLAY_HOURS = 8  # 2026-09-23 6 -> 8 (짝: SNAPSHOT_CHART_MAX_CANDLES=96) # 2026-08-25 user request: density-history snapshot count for the
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
LIVE_DIR = REPO_ROOT / "data" / "live"
DASHBOARD_DIR = REPO_ROOT / "dashboard" / "live"

# 2026-09-12: 이 띠는 프로세스 메모리(deque)에만 있었다. 배포 워처가 main 전진마다 대시보드를
# 재기동하는데(실측 2026-09-11 하루 12회) 48칸 × 5분 = 4시간을 다시 채워야 해서, 사용자가
# 새로고침할 때마다 «브라우저에 쌓여 있던 과거가 갑자기 사라지는» 증상이 났다. 브라우저는
# 라이브 틱으로 40칸까지 누적하는데 새로고침 후 씨앗(서버 deque)은 10칸뿐이었기 때문이다.


# 2026-09-12: 바이낸스 userTrades 는 **7일 롤링**이라 그 앞의 왕복은 API 에서 그냥 사라진다
# (실측: 09-11 에 보이던 2026-09-03~04 건이 09-12 조회에는 없다). 이 왕복이 «크기 배분이
# 손익을 얼마나 갈랐나»를 재는 유일한 표본인데, 원천이 스스로를 지우고 있어서 표본이 15건
# 근처에 정체한다. 보이는 동안 디스크에 붙여 둔다. 화면에는 쓰지 않는다 -- 축적이 목적이다.
ACCOUNT_TRIP_LEDGER_PATH = LIVE_DIR / "account_round_trips.jsonl"
ACCOUNT_TRIP_RECORD_SECONDS = 300.0
ACCOUNT_TRIP_CHECK_TOL_BP = 0.5   # 올바른 폴딩은 0.0000 이다. 0.5 는 부동소수 여유일 뿐이다


ACCOUNT_TRIP_CARD_ROWS = 400      # 카드가 그릴 최대 왕복 수. 하루 ~20건이면 20일치다


def load_account_trip_rows() -> list[dict]:
    """카드가 그릴 원장 꼬리. **화면이 거래소 payload 대신 이걸 그린다**(2026-09-22).

    userTrades 는 7일 롤링이라, 창 앞에서 열린 포지션은 진입 체결이 사라져 폴딩이 포지션
    한가운데서 시작한다. 그때의 왕복은 수량·진입가가 틀리고 없던 왕복까지 생긴다(실측:
    ETHUSDT 8건 중 3건이 유령, 6건이 항등식 불일치, 승률 50% vs 실제 80%). 원장은 그날그날
    항등식·겹침을 통과한 것만 담고 7일이 지나도 남으므로, 화면이 믿을 수 있는 유일한 판이다.

    ponytail: 꼬리만 읽는다(liq_events_load 와 같은 패턴). 매 조회마다 파일을 훑지만 30초
    캐시 뒤이고 지금 40KB 다. 수 MB 가 되면 바이트 tail 로 바꾼다."""
    try:
        with open(ACCOUNT_TRIP_LEDGER_PATH, encoding="utf-8") as fh:
            tail = deque(fh, maxlen=ACCOUNT_TRIP_CARD_ROWS)
    except OSError:
        return []
    keep = ("symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price",
            "max_qty", "net_pnl")
    rows = []
    for line in tail:
        try:
            trip = json.loads(line)
        except ValueError:
            continue      # 마지막 줄이 쓰다 만 상태일 수 있다
        rows.append({k: trip.get(k) for k in keep})
    return rows


def trip_key(trip: dict) -> str:
    """왕복 하나의 신원. **이 계좌는 헤지 모드**라 같은 심볼에 LONG/SHORT 가 동시에 열린다 --
    side 를 빼면 서로 다른 두 왕복이 한 건으로 뭉개진다."""
    return f"{trip.get('symbol')}|{trip.get('side')}|{trip.get('entry_time')}"


def trip_span(trip: dict) -> tuple[str, str, int, int]:
    """(심볼, 측면, 진입ms, 청산ms). 청산이 없으면 진입 한 점으로 본다."""
    entry = int(trip.get("entry_time") or 0)
    exit_ms = trip.get("exit_time")
    return (str(trip.get("symbol")), str(trip.get("side")), entry,
            int(exit_ms) if exit_ms else entry)


def load_account_trip_keys() -> dict[str, tuple[str, str, int, int]]:
    """이미 적어 둔 왕복의 신원 → 구간. 깨진 줄 하나가 원장 전체를 버리게 두지 않는다 --
    그 줄의 왕복은 seen 에 안 들어가므로 아직 API 창 안에 있으면 다음 주기에 다시 적힌다.

    2026-09-12: 키만 담던 것을 **구간까지** 담게 바꿨다. 키(진입시각)만으로는 중복을 못 막는다 --
    아래 record_account_trips 주석 참조."""
    try:
        lines = ACCOUNT_TRIP_LEDGER_PATH.read_text().splitlines()
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"account_round_trips read failed: {exc}", flush=True)
        return {}
    index: dict[str, tuple[str, str, int, int]] = {}
    for line in lines:
        try:
            trip = json.loads(line)
            index[trip_key(trip)] = trip_span(trip)
        except Exception:
            continue
    return index


def overlaps_recorded(trip: dict, seen: dict[str, tuple[str, str, int, int]]) -> bool:
    """같은 심볼·측면으로 **시간이 겹치는** 왕복이 이미 있는가.

    한 방향 포지션은 같은 시각에 둘일 수 없으므로 겹침은 «같은 왕복» 이라는 뜻이다.
    """
    symbol, side, lo, hi = trip_span(trip)
    for sym2, side2, lo2, hi2 in seen.values():
        if sym2 == symbol and side2 == side and lo <= hi2 and lo2 <= hi:
            return True
    return False


def record_account_trips(payload: dict, seen: dict[str, tuple[str, str, int, int]]) -> int:
    """종료된 왕복 중 처음 보는 것만 덧붙인다(append-only). 반환값은 새로 적은 건수.

    미청산 왕복은 적지 않는다 -- 나중에 청산되면 exit/net_pnl 이 채워지므로 지금 적으면
    같은 왕복의 반쪽 판이 원장에 영구히 남는다.

    🔴키(진입시각)만으로는 중복을 못 막는다. 거래소 조회가 체결 스트림의 **시작을 자르면**
    폴딩이 포지션 한가운데서 시작해 같은 포지션을 «더 늦게 진입한 더 작은 왕복» 으로 만든다.
    키가 다르니 그대로 또 적히고, 원장이 같은 거래를 두 번 센다. 2026-09-12 실측으로 28줄 중
    10줄이 그런 조각이었고, 그 조각들의 +355.02 USDT 가 원장 합계 +232.32 를 만들고 있었다
    (검증된 18줄만 보면 −122.70). 그래서 **시간 겹침**으로도 막는다."""
    if not (isinstance(payload, dict) and payload.get("ok")):
        return 0
    fresh = []
    for trip in payload.get("trades") or []:
        if not trip.get("closed") or trip_key(trip) in seen:
            continue
        if overlaps_recorded(trip, seen):
            print(f"account_round_trips: 겹침으로 건너뜀 {trip_key(trip)} "
                  f"(절단된 체결 스트림의 조각으로 보인다)", flush=True)
            continue
        # 🔴회계 항등식이 안 맞으면 적지 않는다. 완전히 닫힌 왕복은
        #   realizedPnl 합 = (청산VWAP − 진입VWAP) × 수량 × 방향부호 가 **정확히** 성립한다
        # (2026-09-12 재구성한 67건 전부 ±0.0000bp). 그래서 0 이 아닌 값 자체가 폴딩이 잘못됐다는
        # 신호다 -- 겹치지 않는 «빈 구간» 에 생긴 유령 왕복은 겹침 가드가 못 잡는데 이게 잡는다
        # (실측: 재시작 직후 09-10 04:53 SHORT 가 −1.04bp 로 다시 생겼고, 거래소 전체 폴딩에는 없다).
        check = trip.get("pnl_check_bp")
        if check is None or abs(float(check)) > ACCOUNT_TRIP_CHECK_TOL_BP:
            print(f"account_round_trips: 항등식 불일치로 건너뜀 {trip_key(trip)} "
                  f"(check={check}bp) -- 체결 스트림이 잘린 것으로 보인다", flush=True)
            continue
        fresh.append(trip)
    if not fresh:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        with ACCOUNT_TRIP_LEDGER_PATH.open("a") as handle:
            for trip in fresh:
                handle.write(json.dumps({**trip, "recorded_at": now}, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"account_round_trips append failed: {exc}", flush=True)
        return 0
    seen.update({trip_key(t): trip_span(t) for t in fresh})
    return len(fresh)


# 2026-08-27: tail_risk_interceptor.py's event-triggered sibling of dashboard_state.json's
# tail_risk block (see its _write_liq_burst_state() docstring) -- written the instant a new
# liquidation event arrives, not on a 10s timer, for sub-few-second "sudden liquidation" alerting.
LIQ_BURST_STATE_PATH = LIVE_DIR / "liq_burst_state.json"
BTC_EVIDENCE_SHADOW_STATE_PATH = REPO_ROOT / "data" / "live" / "btc_evidence_signal_shadow_state.json"
# 2026-09-10 극점 탐지기 -- 채점은 워커가 하고 대시보드는 읽기만 한다
# (scripts/live_eth_extreme_detector_worker_20260910.py · supervisor_extreme_detector_worker.sh)
REGIME_WIDE24_STATE_PATH = REPO_ROOT / "data" / "live" / "regime_wide24_state.json"
REGIME_BTC_STATE_PATH = REPO_ROOT / "data" / "live" / "regime_btc_state.json"
REGIME_XRP_STATE_PATH = REPO_ROOT / "data" / "live" / "regime_xrp_state.json"
REGIME_MAX_AGE_MIN = 20.0                  # 레짐 워커 주기 300초 + 사이클 13초 여유
MACRO_CALENDAR_STATE_PATH = REPO_ROOT / "data" / "live" / "macro_calendar_state.json"
MACRO_CALENDAR_MAX_AGE_MIN = 90.0          # 달력이라 분 단위 신선도가 의미 없다

# 2026-09-10 24시간 변동성 전망 -- 새 정보원(Deribit DVOL)을 쓰는 첫 지표. 워커가 채점한다
# (scripts/live_eth_vol_forecast_worker_20260910.py · supervisor_vol_forecast_worker.sh).
# ⚠️시간봉 신호라 워커 주기가 300초다 -- 5분봉 카드보다 신선도 기준을 넉넉히 둔다.

# 2026-09-11 횡보→추세 전환 탐지기 -- 방향은 예측하지 않는다(그 축은 닫혔다). 워커가 채점한다
# (scripts/live_eth_breakout_detector_worker_20260911.py · supervisor_breakout_detector.sh).
BREAKOUT_DETECTOR_STATE_PATH = REPO_ROOT / "data" / "live" / "eth_breakout_detector_state.json"
BREAKOUT_DETECTOR_MAX_AGE_MIN = 15.0       # 5분봉 3개 -- 봉 마감 +20초에 도는 워커다

# 2026-09-15 **E|r| 게이트** -- 「지금 큰 움직임이 예상되는가」만 말한다(1일 지평 · 20자산).
# 워커가 채점한다(scripts/live_evr_gate_worker_20260915.py). 봉의 10%만 발동한다.
# 🔴🔴**방향은 말하지 않는다.** 같은 아티팩트의 방향 분류기는 **실계좌 72왕복에서 적중 47.2%**
#   (동전 아래)이고 게이트가 고른 좋은 자리일수록 더 나빴다(−51.18bp) -- 호메로스 §5.36-R.
#   그래서 워커는 `art["dir"]` 를 **아예 호출하지 않는다**. 이름도 direction_gate 가 아니다.
# 게이트 자체의 근거: 두 독립 설계에서 짝지은 증분 +8.25 / +7.44bp/일 · **CI 둘 다 0배제** ·
#   MDD 를 −47.4% -> −8.4% 로 줄인다. **손실 차단기**이지 수익 생성기가 아니다. 표시 전용.
# 2026-09-19 GEX. 🔴duckdb(`deribit_gex.duckdb`)는 **열지 않는다** -- 단일 writer 라
#   매시 cron 과 부딪히면 `Could not set lock` 이다. 수집기가 떨구는 JSON 만 읽는다.
GEX_STATE_PATH = REPO_ROOT / "data" / "live" / "deribit_gex_state.json"
GEX_MAX_AGE_MIN = 150.0   # 매시 cron -- 두 사이클 놓치면 죽은 것으로 본다

# 2026-09-19 호가 히트맵. 🔴**전용 스레드풀**을 쓴다 -- 디스크에서 수 MB 를 읽는 동안
#   기본 executor 를 물면 같은 풀을 쓰는 다른 핸들러가 같이 막히고, 이벤트루프에서 읽으면
#   SSE 가 멈춘다(설계서 dashboard_orderflow_footprint_heatmap_design_20260914 P2).
# 🔴symbol 은 **파일 경로로 들어간다**. 화이트리스트로만 받는다(`../` 탈출 차단).
HEATMAP_SYMBOLS = {"ethusdt", "btcusdt"}
HEATMAP_MAX_COLS = 900          # 캔버스 폭 상한 -- 이보다 넓게 그릴 화면이 없다
HEATMAP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                         thread_name_prefix="heatmap")
EVR_GATE_MAX_AGE_MIN = 30.0          # 워커 주기 300초 x 6

# 2026-09-11 크기 가늠자 -- 역변동성 사이징의 실시간 눈금
# (scripts/live_eth_position_sizing_worker_20260911.py). 읽기만 한다.
# 🔴수익을 예측하지 않는다. 검정된 것은 위험 축뿐이다(무작위 진입 82,167건에서 평균 명목 동일
#   조건에 표준편차 -18% · 50배 청산 도달률 13.9%->9.1%). 카드 문구도 그렇게 쓴다.
POSITION_SIZING_STATE_PATH = REPO_ROOT / "data" / "live" / "eth_position_sizing_state.json"
POSITION_SIZING_MAX_AGE_MIN = 30.0         # 워커 주기 300초 x 6

BTC_EVIDENCE_CTX_REPORT_PATH = REPO_ROOT / "data" / "labels" / "btc_5m_evidence_signal_live_contexts_20260902" / "contexts_report.json"
MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT", "xrp": "XRPUSDT", "hype": "HYPEUSDT"}
# 2026-09-19 수동 진입이 나갈 심볼. **차트·시세·청산맵과 분리한다** -- MARKET_SYMBOLS 는
# 화면 전체가 공유하므로 그걸 뒤집으면 데이터 출처까지 통째로 바뀐다.
# ⭐왜 USDC 인가: 계정 실요율(/fapi/v1/commissionRate) ETHUSDT 메이커 2.0bp vs **ETHUSDC 0bp**.
#   peg-maker 섀도우 실측 전량 왕복 ETHUSDC 1.36bp vs ETHUSDT 5.77bp (2,056 leg, 09-16~).
# 🔴켜기 전 필수: 계정이 단일자산 담보(multiAssetsMargin=False)면 **USDC 잔고가 있어야** 한다.
#   2026-09-19 실측 USDT 1,355 / USDC 0 -- 이 상태로 ETHUSDC 주문은 증거금 부족으로 거절된다.
MANUAL_EXEC_SYMBOL = (os.getenv("DASHBOARD_MANUAL_EXEC_SYMBOL", "").strip().upper()
                      or MARKET_SYMBOLS["eth"])
# 대시보드가 **보여줄** 코인 (2026-09-16 사용자 요청 "나머지 코인은 리소스 먹지 않게 비활성").
# 실측으로 본 실제 절약: 엔드포인트별 코인 계산은 요청이 와야 도는 on-demand(swr_cached, 콜드
# 0.05초)라 안 쓰면 안 돈다. **무조건 도는 건 SSE 루프의 시세 팬아웃 하나뿐**이었다 --
# EVENT_POLL_SECONDS(2.5초)마다 5코인 티커를 전부 가져왔다(= 120요청/분). 그걸 이 목록으로 줄인다.
# 되돌리려면 환경변수 하나: DASHBOARD_ASSETS="eth,btc,sol,xrp,hype"
# ⚠️계좌 패널(fetch_account)은 이 목록을 **쓰지 않는다** -- 다른 코인에 포지션이 있으면 그건
#   보여야 한다. 화면에서 코인을 지우는 것과 «내 돈이 어디 있는지»는 다른 문제다.
DASHBOARD_ASSETS = [a.strip().lower() for a in os.getenv("DASHBOARD_ASSETS", "eth").split(",")
                    if a.strip().lower() in MARKET_SYMBOLS] or ["eth"]
DASHBOARD_TICKER_SYMBOLS = {a: MARKET_SYMBOLS[a] for a in DASHBOARD_ASSETS}
# 볼륨 풋프린트 (2026-09-15) -- ETH 만. 자세한 근거는 collect_footprint() 주석.
FOOTPRINT_SYMBOL = "ETHUSDT"
FOOTPRINT_BAR_SECONDS = 300      # 차트 캔들과 같은 5분봉
FOOTPRINT_BARS = 12              # 1시간
FOOTPRINT_BUCKET = 0.5           # 가격 버킷(달러). 화면 행 크기는 이것의 배수로 클라가 묶는다
# 백필 페이스. 요청당 가중치 20 이므로 2.5초 = ~480/분. 한도는 IP 당 2400/분인데 그 IP 를
# **트레이딩 봇과 대시보드의 다른 폴링이 같이 쓴다** -- 1.2초(~1000/분)로 돌렸더니 2026-09-15
# 실제로 429 가 났다. 차트 하나 빨리 채우자고 주문 경로를 위협할 이유가 없다(백필은 최신 봉부터
# 채우므로 느려도 «지금 보는 자리»는 1분 안에 찬다).
FOOTPRINT_CATCHUP_SECONDS = 2.5
# 받은 테이프를 디스크에 남긴다. 안 남기면 재시작마다 1시간을 REST 로 다시 사와야 하는데,
# 대시보드는 배포마다 재시작된다(2026-09-15 하루에도 여러 번). 남기면 재시작 뒤 메울 구간이
# «꺼져 있던 시간»으로 줄어든다 -- 보통 몇 초다.
FOOTPRINT_SNAPSHOT_PATH = LIVE_DIR / "footprint_eth.json"
FOOTPRINT_SNAPSHOT_SECONDS = 30.0
# 화면 토글이 고를 수 있는 **가장 긴 창**(4h). 저장·복원은 여기에 맞춘다 -- 12봉만 저장하면
# 재시작(배포마다 하루 여러 번) 직후 4h 를 골라도 1h 밖에 안 보인다. 실측 51KB/48봉이고
# 30초마다 덮어써도 하루 150MB 쓰기다(288봉이면 1GB 라 링 전체는 여전히 안 쓴다).
#
# ⭐**복원 경로는 이 스냅샷 하나다**(2026-09-19 사용자 결정). 체결 테이프 duckdb 에서 읽어
#   오는 길도 만들어 봤다 -- 셀 단위 오차 0.0 ETH 로 정확하긴 했는데, 실측 **250ms vs
#   0.8ms** 로 300배 느리고 20.7MB 파일을 여는 대가였다. 같은 4시간이 이 json 안에 이미
#   있으므로 두 벌을 둘 이유가 없다. duckdb 는 수집기가 계속 쌓는 연구용 아카이브로 남는다.
# 🔴백필(REST)도 **넓히지 않는다**. 1시간이 ~200요청인데 4시간이면 예산 400을 넘고, 그 IP 를
#   트레이딩 봇이 같이 쓴다. 스냅샷이 못 덮는 구간은 WS 가 돌면서 자연히 찬다.
# 🔴이 상수의 뜻은 «화면이 고를 수 있는 가장 긴 창»이다(아래 저장 주석). 2026-09-23
#   12h 토글이 생겨 정의상 144 가 됐다 -- 48 로 두면 재시작 뒤 12h 창이 4h 만 보인다
#   (실측: 오늘 배포 직후 144 봉 요청에 50 봉만 나왔다). 스냅샷 85KB -> ~255KB.
FOOTPRINT_MAX_WINDOW_BARS = 144


def footprint_window_bars(request) -> int:
    """`?bars=N` -- 화면의 창 토글(1h/2h/4h = 12/24/48봉, 2026-09-19 사용자 요청).

    봉 링(FOOTPRINT_KEEP_BARS)보다 길게는 줄 수 없다. 신뢰경계 입력이므로 파싱 실패와
    범위를 여기 한 곳에서 닫는다 -- 두 엔드포인트가 같은 함수를 쓰게 두는 이유다.
    있는 봉보다 크게 달라고 해도 문제가 없다: 슬라이스가 알아서 있는 만큼만 준다."""
    try:
        n = int(request.query.get("bars", FOOTPRINT_BARS))
    except (TypeError, ValueError):
        return FOOTPRINT_BARS
    return max(1, min(n, FOOTPRINT_KEEP_BARS))
# ── 리테일/고래 수급 (2026-09-19) ───────────────────────────────────────────
# 셀은 [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도] 6칸이다.
# **중형($10k~$100k)은 칸이 없다** -- 매수 - 고래매수 - 리테일매수로 정확히 나온다.
#
# ⭐경계는 체결 테이프 수집기에서 **가져다 쓴다**. 화면과 저장이 같은 숫자를 써야 나중에
#   「그날 화면의 고래」와 「DB 의 고래」가 같은 것이 된다. 여기 상수를 새로 만들면 두 개가
#   조용히 갈라진다 -- 이 저장소가 레짐 분류기에서 한 번 겪은 일이다(2026-09-17).
# ⭐왜 분위가 아니라 달러 고정인지, 왜 경계가 둘인지는 그 파일의 상수 주석에 실측과 함께 있다.
FOOTPRINT_KEEP_BARS = 288
# 최근 5분을 **1초 해상도**로 보는 화면용 링(2026-09-19 사용자 지시). 5분봉 리본을 걷어내고
# 이걸로 갈음했다 -- 「바로바로」가 뜻하는 건 5분봉이 아니라 초 단위였다.
# 가격빈을 버리고 초마다 한 칸만 남긴다(가격축은 수급 프로파일이 따로 본다). 360초를 두는 건
# 화면이 300초를 그리는데 경계에서 모자라지 않게 하려는 여유다.
# 🔴2026-09-20: 660 으로 올렸다가 다시 내렸다. 660 은 «창 왼쪽이 이전 봉에 걸친다»는
# 시안 G 전제였는데, 채택된 시안 H 는 **현재 5분봉 하나**만 그린다(최악 now-300).
# 두 봉치(600)에 여유를 더해 두면 봉이 바뀌는 순간과 폴링 지각을 모두 덮는다.
SUPPLY_1S_SECONDS = 660
# 같은 5분 창의 **미결제약정(OI)**. 「신규 계약이 몇 개 생겼나」는 OI 의 증분이다.
# 🔴바이낸스는 OI 를 매 초 갱신하지 않는다 -- /fapi/v1/openInterest 를 1초 간격 20회 때려보니
#   서로 다른 스냅샷은 6개, 간격 0.6~6.7초(중앙 ~3.5초)였다(2026-09-19 실측). 그래서 이 링은
#   **값이 바뀐 초에만** 점을 남긴다. 매초 같은 값을 복붙하면 없는 해상도를 있는 척하게 된다.
#   1초 폴링은 그 3~7초 갱신을 가장 빨리 잡기 위한 것이다(가중치 1 x 60/분, IP 한도 2400/분).
OI_1S_URL = "https://fapi.binance.com/fapi/v1/openInterest"
# 2026-09-19 1.0 -> 0.25. 1초로는 갱신의 약 5%를 놓쳤다: 0.2초로 훑으면 3.66초당 1회인데
# 1초 폴링으로 쌓인 건 3.86초당 1회였다(실측 간격 최소 0.63초 · 1초 미만이 40건 중 2건).
# 놓친 갱신은 되살릴 방법이 없다 -- 바이낸스는 1초 OI 이력을 안 준다(openInterestHist 는 5분).
# weight 1 x 240/분, IP 한도 2400/분이라 여유가 10배다.
OI_1S_POLL_SECONDS = 0.25
# 화면 링(6분)은 재기동하면 비고, 5분 누적 패널은 몇 시간을 봐야 한다 -- 그래서 남긴다.
# 🔴체결 테이프 duckdb(TAPE_DB_PATH)에 끼워 넣지 않는다: 저쪽 writer 는 별도 프로세스가
#   **연결을 붙들고** 있어 외부에서는 read_only 조차 거부된다(2026-09-19 실측). 여기서는
#   대시보드가 유일한 writer 이고, 매 flush 마다 연결-작업-닫기라 연구/감시 쪽 읽기를 막지 않는다.
OI_1S_DB_PATH = LIVE_DIR / "oi_1s.duckdb"
OI_1S_TABLE = "oi_1s"
OI_1S_FLUSH_SECONDS = 10.0   # 크래시 시 잃는 최대치(스냅샷 ~3개). 파일 락 잡는 횟수와의 맞교환.
OI_5M_BAR_SECONDS = 300
OI_5M_WINDOW_BARS = 48       # 4시간

# ── 미시 참고 (2026-09-20, 사용자 지시 "있는 것들 모두 붙여줘") ─────────────────
# docs/experiments/eth_realtime_five_stream_1s_joint_analysis_20260920.md §4·가이드가 가리킨 다섯 표시:
# ①QI×OFI 동조 색 ②같은 시간대 분위 게이지 ③S/R 근접 경보 ④거래량 폭발→OI 급감 ⑤@forceOrder 이벤트 로그.
# 원천은 전부 이 프로세스가 이미 갖고 있거나(체결 초 링·OI 링·청산맵 캐시·래스터 read_window)
# 가벼운 것 하나를 더 받는다(REST bookTicker 1/s · @forceOrder WS -- 조용한 스트림).
MICRO_REF_POLL_SECONDS = 1.0
MICRO_BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"   # weight 1 · 1/s
# 🔴경로가 «/market/ws/» 여야 한다(봇 tail_risk_interceptor 와 같은 것). 2026-09-20 서버 실측: 같은 프로세스에서
#   /ws/ethusdt@forceOrder · /market/ws/ethusdt@forceOrder · /ws/!forceOrder@arr 셋을 동시에 열어 두니 22:40:48 의
#   실제 ETH 청산(SELL 1.000 @2575.83)이 **/market/ws/ 에만** 왔다. /ws/ 는 연결은 되는데(핸드셰이크 OK, 오류 0)
#   이벤트를 안 준다 -- 첫 배포에서 tail_risk 가 4건을 적는 동안 이 카드가 0건이었던 원인.
FORCE_ORDER_WS_URL = "wss://fstream.binance.com/market/ws/ethusdt@forceOrder"
LIQ_EVENTS_PATH = LIVE_DIR / "liq_events.jsonl"      # ⑤ 원시 이벤트. 봇의 tail_risk 는 1분 합만 남긴다
MICRO_TAPE_DB_PATH = LIVE_DIR / "trade_tape.duckdb"  # ② 기준선(시간대별 분위). 읽기 전용, 1시간마다
# ⑥ 마크가격 스트림(2026-09-21): 예상 펀딩(«어느 쪽이 갇혔나») + 마크−인덱스 베이시스(«누가 주도하나»).
#    🔴forceOrder 와 같은 함정 -- `/ws/` 는 연결되는데 이벤트 0, `/market/ws/` 만 온다(09-21 dev 실측 0 vs 8건/6초).
MARK_PRICE_WS_URL = "wss://fstream.binance.com/market/ws/ethusdt@markPrice@1s"
MARK_PRICE_DB_PATH = LIVE_DIR / "mark_price_1s.duckdb"
MARK_PRICE_TABLE = "mark_price_1s"
MARK_PRICE_RING_S = 7200          # 메모리 링(초). 베이시스 임계 분위를 «창 Δ» 표본 60개 이상에서 잡으려면 1시간 넘게 필요
MICRO_BASELINE_SECONDS = 3600
SITUATION_EVERY_TICKS = 1                              # micro-ref 1초 루프마다 (09-21 사용자 «급변 때 느리다» -- 실측 비용 30ms, duckdb 둘은 아래서 5초 캐시)
SITUATION_LOG_PATH = LIVE_DIR / "situation_log.jsonl"    # 예측 장부 -- 30분 뒤 결과와 맞춰 적중률을 낸다
SITUATION_LOG_MIN_GAP_S = 300                          # 상태가 안 바뀌어도 이 간격으로 한 줄
SITUATION_VA_ROW_USD = 3.0                             # 가치영역 행 폭(차트의 행과 같다)
SITUATION_HORIZON_S = 1800
SITUATION_LOG_KEEP = 2000              # 메모리에 드는 예측 수(≈하루). 카드가 «표본·에피소드»를 하루 단위로 말하려면 필요
SITUATION_MINUTE_LIMIT = 500            # 해결 전용 1분봉(≈8시간). 5분봉은 창 앞 최대 5분을 버리고 봉 안 순서를 모른다


def oi_1s_persist(rows: list[tuple[int, float]]) -> None:
    """OI 스냅샷을 duckdb 에 남긴다(연결-작업-닫기). rows 는 (ts_ms, open_interest).

    PK 는 **밀리초**다. 초로 잡으면 같은 초에 온 둘째 스냅샷이 조용히 사라진다 -- 2026-09-19
    실측으로 10분간 갱신 163개 중 8개(4.91%)가 그렇게 버려지고 있었다(간격 최소 0.14초).
    재기동 직후 겹치는 구간을 다시 써도 PK 가 무시하므로 호출부는 «어디까지 저장했나»를
    따로 들고 있지 않아도 된다."""
    if not rows:
        return
    with duckdb_path_lock(OI_1S_DB_PATH):
        con = duckdb.connect(str(OI_1S_DB_PATH))
        try:
            # 2026-09-19 이전 스키마(ts_sec PK)를 한 번만 밀리초로 옮긴다. 옛 행은 그 초의
            # 대표값이므로 .000ms 에 놓는다 -- 봉 집계는 초로 접으므로 결과가 안 변한다.
            legacy = con.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = ? AND column_name = 'ts_sec'", [OI_1S_TABLE]).fetchone()[0]
            if legacy:
                # 한 트랜잭션. 중간에 죽으면 통째로 없던 일이 된다 -- 네 문장이 따로 커밋되면
                # RENAME 만 성공한 상태로 남아 옛 행이 서빙 테이블에서 끊긴다.
                con.execute("BEGIN TRANSACTION")
                con.execute(f"ALTER TABLE {OI_1S_TABLE} RENAME TO {OI_1S_TABLE}_sec_legacy")
                con.execute(f"""CREATE TABLE {OI_1S_TABLE} (
                    ts_ms BIGINT, symbol VARCHAR, open_interest DOUBLE,
                    PRIMARY KEY (ts_ms, symbol))""")
                con.execute(f"INSERT INTO {OI_1S_TABLE} "
                            f"SELECT ts_sec * 1000, symbol, open_interest "
                            f"FROM {OI_1S_TABLE}_sec_legacy")
                con.execute(f"DROP TABLE {OI_1S_TABLE}_sec_legacy")
                con.execute("COMMIT")
                print("oi-1s: ts_sec -> ts_ms 스키마 이관 완료", flush=True)
            con.execute(f"""CREATE TABLE IF NOT EXISTS {OI_1S_TABLE} (
                ts_ms BIGINT, symbol VARCHAR, open_interest DOUBLE,
                PRIMARY KEY (ts_ms, symbol))""")
            con.executemany(f"INSERT OR IGNORE INTO {OI_1S_TABLE} VALUES (?, ?, ?)",
                            [(int(ms), FOOTPRINT_SYMBOL.lower(), float(v)) for ms, v in rows])
        finally:
            con.close()


def mark_price_persist(rows: list[tuple[int, float, float, float, int]]) -> None:
    """markPrice@1s 를 duckdb 에 남긴다. rows = (ts_ms, mark, index, funding_rate, next_funding_ms). PK ts_ms(밀리초).
    재기동 직후 겹치는 구간을 다시 써도 PK 가 무시한다(oi_1s_persist 와 같은 계약)."""
    if not rows:
        return
    with duckdb_path_lock(MARK_PRICE_DB_PATH):
        con = duckdb.connect(str(MARK_PRICE_DB_PATH))
        try:
            con.execute(f"""CREATE TABLE IF NOT EXISTS {MARK_PRICE_TABLE} (
                ts_ms BIGINT, symbol VARCHAR, mark DOUBLE, index_px DOUBLE, funding_rate DOUBLE, next_funding_ms BIGINT,
                PRIMARY KEY (ts_ms, symbol))""")
            con.executemany(f"INSERT OR IGNORE INTO {MARK_PRICE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                            [(int(ms), FOOTPRINT_SYMBOL.lower(), float(mk), float(ix), float(fr), int(nf)) for ms, mk, ix, fr, nf in rows])
        finally:
            con.close()


def oi_5m_buckets(bars: int) -> list[list[float]]:
    """5분 봉별 [봉시각, 신규계약(Δ), 봉 끝 OI, 스냅샷 수, 공백여부].

    Δ 는 «직전 봉 끝 -> 이 봉 끝»이다. 봉 안(열림->닫힘)만 재면 봉 사이 3~7초에 일어난 변화가
    통째로 사라진다. 다만 **앞 봉이 비어 있으면**(수집기 정지) 그 공백 동안의 변화를 이 봉에
    몰아주지 않고 봉 안에서만 재고 gap=1 로 알린다 -- 0이 아니라 «모름»이다.
    """
    if not OI_1S_DB_PATH.exists():
        return []
    floor = (int(time.time()) // OI_5M_BAR_SECONDS - (bars - 1)) * OI_5M_BAR_SECONDS
    try:
        with duckdb_path_lock(OI_1S_DB_PATH):
            # 🔴read_only 여야 한다. 쓰기로 열면 이 조회가 도는 동안(클라마다 15초 주기)
            #   연구/감시 쪽 외부 read_only 연결이 거부된다 -- 2026-09-19 실측 30회 중 2회.
            #   쓰기는 oi_1s_persist(10초 flush)뿐이고 같은 in-process 락이 둘을 갈라 준다.
            con = duckdb.connect(str(OI_1S_DB_PATH), read_only=True)
            try:
                rows = con.execute(f"""
                    SELECT (ts_ms // (? * 1000)) * ?          AS bar,
                           arg_min(open_interest, ts_ms)      AS oi_open,
                           arg_max(open_interest, ts_ms)      AS oi_close,
                           count(*)                           AS n
                    FROM {OI_1S_TABLE}
                    WHERE symbol = ? AND ts_ms >= ?
                    GROUP BY 1 ORDER BY 1
                """, [OI_5M_BAR_SECONDS, OI_5M_BAR_SECONDS, FOOTPRINT_SYMBOL.lower(),
                      (floor - OI_5M_BAR_SECONDS) * 1000]).fetchall()  # 한 봉 더: 첫 봉의 기준점
            finally:
                con.close()
    except Exception as exc:  # noqa: BLE001 -- 아직 테이블이 없거나(첫 가동) 잠깐 잠겼다
        print(f"oi-5m read failed: {exc}", flush=True)
        return []
    out: list[list[float]] = []
    prev_bar: int | None = None
    prev_close = 0.0
    for bar, oi_open, oi_close, n in rows:
        bar = int(bar)
        gap = prev_bar is None or bar - prev_bar > OI_5M_BAR_SECONDS
        delta = (oi_close - oi_open) if gap else (oi_close - prev_close)
        prev_bar, prev_close = bar, oi_close
        if bar >= floor:
            out.append([bar, round(delta, 3), round(oi_close, 3), int(n), 1 if gap else 0])
    return out

# 2026-09-16 2.5 -> 1.0 (사용자 "최대한 빠르게"). 이 값이 곧 **현재가 선의 지연**이다 --
# 차트는 SSE 푸시마다 다시 그린다. 코인을 ETH 하나로 줄이면서(DASHBOARD_ASSETS) 티커 요청이
# 2.5초×5코인 = 120/분 이었던 것이 1초×1코인 = 60/분 이 된다 -- **더 빨라지면서 절반이다**.
# 상태 파일은 stat 으로 바뀐 것만 다시 읽으므로(etag) 주기를 당겨도 재직렬화가 늘지 않는다.
EVENT_POLL_SECONDS = 1.0
# How long a cached payload may keep being served while its (expensive) replacement computes.
# Sized to cover a worst-case TabPFN refit (43s measured under GPU contention) with wide margin:
# past this the payload is treated as cold again and the request blocks for a current reading.
STALE_GRACE_SECONDS = 600
SWR_SLOW_LOG_SECONDS = 0.30   # 이 아래는 «파일 읽기» 취급, 로그에 안 남긴다
MARKET_HISTORY_CACHE_SECONDS = 300
# 3 + N개의 서명 GET(weight 5씩)이라 폴링 자체는 싸다. 포지션은 실시간성이 필요하고
# 체결내역은 안 변하지만, 캐시를 둘로 쪼개는 값어치는 없어서 한 페이로드 30초로 묶었다.
BINANCE_ACCOUNT_CACHE_SECONDS = 30
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


NOCACHE = {"Cache-Control": "no-cache"}


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
        return datetime.fromisoformat(str(raw)[:19]).timestamp()
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


COIN_INDICATOR_CACHE_SECONDS = 20
# nif_whale은 간헐적이라 최신 1행만 보면 절반이 빈 값이다 -- 이 창 안의 마지막 값을 쓴다.
MICRO_LOOKBACK_MIN = 15
# ETH 톤 스트립과 같은 모양(app.js MICRO_HISTORY_MAX=48, 5분 간격 = 4시간)
MICRO_STRIP_SAMPLES = 48


def _age_min(ts: Any) -> float | None:
    """UTC 문자열 -> 지금까지 경과 분. 원장은 tz 표기가 없는 UTC 문자열이다."""
    try:
        dt = datetime.fromisoformat(str(ts).replace(" ", "T"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - dt).total_seconds() / 60.0, 1)


def worker_payload(path: Path, max_age_min: float, *, ts_field: str = "updated_utc",
                   require_ok: bool = False, stamp_available: bool = False,
                   bare_missing: bool = False, extra_missing: dict | None = None) -> dict[str, Any]:
    """워커 상태 파일을 신선도까지 판정해 돌려준다.

    ⚠️인라인 폴백은 두지 않는다. 폴백을 두면 워커가 죽어도 표는 정상으로 보이고 대신
      대시보드가 5초씩 느려진다(그게 이 구조를 만든 이유다) -- 죽었으면 죽었다고 보여야 한다.
    `available` 은 워커가 아니라 **여기서** 찍는다. 신선도를 판정하는 쪽이 찍어야 워커
      버전이 달라도 계약이 안 깨진다(api_liq_burst_state 와 같은 방식).
    `require_ok` 는 워커가 외부 조회에 실패한 사이클(`ok: False`)을 오류로 떨어뜨린다 --
      그때 옛 톤을 그대로 보이면 화면이 «지금 조용하다»고 거짓말을 한다.
    """
    # `extra_missing` 은 **결측/조회실패에만** 얹는다. stale 은 워커가 쓴 값을 그대로 보여주므로
    # 빈 history/times 로 덮으면 안 된다(차분 테스트로 확인한 원 동작).
    base = {"available": False, "error": "worker_state_missing",
            **({} if bare_missing else {"tone": "neutral"}), "subText": "데이터 없음"}
    st = load_json(path)
    if not st:
        return {**base, **(extra_missing or {})}
    if require_ok and not st.get("ok"):
        # bare_missing 계열은 워커 상태를 얹지 않고 결측과 같은 모양으로 떨어뜨린다(원 동작).
        return {**base, **(extra_missing or {})} if bare_missing else {
            **st, **base, **(extra_missing or {}),
            "error": "worker_fetch_failed", "subText": "오류"}
    age = _age_min(st.get(ts_field))
    stale = round(age, 1) if age is not None else None
    if age is not None and age > max_age_min:
        return {**st, **base, "error": "worker_stale", "stale_min": stale}
    return {**st, **({"available": True} if stamp_available else {}), "stale_min": stale}


def regime_payload(path: Path) -> dict[str, Any]:
    """레짐 리본 워커 상태(2026-09-14). 셋 다 **매 사이클 13초**로 일정해서(서버 실측
    wide24 13.73/12.86/13.30s) 캐시가 만료될 때마다 그대로 다시 낸다 -- 요청 경로에 둘 이유가 없다.
    셋은 워커 **한 프로세스**가 순서대로 돈다(live_signal_worker.py 의 --compute/--state 반복)."""
    return worker_payload(path, REGIME_MAX_AGE_MIN, extra_missing={"regime": None, "history": []})


def macro_calendar_payload() -> dict[str, Any]:
    """거시 달력 워커 상태(2026-09-14, 콜드 51.03초 -- 외부 6개 소스를 동기 requests 로 친다)."""
    return worker_payload(MACRO_CALENDAR_STATE_PATH, MACRO_CALENDAR_MAX_AGE_MIN,
                          ts_field="generated_at", extra_missing={"events": []})


def breakout_detector_payload() -> dict[str, Any]:
    """횡보->추세 전환 탐지기 워커 상태."""
    return worker_payload(BREAKOUT_DETECTOR_STATE_PATH, BREAKOUT_DETECTOR_MAX_AGE_MIN,
                          require_ok=True, stamp_available=True,
                          extra_missing={"history": [], "times": []})


def gex_payload() -> dict[str, Any]:
    """옵션 감마 노출(GEX). **참고 표시 전용이고 신호가 아니다.**

    🔴라벨을 이론대로 붙이면 안 된다 -- 실측 rho(GEX, 전방RV) = **+0.44~+0.51** 로 부호가
      반대다(후행RV 통제 후에도 +0.25~+0.34). GEX 는 명목 달러라 «옵션시장 활동 수준 =
      변동성»의 **결과 대리변수**로 작동한다. 그래서 두 축으로 가른다:
        level  = total            -- 변동성 대리(이론 부호 아님)
        struct = front / total    -- total 을 통제하면 front 가 이론 부호를 회복한다(t -6.22)
    판정일: 1h 2026-09-28 / 4h 10-17 (독립일이 그때 44/63 이 된다). 그 전까지 CI 는 전부
    0 을 포함하므로 화면은 «참고»라고 말해야 한다.
    출처: docs/experiments/eth_gamma_zomma_graphic_research_20260916.md
    """
    return worker_payload(GEX_STATE_PATH, GEX_MAX_AGE_MIN, ts_field="generated_at",
                          stamp_available=True, bare_missing=True,
                          extra_missing={"currencies": {}})


# 2026-09-14 사용자 요청: 사이징 변동성 모델(A)을 «모델 내부 지표»에 신호로 띄운다.
# 🔴여기서 **모델을 돌리지 않는다** -- 사이징 워커(`live_eth_position_sizing_worker_20260911`)가
#   이미 매 300초 `sizing_model.pred_vol` 을 상태파일에 남긴다. 요청 경로 계산 금지(2026-09-10 실장애).
# 🔴컷은 **절대값이 아니라 비율(pred / ref_pred)** 이다. 이 아티팩트는 **기계마다 다르게 학습된다**
#   -- 학습 CSV 길이가 다르기 때문이다(모듈 verify() 주석: 로컬 385k · 서버 175k). 2026-09-14 실측
#   md5 도 달랐고 ref_pred 가 0.0015594(로컬) vs 0.0016025(서버)였다. 절대 분위를 박으면 배포 기계에서
#   어긋난다. 비율은 같은 분포의 중앙으로 나눈 값이라 기계가 달라도 거의 같다:
#   ref_pred 는 워커가 내려주는 그 아티팩트 자신의 값을 쓴다(자기보정).
# 🔴2026-09-14 재설정(사용자 지적 "텍스트는 평소라는데 라벨은 주의"): 처음엔 학습창 40%/80%
#   분위를 경계로 썼는데, 그러면 **40~80% 구간 = 평소**를 「주의」라고 부르고 시간의 40%를
#   주황으로 칠하게 된다. 위험도 어휘(안정/주의/위험)는 «지금 위험한가»를 말하는 자리다.
#   그래서 **평소는 안정**으로 옮긴다: 안정 = 학습창 하위 80%(<1.39배) · 주의 = 상위 20~5%
#   · 위험 = 상위 5%(>=1.88배). 서버 실측 비율 분위 80/90/95 = 1.392 / 1.646 / 1.882.
#   최근 구간 점유율: OOS 74.7/15.6/9.7% · TEST 94.0/4.4/1.5%.
# 근거: research_eth_vol_forecast_head_to_head_20260914.py — 이 모델은 앞으로 4시간 실현변동성
#   **수준**을 맞히고(1시간 앞 ρ .642 · 4시간 .712, 단순 rv48 은 .553/.486), 「확장」은 못 맞힌다
#   (AUC .46~.52). 그래서 어휘는 위험도 3등급이고 **방향 신호가 아니다**.
# 🔴2026-09-15 «안정만 뜬다»(사용자). 버그가 아니라 **기준이 고정**이어서였다 -- `ref_pred` 는
#   학습창(2021-12~2025-08) 예측 중앙값으로 아티팩트에 박히는데 2026-04 이후 ETH 가 그 「평소」보다
#   25% 조용하다(비율 중앙값 0.78). 고정 기준 ÷ 내려앉은 분포 = 최근 7일 98.3% 「안정」.
#   실현변동성 비율도 같은 방향(p50 0.73)이라 **모델이 틀린 게 아니라 레짐이 내려앉은 것**이다.
#   그래서 등급은 워커가 주는 **최근 30일 예측 중앙값**(`ref_pred_recent`)으로 자른다.
#   재현: 컷 그대로 두고 기준만 롤링하면 전구간(2022-06~) 81.2/13.5/5.3%, 최근 30일 80.3/14.6/5.1%.
#   ⚠️**수량 배수는 그대로 고정 기준**이다 -- 사이징은 «평소보다 조용하면 크게»가 의도이고
#   배포 공식(`ref_pred/pred`)이 그것이다. 둘을 같은 값으로 합치면 안 된다.
VOL_LEVEL_RATIO_CALM = 1.39        # 이 아래 = 안정(평소 이하) — 상위 20% 경계
VOL_LEVEL_RATIO_HOT = 1.88         # 이 위 = 위험 — 상위 5% 경계
VOL_LEVEL_REF_FALLBACK = 0.00160   # 워커가 ref_pred 를 안 줄 때만 (서버 아티팩트 기준)

# 2026-09-21 ⭐**확장 배수 + 확률**. 위 `ratio`(평소 대비)와 **다른 축**이다 --
# 이건 `pred / rv48` = «앞으로 4시간이 직전 4시간 대비 몇 배».
# 🔴카드가 오래 「곧 커진다는 못 맞힌다(AUC .46~.52)」라고 적어 왔는데 **채점을 잘못한 것**이었다:
#   확장은 «비율» 질문인데 모델의 «수준»으로 채점했다(수준은 스케일을 갖고 비율은 안 갖는다).
#   올바른 점수인 pred/rv48 로 재면 OOS 109,267행 AUC **0.8237**, rv48 십분위 통제 후 **0.7865**
#   (10/10 셀 0.69 이상 · 분기 5/5 0.729~0.791 · 일블록 부트 CI [0.754, 0.822] 로 0.5 배제).
#   보정도 편향 없다: log(실제배수) = -0.0091 + **1.0365**·log(예측배수).
# 아래 두 상수는 TRAIN(<=2025-08-31, 385,453행)에서 적합한 로지스틱이다.
#   OOS Brier 0.1388(상수예측 0.1961) · 십분위 보정오차 <=4pp.
# ⚠️점으로 읽으면 안 된다 -- 예측 1.5배일 때 실제 68% 구간이 [1.07, 2.12]배다(잔차 SD 0.34).
#   그래서 화면은 배수와 **확률을 같이** 띄운다.
#   근거: docs/experiments/eth_dashboard_signal_cull_and_vol_expansion_rescore_20260921.md
VOL_EXPAND_K = 1.3                 # «확장» 정의: 앞 4시간 RV >= 1.3 x 직전 4시간 RV
VOL_EXPAND_LOGIT_A = -1.6560       # sigmoid(A + B*ln(pred/rv48))
VOL_EXPAND_LOGIT_B = 6.1834


def vol_level_item(state: dict[str, Any]) -> dict[str, Any]:
    """예측 변동성 등급(기준 = 최근 30일) + **수량 배수**(배포 공식 = 고정 ref_pred / pred)."""
    sm = (state or {}).get("sizing_model") or {}
    pred = sm.get("pred_vol")
    if not sm.get("used") or not isinstance(pred, (int, float)) or not (float(pred) > 0):
        return {"available": False, "grade": "데이터 없음", "tone": "neutral"}
    pred = float(pred)
    ref_fixed = float(sm.get("ref_pred") or VOL_LEVEL_REF_FALLBACK)
    recent = sm.get("ref_pred_recent")
    use_recent = isinstance(recent, (int, float)) and float(recent) > 0
    ref = float(recent) if use_recent else ref_fixed
    if not (ref > 0 and ref_fixed > 0):
        return {"available": False, "grade": "데이터 없음", "tone": "neutral"}
    ratio = pred / ref
    if ratio < VOL_LEVEL_RATIO_CALM:
        grade, tone = "안정", "good"
    elif ratio >= VOL_LEVEL_RATIO_HOT:
        grade, tone = "위험", "bad"
    else:
        grade, tone = "주의", "warn"
    # ⭐확장 축(«직전 4시간 대비»). rv48 을 워커가 안 주는 옛 상태파일이면 조용히 생략한다.
    rv48 = sm.get("rv48")
    expand: dict[str, Any] = {}
    if isinstance(rv48, (int, float)) and float(rv48) > 0:
        _m = pred / float(rv48)
        expand = {"mult": _m, "expand_k": VOL_EXPAND_K, "rv48": float(rv48),
                  "p_expand": 1.0 / (1.0 + math.exp(
                      -(VOL_EXPAND_LOGIT_A + VOL_EXPAND_LOGIT_B * math.log(_m))))}
    return {"available": True, "grade": grade, "tone": tone, "pred_vol": pred, "ref_pred": ref,
            **expand,
            "ratio": ratio, "qty_mult": ref_fixed / pred,
            # 어느 기준으로 잘랐는지 숨기지 않는다(고정으로 떨어졌으면 화면도 그렇게 말한다).
            "ref_source": "recent" if use_recent else "train",
            "ref_days": sm.get("ref_recent_days"), "ref_asof": sm.get("ref_recent_asof"),
            "ref_error": sm.get("ref_recent_error"),
            "cuts": {"calm": VOL_LEVEL_RATIO_CALM, "hot": VOL_LEVEL_RATIO_HOT}}



# 2026-09-19 Zeus 섀도우 페이로드·엔드포인트 제거(사용자 지시로 대시보드 카드 삭제).
# 러너 v3/v4 는 같은 날 다른 세션이 정지시켰고 감시표 줄도 함께 뺐다 -- 카드를 지우면 이
# API 는 소비자가 없다. 원장 파일(data/live/zeus_v*_shadow_*)과 연구 문서는 그대로 둔다.
def position_sizing_payload() -> dict[str, Any]:
    """크기 가늠자 상태. **계좌 포지션과의 결합은 프런트가 한다** -- 프런트는 이미
    `/api/binance-account` 를 들고 있어(app.js latestBinanceAccount) 서버에 비동기 의존을
    새로 만들 이유가 없다. 여기서는 파일 하나만 읽는다(요청 경로 계산 금지, 2026-09-10 실장애)."""
    out = worker_payload(POSITION_SIZING_STATE_PATH, POSITION_SIZING_MAX_AGE_MIN,
                         ts_field="generated_at", require_ok=True, stamp_available=True,
                         bare_missing=True)
    if not out.get("available"):
        return {**out, "vol_level": {"available": False, "grade": "웜업", "tone": "neutral"}}
    return {**out, "cap": sizing_cap(), "vol_level": vol_level_item(out)}


# 2026-09-12 단건 상한. ⚠️이 값은 처음 19왕복에서 골랐는데 **그 원장이 틀려 있었다** --
# 진입/청산가가 첫/마지막 체결가였고(다리별 VWAP 아님) 절단된 체결 스트림이 만든 중복 왕복이
# 섞여 있었다. 수리·재구성 후 68왕복으로 다시 쟀고(research_sizing_cap_mult_68trips_20260912.py)
# **2.0x 를 유지**한다. 근거가 바뀌었으므로 옛 숫자는 쓰지 말 것:
#   · 🔴크기 매칭 없이 비교하면 답이 뒤집힌다 -- 미매칭 최고는 4.0x(누적 483.80)인데 평균 노출을
#     맞추면 4.0x 는 531.55 로 하위권이고 0.5x 가 637.59 로 최고다(2026-09-06 짝비교 규율).
#   · 크기 매칭 후 곡선은 **단조**(조일수록 누적↑·t↑·MDD↓)라 내부 최적점이 없다. 다만 1.0~2.5x 는
#     565~577 로 사실상 동률 고원이고 2.0x 가 그 안에 있다.
#   · 2.0x 를 쓰는 이유는 최고점이어서가 아니라 **부트스트랩 2.5% 하한이 양수로 남는 가장 느슨한
#     값**이기 때문이다(2.0x +22.48 / 2.5x −62.00, B=2000 왕복 재표집).
#   · 시간 앞뒤 절반 **양방향 이월** 확인(앞→뒤 451.96 vs 그쪽 상한없음 146.53).
# 더 조이면 기대값은 낫지만 그건 상한이 아니라 사실상 균일 크기다(68건 중 34~42건을 자른다).
# 진짜 지렛대는 배수가 아니라 균일성이고, 재량을 얼마나 남길지는 사람이 정한다.
SIZING_CAP_MULT = 2.0
SIZING_CAP_MIN_TRIPS = 10   # 이보다 적으면 중앙값이 표본 하나에 휘둘린다 -- 상한을 만들지 않는다
SIZING_CAP_WINDOW = 30      # 최근 N 왕복만 본다(2026-09-13). 아래 sizing_cap 주석 참조

# 🔴2026-09-12 저녁 수정: 원장 기반 상한만 두면 세 가지가 깨진다(실제로 터졌다).
#   ① **기준이 움직인다** — 왕복이 19→68 건이 되자 중앙 명목이 6,790→4,346 으로 내려가
#      상한이 13,579→8,692 가 됐고, 낮에 정당하게 잡은 포지션이 **소급해서 초과**가 됐다.
#   ② **자기참조다** — 상한을 «사용자가 해온 크기의 중앙값»에 묶었는데 고치려는 대상이
#      바로 그 습관이다.
#   ③ **자산과 무관하다** — 계좌가 반토막 나도 상한은 그대로인데, 실제 위험은 명목이 아니라
#      **명목 ÷ 순자산**이다.
# ⇒ 교차 마진에서 `청산까지 거리 ≈ 순자산 / 총명목` 이므로(실측 1,065/7,642=13.9% vs
#   거래소 13.48%), 총명목을 순자산의 N 배로 캡하면 **청산 거리에 1/N 하한**이 생긴다.
# 🔴2026-09-13 하향 12.5 -> 6.7 (사용자 결정). 12.5 는 «역행폭 **95분위**»로 골랐는데
#   그게 틀린 잣대였다 — 청산은 꼬리 사건이라 95분위가 아니라 꼬리를 봐야 하고,
#   보유시간도 «24시간»으로 가정했지 실제 분포를 안 봤다(실제: 중앙 1.12시간이지만
#   **최대 9일**, 1일 초과가 7.4%).
#   실측 재계산 — **상한이 실제로 묶이는 봉**(배포 모델의 권고 > 상한)에서 5분봉 MAE 를
#   **원장 68왕복의 실제 보유시간에 대입**한 «68건 중 최소 1회 이상 청산»:
#        청산거리  레버리지   묶임비율   건당    최소1회
#           8%      12.5배     9.9%    1.70%    85.9%   <- 옛 값
#          10%      10.0배    21.7%    1.19%    66.9%
#        12.5%       8.0배    39.5%    0.76%    46.3%
#        14.9%       6.7배    55.0%    0.52%    33.0%
#          20%       5.0배    77.5%    0.29%    18.9%
#   ⭐2026-09-13 재결정(사용자): **수익 우선**으로 목적을 바꿔 8.0배 채택.
#   파산을 «계좌 0» 으로 두고 실제 복리를 시뮬레이션한 결과(research_sizing_growth_optimal_leverage):
#        L    68왕복 중앙배수 / 파산율      200왕복 중앙배수 / 파산율
#        3       1.44 /  2.4%                2.89 /  7.8%
#        5       1.77 / 11.2%                4.77 / 30.4%
#        8       2.12 / 30.2%  <- 채택       0.00 / 65.3%
#       12       0.00 / 57.1%                0.00 / 92.0%
#   🔴**8배는 «한 달 승부» 값이지 지속 가능한 값이 아니다.** 68왕복(≈한 달)에서 중앙배수가
#   최대지만 200왕복이면 파산 65%·중앙 0 이다. 거래를 계속하면 5배가 최적이다.
#   ⚠️엣지 민감도: μ 를 95% 하한(8.57bp)으로 낮추면 8배 1.30 vs 5배 1.27 로 **구분되지 않는다**.
#   8배의 우위는 μ=18.37bp 점추정이 맞을 때만 나온다. μ=0 이면 모든 L 에서 중앙 < 1.
#   🔴같은 날 **초판 계산이 틀렸다**(6.7배를 19.0% 로 봤다). 두 가지였다:
#     ① 저변동을 후행 atr_pct 로 정의 — 배포 경로는 `live_eth_sizing_vol_model_20260912`
#        의 **전방 변동성 예측**이다. 후행으로 재면 «저변동일수록 꼬리가 두껍다»가 나오는데
#        (1일 MAE 99분위 21.66%) 모델 기준으로는 **7.83% 로 오히려 얇다** — 모델이 고치려던 게 그거다.
#     ② 조건을 «하위 5.68%» 로 걸었다 — 맞는 조건은 «상한이 묶이는 상태»다(6.7배에서 55%).
#   ①은 부풀리고 ②는 줄이는데 ②가 커서 합치면 과소평가였다.
#   근거 스크립트: research_sizing_cap_liquidation_risk_20260913.py (두 함정을 assert 로 고정)
# 두 상한의 **작은 쪽**을 쓴다 — 순자산 연동이 주 방어선, 원장 기반은 보조.
# 🔴2026-09-13 8.0 -> 6.0 (사용자 결정). 파산 포함 1년 복리(735건)를 다시 재니 중앙값
# 최대가 6배였다(68건이면 12배 · 180건이면 10배 — **거래를 계속할수록 최적이 내려간다**).
# 엣지를 95% 하한으로 낮춰도 6배가 최적이라 결론이 엣지 추정에 민감하지 않다.
# ⭐지금 처방(1일 보유)에는 영향이 없다 — 생존 제약 4.30배가 이미 더 작게 묶는다.
#   줄어드는 건 짧은 보유(8->6, −25%)뿐이고 1년 중앙 계좌배수는 226 -> 230 으로 오히려 올랐다.
#   근거: scratchpad kelly_cap / tradeoff 계산, research_sizing_growth_optimal_leverage_20260913.
SIZING_CAP_EQUITY_X = 6.0
# 🔴상한 무시 스위치 (2026-09-20 사용자 지시 «우리 사이징 코드는 잠깐 꺼줘»).
# 세 상한(원장·순자산·위험모델)을 **전부 건너뛰고** 「순자산 × 이 배수」 하나만 기준으로 쓴다.
# 🔴상한을 «없애는» 게 아니다. 진입 비율 슬라이더의 100% 가 **상한 명목에 대한 비율**이라
#   기준이 없으면 수량 자체를 만들 수 없다 -- 끄기는 실제로 «기준을 하나로 바꾸기»다.
# 기본값 0 = 꺼짐. 커밋된 코드의 동작은 그대로이고, 서버 환경변수로만 열리며 재기동으로
# 되돌아온다(사용자 선택: 환경변수 스위치).
#   DASHBOARD_SIZING_CAP_EQUITY_X=20 python dashboard/server.py
# ⚠️이 배수를 거래소 레버리지와 **같게** 두면(둘 다 20) 슬라이더 100% 에서 필요 증거금이
#   순자산의 100% 가 되어 수수료 몫이 없다 -- 거래소가 거부한다(leverage_setting 의
#   ENFORCE_SLACK 주석이 같은 실패를 기록한다). 끝까지 쓸 생각이면 배수를 조금 낮춰라.
# ⚠️이 값이 0 보다 크면 크기를 막는 것이 **아무것도 남지 않는다** -- 이 화면이 존재하는
#   이유(재량 사이징이 실계좌 손실의 출처, 크기-수익 상관 -0.494)가 잠시 꺼진 상태다.
try:
    SIZING_CAP_OVERRIDE_X = max(0.0, float(os.getenv("DASHBOARD_SIZING_CAP_EQUITY_X", "0") or 0))
except ValueError:                 # 오타를 «무제한»으로 읽지 않는다 -- 꺼진 쪽으로 떨어뜨린다
    SIZING_CAP_OVERRIDE_X = 0.0


def notional_sum(positions: list[dict[str, Any]] | None, symbol: str | None = None) -> float:
    """|명목| 합. `symbol` 을 주면 그 심볼만, 없으면 **계좌 전체**.

    헤지 모드라 롱·숏이 동시에 열리므로 상쇄를 가정하지 않고 절대값을 더한다 -- 두 다리 다
    증거금을 먹고 둘 다 청산될 수 있다.
    🔴범위를 틀리면 조용히 위험해진다: 상한(교차 마진의 청산거리 = 순자산/총명목)은 계좌
    전체여야 하고, 청산거리 투영은 심볼 안에서만 성립한다. 2026-09-19 에 상한이 심볼 하나만
    세고 있었다(봇 ETHUSDT + 수동 ETHUSDC). 자체점검 test/test_notional_sum_scope_20260919.py.
    """
    return sum(abs(float(p.get("notional") or 0.0)) for p in (positions or [])
               if symbol is None or p.get("symbol") == symbol)


def entry_projection(plan: dict, account: dict, positions: list, existing: float,
                     equity: float) -> dict[str, Any]:
    """«이 주문을 넣으면 계좌 카드가 어떻게 바뀌나» -- 사용자 요청(2026-09-12).

    계좌 카드가 실제로 띄우는 **세 타일과 같은 정의**를 쓴다(app.js 의 tiles 참조):
      · 증거금 사용 = 사용증거금 ÷ 순자산     · 계좌 노출 = 명목 ÷ 순자산
      · 청산까지    = |마크 − 청산가| ÷ 마크
    정의가 어긋나면 «미리보기에서 본 숫자»와 «진입 후 카드»가 달라져 신뢰를 잃는다.

    청산 후 거리는 거래소가 준 **현재 청산가에서 비례 축소**한다 -- 교차 마진에서 거리는
    대략 순자산÷총명목이므로 명목이 커진 비율만큼 좁아진다. 순수 근사식(순자산÷총명목)보다
    실측값에 앵커된 이 쪽이 낫다(실측 13.48% vs 근사 13.9%). 포지션이 없으면 근사로 떨어진다.
    """
    balance = account.get("balance") or {}
    used = balance.get("initial_margin")
    used = float(used) if used is not None else max(
        0.0, float(balance.get("margin") or 0.0) - float(balance.get("available") or 0.0))
    available = float(balance.get("available") or 0.0)
    add_margin = float(plan.get("margin_usdt") or 0.0)
    total = float(plan.get("total_notional_usdt") or 0.0)

    liq_before = None
    for position in positions:
        mark, liq = float(position.get("mark_price") or 0.0), float(position.get("liquidation_price") or 0.0)
        if mark > 0 and liq > 0:
            distance = abs(mark - liq) / mark * 100.0
            liq_before = distance if liq_before is None else min(liq_before, distance)
    if liq_before is not None and total > 0 and existing > 0:
        liq_after = liq_before * existing / total
    elif equity > 0 and total > 0:
        liq_after = 100.0 * equity / total
    else:
        liq_after = None

    def side(margin_used: float, notional: float, liq: float | None) -> dict[str, Any]:
        return {
            "margin_usdt": round(margin_used, 2),
            "margin_used_pct": round(100 * margin_used / equity, 1) if equity else None,
            "available_usdt": round(equity - margin_used, 2) if equity else None,
            "notional_usdt": round(notional, 2),
            "exposure_x": round(notional / equity, 2) if equity else None,
            "liq_pct": round(liq, 2) if liq is not None else None,
        }

    return {
        "equity_usdt": round(equity, 2),
        "available_before_usdt": round(available, 2),
        "before": side(used, existing, liq_before),
        "after": side(used + add_margin, total, liq_after),
    }


def sizing_cap() -> dict[str, Any]:
    """왕복 원장에서 중앙 명목을 읽어 상한을 낸다. **현재가는 곱하지 않는다** --
    ETH 수량 환산은 가격을 이미 들고 있는 프런트가 한다(요청 경로 계산 금지 원칙)."""
    try:
        lines = ACCOUNT_TRIP_LEDGER_PATH.read_text().splitlines()
    except Exception:
        return {"available": False, "reason": "ledger_missing"}
    trips = []
    for line in lines:
        try:
            trip = json.loads(line)
            trips.append((int(trip.get("entry_time") or 0),
                          abs(float(trip["max_qty"]) * float(trip["entry_price"]))))
        except Exception:
            continue          # 깨진 줄 하나가 상한을 통째로 못 내게 하지 않는다
    if len(trips) < SIZING_CAP_MIN_TRIPS:
        return {"available": False, "reason": "not_enough_trips",
                "trips": len(trips), "need": SIZING_CAP_MIN_TRIPS}
    # 🔴2026-09-13: **최근 창**만 본다. 이력 전체에 묶으면 상한이 톱니처럼 계속 조여진다 --
    #   원장은 자라기만 하고, 과거 거래가 지금보다 작으면 중앙값이 계속 내려가 결국 정상
    #   매매까지 막는다. 실제로 왕복이 19→68건이 되자(백필) 중앙 명목 6,790→4,346,
    #   상한 13,579→8,692 로 떨어져 **낮에 정당하게 잡은 포지션이 소급 초과**가 됐다.
    #   실측(68왕복): 전체 중앙 4,346 vs 최근 30건 6,774 -- 옛 거래가 지금 규모와 무관하다
    #   (하위 12건이 646~1,194 USDT). 최근 30건 ×2 = 13,549 는 순자산 ×12.5 = 13,083 과
    #   2% 안에서 만난다 -- 독립인 두 기준이 같은 값을 가리킨다.
    trips.sort()
    notionals = [v for _, v in trips[-SIZING_CAP_WINDOW:]]
    median = statistics.median(notionals)
    return {"available": True, "trips": len(notionals), "trips_total": len(trips),
            "window": SIZING_CAP_WINDOW, "mult": SIZING_CAP_MULT,
            "median_notional_usdt": round(median, 2),
            "cap_notional_usdt": round(SIZING_CAP_MULT * median, 2)}


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


# 🔴SSE 로 나가는 상태는 **화면이 실제로 읽는 세 블록**뿐이다.
# app.js 의 render() 가 보는 것은 state.session / state.microstructure / state.tail_risk 이고
# compactState 는 인자로 넘어가기만 하고 **본문에서 한 번도 안 읽힌다**(2026-09-20 확인).
# 그런데 이 루프는 dashboard_state.json 전체(실측 30,109B)를 상태가 바뀔 때마다 그대로 밀고
# 있었다 -- 세 블록만 추리면 3,498B 로 **8.6배** 줄어든다. SSE 는 gzip 도 안 걸린다
# (json_compress_etag 미들웨어는 StreamResponse 를 건드리지 않는다).
# ⭐옛 app.js 와도 호환된다: 지우는 키는 클라가 애초에 안 읽던 것들이다.
# ⚠️`/api/state` 는 **그대로 전체**를 준다 -- 계약이 다르고 테스트가 그걸 검사한다.
SSE_STATE_KEYS = ("session", "microstructure", "tail_risk")


def sse_state_view(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    state = (payload or {}).get("state")
    if not state:
        return payload      # 없음/빈 상태는 원래 모양 그대로 -- 클라 가드가 그걸 본다
    return {"state": {k: state.get(k) for k in SSE_STATE_KEYS}}


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
            # 🔴immutable 금지. 이 둘은 배포마다 바뀌는데 캐시버스터가 날짜라 같은 날
            # 두 번째 배포는 브라우저/Cloudflare 캐시를 못 뚫는다(2026-09-13 실제 사고).
            # no-cache 는 «캐시하지 마라»가 아니라 «쓰기 전에 물어봐라»다 -- 안 바뀌었으면
            # ETag 로 304(본문 0바이트)라 아끼는 건 조건부 요청 한 번뿐이고, 대가가 사고였다.
            response.headers["Cache-Control"] = "no-cache"
            response.enable_compression()
        elif asset_name.endswith(".ttf"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif asset_name.endswith(".html"):
            # 🔴index.html 은 **절대** 캐시하면 안 된다. 그 안에 캐시버스터가 들어 있어서,
            # 이 파일이 묵으면 옛 `app.js?v=...` 를 가리키고 그 app.js 는 immutable(1년) 이라
            # 브라우저도 Cloudflare 도 새 코드를 영영 안 가져온다.
            # 2026-09-13 실제 증상: 사용자 화면의 «특화 감지기»가 이틀 전 판본(2행)이었고
            # Ctrl+Shift+R 로도 안 바뀌었다. 디렉터리 경로(/dashboard/live/)로 열 때만
            # no-cache 가 붙고, index.html 을 직접 열면 헤더가 아예 없었다.
            response.headers["Cache-Control"] = "no-cache"
        return response

    # 2026-09-15: JSON 응답 압축·ETag 를 **한 곳**에서 건다.
    # 왜: 응답을 만드는 자리가 둘이었다 -- json_response() 헬퍼(11곳)는 압축+ETag 를 붙이는데
    # 날 web.json_response()(61곳)는 둘 다 없었다. 실측: trades 302,580B -> 23,881B(12.7배),
    # state 43,076 -> 6,156 은 압축됐고, model-indicator-history 194,437 · liquidation-map
    # 46,898 · regime-wide24 19,960 은 gzip 을 요청해도 그대로 나갔다.
    # 61곳을 고치는 대신 미들웨어로 잡는다 -- 앞으로 추가되는 엔드포인트도 자동으로 걸린다.
    # ⚠️ETag 는 클라가 조건부 요청을 보내야 뜻이 있다. app.js 의 fetch 가 `cache: "no-store"`
    #   면 브라우저가 HTTP 캐시를 통째로 건너뛰어 If-None-Match 를 **안 보낸다**. 같은 커밋에서
    #   28곳을 "no-cache"(=캐시하되 쓰기 전에 물어봐라)로 바꿨다. 서버가 Cache-Control:
    #   no-cache 를 주므로 재검증 없이 옛 본문이 나갈 일은 없다.
    @web.middleware
    async def json_compress_etag(request: web.Request, handler: Any) -> web.StreamResponse:
        response = await handler(request)
        # SSE(StreamResponse)·POST·비JSON·비200 은 건드리지 않는다. 304 는 본문이 없어야 한다.
        if (request.method != "GET" or not isinstance(response, web.Response)
                or response.status != 200 or response.body is None
                or (response.content_type or "") != "application/json"):
            return response
        if "ETag" not in response.headers:          # json_response() 가 이미 붙인 건 그대로 둔다
            etag = f'W/"body-{hashlib.sha256(response.body).hexdigest()[:16]}"'
            response.headers["ETag"] = etag
            response.headers.setdefault("Cache-Control", "no-cache")
            if etag_matches(request, etag):
                return web.Response(status=web.HTTPNotModified.status_code,
                                    headers={"ETag": etag,
                                             "Cache-Control": response.headers["Cache-Control"]})
        response.enable_compression()               # Accept-Encoding 에 맞춰 협상된다
        return response

    app = web.Application(middlewares=[static_asset_headers, json_compress_etag])
    json_cache: dict[Path, tuple[tuple[int, int] | None, Any]] = {}
    trade_cache: dict[str, Any] = {
        "signature": None,
        "rows": [],
        "payloads": {},
    }
    event_clients: set[asyncio.Queue[str]] = set()
    latest_event_state: dict[str, Any] | None = None
    latest_event_tickers: dict[str, dict[str, Any]] = {}
    # swr_cached 가 캐시를 키로 소유하지만, 이 셋은 swr 밖에서도 읽으므로 남긴다
    # (market_history: 부분 갱신 / evidence_signal: frames / binance_account: 원장 기록).
    market_history_cache: dict[str, dict[str, Any]] = {}
    binance_account_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    # last_at 0.0 + time.monotonic() ⇒ 기동 직후 첫 주기에 바로 한 번 적는다.
    account_trip_state: dict[str, Any] = {"last_at": 0.0, "seen": load_account_trip_keys()}
    # 진행 중인 수동 주문 하나. 동시에 둘을 두지 않는다 -- 겹치면 단건 상한의 뜻이 흐려진다.
    manual_entry_state: dict[str, Any] = {"phase": "idle"}
    evidence_signal_cache: dict[str, Any] = {"ts": 0.0, "payload": None, "frames": None}
    # 2026-08-31: 자산별로 키를 나눈다(원래는 공유 슬롯 하나였다) -- ETH 요청과 BTC 요청이
    # 서로의 캐시를 밀어내지 않게. 2026-09-12 부터 그 dict/Lock 은 swr_cached 가 f"...:{asset}"
    # 키로 직접 소유하므로 여기서 선언하지 않는다.

    # ---------------------------------------------------------------------------------
    # 2026-09-03 perf pass -- stale-while-revalidate.
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
    # again. swr_cached() below keeps the REQUEST path off the COMPUTE path: once a payload
    # exists, a request never blocks on a recompute -- it returns the stale one immediately
    # and refreshes behind it. Only a genuinely cold cache (this process has never computed
    # that asset) still waits.
    #
    # ⚠️A companion background prewarm loop was written and deployed alongside this on
    # 2026-09-03, then REMOVED (user decision) after the server locked up that evening. The
    # hang was traced to the WSL2 host pausing (kernel journal shows "Clock change detected"
    # twice, no OOM, and trading_bot.py still logging normally throughout), NOT to this code
    # -- but the loop refreshed 4 TabPFN endpoints every ~45-90s whenever a browser was
    # connected, which was new background GPU/CPU load on a box that already has documented
    # GPU contention, and it was not worth carrying an unproven suspect. Do not reintroduce
    # a background warm loop here without a dedicated GPU budget for the dashboard.
    #
    # The per-endpoint fetch/compute bodies themselves are UNCHANGED -- they were only moved
    # into `produce()` closures so swr_cached() can drive them.
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
                    payload = await _timed_produce(key, produce)
                    cache["payload"] = payload
                    cache["ts"] = time.monotonic()
            except Exception as exc:  # noqa: BLE001 -- a failed BACKGROUND refresh must never
                # reach the client: the stale payload it was refreshing is still being served,
                # so the correct behaviour is to log it and let the next cycle retry. Letting
                # this raise would only kill an orphan task and lose the reason.
                print(f"cache refresh failed for {key} (still serving stale): {exc}", flush=True)
        refresh_tasks[key] = asyncio.create_task(_run())

    async def _timed_produce(key: str, produce):
        """produce() 한 번의 실제 소요시간을 남긴다 -- 어떤 엔드포인트를 워커로 빼야 하는지
        추측이 아니라 실측으로 정하기 위해서다(2026-09-13). 이미 도는 갱신 경로에만 걸리므로
        계산 횟수가 늘지 않는다. 임계 미만은 안 찍는다 -- 파일 한 줄 읽는 것들이 로그를 덮는다."""
        t0 = time.monotonic()
        try:
            return await produce()
        finally:
            took = time.monotonic() - t0
            if took >= SWR_SLOW_LOG_SECONDS:
                print(f"swr produce {key} {took:.2f}s", flush=True)

    swr_store: dict[str, dict] = {}
    swr_locks: dict[str, asyncio.Lock] = {}

    async def swr_cached(key: str, ttl: float, produce, *, max_stale: float = 0.0,
                         cache: dict | None = None) -> Any:
        """Fresh -> serve it. Stale -> serve stale NOW and refresh behind it. Cold -> await.

        `max_stale` is how far PAST `ttl` a payload may be served while its replacement is still
        being computed, and it defaults to 0 -- i.e. this behaves exactly like the double-checked
        lock it replaced unless a call site opts in. Only endpoints whose recompute is genuinely
        expensive (a Binance round trip, or a TabPFN refit that has been measured at 43s under GPU
        contention) opt in; for the endpoints that just read a local duckdb, blocking was already
        fast and returns FRESHER data, so they keep doing that.

        Past ttl + max_stale the payload is treated as cold again: at that age, blocking for a
        current reading beats silently serving something long out of date.

        캐시와 락은 `key` 로 여기서 소유한다 -- 호출부마다 dict 와 Lock 을 손으로 선언해
        넘길 이유가 없다. `cache` 를 넘기는 건 그 dict 를 swr 밖에서도 읽는 세 곳뿐이다
        (market_history / evidence_signal 의 frames / binance_account)."""
        if cache is None:
            cache = swr_store.setdefault(key, {"ts": 0.0, "payload": None})
        lock = swr_locks.setdefault(key, asyncio.Lock())
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
            payload = await _timed_produce(key, produce)
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

    # 추세 veto (2026-09-22) -- `close > SMA144` 를 ±1.0xATR144 히스테리시스로 디바운스한 상태.
    # 이 캐시가 이미 1500 봉을 들고 있어 별도 fetch/워커가 필요 없다(차트는 tail(100) 만 쓴다).
    # 🔴근거는 **ETH 5m 전용**이다: 겹치지 않는 24h 창 1,719 개에서 순추세−역추세가 TP1.5/SL0.7
    #   기준 롱 +16.2bp [+6.6,+25.3] · 숏 +18.3 [+8.0,+28.3] · 5 연도 x 양측 전부 양수.
    #   같은 산식을 BTC/SOL 프레임에도 그대로 계산하지만 **그 자산에서는 검정한 적이 없다**.
    # 손실 차단기이지 수익 생성기가 아니다. 순추세 비중이 ~50% 라 기회가 절반으로 준다.
    TREND_VETO_N = 144          # 5분봉 144 개 = 12 시간. SMA 와 ATR 이 같은 창이다.
    TREND_VETO_K = 1.0          # 밴드 = K x ATR. 실측 폭 중앙 +-0.238%(p10 0.132 / p90 0.416)

    def trend_veto_rows(df: pd.DataFrame) -> dict[int, dict[str, float | int]]:
        """{봉시각(초): {sma, atr, veto}}. veto = +1 롱만 / -1 숏만 / 0 워밍업.
        인과적: 봉 t 의 값은 봉 t 종가까지만 본다."""
        w = df.tail(TREND_VETO_N * 5)           # ewm 워밍업 여유(720 봉). 1500 봉 전체는 낭비다
        if len(w) < TREND_VETO_N + 1:
            return {}
        c, h, l = w["close"], w["high"], w["low"]
        sma = c.rolling(TREND_VETO_N).mean()
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / TREND_VETO_N, adjust=False).mean()
        dev = ((c - sma) / sma).to_numpy()
        eps = (TREND_VETO_K * atr / c).to_numpy()
        # 형성 중 봉으로 한 칸 전진하려면 창에서 빠질 종가 하나가 필요하다(클라는 100봉만 갖고
        # 있어 SMA144 를 스스로 못 만든다). 봉 i 에서 봉 i+1 로 갈 때 빠지는 건 close[i-143].
        drop = c.shift(TREND_VETO_N - 1).to_numpy()
        out: dict[int, dict[str, float | int]] = {}
        cur = 0
        for i, ts in enumerate(w["timestamp"]):
            if not (np.isfinite(dev[i]) and np.isfinite(eps[i])):
                continue
            if dev[i] > eps[i]:
                cur = 1
            elif dev[i] < -eps[i]:
                cur = -1                        # 밴드 안이면 직전 유지(히스테리시스)
            row: dict[str, float | int] = {"sma": round(float(sma.iloc[i]), 4),
                                           "atr": round(float(atr.iloc[i]), 4), "veto": cur}
            if np.isfinite(drop[i]):
                # n 도 같이 보낸다 -- 클라에 144 를 또 적으면 두 곳을 따로 고쳐야 한다.
                row["drop"], row["vn"] = round(float(drop[i]), 4), TREND_VETO_N
            out[int(ts.timestamp())] = row
        return out

    async def load_market_history_from_evidence_cache(asset: str) -> list[dict[str, float | int]]:
        """ETH/BTC candle-chart data (2026-08-26, user request to de-duplicate) sliced straight out
        of evidence_signal_cache["frames"] instead of a separate Binance klines fetch -- that cache
        already holds EVIDENCE_FETCH_LIMIT (1500) 5m closed bars for both symbols, a strict superset
        of the 100 bars this used to fetch on its own, refreshed every EVIDENCE_SIGNAL_CACHE_SECONDS
        (60s, 5x more often than the old MARKET_HISTORY_CACHE_SECONDS=300s cache as a side effect).
        No separate cache of its own needed -- slicing 100 rows out of an already-in-memory frame is
        cheap enough to just redo on every call."""
        # 🔴2026-09-16: 예전엔 «frames 가 None 일 때만» 데웠다. 그때는 증거신호 엔드포인트가
        # 프런트 폴링에 맞춰 60초마다 load_evidence_signals() 를 불러 캐시를 갱신했기 때문이다.
        # 그 엔드포인트를 내리면서 **갱신 주체가 사라져** 차트가 첫 캔들에서 멈추게 됐다.
        # 이제 매번 부른다 -- swr_cached 가 60초 TTL 로 스스로 throttle 하므로 비용은 같다.
        await load_chart_klines_frames()
        frames = evidence_signal_cache["frames"]
        if frames is None:
            raise web.HTTPBadGateway(reason="market_history_upstream_error")
        closed_df, btc_df, _funding_df = frames
        src = closed_df if asset == "eth" else btc_df
        if src is None or src.empty:
            raise web.HTTPBadGateway(reason="market_history_upstream_error")
        veto = trend_veto_rows(src)
        return [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                **veto.get(int(row["timestamp"].timestamp()), {}),
            }
            # 🔴2026-09-23 100 -> 200. 100 봉이면 8.3시간이라 12h 창을 골라도 8.3시간에서
            #   끊겼다(사용자 «12시간 데이터 추세를 보고 싶다»). closed_df 는 SMA144 를
            #   계산하느라 이미 그보다 길다 -- 자르는 폭만 넓히면 된다. 200 = 16.6시간.
            for _, row in src.tail(200).iterrows()
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
            f"market_history:{asset}", MARKET_HISTORY_CACHE_SECONDS, produce,
            cache=market_history_cache.setdefault(asset, {"ts": 0.0, "payload": None}),
            max_stale=STALE_GRACE_SECONDS,
        )

    # ── 볼륨 풋프린트 체결 테이프 (2026-09-15) ────────────────────────────────────
    # 봉 하나를 가격레벨로 쪼개 **공격적 매수/매도** 체결량을 따로 센다. klines 에는 이 정보가
    # 없다(봉당 taker_buy 합계 하나뿐) -- 그래서 체결 테이프를 직접 누적한다.
    # `m` = "매수자가 메이커" 이므로 m=True 면 **공격자는 매도자**다(check_footprint_tape 로 검증).
    #
    # 실시간은 WS, 과거는 REST 다. 처음엔 REST 폴링만으로 만들었다가 갈아엎었다 -- 2026-09-15
    # 22:35 봉이 275,040 ETH(평소의 8배)로 터지자 1000건/요청 페이스가 **25분** 뒤처졌다.
    # 풋프린트가 가장 필요한 순간이 바로 그때인데 그때 밀린다. WS 는 가중치가 0이라 급증에도
    # 안 밀리고, REST 는 «연결 이전 구간» 백필에만 쓴다(이건 늦어도 라벨로 알리면 그만이다).
    # ⚠️@aggTrade 는 2026-09-02 부터 바이낸스가 배달을 멈췄다(구독은 에러 없이 되고 메시지만
    #   안 온다 -- 2026-09-15 재확인: 10초에 0건). @trade 를 쓴다(개별 체결, 같은 p/q/T/m).
    FOOTPRINT_AGG_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
    FOOTPRINT_WS_URL = f"wss://fstream.binance.com/ws/{FOOTPRINT_SYMBOL.lower()}@trade"
    footprint_state: dict[str, Any] = {"bars": {}, "ready": False, "updated": 0.0,
                                   "last_ms": 0, "saved_at": 0.0,
                                   # aggTrades 로 메운 봉. 출처 표시일 뿐 «덜 정확하다»는 뜻이
                                   # 아니다 -- 크기 구간은 주문 단위라 aggTrades 쪽이 오히려
                                   # 정확하다(footprint_add 도크스트링의 2026-09-19 정정).
                                   "agg_bars": set(),
                                   # 초 -> [리테일매수, 리테일매도, 고래매수, 고래매도,
                                   #        총매수, 총매도, 마지막가격]. 아래 supply_1s_cell 참고.
                                   "sec": {}}

    def supply_1s_cell(ts_ms: int) -> list[float]:
        """그 초의 칸. 오래된 초는 새 초가 생길 때만 버린다(체결마다 돌 일이 아니다)."""
        sec = int(ts_ms) // 1000
        by_sec = footprint_state["sec"]
        cell = by_sec.get(sec)
        if cell is None:
            cell = by_sec[sec] = [0.0] * 7
            cutoff = sec - SUPPLY_1S_SECONDS
            for old in [s for s in by_sec if s < cutoff]:
                del by_sec[old]
        return cell

    def footprint_bar_start(ts_ms: float) -> int:
        return (int(ts_ms) // 1000) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS

    def footprint_add(price: float, qty: float, ts_ms: int, sell: bool,
                      agg: bool = False, order: bool = False) -> None:
        """order=False 는 **개별 체결**(셀 총량만), True 는 **테이커 주문**(크기 구간만).

        백필의 aggTrades 한 줄은 그 자체가 주문이라 둘 다다. 실시간 `@trade` 는 총량을 바로
        넣고, 주문이 닫힐 때 구간을 따로 넣는다 -- 한 번에 못 하는 이유는 주문이 닫히기
        전에는 그 주문이 어느 통에 속하는지 알 수 없기 때문이다(쓸어담는 중에는 계속 자란다).

        🔴여기서 한 번 틀렸다(2026-09-19). 처음엔 개별 체결(`@trade`)로 갈랐는데, 큰 주문이
          호가를 쓸면 그게 작은 체결 수십 건으로 쪼개져 **고래가 사라진다**(같은 11.3초 구간
          실측: aggTrade 기준 37.4% vs @trade 기준 9.6%, 총 명목은 동일). 그다음엔 반대로
          백필 덩어리를 개별 수로 나눠봤는데 그건 **과교정**이라 고래가 0.0% 가 됐다 --
          덩어리는 이미 «주문 하나»라 나눌 것이 아니었다.
          답은 둘 다 **주문 단위로 맞추는 것**이다: 백필은 덩어리를 그대로 쓰고, 실시간은
          같은 규칙(가격·방향·ms)으로 되묶는다. 기존 `nif_whale` 도 @aggTrade 기준이라
          이래야 이 저장소에서 「고래」가 한 뜻이 된다."""
        bars = footprint_state["bars"]
        bar = footprint_bar_start(ts_ms)
        cells = bars.get(bar)
        if cells is None:
            cutoff = (footprint_bar_start(time.time() * 1000)
                      - (FOOTPRINT_KEEP_BARS - 1) * FOOTPRINT_BAR_SECONDS)
            if bar < cutoff:
                return              # 창을 벗어난 봉 -- 넣어봐야 바로 아래에서 지워진다
            cells = bars[bar] = {}
            for old_bar in [b for b in bars if b < cutoff]:
                del bars[old_bar]   # 새 봉이 생길 때만 정리한다 -- 체결마다 돌 일이 아니다(266/s)
                footprint_state["agg_bars"].discard(old_bar)
        cell = cells.setdefault(int(round(price / FOOTPRINT_BUCKET)), [0.0] * 6)
        sec_cell = supply_1s_cell(ts_ms)
        side = 1 if sell else 0
        if not order:
            cell[side] += qty
            sec_cell[4 + side] += qty
            sec_cell[6] = price          # 그 초의 마지막 체결가 = 1초 가격선
        if order or agg:
            notional = price * qty
            if notional >= WHALE_MIN_USD:
                cell[2 + side] += qty
                sec_cell[2 + side] += qty
            elif notional < RETAIL_MAX_USD:
                cell[4 + side] += qty
                sec_cell[side] += qty
        if agg:
            footprint_state["agg_bars"].add(bar)
        # 🔴`now` 는 아래 스냅샷 저장 조건이 쓴다. 2026-09-19 리팩터에서 이 줄을 지웠다가
        #   **ready 가 되는 순간 모든 체결이 NameError** 로 터졌다(단락평가 때문에 ready
        #   이전에는 조용했다). 결과: WS 루프가 크래시->재연결->백필을 무한 반복해 실시간
        #   누적이 아예 안 됐고 REST 를 계속 때렸다. 한 줄이 지워진 걸 테스트가 못 잡은 건
        #   이 경로에 «ready 이후 체결» 을 태우는 시험이 없었기 때문이다.
        now = time.time()
        footprint_state["updated"] = now
        # ⚠️ready 일 때만 저장한다. 백필이 **진행 중인 봉**을 저장하면, 다음 판이 그걸 «이미 있는
        # 봉»으로 보고 건너뛰어 반쪽짜리로 굳는다(2026-09-15 시험에서 한 봉이 -83.7% 로 남았다).
        # 저장된 스냅샷의 계약은 «last_ms 까지 공백이 없다» 이고, 그 보증이 곧 ready 다.
        if footprint_state["ready"] and now - footprint_state["saved_at"] >= FOOTPRINT_SNAPSHOT_SECONDS:
            footprint_save()   # 30초마다 -- 죽어도 잃는 건 30초어치이고 그건 REST 로 메운다

    def footprint_save() -> None:
        """봉 상태를 통째로 덮어쓴다(수십 KB). 증분 append 를 안 쓰는 이유는 진행 중인 봉이
        계속 자라기 때문 -- 어차피 마지막 상태만 쓸모 있다. tmp -> replace 로 원자적으로."""
        try:
            tmp = FOOTPRINT_SNAPSHOT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "bar_seconds": FOOTPRINT_BAR_SECONDS, "bucket": FOOTPRINT_BUCKET,
                "symbol": FOOTPRINT_SYMBOL, "last_ms": footprint_state["last_ms"],
                # 셀 칸수 표식. 2칸 시절 스냅샷을 4칸 코드가 읽으면 IndexError 가 아니라
                # **조용히 고래 0** 이 된다 -- 그래서 버전을 적고 다르면 통째로 버린다.
                "cells_v": 3,
                # 봉 링은 24시간이지만 저장은 **화면이 고를 수 있는 가장 긴 창**만 한다
                # (위 FOOTPRINT_MAX_WINDOW_BARS). 그 밖은 재시작 뒤 다시 찬다 -- 화면이
                # 실제 봉 수를 적으므로 짧아진 걸 숨기지 않는다.
                "bars": {str(bar): {str(k): v for k, v in cells.items()}
                         for bar, cells in sorted(footprint_state["bars"].items())[-FOOTPRINT_MAX_WINDOW_BARS:]},
            }))
            tmp.replace(FOOTPRINT_SNAPSHOT_PATH)
            footprint_state["saved_at"] = time.time()
        except OSError as exc:
            footprint_state["saved_at"] = time.time()   # 매 체결마다 재시도하지 않게
            print(f"footprint snapshot save failed: {exc}", flush=True)

    def footprint_load() -> None:
        """재시작 직후 1회. 창 밖 봉은 버리고, last_ms 를 복원해 백필이 «꺼져 있던 구간»만
        메우게 한다. 봉 길이나 버킷이 바뀌었으면 통째로 무시한다 -- 섞이면 조용히 틀린다."""
        try:
            saved = json.loads(FOOTPRINT_SNAPSHOT_PATH.read_text())
        except (OSError, ValueError):
            return
        if (saved.get("bar_seconds") != FOOTPRINT_BAR_SECONDS
                or saved.get("bucket") != FOOTPRINT_BUCKET
                or saved.get("symbol") != FOOTPRINT_SYMBOL
                or saved.get("cells_v") != 3):
            print("footprint snapshot: 설정이 달라 무시한다", flush=True)
            return
        cutoff = (footprint_bar_start(time.time() * 1000)
                  - (FOOTPRINT_MAX_WINDOW_BARS - 1) * FOOTPRINT_BAR_SECONDS)
        bars = {int(bar): {int(k): [float(x) for x in v] for k, v in cells.items()}
                for bar, cells in (saved.get("bars") or {}).items() if int(bar) >= cutoff}
        if not bars:
            return
        footprint_state["bars"] = bars
        footprint_state["last_ms"] = int(saved.get("last_ms") or 0)
        print(f"footprint snapshot: {len(bars)}봉 복원", flush=True)

    async def footprint_backfill(gap_from_ms: int, until_ms: int) -> None:
        """WS 가 못 준 구간을 aggTrades 로 메운다. 최신 봉부터, 봉마다 «받을 창»을 따로 잡는다.

        창 계산이 이 함수의 전부다:
          - state 에 **없는** 봉  -> 봉 전체(단 until_ms 까지). 처음 뜰 때·WS 가 오래 끊겼을 때.
          - state에 **있는** 봉   -> [gap_from_ms, until_ms] 와 겹치는 부분만. gap_from_ms 는
            끊기기 직전 마지막 WS 체결 시각이라, 이미 센 체결을 다시 세지 않는다.
        이 구분이 없으면 둘 중 하나가 깨진다 -- 「있는 봉은 건너뛴다」로 두면 **서버가 뜬 봉의
        앞부분**이 통째로 빈다(2026-09-15 실측 43% 누락), 무조건 다시 받으면 이중계상이다.

        왜 최신 봉부터인가: 오래된 쪽부터 한 줄로 걸었더니 거래량이 8배로 터진 봉(275,040 ETH
        ≈ 요청 130회)에 막혀 **정작 사용자가 보는 최근 봉이 계속 비어 있었다**.

        요청당 1000건·가중치 20 이라 FOOTPRINT_CATCHUP_SECONDS 로 페이스를 걸고, 총 요청 수에
        상한을 둔다(폭주 구간에서 무한정 긁지 않도록)."""
        budget = 400          # 총 요청 상한 ≈ 17분·가중치 8000. 평소 1시간 백필은 ~200회면 끝난다
        MISS_LIMIT = 10       # 빈 응답(429·5xx) 연속 허용치 -- 한 번에 포기하면 조용히 죽는다
        now_bar = footprint_bar_start(until_ms)
        window_floor = (now_bar - (FOOTPRINT_BARS - 1) * FOOTPRINT_BAR_SECONDS)
        try:
            for bar in [now_bar - i * FOOTPRINT_BAR_SECONDS for i in range(FOOTPRINT_BARS)]:
                if bar < window_floor:
                    continue
                lo = (bar * 1000 if bar not in footprint_state["bars"]
                      else max(bar * 1000, gap_from_ms))
                hi = min((bar + FOOTPRINT_BAR_SECONDS) * 1000, until_ms)
                if lo >= hi:
                    continue                  # 이 봉은 이미 채워져 있다(겹치는 공백이 없다)
                next_id, misses = None, 0
                while budget > 0:
                    budget -= 1
                    rows = None
                    if next_id is None:
                        seed = await fetch_binance_json(FOOTPRINT_AGG_URL, {
                            "symbol": FOOTPRINT_SYMBOL, "startTime": lo,
                            "endTime": lo + 2000, "limit": 1})
                        if seed:
                            next_id, misses = int(seed[0]["a"]), 0
                            continue
                    else:
                        rows = await fetch_binance_json(FOOTPRINT_AGG_URL, {
                            "symbol": FOOTPRINT_SYMBOL, "fromId": next_id, "limit": 1000})
                    if not rows:
                        misses += 1
                        if misses > MISS_LIMIT:
                            print(f"footprint backfill: {bar} 봉 빈 응답 {MISS_LIMIT}회, 포기",
                                  flush=True)
                            break
                        await asyncio.sleep(5.0)
                        continue
                    misses = 0
                    for row in rows:
                        ts_ms = int(row["T"])
                        if lo <= ts_ms < hi:   # 창 밖은 그 창의 차례에 받는다(또는 이미 있다)
                            # aggTrades 한 줄이 곧 주문 하나다 -- 나누지 않는다.
                            footprint_add(float(row["p"]), float(row["q"]), ts_ms,
                                          bool(row["m"]), agg=True)
                    next_id = int(rows[-1]["a"]) + 1
                    if len(rows) < 1000 or int(rows[-1]["T"]) >= hi:
                        break                  # 이 봉 끝
                    await asyncio.sleep(FOOTPRINT_CATCHUP_SECONDS)
                if budget <= 0:
                    print("footprint backfill: 요청 상한 소진 -- 남은 봉은 다음 재연결에", flush=True)
                    return
            footprint_state["ready"] = True    # 공백이 없다 = 화면의 「수집 중」을 내린다
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 실패하면 ready 가 False 로 남아 화면이
            # 계속 「수집 중」이라고 말한다. 다음 재연결이 다시 시도한다.
            print(f"footprint backfill failed: {exc}", flush=True)

    # 진행 중인 테이커 주문(같은 가격·방향·ms). 수집기와 **같은 클래스**를 쓴다 -- 되묶기
    # 규칙이 두 벌이 되면 화면과 DB 의 「고래」가 또 갈라진다.
    fp_orders = TakerOrderAggregator()

    async def collect_footprint(app: web.Application) -> None:
        backfill: asyncio.Task | None = None
        footprint_load()   # 지난 판이 남긴 봉들 -- 이게 있으면 아래 백필은 공백만 메운다
        # 공용 세션(binance_session)은 total=10초라 WS 에 못 쓴다 -- aiohttp 버전에 따라 그
        # 타임아웃이 WS 에도 걸려 10초마다 끊긴다(끊길 때마다 백필이 다시 뜬다). 전용 세션을
        # 쓰되 total=None 을 **명시**한다: aiohttp 기본값은 5분이라 그냥 두면 5분마다 끊긴다.
        ws_session = ClientSession(timeout=ClientTimeout(total=None),
                                   connector=TCPConnector(limit=2))
        try:
            while True:
                try:
                    async with ws_session.ws_connect(FOOTPRINT_WS_URL, heartbeat=30) as ws:
                        # WS 는 연결 이후만 준다 -- 그 앞의 빈 봉은 REST 가 메운다. 메시지를
                        # 읽으면서 **동시에** 채운다(백필을 기다리면 그동안 오는 체결이 aiohttp
                        # 큐에 수만 건 쌓인다).
                        first_ms: int | None = None
                        async for msg in ws:
                            if msg.type is not WSMsgType.TEXT:
                                break
                            trade = json.loads(msg.data)
                            if trade.get("e") != "trade":
                                continue
                            price, qty = float(trade["p"]), float(trade["q"])
                            if not (price > 0 and qty > 0):
                                # 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다
                                # (2026-09-16 실측 0.3%). 수량이 0이라 합계는 안 틀려서
                                # klines 대조로는 안 잡힌다 -- 대신 **가격 0 레벨**이 생긴다.
                                continue
                            ts_ms = int(trade["T"])
                            if first_ms is None:
                                # 백필의 경계를 **로컬 시계가 아니라 첫 체결의 거래소 시각**으로
                                # 잡는다. time.time() 으로 잡았더니 시계 차이만큼 REST 와 WS 가
                                # 겹쳐 그 봉만 +0.54% 더 세어졌다(2026-09-15 klines 대조).
                                first_ms = ts_ms
                                # 돌고 있는 백필이 있으면 새로 띄우지 않는다(REST 가중치가 두 배).
                                # 대가: 첫 백필 도중 WS 가 끊기면 그 공백은 다음 재연결 때 메워진다.
                                if backfill is None or backfill.done():
                                    fp_orders.reset()   # 끊김 전 묶음은 버린다
                                    footprint_state["ready"] = False
                                    backfill = asyncio.create_task(footprint_backfill(
                                        footprint_state["last_ms"], first_ms))
                            sell = bool(trade["m"])
                            footprint_add(price, qty, ts_ms, sell)      # 총량
                            tid = trade.get("t")
                            done = fp_orders.add(price, qty, ts_ms, sell,
                                                 None if tid is None else int(tid))
                            if done is not None:                        # 닫힌 주문의 크기 구간
                                footprint_add(done[0], done[1], done[2], done[3], order=True)
                            footprint_state["last_ms"] = ts_ms   # 다음 재연결이 메울 공백의 시작
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- 한 번의 끊김/에러가 수집기를 영구히
                    # 죽이면 재시작 전까지 풋프린트가 통째로 빈다(publish_dashboard_events 와
                    # 같은 이유). 끊기면 3초 뒤 다시 붙는다.
                    print(f"footprint collector cycle failed (will reconnect): {exc}", flush=True)
                await asyncio.sleep(3.0)
        finally:
            if backfill is not None:
                backfill.cancel()
            await ws_session.close()

    async def start_footprint_collector(app: web.Application) -> None:
        app["footprint_task"] = asyncio.create_task(collect_footprint(app))

    async def stop_footprint_collector(app: web.Application) -> None:
        # 내려가기 직전에 한 번 더 남긴다 -- 이러면 다음 판이 메울 공백이 «재시작에 걸린 시간»
        # (보통 1분 남짓)으로 줄어든다. 배포 재시작이 잦은 저장소라 이 한 줄이 제일 크게 먹는다.
        if footprint_state["ready"]:
            footprint_save()
        task = app["footprint_task"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 초 -> 그 초의 미결제약정(ETH). 값이 **바뀐** 초만 들어간다(OI_1S_URL 주석 참고).
    oi_1s: dict[int, float] = {}

    async def collect_oi_1s(app: web.Application) -> None:
        # 🔴«본 stamp 의 집합»이지 최고수위(last_ms)가 아니다. 바이낸스는 stamp 를 도착 순서대로
        #   주지 않는다 -- 2026-09-19 실측: stamp 가 응답에 나타나기까지 중앙 2.65초·p90 5.08초가
        #   걸리고, 그 편차 때문에 115개 중 4개(3.5%)는 «더 새 stamp 가 먼저» 도착했다. 최고수위로
        #   비교하면 그 4개를 «이미 본 것»으로 오인해 버린다(짝비교에서 실측 손실 4/243 과 일치).
        #   순서가 뒤바뀌어 들어와도 문제없다: 봉 집계는 ts_ms 로 arg_min/arg_max 를 잡고
        #   저장은 PK(ts_ms, symbol) 가 중복을 막는다.
        seen_ms: set[int] = set()
        pending: list[tuple[int, float]] = []      # 아직 duckdb 에 못 넣은 스냅샷
        flushed_at = time.time()
        while True:
            try:
                async with http_session["session"].get(
                        OI_1S_URL, params={"symbol": FOOTPRINT_SYMBOL}) as resp:
                    data = await resp.json()
                ts_ms = int(data["time"])
                if ts_ms not in seen_ms:     # 같은 스냅샷을 다른 초에 복제하지 않는다
                    seen_ms.add(ts_ms)
                    if len(seen_ms) > 8192:  # 메모리 상한만 건다. 10분이면 재도착이 끝난다(최대 6.4초).
                        seen_ms = {m for m in seen_ms if m >= ts_ms - 600_000}
                    sec = ts_ms // 1000
                    value = float(data["openInterest"])
                    oi_1s[sec] = value          # 화면 링은 1초 해상도 그대로
                    pending.append((ts_ms, value))
                    for old in [s for s in oi_1s if s < sec - SUPPLY_1S_SECONDS]:
                        del oi_1s[old]
                now = time.time()
                if pending and now - flushed_at >= OI_1S_FLUSH_SECONDS:
                    # 성공했을 때만 비운다 -- 파일이 잠깐 잠겨 있으면 다음 flush 로 미룬다.
                    await asyncio.to_thread(oi_1s_persist, pending)
                    pending = []
                    flushed_at = now
            except asyncio.CancelledError:
                # 정상 종료(배포 재기동)에서 미저장분을 버리지 않는다 -- 이 경로가 유일하게
                # 자주 도는 손실이었다(재기동마다 최대 OI_1S_FLUSH_SECONDS 만큼). 취소된
                # 태스크에서는 await 가 즉시 다시 취소되므로 to_thread 없이 그 자리에서 쓴다.
                if pending:
                    try:
                        oi_1s_persist(pending)
                    except Exception as exc:  # noqa: BLE001 -- 종료 중엔 알리고 넘어간다
                        print(f"oi-1s final flush failed: {exc}", flush=True)
                raise
            except Exception as exc:  # noqa: BLE001 -- 한 번의 실패로 수집을 영구히 멈추지 않는다
                print(f"oi-1s poll/flush failed (will retry): {exc}", flush=True)
                # 보류분이 끝없이 자라지는 않게 한다(디스크가 통째로 나간 경우). 1시간치면
                # 그건 일시적 잠금이 아니라 사람이 봐야 하는 고장이다.
                del pending[:-1200]
                flushed_at = time.time()
            await asyncio.sleep(OI_1S_POLL_SECONDS)

    async def start_oi_1s_collector(app: web.Application) -> None:
        app["oi_1s_task"] = asyncio.create_task(collect_oi_1s(app))

    async def stop_oi_1s_collector(app: web.Application) -> None:
        app["oi_1s_task"].cancel()
        try:
            await app["oi_1s_task"]
        except asyncio.CancelledError:
            pass

    # ── 미시 참고 (2026-09-20) ──────────────────────────────────────────────
    micro_state: dict[str, Any] = {"payload": {"available": False}, "baseline": None, "baseline_at": 0.0,
                                   "liq_prev": None, "liq_prev_at": 0.0,
                                   # 히스테리시스는 «직전 QI 쪽»을 들고 있어야 이어진다(micro_ref.qi_side_hyst).
                                   "qi_prev": "중립",
                                   # 마지막 동조 방아쇠. 화면은 «지금 상태»가 아니라 이걸 나이와 함께 그린다 --
                                   # 값의 전부가 발동한 그 초에 있어서(나이 0초 +0.55bp · 3초 이후 0) 늦추면 잃는다.
                                   "trigger": {"side": None, "ts": None}}
    agree_ring: deque = deque(maxlen=60)        # 최근 60초 동조 상태(+1/0/-1) -- 표시용 집계
    liq_events: deque = deque(maxlen=5000)      # (ts_ms, side, qty, price, usd) -- side "long" = 롱 포지션 청산(SELL)
    # WS 자체의 상태. 청산은 조용한 스트림이라 «이벤트 없음»과 «연결 없음»을 화면이 구별해야 한다
    # (tail_risk_interceptor 가 2026-07-30 에 77일간 잘못 connected=True 로 있던 그 함정).
    fo_state: dict[str, Any] = {"connected": False, "since": None, "last_event_ms": None, "events": 0, "errors": 0, "last_error": None}
    ofi_hist: deque = deque(maxlen=600)         # |OFI10| 최근 10분 -- 임계는 상수가 아니라 분위(p50)
    mp_state: dict[str, Any] = {"connected": False, "since": None, "last_ms": None, "events": 0, "errors": 0, "last_error": None}
    mark_ring: dict[int, tuple[float, float, float]] = {}   # sec → (mark, index, funding). 상황 읽기의 펀딩·베이시스 입력

    def liq_events_load() -> None:
        """재시작 직후 1회. 청산은 **11분에 몇 건**이라 빈 deque 로 시작하면 새로고침해도
        수급 차트의 청산선이 한동안 안 그려진다(체결·OI 는 초당 들어와 1초면 다시 찬다).
        아래 루프가 이미 쓰고 있는 jsonl 이 그 구간을 들고 있으니 그걸 되읽는다."""
        cut = time.time() * 1000 - SUPPLY_1S_SECONDS * 1000
        try:
            with open(LIQ_EVENTS_PATH, encoding="utf-8") as fh:
                # ponytail: 파일 전체를 훑는다(지금 200KB·연 65MB, 기동 1회). 커지면 tail 바이트만.
                tail = deque(fh, maxlen=liq_events.maxlen)
        except OSError:
            return
        for line in tail:
            try:
                ev = json.loads(line)
            except ValueError:
                continue    # 마지막 줄이 쓰다 만 상태일 수 있다
            if int(ev.get("ts_ms") or 0) >= cut:
                liq_events.append(ev)
        print(f"force-order: 지난 판 {len(liq_events)}건 복원", flush=True)

    async def collect_force_orders(app: web.Application) -> None:
        """⑤ @forceOrder 원시 이벤트를 jsonl 로 남기고 60초 링을 든다. 이벤트가 없으면 조용하다."""
        liq_events_load()
        ws_session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=2))
        try:
            while True:
                try:
                    async with ws_session.ws_connect(FORCE_ORDER_WS_URL, heartbeat=30) as ws:
                        fo_state.update(connected=True, since=time.time())
                        print("force-order ws: connected", flush=True)
                        async for msg in ws:
                            if msg.type is not WSMsgType.TEXT:
                                print(f"force-order ws: non-text {msg.type!r} -> reconnect", flush=True)
                                break
                            o = (json.loads(msg.data) or {}).get("o") or {}
                            if not o:
                                continue
                            fo_state["events"] += 1
                            fo_state["last_event_ms"] = int(o.get("T") or time.time() * 1000)
                            qty, price = float(o.get("z") or o.get("q") or 0.0), float(o.get("ap") or o.get("p") or 0.0)
                            ev = {"ts_ms": int(o.get("T") or time.time() * 1000),
                                  "side": "long" if o.get("S") == "SELL" else "short",   # 롱 청산 = 시장에 SELL
                                  "qty": qty, "price": price, "usd": qty * price, "symbol": o.get("s")}
                            liq_events.append(ev)
                            with open(LIQ_EVENTS_PATH, "a", encoding="utf-8") as fh:   # 분당 몇 줄 -- 블로킹 무시 가능
                                fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- 끊기면 3초 뒤 다시. 봇에 영향 없음
                    fo_state.update(errors=fo_state["errors"] + 1, last_error=repr(exc)[:120])
                    print(f"force-order ws: {exc!r}", flush=True)
                    await asyncio.sleep(3.0)
                finally:
                    fo_state["connected"] = False
        finally:
            await ws_session.close()

    async def collect_mark_price(app: web.Application) -> None:
        """⑥ @markPrice@1s → 메모리 링 + duckdb(10초마다 flush). 끊기면 3초 뒤 다시."""
        ws_session = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=2))
        pending: list[tuple[int, float, float, float, int]] = []
        flushed_at = time.time()
        try:
            while True:
                try:
                    async with ws_session.ws_connect(MARK_PRICE_WS_URL, heartbeat=30) as ws:
                        mp_state.update(connected=True, since=time.time())
                        print("mark-price ws: connected", flush=True)
                        async for msg in ws:
                            if msg.type is not WSMsgType.TEXT:
                                print(f"mark-price ws: non-text {msg.type!r} -> reconnect", flush=True)
                                break
                            o = json.loads(msg.data) or {}
                            if o.get("e") != "markPriceUpdate":
                                continue
                            ts_ms = int(o["E"]); mark, idx, fr = float(o["p"]), float(o["i"]), float(o["r"])
                            mp_state["events"] += 1; mp_state["last_ms"] = ts_ms
                            mark_ring[ts_ms // 1000] = (mark, idx, fr)
                            pending.append((ts_ms, mark, idx, fr, int(o.get("T") or 0)))
                            now = time.time()
                            if now - flushed_at >= OI_1S_FLUSH_SECONDS:
                                for old in [sc for sc in mark_ring if sc < now - MARK_PRICE_RING_S]:
                                    del mark_ring[old]
                                try:
                                    await asyncio.to_thread(mark_price_persist, pending)
                                    pending = []
                                except Exception as exc:  # noqa: BLE001 -- 잠깐 잠기면 다음 flush 로. WS 는 유지
                                    print(f"mark-price flush failed (will retry): {exc}", flush=True)
                                    del pending[:-1200]
                                flushed_at = now
                except asyncio.CancelledError:
                    if pending:
                        try:
                            mark_price_persist(pending)
                        except Exception as exc:  # noqa: BLE001
                            print(f"mark-price final flush failed: {exc}", flush=True)
                    raise
                except Exception as exc:  # noqa: BLE001 -- 끊기면 3초 뒤 다시. 봇에 영향 없음
                    mp_state.update(errors=mp_state["errors"] + 1, last_error=repr(exc)[:120])
                    print(f"mark-price ws: {exc!r}", flush=True)
                    await asyncio.sleep(3.0)
                finally:
                    mp_state["connected"] = False
        finally:
            await ws_session.close()

    async def collect_micro_ref(app: web.Application) -> None:
        from scripts.live_orderflow_raster_collector_20260914 import read_window  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        while True:
            try:
                now = time.time(); now_sec = int(now)
                # 기준선이 없으면 60초마다 다시 시도한다 -- 수집기가 같은 파일에 쓰는 순간 read_only
                # 연결이 거부될 수 있고(2026-09-19 실측 30회 중 2회), 그때 1시간을 «기준없음»으로 두면
                # 카드 절반이 하루 종일 빈다. 실패 사유는 로그에 남긴다(조용히 None 이면 원인을 못 찾는다).
                if now - micro_state["baseline_at"] >= (MICRO_BASELINE_SECONDS if micro_state["baseline"] else 60.0):
                    micro_state["baseline_at"] = now
                    try:
                        micro_state["baseline"] = await asyncio.to_thread(mref.baseline_from_tape, MICRO_TAPE_DB_PATH)
                    except Exception as exc:  # noqa: BLE001
                        micro_state["baseline"] = None
                        print(f"micro-ref baseline: {exc!r}", flush=True)
                    if micro_state["baseline"] is None:
                        print("micro-ref baseline: none (재시도 60초 뒤)", flush=True)
                if now - micro_state["liq_prev_at"] >= 10.0:
                    micro_state["liq_prev_at"] = now
                    micro_state["liq_prev"] = await asyncio.to_thread(mref.liq_prev_minute, LIVE_DIR / "tail_risk.duckdb")
                book = await fetch_binance_json(MICRO_BOOK_URL, {"symbol": FOOTPRINT_SYMBOL})
                qi_val = mref.qi(float(book["bidQty"]), float(book["askQty"]))
                mid = (float(book["bidPrice"]) + float(book["askPrice"])) / 2.0
                w = await loop.run_in_executor(
                    HEATMAP_EXECUTOR, functools.partial(read_window, FOOTPRINT_SYMBOL.lower(), int(now * 1000), 12, 1))
                flow = mref.raster_flow(w)
                if flow.get("ok"):
                    ofi_hist.append(abs(flow["ofi10"]))
                ofi_thr = float(np.median(ofi_hist)) if len(ofi_hist) >= 30 else None
                if flow.get("ok") and ofi_thr:
                    agree, micro_state["qi_prev"] = mref.agree_state(
                        qi_val, flow["ofi10"], ofi_thr, micro_state["qi_prev"])
                else:
                    agree = "기준없음"
                agree_ring.append(1 if agree == "동조매수" else -1 if agree == "동조매도" else 0)
                if agree in ("동조매수", "동조매도"):
                    micro_state["trigger"] = {"side": agree, "ts": now_sec}
                trig = dict(micro_state["trigger"])
                trig["age"] = (now_sec - trig["ts"]) if trig["ts"] else None
                trig["live"] = bool(trig["age"] is not None and trig["age"] <= mref.TRIGGER_LIVE_S)
                by_sec = footprint_state["sec"]
                vol1s = [(sec, cell[4] + cell[5]) for sec, cell in by_sec.items() if sec >= now_sec - 90]
                vol60 = sum(v for sec, v in vol1s if now_sec - 61 < sec < now_sec)   # 진행 중인 초는 뺀다
                hour = int((now_sec % 86400) // 3600)
                base = micro_state["baseline"]
                vol60_pct = mref.pct_rank(vol60, base["minute_vol_sorted"][hour]) if base else None
                med = base["minute_vol_sorted"][hour] if base else None
                vol60_x = (vol60 / med[len(med) // 2]) if med and med[len(med) // 2] > 0 else None
                levels = await load_liquidation_map("eth")
                sr = mref.sr_context(levels if levels.get("warmed_up") else None, mid, flow.get("imb40") if flow.get("ok") else None)
                if levels.get("warmed_up"):   # 상황 읽기의 플러시 목표(청산 군집)용 -- 가까운 순 [{price, weight_pct}]
                    sr["sup_levels"] = [{"price": lv["price"], "weight_pct": lv.get("weight_pct")} for lv in levels.get("support_levels") or []]
                    sr["res_levels"] = [{"price": lv["price"], "weight_pct": lv.get("weight_pct")} for lv in levels.get("resistance_levels") or []]
                burst = mref.burst_state(vol1s, oi_1s, (base or {}).get("sec_vol_p99", {}).get(hour), now_sec)
                cut = now * 1000 - 60_000
                ev60 = [e for e in liq_events if e["ts_ms"] >= cut]
                liq60 = {"long": sum(e["usd"] for e in ev60 if e["side"] == "long"),
                         "short": sum(e["usd"] for e in ev60 if e["side"] == "short"), "n": len(ev60)}
                micro_state["tick"] = micro_state.get("tick", 0) + 1
                micro_state["payload"] = {
                    "available": True, "ts": now_sec, "mid": mid,
                    "qi": qi_val, "qi_side": mref.side_of(qi_val, mref.QI_SIDE_ABS),
                    "ofi10": flow.get("ofi10") if flow.get("ok") else None, "ofi_thr": ofi_thr,
                    "ofi_side": (mref.side_of(flow["ofi10"], ofi_thr) if (flow.get("ok") and ofi_thr) else "기준없음"),
                    "agree": agree, "trigger": trig,
                    # «최근 60초 중 몇 초가 어느 쪽이었나» -- 안 흔들리는 맥락. 🔴예측을 더 해 주지는 않는다
                    # (즉시 동조를 이 부호로 걸러도 fwd15 +0.62 vs +0.58 로 SE 안, 실측). 상태 서술이다.
                    "tally60": {"buy": int(sum(1 for v in agree_ring if v > 0)),
                                "sell": int(sum(1 for v in agree_ring if v < 0)), "n": len(agree_ring)},
                    "qi_enter": mref.QI_ENTER, "qi_exit": mref.QI_EXIT,
                    "imb10": flow.get("imb10") if flow.get("ok") else None, "imb40": flow.get("imb40") if flow.get("ok") else None,
                    "vol60": vol60, "vol60_pct": vol60_pct, "vol60_x_p50": vol60_x, "act": mref.act_label(vol60_pct),
                    "baseline_days": (base or {}).get("days"), "hour_utc": hour,
                    "sr": sr, "burst": burst, "liq60": liq60, "liq_prev": micro_state["liq_prev"],
                    "fo": dict(fo_state), "mp": dict(mp_state),
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 참고 카드 하나가 죽어도 나머지 화면은 그대로
                micro_state["payload"] = {"available": False, "error": repr(exc)[:160]}
            # 상황 읽기는 5틱마다, 그리고 자기 예외는 자기가 삼킨다(미시 참고를 못 죽인다)
            if micro_state.get("tick", 0) % SITUATION_EVERY_TICKS == 0:
                try:
                    await compute_situation(time.time())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    situation_state["now"] = {"ok": False, "reason": repr(exc)[:160]}
            await asyncio.sleep(MICRO_REF_POLL_SECONDS)

    async def start_micro_ref(app: web.Application) -> None:
        _situation_load_log()
        app["micro_ref_task"] = asyncio.create_task(collect_micro_ref(app))
        app["force_order_task"] = asyncio.create_task(collect_force_orders(app))
        app["mark_price_task"] = asyncio.create_task(collect_mark_price(app))

    async def stop_micro_ref(app: web.Application) -> None:
        for key in ("micro_ref_task", "force_order_task", "mark_price_task"):
            app[key].cancel()
            try:
                await app[key]
            except asyncio.CancelledError:
                pass

    async def api_micro_ref(request: web.Request) -> web.Response:
        return web.json_response(micro_state["payload"], headers=NOCACHE)

    # ── 상황 읽기 · 30분 (2026-09-21) ──────────────────────────────────────────
    situation_state: dict[str, Any] = {"now": {"ok": False, "reason": "계산 전"}, "computed_at": 0.0,
                                       "log": [], "last_logged": 0.0, "last_key": None, "last_resolve": 0.0}

    def _situation_append(rec: dict[str, Any]) -> None:
        try:
            with open(SITUATION_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"situation log write: {exc!r}", flush=True)

    def _situation_load_log() -> None:
        """예측 줄({ts, labels, ...})과 해결 줄({ts, outcome})이 한 파일에 섞여 있다. 해결을 영속하지 않으면
        재기동(배포)마다 outcome 이 None 으로 돌아가 캔들 캐시(8시간) 밖의 예측은 영영 못 푼다."""
        try:
            entries: dict[int, dict[str, Any]] = {}
            # 파일은 계속 자란다(P1 부터 피쳐까지 실린다). 꼬리만 파싱한다 -- 해결 줄은 예측 30분 뒤라
            # 같은 꼬리 안에 들어온다. 이보다 오래된 항목은 이미 해결돼 화면 창 밖이다.
            # 🔴read_text 는 슬라이스 전에 파일 전체를 메모리에 올린다. 이 기기는 실거래 봇과 공유하고
            #   24GB OOM 이력이 있다(2026-09-18). deque 로 **꼬리만** 들고 있는다.
            with SITUATION_LOG_PATH.open(encoding="utf-8") as fh:
                tail = deque(fh, maxlen=20 * SITUATION_LOG_KEEP)
            for x in tail:
                if not x.strip():
                    continue
                rec = json.loads(x)
                if "labels" in rec:
                    entries[int(rec["ts"])] = rec
                elif int(rec.get("ts", -1)) in entries:
                    entries[int(rec["ts"])]["outcome"] = rec.get("outcome")
                    entries[int(rec["ts"])]["outcome_sym"] = rec.get("outcome_sym")
                    entries[int(rec["ts"])]["path"] = rec.get("path")
            kept: list[dict[str, Any]] = []
            for rec in entries.values():   # 같은 상태 서명이 300초 안에 이어지면 중복 -- 09-21 이전 파일의 5초 중복을 소급 정리
                if kept and rec["ts"] - kept[-1]["ts"] < SITUATION_LOG_MIN_GAP_S and sit.log_key(rec) == sit.log_key(kept[-1]):
                    continue
                kept.append(rec)
            situation_state["log"] = kept[-SITUATION_LOG_KEEP:]
        except FileNotFoundError:
            situation_state["log"] = []
        except Exception as exc:  # noqa: BLE001
            print(f"situation log load: {exc!r}", flush=True)
            situation_state["log"] = []

    def _situation_inputs(now: float) -> dict[str, Any]:
        """기존 서버 상태를 situation.classify 의 입력 모양으로 접는다. 새 원천은 없다."""
        bar_start = int(now) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS
        fbars = footprint_state["bars"]
        # OI·청산 5분 집계는 compute_situation 이 5초 swr 캐시로 미리 받아 둔다(둘 다 매 호출 duckdb 를 열어 11·18ms).
        oi_map = {int(b): (None if gap else float(dl)) for b, dl, _c, _n, gap in (situation_state.get("oi_5m") or [])}
        liq_map: dict[int, tuple[float, float]] = {}
        try:
            for r in ((situation_state.get("liq_5m") or {}).get("bars") or []):
                ts = int(datetime.fromisoformat(r["ts"]).timestamp())
                liq_map[ts // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS] = (float(r.get("long_usd") or 0), float(r.get("short_usd") or 0))
        except Exception:  # noqa: BLE001 -- 청산 이력이 없으면 «모름»으로 간다
            pass
        candles = {int(c["time"]): c for c in (situation_state.get("candles") or [])}
        bars = []
        levels: dict[float, float] = {}
        want = [b for b in sorted(fbars) if b < bar_start][-(2 * sit.WINDOW + 1):]
        for b in want:
            cells = fbars[b]
            buy = sum(v[0] for v in cells.values()); sell = sum(v[1] for v in cells.values())
            c = candles.get(b)
            prices = [k * FOOTPRINT_BUCKET for k in cells]
            bars.append({"time": b,
                         "high": float(c["high"]) if c else (max(prices) if prices else None),
                         "low": float(c["low"]) if c else (min(prices) if prices else None),
                         "close": float(c["close"]) if c else None,
                         "delta": buy - sell, "vol": buy + sell,
                         "whale_net": sum(v[2] - v[3] for v in cells.values()),
                         "retail_net": sum(v[4] - v[5] for v in cells.values()),
                         "oi_delta": oi_map.get(b), "liq_long": liq_map.get(b, (0, 0))[0], "liq_short": liq_map.get(b, (0, 0))[1],
                         # 추세 veto(2026-09-22) -- load_market_history 가 이미 봉마다 붙여 준다.
                         "veto": (c or {}).get("veto")})
            if b >= (want[-sit.WINDOW] if len(want) >= sit.WINDOW else want[0]):
                for k, v in cells.items():
                    row = round(k * FOOTPRINT_BUCKET / SITUATION_VA_ROW_USD) * SITUATION_VA_ROW_USD
                    levels[row] = levels.get(row, 0.0) + v[0] + v[1]
        # 종가가 캔들 캐시에 없으면(60초 캐시 지연, 또는 캐시 자체가 없는 dev) 초 링의 마지막 체결가로,
        # 그것도 없으면 그 봉 풋프린트의 **VWAP** 으로 메운다 -- 되돌림/지속 판정에 종가 근사는 충분하고,
        # 봉이 비어서 엔진이 통째로 «완결 봉 부족»으로 멈추는 것보다 낫다.
        by_sec = footprint_state["sec"]
        for bar, b in zip(bars, want):
            if bar["close"] is None:
                secs = [sc for sc in by_sec if b <= sc < b + FOOTPRINT_BAR_SECONDS and by_sec[sc][6]]
                if secs:
                    bar["close"] = by_sec[max(secs)][6]
                else:
                    cells = fbars[b]; tot = sum(v[0] + v[1] for v in cells.values())
                    bar["close"] = (sum(k * FOOTPRINT_BUCKET * (v[0] + v[1]) for k, v in cells.items()) / tot) if tot > 0 else None
        cur_cells = [by_sec[sc] for sc in by_sec if sc >= bar_start]
        cur = {"elapsed_s": int(now) - bar_start,
               "whale_net": sum(c[2] - c[3] for c in cur_cells), "retail_net": sum(c[0] - c[1] for c in cur_cells),
               "delta": sum(c[4] - c[5] for c in cur_cells), "oi_delta": oi_map.get(bar_start)}
        # BTC 같은 창: ETH 봉과 같은 시각 구간의 이동·고저폭(bp). 캔들이 없거나 구간이 안 맞으면 생략
        btc: dict[str, float] = {}
        bc = {int(c["time"]): c for c in (situation_state.get("btc_candles") or [])}
        if len(want) >= sit.WINDOW + 1 and want[-sit.WINDOW - 1] in bc and want[-1] in bc:
            c0, c1 = bc[want[-sit.WINDOW - 1]], bc[want[-1]]
            span = [bc[t] for t in want[-sit.WINDOW:] if t in bc]
            btc = {"move_bp": (c1["close"] - c0["close"]) / c0["close"] * 1e4,
                   "range_bp": (max(c["high"] for c in span) - min(c["low"] for c in span)) / c1["close"] * 1e4}
        mp = micro_state["payload"] if micro_state["payload"].get("available") else {}
        bo = situation_state.get("breakout") or {}
        return {"bars": bars, "levels": levels, "cur": cur, "mid": mp.get("mid"),
                "book": situation_state.get("book") or {}, "act_pct": mp.get("vol60_pct"), "sr": mp.get("sr") or {},
                "deriv": mref.deriv_from_ring(mark_ring, sit.WINDOW * FOOTPRINT_BAR_SECONDS, sit.BASIS_PCT), "btc": btc,
                "breakout": {"detect_on": bool((bo.get("detect") or {}).get("on")), "prewarn_on": bool((bo.get("prewarn") or {}).get("on"))}}

    async def load_minute_candles() -> list[dict[str, float]]:
        """예측 해결 전용 1분봉. 🔴5분봉으로 풀면 ①결정 시점이 든 봉을 통째로 건너뛰어 30분 창의
        최대 17%가 죽고 ②같은 봉 안 순서를 몰라 오채점된다 -- 09-21 장부를 두 해상도로 풀어보니
        13.4%가 달라졌다. 20초 캐시라 1초 루프가 매번 부르지 않는다."""
        async def produce() -> list[dict[str, float]]:
            raw = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {"symbol": FOOTPRINT_SYMBOL, "interval": "1m", "limit": SITUATION_MINUTE_LIMIT},
                error_reason="situation_minute_upstream_error")
            # 🔴마지막 행은 **형성 중인 봉**이다. 그대로 두면 해결의 일부가 «t+29분+α 의 임의 시점»으로
            #   확정되고(ret_bp 가 창 끝 종가가 아니게 된다) 마지막 수십 초의 터치를 놓친다.
            #   해결 줄은 영속되므로 되돌릴 수 없다 -- 닫힌 봉만 쓴다.
            return [{"time": int(r[0]) // 1000, "high": float(r[2]), "low": float(r[3]), "close": float(r[4])}
                    for r in raw[:-1]]
        # max_stale: 바이낸스가 느리면 **직전 값을 내주고 뒤에서 갱신**한다. 없으면 1초 루프가
        # 매초 왕복을 기다려 미시 참고까지 같이 멈춘다(09-21 검토).
        return await swr_cached("situation_minutes", 20.0, produce, max_stale=STALE_GRACE_SECONDS)

    async def compute_situation(now: float) -> None:
        loop = asyncio.get_running_loop()
        # 느린 입력 셋은 캐시로 (캔들 60초 · 전환탐지기 60초 · 호가 요약 5초)
        situation_state["candles"] = await load_market_history("eth")
        try:
            situation_state["btc_candles"] = await load_market_history("btc")   # 같은 캐시 프레임에서 자른다
        except Exception:  # noqa: BLE001 -- BTC 가 없으면 그 라벨만 빠진다
            situation_state["btc_candles"] = None
        situation_state["breakout"] = await load_breakout_detector()
        try:
            situation_state["minutes"] = await load_minute_candles()
        except Exception as exc:  # noqa: BLE001 -- 실패하면 직전 1분봉으로 푼다(없으면 해결을 미룬다)
            print(f"situation minutes: {exc!r}", flush=True)
        situation_state["oi_5m"] = await swr_cached("situation_oi5m", 5.0, lambda: asyncio.to_thread(oi_5m_buckets, 2 * sit.WINDOW + 3))
        situation_state["liq_5m"] = await swr_cached("situation_liq5m", 5.0, lambda: asyncio.to_thread(compute_liquidation_5m_history, "eth", 2 * sit.WINDOW + 3))
        try:
            hm = await swr_cached("situation_book", 5.0, lambda: loop.run_in_executor(
                HEATMAP_EXECUTOR, functools.partial(_heatmap_read, "ethusdt", 300, 3, True)))
            situation_state["book"] = (hm or {}).get("summary") or {}
        except Exception:  # noqa: BLE001
            situation_state["book"] = {}
        inp = await asyncio.to_thread(_situation_inputs, now)
        # 슈미트 트리거의 «이전 레짐». classify 는 순수 함수라 상태를 여기서 들고 넘긴다.
        # 재기동 직후엔 0(중립)에서 시작한다 -- 추세로 들어가려면 ENTER 를 넘어야 한다.
        inp["prev_dir"] = (situation_state.get("now") or {}).get("dir") or 0
        res = sit.classify(inp)
        res["computed_at"] = now
        situation_state["now"] = res
        situation_state["computed_at"] = now
        if not res.get("ok"):
            return
        key = sit.log_key(res)   # 라벨의 숫자를 뺀 종류 서명 -- 숫자를 두면 5초마다 새 항목(09-21 1,742건 사고)
        # 🔴그래도 «상태 종류»는 초 단위로 떨린다: 실측 해결 1,867줄이 구별되는 분은 **366분**뿐이고
        #   한 분에 최대 24줄이었다. 같은 분의 줄들은 라벨도 결과도 같다(해결이 보는 1분봉 집합이 같다).
        #   깜빡임이 심한 구간이 표본에서 5배 가중되므로 **분당 1줄**로 막는다.
        minute = int(now) // 60
        if minute != situation_state.get("last_minute") and (
                key != situation_state["last_key"] or now - situation_state["last_logged"] >= SITUATION_LOG_MIN_GAP_S):
            situation_state["last_minute"] = minute
            entry = {"ts": int(now), "mid": inp.get("mid"), "dir": res["dir"], "prob": res["prob"], "targets": res["targets"],
                     "labels": res["labels"], "flips_on": [f["signal"] for f in res["flips"] if f["on"]],
                     # 대칭 라벨(학습 주 타깃)과 목표 거리 -- 거리를 같이 남겨야 «근접 편향»을 나중에 다시 잴 수 있다
                     "sym": res.get("sym"), "dist_bp": res.get("dist_bp"), "outcome": None, "outcome_sym": None,
                     "scorable": res.get("scorable", True), "passed": res.get("passed") or [],
                     # P1 학습 표본: 규칙의 근거 dict 가 그대로 피쳐다(전부 결정 시점 t 까지의 값).
                     # 🔴분위 정규화는 여기서 하지 않는다 -- 원시값을 남겨야 나중에 정규화 방식을 바꿀 수 있다.
                     "feat": res.get("evidence")}
            situation_state["log"].append(entry); situation_state["log"] = situation_state["log"][-SITUATION_LOG_KEEP:]
            situation_state["last_key"], situation_state["last_logged"] = key, now
            _situation_append(entry)
        if now - situation_state["last_resolve"] >= 60:
            situation_state["last_resolve"] = now
            ms = situation_state.get("minutes") or []
            for e in situation_state["log"]:
                if e.get("outcome") is None and e["ts"] + SITUATION_HORIZON_S <= now:
                    e["outcome"] = sit.resolve(e, ms, SITUATION_HORIZON_S)
                    e["outcome_sym"] = sit.resolve_sym(e, ms, SITUATION_HORIZON_S)
                    e["path"] = sit.path_targets(e, ms, SITUATION_HORIZON_S)      # P1 연속 타깃
                    if e["outcome"] is not None:
                        _situation_append({"ts": e["ts"], "outcome": e["outcome"],
                                           "outcome_sym": e.get("outcome_sym"),
                                           "path": e.get("path")})   # 해결 줄 -- 재기동 뒤에도 남는다

    async def api_situation(request: web.Request) -> web.Response:
        log = situation_state["log"]
        return web.json_response({"now": situation_state["now"], "computed_at": situation_state["computed_at"],
                                  "recent": [{k: e.get(k) for k in ("ts", "mid", "dir", "prob", "outcome", "outcome_sym", "flips_on")} for e in log[-12:]],
                                  "calibration": sit.calibration(log, SITUATION_HORIZON_S),
                                  "streams": {"fo": dict(fo_state), "mp": dict(mp_state)}}, headers=NOCACHE)

    async def load_chart_klines_frames() -> dict[str, Any]:
        """차트 캔들용 ETH/BTC 5분봉 프레임 캐시 (1500봉, 닫힌 봉만).

        🔴2026-09-16: 원래 이름은 load_evidence_signals() 였고 세 일을 겸했다 -- klines 수집 +
        증거신호 8종 compute_signals()(실측 8.22초/회, 대시보드 최고 비용 항목) + TabPFN
        메타라벨 병합. 증거신호 칩을 내리면서 **계산만 걷어내고 수집은 남겼다**:
        load_market_history_from_evidence_cache() 가 이 frames 에서 차트 캔들을 슬라이스한다.
        복원 안내: docs/experiments/eth_evidence_signal_chips_removed_20260916.md
        """
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

            # BTC 레그 -- 차트의 BTC 캔들용. 실패해도 ETH 캔들은 떠야 하므로 잡아서 로그만 남긴다
            # (2026-09-16 이전엔 smt_divergence 의 교차자산 레그를 겸했다).
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

            # 세 번째 칸(옛 funding_df)은 orthogonal_combo 전용이었다 -- 신호를 내렸으므로 None.
            evidence_signal_cache["frames"] = (
                df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]],
                btc_df,
                None,
            )
            return {"available": True, "bars": int(len(df)),
                    "btc_bars": int(len(btc_df)) if btc_df is not None else 0}

        # 🔴max_stale=0 이어야 한다. 이 캐시를 데우는 곳은 /api/market-history 하나뿐인데
        # (위 load_market_history_from_evidence_cache 의 유일한 호출부), 프런트 폴링 주기는
        # CANDLE_HISTORY_POLL_MS=300초로 TTL 60초의 5배다 -- 그래서 «매 요청이» age>ttl 로
        # 낡은 가지를 타고 **5분 묵은 프레임**을 받아갔다. 거기에 위의 형성봉 버리기가 겹쳐
        # 마지막 봉이 «현재봉-2」가 되는데, 클라이언트(updateSnapshotCandleLive)는 현재 버킷
        # 한 봉만 밀어 넣으므로 그 사이가 **상시 한 봉짜리 구멍**으로 남았다. 풋프린트·청산맵이
        # 같은 캔들 배열을 그리므로 둘 다 같이 빈다. 2026-09-17 실측: 14:06:02 첫 호출 13:55,
        # 19초 뒤 재호출 14:00 -- 낡은 값을 주고 뒤에서 갱신하던 게 그대로 보였다.
        # 0 으로 두면 매 폴링이 klines 왕복을 기다려 「현재봉-1」을 주고 클라가 한 봉 메워
        # 정확히 맞는다. 5분에 한 번 왕복이라 SWR 도입 이전과 같은 비용이다.
        return await swr_cached(
            "evidence_signal", EVIDENCE_SIGNAL_CACHE_SECONDS, produce,
            cache=evidence_signal_cache,
            max_stale=0.0,
        )
    async def load_breakout_detector() -> dict[str, Any]:
        """횡보→추세 전환 탐지기 -- 워커가 쓴 상태 파일을 읽기만 한다(계산 인라인 금지)."""
        return await swr_cached(
            "breakout_detector", EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(breakout_detector_payload),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_chart_markers(asset: str = "eth") -> dict[str, Any]:
        """청산맵 차트 마커 -- scripts/live_eth_chart_markers_20260909.py 참고.
        ETH 전용이다(다른 코인은 unsupported 로 비운다 -- 빈 레인은 "신호 없음"으로 오독된다).
        2026-09-21: V자·극점·E|r| 게이트를 내려 **돌파 한 줄만** 남았다.
        ⚠️극점을 여기서 인라인으로 채점하면 안 된다 -- 2026-09-10 그 인라인 호출이 TabPFN
          아티팩트(1.08GB)를 60초마다 로드해 to_thread 풀을 고갈시켰고 증거신호를 포함한 모든
          계산 엔드포인트가 멈췄다. 자세한 실측은 live_eth_chart_markers_20260909.py 주석."""
        if (asset or "eth").lower() != "eth":
            return compute_chart_markers(asset)
        # 실패해도 마커 전체를 죽이지 않는다: 그 줄만 비고 나머지는 그려진다.
        try:
            bo = await load_breakout_detector()
        except Exception:  # noqa: BLE001
            bo = None
        return await swr_cached(
            "chart_markers", EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_chart_markers, "eth", None, None, bo, None),
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
        asyncio.to_thread reasoning as load_chart_klines_frames() above.

        asset: 2026-08-31, BTC added -- see coin_config.py for BTC's separate tail-risk file."""
        return await swr_cached(
            f"liquidation_5m:{asset}", LIQUIDATION_5M_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_liquidation_5m_signal, coin=asset),
        )

    async def load_liquidation_direction_signal(asset: str = "eth") -> dict[str, Any]:
        """Directional-only liquidation tilt (liq_net_z_12, contrarian sign) -- model-indicator
        tier, no PnL/economic claim. See scripts/live_liquidation_direction_signal_20260825.py
        docstring. Same 60s cache reasoning as load_liquidation_5m_signal() above (underlying data
        updates once per minute).

        asset: 2026-08-31, BTC added -- see coin_config.py for BTC's separate tail-risk file."""
        return await swr_cached(
            f"liquidation_direction:{asset}", EVIDENCE_SIGNAL_CACHE_SECONDS,
            lambda: asyncio.to_thread(compute_liquidation_direction_signal, coin=asset),
        )

    async def load_liquidation_map(asset: str = "eth") -> dict[str, Any]:
        """Snapshot-tab liquidation map (estimated support/resistance) -- see
        scripts/live_liquidation_map_20260824.py docstring for the estimation methodology and its
        caveats. Mirrors load_chart_klines_frames()'s klines-fetch/cache pattern (own cache, since
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
            f"liquidation_map:{asset}", LIQUIDATION_MAP_CACHE_SECONDS, produce,
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_wide24() -> dict[str, Any]:
        """wide24 HMM regime overlay for the Snapshot tab's liquidation-map chart -- see
        scripts/live_regime_gbm3_signal_20260826.py docstring.

        ⚠️모듈 이름이 화면 이름과 다르다 -- wide24 는 **live_regime_gbm3_signal_20260826** 의
        compute_regime_gbm3_signal() 이다(별칭으로 import 돼 있었다). 워커 spec 을 쓸 때
        파일명을 짐작하면 틀린다.

        2026-09-14: **워커 상태 파일을 읽기만 한다.** 매 사이클 13초로 일정했고(서버 실측
        13.73/12.86/13.30s) 캐시 만료마다 그대로 다시 냈다. BTC/XRP 판과 함께 워커 한 프로세스가
        순서대로 돈다."""
        return await swr_cached(
            "regime_wide24", REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(regime_payload, REGIME_WIDE24_STATE_PATH),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_btc() -> dict[str, Any]:
        """BTC regime overlay for the Snapshot tab when its coin switcher is on BTC -- the BTC twin
        of load_regime_wide24() above, same cache TTL, same asyncio.to_thread offload (the scorer
        fetches from Binance and runs FeatureEngineer, both blocking), and the same never-raises
        contract (degrades to warmed_up=False so the ribbon shows its waiting state rather than
        breaking the chart)."""
        return await swr_cached(
            "regime_btc", REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(regime_payload, REGIME_BTC_STATE_PATH),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_regime_xrp() -> dict[str, Any]:
        """XRP 3-class 레짐(S96_K9, 2026-09-03) -- load_regime_btc()의 XRP판.
        같은 캐시 TTL / asyncio.to_thread / never-raises 계약."""
        return await swr_cached(
            "regime_xrp", REGIME_WIDE24_CACHE_SECONDS,
            lambda: asyncio.to_thread(regime_payload, REGIME_XRP_STATE_PATH),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def load_macro_calendar() -> dict[str, Any]:
        """US macro/corporate event calendar for the Snapshot tab -- see scripts/live_macro_
        calendar_20260826.py docstring for the 6 sources.

        2026-09-14: **워커 상태 파일을 읽기만 한다.** 그 전에는 여기서 compute_macro_calendar()
        (동기 requests 로 외부 6개 소스)를 돌렸고 콜드 51.03초였다(서버 실측). 계산은
        scripts/live_signal_worker.py 가 compute_macro_calendar_from_env() 로 대신 돈다."""
        return await swr_cached(
            "macro_calendar", MACRO_CALENDAR_CACHE_SECONDS,
            lambda: asyncio.to_thread(macro_calendar_payload),
            max_stale=STALE_GRACE_SECONDS,
        )

    async def start_http_session(app: web.Application) -> None:
        http_session["session"] = ClientSession(
            timeout=ClientTimeout(total=10),
            connector=TCPConnector(limit=32, ttl_dns_cache=300, keepalive_timeout=60),
        )

    async def stop_http_session(app: web.Application) -> None:
        # Cancel any in-flight background refresh FIRST: those tasks hold a reference to the
        # session closed below, so letting them outlive it leaves pending tasks to be destroyed
        # mid-request at loop shutdown.
        for pending in tuple(refresh_tasks.values()):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*refresh_tasks.values(), return_exceptions=True)
        refresh_tasks.clear()
        session = http_session["session"]
        http_session["session"] = None
        if session is not None and not session.closed:
            await session.close()

    async def produce_account() -> dict[str, Any]:
        """계좌 조회는 여기 한 곳뿐이다 -- 화면 요청이든 원장 주기든 같은 캐시를 통과하므로
        조회가 두 벌로 갈라지지 않는다. 원장 기록을 이 안에 둔 이유도 같다."""
        # 🔴수동 주문 심볼(ETHUSDC)을 같이 넣는다. 이게 빠져 있어서 그 심볼의 **최근 거래**가
        #   통째로 안 보였다(2026-09-19 사용자 보고). 포지션 자체는 /fapi/v2/positionRisk 를
        #   심볼 필터 없이 부르므로 원래 다 들어온다 -- 빠지는 건 userTrades(왕복) 쪽뿐이다.
        # 🔴잔고를 어느 «자산 지갑»에서 읽을지 정한다. 단일자산 담보 모드에서 최상위 total*
        #   합계는 **USDT 전용**이라, 현금이 USDC 에만 있으면 전부 0 으로 내려온다(2026-09-19
        #   실측: USDC 1,472 인데 잔고 0). 수동 주문이 나가는 심볼의 담보 자산을 따른다 --
        #   증거금을 실제로 먹는 지갑이 거기다.
        quote = "USDC" if MANUAL_EXEC_SYMBOL.endswith("USDC") else "USDT"
        payload = await fetch_account(
            binance_session(),
            list(dict.fromkeys([*MARKET_SYMBOLS.values(), MANUAL_EXEC_SYMBOL])),
            quote_asset=quote)
        # 화면이 «이 코인의 포지션»을 찾을 때 USDT 심볼 하나만 보면 USDC 포지션을 못 본다.
        # 서버가 실제로 쓰는 심볼을 payload 에 실어 보내 화면이 하드코딩하지 않게 한다 --
        # 환경변수(DASHBOARD_MANUAL_EXEC_SYMBOL)로 바뀌는 값이다.
        if isinstance(payload, dict):
            payload["exec_symbol"] = MANUAL_EXEC_SYMBOL
        added = record_account_trips(payload, account_trip_state["seen"])
        if added:
            print(f"account_round_trips: +{added}건 (누적 {len(account_trip_state['seen'])}건)", flush=True)
        # 기록 **뒤에** 싣는다 -- 방금 닫힌 왕복이 같은 응답에 들어간다(원장 주기 300초는
        # 브라우저가 닫혀 있을 때의 보험일 뿐, 화면 요청도 이 함수를 지난다).
        if isinstance(payload, dict):
            payload["ledger"] = load_account_trip_rows()
        return payload

    async def keep_trip_ledger() -> None:
        """브라우저가 닫혀 있어도 돌아야 한다 -- 사라지는 쪽은 거래소의 7일 창이지 화면이 아니다.

        별도 태스크로 떼는 이유: 부르는 쪽(publish_dashboard_events)은 모든 탭의 SSE 유일
        공급원이라, 서명 요청 3~4개를 그 안에서 기다리면 화면 갱신이 함께 밀린다."""
        try:
            await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS, produce_account,
                             max_stale=STALE_GRACE_SECONDS, cache=binance_account_cache)
        except Exception as exc:  # noqa: BLE001 -- 키 만료·네트워크·시계드리프트 전부 여기로 온다
            print(f"account_round_trips cycle failed (will retry next cycle): {exc}", flush=True)

    async def publish_dashboard_events(app: web.Application) -> None:
        nonlocal latest_event_state, latest_event_tickers
        last_state_etag = ""
        timeout = ClientTimeout(total=2)
        async with ClientSession(timeout=timeout) as session:
            while True:
                started = time.monotonic()
                try:
                    state_payload, state_etag = dashboard_state_payload()
                    # 2026-09-21 모델지표 이력 샘플러 제거 -- 수급 흐름/리테일/청산
                    # 캐스케이드 칩 전용이었고 세 칩이 다 내려갔다(소비자 0).
                    if started - account_trip_state["last_at"] >= ACCOUNT_TRIP_RECORD_SECONDS:
                        account_trip_state["last_at"] = started
                        running = refresh_tasks.get("account_trips")
                        if running is None or running.done():
                            # refresh_tasks 에 넣어 두면 stop_http_session 이 세션을 닫기 전에
                            # 이 태스크까지 취소해 준다(따로 종료 코드를 만들지 않는다).
                            refresh_tasks["account_trips"] = asyncio.create_task(keep_trip_ledger())
                    ticker_rows = await asyncio.gather(
                        *(fetch_market_ticker(session, asset, symbol)
                          for asset, symbol in DASHBOARD_TICKER_SYMBOLS.items())
                    )
                    latest_event_tickers = {
                        asset: ticker for asset, ticker in ticker_rows if ticker is not None
                    }
                    state_changed = state_etag != last_state_etag
                    if state_changed:
                        latest_event_state = sse_state_view(state_payload)
                        last_state_etag = state_etag
                    payload = {
                        "state": latest_event_state if state_changed else None,
                        "tickers": latest_event_tickers,
                        # 켜진 코인 목록을 **서버가 알려준다**. 클라가 같은 목록을 따로 들고
                        # 있으면 둘이 어긋나는 날이 온다(SUSTAIN_BARS_OVERRIDE 류의 반복 교훈).
                        "assets": DASHBOARD_ASSETS,
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

    def _asset_buster() -> str:
        """app.js / styles.css 의 **내용**에서 뽑은 캐시 버스터.

        🔴손으로 적는 날짜 슬러그는 두 번 터졌다(2026-09-13, 2026-09-16). 앞단에 cloudflared
        터널이 있어 no-cache 만으로는 부족한데, 같은 날 두 번째 배포는 슬러그가 그대로라
        엣지 캐시를 못 뚫는다. 사람이 안 잊는 방법은 «안 적는 것»이다 -- 파일이 바뀌면
        버스터가 저절로 바뀐다. mtime 이 아니라 내용 해시라 재배포로 mtime 만 변해도 안 흔들린다.
        """
        h = hashlib.sha256()
        for name in ("app.js", "styles.css"):
            try:
                h.update((DASHBOARD_DIR / name).read_bytes())
            except OSError:
                h.update(name.encode())          # 파일이 없어도 페이지는 떠야 한다
        return h.hexdigest()[:12]

    def _hide_off_assets(html: str) -> str:
        """꺼진 코인 탭을 **처음부터** hidden 으로 내보낸다.

        🔴전에는 HTML 이 5 개를 다 보이게 싣고 SSE 가 도착한 뒤에야 JS 가 숨겼다 --
        새로고침할 때마다 «ETH BTC SOL XRP HYPE» 가 번쩍였다가 ETH 만 남았다(사용자 리포트).
        서버는 DASHBOARD_ASSETS 를 이미 알고 이 함수에서 index 를 고쳐 쓰고 있으므로
        여기서 붙이면 깜빡임이 원천 소멸한다. 클라이언트의 fail-open(목록을 못 받으면 전부
        보인다)은 그대로 둔다 -- 이건 «처음 그림»만 맞추는 것이다.
        """
        def mark(m: "re.Match[str]") -> str:
            tag, asset = m.group(0), m.group(1)
            if asset in DASHBOARD_ASSETS or " hidden" in tag:
                return tag
            return tag[:-1] + " hidden>"
        return re.sub(r'<button[^>]*\bdata-asset="([a-z]+)"[^>]*>', mark, html)

    async def dashboard_index(_: web.Request) -> web.Response:
        html = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
        html = re.sub(r'(app\.js|styles\.css)\?v=[A-Za-z0-9._-]+',
                      lambda m: f"{m.group(1)}?v={_asset_buster()}", html)
        html = _hide_off_assets(html)
        response = web.Response(text=html, content_type="text/html")
        response.enable_compression()
        return no_cache(response)

    async def pwa_asset(request: web.Request) -> web.FileResponse:
        """sw.js / manifest를 add_static이 아니라 no-cache로 직접 서빙한다.

        2026-09-01에 이 대시보드는 CSS를 고쳤는데도 화면이 안 바뀌는 사고를 겪었고 원인은
        Cloudflare 엣지 캐시였다(eth_dashboard_low_atr_warning_overflow_fix_20260901). 서비스
        워커가 같은 일을 당하면 훨씬 고약하다 -- 낡은 sw.js는 브라우저에 등록된 채로 남아
        알림 동작을 계속 지배하고, app.js처럼 쿼리스트링 캐시버스터를 붙일 수도 없다
        (등록 URL이 바뀌면 브라우저는 다른 워커로 취급한다).
        """
        name = request.match_info["name"]
        return no_cache(web.FileResponse(DASHBOARD_DIR / name))

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
        # 접속 즉시 보내는 스냅샷에도 코인 목록을 넣는다 -- 빠뜨리면 다음 주기(2.5초)까지
        # 클라가 목록을 모르고 꺼진 코인 탭이 잠깐 보였다가 사라진다(깜빡임).
        initial_payload = {"state": latest_event_state, "tickers": latest_event_tickers,
                           "assets": DASHBOARD_ASSETS}
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
        return web.json_response({"asset": asset, "candles": candles}, headers=NOCACHE)

    async def api_footprint(request: web.Request) -> web.Response:
        """가격레벨별 매수/매도 체결량. 레벨은
        [가격, 매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도] 7칸 배열이다
        -- 키 이름을 반복해 싣지 않으려는 것(12봉 x 수십 레벨을 2초마다 보낸다).
        고래·리테일은 매수/매도의 **부분집합**이고, 중형은 셋을 빼서 얻는다."""
        want = footprint_window_bars(request)
        recent = sorted(footprint_state["bars"].items())[-want:]
        agg_bars = footprint_state["agg_bars"]
        # ── `?since=<봉시각>` 증분 (2026-09-20) ──────────────────────────────
        # 400ms 폴링인데 48봉을 통째로 보내고 있었다. 실측: 400ms 간격 19번 중 내용이 실제로
        # 바뀐 건 12번이고, 바뀌는 건 **거의 언제나 맨 오른쪽 봉 하나**다(닫힌 봉은 5분에 한 번).
        # 화면 실사용 9.8KB/s 중 대부분이 안 바뀐 47봉을 다시 보내는 값이었다.
        # 🔴«현재 봉 하나»로는 부족하다. 봉 경계에서 늦게 도착한 체결이 **직전 봉**에 들어가므로
        #   클라가 since = 최신봉 - 1봉 으로 물어 둘을 받는다. 그래서 여기 비교는 `>=` 다.
        # 🔴백필 중에는 **과거 봉도 바뀐다**(REST 가 몇 분에 걸쳐 메운다). 그 구간의 계약이
        #   바로 `ready`(= 공백 없음)이므로, ready 가 아니면 증분을 주지 않고 전량을 준다.
        #   재연결로 ready 가 다시 False 가 되면 자동으로 전량 재동기화된다.
        try:
            since = int(request.query.get("since", "0"))
        except (TypeError, ValueError):
            since = 0
        full = since <= 0 or not footprint_state["ready"]
        sent = recent if full else [(b, c) for b, c in recent if b >= since]
        return web.json_response({
            "symbol": FOOTPRINT_SYMBOL,
            "bucket": FOOTPRINT_BUCKET,
            "barSeconds": FOOTPRINT_BAR_SECONDS,
            "barsExpected": want,
            "ready": bool(footprint_state["ready"]),
            # 🔴클라가 «이게 전량인가 조각인가»를 payload 에서 알아야 한다. 요청 쿼리로
            #   추측하면 서버가 ready=False 라 전량으로 되돌린 경우를 놓친다.
            "full": full,
            # 🔴`updated`(마지막 체결 시각)를 **뺐다**. 읽는 곳이 없는데(화면·시험 전수 확인)
            #   체결마다 바뀌어서 **ETag 를 매번 깨뜨리고 있었다** -- 봉 내용이 그대로여도
            #   304 가 안 나갔다. 빼고 나면 조용한 구간의 폴링이 본문 0B 로 끝난다
            #   (실측: 조건부 요청 20회 중 이미 11회가 304 였고, 그 비율이 더 올라간다).
            #   서버 내부의 footprint_state["updated"]는 스냅샷 저장 주기에 계속 쓴다.
            # 화면이 「고래 ≥$100k」를 적는 데 쓴다. 경계를 화면에 안 적으면 「고래」가
            # 무슨 뜻인지 보는 사람이 알 방법이 없다.
            "retailMaxUsd": RETAIL_MAX_USD,
            "whaleMinUsd": WHALE_MIN_USD,
            "bars": [
                {"time": bar,
                 # aggTrades 로 메운 봉은 체결이 **묶여** 있어 고래가 과장된다.
                 "agg": bar in agg_bars,
                 "levels": [[round(k * FOOTPRINT_BUCKET, 2)] + [round(x, 3) for x in v]
                            for k, v in sorted(cells.items())]}
                for bar, cells in sent
            ],
        }, headers=NOCACHE)

    async def api_supply_1s(request: web.Request) -> web.Response:
        """최근 5분을 1초 해상도로. `?since=<초>` 면 그 뒤에 **새로 생긴 초만** 보낸다.

        매초 폴링이라 전량을 매번 보내면 안 된다 -- 300초 x 7숫자면 회당 ~18KB 이고, 1초
        주기면 시간당 60MB 가 넘는다. 증분이면 보통 한두 줄(~100B)이다. 첫 요청만 전량이다.
        ⚠️**진행 중인 초는 안 보낸다**. 보내면 그 초가 자라는 동안 클라가 이미 «받은 초»로
          알고 건너뛰어, 반쪽짜리로 굳는다(풋프린트 스냅샷에서 겪은 그 실패와 같은 모양).
        """
        by_sec = footprint_state["sec"]
        try:
            since = int(request.query.get("since", "0"))
        except ValueError:
            since = 0
        # OI 는 별도 `sinceOi` 로 증분한다 -- 체결 초와 갱신 시점이 다르므로(3~7초) 같은 커서를
        # 공유하면 체결 초가 앞서갈 때 그 사이의 OI 점이 통째로 건너뛰어진다.
        try:
            since_oi = int(request.query.get("sinceOi", "0"))
        except ValueError:
            since_oi = 0
        # 2026-09-22 청산도 같은 payload 로 (사용자 요청). **새 스트림을 안 만든다** --
        # collect_force_orders 가 이미 @forceOrder 를 받아 liq_events(maxlen=5000)에 쌓고
        # 있고, 분당 몇 건이라 5분 창은 넉넉히 덮인다.
        # ⭐USD 가 아니라 **수량**을 보낸다. 이 차트의 다른 선(고래·리테일·OI)이 전부
        #   ETH 단위라 수량이어야 같은 축에 그대로 얹힌다. USD 였으면 축이 하나 더 필요하다.
        try:
            since_liq = int(request.query.get("sinceLiq", "0"))
        except ValueError:
            since_liq = 0
        oi_floor = max(since_oi, (max(oi_1s) if oi_1s else 0) - SUPPLY_1S_SECONDS)
        # [초, 미결제약정]. 증분은 클라가 뺀다(창 시작을 0으로 두는 누적선이라 절대값이 필요).
        oi_rows = [[s, oi_1s[s]] for s in sorted(oi_1s) if s > oi_floor]
        if not by_sec:
            return web.json_response({"symbol": FOOTPRINT_SYMBOL, "seconds": [], "now": 0,
                                      "oi": oi_rows, "liq": [],
                                      "retailMaxUsd": RETAIL_MAX_USD,
                                      "whaleMinUsd": WHALE_MIN_USD}, headers=NOCACHE)
        newest = max(by_sec)
        # 🔴**진행 중인 초는 안 보낸다** -- seconds 와 같은 규칙이다. 보내면 그 초가 자라는
        #   동안 클라가 «받은 초»로 알고 건너뛰어 반쪽으로 굳는다.
        # 🔴deque 스냅샷을 뜬다. 같은 이벤트 루프라 순회 중 append 는 안 오지만, 이 함수가
        #   길어질 때 그 불변식에 기대고 싶지 않다.
        liq_floor = max(since_liq, newest - SUPPLY_1S_SECONDS)
        liq_by_sec: dict[int, list[float]] = {}
        for e in list(liq_events):
            s = int(e.get("ts_ms") or 0) // 1000
            if not (liq_floor < s < newest):
                continue
            cell = liq_by_sec.setdefault(s, [0.0, 0.0, 0.0, 0.0])
            long_ = e.get("side") == "long"
            cell[0 if long_ else 1] += float(e.get("qty") or 0.0)
            # 2026-09-22 금액도 같이 보낸다(사용자 «단위나 금액으로 맞춰줘»). 화면의 5분봉
            #   청산은 USD 라 1초 쪽만 ETH 면 같은 카드에서 두 단위가 섞인다.
            # ⭐수량 x «그 초의 마지막 체결가» 로 클라가 곱하지 않는다 -- 이벤트마다 제
            #   체결가(ap)로 이미 계산된 값이 여기 있고, 캐스케이드에서는 그 둘이 벌어진다.
            cell[2 if long_ else 3] += float(e.get("usd") or 0.0)
        # [초, 롱수량, 숏수량, 롱USD, 숏USD]. 롱 청산 = 시장에 강제 SELL, 숏 청산 = 강제 BUY.
        liq_rows = [[s, round(v[0], 3), round(v[1], 3), round(v[2]), round(v[3])]
                    for s, v in sorted(liq_by_sec.items())]
        floor = max(since, newest - SUPPLY_1S_SECONDS)
        return web.json_response({
            "symbol": FOOTPRINT_SYMBOL,
            "now": newest,
            "retailMaxUsd": RETAIL_MAX_USD,
            "whaleMinUsd": WHALE_MIN_USD,
            "oi": oi_rows,
            "liq": liq_rows,
            # [초, 리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]
            "seconds": [[s] + [round(x, 3) for x in by_sec[s]]
                        for s in sorted(by_sec) if floor < s < newest],
        }, headers=NOCACHE)

    async def api_oi_5m(request: web.Request) -> web.Response:
        """OI 5분 누적(신규 계약). duckdb 를 읽으므로 to_thread 로 뺀다(이벤트 루프 블로킹 방지)."""
        try:
            bars = max(1, min(288, int(request.query.get("bars", OI_5M_WINDOW_BARS))))
        except ValueError:
            bars = OI_5M_WINDOW_BARS
        # 🔴이 저장소에서 **매 요청 duckdb 를 여는 유일한 엔드포인트**였다(실측 12~15ms,
        #   따뜻한 다른 엔드포인트가 0.5~4ms). 파일 락을 잡으므로 10초마다 도는 쓰기
        #   (oi_1s_persist)와 탭 수만큼 부딪힌다. TTL 은 클라 폴링(OI_5M_POLL_MS=15초)의
        #   3분의 1이라 **단일 탭에서는 늘 미스**다 -- 신선도는 그대로고 탭이 늘어도 duckdb
        #   열기 횟수만 5초당 한 번으로 묶인다. 값 자체도 5분봉이라 5초는 해상도 아래다.
        buckets = await swr_cached(f"oi_5m:{bars}", 5.0,
                                   lambda: asyncio.to_thread(oi_5m_buckets, bars))
        # 현재 OI 는 링에서 바로 준다 -- duckdb 는 최대 OI_1S_FLUSH_SECONDS 만큼 뒤처져 있다.
        now_oi = oi_1s[max(oi_1s)] if oi_1s else (buckets[-1][2] if buckets else 0.0)
        return web.json_response({
            "symbol": FOOTPRINT_SYMBOL,
            "barSeconds": OI_5M_BAR_SECONDS,
            "openInterest": round(float(now_oi), 3),
            # [봉시각, 신규계약, 봉 끝 OI, 스냅샷 수, 공백여부]
            "bars": buckets,
        }, headers=NOCACHE)

    async def api_supply_profile(request: web.Request) -> web.Response:
        """가격축 수급 프로파일 -- 창 전체를 가격빈으로 접은 것. 시간 축이 없다.

        같은 봉 링(최대 24시간)에서 만든다. 별도 수집기를 두지 않는 이유는 원천이 같아서다 --
        하나를 더 두면 둘이 어긋날 때 어느 쪽이 맞는지 알 방법이 없다."""
        # 2026-09-19 창을 풋프린트와 **같은 토글**이 정한다(사용자 요청). 그전에는 링에
        # 쌓인 것을 전부(최대 24시간) 접었는데, 아래 풋프린트는 1시간이라 위아래 두 그림이
        # 다른 구간을 말하고 있었다 -- 한 카드 안에서 그건 읽는 사람을 속이는 것이다.
        want = footprint_window_bars(request)
        recent = sorted(footprint_state["bars"].items())[-want:]
        merged: dict[int, list[float]] = {}
        for _, cells in recent:
            for k, v in cells.items():
                row = merged.setdefault(k, [0.0] * 6)
                for i in range(6):
                    row[i] += v[i]
        span = len(recent) * FOOTPRINT_BAR_SECONDS
        return web.json_response({
            "symbol": FOOTPRINT_SYMBOL,
            "bucket": FOOTPRINT_BUCKET,
            "barCount": len(recent),
            "spanSeconds": span,
            "spanMaxSeconds": want * FOOTPRINT_BAR_SECONDS,
            "retailMaxUsd": RETAIL_MAX_USD,
            "whaleMinUsd": WHALE_MIN_USD,
            # 창 안의 봉만 센다 -- 링 전체를 세면 1h 를 보는데 24h 치 경고가 뜬다.
            "aggBars": sum(1 for bar, _ in recent if bar in footprint_state["agg_bars"]),
            "levels": [[round(k * FOOTPRINT_BUCKET, 2)] + [round(x, 3) for x in v]
                       for k, v in sorted(merged.items())],
        }, headers=NOCACHE)

    # 🔴이 응답은 **페이지 로드를 막는다** -- app.js 가 seedModelIndicatorHistory() 를 await
    #   한 뒤에야 SSE 를 연다. 그런데 48샘플의 microstructure/tail_risk 를 통째로 실어 보내
    #   194KB(gzip 27KB)였다: 이 저장소에서 **가장 큰 응답**이고, 그 중 클라이언트가 읽는 것은
    #   classifyIndicators() 가 쓰는 **다섯 필드뿐**이다(나머지는 파싱되고 버려진다).
    #   버퍼/파일은 그대로 두고(나중에 다른 필드를 쓸 수 있게) **나갈 때만** 자른다.
    MIH_MICRO_KEYS = ("nif_whale", "nif_retail")
    MIH_TAIL_KEYS = ("hawkes_active", "z_long", "z_short")

    def _mih_slim(sample: dict) -> dict:
        micro = sample.get("microstructure") or {}
        tail = sample.get("tail_risk") or {}
        return {
            "sampled_at": sample.get("sampled_at"),
            "microstructure": {k: micro[k] for k in MIH_MICRO_KEYS if k in micro},
            "tail_risk": {k: tail[k] for k in MIH_TAIL_KEYS if k in tail},
        }

    async def api_liquidation_5m_history(request: web.Request) -> web.Response:
        """봉별 청산 금액 시계열 -- 청산맵 캔들 위에 얹는다(2026-09-11 사용자 요청).
        게이지(/api/liquidation-5m-signal)는 현재 봉 하나만 주므로 지나간 봉은 여기서 온다.
        duckdb 를 읽으므로 to_thread 로 뺀다(이벤트 루프 블로킹 방지)."""
        asset = _query_coin_asset(request)
        payload = await swr_cached(
            f"liq5m_hist_{asset}", 30.0,
            lambda: asyncio.to_thread(compute_liquidation_5m_history, asset, 96),
            max_stale=STALE_GRACE_SECONDS,
        )
        return web.json_response(payload, headers=NOCACHE)

    async def api_position_sizing(request: web.Request) -> web.Response:
        payload = await swr_cached(
            "position_sizing", 30.0, lambda: asyncio.to_thread(position_sizing_payload),
            max_stale=STALE_GRACE_SECONDS,
        )
        return web.json_response(payload, headers=NOCACHE)

    async def api_breakout_detector(request: web.Request) -> web.Response:
        return web.json_response(await load_breakout_detector(),
                                 headers=NOCACHE)

    def _heatmap_read(symbol: str, cols: int, agg: int, rows_only: bool = False) -> dict[str, Any]:
        """래스터 창을 읽어 화면이 바로 그릴 수 있는 모양으로 낸다.

        🔴포맷을 여기서 다시 파싱하지 않는다 -- 수집기의 `read_window()` 가 seek·절대
          가격축 정렬·부호보존 다운샘플(|값| 최대. 평균은 큰 벽 하나를 뭉갠다)까지 한다.
          헤더가 ver=2 로 오르면 그쪽만 고치면 된다.
        qty 부호가 방향이다: + 매수호가(bid), - 매도호가(ask).
        mid=NaN 인 열은 그 초의 북이 무효다 -- 렌더러가 회색으로 그린다. **보간 금지**
          (없는 유동성을 그리는 것이다)."""
        from scripts.live_orderflow_raster_collector_20260914 import read_window  # noqa: PLC0415
        w = read_window(symbol, int(time.time() * 1000), cols, agg)
        qty, mid = w.pop("qty"), w.pop("mid")
        # ⭐int8 로 양자화해 보낸다. 화면은 |q|/scale 만 쓰므로(색 농도) 원값이 필요없고,
        #   f32 면 300x259 에 base64 405KB 인데 int8 은 104KB 다(풋프린트가 12KB).
        #   scale 은 **분위**다 -- 최대값으로 나누면 큰 벽 하나가 나머지를 다 씻어낸다.
        nz = np.abs(qty[qty != 0])
        scale = float(np.percentile(nz, 97)) if nz.size else 1.0
        if not (scale > 0):
            scale = 1.0
        out = {**w, "scale": scale, **_heatmap_rows(w, qty, mid)}
        if rows_only:
            # 2026-09-19 히트맵 «그림»을 걷어낸 뒤로 화면은 행 집계와 요약만 쓴다.
            # 이미지 배열을 계속 보내면 서버 실측 **102KB** 가 매 폴링 그냥 버려진다.
            return out
        q8 = np.rint(np.clip(qty / scale, -1.0, 1.0) * 127.0)   # rint: 절단하면 오차가 2배다
        return {**out,
                "qty_i8": base64.b64encode(np.ascontiguousarray(q8, dtype=np.int8)).decode(),
                "mid_b64": base64.b64encode(np.ascontiguousarray(mid, dtype="<f4")).decode()}

    def _heatmap_rows(w: dict, qty: "np.ndarray", mid: "np.ndarray") -> dict[str, Any]:
        """그림에서 **뽑은 수치**. 히트맵 이미지로는 구별 안 되는 것을 숫자로 가른다.

        ⭐2026-09-20 행마다 다섯 값을 낸다(`row_stats`, 수집기에 있다 -- 래스터 축소는
          거기가 집이고 `--selftest` 가 붙어 있다). 그전에는 `pers` 하나로 막대를 그렸는데,
          min 은 가능한 추정량 중 **가장 보수적**이라 「한 번도 안 빠진 양」밖에 대답하지
          않는다. 실측 600초 창에서 그 값(103,531)은 `refill`(1,029,281)의 **1/10** 이다.
          화면은 이제 길이=`inst`, 농도=`refill/peak`(재깔림)로 갈라 그린다.
        🔴`pers`(창 내내 한 번도 안 빠진 양)도 그대로 낸다. 실측(600초): 순간 잔량의 **36%가
          창을 못 버틴다**. 겉보기 크기가 비슷한 두 벽이 지속률 100%(-0.75%, 8073->8056) 대
          0%(+2.24%, 5461->0) 로 갈리는데 이미지에서는 둘 다 같은 진한 띠다.
        🔴`offtouch_leave_share` 는 «취소율»이 **아니다**. 「터치에서 떨어진 호가가 체결 없이
          빠지는 비율」이다 -- 아래 조인의 두 통제(창 커버·사이드 유지)가 터치 구간을 구조적으로
          제외하기 때문이다. 그 안에는 취소·리프라이싱이 섞여 있고 이 피드로는 못 가른다.
          바이낸스 선물 WS 는 레벨별 총량만 주고 **주문 ID 가 없어**(depthUpdate 가
          [가격, 새 총량]) 개별 «건»은 원리적으로 복원 불가다.
        🔴`obi` 는 **밴드가 값을 정한다**(실측 ±0.1% +0.504 / ±2% +0.071, 7배). 밴드를 같이 낸다.
        🔴이 수치들은 «지지·저항»이 아니다 -- 2026-09-20 에 **실제로 쟀고 없었다**.
          5.8일 래스터 71,293건: 벽에 닿은 뒤 반등률 0.509(동전) · 크기 사분위
          0.502/0.510/0.520/0.503 로 순서조차 없음 · 같은 크기 안에서 지속률 상위−하위
          +0.001(섞기 귀무 z=0.22). 판정폭 $1.5→$5 · 지평 5→15분에서도 같다(z=−0.45).
          기록: docs/experiments/eth_orderbook_wall_support_resistance_20260920.md
          (앞선 «판정가능 2셀·부호반전» 메모는 GEX 스트라이크 벽 건이라 별개다 --
           docs/experiments/eth_gex_strike_walls_vs_liqmap_20260919).
        """
        ok = np.isfinite(mid)
        if not ok.any():
            return {"rows": None, "summary": None}
        from scripts.live_orderflow_raster_collector_20260914 import row_stats  # noqa: PLC0415
        a = np.abs(qty[ok])
        st = row_stats(a, int(w["dt_s"]))
        inst, pers = st["inst"], st["pers"]
        spot = float(np.nanmedian(mid[ok]))
        price = (w["bin_lo"] + np.arange(w["n_bins"])) * w["bin_size"]
        dist = (price - spot) / spot * 100.0 if spot else np.zeros_like(price)

        # ── 체결 vs 이탈: 풋프린트로 가른다 ──────────────────────────────────
        # 잔량 감소 = 체결 + 이탈(취소·리프라이싱·창 밖). 풋프린트가 **같은 $0.5 버킷**으로
        # 가격행별 체결량을 주므로 조인이 공짜다(같은 프로세스 메모리, HTTP 왕복 0).
        # 🔴방향: **테이커 매도가 비드를, 테이커 매수가 아스크를** 먹는다.
        #   풋프린트 셀은 [buy, sell, ...] (footprint_add 의 side = 1 if sell else 0).
        #
        # 🔴통제 둘을 안 걸면 숫자가 통째로 틀린다(2026-09-19 실측으로 잡았다):
        #   ① **창이 mid 를 따라 움직인다.** 한 행이 덮는 건 N_BINS x bin_size(±$60)뿐이라,
        #      mid 가 움직이면 가장자리 빈이 «커버 밖»이 되어 0 으로 보인다 -- 취소가 아니다.
        #   ② **가격이 빈을 통과하면 사이드가 뒤집힌다.** 비드였던 빈이 아스크가 되면
        #      비드 잔량이 0 이 되는데 그것도 취소가 아니다.
        #   통제 전 «취소 89.4%» 였고, 방향 대조군이 5,087 vs 4,909 로 **구분이 안 됐다**.
        #   통제 후 방향이 선다(비드 14.9% 체결설명 vs 아스크 0.4% -- mid 가 내려간 창이라 맞다).
        #
        # ⭐그래서 이 값은 «사라진 유동성의 N%가 취소»가 **아니다**. 「내내 한쪽」 필터가
        #   터치 구간을 구조적으로 뺀다 -- **터치에서 떨어진 호가가 체결 없이 빠지는 비율**이다.
        #   먼 벽을 장벽으로 믿지 말라는 뜻이고, 그게 화면이 말해야 하는 것이다.
        # 🔴봉 단위(300초) 조인이라 창 가장자리가 최대 한 봉 어긋난다. 현재 봉이 부분이라
        #   체결이 과대계상되고 그만큼 **이탈이 과소**로 나온다 -- 안전한 쪽이다.
        from scripts.live_orderflow_raster_collector_20260914 import N_BINS  # noqa: PLC0415
        m0, m1 = float(mid[ok][0]), float(mid[ok][-1])
        cov_half = 0.4 * N_BINS * w["bin_size"]          # 한 행 커버(±$60)의 보수적 안쪽
        cov = (np.abs(price - m0) < cov_half) & (np.abs(price - m1) < cov_half)
        same_bid = cov & (price < m0) & (price < m1)
        same_ask = cov & (price > m0) & (price > m1)

        bid_now, ask_now = np.clip(qty[ok], 0, None), np.clip(-qty[ok], 0, None)
        bid_drop = np.clip(bid_now[0] - bid_now[-1], 0, None)
        ask_drop = np.clip(ask_now[0] - ask_now[-1], 0, None)
        fill_bid = np.zeros(w["n_bins"])   # 비드를 먹은 양 = 테이커 매도
        fill_ask = np.zeros(w["n_bins"])
        fill_source = "footprint"
        try:
            t_start = w["t0_ms"] / 1000.0
            t_end = t_start + w["cols"] * w["dt_s"]
            for bar, cells in footprint_state["bars"].items():
                if bar + FOOTPRINT_BAR_SECONDS < t_start or bar > t_end:
                    continue
                for key, v in cells.items():
                    i = int(round(key * FOOTPRINT_BUCKET / w["bin_size"])) - w["bin_lo"]
                    if 0 <= i < w["n_bins"]:
                        fill_ask[i] += float(v[0])
                        fill_bid[i] += float(v[1])
        except Exception:  # noqa: BLE001 -- 풋프린트 웜업이면 못 가른다
            fill_source = None

        drop = float(bid_drop[same_bid].sum() + ask_drop[same_ask].sum())
        filled = float(np.minimum(bid_drop, fill_bid)[same_bid].sum()
                       + np.minimum(ask_drop, fill_ask)[same_ask].sum())
        # 이름이 곧 정의다 -- 「취소」가 아니라 「터치 밖 이탈」이다.
        offtouch_leave_share = (round(1.0 - filled / drop, 3)
                                if (fill_source and drop > 0) else None)
        tot_drop = float(np.clip(bid_now[0] - bid_now[-1], 0, None).sum()
                         + np.clip(ask_now[0] - ask_now[-1], 0, None).sum())
        band = np.abs(dist) <= 0.5
        b = float(np.clip(qty[ok][-1], 0, None)[band].sum())
        k = float(np.clip(-qty[ok][-1], 0, None)[band].sum())
        return {
            "rows": {"bin_lo": int(w["bin_lo"]), "bin_size": float(w["bin_size"]),
                     **{f"{k}_f4": base64.b64encode(
                         np.ascontiguousarray(v, "<f4")).decode() for k, v in st.items()}},
            "summary": {
                "spot": round(spot, 2),
                "persist_share": round(float(pers.sum() / max(inst.sum(), 1e-9)), 3),
                "offtouch_leave_share": offtouch_leave_share, "fill_source": fill_source,
                "offtouch_bins": int(same_bid.sum() + same_ask.sum()),
                "obi": round((b - k) / max(b + k, 1e-9), 3), "obi_band_pct": 0.5,
                "window_s": int(w["cols"] * w["dt_s"]),
            },
        }

    def _volsig_ref(symbol: str) -> dict[str, Any]:
        """«분위 자» -- 4시간을 60초마다 표집해 정렬해 둔다.

        🔴문턱을 절대값으로 박으면 안 된다(저장소 규칙: 확률 임계값은 분위로 선언한다).
          재깔림도 실현변동도 시간대·레짐에 따라 자릿수가 통째로 움직인다. 「지금이
          최근 4시간 중 몇 분위인가」만 화면이 읽을 수 있는 말이다.
        🔴두 분위의 «합»은 균등분포가 아니다. 그래서 합 자체의 분포도 같이 정렬해 둔다
          -- 이게 없으면 「70%」가 실제로는 상위 15% 같은 값이 된다.
        """
        from scripts.live_orderflow_raster_collector_20260914 import (  # noqa: PLC0415
            read_window, vol_scores)
        w = read_window(symbol, int(time.time() * 1000), 4 * 3600, 1)
        past, refill = vol_scores(w, range(600, 4 * 3600, 60))
        if past.size < 60:          # 워밍업 중이면 자가 없다 -- 없다고 말한다
            return {}
        sp, sr = np.sort(past), np.sort(refill)
        sc = np.sort(np.searchsorted(sp, past) / sp.size
                     + np.searchsorted(sr, refill) / sr.size)
        return {"past": sp, "refill": sr, "score": sc}

    def _volsig_read(symbol: str, ref: dict[str, Any]) -> dict[str, Any]:
        """지금 이 초의 «얼마나 흔들릴까» 분위. 재료도 같이 낸다.

        2026-09-20 에 후보를 전부 재고 **떨어뜨린 뒤** 남은 조합이다(`vol_scores` 의 표).
        🔴이름값 주의: 주역은 «직전 300초 실현변동»이고 호가(재깔림)는 증분이다. 그리고
          그 증분(상위20%/하위33% 1.63 -> 1.77배)은 CI [-0.014, +0.301] 로 0 을 포함한다.
          화면이 둘을 따로 보여야 사용자가 어느 쪽이 켜졌는지 안다.
        🔴mid 도 호가 데이터다(최우선 호가의 중간). 「가격 지표를 섞었다」가 아니다.
        """
        from scripts.live_orderflow_raster_collector_20260914 import (  # noqa: PLC0415
            read_window, vol_scores)
        if not ref:
            return {}
        w = read_window(symbol, int(time.time() * 1000), 660, 1)
        past, refill = vol_scores(w)
        if past.size == 0:
            return {}
        pp = float(np.searchsorted(ref["past"], past[0]) / ref["past"].size)
        rp = float(np.searchsorted(ref["refill"], refill[0]) / ref["refill"].size)
        sc = float(np.searchsorted(ref["score"], pp + rp) / ref["score"].size)
        return {"vol_pct": round(100 * sc), "vol_past_pct": round(100 * pp),
                "vol_refill_pct": round(100 * rp), "vol_ref_n": int(ref["score"].size)}

    def _approach_read(symbol: str) -> dict[str, Any]:
        """4시간 창으로 «가격이 다가왔을 때 이 가격대가 두꺼워졌나»를 낸다.

        🔴화면 창(15분)으로는 못 잰다 -- 같은 빈이 near 와 far 를 둘 다 겪어야 비교가
          성립하는데 자격 빈이 22개뿐이다(4시간이면 82개, 2026-09-20 실측).
        ⭐그래서 **따로 읽고 따로 캐시한다**. 5초 폴링마다 4시간을 읽을 수는 없다
          (실측 읽기 0.27초). agg=5 로 접어 받아 계산량을 5분의 1로 줄인다 -- 이 값은
          구간 평균의 비라 초 해상도가 필요 없다.
        🔴«스푸핑 판정»이 아니다. 지배적 방향이 오히려 «다가오면 커진다»(46%)이고
          «얇아진다»는 16% 꼬리다. 화면은 관측치로만 적는다.
        """
        from scripts.live_orderflow_raster_collector_20260914 import (  # noqa: PLC0415
            approach_ratio, read_window)
        w = read_window(symbol, int(time.time() * 1000), 2880, 5)
        a = approach_ratio(w)
        return {"approach_bin_lo": a["bin_lo"], "approach_bin_size": a["bin_size"],
                "approach_qualifying": a["qualifying"], "approach_window_s": 4 * 3600,
                "approach_f4": base64.b64encode(
                    np.ascontiguousarray(a["ratio"], "<f4")).decode()}

    async def api_flow_heatmap(request: web.Request) -> web.Response:
        symbol = (request.query.get("symbol") or "ethusdt").lower()
        if symbol not in HEATMAP_SYMBOLS:
            raise web.HTTPBadRequest(reason="unsupported_symbol")
        try:
            cols = max(10, min(HEATMAP_MAX_COLS, int(request.query.get("cols", "300"))))
            agg = max(1, min(60, int(request.query.get("agg", "3"))))
        except ValueError:
            raise web.HTTPBadRequest(reason="bad_cols_or_agg") from None
        loop = asyncio.get_running_loop()
        rows_only = request.query.get("mode") == "rows"
        # 2026-09-20 화면이 1초 주기로 받는다(사용자 요청). 실측 한 번 11~32ms(탭별)이라
        # 1Hz 면 코어 1.1~3.2% 인데, **탭 수만큼 곱해진다**. 1초 SWR 로 묶어 몇 명이 보든
        # 비용을 한 번으로 고정한다 -- 래스터가 1초에 한 행이라 그보다 촘촘한 캐시는 뜻이 없다.
        payload = await swr_cached(
            f"heatmap_{symbol}_{cols}_{agg}_{int(rows_only)}", 1.0,
            lambda: loop.run_in_executor(
                HEATMAP_EXECUTOR, functools.partial(_heatmap_read, symbol, cols, agg, rows_only)))
        # 접근행동은 창이 4시간이라 폴링마다 못 읽는다. 120초 캐시 + 낡은 값 먼저 준다.
        if rows_only and payload.get("rows"):
            try:
                payload["rows"].update(await swr_cached(
                    f"approach_{symbol}", 120.0,
                    lambda: loop.run_in_executor(HEATMAP_EXECUTOR, _approach_read, symbol),
                    max_stale=STALE_GRACE_SECONDS))
                # 변동 신호: 자(4시간)는 300초마다, 지금 값(660초)은 매 초. 실측 읽기
                # 0.028초 / 0.002초라 폴링에 얹어도 된다.
                ref = await swr_cached(
                    f"volsig_ref_{symbol}", 300.0,
                    lambda: loop.run_in_executor(HEATMAP_EXECUTOR, _volsig_ref, symbol),
                    max_stale=STALE_GRACE_SECONDS)
                if payload.get("summary"):
                    payload["summary"].update(await swr_cached(
                        f"volsig_{symbol}", 1.0,
                        lambda: loop.run_in_executor(
                            HEATMAP_EXECUTOR, _volsig_read, symbol, ref)))
            except Exception:  # noqa: BLE001, S110
                # 삼킨다 -- 이 값이 없으면 화면은 표식만 안 그리고 나머지는 그대로다.
                # 여기서 던지면 «장식 하나» 때문에 프로파일 전체가 빈다.
                pass
        return web.json_response(payload, headers=NOCACHE)

    async def api_gex(request: web.Request) -> web.Response:
        payload = await swr_cached(
            "gex", 60.0, lambda: asyncio.to_thread(gex_payload), max_stale=STALE_GRACE_SECONDS)
        return web.json_response(payload, headers=NOCACHE)

    async def api_chart_markers(request: web.Request) -> web.Response:
        payload = await load_chart_markers(request.query.get("asset", "eth"))
        return web.json_response(payload, headers=NOCACHE)

    def _query_coin_asset(request: web.Request) -> str:
        """Shared `?asset=` parsing for the 4 Snapshot-tab signals wired to multiple coins
        (2026-08-31) -- raises the same 400 shape as api_market_history()'s existing
        unsupported-asset check."""
        asset = request.query.get("asset", "eth").lower()
        if asset not in COIN_CONFIG:
            raise web.HTTPBadRequest(reason="unsupported_asset")
        return asset

    async def api_liquidation_5m_signal(request: web.Request) -> web.Response:
        payload = await load_liquidation_5m_signal(_query_coin_asset(request))
        return web.json_response(payload, headers=NOCACHE)

    async def api_liquidation_direction_signal(request: web.Request) -> web.Response:
        payload = await load_liquidation_direction_signal(_query_coin_asset(request))
        return web.json_response(payload, headers=NOCACHE)

    async def api_liquidation_map(request: web.Request) -> web.Response:
        try:
            payload = await load_liquidation_map(_query_coin_asset(request))
        except web.HTTPBadGateway:
            return web.json_response(
                {"error": "liquidation_map_upstream_error", "detail": "Binance klines fetch failed."},
                status=web.HTTPBadGateway.status_code,
                headers=NOCACHE,
            )
        return web.json_response(payload, headers=NOCACHE)

    async def api_regime_wide24(request: web.Request) -> web.Response:
        payload = await load_regime_wide24()
        return web.json_response(payload, headers=NOCACHE)

    async def api_regime_btc(request: web.Request) -> web.Response:
        payload = await load_regime_btc()
        return web.json_response(payload, headers=NOCACHE)


    async def load_coin_indicators(asset: str) -> dict[str, Any]:
        return await swr_cached(
            f"coin_indicator:{asset}", COIN_INDICATOR_CACHE_SECONDS,
            lambda: asyncio.to_thread(coin_indicators_payload, asset),
        )

    async def api_coin_indicators(request: web.Request) -> web.Response:
        payload = await load_coin_indicators(_query_coin_asset(request))
        return web.json_response(payload, headers=NOCACHE)

    async def api_regime_xrp(request: web.Request) -> web.Response:
        payload = await load_regime_xrp()
        return web.json_response(payload, headers=NOCACHE)

    async def api_macro_calendar(request: web.Request) -> web.Response:
        payload = await load_macro_calendar()
        return web.json_response(payload, headers=NOCACHE)

    async def api_liq_burst_state(request: web.Request) -> web.Response:
        # load_json_cached() keys off (mtime, size), not a timer -- so this serves the freshest
        # write tail_risk_interceptor.py has made (event-triggered, see its _write_liq_burst_state()
        # docstring) without needing its own cache TTL/lock here.
        payload = load_json_cached(LIQ_BURST_STATE_PATH)
        if not payload:
            return web.json_response({"available": False}, headers=NOCACHE)
        # tail_risk_interceptor.py's _write_liq_burst_state() never sets "available" itself (it
        # always writes on success) -- the frontend's renderLiqBurstAlert() checks payload.available
        # to distinguish this from the {"available": False} fallback above, so stamp it here.
        return web.json_response({**payload, "available": True}, headers=NOCACHE)

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
        return web.json_response(payload, headers=NOCACHE)

    # ---- Web Push (PWA 알림) --------------------------------------------------------
    # 판단 로직은 여기 없다 -- scripts/live_push_notifier_20260904.py 참고. 여기는 구독 수명주기만.
    async def api_push_config(request: web.Request) -> web.Response:
        """브라우저가 pushManager.subscribe()에 넘길 applicationServerKey(=VAPID 공개키)를 준다.
        키가 .env에 없으면 enabled=false로 답해서 프론트가 알림 버튼 자체를 숨기게 한다 --
        키 없이 subscribe()를 호출하면 브라우저가 던지는 예외는 원인을 알아보기 어렵다."""
        public_key = os.getenv("VAPID_PUBLIC_KEY", "")
        return web.json_response(
            {"enabled": bool(public_key and os.getenv("VAPID_PRIVATE_KEY")),
             "vapid_public_key": public_key,
             "subscriber_count": len(load_subscriptions())},
            headers=NOCACHE,
        )

    async def api_push_subscribe(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="invalid json")
        sub = body.get("subscription") or {}
        if not sub.get("endpoint") or not (sub.get("keys") or {}).get("p256dh"):
            raise web.HTTPBadRequest(text="subscription must carry endpoint and keys.p256dh")
        sid = add_subscription(sub, label=str(body.get("label", ""))[:80])
        return web.json_response({"ok": True, "id": sid}, headers=NOCACHE)

    async def api_push_unsubscribe(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text="invalid json")
        sub = body.get("subscription") or {}
        sid = body.get("id") or (subscription_id(sub) if sub.get("endpoint") else None)
        if not sid:
            raise web.HTTPBadRequest(text="id or subscription.endpoint required")
        return web.json_response({"ok": remove_subscription(sid)},
                                 headers=NOCACHE)

    async def api_push_devices(request: web.Request) -> web.Response:
        """등록된 구독 목록. endpoint 원문은 기기 식별 토큰이라 내보내지 않고, 프론트가 자기
        구독인지 대조할 수 있도록 해시 id만 준다(브라우저가 같은 해시를 계산할 수 없으므로
        endpoint의 꼬리 12자도 함께 준다 -- 그 정도면 대조는 되고 재사용은 안 된다)."""
        rows = []
        for sid, sub in load_subscriptions().items():
            endpoint = str(sub.get("endpoint", ""))
            rows.append({"id": sid, "label": sub.get("label") or "",
                         "subscribed_utc": sub.get("subscribed_utc"),
                         "endpoint_tail": endpoint[-12:]})
        rows.sort(key=lambda r: r.get("subscribed_utc") or "")
        return web.json_response({"devices": rows}, headers=NOCACHE)

    async def api_push_test(request: web.Request) -> web.Response:
        """구독 직후 '진짜로 뜨는가'를 확인하는 용도. 이게 없으면 사용자는 실제 신호가 날 때까지
        (조용한 장에서는 몇 시간) 설정이 됐는지 알 수 없다."""
        private = os.getenv("VAPID_PRIVATE_KEY", "")
        if not private:
            raise web.HTTPServiceUnavailable(text="VAPID_PRIVATE_KEY not configured")
        result = await broadcast(
            {"tier": "test", "title": "알림 설정 완료",
             "body": "이 알림이 보이면 정상입니다. 실제 신호가 나면 같은 방식으로 도착합니다.",
             "tag": "push-test", "url": "/dashboard/live/"},
            private_b64=private,
            subject=os.getenv("VAPID_SUBJECT", "mailto:kbj2060@gmail.com"),
            ttl=60,
        )
        return web.json_response(result, headers=NOCACHE)

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

    async def api_binance_account(request: web.Request) -> web.Response:
        """실계좌 잔고/포지션/왕복거래(진입·청산 시각). 키에 Futures 읽기 권한이 없으면
        ok=false + hint로 내려가고, 프런트는 그 문구를 그대로 보여준다."""
        payload = await swr_cached(
            "binance_account", BINANCE_ACCOUNT_CACHE_SECONDS, produce_account,
            cache=binance_account_cache,
            max_stale=STALE_GRACE_SECONDS,
        )
        return web.json_response(payload, headers=NOCACHE)

    def effective_cap(sizing: dict[str, Any], equity: float,
                      safe_mae_pct: float | None) -> tuple[float | None, str | None, dict]:
        """세 상한(원장·순자산·위험모델)의 **작은 쪽**과 어느 쪽이 묶었는지.

        🔴진입과 청산이 이 함수를 **같이** 쓴다(2026-09-13). 예전에는 진입만 세 상한을 쓰고
        청산의 «최소 청산 비율»은 정책상한 25배를 기준으로 계산해서, 20배 포지션에서 같은
        카드가 «최소 청산 0%»와 «예산 사다리 60%»를 나란히 띄웠다."""
        cap = dict(sizing.get("cap") or {})
        # 🔴상한 무시 스위치 -- assemble_entry_plan 의 인라인 블록과 **같은 규칙**이어야 한다.
        #   두 곳이 갈리면 «추천 지평은 옛 상한, 실제 주문은 새 상한»이 된다.
        if SIZING_CAP_OVERRIDE_X > 0 and equity > 0:
            return equity * SIZING_CAP_OVERRIDE_X, "override", cap
        cands = []
        if cap.get("available") and cap.get("cap_notional_usdt"):
            cands.append((float(cap["cap_notional_usdt"]), "ledger"))
        if equity > 0:
            cands.append((equity * SIZING_CAP_EQUITY_X, "equity"))
            if safe_mae_pct and safe_mae_pct > 0:
                cands.append((entry_notional(equity, safe_mae_pct)["total_notional"], "model"))
        if not cands:
            return None, None, cap
        # 🔴값만 비교한다 -- 튜플 min 은 값이 같을 때 이름 알파벳순으로 갈린다.
        notional, who = min(cands, key=lambda t: t[0])
        return notional, who, cap

    async def realized_vol_now(symbol: str) -> float | None:
        """청산 마감·기대 체결시간을 정하는 1분봉 실현변동성. 실패해도 주문을 막지 않는다 --
        None 이면 exit_deadline_sec 이 보수적으로 최대(120초)를 쓴다."""
        try:
            kl = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {"symbol": symbol, "interval": "1m", "limit": EXIT_VOL_WINDOW + 2},
                timeout=5.0, error_reason=None)
            if kl:
                # 마지막 봉은 **미완결**이라 버린다. 종가는 인덱스 4.
                return realized_vol_bpm([float(row[4]) for row in kl[:-1]])
        except Exception:  # noqa: BLE001 -- 변동성은 있으면 좋은 값이지 필수가 아니다
            pass
        return None

    async def assemble_entry_plan(side: str, fraction: float = 1.0,
                                  want_lev: int | None = None):
        """계획 조립은 **여기 한 곳뿐**이다 -- 미리보기와 실주문이 같은 입력·같은 함수를 지난다.
        두 곳에 복사해 두면 언젠가 한쪽만 고쳐져 «미리보기와 다른 주문»이 나간다.

        반환: (plan, cap, sizing, error). error 는 (본문, HTTP상태) 또는 None.
        계좌를 같이 읽는 이유 둘: (1) 상한을 **합산 포지션**에 걸어야 하고
        (2) 화면이 «현금 얼마·레버리지 몇 배·청산까지 몇 %»를 말할 수 있어야 한다."""
        symbol = MANUAL_EXEC_SYMBOL        # 2026-09-19 차트 심볼과 분리(위 상수 주석)
        try:
            sizing = await asyncio.to_thread(position_sizing_payload)
            if not sizing.get("available"):
                return None, {}, {}, ({"error": "sizing_unavailable",
                                       "detail": sizing.get("error")}, 503)
            account = await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS,
                                       produce_account, max_stale=STALE_GRACE_SECONDS)
            positions = [p for p in (account.get("positions") or []) if p.get("symbol") == symbol]
            # 헤지 모드라 롱·숏이 동시에 열린다. 위험 상쇄를 가정하지 않고 **절대값 합**으로 본다
            # -- 두 다리 다 증거금을 먹고, 둘 다 청산될 수 있다.
            existing = notional_sum(positions)
            # 🔴상한은 **계좌 전체**에 걸린다. 교차 마진이라 청산거리 = 순자산 / **총명목**이고,
            #   그 총명목에는 다른 심볼도 들어간다 -- 봇은 ETHUSDT 로, 수동 주문은 ETHUSDC 로
            #   나가므로(2026-09-19 MANUAL_EXEC_SYMBOL) 심볼 하나만 세면 봇 포지션이 열려 있는
            #   동안 상한이 그만큼 헐거워진다. 위 `existing`(이 심볼)은 아래 청산거리 투영에만
            #   쓴다 -- 그 계산은 심볼 안에서만 성립한다(liq_after = liq_before × existing/total).
            exposure = notional_sum(account.get("positions"))
            # 추가 진입 맥락은 **같은 방향**만 본다 -- 헤지 모드에서 반대 다리는 다른 결정이다.
            same = [p for p in positions if p.get("side") == side]
            same_unrealized = sum(float(p.get("unrealized_pnl") or 0.0) for p in same)
            equity = float((account.get("balance") or {}).get("margin") or 0.0)
            # 포지션이 없으면 positions 가 비어 있다 -- 그때도 설정 레버리지는 알아야
            # «증거금 얼마»를 말할 수 있다(교차 마진에서 증거금 = 명목/레버리지).
            leverage = (max((float(p.get("leverage") or 0.0) for p in positions), default=0.0)
                        or float((account.get("leverage_by_symbol") or {}).get(symbol) or 0.0))
            book = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                                            {"symbol": symbol}, error_reason="book_ticker_failed")
            filters = await load_filters(binance_session(), symbol)
            cap = dict(sizing.get("cap") or {})
            # 두 상한의 **작은 쪽**. 순자산 연동은 원장이 자라도 안 변하고 자산이 줄면 같이
            # 줄어든다 -- 소급 초과가 생기지 않는다. 어느 쪽이 묶었는지 화면에 남긴다.
            cap_ledger = cap.get("cap_notional_usdt") if cap.get("available") else None
            cap_equity = equity * SIZING_CAP_EQUITY_X if equity > 0 else None
            # 🔴거래소 레버리지의 기준은 **지평과 무관한** 정책 천장이다 -- 모델 상한을 빼고
            # 원장·순자산만 본다. 모델 상한을 넣으면 보유시간 선택마다 설정이 움직인다
            # (2026-09-13 라이브에서 실제로 그랬다: 설정이 4.49 를 따라 8배로 내려앉았다).
            # ⭐이 값이 **지평 선택의 순환을 끊는다**: 지평에는 상한이, 상한에는 지평이 필요한데
            #   정책 천장은 지평과 무관하므로 먼저 정해진다(planning_hold 주석 참조).
            policy_only = [v for v in (cap_ledger, cap_equity) if v]
            policy_cap_x = (min(policy_only) / equity) if policy_only and equity > 0 else None
            # 🔴상한을 계산할 지평 = 처방이 고를 지평(2026-09-13 감사). 같은 방향 포지션이
            # 있으면 그 포지션의 **남은 시간**이 이긴다 -- 추가한다고 시계가 새로 생기지 않는다
            # (2026-09-13 사용자 질문: "물타기하면 시간이 늘어나나?").
            # 🔴그 시계의 **길이**도 처방 지평이다(2026-09-14). 240분 상수를 쓰면 24시간
            # 처방으로 들어간 포지션의 추가분이 4시간 예산으로 계산된다.
            plan_hold = planning_hold(sizing, side, policy_cap_x)
            hold_min = (remaining_hold(same[0].get("entry_at"), fixed=plan_hold)
                        if same else plan_hold)
            # 2026-09-13 세 번째 상한: 보유시간 조건부 위험 모델. 있으면 같이 경쟁시킨다.
            risk = risk_sizing(sizing, hold_min, side)
            cap_model = None
            if risk.get("available") and equity > 0:
                e = entry_notional(equity, risk["safe_mae_pct"], existing_notional=exposure)
                cap_model = e["total_notional"]
                risk.update(leverage=round(e["leverage"], 2), binding=e["binding"],
                            survival_x=round(e["survival_x"], 2),
                            growth_x=round(e["growth_x"], 2) if e["growth_x"] else None)
            binding = [(v, k) for v, k in ((cap_ledger, "ledger"), (cap_equity, "equity"),
                                           (cap_model, "model")) if v]
            cap_notional = min(v for v, _ in binding) if binding else None
            # 🔴상한 무시 스위치. 위 SIZING_CAP_OVERRIDE_X 주석이 계약이다.
            overridden = SIZING_CAP_OVERRIDE_X > 0 and equity > 0
            if overridden:
                cap_notional = equity * SIZING_CAP_OVERRIDE_X
            # 화면이 «무엇이 묶었나»를 말할 때 쓰는 값. policy_leverage 의 binding 은
            # 순자산·원장 상한을 **모르므로** 그대로 보여주면 «정책상한 25배»라고 거짓말한다.
            # 🔴키로 **값만** 비교한다 -- 튜플 min 은 값이 같을 때 이름 알파벳순으로 갈린다.
            #   (아래 cap.update 도 같은 값을 쓴다. 옛 판은 여기만 고치고 그쪽은 `min(binding)`
            #    이라 동점에서 두 줄이 서로 다른 이름을 말할 수 있었다.)
            who = ("override" if overridden
                   else min(binding, key=lambda t: t[0])[1] if binding else None)
            if risk.get("available") and equity > 0 and cap_notional:
                risk["effective_x"] = round(cap_notional / equity, 2)
                risk["applied_binding"] = who
            if cap_notional is not None:
                cap.update(available=True, cap_notional_usdt=round(cap_notional, 2),
                           cap_equity_usdt=round(cap_equity, 2) if cap_equity else None,
                           cap_ledger_usdt=round(cap_ledger, 2) if cap_ledger else None,
                           cap_model_usdt=round(cap_model, 2) if cap_model else None,
                           risk=risk,
                           equity_x=SIZING_CAP_EQUITY_X,
                           liq_floor_pct=round(100.0 / SIZING_CAP_EQUITY_X, 1),
                           # 🔴꺼져 있다는 사실을 payload 에 **명시**한다. 조용히 흘려보내면
                           #   화면이 «원장 상한이 묶었다»고 거짓말한다.
                           override_x=SIZING_CAP_OVERRIDE_X if overridden else None,
                           binding=who)
            # ⭐권고 수량의 기준점이 «습관 중앙값»에서 «위험»으로 바뀌었다(2026-09-13).
            # 모델이 있으면 순자산 × 허용배수 ÷ 가격, 없으면 옛 1/변동성 경로.
            price_ref = float(book["askPrice"]) if side == "LONG" else float(book["bidPrice"])
            rec_qty = float(sizing.get("vol_equivalent_qty") or 0.0)
            rec_src = "vol_equivalent"
            if risk.get("available") and equity > 0 and price_ref > 0:
                # 🔴E|r| 배수(`× evr_q`)를 여기 얹었다가 **켜기 전에 철회했다**(2026-09-15 당일).
                #   5주 원장 순차 검정에서 증분이 앞뒤로 갈렸다 -- 전반 36건 +11.01bp/건,
                #   후반 36건 **+0.73**(Δ>0 47.2% = 동전). 상위 10건을 빼면 합계 부호가
                #   뒤집힌다(+422.5 → −89.7bp). 두 CI 가 0 을 배제하지만 그 유의성을 만드는 건
                #   전반부다. 원장이 ~100왕복 될 때 다시 본다. 상세 docs/homer §5.36-S.
                rec_qty = (entry_notional(equity, risk["safe_mae_pct"])
                           ["total_notional"] / price_ref)
                rec_src = "risk_model"
            plan = build_entry_plan(
                side=side, best_bid=float(book["bidPrice"]), best_ask=float(book["askPrice"]),
                recommended_qty=rec_qty,
                cap_notional=cap_notional,
                filters=filters, symbol=symbol, existing_notional=exposure,
                equity=equity, leverage=leverage, fraction=fraction)
            plan["recommended_source"] = rec_src
            plan["recommended_qty"] = round(rec_qty, 8)
            plan["projection"] = entry_projection(plan, account, positions, existing, equity)
            # 2026-09-13 «지금 상황» 플랜: 보유시간 권고·집행·분할·예산 사다리. 상한은 적용된 실효 배수.
            plan["trade_plan"] = plan_now(
                side=side, equity=equity, existing_notional=exposure,
                unrealized_pnl=same_unrealized,
                risk_table=sizing.get("risk_mae") or {},
                vol_bpm=await realized_vol_now(symbol),
                cap_x=(cap_notional / equity) if cap_notional and equity > 0 else SIZING_CAP_EQUITY_X,
                atr_pct=sizing.get("atr_pct"), hold_min=hold_min,
                policy_cap_x=policy_cap_x,
                funding_bp_8h=await funding_now(symbol))
            # 집행기는 계획 dict 하나만 받는다. 처방 깊숙이 손을 넣게 하지 않고 여기서 꺼내 준다.
            _rx = (plan["trade_plan"] or {}).get("prescription") or {}
            _lv = (_rx.get("exchange_leverage") or {}) if _rx.get("available") else {}
            # 🔴손절은 **체결 후 평단** 기준이다(2026-09-13, 사용자 결정 가격 3%).
            # 기존 포지션이 있으면 «기존 + 이번 주문»의 가중평균이 새 평단이 된다 --
            # 물타기하면 손절가가 따라 내려온다(기존 주문은 집행기가 지우고 다시 건다).
            _sq = sum(abs(float(p.get("qty") or 0.0)) for p in same)
            _sv = sum(abs(float(p.get("qty") or 0.0)) * float(p.get("entry_price") or 0.0)
                      for p in same)
            _nq = float(plan.get("quantity") or 0.0)
            _np = float(plan.get("price") or 0.0)
            _vwap = ((_sv + _nq * _np) / (_sq + _nq)) if (_sq + _nq) > 0 else 0.0
            if _vwap > 0:
                # 🔴«계좌로 얼마인가»는 **이 주문 뒤 실제 명목**으로 재야 한다(2026-09-13 감사).
                # 상한 배수를 쓰면 분할 진입에서는 과대, **상한을 넘긴 상태에서는 과소**로 나온다
                # (실계좌 실측: 화면 18% vs 실제 45.4%). 손절 손실은 사용자가 «버틸 수 있나»를
                # 판단하는 바로 그 숫자라 상한이 아니라 사실을 적는다.
                _total_x = (float(plan.get("total_notional_usdt") or 0.0) / equity
                            if equity > 0 else 0.0)
                plan["stop_plan"] = build_stop_plan(
                    position_side=side, entry_price=_vwap, filters=filters, symbol=symbol,
                    leverage=_total_x)
                plan["stop_pct"] = STOP_LOSS_PCT
            # 게이지가 값을 주면 그걸 쓰고, «자동»이면 모델 추천을 쓴다. 어느 쪽인지 남긴다 --
            # 안 남기면 나중에 «왜 30배로 걸렸지»를 못 푼다.
            plan["target_leverage"] = want_lev or _lv.get("setting")
            plan["leverage_source"] = "manual" if want_lev else "model"
            plan["leverage_model"] = _lv.get("setting")
            plan["leverage_min_feasible"] = _lv.get("min_feasible")
            # 열린 포지션이 만드는 바닥. 이 아래를 고르면 거래소가 -2028 로 거부한다.
            plan["leverage_position_floor"] = _lv.get("position_floor")
            # 🔴화면이 띄울 **계획 지평**. 상수가 아니라 크기를 실제로 정한 그 값이다
            # (2026-09-14). 상수를 보내면 화면엔 4시간인데 1440분 셀로 크기가 나간다.
            plan["hold_planned_min"] = plan_hold
            plan["hold_remaining_min"] = hold_min
            plan["leverage_steps"] = list(LEVERAGE_STEPS)
        except Exception as exc:  # noqa: BLE001 -- 여기서 터져도 주문은 아직 안 나갔다
            return None, {}, {}, ({"error": f"{type(exc).__name__}: {exc}"}, 502)
        return plan, cap, sizing, None

    async def api_manual_entry_preview(request: web.Request) -> web.Response:
        """«이 버튼을 누르면 나갈 주문». 1단계에서는 게이트가 닫혀 있어 **보내지 않는다**.

        미리보기가 실주문과 **같은 함수**(build_entry_plan)를 통과한다 -- 다른 경로로 만든
        미리보기는 실주문을 검증하지 못한다(대조군이 안 덮는 경로는 검증 안 된 것)."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "진입 비율은 0 초과 100 이하여야 합니다"}, status=400)
        plan, cap, sizing, error = await assemble_entry_plan(side, frac,
                                                             query_leverage(request))
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        return web.json_response({"ok": True, "plan": plan, "cap": cap,
                                  "recommended_qty": plan.get("recommended_qty"),
                                  "recommended_source": plan.get("recommended_source"),
                                  "exec_enabled": exec_enabled()},
                                 headers=NOCACHE)

    async def api_manual_entry_submit(request: web.Request) -> web.Response:
        """실주문. **POST 전용 + confirm=1 필수 + 게이트가 켜져 있어야** 나간다.

        POST 인 이유: GET 이면 링크 미리보기·프리페치·주소창 재방문이 그대로 주문이 된다.
        confirm 을 받는 이유: 프런트의 2단 확인을 서버에서 한 번 더 건다 -- 프런트만 믿으면
        프런트 버그가 곧 오발주다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm=1 required"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "진입 비율은 0 초과 100 이하여야 합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"},
                                     status=403)
        if manual_entry_state.get("phase") in ("working", "submitting"):
            return web.json_response({"ok": False, "error": "already_working",
                                      "state": manual_entry_state}, status=409)
        # 비율은 **여기서 다시** 적용한다 -- 기존 포지션도 다시 읽으므로, 앞 칸이 이미
        # 들어가 있으면 상한 여유가 그만큼 줄어든 상태에서 계산된다.
        plan, cap, sizing, error = await assemble_entry_plan(side, frac,
                                                             query_leverage(request))
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        if plan.get("blocked"):
            return web.json_response({"ok": False, "error": "blocked", "detail": plan["blocked"]},
                                     status=400)
        manual_entry_state.clear()
        manual_entry_state.update(phase="submitting", side=side, plan=plan,
                                  started_at=datetime.now(timezone.utc).isoformat())
        # refresh_tasks 에 넣어 두면 stop_http_session 이 세션을 닫기 전에 취소해 준다.
        refresh_tasks["manual_entry"] = asyncio.create_task(
            run_entry(binance_session(), plan, manual_entry_state))
        return web.json_response({"ok": True, "plan": plan, "state": manual_entry_state},
                                 headers=NOCACHE)

    HOLD_CHOICES = (60, 120, 240, 480, 1440)
    # 🔴2026-09-13 사용자 결정: 보유시간을 **4시간 고정**. 화면 선택지를 없앤다.
    # 근거: 생존 상한이 보유시간에 가파르게 반응한다(1일 4.30배 -> 4시간 8배 이상).
    # «지렛대는 크기가 아니라 보유시간»(09-13 반사실)의 직접 적용이다.
    async def funding_now(symbol: str) -> float | None:
        """현재 펀딩률(bp/8h). 🔴비용 모델에 **없던 항목**이다(2026-09-13 감사) -- 8시간마다
        정산되므로 보유시간에 비례하고 롱/숏 부호가 반대다. 1440분 보유면 왕복비용의 28%다.
        못 읽으면 None -- 그때는 최근 30일 중앙값 폴백을 쓴다(0 으로 두면 긴 지평이 공짜가 된다)."""
        try:
            r = await fetch_binance_json("https://fapi.binance.com/fapi/v1/premiumIndex",
                                         {"symbol": symbol}, timeout=5.0, error_reason=None)
            if r and r.get("lastFundingRate") is not None:
                return float(r["lastFundingRate"]) * 1e4
        except Exception:  # noqa: BLE001 -- 있으면 좋은 값이지 필수가 아니다
            pass
        return None

    # 🔴지평의 단일 출처는 `planning_hold()` 다(2026-09-14 사용자 결정). 이 상수는 그 함수가
    # 고르지 못할 때(워커 없음·표 없음)의 **폴백**일 뿐, 정책 지평이 아니다 -- 상수로 쓰면
    # 화면엔 4시간인데 크기는 1440분 셀로 나간다(실측 허용 배수 6.0 vs 4.3배).
    HOLD_FIXED_MIN = 240

    def remaining_hold(entry_at: str | None, fixed: int = HOLD_FIXED_MIN) -> int:
        """**남은** 보유시간을 모델 지평에 맞춰 올림한다.

        🔴물타기를 해도 시계는 **안 늘어난다**. 약속은 포지션에 걸린 것이지 칸마다 새로 생기지
        않는다. `entry_at` 은 첫 체결 시각이라(positionRisk 의 updateTime 과 달리 물타기에
        안 움직인다) 그대로 쓸 수 있다.
        ⚠️올림은 **보수적**이다 -- 남은 90분이면 120분 표를 본다(더 큰 역행폭 = 더 작은 크기).
        """
        if not entry_at:
            return fixed
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(entry_at)).total_seconds() / 60.0
        except (TypeError, ValueError):
            return fixed
        left = fixed - max(0.0, age)
        if left <= 0:                      # 약속한 시간을 넘겼다 -- 가장 짧은 표를 쓴다
            return min(HOLD_CHOICES)
        return next((h for h in sorted(HOLD_CHOICES) if h >= left), fixed)

    def planning_hold(sizing: dict[str, Any], side: str, policy_cap_x: float | None) -> int:
        """**상한을 계산할 지평**. 처방이 고를 지평과 같아야 한다.

        🔴예전에는 상한을 240분 셀로 계산하면서 화면은 1440분을 권고했다(2026-09-13 감사).
        1440분의 허용 배수는 240분의 3분의 1 수준이라, 권고대로 들면 진입 직후 예산 사다리를
        35% 위반했다. 여기서 한 번 고르고 그 값을 상한·처방·손절 표시가 **공유**한다.

        지평 선택에는 상한이 필요하고 상한에는 지평이 필요한 순환이 있다 -- **지평과 무관한**
        정책 천장(원장 ∧ 순자산)으로 끊는다. 위험모델 상한은 그 뒤에 min 으로 들어간다."""
        table = (sizing or {}).get("risk_mae") or {}
        atr = (sizing or {}).get("atr_pct")
        if not table or not atr:
            return HOLD_FIXED_MIN
        h = recommend_hold(table, side, policy_cap_x or SIZING_CAP_EQUITY_X, atr,
                           acc=PRESCRIBE_ACC)
        return int(h["recommended_min"]) if h.get("available") else HOLD_FIXED_MIN

    def risk_sizing(sizing: dict[str, Any], hold_min: int, side: str) -> dict[str, Any]:
        """이 보유시간에서 각오할 역행폭. **사이징 워커가 300초마다 계산해 둔 값을 읽는다.**

        🔴요청 경로에서 모델을 돌리지 않는다 -- 대시보드는 60초마다 이 미리보기를 폴링하므로
        그러면 to_thread 풀이 고갈된다(2026-09-10 실장애, 워커가 애초에 존재하는 이유).
        워커 상태가 낡았으면 available=False 로 내려가고 호출부는 기존 상한만 쓴다."""
        cell = ((sizing.get("risk_mae") or {}).get(str(hold_min)) or {}).get(side.upper())
        if not cell or not (cell.get("safe_mae_pct", 0) > 0):
            return {"available": False, "hold_min": hold_min,
                    "reason": "worker_no_risk" if sizing.get("available") else "sizing_unavailable"}
        return {"available": True, "hold_min": hold_min,
                "safe_mae_pct": cell["safe_mae_pct"]}

    def query_leverage(request: web.Request) -> int | None:
        """화면 게이지가 고른 거래소 레버리지. 없으면 None -- 그때는 **모델 추천**을 쓴다.

        범위를 벗어나면 조용히 자르지 않고 None 을 돌려 모델값으로 떨어뜨린다.
        (자르면 «10 을 눌렀는데 1 이 걸림» 같은 일이 생긴다.)"""
        raw = request.query.get("lev")
        if raw in (None, ""):
            return None
        try:
            v = int(float(raw))
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= EXCHANGE_MAX_LEVERAGE else None

    def query_fraction(request: web.Request) -> float | None:
        """쿼리의 비율(%)을 0<f<=1 로 바꾼다. 진입 분할과 부분 청산이 **같은 함수**를 쓴다.
        이상하면 None -- 호출부가 400 을 낸다.
        **조용히 1.0 으로 떨어뜨리지 않는다**: 일부만 하려던 요청이 전량이 되면 안 된다."""
        raw = request.query.get("pct")
        if raw in (None, ""):
            return 1.0
        try:
            pct = float(raw)
        except (TypeError, ValueError):
            return None
        return pct / 100.0 if 0.0 < pct <= 100.0 else None

    async def assemble_exit_plan(position_side: str, fraction: float = 1.0,
                                 fresh: bool = False):
        """청산 계획 조립. 진입과 같은 이유로 **여기 한 곳뿐**이다.

        수량은 반드시 **방금 읽은 포지션**에서 온다 -- 헤지 모드라 reduceOnly 를 못 써서
        (-1106) 과청산을 막는 게 수량밖에 없다. 캐시된 값을 쓰면 이미 닫힌 포지션을
        다시 닫으려다 반대 방향으로 열릴 수 있다.

        🔴2026-09-19 심볼도 **설정이 아니라 실제 포지션**에서 온다. 진입 심볼을 USDC 로
        바꿔도 그 전에 연 ETHUSDT 포지션은 계속 닫을 수 있어야 하고, 반대로 설정값을 믿고
        보내면 «없는 포지션을 닫는» 주문이 헤지 모드에서 **반대 방향 신규 진입**이 된다."""
        candidates = list(dict.fromkeys([MANUAL_EXEC_SYMBOL, MARKET_SYMBOLS["eth"]]))
        try:
            # 🔴실주문(fresh=True)은 30초 캐시를 **우회**한다. 헤지 모드라 reduceOnly 가 없어
            # 과청산 방어가 수량뿐인데, 그 수량이 30초 묵으면 방어가 30초 묵는다.
            account = (await produce_account() if fresh else
                       await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS,
                                        produce_account, max_stale=STALE_GRACE_SECONDS))
            # 결정 규칙과 자체점검은 live_manual_peg_entry 모듈에 있다(85/85).
            position, symbol, leftover = resolve_exit_position(
                account.get("positions") or [], position_side, candidates)
            if position is None:
                return None, ({"error": "no_position",
                               "detail": f"{position_side} 포지션이 없습니다"}, 400)
            book = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                                            {"symbol": symbol}, error_reason="book_ticker_failed")
            filters = await load_filters(binance_session(), symbol)
            vol_bpm = await realized_vol_now(symbol)
            plan = build_exit_plan(
                position_side=position_side, position_qty=float(position.get("qty") or 0.0),
                best_bid=float(book["bidPrice"]), best_ask=float(book["askPrice"]),
                filters=filters, symbol=symbol,
                entry_price=float(position.get("entry_price") or 0.0),
                mark_price=float(position.get("mark_price") or 0.0),
                vol_bpm=vol_bpm, fraction=fraction)
            plan["unrealized_pnl"] = position.get("unrealized_pnl")
            if leftover:                      # 같은 방향이 다른 심볼에도 열려 있다
                plan["other_symbol_open"] = leftover
            # 남은 보유시간 기준 위험 한도. 넘었으면 «최소 이만큼은 닫아야 한다»를 준다.
            # 🔴위에서 읽은 **그 계좌**를 쓴다(2026-09-13 병합). 같은 함수 안에서 캐시를 또
            # 조회하면 fresh=True(실주문)일 때 수량은 새 값인데 순자산·명목은 30초 묵은 값이
            # 되어, 청산 한도가 실제와 어긋난 채로 «최소 몇 % 닫아라»가 나간다.
            eq = float((account.get("balance") or {}).get("margin") or 0.0)
            cur_notional = sum(abs(float(p.get("notional") or 0.0))
                               for p in (account.get("positions") or [])
                               if p.get("symbol") == symbol)
            sz = await asyncio.to_thread(position_sizing_payload)
            # 🔴지평을 고를 정책 천장은 **지평과 무관**해야 순환이 끊긴다(진입과 같은 구조).
            # safe_mae_pct=None 이면 effective_cap 이 모델 후보를 빼고 원장∧순자산만 본다.
            policy_n, _, _ = effective_cap(sz, eq, None)
            policy_cap_x = (policy_n / eq) if policy_n and eq > 0 else None
            # 🔴이미 든 시간만큼 깎은 **남은** 보유시간으로 잰다. 물타기로 칸이 늘어도 시계는
            # 그대로라, 늦게 추가할수록 남은 시간이 짧아 허용 배수가 커진다 -- 그건 «그 시각에
            # 실제로 닫는다»는 전제 위에서만 맞다. 화면이 남은 시간을 같이 띄운다.
            # 🔴시계의 **길이**는 진입이 처방한 지평이다(2026-09-14). 예전에는 240분 상수라
            # 24시간 처방으로 들어간 포지션을 4시간 예산으로 재고, 4시간이 지나면 60분 셀로
            # 떨어져 «여유가 더 생겼다»고 말했다 -- 지평이 짧을수록 허용 배수가 커지기 때문이다.
            plan_hold = planning_hold(sz, position_side, policy_cap_x)
            hold_min = remaining_hold(position.get("entry_at"), fixed=plan_hold)
            risk = risk_sizing(sz, hold_min, position_side)
            # 위험모델이 없으면 모델 후보가 빠진 정책 천장이 그대로 실효 상한이다.
            eff_x = policy_cap_x or SIZING_CAP_EQUITY_X
            if risk.get("available") and eq > 0 and cur_notional > 0:
                # 🔴진입과 **같은 상한**을 쓴다. 정책상한 25배로 재면 진입이 8배에서 막은
                # 포지션을 청산은 «닫을 필요 없음»이라고 말한다(같은 카드에 모순된 두 숫자).
                cap_n, who, _ = effective_cap(sz, eq, risk["safe_mae_pct"])
                eff_x = (cap_n / eq) if cap_n else HARD_CAP_X
                r = exit_fraction_required(eq, risk["safe_mae_pct"], cur_notional,
                                           hard_cap=eff_x)
                risk["effective_x"] = round(eff_x, 2)
                risk["applied_binding"] = who
                plan["risk"] = {**risk, "required_fraction": round(r["required_fraction"], 4),
                                "allowed_notional": round(r["allowed_notional"], 2),
                                "excess_notional": round(r["excess_notional"], 2),
                                "current_notional": round(cur_notional, 2),
                                "leverage": round(r["leverage"], 2), "binding": r["binding"]}
            else:
                plan["risk"] = risk
            plan["hold_planned_min"] = plan_hold
            plan["hold_remaining_min"] = hold_min
            # 🔴상한은 **위에서 이미 고른 실효 상한**이다(2026-09-14). 상수 6.0 을 쓰면 같은
            # 카드가 «상한 넘었으니 닫아라»(risk, effective_cap 기준)와 «15,000 더 넣을 여유
            # 있음»(trade_plan, 6.0배 기준)을 나란히 띄운다 -- effective_cap 이 막으려고 생긴
            # 바로 그 모순인데, 74441ad 감사가 risk 블록에만 적용되고 이 줄을 지나쳤다.
            plan["trade_plan"] = plan_now(
                side=position_side, equity=eq, existing_notional=cur_notional,
                unrealized_pnl=float(position.get("unrealized_pnl") or 0.0),
                risk_table=sz.get("risk_mae") or {}, vol_bpm=vol_bpm,
                cap_x=eff_x, atr_pct=sz.get("atr_pct"), hold_min=hold_min,
                policy_cap_x=policy_cap_x or SIZING_CAP_EQUITY_X,
                funding_bp_8h=await funding_now(symbol))
        except Exception as exc:  # noqa: BLE001 -- 여기서 터져도 주문은 아직 안 나갔다
            return None, ({"error": f"{type(exc).__name__}: {exc}"}, 502)
        return plan, None

    async def api_manual_exit_preview(request: web.Request) -> web.Response:
        """«이 버튼을 누르면 나갈 청산 주문». 진입과 같은 함수를 지나므로 미리보기가
        실주문을 실제로 검증한다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "청산 비율은 0 초과 100 이하여야 합니다"}, status=400)
        # 🔴청산 미리보기도 **fresh** 다(2026-09-14, 사용자 요청 «청산 누르면 강제 조회부터»).
        # 사람이 버튼을 눌러야만 오는 경로라 호출이 잦지 않고, 30초 캐시로 그리면 화면이
        # 「2.754 닫는다」고 말한 뒤 submit(이미 fresh)이 다른 수량을 내보낼 수 있다.
        # 미리보기와 실주문이 **같은 수량을 보는 것**이 이 화면의 존재 이유다.
        plan, error = await assemble_exit_plan(side, frac, fresh=True)
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        return web.json_response({"ok": True, "plan": plan, "exec_enabled": exec_enabled()},
                                 headers=NOCACHE)

    async def api_manual_exit_submit(request: web.Request) -> web.Response:
        """실제 청산. 진입과 **같은 방어**를 건다: POST 전용 + confirm=1 + 게이트.
        상태 dict 도 진입과 공유한다 -- 수동 주문이 동시에 둘 나가지 않는 보호가 따라온다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm=1 required"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "청산 비율은 0 초과 100 이하여야 합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"},
                                     status=403)
        if manual_entry_state.get("phase") in ("working", "submitting"):
            return web.json_response({"ok": False, "error": "already_working",
                                      "state": manual_entry_state}, status=409)
        # 비율은 **여기서 다시** 적용한다 -- 포지션도 다시 읽으므로 미리보기 이후에 포지션이
        # 줄었으면 그만큼 줄어든 수량이 나간다(프런트가 계산한 수량을 받지 않는 이유).
        plan, error = await assemble_exit_plan(side, frac, fresh=True)
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        if plan.get("blocked"):
            return web.json_response({"ok": False, "error": "blocked", "detail": plan["blocked"]},
                                     status=400)
        manual_entry_state.clear()
        manual_entry_state.update(phase="submitting", kind="exit", side=side, plan=plan,
                                  started_at=datetime.now(timezone.utc).isoformat())
        refresh_tasks["manual_entry"] = asyncio.create_task(
            run_exit(binance_session(), plan, manual_entry_state))
        return web.json_response({"ok": True, "plan": plan, "state": manual_entry_state},
                                 headers=NOCACHE)

    async def api_manual_entry_status(request: web.Request) -> web.Response:
        """진행 중인 수동 주문 상태. 프런트가 폴링해 «메이커로 채워졌나 / 테이커로 넘어갔나»를
        보여준다. 주문은 최대 하나만 동시에 둔다."""
        return web.json_response({"ok": True, "state": manual_entry_state,
                                  "exec_enabled": exec_enabled()},
                                 headers=NOCACHE)

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
                headers=NOCACHE,
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
                headers=NOCACHE,
            )
        return json_response(request, payload, etag)

    async def api_scalp_reuse_shadow(request: web.Request) -> web.Response:
        mode = request.query.get("mode", "eth_lifecycle").lower()
        config = SCALP_REUSE_MODES.get(mode)
        if config is None:
            return web.json_response(
                {"error": "unsupported_scalp_reuse_mode"},
                status=web.HTTPBadRequest.status_code,
                headers=NOCACHE,
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
                headers=NOCACHE,
            )
        return json_response(request, payload, etag)

    async def api_btc_evidence_shadow(request: web.Request) -> web.Response:
        etag = make_etag(
            "btc-evidence-shadow",
            file_signature(BTC_EVIDENCE_SHADOW_STATE_PATH),
            file_signature(BTC_EVIDENCE_CTX_REPORT_PATH),
        )
        if etag_matches(request, etag):
            return json_response(request, None, etag)
        return json_response(request, btc_evidence_shadow_payload(), etag)

    app.router.add_get("/", index)
    app.router.add_get("/dashboard/live", dashboard_index)
    app.router.add_get("/dashboard/live/", dashboard_index)
    # 🔴`/index.html` 로 직접 열면 add_static 이 원본을 그대로 줘서 **버스터 치환을 우회한다**
    #   (같은 비대칭을 2026-09-13 에 Cache-Control 로 한 번 겪었다). 같은 핸들러로 묶는다.
    app.router.add_get("/dashboard/live/index.html", dashboard_index)
    # add_static("/dashboard/live/") 보다 먼저 등록 -- aiohttp는 등록 순서대로 매칭하므로
    # 이 둘만 no-cache 경로로 빠지고 나머지 정적 파일은 그대로 static 핸들러가 처리한다.
    app.router.add_get("/dashboard/live/{name:sw\\.js|manifest\\.webmanifest}", pwa_asset)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/events", api_events)
    app.router.add_get("/api/market-history", api_market_history)
    app.router.add_get("/api/footprint", api_footprint)
    app.router.add_get("/api/supply-profile", api_supply_profile)
    app.router.add_get("/api/supply-1s", api_supply_1s)
    app.router.add_get("/api/oi-5m", api_oi_5m)
    app.router.add_get("/api/breakout-detector", api_breakout_detector)
    app.router.add_get("/api/gex", api_gex)
    app.router.add_get("/api/flow/heatmap", api_flow_heatmap)
    app.router.add_get("/api/chart-markers", api_chart_markers)
    app.router.add_get("/api/liquidation-5m-signal", api_liquidation_5m_signal)
    app.router.add_get("/api/liquidation-direction-signal", api_liquidation_direction_signal)
    app.router.add_get("/api/liquidation-map", api_liquidation_map)
    app.router.add_get("/api/regime-wide24", api_regime_wide24)
    app.router.add_get("/api/regime-btc", api_regime_btc)
    app.router.add_get("/api/regime-xrp", api_regime_xrp)
    app.router.add_get("/api/coin-indicators", api_coin_indicators)
    app.router.add_get("/api/macro-calendar", api_macro_calendar)
    app.router.add_get("/api/liq-burst-state", api_liq_burst_state)
    app.router.add_get("/api/micro-ref", api_micro_ref)
    app.router.add_get("/api/situation", api_situation)
    app.router.add_get("/api/session-alerts", api_session_alerts)
    app.router.add_get("/api/push/config", api_push_config)
    app.router.add_post("/api/push/subscribe", api_push_subscribe)
    app.router.add_post("/api/push/unsubscribe", api_push_unsubscribe)
    app.router.add_get("/api/push/devices", api_push_devices)
    app.router.add_post("/api/push/test", api_push_test)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/binance-account", api_binance_account)
    app.router.add_get("/api/manual-entry/preview", api_manual_entry_preview)
    app.router.add_post("/api/manual-entry/submit", api_manual_entry_submit)
    app.router.add_get("/api/manual-entry/status", api_manual_entry_status)
    app.router.add_get("/api/manual-exit/preview", api_manual_exit_preview)
    app.router.add_post("/api/manual-exit/submit", api_manual_exit_submit)
    app.router.add_get("/api/position-sizing", api_position_sizing)
    app.router.add_get("/api/liquidation-5m-history", api_liquidation_5m_history)
    app.router.add_get("/api/ops-status", api_ops_status)
    app.router.add_get("/api/scalp-shadow", api_scalp_shadow)
    app.router.add_get("/api/scalp-reuse-shadow", api_scalp_reuse_shadow)
    app.router.add_get("/api/btc-evidence-shadow", api_btc_evidence_shadow)
    # show_index=False to match /data/live/ below: this directory is reachable from the
    # public tunnel, and a listing advertised every file sitting in it (e.g. the
    # *.bak_pre_live_tab_removal_20260831 snapshots) rather than just the three the page
    # actually loads. The explicit /dashboard/live[/] routes above already serve the page
    # itself, and nothing fetches a listing, so this only removes the enumeration.
    app.router.add_static("/dashboard/live/", DASHBOARD_DIR, show_index=False)
    app.router.add_static("/data/live/", LIVE_DIR, show_index=False)
    # start_http_session FIRST: every klines fetch below it needs the pooled session to exist.
    # stop_http_session LAST for the mirror-image reason.
    app.on_startup.append(start_http_session)
    app.on_startup.append(start_dashboard_events)
    app.on_startup.append(start_footprint_collector)
    app.on_startup.append(start_oi_1s_collector)
    app.on_startup.append(start_micro_ref)
    app.on_cleanup.append(stop_micro_ref)
    app.on_cleanup.append(stop_oi_1s_collector)
    app.on_cleanup.append(stop_footprint_collector)
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
