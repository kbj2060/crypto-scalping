#!/usr/bin/env python3
"""Read-only live event-triggered signal: this dashboard's "V자 반등락" specialized-detector chip.

**2026-08-31 upgrade -- 9-trigger multitrigger model replaces the sweep-only v7b model.** Full
history: memory eth_v_rebound_sweep_gated_recall_gap_20260831 (5-stage recall-gap diagnostic that
motivated this, then the label/feature/TabPFN/holdout chain that validated it), docs/homer/
README.md's "V자반등/지지횡보" section. Filename/function name kept unchanged from the v1-v7b
lineage per this project's established convention (dated filenames are not renamed through later
version upgrades -- e.g. build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py stayed named
through its own v1->v4 evolution).

**Why this upgrade happened**: a 90-day recall-gap audit found the OLD sweep-only trigger caught
only ~25.5% of an estimated true population of qualifying V-shaped reversals (sweep is just one of
several conditions that precede a real V자반등; most instances happen without a sweep at all, at
equal or better quality). Widening the candidate pool to 9 triggers (all OR'd, not AND'd) closes
that gap while reusing the EXACT SAME v7b outcome/label formula unchanged (fast_move_atr_mult>=1.5x
within 30min AND giveback_ratio<=0.20 within 60min) -- trigger and label are fully decoupled axes;
only which bars get scored changed, not what "V자반등" means.

The 9 triggers (OR'd; downside=candidate for an upward rebound, upside=mirror):
  1. liquidity_sweep, 2. taker_delta_z_climax, 3. short_term_return_z, 4. orthogonal_combo,
  5. smt_divergence, 6. fib_extension_exhaustion, 7. demarker_extreme, 8. kalman_deviation_meanrev
  -- all 8 reused verbatim via compute_signals() (live_evidence_signal_dashboard_20260823.py),
  the same canonical pre-TabPFN boolean-trigger source this dashboard's own evidence-signal chips
  use (demarker_extreme/kalman_deviation_meanrev were themselves 2026-08-31 Homer candidate-pool
  additions to that function, independently validated and deployed by a concurrent session).
  9. local_extreme -- the only genuinely new trigger, no precondition: bar is the highest/lowest
  in a +-30min (+-6 bar) window. Turned out to be both the single largest AND highest-hit-rate
  trigger of the 9 (22.2% vs the others' 12-20%) in the full-history label build.

═══════════════════════════════════════════════════════════════════════════════════════════════
**2026-09-01 재설계 -- 9트리거 게이트 제거, 매 5분봉 전부 채점(every-bar scoring).**
═══════════════════════════════════════════════════════════════════════════════════════════════

위 9트리거 구성은 **라벨 생성에는 그대로 유지**되지만, 라이브가 "어느 봉을 채점하는가"의 게이트
역할에서는 제거됐다. 이제 매 5분봉이 양방향(bottom/top)으로 채점되고, 현재 봉의 확률이 항상
표시된다. `triggers` 필드는 남지만 **표시 전용 참고정보**로 격하됐다.

왜: 9트리거 중 local_extreme(호출 population의 73~76%를 공급)은 정의상
`low[i]==min(low[i-6:i+7])`이라, 라벨의 fast_move가 요구하는 held_up 선행조건을 **100% 만족한
봉만** 후보로 올렸다. 트리거·자산 무관하게 라벨률을 4.2~4.8배 부풀리는 기계적 얽힘이고, 모델은
그 공짜 크레딧을 성능으로 계상해왔다. 라이브 인과성 자체는 깨지지 않았지만(모델이 미래를 보지는
않음) 헤드라인 수치는 과대평가였다 -- held_up 층화 시 내부 AUC 0.66~0.69로 하락. 부수적으로
local_extreme의 30분 지연확인이 "신호가 갑자기 과거기록과 함께 나타남" UX 문제와 경제성
백테스트의 진입시점 비현실성(+9.28bp -> +4.75bp 정정)의 원인이기도 했다. 게이트를 없애면 모델이
held_up을 스스로 예측해야 하고, 현재 봉에 점수가 나오므로 다음 봉 진입이 정직한 가정이 된다.
전체 기록: docs/homer/v_rebound_open_issues_20260901.md, 메모리
eth_v_rebound_held_up_circularity_proof_and_model_skill_check_20260901.

⚠️ **게이트만 제거하면 안 된다 -- 전체 재학습이 필수다.** 후보풀로 학습한 모델을 그대로 전체 봉에
적용하면 AUC 0.5287로 붕괴한다(이 저장소에서 "kept-only 착시"의 3번째 반복). 재학습 후 실측
(research_eth_v_rebound_every_bar_tabpfn_confirm_20260901.py, TabPFN 3시드, VAL 전체봉 15,000행):
    TabPFN random_18000   VAL AUC 0.6942±0.0023   라이브 사이클 6.56s
    GBM 프록시(전체 TRAIN 182,969행)      0.6953   <- 재현 확인(−0.0011)
    후보풀 학습을 전체봉에 적용            0.5287   <- 붕괴 기준선(+0.1655 개선)
0.6942는 옛 헤드라인 0.8465보다 낮지만, 그 0.8465가 held_up 크레딧을 포함한 값이었고 층화 추정치
0.66~0.69와 여기가 독립 수렴한다 -- **같은 문제를 정직하게 푼 점수**다. 라벨 정의가 달라졌으므로
두 AUC의 직접 비교는 무효(메모리 feedback_cross_model_auc_comparison_requires_matched_label_
difficulty_20260901).

기각된 대안들(전부 실측 근거): shift 변형 라벨, 8트리거를 피쳐로 추가, 사건당 1봉 샘플링
(chop이 배경상태라 156k봉->154사건, base 96.8%, AUC 0.5260), 절대 bp 하한 20/30bp, 층화추출
(무작위와 유의차 없음), event-first 컨텍스트 샘플링(GBM −0.097).

**임계값 0.60** (2026-09-01 0.50에서 상향, 사용자 승인). 실제 서빙 확률(TabPFN + 동결 컨텍스트)로
경제성을 재실행한 결과 0.50이 게이트를 통과하지 못했다 -- 방향뒤집기 대조군이 저ARM 노이즈수확
아티팩트가 닿지 않는 ARM=1.5 구간(40셀)에서도 정방향을 이겼다(VAL 정4/뒤6, OOS 정10/뒤13).
0.55부터 역전되고 0.60이 가장 깨끗하다:

    thr        VAL 정/뒤   OOS 정/뒤   선택셀 VAL / OOS      precision VAL/OOS
    0.50         4 / 6      10 / 13    +0.90 / +6.81bp      0.606 / 0.596
    0.55        29 / 3      19 / 10    +9.28 / +11.93bp     0.664 / 0.647
    0.60        35 / 0      22 / 9     +16.05 / +11.30bp    0.713 / 0.683

빈도 손실은 없다 -- 화면 기준(봉당 max 집계, 라벨 없는 봉 포함) 하루 11~12건, 신호 간격 중앙값
약 1시간, 신호 없는 날 0%, 4시간 스트립에 평균 2.2~2.4칸. 0.50은 하루 18건으로 오히려 과했다.
전체 수치: data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/{report,signal_frequency}.json

⚠️ **precision의 적용 범위 주의**: 위 precision은 라벨이 붙은 행(전체의 52.9%) 기준이다. 라이브는
라벨 유무와 무관하게 모든 봉을 채점하므로, 화면에 뜨는 신호의 상당수는 결과가 v_rebound도 chop도
아니었던(excluded-middle) 봉에 앉는다. 이는 라벨 설계에서 오는 구조적 한계로 모든 임계값에 동일
적용되며 이 상향으로 생긴 문제가 아니다.

이전(후보풀 게이트) 시절의 검증 결과 -- 이제 헤드라인이 아니라 역사 기록:
  cheap_gate VAL/OOS AUC 0.8296/0.8119 -> 4시드 안정성 -> HOLDOUT VAL/OOS/HOLDOUT
  0.8292/0.8127/0.8465 + 트레일링 경제성 VAL+11.97/OOS+20.96/HOLDOUT+9.28bp(진입시점 보정 후
  +4.75bp). 전부 held_up 크레딧을 포함한 수치다.

TabPFN is in-context inference, not a saved/trained model file: every call re-fits on the SAME
FROZEN TRAIN context (data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_
every_bar_20260901.csv -- 전체 봉 TRAIN 182,969행에서 무작위 18,000행, 자연 라벨률 ~14.6%,
재균형 안 함; freeze_eth_v_rebound_every_bar_train_context_20260901.py가 생성). 18,000행은 TabPFN
권장 상한 1만행을 넘으므로 `ignore_pretraining_limits=True`가 필요하고, 검증 파이프라인도 처음부터
같은 플래그로 측정했다. 라이브 사이클 6.56s는 이 엔드포인트 캐시 주기(60s)의 11%.
Single-seed inference (random_state=20260829, matching this script's own convention -- the
multi-seed ensembles were for validation robustness, not live serving).

*** DISCRETIONARY READING AID -- NOT WIRED INTO trading_bot.py, NOT AUTOMATED ENTRY/EXIT. ***
Feature/BTC-fetch/RSI-Wilder machinery below is unchanged from the pre-redesign version (Tier0 22
+ rsi = 23 features, exact same formulas) -- only which bars get scored changed. Values are reused,
not reimplemented, from build_eth_5m_v_rebound_multitrigger_labels_20260831.py::main() and this
script's own pre-existing _build_features()/_rsi_wilder().

**콜 지속성(배지 + 히스토리 스트립 둘 다)** -- 2026-09-01, 매 봉 전환 직후 사용자가 화면에서
"신호가 유지가 안 된다", 이어서 "막대 게이지에 5분 칸 하나만 들어와 있다"고 보고해 추가.
매 봉 채점에서 콜이 봉 하나만 차지하면 사건당 평균 1.2봉, 즉 대부분 5분만 떴다 사라진다
(0.60 기준 하루 13.25 발동봉 / 11.01 사건 실측). 48칸 스트립에서 한 칸은 사실상 안 보인다.
그래서 증거신호 8종 칩의 `_fill_until_tp_or_horizon`과 **같은 계약**으로 맞춘다 -- 콜을 발동봉부터
"익절 도달(라벨의 1.5xATR 빠른 다리) 또는 호라이즌 경과(60분/12봉)" 중 먼저 오는 쪽까지 유지.
정의는 `_call_end_pos()` **한 곳에만** 두고 배지와 스트립이 함께 쓴다. 구간이 겹치면 나중 콜이
덮어쓴다(배지가 최근 콜을 보여주는 것과 일치). 스트립 창 직전에 발동한 콜도 창 안으로 칠해지므로
채점 범위를 `HISTORY_BARS + 1 + BADGE_HORIZON_BARS`로 넓혔다.
⚠️예전 게이트 시절의 지연확인 문제는 재발하지 않는다 -- 그건 local_extreme이 30분 뒤에야
확정되던 데서 왔고, 지금은 봉 자체에서 즉시 점수가 나온다. 여기서 늘리는 건 "이미 현재인 신호의
표시 시간"이지 "확정이 늦는 신호"가 아니다. 룩어헤드도 없다: `_fetch_klines()`가 형성중 봉을
이미 버리므로 익절 판정이 보는 close는 전부 확정된 과거 봉이다.

응답 스키마는 그대로 유지된다(dashboard/server.py와 app.js 무변경). `minutes_ago`는 배지가
가리키는 봉의 나이로 **원래 의미를 회복**했고(현재 봉이면 0, 유지중인 과거 콜이면 그 경과분),
`early_confirmed`는 항상 None이다(익절 도달한 콜은 배지에서 이미 내려가므로 표시중인 콜에는
해당 없음 -- 키만 남긴다). `_early_confirm_pos()`와 `_multitrigger_rows()`는 같은 날 삭제됐다 --
둘 다 이벤트/게이트 구조에만 쓰이던 함수라 호출자가 사라졌다.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

TRAIN_CONTEXT_CSV = ROOT / "data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_every_bar_20260901.csv"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
BTC_SYMBOL = "BTCUSDT"
FETCH_LIMIT = 1500          # ~5.2 days at 5m -- clears the 864-bar longest indicator warmup with margin
SWEEP_LOOKBACK_BARS = 48    # 4h -- matches the live liquidity_sweep definition elsewhere on this dashboard
HISTORY_BARS = 48           # 4h sparkline strip, matches this dashboard's HISTORY_BARS convention
LOCAL_EXTREME_W = 6         # +-30min, matches build_eth_5m_v_rebound_multitrigger_labels_20260831.py

# 2026-09-01: 0.50 -> 0.60. 0.50은 분류 운영점으로는 타당했지만(precision 0.606/0.596) **경제성
# 게이트를 통과하지 못했다** -- 방향뒤집기 대조군이 저ARM 아티팩트가 닿지 않는 ARM=1.5 구간에서도
# 정방향을 이겼다(VAL 정4/뒤6, OOS 정10/뒤13). 0.60은 두 구간 모두 깨끗이 통과하고
# (VAL 정35/뒤0, OOS 정22/뒤9; 선택셀 VAL+16.05bp/OOS+11.30bp) precision도 가장 높다
# (0.713/0.683). 빈도 손실도 없다 -- 화면 기준 하루 11~12건, 신호 간격 중앙값 ~1시간,
# 신호 없는 날 0%(0.50은 하루 18건으로 오히려 과했다). 근거:
# data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/{report,signal_frequency}.json
PROBA_THRESHOLD = 0.60

# 2026-09-01 배지 지속성. 매 봉 스코어링에서 배지가 현재 봉만 반영하면 사건당 평균 1.2봉,
# 즉 대부분 5분만 떴다 사라진다(0.60 기준 하루 13.25 발동봉 / 11.01 사건 실측; 사용자가
# 실제 화면에서 "유지가 안 된다"고 보고). 증거신호 8종 칩의 `_fill_until_tp_or_horizon`
# (live_evidence_signal_dashboard_20260823.py:656)과 **같은 계약**으로 맞춘다 -- 마지막 콜을
# "익절 도달 또는 호라이즌 경과" 중 먼저 오는 쪽까지 유지. 값은 이 신호 자신의 라벨 정의에서
# 가져온다(research_eth_v_rebound_label_redesign_variant_screen_20260901.py: FULL_BARS/ATR_MULT).
# 배지와 히스토리 스트립 **둘 다** 이 지속 규칙을 따른다(정의는 `_call_end_pos()` 한 곳).
# ⚠️예전 게이트 시절의 지연확인 문제는 재발하지 않는다 -- 그건 local_extreme이 30분 뒤에야
# 확정되던 데서 왔고, 지금은 봉 자체에서 즉시 점수가 나온다. 여기서 늘리는 건 "이미 현재인
# 신호의 표시 시간"이지 "확정이 늦는 신호"가 아니다.
BADGE_HORIZON_BARS = 12   # 60분 -- 라벨의 FULL_BARS
BADGE_ATR_MULT = 1.5      # 라벨의 ATR_MULT (빠른 다리 목표)

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]

NAMED_TRIGGERS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
                  "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion",
                  "demarker_extreme", "kalman_deviation_meanrev"]

_TRAIN_CACHE: pd.DataFrame | None = None
_SWEEP_IMPL = None


def _load_sweep_impl():
    global _SWEEP_IMPL
    if _SWEEP_IMPL is None:
        spec = importlib.util.spec_from_file_location("sweep_impl_live_20260829", SWEEP_IMPL_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SWEEP_IMPL = module
    return _SWEEP_IMPL


def _load_train_context() -> pd.DataFrame:
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        df = pd.read_csv(TRAIN_CONTEXT_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _TRAIN_CACHE = df
    return _TRAIN_CACHE


def _empty(error: str) -> dict:
    return {"warmed_up": False, "error": error, "event_active": False, "call": None,
            "direction": None, "proba_rebound": None, "minutes_ago": None,
            "sweep_ts_utc": None, "price": None, "tone": "neutral", "history": [], "times": [],
            "triggers": None, "early_confirmed": None}


def _fetch_klines(symbol: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(FUTURES_KLINES_URL,
                             params={"symbol": symbol, "interval": "5m", "limit": FETCH_LIMIT},
                             timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar
    return df


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _build_features(kl: pd.DataFrame) -> pd.DataFrame:
    """Unchanged by the 2026-08-31 trigger upgrade -- exact port of
    build_eth_5m_sweep_v_rebound_features_tier0_20260829.py::build_indicator_frame + main()'s
    per-row feature derivation, fed with freshly-fetched klines instead of the static training
    CSV. Produces sweep_level_low/high/atr (used generically for ANY trigger's direction-relative
    features now, not sweep-specific) plus all Tier0 columns."""
    sweep_impl = _load_sweep_impl()
    frame = compute_indicators(kl)
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)

    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)

    causal = sweep_impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    frame["sweep_level_low"] = causal["sweep_level_low"]
    frame["sweep_level_high"] = causal["sweep_level_high"]
    frame["atr"] = causal["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    frame["rsi"] = _rsi_wilder(frame["close"])
    return frame


def _every_bar_rows(frame: pd.DataFrame, sig: pd.DataFrame, n_tail: int) -> pd.DataFrame:
    """매 봉 스코어링(2026-09-01 재설계): 트리거 발동 여부와 무관하게 **최근 n_tail개 봉 전부**를
    양쪽 방향(bottom/top)으로 채점 대상으로 만든다.

    ⚠️ 이 함수가 이전 `_multitrigger_rows()`를 대체한다(같은 날 삭제 -- 게이트가 없어져 호출자가
    사라짐). 여러 BTC/ETH 연구 스크립트 주석이 `_multitrigger_rows()`를 부호 규약의 정본으로
    인용하는데, 그 공식(is_downside/sweep_penetration_atr/flow_aligned_delta_z --
    penetration = (level-low) 또는 (high-level), 즉 진짜 관통에서 양수)은 아래에 **문자 그대로
    동일하게** 옮겨져 있다. 바뀐 것은 "어느 봉을 넣는가"뿐이다.

    왜 게이트를 없앴는가: 9트리거 게이트는 held_up 얽힘의 원인이었다 -- local_extreme은 정의상
    `low[i]==min(low[i-6:i+7])`이라 라벨의 fast_move가 요구하는 선행조건(held_up)을 100%
    만족한 봉만 공급했고, 모델은 그 공짜 크레딧을 성능으로 계상해왔다(헤드라인 AUC 0.83~0.85 중
    상당분). 게이트를 없애면 모델이 held_up을 스스로 예측해야 해서 그 크레딧이 사라진다.
    부수적으로 local_extreme의 30분 지연확인에서 오던 UX 문제("신호가 갑자기 과거기록과 함께
    나타남")와 경제성 백테스트의 진입시점 비현실성도 함께 해소된다 -- 이제 현재 봉에 점수가
    나오므로 다음 봉 진입이 정직한 가정이 된다.

    `triggers` 컬럼은 유지하되 **표시 전용 정보**로 격하된다(어느 트리거가 겹쳤는지 참고용,
    채점 대상 선정에는 더 이상 관여하지 않음). 발동이 없으면 빈 문자열."""
    n = len(frame)
    lo_i = max(0, n - n_tail)
    low, high = frame["low"].to_numpy(), frame["high"].to_numpy()

    down = {name: sig[f"bottom_{name}"].fillna(False).to_numpy() for name in NAMED_TRIGGERS}
    up = {name: sig[f"top_{name}"].fillna(False).to_numpy() for name in NAMED_TRIGGERS}
    W = LOCAL_EXTREME_W
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            local_low[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            local_high[i] = True
    down["local_extreme"] = local_low
    up["local_extreme"] = local_high

    level_low = frame["sweep_level_low"].to_numpy()
    level_high = frame["sweep_level_high"].to_numpy()
    atr = frame["atr"].to_numpy()
    delta_z = frame["delta_z"].to_numpy()

    rows = []
    for is_down, triggers in ((True, down), (False, up)):
        for i in range(lo_i, n):
            level = level_low[i] if is_down else level_high[i]
            penetration = (level - low[i]) if is_down else (high[i] - level)
            fired = sorted(name for name, arr in triggers.items() if arr[i])
            rows.append({
                "pos": i, "timestamp": frame["timestamp"].iloc[i], "is_downside": int(is_down),
                "sweep_penetration_atr": penetration / atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan,
                "flow_aligned_delta_z": delta_z[i] if is_down else -delta_z[i],
                "triggers": ",".join(fired),
            })
    return pd.DataFrame(rows)


def _call_end_pos(pos: int, direction: str, frame: pd.DataFrame, last_pos: int) -> int:
    """콜 하나가 표시되는 **마지막 봉 위치**. 익절에 닿은 봉, 없으면 호라이즌 끝(확정봉 범위 내).

    증거신호 8종 칩의 `_fill_until_tp_or_horizon`
    (live_evidence_signal_dashboard_20260823.py:656)과 같은 계약이다 -- 배지와 히스토리 스트립
    **둘 다** 이 함수 하나를 기준으로 삼는다("콜이 얼마나 지속되는가"의 정의는 한 곳에만 둔다).

    익절 판정은 이 신호 자신의 라벨 정의를 그대로 쓴다 -- 라벨의 빠른 다리가
    `fast_close_max - extreme >= ATR_MULT * pre_atr`(종가 기준, 앵커는 그 봉의 저가/고가)이므로
    여기서도 **종가 기준**으로 본다. 증거신호는 intrabar 고가/저가 터치를 쓰지만 그건 그쪽 라벨이
    그렇게 정의돼서다 -- 컨벤션을 신호 간에 옮기지 않는다(CLAUDE.md 배리어 컨벤션 항목).

    ⚠️룩어헤드 없음: `frame`은 `_fetch_klines()`가 형성중 봉을 이미 버린 **확정 봉만** 담고
    있고, 여기서 보는 close[pos+1..horizon_end]는 전부 과거의 확정 봉이다."""
    horizon_end = min(pos + BADGE_HORIZON_BARS, last_pos)
    atr = frame["atr"].to_numpy()
    pre_atr = atr[pos - 1] if pos >= 1 else np.nan
    if not np.isfinite(pre_atr) or pre_atr <= 0:
        return horizon_end  # ATR을 못 구하면 호라이즌으로만 판정
    close = frame["close"].to_numpy()
    if direction == "down":  # 지지쪽 -- 위로 반등하는 게 익절
        target = frame["low"].to_numpy()[pos] + BADGE_ATR_MULT * pre_atr
        for b in range(pos + 1, horizon_end + 1):
            if close[b] >= target:
                return b
    else:                    # 저항쪽 -- 아래로 반전하는 게 익절
        target = frame["high"].to_numpy()[pos] - BADGE_ATR_MULT * pre_atr
        for b in range(pos + 1, horizon_end + 1):
            if close[b] <= target:
                return b
    return horizon_end


def _call_spans(best_by_pos: dict, frame: pd.DataFrame, last_pos: int,
                threshold: float) -> list[tuple[int, int, dict]]:
    """(시작봉, 끝봉, 콜) 목록을 오래된 순으로. 끝봉은 `_call_end_pos()` 정의."""
    spans = []
    for pos in sorted(best_by_pos):
        b = best_by_pos[pos]
        if b["proba"] < threshold:
            continue
        spans.append((pos, _call_end_pos(pos, b["direction"], frame, last_pos), b))
    return spans


def _predicted_tone(direction: str | None, call: str | None) -> str:
    """direction is which side the candidate leans (down=support-side, expecting an upward
    rebound; up=resistance-side, expecting a downward reversal); call is the model's rebound-vs-
    continuation read. Unchanged by the 2026-08-31 trigger upgrade -- semantics identical, see
    dashboard MODEL_INDICATOR_DETAIL.v_rebound's "[배지 표시]" paragraph for the full history of
    this tone-mapping decision."""
    if call == "rebound" and direction in ("up", "down"):
        return "good" if direction == "down" else "bad"
    if call == "continuation" and direction in ("up", "down"):
        return "flat"
    return "neutral"


def compute_eth_sweep_v_rebound_signal() -> dict:
    """Returns {"warmed_up", "error", "event_active", "call" ("rebound"|"continuation"|None),
    "direction" ("up"|"down"|None), "proba_rebound" (0-1 or None), "minutes_ago",
    "sweep_ts_utc", "price", "tone" ("good"|"bad"|"flat"|"neutral", direction x call resolved via
    _predicted_tone), "history" (oldest-to-newest tone strings, HISTORY_BARS long), "times"
    (matching ISO timestamps), "triggers" (comma-joined names of which of the 9 triggers happen to
    have fired on the CURRENT bar -- display-only since the 2026-09-01 every-bar redesign, no longer
    gates scoring; empty string when none fired). Never raises.

    2026-09-01 매 봉 스코어링 + 배지 지속성 이후 필드 의미(키 이름/타입은 그대로):
      event_active -- "표시할 점수가 있는가"(사실상 warmed_up이면 항상 True).
      minutes_ago  -- **배지가 가리키는 봉의 나이**. 방금 마감된 봉의 콜이면 0, 익절/호라이즌
                      전까지 유지되는 과거 콜이면 그 경과분. 지속성 도입으로 원래 의미 회복.
      sweep_ts_utc -- 배지가 가리키는 봉의 타임스탬프(현재 봉일 수도, 최대 60분 전일 수도).
      early_confirmed -- 항상 None. 익절에 도달한 콜은 배지에서 이미 내려가므로 표시중인
                      콜에는 해당이 없다. 스키마 유지를 위해 키만 남긴다.
      history/times -- **배지와 무관하게 봉별 기록**. 지속성은 배지에만 적용된다."""
    try:
        kl = _fetch_klines(SYMBOL)
        if kl is None or len(kl) < 900:
            return _empty("price_fetch_failed_or_insufficient_history")
        btc_kl = _fetch_klines(BTC_SYMBOL)  # None on failure -- smt_divergence just won't fire this cycle

        frame = _build_features(kl)
        sig = compute_signals(kl, btc_df=btc_kl, funding_df=None)
        # 2026-09-01 매 봉 스코어링: 트리거 게이트 제거. 최근 HISTORY_BARS+1 봉 전부를 양방향으로
        # 채점한다(예측 비용은 예측 행수가 아니라 컨텍스트 크기가 지배 -- 314행이 1행과 같은
        # 시간이었던 실측, research_eth_v_rebound_pool_a_context_size_tabpfn_20260901.py).
        # +BADGE_HORIZON_BARS: 스트립 창 **직전**에 발동한 콜도 창 안으로 칠해져 들어오므로 그만큼
        # 더 거슬러 채점해야 한다. 예측 비용은 예측 행수가 아니라 컨텍스트 크기가 지배하므로
        # (314행이 1행과 같은 시간이었던 실측) 이 확장은 사실상 공짜다.
        candidates = _every_bar_rows(frame, sig, HISTORY_BARS + 1 + BADGE_HORIZON_BARS)
        candidates = candidates.merge(frame[["timestamp"] + [c for c in FEATURES if c not in
                                       ("is_downside", "sweep_penetration_atr", "flow_aligned_delta_z")]],
                                       on="timestamp", how="left")
        candidates = candidates.dropna(subset=FEATURES)
        if candidates.empty:
            return _empty("indicators_not_warmed_up")

        price = float(frame["close"].iloc[-1])

        train = _load_train_context()
        from tabpfn import TabPFNClassifier
        # ignore_pretraining_limits: TabPFN 권장 상한은 1만행인데 이 컨텍스트는 18,000행이다
        # (전체 봉 TRAIN 182,969행에서 무작위 추출 -- 층화/event-first보다 나음이 실측됨).
        # 검증 파이프라인도 처음부터 이 플래그를 썼다.
        clf = TabPFNClassifier(device="cuda", random_state=20260829,
                                ignore_pretraining_limits=True)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        proba = clf.predict_proba(candidates[FEATURES])[:, 1]
        candidates = candidates.assign(proba=proba)

        def call_of(p: float) -> str:
            return "rebound" if p >= PROBA_THRESHOLD else "continuation"

        # 매 봉 스코어링에서는 봉마다 bottom/top 두 점수가 나온다. 그 봉의 "그림"은 둘 중 확률이
        # 높은 쪽으로 정한다(현행 칩이 방향 하나 + 확률 하나를 보여주는 계약을 그대로 유지).
        best_by_pos: dict[int, dict] = {}
        for row in candidates.itertuples():
            pos = int(row.pos)
            prev = best_by_pos.get(pos)
            if prev is None or row.proba > prev["proba"]:
                best_by_pos[pos] = {
                    "t0": frame["timestamp"].iloc[pos], "proba": float(row.proba),
                    "direction": "down" if int(row.is_downside) == 1 else "up",
                    "triggers": row.triggers or None,
                }

        # 콜 구간: 발동봉부터 "익절 도달 또는 호라이즌 경과"까지. 배지와 스트립이 같은 정의를
        # 공유한다(증거신호 8종의 _fill_until_tp_or_horizon과 동일 계약).
        last_pos = len(frame) - 1
        spans = _call_spans(best_by_pos, frame, last_pos, PROBA_THRESHOLD)

        # 히스토리 스트립: 콜 구간을 앞으로 칠한다. 겹치면 **나중 콜이 덮어쓴다**(배지가 최근
        # 콜을 보여주는 것과 일치 -- spans가 오래된 순이라 그냥 순서대로 쓰면 된다).
        fill: dict[int, str] = {}
        for start, end, b in spans:
            tone = _predicted_tone(b["direction"], call_of(b["proba"]))
            for q in range(start, end + 1):
                fill[q] = tone

        history, times = [], []
        for pos in range(max(0, len(frame) - HISTORY_BARS), len(frame)):
            times.append(frame["timestamp"].iloc[pos].isoformat())
            if pos in fill:
                history.append(fill[pos])
            else:
                b = best_by_pos.get(pos)
                # 콜 구간 밖: 그 봉 자신의 읽기(대개 '미반등'). 점수 자체가 없으면 중립.
                history.append(_predicted_tone(b["direction"], call_of(b["proba"])) if b else "neutral")

        # 배지: 아직 살아있는(끝봉이 현재 봉 이상인) 콜 중 가장 최근 것. 없으면 현재 봉 자신의
        # 읽기로 떨어진다(대개 '미반등').
        badge_pos = next((start for start, end, _ in reversed(spans) if end >= last_pos), last_pos)

        cur = best_by_pos.get(badge_pos)
        if cur is None:
            return {"warmed_up": True, "error": None, "event_active": False, "call": None,
                    "direction": None, "proba_rebound": None, "minutes_ago": None,
                    "sweep_ts_utc": None, "price": price, "tone": "neutral",
                    "history": history, "times": times, "triggers": None, "early_confirmed": None}

        # minutes_ago: 배지가 가리키는 봉의 나이(현재 봉이면 0) -- 지속성 도입으로 이 필드가
        # 원래 의미를 되찾았다. early_confirmed는 표시중인 콜에는 해당 없음(익절 도달한 콜은
        # 애초에 배지에 안 뜨므로) -- 스키마 유지를 위해 None.
        return {
            "warmed_up": True, "error": None, "event_active": True, "call": call_of(cur["proba"]),
            "direction": cur["direction"],
            "proba_rebound": round(cur["proba"], 4),
            "minutes_ago": int((frame["timestamp"].iloc[last_pos]
                                - frame["timestamp"].iloc[badge_pos]).total_seconds() // 60),
            "sweep_ts_utc": cur["t0"].isoformat(), "price": price,
            "tone": _predicted_tone(cur["direction"], call_of(cur["proba"])),
            "history": history, "times": times, "triggers": cur["triggers"],
            "early_confirmed": None,
        }
    except Exception as e:  # noqa: BLE001 -- never raise, same contract as sibling live_* modules
        return _empty(f"compute_error: {e}")


if __name__ == "__main__":
    import json
    result = compute_eth_sweep_v_rebound_signal()
    history = result.pop("history", [])
    times = result.pop("times", [])
    print(json.dumps(result, indent=2, default=str))
    print(f"history: {len(history)} bars")
