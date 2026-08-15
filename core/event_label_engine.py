"""
이벤트 기반 라벨 생성 엔진 (Event-Driven Label Engine)
================================================================================
"오디세이"류 개별 실험 스크립트와 무관하게, 금융 ML 라벨링 문헌에서 검증된 기법들을
하나로 결합한 범용 학습 라벨 생성 로직. 특정 자산/모델에 종속되지 않고 OHLCV bar
DataFrame(columns: timestamp, open, high, low, close, volume) 하나만 있으면 동작한다.

핵심 구성 요소 (전부 아래 "참고문헌" 절의 실제 문헌에 근거):
  1. 변동성 추정      : EWMA / ATR 기반 로컬 변동성 (배리어 폭을 레짐에 맞게 스케일링)
  2. 이벤트 샘플링    : 전체 bar / CUSUM 필터 / Directional-Change(intrinsic time)
  3. Triple-Barrier   : 동적(변동성 스케일) PT/SL + 수직(시간) 배리어, 표준 3-way 라벨과
                        메타라벨링(이진, 주어진 side의 적중 여부) 두 모드를 하나의 커널로 지원
  4. Trend-Scanning   : 여러 전방 구간에 OLS를 적합해 |t-value|가 최대인 구간을 선택 →
                        고정 구간 편향 없는 연속형 추세 라벨(부호+강도), meta-label의 side
                        후보로도 사용 가능
  5. 표본 가중치      : 배리어 구간이 겹치는 라벨들의 비독립성을 보정하는 concurrency 기반
                        uniqueness 가중치 + 절대수익 기여도 가중치
  6. Purged K-Fold    : 배리어 구간이 test 구간과 겹치는 train 표본을 제거 + embargo,
                        라벨 누수 없는 교차검증 분할

이 저장소의 causal_event_labels.py / generate_oracle_labels.py 와는 별도의, 문헌 원리에서
새로 설계한 독립 구현이다. 기존 스크립트의 자산별 하드코딩된 매직 넘버 대신, 모든 임계값이
설정(config)으로 노출되어 있고 calibrate_barriers()로 데이터에서 직접 보정할 수 있다.

사용 (표준 3-way 라벨):
  from core.event_label_engine import generate_labels, LabelEngineConfig
  labels = generate_labels(df, LabelEngineConfig(event_method='cusum'))

사용 (메타라벨링 2-pass — primary side 결정 후 secondary 적중여부 라벨):
  primary = generate_labels(df, cfg)                       # 1st pass: side 후보 획득
  side = np.sign(primary['trend_tstat']).where(primary['trend_tstat'].abs() > 1.0, 0)
  meta = generate_labels(df, cfg, side=side)                # 2nd pass: 메타라벨(0/1)

참고문헌:
  - Triple-Barrier Method: Lopez de Prado, "Advances in Financial Machine Learning" (2018).
    https://mlfinpy.readthedocs.io/en/latest/Labelling.html
    https://quantstrategy.io/blog/the-triple-barrier-method-revolutionizing-how-we-label/
  - Trend-Scanning Labels: Lopez de Prado, AFML 강의노트 / "Machine Learning for Asset
    Managers". https://random-docs.readthedocs.io/en/latest/implementations/labeling_trend_scanning.html
  - Meta-Labeling: Lopez de Prado (2017, Guggenheim/Cornell).
    https://en.wikipedia.org/wiki/Meta-Labeling
  - CUSUM 필터 기반 이벤트 샘플링 + Triple-Barrier + 딥러닝 (BTC/ETH, 2025):
    "Algorithmic crypto trading using information-driven bars, triple barrier labeling and
    deep learning", Financial Innovation (2025).
    https://ideas.repec.org/a/spr/fininn/v11y2025i1d10.1186_s40854-025-00866-w.html
  - Directional-Change / intrinsic time: https://en.wikipedia.org/wiki/Directional-change_intrinsic_time
    "A Modern Paradigm for Algorithmic Trading" (2025). https://arxiv.org/pdf/2501.06032
    "Adaptive Crypto Trading Using Directional Change and Meta-Learning", Razmi & Barak (2024/2025).
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5017215
  - 동적(변동성 스케일) 배리어: 다수 문헌에서 공통 확인. 예)
    "Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method ... Cryptocurrency
    Markets", Mathematics (MDPI, 2024). https://www.mdpi.com/2227-7390/12/5/780
  - 표본 uniqueness 가중치 / Sequential Bootstrap: Lopez de Prado, AFML Ch.4.
    https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/
  - Purged K-Fold + Embargo: Lopez de Prado, AFML Ch.7 (배리어 구간 중첩으로 인한 라벨
    누수를 막는 교차검증). 동일 계열: https://www.mql5.com/en/articles/19850

의도적으로 구현하지 않은 최신 기법 (근거 문헌은 있으나 복잡도 대비 검증된 이득이 아직
불확실하여 보류 — 필요해지면 이 모듈 위에 얹는 형태로 확장):
  - GA로 pt/sl/max_hold를 직접 탐색 (MDPI 2024, 위 참고문헌) → calibrate_barriers()의
    grid search로 핵심 아이디어(매직넘버 배제)만 단순하게 반영.
  - Multi-scale Granger-causality + MAML 적응형 라벨링(AEDL, MDPI Applied Sciences 2025,
    https://www.mdpi.com/2076-3417/15/24/13204) → 프레임워크 자체가 별도 연구 과제 수준이라 제외.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit


# ════════════════════════════════════════════════════════════════
# 1. 변동성 추정 (Volatility Estimators)
# ════════════════════════════════════════════════════════════════
def ewma_volatility(close: pd.Series, span: int = 100, min_periods: int = 20) -> pd.Series:
    """로그수익률의 EWMA 표준편차. 워밍업 구간(min_periods 미만)은 NaN으로 남겨
    이후 이벤트 샘플링 단계에서 자연스럽게 제외되도록 한다(미래 데이터로 채우지 않음)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.ewm(span=span, min_periods=min_periods).std()


def atr_volatility(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR을 종가 대비 비율로 정규화한 변동성 프록시."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=window, min_periods=window).mean()
    return atr / close


def return_dispersion_volatility(close: pd.Series, window: int = 12, lookback: int = 288) -> pd.Series:
    """window-bar 누적 로그수익률의 표준편차(lookback 구간 기준). 단일 bar EWMA/ATR은 배리어
    폭이 노이즈 한 bar에 좌우된다는 문제가 이 저장소에서 실제로 두 번 독립적으로 진단된 바
    있다 — BTC 'race' triple-barrier(scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py)와
    zigzag corrected-vol 리빌드(scripts/build_btc_5m_zigzag_correctedvol_20260806.py) 모두
    같은 근본원인 문서(project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806)를 인용해 단일 bar
    변동성 대신 이 방식으로 교체했다. 스캘핑처럼 배리어가 여러 bar에 걸쳐 판정될 때는
    ewma_volatility/atr_volatility보다 이 방법을 우선 검토할 것."""
    log_ret = np.log(close / close.shift(1))
    cumret = log_ret.rolling(window).sum()
    return cumret.rolling(lookback, min_periods=window * 2).std()


# ════════════════════════════════════════════════════════════════
# 2. 이벤트 샘플링 (Event Sampling)
# ════════════════════════════════════════════════════════════════
@njit(cache=True)
def _cusum_events_numba(log_ret: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    n = len(log_ret)
    flags = np.zeros(n, dtype=np.bool_)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        s_pos = max(0.0, s_pos + log_ret[i])
        s_neg = min(0.0, s_neg + log_ret[i])
        h = threshold[i]
        if np.isnan(h):
            continue
        if s_pos > h:
            s_pos = 0.0
            flags[i] = True
        elif s_neg < -h:
            s_neg = 0.0
            flags[i] = True
    return flags


def cusum_filter(close: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """AFML Ch.2.5.2.1 CUSUM 필터. threshold는 bar별 배열(예: k * vol)로, 누적 signed
    log-return이 threshold를 넘는 시점만 이벤트로 샘플링해 매 bar를 라벨링할 때 생기는
    극심한 중첩/자기상관을 줄인다. threshold가 NaN인 구간(변동성 워밍업)은 건너뛴다.
    Returns: 이벤트가 발생한 정수 위치(bar index) 배열.
    """
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    flags = _cusum_events_numba(log_ret, threshold)
    return np.flatnonzero(flags)


@njit(cache=True)
def _directional_change_numba(price: np.ndarray, theta: float):
    n = len(price)
    event_idx = np.empty(n, dtype=np.int64)
    event_dir = np.empty(n, dtype=np.int64)
    n_events = 0
    p_ext = price[0]
    trend = 1  # 임의의 초기값(첫 실제 DC 이벤트에서 스스로 교정됨)
    for i in range(1, n):
        p = price[i]
        if trend == 1:
            if p > p_ext:
                p_ext = p
            elif p <= p_ext * (1.0 - theta):
                event_idx[n_events] = i
                event_dir[n_events] = -1
                n_events += 1
                trend = -1
                p_ext = p
        else:
            if p < p_ext:
                p_ext = p
            elif p >= p_ext * (1.0 + theta):
                event_idx[n_events] = i
                event_dir[n_events] = 1
                n_events += 1
                trend = 1
                p_ext = p
    return event_idx[:n_events], event_dir[:n_events]


def directional_change_events(close: np.ndarray, theta: float):
    """Directional-Change intrinsic-time 이벤트 검출 (Tsang et al.).
    theta: DC 확정 임계값(비율, 예 0.004 = 0.4%). 극값 대비 theta만큼 반대로 움직이면
    그 시점에 방향전환(DC)이 "확정"된다. 시간축(bar) 대신 가격 변동 자체를 이벤트로
    샘플링하는 방식으로, crypto에 특화 검증된 사례가 있다(Razmi & Barak 2024/2025).
    Returns: (event_idx, event_dir) — event_dir=+1(상승전환 확정)/-1(하락전환 확정).
    """
    return _directional_change_numba(close, theta)


# ════════════════════════════════════════════════════════════════
# 3. Triple-Barrier 라벨링 (표준 3-way + 메타라벨링 통합 커널)
# ════════════════════════════════════════════════════════════════
@dataclass
class TripleBarrierConfig:
    pt_mult: Optional[float] = 2.0   # 익절 배리어 = pt_mult * vol_at_entry (None = 비활성화)
    sl_mult: Optional[float] = 2.0   # 손절 배리어 = sl_mult * vol_at_entry (None = 비활성화)
    max_hold: int = 48               # 수직 배리어(최대 보유 bar 수)
    min_vol: float = 0.0             # 이 값 미만의 변동성 구간은 이벤트에서 제외(min_ret 필터)


@njit(cache=True)
def _triple_barrier_numba(high, low, close, start_idx, side, vol_at_start, pt_level, sl_level, max_hold):
    """side-보정 수익 공간에서 배리어 접촉을 탐지하는 단일 커널.
    side=+1(표준모드는 항상 +1)이면 물리적 상승=favorable, 하락=adverse.
    side=-1(메타모드 숏 베팅)이면 하락이 favorable로 뒤집힌다 — 이 변환 덕분에 표준/메타
    두 모드를 같은 스캔 루프로 처리한다. 한 bar 안에서 pt/sl이 동시에 조건을 만족하면
    보수적으로 sl(adverse)을 먼저 접촉한 것으로 간주한다(둘 다 걸릴 때 낙관적 순서를
    가정하지 않기 위함 — 대부분의 백테스트 엔진이 쓰는 관례).
    """
    n_events = len(start_idx)
    n = len(close)
    touch_offset = np.zeros(n_events, dtype=np.int64)
    touch_type = np.zeros(n_events, dtype=np.int64)      # +1 pt / -1 sl / 0 timeout
    realized_ret = np.zeros(n_events, dtype=np.float64)  # side-보정 signed return

    for i in range(n_events):
        t0 = start_idx[i]
        p0 = close[t0]
        s = side[i]
        pt_lvl = pt_level[i]
        sl_lvl = sl_level[i]
        end = t0 + max_hold
        if end >= n:
            end = n - 1

        touched = False
        for j in range(t0 + 1, end + 1):
            ret_hi = s * (high[j] / p0 - 1.0)
            ret_lo = s * (low[j] / p0 - 1.0)
            if ret_hi > ret_lo:
                favorable = ret_hi
                adverse = ret_lo
            else:
                favorable = ret_lo
                adverse = ret_hi

            if adverse <= -sl_lvl:
                touch_offset[i] = j - t0
                touch_type[i] = -1
                realized_ret[i] = -sl_lvl
                touched = True
                break
            if favorable >= pt_lvl:
                touch_offset[i] = j - t0
                touch_type[i] = 1
                realized_ret[i] = pt_lvl
                touched = True
                break

        if not touched:
            touch_offset[i] = end - t0
            touch_type[i] = 0
            realized_ret[i] = s * (close[end] / p0 - 1.0)

    return touch_offset, touch_type, realized_ret


def apply_triple_barrier(
    df: pd.DataFrame,
    event_idx: np.ndarray,
    vol: pd.Series,
    config: TripleBarrierConfig,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """event_idx의 각 이벤트에 triple-barrier를 적용한다.

    side=None  → 표준 3-way 라벨 {-1, 0, +1}. 배리어가 대칭이라 "상승 배리어를 먼저
                 접촉했다면 롱, 하락 배리어를 먼저 접촉했다면 숏이 이겼을 경로"를 의미한다.
    side 지정  → 메타라벨링. side(+1/-1, 이벤트별)는 이미 정해진 "베팅 방향"이며, 출력
                 라벨은 {0,1}로 "그 베팅이 옳았는가"만 나타낸다(포지션 크기는 이 라벨의
                 몫이 아니라 이 라벨을 학습한 2차 모델의 몫 — Lopez de Prado meta-labeling).
    """
    n = len(df)
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    vol_arr = vol.to_numpy()

    valid = ~np.isnan(vol_arr[event_idx])
    valid &= (vol_arr[event_idx] >= config.min_vol)
    valid &= (event_idx + 1 < n)  # 최소 1개 이후 bar가 있어야 접촉 판정 가능
    event_idx = event_idx[valid]

    is_meta = side is not None
    if is_meta:
        side_arr = side.to_numpy()[event_idx].astype(np.float64)
        side_valid = ~np.isnan(side_arr) & (side_arr != 0)
        event_idx = event_idx[side_valid]
        side_arr = np.sign(side_arr[side_valid])
    else:
        side_arr = np.ones(len(event_idx), dtype=np.float64)

    event_vol = vol_arr[event_idx]
    pt_mult = config.pt_mult if config.pt_mult is not None else np.inf
    sl_mult = config.sl_mult if config.sl_mult is not None else np.inf
    pt_level = np.full(len(event_idx), pt_mult) * event_vol
    sl_level = np.full(len(event_idx), sl_mult) * event_vol

    touch_offset, touch_type, realized_ret = _triple_barrier_numba(
        high, low, close, event_idx.astype(np.int64), side_arr, event_vol,
        pt_level, sl_level, config.max_hold,
    )

    t1_idx = event_idx + touch_offset
    touch_name = np.array(['timeout', 'pt', 'sl'])[np.where(touch_type == 1, 1, np.where(touch_type == -1, 2, 0))]

    out = pd.DataFrame({
        'event_idx': event_idx,
        'event_time': df['timestamp'].to_numpy()[event_idx],
        't1_idx': t1_idx,
        't1_time': df['timestamp'].to_numpy()[t1_idx],
        'bars_held': touch_offset,
        'side': side_arr,
        'touch_type': touch_name,
        'realized_ret': realized_ret,
        'vol_at_entry': event_vol,
    })

    if is_meta:
        out['label'] = (realized_ret > 0).astype(np.int64)
    else:
        terminal_sign = np.sign(realized_ret)
        out['label'] = np.where(touch_type != 0, touch_type, terminal_sign).astype(np.int64)

    return out


def calibrate_barriers(
    df: pd.DataFrame,
    event_idx: np.ndarray,
    vol: pd.Series,
    pt_mult_grid: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    sl_mult_grid: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
    max_hold_grid: Sequence[int] = (24, 48, 96),
    target_balance: float = 0.30,
    min_events: int = 200,
) -> TripleBarrierConfig:
    """pt/sl/max_hold를 임의로 고정하는 대신, 그리드에서 "타임아웃 비율이 과반을 넘지
    않고, 클래스가 target_balance 이상으로 갈리는" 조합 중 가장 촘촘한(=min_hold가 작아
    표본이 많이 나오는) 설정을 고른다. GA/베이지안 최적화(MDPI 2024)까지는 아니지만,
    "매직넘버를 그냥 손으로 고정"하는 이 저장소의 반복된 실패 패턴(라벨 재검토 시마다
    드러난 문제들)을 피하기 위한 최소한의 데이터 기반 보정.
    """
    close = df['close'].to_numpy()
    best = None
    best_score = -np.inf
    for max_hold in max_hold_grid:
        for pt_mult in pt_mult_grid:
            for sl_mult in sl_mult_grid:
                cfg = TripleBarrierConfig(pt_mult=pt_mult, sl_mult=sl_mult, max_hold=max_hold)
                labels = apply_triple_barrier(df, event_idx, vol, cfg, side=None)
                if len(labels) < min_events:
                    continue
                frac_pt = (labels['label'] == 1).mean()
                frac_sl = (labels['label'] == -1).mean()
                frac_timeout = (labels['touch_type'] == 'timeout').mean()
                min_class_frac = min(frac_pt, frac_sl)
                if min_class_frac < target_balance * 0.5:
                    continue
                # 점수: 클래스 균형이 좋을수록, 타임아웃(정보량 적은 경로)이 적을수록,
                # 보유기간이 짧을수록(회전율 높은 스캘핑에 유리) 우대
                score = min_class_frac - 0.3 * frac_timeout - 0.001 * max_hold
                if score > best_score:
                    best_score = score
                    best = cfg
    if best is None:
        raise ValueError("주어진 그리드에서 min_events/target_balance 조건을 만족하는 배리어 설정을 찾지 못함")
    return best


# ════════════════════════════════════════════════════════════════
# 4. Trend-Scanning 라벨 (고정 구간 편향 없는 추세 방향/강도)
# ════════════════════════════════════════════════════════════════
@njit(cache=True)
def _trend_scan_numba(y: np.ndarray, horizons: np.ndarray):
    n = len(y)
    n_h = len(horizons)
    best_t = np.zeros(n, dtype=np.float64)
    best_slope = np.zeros(n, dtype=np.float64)
    best_horizon = np.zeros(n, dtype=np.int64)

    for i in range(n):
        max_abs_t = -1.0
        for hi in range(n_h):
            L = horizons[hi]
            if i + L > n or L <= 2:
                continue
            sx = 0.0
            sxx = 0.0
            sy = 0.0
            sxy = 0.0
            for k in range(L):
                xv = float(k)
                yv = y[i + k]
                sx += xv
                sxx += xv * xv
                sy += yv
                sxy += xv * yv
            xbar = sx / L
            ybar = sy / L
            sxx_c = sxx - L * xbar * xbar
            if sxx_c <= 1e-12:
                continue
            sxy_c = sxy - L * xbar * ybar
            slope = sxy_c / sxx_c

            sse = 0.0
            for k in range(L):
                xv = float(k)
                yhat = ybar + slope * (xv - xbar)
                resid = y[i + k] - yhat
                sse += resid * resid
            se = np.sqrt(sse / (L - 2) / sxx_c)
            if se <= 1e-12:
                continue
            tstat = slope / se
            if abs(tstat) > max_abs_t:
                max_abs_t = abs(tstat)
                best_t[i] = tstat
                best_slope[i] = slope
                best_horizon[i] = L

    return best_t, best_slope, best_horizon


def trend_scanning_labels(close: pd.Series, horizons: Sequence[int] = (6, 12, 24, 48, 96, 144)) -> pd.DataFrame:
    """Lopez de Prado의 trend-scanning: 각 시점 t에서 t..t+L 구간에 log-price ~ time OLS를
    적합해, 여러 L(horizons) 중 |t-value|가 최대인 구간을 선택한다. 사전에 정해진 하나의
    보유기간에 편향되지 않고, 통계적으로 가장 뚜렷한 추세 구간을 스스로 고르는 것이 핵심.
    Returns: DataFrame(trend_tstat, trend_slope, trend_horizon) — index는 close와 동일.
    구간 끝(len(close)-min(horizons) 이후)은 앞으로 볼 데이터가 부족해 NaN이 될 수 있다.

    ⚠ 반드시 라벨/메타라벨링 side 구성 등 "학습 시점에 오프라인으로만" 쓸 것 — 절대 시점 i의
    실시간 causal feature로 소비하지 말 것. index i의 값은 i..i+L(선택된 horizon)의 미래
    구간을 회귀해서 나온 값이라 시점 i에는 원천적으로 존재할 수 없는 정보다. 이 저장소에서
    바로 이 지점에서 실제 leakage 버그가 반복 발생했다: 최소 3개의 독립 trend-scan 커널이
    구간의 "끝" 인덱스가 아니라 "시작" 인덱스에 결과를 기록해, 그 산출물이(라벨이 아니라)
    "전체 feature parquet"에 실시간 피처처럼 섞여 들어갔다가 2026-08-04에야 발견·수정됐다
    (Sigma3/Sigma6/Sigma9 라인, `scripts/build_btc_1h_trendscan_causal_fix_20260804.py`).
    """
    log_price = np.log(close.to_numpy())
    horizons_arr = np.array(sorted(horizons), dtype=np.int64)
    t_stat, slope, horizon = _trend_scan_numba(log_price, horizons_arr)

    out = pd.DataFrame({
        'trend_tstat': t_stat,
        'trend_slope': slope,
        'trend_horizon': horizon,
    }, index=close.index)
    out.loc[out['trend_horizon'] == 0, ['trend_tstat', 'trend_slope']] = np.nan
    return out


# ════════════════════════════════════════════════════════════════
# 5. 표본 가중치 (겹치는 배리어 구간의 비독립성 보정)
# ════════════════════════════════════════════════════════════════
def _bar_concurrency(event_idx: np.ndarray, t1_idx: np.ndarray, n_bars: int) -> np.ndarray:
    """각 bar를 몇 개의 이벤트 구간 [event_idx, t1_idx]가 덮고 있는지(concurrency)."""
    diff = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(diff, event_idx, 1.0)
    np.add.at(diff, t1_idx + 1, -1.0)
    return np.cumsum(diff)[:n_bars]


def sample_uniqueness_weights(event_idx: np.ndarray, t1_idx: np.ndarray, n_bars: int) -> np.ndarray:
    """AFML Ch.4: 라벨 i의 가중치 = 그 구간 [t0,t1] 동안의 평균 (1/concurrency).
    다른 라벨과 구간이 많이 겹칠수록(=독립 정보가 적을수록) 가중치가 낮아진다."""
    concurrency = _bar_concurrency(event_idx, t1_idx, n_bars)
    weights = np.empty(len(event_idx), dtype=np.float64)
    for i in range(len(event_idx)):
        span = concurrency[event_idx[i]:t1_idx[i] + 1]
        weights[i] = np.mean(1.0 / np.maximum(span, 1.0))
    return weights


def return_attribution_weights(close: np.ndarray, event_idx: np.ndarray, t1_idx: np.ndarray) -> np.ndarray:
    """AFML Ch.4 "absolute return attribution": 구간이 겹치는 여러 라벨에 그 bar의 수익
    기여분을 concurrency로 나눠 분배한 뒤 절대값을 합산 — 가격이 크게 움직인 구간을
    포착한 라벨일수록(단순 시간 길이가 아니라) 더 큰 가중치를 받는다."""
    n_bars = len(close)
    concurrency = _bar_concurrency(event_idx, t1_idx, n_bars)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    weights = np.empty(len(event_idx), dtype=np.float64)
    for i in range(len(event_idx)):
        s, e = event_idx[i], t1_idx[i]
        span_ret = log_ret[s + 1:e + 1]
        span_conc = np.maximum(concurrency[s + 1:e + 1], 1.0)
        weights[i] = np.abs(np.sum(span_ret / span_conc))
    return weights


def combine_sample_weights(uniqueness: np.ndarray, return_attr: np.ndarray) -> np.ndarray:
    """두 가중치를 곱한 뒤 평균 1로 정규화(트레이너의 sample_weight에 바로 투입 가능)."""
    combined = uniqueness * (return_attr + 1e-12)
    return combined / combined.mean()


# ════════════════════════════════════════════════════════════════
# 6. Purged K-Fold + Embargo (배리어 구간 중첩으로 인한 라벨 누수 방지)
# ════════════════════════════════════════════════════════════════
def purged_kfold_splits(
    event_idx: np.ndarray,
    t1_idx: np.ndarray,
    n_bars: int,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
):
    """AFML Ch.7: 각 fold를 test로 쓸 때, [event_idx,t1_idx] 구간이 test 구간과 겹치는
    train 표본을 전부 제거(purge)하고, test 구간 직후 embargo_frac*n_bars bar를 추가로
    train에서 제외한다. 배리어 라벨은 결과가 미래로 t1까지 걸쳐 있으므로, 이 처리 없이
    표준 K-fold를 쓰면 test 정보가 인접 train fold로 새어 들어간다.
    Yields: (train_mask, test_mask) — 둘 다 이벤트 배열과 같은 길이의 bool 배열.
    """
    n_events = len(event_idx)
    order = np.argsort(event_idx)
    event_idx = event_idx[order]
    t1_idx = t1_idx[order]
    embargo = int(n_bars * embargo_frac)

    fold_bounds = np.linspace(0, n_events, n_splits + 1).astype(int)
    for k in range(n_splits):
        test_start, test_end = fold_bounds[k], fold_bounds[k + 1]
        test_mask = np.zeros(n_events, dtype=bool)
        test_mask[test_start:test_end] = True

        test_bar_lo = event_idx[test_start]
        test_bar_hi = t1_idx[test_start:test_end].max() + embargo

        train_mask = ~test_mask
        overlap = (event_idx < test_bar_hi) & (t1_idx >= test_bar_lo)
        train_mask &= ~overlap

        inv_order = np.argsort(order)
        yield train_mask[inv_order], test_mask[inv_order]


# ════════════════════════════════════════════════════════════════
# 7. 오케스트레이터
# ════════════════════════════════════════════════════════════════
@dataclass
class LabelEngineConfig:
    event_method: Literal['all_bars', 'cusum', 'directional_change'] = 'cusum'
    cusum_k: float = 1.0            # cusum 임계값 = cusum_k * vol
    dc_theta: float = 0.004         # directional-change 확정 임계값(비율)
    vol_method: Literal['ewma', 'atr', 'return_dispersion'] = 'ewma'
    vol_span: int = 100             # ewma/atr의 span·window. return_dispersion에서는 lookback으로 쓰임
    vol_window: int = 12            # return_dispersion 전용: 누적수익을 구할 내부 window(bar)
    barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    trend_horizons: tuple = (6, 12, 24, 48, 96, 144)
    compute_trend_scan: bool = True
    compute_weights: bool = True


def generate_labels(
    df: pd.DataFrame,
    config: Optional[LabelEngineConfig] = None,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """OHLCV DataFrame(columns: timestamp, open, high, low, close, volume) 하나로부터
    전체 라벨 파이프라인(변동성 → 이벤트 샘플링 → triple-barrier → trend-scan →
    표본가중치)을 실행해 이벤트당 1행짜리 라벨 DataFrame을 반환한다.

    side=None이면 표준 3-way 라벨(label ∈ {-1,0,+1})을, side를 주면(이벤트 시각과 같은
    인덱스를 가진 +1/-1 Series) 메타라벨(label ∈ {0,1})을 생성한다.

    주의(메타라벨링, side가 "학습된 1차 모델"의 예측일 때만 해당 — 이 데모처럼 trend-scan
    같은 비학습 통계량이면 해당 없음): side를 그 모델이 같은 구간에서 in-sample로 학습한
    예측치로 채우면 1차 모델의 과적합 확신이 그대로 메타라벨에 새는 leakage가 된다
    (Lopez de Prado AFML Ch.3). 이 저장소에서 가장 엄밀하게 처리한 사례
    (`scripts/train_eval_scalp_1m_meta_label_20260716.py`)는 purged CV로 만든
    out-of-fold 예측만 side로 사용한다 — 실제 학습된 모델로 메타라벨링할 때는
    `purged_kfold_splits()`로 fold를 나눠 fold별로 학습한 뒤, 그 fold의 test 구간에 대한
    예측만 모아 side를 구성할 것.
    """
    config = config or LabelEngineConfig()
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df = df.reset_index(drop=True)
    n = len(df)
    close = df['close']

    if config.vol_method == 'ewma':
        vol = ewma_volatility(close, span=config.vol_span)
    elif config.vol_method == 'atr':
        vol = atr_volatility(df['high'], df['low'], close, window=config.vol_span)
    elif config.vol_method == 'return_dispersion':
        vol = return_dispersion_volatility(close, window=config.vol_window, lookback=config.vol_span)
    else:
        raise ValueError(f"알 수 없는 vol_method: {config.vol_method}")

    if config.event_method == 'all_bars':
        event_idx = np.arange(n)
    elif config.event_method == 'cusum':
        threshold = (config.cusum_k * vol).to_numpy()
        event_idx = cusum_filter(close.to_numpy(), threshold)
    elif config.event_method == 'directional_change':
        event_idx, _ = directional_change_events(close.to_numpy(), config.dc_theta)
    else:
        raise ValueError(f"알 수 없는 event_method: {config.event_method}")

    labels = apply_triple_barrier(df, event_idx, vol, config.barrier, side=side)

    if config.compute_trend_scan:
        trend = trend_scanning_labels(close, horizons=config.trend_horizons)
        labels = labels.merge(
            trend.reset_index().rename(columns={'index': 'event_idx'}),
            on='event_idx', how='left',
        )

    if config.compute_weights and len(labels) > 0:
        event_arr = labels['event_idx'].to_numpy()
        t1_arr = labels['t1_idx'].to_numpy()
        w_uniq = sample_uniqueness_weights(event_arr, t1_arr, n)
        w_ret = return_attribution_weights(close.to_numpy(), event_arr, t1_arr)
        labels['weight_uniqueness'] = w_uniq
        labels['weight_return_attr'] = w_ret
        labels['weight'] = combine_sample_weights(w_uniq, w_ret)

    return labels


# ════════════════════════════════════════════════════════════════
# 8. CLI / 검증
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os
    import time

    print("=" * 70)
    print("이벤트 기반 라벨 생성 엔진 — 검증 실행")
    print("=" * 70)

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'eth_5m_1year.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"\n실데이터 로드: {data_path}")
    else:
        print("\n실데이터 없음 — 합성 데이터로 대체")
        np.random.seed(42)
        n = 20000
        close = 3000 + np.cumsum(np.random.randn(n) * 5)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'open': close, 'high': close + np.abs(np.random.randn(n) * 5),
            'low': close - np.abs(np.random.randn(n) * 5), 'close': close,
            'volume': np.random.exponential(1000, n),
        })

    print(f"bar 수: {len(df):,}  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    for event_method in ['cusum', 'directional_change', 'all_bars']:
        cfg = LabelEngineConfig(event_method=event_method, barrier=TripleBarrierConfig(pt_mult=2.0, sl_mult=2.0, max_hold=48))
        t0 = time.time()
        labels = generate_labels(df, cfg)
        elapsed = time.time() - t0

        print(f"\n--- event_method={event_method} ---")
        print(f"이벤트 수: {len(labels):,}  (실행 {elapsed:.2f}s)")
        if len(labels) == 0:
            continue
        dist = labels['label'].value_counts(normalize=True).sort_index()
        print(f"라벨 분포: {dict(dist.round(3))}")
        print(f"평균 보유 bar: {labels['bars_held'].mean():.1f}  touch_type 분포: {dict(labels['touch_type'].value_counts(normalize=True).round(3))}")
        if 'weight' in labels.columns:
            print(f"가중치(uniqueness) 평균: {labels['weight_uniqueness'].mean():.3f}  (1.0=완전 독립, 낮을수록 배리어 구간이 많이 겹침)")
        if 'trend_tstat' in labels.columns:
            print(f"trend_tstat 통계: mean={labels['trend_tstat'].mean():.2f} std={labels['trend_tstat'].std():.2f}")

    # vol_method 비교 데모: 단일-bar(ewma) vs 다중-bar 누적수익 분산(return_dispersion)
    print(f"\n--- vol_method 비교 (event_method=cusum 고정) ---")
    for vol_method in ['ewma', 'return_dispersion']:
        cfg = LabelEngineConfig(event_method='cusum', vol_method=vol_method,
                                 barrier=TripleBarrierConfig(pt_mult=2.0, sl_mult=2.0, max_hold=48))
        labels = generate_labels(df, cfg)
        print(f"vol_method={vol_method}: 이벤트 수={len(labels):,}  평균 vol_at_entry={labels['vol_at_entry'].mean():.5f}  평균 보유={labels['bars_held'].mean():.1f}bar")

    # 메타라벨링 2-pass 데모: 1차 CUSUM+trend-scan → side 결정 → 2차 메타라벨
    print(f"\n--- 메타라벨링 2-pass 데모 ---")
    cfg = LabelEngineConfig(event_method='cusum')
    primary = generate_labels(df, cfg)
    vol = ewma_volatility(df['close'], span=cfg.vol_span)
    side_full = pd.Series(np.nan, index=df.index)
    side_full.loc[primary['event_idx']] = np.where(
        primary['trend_tstat'].abs() > 1.0, np.sign(primary['trend_tstat']), np.nan
    )
    meta = generate_labels(df, cfg, side=side_full)
    print(f"1차 이벤트 수: {len(primary):,}  side 확정(|t|>1.0) 후 2차(메타) 이벤트 수: {len(meta):,}")
    if len(meta) > 0:
        print(f"메타라벨 적중률(label==1 비율): {(meta['label'] == 1).mean():.3f}")

    # calibrate_barriers 데모
    print(f"\n--- calibrate_barriers 데모 (CUSUM 이벤트 기준) ---")
    threshold = (cfg.cusum_k * vol).to_numpy()
    event_idx = cusum_filter(df['close'].to_numpy(), threshold)
    best_cfg = calibrate_barriers(df, event_idx, vol)
    print(f"보정된 배리어: pt_mult={best_cfg.pt_mult} sl_mult={best_cfg.sl_mult} max_hold={best_cfg.max_hold}")

    # purged K-fold 데모
    print(f"\n--- purged_kfold_splits 데모 ---")
    for k, (train_mask, test_mask) in enumerate(purged_kfold_splits(
        primary['event_idx'].to_numpy(), primary['t1_idx'].to_numpy(), len(df), n_splits=5,
    )):
        print(f"fold {k}: train={train_mask.sum():,} test={test_mask.sum():,} purge로 제외={(~train_mask & ~test_mask).sum():,}")
