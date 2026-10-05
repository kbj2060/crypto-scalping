"""«시장 맥락» 카드 계산부 (2026-09-29). 서버는 원천을 모아 이 함수들에 넣고 결과를 그대로 보낸다.

왜: ETH 오더플로 레포트 105항목을 대시보드와 대조한 뒤 사용자 지시 «불합격된 것들도 쓸모 있으면 넣어줘 --
  우리 검증 방식이 틀렸을 수도 있다». 그래서 이 카드의 값은 **전부 서술**이다(방향 신호 아님). 칸마다
  우리 검정 판정은 화면의 «?» 설명이 말로 한다(DESIGN.md: 매매 화면에 연구 원문 숫자를 올리지 않는다).
원천: 바이낸스 markPrice@1s·OI·래스터(이미 이 프로세스에 있다) + OKX·HL 맥락 수집기 duckdb + 봇 롱숏비 duckdb.
ponytail: 부호 규약 -- 매수·매수벽·롱 쪽이 양수(micro_ref.py 와 같다).
"""
from __future__ import annotations

import math
from bisect import bisect_right
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np

SWEEP_BPS = (25, 50, 100)  # «여기까지 쓸어올리려면 몇 ETH 가 필요한가» -- 현재 미드에서 한쪽으로.
# 🔴±10bp 는 뺐다: 칸($0.5 ≈ 1.8bp)이 미드를 품은 칸에서 매수·매도를 상계하므로 10bp 띠는 오차가 20% 넘게 난다(09-29 서버에서
#   REST 호가 1000단계와 같은 순간 3회 대조: ±25bp 는 ±3% 안, ±10bp 매도는 −23~−26%). ±50bp 넘게는 REST 1000단계가 못 덮는다(±40bp).
HL_LIQ_BIN_USD = 5.0       # HL 고래 청산가 묶음 폭($). ETH 2,700 에서 ~1.9bp
HL_LIQ_MAX_PCT = 0.10      # 이보다 먼 청산가는 버린다(±10%)
LIQ_PROFILE_BIN_USD = 2.5  # 실측 청산 가격 프로파일 묶음 폭($)
PROFILE_BIN_FRAC = 5e-4    # 거래량 프로파일 칸 = 가격의 0.05%(연구 규약 -- ETH 2,600 에서 $1.3)
NODE_WIN, NODE_SIG, NODE_MAX = 6, 2.0, 8   # HVN/LVN: 평활 σ 2칸 · ±6칸(±0.3%) 국소 극값 · 종류마다 최대 8개


def pct_rank(x: float | None, hist: Iterable[float], min_n: int = 30) -> float | None:
    """x 가 hist 에서 몇 분위인가(0~1, 자기 이하 비율). 표본이 min_n 미만이면 None(판정 보류)."""
    a = np.asarray([v for v in hist if v is not None and math.isfinite(v)], dtype=float)
    if x is None or not math.isfinite(x) or a.size < min_n:
        return None
    return float(np.count_nonzero(a <= x) / a.size)


def sweep_depth(row: np.ndarray, px: np.ndarray, mid: float, bps: tuple[int, ...] = SWEEP_BPS,
                top: tuple[float, float, float, float] | None = None) -> dict[str, list[float]]:
    """래스터 한 초(row: +매수호가 / −매도호가 수량, px: 칸 **아래끝** 가격 -- 수집기가 floor(p/칸)로 넣는다) → 미드에서 ±bp 까지
    걸린 호가 합(ETH). ask[i] = 위로 bps[i] 까지 쓸어올리는 데 먹어야 할 매도호가 · bid[i] = 아래로 같은 폭의 매수호가.
    🔴미드를 품은 칸은 **매수·매도가 상계**돼 있다(수집기: «스프레드가 걸친 한 빈만 상계») -- 그 칸은 빼고, 그 칸 안의 최우선
      호가는 top = (최우선 매수가, 수량, 최우선 매도가, 수량)(bookTicker)으로 정확히 더한다. 먼 쪽 경계는 칸 **가운데**로 판정한다.
      (09-29 첫 판은 상계된 칸을 매수에 통째로 넣어 ±10bp 매수가 바이낸스 REST 호가보다 20~30% 작았다 -- 같은 순간 3회 대조)
    취소·재보충은 모른다(한 초 스냅샷) -- «지금 걸려 있는 양»이다."""
    bs = float(px[1] - px[0]) if len(px) > 1 else 0.5
    ctr = px + bs / 2
    s_lo = np.floor(mid / bs) * bs                      # 미드를 품은(상계된) 칸의 아래끝
    out: dict[str, list[float]] = {"bid": [], "ask": []}
    for bp in bps:
        w = mid * bp / 1e4
        up = (px >= s_lo + bs) & (ctr <= mid + w) & (row < 0)
        dn = (px + bs <= s_lo) & (ctr >= mid - w) & (row > 0)
        a, b = float(-row[up].sum()), float(row[dn].sum())
        if top:
            bpx, bq, apx, aq = top
            b += bq if bpx >= s_lo else 0.0            # 최우선 호가가 상계 칸 안일 때만(밖이면 위 합에 이미 들어 있다)
            a += aq if apx < s_lo + bs else 0.0
        out["ask"].append(a)
        out["bid"].append(b)
    return out


def quad_1h(move_bp: float | None, doi: float | None, p75: float | None) -> dict[str, str] | None:
    """1시간 가격 × OI 사분면. flow_read ③ 과 **같은 기준**(|이동| 이 하루 1시간 이동의 p75 이상일 때만 판정).
    note = 우리 검정의 말(1시간·청산 프레임에서만 근거, 상승 쪽 두 칸은 미측정)."""
    if move_bp is None or doi is None:
        return None
    if p75 is None or abs(move_bp) < p75:
        return {"key": "small", "label": "이동이 작다 — 판정 보류", "note": "1시간 이동이 하루 상위 25% 밖"}
    if move_bp < 0 and doi > 0:
        return {"key": "dn_up", "label": "하락 + OI↑ · 새 숏 유입", "note": "숏은 서두르지 않는 쪽(근거)"}
    if move_bp < 0:
        return {"key": "dn_dn", "label": "하락 + OI↓ · 롱 정리", "note": "숏은 거둘 쪽(근거)"}
    if doi > 0:
        return {"key": "up_up", "label": "상승 + OI↑ · 새 롱 유입", "note": "미측정 · 서술"}
    return {"key": "up_dn", "label": "상승 + OI↓ · 숏 커버", "note": "미측정 · 서술"}


OI_Z_MAD_SCALE = 1.55


def oi_stats(series: list[tuple[int, float]], step_s: int = 300) -> dict[str, float | None]:
    """5분 격자 OI [(봉 시각, OI)] → 1시간·24시간 변화율(%)과 1시간 변화의 z(과거 1시간 변화들 대비).
    격자에 구멍이 있으면 그 칸은 NaN 으로 두어 변화가 구멍을 건너뛰지 않게 한다."""
    out: dict[str, float | None] = {"oi": None, "d1h_pct": None, "d24h_pct": None, "z1h": None}
    if not series:
        return out
    t0, t1 = series[0][0], series[-1][0]
    grid = np.full((t1 - t0) // step_s + 1, np.nan)
    for t, v in series:
        grid[(t - t0) // step_s] = v
    out["oi"] = float(grid[-1])
    k1, k24 = 3600 // step_s, 86400 // step_s
    if grid.size > k1 and math.isfinite(grid[-1 - k1]):
        out["d1h_pct"] = float((grid[-1] / grid[-1 - k1] - 1) * 100)
    if grid.size > k24 and math.isfinite(grid[-1 - k24]):
        out["d24h_pct"] = float((grid[-1] / grid[-1 - k24] - 1) * 100)
    d = grid[k1:] / grid[:-k1] - 1 if grid.size > k1 else np.array([])
    # 2026-10-05 요일유형 분리(사용자 «주말에 너무 작게»): 7일 중 5일이 평일이라 주말 |z|≥1.5 가 평일의 1/3
    #   (8주 실측 12.6 vs 4.0%) → 지금과 같은 요일유형(UTC 토·일 / 평일)의 과거 변화만 분포로(평일유형 14일판 9.8 vs 8.6%).
    #   그 유형 표본이 72 미만이면 전체로.
    wk = ((t0 + (np.arange(d.size) + k1) * step_s) // 86400 + 3) % 7 >= 5
    same = d[:-1][np.isfinite(d[:-1]) & (wk[:-1] == wk[-1])] if d.size else np.array([])
    past = same if same.size >= 72 else d[:-1][np.isfinite(d[:-1])]
    # 2026-10-06 평균·표준편차 -> 중앙값·MAD(사용자 «저번 주 같은 요일에 이벤트가 있었으면 이번 주는 작게 나오나?» -> «그렇게 해줘»).
    #   7일 창에 큰 급변(상위 0.1%)이 끼면 표준편차가 부풀어 같은 크기 변화의 |z|≥1.5 가 ETH 99→54% · SOL 97→38% · XRP 94→30%
    #   (2025-01~2026-09, 같은 크기 = 상위 1~5% 변화). MAD 는 84 · 80 · 69% 로 덜 눌린다.
    #   MAD z 는 꼬리가 두꺼워 같은 문턱이면 2배 자주 뜬다 -> 배율 OI_Z_MAD_SCALE 로 전체 빈도(|z|≥1.5 ~10% · z≥1 ~10% · z≤-1.5 ~5%)를
    #   옛 식과 맞춘다(문턱별·코인별·기간별 1.48~1.68, 2025~ 세 코인 평균). scratchpad oi_z_robust.py.
    med = float(np.median(past)) if past.size else 0.0
    mad = 1.4826 * float(np.median(np.abs(past - med))) if past.size else 0.0
    mad = max(mad, 0.25 * float(past.std())) if past.size else mad   # 하한: OI 피드가 일부(30~49%) 멈춰 0 변화가 쌓이면 MAD 만 0 근처로 줄어 |z| 가 폭증 -- 정상 분포에선 거의 안 걸린다
    if d.size and math.isfinite(d[-1]) and past.size >= 72 and mad > 0:
        out["z1h"] = float((d[-1] - med) / (OI_Z_MAD_SCALE * mad))
    return out


def oi_series_live(series: list[tuple[int, float]], sec: int, value: float,
                   step_s: int = 300) -> list[tuple[int, float]]:
    """2026-09-30: 7일 OI 격자(5분 캐시, 최대 30분 낡음) 끝에 **라이브 OI** 를 그 5분 칸의 끝값으로 붙인다(같은 칸이면 교체).
    옛 판은 격자 끝값을 «지금 OI»로 써 1시간 변화·z 가 캐시 나이만큼 낡았다. 기준(1시간 전 칸·과거 분포)은 격자 그대로."""
    b = sec // step_s * step_s
    return [r for r in series if r[0] < b] + [(b, value)]


def oi_sum_series(per: dict[str, list[tuple[int, float]]], since: int, tol_s: int = 3600) -> tuple[list[tuple[int, float]], list[str]]:
    """거래소별 5분 OI [(봉 시각, OI)] → 합 시계열과 넣은 거래소(2026-10-06 사용자 «바로 합산»).
    창 시작(since) 부근(tol_s 안)부터 덮는 거래소만 넣는다 -- 새 거래소(Bybit 10-06~)는 7일 쌓인 뒤 저절로 합류한다
    (안 그러면 z 의 과거 분포가 그 거래소 없이 만들어져 합류 순간 «급증»으로 읽힌다). 바이낸스가 없으면 빈 결과.
    한 거래소라도 빈 칸은 합에서 뺀다 -- oi_stats 가 구멍 칸을 NaN 으로 두어 변화가 구멍을 건너뛰지 않는다."""
    use = [k for k, rows in per.items() if rows and rows[0][0] <= since + tol_s]
    if "binance" not in use:
        return [], []
    maps = [dict(per[k]) for k in use]
    ts = sorted(set.intersection(*(set(m) for m in maps)))
    return [(t, sum(m[t] for m in maps)) for t in ts], use


def ring_at(ring: dict, sec: int, fallback, tol_s: int = 2):
    """2026-09-30: 초별 링(sec→값)에서 sec 에 가장 가까운 값(±tol_s초). 없으면 fallback(지금 값).
    거래소 간 가격차를 **같은 순간**끼리 재려고 -- 30초 묵은 HL/OKX 표본을 바이낸스 «지금»과 견주면
    그 사이 바이낸스 움직임이 가격차로 섞였다(30초 새 −6.1 → +11.4bp)."""
    for d in sorted(range(-tol_s, tol_s + 1), key=abs):
        if sec + d in ring:
            return ring[sec + d]
    return fallback


def lev_state(basis_pct: float | None, oi_z: float | None) -> dict[str, str]:
    """레버리지 상태 한 줄. 🔴바이낸스 펀딩은 평온장에서 0.01% 에 클램프돼 거의 안 움직인다(2025~26 실측 최댓값 = 0.01%)
    -- 그래서 «비싸게 들고 있나»는 펀딩이 아니라 베이시스(마크−인덱스)의 7일 분위로 본다. 방향 신호 아님(극단 펀딩 검정 0/10)."""
    hot = oi_z is not None and oi_z >= 1.0
    if basis_pct is None:
        return {"key": "na", "label": "베이시스 이력 쌓는 중"}
    if basis_pct >= 0.9 and hot:
        return {"key": "long_crowd", "label": "롱 쏠림 — 프리미엄 높고 OI 급증"}
    if basis_pct <= 0.1 and hot:
        return {"key": "short_crowd", "label": "숏 쏠림 — 할인 깊고 OI 급증"}
    if oi_z is not None and oi_z <= -1.5:
        return {"key": "deleverage", "label": "레버리지 빠지는 중 — OI 급감"}
    if basis_pct >= 0.9:
        return {"key": "premium", "label": "선물 프리미엄 높음"}
    if basis_pct <= 0.1:
        return {"key": "discount", "label": "선물 할인 깊음"}
    return {"key": "normal", "label": "보통"}


def hl_liq_levels(rows: Iterable[tuple[float, float]], mid: float, bin_usd: float = HL_LIQ_BIN_USD,
                  top: int = 3, max_pct: float = HL_LIQ_MAX_PCT) -> dict[str, list[dict[str, float]]]:
    """HL 고래 포지션 [(szi, liq_px)] → 가격 묶음별 청산 예정 명목($). below = 롱 청산가(가격 아래), above = 숏.
    각 쪽 금액 상위 top 개를 **가까운 순**으로. 추정이 아니라 거래소가 준 실제 청산가다(추적 300주소만)."""
    agg: dict[tuple[str, float], list[float]] = {}
    for szi, liq in rows:
        if not (szi and liq and liq > 0) or abs(liq / mid - 1) > max_pct:
            continue
        side = "below" if szi > 0 else "above"
        if (side == "below") != (liq < mid):
            continue                     # 이미 넘어선 청산가(곧 사라질 행) -- 방향이 뒤집힌 건 버린다
        k = (side, round(round(liq / bin_usd) * bin_usd, 10))   # 10자리: XRP 칸(0.003)의 부동소수 꼬리(1.4970000000000001)를 뗀다
        a = agg.setdefault(k, [0.0, 0])
        a[0] += abs(szi) * mid
        a[1] += 1
    out: dict[str, list[dict[str, float]]] = {"below": [], "above": []}
    for side in out:
        lv = sorted(((px, a) for (s, px), a in agg.items() if s == side), key=lambda t: -t[1][0])[:top]
        out[side] = [{"px": px, "usd": round(a[0]), "n": a[1]} for px, a in sorted(lv, key=lambda t: abs(t[0] - mid))]
    return out


def liq_profile(events: Iterable[tuple[float, float, bool]], bin_usd: float = LIQ_PROFILE_BIN_USD) -> list[list[float]]:
    """실측 청산 [(체결가, USD, 롱청산?)] → [[가격, 롱USD, 숏USD]] 가격 오름차순. 추정 청산맵과 따로 그린다."""
    agg: dict[float, list[float]] = {}
    for px, usd, is_long in events:
        if px > 0 and usd > 0:
            a = agg.setdefault(round(round(px / bin_usd) * bin_usd, 10), [0.0, 0.0])
            a[0 if is_long else 1] += usd
    return [[k, round(a[0]), round(a[1])] for k, a in sorted(agg.items())]


# 2026-09-30 사용자 «세션은 미국·유럽·아시아 시장 VWAP 으로» -- 현지 개장 시각(서머타임 반영). 셋 다 같은 UTC 날짜 안에 떨어진다
#   (도쿄 09:00 JST = 00:00 UTC · 런던 08:00 = 07/08 UTC · 뉴욕 09:30 = 13:30/14:30 UTC).
#   ponytail: 주말·휴장일에도 같은 시각에 다시 센다(코인은 24시간) -- 거래소 달력을 따르려면 여기서 날짜를 거른다.
MARKET_OPENS = (("아시아", "Asia/Tokyo", 9, 0), ("유럽", "Europe/London", 8, 0), ("미국", "America/New_York", 9, 30))


@lru_cache(maxsize=64)
def session_starts(day: int) -> tuple[int, ...]:
    """UTC 날짜 번호 day(= t // 86400) 의 세 세션 시작 초 (아시아, 유럽, 미국)."""
    d = datetime.fromtimestamp(day * 86400, timezone.utc).date()
    return tuple(int(datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo(tz)).timestamp()) for _, tz, h, m in MARKET_OPENS)


def market_session(t: int) -> tuple[int, str]:
    """초 t 가 속한 세션의 (시작 초, 이름). 다음 세션이 열리면 넘어간다 -- 미국장 뒤 ~00시는 미국 세션이 이어진다."""
    st = session_starts(t // 86400)
    i = bisect_right(st, t) - 1
    return (st[i], MARKET_OPENS[i][0]) if i >= 0 else (st[0], MARKET_OPENS[0][0])


def session_vwap(ts: list[int], high: list[float], low: list[float], close: list[float],
                 vol: list[float], key: Callable[[int], object] | None = None) -> tuple[list[float | None], list[float | None]]:
    """key(t) 가 바뀔 때 다시 시작하는 VWAP 과 거래량 가중 표준편차(대표가 = (고+저+종)/3). 봉 t 의 값은 봉 t 까지만 본다.
    기본 key = UTC 날짜(00:00 에 다시 시작)."""
    key = key or (lambda t: t // 86400)
    vw: list[float | None] = []
    sd: list[float | None] = []
    day, sv, spv, sp2v = object(), 0.0, 0.0, 0.0
    for t, h, lo, c, v in zip(ts, high, low, close, vol):
        if key(t) != day:
            day, sv, spv, sp2v = key(t), 0.0, 0.0, 0.0
        tp = (h + lo + c) / 3.0
        sv += v; spv += tp * v; sp2v += tp * tp * v
        if sv > 0:
            m = spv / sv
            vw.append(m); sd.append(math.sqrt(max(0.0, sp2v / sv - m * m)))
        else:
            vw.append(None); sd.append(None)
    return vw, sd



def value_area(k: Iterable[int], vol: Iterable[float], bw: float, frac: float = 0.7) -> dict[str, float] | None:
    """가격 칸 번호 k(= floor(가격/bw))별 거래량 → VAL·POC·VAH. POC 에서 위/아래 중 큰 쪽부터 한 칸씩 넓혀 frac 을 채운다
    (research_eth_fp_event_response_20260930.value_area 와 같은 규약). 09-30 검정: 닿아도 되돌림은 위약 레벨과 같다 -- 지도일 뿐."""
    k = np.asarray(list(k), dtype=np.int64); v = np.asarray(list(vol), dtype=float)
    if not len(k) or v.sum() <= 0:
        return None
    k0 = int(k.min()); hist = np.bincount(k - k0, weights=v)
    poc = int(np.argmax(hist)); lo = hi = poc; tot = hist[poc]; target = frac * hist.sum()
    while tot < target and (lo > 0 or hi < len(hist) - 1):
        up = hist[hi + 1] if hi < len(hist) - 1 else -1.0
        dn = hist[lo - 1] if lo > 0 else -1.0
        if up >= dn:
            hi += 1; tot += up
        else:
            lo -= 1; tot += dn
    return {"val": (k0 + lo) * bw, "poc": (k0 + poc + 0.5) * bw, "vah": (k0 + hi + 1) * bw}


def profile_nodes(k: Iterable[int], vol: Iterable[float], bw: float) -> dict[str, list[dict[str, float]]]:
    """같은 입력 → HVN(평활 거래량의 ±6칸 국소 최대 · 최대의 25% 이상) · LVN(±6칸 국소 최소 · 양쪽 봉우리 중 낮은 쪽의 50% 이하).
    연속 칸은 가운데 하나, 가격 = 칸 가운데. rel = 평활 거래량 ÷ 최대(굵기용). research_eth_profile_vwap_levels_20260930.hvn_lvn 과 같은 규약.
    09-30 검정: HVN 되돌림·LVN 빠른 통과 둘 다 위약과 같다(LVN 돌파가 빨라 보이는 건 변동성 큰 때 닿아서) -- 지도일 뿐."""
    k = np.asarray(list(k), dtype=np.int64); v = np.asarray(list(vol), dtype=float)
    if len(k) < 2 * NODE_WIN or v.sum() <= 0:
        return {"hvn": [], "lvn": []}
    k0 = int(k.min()); hist = np.bincount(k - k0, weights=v)
    x = np.arange(-3 * NODE_SIG, 3 * NODE_SIG + 1)
    ker = np.exp(-0.5 * (x / NODE_SIG) ** 2); ker /= ker.sum()
    s = np.convolve(hist, ker, "same")
    swv = np.lib.stride_tricks.sliding_window_view
    mx = swv(np.pad(s, NODE_WIN, constant_values=-np.inf), 2 * NODE_WIN + 1).max(1)
    mn = swv(np.pad(s, NODE_WIN, constant_values=np.inf), 2 * NODE_WIN + 1).min(1)
    left, right = np.maximum.accumulate(s), np.maximum.accumulate(s[::-1])[::-1]
    out = {}
    for name, isx, key in (("hvn", (s >= mx) & (s >= 0.25 * s.max()), -s), ("lvn", (s <= mn) & (s <= 0.5 * np.minimum(left, right)), s)):
        d = np.diff(np.r_[0, isx.astype(np.int8), 0])
        idx = (np.flatnonzero(d == 1) + np.flatnonzero(d == -1) - 1) // 2
        idx = idx[np.argsort(key[idx], kind="stable")[:NODE_MAX]]
        out[name] = [{"px": (k0 + int(i) + 0.5) * bw, "rel": round(float(s[i] / s.max()), 3)} for i in sorted(idx)]
    return out



def vwap_rows(ts: list[int], high: list[float], low: list[float], close: list[float], vol: list[float],
              nd: int = 4) -> dict[int, dict]:
    """봉 시각 → 캔들 행에 붙일 VWAP 두 벌: vwap/vsd = UTC 00시부터 · svwap/svsd = 지금 시장 세션 개장부터(sstart 초, sname 이름).
    화면의 «세션 시작부터» 스위치가 둘 중 하나를 고른다."""
    vw, vsd = session_vwap(ts, high, low, close, vol)
    sw, ssd = session_vwap(ts, high, low, close, vol, key=lambda t: market_session(t)[0])
    out: dict[int, dict] = {}
    for t, a, b, c, d in zip(ts, vw, vsd, sw, ssd):
        row: dict = {}
        if a is not None:
            row.update(vwap=round(a, nd), vsd=round(b, nd))
        if c is not None:
            st, nm = market_session(t)
            row.update(svwap=round(c, nd), svsd=round(d, nd), sstart=st, sname=nm)
        if row:
            out[t] = row
    return out


if __name__ == "__main__":  # 자체점검 -- 부호·경계·보류 조건
    assert pct_rank(5, range(100)) == 0.06 and pct_rank(5, range(10)) is None and pct_rank(None, range(100)) is None
    px = np.arange(2690.0, 2710.5, 0.5)                    # 칸 아래끝 · 미드 2700.005 는 칸 [2700.0, 2700.5) 안
    row = np.where(px < 2700, 2.0, np.where(px > 2700, -1.0, 0.5))   # 상계된 칸(2700.0)은 순 +0.5 -- 믿으면 안 되는 값
    s = sweep_depth(row, px, 2700.005, (10, 25))           # 10bp = 2.7$: 가운데가 [2697.3, 2702.7] 안인 칸
    assert s == {"bid": [5 * 2.0, 13 * 2.0], "ask": [4 * 1.0, 13 * 1.0]}, s   # 상계 칸은 뺀다 · 25bp 는 양쪽 13칸(대칭)
    s = sweep_depth(row, px, 2700.005, (10,), top=(2700.0, 30.0, 2700.01, 7.0))  # 최우선 호가가 상계 칸 안 → 정확한 수량을 더한다
    assert s == {"bid": [10.0 + 30.0], "ask": [4.0 + 7.0]}, s
    s = sweep_depth(row, px, 2700.005, (10,), top=(2699.9, 30.0, 2700.6, 7.0))   # 최우선이 상계 칸 밖 → 이미 합에 있다(두 번 안 센다)
    assert s == {"bid": [10.0], "ask": [4.0]}, s
    assert quad_1h(-40, 500, 30)["key"] == "dn_up" and quad_1h(-40, -5, 30)["key"] == "dn_dn"
    assert quad_1h(40, 5, 30)["key"] == "up_up" and quad_1h(40, -5, 30)["key"] == "up_dn"
    assert quad_1h(10, 5, 30)["key"] == "small" and quad_1h(10, 5, None)["key"] == "small" and quad_1h(None, 5, 30) is None
    ser = [(i * 300, 1000.0 + (i * 7 % 5) + (20.0 if i == 299 else 0.0)) for i in range(300)]   # 마지막 봉에 급증
    st = oi_stats(ser)
    assert st["oi"] == 1000.0 + 299 * 7 % 5 + 20 and st["z1h"] is not None and st["z1h"] > 3 and st["d24h_pct"] is not None, st
    holes = [(t, v) for t, v in ser if t != 299 * 300 - 12 * 300]                          # 1시간 전 칸이 비었다
    assert oi_stats(holes)["d1h_pct"] is None and oi_stats([])["oi"] is None
    # 2026-10-05 요일유형: 평일엔 1시간 변화가 크고 주말엔 작은 7일 -- 주말 끝의 «주말치고 큰» 변화가 평일 분포에 묻히지 않는다
    _t0, _g = 1_759_536_000 - 5 * 86400, np.random.default_rng(1)       # 월요일 00:00 UTC 부터 7일(토·일로 끝남)
    _lv = 1000.0 + np.cumsum([_g.normal(0, 4.0 if ((_t0 + i * 300) // 86400 + 3) % 7 < 5 else 0.4) for i in range(7 * 288)])
    _lv[-1] = _lv[-13] * 1.015                                          # 주말 마지막 봉: 1시간 +1.5%
    _ser = [(_t0 + i * 300, float(v)) for i, v in enumerate(_lv)]
    _z = oi_stats(_ser)["z1h"]
    _d = _lv[12:] / _lv[:-12] - 1; _zall = (_d[-1] - _d[:-1].mean()) / _d[:-1].std()
    assert _z > 2.5 and _zall < 1.5, (_z, _zall)                          # 주말 분포 기준이면 튀고, 전체 기준이면 평일 흔들림에 묻힌다
    # 2026-10-06 7일 창에 큰 급변 하나(창 안 한 시간 +8%)가 끼어도 그 뒤 «평소의 3배 급증»은 계속 급증으로 잡힌다(평균·표준편차는 묻힌다)
    _g2 = np.random.default_rng(2); _lv2 = 1000.0 * np.cumprod(1 + _g2.normal(0, 0.0005, 7 * 288))
    _lv2[1000:1012] *= np.linspace(1.0, 1.08, 12); _lv2[1012:] *= 1.08          # 창 중간의 큰 급변
    _lv2[-1] = _lv2[-13] * 1.006                                                  # 마지막: 1시간 +0.6%(평소 σ≈0.17% 의 ~3.5배)
    _s2 = [(_t0 + i * 300, float(v)) for i, v in enumerate(_lv2)]
    _d2 = _lv2[12:] / _lv2[:-12] - 1; _zstd = (_d2[-1] - _d2[:-1].mean()) / _d2[:-1].std()
    assert oi_stats(_s2)["z1h"] > 1.5 and _zstd < oi_stats(_s2)["z1h"], (oi_stats(_s2)["z1h"], _zstd)
    _g3 = np.random.default_rng(2); _lv3 = 1000.0 * np.cumprod(1 + _g3.normal(0, 0.0005, 7 * 288))
    _lv3[-295:-1] = _lv3[-296]; _lv3[-1] = _lv3[-13] * 1.0005                    # 피드 일부 멈춤: 같은 요일유형 1시간 변화의 49% 가 0
    _z3 = oi_stats([(_t0 + i * 300, float(v)) for i, v in enumerate(_lv3)])["z1h"]
    assert _z3 is not None and abs(_z3) < 2, _z3                                  # 하한 없으면 4.8(거짓 «급증») · 있으면 1.19
    _ss, _sv = oi_sum_series({"binance": [(0, 10.0), (300, 11.0), (600, 12.0)], "okx": [(0, 1.0), (600, 2.0)],
                              "bybit": [(300, 5.0), (600, 5.0)]}, since=0, tol_s=0)       # bybit 은 창 시작을 못 덮어 빠진다
    assert _sv == ["binance", "okx"] and _ss == [(0, 11.0), (600, 14.0)], (_ss, _sv)          # okx 빈 칸(300)은 합에서 뺀다
    assert oi_sum_series({"okx": [(0, 1.0)]}, since=0) == ([], [])                            # 바이낸스 없으면 합 없음
    assert lev_state(0.95, 1.2)["key"] == "long_crowd" and lev_state(0.05, 1.2)["key"] == "short_crowd"
    assert lev_state(0.5, -2)["key"] == "deleverage" and lev_state(0.95, 0)["key"] == "premium" and lev_state(None, 3)["key"] == "na"
    lv = hl_liq_levels([(10, 2601), (5, 2602), (-2, 2800), (3, 2710), (-1, 2500), (1, 1000)], mid=2700, bin_usd=5)
    assert [l["px"] for l in lv["below"]] == [2600.0] and lv["below"][0]["usd"] == 15 * 2700 and lv["below"][0]["n"] == 2, lv
    assert [l["px"] for l in lv["above"]] == [2800.0], lv   # 롱인데 청산가가 위(2710) · 숏인데 아래(2500) · 너무 먼 1000 은 버린다
    assert liq_profile([(2700.4, 100, True), (2701.4, 50, False), (2701.3, 20, True), (0, 5, True)]) == [[2700.0, 100, 0], [2702.5, 20, 50]]
    t = [86400 - 600, 86400 - 300, 86400, 86400 + 300]
    vw, sd = session_vwap(t, [11, 13, 21, 23], [9, 11, 19, 21], [10, 12, 20, 22], [1, 1, 1, 3])
    assert vw[0] == 10 and vw[1] == 11 and abs(sd[1] - 1) < 1e-9 and vw[2] == 20 and abs(vw[3] - 21.5) < 1e-9, (vw, sd)
    # 가치영역: 합 110 의 70% = 77. POC 칸 10(50)에서 큰 쪽(11, 30)을 먼저 붙이면 80 으로 채워진다 -- 작은 쪽(9)은 안 붙는다
    va = value_area([8, 9, 10, 11, 12], [5, 20, 50, 30, 5], bw=2.0)
    assert va == {"val": 20.0, "poc": 21.0, "vah": 24.0}, va
    assert value_area([8, 9, 10, 11, 12], [5, 20, 50, 30, 5], bw=2.0, frac=0.9)["val"] == 18.0   # 90% = 99 → 9(20)까지
    assert value_area([], [], 1.0) is None and value_area([3], [0], 1.0) is None
    # 봉우리 둘(칸 20·60) 사이 골(칸 40) → HVN 두 개 · LVN 하나. 칸 가운데 가격
    kk = np.arange(0, 81); vv = 100 * np.exp(-0.5 * ((kk - 20) / 4) ** 2) + 80 * np.exp(-0.5 * ((kk - 60) / 4) ** 2) + 1
    nd = profile_nodes(kk, vv, bw=1.0)
    assert [round(h["px"]) for h in nd["hvn"]] == [20, 60] and nd["hvn"][0]["rel"] == 1.0, nd
    assert len(nd["lvn"]) == 1 and abs(nd["lvn"][0]["px"] - 40.5) <= 3, nd       # 골 바닥이 넓으면(연속 칸) 가운데
    assert profile_nodes([1, 2], [1, 1], 1.0) == {"hvn": [], "lvn": []}
    # 세션 시작: 2026-09-30(서머타임) 00:00 · 07:00 · 13:30 UTC / 2026-12-15(표준시) 00:00 · 08:00 · 14:30 UTC
    D = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())   # noqa: E731
    assert session_starts(D("2026-09-30 00:00") // 86400) == (D("2026-09-30 00:00"), D("2026-09-30 07:00"), D("2026-09-30 13:30"))
    assert session_starts(D("2026-12-15 00:00") // 86400) == (D("2026-12-15 00:00"), D("2026-12-15 08:00"), D("2026-12-15 14:30"))
    assert market_session(D("2026-09-30 06:59")) == (D("2026-09-30 00:00"), "아시아")
    assert market_session(D("2026-09-30 13:30")) == (D("2026-09-30 13:30"), "미국")
    assert market_session(D("2026-09-30 23:55")) == (D("2026-09-30 13:30"), "미국")      # 미국장 뒤도 다음 00시까지 미국 세션
    t5 = [D("2026-09-30 06:55"), D("2026-09-30 07:00"), D("2026-09-30 07:05")]
    vw, _ = session_vwap(t5, [11, 21, 23], [9, 19, 21], [10, 20, 22], [1, 1, 1], key=lambda t: market_session(t)[0])
    assert vw == [10, 20, 21], vw                                                          # 07:00(런던 개장)에 다시 센다
    vw, _ = session_vwap(t5, [11, 21, 23], [9, 19, 21], [10, 20, 22], [1, 1, 1])
    assert vw[:2] == [10, 15] and abs(vw[2] - 52 / 3) < 1e-9, vw                            # 기본(UTC 날짜)은 이어서 센다
    r = vwap_rows(t5, [11, 21, 23], [9, 19, 21], [10, 20, 22], [1, 1, 1])
    assert r[t5[2]]["svwap"] == 21 and r[t5[2]]["sname"] == "유럽" and r[t5[2]]["sstart"] == D("2026-09-30 07:00") and r[t5[0]]["sname"] == "아시아", r
    print("market_ctx selftest ok")
