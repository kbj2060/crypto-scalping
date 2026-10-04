"""시장 프로파일 레벨(VAH/VAL·HVN/LVN)과 VWAP 계열 레벨(주간·앵커드)이 «트레이딩에 도움이 되는가» (2026-09-30).

모델 학습이 아니라 조건식 규칙을 과거에 적용해 그 뒤 가격을 센다. 사용자가 2024년 이전을 신뢰하지 않으므로
평가 = 2025-01-01 ~ 2026-09-14 만. 앞 = 2025년, 뒤 = 2026년. 2024-08~12 는 프로파일·VWAP·위약(placebo) 지연 예열에만 쓴다.

사전 고정(결과 보기 전, 2026-09-30 작성):
  데이터   ETHUSDT 선물 1분봉(t,h,l,c,v,tb). 빈 분은 NaN. 대표가 tp = (h+l+c)/3. 시가 열 없음 → «직전 가격» = 직전 분 종가.
  레벨     전부 «분 j 가 시작할 때 알려진 값»(분 j 의 데이터는 안 들어간다).
    VAH/VAL   전일(UTC) 1분 tp 거래량 프로파일, 칸 = 그날 중앙가의 0.05%, POC 에서 큰 쪽부터 넓혀 70%(E.value_area 그대로). 그날 내내 고정.
    HVN/LVN   매 정시 h 에 [h−24h, h) 1분 tp 거래량 프로파일(칸 0.05%) → 가우스 평활(σ=2칸) s.
              HVN = s 가 ±6칸(±0.3%) 안 최대 · s ≥ 0.25·max(s). LVN = s 가 ±6칸 안 최소 · s ≤ 0.5·min(왼쪽 최대, 오른쪽 최대)(양쪽이 막힌 골).
              평평한 봉우리/골(연속 칸)은 가운데 칸 하나. 레벨가 = 칸 중앙. [h, h+1h) 동안 유효. 분 j 에서 쓰는 레벨 = 직전 종가 위 가장 가까운 것 / 아래 가장 가까운 것.
    주간 VWAP  월요일 00:00 UTC 부터 분 j 직전까지 Σtp·v/Σv. 앵커 뒤 60분 미만이면 없음.
    앵커드 VWAP 앵커 = 전일(UTC) 고가를 처음 찍은 분 / 전일 저가를 처음 찍은 분(두 개). 그 분부터 분 j 직전까지. 전일이 닫혀야 앵커를 알므로 당일 00:00 부터 유효.
  닿음     직전 종가 c[j−1] < L 이면 저항 쪽: 고가 h[j] ≥ L. c[j−1] > L 이면 지지 쪽: 저가 l[j] ≤ L. 한 분이 위·아래 레벨을 둘 다 닿으면 버린다.
           VAH/VAL 주장 a 는 «가치영역 안에서»만: VAH 는 VAL ≤ c[j−1] < VAH 에서, VAL 은 VAL < c[j−1] ≤ VAH 에서.
           반복: 닿은 뒤 종가가 그 레벨 값에서 30bp 이상 멀어진 분이 지나야 다시 센다(재무장). 레벨 인스턴스(일/주)가 바뀌면 초기화.
           강건성 변형 «첫닿음»(VAH/VAL·VWAP 계열만): UTC 일·방향·출처(실제/지연 k)별 첫 닿음만.
           단일 레벨(VAH·VAL·VWAP)은 무장 해제 중 그 계열의 모든 닿음을 버린다. HVN/LVN 은 해제된 레벨 ±10bp 안의 닿음만 버린다(정시마다 바뀌므로 초기화 없음).
  라벨     닿음은 분 j 가 닫혀야 안다 → 기준가 = c[j], 탐색 = 분 j+1 부터 240분(한 봉 미래참조 금지).
           반전 쪽 +X bp 와 통과 쪽 +X bp 중 먼저 닿는 쪽(1분 고·저). X = 15 · 30 · vol(=직전 60분 1분 로그수익 실현변동성 bp, [10,100] 클립).
           240분 안에 둘 다 안 닿음 = 제외(제외율 보고). 같은 분에 둘 다 = 제외(동률율 보고). 판정 기준 X = 30bp, 나머지는 강건성.
  방향수익 닿은 다음 분 종가 c[j+1] 진입 → 30·60·240분 뒤 종가(bp), 반전 방향 부호. 대조군 대비.
  대조군   «지연 기하 위약»: 같은 레벨을 k = 7·14·21·28일 전 것으로 가져와 인스턴스 시작가 비율로 옮긴다
           L'(t) = L(t−k일) × A(t)/A(t−k일), A = 인스턴스 시작 직전 종가(VAH/VAL·앵커드 = 일 · 주간 VWAP = 주 · HVN/LVN = 정시).
           → 같은 시각대·같은 «시작가 대비 거리» 분포·같은 모양이지만 오늘 거래와 무관. 닿음·방향·재무장·라벨 절차를 똑같이 적용하고
           같은 방향(저항/지지)에서 닿은 것끼리 비교되도록 사건마다 방향을 갖는다. 4개 지연을 풀링. 효과 = 실제 − 대조.
  주장별 지표 (기대 부호 +)
    1a VAH/VAL 거부      반전율(해결된 사건 중 반전 쪽 먼저) 실제 − 대조.
    1b 80% 규칙           그날 시가(전일 마지막 종가)가 VA 밖 → 안으로 들어와 1분 종가 30개 연속 VA 안 = 확인 분 m.
                          라벨 = m+1 ~ 그날 끝(23:59)에 반대편 경계 닿음(위에서 왔으면 저가 ≤ VAL). 남은 시간 30분 미만이면 제외.
                          대조 = 같은 해 «시가가 VA 안»인 날의 분 m' ∈ [m−30, m+30] 중 직전 30종가가 VA 안 · 같은 목표 쪽 상대위치 u 가 ±0.1 ·
                          VA 폭/전일 실현변동성(로그) ±0.3 인 점들의 같은 목표 도달률(날마다 평균 → 날 평균). 대조 날 5개 미만이면 사건 제외.
                          효과 = 사건 도달 − 짝 대조 도달, 사건 일 블록 부트스트랩.
    2  HVN 지지·저항      반전율 실제 − 대조.
    2  LVN 빠른 통과      통과율(통과 쪽 먼저) 실제 − 대조. 보조: «30분 안 통과» 비율(전체 사건 분모), 통과까지 분 중앙값.
    3a 주간·앵커드 VWAP   반전율 실제 − 대조.
    3b 주간 VWAP 방향     표본 = 매 4시간 경계(4h 지평) · 매일 00:00(24h 지평), 겹치지 않음. 신호 = sign(c[t−1] − VWAP(t)),
                          수익 = c[t−1+H]/c[t−1]. 지표 = 방향 적중률. 대조 = 같은 창(월요일부터)의 단순평균 tp(SMA_same) — 차이는 거래량 가중뿐.
                          참고 대조 = 7일 이동 SMA. 효과 = 적중(VWAP) − 적중(SMA), 같은 표본 짝 차이. bp 차이도 보고.
  CI       일 단위 포아송 블록 부트스트랩 B=1000(R.block_ci · E.mean_ci).
  판정     2025·2026 둘 다 같은 부호로 CI 가 0 배제 → 기대 부호면 «통과», 반대면 «반대로 통과(주장 기각)».
           한 해만 → «한 해만». 아니면 «불합격». 한 해라도 CI 반폭 > 4pp 면 «검정력 부족» 표시.
  누수점검 라벨 시작을 한 분 더 늦춰(기준 c[j+1], 탐색 j+2~, 진입 c[j+2]; 80% 규칙은 m+2~) 결과가 유지되는지 모든 항목에서 본다.

실행: python scripts/research_eth_profile_vwap_levels_20260930.py [--selftest] [--data DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_fp_event_response_20260930 as E    # noqa: E402  load_1m · value_area · mean_ci
import research_eth_fp_pattern_lookback_20260930 as R  # noqa: E402  default_data_dir · block_ci

WARM = pd.Timestamp("2024-08-01")
SPLITS = (("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
          ("2026", pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-01")))
T_LIM, REARM_BP, HVN_TOL_BP, MIN_LEN = 240, 30.0, 10.0, 60
XS = ("15", "30", "vol")
LAGS_D = (7, 14, 21, 28)
HS = (30, 60, 240)
D1 = 1440
swv = np.lib.stride_tricks.sliding_window_view


# ---------- 레벨 ----------
def seg_ratio(num: np.ndarray, den: np.ndarray, a: np.ndarray) -> np.ndarray:
    """분 j 마다 [a[j], j) 구간 Σnum/Σden -- 분 j 는 안 들어간다. 구간 < MIN_LEN 이거나 a<0 이면 NaN."""
    Gn = np.r_[0.0, np.cumsum(np.nan_to_num(num))]; Gd = np.r_[0.0, np.cumsum(np.nan_to_num(den))]
    j = np.arange(len(num)); aa = np.clip(a, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (Gn[j] - Gn[aa]) / (Gd[j] - Gd[aa])
    out[(a < 0) | (j - a < MIN_LEN) | ~np.isfinite(out)] = np.nan
    return out


def open_before(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """인스턴스 시작 직전 종가 c[a−1]."""
    return np.where(a >= 1, c[np.clip(a - 1, 0, None)], np.nan)


def shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    out[k:] = x[:len(x) - k]
    return out


def day_anchor(x: np.ndarray, fn) -> np.ndarray:
    """분 j 마다 «전일» 안에서 fn(argmax/argmin) 이 가리키는 분의 전역 위치. 배열은 00:00 에서 시작하고 1440 의 배수 길이."""
    nd = len(x) // D1
    X = x.reshape(nd, D1)
    bad = ~np.isfinite(X).any(1)
    k = fn(np.where(np.isfinite(X), X, -np.inf if fn is np.argmax else np.inf), 1)
    a = np.arange(nd) * D1 + k
    a = np.where(bad, -1, a)
    prev = np.r_[-1, a[:-1]]                      # 당일 d 의 앵커 = 전일 d−1 의 극값 분
    return np.repeat(prev, D1)


def hvn_lvn(tp: np.ndarray, v: np.ndarray, K: int = 16, win: int = 6, sig: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """정시 r 마다 [r−24h, r) 프로파일의 HVN·LVN 가격 (시간 수 × K, NaN 채움). 행 r 은 분 [60r, 60r+60) 에 쓴다."""
    nh = len(tp) // 60
    H = np.full((nh, K), np.nan); L = np.full((nh, K), np.nan)
    x = np.arange(-3 * sig, 3 * sig + 1)
    ker = np.exp(-0.5 * (x / sig) ** 2); ker /= ker.sum()
    for r in range(24, nh):
        p, w = tp[(r - 24) * 60:r * 60], v[(r - 24) * 60:r * 60]
        ok = np.isfinite(p) & np.isfinite(w)
        if ok.sum() < 720:
            continue
        p, w = p[ok], w[ok]
        bw = np.median(p) * 5e-4; lo = p.min()
        s = np.convolve(np.bincount(((p - lo) / bw).astype(int), weights=w), ker, "same")
        mx = swv(np.pad(s, win, constant_values=-np.inf), 2 * win + 1).max(1)
        mn = swv(np.pad(s, win, constant_values=np.inf), 2 * win + 1).min(1)
        left = np.maximum.accumulate(s); right = np.maximum.accumulate(s[::-1])[::-1]
        ih = (s >= mx) & (s >= 0.25 * s.max())
        il = (s <= mn) & (s <= 0.5 * np.minimum(left, right))
        for M, isx, key in ((H, ih, -s), (L, il, s)):
            d = np.diff(np.r_[0, isx.astype(np.int8), 0])
            idx = (np.flatnonzero(d == 1) + np.flatnonzero(d == -1) - 1) // 2     # 연속 칸은 가운데 하나
            idx = idx[np.argsort(key[idx], kind="stable")[:K]]
            M[r, :len(idx)] = lo + (idx + 0.5) * bw
    return H, L


def nearest(Lv: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """분 j: 직전 종가 위 가장 가까운 레벨 U, 아래 가장 가까운 레벨 D (정시 행 = j//60)."""
    n = len(c); cp = np.r_[np.nan, c[:-1]]
    U = np.full(n, np.nan); D = np.full(n, np.nan)
    for a in range(0, n, 200_000):
        sl = slice(a, min(n, a + 200_000))
        M = Lv[np.arange(sl.start, sl.stop) // 60]; p = cp[sl][:, None]
        with np.errstate(invalid="ignore"):
            up = np.where(M > p, M, np.inf).min(1); dn = np.where(M < p, M, -np.inf).max(1)
        U[sl] = np.where(np.isfinite(up), up, np.nan); D[sl] = np.where(np.isfinite(dn), dn, np.nan)
    return U, D


# ---------- 사건·라벨 ----------
def rearm_at(c: np.ndarray, j: int, L: float) -> int:
    for w in (240, D1, 7 * D1, 28 * D1):
        seg = c[j + 1:j + 1 + w]
        with np.errstate(invalid="ignore"):
            far = np.abs(seg / L - 1) * 1e4 >= REARM_BP
        if far.any():
            return j + 1 + int(np.argmax(far))
    return len(c)


def touches(U: np.ndarray, D: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray,
            inst: np.ndarray | None = None, tol: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """닿음 사건 (분 j, 레벨, 반전 방향 s: 저항 닿음 −1 · 지지 닿음 +1). U/D 는 분 j 시작 때 알려진 위/아래 레벨."""
    with np.errstate(invalid="ignore"):
        up = np.isfinite(U) & (h >= U); dn = np.isfinite(D) & (l <= D)
    cand = np.flatnonzero((up | dn) & ~(up & dn))
    J, LV, S, dis, cur = [], [], [], [], None
    for j in cand:
        if inst is not None and inst[j] != cur:
            cur, dis = inst[j], []
        L = U[j] if up[j] else D[j]
        dis = [(a, u) for a, u in dis if u >= j]
        if any(abs(L / a - 1) * 1e4 < tol for a, _ in dis):
            continue
        dis.append((L, rearm_at(c, j, L)))
        J.append(j); LV.append(L); S.append(-1 if up[j] else 1)
    return np.array(J, np.int64), np.array(LV, float), np.array(S, np.int8)


def outcome(h: np.ndarray, l: np.ndarray, c: np.ndarray, j: np.ndarray, s: np.ndarray, X: np.ndarray,
            extra: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """기준 c[j+extra], 탐색 분 j+1+extra ~ +240. code 1 반전 먼저 · 0 통과 먼저 · −1 둘 다 없음 · −2 같은 분 · −3 평가 불가. tmin = 해결까지 분."""
    n = len(c); S = j + 1 + extra
    code = np.full(len(j), -3, np.int8); tmin = np.full(len(j), -1, np.int32)
    Wh, Wl = swv(h, T_LIM), swv(l, T_LIM)
    idx = np.flatnonzero(S + T_LIM <= n)
    for a in range(0, len(idx), 20_000):
        q = idx[a:a + 20_000]; ss = S[q]; ref = c[ss - 1]
        with np.errstate(invalid="ignore"):
            up = Wh[ss] >= (ref * (1 + X[q] / 1e4))[:, None]; dn = Wl[ss] <= (ref * (1 - X[q] / 1e4))[:, None]
        iu = np.where(up.any(1), up.argmax(1), T_LIM); idn = np.where(dn.any(1), dn.argmax(1), T_LIM)
        rev = np.where(s[q] > 0, iu, idn); pas = np.where(s[q] > 0, idn, iu)
        cd = np.where(rev < pas, 1, np.where(pas < rev, 0, np.where(rev == T_LIM, -1, -2)))
        cd[~np.isfinite(ref)] = -3
        code[q] = cd; tmin[q] = np.minimum(rev, pas) + 1
    return code, tmin


def fwd_ret(c: np.ndarray, e: np.ndarray, H: int) -> np.ndarray:
    out = np.full(len(e), np.nan)
    ok = e + H < len(c)
    out[ok] = (c[e[ok] + H] / c[e[ok]] - 1) * 1e4
    return out


# ---------- 80% 규칙 ----------
def rule80(C, Hh, Ll, vah, val, opn, sig, extra: int = 0, run: int = 30):
    """일별 (nd,1440) 배열. 반환: 사건 dict 배열들, 대조 풀 도구. 목표 0 = VAL(위에서 옴), 1 = VAH(아래에서 옴)."""
    nd = len(C)
    with np.errstate(invalid="ignore"):
        inside = (C >= val[:, None]) & (C <= vah[:, None])
        width = vah - val
        u = np.stack([(C - val[:, None]) / width[:, None], (vah[:, None] - C) / width[:, None]])      # 목표까지 남은 폭 비율
        w = np.log(width / ((vah + val) / 2) * 1e4 / sig)
        hit = np.stack([Ll <= val[:, None], Hh >= vah[:, None]])
    cnt = np.zeros(C.shape, np.int32)
    cnt[:, 0] = inside[:, 0]
    for m in range(1, D1):
        cnt[:, m] = (cnt[:, m - 1] + 1) * inside[:, m]
    ok30 = cnt >= run
    last = np.where(hit.any(2), D1 - 1 - np.argmax(hit[:, :, ::-1], 2), -1)                          # (2, nd) 그날 마지막 닿은 분
    reach = last[:, :, None] >= (np.arange(D1) + 1 + extra)[None, None, :]                          # (2, nd, 1440) m 뒤에 닿나
    with np.errstate(invalid="ignore"):
        tgt = np.where(opn > vah, 0, np.where(opn < val, 1, -1))
        inside_open = (opn >= val) & (opn <= vah)
    ev = []
    for d in np.flatnonzero(tgt >= 0):
        ms = np.flatnonzero(ok30[d, :D1 - 30])
        if len(ms) == 0 or not np.isfinite(w[d]):
            continue
        m = int(ms[0]); T = int(tgt[d])
        ev.append((d, m, T, u[T, d, m], w[d], bool(reach[T, d, m])))
    return ev, dict(ok30=ok30, u=u, w=w, reach=reach, inside_open=inside_open)


def rule80_match(ev, pool, days_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """사건마다 짝 대조 도달률. days_ok = 대조로 쓸 날 마스크(같은 해·시가 VA 안)."""
    P = np.flatnonzero(days_ok & pool["inside_open"] & np.isfinite(pool["w"]))
    y, cr, dd = [], [], []
    for d, m, T, ue, we, ye in ev:
        a, b = max(29, m - 30), min(D1 - 31, m + 30) + 1
        mk = pool["ok30"][P, a:b] & (np.abs(pool["u"][T][P, a:b] - ue) <= 0.1) & (np.abs(pool["w"][P] - we) <= 0.3)[:, None]
        nday = mk.any(1)
        if nday.sum() < 5:
            continue
        rr = (pool["reach"][T][P, a:b] * mk).sum(1)[nday] / mk.sum(1)[nday]
        y.append(float(ye)); cr.append(float(rr.mean())); dd.append(d)
    return np.array(y), np.array(cr), np.array(dd, np.int64)


# ---------- 판정 ----------
def verdict(cis: list[tuple[float, float, float]], sign: int = 1) -> str:
    sg = [int(np.sign(p)) if (lo > 0 or hi < 0) else 0 for p, lo, hi in cis]
    if sg[0] != 0 and sg[0] == sg[1]:
        v = "통과" if sg[0] == sign else "반대로 통과(주장 기각)"
    elif sg[0] != 0 and sg[1] != 0:
        v = "불합격(해마다 부호 반대)"
    elif sg[0] != 0 or sg[1] != 0:
        v = "한 해만(" + ("2025" if sg[0] else "2026") + (", 기대 부호" if (sg[0] or sg[1]) == sign else ", 반대 부호") + ")"
    else:
        v = "불합격"
    if any((hi - lo) / 2 > 0.04 for _, lo, hi in cis):
        v += " · 검정력 부족"
    return v


def run(data_dir: Path, out_dir: Path) -> None:
    m1 = E.load_1m(data_dir)
    m1 = m1[m1.index >= WARM.value // 10**6]
    pad = (-len(m1)) % D1
    if pad:
        m1 = m1.reindex(np.r_[m1.index.to_numpy(), m1.index[-1] + 60_000 * np.arange(1, pad + 1)])
    t = m1.index.to_numpy(np.int64)
    assert t[0] % 86_400_000 == 0 and (np.diff(t) == 60_000).all()
    h, l, c, v = (m1[k].to_numpy(float) for k in ("h", "l", "c", "v"))
    tp = (h + l + c) / 3
    n = len(c); j_all = np.arange(n)
    mn = t // 60_000
    day_ep = mn // D1
    a_day = j_all - mn % D1
    a_week = j_all - (mn + 3 * D1) % (7 * D1)
    a_week = np.where(a_week >= 0, a_week, -1)
    tsd = pd.to_datetime(t, unit="ms")
    yr = np.full(n, "", dtype=object)
    for name, a, b in SPLITS:
        yr[(tsd >= a) & (tsd < b)] = name
    day_id = t // 86_400_000
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.r_[np.nan, np.diff(np.log(c))]
    G = np.r_[0.0, np.cumsum(np.nan_to_num(r1) ** 2)]
    rv60 = np.full(n, np.nan); rv60[60:] = np.sqrt(G[61:] - G[1:-60]) * 1e4          # 분 j 까지(포함) 60분
    Xs = {"15": np.full(n, 15.0), "30": np.full(n, 30.0), "vol": np.clip(np.nan_to_num(rv60, nan=30.0), 10, 100)}
    print(f"1분봉 {n:,} ({tsd[0]} ~ {tsd[-1]})", flush=True)

    # 레벨
    va = E.value_area(m1)
    vah = pd.Series(day_ep - 1).map(va["vah"]).to_numpy(float); val = pd.Series(day_ep - 1).map(va["val"]).to_numpy(float)
    A_day, A_week = open_before(c, a_day), open_before(c, a_week)
    vw_w = seg_ratio(tp * v, v, a_week)
    aH, aL = day_anchor(h, np.argmax), day_anchor(-l, np.argmax)
    vw_h, vw_l = seg_ratio(tp * v, v, aH), seg_ratio(tp * v, v, aL)
    HV, LV = hvn_lvn(tp, v)
    P_hr = open_before(c, np.arange(len(HV)) * 60)
    print(f"레벨 준비 끝 · VA {va.shape[0]}일 · HVN 행당 평균 {np.isfinite(HV).sum(1)[24:].mean():.2f} · LVN {np.isfinite(LV).sum(1)[24:].mean():.2f}", flush=True)
    cp = np.r_[np.nan, c[:-1]]
    week_id = (mn + 3 * D1) // (7 * D1)

    def va_events(VH, VL):
        with np.errstate(invalid="ignore"):
            U = np.where((cp >= VL) & (cp < VH), VH, np.nan); Dn = np.where((cp > VL) & (cp <= VH), VL, np.nan)
        e1 = touches(U, np.full(n, np.nan), c, h, l, day_id); e2 = touches(np.full(n, np.nan), Dn, c, h, l, day_id)
        return tuple(np.r_[a, b] for a, b in zip(e1, e2))

    def single(Lv, inst):
        with np.errstate(invalid="ignore"):
            return touches(np.where(cp < Lv, Lv, np.nan), np.where(cp > Lv, Lv, np.nan), c, h, l, inst)

    def multi(M):
        U, Dn = nearest(M, c)
        return touches(U, Dn, c, h, l, None, HVN_TOL_BP)

    def plac(L, A, k):
        return shift(L, k * D1) * A / shift(A, k * D1)

    def plac_rows(M, k):
        kk = 24 * k
        out = np.full_like(M, np.nan); out[kk:] = M[:-kk] * (P_hr[kk:] / P_hr[:-kk])[:, None]
        return out

    tests = {
        "VAH/VAL": ("1a 거부(가치영역 안에서 닿으면 되돌림)", "rev",
                    lambda: va_events(vah, val), lambda k: va_events(plac(vah, A_day, k), plac(val, A_day, k))),
        "HVN": ("2 지지·저항(닿으면 멈춤/되돌림)", "rev", lambda: multi(HV), lambda k: multi(plac_rows(HV, k))),
        "LVN": ("2 빠른 통과", "pass", lambda: multi(LV), lambda k: multi(plac_rows(LV, k))),
        "주간 VWAP": ("3a 지지·저항 되돌림", "rev", lambda: single(vw_w, week_id), lambda k: single(plac(vw_w, A_week, k), week_id)),
        "AVWAP 전일고가": ("3a 지지·저항 되돌림", "rev", lambda: single(vw_h, day_id), lambda k: single(plac(vw_h, A_day, k), day_id)),
        "AVWAP 전일저가": ("3a 지지·저항 되돌림", "rev", lambda: single(vw_l, day_id), lambda k: single(plac(vw_l, A_day, k), day_id)),
    }
    rows, summ = [], []
    for name, (claim, metric, f_real, f_ctl) in tests.items():
        J, _, S = f_real()
        cj, cs = [], []
        for k in LAGS_D:
            a, _, b = f_ctl(k); cj.append(a); cs.append(b)
        CJ, CS = np.concatenate(cj), np.concatenate(cs); CK = np.concatenate([np.full(len(a), k) for k, a in zip(LAGS_D, cj)])
        J, S = J[yr[J] != ""], S[yr[J] != ""]; keep = yr[CJ] != ""; CJ, CS, CK = CJ[keep], CS[keep], CK[keep]
        allj, alls = np.r_[J, CJ], np.r_[S, CS].astype(int); grp = np.r_[np.ones(len(J), int), np.zeros(len(CJ), int)]
        src = np.r_[np.zeros(len(J), int), CK]
        first = pd.DataFrame({"s": src, "d": day_id[allj], "v": alls, "j": allj}).groupby(["s", "d", "v"])["j"].transform("min").to_numpy() == allj
        print(f"{name}: 실제 {len(J):,} · 대조 {len(CJ):,}", flush=True)
        prim = {}
        for extra in (0, 1):
            for X in (XS if extra == 0 else ("30",)):
                code, tmin = outcome(h, l, c, allj, alls, Xs[X][allj], extra)
                for side in ("전체", "저항", "지지", "첫닿음"):
                    if side != "전체" and (extra or X != "30"):
                        continue
                    if side == "첫닿음" and name in ("HVN", "LVN"):
                        continue                                          # 정시마다 바뀌는 레벨 -- «하루 첫 닿음»이 정의되지 않는다
                    sm = {"전체": np.ones(len(allj), bool), "저항": alls == -1, "지지": alls == 1, "첫닿음": first}[side]
                    for sp, _, _ in SPLITS:
                        m = sm & (yr[allj] == sp) & (code != -3)
                        res = m & ((code == 0) | (code == 1))
                        y = ((code == 1) if metric == "rev" else (code == 0)).astype(float)
                        row = {"test": name, "claim": claim, "metric": "반전율" if metric == "rev" else "통과율", "X": X, "extra": extra,
                               "side": side, "split": sp, "n": int((m & (grp == 1)).sum()), "n_ctl": int((m & (grp == 0)).sum()),
                               "excl": float((code[m & (grp == 1)] == -1).mean()), "excl_ctl": float((code[m & (grp == 0)] == -1).mean()),
                               "tie": float((code[m & (grp == 1)] == -2).mean())}
                        row["rate"] = float(y[res & (grp == 1)].mean()); row["rate_ctl"] = float(y[res & (grp == 0)].mean())
                        row["diff"], row["lo"], row["hi"] = R.block_ci(y[res], grp[res], day_id[allj[res]])
                        if metric == "pass" and extra == 0 and side == "전체":
                            p30 = ((code == 0) & (tmin <= 30)).astype(float)
                            row["pass30"], row["pass30_ctl"] = float(p30[m & (grp == 1)].mean()), float(p30[m & (grp == 0)].mean())
                            row["pass30_diff"], row["pass30_lo"], row["pass30_hi"] = R.block_ci(p30[m], grp[m], day_id[allj[m]])
                            row["tpass_med"] = float(np.median(tmin[m & (grp == 1) & (code == 0)]))
                            row["tpass_med_ctl"] = float(np.median(tmin[m & (grp == 0) & (code == 0)]))
                        if side == "전체" and X == "30":
                            e = allj + 1 + extra
                            for H in HS:
                                r = fwd_ret(c, e, H) * alls
                                mm = m & np.isfinite(r)
                                row[f"ret{H}"] = float(r[mm & (grp == 1)].mean()); row[f"ret{H}_ctl"] = float(r[mm & (grp == 0)].mean())
                                row[f"ret{H}_diff"], row[f"ret{H}_lo"], row[f"ret{H}_hi"] = R.block_ci(r[mm], grp[mm], day_id[allj[mm]])
                        rows.append(row)
                        if side in ("전체", "첫닿음"):
                            prim[(X, extra if side == "전체" else "first", sp)] = row
        for X, extra in [(x, 0) for x in XS] + [("30", 1)] + ([("30", "first")] if name not in ("HVN", "LVN") else []):
            cis = [(prim[(X, extra, sp)]["diff"], prim[(X, extra, sp)]["lo"], prim[(X, extra, sp)]["hi"]) for sp, _, _ in SPLITS]
            summ.append({"test": name, "claim": claim, "X": X, "extra": extra, "verdict": verdict(cis),
                         **{f"{k}_{sp}": prim[(X, extra, sp)][k] for sp, _, _ in SPLITS for k in ("n", "n_ctl", "rate", "rate_ctl", "diff", "lo", "hi", "excl")}})

    # 1b 80% 규칙
    nd = n // D1
    rs = lambda x: x.reshape(nd, D1)   # noqa: E731
    vahd, vald = vah[::D1], val[::D1]
    opn = A_day[::D1]
    sig = np.r_[np.nan, np.sqrt(np.nansum(rs(r1) ** 2, 1))[:-1]] * 1e4
    yrd = yr[::D1]
    for extra in (0, 1):
        ev, pool = rule80(rs(c), rs(h), rs(l), vahd, vald, opn, sig, extra)
        cis, rec = [], {"test": "VAH/VAL", "claim": "1b 80% 규칙(밖→안 30분 머묾 → 반대편 도달)", "X": "-", "extra": extra}
        for sp, _, _ in SPLITS:
            evs = [e for e in ev if yrd[e[0]] == sp]
            y, cr, dd = rule80_match(evs, pool, yrd == sp)
            p, lo, hi = E.mean_ci(y - cr, dd)
            cis.append((p, lo, hi))
            rec.update({f"n_{sp}": len(y), f"n_raw_{sp}": len(evs), f"rate_{sp}": float(y.mean()), f"rate_ctl_{sp}": float(cr.mean()),
                        f"diff_{sp}": p, f"lo_{sp}": lo, f"hi_{sp}": hi})
        rec["verdict"] = verdict(cis)
        summ.append(rec)
    print("80% 규칙 끝", flush=True)

    # 3b 주간 VWAP 위/아래 → 방향
    sma_same = seg_ratio(tp, np.isfinite(tp).astype(float), a_week)
    sma_7d = seg_ratio(tp, np.isfinite(tp).astype(float), j_all - 7 * D1)
    modd = mn % D1
    for Hh, sel in ((240, modd % 240 == 0), (D1, modd == 0)):
        ii = np.flatnonzero(sel & (yr != "") & (j_all >= 1) & (j_all - 1 + Hh < n))
        ref = c[ii - 1]; r = (c[ii - 1 + Hh] / ref - 1) * 1e4
        for cname, ctl in (("SMA_same", sma_same), ("SMA_7d", sma_7d)):
            sv, sc = np.sign(ref - vw_w[ii]), np.sign(ref - ctl[ii])
            ok = np.isfinite(r) & np.isfinite(sv) & np.isfinite(sc) & (sv != 0) & (sc != 0)
            cis, rec = [], {"test": "주간 VWAP", "claim": f"3b 위/아래 → 다음 {Hh // 60}h 방향 (대조 {cname})", "X": "-", "extra": 0}
            for sp, _, _ in SPLITS:
                m = ok & (yr[ii] == sp)
                hv, hc = (sv[m] * r[m] > 0).astype(float), (sc[m] * r[m] > 0).astype(float)
                dd = day_id[ii[m]]
                p, lo, hi = E.mean_ci(hv - hc, dd)
                pb, lob, hib = E.mean_ci((sv[m] - sc[m]) * r[m], dd)
                pv, lov, hiv = E.mean_ci(hv - 0.5, dd)
                cis.append((p, lo, hi))
                rec.update({f"n_{sp}": int(m.sum()), f"rate_{sp}": float(hv.mean()), f"rate_ctl_{sp}": float(hc.mean()),
                            f"diff_{sp}": p, f"lo_{sp}": lo, f"hi_{sp}": hi, f"agree_{sp}": float((sv[m] == sc[m]).mean()),
                            f"bp_vwap_{sp}": float((sv[m] * r[m]).mean()), f"bp_ctl_{sp}": float((sc[m] * r[m]).mean()),
                            f"bpdiff_{sp}": pb, f"bplo_{sp}": lob, f"bphi_{sp}": hib, f"hit_vs50_{sp}": (pv, lov, hiv)})
            rec["verdict"] = verdict(cis)
            summ.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "levels_grid.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    for s in summ:
        f = lambda sp: (f"n={s.get(f'n_{sp}')} 실제 {s[f'rate_{sp}']:.3f} 대조 {s[f'rate_ctl_{sp}']:.3f} "   # noqa: E731
                        f"차 {100 * s[f'diff_{sp}']:+.1f}pp[{100 * s[f'lo_{sp}']:+.1f},{100 * s[f'hi_{sp}']:+.1f}]")
        print(f"{s['test']:<14} {s['claim'][:28]:<28} X={s['X']:<3} +{s['extra']} | 2025 {f('2025')} | 2026 {f('2026')} | {s['verdict']}")


def selftest() -> None:
    # 1) 라벨은 닿은 다음 분부터, 기준가 = 닿은 분 종가. 저항 닿음(s=−1) at j=10.
    n = 400
    h = np.full(n, 100.05); l = np.full(n, 99.95); c = np.full(n, 100.0)
    l[10] = 99.0                                       # 닿은 분 자체의 급락 -- 세면 안 된다
    h[12] = 100.5                                      # 통과 쪽 30bp(100.3) 먼저
    code, tm = outcome(h, l, c, np.array([10]), np.array([-1]), np.array([30.0]))
    assert code[0] == 0 and tm[0] == 2, (code, tm)
    l[11] = 99.0                                       # 다음 분 반전 → 반전 먼저
    assert outcome(h, l, c, np.array([10]), np.array([-1]), np.array([30.0]))[0][0] == 1
    code, tm = outcome(h, l, c, np.array([10]), np.array([-1]), np.array([30.0]), extra=1)   # 한 분 더 늦추면 11분 반전은 안 보인다
    assert code[0] == 0 and tm[0] == 1, (code, tm)
    h2 = h.copy(); h2[12] = 100.05
    assert outcome(h2, np.full(n, 99.95), c, np.array([10]), np.array([-1]), np.array([30.0]))[0][0] == -1   # 둘 다 없음 = 제외

    # 2) VWAP 은 분 j 를 안 쓴다(분 j 에 거대 거래량을 넣어도 VWAP[j] 불변, VWAP[j+1] 은 움직임).
    tp = np.full(n, 100.0); v = np.ones(n); a = np.zeros(n, np.int64)
    base = seg_ratio(tp * v, v, a)
    tp2, v2 = tp.copy(), v.copy(); tp2[200], v2[200] = 150.0, 1e6
    vv = seg_ratio(tp2 * v2, v2, a)
    assert abs(vv[200] - base[200]) < 1e-12 and vv[201] > 149 and np.isnan(vv[MIN_LEN - 1]) and np.isfinite(vv[MIN_LEN])

    # 3) 앵커드 VWAP 앵커 = 전일 고가 분. 이틀 데이터, 첫날 고가 분 500.
    hh = np.full(2 * D1, 100.0); hh[500] = 105.0
    aH = day_anchor(hh, np.argmax)
    assert (aH[:D1] == -1).all() and (aH[D1:] == 500).all()

    # 4) VA 는 전일 프로파일만: 둘째 날 첫 분들에 120 에 거대 거래량 → 둘째 날 VAH/VAL 은 첫째 날(≈100) 값.
    t0 = 1_735_689_600_000                              # 2025-01-01 00:00 UTC
    idx = t0 + np.arange(3 * D1) * 60_000
    px = 100 + 0.5 * np.sin(np.arange(3 * D1) / 50)
    m1 = pd.DataFrame({"h": px + 0.01, "l": px - 0.01, "c": px, "v": 1.0}, index=idx)
    m1.iloc[D1:D1 + 5, :3] = 120.0; m1.iloc[D1:D1 + 5, 3] = 1e6
    va = E.value_area(m1); dd = t0 // 86_400_000
    day_ep = (idx // 60_000) // D1
    vah = pd.Series(day_ep - 1).map(va["vah"]).to_numpy()
    assert np.isnan(vah[:D1]).all() and vah[D1] < 101 and vah[2 * D1] > 119, (vah[D1], vah[2 * D1], va.loc[dd + 1])

    # 5) HVN/LVN 은 [h−24h, h) 만: 26h 지점 첫 분에 150 거대 거래량 → 행 26 에는 없고 행 27 에는 있다.
    nn = 28 * 60
    tpp = np.where(np.arange(nn) % 2 == 0, 99.0, 101.0).astype(float); vv2 = np.ones(nn)
    tpp[26 * 60], vv2[26 * 60] = 150.0, 1e6
    HV, LV = hvn_lvn(tpp, vv2)
    assert not (np.abs(HV[26] - 150) < 0.2).any() and (np.abs(HV[27] - 150) < 0.2).any(), (HV[26], HV[27])
    assert (np.abs(HV[25] - 99) < 0.1).any() and (np.abs(HV[25] - 101) < 0.1).any() and (np.abs(LV[25] - 100) < 0.6).any(), (HV[25], LV[25])

    # 6) 닿음·재무장: 레벨 100.1, 종가 100 에서 두 번 닿음 → 첫 번만. 30bp 멀어졌다 다시 닿으면 센다.
    c3 = np.full(100, 100.0); h3 = np.full(100, 100.05); l3 = np.full(100, 99.95)
    h3[[10, 20, 60]] = 100.12; c3[40] = 100.5; h3[40] = 100.5; c3[39] = 100.0
    U = np.full(100, 100.1); U[40:42] = np.nan
    J, _, S = touches(U, np.full(100, np.nan), c3, h3, l3)
    assert list(J) == [10, 60] and list(S) == [-1, -1], J

    # 7) 80% 규칙: VA [99,101], 시가 102, 5분부터 100 에 머묾 → 확인 분 34(5..34 = 30개), 200분에 VAL 이탈 → 도달.
    C = np.full((1, D1), 100.0); Hh = C + 0.05; Ll = C - 0.05
    C[0, :5] = 102.0; Hh[0, :5] = 102.05; Ll[0, :5] = 101.95
    Ll[0, 200] = 98.9
    ev, pool = rule80(C, Hh, Ll, np.array([101.0]), np.array([99.0]), np.array([102.0]), np.array([100.0]))
    assert len(ev) == 1 and ev[0][1] == 34 and ev[0][2] == 0 and ev[0][5], ev
    Ll[0, 200] = 99.5; Ll[0, 35] = 98.9                 # 확인 분 바로 다음 분 도달: extra=1 이면 못 센다
    ev1, _ = rule80(C, Hh, Ll, np.array([101.0]), np.array([99.0]), np.array([102.0]), np.array([100.0]), extra=1)
    ev0, _ = rule80(C, Hh, Ll, np.array([101.0]), np.array([99.0]), np.array([102.0]), np.array([100.0]))
    assert ev0[0][5] and not ev1[0][5]

    # 8) 판정
    assert verdict([(0.05, 0.01, 0.09), (0.03, 0.005, 0.06)]).startswith("통과")
    assert verdict([(-0.05, -0.09, -0.01), (-0.03, -0.06, -0.005)]).startswith("반대로")
    assert verdict([(0.05, 0.01, 0.09), (0.0, -0.02, 0.02)]).startswith("한 해만")
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("tmp/profile_vwap_levels_20260930"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.data or R.default_data_dir(), a.out)
