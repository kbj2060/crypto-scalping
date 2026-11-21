"""4시간 방향 피쳐(scripts/live_eth_dir4h_20261015.build_features)가 쓰는 pandas·numpy 연산의 계산 순서 실측 (2026-11-21 · R5 dir4h).
설치본(pandas 2.3.3)에 .pyx 소스가 없어 기억한 식을 파이썬으로 옮겨 실제 값과 비트 대조한다. 여기서 통과한 식만 src/dir4h.rs 가 따른다.

    python rtmodel/tools/check_pandas_ops.py
"""
import math

import numpy as np
import pandas as pd


def roll_mean(x, w, minp):
    """roll_mean: Kahan 더하기·빼기(보정 따로) · 음수 개수(neg_ct) 부호 보정 · 같은 값 연속이면 그 값."""
    out = [math.nan] * len(x); nobs = 0; sx = ca = cr = 0.0; neg = 0; same = 0; prev = x[0]
    for i in range(len(x)):
        if i >= w and x[i - w] == x[i - w]:
            v = x[i - w]; nobs -= 1; y = -v - cr; t = sx + y; cr = t - sx - y; sx = t
            neg -= math.copysign(1, v) < 0
        v = x[i]
        if v == v:
            nobs += 1; y = v - ca; t = sx + y; ca = t - sx - y; sx = t
            neg += math.copysign(1, v) < 0
            same = same + 1 if v == prev else 1; prev = v
        if nobs >= max(minp, 1) and nobs > 0:
            r = sx / nobs
            if same >= nobs:
                r = prev
            elif neg == 0 and r < 0:
                r = 0.0
            elif neg == nobs and r > 0:
                r = 0.0
            out[i] = r
    return out


def _window_sorted(x, i, w):
    return sorted(v for v in x[max(0, i - w + 1):i + 1] if v == v)


def roll_quantile(x, w, minp, q):
    """roll_quantile(linear): idx = q·(n−1) · 정수면 그 값 · 아니면 lo + (hi − lo)·(idx − ⌊idx⌋)."""
    out = [math.nan] * len(x)
    for i in range(len(x)):
        s = _window_sorted(x, i, w); n = len(s)
        if n >= max(minp, 1):
            f = q * (n - 1); k = int(f)
            out[i] = s[k] if k == f else s[k] + (s[k + 1] - s[k]) * (f - k)
    return out


def roll_median(x, w, minp):
    """roll_median_c: 홀수면 가운데 · 짝수면 (가운데 둘 합)/2."""
    out = [math.nan] * len(x)
    for i in range(len(x)):
        s = _window_sorted(x, i, w); n = len(s)
        if n >= max(minp, 1):
            out[i] = s[n // 2] if n % 2 else (s[n // 2] + s[n // 2 - 1]) / 2
    return out


def ewm_mean(x, alpha):
    """ewm(alpha, adjust=True, ignore_na=False, min_periods=0).mean()."""
    out = [math.nan] * len(x); w = x[0]; old = 1.0; f = 1.0 - alpha; nobs = int(x[0] == x[0])
    out[0] = w if nobs else math.nan
    for i in range(1, len(x)):
        v = x[i]; obs = v == v; nobs += obs
        if w == w:
            old *= f
            if obs:
                if w != v:
                    w = (old * w + 1.0 * v) / (old + 1.0)
                old += 1.0
        elif obs:
            w = v
        out[i] = w if nobs >= 1 else math.nan
    return out


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int(((a == b) | (np.isnan(a) & np.isnan(b))).sum()), len(a)


def main():
    rng = np.random.default_rng(5); tot = {}
    def add(k, r):
        s, n = tot.get(k, (0, 0)); tot[k] = (s + r[0], n + r[1])
    for k in range(12):
        n = 2500; x = rng.normal(0, 1, n) * np.exp(rng.normal(0, 1, n))
        if k % 3 == 1:
            x = 3000 + np.round(rng.normal(0, 1, n).cumsum(), 2)          # 가격처럼(평균 ≫ 산포 · 같은 값 반복)
        if k % 3 == 2:
            x[300:700] = 0.5; x[900:950] = -0.25
        x[rng.random(n) < 0.05] = np.nan
        s = pd.Series(x); L = x.tolist()
        for w, mp in ((20, None), (144, None), (288, 144), (12, None)):
            add("rolling mean", same(s.rolling(w, min_periods=mp).mean(), roll_mean(L, w, mp if mp is not None else w)))
        for w, mp in ((300, 60), (500, 288)):
            add("rolling median", same(s.rolling(w, min_periods=mp).median(), roll_median(L, w, mp)))
            for q in (.25, .75):
                add("rolling quantile", same(s.rolling(w, min_periods=mp).quantile(q), roll_quantile(L, w, mp, q)))
        for a in (1 / 14, 1 / 144):
            add("ewm mean", same(s.ewm(alpha=a).mean(), ewm_mean(L, a)))
    for k, (a, b) in tot.items():
        print(f"{k:<18} 비트 같음 {a}/{b}")


if __name__ == "__main__":
    main()
