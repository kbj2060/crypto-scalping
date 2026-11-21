"""pandas rolling(w, min_periods).std() 의 계산 순서 실측(설치본엔 .pyx 소스가 없어 기억한 roll_var 를 파이썬으로 옮겨 비트 대조 · 2026-11-20).
    python rtmodel/tools/check_rollvar.py           # 옮긴 식 대 pandas
    python rtmodel/tools/check_rollvar.py --write   # + tests/fixtures/sr_roll.json(pandas 값 -- tests/sr_lean.rs roll_std_pandas 가 Rust 쪽을 대조)"""
import json
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd


def roll_std(x, w, minp):
    """pandas _libs/window/aggregations.pyx roll_var(ddof=1) + zsqrt -- Welford + Kahan(더하기·빼기 보정 따로) + 같은 값 연속 수."""
    minp = max(minp, 1); n = len(x); out = [math.nan] * n
    nobs = mean = ssq = comp_add = comp_rem = 0.0; same = 0; prev = x[0]
    for i in range(n):
        s, e = max(0, i + 1 - w), i + 1
        if i == 0:
            for j in range(s, e):
                nobs, mean, ssq, comp_add, same, prev = add(x[j], nobs, mean, ssq, comp_add, same, prev)
        else:
            for j in range(max(0, i - w), s):
                nobs, mean, ssq, comp_rem = rem(x[j], nobs, mean, ssq, comp_rem)
            nobs, mean, ssq, comp_add, same, prev = add(x[i], nobs, mean, ssq, comp_add, same, prev)
        if nobs >= minp and nobs > 1:
            v = 0.0 if (nobs == 1 or same >= nobs) else ssq / (nobs - 1)
        else:
            v = math.nan
        out[i] = v if math.isnan(v) else (math.sqrt(v) if v >= 0 else 0.0)
    return out


def add(v, nobs, mean, ssq, comp, same, prev):
    if v != v:
        return nobs, mean, ssq, comp, same, prev
    nobs += 1
    same = same + 1 if v == prev else 1
    prev = v
    pm = mean - comp; y = v - comp; t = y - mean; comp = t + mean - y
    mean = mean + t / nobs if nobs else 0.0
    ssq = ssq + (v - pm) * (v - mean)
    return nobs, mean, ssq, comp, same, prev


def rem(v, nobs, mean, ssq, comp):
    if v == v:
        nobs -= 1
        if nobs:
            pm = mean - comp; y = v - comp; t = y - mean; comp = t + mean - y
            mean = mean - t / nobs
            ssq = ssq - (v - pm) * (v - mean)
        else:
            mean = ssq = 0.0
    return nobs, mean, ssq, comp


rng = np.random.default_rng(3); bad = tot = 0
for k in range(40):
    n = int(rng.integers(50, 3000)); x = rng.normal(0, 1e-3, n) * np.exp(rng.normal(0, 1, n))
    x[rng.random(n) < rng.uniform(0, .3)] = np.nan
    if k % 5 == 0:
        x[100:300] = 1e-4                                   # 같은 값 연속
    w, mp = int(rng.integers(5, 1500)), None
    mp = int(rng.integers(1, w + 1))
    ref = pd.Series(x).rolling(w, min_periods=mp).std().to_numpy(); got = np.array(roll_std(x.tolist(), w, mp))
    same = (ref == got) | (np.isnan(ref) & np.isnan(got)); bad += int((~same).sum()); tot += n
print(f"pandas {pd.__version__} rolling std 대 옮긴 roll_var: 다른 칸 {bad} / {tot}")

if "--write" in sys.argv:   # 끝자리를 가르는 입력: 평균 ≫ 산포(Kahan 보정 몫) · 창 전체가 같은 값(같은 값 연속 규칙) · NaN 섞임 · 실제 창(1440, 1200)
    sets = []
    for k in range(24):
        n = 3000; x = rng.normal(0, 1e-3, n)
        if k % 4 == 1:
            x = 1000.0 + rng.normal(0, 1e-6, n)
        if k % 4 == 2:
            x[500:2200] = 2.5e-4
        if k % 4 == 3:
            x = np.round(rng.normal(0, 3, n)) * 1e-4
        x[rng.random(n) < (0.02 if k % 2 else 0.2)] = np.nan
        w, mp = (1440, 1200) if k < 12 else (int(rng.integers(5, 1500)), None)
        mp = mp or int(rng.integers(1, w + 1))
        ref = pd.Series(x).rolling(w, min_periods=mp).std().to_numpy()
        b = lambda a: np.asarray(a, np.float64).view(np.uint64).tolist()  # noqa: E731
        sets.append({"w": w, "minp": mp, "x": b(x), "std": b(ref)})
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sr_roll.json"
    out.write_text(json.dumps(sets)); print(out.name, out.stat().st_size, "바이트")
