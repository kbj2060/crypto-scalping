"""numpy 합·누적합 순서 실측(rtmodel 이 따를 순서 고르기 · 2026-11-20). python rtmodel/tools/check_npsum.py"""
import numpy as np


def pw(a):
    n = len(a)
    if n < 8:
        s = 0.0
        for v in a:
            s += v
        return s
    if n <= 128:
        r = list(a[:8]); m = n - n % 8
        for i in range(8, m, 8):
            for j in range(8):
                r[j] += a[i + j]
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in a[m:]:
            res += v
        return res
    n2 = n // 2; n2 -= n2 % 8
    return pw(a[:n2]) + pw(a[n2:])


rng = np.random.default_rng(1)
bad_pw = bad_first = 0
def blk(a):                              # 2026-11-21: 8,192칸 넘으면 묶음마다 짝 합 → 묶음 합 차례로(rtcollect::ob5m::np_sum)
    s = 0.0
    for j in range(0, len(a), 8192):
        s += pw(a[j:j + 8192])
    return s


bad_blk = 0
for k in range(300):
    n = int(rng.integers(8000, 40000)); a = rng.normal(0, 1, n) * 10 ** rng.uniform(-3, 3, n)
    bad_blk += float(np.sum(a)) != blk(a.tolist())
print("8192 묶음 규칙 다름", bad_blk, "/ 300 (n 8000~40000)")
for k in range(3000):
    n = int(rng.integers(1, 3000)); a = rng.normal(0, 1, n) * 10 ** rng.uniform(-3, 3, n)
    s = float(np.sum(a))
    bad_pw += s != pw(a.tolist())
    bad_first += s != (a[0] + pw(a[1:].tolist()) if n > 1 else a[0])
print("pairwise 전체", bad_pw, "· 첫 값 + 나머지 pairwise", bad_first, "/ 3000")
print("sum([-0.0]) =", np.sum(np.array([-0.0])), "· nansum([nan]) =", np.nansum(np.array([np.nan])), "· cumsum([-0.0, -0.0]) =", np.cumsum(np.array([-0.0, -0.0])))
c = rng.normal(0, 1, 5000); cs = np.cumsum(c); acc = []; t = 0.0
for i, v in enumerate(c):
    t = v if i == 0 else t + v; acc.append(t)
print("cumsum 차례 합과 비트 같음", bool((cs == np.array(acc)).all()))
