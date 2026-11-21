"""레벨 5분 그림자 대조: 파이썬 워커(data/live/sr_lean) 대 Rust(data/live/rt_sr_lean)의 기록을 «양쪽 다 채점한 id» 끼리 줄 글자로 비교 (2026-11-21).
읽기만 한다. 두 프로세스는 시작 시각(t0)이 달라 준비 끝·건너뜀이 다를 수 있다 -- 그래서 한쪽에만 있는 id 는 수만 센다(분모 병기).

    python rtmodel/tools/shadow_sr_lean.py [파이썬 디렉터리] [Rust 디렉터리] [--since <epoch 초>]
    종료코드 0 = 양쪽 다 있는 id 가 전부 같음(ev·feat 실수 칸은 상대 1e-12 안 · 아래 RTOL) · 1 = 다른 줄 있음 · 2 = 비교할 id 없음
"""
import json
import sys
from pathlib import Path


# 2026-11-22 서버 대조(그림자 01:51~ · 14 id): ev·feat 줄이 실수 끝자리(상대 ≤ 6.4e-15)로만 갈렸다 -- 전부 s1h(= 1분봉 로그수익의 rolling(1440, min_periods=1200).std())와
#   그것에서 나온 칸(s_ret*·s_mom*·s_pos·s_ext·m_rv*·m_rng·m_ls). 두 프로세스는 같은 3,000분 틀(reindex)을 쓰지만 pandas 의 rolling 은 온라인(Welford + Kahan)이라
#   창에서 빠져나간 값들의 반올림 잔차를 들고 간다 -- 창(1,440분) 밖 앞쪽 분봉의 출처(klines 로 채운 분 대 그 프로세스가 체결로 만든 분 · 시작 시각이 다름)가
#   다르면 둘 다 맞는 계산이 끝자리로 갈린다. 창 안 값이 하나라도 다르면 차이는 1e-12 보다 훨씬 크다(양성 대조: takeover_selftest 11단계).
#   그래서 ev·feat 의 실수 칸은 상대 RTOL 안이면 같다고 본다. 판정에 쓰는 칸(EXACT)·정수·문자열·결과(res) 줄은 글자 일치 그대로.
RTOL = 1e-12
EXACT = {"p", "grp"}


def close_enough(a: str | None, b: str | None):
    """두 줄이 실수 끝자리로만 다르면 최대 상대차, 아니면 None."""
    if a is None or b is None:
        return None
    x, y = json.loads(a), json.loads(b)
    if x.keys() != y.keys():
        return None
    worst = 0.0
    for k in x:
        u, v = x[k], y[k]
        if u == v:
            continue
        if k in EXACT or not (isinstance(u, float) and isinstance(v, float)):
            return None
        r = abs(u - v) / max(abs(u), abs(v))
        if not r <= RTOL:
            return None
        worst = max(worst, r)
    return worst


def load(d: Path, since: int):
    ev, res, feat, skip = {}, {}, {}, set()
    for name, into in (("ethusdt.jsonl", None), ("ethusdt_feat.jsonl", feat)):
        p = d / name
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if into is not None:
                into[r["id"]] = ln
            elif r["k"] == "ev" and r["a"] >= since:
                ev[r["id"]] = ln
            elif r["k"] == "res":
                res[r["id"]] = ln
            elif r["k"] == "skip" and r["a"] >= since:
                skip.add((r["a"], r["look"], r["kind"], r["side"]))
    return ev, res, feat, skip


def main() -> int:
    a, since = list(sys.argv[1:]), 0
    if "--since" in a:
        k = a.index("--since"); since = int(a[k + 1]); del a[k:k + 2]
    py, rs = Path(a[0] if a else "data/live/sr_lean"), Path(a[1] if len(a) > 1 else "data/live/rt_sr_lean")
    P, R = load(py, since), load(rs, since)
    both = sorted(set(P[0]) & set(R[0]))
    bad = []
    near, worst = 0, 0.0
    for i in both:
        for k, nm in ((0, "ev"), (1, "res"), (2, "feat")):
            if (i in P[k]) != (i in R[k]) and k == 1:
                continue        # 결과는 300초 뒤 -- 한쪽이 아직 안 쓴 것일 수 있다
            if P[k].get(i) != R[k].get(i):
                r = None if k == 1 else close_enough(P[k].get(i), R[k].get(i))   # 결과 줄은 글자 일치 그대로
                if r is None:
                    bad.append((nm, i, P[k].get(i), R[k].get(i)))
                else:
                    near, worst = near + 1, max(worst, r)
    nres = sum(1 for i in both if i in P[1] and i in R[1])
    print(f"사건 id: 파이썬 {len(P[0])} · Rust {len(R[0])} · 양쪽 {len(both)} · 파이썬만 {len(set(P[0]) - set(R[0]))} · Rust만 {len(set(R[0]) - set(P[0]))}")
    print(f"양쪽 다 있는 id 의 줄: ev {len(both)} · 결과 {nres} · 피쳐 {sum(1 for i in both if i in P[2] and i in R[2])} → 다른 줄 {len(bad)}")
    print(f"건너뜀(warm): 파이썬 {len(P[3])} · Rust {len(R[3])} · 양쪽 {len(P[3] & R[3])}")
    if near:
        print(f"끝자리만 다른 줄 {near} (실수 칸 최대 상대 {worst:.1e} ≤ {RTOL:g} · 판정 칸 {'·'.join(sorted(EXACT))}·정수·문자열은 같음) -- 다른 줄로 세지 않음")
    for nm, i, x, y in bad[:5]:
        print(f"  {nm} {i}\n    파이썬 {x}\n    Rust   {y}")
    return 2 if not both else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
