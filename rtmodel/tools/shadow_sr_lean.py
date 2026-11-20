"""레벨 5분 그림자 대조: 파이썬 워커(data/live/sr_lean) 대 Rust(data/live/rt_sr_lean)의 기록을 «양쪽 다 채점한 id» 끼리 줄 글자로 비교 (2026-11-21).
읽기만 한다. 두 프로세스는 시작 시각(t0)이 달라 준비 끝·건너뜀이 다를 수 있다 -- 그래서 한쪽에만 있는 id 는 수만 센다(분모 병기).

    python rtmodel/tools/shadow_sr_lean.py [파이썬 디렉터리] [Rust 디렉터리] [--since <epoch 초>]
    종료코드 0 = 양쪽 다 있는 id 가 전부 같음 · 1 = 다른 줄 있음 · 2 = 비교할 id 없음
"""
import json
import sys
from pathlib import Path


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
    for i in both:
        for k, nm in ((0, "ev"), (1, "res"), (2, "feat")):
            if (i in P[k]) != (i in R[k]) and k == 1:
                continue        # 결과는 300초 뒤 -- 한쪽이 아직 안 쓴 것일 수 있다
            if P[k].get(i) != R[k].get(i):
                bad.append((nm, i, P[k].get(i), R[k].get(i)))
    nres = sum(1 for i in both if i in P[1] and i in R[1])
    print(f"사건 id: 파이썬 {len(P[0])} · Rust {len(R[0])} · 양쪽 {len(both)} · 파이썬만 {len(set(P[0]) - set(R[0]))} · Rust만 {len(set(R[0]) - set(P[0]))}")
    print(f"양쪽 다 있는 id 의 줄: ev {len(both)} · 결과 {nres} · 피쳐 {sum(1 for i in both if i in P[2] and i in R[2])} → 다른 줄 {len(bad)}")
    print(f"건너뜀(warm): 파이썬 {len(P[3])} · Rust {len(R[3])} · 양쪽 {len(P[3] & R[3])}")
    for nm, i, x, y in bad[:5]:
        print(f"  {nm} {i}\n    파이썬 {x}\n    Rust   {y}")
    return 2 if not both else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
