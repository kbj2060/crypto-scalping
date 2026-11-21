"""takeover.sh 의 대조기 (2026-11-21) -- 파이썬이 실제로 쓴 출력 vs Rust 한 회차 출력. 표준 라이브러리만(서버 파이썬 어느 것이든).
  lines <파이썬.jsonl> <Rust.jsonl> --key a,b [--skip x,y] [--rtol 0]   Rust 줄마다 같은 키의 파이썬 줄(마지막)과 칸별 대조
  json  <파이썬.json>  <Rust.json>  [--skip x,y] [--rtol 0]              한 파일 통째로 칸별 대조
  text  <파이썬>       <Rust>                                            글자 그대로
  last  <jsonl> <칸>                                                     마지막 줄의 그 칸(셸용)
출력 한 줄: «칸 N · 어긋남 M · 짝 없음 K» (+ 처음 어긋난 몇 개). 종료코드 0 = N>0 · M=0 · K=0.
skip 은 이름으로만 뺀다(시각·실행 시간처럼 달라도 되는 칸). rtol>0 은 실수 칸만(위험 레이더 X@β 의 BLAS 합 순서 끝자리 -- REMAINING.md).
--text 를 주면 skip 이 없을 때 줄 글자도 같아야 한다(칸이 같아도 서식이 다르면 어긋남).
"""
import json
import math
import sys


def walk(a, b, path, skip, rtol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a or k not in b:
                out[0] += 1; out[1].append(f"{path}.{k}: 한쪽에만")
                continue
            walk(a[k], b[k], f"{path}.{k}", skip, rtol, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[0] += 1; out[1].append(f"{path}: 길이 {len(a)} 대 {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", skip, rtol, out)
        return
    out[0] += 1
    if isinstance(a, float) and isinstance(b, float) and rtol > 0:
        ok = a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= rtol * max(abs(a), abs(b))
    else:
        ok = type(a) is type(b) and (a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)))
    if not ok:
        out[1].append(f"{path}: 파이썬 {a!r} 대 Rust {b!r}")


def opts(argv):
    o = {"key": [], "skip": set(), "rtol": 0.0, "text": False}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--text":
            o["text"] = True; i += 1; continue
        v = argv[i + 1]; i += 2
        if k == "--key":
            o["key"] = v.split(",")
        elif k == "--skip":
            o["skip"] = set(x for x in v.split(",") if x)
        elif k == "--rtol":
            o["rtol"] = float(v)
    return o


def report(n, bad, nopair, nokeys=()):
    """한 줄 요약 + 어긋난 칸의 예(처음 5개 · 파이썬 값 대 Rust 값) + 짝 없는 키의 예(처음 3개) -- 원인을 그 자리에서 가리게."""
    print(f"칸 {n} · 어긋남 {len(bad)} · 짝 없음 {nopair}" + ("" if not bad else " · 예: " + " | ".join(bad[:5]))
          + ("" if not nokeys else " · 짝 없는 키 예: " + ", ".join("/".join(k) for k in list(nokeys)[:3])))
    return 0 if n > 0 and not bad and nopair == 0 else 1


def first_diff(a, b):
    """두 글자에서 처음 다른 줄(번호 · 앞 100자씩)."""
    la, lb = a.splitlines(), b.splitlines()
    for i in range(max(len(la), len(lb))):
        x, y = (la[i] if i < len(la) else "<없음>"), (lb[i] if i < len(lb) else "<없음>")
        if x != y:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            return f"{i + 1}번째 줄 {j + 1}자부터 -- 파이썬 {x[max(0, j - 20):j + 80]!r} 대 Rust {y[max(0, j - 20):j + 80]!r}"
    return "줄 끝 공백·개행 차이"


def main(argv):
    mode = argv[0]
    if mode == "last":
        rows = [ln for ln in open(argv[1], encoding="utf-8").read().splitlines() if ln.strip()]
        print(json.loads(rows[-1])[argv[2]])
        return 0
    if mode == "text":
        a, b = open(argv[1], encoding="utf-8").read(), open(argv[2], encoding="utf-8").read()
        return report(1, [] if a == b else [first_diff(a, b)], 0)
    o = opts(argv[3:])
    if mode == "json":
        out = [0, []]
        walk(json.load(open(argv[1], encoding="utf-8")), json.load(open(argv[2], encoding="utf-8")), "", o["skip"], o["rtol"], out)
        return report(out[0], out[1], 0)
    py = {}
    for ln in open(argv[1], encoding="utf-8").read().splitlines():
        if ln.strip():
            r = json.loads(ln)
            py[tuple(str(r.get(k)) for k in o["key"])] = (ln, r)
    out, nokeys = [0, []], []
    for ln in open(argv[2], encoding="utf-8").read().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        k = tuple(str(r.get(x)) for x in o["key"])
        if k not in py:
            nokeys.append(k)
            continue
        pl, pr = py[k]
        nb = len(out[1])
        walk(pr, r, "/".join(k), o["skip"], o["rtol"], out)
        if o["text"] and not o["skip"] and pl != ln and len(out[1]) == nb:     # 값은 같은데 서식(1e-05·키 순서·공백)이 다름
            out[1].append(f"{'/'.join(k)}: 줄 글자 {first_diff(pl, ln)}")
    return report(out[0], out[1], len(nokeys), nokeys)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
