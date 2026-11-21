"""takeover.sh 의 대조기 (2026-11-21) -- 파이썬이 실제로 쓴 출력 vs Rust 한 회차 출력. 표준 라이브러리만(서버 파이썬 어느 것이든).
  lines <파이썬.jsonl> <Rust.jsonl> --key a,b [--skip x,y] [--rtol 0]   Rust 줄마다 같은 키의 파이썬 줄(마지막)과 칸별 대조
  json  <파이썬.json>  <Rust.json>  [--skip x,y] [--rtol 0]              한 파일 통째로 칸별 대조
  text  <파이썬>       <Rust>                                            글자 그대로
  radar <파이썬 latest> <Rust latest> [--skip x] [--rtol r]             위험 레이더(가격 다른 코인은 마지막 봉 칸만 뺌 -- 아래 LASTBAR)
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


def rows(path):
    """(줄, 값) 목록 + 깨진 줄 수. 원본 pred_record._tail 처럼 바이트를 utf-8(replace)로 읽고 json 이 안 되는 줄(NUL 낀 줄 등)은 건너뛴다.
    (서버 실측 11-22: pred_record BTC·SOL·XRP 에 NUL 59~115개 -- grep 은 «binary file matches» 로 줄을 못 냈다)"""
    out, bad = [], 0
    for ln in open(path, "rb").read().decode("utf-8", "replace").split("\n"):
        if not ln.strip():
            continue
        try:
            out.append((ln, json.loads(ln)))
        except ValueError:
            bad += 1
    return out, bad


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


# 위험 레이더: 원본은 «닫힌» 1분봉만 쓰지만 마지막 봉은 닫힌 직후 몇 초 동안 거래소가 값을 고친다 -- 원본도 다음 회차에 그 봉을 다시 받아 덮는다
#   (live_risk_radar_20261115.update_cache «마지막 봉을 한 번 더 받아 덮는다»). Rust 한 회차는 몇십 초 뒤에 받으니 그 봉의 종가가 다를 수 있다 = 원리상 재현 불가.
#   그래서 price 가 다른 코인은 마지막 봉에서 나오는 칸만 이름으로 빼고 나머지를 견주고, price 가 같은 코인은 전부 견준다(같은 코인이 하나도 없으면 불통과).
LASTBAR = ("price", "sigma15_pct", "band80_pct", "band95_pct", "sigma60_pct", "ratio60", "state60", "alert", "valid", "usual60_pct", "baseline7d_sigma15_pct")


def radar(pa, pb, o):
    a, b = json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
    out = [0, []]
    walk({k: v for k, v in a.items() if k != "coins"}, {k: v for k, v in b.items() if k != "coins"}, "", o["skip"], o["rtol"], out)
    ca, cb = a.get("coins", {}), b.get("coins", {})
    nopair = len(set(ca) ^ set(cb))
    same, moved = [], []
    for c in sorted(set(ca) & set(cb)):
        x, y = ca[c], cb[c]
        if x.get("price") == y.get("price"):
            same.append(c)
            walk(x, y, f".coins.{c}", o["skip"], o["rtol"], out)
        else:
            moved.append(f"{c}(파이썬 {x.get('price')} 대 Rust {y.get('price')})")
            walk(x, y, f".coins.{c}", o["skip"] | set(LASTBAR), o["rtol"], out)
    if moved and not same:
        out[1].append("모든 코인 가격이 다름 -- 견줄 코인이 없다(파이썬 회차 직후 다시)")
    rc = report(out[0], out[1], nopair, [(c,) for c in sorted(set(ca) ^ set(cb))])
    if moved:
        print(f"  가격 다른 코인 {', '.join(moved)} → 그 코인만 {'·'.join(LASTBAR)} 뺌: 마지막 1분봉 종가를 받은 순간이 다름(원본도 다음 회차에 덮음) · 가격 같은 코인 {len(same)}개는 전부 견줌")
    return rc


def main(argv):
    mode = argv[0]
    if mode == "last":
        print(rows(argv[1])[0][-1][1][argv[2]])
        return 0
    if mode == "maxt":                                                          # maxt <디렉터리> <k> -- *.jsonl 의 그 k 줄 중 가장 큰 t(깨진 줄 건너뜀)
        import glob
        ts = [r["t"] for f in sorted(glob.glob(argv[1] + "/*.jsonl")) for _, r in rows(f)[0] if isinstance(r, dict) and r.get("k") == argv[2]]
        print(max(ts) if ts else "")
        return 0
    if mode == "text":
        a, b = open(argv[1], encoding="utf-8").read(), open(argv[2], encoding="utf-8").read()
        return report(1, [] if a == b else [first_diff(a, b)], 0)
    o = opts(argv[3:])
    if mode == "radar":
        return radar(argv[1], argv[2], o)
    if mode == "json":
        out = [0, []]
        walk(json.load(open(argv[1], encoding="utf-8")), json.load(open(argv[2], encoding="utf-8")), "", o["skip"], o["rtol"], out)
        return report(out[0], out[1], 0)
    py = {}
    pa, bad_a = rows(argv[1])
    for ln, r in pa:
        py[tuple(str(r.get(k)) for k in o["key"])] = (ln, r)
    out, nokeys = [0, []], []
    rb, bad_b = rows(argv[2])
    if bad_a or bad_b:
        print(f"  깨진 줄(NUL 등 · 원본 _tail 처럼 건너뜀) 파이썬 {bad_a} · Rust {bad_b}")
    for ln, r in rb:
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
