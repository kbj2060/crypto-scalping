"""상황 읽기 장부 감사 — 「청산(C)·이탈(B)을 못 맞히나」 (2026-09-21).

읽기 전용. data/live/situation_log.jsonl 만 본다.
🔴세 가지를 강제한다:
  1) **P0 이전 줄을 섞지 않는다** — 09-21 채점 기하 수정(먼 변·1분봉·대칭 라벨) 전후는 다른 실험이다.
     식별자는 시각이 아니라 스키마다(`scorable` 키는 P0 에서 생겼다).
  2) **선점(scorable=false)은 뺀다** — 목표가 이미 지나간 예측은 배리어 경주로 답할 수 없다.
  3) **분모는 겹치지 않는 30분 창** — 예측이 5분마다 나오고 지평이 30분이라 그냥 세면
     같은 움직임을 6번 센다(검정력 부풀림).
"""
import json, math, collections, sys
from pathlib import Path

LOG = Path("data/live/situation_log.jsonl")
HORIZON_S = 1800


def load():
    preds, outs = {}, {}
    for ln in LOG.open(encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        ts = int(r["ts"])
        (preds if "labels" in r else outs)[ts] = r
    for ts, o in outs.items():
        if ts in preds:
            preds[ts]["outcome"] = o.get("outcome")
            preds[ts]["outcome_sym"] = o.get("outcome_sym")
            preds[ts]["path"] = o.get("path")
    return [preds[t] for t in sorted(preds)]


def disjoint(rows, gap=HORIZON_S):
    """겹치지 않는 창만 남긴다 — 앞에서부터 욕심껏."""
    out, last = [], -10**9
    for r in rows:
        if r["ts"] - last >= gap:
            out.append(r); last = r["ts"]
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n):
    return f"{k}/{n} = {100*k/n:4.1f}%" if n else f"0/0 = —"


def block(title):
    print("\n" + "=" * 74); print(title); print("=" * 74)


rows = load()
p1 = [r for r in rows if "scorable" in r]
pre = [r for r in rows if "scorable" not in r]
print(f"장부 {len(rows)}건 = P0 이전 {len(pre)} + P0 이후 {len(p1)}")
if not p1:
    sys.exit("P0 이후 표본이 없다")
span = (p1[-1]["ts"] - p1[0]["ts"]) / 3600
print(f"P0 이후 구간 {span:.1f}h · 예측 간격 중앙 "
      f"{sorted(b['ts']-a['ts'] for a, b in zip(p1, p1[1:]))[len(p1)//2]}s")

RES = ("A", "B", "C", "none")
ok = [r for r in p1 if r.get("outcome") in RES and r.get("scorable")]
drop_unres = sum(1 for r in p1 if r.get("outcome") is None)
drop_amb = sum(1 for r in p1 if r.get("outcome") == "amb")
drop_pass = sum(1 for r in p1 if r.get("outcome") in RES and not r.get("scorable"))
print(f"해결 {len(ok)}건 (대기 {drop_unres} · 판정불가 {drop_amb} · 선점제외 {drop_pass})")
dj = disjoint(ok)
print(f"겹치지 않는 창: {len(dj)}건  ← 유의성은 전부 이 분모로 낸다")

# ── 1. 시나리오별 «말한 것 vs 실제» ────────────────────────────────────────
block("1. 캘리브레이션 — 말한 확률 vs 실제 발생률 (겹치지 않는 창)")
print(f"{'':22} {'말한(평균)':>10} {'실제':>16} {'95% CI':>16}  차")
for k, nm in (("A", "A 되돌림/유지"), ("B", "B 지속·상단이탈"), ("C", "C 플러시·하단이탈"), ("none", "아무것도 안 닿음")):
    if k == "none":
        said = sum(1 - (r["prob"]["A"] + r["prob"]["B"] + r["prob"]["C"]) / 100 for r in dj) / len(dj)
        said = None
    else:
        said = sum(r["prob"][k] for r in dj) / len(dj) / 100
    hit = sum(1 for r in dj if r["outcome"] == k)
    lo, hi = wilson(hit, len(dj))
    s = f"{said*100:9.1f}%" if said is not None else "        —"
    d = f"{(hit/len(dj)-said)*100:+5.1f}pp" if said is not None else ""
    print(f"{nm:22} {s} {pct(hit, len(dj)):>16} [{lo*100:4.1f},{hi*100:5.1f}]  {d}")

# ── 2. 1순위로 찍었을 때 ──────────────────────────────────────────────────
block("2. 1순위로 찍었을 때 그게 실제로 일어났나 (겹치지 않는 창)")
top = collections.defaultdict(lambda: [0, 0])
for r in dj:
    k = max(("A", "B", "C"), key=lambda x: r["prob"][x])
    top[k][1] += 1
    top[k][0] += (r["outcome"] == k)
tot = [sum(v[0] for v in top.values()), sum(v[1] for v in top.values())]
for k in ("A", "B", "C"):
    h, n = top[k]
    lo, hi = wilson(h, n)
    print(f"  1순위 {k}: {pct(h, n):>16}  [{lo*100:4.1f},{hi*100:5.1f}]")
lo, hi = wilson(*tot)
print(f"  전체    : {pct(*tot):>16}  [{lo*100:4.1f},{hi*100:5.1f}]   무작위 기준 33.3%")

# ── 3. 방향별 (횡보 vs 추세) ─────────────────────────────────────────────
block("3. 방향별 — 횡보(dir=0)와 추세(dir≠0)는 다른 문제다")
for lab, sel in (("횡보 dir=0", lambda r: r["dir"] == 0), ("추세 dir≠0", lambda r: r["dir"] != 0)):
    sub = [r for r in dj if sel(r)]
    if not sub:
        continue
    c = collections.Counter(r["outcome"] for r in sub)
    said = {k: sum(r["prob"][k] for r in sub) / len(sub) for k in "ABC"}
    print(f"  {lab}  n={len(sub)}")
    for k in ("A", "B", "C", "none"):
        s = f"말한 {said[k]:4.1f}%" if k in said else "말한    —"
        print(f"     {k:5} {s}   실제 {pct(c[k], len(sub)):>16}")

# ── 4. C(청산) 목표의 출처가 성능을 가르나 ────────────────────────────────
block("4. C 목표를 «청산 군집»에서 뽑았을 때 vs «베이스»에서 뽑았을 때")
for src in ("청산군집", "베이스"):
    sub = [r for r in dj if (r.get("feat") or {}).get("c_target_src") == src]
    if not sub:
        print(f"  {src}: 표본 없음"); continue
    h = sum(1 for r in sub if r["outcome"] == "C")
    said = sum(r["prob"]["C"] for r in sub) / len(sub)
    lo, hi = wilson(h, len(sub))
    print(f"  {src:6} n={len(sub):3}  말한 {said:4.1f}%  실제 C {pct(h, len(sub)):>15} [{lo*100:4.1f},{hi*100:5.1f}]")

# ── 5. 거리 통제 — «못 맞힌다»가 «멀다»의 다른 말인가 ─────────────────────
block("5. 거리 통제 — 목표까지 |bp| 와 실제 도달")
print("   같은 시나리오라도 목표가 멀면 안 닿는 게 당연하다. 거리를 3분위로 갈라 본다.")
for k in ("A", "B", "C"):
    have = [r for r in dj if (r.get("dist_bp") or {}).get(k) is not None]
    if len(have) < 9:
        print(f"  {k}: 표본 {len(have)} — 건너뜀"); continue
    have.sort(key=lambda r: abs(r["dist_bp"][k]))
    n3 = len(have) // 3
    for i, (lab, sub) in enumerate((("가까움", have[:n3]), ("중간", have[n3:2*n3]), ("멂", have[2*n3:]))):
        if not sub: continue
        d = sorted(abs(r["dist_bp"][k]) for r in sub)[len(sub)//2]
        h = sum(1 for r in sub if r["outcome"] == k)
        print(f"  {k} {lab:4} 중앙 {d:6.1f}bp  실제 {pct(h, len(sub)):>15}")

# ── 6. 겹침을 세면 얼마나 부풀어 보이나 ───────────────────────────────────
block("6. 참고 — 겹치는 분모로 세면 (화면이 지금 쓰는 방식)")
c = collections.Counter(r["outcome"] for r in ok)
for k in ("A", "B", "C", "none"):
    print(f"  {k:5} {pct(c[k], len(ok)):>16}   ← n 이 {len(ok)/max(1,len(dj)):.1f}배로 보인다")


# ══════════════════════════════════════════════════════════════════════════
# 7~10. 30분 블록 부트스트랩 — 모든 해결 줄을 쓰되 의존성을 정직하게 센다.
#   겹치는 줄을 그냥 세면 n 이 95배로 보이고, 겹치지 않는 창만 쓰면 n=14 다.
#   같은 30분 창의 줄들은 **한 덩어리로** 리샘플한다(창 안은 완전 상관, 창 사이는 독립 취급).
# ══════════════════════════════════════════════════════════════════════════
import random
random.seed(20260921)
BLOCK_S = HORIZON_S
REPS = 4000


def blocks_of(rows):
    b = collections.defaultdict(list)
    for r in rows:
        b[r["ts"] // BLOCK_S].append(r)
    return list(b.values())


def boot(rows, stat, reps=REPS):
    """stat(rows) -> float|None. 블록 리샘플 분포의 2.5/97.5 분위."""
    bl = blocks_of(rows)
    if not bl:
        return (None, None, None)
    base = stat(rows)
    vals = []
    for _ in range(reps):
        pick = [x for _ in bl for x in random.choice(bl)]
        v = stat(pick)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return (base, None, None)
    return (base, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))])


def rate(k):
    return lambda rs: (sum(1 for r in rs if r["outcome"] == k) / len(rs)) if rs else None


def gap(k):
    return lambda rs: ((sum(1 for r in rs if r["outcome"] == k) - sum(r["prob"][k] / 100 for r in rs)) / len(rs)) if rs else None


block(f"7. 블록 부트스트랩 — 해결 {len(ok)}줄 / 독립 블록 {len(blocks_of(ok))}개")
print(f"{'':22} {'말한':>7} {'실제':>7} {'차':>8}   차의 95% CI (0 포함이면 미달 주장 불가)")
for k, nm in (("A", "A 되돌림/유지"), ("B", "B 지속·상단이탈"), ("C", "C 플러시·하단이탈"), ("none", "아무것도 안 닿음")):
    r0, lo, hi = boot(ok, rate(k))
    said = sum(x["prob"][k] for x in ok) / len(ok) if k != "none" else None
    if k == "none":
        print(f"{nm:22} {'—':>7} {r0*100:6.1f}% {'':>8}   [{lo*100:5.1f}, {hi*100:5.1f}]")
        continue
    g0, glo, ghi = boot(ok, gap(k))
    star = "" if (glo < 0 < ghi) else "  ← 0 배제"
    print(f"{nm:22} {said:6.1f}% {r0*100:6.1f}% {g0*100:+7.1f}pp   [{glo*100:+6.1f}, {ghi*100:+6.1f}]{star}")

block("8. 1순위 적중 (블록 부트스트랩)")
def top_rate(rs):
    if not rs: return None
    return sum(1 for r in rs if max(("A","B","C"), key=lambda x: r["prob"][x]) == r["outcome"]) / len(rs)
r0, lo, hi = boot(ok, top_rate)
print(f"  전체 1순위 적중 {r0*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]   무작위 33.3%")
for k in ("A", "B", "C"):
    sub = [r for r in ok if max(("A","B","C"), key=lambda x: r["prob"][x]) == k]
    if len(sub) < 20:
        print(f"  1순위 {k}: n={len(sub)} — 건너뜀"); continue
    r0, lo, hi = boot(sub, rate(k))
    print(f"  1순위 {k}: n={len(sub):4} ({len(blocks_of(sub))}블록)  적중 {r0*100:5.1f}%  [{lo*100:5.1f}, {hi*100:5.1f}]")

block("9. 목표까지의 거리 — B·C 가 «못 맞히는» 게 «멀다»의 다른 말인가")
for k in ("A", "B", "C"):
    have = [r for r in ok if (r.get("dist_bp") or {}).get(k) is not None]
    if not have: continue
    ds = sorted(abs(r["dist_bp"][k]) for r in have)
    print(f"  {k}: |거리| 중앙 {ds[len(ds)//2]:5.1f}bp  (사분위 {ds[len(ds)//4]:5.1f} ~ {ds[3*len(ds)//4]:5.1f})  n={len(have)}")
print()
for k in ("B", "C"):
    have = sorted([r for r in ok if (r.get("dist_bp") or {}).get(k) is not None],
                  key=lambda r: abs(r["dist_bp"][k]))
    n3 = len(have) // 3
    for lab, sub in (("가까움", have[:n3]), ("중간", have[n3:2*n3]), ("멂", have[2*n3:])):
        if len(sub) < 20: continue
        d = sorted(abs(r["dist_bp"][k]) for r in sub)[len(sub)//2]
        r0, lo, hi = boot(sub, rate(k))
        said = sum(x["prob"][k] for x in sub) / len(sub)
        print(f"  {k} {lab:4} 중앙 {d:6.1f}bp  말한 {said:4.1f}%  실제 {r0*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]  n={len(sub)}/{len(blocks_of(sub))}블록")

block("10. C 목표 출처 · 방향별 (블록 부트스트랩)")
for src in ("청산군집", "베이스"):
    sub = [r for r in ok if (r.get("feat") or {}).get("c_target_src") == src]
    if len(sub) < 20:
        print(f"  C 출처 {src}: n={len(sub)} — 건너뜀"); continue
    r0, lo, hi = boot(sub, rate("C"))
    said = sum(x["prob"]["C"] for x in sub) / len(sub)
    print(f"  C 출처 {src:5}: n={len(sub):4}/{len(blocks_of(sub))}블록  말한 {said:4.1f}%  실제 {r0*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]")
print()
for lab, sel in (("횡보 dir=0", lambda r: r["dir"] == 0), ("추세 dir≠0", lambda r: r["dir"] != 0)):
    sub = [r for r in ok if sel(r)]
    if len(sub) < 20: continue
    print(f"  {lab}  n={len(sub)}/{len(blocks_of(sub))}블록")
    for k in ("A", "B", "C", "none"):
        r0, lo, hi = boot(sub, rate(k))
        said = f"{sum(x['prob'][k] for x in sub)/len(sub):5.1f}%" if k != "none" else "    —"
        print(f"     {k:5} 말한 {said}  실제 {r0*100:5.1f}% [{lo*100:5.1f}, {hi*100:5.1f}]")
