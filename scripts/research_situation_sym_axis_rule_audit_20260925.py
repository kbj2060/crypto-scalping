"""상황 카드 채점축 통일 — 등거리(Y_sym) 하나로 엔진과 점수표를 다시 잰다 (2026-09-25).

읽기 전용. data/live/situation_log.jsonl 만 본다.

**왜 축을 바꾸나**: Y_geo(A/B/C 기하 목표)는 목표 거리가 중앙 17~44bp 로 흩어져 있어
«가장 가까운 목표가 이긴다»를 채점한다 — 09-25 감사에서 엔진 1순위 43.2% < 무모델 최근접
50.4%, 다르게 찍은 632줄에서는 27.5% vs 53.3% 였다. 등거리 배리어(±SYM_K×창폭)로 통일하면
그 인공물이 사라지고, **09-22 의 4.7년 SCORES 감사와 같은 축**이 되어 숫자를 직접 비교할 수 있다
(그 감사도 «추세 구간 B=지속, Y_sym» 으로 쟀다).

**재는 것 둘**
  1) 통일축에서 엔진이 무모델 대조군을 이기나.
  2) 점수표의 각 규칙이 **주장하는 pp** 를 원장이 배제하나. 09-22 재가중이 «1점 = 실측 1pp»
     눈금을 선언했는데, 그때 «미측정»으로 남은 규칙 10여 개가 5~20pp 를 주장하고 있다.
     독립 창이 적어도 그 **크기**는 판정할 수 있다 — 작은 효과를 확인하는 검정력은 없다.

🔴분모는 겹치지 않는 30분 창이다. 예측이 분마다 나오고 지평이 30분이라 그냥 세면 같은 움직임을
  여러 번 센다. 🔴'none'(둘 다 미도달)·'amb'(같은 봉 양쪽)은 방향 질문에 답이 아니라 뺀다.

발동 규칙은 `feat` 에서 재구성한다(장부는 규칙 키를 안 남긴다). 재구성이 맞는지는
**장부에 적힌 prob 를 그대로 되만들어** 확인한다 — §1 자체점검이 그것이고, 틀리면 멈춘다.
"""
import collections
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import situation as sit  # noqa: E402 -- SCORES·BASE·임계를 한 곳에서만 읽는다

LOG = Path(os.environ.get("SIT_LOG", "data/live/situation_log.jsonl"))
# 🔴기본 시작 = 현행 점수표가 **실제로 서빙되기 시작한** 시각(2026-09-22 16:00 UTC). SCORES·BASE 를
#   지금 파일에서 읽으므로 그 이전 줄은 다른 표로 만들어진 것이고, 섞으면 §1 재구성이 0% 로 떨어진다.
#   시각은 커밋(09-22 14:48 UTC)이 아니라 §1 일치율이 100% 로 붙는 지점에서 읽었다 — 배포 지연 때문.
SINCE = int(os.environ.get("SIT_SINCE", "1790092800"))
BLOCK_S = 1800
REPS = 2000
MIN_ON = 30        # 이보다 적게 발동한 규칙은 크기를 말할 수 없다
random.seed(7)


# ── 장부 로더·블록 부트스트랩: research_situation_ledger_scenario_audit_20260921.py 와 같은 규약.
#    그 파일은 임포트하면 감사 전체가 돌아버려서(모듈 최상단 실행) 세 함수만 옮겨 왔다.
def load():
    preds, outs = {}, {}
    for ln in LOG.open(encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        (preds if "labels" in r else outs)[int(r["ts"])] = r
    for ts, o in outs.items():
        if ts in preds:
            preds[ts]["outcome_sym"] = o.get("outcome_sym")
            preds[ts]["outcome"] = o.get("outcome")       # §2b 의 Y_geo 대조용
    return [preds[t] for t in sorted(preds) if t >= SINCE]


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
    base, vals = stat(rows), []
    for _ in range(reps):
        v = stat([x for _ in bl for x in random.choice(bl)])
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return (base, None, None)
    return (base, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))])


# ── 점수표 재구성 (dashboard/situation.py:343-375 와 같은 조건) ──
def fired(f, mid):
    d = int(f.get("dir") or 0)
    out = []
    if f.get("fuel") in ("스퀴즈", "롱이탈"):
        out.append("스퀴즈")
    elif f.get("fuel") == "신규유입":
        out.append("신규유입")
    if f.get("climax"):
        out.append("클라이맥스")
    if f.get("breakout_detect"):
        out.append("전환탐지")
    sig = f.get("cur_sig")
    if (sig == "분배" and d > 0) or (sig == "축적" and d < 0):
        out.append("분배")
    if (sig == "축적" and d > 0) or (sig == "분배" and d < 0):
        out.append("축적")
    if f.get("reject"):
        out.append("거부봉")
    if f.get("cvd_div"):
        out.append("CVD_역행")
    wall, pers = int(f.get("wall") or 0), f.get("persist")
    if wall and d != 0:
        if wall == d:
            thick = pers is not None and pers >= sit.PERSIST_THICK
            thin = pers is not None and pers < sit.PERSIST_THIN
            out.append("벽_동방향_두꺼움" if thick else "벽_동방향_얇음" if thin else "벽_동방향_중간")
        else:
            out.append("벽_역방향")
    va_lo, va_hi = f.get("va_lo"), f.get("va_hi")
    if va_lo is not None and mid is not None and (mid > va_hi or mid < va_lo):
        out.append("가치영역_밖")
    if d == 0:
        for flag, key in (("near_res", "저항근접"), ("near_sup", "지지근접"),
                          ("no_cushion", "쿠션없음"), ("no_cushion_up", "쿠션없음_위")):
            if f.get(flag):
                out.append(key)
    else:
        if f.get("near_res") or f.get("near_sup"):
            out.append("저항근접")
        if f.get("no_cushion"):
            out.append("쿠션없음")
        v = int(f.get("veto") or 0)
        if v:
            out.append("추세정렬" if v == d else "추세역행")
    if f.get("hot"):
        out.append("활발")
    if f.get("trapped"):
        out.append("펀딩_반대쏠림")
    lead = int(f.get("lead") or 0)
    if lead > 0:
        out.append("선물주도")
    elif lead < 0:
        out.append("현물주도")
    if f.get("btc_rel") == "동행":
        out.append("BTC_동행")
    if f.get("btc_rel") == "단독":
        out.append("BTC_단독")
    return out


def prob_of(keys, d, rng_bp):
    sc = dict(sit.BASE) if d else dict(sit.base_range(rng_bp))
    for k in keys:
        for kk, v in sit.SCORES[k].items():
            sc[kk] += v
    tot = sum(max(v, 1) for v in sc.values())
    return {k: round(max(v, 1) / tot * 100) for k, v in sc.items()}


def hdr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


rows = [r for r in load() if r.get("feat") and r.get("outcome_sym") in ("up", "down")]
for r in rows:
    f = r["feat"]
    r["keys"] = fired(f, r.get("mid"))
    r["d"] = int(f.get("dir") or 0)
    # 통일축 라벨: 추세는 «지속»(이동 방향으로 먼저 닿음), 횡보는 «위로 이탈».
    r["cont"] = (r["outcome_sym"] == ("up" if r["d"] > 0 else "down")) if r["d"] else None
    r["up"] = r["outcome_sym"] == "up"

hdr("1. 자체점검 — feat 로 장부의 prob 를 되만들 수 있나 (틀리면 아래 전부 무효)")
ok = [r for r in rows if prob_of(r["keys"], r["d"], r["feat"].get("range_bp")) == r.get("prob")]
print(f"  일치 {len(ok)}/{len(rows)} = {100 * len(ok) / max(len(rows), 1):.1f}%")
print("  남는 불일치는 전부 range_bp 가 HOLD_BY_RANGE 경계(20/30/40/55/75)에 **반올림해서** 걸린 횡보 줄이다"
      " — ev 가 round(rng_bp,1) 로 적어(situation.py:197) 경계 바로 아래 값이 복원되지 않는다. 0.4% 라 둔다.")
assert len(ok) >= 0.98 * len(rows), "prob 재구성 불일치 — 규칙 재구성이 틀렸거나 feat 가 불완전하다"

tr = [r for r in rows if r["d"]]
rg = [r for r in rows if not r["d"]]


def show(lab, rs, stat, ref=None):
    b, lo, hi = boot(rs, stat)
    if b is None:
        print(f"  {lab:34} 표본 없음")
        return
    mark = "  ← 기준 배제" if (ref is not None and not (lo <= ref <= hi)) else ""
    print(f"  {lab:34} {b * 100:5.1f}%  [{lo * 100:5.1f}, {hi * 100:5.1f}]"
          f"  n={len(rs):4}/{len(blocks_of(rs))}블록{mark}")


hdr("2. 통일축 헤드라인 — 등거리 배리어에서 엔진 vs 무모델. 동전 = 50%")
print(f"  추세 n={len(tr)} · 횡보 n={len(rg)}  (기간 {(rows[-1]['ts'] - rows[0]['ts']) / 86400:.1f}일)")
print("  [추세] 목표 = 이동 방향으로 먼저 닿나(지속)")
show("실제 지속률 (무조건부)", tr, lambda rs: sum(r["cont"] for r in rs) / len(rs) if rs else None, 0.5)
show("엔진 1순위가 함의한 방향 적중", tr,
     lambda rs: sum((max(r["prob"], key=r["prob"].get) == "B") == r["cont"] for r in rs) / len(rs) if rs else None, 0.5)
show("무모델: 항상 되돌림", tr, lambda rs: sum(not r["cont"] for r in rs) / len(rs) if rs else None, 0.5)
show("무모델: SMA144 veto 방향", [r for r in tr if r["feat"].get("veto")],
     lambda rs: sum((r["feat"]["veto"] == r["d"]) == r["cont"] for r in rs) / len(rs) if rs else None, 0.5)
print("  [횡보] 목표 = 위로 이탈하나")
show("실제 상단 이탈률", rg, lambda rs: sum(r["up"] for r in rs) / len(rs) if rs else None, 0.5)
show("엔진 1순위가 함의한 방향 적중", rg,
     lambda rs: sum((max(r["prob"], key=r["prob"].get) == "B") == r["up"] for r in rs) / len(rs) if rs else None, 0.5)

hdr("2b. 두 축을 나란히 — 🔴분모가 다르다는 점을 먼저 본다")
geo = [r for r in tr if r.get("outcome") in ("A", "B", "C") and r.get("scorable") is not False]
show("Y_geo 1순위 적중 (양쪽 다 해결된 줄)", geo,
     lambda rs: sum(max(r["prob"], key=r["prob"].get) == r["outcome"] for r in rs) / len(rs) if rs else None)
show("Y_sym 방향 적중 (같은 줄)", geo,
     lambda rs: sum((max(r["prob"], key=r["prob"].get) == "B") == r["cont"] for r in rs) / len(rs) if rs else None, 0.5)
print("  🔴이 부분집합은 **두 배리어가 다 닿은 줄**이라 «크게 움직인 창»으로 치우쳐 있다 — 둘 다 후하게 나온다.")
print("  두 축을 «어느 쪽이 후한가»로 비교하면 안 된다(미도달 처리가 달라 분모가 다르다).")
print("  통일축의 값은 숫자를 올리는 데 있지 않고, 목표 거리라는 교란을 **없애는** 데 있다 — 판정은 §2 헤드라인이다.")

hdr("3. 규칙별 ON−OFF (추세, 통일축) — 점수표가 주장하는 pp 를 원장이 배제하나")
print("  주장 = 그 규칙을 빼면 P(지속) 이 몇 pp 움직이나(장부 줄마다 계산한 평균).")
print("  측정 = 지속률(ON) − 지속률(OFF). CI 는 겹치지 않는 30분 창 블록 부트스트랩.\n")
print(f"  {'규칙':22}{'n':>5} {'주장':>7} {'측정':>8}  {'95% CI':>18}   판정")
half = []
seen = collections.Counter(k for r in tr for k in r["keys"])
for key, n_on in seen.most_common():
    on = [r for r in tr if key in r["keys"]]
    off = [r for r in tr if key not in r["keys"]]
    if n_on < MIN_ON or len(off) < MIN_ON:
        print(f"  {key:22}{n_on:5} {'':>7} {'':>8}  {'표본 부족':>18}")
        continue
    claim = sum(prob_of(r["keys"], r["d"], None)["B"]
                - prob_of([k for k in r["keys"] if k != key], r["d"], None)["B"] for r in on) / len(on)

    def diff(rs, _key=key):
        a = [r for r in rs if _key in r["keys"]]
        b = [r for r in rs if _key not in r["keys"]]
        return (sum(r["cont"] for r in a) / len(a) - sum(r["cont"] for r in b) / len(b)) if a and b else None

    m, lo, hi = boot(tr, diff)
    if m is None or lo is None:
        print(f"  {key:22}{n_on:5} {claim:+6.1f}pp {'':>8}  {'CI 없음':>18}")
        continue
    m, lo, hi = m * 100, lo * 100, hi * 100
    half.append((hi - lo) / 2)
    v = "주장 배제" if not (lo <= claim <= hi) else ("0 배제" if not (lo <= 0 <= hi) else "미달")
    print(f"  {key:22}{n_on:5} {claim:+6.1f}pp {m:+7.1f}pp  [{lo:+6.1f},{hi:+6.1f}]   {v}")

print("\n  «주장 배제» = 점수표가 말하는 크기가 CI 밖 = 그 점수는 너무 크다(또는 부호가 틀렸다).")
print("  «미달» = CI 가 0 과 주장을 둘 다 품는다 = 이 표본으로는 판정 불가(검정력 부족).")

hdr("4. 검정력 — 이 표를 판정하려면 얼마나 더 필요한가")
half.sort()
hw = half[len(half) // 2]
nb, days = len(blocks_of(tr)), (tr[-1]["ts"] - tr[0]["ts"]) / 86400
print(f"  지금 CI 반폭 중앙 {hw:.1f}pp ({nb}블록 · {days:.1f}일).")
print(f"  주장의 크기는 정규화 때문에 통일축에서 −4~+6pp 다 — 점수표의 «10점·20점»이 그만큼으로 줄어든다.")
for tgt in (6, 3):
    need = nb * (hw / tgt) ** 2
    print(f"  ±{tgt}pp 까지 좁히려면 블록 {need:.0f}개 = 약 {days * need / nb:.0f}일 (반폭은 √블록에 반비례)")
