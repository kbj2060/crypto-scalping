"""상황 읽기 A/B/C 의 «무조건부 기하 기저율» — 4.7년 5분 패널 (2026-09-22).

왜: 추세에서 엔진의 1순위 적중이 무작위 이하다(원장 새 시대 15.4%, 「항상 되돌림」 56.4%).
   그런데 독립 창이 14개뿐이라 **이 원장으로 계수를 맞추면 과적합**이다. 횡보를 고친 방식이
   본보기다 -- 표본이 아니라 **긴 패널의 기하 기저율**로 사전확률을 다시 깔았다(BASE_RANGE).
   여기서 추세에 같은 것을 한다. 원장은 «목표까지의 거리»만 빌려 쓰고 결과는 안 쓴다.

방법: situation.classify 의 기하를 klines 만으로 재현한다(WINDOW=6·MOVE_THR_FRAC=0.35).
   ⚠️A(가치영역 먼 변)와 B(청산맵 저항)는 라이브 전용 입력이라 패널로 못 만든다. 그래서
     **거리를 배리어로 추상화**한다: 각 과거 창의 30분 경로에서 «아래로 a bp / 위로 b bp /
     아래로 c bp(>a)» 세 배리어의 선착을 센다. 라이브가 실제로 쓰는 거리를 원장에서 가져와
     그 분포 위에서 평균내면 그게 무조건부 기저율이다.
   ⚠️5분봉이라 **같은 봉 안 순서를 모른다**. 같은 봉에서 양쪽이 닿으면 amb 로 빼고 그 비율을
     같이 보고한다(라이브는 1분봉이라 이 모호함이 작다).
   채점 규칙은 **현행과 같다**: 쪽끼리는 선착, 이긴 쪽 안에서는 더 깊은 것(A⊂C).

검증: 같은 코드로 횡보(B=창 고가·C=창 저가·A=잔여)를 돌려 피어의 BASE_RANGE 근거인
   «유지 16.0%»가 재현되는지 먼저 본다. 재현되면 추세 숫자를 믿는다.
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq

WINDOW, MOVE_THR_FRAC, HOR_BARS = 6, 0.35, 6
PANEL = "tmp/eth5m_ohlc_for_base_20260922.parquet"


def load():
    t = pq.read_table(PANEL, columns=["high", "low", "close"])
    return (t.column("high").to_numpy().astype(np.float64),
            t.column("low").to_numpy().astype(np.float64),
            t.column("close").to_numpy().astype(np.float64))


def windows(hi, lo, cl):
    """각 결정봉 i 의 dir·창 고저·직전창 극값·앞 30분 누적편위(bp)."""
    n = len(cl)
    i = np.arange(2 * WINDOW, n - HOR_BARS)          # 직전창까지 있고 앞 30분도 있는 구간
    mid = cl[i]
    w_hi = np.array([hi[i - k] for k in range(WINDOW)]).max(0)
    w_lo = np.array([lo[i - k] for k in range(WINDOW)]).min(0)
    rng_bp = (w_hi - w_lo) / mid * 1e4
    move_bp = (cl[i] - cl[i - WINDOW]) / cl[i - WINDOW] * 1e4
    thr = MOVE_THR_FRAC * rng_bp
    d = np.where(move_bp > thr, 1, np.where(move_bp < -thr, -1, 0))
    p_hi = np.array([hi[i - WINDOW - k] for k in range(WINDOW)]).max(0)
    p_lo = np.array([lo[i - WINDOW - k] for k in range(WINDOW)]).min(0)
    # 앞 30분 누적 편위(bp, 양수). up[:,j] = 봉 j 까지의 최대 상승, dn = 최대 하락
    up = np.maximum.accumulate(np.stack([(hi[i + j] - mid) / mid * 1e4 for j in range(1, HOR_BARS + 1)], 1), 1)
    dn = np.maximum.accumulate(np.stack([(mid - lo[i + j]) / mid * 1e4 for j in range(1, HOR_BARS + 1)], 1), 1)
    return dict(i=i, mid=mid, d=d, rng_bp=rng_bp, w_hi=w_hi, w_lo=w_lo, p_hi=p_hi, p_lo=p_lo, up=up, dn=dn)


NEVER = 99


def first_bar(exc, thr):
    """편위 누적 배열에서 thr 을 처음 넘는 봉 index (없으면 NEVER). thr 은 (N,) 또는 스칼라."""
    t = np.asarray(thr, dtype=np.float64)
    hit = exc >= (t[:, None] if t.ndim else t)
    return np.where(hit.any(1), hit.argmax(1), NEVER)


def resolve(tA, tB, tC):
    """현행 resolve 와 같은 규칙. A·C 는 아래(같은 쪽), B 는 위.
    반환: 0=A 1=B 2=C 3=none 4=amb(같은 봉에 양쪽)."""
    down = np.minimum(tA, tC)
    amb = (down == tB) & (down != NEVER)
    out = np.full(len(tA), 3)
    win_down = (down < tB) & (down != NEVER)
    win_up = (tB < down) & (tB != NEVER)
    out[win_up] = 1
    # 아래쪽이 이겼으면 «반대쪽이 넘겨받기 전까지» 더 깊은 C 를 인정한다
    deep = win_down & (tC < tB)
    out[win_down] = 0
    out[deep] = 2
    out[amb] = 4
    return out


def tally(res, label):
    n = len(res)
    amb = int((res == 4).sum())
    ok = res[res != 4]
    parts = {k: int((ok == v).sum()) for k, v in (("A", 0), ("B", 1), ("C", 2), ("none", 3))}
    m = len(ok)
    print(f"  {label}  n={n:,} (판정불가 {100*amb/n:.1f}% 제외 -> {m:,})")
    for k in ("A", "B", "C", "none"):
        print(f"     {k:5} {100*parts[k]/m:5.1f}%")
    touched = m - parts["none"]
    print("     ─ 닿은 것만 분모(엔진 확률이 사는 조건):",
          " · ".join(f"{k} {100*parts[k]/touched:.1f}%" for k in ("A", "B", "C")))
    return {k: parts[k] / m for k in parts} | {"amb": amb / n, "n": m,
            "touched": {k: parts[k] / touched for k in ("A", "B", "C")}}


def main() -> None:
    hi, lo, cl = load()
    W = windows(hi, lo, cl)
    print(f"패널 {len(cl):,} 봉 · 결정봉 {len(W['i']):,} · "
          f"추세 {int((W['d']!=0).sum()):,} · 횡보 {int((W['d']==0).sum()):,}")

    # ── 검증: 횡보. B=창 고가 · C=창 저가 · A=잔여(둘 다 안 닿음) ──
    print("\n[검증] 횡보 — 피어의 BASE_RANGE 근거(«유지» 16.0%)가 재현되나")
    m = W["d"] == 0
    up_b = (W["w_hi"][m] - W["mid"][m]) / W["mid"][m] * 1e4
    dn_b = (W["mid"][m] - W["w_lo"][m]) / W["mid"][m] * 1e4
    tB = first_bar(W["up"][m], up_b)
    tC = first_bar(W["dn"][m], dn_b)
    tA = np.full(len(tB), NEVER)                      # 횡보 A 는 배리어가 아니라 잔여
    r = resolve(tA, tB, tC)
    r = np.where(r == 3, 0, r)                        # 아무것도 안 닿음 = A(유지)
    chop = tally(r, "횡보(창 자체가 레인지)")
    print(f"     ⇒ «유지» {100*chop['A']:.1f}%  (피어 보고 16.0%)")

    # ── 추세: 거리를 원장에서 빌려 온다 ──
    print("\n[본계산] 추세 — 라이브가 실제로 쓰는 목표 거리 분포 위에서")
    P, O = {}, {}
    for line in open("data/live/situation_log.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        (P if "labels" in rec else O)[int(rec["ts"])] = rec
    live = [r for r in P.values() if "scorable" in r and r["ts"] >= 1789984800
            and r.get("dir") and (r.get("targets_raw") or r.get("targets"))]
    ds = []
    for r in live:
        tg = r.get("targets_raw") or r["targets"]
        mid = r.get("mid")
        if not mid or any(tg.get(k) is None for k in "ABC"):
            continue
        sgn = 1 if r["dir"] > 0 else -1
        a = sgn * (mid - tg["A"]) / mid * 1e4          # A·C 는 이동 반대쪽 = 양수여야 한다
        b = sgn * (tg["B"] - mid) / mid * 1e4
        c = sgn * (mid - tg["C"]) / mid * 1e4
        if a > 0 and b > 0 and c > 0:
            ds.append((a, b, c))
    print(f"  원장에서 가져온 거리 {len(ds)}쌍 "
          f"(중앙 A {np.median([x[0] for x in ds]):.0f} · B {np.median([x[1] for x in ds]):.0f} · C {np.median([x[2] for x in ds]):.0f} bp)")
    tr = W["d"] != 0
    UP, DN = W["up"][tr], W["dn"][tr]
    acc = np.zeros(5)
    for a, b, c in ds:
        r = resolve(first_bar(DN, a), first_bar(UP, b), first_bar(DN, c))
        acc += np.bincount(r, minlength=5)
    tot = acc.sum(); ok = tot - acc[4]
    print(f"  추세 창 {UP.shape[0]:,} × 거리 {len(ds)}쌍 = {int(tot):,} 판정 "
          f"(판정불가 {100*acc[4]/tot:.1f}% 제외)")
    for k, v in (("A", 0), ("B", 1), ("C", 2), ("none", 3)):
        print(f"     {k:5} {100*acc[v]/ok:5.1f}%")
    t = ok - acc[3]
    print("     ─ 닿은 것만 분모:", " · ".join(f"{k} {100*acc[v]/t:.1f}%" for k, v in (("A",0),("B",1),("C",2))))
    print(f"\n  ⇒ 현행 BASE = {{A:34, B:33, C:33}} · 기하 기저율 = "
          f"{{A:{round(100*acc[0]/t)}, B:{round(100*acc[1]/t)}, C:{round(100*acc[2]/t)}}}")


if __name__ == "__main__":
    main()
