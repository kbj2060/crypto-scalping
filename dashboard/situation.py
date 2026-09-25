"""30분 카드의 **입력 근거** — 2026-09-25 옛 시나리오 로직을 걷어낸 뒤 남은 것.

2026-09-21 에 트레이더 읽기를 규칙으로 옮긴 «상황 읽기 · 30분»(A/B/C 휴리스틱 점수표 · 기하 목표 · 뒤집기 신호 ·
예측 장부 · 해결 · 보정)은 **통째로 제거**했다(사용자 지시). 이유: 09-25 통일축 재측정에서 그 확률이 «항상 되돌림»과
소수점까지 같았고(변별 0), 카드가 «융합 3결과»(dashboard/flow_read.py — 같은 채점축의 3.7년 실측 확률)로 바뀌었다.
옛 코드·장부 감사 스크립트는 git 이력(이 커밋 직전)에 있다.

남은 일: 완결 5분봉 30분 창에서 융합 신호와 근본 신호가 쓰는 원시 근거를 뽑는다 — 30분 방향(슈미트 트리거)·
창 CVD·거부 봉·청산·12h 추세 veto·호가·펀딩/베이시스·BTC. 부호 규약: 매수·상승 = +.
"""
from __future__ import annotations
from typing import Any

WINDOW = 6                 # «이동»을 재는 완결 봉 수 (30분)
# 슈미트 트리거. 단일 임계면 분자(이동)·분모(창폭)가 매 봉 움직여 선을 스칠 때마다 방향이 뒤집힌다 —
#   4.7년 실측으로 «바꿨다 되돌아오는» 변경이 45% → 28%. 방향 전환에도 ENTER 를 요구한다(중립을 건너뛰는 직접 반전 방지).
#   🔴«최소 체류»는 넣지 않는다(09-20 호가 방아쇠 연구에서 역효과).
TREND_ENTER, TREND_EXIT = 0.45, 0.25
REJECT_FRAC = 0.5          # 거부 봉: 마지막 봉 델타가 방향 반대 & |델타| ≥ 이 비율 × 창 최대 |델타| (4.7년 −2.2pp · 6/6년)
FUNDING_NEUTRAL = 0.0001   # 바이낸스 펀딩의 이자 성분(0.01%/8h). 이만큼 더 벗어나야 «쏠림»
BASIS_PCT = 0.75           # |Δ베이시스(창 동안)| 이 마크가격 링 분포의 이 분위 이상이면 «선물/현물 주도»


def _sign(x: float | None) -> int:
    return 0 if x is None else (1 if x > 0 else -1 if x < 0 else 0)


def classify(inp: dict[str, Any]) -> dict[str, Any]:
    """inp: bars(완결 5분봉 오래된→최신, {time, high, low, close, delta, liq_long, liq_short, veto, oi_delta}) ·
    mid · prev_dir(직전 방향, 슈미트 상태) · book{obi, persist_share} · deriv{funding, basis_bp, basis_d_bp, basis_thr_bp} · btc{move_bp, range_bp}.
    반환 {ok, dir, labels[머리글 한 줄], regime{ratio, thr, margin, enter, exit}, evidence{…}}."""
    bars = [b for b in inp.get("bars", []) if b.get("close")]
    if len(bars) < WINDOW + 1:
        return {"ok": False, "reason": f"완결 봉 {len(bars)} < {WINDOW + 1}"}
    w = bars[-WINDOW:]
    mid = float(inp.get("mid") or w[-1]["close"])
    ev: dict[str, Any] = {"mid": mid}

    # ── 30분 방향(슈미트) ──
    rng_bp = (max(b["high"] for b in w) - min(b["low"] for b in w)) / mid * 1e4
    move_bp = (w[-1]["close"] - bars[-WINDOW - 1]["close"]) / bars[-WINDOW - 1]["close"] * 1e4
    ratio = abs(move_bp) / rng_bp if rng_bp > 0 else 0.0
    sgn = _sign(move_bp)
    prev = int(inp.get("prev_dir") or 0)
    if prev == 0:
        d = sgn if ratio > TREND_ENTER else 0
    elif sgn == prev:
        d = 0 if ratio < TREND_EXIT else prev
    else:
        d = sgn if ratio > TREND_ENTER else (0 if ratio < TREND_EXIT else prev)
    thr_next = TREND_EXIT if (d != 0 and sgn == d) else TREND_ENTER
    margin = (ratio - thr_next) if (d != 0 and sgn == d) else (thr_next - ratio)
    ev.update(move_bp=round(move_bp, 1), range_bp=round(rng_bp, 1), dir=d, move_ratio=round(ratio, 3))
    head = {1: f"상승 {move_bp:+.0f}bp", -1: f"하락 {move_bp:+.0f}bp", 0: f"횡보 (±{thr_next * rng_bp:.0f}bp 안)"}[d]

    # ── 체결 · 거부 봉 · 청산 · 12h 추세 ──
    deltas = [b.get("delta") or 0.0 for b in w]
    maxd = max((abs(x) for x in deltas), default=0.0)
    last = deltas[-1]
    ev["cvd"] = round(sum(deltas))
    ev["reject"] = bool(d != 0 and _sign(last) == -d and maxd > 0 and abs(last) >= REJECT_FRAC * maxd)
    ev["liq_long"] = round(sum(b.get("liq_long") or 0 for b in w))
    ev["liq_short"] = round(sum(b.get("liq_short") or 0 for b in w))
    ev["veto"] = w[-1].get("veto") or 0

    # ── 호가 · 펀딩/베이시스 · BTC ──
    book = inp.get("book") or {}
    ev.update(obi=book.get("obi"), persist=book.get("persist_share"))
    dv = inp.get("deriv") or {}
    fr, bd, bthr = dv.get("funding"), dv.get("basis_d_bp"), dv.get("basis_thr_bp")
    ev.update(funding=fr, basis_bp=dv.get("basis_bp"), basis_d_bp=bd, basis_thr_bp=bthr,
              crowd=0 if fr is None or abs(fr - FUNDING_NEUTRAL) < FUNDING_NEUTRAL else _sign(fr - FUNDING_NEUTRAL),
              lead=0 if (d == 0 or bd is None or bthr is None or abs(bd) < bthr) else (1 if _sign(bd) == d else -1))
    btc = inp.get("btc") or {}
    bm, brg = btc.get("move_bp"), btc.get("range_bp")
    d_btc = None if bm is None or brg is None else (1 if bm > 0.35 * brg else -1 if bm < -0.35 * brg else 0)
    ev.update(btc_move_bp=None if bm is None else round(bm, 1),
              btc_rel=None if (d == 0 or d_btc is None) else ("동행" if d_btc == d else "단독"))

    return {"ok": True, "dir": d, "labels": [head], "evidence": ev,
            "regime": {"ratio": round(ratio, 3), "thr": thr_next, "margin": round(margin, 3),
                       "enter": TREND_ENTER, "exit": TREND_EXIT}}


if __name__ == "__main__":   # 자체점검 — 슈미트·거부 봉·청산·베이시스·BTC 가 융합이 기대하는 모양인가
    def bar(t, c, dl, ll=0.0, ls=0.0, h=None, lo=None, veto=0):
        return dict(time=t, high=h or c + 3, low=lo or c - 3, close=c, delta=dl, liq_long=ll, liq_short=ls, veto=veto)
    bars = [bar(0, 2597, -1900), bar(300, 2594, -3400, ll=150e3), bar(600, 2600, 3400, ls=80e3),
            bar(900, 2606, 5300, ls=90e3), bar(1200, 2609, 934, ls=40e3), bar(1500, 2611, 2200, ls=60e3),
            bar(1800, 2614, 1500, ls=50e3), bar(2100, 2615, 5900, ls=120e3, h=2616), bar(2400, 2612, -3000, ls=30e3, veto=1)]
    inp = dict(bars=bars, mid=2612.5, book=dict(obi=0.6, persist_share=0.2),
               deriv=dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=1.0), btc=dict(move_bp=40.0, range_bp=60.0))
    r = classify(inp); e = r["evidence"]
    assert r["ok"] and r["dir"] == 1 and r["labels"][0].startswith("상승")
    assert e["reject"] and e["cvd"] == 5300 + 934 + 2200 + 1500 + 5900 - 3000 and e["veto"] == 1   # 창 = 마지막 6봉
    assert e["liq_short"] == 390000 and e["liq_long"] == 0 and e["crowd"] == -1 and e["lead"] == 1 and e["btc_rel"] == "동행"
    assert classify(dict(inp, btc=dict(move_bp=5.0, range_bp=60.0)))["evidence"]["btc_rel"] == "단독"
    assert classify(dict(inp, deriv=dict(funding=-0.0002, basis_bp=3.0, basis_d_bp=2.0, basis_thr_bp=None)))["evidence"]["lead"] == 0
    # 슈미트: 같은 입력도 이전 방향에 따라 다르다(비율이 두 문턱 사이면 유지)
    sb = [dict(time=i * 300, high=2603 + i, low=2597 + i, close=2600 + i, delta=100) for i in range(6)] \
        + [dict(time=1800, high=2610, low=2600, close=2604.5, delta=-10)]                      # 비율 0.376 = 두 문턱 사이
    assert classify(dict(bars=sb, prev_dir=1))["dir"] == 1 and classify(dict(bars=sb, prev_dir=0))["dir"] == 0
    assert classify(dict(bars=sb, prev_dir=1))["regime"]["thr"] == TREND_EXIT
    assert classify(dict(inp, bars=bars[:5]))["ok"] is False
    print("situation selfcheck ok")
